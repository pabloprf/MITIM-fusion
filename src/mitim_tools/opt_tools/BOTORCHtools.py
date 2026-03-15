"""
**************************************************************************************************************
This set of tools are custom modifications to BOTORCH or GPYTORCH ones to satisfy my needs
**************************************************************************************************************
"""

import torch
import botorch
import gpytorch
from IPython import embed
from mitim_tools.misc_tools.LOGtools import printMsg as print

# ----------------------------------------------------------------------------------------------------------------------------
# Performance helpers
# ----------------------------------------------------------------------------------------------------------------------------

_original_thread_state = {}

def configure_performance_settings(n_threads=None):
    """
    Restrict thread counts for GP inference and save the original state so that
    restore_performance_settings() can hand full threads back to physics codes.

    On HPC clusters OMP_NUM_THREADS is typically set to the full node core count by the
    scheduler (e.g. 64) for the benefit of the physics code.  Passing that value straight
    to PyTorch causes massive thread oversubscription on the tiny GP matrices used here
    (e.g. 5×5 Cholesky with 64 threads is ~6× *slower* than with 4 threads).
    We therefore cap at MITIM_GP_THREADS (default 4) regardless of OMP_NUM_THREADS.

    NOTE: only in-process thread pools (PyTorch, MKL, OpenBLAS via threadpoolctl) are
    restricted.  Environment variables are NOT modified, so child processes launched for
    physics evaluation inherit the original scheduler allocation unchanged.
    """
    import os
    import linear_operator

    if n_threads is None:
        n_threads = int(os.environ.get("MITIM_GP_THREADS", 4))

    # Save original in-process thread counts so restore_performance_settings() can undo this
    _original_thread_state["torch_num_threads"] = torch.get_num_threads()
    try:
        import threadpoolctl
        _original_thread_state["blas_info"] = threadpoolctl.threadpool_info()
    except ImportError:
        pass

    # PyTorch intraop / interop (in-process only — does not affect child processes)
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # can only be set once before any parallel work

    # BLAS/OpenMP thread pools via threadpoolctl (in-process only — child processes
    # inherit the environment unchanged and use their full SLURM allocation)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=n_threads, user_api="blas")
        threadpoolctl.threadpool_limits(limits=n_threads, user_api="openmp")
        blas_info = {lib["prefix"]: lib["num_threads"] for lib in threadpoolctl.threadpool_info()}
    except ImportError:
        blas_info = {"note": "threadpoolctl not available"}

    linear_operator.settings.max_cholesky_size._set_value(2000)
    print(f"\t[perf] torch_num_threads={n_threads}, blas_threads={blas_info}, max_cholesky_size=2000", typeMsg="i")


def restore_performance_settings():
    """
    Restore in-process thread counts to the values saved by configure_performance_settings().

    Call this before any in-process physics evaluation so that physics code running in the
    same Python process gets the full thread allocation back.  Not needed when physics runs
    as subprocesses (they were never affected).
    """
    if not _original_thread_state:
        return  # configure_performance_settings() was never called

    torch.set_num_threads(_original_thread_state["torch_num_threads"])

    try:
        import threadpoolctl
        orig = _original_thread_state.get("blas_info", [])
        for lib in orig:
            try:
                threadpoolctl.threadpool_limits(limits=lib["num_threads"], user_api=lib.get("user_api"))
            except Exception:
                pass
        restored = {lib["prefix"]: lib["num_threads"] for lib in threadpoolctl.threadpool_info()}
    except ImportError:
        restored = {}

    print(f"\t[perf] thread pools restored: torch={_original_thread_state['torch_num_threads']}, blas={restored}", typeMsg="i")


# ----------------------------------------------------------------------------------------------------------------------------
# SingleTaskGP needs to be modified because I want to input options and outcome transform taking X, otherwise it should be a copy
# ----------------------------------------------------------------------------------------------------------------------------


class ExactGPcustom(botorch.models.gp_regression.SingleTaskGP):
    def __init__(
        self,
        train_X,
        train_Y,
        train_Yvar,
        input_transform=None,
        outcome_transform=None,
        surrogate_options={},
        variables=None,
        output=None,
        train_X_added=torch.Tensor([]),
        train_Y_added=torch.Tensor([]),
        train_Yvar_added=torch.Tensor([]),
    ):
        """
        _added refers to already-transformed variables that are added from table
        """

        TypeMean = surrogate_options.get("TypeMean", 0)
        TypeKernel = surrogate_options.get("TypeKernel", 0)
        FixedNoise = surrogate_options.get("FixedNoise", False)
        ConstrainNoise = surrogate_options.get("ConstrainNoise", -1e-4)
        learn_additional_noise = surrogate_options.get("ExtraNoise", False)
        additional_constraints = surrogate_options.get("additional_constraints", None)
        print("\t\t* Surrogate model options:")
        print(f"\t\t\t- FixedNoise: {FixedNoise} (extra noise: {learn_additional_noise}), TypeMean: {TypeMean}, TypeKernel: {TypeKernel}, ConstrainNoise: {ConstrainNoise:.1e}")

        self.store_training(
            train_X,
            train_X_added,
            train_Y,
            train_Y_added,
            train_Yvar,
            train_Yvar_added,
            input_transform,
            outcome_transform,
        )

        """
		----------------------------------------------------------------------------------------
		What set_dimensions did, and select things to train (already transformed and normalized)
		----------------------------------------------------------------------------------------
		"""

        # Grab num_outputs
        self._num_outputs = train_Y.shape[-1]

        # Grab ard_num_dims
        if train_X.shape[0] > 0:
            with torch.no_grad():
                transformed_X = self.transform_inputs(
                    X=train_X, input_transform=input_transform
                )
            self.ard_num_dims = transformed_X.shape[-1]
        else:
            self.ard_num_dims = train_X_added.shape[-1]
            transformed_X = torch.empty((0, self.ard_num_dims)).to(train_X)

        # Transform outcomes
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_X, train_Y, train_Yvar)

        # Added points are raw transformed, so I need to normalize them
        if train_X_added.shape[0] > 0:
            train_X_added = input_transform["tf2"](train_X_added)
            train_Y_added, train_Yvar_added = outcome_transform["tf2"](
                train_Y_added, train_Yvar_added
            )
        # -----

        train_X_usedToTrain = torch.cat((transformed_X, train_X_added), axis=0)
        train_Y_usedToTrain = torch.cat((train_Y, train_Y_added), axis=0)
        train_Yvar_usedToTrain = torch.cat((train_Yvar, train_Yvar_added), axis=0)

        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X_usedToTrain, train_Y=train_Y_usedToTrain
        )

        train_Y_usedToTrain = train_Y_usedToTrain.squeeze(-1)
        train_Yvar_usedToTrain = train_Yvar_usedToTrain.squeeze(-1)

        """
		-----------------------------------------------------------------------
		Likelihood and Noise
		-----------------------------------------------------------------------
		"""

        self._subset_batch_dict = {}

        if FixedNoise:
            # Noise not inferred, given by data
            
            likelihood = (
                gpytorch.likelihoods.gaussian_likelihood.FixedNoiseGaussianLikelihood(
                    noise=train_Yvar_usedToTrain.clip(1e-6), # I clip the noise to avoid numerical issues (gpytorch would do it anyway, but this way it doesn't throw a warning)
                    batch_shape=self._aug_batch_shape,
                    learn_additional_noise=learn_additional_noise,
                )
            )

        else:
            # Infer Noise

            noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate

            if ConstrainNoise < 0:
                noise_constraint = gpytorch.constraints.constraints.GreaterThan(
                    -ConstrainNoise, transform=None, initial_value=noise_prior_mode
                )
            else:
                noise_constraint = gpytorch.constraints.constraints.Interval(
                    1e-6, ConstrainNoise, transform=None, initial_value=noise_prior_mode
                )

            likelihood = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=noise_constraint,
            )

            self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2

        """
		-----------------------------------------------------------------------
		Initialize ExactGP
		-----------------------------------------------------------------------
		"""

        gpytorch.models.exact_gp.ExactGP.__init__(
            self,
            train_inputs=train_X_usedToTrain,
            train_targets=train_Y_usedToTrain,
            likelihood=likelihood,
        )

        """
		-----------------------------------------------------------------------
		GP Mean
		-----------------------------------------------------------------------
		"""

        if TypeMean == 0:
            self.mean_module = gpytorch.means.constant_mean.ConstantMean(
                batch_shape=self._aug_batch_shape
            )
        elif TypeMean == 1:
            self.mean_module = gpytorch.means.linear_mean.LinearMean(
                self.ard_num_dims, batch_shape=self._aug_batch_shape, bias=True
            )
        elif TypeMean == 2:
            self.mean_module = MITIM_LinearMeanGradients(
                batch_shape=self._aug_batch_shape, variables=variables, output=output
            )

        """
		-----------------------------------------------------------------------
		GP Kernel - Covariance
		-----------------------------------------------------------------------
		"""

        # Priors
        lengthscale_prior = gpytorch.priors.torch_priors.GammaPrior(3.0, 6.0)
        outputscale_prior = gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15)

        # Do not allow too small lengthscales?
        if (additional_constraints is not None) and ("lenghtscale_constraint" in additional_constraints):
            lengthscale_constraint = additional_constraints["lenghtscale_constraint"]
        else:
            lengthscale_constraint = None

        self._subset_batch_dict["covar_module.raw_outputscale"] = -1
        self._subset_batch_dict["covar_module.base_kernel.raw_lengthscale"] = -3

        if TypeKernel == 0:
            self.covar_module = gpytorch.kernels.scale_kernel.ScaleKernel(
                base_kernel=gpytorch.kernels.matern_kernel.MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=lengthscale_constraint,
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=outputscale_prior,
            )
        elif TypeKernel == 1:
            self.covar_module = gpytorch.kernels.scale_kernel.ScaleKernel(
                base_kernel=gpytorch.kernels.rbf_kernel.RBFKernel(
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=lengthscale_constraint,
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=outputscale_prior,
            )
        elif TypeKernel == 2:
            self.covar_module = MITIM_ConstantKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            )
        elif TypeKernel == 3:
            self.covar_module = gpytorch.kernels.scale_kernel.ScaleKernel(
                base_kernel=MITIM_NNKernel(
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=lengthscale_constraint,
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=outputscale_prior,
            )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

        self.to(train_X)

    def store_training(self, x, xa, y, ya, yv, yva, input_transform, outcome_transform):

        # x, y are raw untransformed, and I want raw transformed
        if input_transform is not None:
            x_tr = input_transform["tf1"](x)
        else:
            x_tr = x
        if outcome_transform is not None:
            y_tr, yv_tr = outcome_transform["tf1"](x, y, yv)
        else:
            y_tr, yv_tr = y, yv

        # xa, ya are raw transformed
        xa_tr = xa
        ya_tr, yva_tr = ya, yva

        self.train_X_usedToTrain = torch.cat((xa_tr, x_tr), axis=0)
        self.train_Y_usedToTrain = torch.cat((ya_tr, y_tr), axis=0)
        self.train_Yvar_usedToTrain = torch.cat((yva_tr, yv_tr), axis=0)

    # Modify posterior call from BatchedMultiOutputGPyTorchModel to call posterior untransform with "X"

    def posterior(
        self,
        X,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs,
    ):
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        Xtr = self.transform_inputs(X)
        with botorch.models.utils.gpt_posterior_settings():
            # insert a dimension for the output dimension
            if self._num_outputs > 1:
                Xtr, output_dim_idx = botorch.models.utils.add_output_dim(
                    X=Xtr, original_batch_shape=self._input_batch_shape
                )
            # NOTE: BoTorch's GPyTorchModels also inherit from GPyTorch's ExactGP, thus
            # self(X) calls GPyTorch's ExactGP's __call__, which computes the posterior,
            # rather than e.g. SingleTaskGP's forward, which computes the prior.
            mvn = self(Xtr)
            mvn = self._apply_noise(X=Xtr, mvn=mvn, observation_noise=observation_noise)
            if self._num_outputs > 1:
                mean_x = mvn.mean
                covar_x = mvn.lazy_covariance_matrix
                output_indices = output_indices or range(self._num_outputs)
                mvns = [
                    gpytorch.distributions.MultivariateNormal(
                        mean_x.select(dim=output_dim_idx, index=t),
                        covar_x[(slice(None),) * output_dim_idx + (t,)],
                    )
                    for t in output_indices
                ]
                mvn = gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)

        posterior = botorch.posteriors.gpytorch.GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(X, posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

# ----------------------------------------------------------------------------------------------------------------------------
# ModelListGP needs to be modified to allow me to have "common" parameters to models, to not run at every transformation again
# ----------------------------------------------------------------------------------------------------------------------------


class ModifiedModelListGP(botorch.models.model_list_gp_regression.ModelListGP):
    def __init__(self, *gp_models):
        super().__init__(*gp_models)

    def prepareToGenerateCommons(self):
        self.models[0].input_transform.tf1.flag_to_store = True
        # Make sure that this ModelListGP evaluation is fresh
        if ("parameters_combined" in self.models[0].input_transform.tf1.surrogate_parameters):
            del self.models[0].input_transform.tf1.surrogate_parameters["parameters_combined"]

    def cold_startCommons(self):
        self.models[0].input_transform.tf1.flag_to_store = False
        if ("parameters_combined" in self.models[0].input_transform.tf1.surrogate_parameters):
            del self.models[0].input_transform.tf1.surrogate_parameters["parameters_combined"]

    def transform_inputs(self, X):
        self.prepareToGenerateCommons()
        X_tr = super().transform_inputs(X)
        self.cold_startCommons()

        return X_tr

    def setup_batched_inference(self):
        """Pre-compute batched kernel and cached Cholesky/alpha for vectorised posterior.

        Supports two kernel types (may be mixed across models):
          - ScaleKernel(Matern/RBF): batched via stacked raw parameters
          - MITIM_ConstantKernel:    K(x1,x2)=1 always; no kernel call needed at inference

        Any mean module is supported (ConstantMean, LinearMean, MITIM_LinearMeanGradients, …).
        Models are grouped by (kernel_type, n_train, ard) so that heterogeneous training-set
        sizes are handled — each group is batched independently.
        Falls back gracefully on any incompatibility.
        Call once after all sub-models are fitted.
        """
        import copy

        self._batched_ready = False
        models = self.models
        N = len(models)
        if N == 0:
            return

        # --- Classify each model; group by (kernel_type, n_train, ard) ---
        # sc_buckets: dict (n_train, ard) -> list of model indices
        # ck_buckets: dict (n_train,)     -> list of model indices
        sc_buckets = {}
        ck_buckets = {}

        for i, m in enumerate(models):
            covar  = m.covar_module
            n_i    = m.train_inputs[0].shape[0]
            if isinstance(covar, gpytorch.kernels.ScaleKernel) and isinstance(
                covar.base_kernel, (gpytorch.kernels.MaternKernel, gpytorch.kernels.RBFKernel)
            ):
                ard_i = covar.base_kernel.raw_lengthscale.shape[-1]
                key   = (n_i, ard_i)
                sc_buckets.setdefault(key, []).append(i)
            elif isinstance(covar, MITIM_ConstantKernel):
                ck_buckets.setdefault((n_i,), []).append(i)
            else:
                print(
                    f"\t[MITIM: GP batching] Cannot batch: model[{i}] covar_module is "
                    f"{type(covar).__name__} (not supported)",
                    typeMsg="w",
                )
                return

        try:
            with torch.no_grad():
                ref = models[0].train_inputs[0]

                # --- Outcome tf2 stats for ALL models (in original order) ---
                std_stdvs = torch.cat(
                    [m.outcome_transform["tf2"].stdvs.reshape(1) for m in models], dim=0
                )  # (N,)
                std_means = torch.cat(
                    [m.outcome_transform["tf2"].means.reshape(1) for m in models], dim=0
                )  # (N,)

                def _noise(m, n):
                    lhd = m.likelihood
                    if isinstance(lhd, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
                        return lhd.noise_covar.noise.reshape(n)
                    return lhd.noise.reshape(1).expand(n)

                def _chol_alpha(mods, K_train_batch, n):
                    noise   = torch.stack([_noise(m, n) for m in mods])
                    y       = torch.stack([m.train_targets.reshape(n) for m in mods])
                    pm      = torch.stack([m.mean_module(m.train_inputs[0]) for m in mods])
                    K_noisy = K_train_batch + torch.diag_embed(noise)
                    L       = torch.linalg.cholesky(K_noisy)
                    alpha   = torch.cholesky_solve((y - pm).unsqueeze(-1), L)
                    return L, alpha

                # --- Build one batched group per (n_train, ard) bucket ---
                sc_groups = []
                for (n, ard), indices in sc_buckets.items():
                    mods   = [models[i] for i in indices]
                    raw_ls = torch.stack(
                        [m.covar_module.base_kernel.raw_lengthscale for m in mods], dim=0
                    )  # (G, 1, ard)
                    raw_os = torch.stack(
                        [m.covar_module.raw_outputscale for m in mods], dim=0
                    )  # (G,)
                    bc = copy.deepcopy(mods[0].covar_module)
                    bc.base_kernel.raw_lengthscale = torch.nn.Parameter(raw_ls.clone())
                    bc.raw_outputscale             = torch.nn.Parameter(raw_os.clone())
                    bc.requires_grad_(False)
                    bc.eval()
                    X_tr = torch.stack([m.train_inputs[0] for m in mods], dim=0)  # (G, n, ard)
                    K_tr = bc(X_tr, X_tr).to_dense()
                    L, alpha = _chol_alpha(mods, K_tr, n)
                    sc_groups.append({
                        "indices":     indices,
                        "n_train":     n,
                        "ard":         ard,
                        "batch_covar": bc,
                        "X_train":     X_tr,
                        "L":           L,
                        "alpha":       alpha,
                        "os":          bc.outputscale.detach(),  # (G,)
                    })

                # --- Build one batched group per n_train bucket (ck) ---
                # Since K(x1,x2)=1 always, K_star = ones for any test point.
                # Both the mean correction (K_star @ alpha) and the predictive variance
                # (1 - ||L^{-1} @ 1_n||^2) are independent of X, so precompute them once.
                ck_groups = []
                for (n,), indices in ck_buckets.items():
                    mods    = [models[i] for i in indices]
                    G       = len(mods)
                    K_tr    = torch.ones(G, n, n, dtype=ref.dtype, device=ref.device)
                    L, alpha = _chol_alpha(mods, K_tr, n)
                    # mean correction: sum over training points (G, 1)
                    mean_correction = alpha.sum(dim=-2)          # (G, 1)
                    # variance constant: 1 - ||L^{-1} @ 1_n||^2, shape (G, 1)
                    ones_n   = torch.ones(G, n, 1, dtype=ref.dtype, device=ref.device)
                    v_ck     = torch.linalg.solve_triangular(L, ones_n, upper=False)  # (G, n, 1)
                    var_const = (1.0 - (v_ck * v_ck).sum(dim=-2)).clamp(min=0)       # (G, 1)
                    ck_groups.append({
                        "indices":         indices,
                        "n_train":         n,
                        "mean_correction": mean_correction,
                        "var_const":       var_const,
                    })

            self._sc_groups  = sc_groups
            self._ck_groups  = ck_groups
            self._std_stdvs  = std_stdvs
            self._std_means  = std_means
            self._n_batched  = N
            self._batched_ready = True

            sc_summary = ", ".join(
                f"n={g['n_train']}/ard={g['ard']}:{len(g['indices'])}" for g in sc_groups
            )
            ck_summary = ", ".join(
                f"n={g['n_train']}:{len(g['indices'])}" for g in ck_groups
            )
            print(
                f"\t[MITIM: GP batching] setup: N={N} "
                f"sc=[{sc_summary}] ck=[{ck_summary}]",
                typeMsg="i",
            )

        except Exception as e:
            print(f"\t[MITIM: GP batching] setup failed ({e}) — sequential path will be used", typeMsg="w")
            self._batched_ready = False

    def _batched_posterior(self, X):
        """Vectorised GP posterior for all N models.

        Handles two kernel groups:
          - ScaleKernel group: one batched kernel call
          - MITIM_ConstantKernel group: K_star = ones (no kernel call)

        X: raw (pre-transform) test inputs, shape (*batch_dims, M, d_raw).
        Returns GPyTorchPosterior with MultitaskMultivariateNormal distribution.
        """
        from gpytorch.distributions import MultitaskMultivariateNormal
        from linear_operator.operators import DiagLinearOperator, BlockDiagLinearOperator
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        N = self._n_batched

        # Transform inputs per model. prepareToGenerateCommons sets flag_to_store=True so
        # model[0] computes and caches the powerstate in parameters_combined["powerstate"];
        # models 1..N-1 reuse that cache.
        # cold_startCommons() is deferred until after the tf1_factors loop below so that
        # output_transform_portals can also reuse the cached powerstate via
        # constructEvaluationProfiles (otherwise it would rebuild the powerstate 63 times).
        self.prepareToGenerateCommons()
        Xtr_per_model = [m.transform_inputs(X) for m in self.models]

        orig_shape = X.shape[:-1]   # (*batch_dims, M)
        M_flat = X[..., 0].numel()  # total test points, flattened

        # --- Process each ScaleKernel group (one batched kernel call per group) ---
        mean_by_model = [None] * N
        var_by_model  = [None] * N

        _ref_dtype = None
        for grp in self._sc_groups:
            indices = grp["indices"]
            d       = grp["ard"]
            Xtr_g   = torch.stack(
                [Xtr_per_model[i].reshape(M_flat, d) for i in indices], dim=0
            )  # (G, M_flat, d)
            K_star  = grp["batch_covar"](Xtr_g, grp["X_train"]).to_dense()  # (G, M_flat, n)
            prior   = torch.stack(
                [self.models[i].mean_module(Xtr_per_model[i].reshape(M_flat, d))
                 for i in indices]
            )  # (G, M_flat)
            mean_g  = prior + (K_star @ grp["alpha"]).squeeze(-1)
            v_g     = torch.linalg.solve_triangular(
                grp["L"], K_star.transpose(-1, -2), upper=False
            )  # (G, n, M_flat)
            var_g   = (grp["os"].unsqueeze(-1) - (v_g * v_g).sum(dim=-2)).clamp(min=0)
            for j, i in enumerate(indices):
                mean_by_model[i] = mean_g[j]
                var_by_model[i]  = var_g[j]
            _ref_dtype = Xtr_g.dtype

        # --- Process each MITIM_ConstantKernel group ---
        # K_star = ones for any test X, so mean correction and variance are X-independent
        # and were precomputed in setup_batched_inference.
        for grp in self._ck_groups:
            indices = grp["indices"]
            G       = len(indices)
            prior   = torch.stack(
                [self.models[i].mean_module(
                     Xtr_per_model[i].reshape(M_flat, Xtr_per_model[i].shape[-1])
                 ) for i in indices]
            )  # (G, M_flat)
            mean_g = prior + grp["mean_correction"].to(X.device)     # (G, M_flat)
            var_g  = grp["var_const"].to(X.device).expand(G, M_flat) # (G, M_flat)
            for j, i in enumerate(indices):
                mean_by_model[i] = mean_g[j]
                var_by_model[i]  = var_g[j]

        pred_mean = torch.stack(mean_by_model).reshape(N, *orig_shape)
        pred_var  = torch.stack(var_by_model).reshape(N, *orig_shape)

        # Un-normalise tf2 (Standardize)
        sh       = [N] + [1] * (pred_mean.dim() - 1)
        mean_tf2 = pred_mean * self._std_stdvs.view(*sh) + self._std_means.view(*sh)
        var_tf2  = pred_var  * self._std_stdvs.view(*sh) ** 2

        # Un-normalise tf1 (physics output scaling) per model.
        # The loop calls transformationOutputs once per model; constructEvaluationProfiles
        # inside it reuses parameters_combined["powerstate"] (still alive from transform_inputs
        # above) so the powerstate is built only once across all 63 models.
        factors_list = []
        for i, m in enumerate(self.models):
            tf1 = m.outcome_transform["tf1"]
            if not (hasattr(tf1, "_cached_factor_X") and tf1._cached_factor_X is X):
                tf1._cached_factor = tf1.surrogate_parameters["transformationOutputs"](
                    X, tf1.surrogate_parameters, tf1.output
                ).to(X.device)
                tf1._cached_factor_X = X
            factors_list.append(tf1._cached_factor.squeeze(-1))  # (*orig_shape,)
        self.cold_startCommons()  # deferred: safe now that both transform_inputs and tf1 factors are done

        factors  = torch.stack(factors_list)                              # (N, *orig_shape)
        mean_fin = (mean_tf2 * factors).reshape(N, M_flat)               # (N, M_flat)
        var_fin  = (var_tf2 * factors ** 2).clamp(min=0).reshape(N, M_flat)  # (N, M_flat)

        # Build MultitaskMultivariateNormal directly from batched tensors — avoids constructing
        # N individual MultivariateNormal objects (O(N*M) Python overhead).
        # interleaved=False: BlockDiagLinearOperator block i covers all M points for task i.
        mean_out    = mean_fin.reshape(N, *orig_shape).permute(*range(1, len(orig_shape) + 1), 0)
        block_covar = BlockDiagLinearOperator(DiagLinearOperator(var_fin))
        mtmvn = MultitaskMultivariateNormal(
            mean_out, block_covar, validate_args=False, interleaved=False
        )

        return GPyTorchPosterior(distribution=mtmvn)

    def posterior(
        self,
        X,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs,
    ):
        # Fast batched path: one kernel call for all N models
        if (
            getattr(self, "_batched_ready", False)
            and output_indices is None
            and not observation_noise
            and posterior_transform is None
        ):
            try:
                return self._batched_posterior(X)
            except Exception as e:
                print(f"\t[MITIM: GP batching] posterior failed ({e}) — falling back", typeMsg="w")

        # Sequential fallback
        self.prepareToGenerateCommons()
        posterior = super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        self.cold_startCommons()

        return posterior


# ----------------------------------------------------------------------------------------------------------------------------
# I need my own transformation based on physics
# ----------------------------------------------------------------------------------------------------------------------------


class Transformation_Inputs(
    botorch.models.transforms.input.ReversibleInputTransform, torch.nn.Module
):
    def __init__(
        self,
        output,
        surrogate_parameters,
        surrogate_transformation_variables,
        indices=None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        if (indices is not None) and (len(indices) > 0):
            indices = torch.tensor(indices, dtype=torch.long)
        self.register_buffer("indices", indices)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

        # Custom parameters
        self.output = output
        self.surrogate_parameters = surrogate_parameters
        self.surrogate_transformation_variables = surrogate_transformation_variables
        self.flag_to_store = False
        self.flag_to_evaluate = True

    @botorch.models.transforms.utils.subset_transform
    def _transform(self, X):
        if (self.output is not None) and (self.flag_to_evaluate):
            Xtr, parameters_combined = self.surrogate_parameters["transformationInputs"](
                X,
                self.output,
                self.surrogate_parameters,
                self.surrogate_transformation_variables,
            )

            # Store the expensive parameters (not for training, or to call outside of ModelList)
            if self.flag_to_store:
                self.surrogate_parameters["parameters_combined"] = parameters_combined

        else:
            Xtr = X

        return Xtr

    @botorch.models.transforms.utils.subset_transform
    def _untransform(self, X):
        raise NotImplementedError("[MITIM] This situation has not been implemented yet")


# ----------------------------------------------------------------------------------------------------------------------------
# I need my own outcome transformation based on physics and that takes "X" as well
# ----------------------------------------------------------------------------------------------------------------------------


# Copy standardize but modify in untransform the "std" which is my factor!
class Transformation_Outcomes(botorch.models.transforms.outcome.Standardize):
    def __init__(self, m, output, surrogate_parameters):
        super().__init__(m)

        self.output = output
        self.surrogate_parameters = surrogate_parameters
        self.flag_to_evaluate = True

    def forward(self, X, Y, Yvar):
        if (self.output is not None) and (self.flag_to_evaluate):
            factor = self.surrogate_parameters["transformationOutputs"](
                X, self.surrogate_parameters, self.output
            ).to(X.device)
        else:
            factor = Y.mean(dim=-2, keepdim=True).to(Y.device) * 0.0 + 1.0

        self.stdvs = factor
        self.means = self.stdvs * 0.0
        self._stdvs_sq = self.stdvs.pow(2)

        # When calling the forward method of Standardize, do not recalculate mean and stdvs (never be on training)
        self._is_trained = torch.tensor(True)
        self.training = False
        # ----------------------------------------

        return super().forward(Y, Yvar)

    def untransform_posterior(self, X, posterior):
        if (self.output is not None) and (self.flag_to_evaluate):
            # Cache factor by X tensor identity: avoids recomputing expensive physics
            # transform when the same X object is passed multiple times (e.g. inner
            # optimizer iterations, MC acquisition sampling).
            if not (hasattr(self, "_cached_factor_X") and self._cached_factor_X is X):
                self._cached_factor = self.surrogate_parameters["transformationOutputs"](
                    X, self.surrogate_parameters, self.output
                ).to(X.device)
                self._cached_factor_X = X
            factor = self._cached_factor

            self.stdvs = factor
            self.means = self.stdvs * 0.0
            self._stdvs_sq = self.stdvs.pow(2)
            return super().untransform_posterior(posterior)

        else:
            return posterior

    def untransform(self, Y, Yvar):
        raise NotImplementedError("[MITIM] This situation has not been implemented yet")


# Because I need it to take X too (for physics only, which is always the first tf)
class ChainedOutcomeTransform(
    botorch.models.transforms.outcome.ChainedOutcomeTransform
):
    def __init__(self, **transforms):
        super().__init__(**transforms)

    def forward(self, X, Y, Yvar):
        for i, tf in enumerate(self.values()):
            Y, Yvar = (
                tf.forward(X, Y, Yvar) if i == 0 else tf.forward(Y, Yvar)
            )  # Only physics transformation (tf1) takes X

        return Y, Yvar

    def untransform_posterior(self, X, posterior):
        for i, tf in enumerate(reversed(self.values())):
            posterior = (
                tf.untransform_posterior(X, posterior)
                if i == 1
                else tf.untransform_posterior(posterior)
            )  # Only physics transformation (tf1) takes X

        return posterior

    def untransform(self, X, Y, Yvar):
        raise NotImplementedError("[MITIM] This situation has not been implemented yet")


# ----------------------------------------------------------------------------------------------------------------------------
# Mean acquisition function in botorch doesn't allow objectives because it's analytic
# ----------------------------------------------------------------------------------------------------------------------------

class PosteriorMean(botorch.acquisition.monte_carlo.MCAcquisitionFunction):
    def __init__(
        self,
        model,
        sampler=None,
        objective=None,
        posterior_transform=None,
        X_pending=None,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

    @botorch.utils.transforms.t_batch_mode_transform()  # This ensures the t-batch dimension. Example: X of (q=5,dim=1) will be (batch=1,q=5,dim=1)
    def forward(self, X):
        """
        Notes:
                - X in the form of [batch,cold_starts,q,dim]
                - The output of the acquisition must be something to MAXIMIZE. That's something that should be given in objective
        """

        # Posterior distribution
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )

        # mean as [batch1...N,q,dimY]
        mean = posterior.mean

        # objective [batch1...N,q] -> This assumes the nonlinearity of the objective is not significant, so obj(mean) = mean(obj)
        obj = self.objective(mean)

        # max over q
        acq = obj.max(dim=1)[0]

        return acq

# ----------------------------------------------------------------------------------------------------------------------------
# Custom kernels
# ----------------------------------------------------------------------------------------------------------------------------

class MITIM_NNKernel(gpytorch.kernels.Kernel):
    has_lengthscale, is_stationary = True, False

    def __init__(self, tau_prior=None, tau_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(
            name="raw_tau",
            parameter=torch.nn.Parameter(
                torch.zeros(*self.batch_shape, 1, self.ard_num_dims)
            ),
        )

        # set the parameter constraint to be [0,1], when nothing is specified
        if tau_constraint is None:
            tau_constraint = gpytorch.constraints.constraints.Interval(-0.5, 1.5)

        # register the constraint
        self.register_constraint("raw_tau", tau_constraint)

        # set the parameter prior, see
        if tau_prior is not None:
            self.register_prior(
                "length_prior",
                tau_prior,
                lambda m: m.tau,
                lambda m, v: m._set_length(v),
            )

    # now set up the 'individual_models' paramter
    @property
    def tau(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_tau_constraint.transform(self.raw_tau)

    @tau.setter
    def tau(self, value):
        return self._set_tau(value)

    def _set_tau(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_tau)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_tau_constraint.inverse_transform(value))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        **params,
    ) -> torch.Tensor:
        # print(self.lengthscale,self.tau,x1.shape,x2.shape)

        x1o = x1.clone()
        x2o = x2.clone()
        if len(x1o.shape) > 2:
            x1 = x1o[:, 0, :]  # x1o.view(x1o.shape[0]*x1o.shape[1],1)
        if len(x2o.shape) > 2:
            x2 = x2o[0, :, :]  # .view(x2o.shape[0]*x2o.shape[1],1)

        x1 = torch.cat((torch.ones(x1.shape[0]).unsqueeze(1), x1.sub(self.tau)), dim=-1)

        x2 = torch.cat((torch.ones(x2.shape[0]).unsqueeze(1), x2.sub(self.tau)), dim=-1)
        S = torch.cat((torch.ones(1).unsqueeze(0), self.lengthscale.pow(-2)), dim=-1)[
            0
        ].diag()

        prod_x1x2 = torch.matmul(x1.matmul(S), x2.transpose(-2, -1)).mul(2)

        aux1 = x1.matmul(S).matmul(x1.transpose(-2, -1)).diag().mul(2).add(1)
        aux2 = x2.matmul(S).matmul(x2.transpose(-2, -1)).diag().mul(2).add(1)

        denom = aux1.unsqueeze(1).matmul(aux2.unsqueeze(0))

        pi = torch.acos(torch.zeros(1)).item() * 2
        val = torch.arcsin(prod_x1x2 * denom.pow(-0.5)).mul(2).div(pi)

        if params["diag"]:
            val = torch.diag(val)

        if len(x1o.shape) > 2:
            val = val.unsqueeze(1)

        return val


class MITIM_ConstantKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        **params,
    ) -> torch.Tensor:
        dist = gpytorch.kernels.kernel.Distance()

        x1_eq_x2 = torch.equal(x1, x2)

        if "diag" in params and params["diag"]:
            if x1_eq_x2:
                res = (
                    torch.zeros(
                        *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
                    )
                    * 0.0
                )
            else:
                res = torch.norm(x1 - x2, p=2, dim=-1).pow(2) * 0.0
        else:
            res = dist._sq_dist(x1, x2, x1_eq_x2=x1_eq_x2, postprocess=False) * 0.0

        val = res.div_(-2).exp_()

        return val


# ----------------------------------------------------------------------------------------------------------------------------
# Custom means
# ----------------------------------------------------------------------------------------------------------------------------

# mitim application: If a variable is a gradient, do linear, if not, do just bias
class MITIM_LinearMeanGradients(gpytorch.means.mean.Mean):
    def __init__(
        self,
        batch_shape=torch.Size(),
        variables=None,
        output=None,
        only_diffusive=True,
        **kwargs
        ):
        super().__init__()

        # Indeces of variables that are gradient, so subject to CG behavior
        grad_vector = []
        if variables is not None:

            if not only_diffusive:
                for i, variable in enumerate(variables):
                    if ("aL" in variable) or ("dw" in variable):
                        grad_vector.append(i)
            else:

                mapping = {
                    'Qe_': 'aLte',
                    'Qi_': 'aLti',
                    'Ge_': 'aLne',
                    'GZ_': 'aLnZ',
                    'Mt_': 'aLw0_n',
                    'Qie': None  # Referring to energy exchange
                }

                for i, variable in enumerate(variables):
                    if (mapping[output[:3]] is not None) and (mapping[output[:3]] == variable):
                        grad_vector.append(i)

        self.indeces_grad = tuple(grad_vector)
        # ----------------------------------------------------------------

        self.register_parameter(name="raw_weights_lin",parameter=torch.nn.Parameter(torch.randn(*batch_shape, len(self.indeces_grad), 1)),)
        self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))

        # set the parameter constraint to be [0,1], when nothing is specified
        diffusion_constraint = gpytorch.constraints.constraints.Positive()

        # positive diffusion coefficient
        if only_diffusive:
            self.register_constraint("raw_weights_lin", diffusion_constraint)

    def forward(self, x):
        weights_lin = self.weights_lin
        res = x[..., self.indeces_grad].matmul(weights_lin).squeeze(-1) + self.bias
        return res
    
    # This follows the exact same pattern as in gpytorch's constant_mean.py

    @property
    def weights_lin(self):
        return self._weights_lin_param(self)

    @weights_lin.setter
    def weights_lin(self, value):
        self._weights_lin_closure(self, value)

    def _weights_lin_param(self, m):
        if hasattr(m, "raw_weights_lin_constraint"):
            return m.raw_weights_lin_constraint.transform(m.raw_weights_lin)
        return m.raw_weights_lin

    def _weights_lin_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_weights_lin)

        if hasattr(m, "raw_weights_lin_constraint"):
            m.initialize(raw_weights_lin=m.raw_weights_lin_constraint.inverse_transform(value))
        else:
            m.initialize(raw_weights_lin=value)
