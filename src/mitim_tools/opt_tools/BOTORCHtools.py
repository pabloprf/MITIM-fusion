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
from mitim_tools.misc_tools.LOGtools import printMsg as print

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
        surrogateOptions={},
        variables=None,
        train_X_added=torch.Tensor([]),
        train_Y_added=torch.Tensor([]),
        train_Yvar_added=torch.Tensor([]),
    ):
        """
        _added refers to already-transformed variables that are added from table
        """

        TypeMean = surrogateOptions.get("TypeMean", 0)
        TypeKernel = surrogateOptions.get("TypeKernel", 0)
        FixedNoise = surrogateOptions.get("FixedNoise", False)
        ConstrainNoise = surrogateOptions.get("ConstrainNoise", -1e-4)
        learn_additional_noise = surrogateOptions.get("ExtraNoise", False)
        print("\t\t* Surrogate model options:")
        print(
            f"\t\t\t- FixedNoise: {FixedNoise} (extra noise: {learn_additional_noise}), TypeMean: {TypeMean}, TypeKernel: {TypeKernel}, ConstrainNoise: {ConstrainNoise:.1e}"
        )

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
            self.mean_module = PRF_LinearMeanGradients(
                batch_shape=self._aug_batch_shape, variables=variables
            )
        elif TypeMean == 3:
            self.mean_module = PRF_CriticalGradient(
                batch_shape=self._aug_batch_shape, variables=variables
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
        lengthscale_constraint = (
            None  # gpytorch.constraints.constraints.GreaterThan(0.05)
        )

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
            self.covar_module = PRF_ConstantKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            )
        elif TypeKernel == 3:
            self.covar_module = gpytorch.kernels.scale_kernel.ScaleKernel(
                base_kernel=PRF_NNKernel(
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
        self.eval()
        Xtr = self.transform_inputs(X)
        with botorch.models.utils.gpt_posterior_settings():
            # insert a dimension for the output dimension
            if self._num_outputs > 1:
                Xtr, output_dim_idx = botorch.models.utils.add_output_dim(
                    X=Xtr, original_batch_shape=self._input_batch_shape
                )
            mvn = self(Xtr)
            if observation_noise is not False:
                if torch.is_tensor(observation_noise):
                    #TODO: Validate noise shape
                    # make observation_noise `batch_shape x q x n`
                    if self.num_outputs > 1:
                        obs_noise = observation_noise.transpose(-1, -2)
                    else:
                        obs_noise = observation_noise.squeeze(-1)
                    mvn = self.likelihood(mvn, Xtr, noise=obs_noise)
                elif isinstance(
                    self.likelihood,
                    gpytorch.likelihoods.gaussian_likelihood.FixedNoiseGaussianLikelihood,
                ):
                    # Use the mean of the previous noise values (TODO: be smarter here).
                    noise = self.likelihood.noise.mean().expand(X.shape[:-1])
                    mvn = self.likelihood(mvn, Xtr, noise=noise)
                else:
                    mvn = self.likelihood(mvn, Xtr)
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
                mvn = gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(
                    mvns=mvns
                )

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
        if (
            "parameters_combined"
            in self.models[0].input_transform.tf1.surrogate_parameters
        ):
            del self.models[0].input_transform.tf1.surrogate_parameters[
                "parameters_combined"
            ]

    def restartCommons(self):
        self.models[0].input_transform.tf1.flag_to_store = False
        if (
            "parameters_combined"
            in self.models[0].input_transform.tf1.surrogate_parameters
        ):
            del self.models[0].input_transform.tf1.surrogate_parameters[
                "parameters_combined"
            ]

    def transform_inputs(self, X):
        self.prepareToGenerateCommons()
        X_tr = super().transform_inputs(X)
        self.restartCommons()

        return X_tr

    def posterior(
        self,
        X,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs,
    ):
        self.prepareToGenerateCommons()
        posterior = super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        self.restartCommons()

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
            Xtr, parameters_combined = self.surrogate_parameters[
                "transformationInputs"
            ](
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
            )
        else:
            factor = Y.mean(dim=-2, keepdim=True) * 0.0 + 1.0

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
            factor = self.surrogate_parameters["transformationOutputs"](
                X, self.surrogate_parameters, self.output
            )

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
                - X in the form of [batch,restarts,q,dim]
                - The output of the acquisition must be something to MAXIMIZE. That's something that should be given in objective
        """

        # Posterior distribution
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )

        # mean as [batch1...N,q,dimY]
        mean = posterior.mean

        # objective [batch1...N,q]
        obj = self.objective(mean)

        # max over q
        acq = obj.max(dim=1)[0]

        return acq

class PosteriorMeanMC(botorch.acquisition.monte_carlo.MCAcquisitionFunction):
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
                - X in the form of [batch,restarts,q,dim]
                - The output of the acquisition must be something to MAXIMIZE. That's something that should be given in objective
        """

        # Posterior distribution
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )

        # samples as [samples,batch1...N,q,dimY]
        samples = self.get_posterior_samples(posterior)

        # objective [samples,batch1...N,q]
        obj = self.objective(samples=samples)

        # mean over samples [batch1...N,q]
        obj_mean = obj.mean(axis=0)
        
        # max over q
        acq = obj_mean.max(axis=-1)[0]

        return acq


# ----------------------------------------------------------------------------------------------------------------------------
# My own IC generator that uses previous points too
# ----------------------------------------------------------------------------------------------------------------------------


def ic_generator_wrapper(batch_initial_conditions):
    def ic_generator(acq_function, bounds, q, num_restarts, raw_samples, **kwargs):
        if q > 1:
            raise NotImplementedError(
                "[MITIM] This situation has not been implemented yet"
            )

        # Points already provided
        provided_points = batch_initial_conditions.unsqueeze(1)

        # Only generate the rest
        num_restarts_new = num_restarts - provided_points.shape[0]

        if num_restarts_new < 1:
            print(
                f"\t- More or same points provided than num_restarts ({provided_points.shape[0]} vs {num_restarts}), clipping...",
                typeMsg="w",
            )
            return provided_points[provided_points.shape[0] - num_restarts :, ...]
        else:
            new_points = botorch.optim.initializers.gen_batch_initial_conditions(
                acq_function, bounds, q, num_restarts_new, raw_samples, **kwargs
            )

            return torch.cat([provided_points, new_points], dim=0)

    return ic_generator


# ----------------------------------------------------------------------------------------------------------------------------
# Custom kernels
# ----------------------------------------------------------------------------------------------------------------------------


class PRF_NNKernel(gpytorch.kernels.Kernel):
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


class PRF_ConstantKernel(gpytorch.kernels.Kernel):
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
class PRF_LinearMeanGradients(gpytorch.means.mean.Mean):
    def __init__(self, batch_shape=torch.Size(), variables=None, **kwargs):
        super().__init__()

        # Indeces of variables that are gradient, so subject to CG behavior
        grad_vector = []
        if variables is not None:
            for i, variable in enumerate(variables):
                if ("aL" in variable) or ("dw" in variable):
                    grad_vector.append(i)
        self.indeces_grad = tuple(grad_vector)
        # ----------------------------------------------------------------

        self.register_parameter(
            name="weights_lin",
            parameter=torch.nn.Parameter(
                torch.randn(*batch_shape, len(self.indeces_grad), 1)
            ),
        )
        self.register_parameter(
            name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )

    def forward(self, x):
        res = x[..., self.indeces_grad].matmul(self.weights_lin).squeeze(-1) + self.bias
        return res


class PRF_CriticalGradient(gpytorch.means.mean.Mean):
    def __init__(self, batch_shape=torch.Size(), variables=None, **kwargs):
        super().__init__()

        # Indeces of variables that are gradient, so subject to CG behavior
        grad_vector = []
        if variables is not None:
            for i, variable in enumerate(variables):
                if ("aL" in variable) or ("dw" in variable):
                    grad_vector.append(i)
        self.indeces_grad = tuple(grad_vector)
        # ----------------------------------------------------------------

        self.register_parameter(
            name="weights_lin",
            parameter=torch.nn.Parameter(
                torch.randn(*batch_shape, len(self.indeces_grad), 1)
            ),
        )
        self.register_parameter(
            name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
        )

        self.NNfunc = (
            lambda x: x * (1 + torch.erf(x / 0.01)) / 2.0
        )  # https://paperswithcode.com/method/gelu

        self.register_parameter(
            name="relu_lin",
            parameter=torch.nn.Parameter(
                torch.randn(*batch_shape, len(self.indeces_grad), 1)
            ),
        )
        self.register_constraint(
            "relu_lin", gpytorch.constraints.constraints.Interval(0, 1)
        )

    def forward(self, x):
        res = (
            self.NNfunc(x[..., self.indeces_grad] - self.relu_lin.transpose(0, 1))
            .matmul(self.weights_lin)
            .squeeze(-1)
            + self.bias
        )
        return res
