import copy
from mitim_modules.powertorch.physics_models import transport_analytic
import torch
import shutil
import random
from functools import partial
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_tools.misc_tools import IOtools
from mitim_modules.powertorch import STATEtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
*********************************************************************************************************************
	Initialization
*********************************************************************************************************************
"""


_SIMPLE_RELAX_DEFAULTS = {
    "tol": None,
    "relax": 0.2,          # Defines relationship between flux and gradient
    "dx_max": 0.2,         # Max relative step in gradient (20% of a/Lx per iter)
    "dx_max_abs": None,    # Max absolute step in gradient
    "dx_min_abs": 0.1,     # Min absolute step in gradient
    "relax_dyn": False,
    "print_each": 1,
}

# Keys that may appear in user-provided `initialization_params` entries. Anything else is rejected
# so typos surface loudly instead of being silently ignored.
_SIMPLE_RELAX_ALLOWED_KEYS = {
    "relax", "dx_max", "dx_max_abs", "dx_min_abs",
    "relax_dyn", "relax_dyn_decrease", "relax_dyn_num",
    "tol", "tol_rel", "print_each",
    "perturbation_base",
}


def _normalize_initialization_params(raw):
    """Normalize the namelist `initialization_params` field into a list of dicts.

    Accepts three shapes:
        - None (or missing) → [{}]   (single trajectory, all defaults)
        - a bare dict        → [dict] (shorthand for one trajectory)
        - a list of dicts    → returned as-is

    Unknown keys raise a ValueError so typos don't silently vanish.
    """
    if raw is None:
        return [{}]
    if isinstance(raw, dict):
        raw_list = [raw]
    else:
        raw_list = list(raw)
        if len(raw_list) == 0:
            return [{}]

    normalized = []
    for idx, entry in enumerate(raw_list):
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            raise ValueError(
                f"initialization_params[{idx}] must be a dict, got {type(entry).__name__}"
            )
        entry = dict(entry)
        if "maxiter" in entry:
            print(
                f"\t- initialization_params[{idx}] contains a 'maxiter' key; the count of "
                "simple-relax points is driven by initial_training, so this value is ignored.",
                typeMsg="w",
            )
            entry.pop("maxiter")
        bad = set(entry) - _SIMPLE_RELAX_ALLOWED_KEYS
        if bad:
            raise ValueError(
                f"initialization_params[{idx}] has unknown key(s): {sorted(bad)}. "
                f"Allowed: {sorted(_SIMPLE_RELAX_ALLOWED_KEYS)}"
            )
        normalized.append(entry)
    return normalized


def initialization_simple_relax(self):
    # ------------------------------------------------------------------------------------
    # Perform flux matching using powerstate.
    #
    # A user may request N parallel simple-relax trajectories through the namelist field
    # `optimization_options.initialization_options.initialization_params` (a list of
    # per-trajectory solver-option dicts). Each trajectory walks toward flux matching with
    # its own (relax, dx_max, …) and contributes `initial_training / N` points to the
    # initial training set.
    #
    # Implementation — one batched flux_match call, not a Python for-loop over trajectories:
    #
    #   1. Deep-copy the caller's powerstate once and tile its plasma tensors to N batches
    #      via `_repeat_tensors(batch_size=n_traj)`. Every `plasma[*]` tensor now carries an
    #      explicit leading batch axis of length n_traj.
    #   2. Build per-trajectory tensor versions of the simple-relax solver knobs
    #      (relax, dx_max, dx_max_abs, dx_min_abs) of shape (n_traj, 1) — `_sr_step` already
    #      accepts per-batch tensors via its torch.where clamps, so the relax math iterates
    #      in lockstep over the N trajectories for free.
    #   3. Call `powerstate.flux_match(algorithm="simple_relax", solver_options=...)` exactly
    #      once. Inside STATEtools.flux_match the evaluator constructs `x0` of shape
    #      (n_traj, dvs) straight from the batched plasma and calls `self.calculate(X, ...)`
    #      on the N-wide batch. `power_transport.evaluate` dispatches to `_evaluate_batched`,
    #      which fans every (plasma, rho) work unit through `mitim_simulation.run_over_plasmas`
    #      — the same FARMINGtools pipeline TGLFmulti_plasma_workflow.py exercises. One
    #      parallel submission per relax step, so N TGLF calls fire concurrently.
    #   4. Read the full `FluxMatch_Xopt_batches` tensor (shape (maxiter, n_traj, dvs))
    #      instead of the scalar-best column `FluxMatch_Xopt`, and reshape to step-major
    #      order for the downstream Execution/Evaluation.{i} copy loop.
    #
    # Folder layout under this scheme: each SR step writes ONE
    # `MainFolder/portals_sr_ev_{s}/transport_simulation_folder` directory whose children
    # are per-plasma sub-folders `plasma_{t}` (one per trajectory) holding the single-plasma
    # `fluxes_{turb,neoc}.json` pair plus an `input.gacode`. The copy loop walks (step,
    # trajectory) pairs and moves each `plasma_{t}` sub-folder into the matching
    # `Execution/Evaluation.{s*n_traj + t}/transport_simulation_folder`, matching the
    # step-major order of the flattened Xopt.
    #
    # Multi-trajectory `base_X` duplicate guard: `simple_relaxation` always records
    # `x_initial` (= base_X) as the first entry of `x_history`. For n_traj>1 every
    # trajectory shares the same base_X row, which would land as N identical training
    # rows and crash `optimization_data.find_point(base_X)` downstream via `.item()` on a
    # multi-row Series. We run one extra relax iteration (`effective_maxiter = steps_per_traj + 1`)
    # and drop `Xopt_batches[0]` when skip_initial is True. Step 0 lands in a throwaway
    # sub-folder `portals_sr_base_step0` whose name is deliberately *not* matched by the
    # `portals_sr_ev_*` glob used by the copy loop and PORTALSinitializer. For
    # single-trajectory runs no duplication is possible and the historical behaviour
    # (include base_X, no throwaway folder) is preserved.
    # ------------------------------------------------------------------------------------

    traj_params = _normalize_initialization_params(
        self.optimization_options["initialization_options"].get("initialization_params")
    )
    n_traj = len(traj_params)

    if self.Originalinitial_training % n_traj != 0:
        raise ValueError(
            f"initial_training ({self.Originalinitial_training}) must be divisible by the "
            f"number of initialization_params trajectories ({n_traj}). Adjust one of the two."
        )
    steps_per_traj = self.Originalinitial_training // n_traj

    folderExecution = IOtools.expandPath(self.folderExecution, ensurePathValid=True)
    MainFolder = folderExecution / "Initialization" / "initialization_simple_relax"
    MainFolder.mkdir(parents=True, exist_ok=True)

    if self.seed is not None and self.seed != 0:
        random.seed(self.seed)
        addon_relax = random.uniform(-0.03, 0.03)
    else:
        addon_relax = 0.0

    skip_initial = n_traj > 1
    effective_maxiter = steps_per_traj + (1 if skip_initial else 0)

    # Folder_namer hook: each relax step gets one folder. When we are running with
    # skip_initial=True, step 0 is the throwaway base_X evaluation — route it to a
    # differently-prefixed folder so the `portals_sr_ev_*` glob used below (and by
    # PORTALSinitializer) does not see it.
    if skip_initial:
        def folder_namer(step):
            return (
                "portals_sr_base_step0" if step == 0
                else f"portals_sr_ev_{step - 1}"
            )
    else:
        def folder_namer(step):
            return f"portals_sr_ev_{step}"

    # Build per-trajectory tensor versions of each solver-option knob.
    def _per_batch_tensor(key, scalar_default):
        vals = []
        any_value = False
        for tp in traj_params:
            v = tp.get(key, scalar_default)
            if v is None:
                vals.append(None)
            else:
                vals.append(float(v))
                any_value = True
        if not any_value:
            return None
        if any(v is None for v in vals):
            raise ValueError(
                f"initialization_params[*]['{key}'] mixes None and numeric entries; either "
                "set a numeric value for every trajectory or omit the key from all of them."
            )
        return torch.tensor(vals).to(self.dfT).view(n_traj, 1)

    relax_tensor       = _per_batch_tensor('relax',       _SIMPLE_RELAX_DEFAULTS['relax'])
    if relax_tensor is not None:
        relax_tensor = relax_tensor + addon_relax
    dx_max_tensor      = _per_batch_tensor('dx_max',      _SIMPLE_RELAX_DEFAULTS['dx_max'])
    dx_max_abs_tensor  = _per_batch_tensor('dx_max_abs',  _SIMPLE_RELAX_DEFAULTS['dx_max_abs'])
    dx_min_abs_tensor  = _per_batch_tensor('dx_min_abs',  _SIMPLE_RELAX_DEFAULTS['dx_min_abs'])

    # Scalar knobs inherit from the first trajectory's overrides (they must agree across
    # trajectories because simple_relaxation reads them as scalars).
    scalar_defaults = dict(_SIMPLE_RELAX_DEFAULTS)
    scalar_defaults.update(traj_params[0])
    scalar_defaults.pop('relax', None)
    scalar_defaults.pop('dx_max', None)
    scalar_defaults.pop('dx_max_abs', None)
    scalar_defaults.pop('dx_min_abs', None)
    scalar_defaults.pop('perturbation_base', None)

    solver_options = {
        'tol':             scalar_defaults.get('tol', None),
        'maxiter':         effective_maxiter,
        'relax':           relax_tensor,
        'dx_max':          dx_max_tensor,
        'dx_max_abs':      dx_max_abs_tensor,
        'dx_min_abs':      dx_min_abs_tensor,
        'relax_dyn':       scalar_defaults.get('relax_dyn', False),
        'print_each':      scalar_defaults.get('print_each', 1),
        'folder':          MainFolder,
        'folder_namer':    folder_namer,
        'namingConvention': 'portals_sr_ev_batched',
    }

    # Replicate the caller's powerstate to an N-batched view. Each trajectory's starting
    # point is base_X * (1 + perturbation_base), where perturbation_base defaults to 0
    # (exact base) and can be set per-trajectory to create diversity (e.g. 0.1 = gradients
    # increased by 10%).
    traj_state = copy.deepcopy(self.optimization_object.powerstate)
    traj_state._repeat_tensors(batch_size=n_traj)

    base_X_row = torch.from_numpy(
        self.optimization_options["problem_options"]["dvs_base"]
    ).to(self.dfT)
    base_X_batched = base_X_row.unsqueeze(0).repeat(n_traj, 1)

    perturbations = [float(tp.get('perturbation_base', 0.0)) for tp in traj_params]
    for t in range(n_traj):
        if perturbations[t] != 0.0:
            base_X_batched[t, :] = base_X_row * (1.0 + perturbations[t])

    traj_state.modify(base_X_batched)

    # ---------------------------------------------------------------------------
    # Print SR information block before starting
    # ---------------------------------------------------------------------------
    n_radii = traj_state.plasma["rho"].shape[1] - 1
    n_channels = len(traj_state.predicted_channels)

    print("\n" + "=" * 80)
    print("  PORTALS Simple-Relax Initialization")
    print("=" * 80)
    print(f"  Trajectories:         {n_traj}")
    print(f"  Steps per trajectory: {steps_per_traj}")
    print(f"  Total training pts:   {self.Originalinitial_training}")
    print(f"  Effective maxiter:    {effective_maxiter} (skip_initial={skip_initial})")
    print(f"  Radii (per plasma):   {n_radii}  ({', '.join(traj_state.predicted_channels)})")
    print(f"  Seed addon_relax:     {addon_relax:+.4f}")
    print(f"  Base X (dvs_base):    [{', '.join(f'{v:.3f}' for v in base_X_row.tolist())}]")
    print(f"  ----- Per-step parallelism: {n_traj} trajectories x {n_radii} radii "
          f"= {n_traj * n_radii} concurrent transport calls (TGLF + NEO)")
    for t, tp in enumerate(traj_params):
        pert = perturbations[t]
        relax_val = relax_tensor[t].item() if relax_tensor is not None else _SIMPLE_RELAX_DEFAULTS['relax'] + addon_relax
        dx_max_val = dx_max_tensor[t].item() if dx_max_tensor is not None else _SIMPLE_RELAX_DEFAULTS['dx_max']
        dx_min_val = dx_min_abs_tensor[t].item() if dx_min_abs_tensor is not None else _SIMPLE_RELAX_DEFAULTS['dx_min_abs']
        print(f"  Trajectory {t}: relax={relax_val:.4f}  dx_max={dx_max_val:.3f}  "
              f"dx_min_abs={dx_min_val:.3f}  perturbation_base={pert:+.2%}")
    print("=" * 80 + "\n")

    traj_state.flux_match(
        algorithm="simple_relax",
        solver_options=solver_options,
    )

    # Full batched trajectory: shape (effective_maxiter, n_traj, dvs). For skip_initial
    # we drop the very first step (the shared base_X row) so the downstream
    # Execution/Evaluation.{i} layout never carries duplicate training rows.
    Xopt_batches = traj_state.FluxMatch_Xopt_batches
    if skip_initial:
        Xopt_batches = Xopt_batches[1:]   # (steps_per_traj, n_traj, dvs)

    # -------------------------------------------------------------------------------------------
    # Fan the per-step batched folders into per-(step, trajectory) Execution/Evaluation.{i}
    # entries. Each `MainFolder/portals_sr_ev_{s}/transport_simulation_folder` carries
    # `plasma_{0..n_traj-1}/` sub-folders with per-plasma single-plasma flux JSON pairs
    # produced by `power_transport._evaluate_batched`.
    # -------------------------------------------------------------------------------------------

    (self.folderExecution / "Execution").mkdir(parents=True, exist_ok=True)

    for s in range(steps_per_traj):
        for t in range(n_traj):
            i = s * n_traj + t
            ff = self.folderExecution / "Execution" / f"Evaluation.{i}"
            ff.mkdir(parents=True, exist_ok=True)
            source = MainFolder / f"portals_sr_ev_{s}" / "transport_simulation_folder" / f"plasma_{t}"

            if (ff / "transport_simulation_folder").exists():
                IOtools.shutil_rmtree(ff / "transport_simulation_folder")

            shutil.copytree(source, ff / "transport_simulation_folder")

    # Flatten the (steps_per_traj, n_traj, dvs) Xopt tensor into step-major (N, dvs) training
    # rows that line up with the Evaluation.{i} layout above (i = s * n_traj + t).
    Xopt = Xopt_batches.reshape(self.Originalinitial_training, -1)

    return Xopt.cpu().numpy()

"""
*********************************************************************************************************************
	External Flux Match Surrogate
*********************************************************************************************************************
"""


def flux_match_surrogate(
        step,
        profiles,
        plot_results=False,
        fn = None,
        file_write_csv=None,
        algorithm = None,
        solver_options = None,
        keep_within_bounds = True,
        target_options_use = None,
        ):
    '''
    Technique to reutilize flux surrogates to predict new conditions
    ----------------------------------------------------------------
    Usage:
        - Requires "step" to be a MITIM step with the proper surrogate parameters, the surrogates fitted and residual function defined
        - Requires "profiles" to be an object with the new profiles to be predicted (e.g. can have different BC)

    '''

    if algorithm is None:
        algorithm  = 'simple_relax'
        solver_options = {
            "tol": -1e-4,
            "tol_rel": 1e-3,        # Residual residual by 1000x (superseeds tol)
            "maxiter": 2000,
            "relax": 0.1,          # Defines relationship between flux and gradient
            "relax_dyn": True,     # If True, relax will be adjusted dynamically
            "print_each": 100,
        }

    # Prepare tensor bounds
    if keep_within_bounds:
        bounds = torch.zeros((2, len(step.GP['combined_model'].bounds))).to(step.GP['combined_model'].train_X)
        for i, ikey in enumerate(step.GP['combined_model'].bounds):
            bounds[0, i] = copy.deepcopy(step.GP['combined_model'].bounds[ikey][0])
            bounds[1, i] = copy.deepcopy(step.GP['combined_model'].bounds[ikey][1])
    else:
        bounds = None

    # ----------------------------------------------------
    # Create powerstate with new profiles
    # ----------------------------------------------------

    transport_options = copy.deepcopy(step.surrogate_parameters["powerstate"].transport_options)

    # Define transport calculation function as a surrogate model
    transport_options['evaluator'] = transport_analytic.surrogate
    transport_options["options"] = {'flux_fun': partial(step.evaluators['residual_function'],outputComponents=True)}

    # Create powerstate with the same options as the original portals but with the new profiles
    powerstate = STATEtools.powerstate(
        profiles,
        evolution_options={
            "ProfilePredicted": step.surrogate_parameters["powerstate"].predicted_channels,
            "rhoPredicted": step.surrogate_parameters["powerstate"].plasma["rho"][0,1:],
            "impurityPosition": step.surrogate_parameters["powerstate"].impurityPosition,
        },
        transport_options=transport_options,
        target_options= step.surrogate_parameters["powerstate"].target_options if target_options_use is None else target_options_use,
        tensor_options = {
            "dtype": step.surrogate_parameters["powerstate"].dfT.dtype,
            "device": step.surrogate_parameters["powerstate"].dfT.device
            },
    )

    # Pass powerstate as part of the surrogate_parameters such that transformations now occur with the new profiles
    step.surrogate_parameters['powerstate'] = powerstate

    # ----------------------------------------------------
    # Flux match
    # ----------------------------------------------------
    
    # Calculate original powerstate (for later comparison in plot)
    if plot_results:
        powerstate_orig = copy.deepcopy(powerstate)
        powerstate_orig.calculate(None)

    # Flux match
    powerstate.flux_match(
        algorithm=algorithm,
        solver_options=solver_options,
        bounds=bounds
    )

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------

    if plot_results:
        powerstate.plot(label='optimized',c='r',compare_to_state=powerstate_orig, c_orig = 'b', fn = fn)

    # ----------------------------------------------------
    # Write In Table
    # ----------------------------------------------------

    if file_write_csv is not None:

        X = powerstate.Xcurrent[-1,:].unsqueeze(0).cpu().numpy()

        inputs = []
        for i in step.bounds:
            inputs.append(i)
        optimization_data = BOgraphics.optimization_data(
            inputs,
            step.outputs,
            file=file_write_csv,
            forceNew=True,
        )

        optimization_data.update_points(X)

        print(f'> File {file_write_csv} written with optimum point')

    return powerstate
