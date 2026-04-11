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
    # initial training set. All trajectories land in a single step-major folder sequence
    # `portals_sr_ev_0..initial_training-1` so downstream code (Execution.{i} copy loop,
    # PORTALSinitializer plotting) sees a flat list of evaluations exactly as before.
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

    base_X = torch.from_numpy(
        self.optimization_options["problem_options"]["dvs_base"]
    ).to(self.dfT).unsqueeze(0)

    # Each trajectory is run as an independent flux_match call. The folder_namer hook
    # (see STATEtools.flux_match) interleaves the step-s outputs of every trajectory into
    # positions `s * n_traj + t` so the resulting sequence portals_sr_ev_0..N-1 is step-major
    # and matches the Execution.{i} numbering downstream.
    def _make_namer(traj_idx, n):
        return lambda step: f"portals_sr_ev_{step * n + traj_idx}"

    Xopt_per_traj = []
    for t, user_overrides in enumerate(traj_params):
        solver_options = dict(_SIMPLE_RELAX_DEFAULTS)
        solver_options.update(user_overrides)
        # The seed-driven jitter is applied on top of the (possibly user-overridden) relax value
        # so setting `relax: 0.1` in the namelist lands at 0.1 ± addon_relax, not 0.2 ± addon_relax.
        solver_options["relax"] = solver_options["relax"] + addon_relax
        solver_options["maxiter"] = steps_per_traj
        solver_options["folder"] = MainFolder
        solver_options["folder_namer"] = _make_namer(t, n_traj)
        # `namingConvention` is still consulted by the evaluator for `nameRun` logging — keep it
        # informative even though folder layout is owned by folder_namer.
        solver_options["namingConvention"] = f"portals_sr_ev_traj{t}"

        traj_state = copy.deepcopy(self.optimization_object.powerstate)
        traj_state.modify(base_X)
        traj_state.flux_match(
            algorithm="simple_relax",
            solver_options=solver_options,
        )
        Xopt_per_traj.append(traj_state.FluxMatch_Xopt)

    # -------------------------------------------------------------------------------------------
    # Once every trajectory has completed, copy the step-major folder sequence into
    # Execution/Evaluation.{i} as if they were ordinary MITIM evaluations.
    # -------------------------------------------------------------------------------------------

    (self.folderExecution / "Execution").mkdir(parents=True, exist_ok=True)

    for i in range(self.Originalinitial_training):
        ff = self.folderExecution / "Execution" / f"Evaluation.{i}"
        ff.mkdir(parents=True, exist_ok=True)
        source = MainFolder / f"portals_sr_ev_{i}" / "transport_simulation_folder"

        if (ff / "transport_simulation_folder").exists():
            IOtools.shutil_rmtree(ff / "transport_simulation_folder")

        shutil.copytree(source, ff / "transport_simulation_folder")

    # Assemble the train_X array in step-major order so it lines up with the folder naming:
    # Xopt_per_traj[t] has shape (steps_per_traj, dvs); stacking gives (steps_per_traj, n_traj, dvs)
    # and the reshape below flattens to (steps_per_traj * n_traj, dvs) = (initial_training, dvs).
    Xopt_stack = torch.stack(Xopt_per_traj, dim=1)  # (steps_per_traj, n_traj, dvs)
    Xopt = Xopt_stack.reshape(self.Originalinitial_training, -1)

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
