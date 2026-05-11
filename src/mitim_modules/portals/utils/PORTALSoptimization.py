import copy
from mitim_modules.powertorch.physics_models import transport_analytic
import numpy as np
import torch
import shutil
import random
from functools import partial
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_tools.misc_tools import IOtools
from mitim_modules.powertorch import STATEtools
from mitim_tools.opt_tools.utils import BOgraphics, SAMPLINGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
*********************************************************************************************************************
	Initialization
*********************************************************************************************************************
"""


def initialization_simple_relax(self):
    # ------------------------------------------------------------------------------------
    # Perform flux matched using powerstate
    # ------------------------------------------------------------------------------------

    powerstate = copy.deepcopy(self.optimization_object.powerstate)

    folderExecution = IOtools.expandPath(self.folderExecution, ensurePathValid=True)

    MainFolder = folderExecution / "Initialization" / "initialization_simple_relax"
    MainFolder.mkdir(parents=True, exist_ok=True)

    a, b = IOtools.reducePathLevel(self.folderExecution, level=1)
    namingConvention = "portals_sr_ev"

    initialization_options = self.optimization_options.get("initialization_options", {})
    simple_relax_options = copy.deepcopy(initialization_options.get("simple_relax_options", {}))
    relax_jitter = initialization_options.get("simple_relax_relax_jitter", 0.03)

    if self.seed is not None and self.seed != 0 and relax_jitter > 0:
        random.seed(self.seed)
        addon_relax = random.uniform(-relax_jitter, relax_jitter)
    else:
        addon_relax = 0.0

    # Solver options tuned for simple relax of beginning of PORTALS (big jumps)
    solver_options = {
        "tol": None,
        "maxiter": self.Originalinitial_training,
        "relax": 0.2+addon_relax,   # Defines relationship between flux and gradient
        "dx_max": 0.2,              # Maximum step size in gradient, relative (e.g. a/Lx can only increase by 20% each time)
        "relax_dyn": False,
        "dx_max_abs": None,         # Maximum step size in gradient, absolute (e.g. a/Lx can only increase by 0.1 each time)
        "dx_min_abs": 0.1,          # Minimum step size in gradient, absolute (e.g. a/Lx can only increase by 0.01 each time)
        "print_each": 1,
        "folder": MainFolder,
        "namingConvention": namingConvention,
    }

    # Optional workflow-level overrides for warm-up aggressiveness.
    # Example in PORTALSedge_workflow.py:
    # optimization_options["initialization_options"]["simple_relax_options"] = {
    #     "relax": 0.35,
    #     "dx_max": 0.5,
    #     "dx_max_abs": 0.3,
    #     "dx_min_abs": 0.05,
    #     "relax_dyn": True,
    #     "relax_dyn_decrease": 2,
    #     "relax_dyn_num": 30,
    # }
    # optimization_options["initialization_options"]["simple_relax_relax_jitter"] = 0.0
    if simple_relax_options:
        solver_options.update(simple_relax_options)

    # Keep initialization bookkeeping controlled by this routine.
    solver_options["maxiter"] = self.Originalinitial_training
    solver_options["folder"] = MainFolder
    solver_options["namingConvention"] = namingConvention

    # Trick to actually start from different gradients than those in the initial_input_gacode

    X = torch.from_numpy(self.optimization_options["problem_options"]["dvs_base"]).to(self.dfT).unsqueeze(0)
    powerstate.modify(X)

    # Flux matching process

    powerstate.flux_match(
        algorithm="simple_relax",
        solver_options=solver_options,
    )
    Xopt = powerstate.FluxMatch_Xopt

    # -------------------------------------------------------------------------------------------
    # Once flux matching has been attained, copy those as if they were direct MITIM evaluations
    # -------------------------------------------------------------------------------------------

    (self.folderExecution / "Execution").mkdir(parents=True, exist_ok=True)

    for i in range(self.Originalinitial_training):
        ff = self.folderExecution / "Execution" / f"Evaluation.{i}"
        ff.mkdir(parents=True, exist_ok=True)
        newname = f"{namingConvention}_{i}"

        # Delte destination first
        if (ff / "transport_simulation_folder").exists():
            IOtools.shutil_rmtree(ff / "transport_simulation_folder")

        shutil.copytree(MainFolder / newname / "transport_simulation_folder", ff / "transport_simulation_folder") #### delete first

    return Xopt.cpu().numpy()


_FEASIBILITY_PENALTY = 1e3


def initialization_feasible_lhs(self):
    """
    LHS initialization with online feasibility rejection.

    For each candidate drawn from LHS, a cheap ``powerstate_edge.modify()`` call
    reconstructs the predicted profiles without running any transport.  If any
    profile value equals or exceeds the ``NaN`` infeasibility sentinel produced by
    the mtanh parameterizer for unphysical parameter sets, the candidate is
    discarded and a replacement is drawn.

    The loop continues until exactly ``initial_training`` feasible points have
    been collected, then the powerstate is restored to its original state.

    Intended for use with the ``mtanh`` parameterizer in ``portals_edge``
    workflows.  Returns a numpy array of shape ``(initial_training, n_dvs)``.
    """
    powerstate = self.optimization_object.powerstate
    predicted_channels = self.optimization_object.portals_parameters["solution"]["predicted_channels"]
    n_target = self.Originalinitial_training
    seed = getattr(self, "seed", 0) or 0

    print(
        f"\t- [initialization_feasible_lhs] Collecting {n_target} feasible LHS points "
        f"(channels checked: {predicted_channels})",
        typeMsg="i",
    )

    # Save plasma keys overwritten by modify() so the powerstate can be restored afterwards.
    _keys_to_backup = (
        list(predicted_channels)
        + [f"aL{ch}" for ch in predicted_channels]
        + [f"curvature_{ch}" for ch in predicted_channels]
        + [k for k in powerstate.plasma if k == "ni" or k.startswith("aLni")]
    )
    plasma_backup = {
        k: powerstate.plasma[k].clone()
        for k in _keys_to_backup
        if k in powerstate.plasma and isinstance(powerstate.plasma[k], torch.Tensor)
    }
    xcurrent_backup = powerstate.Xcurrent

    feasible_X = []
    n_rejected = 0
    total_draws = 0
    max_total_draws = 1000 * n_target
    seed_offset = 0

    while len(feasible_X) < n_target:
        if total_draws >= max_total_draws:
            raise RuntimeError(
                f"[initialization_feasible_lhs] Failed to collect {n_target} feasible points "
                f"after {total_draws} draws ({n_rejected} rejected). "
                "Check that the DV bounds contain a feasible region for the mtanh parameterizer."
            )

        n_still_needed = n_target - len(feasible_X)
        n_draw = max(n_still_needed * 4, n_target)
        candidates = SAMPLINGtools.LHS(n_draw, self.boundsInitialization, seed=seed + seed_offset)
        seed_offset += 1
        total_draws += n_draw

        for i in range(candidates.shape[0]):
            if len(feasible_X) >= n_target:
                break

            x_row = candidates[i].unsqueeze(0).to(self.dfT)
            powerstate.modify(x_row)

            feasible = all(
                not (powerstate.plasma[ch] >= _FEASIBILITY_PENALTY).any().item()
                for ch in predicted_channels
                if ch in powerstate.plasma
            )

            if feasible:
                feasible_X.append(candidates[i].cpu().numpy())
            else:
                n_rejected += 1

    # Restore powerstate to its original state before returning.
    powerstate.plasma.update(plasma_backup)
    powerstate.Xcurrent = xcurrent_backup

    print(
        f"\t- [initialization_feasible_lhs] Done: {n_target} accepted, {n_rejected} rejected "
        f"(feasibility rate \u2248 {n_target / max(n_target + n_rejected, 1):.1%})",
        typeMsg="i",
    )

    return np.array(feasible_X[:n_target])


def initialization_feasible_simple_relax(self):
    """
    Feasible initialization by local relaxation / perturbation from the base DV.

    This avoids global-space LHS exploration. The routine attempts to walk along
    the accepted-step trajectory (when available), while enforcing mtanh
    feasibility through cheap ``powerstate.modify()`` checks only.

    Strategy
    --------
    1) Start from ``dvs_base`` and ensure feasibility.
    2) Propose small bounded perturbations.
    3) Prefer direction of previous accepted step; add jitter for diversity.
    4) Backtrack (reduce step size) on infeasible proposals.
    5) Continue until exactly ``initial_training`` feasible points are collected.
    """
    powerstate = self.optimization_object.powerstate
    predicted_channels = self.optimization_object.portals_parameters["solution"]["predicted_channels"]
    n_target = self.Originalinitial_training
    seed = getattr(self, "seed", 0) or 0

    rng = np.random.default_rng(seed)

    bounds = np.asarray(self.boundsInitialization, dtype=float)
    x_min = bounds[0]
    x_max = bounds[1]
    x_span = np.maximum(x_max - x_min, 1e-12)

    x_base = np.asarray(self.optimization_options["problem_options"]["dvs_base"], dtype=float).reshape(-1)
    x_base = np.clip(x_base, x_min, x_max)

    print(
        f"\t- [initialization_feasible_simple_relax] Collecting {n_target} local feasible points "
        f"(channels checked: {predicted_channels})",
        typeMsg="i",
    )

    # Save plasma keys overwritten by modify() so the powerstate can be restored afterwards.
    _keys_to_backup = (
        list(predicted_channels)
        + [f"aL{ch}" for ch in predicted_channels]
        + [f"curvature_{ch}" for ch in predicted_channels]
        + [k for k in powerstate.plasma if k == "ni" or k.startswith("aLni")]
    )
    plasma_backup = {
        k: powerstate.plasma[k].clone()
        for k in _keys_to_backup
        if k in powerstate.plasma and isinstance(powerstate.plasma[k], torch.Tensor)
    }
    xcurrent_backup = powerstate.Xcurrent

    def _is_feasible(x_np):
        x_row = torch.from_numpy(np.asarray(x_np, dtype=float)).to(self.dfT).unsqueeze(0)
        powerstate.modify(x_row)
        return all(
            not (powerstate.plasma[ch] >= _FEASIBILITY_PENALTY).any().item()
            for ch in predicted_channels
            if ch in powerstate.plasma
        )

    # Ensure start point is feasible; if not, perform tiny random nudges around base.
    feasible_points = []
    n_rejected = 0

    if _is_feasible(x_base):
        feasible_points.append(x_base.copy())
    else:
        found_start = False
        for _ in range(200):
            jitter = rng.normal(size=x_base.shape[0]) * (0.01 * x_span)
            x_try = np.clip(x_base + jitter, x_min, x_max)
            if _is_feasible(x_try):
                feasible_points.append(x_try.copy())
                found_start = True
                break
            n_rejected += 1
        if not found_start:
            powerstate.plasma.update(plasma_backup)
            powerstate.Xcurrent = xcurrent_backup
            raise RuntimeError(
                "[initialization_feasible_simple_relax] Could not find a feasible start near dvs_base. "
                "Please tighten bounds or adjust base values."
            )

    # Local stepping parameters (small, conservative by default).
    step_frac = 0.05
    step_frac_min = 0.025
    step_frac_max = 0.1
    max_tries_per_point = 60

    while len(feasible_points) < n_target:
        x_ref = feasible_points[-1]
        accepted_this_point = False

        local_step = step_frac
        for _ in range(max_tries_per_point):
            if len(feasible_points) >= 2:
                trend = feasible_points[-1] - feasible_points[-2]
                trend_norm = np.linalg.norm(trend)
                if trend_norm > 1e-14:
                    direction = trend / trend_norm
                else:
                    direction = rng.normal(size=x_ref.shape[0])
            else:
                direction = rng.normal(size=x_ref.shape[0])

            direction = direction / max(np.linalg.norm(direction), 1e-14)
            noise = rng.normal(size=x_ref.shape[0])
            noise = noise / max(np.linalg.norm(noise), 1e-14)

            # Follow trajectory if available, with mild noise to avoid collapse.
            move = 0.75 * direction + 0.25 * noise
            move = move / max(np.linalg.norm(move), 1e-14)

            x_try = np.clip(x_ref + (local_step * x_span) * move, x_min, x_max)

            if _is_feasible(x_try):
                feasible_points.append(x_try.copy())
                step_frac = min(step_frac * 1.10, step_frac_max)
                accepted_this_point = True
                break

            n_rejected += 1
            local_step = max(local_step * 0.5, step_frac_min)

        if not accepted_this_point:
            # Fallback: tiny isotropic step around the initial feasible point.
            x_anchor = feasible_points[0]
            fallback_ok = False
            for _ in range(80):
                jitter = rng.normal(size=x_anchor.shape[0]) * (step_frac_min * x_span)
                x_try = np.clip(x_anchor + jitter, x_min, x_max)
                if _is_feasible(x_try):
                    feasible_points.append(x_try.copy())
                    fallback_ok = True
                    break
                n_rejected += 1

            if not fallback_ok:
                powerstate.plasma.update(plasma_backup)
                powerstate.Xcurrent = xcurrent_backup
                raise RuntimeError(
                    f"[initialization_feasible_simple_relax] Failed to collect {n_target} feasible points "
                    f"from local perturbations. Collected {len(feasible_points)} points."
                )

    # Restore powerstate to its original state before returning.
    powerstate.plasma.update(plasma_backup)
    powerstate.Xcurrent = xcurrent_backup

    print(
        f"\t- [initialization_feasible_simple_relax] Done: {n_target} accepted, {n_rejected} rejected",
        typeMsg="i",
    )

    return np.array(feasible_points[:n_target])


def mtanh_feasibility_constraint(portals_obj, safety_margin=1e-10, powerstate=None):
    """
    Build BoTorch nonlinear feasibility constraints for mtanh solver-space DVs.

    Returns one aggregated constraint that is feasible only when ALL
    predicted channels satisfy:
        y_bc * aLy_bc - exp(log_alpha + log_Ralpha) >= safety_margin

    Returns
    -------
    list[tuple[callable, bool]]
        BoTorch ``nonlinear_inequality_constraints`` payload with a single
        ``(constraint_callable, is_intra_point=True)`` tuple.
    """
    powerstate = portals_obj.powerstate if powerstate is None else powerstate
    parameterizer = powerstate.parameterizer

    if parameterizer.__class__.__name__.lower() != "mtanh":
        return []

    predicted_channels = portals_obj.portals_parameters["solution"]["predicted_channels"]
    dv_names = portals_obj.optimization_options["problem_options"]["dvs"]

    def _bc_value(name):
        bc = powerstate.bc_dict.get(name, None)

        if bc is None:
            return None

        if isinstance(bc, dict):
            return float(bc.get("val", bc.get("value", np.nan)))

        if isinstance(bc, (list, tuple, np.ndarray)):
            if len(bc) == 0:
                return None
            if isinstance(bc[0], dict):
                return float(bc[0].get("val", np.nan))
            return float(bc[0])

        return float(bc)

    channel_specs = []
    for ch in predicted_channels:
        log_alpha_name = f"{ch}_log_alpha"
        log_ralpha_name = f"{ch}_log_Ralpha"

        if (log_alpha_name not in dv_names) or (log_ralpha_name not in dv_names):
            continue

        y_bc = _bc_value(ch)
        aly_bc = _bc_value(f"aL{ch}")

        if y_bc is None and ch in powerstate.plasma:
            y_bc = float(powerstate.plasma[ch][0, -1].item())
        if aly_bc is None and (f"aL{ch}") in powerstate.plasma:
            aly_bc = float(powerstate.plasma[f"aL{ch}"][0, -1].item())

        if y_bc is None or aly_bc is None:
            raise ValueError(
                f"[mtanh_feasibility_constraint] Missing BCs for channel '{ch}' "
                "(need both y and aLy boundary values)."
            )

        channel_specs.append(
            {
                "idx_alpha": dv_names.index(log_alpha_name),
                "idx_ralpha": dv_names.index(log_ralpha_name),
                "limit": float(y_bc * aly_bc - safety_margin),
            }
        )

    if len(channel_specs) == 0:
        return []

    def _constraint_all_profiles(X, specs=channel_specs):
        # Feasible iff min margin across channels is >= 0.
        margins = []
        for spec in specs:
            log_alpha = X[..., spec["idx_alpha"]]
            log_ralpha = X[..., spec["idx_ralpha"]]
            lim_t = torch.as_tensor(spec["limit"], dtype=log_alpha.dtype, device=log_alpha.device)
            margins.append(lim_t - torch.exp(log_alpha + log_ralpha))

        return torch.stack(margins, dim=0).amin(dim=0)

    return [(_constraint_all_profiles, True)]


def mtanh_feasibility_constraint_builder(portals_obj, safety_margin=1e-10):
    """
    Return a callable that rebuilds mtanh feasibility constraints at runtime.

    This is used so boundary conditions (y_bc, aLy_bc) can evolve and constraints
    refresh each optimization iteration.
    """

    def _build_constraints(_fun=None):
        runtime_powerstate = None

        if _fun is not None:
            try:
                runtime_powerstate = _fun.evaluators["GP"].surrogate_parameters.get("powerstate", None)
            except Exception:
                runtime_powerstate = None

        return mtanh_feasibility_constraint(
            portals_obj,
            safety_margin=safety_margin,
            powerstate=runtime_powerstate,
        )

    return _build_constraints


def enable_mtanh_feasibility_constraint(portals_obj, safety_margin=1e-10):
    """
    Attach mtanh nonlinear inequality constraints to the BoTorch optimizer.

    Returns
    -------
    bool
        True when constraints were attached, False otherwise.
    """
    parameterizer_name = portals_obj.powerstate.parameterizer.__class__.__name__.lower()
    if parameterizer_name != "mtanh":
        return False

    constraints = mtanh_feasibility_constraint(portals_obj, safety_margin=safety_margin)
    if len(constraints) == 0:
        return False

    acq_opts = portals_obj.optimization_options["acquisition_options"]
    optimizer_options = acq_opts.setdefault("optimizer_options", {})
    botorch_options = optimizer_options.setdefault("botorch", {})
    botorch_options["nonlinear_inequality_constraints"] = constraints
    botorch_options["nonlinear_inequality_constraints_builder"] = mtanh_feasibility_constraint_builder(
        portals_obj,
        safety_margin=safety_margin,
    )

    print(
        "\t- [mtanh_feasibility_constraint] Enabled dynamic nonlinear mtanh feasibility constraints",
        typeMsg="i",
    )

    return True


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
