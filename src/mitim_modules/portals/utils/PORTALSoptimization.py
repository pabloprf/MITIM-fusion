import copy
import torch
import shutil
import random
from functools import partial
from mitim_modules.powertorch.physics import TRANSPORTtools
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


def initialization_simple_relax(self):
    # ------------------------------------------------------------------------------------
    # Perform flux matched using powerstate
    # ------------------------------------------------------------------------------------

    powerstate = copy.deepcopy(self.optimization_object.powerstate)

    folderExecution = IOtools.expandPath(self.folderExecution, ensurePathValid=True)

    MainFolder = folderExecution / "Initialization" / "initialization_simple_relax"
    MainFolder.mkdir(parents=True, exist_ok=True)

    a, b = IOtools.reducePathLevel(self.folderExecution, level=1)
    namingConvention = f"portals_sr_{b}_ev"

    if self.seed is not None and self.seed != 0:
        random.seed(self.seed)
        addon_relax = random.uniform(-0.03, 0.03) 
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
        if (ff / "model_complete").exists():
            IOtools.shutil_rmtree(ff / "model_complete")

        shutil.copytree(MainFolder / newname / "model_complete", ff / "model_complete") #### delete first

    return Xopt.cpu().numpy()

"""
*********************************************************************************************************************
	External Flux Match Surrogate
*********************************************************************************************************************
"""


def flux_match_surrogate(step,profiles, plot_results=False, fn = None, file_write_csv=None, algorithm = None, solver_options = None, keep_within_bounds = True):
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

    TransportOptions = copy.deepcopy(step.surrogate_parameters["powerstate"].TransportOptions)

    # Define transport calculation function as a surrogate model
    TransportOptions['transport_evaluator'] = TRANSPORTtools.surrogate_model
    TransportOptions['ModelOptions'] = {'flux_fun': partial(step.evaluators['residual_function'],outputComponents=True)}

    # Create powerstate with the same options as the original portals but with the new profiles
    powerstate = STATEtools.powerstate(
        profiles,
        EvolutionOptions={
            "ProfilePredicted": step.surrogate_parameters["powerstate"].ProfilesPredicted,
            "rhoPredicted": step.surrogate_parameters["powerstate"].plasma["rho"][0,1:],
            "useConvectiveFluxes": step.surrogate_parameters["powerstate"].useConvectiveFluxes,
            "impurityPosition": step.surrogate_parameters["powerstate"].impurityPosition,
            "fineTargetsResolution": step.surrogate_parameters["powerstate"].fineTargetsResolution,
        },
        TransportOptions=TransportOptions,
        TargetOptions=step.surrogate_parameters["powerstate"].TargetOptions,
        tensor_opts = {
            "dtype": step.surrogate_parameters["powerstate"].dfT.dtype,
            "device": step.surrogate_parameters["powerstate"].dfT.device},
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
