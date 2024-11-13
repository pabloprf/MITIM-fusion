import copy
import torch
import shutil
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
    digitized = IOtools.string_to_sequential_number(f'{self.folderExecution}', num_digits=10)
    namingConvention = f"portals_{b}_{digitized}_ev"

    algorithmOptions = {
        "tol": 1e-6,
        "max_it": self.Originalinitial_training,
        "relax": 0.2,
        "dx_max": 0.2,
        "print_each": 1,
        "MainFolder": MainFolder,
        "storeValues": True,
        "namingConvention": namingConvention,
    }

    # Trick to actually start from different gradients than those in the initial_input_gacode

    X = torch.from_numpy(self.optimization_options["dvs_base"]).to(self.dfT).unsqueeze(0)
    powerstate.modify(X)

    # Flux matching process

    powerstate.findFluxMatchProfiles(
        algorithm="simple_relax",
        algorithmOptions=algorithmOptions,
    )
    Xopt = powerstate.FluxMatch_Xopt

    # -------------------------------------------------------------------------------------------
    # Once flux matching has been attained, copy those as if they were direct MITIM evaluations
    # -------------------------------------------------------------------------------------------

    (self.folderExecution / "Execution").mkdir(parents=True, exist_ok=True)

    for i in range(self.Originalinitial_training):
        ff = self.folderExecution / "Execution" / f"Evaluation.{i}"
        ff.mkdir(parents=True, exist_ok=True)
        newname = f"{namingConvention}{i}"
        shutil.copytree(MainFolder / newname / "model_complete", ff / "model_complete")

    return Xopt.cpu().numpy()

"""
*********************************************************************************************************************
	External Flux Match Surrogate
*********************************************************************************************************************
"""


def flux_match_surrogate(step,profiles_new, plot_results=True, file_write_csv=None,
    algorithm = {'root':{'storeValues':True}}):
    '''
    Technique to reutilize flux surrogates to predict new conditions
    ----------------------------------------------------------------
    Usage:
        - Requires "step" to be a MITIM step with the proper surrogate parameters, the surrogates fitted and residual function defined
        - Requires "profiles_new" to be an object with the new profiles to be predicted (e.g. can have different BC)

    Notes:
        #TODO: So far only works if Te,Ti,ne

    '''

    # ----------------------------------------------------
    # Create powerstate with new profiles
    # ----------------------------------------------------

    TransportOptions = copy.deepcopy(step.surrogate_parameters["powerstate"].TransportOptions)

    # Define transport calculation function as a surrogate model
    TransportOptions['transport_evaluator'] = TRANSPORTtools.surrogate_model
    TransportOptions['ModelOptions'] = {'flux_fun': partial(step.evaluators['residual_function'],outputComponents=True)}


    # Create powerstate with the same options as the original portals but with the new profiles
    powerstate = STATEtools.powerstate(
        profiles_new,
        EvolutionOptions={
            "ProfilePredicted": step.surrogate_parameters["powerstate"].ProfilesPredicted,
            "rhoPredicted": step.surrogate_parameters["powerstate"].plasma["rho"][0,1:],
            "useConvectiveFluxes": step.surrogate_parameters["powerstate"].useConvectiveFluxes,
            "impurityPosition": step.surrogate_parameters["powerstate"].impurityPosition,
            "fineTargetsResolution": step.surrogate_parameters["powerstate"].fineTargetsResolution,
        },
        TransportOptions=TransportOptions,
        TargetOptions=step.surrogate_parameters["powerstate"].TargetOptions,
    )

    # Pass powerstate as part of the surrogate_parameters such that transformations now occur with the new profiles
    step.surrogate_parameters['powerstate'] = powerstate

    # ----------------------------------------------------
    # Flux match
    # ----------------------------------------------------
    
    powerstate_orig = copy.deepcopy(powerstate)
    powerstate_orig.calculate(None)

    powerstate.findFluxMatchProfiles(
        algorithm=list(algorithm.keys())[0],
        algorithmOptions=algorithm[list(algorithm.keys())[0]])

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------

    if plot_results:
        powerstate.plot(label='optimized',c='r',compare_to_orig=powerstate_orig, c_orig = 'b')

    # ----------------------------------------------------
    # Write In Table
    # ----------------------------------------------------

    X = powerstate.Xcurrent[-1,:].unsqueeze(0).numpy()

    if file_write_csv is not None:
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
