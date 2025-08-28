import os
import torch
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSoptimization
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Inputs
inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "portals_test"

if cold_start and folderWork.exists():
    os.system(f"rm -r {folderWork.resolve()}")

# Let's not consume the entire computer resources when running test... limit threads
torch.set_num_threads(8)

# --------------------------------------------------------------------------------------------
# Optimization Class
# --------------------------------------------------------------------------------------------

# Initialize class
portals_fun = PORTALSmain.portals(folderWork)
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 1
portals_fun.optimization_options["initialization_options"]["initial_training"] = 2

portals_fun.portals_parameters["main_parameters"]['turbulent_exchange_as_surrogate'] = True

portals_fun.portals_parameters["initialization_parameters"]["remove_fast"] = True
portals_fun.portals_parameters["initialization_parameters"]["quasineutrality"] = True
portals_fun.portals_parameters["initialization_parameters"]["enforce_same_aLn"] = True

portals_fun.portals_parameters["model_parameters"]["radii_rho"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.portals_parameters["model_parameters"]['predicted_channels'] = ["te", "ti", "ne", "nZ", 'w0'] 
portals_fun.portals_parameters["model_parameters"]['ImpurityOfInterest'] = 'N'
portals_fun.portals_parameters["model_parameters"]["transport_parameters"]["transport_evaluator_options"]["tglf"]["run"]["code_settings"] = 2

# Prepare run
portals_fun.prep(inputgacode)

# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------

# Run
mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# Plot
portals_fun.plot_optimization_results(analysis_level=4)

# For fun and to show capabilities, let's do a flux match of the current surrogates and plot in the same notebook
PORTALSoptimization.flux_match_surrogate(
    mitim_bo.steps[-1],PROFILEStools.gacode_state(inputgacode),
    fn = portals_fun.fn,
    plot_results = True,
    keep_within_bounds = False
    )

# Required if running in non-interactive mode
portals_fun.fn.show()
