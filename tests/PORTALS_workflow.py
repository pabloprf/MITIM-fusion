import os
import torch
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Inputs
inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "portals_test"

if cold_start and folderWork.exists():
    os.system(f"rm -r {folderWork.resolve()}")

# Let's not consume the entire computer resources when running test... limit to 4 threads
torch.set_num_threads(4)

# --------------------------------------------------------------------------------------------
# Optimization Class
# --------------------------------------------------------------------------------------------

# Initialize class
portals_fun = PORTALSmain.portals(folderWork)
portals_fun.optimization_options["BO_iterations"] = 1
portals_fun.optimization_options["initial_training"] = 3
portals_fun.MODELparameters["RhoLocations"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.INITparameters["removeFast"] = True
portals_fun.INITparameters["quasineutrality"] = True
portals_fun.INITparameters["sameDensityGradients"] = True
portals_fun.MODELparameters["transport_model"]["TGLFsettings"] = 2

# Prepare run
portals_fun.prep(inputgacode)

# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------

# Run
prf_bo = STRATEGYtools.PRF_BO(portals_fun, cold_start=cold_start, askQuestions=False)
prf_bo.run()

# Plot
portals_fun.plot_optimization_results(analysis_level=4)

# Required if running in non-interactive mode
portals_fun.fn.show()