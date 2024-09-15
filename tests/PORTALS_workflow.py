import os
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools import __mitimroot__

restart = True

os.makedirs(os.path.join(__mitimroot__, "tests/scratch/"), exist_ok=True)

# Inputs
inputgacode = __mitimroot__ + "/tests/data/input.gacode"
folderWork = __mitimroot__ + "/tests/scratch/portals_test/"

if restart and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

# --------------------------------------------------------------------------------------------
# Optimization Class
# --------------------------------------------------------------------------------------------

# Initialize class
portals_fun = PORTALSmain.portals(folderWork)
portals_fun.optimization_options["BO_iterations"] = 2
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
prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN=restart, askQuestions=False)
prf_bo.run()

# Plot
portals_fun.plot_optimization_results(analysis_level=4)

# Required if running in non-interactive mode
portals_fun.fn.show()