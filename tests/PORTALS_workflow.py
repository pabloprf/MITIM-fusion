import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

# Inputs
inputgacode = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")
folderWork = IOtools.expandPath("$MITIM_PATH/tests/scratch/portals_test/")

if restart and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

# --------------------------------------------------------------------------------------------
# Optimization Class
# --------------------------------------------------------------------------------------------

# Initialize class
portals_fun = PORTALSmain.portals(folderWork)
portals_fun.Optim["BOiterations"] = 2
portals_fun.Optim["initialPoints"] = 3
portals_fun.MODELparameters["RhoLocations"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.INITparameters["removeFast"] = True
portals_fun.INITparameters["quasineutrality"] = True
portals_fun.INITparameters["sameDensityGradients"] = True
portals_fun.MODELparameters["transport_model"]["TGLFsettings"] = 2 # Run with TGLF SAT 0 

# Prepare run
portals_fun.prep(inputgacode, folderWork)

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