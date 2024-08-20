import os
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools import __mitimroot__

restart = True

if not os.path.exists(__mitimroot__ + "/tests/scratch/"):
    os.system("mkdir " + __mitimroot__ + "/tests/scratch/")

# Inputs
inputgacode = __mitimroot__ + "/tests/data/input.gacode"
folderWork = __mitimroot__ + "/tests/scratch/portals_test/"

if restart and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

# --------------------------------------------------------------------------------------------
# Optimization Class
# --------------------------------------------------------------------------------------------

# Initialize class
portals_fun = PORTALSmain.portals(folderWork, additional_params_in_surrogate = ['q'])
portals_fun.optimization_options["BO_iterations"] = 5
portals_fun.optimization_options["initial_training"] = 3
portals_fun.MODELparameters["RhoLocations"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.INITparameters["removeFast"] = True
portals_fun.INITparameters["quasineutrality"] = True
portals_fun.INITparameters["sameDensityGradients"] = True
portals_fun.MODELparameters["transport_model"]["TGLFsettings"] = 2 # Run with TGLF SAT 0 


# Do not consider first set of variables
# del portals_fun.PORTALSparameters['physicsBasedParams'][10]
# del portals_fun.PORTALSparameters['physicsBasedParams'][30]
portals_fun.optimization_options["surrogateOptions"]['extrapointsFile'] = '/Users/pablorf/MITIM-fusion/tests/scratch/surrogate_data.csv'

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