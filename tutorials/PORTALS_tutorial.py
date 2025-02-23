from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools import __mitimroot__

# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

# Starting input.gacode file
inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "portals_tut"

# Initialize PORTALS class
portals_fun = PORTALSmain.portals(folder)

# Radial locations (RhoLocations or RoaLocations [last one preceeds])
portals_fun.MODELparameters["RhoLocations"] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]

# Profiles to predict
portals_fun.MODELparameters["ProfilesPredicted"] = ["te", "ti", "ne"]

# Codes to use
from mitim_modules.powertorch.physics import TRANSPORTtools
portals_fun.PORTALSparameters["transport_evaluator"] = TRANSPORTtools.tgyro_model

# TGLF specifications
portals_fun.MODELparameters["transport_model"] = {
	"turbulence":'TGLF',
	"TGLFsettings": 6,							# Check out templates/input.tglf.models.json for more options
	"extraOptionsTGLF": {"USE_BPER": False}  	# Turn off BPER
	}

# Plasma preparation: remove fast species, adjust quasineutrality
portals_fun.INITparameters["removeFast"] = True
portals_fun.INITparameters["quasineutrality"] = True

# Stopping criterion 1: 100x improvement in residual
portals_fun.optimization_options['convergence_options']['stopping_criteria_parameters']["maximum_value"] = 1e-2
portals_fun.optimization_options['convergence_options']['stopping_criteria_parameters']["maximum_value_is_rel"] = True

# Prepare run: search +-100% the original gradients
portals_fun.prep(inputgacode, ymax_rel=1.0, ymin_rel=1.0)

# --------------------------------------------------------------------------------------------
# Run (optimization following namelist: templates/main.namelists.json)
# --------------------------------------------------------------------------------------------

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun)
mitim_bo.run()

"""
To plot while it's going or has finished (via alias in portals.bashrc):
	mitim_plot_portals portals_tut

"""
