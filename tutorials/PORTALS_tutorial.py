from mitim_tools.misc_tools  import IOtools
from mitim_tools.opt_tools 	 import STRATEGYtools
from mitim_modules.portals   import PORTALSmain

# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

# Starting input.gacode file
inputgacode = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")

folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/portals_tut/")

# Initialize PORTALS class
PORTALS_fun  = PORTALSmain.evaluatePORTALS(folder)

# Radial locations (RhoLocations or RoaLocations)
PORTALS_fun.TGYROparameters['RhoLocations']  	  = [0.3,0.4,0.5,0.6,0.7,0.8,0.85]

# Profiles to predict
PORTALS_fun.TGYROparameters['ProfilesPredicted']  = [ 'te', 'ti', 'ne']

# Codes to use
PORTALS_fun.PORTALSparameters['model_used']    = 'tglf_neo-tgyro'

# TGLF specifications
PORTALS_fun.TGLFparameters['TGLFsettings'] = 5      # SAT2 EM, check out mitim_tools/gacode_tools/aux/GACODEdefaults.py for more options
PORTALS_fun.TGLFparameters['extraOptionsTGLF'] = {'BPER_USE':False} # Turn off BPER

# Plasma preparation: remove fast species, adjust quasineutrality
PORTALS_fun.INITparameters['removeFast']  	= True

# Stopping criterion 1: 200x improvement in residual
PORTALS_fun.Optim['minimumResidual'] = -5E-3

# Stopping criterion 2: inputs vary less than 0.1% for 3 consecutive iterations after 10 evaluations
PORTALS_fun.Optim['minimumDVvariation'] = [ 10, 3, 1E-1 ]

# Prepare run: search +-100% the original gradients
PORTALS_fun.prep(inputgacode,folder,ymax_rel = 1.0,ymin_rel = 1.0)

# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------

prf_bo = STRATEGYtools.PRF_BO(PORTALS_fun)
prf_bo.run()

'''
To plot while it's going or has finished (via alias in portals.bashrc):
	mitim_plot_portals portals_tut

To plot while it's still in the simple relaxation phase:
	mitim_plot_portalsSR portals_tut

Debug what's going on with TGLF
	run ~/PORTALS/portals_opt/PORTALS_tools/exe/runTGLFdrivesfromPORTALS.py --folder portals_tut/ --ev 27 --pos 0 2

'''
