import argparse
import torch
import os
from mitim_tools.misc_tools import IOtools, CONFIGread
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.powertorch.physics import TRANSPORTtools
from mitim_tools.misc_tools.IOtools import printMsg as print

# Get test number
parser = argparse.ArgumentParser()
parser.add_argument("test", type=int)
args = parser.parse_args()
test = args.test

if test == 0:
    tests = [1,2]
else:
    tests = [test]

# Set up case
inputgacode = IOtools.expandPath("$MITIM_PATH/regressions/data/input.gacode")

# ---------------------------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------------------------

def conditions_regressions(variables):

    conditions = True

    # Checks
    for var in variables:
        conditions &= var[0] == var[1]

    # Results
    if conditions:
        print("\t PASSED")
    else:
        print("\t FAILED",typeMsg='w')


for test in tests:

    folderWork  = IOtools.expandPath(f"$MITIM_PATH/regressions/scratch/portals_regression_{test}/")

    if test == 1:

        print("\n>>>>> Running PORTALS test 1: Standard quick run with constant diffusivities")
        
        os.system(f"rm -rf {folderWork} && mkdir {folderWork}")
        with CONFIGread.redirect_all_output_to_file(f'{folderWork}/regression.log'):

            portals_fun = PORTALSmain.portals(folderWork)
            portals_fun.Optim["BOiterations"] = 2
            portals_fun.Optim["initialPoints"] = 3
            portals_fun.INITparameters["removeFast"] = True

            portals_fun.MODELparameters["ProfilesPredicted"] = ["te", "ti"]
            portals_fun.Optim["optimizers"] = "botorch"

            portals_fun.PORTALSparameters["transport_evaluator"] = TRANSPORTtools.diffusion_model
            ModelOptions = {'chi_e': torch.ones(5)*0.5,'chi_i':  torch.ones(5)*2.0}

            portals_fun.prep(inputgacode, folderWork, ModelOptions=ModelOptions)
            prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN=False, askQuestions=False)
            prf_bo.run()

        # Checks
        conditions_regressions([
            [prf_bo.optimization_data.data['QeTurb_1'][2],0.0203659938088274]
        ])

    if test == 2:

        print("\n>>>>> Running PORTALS test 2: Standard quick run with TGLF")
        
        os.system(f"rm -rf {folderWork} && mkdir {folderWork}")
        with CONFIGread.redirect_all_output_to_file(f'{folderWork}/regression.log'):

            portals_fun = PORTALSmain.portals(folderWork)
            portals_fun.Optim["BOiterations"] = 2
            portals_fun.Optim["initialPoints"] = 3
            portals_fun.INITparameters["removeFast"] = True

            portals_fun.MODELparameters["ProfilesPredicted"] = ["te", "ti", "ne"]

            portals_fun.prep(inputgacode, folderWork)
            prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN=False, askQuestions=False)
            prf_bo.run()

        # Checks
        conditions_regressions([
            [prf_bo.optimization_data.data['QeTurb_1'][2],0.01802725581511],
            [prf_bo.optimization_data.data['GeTurb_3'][2],0.0004633298635944]
        ])
