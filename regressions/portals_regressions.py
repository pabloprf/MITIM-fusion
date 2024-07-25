import argparse
import torch
import os
import dill as pickle_dill
from mitim_tools.misc_tools import IOtools, CONFIGread
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.powertorch.physics import TRANSPORTtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

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

        print("\n>>>>> Running PORTALS test 1: Standard run with constant diffusivities")
        
        os.system(f"rm -rf {folderWork} && mkdir {folderWork}")
        with CONFIGread.redirect_all_output_to_file(f'{folderWork}/regression.log'):
            portals_fun = PORTALSmain.portals(folderWork)
            portals_fun.Optim["BO_iterations"] = 2
            portals_fun.Optim["initial_training"] = 3
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
            [prf_bo.optimization_data.data['QeTurb_1'][0],0.0129878663484079],
            [prf_bo.optimization_data.data['QeTurb_1'][1],0.0174629359509858],
            [prf_bo.optimization_data.data['QeTurb_1'][2],0.0222306543202599],
            [prf_bo.optimization_data.data['QeTurb_1'][3],0.0037220182305746],
            [prf_bo.optimization_data.data['QeTurb_1'][4],0.0301250769357799],
            [prf_bo.optimization_data.data['QeTurb_1'][5],0.0436471750834417],
            [prf_bo.optimization_data.data['QiTurb_5'][0],0.0114099018688661],
            [prf_bo.optimization_data.data['QiTurb_5'][1],0.0103728562456646],
            [prf_bo.optimization_data.data['QiTurb_5'][2],0.0095916319760464],
            [prf_bo.optimization_data.data['QiTurb_5'][3],0.0063868247281859],
            [prf_bo.optimization_data.data['QiTurb_5'][4],0.0062216868661381],
            [prf_bo.optimization_data.data['QiTurb_5'][5],0.0061692702220821],
        ])

    if test == 2:

        print("\n>>>>> Running PORTALS test 2: Standard run with TGLF")
        
        os.system(f"rm -rf {folderWork} && mkdir {folderWork}")
        with CONFIGread.redirect_all_output_to_file(f'{folderWork}/regression.log'):

            portals_fun = PORTALSmain.portals(folderWork)
            portals_fun.Optim["BO_iterations"] = 1
            portals_fun.Optim["initial_training"] = 3
            portals_fun.MODELparameters["RhoLocations"] = [0.25, 0.45, 0.65, 0.85]
            portals_fun.INITparameters["removeFast"] = True
            portals_fun.INITparameters["quasineutrality"] = True
            portals_fun.INITparameters["sameDensityGradients"] = True
            portals_fun.MODELparameters["transport_model"]["TGLFsettings"] = 2

            portals_fun.MODELparameters["ProfilesPredicted"] = ["te", "ti", "ne"]

            portals_fun.prep(inputgacode, folderWork)
            prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN=False, askQuestions=False)
            prf_bo.run()

        # Checks
        conditions_regressions([
            [prf_bo.optimization_data.data['QeTar_3'][0],0.0276660734686889],
            [prf_bo.optimization_data.data['QeTar_3'][1],0.026050457428488],
            [prf_bo.optimization_data.data['QeTar_3'][2],0.0245681162983153],
            [prf_bo.optimization_data.data['QeTar_3'][3],0.0225138750256145],
            [prf_bo.optimization_data.data['QeTar_3'][4],0.0238676726307135],
            [prf_bo.optimization_data.data['QiTurb_4'][0],0.01904210194957 ],
            [prf_bo.optimization_data.data['QiTurb_4'][1],0.015054384849328], 
            [prf_bo.optimization_data.data['QiTurb_4'][2],0.012453620533174], 
            [prf_bo.optimization_data.data['QiTurb_4'][3],0.009167817359775], 
            [prf_bo.optimization_data.data['QiTurb_4'][4],0.010592748091966],
            [prf_bo.optimization_data.data['QeTurb_1'][0],0.0008148021791468 ],
            [prf_bo.optimization_data.data['QeTurb_1'][1],0.005048271135896 ],
            [prf_bo.optimization_data.data['QeTurb_1'][2],0.0316597732275 ],
            [prf_bo.optimization_data.data['QeTurb_1'][3],0.4672666906836 ],
            [prf_bo.optimization_data.data['QeTurb_1'][4],-0.0006023859321252],
        ])

    if test == 3:

        print("\n>>>>> Running PORTALS test 3: Run with TGLF multi-channel")
        
        # os.system(f"rm -rf {folderWork} && mkdir {folderWork}")
        # with CONFIGread.redirect_all_output_to_file(f'{folderWork}/regression.log'):

        #     portals_fun = PORTALSmain.portals(folderWork)
        #     portals_fun.Optim["BO_iterations"] = 2
        #     portals_fun.Optim["initial_training"] = 3
        #     portals_fun.INITparameters["removeFast"] = True

        #     portals_fun.MODELparameters["ProfilesPredicted"] = ["te", "ti", "ne",'nZ','w0']

        #     portals_fun.PORTALSparameters["ImpurityOfInterest"] = 3
        #     portals_fun.PORTALSparameters["surrogateForTurbExch"] = True

        #     portals_fun.prep(inputgacode, folderWork)
        #     prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN=False, askQuestions=False)
        #     prf_bo.run()

        #     with open(prf_bo.mainFunction.optimization_extra, "rb") as f:
        #         mitim_runs = pickle_dill.load(f)

        # # Checks
        # conditions_regressions([
        #     [prf_bo.optimization_data.data['QeTurb_1'][5],0.0713711320661],
        #     [mitim_runs[5]['powerstate'].plasma['PexchTurb'][0,3].item(),-0.0009466626542564001]
        # ])
