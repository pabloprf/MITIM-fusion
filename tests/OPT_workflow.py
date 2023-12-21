"""
This example runs the complete MITIM optimization algorithm on a simple test function, and plot results
To run: python3  $MITIM_PATH/tests/MITIM_workflow.py
"""

import os, torch
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

# -----------------------------------------------------------------------------------------------------
# ----- Inputs (function to optimize)
# -----------------------------------------------------------------------------------------------------


class opt_class(STRATEGYtools.FUNmain):
    def __init__(self, folder, namelist):
        # Store folder, namelist. Read namelist
        super().__init__(folder, namelist=namelist)
        # ----------------------------------------

        # Problem description (rest of problem parameters are taken from namelist)
        self.Optim["dvs"] = ["x"]
        self.Optim["dvs_min"] = [0.0]
        self.Optim["dvs_max"] = [20.0]

        self.Optim["ofs"] = ["z", "zval"]
        self.name_objectives = ["zval_match"]

    def run(self, paramsfile, resultsfile):
        # Read stuff
        folderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        # Operations
        dictOFs["z"]["value"] = dictDVs["x"]["value"] ** 2
        dictOFs["z"]["error"] = dictOFs["z"]["value"] * 2e-2

        dictOFs["zval"]["value"] = 15.0
        dictOFs["zval"]["error"] = 0.0

        # Write stuff
        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        ofs_ordered_names = np.array(self.Optim["ofs"])

        of = Y[..., ofs_ordered_names == "z"]
        cal = Y[..., ofs_ordered_names == "zval"]

        # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L1
        res = -1 / of.shape[-1] * torch.norm((of - cal), p=1, dim=-1)

        return of, cal, res


# -----------------------------------------------------------------------------------------------------
# ----- Inputs
# -----------------------------------------------------------------------------------------------------

namelist = IOtools.expandPath("$MITIM_PATH/templates/main.namelist")
folderWork = IOtools.expandPath("$MITIM_PATH/tests/scratch/opt_test/")

if restart and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

# -----------------------------------------------------------------------------------------------------
# ----- Workflow
# -----------------------------------------------------------------------------------------------------

# Initialize class
opt_fun1D = opt_class(folderWork, namelist)

# Changes to namelist in MITIM_PATH/templates/main.namelist
opt_fun1D.Optim["initialPoints"] = 2

# Initialize BO framework
PRF_BO = STRATEGYtools.PRF_BO(opt_fun1D, restartYN=True, askQuestions=False)

# Run BO framework
PRF_BO.run()

# -----------------------------------------------------------------------------------------------------
# ----- Plotting
# -----------------------------------------------------------------------------------------------------

opt_fun1D.plot_optimization_results(analysis_level=2)
