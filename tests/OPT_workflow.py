"""
This example runs the complete MITIM optimization algorithm on a simple test function, and plot results
To run: python3  tests/MITIM_workflow.py
"""

import os
import torch
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------------------------------
# ----- Inputs (function to optimize)
# -----------------------------------------------------------------------------------------------------


class opt_class(STRATEGYtools.opt_evaluator):
    def __init__(self, folder, namelist):
        # Store folder, namelist. Read namelist
        super().__init__(folder, namelist=namelist)
        # ----------------------------------------

        # Problem description (rest of problem parameters are taken from namelist)
        self.optimization_options["dvs"] = ["x"]
        self.optimization_options["dvs_min"] = [0.0]
        self.optimization_options["dvs_max"] = [20.0]

        self.optimization_options["ofs"] = ["z", "zval"]
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
        import numpy as np
        ofs_ordered_names = np.array(self.optimization_options["ofs"])

        of = Y[..., ofs_ordered_names == "z"]
        cal = Y[..., ofs_ordered_names == "zval"]

        # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L1
        res = -1 / of.shape[-1] * torch.norm((of - cal), p=1, dim=-1)

        return of, cal, res


# -----------------------------------------------------------------------------------------------------
# ----- Inputs
# -----------------------------------------------------------------------------------------------------

namelist = __mitimroot__ / "templates" / "main.namelist.json"
folderWork = __mitimroot__ / "tests" / "scratch" / "opt_test"

if cold_start and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

# -----------------------------------------------------------------------------------------------------
# ----- Workflow
# -----------------------------------------------------------------------------------------------------

# Initialize class
opt_fun1D = opt_class(folderWork, namelist)

# Changes to namelist in templates/main.namelist.json
opt_fun1D.optimization_options["initial_training"] = 2

# Initialize BO framework
PRF_BO = STRATEGYtools.PRF_BO(opt_fun1D, cold_start=cold_start, askQuestions=False)

# Run BO framework
PRF_BO.run()

# -----------------------------------------------------------------------------------------------------
# ----- Plotting
# -----------------------------------------------------------------------------------------------------

opt_fun1D.plot_optimization_results(analysis_level=2)

# Required if running in non-interactive mode
opt_fun1D.fn.show()
