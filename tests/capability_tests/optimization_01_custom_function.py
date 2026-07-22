"""
CAPABILITY: Generic MITIM Bayesian optimization on a custom function
--------------------------------------------------------------------
This script teaches how to use the MITIM surrogate-based optimization engine
(the same one that powers PORTALS) on your own function. The problem solved
here is trivial on purpose — find z such that z^2 matches the target value 15
— so that the focus is on the framework, not the physics.

Key teaching points:
    1. A problem is defined by subclassing STRATEGYtools.opt_evaluator and
       declaring, on top of the default namelist
       (templates/namelist.optimization.yaml), the design variables `dvs`
       (with bounds) and the outputs `ofs`.
    2. MITIM optimizes MATCHES between pairs of outputs: here "z" (the model
       output, x^2) is driven towards "zval" (the target, 15). This is
       exactly how PORTALS drives transport fluxes towards target fluxes.
    3. run() is the black box: it receives the design variables (dictDVs),
       fills in values AND errors (1-sigma, used by the Gaussian-process
       surrogates) of every output, and writes them back.
    4. scalarized_objective() defines the scalar residual that the optimizer
       maximizes — here the negative L1 norm of (z - zval), so the best
       point is the closest match (x = sqrt(15) ~ 3.873).
"""

import numpy as np
import torch
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# the evaluations already present in the folder instead of re-running them
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: evaluations, surrogates, logs and results live in it
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_optimization"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Define the optimization problem by subclassing opt_evaluator
# ---------------------------------------------------------------------------------------------------------------------

class opt_class(STRATEGYtools.opt_evaluator):
    def __init__(self, folder, namelist):
        # Store folder and namelist, and read the namelist into self.optimization_options
        super().__init__(folder, namelist=namelist)

        # Problem description (the rest of the optimization parameters come from the namelist):
        # one design variable "x" within [0, 20]
        self.optimization_options["problem_options"]["dvs"] = ["x"]
        self.optimization_options["problem_options"]["dvs_min"] = [0.0]
        self.optimization_options["problem_options"]["dvs_max"] = [20.0]

        # Two outputs forming a match pair: "z" (model) is driven towards "zval" (target)
        self.optimization_options["problem_options"]["ofs"] = ["z", "zval"]
        self.name_objectives = ["zval_match"]

    def run(self, paramsfile, resultsfile):
        # Read the design variables of this evaluation
        folderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        # The actual black-box evaluation: any code can go here (calling external codes,
        # simulations, etc.). Each output needs a value and a 1-sigma error, which is
        # what the Gaussian-process surrogates fit
        dictOFs["z"]["value"] = dictDVs["x"]["value"] ** 2
        dictOFs["z"]["error"] = dictOFs["z"]["value"] * 2e-2  # 2% error

        # The target to match: a fixed value with no uncertainty
        dictOFs["zval"]["value"] = 15.0
        dictOFs["zval"]["error"] = 0.0

        # Write the outputs of this evaluation
        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        # From the full output tensor Y, separate model outputs (of) and targets (cal)
        ofs_ordered_names = np.array(self.optimization_options["problem_options"]["ofs"])

        of = Y[..., ofs_ordered_names == "z"]
        cal = Y[..., ofs_ordered_names == "zval"]

        # Scalar residual to MAXIMIZE: negative (1/N-normalized) L1 norm of the mismatch
        res = -1 / of.shape[-1] * torch.norm((of - cal), p=1, dim=-1)

        return of, cal, res

# ---------------------------------------------------------------------------------------------------------------------
# 2. Initialize the problem from the default optimization namelist
# ---------------------------------------------------------------------------------------------------------------------

# templates/namelist.optimization.yaml carries all the optimizer defaults (acquisition,
# surrogate options, convergence criteria, ...); read it and modify in-situ, exactly as
# the PORTALS namelist is modified in portals_01_tglf_standard.py
namelist = __mitimroot__ / "templates" / "namelist.optimization.yaml"
opt_fun1D = opt_class(folderWork, namelist)

# Number of initial (random/LHS) evaluations used to seed the surrogates before the
# Bayesian-optimization iterations start
opt_fun1D.optimization_options["initialization_options"]["initial_training"] = 2

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

# MITIM_BO is the generic optimization driver; askQuestions=False avoids interactive prompts
MITIM_BO = STRATEGYtools.MITIM_BO(opt_fun1D, cold_start=cold_start, askQuestions=False)
MITIM_BO.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot results (residual evolution, surrogate behavior, DV trajectories)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (opt_fun1D.fn); show() opens the GUI
opt_fun1D.plot_optimization_results(analysis_level=2)
opt_fun1D.fn.show()
