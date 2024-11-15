import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools

"""
Script to grab the GP object from a full run, at a given step, for a given output.
This way, you can try plot, re-ft, find best parameters, etc.
It calculates speed, and generates profile file to look at bottlenecks
e.g.
	evaluate_model.py --folder run1/ --output QiTurb_5 --input aLti_5
	evaluate_model.py --folder run1/ --step -1 --output QiTurb_5 --file figure.eps
"""

# ***************** Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--step", type=int, required=False, default=-1)
parser.add_argument("--output", required=False, type=str, default="QiTurb_1")
parser.add_argument("--input", required=False, type=str, default="aLti_1")
parser.add_argument(
    "--file", type=str, required=False, default=None
)  # File to save .eps
parser.add_argument("--plot", type=bool, required=False, default=True)
args = parser.parse_args()

folderWork = IOtools.expandPath(args.folder)
step_num = args.step
output_label = args.output
input_label = args.input
file = args.file
plotYN = args.plot

# ***************** Read

opt_fun = STRATEGYtools.opt_evaluator(folderWork)
opt_fun.read_optimization_results(analysis_level=4)
strat = opt_fun.prfs_model
step = strat.steps[step_num]
gpA = step.GP["combined_model"]
gp = step.GP["individual_models"][
    np.where(np.array(opt_fun.prfs_model.outputs) == output_label)[0][0]
]

# ***************** Plot

if plotYN:
    gp.plot()
    if file is not None:
        plt.savefig(file, transparent=True, dpi=300)

    gp.localBehavior_scan(gpA.train_X[-1, :], dimension_label=input_label)

    # gp.plot(plotFundamental=False)
    # gp.plotTraining()

# ***************** Speed tester

else:
    num = 1000

    x = torch.rand(num, gp.train_X.shape[-1]).to(gpA.train_X)

    with IOtools.speeder("profiler.prof"), torch.no_grad():
        mean, upper, lower, _ = gpA.predict(x)

    with IOtools.speeder("profiler_jac.prof"):
        gpA.localBehavior(x, plotYN=False)
