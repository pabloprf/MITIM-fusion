import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools

"""
Script to grab the GP object from a full run, at a given step, for a given output.
This way, you can try plot, re-ft, find best parameters, etc.
It calculates speed, and generates profile file to look at bottlenecks
e.g.
	evaluate_model.py --folder run1/ --output Qi_tr_turb_5 --inputs aLti_5 --around -3
	evaluate_model.py --folder run1/ --step -1 --output Qi_tr_turb_5 --file figure.eps
"""

# ***************** Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--step", type=int, required=False, default=-1)
parser.add_argument("--output", required=False, type=str, default="Qi_tr_turb_1")
parser.add_argument("--inputs", required=False, type=str,nargs='*', default=["aLti_1"])
parser.add_argument("--around", type=int, required=False, default=-1)
parser.add_argument("--xrange", type=float, required=False, default=0.5)
parser.add_argument("--file", type=str, required=False, default=None)  # File to save .eps
parser.add_argument("--plot", type=bool, required=False, default=True)

args = parser.parse_args()

folderWork = IOtools.expandPath(args.folder)
step_num = args.step
output_label = args.output
input_labels = args.inputs
file = args.file
plotYN = args.plot
around = args.around
xrange = args.xrange

# ***************** Read

opt_fun = STRATEGYtools.opt_evaluator(folderWork)
opt_fun.read_optimization_results(analysis_level=4)
strat = opt_fun.mitim_model
step = strat.steps[step_num]
gpA = step.GP["combined_model"]
gp = step.GP["individual_models"][np.where(np.array(opt_fun.mitim_model.outputs) == output_label)[0][0]]

# ***************** Plot

cols = GRAPHICStools.listColors()

if plotYN:
    gp.plot()
    if file is not None:
        plt.savefig(file, transparent=True, dpi=300)

    fig, axs = plt.subplots(nrows=2, figsize=(6, 9))
    for i,input_label in enumerate(input_labels):
        gp.localBehavior_scan(gpA.train_X[around, :], dimension_label=input_label,xrange=xrange, axs=axs, c=cols[i], label=input_label)

    axs[0].legend()
    axs[0].set_title("Full behavior (untransformed space)")

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
