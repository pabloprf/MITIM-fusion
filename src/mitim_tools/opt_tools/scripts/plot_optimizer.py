import argparse
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import GRAPHICStools,GUItools
from IPython import embed	

"""
e.g.
	~/MITIM/mitim_opt/opt_tools/scripts/plot_optimizationl.py --folder run1/

"""

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--from_step", type=int, required=False, default=0)
parser.add_argument("--to_step", type=int, required=False, default=-1)

args = parser.parse_args()

# Inputs

folderWork = args.folder
step_from = args.from_step
step_to = args.to_step

# Read

opt_fun = STRATEGYtools.opt_evaluator(folderWork)
opt_fun.read_optimization_results(analysis_level=1)
strat = opt_fun.prfs_model


if step_to == -1:
	step_to = len(strat.steps)

step_num = np.arange(step_from, step_to)


fn = GUItools.FigureNotebook("MITIM BO Acquisition Optimization Analysis")
fig = fn.add_figure(label='Optimization Convergence')

axs = GRAPHICStools.producePlotsGrid(len(step_num), fig=fig, hspace=0.6, wspace=0.6, sharex=False, sharey=False)

for step in step_num:

	if 'InfoOptimization' not in strat.steps[step].__dict__: break
	infoOPT = strat.steps[step].InfoOptimization

	y_acq = infoOPT[0]['info']['acq_evaluated'].numpy()

	ax = axs[step]
	ax.plot(y_acq)
	ax.set_title(f'Step #{step}')

	GRAPHICStools.addDenseAxis(ax)
	ax.set_ylim(top=0.0)

fn.show()

