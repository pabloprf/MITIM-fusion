import argparse
import torch
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import GRAPHICStools,GUItools

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--from_step", type=int, required=False, default=0)
parser.add_argument("--to_step", type=int, required=False, default=-1)

args = parser.parse_args()

# Inputs

folderWork = IOtools.expandPath(args.folder)
step_from = args.from_step
step_to = args.to_step

# Read

opt_fun = STRATEGYtools.opt_evaluator(folderWork)
opt_fun.read_optimization_results(analysis_level=1)
strat = opt_fun.mitim_model


if step_to == -1:
	step_to = len(strat.steps)

step_to = np.min([step_to, len(strat.steps)])

step_num = np.arange(step_from, step_to)


fn = GUItools.FigureNotebook("MITIM BO Acquisition Optimization Analysis")
fig = fn.add_figure(label='Optimization Convergence')

axs = GRAPHICStools.producePlotsGrid(len(step_num), fig=fig, hspace=0.6, wspace=0.3, sharex=False, sharey=False)

for step in step_num:

	ax = axs[step]

	if 'InfoOptimization' not in strat.steps[step].__dict__: break

	# Grab info from optimization
	infoOPT = strat.steps[step].InfoOptimization
	y_acq = infoOPT[0]['info']['acq_evaluated'].cpu().numpy()

	# Operate
	acq = strat.steps[step].evaluators['acq_function']

	acq_trained = np.zeros(strat.steps[step].train_X.shape[0])
	for ix in range(strat.steps[step].train_X.shape[0]):
		acq_trained[ix] = acq(torch.Tensor(strat.steps[step].train_X[ix,:]).unsqueeze(0)).item()


	# Plot
	ax.plot(y_acq, c='b')
	ax.axhline(y=acq_trained.max(), c='r', ls='--', lw=0.5, label='max acq of trained points')

	ax.set_title(f'BO Step #{step}')
	ax.set_ylabel('acquisition')
	ax.set_xlabel('iteration')
	if step == step_num[0]:
		ax.legend()

	GRAPHICStools.addDenseAxis(ax)
	ax.set_ylim(top=0.0)

fn.show()

