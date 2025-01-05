import torch
import argparse
import copy
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools, OPTtools
from mitim_tools.opt_tools.utils import TESTtools
from mitim_tools.opt_tools.optimizers import ROOTtools
from IPython import embed

"""
This script performs a series of tests using BOTORCH optimizer for the last step of the MITIM folder.
e.g.
	optimizer_tester.py --folder run1/ [--step 0]
"""

# ***************************************************************************************************
# Inputs
# ***************************************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--step", required=False, type=int, default=-1)
args = parser.parse_args()

folder = IOtools.expandPath(args.folder)
step = args.step

# ***************************************************************************************************
# Preparation
# ***************************************************************************************************

opt_fun = STRATEGYtools.opt_evaluator(folder)
opt_fun.read_optimization_results(analysis_level=4)
step = opt_fun.mitim_model.steps[step]

fun = OPTtools.fun_optimization(
    step.stepSettings, step.evaluators, step.strategy_options_use
)
fun.prep()

# xGuesses 			= fun.evaluators['GP'].train_X[:-1,:] #[-1,:]
# plotX 				= False

xGuesses = fun.evaluators["GP"].train_X[0, :]
plotX = True

plt.ion()
if plotX:
    fig, axs = plt.subplots(nrows=3, sharex=True)
else:
    fig, axs = plt.subplots(nrows=1, sharex=True)
    axs = [axs]
colors = GRAPHICStools.listColors()
colors1 = GRAPHICStools.listColors()

bounds_logi = fun.bounds_mod

# for opt,lab in enumerate(['vectorize=True']): #,'vectorize=False']):
for opt, lab in enumerate(["x0=0"]):  # ,'x0=1.0']): #,'vectorize=False']):
    logi = ROOTtools.logistic(l=bounds_logi[0, :], u=bounds_logi[1, :], k=0.5, x0=0)
    # ***************************************************************************************************
    # OPTIMIZER
    # ***************************************************************************************************

    def channel_residual_evaluator(x):
        X = x.view(
            (x.shape[0] // xGuesses.shape[-1], xGuesses.shape[-1])
        )  # [batch*dim]->[batch,dim]

        X = logi.transform(X)

        # *** Evaluate residuals ***
        yM, y1, y2, _ = fun.evaluators["residual_function"](X, outputComponents=True)
        y = y1 - y2

        # Compress again  [batch,dim] -> [batch*dim]
        y = y.view(x.shape)

        return y

    x_evaluated0, acq_evaluated0 = [], []

    def func(x):
        # Root will work with arrays, convert to tensor with AD
        X = torch.tensor(x, requires_grad=True).to(xGuesses)

        # Evaluate value and local jacobian
        QhatD = channel_residual_evaluator(X)
        JD = torch.autograd.functional.jacobian(
            channel_residual_evaluator, X, strict=False, vectorize=True
        )

        acq_evaluated0.append(QhatD.detach().cpu().numpy())
        x_evaluated0.append(copy.deepcopy(x))

        # Back to arrays
        return QhatD.detach().cpu().numpy(), JD.detach().cpu().numpy()

    xGuesses = logi.untransform(xGuesses)

    xGuesses_extended = xGuesses.view(-1).unsqueeze(0)

    xGuess0 = (
        xGuesses_extended.squeeze(0).cpu().numpy()
        if xGuesses_extended.dim() > 1
        else xGuesses_extended.cpu().numpy()
    )

    # ************
    # Root process
    # ************

    f0 = func(xGuess0)[0]
    print(
        f"\t- |f-fT|*w (mean = {np.mean(np.abs(f0)):.3e} of {f0.shape[0]} channels):\n\t{f0}"
    )

    with IOtools.speeder(f"profiler_root.prof"):
        sol = scipy.optimize.root(
            func, xGuess0, jac=True, method="lm", tol=None, options={"disp": True}
        )

    f = func(sol.x)[0]
    print(
        f"\t- |f-fT|*w (mean = {np.mean(np.abs(f)):.3e} of {f.shape[0]} channels):\n\t{f}"
    )
    # ************

    x_best0 = torch.tensor(sol.x).to(xGuesses)

    x_best = x_best0.view((x_best0.shape[0] // xGuesses.shape[-1], xGuesses.shape[-1]))

    x_best = logi.transform(x_best)

    yM = fun.evaluators["residual_function"](x_best)
    print(f"MAXIMUM: {yM.max().item():.3e}")

    print(
        f"inside = {TESTtools.checkSolutionIsWithinBounds(x_best,fun.bounds).item()}",
        sol,
    )

    # ***************************************************************************************************
    # Postprocess
    # ***************************************************************************************************

    acq_evaluated1 = torch.Tensor(acq_evaluated0).to(xGuesses)

    acq_evaluated1 = acq_evaluated1.view(
        (
            acq_evaluated1.shape[0],
            x_best0.shape[0] // xGuesses.shape[-1],
            xGuesses.shape[-1],
        )
    )

    acq_evaluated = acq_evaluated1.abs().mean(axis=-1)
    x_evaluated = torch.Tensor(x_evaluated0).to(xGuesses)

    # ***************************************************************************************************
    # Plotting
    # ***************************************************************************************************
    if plotX:
        ax = axs[opt]
        for i in range(x_evaluated.shape[-1]):
            val = x_evaluated[:, i] / xGuesses[i]
            ax.plot(val.cpu(), "-o", lw=0.5, c=colors[i], markersize=2)

    ax = axs[-1]
    for i in range(acq_evaluated.shape[-1]):
        ax.plot(
            acq_evaluated[:, i].cpu(), "-o", markersize=4, label=lab + f" {i}"
        )  # ,c=colors1[opt])

axs[-1].set_yscale("log")
axs[-1].legend()

if plotX:
    bounds_upper = fun.bounds[1, :] / xGuesses
    bounds_lower = fun.bounds[0, :] / xGuesses

    ax = axs[0]
    for i in range(xGuesses.shape[-1]):
        ax.axhline(y=bounds_upper[i].cpu(), ls="--", lw=0.5, c=colors[i])
        ax.axhline(y=bounds_lower[i].cpu(), ls="--", lw=0.5, c=colors[i])

    ax = axs[1]
    for i in range(xGuesses.shape[-1]):
        ax.axhline(y=bounds_upper[i].cpu(), ls="--", lw=0.5, c=colors[i])
        ax.axhline(y=bounds_lower[i].cpu(), ls="--", lw=0.5, c=colors[i])
