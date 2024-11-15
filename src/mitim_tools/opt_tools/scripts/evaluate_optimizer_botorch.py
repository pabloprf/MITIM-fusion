import datetime
import argparse
import botorch
import dill as pickle_dill
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools, BOTORCHtools, OPTtools

from IPython import embed

"""
This script performs a series of tests using BOTORCH optimizer for the last step of the MITIM folder.
e.g.
    optimizer_tester.py --folder run1/ --test 1 --save True --seeds 10 --var 9
"""

# ***************************************************************************************************
# Inputs
# ***************************************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--test", required=False, type=int, default=1)
parser.add_argument("--save", required=False, default=False, action="store_true")
parser.add_argument("--seeds", required=False, type=int, default=1)
parser.add_argument("--var", required=False, type=int, default=9)
args = parser.parse_args()

folder = IOtools.expandPath(args.folder)
test = args.test
save = args.save
numSeeds = args.seeds
numVariations = args.var

# ***************************************************************************************************
# Preparation
# ***************************************************************************************************

opt_fun = STRATEGYtools.opt_evaluator(folder)
opt_fun.read_optimization_results(analysis_level=4)
step = opt_fun.prfs_model.steps[-1]

fun = OPTtools.fun_optimization(
    step.stepSettings, step.evaluators, step.StrategyOptions_use
)
fun.prep()

# ***************************************************************************************************
# Options
# ***************************************************************************************************

# ---------------
rws_exponent = 10  # 1024
nr_exponent = 6  # 64
# ---------------

if test == 1:
    print(f"Testing one case ({int(numSeeds)} seeds")

    nr = 2**nr_exponent
    rws = 2**rws_exponent

    timeout_sec = None
    its = 100

    num_restarts_cases = np.array([nr])
    iterations_cases = np.array([its])
    raw_samples_cases = np.array([rws])

    title = ""

    numVariations = 1

elif test == 2:
    rws = 2**rws_exponent

    timeout_sec = 30
    its = None

    num_restarts_cases = 2 ** np.linspace(
        3, rws_exponent, numVariations
    )  # max number of cold_starts has to be the raw_samples
    iterations_cases = np.array([its] * numVariations)
    raw_samples_cases = np.ones(numVariations) * rws

    title = f"{int(numVariations)} cases ({int(numSeeds)} seeds): raw_samples = {rws}, timeout_sec = {timeout_sec}"

elif test == 3:
    nr = 2**nr_exponent

    timeout_sec = None
    its = 20

    print(f"Testing raw_samples with num_restarts = {nr}, iterations = {its}")
    raw_samples_cases = 2 ** np.linspace(nr_exponent, 14, numVariations)
    num_restarts_cases = np.ones(numVariations) * nr
    iterations_cases = np.ones(numVariations) * its

    title = f"{int(numVariations)} cases ({int(numSeeds)} seeds): num_restarts = {nr}, iterations = {its}"

print(
    "\n*************************************\n",
    title,
    "\n*************************************\n",
)
seeds = np.linspace(0, 100, numSeeds)

# ***************************************************************************************************
# OPTIMIZER
# ***************************************************************************************************

batch_initial_conditions = fun.evaluators["GP"].train_X  # fun.xGuesses
ic_generator = BOTORCHtools.ic_generator_wrapper(batch_initial_conditions)

times = []
yacq = []
traj = []
for i, (num_restarts, iterations, raw_samples) in enumerate(
    zip(num_restarts_cases, iterations_cases, raw_samples_cases)
):
    times0 = []
    yacq0 = []
    traj0 = []
    for seed in seeds:
        arr = []

        def fun_opt(x, v=arr):
            f = fun.evaluators["acq_function"](x)
            v.append(f.max().item())
            return f

        # fun_opt = fun.evaluators['acq_function']

        options = {"sample_around_best": True, "disp": True, "seed": int(seed)}
        if iterations is not None:
            options["maxiter"] = int(iterations)

        timeBeginning = datetime.datetime.now()
        x_opt, _ = botorch.optim.optimize_acqf(
            acq_function=fun_opt,
            bounds=fun.bounds,
            options=options,
            raw_samples=int(raw_samples),
            ic_generator=None,  # ic_generator,
            batch_initial_conditions=None,
            q=1,
            num_restarts=int(num_restarts),
            return_best_only=True,
            timeout_sec=timeout_sec,
        )
        time = IOtools.getTimeDifference(timeBeginning, niceText=False)

        y_res = OPTtools.summarizeSituation(x_opt, fun, x_opt)[0]

        traj0.append(arr)
        times0.append(time)
        yacq0.append(y_res)

    yacq.append(yacq0)
    times.append(times0)
    traj.append(traj0)

    print(
        f"\nnum_restarts = {num_restarts}, raw_samples = {raw_samples} time = {time:.1f}, acq = {y_res:.3e}"
    )

# ***************************************************************************************************
# Postprocess
# ***************************************************************************************************

ybest = -np.array(yacq)  # [variation,seeds]
ybest_m = ybest.mean(axis=1)  # [variations]
ybest_s = ybest.std(axis=1)  # [variations]

timeTotal = np.array(times)  # [variation,seeds]
timeTotal_m = timeTotal.mean(axis=1)  # [variations]
timeTotal_s = timeTotal.std(axis=1)  # [variations]

# ***************************************************************************************************
# Plotting
# ***************************************************************************************************

from mitim_tools.misc_tools.GRAPHICStools import fillGraph

alpha = 0.1
lw = 1.0
lw_seeds = 1.0
alpha_seeds = 0.5
plt.close("all")
plt.ion()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
cols, _ = GRAPHICStools.colorTableFade(
    len(num_restarts_cases), startcolor="b", endcolor="r"
)

ax = axs[0, 0]
if test in [1, 2]:
    x, tit = num_restarts_cases, "Number of cold_starts"
else:
    x, tit = raw_samples_cases, "Raw Samples"

fillGraph(
    ax,
    x,
    ybest_m,
    ls="-o",
    ms=5,
    y_down=ybest_m - ybest_s,
    y_up=ybest_m + ybest_s,
    alpha=alpha,
    color="r",
    lw=lw,
    islwOnlyMean=True,
)

ax.set_xlabel(tit)
ax.set_xscale("log")
ax.set_ylabel("Acquisition function value (to minimize)")
# ax.set_yscale('log')
GRAPHICStools.addDenseAxis(ax)
ax.set_title(title)

ax = axs[1, 0]
if test in [1, 2]:
    x, tit = num_restarts_cases, "Number of cold_starts"
else:
    x, tit = raw_samples_cases, "Raw Samples"

fillGraph(
    ax,
    x,
    timeTotal_m,
    ls="-o",
    ms=5,
    y_down=timeTotal_m - timeTotal_s,
    y_up=timeTotal_m + timeTotal_s,
    alpha=alpha,
    color="r",
    lw=lw,
    islwOnlyMean=True,
)

ax.set_xlabel(tit)
ax.set_xscale("log")
ax.set_ylabel("Total time consumed (seconds)")
# ax.set_yscale('log')
GRAPHICStools.addDenseAxis(ax)

for i in range(numVariations):
    label = (
        f"num_restarts = {int(num_restarts_cases[i])}"
        if test == 2
        else f"raw_samples = {int(raw_samples_cases[i])}" if test == 3 else ""
    )

    for s in range(numSeeds):
        y1 = -np.array(traj[i][s])
        x1 = np.linspace(0, y1.shape[0], y1.shape[0])
        axs[0, 1].plot(x1, y1, c=cols[i], lw=lw_seeds, label=label if s == 0 else "")

        x1 = np.linspace(0, timeTotal[i, s], y1.shape[0])
        axs[1, 1].plot(x1, y1, c=cols[i], lw=lw_seeds)

ax = axs[0, 1]
ax.set_xlabel("Calls to evaluator")
# ax.set_xscale('l')
ax.set_ylabel("Acquisition function value (to minimize)")
ax.set_yscale("log")
GRAPHICStools.addDenseAxis(ax)
GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=True, size=7)

ax = axs[1, 1]
ax.set_xlabel("Time (seconds) - Equally splitted")
# ax.set_xscale('l')
ax.set_ylabel("Acquisition function value (to minimize)")
ax.set_yscale("log")
GRAPHICStools.addDenseAxis(ax)
GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=False, size=7)


if save:
    name = folder / f"test{test}_seeds{numSeeds}_variations{numVariations}.pkl"
    with open(name, "wb") as handle:
        pickle_dill.dump({"traj": traj, "ybest": ybest, "timeTotal": timeTotal}, handle, protocol=4)
