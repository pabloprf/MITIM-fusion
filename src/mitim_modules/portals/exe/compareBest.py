"""
This plots final profiles
"""

import sys, argparse, copy
import matplotlib.pyplot as plt
import dill as pickle_dill
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.opt_tools import STRATEGYtools


parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="*")
args = parser.parse_args()
folders = args.folders


p = []
for folderWork in folders:
    opt_fun = STRATEGYtools.FUNmain(folderWork)
    opt_fun.read_optimization_results(analysis_level=4)

    with open(opt_fun.prfs_model.MITIMextra, "rb") as handle:
        MITIMextra_dict = pickle_dill.load(handle)

    MITIMextra_dict[opt_fun.res.best_absolute_index]["tgyro"].profiles
    p.append(MITIMextra_dict[opt_fun.res.best_absolute_index]["tgyro"].profiles)

labels = []
for i in folders:
    labels.append(i.split("/")[-1])

plt.ion()
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 9))

colors = GRAPHICStools.listColors()

# porig = MITIMextra_dict[0]['tgyro'].profiles
# porig.plotGradients(axs.flatten(),color='k',label='original',lastRho=MITIMextra_dict[opt_fun.res.best_absolute_index]['tgyro'].rhosToSimulate[-1])

for p0, color, label in zip(p, colors, labels):
    p0.plotGradients(
        axs.flatten(),
        color=color,
        label=label,
        lastRho=MITIMextra_dict[opt_fun.res.best_absolute_index][
            "tgyro"
        ].rhosToSimulate[-1],
    )

for ax in axs.flatten():
    for i in MITIMextra_dict[opt_fun.res.best_absolute_index]["tgyro"].rhosToSimulate:
        ax.axvline(x=i, ls="--", c="k", lw=0.2)

axs[0, 0].legend()

for ax in axs.flatten():
    ax.set_ylim(bottom=0)
