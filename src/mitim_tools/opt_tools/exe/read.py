import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.opt_tools.aux import BOgraphics
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

from IPython import embed

verbose_level = read_verbose_level()

"""
This script is to read results of a MITIM optimization, and to compare among optimizations.

* Basic use:

	python3 read.py --type [analysis_level] [optional: --remote remoteFolder] --folders [folder_run1, folder_run2, etc] [optional: --seeds 10] [optional: --conv 1E-3]
		- Full analysis (analysis level 2) performs analysis in the current machine for the base case and optimized case
		- If remote folder is provided, read from machine and copy stuff to folders
		- If more than one run specified, analysis_level will forced to -1, meaning... only compare them.

* Examples:

	Local:
		 run ~/MITIM/mitim_opt/opt_tools/exe/read.py --type -1 --folders run44 run45

	Remote:
		run ~/MITIM/mitim_opt/opt_tools/exe/read.py --type -1 --remote eofe7.mit.edu:/nobackup1/pablorf/runs_portals/dev/ --folders run44 run45
		run ~/MITIM/mitim_opt/opt_tools/exe/read.py --type 2 --remote  mferws01.psfc.mit.edu-9224:/nobackup1/pablorf/runs_portals/dev/ --folders run44

* Notes:
	- Analysis higher than 2 may be enabling other options for mitim and others
	- Seeds indicate that the slurm was run with --seeds, so folders of the type of run44_s0...9 (if 10 seeds) will be searched for
	- Save full notebook to --save folder

"""

# ----- Inputs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type", type=int, required=False, default=-1
)  # 0: Only ResultsOpt plotting, 1: Also pickle, 2: Also final analysis, 3: Others
parser.add_argument("folders", type=str, nargs="*")
parser.add_argument("--remote","-r", type=str, required=False, default=None)
parser.add_argument("--seeds", type=int, required=False, default=None)
parser.add_argument("--resolution", type=int, required=False, default=50)
parser.add_argument("--save", type=str, required=False, default=None)
parser.add_argument("--conv", type=float, required=False, default=-1e-2)

args = parser.parse_args()

analysis_level = args.type
folders_reduced = args.folders
folderRemote_reduced = args.remote
seeds = args.seeds
resolution = args.resolution
save_folder = args.save
conv = args.conv

# -----------------------------------------


def plotCompare(folders, plotMeanMax=[True, False]):
    folderWorks = []
    names = []
    for cont, i in enumerate(folders):
        folderWorks.append(IOtools.expandPath(i + "/"))
        names.append("run" + folderWorks[-1].replace("/", "").split("run")[-1])
    colors = GRAPHICStools.listColors()

    plt.close("all")
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    grid = plt.GridSpec(3, 2, hspace=0.2, wspace=0.1)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1])
    ax1i = fig.add_subplot(grid[2, 0], sharex=ax0)

    types_ls = [
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        ":",
    ]
    types_m = [
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
        "o",
        "s",
        "^",
        "v",
        "*",
    ]

    maxEv = -np.inf
    yCummMeans = []
    xes = []
    resS = []
    logS = []
    for i, (color, name, folderWork) in enumerate(zip(colors, names, folderWorks)):
        res = BOgraphics.ResultsOptimization(
            f"{folderWork}/Outputs/ResultsOptimization.out"
        )
        res.readClass(
            STRATEGYtools.read_from_scratch(f"{folderWork}/Outputs/MITIMstate.pkl")
        )
        res.read()

        log_class = BOgraphics.LogFile(folderWork + "/Outputs/MITIM.log")

        try:
            log_class.interpret()
        except:
            print("Could not read log", typeMsg="w")
            log_class = None

        plotAllmembers = len(folderWorks) <= 3
        xe, yCummMean = res.plotImprovement(
            axs=[ax0, ax1, ax1i],
            color=color,
            extralab=name + " ",
            plotAllmembers=plotAllmembers,
            plotMeanMax=plotMeanMax,
        )
        if xe[-1] > maxEv:
            maxEv = xe[-1]

        compared = -yCummMean[0] * conv if conv < 0 else conv

        ax1.axhline(y=compared, ls="-.", lw=0.3, color=color)

        if log_class is not None:
            log_class.plot(
                axs=[ax2, ax3],
                ls=types_ls[i],
                lab=name,
                marker=types_m[i],
                color=colors[i],
            )

        yCummMeans.append(yCummMean)
        xes.append(xe)
        resS.append(res)
        logS.append(log_class)

    ax0.set_xlim([0, maxEv])

    ax2.legend(prop={"size": 6})
    ax3.legend(prop={"size": 6})

    return yCummMeans, xes, resS, logS, fig


folders_reduced_original = copy.deepcopy(folders_reduced)

if seeds is not None:
    for i in range(len(folders_reduced)):
        if folders_reduced[i][-1] == "/":
            folders_reduced[i] = folders_reduced[i][:-1]

        aux = [f"{folders_reduced[i]}_s{k}" for k in range(seeds)]
        del folders_reduced[i]
        folders_reduced.extend(aux)

foldersWork = [
    IOtools.expandPath(folder_reduced + "/", ensurePathValid=True)
    for folder_reduced in folders_reduced
]
reduced_folders = [
    IOtools.reducePathLevel(folderWork)[-1] for folderWork in foldersWork
]

if len(foldersWork) > 1:
    retrieval_level = copy.deepcopy(analysis_level)
    analysis_level = -1
else:
    retrieval_level = analysis_level

txt = f"***************************************************************************\n"
for folderWork in foldersWork:
    txt += f"* Reading results in {folderWork}\n"

if folderRemote_reduced is None:
    foldersRemote = [None] * len(foldersWork)
else:
    foldersRemote = [
        f"{folderRemote_reduced}/{reduced_folder}/"
        for reduced_folder in reduced_folders
    ]
    txt += f"\n\t...From remote folder {folderRemote_reduced}\n"

print(
    "\n"
    + txt
    + "***************************************************************************"
)
print(f"(Analysis level {analysis_level})\n")

if len(foldersWork) == 1:
    opt_fun = STRATEGYtools.FUNmain(foldersWork[0])
    opt_fun.plot_optimization_results(
        analysis_level=analysis_level,
        folderRemote=foldersRemote[0],
        retrieval_level=retrieval_level,
        pointsEvaluateEachGPdimension=resolution,
        save_folder=save_folder,
    )
else:
    opt_funs = []
    for folderWork, folderRemote in zip(foldersWork, foldersRemote):
        opt_fun = STRATEGYtools.FUNmain(folderWork)
        try:
            opt_fun.plot_optimization_results(
                analysis_level=analysis_level,
                folderRemote=folderRemote,
                retrieval_level=retrieval_level,
                save_folder=save_folder,
            )
        except:
            print(f"Could not retrieve #{folderWork}", typeMsg="w")
        opt_funs.append(opt_fun)

if analysis_level == -1:
    yCummMeans, xes, resS, logS, fig = plotCompare(
        foldersWork, plotMeanMax=[True, len(foldersWork) < 2]
    )

# ------
if seeds is not None:
    grid = plt.GridSpec(3, 2, hspace=0.2, wspace=0.1)
    ax = fig.add_subplot(grid[2, 1])
    percent = 1e-2

    xf = []
    for i in range(len(xes)):
        try:
            compared = -yCummMeans[i][0] * conv if conv < 0 else conv
            xf.append(xes[i][yCummMeans[i] < compared][0])
        except:
            pass  # xf.append(np.nan)
    xf = np.array(xf)

    if xf.shape[0] > 0:
        print(f"Plotting Violin with {xf.shape[0]} points")
        GRAPHICStools.plotViolin([xf], labels=["run"], ax=ax, colors=["b"])

        ax.set_xlabel(f"Number of evaluations to converge")
        # ax.set_title(f'Residual reduced by x{1/percent:.0f}')
        ax.set_xlim([0, 50])

        GRAPHICStools.addDenseAxis(ax)
    else:
        print(
            f"Could not produce Violin-plot because no point reached the convergence criterion (factor of {percent})",
            typeMsg="w",
        )
