import argparse
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.utils import remote_tools
from mitim_tools.misc_tools.CONFIGread import read_dpi


# These import are usually needed if they are called within the pickling object
import torch  
import numpy as np


from IPython import embed

"""
This script is to read results of a MITIM optimization, and to compare among optimizations.

* Basic use:

	python3 read.py --type [analysis_level] [folder_run1, folder_run2, etc] [optional: --remote remoteFolder] [optional: --seeds 10] [optional: --conv 1E-3]
		- Full analysis (analysis level 2) performs analysis in the current machine for the base case and optimized case
		- If remote folder is provided, read from machine and copy stuff to folders
		- If more than one run specified, all of them are plotted in the SAME notebook, one tab color and
		  one tab-name prefix per run, plus a "Comparison" tab with the cross-run residual/timing comparison.
		- --simple skips the full notebook and produces only that cross-run comparison figure.

* Examples:

	Local:
		 run ~/MITIM/mitim_opt/opt_tools/scripts/read.py run44 run45 --type -1

	Remote:
		run ~/MITIM/mitim_opt/opt_tools/scripts/read.py run44 run45 --type -1 --remote eofe7.mit.edu:/nobackup1/pablorf/runs_portals/dev/ 
		run ~/MITIM/mitim_opt/opt_tools/scripts/read.py run44 --type 2 --remote  mferws01.psfc.mit.edu-9224:/nobackup1/pablorf/runs_portals/dev/

* Notes:
	- Analysis higher than 2 may be enabling other options for mitim and others
	- Seeds indicate that the slurm was run with --seeds, so folders of the type of run44_s0...9 (if 10 seeds) will be searched for
	- Save full notebook to --save folder

"""

def plotCompare(folders, plotMeanMax=[True, False], fig=None):
    '''
    Cross-run comparison of residual improvement and timings.
    If `fig` is provided (e.g. a tab of an existing notebook), it is populated
    instead of creating a standalone figure (so other figures are not closed).
    '''
    folderWorks = []
    names = []
    for cont, i in enumerate(folders):
        folderWorks.append(IOtools.expandPath(i))
        names.append("run" + f"{folderWorks[-1].name}".replace("/", "").split("run")[-1])
    colors = GRAPHICStools.listColors()

    if fig is None:
        plt.close("all")
        fig = plt.figure(figsize=(16, 10))
    grid = plt.GridSpec(3, 2, hspace=0.2, wspace=0.1)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1],sharex=ax2)
    ax1i = fig.add_subplot(grid[2, 0], sharex=ax0)

    types_ls = GRAPHICStools.listLS()
    types_m = GRAPHICStools.listmarkers

    maxEv = -np.inf
    yCummMeans = []
    xes = []
    resS = []
    for i, (color, name, folderWork) in enumerate(zip(colors, names, folderWorks)):
        res = BOgraphics.optimization_results(folderWork / "Outputs" / "optimization_results.out")
        res.readClass(STRATEGYtools.read_from_scratch(folderWork / "Outputs" / "optimization_object.pkl"))
        res.read()

        plotAllmembers = len(folderWorks) <= 3
        xe, yCummMean = res.plotImprovement(
            axs=[ax0, ax1, ax1i, None],
            color=color,
            extralab=name + " ",
            plotAllmembers=plotAllmembers,
            plotMeanMax=plotMeanMax,
        )
        if xe[-1] > maxEv:
            maxEv = xe[-1]

        #compared = -yCummMean[0] * conv if conv < 0 else conv
        #ax1.axhline(y=compared, ls="-.", lw=0.3, color=color)

        IOtools.plot_timings(
            folderWork / "Outputs" / "timing.jsonl", axs=[ax2, ax3], label=name, color=color
        )

        yCummMeans.append(yCummMean)
        xes.append(xe)
        resS.append(res)

    ax0.set_xlim([0, maxEv])

    ax2.legend(prop={"size": 6})
    ax3.legend(prop={"size": 6})

    return yCummMeans, xes, resS, fig


def main():

# ----- Inputs

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, required=False, default=2)  # 0: Only ResultsOpt plotting, 1: Also pickle, 2: Also all, 3: additional, 4: addtional + extra
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--seeds", type=int, required=False, default=None)
    parser.add_argument("--resolution", type=int, required=False, default=50)
    parser.add_argument("--conv", type=float, required=False, default=-1e-2)
    parser.add_argument("--its", type=int, nargs="*", required=False, default=None)

    parser.add_argument("--simple", required=False, default=False, action="store_true",
                        help="If set, do not build the full notebook; only produce the simple cross-run comparison figure (residual improvement + timings).")

    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")
    parser.add_argument("--save", type=str, nargs="?", const=IOtools.SAVE_FOLDER_AUTO_SENTINEL, required=False, default=None,
                        help=f"Folder to save the figures. If flag given without a value, defaults to '<first folder>/{IOtools.SAVE_FOLDER_DEFAULT_SUBDIR}'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")

    # Remote options
    parser.add_argument("--remote",type=str, required=False, default=None,
                        help="Remote machine to retrieve the folders from. If not provided, it will read the local folders.")
    parser.add_argument("--remote_folder_parent",type=str, required=False, default=None,
                        help="Parent folder in the remote machine where the folders are located. If not provided, it will use --remote_folders.")
    parser.add_argument("--remote_folders",type=str, nargs="*", required=False, default=None,
                        help="List of folders in the remote machine to retrieve. If not provided, it will use the local folder structures.")
    # parser.add_argument("--remote_minimal", required=False, default=False, action="store_true",
    #                     help="If set, it will only retrieve the folder structure with a few key files")
    parser.add_argument('--fix', required=False, default=False, action='store_true',
                        help="If set, it will fix the pkl optimization portals in the remote folders.")

    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    if args.save == IOtools.SAVE_FOLDER_AUTO_SENTINEL and not args.folders and not (args.remote_folder_parent or args.remote_folders):
        parser.error("--save without a value needs at least one positional folder argument")

    analysis_level = args.type
    seeds = args.seeds
    resolution = args.resolution
    conv = args.conv
    rangePlot = args.its

    noshow = args.noshow
    dpi_fig = args.dpi

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Retrieve from remote
    # --------------------------------------------------------------------------------------------------------------------------------------------

    folders = remote_tools.retrieve_remote_folders(args.folders, args.remote, args.remote_folder_parent, args.remote_folders, None)

    folder_save = IOtools.resolve_save_folder(args.save, folders[0] if folders else None)

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Fix pkl optimization portals in remote
    # --------------------------------------------------------------------------------------------------------------------------------------------

    if args.fix:
        for folder in folders:
            STRATEGYtools.clean_state(folder)


    folders_complete = folders
    several_runs = len(folders_complete) > 1

    # --simple: only read the results (no per-run plotting), so that the sole
    # output is the cross-run comparison figure
    if args.simple:
        retrieval_level = copy.deepcopy(analysis_level)
        analysis_level = -1
    else:
        retrieval_level = analysis_level

    print(f"(Analysis level {analysis_level})\n")

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Plot each run, all of them into the SAME notebook (one color and one tab-name prefix per run)
    # --------------------------------------------------------------------------------------------------------------------------------------------

    fn = GUItools.FigureNotebook("MITIM Optimization Results", show=not noshow) if (analysis_level >= 0) else None

    opt_funs = []
    for i, folderWork in enumerate(folders_complete):
        opt_fun = STRATEGYtools.opt_evaluator(folderWork)
        if fn is not None:
            opt_fun.fn = fn
            fn.label_prefix = f"{folderWork.name}: " if several_runs else ""

        plotting_arguments = {
            "analysis_level": analysis_level,
            "retrieval_level": retrieval_level,
            "pointsEvaluateEachGPdimension": resolution,
            "rangesPlot": rangePlot,
            "noshow": noshow,
            "tabs_colors": i if several_runs else None,
        }

        if several_runs:
            # One broken run must not prevent the others from being plotted
            try:
                opt_fun.plot_optimization_results(**plotting_arguments)
            except Exception as e:
                print(f"Could not retrieve #{folderWork}: {e}", typeMsg="w")
            if fn is not None:
                fn.tab_color_forced = None  # in case the run died mid-plotting
        else:
            opt_fun.plot_optimization_results(**plotting_arguments)

        opt_funs.append(opt_fun)

    if fn is not None:
        fn.label_prefix = ""

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Cross-run comparison (the only figure produced with --simple, an extra first tab otherwise)
    # --------------------------------------------------------------------------------------------------------------------------------------------

    fig = yCummMeans = xes = resS = None

    if several_runs or (analysis_level == -1):
        # Figure created outside the notebook and only added as a tab if the comparison
        # succeeds, so that a failure does not leave an empty tab behind
        fig_compare = plt.figure(figsize=(16, 10), dpi=read_dpi()) if fn is not None else None
        try:
            yCummMeans, xes, resS, fig = plotCompare(
                folders_complete, plotMeanMax=[True, not several_runs], fig=fig_compare
            )
            if fn is not None:
                fn.addPlot("Comparison", fig)
                # Cross-run comparison first, then each run's tabs
                if not fn._headless:
                    fn.move_tabs_block_to_front(fn.tabs.count() - 1, 1)
        except Exception as e:
            print(f"Could not produce the cross-run comparison: {e}", typeMsg="w")
            if fig_compare is not None:
                plt.close(fig_compare)
            fig = None

    # ------
    if seeds is not None and xes is not None:
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

            ax.set_xlabel("Number of evaluations to converge")
            # ax.set_title(f'Residual reduced by x{1/percent:.0f}')
            ax.set_xlim([0, 50])

            GRAPHICStools.addDenseAxis(ax)
        else:
            print(f"Could not produce Violin-plot because no point reached the convergence criterion (factor of {percent})",typeMsg="w",)
            
            
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Show figures?
    # --------------------------------------------------------------------------------------------------------------------------------------------
    
    if not noshow:
        if fn is not None:
            fn.show()
        else:
            plt.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Save figures?
    # --------------------------------------------------------------------------------------------------------------------------------------------

    if folder_save:
        if fn is not None:
            # Use Notebook save method, to collect all figures into a single folder
            fn.save(folder_save, dpi=dpi_fig)
        else:
            if not folder_save.exists():
                folder_save.mkdir(parents=True)
            GRAPHICStools.output_figure_papers(f"{folder_save}/figure", fig=fig, dpi=dpi_fig)
            
    embed()

if __name__ == "__main__":
    main()
