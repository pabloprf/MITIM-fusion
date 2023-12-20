import sys, argparse
import dill as pickle_dill
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals.aux import PORTALSplot

"""
This script is to plot only the convergence figure, not the rest of surrogates that takes long.
It also does it on a separate figure, so easy to manage (e.g. for saving as .eps)
"""

parser = argparse.ArgumentParser()
parser.add_argument("--folders", required=True, type=str, nargs="*")
parser.add_argument("--remote", type=str, required=False, default=None)

parser.add_argument(
    "--max", type=int, required=False, default=None
)  # Define max bounds of fluxes based on this one, like 0, -1 or None(best)
parser.add_argument("--index_extra", type=int, required=False, default=None)
parser.add_argument(
    "--all", type=bool, required=False, default=False
)  # Plot all fluxes?
parser.add_argument(
    "--file", type=str, required=False, default=None
)  # File to save .eps

args = parser.parse_args()


folders = args.folders


size = 8
plt.rc("font", family="serif", serif="Times", size=size)
plt.rc("xtick.minor", size=size)
plt.close("all")

for folderWork in folders:
    folderRemote_reduced = args.remote
    file = args.file
    indexToMaximize = args.max
    index_extra = args.index_extra
    plotAllFluxes = args.all

    folderRemote = (
        f"{folderRemote_reduced}/{IOtools.reducePathLevel(folderWork)[-1]}/"
        if folderRemote_reduced is not None
        else None
    )

    # Read standard OPT
    opt_fun = STRATEGYtools.FUNmain(folderWork)
    opt_fun.read_optimization_results(
        analysis_level=4, plotYN=False, folderRemote=folderRemote
    )

    # Analyze mitim results
    self_complete = opt_fun.plot_optimization_results(analysis_level=4, plotYN=False)

    # Interpret results
    with open(self_complete.MITIMextra, "rb") as handle:
        MITIMextra_dict = pickle_dill.load(handle)
    portals_plot = PORTALSplot.PORTALSresults(
        self_complete.folder,
        opt_fun.prfs_model,
        opt_fun.res,
        MITIMextra_dict=MITIMextra_dict,
        indecesPlot=[opt_fun.res.best_absolute_index, 0, index_extra],
    )

    # Plot
    plt.ion()
    fig = plt.figure(figsize=(18, 9))
    PORTALSplot.plotConvergencePORTALS(
        portals_plot,
        fig=fig,
        indexToMaximize=indexToMaximize,
        plotAllFluxes=plotAllFluxes,
    )

    # Save

if file is not None:
    plt.savefig(file, transparent=True, dpi=300)
