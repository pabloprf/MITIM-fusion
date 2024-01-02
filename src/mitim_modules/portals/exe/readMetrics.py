import argparse
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools
from mitim_modules.portals.aux import PORTALSanalysis

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

    # Read PORTALS
    portals = PORTALSanalysis.PORTALSanalyzer.from_folder(
        folderWork, folderRemote=folderRemote
    )
    
    portals.plotMetrics(
        indexToMaximize=indexToMaximize,
        plotAllFluxes=plotAllFluxes,
        index_extra=index_extra,
        file_save=file,
    )
