"""
To compare mitim and TGYRO results
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from mitim_tools.gacode_tools import TGYROtools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.aux import BOgraphics
import dill as pickle_dill

# run ~/MITIM/mitim_opt/mitim/exe/comparemitim.py  ~/runs_portals/set6/run38/ ~/runs_portals/set6/run38/TGYRO_test/tgyro1_1/
# run comparemitim.py ~/PRF/mitim_dev/run1/ ~/PRF/mitim_dev/tgyro1_1/ mfews01.psfc.mit.edu-9224:runs_portals/set6/run38/


def compareSolvers(
    foldermitim, foldersTGYRO, labelsTGYRO, foldermitim_remote=None, fig=None
):
    plt.ion()
    if fig is None:
        fig = plt.figure(figsize=(6, 8))
    grid = plt.GridSpec(1, 2, hspace=0.3, wspace=0.3)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1], sharex=ax0)

    c = GRAPHICStools.listColors()

    # ----------------------------------------------------------------------------------------------------
    # ----- Plot TGYRO
    # ----------------------------------------------------------------------------------------------------

    resTGYRO = TGYROtools.TGYROoutput(foldersTGYRO[0])
    i = 0
    for folderTGYRO, labelTGYRO in zip(foldersTGYRO, labelsTGYRO):
        # ----- Read TGYRO
        try:
            resTGYRO = TGYROtools.TGYROoutput(folderTGYRO)
        except:
            continue

        iterations_array = resTGYRO.calls_solver / (resTGYRO.rho.shape[1] - 1)
        residual_array = resTGYRO.residual_manual_real
        ax0.plot(
            iterations_array,
            residual_array,
            "*-",
            color=c[i],
            label=labelTGYRO,
            lw=0.5,
            markersize=3,
        )
        i += 1

    # ----------------------------------------------------------------------------------------------------
    # ----- Plot mitim
    # ----------------------------------------------------------------------------------------------------

    opt_fun = STRATEGYtools.FUNmain(foldermitim)
    opt_fun.read_optimization_results(
        plotYN=False, folderRemote=foldermitim_remote, analysis_level=4
    )
    self_complete = opt_fun.plot_optimization_results(analysis_level=4, plotYN=False)
    with open(self_complete.MITIMextra, "rb") as handle:
        MITIMextra_dict = pickle_dill.load(handle)

    iterationsMultiplier = 1.0  # resTGYRO.rho.shape[1]-1
    iterationsOffset = 1  # resTGYRO.rho.shape[1]-1

    # Residual based on TGYRO definition
    resmitim = []
    iterations_arrayP = []
    for i in MITIMextra_dict:
        try:
            resmitim.append(
                MITIMextra_dict[i]["tgyro"]
                .results[tuple(MITIMextra_dict[i]["tgyro"].results.keys())[0]]
                .residual_manual_real[0]
            )
            iterations_arrayP.append(i)
        except:
            break
    resmitim = np.array(resmitim)
    iterations_arrayP = np.array(iterations_arrayP)

    ax0.plot(
        iterations_arrayP * iterationsMultiplier + iterationsOffset,
        resmitim,
        "s-",
        color="g",
        label="MITIM",
        lw=1.0,
        markersize=5,
    )

    # ----------------------------------------------------------------------------------------------------
    # Decor
    # ----------------------------------------------------------------------------------------------------

    plotPercent = 1e-2

    ax = ax0

    ax.legend(loc="upper right", prop={"size": 8})

    if iterationsMultiplier > 1.0:
        ax.set_xlabel("Calls to TGLF model")
    else:
        ax.set_xlabel("Profile evaluations (Calls to TGLF per radius)")
    if plotPercent > 0:
        GRAPHICStools.drawLineWithTxt(
            ax,
            resmitim[0] * plotPercent,
            label="$10^{-2}$x Original",
            orientation="horizontal",
            color="k",
            lw=0.5,
            ls="--",
            alpha=1.0,
            fontsize=8,
            fromtop=0.8,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="left",
            separation=0,
        )

    ax.set_ylabel("$\\widehat{L_1}$ Residue ($MW/m^2$), TGYRO definition")

    ax.set_yscale("log")
    ax.set_xlim(left=0.0)
    GRAPHICStools.addDenseAxis(ax)

    return ax0, ax1


if __name__ == "__main__":
    foldermitim = sys.argv[1]
    folderTGYRO = sys.argv[2]
    try:
        foldermitim_remote = sys.argv[3]
    except:
        foldermitim_remote = None

    ax0, ax1 = compareSolvers(
        foldermitim,
        [folderTGYRO],
        labelsTGYRO=["TGYRO"],
        foldermitim_remote=foldermitim_remote,
    )
