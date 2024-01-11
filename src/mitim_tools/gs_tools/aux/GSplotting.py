import copy, pandas, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import freegs
from mitim_tools.gs_tools import FREEGStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import GRAPHICStools, MATHtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools.aux import BOgraphics

from mitim_tools.misc_tools.IOtools import printMsg as print

from IPython import embed


def writeResults(
    whereOutput,
    prfs,
    metrics,
    function_parameters,
    divPlate=None,
    namePkl="run",
    plotGs=False,
    strikes=True,
    labelEq="time (s)",
    params=None,
    fn=None,
):
    if not os.path.exists(whereOutput):
        IOtools.askNewFolder(whereOutput, force=False, move=None)

    metrics["prfs"], metrics["params"], metrics["function_parameters"] = (
        prfs,
        params,
        function_parameters,
    )

    # ------------- Write table
    filename = "Summary"
    writeTable(
        metrics,
        f"{whereOutput}/{filename}.xlsx",
        metrics["params"]["times"],
        divPlate=divPlate,
        strikes=strikes,
        labelEq=labelEq,
    )

    # ------------- Write g-files
    gs = write_gfiles(prfs, whereOutput)

    if plotGs:
        GEQtools.compareGeqdsk(gs, fn=fn, plotAll=True, labelsGs=None)

    # ------------- Pickles
    file = f"{whereOutput}/{namePkl}.pkl"
    try:
        with open(file, "wb") as handle:
            pickle.dump(metrics, handle, protocol=2)
    except:
        for i in range(len(prfs)):
            del metrics["prfs"][i].sweep.eq
            del metrics["prfs"][i].sweep.profiles.eq
        with open(file, "wb") as handle:
            pickle.dump(metrics["prfs"][i].sweep, handle, protocol=2)

    return gs


def plotAllEquilibria(
    ax,
    prf_classes,
    ProblemExtras,
    legYN=True,
    extralab="",
    startcol=0,
    colors=None,
    onlySeparatrix=False,
    plotXpoints=True,
    lwSep=1,
    plotDIV=True,
    plotLIM=True,
    plotStrike=True,
    limitContours=True,
    plotSOL=False,
    plotAxis=False,
):
    if colors is None:
        cols = GRAPHICStools.listColors()
    else:
        cols = colors
    cols = cols[startcol:]

    if type(prf_classes) != list:
        prf_classes = [prf_classes]

    zorderBase = 0

    if plotLIM:
        prf_classes[0].lim.plot(ax)
    if plotDIV:
        prf_classes[0].div.plot(ax, color=cols[0])

    colorsChosen = []
    for cont, prf in enumerate(prf_classes):
        if limitContours:
            limitPatch = [prf.lim.limRZ, prf.sweep.Rgrid, prf.sweep.Zgrid]
        else:
            limitPatch = None

        colorsChosen.append(cols[cont])
        prf.sweep.plot(
            ax,
            color=cols[cont],
            onlySeparatrix=onlySeparatrix,
            plotXpoints=plotXpoints,
            plotStrike=plotStrike,
            lwSep=lwSep,
            limitPatch=limitPatch,
            zorderBase=zorderBase,
            plotSOL=plotSOL,
            plotAxis=plotAxis,
        )

    if ProblemExtras is not None and "HHF" in ProblemExtras:
        ax.plot(
            ProblemExtras["HHF"][0],
            ProblemExtras["HHF"][1],
            "s",
            c="c",
            markersize=4,
            zorder=zorderBase + 3,
        )

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")

    for cont, i in enumerate(colorsChosen):
        ax.plot([-1, -1], [-1, -1], c=i, label=f"{extralab}#{cont}")
    if legYN:
        ax.legend(loc="best", prop={"size": 7})
    ax.set_xlim([1.2, 2.5])
    ax.set_ylim([-1.6, 1.6])

    ax.set_aspect("equal")


def plotResult(
    prf_classes,
    metrics,
    Constraints,
    figs=None,
    coils_orig=None,
    ProblemExtras=None,
    colorOffset=0,
    limitContours=False,
    ParamsAll=None,
):
    if Constraints is not None:
        midplaneR = Constraints["midplaneR"]
        xpoints = Constraints["xpoints"]
    else:
        midplaneR = xpoints = None

    cols = GRAPHICStools.listColors()

    cols = cols[colorOffset:]

     
    if figs is None:
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        fn = FigureNotebook(0, "FreeGS Notebook", geometry="1600x1000")

        fig1 = fn.add_figure(label="Eq. & Coils")
        fig2 = fn.add_figure(label="Metrics")
        fig3 = fn.add_figure(label="Solution")
        figP = fn.add_figure(label="Profiles")
        figa = fn.add_figure(label="Powers")
        figb = fn.add_figure(label="Maxima")
        figMach = fn.add_figure(label="Machine")
        figRes = fn.add_figure(label="Summary")

    else:
        fig1 = figs[0]
        fig2 = figs[1]
        fig3 = figs[2]
        figP = figs[3]
        figa = figs[4]
        figb = figs[5]
        figMach = figs[6]
        figRes = figs[7]

    grid = plt.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.4)
    ax1 = fig1.add_subplot(grid[:, 0])
    axs2 = [
        fig1.add_subplot(grid[0, 1]),
        fig1.add_subplot(grid[1, 1]),
        fig1.add_subplot(grid[0, 2]),
        fig1.add_subplot(grid[1, 2]),
    ]

    if prf_classes is not None:
        plotAllEquilibria(
            ax1,
            prf_classes,
            ProblemExtras,
            limitContours=limitContours,
            onlySeparatrix=True,
            plotSOL=False,
        )

    if prf_classes is not None:
        plotCoils_organized(axs2, prf_classes, metrics, coils_orig=coils_orig, kA=True)

    grid = plt.GridSpec(nrows=1, ncols=1, hspace=0.3, wspace=0.4)
    ax1 = figMach.add_subplot(grid[0, 0])
    if prf_classes is not None:
        for i in range(len(prf_classes)):
            prf_classes[i].plot(ax=ax1, colorContours=cols[i])

    cm_span = 0.1
    m_span = cm_span * 1e-2

    aalpha = 0.1
    lw = 1.0  # 0.5
    cols = GRAPHICStools.listColors()

    grid = plt.GridSpec(nrows=2, ncols=5, hspace=0.3, wspace=0.4)

    cont = 0
    axs = [
        fig2.add_subplot(grid[0, 0]),
        fig2.add_subplot(grid[1, 0]),
        fig2.add_subplot(grid[0, 1]),
        fig2.add_subplot(grid[1, 1]),
        fig2.add_subplot(grid[0, 2]),
        fig2.add_subplot(grid[1, 2]),
        fig2.add_subplot(grid[0, 3]),
        fig2.add_subplot(grid[1, 3]),
        fig2.add_subplot(grid[0, 4]),
        fig2.add_subplot(grid[1, 4]),
    ]

    color = cols[cont]
    plotMetricsPRFS(
        axs,
        metrics,
        color=color,
        midplaneR=midplaneR,
        xpoints=xpoints,
        lw=lw,
        aalpha=aalpha,
        cm_span=cm_span,
    )

    grid = plt.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.4)

    axs = [
        fig3.add_subplot(grid[0, 0]),
        fig3.add_subplot(grid[0, 1]),
        fig3.add_subplot(grid[1, 0]),
        fig3.add_subplot(grid[1, 1]),
    ]

    if prf_classes is not None:
        plotCoils_organized(axs, prf_classes, metrics, relativeToMin=True)

    try:
        ax = fig3.add_subplot(grid[0, 2])
        y = 1e3 * (np.array(metrics["midplane_sepout"]) - midplaneR[1])
        ax.plot(y, "-o", label="$R_{out}$", lw=lw, markersize=3)
        y = 1e3 * (np.array(metrics["midplane_sepin"]) - midplaneR[0])
        ax.plot(y, "-o", label="$R_{in}$", lw=lw, markersize=3)
    except:
        pass
    try:
        y = 1e3 * (np.array(metrics["xpoints_up"][:, 0]) - np.array(xpoints)[:, 0])
        ax.plot(y, "-o", label="$R_{x}$", lw=lw, markersize=3)
        y = 1e3 * (np.array(metrics["xpoints_up"][:, 1]) - np.array(xpoints)[:, 1])
        ax.plot(y, "-o", label="$Z_{x}$", lw=lw, markersize=3)
    except:
        pass

    ax.set_title("Deviations (mm)")
    ax.legend(loc="best")
    ax.axhline(y=0, ls="--", lw=0.5, c="k")
    GRAPHICStools.gradientSPAN(
        ax,
        -m_span * 1e3,
        m_span * 1e3,
        color="k",
        startingalpha=aalpha,
        endingalpha=aalpha,
        orientation="horizontal",
    )
    ax.set_ylim([-m_span * 1e3 * 10, m_span * 1e3 * 10])
    ax.set_xlabel("Equilibrium #")

    # Profiles

    grid = plt.GridSpec(nrows=2, ncols=5, hspace=0.3, wspace=0.3)
    ax00 = figP.add_subplot(grid[0, 0])
    ax01l = figP.add_subplot(grid[0, 1])
    # ax00.twinx()
    ax01 = figP.add_subplot(grid[0, 2])
    ax02 = figP.add_subplot(grid[0, 3])
    ax03 = figP.add_subplot(grid[0, 4])
    ax10 = figP.add_subplot(grid[1, 0])
    ax10p = figP.add_subplot(grid[1, 1])
    ax11 = figP.add_subplot(grid[1, 2])
    ax12 = figP.add_subplot(grid[1, 3])
    ax13 = figP.add_subplot(grid[1, 4])

    if prf_classes is not None:
        for cont, prf in enumerate(prf_classes):
            psinorm = prf.profiles["psinorm"]
            q = prf.profiles["qpsi"]
            ax00.plot(psinorm, q, lw=2, ls="-", color=cols[cont], label=f"#{cont}")
            pres = prf.profiles["pres"]
            ax01.plot(psinorm, pres, lw=2, ls="-", color=cols[cont], label=f"#{cont}")
            fpol = prf.profiles["fpol"]
            ax01l.plot(
                psinorm,
                fpol,
                lw=2,
                ls="--",
                color=cols[cont],
                label=f"#{cont}",
            )
            pprime = prf.profiles["pprime"]
            ax10.plot(
                psinorm,
                pprime,
                lw=2,
                ls="-",
                color=cols[cont],
                label=f"#{cont}",
            )
            ffprime = prf.profiles["ffprime"] / (4 * np.pi * 1e-7)
            ax10p.plot(
                psinorm, ffprime, lw=2, ls="-", color=cols[cont]
            )  # ,label='#{0}'.format(cont))
            # ax11.plot(psinorm,ffprime,lw=2,ls='-',color=cols[cont],label='#{0}'.format(cont))

    ax = ax02
    psi = np.array(metrics["xpoints_up"][:, 2])
    ax.plot(psi, "-o", label="upper", c=cols[0], lw=lw, markersize=3)
    ax.plot(
        metrics["xpoints_low"][:, 2],
        "--o",
        label="lower",
        c=cols[1],
        lw=lw,
        markersize=3,
    )
    ax.set_title("Psi at x-points (Wb/rad)")
    ax.set_xlabel("Equilibrium #")
    ax.legend(loc="best")
    ax.set_ylim([np.min(psi) - 0.1, np.max(psi) + 0.1])

    ax = ax11
    try:
        dRsep = np.array(metrics["xpoints_up"][:, 3])
        ax.plot(dRsep, "-o", label="upper", c=cols[0], lw=lw, markersize=3)
        dRsep = np.array(metrics["xpoints_low"][:, 3])
        ax.plot(dRsep, "--o", label="lower", c=cols[1], lw=lw, markersize=3)
    except:
        pass
    ax.set_title("dRsep (cm)")
    ax.set_xlabel("Equilibrium #")
    ax.legend(loc="best")
    ax.axhline(y=0, ls="--", lw=0.5, c="k")

    ax = ax03
    psi = np.array(metrics["xpoints_up"][:, 2])
    t = ProblemExtras["times"]
    t0 = np.linspace(t[0], t[-1], 100)
    psi0 = np.interp(t0, t, psi)
    V = -2 * np.pi * MATHtools.deriv(t0, psi0)
    ax.plot(t0, V, "-o", c=cols[0], lw=lw, markersize=3)
    ax.set_title("$V=2\\pi*\\partial\\psi/\\partial t$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V")
    try:
        ax.set_ylim([np.min(V) - 0.5, np.max(V) + 0.5])
    except:
        pass

    ax = ax12
    ax.plot(metrics["Ip"], "-o", label="$1$", c=cols[0], lw=lw, markersize=3)
    ax.set_title("Plasma Current")
    ax.set_xlabel("Equilibrium #")
    ax.legend(loc="best")
    ax.set_ylabel("$I_p$ (MA)")
    ax.set_ylim([0, 10])

    ax = ax13
    for i in range(len(prf_classes)):
        ax.plot(
            prf_classes[i].sweep.psiEvolution_rel, "-o", c=cols[i], lw=2, markersize=3
        )

    ax.axhline(y=1e-4, lw=0.5, ls="--", c="k")

    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_xlabel("Picard Iterations")
    ax.set_ylabel("Relative change in Psi")

    ax = ax00
    ax.set_xlim([0, 1])
    ax.set_xlabel("psi_norm")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("q-profile")
    ax.legend(loc="best")

    ax = ax01
    ax.set_xlim([0, 1])
    ax.set_xlabel("psi_norm")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("pressure (Pa)")
    ax.legend(loc="best")
    ax = ax01l
    ax.set_ylabel("R*Bt (m*T)")

    ax = ax10
    ax.set_xlim([0, 1])
    ax.set_xlabel("psi_norm")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("p'")

    ax = ax10p
    ax.set_ylabel("ff'/$\\mu_0$")
    ax.legend(loc="best")

    ax = ax11
    # ax.set_xlim([0,1]); ax.set_xlabel('psi_norm');ax.set_ylim(bottom=0); ax.set_ylabel("ff'"); ax.legend(loc='best')

    # Powers
    if (ProblemExtras is not None) and ("InductanceMatrix" in ProblemExtras):
        fig = figa
        figMax = figb

        # Plotting settings
        if fig is None:
            plt.ion()
            fig = plt.figure(figsize=(15, 9))
        if figMax is None:
            plt.ion()
            figMax = plt.figure(figsize=(15, 9))

        grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.3, wspace=0.2)
        axs = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]),
        ]
        grid = plt.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.2)
        axsM = [
            figMax.add_subplot(grid[0, 0]),
            figMax.add_subplot(grid[1, 0]),
            figMax.add_subplot(grid[0, 1]),
            figMax.add_subplot(grid[1, 1]),
            figMax.add_subplot(grid[0, 2]),
            figMax.add_subplot(grid[1, 2]),
        ]

        t, Power_total, Power, V = plotPowers(
            metrics, ProblemExtras, axs=axs, axsM=axsM
        )
        # except: print('could not plot powers')

        ax = axsM[-2]
        if prf_classes is not None:
            plotAllEquilibria(
                ax,
                prf_classes,
                ProblemExtras,
                limitContours=False,
                onlySeparatrix=True,
                plotSOL=False,
            )

        ax.set_xlim([1.2, 1.9])
        ax.set_ylim([-1.6, -1.0])
        # ax.set_ylim([1.0,1.6])

        ax = axsM[-1]
        ax.plot(
            np.arange(0, len(metrics["midplane_sepin"])),
            metrics["strike_outAngle"],
            "-o",
            c=color,
            lw=0.5,
            markersize=3,
        )
        ax.set_title("Strike Angle (Degrees)")
        ax.set_xlabel("Equilibrium #")

    try:
        plotShapingComparison(
            prf_classes,
            metrics,
            fig=figRes,
            ParamsAll=ParamsAll,
            ProblemExtras=ProblemExtras,
        )
    except:
        pass


def plotMetricsPRFS(
    axs,
    metrics,
    color="b",
    midplaneR=[1.28, 2.42],
    xpoints=[0] * 10,
    lw=1,
    aalpha=0.1,
    cm_span=0.1,
    legYN=True,
):
    try:
        xaxis = metrics["params"]["times"]
        xaxis_label = "Times (s)"
    except:
        xaxis = np.arange(0, len(metrics["midplane_sepin"]))
        xaxis_label = "Equilibrium #"
    # xaxis = np.arange(0,len(metrics['midplane_sepin']))
    # xaxis_label = 'Equilibrium #'

    gradSpan = True
    m_span = cm_span * 1e-2
    lims = m_span * 10

    ax = axs[0]
    ax.plot(
        xaxis,
        metrics["midplane_sepin"],
        "-o",
        label="$R_{in}$",
        c=color,
        lw=lw,
        markersize=3,
    )
    if midplaneR is not None:
        yy = midplaneR[0]
        ax.axhline(y=yy, ls="--", c="k", lw=0.5)

        if gradSpan:
            GRAPHICStools.gradientSPAN(
                ax,
                yy - m_span,
                yy + m_span,
                color="k",
                startingalpha=aalpha,
                endingalpha=aalpha,
                orientation="horizontal",
            )

            GRAPHICStools.drawLineWithTxt(
                ax,
                yy + m_span,
                label=f"+{m_span * 1000.0:.1f}mm",
                orientation="horizontal",
                color="k",
                lw=0,
                ls="--",
                alpha=1.0,
                fontsize=5,
                fromtop=0.1,
                fontweight="normal",
                verticalalignment="bottom",
                horizontalalignment="left",
                separation=0,
            )

        ax.set_ylim([yy - lims, yy + lims])

    ax.set_title("R (m) of inner midplane")
    ax.set_xlabel(xaxis_label)

    ax = axs[1]
    ax.plot(
        xaxis,
        metrics["midplane_sepout"],
        "-o",
        label="$R_{out}$",
        c=color,
        lw=lw,
        markersize=3,
    )
    ax.set_title("R (m) of outer midplane")
    if midplaneR is not None:
        yy = midplaneR[1]
        ax.axhline(y=yy, ls="--", c="k", lw=0.5)
        if gradSpan:
            GRAPHICStools.gradientSPAN(
                ax,
                yy - m_span,
                yy + m_span,
                color="k",
                startingalpha=aalpha,
                endingalpha=aalpha,
                orientation="horizontal",
            )
        ax.set_ylim([yy - lims, yy + lims])
    ax.set_xlabel(xaxis_label)

    ax = axs[2]
    ax.plot(
        xaxis,
        metrics["xpoints_up"][:, 0],
        "-o",
        label="upper",
        c=color,
        lw=lw,
        markersize=3,
    )
    ax.plot(
        xaxis,
        metrics["xpoints_low"][:, 0],
        "--o",
        label="lower",
        c=color,
        lw=lw,
        markersize=3,
    )
    ax.set_title("X-points: R (m)")
    # if xpoints is not None:
    #     for xpoint in xpoints:
    #         try:
    #             yy = xpoint[0]
    #             ax.axhline(y=yy,ls='--',c=color,lw=0.5)
    #             GRAPHICStools.gradientSPAN(ax,yy-m_span,yy+m_span,color=color,startingalpha=aalpha,endingalpha=aalpha,orientation='horizontal')
    #             ax.set_ylim([yy-lims,yy+lims])
    #         except: pass

    ax.set_xlabel(xaxis_label)
    ax.legend()

    ax = axs[3]
    ax.plot(
        xaxis,
        np.abs(metrics["xpoints_up"][:, 1]),
        "-o",
        label="upper",
        c=color,
        lw=lw,
        markersize=3,
    )
    ax.plot(
        xaxis,
        np.abs(metrics["xpoints_low"][:, 1]),
        "--o",
        label="lower",
        c=color,
        lw=lw,
        markersize=3,
    )
    ax.set_title("X-points: |Z| (m)")
    # if xpoints is not None:
    #     for xpoint in xpoints:
    #         try:
    #             yy = xpoint[1]
    #             ax.axhline(y=yy,ls='--',c=color,lw=0.5)
    #             GRAPHICStools.gradientSPAN(ax,yy-m_span,yy+m_span,color=color,startingalpha=aalpha,endingalpha=aalpha,orientation='horizontal')
    #             ax.set_ylim([yy-lims,yy+lims])
    #         except: pass
    ax.set_xlabel(xaxis_label)
    ax.legend()

    ax = axs[4]
    ax.plot(
        xaxis,
        metrics["strike_inR"],
        "--o",
        label="inner",
        c=color,
        lw=0.5,
        markersize=3,
    )
    ax.plot(
        xaxis,
        metrics["strike_outR"],
        "-o",
        label="outer",
        c=color,
        lw=0.5,
        markersize=3,
    )
    ax.set_title("Main Strike Point: R (m)")
    ax.set_xlabel(xaxis_label)
    if legYN:
        ax.legend(loc="best")

    ax = axs[5]
    ax.plot(
        xaxis,
        metrics["strike_inZ"],
        "--o",
        label="inner",
        c=color,
        lw=0.5,
        markersize=3,
    )
    ax.plot(
        xaxis,
        metrics["strike_outZ"],
        "-o",
        label="outer",
        c=color,
        lw=0.5,
        markersize=3,
    )
    ax.set_title("Main Strike Point: Z (m)")
    if legYN:
        ax.legend(loc="best")
    ax.set_xlabel(xaxis_label)

    ax = axs[6]
    ax.plot(
        xaxis,
        (metrics["Rmajors"] - 1.85) * 1000,
        "--o",
        c=color,
        lw=0.5,
        markersize=3,
        label="$R_{major}-1.85$",
    )
    ax.plot(
        xaxis,
        (metrics["aminors"] - 0.57) * 1000,
        "-o",
        c=color,
        lw=0.5,
        markersize=3,
        label="$a-0.57$",
    )
    ax.set_title("(mm)")
    if legYN:
        ax.legend(loc="best")
    ax.set_xlabel(xaxis_label)
    ax.set_ylim([-5, 5])
    ax.axhline(y=1.0, lw=0.5, ls="--", c="r")
    ax.axhline(y=0.0, lw=0.5, ls="-.", c="k")
    ax.axhline(y=-1.0, lw=0.5, ls="--", c="r")

    ax = axs[7]
    # ax.plot(xaxis,metrics['strike_inAngle'],'--o',c=color,lw=0.5,markersize=3,label='inner')
    ax.plot(
        xaxis,
        metrics["strike_outAngle"],
        "-o",
        c=color,
        lw=0.5,
        markersize=3,
        label="outer",
    )
    ax.set_title("Strike Angle (Degrees)")
    # if legYN: ax.legend(loc='best')
    ax.set_xlabel(xaxis_label)
    ax.axhline(y=1, ls="--", c="k", lw=0.5)
    # ax.set_ylim(bottom=0)

    ax = axs[8]
    try:
        ax.plot(xaxis, metrics["kappaM"], "-o", c=color, lw=0.5, markersize=3)
    except:
        ax.plot(xaxis, metrics["elongs"], "-o", c=color, lw=0.5, markersize=3)
    ax.set_ylabel("Elongation")
    ax.set_xlabel(xaxis_label)
    ax.axhline(y=1.9649, lw=0.5, ls="-.", c="k")
    ax.set_ylim([1.85, 2.0])

    ax = axs[8].twinx()  # axs[9]
    try:
        ax.plot(xaxis, metrics["deltaM"], "--o", c=color, lw=0.5, markersize=3)
    except:
        ax.plot(xaxis, metrics["triangs"], "-o", c=color, lw=0.5, markersize=3)
    ax.set_ylabel("Triangularity")
    # ax.set_xlabel(xaxis_label)
    ax.axhline(y=0.5439, lw=0.5, ls="-.", c="k")
    ax.set_ylim([0.45, 0.65])

    ax = axs[9]
    try:
        ax.plot(
            xaxis,
            metrics["xpoints_up"][:, 3],
            "-o",
            c=color,
            lw=0.5,
            markersize=3,
            label="upper",
        )
        ax.plot(
            xaxis,
            metrics["xpoints_low"][:, 3],
            "--o",
            c=color,
            lw=0.5,
            markersize=3,
            label="lower",
        )
    except:
        print("This run did not have dRsep calculated")
    ax.set_title("X-points: dRsep (cm)")
    ax.set_xlabel(xaxis_label)
    ax.axhline(y=0, lw=0.5, ls="-.", c="k")


def plotPowers(
    metrics, ProblemExtras, axs=None, axsM=None, figMax=None, legYN=True, char="o"
):
    if axs is None:
        fig = plt.figure(figsize=(15, 9))
        figMax = plt.figure(figsize=(15, 9))

        grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.3, wspace=0.1)
        axs = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]),
        ]
        grid = plt.GridSpec(nrows=3, ncols=2, hspace=0.3, wspace=0.1)
        axsM = [
            figMax.add_subplot(grid[0, 0]),
            figMax.add_subplot(grid[1, 0]),
            figMax.add_subplot(grid[0, 1]),
            figMax.add_subplot(grid[1, 1]),
            figMax.add_subplot(grid[0, 2]),
            figMax.add_subplot(grid[1, 2]),
        ]

    (
        t,
        coils,
        coils_terminal,
        Power,
        V,
        dIdt,
        RampUp_limits,
        Supplies_limits,
    ) = FREEGStools.calculatePowerQuantities(metrics, ProblemExtras)

    [maxVar_dIdT_ramp, maxVar_V_ramp, maxVar_P_ramp] = RampUp_limits
    [maxVar_I_supp, maxVar_V_supp, minVar_I_supp, minVar_V_supp] = Supplies_limits

    colors = GRAPHICStools.listColors()

    # Plotting
    (
        maxV,
        maxV_coilcalcs,
        maxP,
        maxP_coilcalcs,
        maxdIdt,
        coil_names2,
        colors2,
        maxdIdt_coilcalcs,
    ) = ([], [], [], [], [], [], [], [])
    maxkA = []
    maxV_supp, maxI_supp = [], []
    ms = 3
    for cont, icoil in enumerate(coils):
        if icoil[-1].upper() == "U":  # coil_names[cont][-1] == 'U':
            col = colors[cont]
            ax = axs[0]
            ax.plot(
                t, coils_terminal[icoil] * 1e-3, color=col, label=icoil, markersize=ms
            )  # coil_names[cont])
            ax = axs[1]
            ax.plot(
                t, coils[icoil], color=col, label=icoil, markersize=ms
            )  # coil_names[cont])

            # ---- Original TSC value
            if "TSC" in ProblemExtras:
                x = [-1, 1]
                try:
                    y = [ProblemExtras["TSC"][icoil[:-1]]] * 2
                    ax.plot(x, y, "-.", lw=0.5, color=col)
                except:
                    pass
            # -------------------------

            ax = axs[2]
            ax.plot(
                t, V[icoil] * 1e-3, color=col, label=icoil, markersize=ms
            )  # coil_names[cont])
            ax = axs[3]
            ax.plot(
                t, Power[icoil] * 1e-6, color=col, label=icoil, markersize=ms
            )  # coil_names[cont])

            coil_names2.append(icoil)  # coil_names[cont])
            colors2.append(col)

            maxV.append(np.max(V[icoil] * 1e-3))
            maxkA.append(np.max(np.abs(coils_terminal[icoil] * 1e-3)))
            maxdIdt.append(np.max(np.abs(dIdt[icoil])))
            maxP.append(np.max(np.abs(Power[icoil] * 1e-6)))

            if maxVar_dIdT_ramp is not None:
                maxdIdt_coilcalcs.append(maxVar_dIdT_ramp[icoil[:-1].lower()])
                maxV_coilcalcs.append(maxVar_V_ramp[icoil[:-1].lower()])
                maxP_coilcalcs.append(maxVar_P_ramp[icoil[:-1].lower()])
            else:
                maxdIdt_coilcalcs.append(np.nan)
                maxV_coilcalcs.append(np.nan)
                maxP_coilcalcs.append(np.nan)

            if maxVar_V_supp is not None:
                maxV_supp.append(
                    np.max(
                        [
                            maxVar_V_supp[icoil[:-1].lower()],
                            -minVar_V_supp[icoil[:-1].lower()],
                        ]
                    )
                )
                maxI_supp.append(
                    np.max(
                        [
                            maxVar_I_supp[icoil[:-1].lower()],
                            -minVar_I_supp[icoil[:-1].lower()],
                        ]
                    )
                )
            else:
                maxV_supp.append(np.nan)
                maxI_supp.append(np.nan)

    maxdIdt = np.array(maxdIdt)
    maxdIdt_coilcalcs = np.array(maxdIdt_coilcalcs)
    coil_names2 = np.array(coil_names2)
    maxV = np.array(maxV)
    maxV_coilcalcs = np.array(maxV_coilcalcs)
    maxP = np.array(maxP)
    maxP_coilcalcs = np.array(maxP_coilcalcs)
    maxkA = np.array(maxkA)
    maxV_supp = np.array(maxV_supp)
    maxI_supp = np.array(maxI_supp)

    legsize = 8

    # Extra quantities

    # Total Power
    ax = axs[3]
    Power_total = 0
    for i in Power:
        Power_total += Power[i]
    ax.plot(t, Power_total * 1e-6, lw=2, ls="-", color=colors[cont + 1], label="total")

    # Plasma
    ax = axs[2]
    ax.plot(
        t, V["plasma"] * 1e-3, lw=0.5, ls="--", color=colors[cont + 1], label="plasma"
    )

    # ----------------------

    ax = axs[0]
    ax.set_xlabel("Time (s)")
    ax.set_title("Currents (kA)")
    if legYN:
        GRAPHICStools.addLegendApart(
            ax, ratio=0.8, withleg=True, extraPad=0, size=legsize
        )
    ax = axs[1]
    ax.set_xlabel("Time (s)")
    ax.set_title("Currents (MA-t)")
    if legYN:
        GRAPHICStools.addLegendApart(
            ax, ratio=0.8, withleg=True, extraPad=0, size=legsize
        )
    ax = axs[2]
    ax.set_xlabel("Time (s)")
    ax.set_title("Voltage (kV)")
    if legYN:
        GRAPHICStools.addLegendApart(
            ax, ratio=0.8, withleg=True, extraPad=0, size=legsize
        )
    ax = axs[3]
    ax.set_xlabel("Time (s)")
    ax.set_title("Powers (MVA)")
    if legYN:
        GRAPHICStools.addLegendApart(
            ax, ratio=0.8, withleg=True, extraPad=0, size=legsize
        )
    ax.set_ylim([-200, 200])

    #
    ax = axsM[0]
    x = np.arange(0, len(maxV))
    ax.scatter(x, maxV, marker=char, facecolors=colors2)
    for i in range(len(x)):
        ax.plot(
            [x[i] - 0.5, x[i], x[i] + 0.5],
            [maxV_coilcalcs[i], maxV_coilcalcs[i], maxV_coilcalcs[i]],
            "--*",
            lw=0.5,
            color=colors2[i],
            markersize=3,
            label=coil_names2[i],
        )
    ax.set_title("Max |V| (kV)")
    # if legYN: GRAPHICStools.addLegendApart(ax,ratio=0.8,withleg=True,extraPad=0,size=legsize)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 2.3])

    lim = maxV_supp
    for i in range(len(x)):
        ax.plot(
            [x[i] - 0.5, x[i], x[i] + 0.5],
            [lim[i], lim[i], lim[i]],
            "-",
            lw=0.5,
            color=colors2[i],
            markersize=3,
            label=coil_names2[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(coil_names2, rotation=90)

    ax = axsM[3]
    x = np.arange(0, len(maxdIdt))
    ax.scatter(x, maxdIdt, marker=char, facecolors=colors2)
    for i in range(len(x)):
        ax.plot(
            [x[i] - 0.5, x[i], x[i] + 0.5],
            [maxdIdt_coilcalcs[i], maxdIdt_coilcalcs[i], maxdIdt_coilcalcs[i]],
            "--*",
            lw=0.5,
            color=colors2[i],
            markersize=3,
            label=coil_names2[i],
        )
    ax.set_title("Max |dI/dt| (MA-t/s)")
    # if legYN: GRAPHICStools.addLegendApart(ax,ratio=0.8,withleg=True,extraPad=0,size=legsize)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 8])

    ax.set_xticks(x)
    ax.set_xticklabels(coil_names2, rotation=90)

    ax = axsM[2]
    x = np.arange(0, len(maxP))
    ax.scatter(x, maxP, marker=char, facecolors=colors2)
    for i in range(len(x)):
        ax.plot(
            [x[i] - 0.5, x[i], x[i] + 0.5],
            [maxP_coilcalcs[i], maxP_coilcalcs[i], maxP_coilcalcs[i]],
            "--*",
            lw=0.5,
            color=colors2[i],
            markersize=3,
            label=coil_names2[i],
        )
    ax.set_title("Max |P| (MVA)")
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 80])
    ax.set_xticks(x)
    ax.set_xticklabels(coil_names2, rotation=90)

    ax = axsM[1]
    x = np.arange(0, len(maxkA))
    ax.scatter(x, maxkA, marker=char, facecolors=colors2)
    lim = maxI_supp
    for i in range(len(x)):
        ax.plot(
            [x[i] - 0.5, x[i], x[i] + 0.5],
            [lim[i], lim[i], lim[i]],
            "-",
            lw=0.5,
            color=colors2[i],
            markersize=3,
            label=coil_names2[i],
        )
    ax.set_title("Max |$I_{c}$| (kA)")
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 80])
    ax.set_xticks(x)
    ax.set_xticklabels(coil_names2, rotation=90)

    return t, Power_total, Power, V


def plotCoils_organized(
    axs, prf_classes, metrics, coils_orig=None, relativeToMin=False, kA=False
):
    sets = ["cs", "pf", "dv", "vs"]
    lw = 1.0  # 0.5
    cols = GRAPHICStools.listColors()

    if kA:
        which = "coils_terminal"
        lab = "kA"
        mult = 1e-3
    else:
        which = "coils"
        lab = "MA*turns"
        mult = 1.0

    for conts, sett in enumerate(sets):
        ax = axs[conts]
        contColor = -1

        for coil in metrics[which]:
            if sett in coil and (
                "u" == coil[-1] or not prf_classes[0].sparc_coils.upDownSymmetric
            ):
                tot = metrics[which][coil] * mult
                if relativeToMin:
                    totMin = np.min(tot)
                else:
                    totMin = 0
                if coil[-1] == "u":
                    ls = "-"
                    mark = "-o"
                    contColor += 1
                else:
                    ls = "--"
                    mark = "-*"
                ax.plot(
                    tot - totMin,
                    mark,
                    label=coil,
                    c=cols[contColor],
                    lw=lw,
                    markersize=3,
                )  # ,ls=ls)

                # Original ---
                # if coils_orig is not None:  ax.axhline(y=coils_orig[stro],ls='--',c=cols[cont],lw=0.5)
                # ------

        GRAPHICStools.addLegendApart(ax, ratio=0.8, withleg=True, extraPad=0, size=10)
        ax.set_xlabel("Equilibrium #")
        if not relativeToMin:
            ax.set_ylabel(lab)
        else:
            ax.set_ylabel("Variation in MA*turns")

        ax.set_title(f"{sett.upper()} currents (upper and lower)")


def writeExtensiveTraces(metrics, file):
    (
        t,
        coils,
        coils_terminal,
        Power,
        V,
        dIdt,
        RampUp_limits,
        Supplies_limits,
    ) = FREEGStools.calculatePowerQuantities(metrics, metrics["params"])

    writer = pandas.ExcelWriter(file, engine="xlsxwriter")

    # Currents
    dictExcel = OrderedDict({"time (s)": t})
    for i in coils:
        dictExcel[i.upper() + " (MA-t)"] = coils[i]
    df = pandas.DataFrame(dictExcel)
    df.to_excel(
        writer, sheet_name="Currents", header=True, index=False, startrow=0, startcol=0
    )

    # Voltages
    dictExcel = OrderedDict({"time (s)": t})
    for i in coils:
        dictExcel[i.upper() + " (kV)"] = V[i] * 1e-3
    df = pandas.DataFrame(dictExcel)
    df.to_excel(
        writer, sheet_name="Voltages", header=True, index=False, startrow=0, startcol=0
    )

    # Powers
    dictExcel = OrderedDict({"time (s)": t})
    tot = 0
    for i in coils:
        dictExcel[i.upper() + " (MVA)"] = Power[i] * 1e-6
        tot += Power[i] * 1e-6
    dictExcel["total"] = tot
    df = pandas.DataFrame(dictExcel)
    df.to_excel(
        writer, sheet_name="Powers", header=True, index=False, startrow=0, startcol=0
    )

    writer.close()


def writeTable(metrics, file, times, divPlate=None, strikes=True, labelEq="time (s)"):
    writer = pandas.ExcelWriter(file, engine="xlsxwriter")

    # Coil currents
    dictExcel = OrderedDict({labelEq: times})
    for i in metrics["coils_terminal"]:
        if "plasma" not in i:
            dictExcel[i.upper() + " (kA)"] = metrics["coils_terminal"][i] * 1e-3
    df = pandas.DataFrame(dictExcel)
    df.to_excel(
        writer,
        sheet_name="Coil Currents",
        header=True,
        index=False,
        startrow=0,
        startcol=0,
    )

    dictExcel = OrderedDict({labelEq: times})
    for i in metrics["coils"]:
        if "plasma" not in i:
            dictExcel[i.upper() + " (MA-t)"] = metrics["coils"][i]
    df = pandas.DataFrame(dictExcel)
    df.to_excel(
        writer,
        sheet_name="Coil Currents (MA-t)",
        header=True,
        index=False,
        startrow=0,
        startcol=0,
    )

    Zo = metrics["strike_outZ"]
    Zi = metrics["strike_inZ"]
    Ro = metrics["strike_outR"]
    Ri = metrics["strike_inR"]
    Ao = metrics["strike_outAngle"]
    Ai = metrics["strike_inAngle"]

    if divPlate is not None:
        [limRZ_d, di, do] = divPlate
    else:
        di = do = np.zeros(len(times))

    listwhich = ["Coil Currents", "Coil Currents (MA-t)"]

    if strikes:
        # Inner strike
        dictExcel = {
            "time (s)": times,
            "R (m)": Ri,
            "Z (m)": Zi,
            "angle (deg)": Ai,
            "L (m)": di,
        }
        df = pandas.DataFrame(dictExcel)
        df.to_excel(
            writer,
            sheet_name="Inner Strike",
            header=True,
            index=False,
            startrow=0,
            startcol=0,
        )

        # Outer strike
        dictExcel = {
            "time (s)": times,
            "R (m)": Ro,
            "Z (m)": Zo,
            "angle (deg)": Ao,
            "L (m)": do,
        }
        df = pandas.DataFrame(dictExcel)
        df.to_excel(
            writer,
            sheet_name="Outer Strike",
            header=True,
            index=False,
            startrow=0,
            startcol=0,
        )

        listwhich.append("Inner Strike")
        listwhich.append("Outer Strike")

    for i in listwhich:
        worksheet = writer.sheets[i]
        workbook = writer.book
        format1 = workbook.add_format({"num_format": "0.0000"})
        worksheet.set_column("A:W", 10, format1)

    writer.close()

    print(f"\t~ Excel file {file} written with coil currents and strike points")


def write_gfiles(prfs, whereOutput, check=True):
    from freegs import geqdsk

    gs = []
    for i in range(len(prfs)):
        filename = f"geqdsk_freegsu_run{i}.geq"
        with open(f"{whereOutput}/{filename}", "w") as f:
            geqdsk.write(prfs[i].sweep.eq, f)
        print(f"\t~ g-file {filename} written")

        if check:
            try:
                gs.append(GEQtools.MITIMgeqdsk(f"{whereOutput}/{filename}"))
            except:
                print("\t- Problem checking gfile", typeMsg="w")

    return gs


def extendSweep(CoilCurrents, Constraints, n=10, orderInEquil=None, times=None):
    """
    orderInEquil is something like [0,6,9] indicating the time difference in between each
    """

    if orderInEquil is None:
        orderInEquil = [0, n - 1]
    orderInEquilNew = np.linspace(orderInEquil[0], orderInEquil[-1], n)

    for i in CoilCurrents:
        CoilCurrents[i] = np.interp(orderInEquilNew, orderInEquil, CoilCurrents[i])

    for i in Constraints["xpoints"]:
        for j in range(n - len(Constraints["xpoints"])):
            Constraints["xpoints"].append(Constraints["xpoints"][0])

    if times is not None:
        times_mod = np.interp(orderInEquilNew, orderInEquil, times)
    else:
        times_mod = None

    print(f"Sweep interpolated to {n} points, from {times} to {times_mod}")

    return CoilCurrents, Constraints, times_mod


def plotCoilsBars(
    coils,
    equil_names,
    ax=None,
    onlyUpper=True,
    plotType="u",
    extralab="",
    plotVS=False,
    ProblemExtras=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    # for j in coils:
    #     if j not in ['plasma','vs1u'] and j[-1] not in ['l']:
    #         ax.plot(equil_names,coils[j]*1E-3,'s',label=j,markersize=5)
    # ax.set_xlabel('Eq #')
    # ax.set_ylabel('kA')
    # GRAPHICStools.addLegendApart(ax,ratio=0.9,withleg=True,extraPad=0,size=7,loc='upper left')
    # ax.set_xlim(equil_names[0],equil_names[-1])
    # for i in [-42,-30,0,30,42]:
    #     ax.axhline(y=i,c='k',ls='--',lw=0.3)

    avoid = ["plasma"]
    if not plotVS:
        avoid.append("vs1u")

    coils_names = []
    coils_names_x = []
    coils_organized = []
    cont = 0
    for j in coils:
        if j not in avoid:
            if not onlyUpper or (j[-1] in [plotType]):
                coils_organized.append(coils[j] * 1e-3)
                coils_names.append(str(j))
                coils_names_x.append(cont)
                cont += 1
    coils_organized = np.transpose(np.array(coils_organized))

    width = 1 / (len(equil_names))
    var = np.linspace(-0.5, 0.5 - width, len(equil_names))

    colors = GRAPHICStools.listColors()

    for i in range(len(equil_names)):
        ax.bar(
            coils_names_x + var[i],
            coils_organized[i],
            width,
            label=equil_names[i],
            color=colors[i],
            align="edge",
        )

    ax.set_xlim([coils_names_x[0] - 0.5, coils_names_x[-1] + 0.5])
    ax.set_xticks(coils_names_x)
    ax.set_xticklabels(coils_names)
    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=True, extraPad=0, size=7, loc="upper left"
    )

    for i in coils_names_x:
        ax.axvline(x=i + 0.5, ls="--", c="k", lw=0.5)

    ax.axhline(y=0, ls="--", c="k", lw=0.5)
    # ax.set_ylim([-50,50])

    ax.set_ylabel("kA " + extralab)

    if (ProblemExtras is not None) and ("RequirementsFile" in ProblemExtras):
        (
            maxVar_I,
            maxVar_V,
            minVar_I,
            minVar_V,
        ) = FREEGStools.readCoilCalcsMatrices_Supplies(
            IOtools.expandPath(ProblemExtras["RequirementsFile"])
        )

        for i in coils_names_x:
            lims = [minVar_I[coils_names[i][:-1]], maxVar_I[coils_names[i][:-1]]]
            ax.plot([i - 0.5, i + 0.5], [lims[0], lims[0]], c="k", lw=2)
            ax.plot([i - 0.5, i + 0.5], [lims[1], lims[1]], c="k", lw=2)


def plotShapingComparison(prfs, metrics, fig=None, ParamsAll=None, ProblemExtras=None):
    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(14, 7))

    # Determine if it's symmetric
    includeLower = False
    for i in metrics["coils_terminal"]:
        if (i[-1] == "u") and (
            metrics["coils_terminal"][i] != metrics["coils_terminal"][i[:-1] + "l"]
        ).any():
            includeLower = True
            break

    xaxis_lab = np.arange(len(metrics["elongs"]))

    if includeLower:
        nrows = 3
    else:
        nrows = 2

    grid = plt.GridSpec(nrows=nrows, ncols=7, hspace=0.3, wspace=0.4)

    atleastone = False
    if ParamsAll is not None:
        for i in range(len(metrics["kappaM"])):
            if (
                "extraSeparatrixPoint" in ParamsAll[i]
                or "extraSeparatrixPoint_plot" in ParamsAll[i]
            ):
                atleastone = True

    if atleastone:
        ax = fig.add_subplot(grid[:-1, 0:2])
    else:
        ax = fig.add_subplot(grid[:, 0:2])
    cols = GRAPHICStools.listColors()
    plotAllEquilibria(
        ax,
        prfs,
        None,
        colors=cols,
        onlySeparatrix=True,
        plotSOL=False,
        limitContours=True,
        plotDIV=False,
        plotLIM=True,
        plotStrike=False,
        plotAxis=True,
    )

    if atleastone:
        ax = fig.add_subplot(grid[-1, 0:2])
        cols = GRAPHICStools.listColors()
        plotAllEquilibria(
            ax,
            prfs,
            None,
            colors=cols,
            onlySeparatrix=True,
            plotSOL=False,
            limitContours=True,
            plotDIV=False,
            plotLIM=False,
            plotStrike=False,
            plotAxis=True,
            legYN=False,
        )
        ax.set_xlim([2.27, 2.31])
        ax.set_ylim([-0.51, -0.47])

        if ParamsAll is not None:
            for i in range(len(metrics["kappaM"])):
                if "extraSeparatrixPoint" in ParamsAll[i]:
                    for j in range(len(ParamsAll[i]["extraSeparatrixPoint"])):
                        ax.plot(
                            [ParamsAll[i]["extraSeparatrixPoint"][j][0]],
                            [ParamsAll[i]["extraSeparatrixPoint"][j][1]],
                            "-s",
                            markersize=3,
                            c=cols[i],
                        )
                if "extraSeparatrixPoint_plot" in ParamsAll[i]:
                    for j in range(len(ParamsAll[i]["extraSeparatrixPoint_plot"])):
                        ax.plot(
                            [ParamsAll[i]["extraSeparatrixPoint_plot"][j][0]],
                            [ParamsAll[i]["extraSeparatrixPoint_plot"][j][1]],
                            "-s",
                            markersize=3,
                            c=cols[i],
                        )

    contT = 2

    # -------------- Plot of coils

    ax = fig.add_subplot(grid[1, contT:])
    equil_names = []
    for i in xaxis_lab:
        equil_names.append(f"Eq #{i:.0f}")

    plotCoilsBars(
        metrics["coils_terminal"],
        equil_names,
        ax=ax,
        plotType="u",
        plotVS=includeLower,
        ProblemExtras=ProblemExtras,
    )

    if includeLower:
        ax = fig.add_subplot(grid[2, contT:])
        plotCoilsBars(
            metrics["coils_terminal"],
            equil_names,
            ax=ax,
            plotType="l",
            extralab="(Lower)",
            plotVS=includeLower,
            ProblemExtras=ProblemExtras,
        )

    # -------------- Plot of shaping properties

    ax = fig.add_subplot(grid[0, 0 + contT])
    # ax.plot(xaxis_lab,metrics['elongs'],'-s',color='b',markersize=5)
    ax.plot(
        xaxis_lab,
        metrics["kappaM"],
        "-s",
        color="b",
        markersize=3,
        lw=1,
        label="$\\kappa$",
    )
    ax.plot(
        xaxis_lab,
        metrics["kappaU"],
        "-o",
        color="r",
        markersize=3,
        lw=0.5,
        label="$\\kappa_U$",
    )
    ax.plot(
        xaxis_lab,
        metrics["kappaL"],
        "-^",
        color="g",
        markersize=3,
        lw=0.5,
        label="$\\kappa_L$",
    )
    ax.plot(
        xaxis_lab,
        metrics["elongsA"],
        "-v",
        color="c",
        markersize=3,
        lw=0.5,
        label="$\\kappa_A$",
    )

    ax.set_title("Elongation")
    ax.set_ylim(
        [
            np.min(
                [
                    1.3,
                    np.min(metrics["elongsA"]),
                    np.min(metrics["elongs"]),
                    np.min(metrics["kappaU"]),
                    np.min(metrics["kappaL"]),
                ]
            ),
            2.2,
        ]
    )
    ax.set_xlabel("Eq #")
    ax.set_xlim(xaxis_lab[0] - 0.5, xaxis_lab[-1] + 0.5)
    ax.legend(prop={"size": 5})

    ax = fig.add_subplot(grid[0, 1 + contT])
    # ax.plot(xaxis_lab,metrics['triangs'],'-s',color='b',markersize=5)
    ax.plot(
        xaxis_lab,
        metrics["deltaM"],
        "-s",
        color="b",
        markersize=3,
        lw=1,
        label="$\\delta$",
    )
    ax.plot(
        xaxis_lab,
        metrics["deltaU"],
        "-o",
        color="r",
        markersize=3,
        lw=0.5,
        label="$\\delta_U$",
    )
    ax.plot(
        xaxis_lab,
        metrics["deltaL"],
        "-^",
        color="g",
        markersize=3,
        lw=0.5,
        label="$\\delta_L$",
    )
    ax.legend(prop={"size": 5})
    ax.set_ylim([0.0, 0.9])
    ax.set_title("Triangularity")
    ax.set_xlabel("Eq #")
    ax.set_xlim(xaxis_lab[0] - 0.5, xaxis_lab[-1] + 0.5)

    ax = fig.add_subplot(grid[0, 2 + contT])
    ax.plot(
        xaxis_lab,
        metrics["zetaM"],
        "-s",
        color="b",
        markersize=3,
        lw=1,
        label="$\\zeta$",
    )
    ax.plot(
        xaxis_lab,
        metrics["zetaU"],
        "-o",
        color="r",
        markersize=3,
        lw=0.5,
        label="$\\zeta_U$",
    )
    ax.plot(
        xaxis_lab,
        metrics["zetaL"],
        "-^",
        color="g",
        markersize=3,
        lw=0.5,
        label="$\\zeta_L$",
    )
    ax.set_ylim([np.min([0.25, np.min(metrics["zetaM"])]), 0.5])
    ax.set_title("Squareness")
    ax.set_xlabel("Eq #")
    ax.set_xlim(xaxis_lab[0] - 0.5, xaxis_lab[-1] + 0.5)
    ax.legend(prop={"size": 5})

    ax = fig.add_subplot(grid[0, 3 + contT])
    ax.plot(
        xaxis_lab,
        metrics["betaN"],
        "-s",
        color="b",
        markersize=3,
        lw=1,
        label="$\\beta_N$",
    )
    ax.plot(
        xaxis_lab, metrics["li1"], "-s", color="r", markersize=3, lw=1, label="$l_i$"
    )
    ax.plot(
        xaxis_lab,
        metrics["ShafParam"],
        "-s",
        color="g",
        markersize=3,
        lw=1,
        label="$\\beta_\\theta+l_i/2$",
    )
    ax.set_ylim([0.0, 4.0])
    ax.set_title("Beta & inductance")
    ax.set_xlabel("Eq #")
    ax.set_xlim(xaxis_lab[0] - 0.5, xaxis_lab[-1] + 0.5)
    ax.legend(prop={"size": 5})
    ax.axhline(y=1, ls="--", lw=0.5, c="k")

    ax = fig.add_subplot(grid[0, 4 + contT])
    ax.plot(xaxis_lab, metrics["q0"], "-s", color="b", markersize=3, lw=1, label="q0")
    ax.plot(xaxis_lab, metrics["q95"], "-s", color="r", markersize=3, lw=1, label="q95")
    ax.set_ylim([0.0, 6.0])
    ax.set_title("q-values")
    ax.set_xlabel("Eq #")
    ax.set_xlim(xaxis_lab[0] - 0.5, xaxis_lab[-1] + 0.5)
    ax.legend(prop={"size": 5})
    ax.axhline(y=1, ls="--", lw=0.5, c="k")
    ax.axhline(y=3, ls="--", lw=0.5, c="k")
