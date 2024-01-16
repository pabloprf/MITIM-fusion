import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools


def plotDB(cdfs, runIDs=None, fn=None, extralab="", timePlot=None, avTime=0.0):
    colors = GRAPHICStools.listColors()

    if runIDs is None:
        runIDs = np.arange(0, len(cdfs))

    if fn is None:
        figsize = (20, 10)
        fig1 = plt.figure(figsize=figsize)
        fig4 = plt.figure(figsize=figsize)
        fig5 = plt.figure(figsize=figsize)
        fig2 = plt.figure(figsize=figsize)
        fig3 = plt.figure(figsize=figsize)
        fig6 = plt.figure(figsize=figsize)
    else:
        fig1 = fn.add_figure(label="Data1" + extralab)
        fig4 = fn.add_figure(label="Data2" + extralab)
        fig5 = fn.add_figure(label="Checks" + extralab)
        fig2 = fn.add_figure(label="Profiles" + extralab)
        fig3 = fn.add_figure(label="Profiles2" + extralab)
        fig6 = fn.add_figure(label="Profiles3" + extralab)

    grid = plt.GridSpec(nrows=2, ncols=7, hspace=0.3, wspace=0.25)
    ax00 = fig1.add_subplot(grid[0, 0])
    ax01 = fig1.add_subplot(grid[0, 1], sharex=ax00)
    ax02 = fig1.add_subplot(grid[0, 2], sharex=ax00)
    ax03 = fig1.add_subplot(grid[0, 3], sharex=ax00)
    ax04 = fig1.add_subplot(grid[0, 4], sharex=ax00)
    ax05 = fig1.add_subplot(grid[0, 5], sharex=ax00)
    ax06 = fig1.add_subplot(grid[0, 6], sharex=ax00)
    ax10 = fig1.add_subplot(grid[1, 0], sharex=ax00)
    ax11 = fig1.add_subplot(grid[1, 1], sharex=ax00)
    ax12 = fig1.add_subplot(grid[1, 2], sharex=ax00)
    ax13 = fig1.add_subplot(grid[1, 3], sharex=ax00)
    ax14 = fig1.add_subplot(grid[1, 4], sharex=ax00)
    ax15 = fig1.add_subplot(grid[1, 5], sharex=ax00)
    ax16 = fig1.add_subplot(grid[1, 6], sharex=ax00)

    grid = plt.GridSpec(nrows=2, ncols=7, hspace=0.3, wspace=0.25)
    ax00u = fig4.add_subplot(grid[0, 0])
    ax01u = fig4.add_subplot(grid[0, 1])
    ax02u = fig4.add_subplot(grid[0, 2])
    ax03u = fig4.add_subplot(grid[0, 3])
    ax04u = fig4.add_subplot(grid[0, 4])
    ax05u = fig4.add_subplot(grid[0, 5])
    ax06u = fig4.add_subplot(grid[0, 6])
    ax10u = fig4.add_subplot(grid[1, 0])
    ax11u = fig4.add_subplot(grid[1, 1])
    ax12u = fig4.add_subplot(grid[1, 2])
    ax13u = fig4.add_subplot(grid[1, 3])
    ax14u = fig4.add_subplot(grid[1, 4])
    ax15u = fig4.add_subplot(grid[1, 5])
    ax16u = fig4.add_subplot(grid[1, 6])

    grid = plt.GridSpec(nrows=2, ncols=5, hspace=0.3, wspace=0.25)
    ax00g = fig5.add_subplot(grid[0, 0])
    ax01g = fig5.add_subplot(grid[0, 1])
    ax10g = fig5.add_subplot(grid[1, 0])
    ax11g = fig5.add_subplot(grid[1, 1])
    ax20g = fig5.add_subplot(grid[0, 2])
    ax21g = fig5.add_subplot(grid[1, 2])
    axg = fig5.add_subplot(grid[0, 3:])
    axg2 = fig5.add_subplot(grid[1, 3:])

    grid = plt.GridSpec(nrows=2, ncols=7, hspace=0.3, wspace=0.25)
    ax00r = fig2.add_subplot(grid[0, 0])
    ax01r = fig2.add_subplot(grid[0, 1])
    ax02r = fig2.add_subplot(grid[0, 2])
    ax03r = fig2.add_subplot(grid[0, 3])
    ax10r = fig2.add_subplot(grid[1, 0])
    ax11r = fig2.add_subplot(grid[1, 1])
    ax12r = fig2.add_subplot(grid[1, 2])
    ax13r = fig2.add_subplot(grid[1, 3])

    ax04r = fig2.add_subplot(grid[0, 4])
    ax14r = fig2.add_subplot(grid[1, 4])

    ax05r = fig2.add_subplot(grid[0, 5])
    ax15r = fig2.add_subplot(grid[1, 5])

    ax06r = fig2.add_subplot(grid[0, 6])
    ax16r = fig2.add_subplot(grid[1, 6])

    grid = plt.GridSpec(nrows=2, ncols=6, hspace=0.3, wspace=0.25)
    ax00f = fig3.add_subplot(grid[0, 0])
    ax01f = fig3.add_subplot(grid[0, 1])
    ax02f = fig3.add_subplot(grid[0, 2])
    ax03f = fig3.add_subplot(grid[0, 3])
    ax04f = fig3.add_subplot(grid[0, 4])
    ax05f = fig3.add_subplot(grid[0, 5])

    ax10f = fig3.add_subplot(grid[1, 0])
    ax11f = fig3.add_subplot(grid[1, 1])
    ax12f = fig3.add_subplot(grid[1, 2])
    ax13f = fig3.add_subplot(grid[1, 3])
    ax14f = fig3.add_subplot(grid[1, 4])
    ax15f = fig3.add_subplot(grid[1, 5])

    grid = plt.GridSpec(nrows=2, ncols=6, hspace=0.3, wspace=0.25)
    ax00f2 = fig6.add_subplot(grid[0, 0])
    ax01f2 = fig6.add_subplot(grid[0, 1])
    ax02f2 = fig6.add_subplot(grid[0, 2])
    ax03f2 = fig6.add_subplot(grid[0, 3])
    ax04f2 = fig6.add_subplot(grid[0, 4])
    ax05f2 = fig6.add_subplot(grid[0, 5])

    ax10f2 = fig6.add_subplot(grid[1, 0])
    ax11f2 = fig6.add_subplot(grid[1, 1])
    ax12f2 = fig6.add_subplot(grid[1, 2])
    ax13f2 = fig6.add_subplot(grid[1, 3])
    ax14f2 = fig6.add_subplot(grid[1, 4])
    ax15f2 = fig6.add_subplot(grid[1, 5])

    labx = "Run #"
    for cont, c in enumerate(cdfs):
        print("**", cont, runIDs[cont])

        try:
            if timePlot is None:
                it1 = c.ind_saw
                it2 = c.ind_saw
            else:
                it1 = np.argmin(np.abs(c.t - (timePlot - avTime)))
                it2 = np.argmin(np.abs(c.t - (timePlot + avTime)))
        except:
            continue

        it = c.ind_saw
        size = 7

        xPlot = runIDs[
            cont
        ]  # np.linspace(1,len(cdfs),len(cdfs))[cont]#c.PradT[it]/(c.PradT[it]+c.P_LCFS[it])

        iw = np.argmin(np.abs(c.x_lw - (1 - c.Te_width[it])))

        # ------------------------------------------------------------------------------------

        if c.Plh_ratio[it] > 1.0:
            alpha = 1.0
            mar = "-s"
        else:
            alpha = 1.0
            mar = "-*"

        ax = ax00
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Ip[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        cur = [8.7, 6.96, 3.48]
        for i in cur:
            ax.axhline(y=i, ls="--", c="g", lw=0.5)
        cur = [5.7, 4.56, 2.28]
        for i in cur:
            ax.axhline(y=i, ls="--", c="m", lw=0.5)

        ax = ax01
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Bt[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=8, ls="--", c="m", lw=0.5)
        ax.axhline(y=12.16, ls="--", c="g", lw=0.5)

        ax = ax02
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.PichT[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=25, ls="--", c="g", lw=0.5)
        ax.axhline(y=11.1, ls="--", c="g", lw=0.5)

        ax = ax03
        ax.plot(
            xPlot, c.mi / c.u, mar, color=colors[cont], markersize=size, alpha=alpha
        )
        ax.axhline(y=2, ls="--", c="m", lw=0.5)
        ax.axhline(y=2.5, ls="--", c="g", lw=0.5)

        # ax.plot(xPlot,c.fmini_avolAVE_Z[it]/c.u,mar,color=colors[cont],markersize=size,alpha=alpha)

        ax = ax04
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.fGh[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax05
        if "W" in c.fZs_avol:
            z = CDFtools.timeAverage(
                c.t[it1 : it2 + 1], c.fZs_avol["W"]["total"][it1 : it2 + 1]
            )
            ax.plot(
                xPlot, z * 1e5, mar, color=colors[cont], markersize=size, alpha=alpha
            )
        ax.axhline(y=1.5, ls="--", c="g", lw=0.5)

        ax = ax10
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.fGv[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=0.4, ls="--", c="g", lw=0.5)
        ax.axhline(y=0.8, ls="--", c="g", lw=0.5)

        ax = ax11
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Wtot[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax13
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            c.PradT[it1 : it2 + 1] / (c.PradT[it1 : it2 + 1] + c.P_LCFS[it1 : it2 + 1]),
        )
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax12
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Plh_ratio[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=1, ls="--", c="g", lw=0.5)

        ax = ax14
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.P_LCFS[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax06
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Zeff_avol[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax15
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Q[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=2, ls="--", c="g", lw=0.5)
        ax.axhline(y=11, ls="--", c="g", lw=0.5)

        ax = ax16
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Pout[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=140, ls="--", c="g", lw=0.5)

        #

        ax = ax00g
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            np.abs(c.PichT[it1 : it2 + 1] - c.PichT_check[it1 : it2 + 1])
            / c.PichT[it1 : it2 + 1]
            * 100.0,
        )
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax01g
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            np.abs(c.Ip[it1 : it2 + 1] - c.Ip_eq[it1 : it2 + 1])
            / c.Ip[it1 : it2 + 1]
            * 100.0,
        )
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax10g
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            np.abs(c.qe_obs[it1 : it2 + 1, :iw] - c.qe_tr[it1 : it2 + 1, :iw])
            / c.qe_obs[it1 : it2 + 1, :iw]
            * 100,
        )
        ax.plot(c.x_lw[:iw], z, "-", color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=10, ls="--", c="g", lw=0.5)

        ax = ax11g
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            np.abs(c.qi_obs[it1 : it2 + 1, :iw] - c.qi_tr[it1 : it2 + 1, :iw])
            / c.qi_obs[it1 : it2 + 1, :iw]
            * 100,
        )
        ax.plot(c.x_lw[:iw], z, "-", color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=10, ls="--", c="g", lw=0.5)

        ax = ax20g
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.V[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax21g
        # ax.plot(c.x_lw,c.V[it],'-',color=colors[cont],markersize=size,alpha=alpha)

        ax = axg
        ax.plot(c.t - c.t[0], c.Te0, "-", color=colors[cont], alpha=alpha)

        ax = axg2
        ax.plot(c.t - c.t[0], c.Piich[:, 20], "-", color=colors[cont], alpha=alpha)

        #

        ix = np.argmin(np.abs(c.x_lw - 0.15))
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            c.aLnD[it1 : it2 + 1, ix] - 0.5 * c.aLTi[it1 : it2 + 1, ix],
        )
        ax = ax00u
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax01u
        z = CDFtools.timeAverage(
            c.t[it1 : it2 + 1],
            c.aLnD[it1 : it2 + 1, :iw] - 0.5 * c.aLTi[it1 : it2 + 1, :iw],
        )
        ax.plot(c.x_lw[:iw], z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax10u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.IpB_fraction[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax11u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Q[it1 : it2 + 1])
        ax.plot(c.P_LCFS[it], z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=2.0, ls="--", c="g", lw=1)
        ax.axvline(x=30.0, ls="--", c="g", lw=1)
        # ax.set_ylim(bottom=0)

        ax = ax12u
        z1 = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Plh_ratio[it1 : it2 + 1])
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Q[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=2.0, ls="--", c="g", lw=1)
        ax.axvline(x=1.0, ls="--", c="g", lw=1)
        # ax.set_ylim(bottom=0);

        ax = ax13u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.rhos_a_avol[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax14u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Wth_frac[it1 : it2 + 1])
        ax.plot(xPlot, 1 - z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax02u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.taue[it1 : it2 + 1])
        ax.plot(xPlot, z * 1e-3, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax03u
        z2 = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.H98y2_check[it1 : it2 + 1])
        ax.plot(z * 1e-3, z2, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax04u
        z1 = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Ip[it1 : it2 + 1])
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.q95[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        # ax.plot(c.Ip[it],c.qstar[it],mar,color=colors[cont],markersize=size/2,alpha=alpha/2)

        ax = ax05u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.fmain_avol[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=0.85, ls="--", c="k", lw=0.5)

        ax = ax15u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.fZ_avolAVE_Z[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax06u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.ShafShift[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax16u
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Vsurf[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        # ------------------------------------------------------------------------------

        ax = ax00r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Te[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax01r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Ti[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax02r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.ne[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax10r
        z1 = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.nu_eff_avol[it1 : it2 + 1])
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.ne_peaking[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax11r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.QiQe_ratio[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=1, ls="--", c="k", lw=0.5)

        ax = ax12r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.TiTe[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=1, ls="--", c="k", lw=0.5)

        ax = ax03r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Prad[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax13r
        z1 = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Plh_ratio[it1 : it2 + 1])
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.H98y2_check[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)
        # ax.plot(c.Plh_ratio[it],c.H98y2[it],mar,color=colors[cont],markersize=size,alpha=0.2)
        ax.axhline(y=1, ls="--", c="g", lw=1.0)
        ax.axhline(y=1.15, ls="--", c="g", lw=0.5)
        ax.axhline(y=0.85, ls="--", c="g", lw=0.5)
        ax.axvline(x=1, ls="--", c="k", lw=1.0)

        ax = ax04r
        z1 = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.fGv[it1 : it2 + 1])
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Te_height[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax14r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Te_width[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax05r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.p_height[it1 : it2 + 1])
        ax.plot(z1, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax15r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.BetaN[it1 : it2 + 1])
        ax.plot(xPlot, z, mar, color=colors[cont], markersize=size, alpha=alpha)

        ax = ax06r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.aLTi_gR[it1 : it2 + 1])
        ax.plot(
            c.x_lw[:iw], z[:iw], "-", color=colors[cont], markersize=size, alpha=alpha
        )

        ax = ax16r
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Chi_i[it1 : it2 + 1])
        ax.plot(
            c.x_lw[:iw], z[:iw], "-", color=colors[cont], markersize=size, alpha=alpha
        )

        # ------------------------------------------------------------------------------

        ax = ax00f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.pFast_fus[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax01f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.pFast_mini[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax02f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Tfus[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax03f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Tmini[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax04f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Piich[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax10f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.jB[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax11f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.q[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)
        ax.axhline(y=1, ls="--", c="g", lw=0.5)

        ax = ax12f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Poh[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax14f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.Peich[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax13f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.xpol[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax05f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.j[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)
        ax = ax15f
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.jOh[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        # ----
        ax = ax00f2
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.qe_obs[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

        ax = ax01f2
        z = CDFtools.timeAverage(c.t[it1 : it2 + 1], c.qi_obs[it1 : it2 + 1])
        ax.plot(c.x_lw, z, "-", color=colors[cont], markersize=size, alpha=alpha)

    def addlab(ax, label, typeA=1):
        if typeA == 1:
            ax.set_title(label)
        else:
            ax.set_ylabel(label)

    ax = ax00
    ax.set_xlabel(labx)
    addlab(ax, "$I_p$ (MA)")
    ax.set_ylim([0, 10.0])
    ax = ax10
    ax.set_xlabel(labx)
    addlab(ax, "$f_G=\\langle n_e\\rangle/n_G$")
    ax.set_ylim([0, 1.0])
    ax = ax02
    ax.set_xlabel(labx)
    addlab(ax, "$P_{ICRF}$ (MW)")
    ax.set_ylim([0, 30])
    ax = ax01
    ax.set_xlabel(labx)
    addlab(ax, "$B_T$ (T)")
    ax.set_ylim([0, 15.0])
    ax = ax04
    ax.set_xlabel(labx)
    addlab(ax, "$n_{e,top}/n_G$")
    ax.set_ylim([0, 1.0])
    ax = ax11
    ax.set_xlabel(labx)
    addlab(ax, "$W_{TOT}$ (MJ)")
    ax.set_ylim(bottom=0)
    ax = ax13
    ax.set_xlabel(labx)
    addlab(ax, "$f_{rad,core}$")
    ax.set_ylim([0, 1.0])
    ax = ax03
    ax.set_xlabel(labx)
    addlab(ax, "fuel mass (u)", typeA=1)
    ax.set_ylim([1, 3.0])
    ax = ax05
    ax.set_xlabel(labx)
    addlab(ax, "$\\langle f_{W}\\rangle$ ($10^{-5}$)", typeA=1)
    ax.set_ylim(bottom=0)
    ax = ax12
    ax.set_xlabel(labx)
    addlab(ax, "$P_{net}/P_{LH}$", typeA=1)
    ax.set_ylim(bottom=0)
    ax = ax15
    ax.set_xlabel(labx)
    addlab(ax, "$Q$", typeA=1)
    ax.set_ylim(bottom=0)
    ax = ax06
    ax.set_xlabel(labx)
    addlab(ax, "$Z_{eff}$", typeA=1)
    ax.set_ylim([0, 2.5])
    ax = ax14
    ax.set_xlabel(labx)
    addlab(ax, "$P_{sol}$ (MW)", typeA=1)
    ax.set_ylim(bottom=0)
    ax = ax16
    ax.set_xlabel(labx)
    addlab(ax, "$P_{fus}$ (MW)", typeA=1)
    ax.set_ylim(bottom=0)

    #
    ax = ax00g
    addlab(ax, "PICRF error (%)")
    ax.set_ylim([0, 25])
    ax.set_xlabel(labx)
    ax = ax01g
    addlab(ax, "Ip error (%)")
    ax.set_ylim([0, 25])
    ax.set_xlabel(labx)
    ax = ax10g
    addlab(ax, "Qe error (%)")
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax11g
    addlab(ax, "Qi error (%)")
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax20g
    addlab(ax, "Toroidal Voltage")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = axg
    addlab(ax, "Central electron temperature")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time from beginning (s)")
    ax = axg2
    addlab(ax, "ICH power to Ions at rho=0.3")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time from beginning (s)")
    #

    ax = ax00u
    addlab(ax, "$a/L_{nD}-0.5*a/L_{Ti}$, 0.15")
    ax.set_xlabel(labx)
    ax = ax01u
    addlab(ax, "$a/L_{nD}-0.5*a/L_{Ti}$")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax10u
    addlab(ax, "$f_{Bootstrap}$")
    ax.set_xlabel(labx)
    ax = ax11u
    addlab(ax, "$Q$")
    ax.set_xlabel("$P_{sol}$ (MW)")
    ax = ax12u
    addlab(ax, "$Q$")
    ax.set_xlabel("$P_{net}/P_{LH}$")
    ax = ax13u
    addlab(ax, "$\\langle\\rho^*\\rangle$")
    ax.set_xlabel(labx)
    ax = ax02u
    addlab(ax, "$\\tau_E$")
    ax.set_xlabel(labx)
    ax = ax03u
    addlab(ax, "$H_{98y2}$")
    ax.set_xlabel("$\\tau_E$")
    ax = ax04u
    addlab(ax, "$q_{95}$")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$I_p$ (MA)")
    ax = ax14u
    addlab(ax, "$W_{fast}/W_{tot}$")
    ax.set_xlabel(labx)
    ax = ax05u
    addlab(ax, "$f_{main}$")
    ax.set_xlabel(labx)
    ax.set_ylim([0, 1])
    ax = ax15u
    addlab(ax, "$Z_{lump}$")
    ax.set_xlabel(labx)
    ax = ax06u
    addlab(ax, "$A_{shaf}$ (m)")
    ax.set_xlabel(labx)
    ax = ax16u
    addlab(ax, "$V_{surf}$")
    ax.set_xlabel(labx)
    #

    ax = ax00r
    addlab(ax, "$T_e$ (keV)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax01r
    addlab(ax, "$T_i$ (keV)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax02r
    addlab(ax, "$n_e$ ($10^{20}m^{-3}$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax11r
    addlab(ax, "$q_i/q_e$")
    ax.set_ylim([0, 10])
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax03r
    addlab(ax, "$P_{rad}$ ($MWm^{-3}$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax12r
    addlab(ax, "$T_i/T_e$")
    ax.set_ylim([0, 2])
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax10r
    addlab(ax, "$\\nu_{ne}$")
    ax.set_xlabel("$\\nu_{eff}$")

    ax = ax13r
    addlab(ax, "$H_{98,y2}$")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$P_{net}/P_{LH}$")

    ax = ax04r
    addlab(ax, "$Te_{top}$ (keV)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$f_G=\\langle n_e\\rangle/n_G$")
    ax = ax14r
    addlab(ax, "$Te_{width}$")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$f_G=\\langle n_e\\rangle/n_G$")

    ax = ax05r
    addlab(ax, "$p_{top}$ (MPa)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$f_G=\\langle n_e\\rangle/n_G$")
    ax = ax15r
    ax.set_xlabel(labx)
    addlab(ax, "$\\beta_N$")
    ax.set_ylim([0, 3.0])

    ax = ax06r
    addlab(ax, "$a/L_{Ti}$")
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax16r
    addlab(ax, "$\\chi_i$ ($m^2/s$)")
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    # -----------------------

    ax = ax00f
    addlab(ax, "$p_{f,fus.}$ (MPa)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax01f
    addlab(ax, "$p_{f,min.}$ (MPa)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax02f
    addlab(ax, "$T_{f,fus.}$ (keV)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax03f
    addlab(ax, "$T_{f,min.}$ (keV)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax13f
    addlab(ax, "$\\rho_{pol}$")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax04f
    addlab(ax, "$P_{ICH,i}$ $MW/m^{3}$")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax14f
    addlab(ax, "$P_{ICH,e}$ $MW/m^{3}$")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax05f
    addlab(ax, "$J$ ($MA/m^2$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax15f
    addlab(ax, "$J_{OH}$ ($MA/m^2$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax10f
    addlab(ax, "$J_B$ ($MA/m^2$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax11f
    addlab(ax, "q-profile")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax12f
    addlab(ax, "$P_{OH}$ ($MW/m^{3}$)")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    ax = ax00f2
    addlab(ax, "$Q_{e}$ ($MW/m^{2}$)")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")
    ax = ax01f2
    addlab(ax, "$Q_{i}$ ($MW/m^{2}$)")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho_N$")

    return fig1, fig2, fig3, fig4, fig5


if __name__ == "__main__":
    folder_in = "~/mitim_runs/sims/sparc2/div_set2/"
    sims = np.arange(37, 48 + 1, 1)

    cdfs, runIDs = [], []
    for i in sims:
        subfolder = f"{folder_in}/Evaluation.{i}/FolderTRANSP/"

        nam = IOtools.findFileByExtension(subfolder, ".CDF")
        file = f"{subfolder}/{nam}.CDF"
        try:
            runIDs.append(i)
            cdfs.append(CDFtools.CDFreactor(file))
        except:
            print("problem " + str(i))

    fn = GUItools.FigureNotebook("SPARC scenarios", geometry="1700x900")
    figs = plotDB(cdfs, runIDs=runIDs, fn=fn)
