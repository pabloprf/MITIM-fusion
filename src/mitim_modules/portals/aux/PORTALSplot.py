import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.portals import PORTALStools

from mitim_tools.misc_tools.IOtools import printMsg as print

factor_dw0dr = 1e-5
label_dw0dr = "$-d\\omega_0/dr$ (krad/s/cm)"

# ---------------------------------------------------------------------------------------------------------------------
# Plotting methods for PORTALS class
# ---------------------------------------------------------------------------------------------------------------------


def PORTALSanalyzer_plotMetrics(
    self,
    fig=None,
    indexToMaximize=None,
    plotAllFluxes=False,
    index_extra=None,
    stds=2,
    plotFlows=True,
    fontsize_leg=5,
    includeRicci=True,
    useConvectiveFluxes=False,  # By default, plot in real particle units
    file_save=None,
):
    print("- Plotting PORTALS Metrics")

    if index_extra is not None:
        self.iextra = index_extra

    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(15, 8))

    numprofs = len(self.ProfilesPredicted)

    grid = plt.GridSpec(nrows=8, ncols=numprofs + 1, hspace=0.3, wspace=0.35)

    # Te
    axTe = fig.add_subplot(grid[:4, 0])
    axTe.set_title("Electron Temperature")
    axTe_g = fig.add_subplot(grid[4:6, 0])
    axTe_f = fig.add_subplot(grid[6:, 0])

    axTi = fig.add_subplot(grid[:4, 1])
    axTi.set_title("Ion Temperature")
    axTi_g = fig.add_subplot(grid[4:6, 1])
    axTi_f = fig.add_subplot(grid[6:, 1])

    cont = 0
    if "ne" in self.ProfilesPredicted:
        axne = fig.add_subplot(grid[:4, 2 + cont])
        axne.set_title("Electron Density")
        axne_g = fig.add_subplot(grid[4:6, 2 + cont])
        axne_f = fig.add_subplot(grid[6:, 2 + cont])
        cont += 1
    else:
        axne = axne_g = axne_f = None

    if self.runWithImpurity:
        labIon = f"Ion {self.runWithImpurity+1} ({self.profiles[0].Species[self.runWithImpurity]['N']}{int(self.profiles[0].Species[self.runWithImpurity]['Z'])},{int(self.profiles[0].Species[self.runWithImpurity]['A'])})"
        axnZ = fig.add_subplot(grid[:4, 2 + cont])
        axnZ.set_title(f"{labIon} Density")
        axnZ_g = fig.add_subplot(grid[4:6, 2 + cont])
        axnZ_f = fig.add_subplot(grid[6:, 2 + cont])
        cont += 1
    else:
        axnZ = axnZ_g = axnZ_f = None

    if self.runWithRotation:
        axw0 = fig.add_subplot(grid[:4, 2 + cont])
        axw0.set_title("Rotation")
        axw0_g = fig.add_subplot(grid[4:6, 2 + cont])
        axw0_f = fig.add_subplot(grid[6:, 2 + cont])
    else:
        axw0 = axw0_g = axw0_f = None

    axQ = fig.add_subplot(grid[:2, numprofs])
    axA = fig.add_subplot(grid[2:4, numprofs])
    axC = fig.add_subplot(grid[4:6, numprofs])
    axR = fig.add_subplot(grid[6:8, numprofs])

    if indexToMaximize is None:
        indexToMaximize = self.ibest
    if indexToMaximize < 0:
        indexToMaximize = self.ilast + 1 + indexToMaximize

    # ---------------------------------------------------------------------------------------------------------
    # Plot all profiles
    # ---------------------------------------------------------------------------------------------------------

    lwt = 0.1
    lw = 0.2
    alph = 0.6
    for i, p in enumerate(self.profiles):
        if p is not None:
            if i < 5:
                col = "k"
            else:
                col = "b"

            if i == 0:
                lab = "Training"
            elif i == 5:
                lab = "Optimization"
            else:
                lab = ""

            ix = np.argmin(
                np.abs(p.profiles["rho(-)"] - self.tgyros[self.i0].rho[0][-1])
            )
            axTe.plot(
                p.profiles["rho(-)"],
                p.profiles["te(keV)"],
                lw=lw,
                color=col,
                label=lab,
                alpha=alph,
            )
            axTe_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLTe"][:ix],
                lw=lw,
                color=col,
                alpha=alph,
            )
            axTi.plot(
                p.profiles["rho(-)"],
                p.profiles["ti(keV)"][:, 0],
                lw=lw,
                color=col,
                label=lab,
                alpha=alph,
            )
            axTi_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLTi"][:ix, 0],
                lw=lw,
                color=col,
                alpha=alph,
            )
            if axne is not None:
                axne.plot(
                    p.profiles["rho(-)"],
                    p.profiles["ne(10^19/m^3)"] * 1e-1,
                    lw=lw,
                    color=col,
                    label=lab,
                    alpha=alph,
                )
                axne_g.plot(
                    p.profiles["rho(-)"][:ix],
                    p.derived["aLne"][:ix],
                    lw=lw,
                    color=col,
                    alpha=alph,
                )

            if axnZ is not None:
                axnZ.plot(
                    p.profiles["rho(-)"],
                    p.profiles["ni(10^19/m^3)"][:, self.runWithImpurity] * 1e-1,
                    lw=lw,
                    color=col,
                    label=lab,
                    alpha=alph,
                )
                axnZ_g.plot(
                    p.profiles["rho(-)"][:ix],
                    p.derived["aLni"][:ix, self.runWithImpurity],
                    lw=lw,
                    color=col,
                    alpha=alph,
                )

            if axw0 is not None:
                axw0.plot(
                    p.profiles["rho(-)"],
                    p.profiles["w0(rad/s)"] * 1e-3,
                    lw=lw,
                    color=col,
                    label=lab,
                    alpha=alph,
                )
                axw0_g.plot(
                    p.profiles["rho(-)"][:ix],
                    p.derived["dw0dr"][:ix] * factor_dw0dr,
                    lw=lw,
                    color=col,
                    alpha=alph,
                )

        t = self.tgyros[i]
        if (t is not None) and plotAllFluxes:
            axTe_f.plot(
                t.rho[0],
                t.Qe_sim_turb[0] + t.Qe_sim_neo[0],
                "-",
                c=col,
                lw=lwt,
                alpha=alph,
            )
            axTe_f.plot(t.rho[0], t.Qe_tar[0], "--", c=col, lw=lwt, alpha=alph)
            axTi_f.plot(
                t.rho[0],
                t.QiIons_sim_turb_thr[0] + t.QiIons_sim_neo_thr[0],
                "-",
                c=col,
                lw=lwt,
                alpha=alph,
            )
            axTi_f.plot(t.rho[0], t.Qi_tar[0], "--", c=col, lw=lwt, alpha=alph)

            if useConvectiveFluxes:
                Ge, Ge_tar = t.Ce_sim_turb + t.Ce_sim_neo, t.Ce_tar
            else:
                Ge, Ge_tar = (t.Ge_sim_turb + t.Ge_sim_neo), t.Ge_tar

            if axne_f is not None:
                axne_f.plot(t.rho[0], Ge[0], "-", c=col, lw=lwt, alpha=alph)
                axne_f.plot(t.rho[0], Ge_tar[0]*(1-int(self.forceZeroParticleFlux)), "--", c=col, lw=lwt, alpha=alph)

            if axnZ_f is not None:
                if useConvectiveFluxes:
                    GZ, GZ_tar = (
                        t.Ci_sim_turb[self.runWithImpurity, :, :]
                        + t.Ci_sim_turb[self.runWithImpurity, :, :],
                        t.Ci_tar[self.runWithImpurity, :, :],
                    )
                else:
                    GZ, GZ_tar = (
                        t.Gi_sim_turb[self.runWithImpurity, :, :]
                        + t.Gi_sim_neo[self.runWithImpurity, :, :]
                    ), t.Gi_tar[self.runWithImpurity, :, :]

                axnZ_f.plot(t.rho[0], GZ[0], "-", c=col, lw=lwt, alpha=alph)
                axnZ_f.plot(t.rho[0], GZ_tar[0], "--", c=col, lw=lwt, alpha=alph)

            if axw0_f is not None:
                axw0_f.plot(
                    t.rho[0],
                    t.Mt_sim_turb[0] + t.Mt_sim_neo[0],
                    "-",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )
                axw0_f.plot(t.rho[0], t.Mt_tar[0], "--", c=col, lw=lwt, alpha=alph)

    # ---------------------------------------------------------------------------------------------------------

    msFlux = 3

    for cont, (indexUse, col, lab) in enumerate(
        zip(
            [self.i0, self.ibest, self.iextra],
            ["r", "g", "m"],
            [
                f"Initial (#{self.i0})",
                f"Best (#{self.ibest})",
                f"Last (#{self.iextra})",
            ],
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.profiles)):
            continue

        p = self.profiles[indexUse]
        t = self.tgyros[indexUse]

        ix = np.argmin(np.abs(p.profiles["rho(-)"] - t.rho[0][-1]))
        axTe.plot(
            p.profiles["rho(-)"], p.profiles["te(keV)"], lw=2, color=col, label=lab
        )
        axTe_g.plot(
            p.profiles["rho(-)"][:ix],
            p.derived["aLTe"][:ix],
            "-",
            markersize=msFlux,
            lw=2,
            color=col,
        )
        axTi.plot(
            p.profiles["rho(-)"],
            p.profiles["ti(keV)"][:, 0],
            lw=2,
            color=col,
            label=lab,
        )
        axTi_g.plot(
            p.profiles["rho(-)"][:ix],
            p.derived["aLTi"][:ix, 0],
            "-",
            markersize=msFlux,
            lw=2,
            color=col,
        )
        if axne is not None:
            axne.plot(
                p.profiles["rho(-)"],
                p.profiles["ne(10^19/m^3)"] * 1e-1,
                lw=2,
                color=col,
                label=lab,
            )
            axne_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLne"][:ix],
                "-",
                markersize=msFlux,
                lw=2,
                color=col,
            )

        if axnZ is not None:
            axnZ.plot(
                p.profiles["rho(-)"],
                p.profiles["ni(10^19/m^3)"][:, self.runWithImpurity] * 1e-1,
                lw=2,
                color=col,
                label=lab,
            )
            axnZ_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["aLni"][:ix, self.runWithImpurity],
                markersize=msFlux,
                lw=2,
                color=col,
            )

        if axw0 is not None:
            axw0.plot(
                p.profiles["rho(-)"],
                p.profiles["w0(rad/s)"] * 1e-3,
                lw=2,
                color=col,
                label=lab,
            )
            axw0_g.plot(
                p.profiles["rho(-)"][:ix],
                p.derived["dw0dr"][:ix] * factor_dw0dr,
                "-",
                markersize=msFlux,
                lw=2,
                color=col,
            )

        plotFluxComparison(
            p,
            t,
            axTe_f,
            axTi_f,
            axne_f,
            axnZ_f,
            axw0_f,
            runWithImpurity=self.runWithImpurity,
            fontsize_leg=fontsize_leg,
            stds=stds,
            col=col,
            lab=lab,
            msFlux=msFlux,
            forceZeroParticleFlux=self.forceZeroParticleFlux,
            useConvectiveFluxes=useConvectiveFluxes,
            maxStore=indexToMaximize == indexUse,
            decor=self.ibest == indexUse,
            plotFlows=plotFlows and (self.ibest == indexUse),
        )

    ax = axTe
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$T_e$ (keV)")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])
    ax.legend(prop={"size": fontsize_leg * 1.5})

    ax = axTe_g
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$a/L_{Te}$")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])

    ax = axTi
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$T_i$ (keV)")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])

    ax = axTi_g
    GRAPHICStools.addDenseAxis(ax)
    # ax.set_xlabel('$\\rho_N$')
    ax.set_ylabel("$a/L_{Ti}$")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])

    if axne is not None:
        ax = axne
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

        ax = axne_g
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$a/L_{ne}$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

    if axnZ is not None:
        ax = axnZ
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

        GRAPHICStools.addScientificY(ax)

    if axnZ_g is not None:
        ax = axnZ_g
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$a/L_{nZ}$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])

    if axw0 is not None:
        ax = axw0
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel("$w_0$ (krad/s)")
        ax.set_xlim([0, 1])
        ax.set_xticklabels([])

    if axw0_g is not None:
        ax = axw0_g
        GRAPHICStools.addDenseAxis(ax)
        # ax.set_xlabel('$\\rho_N$')
        ax.set_ylabel(label_dw0dr)
        ax.set_xlim([0, 1])
        ax.set_xticklabels([])

    ax = axC
    if "te" in self.ProfilesPredicted:
        v = self.resTeM
        ax.plot(
            self.evaluations,
            v,
            "-o",
            lw=0.5,
            c="b",
            markersize=2,
            label=self.labelsFluxes["te"],
        )
    if "ti" in self.ProfilesPredicted:
        v = self.resTiM
        ax.plot(
            self.evaluations,
            v,
            "-s",
            lw=0.5,
            c="m",
            markersize=2,
            label=self.labelsFluxes["ti"],
        )
    if "ne" in self.ProfilesPredicted:
        v = self.resneM
        ax.plot(
            self.evaluations,
            v,
            "-*",
            lw=0.5,
            c="k",
            markersize=2,
            label=self.labelsFluxes["ne"],
        )
    if "nZ" in self.ProfilesPredicted:
        v = self.resnZM
        ax.plot(
            self.evaluations,
            v,
            "-v",
            lw=0.5,
            c="c",
            markersize=2,
            label=self.labelsFluxes["nZ"],
        )
    if "w0" in self.ProfilesPredicted:
        v = self.resw0M
        ax.plot(
            self.evaluations,
            v,
            "-v",
            lw=0.5,
            c="darkred",
            markersize=2,
            label=self.labelsFluxes["w0"],
        )

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            [self.i0, self.ibest, self.iextra],
            ["r", "g", "m"],
            ["Initial", "Best", "Last"],
            ["o", "s", "*"],
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.profiles)):
            continue
        if "te" in self.ProfilesPredicted:
            v = self.resTeM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "ti" in self.ProfilesPredicted:
            v = self.resTiM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "ne" in self.ProfilesPredicted:
            v = self.resneM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "nZ" in self.ProfilesPredicted:
            v = self.resnZM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "w0" in self.ProfilesPredicted:
            v = self.resw0M
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )

    # Plot las point as check
    ax.plot([self.evaluations[-1]], [self.resCheck[-1]], "-o", markersize=2, color="k")

    separator = self.opt_fun.prfs_model.Optim["initialPoints"] + 0.5 - 1

    if self.evaluations[-1] < separator:
        separator = None

    GRAPHICStools.addDenseAxis(ax, n=5)

    ax.set_ylabel("Channel residual")
    ax.set_xlim(left=-0.2)
    # ax.set_ylim(bottom=0)
    try:
        ax.set_yscale("log")
    except:
        pass
    GRAPHICStools.addLegendApart(
        ax,
        ratio=0.9,
        withleg=True,
        size=fontsize_leg * 1.5,
        title="Channels $\\frac{1}{N_c}L_1$",
    )
    ax.set_xticklabels([])

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=8,
            fromtop=0.1,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    ax = axR

    for resChosen, label, c in zip(
        [self.resM, self.resCheck],
        ["OF: $\\frac{1}{N}L_2$", "$\\frac{1}{N}L_1$"],
        ["olive", "rebeccapurple"],
    ):
        ax.plot(
            self.evaluations, resChosen, "-o", lw=1.0, c=c, markersize=2, label=label
        )
        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                [self.i0, self.ibest, self.iextra],
                ["r", "g", "m"],
                ["Initial", "Best", "Last"],
                ["o", "s", "*"],
            )
        ):
            if (indexUse is None) or (indexUse >= len(self.profiles)):
                continue
            ax.plot(
                [self.evaluations[indexUse]],
                [resChosen[indexUse]],
                "o",
                color=col,
                markersize=4,
            )

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=12,
            fromtop=0.75,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    GRAPHICStools.addDenseAxis(ax, n=5)
    ax.set_xlabel("Iterations (calls/radius)")
    ax.set_ylabel("Residual")
    ax.set_xlim(left=0)
    try:
        ax.set_yscale("log")
    except:
        pass
    GRAPHICStools.addLegendApart(
        ax,
        ratio=0.9,
        withleg=True,
        size=fontsize_leg * 2.0,
        title="Residuals",
    )

    ax = axA

    ax.plot(
        self.DVdistMetric_x,
        self.DVdistMetric_y,
        "-o",
        c="olive",
        lw=1.0,
        markersize=2,
        label=r"$||\Delta x||_\infty$",
    )  #'$\\Delta$ $a/L_{X}$ (%)')

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            [self.i0, self.ibest, self.iextra],
            ["r", "g", "m"],
            ["Initial", "Best", "Last"],
            ["o", "s", "*"],
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.profiles)):
            continue
        v = self.chiR_Ricci
        # try:
        #     axt.plot(
        #         [self.evaluations[indexUse]],
        #         [self.DVdistMetric_y[indexUse]],
        #         "o",
        #         color=col,
        #         markersize=4,
        #     )
        # except:
        #     pass

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=12,
            fromtop=0.75,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    ax.set_ylabel("$\\Delta$ $a/L_{X}$ (%)")
    ax.set_xlim(left=0)
    try:
        ax.set_yscale("log")
    except:
        pass
    ax.set_xticklabels([])

    if includeRicci and self.chiR_Ricci is not None:
        axt = axA.twinx()
        (l2,) = axt.plot(
            self.DVdistMetric_x,
            self.DVdistMetric_y,
            "-o",
            c="olive",
            lw=1.0,
            markersize=2,
            label="$\\Delta$ $a/L_{X}$",
        )
        axt.plot(
            self.evaluations,
            self.chiR_Ricci,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$\\chi_R$",
        )
        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                [self.i0, self.ibest, self.iextra],
                ["r", "g", "m"],
                ["Initial", "Best", "Last"],
                ["o", "s", "*"],
            )
        ):
            if (indexUse is None) or (indexUse >= len(self.profiles)):
                continue
            v = self.chiR_Ricci
            axt.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                "o",
                color=col,
                markersize=4,
            )
        axt.set_ylabel("Ricci Metric, $\\chi_R$")
        axt.set_ylim([0, 1])
        axt.legend(loc="best", prop={"size": fontsize_leg * 1.5})
        l2.set_visible(False)
    elif self.aLTn_perc is not None:
        ax = axA  # .twinx()

        x = self.evaluations

        if len(x) > len(self.aLTn_perc):
            x = x[:-1]

        x0, aLTn_perc0 = [], []
        for i in range(len(self.aLTn_perc)):
            if self.aLTn_perc[i] is not None:
                x0.append(x[i])
                aLTn_perc0.append(self.aLTn_perc[i].mean())
        ax.plot(
            x0,
            aLTn_perc0,
            "-o",
            c="rebeccapurple",
            lw=1.0,
            markersize=2,
            label="$\\Delta$ $a/L_{X}^*$ (%)",
        )

        v = self.aLTn_perc[self.i0].mean()
        ax.plot([self.evaluations[self.i0]], v, "o", color="r", markersize=4)
        try:
            v = self.aLTn_perc[self.ibest].mean()
            ax.plot(
                [self.evaluations[self.ibest]],
                [v],
                "o",
                color="g",
                markersize=4,
            )
        except:
            pass

        ax.set_ylabel("$\\Delta$ $a/L_{X}^*$ (%)")
        try:
            ax.set_yscale("log")
        except:
            pass

        (l2,) = axA.plot(
            x0,
            aLTn_perc0,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$\\Delta$ $a/L_{X}^*$ (%)",
        )
        axA.legend(loc="upper center", prop={"size": 7})
        l2.set_visible(False)

    else:
        GRAPHICStools.addDenseAxis(ax, n=5)

    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=False, size=fontsize_leg
    )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')

    ax = axQ

    isThereFusion = np.nanmax(self.FusionGain) > 0

    if isThereFusion:
        v = self.FusionGain
        axt6 = ax.twinx()  # None
    else:
        v = self.tauE
        axt6 = None
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")

    ax.plot(self.evaluations, v, "-o", lw=1.0, c="olive", markersize=2, label="$Q$")
    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            [self.i0, self.ibest, self.iextra],
            ["r", "g", "m"],
            ["Initial", "Best", "Last"],
            ["o", "s", "*"],
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.profiles)):
            continue
        ax.plot(
            [self.evaluations[indexUse]], [v[indexUse]], "o", color=col, markersize=4
        )

    vmin, vmax = np.max([0, np.nanmin(v)]), np.nanmax(v)
    ext = 0.8
    ax.set_ylim([vmin * (1 - ext), vmax * (1 + ext)])
    ax.set_ylim([0, vmax * (1 + ext)])

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="",
            orientation="vertical",
            color="k",
            lw=0.5,
            ls="-.",
            alpha=1.0,
            fontsize=8,
            fromtop=0.1,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    if axt6 is None:
        GRAPHICStools.addDenseAxis(ax, n=5, grid=axt6 is None)

    if isThereFusion:
        ax.set_ylabel("$Q$")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=True, size=fontsize_leg
        )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')
    else:
        ax.set_ylabel("$\\tau_E$ (s)")
        GRAPHICStools.addLegendApart(
            ax, ratio=0.9, withleg=False, size=fontsize_leg
        )  # ax.legend(prop={'size':fontsize_leg},loc='lower left')
    ax.set_xlim(left=0)
    ax.set_xticklabels([])

    if separator is not None:
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="surrogate",
            orientation="vertical",
            color="b",
            lw=0.25,
            ls="--",
            alpha=1.0,
            fontsize=7,
            fromtop=0.72,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="left",
            separation=0.2,
        )
        GRAPHICStools.drawLineWithTxt(
            ax,
            separator,
            label="training",
            orientation="vertical",
            color="k",
            lw=0.01,
            ls="--",
            alpha=1.0,
            fontsize=7,
            fromtop=0.72,
            fontweight="normal",
            verticalalignment="bottom",
            horizontalalignment="right",
            separation=-0.2,
        )

    if (axt6 is not None) and (isThereFusion):
        v = self.FusionPower
        axt6.plot(
            self.evaluations,
            v,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$P_{fus}$",
        )
        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                [self.i0, self.ibest, self.iextra],
                ["r", "g", "m"],
                ["Initial", "Best", "Last"],
                ["o", "s", "*"],
            )
        ):
            if (indexUse is None) or (indexUse >= len(self.profiles)):
                continue
            axt6.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                "s",
                color=col,
                markersize=4,
            )

        axt6.set_ylabel("$P_{fus}$ (MW)")
        axt6.set_ylim(bottom=0)

        (l2,) = ax.plot(
            self.evaluations,
            v,
            "-o",
            lw=1.0,
            c="rebeccapurple",
            markersize=2,
            label="$P_{fus}$",
        )
        ax.legend(loc="lower left", prop={"size": fontsize_leg})
        l2.set_visible(False)

    for ax in [axQ, axA, axR, axC]:
        ax.set_xlim([0, len(self.FusionGain) + 2])

    # for ax in [axA,axR,axC]:
    # 	ax.yaxis.tick_right()
    # 	ax.yaxis.set_label_position("right")

    # print(
    #     "\t* Reminder: With the exception of the Residual plot, the rest are calculated with the original profiles, not necesarily modified by targets",
    #     typeMsg="i",
    # )

    # Save plot
    if file_save is not None:
        plt.savefig(file_save, transparent=True, dpi=300)


def PORTALSanalyzer_plotExpected(
    self, fig=None, stds=2, max_plot_points=4, plotNext=True
):
    print("- Plotting PORTALS Expected")

    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(18, 9))

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------

    trained_points = self.ilast + 1
    self.ibest = self.opt_fun.res.best_absolute_index

    # Best point
    plotPoints = [self.ibest]
    labelAssigned = [f"#{self.ibest} (best)"]

    # Last point
    if (trained_points - 1) != self.ibest:
        plotPoints.append(trained_points - 1)
        labelAssigned.append(f"#{trained_points-1} (last)")

    # Last ones
    i = 0
    while len(plotPoints) < max_plot_points:
        if (trained_points - 2 - i) < 1:
            break
        if (trained_points - 2 - i) != self.ibest:
            plotPoints.append(trained_points - 2 - i)
            labelAssigned.append(f"#{trained_points-2-i}")
        i += 1

    # First point
    if 0 not in plotPoints:
        if len(plotPoints) == max_plot_points:
            plotPoints[-1] = 0
            labelAssigned[-1] = "#0 (base)"
        else:
            plotPoints.append(0)
            labelAssigned.append("#0 (base)")

    if fig is None:
        fig = plt.figure(figsize=(12, 8))

    model = self.step.GP["combined_model"]

    x_train_num = self.step.train_X.shape[0]

    # ---- Training
    x_train = torch.from_numpy(self.step.train_X).to(model.train_X)
    y_trainreal = torch.from_numpy(self.step.train_Y).to(model.train_X)
    yL_trainreal = torch.from_numpy(self.step.train_Ystd).to(model.train_X)
    yU_trainreal = torch.from_numpy(self.step.train_Ystd).to(model.train_X)

    y_train = model.predict(x_train)[0]

    # ---- Next
    y_next = yU_next = yL_next = None
    if plotNext:
        try:
            y_next, yU_next, yL_next, _ = model.predict(self.step.x_next)
        except:
            pass

    # ---- Plot

    numprofs = len(self.ProfilesPredicted)

    if numprofs <= 4:
        wspace = 0.3
    else:
        wspace = 0.5

    grid = plt.GridSpec(nrows=4, ncols=numprofs, hspace=0.2, wspace=wspace)

    axTe = fig.add_subplot(grid[0, 0])
    axTe.set_title("Electron Temperature")
    axTe_g = fig.add_subplot(grid[1, 0], sharex=axTe)
    axTe_f = fig.add_subplot(grid[2, 0], sharex=axTe)
    axTe_r = fig.add_subplot(grid[3, 0], sharex=axTe)

    axTi = fig.add_subplot(grid[0, 1], sharex=axTe)
    axTi.set_title("Ion Temperature")
    axTi_g = fig.add_subplot(grid[1, 1], sharex=axTe)
    axTi_f = fig.add_subplot(grid[2, 1], sharex=axTe)
    axTi_r = fig.add_subplot(grid[3, 1], sharex=axTe)

    cont = 0
    if "ne" in self.ProfilesPredicted:
        axne = fig.add_subplot(grid[0, 2 + cont], sharex=axTe)
        axne.set_title("Electron Density")
        axne_g = fig.add_subplot(grid[1, 2 + cont], sharex=axTe)
        axne_f = fig.add_subplot(grid[2, 2 + cont], sharex=axTe)
        axne_r = fig.add_subplot(grid[3, 2 + cont], sharex=axTe)
        cont += 1
    else:
        axne = axne_g = axne_f = axne_r = None
    if self.runWithImpurity:
        labIon = f"Ion {self.runWithImpurity+1} ({self.profiles[0].Species[self.runWithImpurity]['N']}{int(self.profiles[0].Species[self.runWithImpurity]['Z'])},{int(self.profiles[0].Species[self.runWithImpurity]['A'])})"
        axnZ = fig.add_subplot(grid[0, 2 + cont], sharex=axTe)
        axnZ.set_title(f"{labIon} Density")
        axnZ_g = fig.add_subplot(grid[1, 2 + cont], sharex=axTe)
        axnZ_f = fig.add_subplot(grid[2, 2 + cont], sharex=axTe)
        axnZ_r = fig.add_subplot(grid[3, 2 + cont], sharex=axTe)
        cont += 1
    else:
        axnZ = axnZ_g = axnZ_f = axnZ_r = None

    if self.runWithRotation:
        axw0 = fig.add_subplot(grid[0, 2 + cont], sharex=axTe)
        axw0.set_title("Rotation")
        axw0_g = fig.add_subplot(grid[1, 2 + cont], sharex=axTe)
        axw0_f = fig.add_subplot(grid[2, 2 + cont], sharex=axTe)
        axw0_r = fig.add_subplot(grid[3, 2 + cont], sharex=axTe)
        cont += 1
    else:
        axw0 = axw0_g = axw0_f = axw0_r = None

    colorsA = GRAPHICStools.listColors()
    colors = []
    coli = -1
    for label in labelAssigned:
        if "best" in label:
            colors.append("g")
        elif "last" in label:
            colors.append("m")
        elif "base" in label:
            colors.append("r")
        else:
            coli += 1
            while colorsA[coli] in ["g", "m", "r"]:
                coli += 1
            colors.append(colorsA[coli])

    rho = self.profiles[0].profiles["rho(-)"]
    roa = self.profiles[0].derived["roa"]
    rhoVals = self.TGYROparameters["RhoLocations"]
    roaVals = np.interp(rhoVals, rho, roa)
    lastX = roaVals[-1]

    # ---- Plot profiles
    cont = -1
    for i in plotPoints:
        cont += 1

        p = self.profiles[i]

        ix = np.argmin(np.abs(p.derived["roa"] - lastX)) + 1

        lw = 1.0 if cont > 0 else 1.5

        ax = axTe
        ax.plot(
            p.derived["roa"],
            p.profiles["te(keV)"],
            "-",
            c=colors[cont],
            label=labelAssigned[cont],
            lw=lw,
        )
        ax = axTi
        ax.plot(
            p.derived["roa"], p.profiles["ti(keV)"][:, 0], "-", c=colors[cont], lw=lw
        )
        if axne is not None:
            ax = axne
            ax.plot(
                p.derived["roa"],
                p.profiles["ne(10^19/m^3)"] * 1e-1,
                "-",
                c=colors[cont],
                lw=lw,
            )
        if axnZ is not None:
            ax = axnZ
            ax.plot(
                p.derived["roa"],
                p.profiles["ni(10^19/m^3)"][:, self.runWithImpurity] * 1e-1,
                "-",
                c=colors[cont],
                lw=lw,
            )
        if axw0 is not None:
            ax = axw0
            ax.plot(
                p.derived["roa"],
                p.profiles["w0(rad/s)"] * 1e-3,
                "-",
                c=colors[cont],
                lw=lw,
            )

        ax = axTe_g
        ax.plot(
            p.derived["roa"][:ix],
            p.derived["aLTe"][:ix],
            "-o",
            c=colors[cont],
            markersize=0,
            lw=lw,
        )
        ax = axTi_g
        ax.plot(
            p.derived["roa"][:ix],
            p.derived["aLTi"][:ix, 0],
            "-o",
            c=colors[cont],
            markersize=0,
            lw=lw,
        )
        if axne_g is not None:
            ax = axne_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["aLne"][:ix],
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )

        if axnZ_g is not None:
            ax = axnZ_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["aLni"][:ix, self.runWithImpurity],
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )
        if axw0_g is not None:
            ax = axw0_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["dw0dr"][:ix] * factor_dw0dr,
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )

    cont += 1

    # ---- Plot profiles next

    if self.profiles_next is not None:
        p = self.profiles_next
        roa = self.profiles_next_new.derived["roa"]
        dw0dr = self.profiles_next_new.derived["dw0dr"]

        ix = np.argmin(np.abs(roa - lastX)) + 1

        lw = 1.5

        ax = axTe
        ax.plot(
            roa,
            p.profiles["te(keV)"],
            "-",
            c="k",
            label=f"#{x_train_num} (next)",
            lw=lw,
        )
        ax = axTi
        ax.plot(roa, p.profiles["ti(keV)"][:, 0], "-", c="k", lw=lw)
        if axne is not None:
            ax = axne
            ax.plot(roa, p.profiles["ne(10^19/m^3)"] * 1e-1, "-", c="k", lw=lw)

        if axnZ is not None:
            ax = axnZ
            ax.plot(
                roa,
                p.profiles["ni(10^19/m^3)"][:, self.runWithImpurity] * 1e-1,
                "-",
                c="k",
                lw=lw,
            )
        if axw0 is not None:
            ax = axw0
            ax.plot(roa, p.profiles["w0(rad/s)"] * 1e-3, "-", c="k", lw=lw)

        ax = axTe_g
        ax.plot(roa[:ix], p.derived["aLTe"][:ix], "o-", c="k", markersize=0, lw=lw)
        ax = axTi_g
        ax.plot(roa[:ix], p.derived["aLTi"][:ix, 0], "o-", c="k", markersize=0, lw=lw)

        if axne_g is not None:
            ax = axne_g
            ax.plot(roa[:ix], p.derived["aLne"][:ix], "o-", c="k", markersize=0, lw=lw)

        if axnZ_g is not None:
            ax = axnZ_g
            ax.plot(
                roa[:ix],
                p.derived["aLni"][:ix, self.runWithImpurity],
                "-o",
                c="k",
                markersize=0,
                lw=lw,
            )
        if axw0_g is not None:
            ax = axw0_g
            ax.plot(
                roa[:ix], dw0dr[:ix] * factor_dw0dr, "-o", c="k", markersize=0, lw=lw
            )

        axTe_g_twin = axTe_g.twinx()
        axTi_g_twin = axTi_g.twinx()

        ranges = [-30, 30]

        rho = self.profiles_next_new.profiles["rho(-)"]
        rhoVals = self.TGYROparameters["RhoLocations"]
        roaVals = np.interp(rhoVals, rho, roa)

        p0 = self.profiles[0]
        zVals = []
        z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
        for roai in roaVals:
            zVals.append(np.interp(roai, roa, z))
        axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

        if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
            p0 = self.profiles[1]
            zVals = []
            z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4)

        axTe_g_twin.set_ylim(ranges)
        axTe_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        p0 = self.profiles[0]
        zVals = []
        z = (
            (p.derived["aLTi"][:, 0] - p0.derived["aLTi"][:, 0])
            / p0.derived["aLTi"][:, 0]
        ) * 100.0
        for roai in roaVals:
            zVals.append(np.interp(roai, roa, z))
        axTi_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

        if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
            p0 = self.profiles[1]
            zVals = []
            z = (
                (p.derived["aLTi"][:, 0] - p0.derived["aLTi"][:, 0])
                / p0.derived["aLTi"][:, 0]
            ) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTi_g_twin.plot(roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4)

        axTi_g_twin.set_ylim(ranges)
        axTi_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        for ax in [axTe_g_twin, axTi_g_twin]:
            ax.axhline(y=0, ls="-.", lw=0.2, c="k")

        if axne_g is not None:
            axne_g_twin = axne_g.twinx()

            p0 = self.profiles[0]
            zVals = []
            z = ((p.derived["aLne"] - p0.derived["aLne"]) / p0.derived["aLne"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axne_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.profiles[1]
                zVals = []
                z = (
                    (p.derived["aLne"] - p0.derived["aLne"]) / p0.derived["aLne"]
                ) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axne_g_twin.plot(
                    roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4
                )

            axne_g_twin.set_ylim(ranges)
            axne_g_twin.set_ylabel("(%) from last or best", fontsize=8)

            axne_g_twin.axhline(y=0, ls="-.", lw=0.2, c="k")

        if axnZ_g is not None:
            axnZ_g_twin = axnZ_g.twinx()

            p0 = self.profiles[0]
            zVals = []
            z = (
                (
                    p.derived["aLni"][:, self.runWithImpurity]
                    - p0.derived["aLni"][:, self.runWithImpurity]
                )
                / p0.derived["aLni"][:, self.runWithImpurity]
            ) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axnZ_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.profiles[1]
                zVals = []
                z = (
                    (
                        p.derived["aLni"][:, self.runWithImpurity]
                        - p0.derived["aLni"][:, self.runWithImpurity]
                    )
                    / p0.derived["aLni"][:, self.runWithImpurity]
                ) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axnZ_g_twin.plot(
                    roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4
                )

            axnZ_g_twin.set_ylim(ranges)
            axnZ_g_twin.set_ylabel("(%) from last or best", fontsize=8)
        else:
            axnZ_g_twin = None

        if axw0_g is not None:
            axw0_g_twin = axw0_g.twinx()

            p0 = self.profiles[0]
            zVals = []
            z = ((dw0dr - p0.derived["dw0dr"]) / p0.derived["dw0dr"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axw0_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.profiles[1]
                zVals = []
                z = ((dw0dr - p0.derived["dw0dr"]) / p0.derived["dw0dr"]) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axw0_g_twin.plot(
                    roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4
                )

            axw0_g_twin.set_ylim(ranges)
            axw0_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        else:
            axw0_g_twin = None

        for ax in [axnZ_g_twin, axw0_g_twin]:
            if ax is not None:
                ax.axhline(y=0, ls="-.", lw=0.2, c="k")

    else:
        axTe_g_twin = axTi_g_twin = axne_g_twin = axnZ_g_twin = axw0_g_twin = None

    # ---- Plot fluxes
    cont = plotVars(
        self.opt_fun.prfs_model,
        y_trainreal,
        [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
        [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r],
        contP=-1,
        lines=["-s", "--o"],
        plotPoints=plotPoints,
        yerr=[yL_trainreal * stds, yU_trainreal * stds],
        lab="",
        plotErr=np.append([True], [False] * len(y_trainreal)),
        colors=colors,
    )
    _ = plotVars(
        self.opt_fun.prfs_model,
        y_train,
        [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
        [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r],
        contP=-1,
        lines=["-.*", None],
        plotPoints=plotPoints,
        plotResidual=False,
        lab=" (surr)",
        colors=colors,
    )  # ,yerr=[yL_train,yU_train])

    if y_next is not None:
        cont = plotVars(
            self.opt_fun.prfs_model,
            y_next,
            [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
            [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r],
            contP=cont,
            lines=["-s", "--o"],
            yerr=[y_next - yL_next * stds / 2.0, yU_next - y_next * stds / 2.0],
            plotPoints=None,
            color="k",
            plotErr=[True],
            colors=colors,
        )

    # ---------------
    n = 10  # 5
    ax = axTe
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylabel("Te (keV)")
    ax.set_ylim(bottom=0)
    GRAPHICStools.addDenseAxis(ax, n=n)
    # ax.	set_xticklabels([])
    ax = axTi
    ax.set_xlim([0, 1])
    ax.set_ylabel("Ti (keV)")
    ax.set_ylim(bottom=0)
    GRAPHICStools.addDenseAxis(ax, n=n)
    # ax.set_xticklabels([])
    if axne is not None:
        ax = axne
        ax.set_xlim([0, 1])
        ax.set_ylabel("ne ($10^{20}m^{-3}$)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax, n=n)
    # ax.set_xticklabels([])

    if axnZ is not None:
        ax = axnZ
        ax.set_xlim([0, 1])
        ax.set_ylabel("nZ ($10^{20}m^{-3}$)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axw0 is not None:
        ax = axw0
        ax.set_xlim([0, 1])
        ax.set_ylabel("$w_0$ (krad/s)")
        GRAPHICStools.addDenseAxis(ax, n=n)

    roacoarse = self.powerstate.plasma["roa"][0, 1:].cpu().numpy()

    ax = axTe_g
    ax.set_xlim([0, 1])
    ax.set_ylabel("$a/L_{Te}$")
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0,5]);
    # ax.set_xticklabels([])
    if axTe_g_twin is not None:
        axTe_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
        if len(roacoarse) < 6:
            axTe_g_twin.set_xticks([round(i, 2) for i in roacoarse])
        GRAPHICStools.addDenseAxis(axTe_g_twin, n=n)
    else:
        GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTi_g
    ax.set_xlim([0, 1])
    ax.set_ylabel("$a/L_{Ti}$")
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0,5]);
    # ax.set_xticklabels([])
    if axTi_g_twin is not None:
        axTi_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
        if len(roacoarse) < 6:
            axTi_g_twin.set_xticks([round(i, 2) for i in roacoarse])
        GRAPHICStools.addDenseAxis(axTi_g_twin, n=n)
    else:
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axne_g is not None:
        ax = axne_g
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{ne}$")
        ax.set_ylim(bottom=0)
        # ax.set_ylim([0,5]);
        # ax.set_xticklabels([])
        if axne_g_twin is not None:
            axne_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
            if len(roacoarse) < 6:
                axne_g_twin.set_xticks([round(i, 2) for i in roacoarse])
            GRAPHICStools.addDenseAxis(axne_g_twin, n=n)
        else:
            GRAPHICStools.addDenseAxis(ax, n=n)

    if axnZ_g is not None:
        ax = axnZ_g
        ax.set_xlim([0, 1])
        ax.set_ylabel("$a/L_{nZ}$")
        ax.set_ylim(bottom=0)
        # ax.set_ylim([0,5]);
        if axnZ_g_twin is not None:
            axnZ_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
            if len(roacoarse) < 6:
                axnZ_g_twin.set_xticks([round(i, 2) for i in roacoarse])
            GRAPHICStools.addDenseAxis(axnZ_g_twin, n=n)
        else:
            GRAPHICStools.addDenseAxis(ax, n=n)

    if axw0_g is not None:
        ax = axw0_g
        ax.set_xlim([0, 1])
        ax.set_ylabel(label_dw0dr)
        # ax.set_ylim(bottom=0); #ax.set_ylim([0,5]);
        if axw0_g_twin is not None:
            axw0_g_twin.set_yticks(np.arange(ranges[0], ranges[1], 5))
            if len(roacoarse) < 6:
                axw0_g_twin.set_xticks([round(i, 2) for i in roacoarse])
            GRAPHICStools.addDenseAxis(axw0_g_twin, n=n)
        else:
            GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTe_f
    ax.set_xlim([0, 1])
    ax.set_ylabel(self.labelsFluxes["te"])
    ax.set_ylim(bottom=0)
    # ax.legend(loc='best',prop={'size':6})
    # ax.set_xticklabels([])
    GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTi_f
    ax.set_xlim([0, 1])
    ax.set_ylabel(self.labelsFluxes["ti"])
    ax.set_ylim(bottom=0)
    # ax.set_xticklabels([])
    GRAPHICStools.addDenseAxis(ax, n=n)

    if axne_f is not None:
        ax = axne_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(self.labelsFluxes["ne"])
        # GRAPHICStools.addDenseAxis(ax,n=n)
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axnZ_f is not None:
        ax = axnZ_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(self.labelsFluxes["nZ"])
        # GRAPHICStools.addDenseAxis(ax,n=n)
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axw0_f is not None:
        ax = axw0_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(self.labelsFluxes["w0"])
        # GRAPHICStools.addDenseAxis(ax,n=n)
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    ax = axTe_r
    ax.set_xlim([0, 1])
    ax.set_xlabel("$r/a$")
    ax.set_ylabel("Residual " + self.labelsFluxes["te"])
    GRAPHICStools.addDenseAxis(ax, n=n)
    ax.axhline(y=0, lw=0.5, ls="--", c="k")

    ax = axTi_r
    ax.set_xlim([0, 1])
    ax.set_xlabel("$r/a$")
    ax.set_ylabel("Residual " + self.labelsFluxes["ti"])
    GRAPHICStools.addDenseAxis(ax, n=n)
    ax.axhline(y=0, lw=0.5, ls="--", c="k")

    if axne_r is not None:
        ax = axne_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + self.labelsFluxes["ne"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")  #

    if axnZ_r is not None:
        ax = axnZ_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + self.labelsFluxes["nZ"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

    if axw0_r is not None:
        ax = axw0_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + self.labelsFluxes["w0"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

    try:
        Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = varToReal(
            y_trainreal[self.opt_fun.prfs_model.BOmetrics["overall"]["indBest"], :]
            .detach()
            .cpu()
            .numpy(),
            self.opt_fun.prfs_model,
        )
        rangePlotResidual = np.max([np.max(Qe_tar), np.max(Qi_tar), np.max(Ge_tar)])
        for ax in [axTe_r, axTi_r, axne_r]:
            ax.set_ylim(
                [-rangePlotResidual * 0.5, rangePlotResidual * 0.5]
            )  # 50% of max targets
    except:
        pass


def PORTALSanalyzer_plotSummary(self, fn=None):
    print("- Plotting PORTALS summary of TGYRO and PROFILES classes")

    indecesPlot = [
        self.ibest,
        self.i0,
        self.iextra,
    ]

    # -------------------------------------------------------
    # Plot TGYROs
    # -------------------------------------------------------

    self.tgyros[indecesPlot[1]].plot(fn=fn, prelabel=f"({indecesPlot[1]}) TGYRO - ")
    if indecesPlot[0] < len(self.tgyros):
        self.tgyros[indecesPlot[0]].plot(fn=fn, prelabel=f"({indecesPlot[0]}) TGYRO - ")

    # -------------------------------------------------------
    # Plot PROFILES
    # -------------------------------------------------------

    figs = [
        fn.add_figure(label="PROFILES - Profiles"),
        fn.add_figure(label="PROFILES - Powers"),
        fn.add_figure(label="PROFILES - Geometry"),
        fn.add_figure(label="PROFILES - Gradients"),
        fn.add_figure(label="PROFILES - Flows"),
        fn.add_figure(label="PROFILES - Other"),
        fn.add_figure(label="PROFILES - Impurities"),
    ]

    if indecesPlot[0] < len(self.profiles):
        PROFILEStools.plotAll(
            [
                self.profiles[indecesPlot[1]],
                self.profiles[indecesPlot[0]],
            ],
            figs=figs,
            extralabs=[f"{indecesPlot[1]}", f"{indecesPlot[0]}"],
        )

    # -------------------------------------------------------
    # Plot Comparison
    # -------------------------------------------------------

    profile_original = self.mitim_runs[0]["tgyro"].results["tglf_neo"].profiles
    profile_best = self.mitim_runs[self.ibest]["tgyro"].results["tglf_neo"].profiles

    profile_original_unCorrected = self.mitim_runs["profiles_original_un"]
    profile_original_0 = self.mitim_runs["profiles_original"]

    fig4 = fn.add_figure(label="PROFILES Comparison")
    grid = plt.GridSpec(
        2,
        np.max([3, len(self.ProfilesPredicted)]),
        hspace=0.3,
        wspace=0.3,
    )
    axs4 = [
        fig4.add_subplot(grid[0, 0]),
        fig4.add_subplot(grid[1, 0]),
        fig4.add_subplot(grid[0, 1]),
        fig4.add_subplot(grid[1, 1]),
        fig4.add_subplot(grid[0, 2]),
        fig4.add_subplot(grid[1, 2]),
    ]

    cont = 1
    if self.runWithImpurity:
        axs4.append(fig4.add_subplot(grid[0, 2 + cont]))
        axs4.append(fig4.add_subplot(grid[1, 2 + cont]))
        cont += 1
    if self.runWithRotation:
        axs4.append(fig4.add_subplot(grid[0, 2 + cont]))
        axs4.append(fig4.add_subplot(grid[1, 2 + cont]))

    colors = GRAPHICStools.listColors()

    for i, (profiles, label, alpha) in enumerate(
        zip(
            [
                profile_original_unCorrected,
                profile_original_0,
                profile_original,
                profile_best,
            ],
            ["Original", "Corrected", "Initial", "Final"],
            [0.2, 1.0, 1.0, 1.0],
        )
    ):
        profiles.plotGradients(
            axs4,
            color=colors[i],
            label=label,
            lastRho=self.TGYROparameters["RhoLocations"][-1],
            alpha=alpha,
            useRoa=True,
            RhoLocationsPlot=self.TGYROparameters["RhoLocations"],
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
        )

    axs4[0].legend(loc="best")


def PORTALSanalyzer_plotRanges(self, fig=None):
    if fig is None:
        plt.ion()
        fig = plt.figure()

    pps = np.max(
        [3, len(self.ProfilesPredicted)]
    )  # Because plotGradients require at least Te, Ti, ne
    grid = plt.GridSpec(2, pps, hspace=0.3, wspace=0.3)
    axsR = []
    for i in range(pps):
        axsR.append(fig.add_subplot(grid[0, i]))
        axsR.append(fig.add_subplot(grid[1, i]))

    produceInfoRanges(
        self.opt_fun.prfs_model.mainFunction,
        self.opt_fun.prfs_model.bounds_orig,
        axsR=axsR,
        color="k",
        lw=0.2,
        alpha=0.05,
        label="original",
    )
    produceInfoRanges(
        self.opt_fun.prfs_model.mainFunction,
        self.opt_fun.prfs_model.bounds,
        axsR=axsR,
        color="c",
        lw=0.2,
        alpha=0.05,
        label="final",
    )

    p = self.mitim_runs[0]["tgyro"].results["tglf_neo"].profiles
    p.plotGradients(
        axsR,
        color="b",
        lastRho=self.TGYROparameters["RhoLocations"][-1],
        ms=0,
        lw=1.0,
        label="#0",
        ls="-o" if self.opt_fun.prfs_model.avoidPoints else "--o",
        plotImpurity=self.runWithImpurity,
        plotRotation=self.runWithRotation,
    )

    for ikey in self.mitim_runs:
        if type(self.mitim_runs[ikey]) != dict:
            break

        p = self.mitim_runs[ikey]["tgyro"].results["tglf_neo"].profiles
        p.plotGradients(
            axsR,
            color="r",
            lastRho=self.TGYROparameters["RhoLocations"][-1],
            ms=0,
            lw=0.3,
            ls="-o" if self.opt_fun.prfs_model.avoidPoints else "-.o",
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
        )

    p.plotGradients(
        axsR,
        color="g",
        lastRho=self.TGYROparameters["RhoLocations"][-1],
        ms=0,
        lw=1.0,
        label=f"#{self.opt_fun.res.best_absolute_index} (best)",
        plotImpurity=self.runWithImpurity,
        plotRotation=self.runWithRotation,
    )

    axsR[0].legend(loc="best")


def PORTALSanalyzer_plotModelComparison(self, axs=None, GB=True, radial_label=True):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=3, figsize=(12, 6))

    self.plotModelComparison_quantity(
        axs[0],
        quantity=f'Qe{"GB" if GB else ""}_sim_turb',
        quantity_stds=f'Qe{"GB" if GB else ""}_sim_turb_stds',
        labely="$Q_e^{GB}$" if GB else "$Q_e$",
        title=f"Electron energy flux {'(GB)' if GB else '($MW/m^2$)'}",
        typeScale="log" if GB else "linear",
        radial_label=radial_label,
    )

    self.plotModelComparison_quantity(
        axs[1],
        quantity=f'Qi{"GB" if GB else ""}Ions_sim_turb_thr',
        quantity_stds=f'Qi{"GB" if GB else ""}Ions_sim_turb_thr_stds',
        labely="$Q_i^{GB}$" if GB else "$Q_i$",
        title=f"Ion energy flux {'(GB)' if GB else '($MW/m^2$)'}",
        typeScale="log" if GB else "linear",
        radial_label=radial_label,
    )

    self.plotModelComparison_quantity(
        axs[2],
        quantity=f'Ge{"GB" if GB else ""}_sim_turb',
        quantity_stds=f'Ge{"GB" if GB else ""}_sim_turb_stds',
        labely="$\\Gamma_e^{GB}$" if GB else "$\\Gamma_e$",
        title=f"Electron particle flux {'(GB)' if GB else '($MW/m^2$)'}",
        typeScale="linear",
        radial_label=radial_label,
    )

    plt.tight_layout()

    return axs


def plotModelComparison_quantity(
    self,
    ax,
    quantity="QeGB_sim_turb",
    quantity_stds="QeGB_sim_turb_stds",
    labely="",
    title="",
    typeScale="linear",
    radial_label=True,
):
    resultsX = "tglf_neo"
    if "cgyro_neo" in self.mitim_runs[0]["tgyro"].results:
        resultsY = "cgyro_neo"
        labely_resultsY = "(CGYRO)"
    else:
        resultsY = "tglf_neo"
        labely_resultsY = "(TGLF)"

    F_tglf = []
    F_cgyro = []
    F_tglf_stds = []
    F_cgyro_stds = []
    for i in range(len(self.mitim_runs) - 2):
        try:
            F_tglf.append(
                self.mitim_runs[i]["tgyro"].results[resultsX].__dict__[quantity][0, 1:]
            )
            F_tglf_stds.append(
                self.mitim_runs[i]["tgyro"]
                .results[resultsX]
                .__dict__[quantity_stds][0, 1:]
            )
            F_cgyro.append(
                self.mitim_runs[i]["tgyro"].results[resultsY].__dict__[quantity][0, 1:]
            )
            F_cgyro_stds.append(
                self.mitim_runs[i]["tgyro"]
                .results[resultsY]
                .__dict__[quantity_stds][0, 1:]
            )
        except TypeError:
            break
    F_tglf = np.array(F_tglf)
    F_cgyro = np.array(F_cgyro)
    F_tglf_stds = np.array(F_tglf_stds)
    F_cgyro_stds = np.array(F_cgyro_stds)

    colors = GRAPHICStools.listColors()

    for ir in range(F_tglf.shape[1]):
        ax.errorbar(
            F_tglf[:, ir],
            F_cgyro[:, ir],
            yerr=F_cgyro_stds[:, ir],
            c=colors[ir],
            markersize=2,
            capsize=2,
            fmt="s",
            elinewidth=1.0,
            capthick=1.0,
            label=f"$r/a={self.roa[ir]:.2f}$" if radial_label else "",
        )

    minFlux = np.min([F_tglf.min(), F_cgyro.min()])
    maxFlux = np.max([F_tglf.max(), F_cgyro.max()])

    if radial_label:
        ax.plot([minFlux, maxFlux], [minFlux, maxFlux], "--", color="k")
    if typeScale == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif typeScale == "symlog":
        ax.set_xscale("symlog")  # ,linthresh=1E-2)
        ax.set_yscale("symlog")  # ,linthresh=1E-2)
    ax.set_xlabel(f"{labely} (TGLF)")
    ax.set_ylabel(f"{labely} {labely_resultsY}")
    ax.set_title(title)
    GRAPHICStools.addDenseAxis(ax)

    ax.legend()


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def varToReal(y, prfs_model):

    of, cal, res = prfs_model.mainFunction.scalarized_objective(
        torch.Tensor(y).to(prfs_model.mainFunction.dfT).unsqueeze(0)
    )

    cont = 0
    Qe, Qi, Ge, GZ, Mt = [], [], [], [], []
    Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = [], [], [], [], []
    for prof in prfs_model.mainFunction.TGYROparameters["ProfilesPredicted"]:
        for rad in prfs_model.mainFunction.TGYROparameters["RhoLocations"]:
            if prof == "te":
                Qe.append(of[0, cont])
                Qe_tar.append(cal[0, cont])
            if prof == "ti":
                Qi.append(of[0, cont])
                Qi_tar.append(cal[0, cont])
            if prof == "ne":
                Ge.append(of[0, cont])
                Ge_tar.append(cal[0, cont])
            if prof == "nZ":
                GZ.append(of[0, cont])
                GZ_tar.append(cal[0, cont])
            if prof == "w0":
                Mt.append(of[0, cont])
                Mt_tar.append(cal[0, cont])

            cont += 1

    Qe, Qi, Ge, GZ, Mt = (
        np.array(Qe),
        np.array(Qi),
        np.array(Ge),
        np.array(GZ),
        np.array(Mt),
    )
    Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = (
        np.array(Qe_tar),
        np.array(Qi_tar),
        np.array(Ge_tar),
        np.array(GZ_tar),
        np.array(Mt_tar),
    )

    return Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar


def plotVars(
    prfs_model,
    y,
    axs,
    axsR,
    contP=0,
    lines=["-s", "--o"],
    yerr=None,
    plotPoints=None,
    plotResidual=True,
    lab="",
    color=None,
    plotErr=[False] * 10,
    colors=GRAPHICStools.listColors(),
):
    [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f] = axs
    [axTe_r, axTi_r, axne_r, axnZ_r, axw0_r] = axsR

    ms, cp, lwc = 4, 2, 0.5

    if plotPoints is None:
        plotPoints = range(y.shape[0])

    cont = -1
    for i in plotPoints:
        cont += 1

        lw = 1.5 if i == 0 else 1.0

        contP += 1

        x_var = (
            prfs_model.mainFunction.surrogate_parameters["powerstate"]
            .plasma["roa"][0, 1:]
            .cpu()
            .numpy()
        )  # prfs_model.mainFunction.TGYROparameters['RhoLocations']

        try:
            Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = varToReal(
                y[i, :].detach().cpu().numpy(), prfs_model
            )
        except:
            continue

        if yerr is not None:
            (
                QeEl,
                QiEl,
                GeEl,
                GZEl,
                MtEl,
                Qe_tarEl,
                Qi_tarEl,
                Ge_tarEl,
                GZ_tarEl,
                Mt_tarEl,
            ) = varToReal(yerr[0][i, :].detach().cpu().numpy(), prfs_model)
            (
                QeEu,
                QiEu,
                GeEu,
                GZEu,
                MtEu,
                Qe_tarEu,
                Qi_tarEu,
                Ge_tarEu,
                GZ_tarEu,
                Mt_tarEu,
            ) = varToReal(yerr[1][i, :].detach().cpu().numpy(), prfs_model)

        ax = axTe_f

        if lines[0] is not None:
            ax.plot(
                x_var,
                Qe,
                lines[0],
                c=colors[contP] if color is None else color,
                label="$Q$" + lab if i == 0 else "",
                lw=lw,
                markersize=ms,
            )
        if lines[1] is not None:
            ax.plot(
                x_var,
                Qe_tar,
                lines[1],
                c=colors[contP] if color is None else color,
                lw=lw,
                markersize=ms,
                label="$Q^T$" + lab if i == 0 else "",
            )
        if yerr is not None:
            ax.errorbar(
                x_var,
                Qe,
                c=colors[contP] if color is None else color,
                yerr=[QeEl, QeEu],
                capsize=cp,
                capthick=lwc,
                fmt="none",
                lw=lw,
                markersize=ms,
                label="$Q$" + lab if i == 0 else "",
            )

        ax = axTi_f
        if lines[0] is not None:
            ax.plot(
                x_var,
                Qi,
                lines[0],
                c=colors[contP] if color is None else color,
                label=f"#{i}",
                lw=lw,
                markersize=ms,
            )
        if lines[1] is not None:
            ax.plot(
                x_var,
                Qi_tar,
                lines[1],
                c=colors[contP] if color is None else color,
                lw=lw,
                markersize=ms,
            )
        if yerr is not None:
            ax.errorbar(
                x_var,
                Qi,
                c=colors[contP] if color is None else color,
                yerr=[QiEl, QiEu],
                capsize=cp,
                capthick=lwc,
                fmt="none",
                lw=lw,
                markersize=ms,
            )

        if axne_f is not None:
            ax = axne_f
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    Ge,
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
            if lines[1] is not None:
                ax.plot(
                    x_var,
                    Ge_tar,
                    lines[1],
                    c=colors[contP] if color is None else color,
                    lw=lw,
                    markersize=ms,
                )
            if yerr is not None:
                ax.errorbar(
                    x_var,
                    Ge,
                    c=colors[contP] if color is None else color,
                    yerr=[GeEl, GeEu],
                    capsize=cp,
                    capthick=lwc,
                    fmt="none",
                    lw=lw,
                    markersize=ms,
                )

        if axnZ_f is not None:
            ax = axnZ_f
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    GZ,
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
            if lines[1] is not None:
                ax.plot(
                    x_var,
                    GZ_tar,
                    lines[1],
                    c=colors[contP] if color is None else color,
                    lw=lw,
                    markersize=ms,
                )
            if yerr is not None:
                ax.errorbar(
                    x_var,
                    GZ,
                    c=colors[contP] if color is None else color,
                    yerr=[GZEl, GZEu],
                    capsize=cp,
                    capthick=lwc,
                    fmt="none",
                    lw=lw,
                    markersize=ms,
                )

        if axw0_f is not None:
            ax = axw0_f
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    Mt,
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
            if lines[1] is not None:
                ax.plot(
                    x_var,
                    Mt_tar,
                    lines[1],
                    c=colors[contP] if color is None else color,
                    lw=lw,
                    markersize=ms,
                )
            if yerr is not None:
                ax.errorbar(
                    x_var,
                    Mt,
                    c=colors[contP] if color is None else color,
                    yerr=[MtEl, MtEu],
                    capsize=cp,
                    capthick=lwc,
                    fmt="none",
                    lw=lw,
                    markersize=ms,
                )

        if plotResidual:
            ax = axTe_r
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    (Qe - Qe_tar),
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label="$Q-Q^T$" + lab if i == 0 else "",
                    lw=lw,
                    markersize=ms,
                )
                if plotErr[cont]:
                    ax.errorbar(
                        x_var,
                        (Qe - Qe_tar),
                        c=colors[contP] if color is None else color,
                        yerr=[QeEl, QeEu],
                        capsize=cp,
                        capthick=lwc,
                        fmt="none",
                        lw=0.5,
                        markersize=0,
                    )

            ax = axTi_r
            if lines[0] is not None:
                ax.plot(
                    x_var,
                    (Qi - Qi_tar),
                    lines[0],
                    c=colors[contP] if color is None else color,
                    label=f"#{i}",
                    lw=lw,
                    markersize=ms,
                )
                if plotErr[cont]:
                    ax.errorbar(
                        x_var,
                        (Qi - Qi_tar),
                        c=colors[contP] if color is None else color,
                        yerr=[QiEl, QiEu],
                        capsize=cp,
                        capthick=lwc,
                        fmt="none",
                        lw=0.5,
                        markersize=0,
                    )

            if axne_r is not None:
                ax = axne_r
                if lines[0] is not None:
                    ax.plot(
                        x_var,
                        (Ge - Ge_tar),
                        lines[0],
                        c=colors[contP] if color is None else color,
                        label=f"#{i}",
                        lw=lw,
                        markersize=ms,
                    )
                    if plotErr[cont]:
                        ax.errorbar(
                            x_var,
                            (Ge - Ge_tar),
                            c=colors[contP] if color is None else color,
                            yerr=[GeEl, GeEu],
                            capsize=cp,
                            capthick=lwc,
                            fmt="none",
                            lw=0.5,
                            markersize=0,
                        )

            if axnZ_r is not None:
                ax = axnZ_r
                if lines[0] is not None:
                    ax.plot(
                        x_var,
                        (GZ - GZ_tar),
                        lines[0],
                        c=colors[contP] if color is None else color,
                        label=f"#{i}",
                        lw=lw,
                        markersize=ms,
                    )
                    if plotErr[cont]:
                        ax.errorbar(
                            x_var,
                            (GZ - GZ_tar),
                            c=colors[contP] if color is None else color,
                            yerr=[GZEl, GZEu],
                            capsize=cp,
                            capthick=lwc,
                            fmt="none",
                            lw=0.5,
                            markersize=0,
                        )
            if axw0_r is not None:
                ax = axw0_r
                if lines[0] is not None:
                    ax.plot(
                        x_var,
                        (Mt - Mt_tar),
                        lines[0],
                        c=colors[contP] if color is None else color,
                        label=f"#{i}",
                        lw=lw,
                        markersize=ms,
                    )
                    if plotErr[cont]:
                        ax.errorbar(
                            x_var,
                            (Mt - Mt_tar),
                            c=colors[contP] if color is None else color,
                            yerr=[MtEl, MtEu],
                            capsize=cp,
                            capthick=lwc,
                            fmt="none",
                            lw=0.5,
                            markersize=0,
                        )

    return contP


def plotFluxComparison(
    p,
    t,
    axTe_f,
    axTi_f,
    axne_f,
    axnZ_f,
    axw0_f,
    forceZeroParticleFlux=False,
    runWithImpurity=3,
    labZ="Z",
    includeFirst=True,
    alpha=1.0,
    stds=2,
    col="b",
    lab="",
    msFlux=1,
    useConvectiveFluxes=False,
    maxStore=False,
    plotFlows=True,
    plotTargets=True,
    decor=True,
    fontsize_leg=12,
    useRoa=False,
    locLeg="upper left",
):
    labelsFluxesF = {
        "te": "$Q_e$ ($MW/m^2$)",
        "ti": "$Q_i$ ($MW/m^2$)",
        "ne": "$\\Gamma_e$ ($10^{20}/s/m^2$)",
        "nZ": f"$\\Gamma_{labZ}$ ($10^{{20}}/s/m^2$)",
        "w0": "$M_T$ ($J/m^2$)",
    }

    r = t.rho if not useRoa else t.roa

    ixF = 0 if includeFirst else 1

    (
        QeBest_min,
        QeBest_max,
        QiBest_min,
        QiBest_max,
        GeBest_min,
        GeBest_max,
        GZBest_min,
        GZBest_max,
        MtBest_min,
        MtBest_max,
    ) = [None] * 10

    axTe_f.plot(
        r[0][ixF:],
        t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:],
        "-s",
        c=col,
        lw=2,
        markersize=msFlux,
        label="Transport",
        alpha=alpha,
    )
    if plotTargets:
        axTe_f.plot(
            r[0][ixF:],
            t.Qe_tar[0][ixF:],
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

    try:
        sigma = t.Qe_sim_turb_stds[0][ixF:] + t.Qe_sim_neo_stds[0][ixF:]
    except:
        print("Could not find errors to plot!", typeMsg="w")
        sigma = t.Qe_sim_turb[0][ixF:] * 0.0

    m, M = (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) - stds * sigma, (
        t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]
    ) + stds * sigma
    axTe_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

    if maxStore:
        QeBest_max = np.max([M.max(), t.Qe_tar[0][ixF:].max()])
        QeBest_min = np.min([m.min(), t.Qe_tar[0][ixF:].min()])

    axTi_f.plot(
        r[0][ixF:],
        t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:],
        "-s",
        markersize=msFlux,
        c=col,
        lw=2,
        label="Transport",
        alpha=alpha,
    )
    if plotTargets:
        axTi_f.plot(
            r[0][ixF:],
            t.Qi_tar[0][ixF:],
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

    try:
        sigma = t.QiIons_sim_turb_thr_stds[0][ixF:] + t.QiIons_sim_neo_thr_stds[0][ixF:]
    except:
        sigma = t.Qe_sim_turb[0][ixF:] * 0.0

    m, M = (
        t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:]
    ) - stds * sigma, (
        t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:]
    ) + stds * sigma
    axTi_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

    if maxStore:
        QiBest_max = np.max([M.max(), t.Qi_tar[0][ixF:].max()])
        QiBest_min = np.min([m.min(), t.Qi_tar[0][ixF:].min()])

    if useConvectiveFluxes:
        Ge, Ge_tar = t.Ce_sim_turb + t.Ce_sim_neo, t.Ce_tar
        try:
            sigma = t.Ce_sim_turb_stds[0][ixF:] + t.Ce_sim_neo_stds[0][ixF:]
        except:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0
    else:
        Ge, Ge_tar = (t.Ge_sim_turb + t.Ge_sim_neo), t.Ge_tar
        try:
            sigma = t.Ge_sim_turb_stds[0][ixF:] + t.Ge_sim_neo_stds[0][ixF:]
        except:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

    if forceZeroParticleFlux:
        Ge_tar = Ge_tar * 0.0

    if axne_f is not None:
        axne_f.plot(
            r[0][ixF:],
            Ge[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )
        if plotTargets:
            axne_f.plot(
                r[0][ixF:],
                Ge_tar[0][ixF:],
                "--",
                c=col,
                lw=2,
                label="Target",
                alpha=alpha,
            )

        m, M = Ge[0][ixF:] - stds * sigma, Ge[0][ixF:] + stds * sigma
        axne_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

    if maxStore:
        GeBest_max = np.max([M.max(), Ge_tar[0][ixF:].max()])
        GeBest_min = np.min([m.min(), Ge_tar[0][ixF:].min()])

    if axnZ_f is not None:
        if useConvectiveFluxes:
            GZ, GZ_tar = (
                t.Ci_sim_turb[runWithImpurity, :, :]
                + t.Ci_sim_neo[runWithImpurity, :, :],
                t.Ge_tar * 0.0,
            )
            try:
                sigma = (
                    t.Ci_sim_turb_stds[runWithImpurity, 0][ixF:]
                    + t.Ci_sim_neo_stds[runWithImpurity, 0][ixF:]
                )
            except:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0
        else:
            GZ, GZ_tar = (
                t.Gi_sim_turb[runWithImpurity, :, :]
                + t.Gi_sim_neo[runWithImpurity, :, :]
            ), t.Ge_tar * 0.0
            try:
                sigma = (
                    t.Gi_sim_turb_stds[runWithImpurity, 0][ixF:]
                    + t.Gi_sim_neo_stds[runWithImpurity, 0][ixF:]
                )
            except:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0
        axnZ_f.plot(
            r[0][ixF:],
            GZ[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )
        if plotTargets:
            axnZ_f.plot(
                r[0][ixF:],
                GZ_tar[0][ixF:],
                "--",
                c=col,
                lw=2,
                label="Target",
                alpha=alpha,
            )

        m, M = (
            GZ[0][ixF:] - stds * sigma,
            GZ[0][ixF:] + stds * sigma,
        )
        axnZ_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

        if maxStore:
            GZBest_max = np.max([M.max(), GZ_tar[0][ixF:].max()])
            GZBest_min = np.min([m.min(), GZ_tar[0][ixF:].min()])

    if axw0_f is not None:
        axw0_f.plot(
            r[0][ixF:],
            t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )
        if plotTargets:
            axw0_f.plot(
                r[0][ixF:],
                t.Mt_tar[0][ixF:],
                "--*",
                c=col,
                lw=2,
                markersize=0,
                label="Target",
                alpha=alpha,
            )

        try:
            sigma = t.Mt_sim_turb_stds[0][ixF:] + t.Mt_sim_neo_stds[0][ixF:]
        except:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

        m, M = (t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:]) - stds * sigma, (
            t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:]
        ) + stds * sigma
        axw0_f.fill_between(r[0][ixF:], m, M, facecolor=col, alpha=alpha / 3)

        if maxStore:
            MtBest_max = np.max([M.max(), t.Mt_tar[0][ixF:].max()])
            MtBest_min = np.min([m.min(), t.Mt_tar[0][ixF:].min()])

    if plotFlows:
        tBest = t.profiles_final
        for ax, var, mult in zip(
            [axTe_f, axTi_f, axne_f, axnZ_f, axw0_f],
            ["qe_MWm2", "qi_MWm2", "ge_10E20m2", None, "mt_Jm2"],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ):
            if ax is not None:
                if var is None:
                    y = tBest.profiles["rho(-)"] * 0.0
                else:
                    y = tBest.derived[var] * mult
                if plotTargets:
                    ax.plot(
                        tBest.profiles["rho(-)"],
                        y,
                        "-.",
                        lw=0.5,
                        c=col,
                        label="Flow",
                        alpha=alpha,
                    )
                else:
                    ax.plot(
                        tBest.profiles["rho(-)"],
                        y,
                        "--",
                        lw=2,
                        c=col,
                        label="Target",
                        alpha=alpha,
                    )

    # -- for legend
    (l1,) = axTe_f.plot(
        r[0][ixF:],
        t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:],
        "-",
        c="k",
        lw=2,
        markersize=0,
        label="Transport",
    )
    (l2,) = axTe_f.plot(
        r[0][ixF:], t.Qe_tar[0][ixF:], "--*", c="k", lw=2, markersize=0, label="Target"
    )
    l3 = axTe_f.fill_between(
        t.roa[0][ixF:],
        (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) - stds,
        (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) + stds,
        facecolor="k",
        alpha=0.3,
    )
    if plotTargets:
        setl = [l1, l3, l2]
        setlab = ["Transport", f"$\\pm{stds}\\sigma$", "Target"]
    else:
        setl = [l1, l3]
        setlab = ["Transport", f"$\\pm{stds}\\sigma$"]
    if plotFlows:
        (l4,) = axTe_f.plot(
            tBest.profiles["rho(-)"],
            tBest.derived["qe_MWm2"],
            "-.",
            c="k",
            lw=1,
            markersize=0,
            label="Transport",
        )
        setl.append(l4)
        setlab.append("Target HR")
    else:
        l4 = l3

    axTe_f.legend(setl, setlab, loc=locLeg, prop={"size": fontsize_leg})
    l1.set_visible(False)
    l2.set_visible(False)
    l3.set_visible(False)
    l4.set_visible(False)
    # ---------------

    if decor:
        ax = axTe_f
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
        ax.set_ylabel(labelsFluxesF["te"])
        ax.set_xlim([0, 1])

        ax = axTi_f
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
        ax.set_ylabel(labelsFluxesF["ti"])
        ax.set_xlim([0, 1])

        if axne_f is not None:
            ax = axne_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["ne"])
            ax.set_xlim([0, 1])

        if axnZ_f is not None:
            ax = axnZ_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["nZ"])
            ax.set_xlim([0, 1])

            GRAPHICStools.addScientificY(ax)

        if axw0_f is not None:
            ax = axw0_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["w0"])
            ax.set_xlim([0, 1])

        if maxStore:
            Qmax = QeBest_max
            Qmax += np.abs(Qmax) * 0.5
            Qmin = QeBest_min
            Qmin -= np.abs(Qmin) * 0.5
            axTe_f.set_ylim([0, Qmax])

            Qmax = QiBest_max
            Qmax += np.abs(Qmax) * 0.5
            Qmin = QiBest_min
            Qmin -= np.abs(Qmin) * 0.5
            axTi_f.set_ylim([0, Qmax])

            if axne_f is not None:
                Qmax = GeBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = GeBest_min
                Qmin -= np.abs(Qmin) * 0.5
                Q = np.max([np.abs(Qmin), np.abs(Qmax)])
                axne_f.set_ylim([-Q, Q])

            if axnZ_f is not None:
                Qmax = GZBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = GZBest_min
                Qmin -= np.abs(Qmin) * 0.5
                Q = np.max([np.abs(Qmin), np.abs(Qmax)])
                axnZ_f.set_ylim([-Q, Q])

            if axw0_f is not None:
                Qmax = MtBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = MtBest_min
                Qmin -= np.abs(Qmin) * 0.5
                Q = np.max([np.abs(Qmin), np.abs(Qmax)])
                axw0_f.set_ylim([-Q, Q])


def produceInfoRanges(
    self_complete, bounds, axsR, label="", color="k", lw=0.2, alpha=0.05
):
    rhos = np.append([0], self_complete.TGYROparameters["RhoLocations"])
    aLTe, aLTi, aLne, aLnZ, aLw0 = (
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
    )
    for i in range(len(rhos) - 1):
        if f"aLte_{i+1}" in bounds:
            aLTe[i + 1, :] = bounds[f"aLte_{i+1}"]
        if f"aLti_{i+1}" in bounds:
            aLTi[i + 1, :] = bounds[f"aLti_{i+1}"]
        if f"aLne_{i+1}" in bounds:
            aLne[i + 1, :] = bounds[f"aLne_{i+1}"]
        if f"aLnZ_{i+1}" in bounds:
            aLnZ[i + 1, :] = bounds[f"aLnZ_{i+1}"]
        if f"aLw0_{i+1}" in bounds:
            aLw0[i + 1, :] = bounds[f"aLw0_{i+1}"]

    X = torch.zeros(
        ((len(rhos) - 1) * len(self_complete.TGYROparameters["ProfilesPredicted"]), 2)
    )
    l = len(rhos) - 1
    X[0:l, :] = torch.from_numpy(aLTe[1:, :])
    X[l : 2 * l, :] = torch.from_numpy(aLTi[1:, :])

    cont = 0
    if "ne" in self_complete.TGYROparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLne[1:, :])
        cont += 1
    if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLnZ[1:, :])
        cont += 1
    if "w0" in self_complete.TGYROparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLw0[1:, :])
        cont += 1

    X = X.transpose(0, 1)

    powerstate = PORTALStools.constructEvaluationProfiles(
        X, self_complete.surrogate_parameters, recalculateTargets=False
    )

    GRAPHICStools.fillGraph(
        axsR[0],
        powerstate.plasma["rho"][0],
        powerstate.plasma["te"][0],
        y_up=powerstate.plasma["te"][1],
        alpha=alpha,
        color=color,
        lw=lw,
        label=label,
    )
    GRAPHICStools.fillGraph(
        axsR[1],
        rhos,
        aLTe[:, 0],
        y_up=aLTe[:, 1],
        alpha=alpha,
        color=color,
        label=label,
        lw=lw,
    )

    GRAPHICStools.fillGraph(
        axsR[2],
        powerstate.plasma["rho"][0],
        powerstate.plasma["ti"][0],
        y_up=powerstate.plasma["ti"][1],
        alpha=alpha,
        color=color,
        label=label,
        lw=lw,
    )
    GRAPHICStools.fillGraph(
        axsR[3],
        rhos,
        aLTi[:, 0],
        y_up=aLTi[:, 1],
        alpha=alpha,
        color=color,
        label=label,
        lw=lw,
    )

    cont = 0
    if "ne" in self_complete.TGYROparameters["ProfilesPredicted"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["ne"][0] * 0.1,
            y_up=powerstate.plasma["ne"][1] * 0.1,
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 2],
            rhos,
            aLne[:, 0],
            y_up=aLne[:, 1],
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        cont += 2

    if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["nZ"][0] * 0.1,
            y_up=powerstate.plasma["nZ"][1] * 0.1,
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 2],
            rhos,
            aLnZ[:, 0],
            y_up=aLnZ[:, 1],
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        cont += 2

    if "w0" in self_complete.TGYROparameters["ProfilesPredicted"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["w0"][0] * 1e-3,
            y_up=powerstate.plasma["w0"][1] * 1e-3,
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 2],
            rhos,
            aLw0[:, 0],
            y_up=aLw0[:, 1],
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        cont += 2
