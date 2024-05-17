import torch
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.portals import PORTALStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

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
    indeces_extra=[],
    stds=2,
    plotFlows=True,
    fontsize_leg=5,
    includeRicci=True,
    useConvectiveFluxes=False,  # By default, plot in real particle units
    file_save=None,
    **kwargs,  # To allow pass fn that may be used in another plotMetrics method
):
    print("- Plotting PORTALS Metrics")

    self.iextra = indeces_extra

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
        p = self.powerstates[0].model_results.profiles_final,
        labIon = f"Ion {self.runWithImpurity+1} ({p.Species[self.runWithImpurity]['N']}{int(p.Species[self.runWithImpurity]['Z'])},{int(p.Species[self.runWithImpurity]['A'])})"
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
    for i, power in enumerate(self.powerstates):
        if power is not None:
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

            t = self.powerstates[self.i0].model_results
            p = t.profiles_final

            ix = np.argmin(
                np.abs(p.profiles["rho(-)"] - t.rho[0][-1])
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
                axne_f.plot(
                    t.rho[0],
                    Ge_tar[0] * (1 - int(self.forceZeroParticleFlux)),
                    "--",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )

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

    indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(self)

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            indeces_plot,
            colors_plot,
            labels_plot,
            markers_plot,
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.powerstates)):
            continue

        power = self.powerstates[indexUse]
        t = power.model_results
        p = t.profiles_final
        

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

        if self.MODELparameters["Physics_options"]["TargetType"] < 3:
            if cont == 0:
                print(
                    "- This run uses partial targets, using POWERSTATE to plot target fluxes, otherwise TGYRO plot will have wrong targets",
                    typeMsg="w",
                )
            powerstate = power
        else:
            powerstate = None

        plotFluxComparison(
            p,
            t,
            axTe_f,
            axTi_f,
            axne_f,
            axnZ_f,
            axw0_f,
            powerstate=powerstate,
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
            addFlowLegend=cont == len(indeces_plot) - 1,
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

    indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(self)

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            indeces_plot,
            colors_plot,
            labels_plot,
            markers_plot,
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.powerstates)):
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
        indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(
            self
        )

        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                indeces_plot,
                colors_plot,
                labels_plot,
                markers_plot,
            )
        ):
            if (indexUse is None) or (indexUse >= len(self.powerstates)):
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

    indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(self)

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            indeces_plot,
            colors_plot,
            labels_plot,
            markers_plot,
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.powerstates)):
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
    if not np.isinf(self.DVdistMetric_y).all():
        ax.set_yscale("log")
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
        indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(
            self
        )

        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                indeces_plot,
                colors_plot,
                labels_plot,
                markers_plot,
            )
        ):
            if (indexUse is None) or (indexUse >= len(self.powerstates)):
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
        # try:
        #     ax.set_yscale("log")
        # except:
        #     pass

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
    indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(self)

    for cont, (indexUse, col, lab, mars) in enumerate(
        zip(
            indeces_plot,
            colors_plot,
            labels_plot,
            markers_plot,
        )
    ):
        if (indexUse is None) or (indexUse >= len(self.powerstates)):
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
        indeces_plot, colors_plot, labels_plot, markers_plot = define_extra_iterators(
            self
        )

        for cont, (indexUse, col, lab, mars) in enumerate(
            zip(
                indeces_plot,
                colors_plot,
                labels_plot,
                markers_plot,
            )
        ):
            if (indexUse is None) or (indexUse >= len(self.powerstates)):
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


def define_extra_iterators(self):

    # Always plot initial and best
    indeces_plot = [self.i0, self.ibest]
    colors_plot = ["r", "g"]
    labels_plot = [f"Initial (#{self.i0})", f"Best (#{self.ibest})"]

    if (len(self.iextra) == 0) and (self.ibest != self.evaluations[-1]):
        self.iextra = [-1]
        if self.ibest != self.evaluations[-2]:
            self.iextra = self.iextra + [self.evaluations[-2]]

    # Add extra points
    colors = GRAPHICStools.listColors()
    colors = [color for color in colors if color not in ["r", "b"]]
    indeces_plot = indeces_plot + self.iextra
    colors_plot = colors_plot + colors[: len(self.iextra)]

    for i in range(len(self.iextra)):

        if self.iextra[i] == -1 or self.iextra[i] == self.evaluations[-1]:
            ll = "Last"
        else:
            ll = "Extra"
        labels_plot = labels_plot + [f"{ll} (#{self.evaluations[self.iextra[i]]})"]

    markers_plot = GRAPHICStools.listmarkers()[: len(indeces_plot)]

    return indeces_plot, colors_plot, labels_plot, markers_plot


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
        p = self.powerstates[0].model_results.profiles_final
        labIon = f"Ion {self.runWithImpurity+1} ({p.Species[self.runWithImpurity]['N']}{int(p.Species[self.runWithImpurity]['Z'])},{int(p.Species[self.runWithImpurity]['A'])})"
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

    p = self.powerstates[0].model_results.profiles_final

    rho = p.profiles["rho(-)"]
    roa = p.derived["roa"]
    rhoVals = self.MODELparameters["RhoLocations"]
    roaVals = np.interp(rhoVals, rho, roa)
    lastX = roaVals[-1]

    # ---- Plot profiles
    cont = -1
    for i in plotPoints:
        cont += 1

        p = p = self.powerstates[i].model_results.profiles_final

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
        rhoVals = self.MODELparameters["RhoLocations"]
        roaVals = np.interp(rhoVals, rho, roa)

        p0 = self.powerstates[plotPoints[0]].model_results.profiles_final
        zVals = []
        z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
        for roai in roaVals:
            zVals.append(np.interp(roai, roa, z))
        axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

        if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
            p0 = self.powerstates[plotPoints[1]].model_results.profiles_final
            zVals = []
            z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4)

        axTe_g_twin.set_ylim(ranges)
        axTe_g_twin.set_ylabel("(%) from last or best", fontsize=8)

        p0 = self.powerstates[plotPoints[0]].model_results.profiles_final
        zVals = []
        z = (
            (p.derived["aLTi"][:, 0] - p0.derived["aLTi"][:, 0])
            / p0.derived["aLTi"][:, 0]
        ) * 100.0
        for roai in roaVals:
            zVals.append(np.interp(roai, roa, z))
        axTi_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

        if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
            p0 = self.powerstates[plotPoints[1]].model_results.profiles_final
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

            p0 = self.powerstates[plotPoints[0]].model_results.profiles_final
            zVals = []
            z = ((p.derived["aLne"] - p0.derived["aLne"]) / p0.derived["aLne"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axne_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.powerstates[plotPoints[1]].model_results.profiles_final
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

            p0 = self.powerstates[plotPoints[0]].model_results.profiles_final
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
                p0 = self.powerstates[plotPoints[1]].model_results.profiles_final
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

            p0 = self.powerstates[plotPoints[0]].model_results.profiles_final
            zVals = []
            z = ((dw0dr - p0.derived["dw0dr"]) / p0.derived["dw0dr"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axw0_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.powerstates[plotPoints[1]].model_results.profiles_final
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


def PORTALSanalyzer_plotSummary(self, fn=None, fn_color=None):
    print("- Plotting PORTALS summary of TGYRO and PROFILES classes")

    indecesPlot = [
        self.ibest,
        self.i0,
        self.iextra,
    ]

    # -------------------------------------------------------
    # Plot TGYROs
    # -------------------------------------------------------

    power = self.powerstates[indecesPlot[1]]
    t = power.model_results

    t.plot(
        fn=fn, prelabel=f"({indecesPlot[1]}) TGYRO - ", fn_color=fn_color
    )
    if indecesPlot[0] < len(self.powerstates):

        power = self.powerstates[indecesPlot[0]]
        t = power.model_results
        t.plot(
            fn=fn, prelabel=f"({indecesPlot[0]}) TGYRO - ", fn_color=fn_color
        )

    # -------------------------------------------------------
    # Plot PROFILES
    # -------------------------------------------------------

    figs = [
        fn.add_figure(label="PROFILES - Profiles", tab_color=fn_color),
        fn.add_figure(label="PROFILES - Powers", tab_color=fn_color),
        fn.add_figure(label="PROFILES - Geometry", tab_color=fn_color),
        fn.add_figure(label="PROFILES - Gradients", tab_color=fn_color),
        fn.add_figure(label="PROFILES - Flows", tab_color=fn_color),
        fn.add_figure(label="PROFILES - Other", tab_color=fn_color),
        fn.add_figure(label="PROFILES - Impurities", tab_color=fn_color),
    ]

    if indecesPlot[0] < len(self.powerstates):
        _ = PROFILEStools.plotAll(
            [
                self.powerstates[indecesPlot[1]].model_results.profiles_final,
                self.powerstates[indecesPlot[0]].model_results.profiles_final,
            ],
            figs=figs,
            extralabs=[f"{indecesPlot[1]}", f"{indecesPlot[0]}"],
        )

    # -------------------------------------------------------
    # Plot Comparison
    # -------------------------------------------------------

    profile_original = self.mitim_runs[0]["powerstate"].model_results.profiles
    profile_best =  self.mitim_runs[self.ibest]["powerstate"].model_results.profiles

    profile_original_unCorrected = self.mitim_runs["profiles_original_un"]
    profile_original_0 = self.mitim_runs["profiles_original"]

    fig4 = fn.add_figure(label="PROFILES Comparison", tab_color=fn_color)
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
            lastRho=self.MODELparameters["RhoLocations"][-1],
            alpha=alpha,
            useRoa=True,
            RhoLocationsPlot=self.MODELparameters["RhoLocations"],
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

    ms = 0

    p = self.mitim_runs[self.i0]["powerstate"].model_results.profiles
    p.plotGradients(
        axsR,
        color="b",
        lastRho=self.MODELparameters["RhoLocations"][-1],
        ms=ms,
        lw=1.0,
        label="Initial (#0)",
        ls="-o" if self.opt_fun.prfs_model.avoidPoints else "--o",
        plotImpurity=self.runWithImpurity,
        plotRotation=self.runWithRotation,
    )

    for ikey in self.mitim_runs:
        if not isinstance(self.mitim_runs[ikey], dict):
            break

        p = self.mitim_runs[ikey]["powerstate"].model_results.profiles
        p.plotGradients(
            axsR,
            color="r",
            lastRho=self.MODELparameters["RhoLocations"][-1],
            ms=ms,
            lw=0.3,
            ls="-o" if self.opt_fun.prfs_model.avoidPoints else "-.o",
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
        )

    p = self.mitim_runs[self.ibest]["powerstate"].model_results.profiles
    p.plotGradients(
        axsR,
        color="g",
        lastRho=self.MODELparameters["RhoLocations"][-1],
        ms=ms,
        lw=1.0,
        label=f"Best (#{self.opt_fun.res.best_absolute_index})",
        plotImpurity=self.runWithImpurity,
        plotRotation=self.runWithRotation,
    )

    axsR[0].legend(loc="best")


def PORTALSanalyzer_plotModelComparison(
    self,
    fig=None,
    axs=None,
    UseTGLFfull_x=None,
    includeErrors=True,
    includeMetric=True,
    includeLegAll=True,
):
    print("- Plotting PORTALS Simulations - Model comparison")

    if (fig is None) and (axs is None):
        plt.ion()
        fig = plt.figure(figsize=(15, 6 if len(self.ProfilesPredicted) < 4 else 10))

    if axs is None:
        if len(self.ProfilesPredicted) < 4:
            axs = fig.subplots(ncols=3)
        else:
            axs = fig.subplots(ncols=3, nrows=2)

        plt.subplots_adjust(wspace=0.25, hspace=0.25)

    axs = axs.flatten()

    metrics = {}

    # te
    quantityX = "QeGB_sim_turb" if UseTGLFfull_x is None else "[TGLF]Qe"
    quantityX_stds = "QeGB_sim_turb_stds" if UseTGLFfull_x is None else None
    quantityY = "QeGB_sim_turb"
    quantityY_stds = "QeGB_sim_turb_stds"
    metrics["Qe"] = plotModelComparison_quantity(
        self,
        axs[0],
        quantityX=quantityX,
        quantityX_stds=quantityX_stds,
        quantityY=quantityY,
        quantityY_stds=quantityY_stds,
        quantity_label="$Q_e^{GB}$",
        title="Electron energy flux (GB)",
        includeErrors=includeErrors,
        includeMetric=includeMetric,
        includeLeg=True,
    )

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    # ti
    quantityX = "QiGBIons_sim_turb_thr" if UseTGLFfull_x is None else "[TGLF]Qi"
    quantityX_stds = "QiGBIons_sim_turb_thr_stds" if UseTGLFfull_x is None else None
    quantityY = "QiGBIons_sim_turb_thr"
    quantityY_stds = "QiGBIons_sim_turb_thr_stds"
    metrics["Qi"] = plotModelComparison_quantity(
        self,
        axs[1],
        quantityX=quantityX,
        quantityX_stds=quantityX_stds,
        quantityY=quantityY,
        quantityY_stds=quantityY_stds,
        quantity_label="$Q_i^{GB}$",
        title="Ion energy flux (GB)",
        includeErrors=includeErrors,
        includeMetric=includeMetric,
        includeLeg=includeLegAll,
    )

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    # ne
    quantityX = "GeGB_sim_turb" if UseTGLFfull_x is None else "[TGLF]Ge"
    quantityX_stds = "GeGB_sim_turb_stds" if UseTGLFfull_x is None else None
    quantityY = "GeGB_sim_turb"
    quantityY_stds = "GeGB_sim_turb_stds"
    metrics["Ge"] = plotModelComparison_quantity(
        self,
        axs[2],
        quantityX=quantityX,
        quantityX_stds=quantityX_stds,
        quantityY=quantityY,
        quantityY_stds=quantityY_stds,
        quantity_label="$\\Gamma_e^{GB}$",
        title="Electron particle flux (GB)",
        includeErrors=includeErrors,
        includeMetric=includeMetric,
        includeLeg=includeLegAll,
    )

    if UseTGLFfull_x is None:
        val_calc = self.mitim_runs[0]["powerstate"].model_results.__dict__[quantityX][0, 1:]
    else:
        val_calc = np.array(
            [
                self.tglf_full.results["ev0"]["TGLFout"][j].__dict__[
                    quantityX.replace("[TGLF]", "")
                ]
                for j in range(len(self.rhos))
            ]
        )

    thre = 10 ** round(np.log10(np.abs(val_calc).min()))
    axs[2].set_xscale("symlog", linthresh=thre)
    axs[2].set_yscale("symlog", linthresh=thre)
    # axs[2].tick_params(axis="both", which="major", labelsize=8)

    cont = 1
    if "nZ" in self.ProfilesPredicted:
        # nZ
        quantityX = "GiGB_sim_turb" if UseTGLFfull_x is None else "[TGLF]GiAll"
        quantityX_stds = "GiGB_sim_turb_stds" if UseTGLFfull_x is None else None
        quantityY = "GiGB_sim_turb"
        quantityY_stds = "GiGB_sim_turb_stds"
        metrics["Gi"] = plotModelComparison_quantity(
            self,
            axs[2 + cont],
            quantityX=quantityX,
            quantityX_stds=quantityX_stds,
            quantityY=quantityY,
            quantityY_stds=quantityY_stds,
            quantity_label="$\\Gamma_Z^{GB}$",
            title="Impurity particle flux (GB)",
            runWithImpurity=self.runWithImpurity,
            includeErrors=includeErrors,
            includeMetric=includeMetric,
            includeLeg=includeLegAll,
        )

        if UseTGLFfull_x is None:
            val_calc = (
                self.mitim_runs[0]["powerstate"].model_results
                .__dict__[quantityX][self.runWithImpurity, 0, 1:]
            )
        else:
            val_calc = np.array(
                [
                    self.tglf_full.results["ev0"]["TGLFout"][j].__dict__[
                        quantityX.replace("[TGLF]", "")
                    ]
                    for j in range(len(self.rhos))
                ]
            )[self.runWithImpurity]

        thre = 10 ** round(np.log10(np.abs(val_calc).min()))
        axs[2 + cont].set_xscale("symlog", linthresh=thre)
        axs[2 + cont].set_yscale("symlog", linthresh=thre)
        axs[2 + cont].tick_params(axis="both", which="major", labelsize=8)

        cont += 1

    if "w0" in self.ProfilesPredicted:
        if UseTGLFfull_x is not None:
            raise Exception("Momentum plot not implemented yet")
        # w0
        quantityX = "MtGB_sim_turb"
        quantityX_stds = "MtGB_sim_turb_stds"
        quantityY = "MtGB_sim_turb"
        quantityY_stds = "MtGB_sim_turb_stds"
        metrics["Mt"] = plotModelComparison_quantity(
            self,
            axs[2 + cont],
            quantityX=quantityX,
            quantityX_stds=quantityX_stds,
            quantityY=quantityY,
            quantityY_stds=quantityY_stds,
            quantity_label="$M_T^{GB}$",
            title="Momentum Flux (GB)",
            includeErrors=includeErrors,
            includeMetric=includeMetric,
            includeLeg=includeLegAll,
        )

        thre = 10 ** round(
            np.log10(
                np.abs(
                    self.mitim_runs[0]["powerstate"].model_results
                    .__dict__[quantityX][0, 1:]
                ).min()
            )
        )
        axs[2 + cont].set_xscale("symlog", linthresh=thre)
        axs[2 + cont].set_yscale("symlog", linthresh=thre)
        axs[2 + cont].tick_params(axis="both", which="major", labelsize=8)

        cont += 1

    if self.PORTALSparameters["surrogateForTurbExch"]:
        if UseTGLFfull_x is not None:
            raise Exception("Turbulent exchange plot not implemented yet")
        # Sexch
        quantityX = "EXeGB_sim_turb"
        quantityX_stds = "EXeGB_sim_turb_stds"
        quantityY = "EXeGB_sim_turb"
        quantityY_stds = "EXeGB_sim_turb_stds"
        metrics["EX"] = plotModelComparison_quantity(
            self,
            axs[2 + cont],
            quantityX=quantityX,
            quantityX_stds=quantityX_stds,
            quantityY=quantityY,
            quantityY_stds=quantityY_stds,
            quantity_label="$S_{exch}^{GB}$",
            title="Turbulent Exchange (GB)",
            includeErrors=includeErrors,
            includeMetric=includeMetric,
            includeLeg=includeLegAll,
        )

        thre = 10 ** round(
            np.log10(
                np.abs(
                    self.mitim_runs[0]["powerstate"].model_results
                    .__dict__[quantityX][0, 1:]
                ).min()
            )
        )
        axs[2 + cont].set_xscale("symlog", linthresh=thre)
        axs[2 + cont].set_yscale("symlog", linthresh=thre)
        axs[2 + cont].tick_params(axis="both", which="major", labelsize=8)

        cont += 1

    return axs, metrics


def plotModelComparison_quantity(
    self,
    ax,
    quantityX="QeGB_sim_turb",
    quantityX_stds="QeGB_sim_turb_stds",
    quantityY="QeGB_sim_turb",
    quantityY_stds="QeGB_sim_turb_stds",
    quantity_label="",
    title="",
    runWithImpurity=None,
    includeErrors=True,
    includeMetric=True,
    includeLeg=True,
):
    resultsX = "tglf_neo"
    quantity_label_resultsX = "(TGLF)"

    if "cgyro_neo" in self.mitim_runs[0]["powerstate"].model_results:
        resultsY = "cgyro_neo"
        quantity_label_resultsY = "(CGYRO)"
    else:
        resultsY = resultsX
        quantity_label_resultsY = quantity_label_resultsX

    X, X_stds = [], []
    Y, Y_stds = [], []
    for i in range(self.ilast + 1):
        """
        Read the fluxes to be plotted in Y from the TGYRO results
        """
        t = self.mitim_runs[i]["powerstate"].model_results
        Y.append(
            t[resultsY].__dict__[quantityY][
                ... if runWithImpurity is None else runWithImpurity, 0, 1:
            ]
        )
        Y_stds.append(
            t[resultsY].__dict__[quantityY_stds][
                ... if runWithImpurity is None else runWithImpurity, 0, 1:
            ]
        )

        """
        Read the fluxes to be plotted in X from...
        """

        # ...from the TGLF full results
        if "[TGLF]" in quantityX:
            X.append(
                [
                    self.tglf_full.results[f"ev{i}"]["TGLFout"][j].__dict__[
                        quantityX.replace("[TGLF]", "")
                    ]
                    for j in range(len(self.rhos))
                ]
            )
            X_stds.append([np.nan for j in range(len(self.rhos))])

        # ...from the TGLF results
        else:
            X.append(
                t[resultsX].__dict__[quantityX][
                    (... if runWithImpurity is None else runWithImpurity), 0, 1:
                ]
            )
            X_stds.append(
                t[resultsX].__dict__[quantityX_stds][
                    ... if runWithImpurity is None else runWithImpurity, 0, 1:
                ]
            )

    X = np.array(X)
    Y = np.array(Y)
    X_stds = np.array(X_stds)
    Y_stds = np.array(Y_stds)

    colors = GRAPHICStools.listColors()

    metrics = {}
    for ir in range(X.shape[1]):
        label = f"$r/a={self.roa[ir]:.2f}$"
        if includeMetric:
            metric, lab_metric = add_metric(None, X[:, ir], Y[:, ir])
            label += f", {lab_metric}: {metric:.2f}"
            metrics[self.roa[ir]] = metric

        ax.errorbar(
            X[:, ir],
            Y[:, ir],
            xerr=X_stds[:, ir] if includeErrors else None,
            yerr=Y_stds[:, ir] if includeErrors else None,
            c=colors[ir],
            markersize=2,
            capsize=2,
            fmt="s",
            elinewidth=1.0,
            capthick=1.0,
            label=label,
        )

    # -------------------------------------------------------
    # Decorations
    # -------------------------------------------------------

    minFlux = np.min([X.min(), Y.min()])
    maxFlux = np.max([X.max(), Y.max()])

    minFlux = minFlux - 0.25 * (maxFlux - minFlux)
    maxFlux = maxFlux + 0.25 * (maxFlux - minFlux)

    ax.plot([minFlux, maxFlux], [minFlux, maxFlux], "-", color="k", lw=0.5)

    ax.set_xlabel(f"{quantity_label} {quantity_label_resultsX}")
    ax.set_ylabel(f"{quantity_label} {quantity_label_resultsY}")
    ax.set_title(title)
    GRAPHICStools.addDenseAxis(ax)

    sizeLeg = 7

    if includeLeg:
        legend = ax.legend(loc="best", prop={"size": sizeLeg})

    if includeMetric:
        metric, lab_metric = add_metric(
            ax if not includeLeg else None, X, Y, fontsize=sizeLeg
        )
        if includeLeg:
            legend.set_title(f"{lab_metric}: {metric:.2f}")
            plt.setp(
                legend.get_title(),
                bbox=dict(
                    facecolor="lightgreen",
                    alpha=0.3,
                    edgecolor="black",
                    boxstyle="round,pad=0.2",
                ),
            )
            legend.get_title().set_fontsize(sizeLeg)

    return metrics


# ---------------------------------------------------------------------------------------------------------------------


def add_metric(ax, X, Y, typeM="RMSE", fontsize=8):
    if typeM == "RMSE":
        metric = np.sqrt(np.mean((X - Y) ** 2))
        metric_lab = "RMSE"
        if ax is not None:
            ax.text(
                0.05,
                0.95,
                f"{metric_lab}: {metric:.2f}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                bbox=dict(
                    facecolor="lightgreen",
                    alpha=0.3,
                    edgecolor="black",
                    boxstyle="round,pad=0.2",
                ),
                fontsize=fontsize,
            )

    return metric, metric_lab


def varToReal(y, prfs_model):
    of, cal, res = prfs_model.mainFunction.scalarized_objective(
        torch.Tensor(y).to(prfs_model.mainFunction.dfT).unsqueeze(0)
    )

    cont = 0
    Qe, Qi, Ge, GZ, Mt = [], [], [], [], []
    Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = [], [], [], [], []
    for prof in prfs_model.mainFunction.MODELparameters["ProfilesPredicted"]:
        for rad in prfs_model.mainFunction.MODELparameters["RhoLocations"]:
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
        )  # prfs_model.mainFunction.MODELparameters['RhoLocations']

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
    powerstate=None,
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
    addFlowLegend=True,
    decor=True,
    fontsize_leg=12,
    useRoa=False,
    locLeg="upper left",
):
    """
    By default this plots the fluxes and targets from tgyro
    If powerstate is provided, it will grab the targets from it
    """

    r = t.rho if not useRoa else t.roa

    ixF = 0 if includeFirst else 1

    # Prep

    labelsFluxesF = {
        "te": "$Q_e$ ($MW/m^2$)",
        "ti": "$Q_i$ ($MW/m^2$)",
        "ne": "$\\Gamma_e$ ($10^{20}/s/m^2$)",
        "nZ": f"$\\Gamma_{labZ}$ ($10^{{20}}/s/m^2$)",
        "w0": "$M_T$ ($J/m^2$)",
    }

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

    # -----------------------------------------------------------------------------------------------
    # Electron energy flux
    # -----------------------------------------------------------------------------------------------

    if axTe_f is not None:
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

        if "Qe_sim_turb_stds" in t.__dict__:
            sigma = t.Qe_sim_turb_stds[0][ixF:] + t.Qe_sim_neo_stds[0][ixF:]
        else:
            print("Could not find errors to plot!", typeMsg="w")
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

        m_Qe, M_Qe = (t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]) - stds * sigma, (
            t.Qe_sim_turb[0][ixF:] + t.Qe_sim_neo[0][ixF:]
        ) + stds * sigma
        axTe_f.fill_between(r[0][ixF:], m_Qe, M_Qe, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Ion energy flux
    # -----------------------------------------------------------------------------------------------

    if axTi_f is not None:
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

        if "QiIons_sim_turb_thr_stds" in t.__dict__:
            sigma = (
                t.QiIons_sim_turb_thr_stds[0][ixF:] + t.QiIons_sim_neo_thr_stds[0][ixF:]
            )
        else:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

        m_Qi, M_Qi = (
            t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:]
        ) - stds * sigma, (
            t.QiIons_sim_turb_thr[0][ixF:] + t.QiIons_sim_neo_thr[0][ixF:]
        ) + stds * sigma
        axTi_f.fill_between(r[0][ixF:], m_Qi, M_Qi, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Electron particle flux
    # -----------------------------------------------------------------------------------------------

    if axne_f is not None:

        if useConvectiveFluxes:
            Ge = t.Ce_sim_turb + t.Ce_sim_neo
            if "Ce_sim_turb_stds" in t.__dict__:
                sigma = t.Ce_sim_turb_stds[0][ixF:] + t.Ce_sim_neo_stds[0][ixF:]
            else:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0
        else:
            Ge = t.Ge_sim_turb + t.Ge_sim_neo
            if "Ge_sim_turb_stds" in t.__dict__:
                sigma = t.Ge_sim_turb_stds[0][ixF:] + t.Ge_sim_neo_stds[0][ixF:]
            else:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0

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

        m_Ge, M_Ge = Ge[0][ixF:] - stds * sigma, Ge[0][ixF:] + stds * sigma
        axne_f.fill_between(r[0][ixF:], m_Ge, M_Ge, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Impurity flux
    # -----------------------------------------------------------------------------------------------

    if axnZ_f is not None:
        if useConvectiveFluxes:
            GZ = (
                t.Ci_sim_turb[runWithImpurity, :, :]
                + t.Ci_sim_neo[runWithImpurity, :, :]
            )

            if "Ci_sim_turb_stds" in t.__dict__:
                sigma = (
                    t.Ci_sim_turb_stds[runWithImpurity, 0][ixF:]
                    + t.Ci_sim_neo_stds[runWithImpurity, 0][ixF:]
                )
            else:
                sigma = t.Qe_sim_turb[0][ixF:] * 0.0
        else:
            GZ = (
                t.Gi_sim_turb[runWithImpurity, :, :]
                + t.Gi_sim_neo[runWithImpurity, :, :]
            )
            if "Gi_sim_turb_stds" in t.__dict__:
                sigma = (
                    t.Gi_sim_turb_stds[runWithImpurity, 0][ixF:]
                    + t.Gi_sim_neo_stds[runWithImpurity, 0][ixF:]
                )
            else:
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

        m_Gi, M_Gi = (
            GZ[0][ixF:] - stds * sigma,
            GZ[0][ixF:] + stds * sigma,
        )
        axnZ_f.fill_between(r[0][ixF:], m_Gi, M_Gi, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Momentum flux
    # -----------------------------------------------------------------------------------------------

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

        if "Mt_sim_turb_stds" in t.__dict__:
            sigma = t.Mt_sim_turb_stds[0][ixF:] + t.Mt_sim_neo_stds[0][ixF:]
        else:
            sigma = t.Qe_sim_turb[0][ixF:] * 0.0

        m_Mt, M_Mt = (t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:]) - stds * sigma, (
            t.Mt_sim_turb[0][ixF:] + t.Mt_sim_neo[0][ixF:]
        ) + stds * sigma
        axw0_f.fill_between(r[0][ixF:], m_Mt, M_Mt, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Plot targets
    # -----------------------------------------------------------------------------------------------

    # Retrieve targets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    rad = r[0][ixF:] if powerstate is None else r[0][1:]
    Qe_tar = (
        t.Qe_tar[0][ixF:]
        if powerstate is None
        else powerstate.plasma["Pe"].numpy()[0][:]
    )
    Qi_tar = (
        t.Qi_tar[0][ixF:]
        if powerstate is None
        else powerstate.plasma["Pi"].numpy()[0][:]
    )

    if forceZeroParticleFlux:
        Ge_tar = Qe_tar * 0.0
    else:
        if powerstate is not None:
            Ge_tar = powerstate.plasma["GauxE"].numpy()[0][
                1:
            ]  # Special because Ge is not stored in powerstate
            if useConvectiveFluxes:
                Ge_tar = PLASMAtools.convective_flux(
                    powerstate.plasma["te"][0][1:], Ge_tar
                ).numpy()
        else:
            if useConvectiveFluxes:
                Ge_tar = t.Ce_tar[0][ixF:]
            else:
                Ge_tar = t.Ge_tar[0][ixF:]

    GZ_tar = rad * 0.0
    Mt_tar = (
        t.Mt_tar[0][ixF:]
        if powerstate is None
        else powerstate.plasma["Mt"].numpy()[0][:]
    )

    # Plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if axTe_f is not None:
        axTe_f.plot(
            rad,
            Qe_tar,
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

        if maxStore:
            QeBest_max = np.max([M_Qe.max(), Qe_tar.max()])
            QeBest_min = np.min([m_Qe.min(), Qe_tar.min()])

    if axTi_f is not None:
        axTi_f.plot(
            rad,
            Qi_tar,
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

        if maxStore:
            QiBest_max = np.max([M_Qi.max(), Qi_tar.max()])
            QiBest_min = np.min([m_Qi.min(), Qi_tar.min()])

    if axne_f is not None:
        axne_f.plot(
            rad,
            Ge_tar,
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

        if maxStore:
            GeBest_max = np.max([M_Ge.max(), Ge_tar.max()])
            GeBest_min = np.min([m_Ge.min(), Ge_tar.min()])

    if axnZ_f is not None:
        axnZ_f.plot(
            rad,
            GZ_tar,
            "--",
            c=col,
            lw=2,
            label="Target",
            alpha=alpha,
        )

        if maxStore:
            GZBest_max = np.max([M_Gi.max(), GZ_tar.max()])
            GZBest_min = np.min([m_Gi.min(), GZ_tar.min()])

    if axw0_f is not None:
        axw0_f.plot(
            rad,
            Mt_tar,
            "--*",
            c=col,
            lw=2,
            markersize=0,
            label="Target",
            alpha=alpha,
        )

        if maxStore:
            MtBest_max = np.max([M_Mt.max(), Mt_tar.max()])
            MtBest_min = np.min([m_Mt.min(), Mt_tar.min()])

    # Plot HR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tBest = t.profiles_final
    if plotFlows:
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
                ax.plot(
                    (tBest.profiles["rho(-)"] if not useRoa else tBest.derived["roa"]),
                    y,
                    "--",
                    lw=0.5,
                    c=col,
                    label="Flow",
                    alpha=alpha,
                )

    # -----------------------------------------------------------------------------------------------
    # Some decor
    # -----------------------------------------------------------------------------------------------

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

    setl = [l1, l3, l2]
    setlab = ["Transport", f"$\\pm{stds}\\sigma$"] #, "Target"]

    if addFlowLegend:
        (l4,) = axTe_f.plot(
            tBest.profiles["rho(-)"] if not useRoa else tBest.derived["roa"],
            tBest.derived["qe_MWm2"],
            "-.",
            c="k",
            lw=1,
            markersize=0,
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
    rhos = np.append([0], self_complete.MODELparameters["RhoLocations"])
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
        ((len(rhos) - 1) * len(self_complete.MODELparameters["ProfilesPredicted"]), 2)
    )
    l = len(rhos) - 1
    X[0:l, :] = torch.from_numpy(aLTe[1:, :])
    X[l : 2 * l, :] = torch.from_numpy(aLTi[1:, :])

    cont = 0
    if "ne" in self_complete.MODELparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLne[1:, :])
        cont += 1
    if "nZ" in self_complete.MODELparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLnZ[1:, :])
        cont += 1
    if "w0" in self_complete.MODELparameters["ProfilesPredicted"]:
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
    if "ne" in self_complete.MODELparameters["ProfilesPredicted"]:
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

    if "nZ" in self_complete.MODELparameters["ProfilesPredicted"]:
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

    if "w0" in self_complete.MODELparameters["ProfilesPredicted"]:
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
