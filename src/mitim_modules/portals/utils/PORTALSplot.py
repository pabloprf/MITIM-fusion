from mitim_tools.plasmastate_tools.utils import state_plotting
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools
from mitim_modules.portals import PORTALStools
from mitim_modules.powertorch import STATEtools
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_modules.powertorch.utils import POWERplot
from mitim_tools.misc_tools.LOGtools import printMsg as print
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
    indeces_extra=None,
    stds=2,
    plotFlows=True,
    fontsize_leg=5,
    includeRicci=True,
    file_save=None,
    **kwargs,  # To allow pass fn that may be used in another plotMetrics method
    ):
    print("- Plotting PORTALS Metrics")

    self.iextra = indeces_extra if indeces_extra is not None else []

    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(15, 8))

    numprofs = len(self.predicted_channels)

    grid = plt.GridSpec(nrows=8, ncols=numprofs + 1, hspace=0.3, wspace=0.35)

    cont = 0

    # Te
    if "te" in self.predicted_channels:
        axTe = fig.add_subplot(grid[:4, cont])
        axTe.set_title("Electron Temperature")
        axTe_g = fig.add_subplot(grid[4:6, cont])
        axTe_f = fig.add_subplot(grid[6:, cont])
        cont += 1
    else:
        axTe = axTe_g = axTe_f = None

    if "ti" in self.predicted_channels:
        axTi = fig.add_subplot(grid[:4, cont])
        axTi.set_title("Ion Temperature")
        axTi_g = fig.add_subplot(grid[4:6, cont])
        axTi_f = fig.add_subplot(grid[6:, cont])
        cont += 1
    else:
        axTi = axTi_g = axTi_f = None

    
    if "ne" in self.predicted_channels:
        axne = fig.add_subplot(grid[:4, cont])
        axne.set_title("Electron Density")
        axne_g = fig.add_subplot(grid[4:6, cont])
        axne_f = fig.add_subplot(grid[6:, cont])
        cont += 1
    else:
        axne = axne_g = axne_f = None

    if self.runWithImpurity:
        p = self.powerstates[0].profiles
        labIon = f"{p.Species[self.runWithImpurity]['N']}{int(p.Species[self.runWithImpurity]['Z'])},{int(p.Species[self.runWithImpurity]['A'])}"
        axnZ = fig.add_subplot(grid[:4, cont])
        axnZ.set_title(f"{labIon} Density")
        axnZ_g = fig.add_subplot(grid[4:6, cont])
        axnZ_f = fig.add_subplot(grid[6:, cont])
        cont += 1
    else:
        axnZ = axnZ_g = axnZ_f = None

    if self.runWithRotation:
        axw0 = fig.add_subplot(grid[:4, cont])
        axw0.set_title("Rotation")
        axw0_g = fig.add_subplot(grid[4:6, cont])
        axw0_f = fig.add_subplot(grid[6:, cont])
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

            p = power.profiles
            rho = power.plasma['rho'][0].cpu().numpy()

            ix = np.argmin(
                np.abs(p.profiles["rho(-)"] - rho[-1].item())
            )
            if axTe is not None:
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
            if axTi is not None:
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

        if plotAllFluxes:
            if axTe_f is not None:
                axTe_f.plot(
                    rho,
                    power.plasma['QeMWm2_tr_turb'][0].cpu().numpy() + power.plasma['QeMWm2_tr_neoc'][0].cpu().numpy(),
                    "-",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )
                axTe_f.plot(rho, power.plasma['QeMWm2'][0].cpu().numpy(), "--", c=col, lw=lwt, alpha=alph)
            if axTi_f is not None:
                axTi_f.plot(
                    rho,
                    power.plasma['QiMWm2_tr_turb'][0].cpu().numpy() + power.plasma['QiMWm2_tr_neoc'][0].cpu().numpy(),
                    "-",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )
                axTi_f.plot(rho, power.plasma['QiMWm2'][0].cpu().numpy(), "--", c=col, lw=lwt, alpha=alph)

            
            if axne_f is not None:
                # By default, use particle fluxes (_raw)

                axne_f.plot(
                    rho, 
                    power.plasma['Ge1E20m2_tr_turb'][0].cpu().numpy()+power.plasma['Ge1E20m2_tr_neoc'][0].cpu().numpy(),
                     "-", c=col, lw=lwt, alpha=alph)
                axne_f.plot(
                    rho,
                    power.plasma['Ge1E20m2'][0].cpu().numpy() * (1 - int(self.force_zero_particle_flux)),
                    "--",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )

            if axnZ_f is not None:

                axnZ_f.plot(rho, power.plasma['GZ1E20m2_tr_turb'][0].cpu().numpy()+power.plasma['GZ1E20m2_tr_neoc'][0].cpu().numpy(), "-", c=col, lw=lwt, alpha=alph)
                axnZ_f.plot(rho, power.plasma['GZ1E20m2'][0].cpu().numpy(), "--", c=col, lw=lwt, alpha=alph)

            if axw0_f is not None:
                axw0_f.plot(
                    rho,
                    power.plasma['MtJm2_tr_turb'][0].cpu().numpy() + power.plasma['MtJm2_tr_neoc'][0].cpu().numpy(),
                    "-",
                    c=col,
                    lw=lwt,
                    alpha=alph,
                )
                axw0_f.plot(rho, power.plasma['MtJm2'][0].cpu().numpy(), "--", c=col, lw=lwt, alpha=alph)

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
        p = power.profiles
        
        ix = np.argmin(np.abs(p.profiles["rho(-)"] - rho[-1]))

        if axTe_g is not None:
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
        if axTi_g is not None:
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
            power,
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
            force_zero_particle_flux=self.force_zero_particle_flux,
            maxStore=indexToMaximize == indexUse,
            decor=self.ibest == indexUse,
            plotFlows=plotFlows and (self.ibest == indexUse),
            addFlowLegend=cont == len(indeces_plot) - 1,
        )
    
    if axTe is not None:
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

    if axTi is not None:
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
    if "te" in self.predicted_channels:
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
    if "ti" in self.predicted_channels:
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
    if "ne" in self.predicted_channels:
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
    if "nZ" in self.predicted_channels:
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
    if "w0" in self.predicted_channels:
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
        if "te" in self.predicted_channels:
            v = self.resTeM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "ti" in self.predicted_channels:
            v = self.resTiM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "ne" in self.predicted_channels:
            v = self.resneM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "nZ" in self.predicted_channels:
            v = self.resnZM
            ax.plot(
                [self.evaluations[indexUse]],
                [v[indexUse]],
                mars,
                color=col,
                markersize=4,
            )
        if "w0" in self.predicted_channels:
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

    separator = self.opt_fun.mitim_model.optimization_options["initialization_options"]["initial_training"] + 0.5 - 1

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
        if self.chiR_Ricci_thr is not None:
            axt.axhline(self.chiR_Ricci_thr, color="rebeccapurple", lw=0.5, ls="-.")

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

    isThereFusion = (np.nanmax(self.FusionGain) > 1E-2) and (np.nanmax(self.FusionGain) != np.inf)

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
        ax.plot([self.evaluations[indexUse]], [v[indexUse]], "o", color=col, markersize=4)

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

    # Save plot
    if file_save is not None:
        plt.savefig(file_save, transparent=True, dpi=300)

def define_extra_iterators(self):

    # Always plot initial and best
    if self.ibest != self.i0:
        indeces_plot = [self.i0, self.ibest]
        colors_plot = ["r", "g"]
        labels_plot = [f"Initial (#{self.i0})", f"Best (#{self.ibest})"]
    else:
        indeces_plot = [self.ibest]
        colors_plot = ["g"]
        labels_plot = [f"Best (#{self.ibest})"]

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

    # Only predict the cases I will plot (to speed up)
    y_train = torch.zeros_like(y_trainreal)
    for i in plotPoints:
        print(f"\t\t* Predicting training point #{i} for plotting 'Expected' tab")
        x_use = x_train[i : i + 1, :]
        y_pred, _, _, _ = model.predict(x_use)
        y_train[i : i + 1, :] = y_pred

    # ---- Next
    y_next = yU_next = yL_next = None
    if plotNext:
        try:
            print(f"\t\t* Predicting next point for plotting 'Expected' tab")
            y_next, yU_next, yL_next, _ = model.predict(self.step.x_next)
        except:
            pass

    # ---- Plot

    numprofs = len(self.predicted_channels)

    if numprofs <= 4:
        wspace = 0.3
    else:
        wspace = 0.5

    grid = plt.GridSpec(nrows=4, ncols=numprofs, hspace=0.2, wspace=wspace)

    cont = 0
    if "te" in self.predicted_channels:
        axTe = fig.add_subplot(grid[0, cont])
        axTe.set_title("Electron Temperature")
        axTe_g = fig.add_subplot(grid[1, cont], sharex=axTe)
        axTe_f = fig.add_subplot(grid[2, cont], sharex=axTe)
        axTe_r = fig.add_subplot(grid[3, cont], sharex=axTe)
        cont += 1
    else:
        axTe = axTe_g = axTe_f = axTe_r = None
    if "ti" in self.predicted_channels:
        axTi = fig.add_subplot(grid[0, cont], sharex=axTe)
        axTi.set_title("Ion Temperature")
        axTi_g = fig.add_subplot(grid[1, cont], sharex=axTe)
        axTi_f = fig.add_subplot(grid[2, cont], sharex=axTe)
        axTi_r = fig.add_subplot(grid[3, cont], sharex=axTe)
        cont += 1
    else:
        axTi = axTi_g = axTi_f = axTi_r = None
    if "ne" in self.predicted_channels:
        axne = fig.add_subplot(grid[0, cont], sharex=axTe)
        axne.set_title("Electron Density")
        axne_g = fig.add_subplot(grid[1, cont], sharex=axTe)
        axne_f = fig.add_subplot(grid[2, cont], sharex=axTe)
        axne_r = fig.add_subplot(grid[3, cont], sharex=axTe)
        cont += 1
    else:
        axne = axne_g = axne_f = axne_r = None
    if self.runWithImpurity:
        p = self.powerstates[0].profiles
        labIon = f"{p.Species[self.runWithImpurity]['N']}{int(p.Species[self.runWithImpurity]['Z'])},{int(p.Species[self.runWithImpurity]['A'])}"
        axnZ = fig.add_subplot(grid[0, cont], sharex=axTe)
        axnZ.set_title(f"{labIon} Density")
        axnZ_g = fig.add_subplot(grid[1, cont], sharex=axTe)
        axnZ_f = fig.add_subplot(grid[2, cont], sharex=axTe)
        axnZ_r = fig.add_subplot(grid[3, cont], sharex=axTe)
        cont += 1
    else:
        axnZ = axnZ_g = axnZ_f = axnZ_r = None

    if self.runWithRotation:
        axw0 = fig.add_subplot(grid[0, cont], sharex=axTe)
        axw0.set_title("Rotation")
        axw0_g = fig.add_subplot(grid[1, cont], sharex=axTe)
        axw0_f = fig.add_subplot(grid[2, cont], sharex=axTe)
        axw0_r = fig.add_subplot(grid[3, cont], sharex=axTe)
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

    p = self.powerstates[0].profiles

    rho = p.profiles["rho(-)"]
    roa = p.derived["roa"]
    rhoVals = self.portals_parameters["solution"]["predicted_rho"]
    roaVals = np.interp(rhoVals, rho, roa)
    lastX = roaVals[-1]

    # ---- Plot profiles
    cont = -1
    for i in plotPoints:
        cont += 1

        p = self.powerstates[i].profiles

        ix = np.argmin(np.abs(p.derived["roa"] - lastX)) + 1

        lw = 1.0 if cont > 0 else 1.5

        if axTe is not None:
            ax = axTe
            ax.plot(
                p.derived["roa"],
                p.profiles["te(keV)"],
                "-",
                c=colors[cont],
                label=labelAssigned[cont],
                lw=lw,
            )
        if axTi is not None:
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

        if axTe_g is not None:
            ax = axTe_g
            ax.plot(
                p.derived["roa"][:ix],
                p.derived["aLTe"][:ix],
                "-o",
                c=colors[cont],
                markersize=0,
                lw=lw,
            )
        if axTi_g is not None:
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

        if axTe is not None:
            ax = axTe
            ax.plot(
                roa,
                p.profiles["te(keV)"],
                "-",
                c="k",
                label=f"#{x_train_num} (next)",
                lw=lw,
            )
        if axTi is not None:
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

        if axTe_g is not None:
            ax = axTe_g
            ax.plot(roa[:ix], p.derived["aLTe"][:ix], "o-", c="k", markersize=0, lw=lw)
        if axTi_g is not None:
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

        ranges = [-30, 30]

        if axTe_g is not None:
            axTe_g_twin = axTe_g.twinx()

            

            rho = self.profiles_next_new.profiles["rho(-)"]
            rhoVals = self.portals_parameters["solution"]["predicted_rho"]
            roaVals = np.interp(rhoVals, rho, roa)

            p0 = self.powerstates[plotPoints[0]].profiles
            zVals = []
            z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.powerstates[plotPoints[1]].profiles
                zVals = []
                z = ((p.derived["aLTe"] - p0.derived["aLTe"]) / p0.derived["aLTe"]) * 100.0
                for roai in roaVals:
                    zVals.append(np.interp(roai, roa, z))
                axTe_g_twin.plot(roaVals, zVals, "--s", c=colors[1], lw=0.5, markersize=4)

            axTe_g_twin.set_ylim(ranges)
            axTe_g_twin.set_ylabel("(%) from last or best", fontsize=8)
            axTe_g_twin.axhline(y=0, ls="-.", lw=0.2, c="k")

        if axTi_g is not None:
            axTi_g_twin = axTi_g.twinx()
            p0 = self.powerstates[plotPoints[0]].profiles
            zVals = []
            z = (
                (p.derived["aLTi"][:, 0] - p0.derived["aLTi"][:, 0])
                / p0.derived["aLTi"][:, 0]
            ) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axTi_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.powerstates[plotPoints[1]].profiles
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

            
            axTi_g_twin.axhline(y=0, ls="-.", lw=0.2, c="k")

        if axne_g is not None:
            axne_g_twin = axne_g.twinx()

            p0 = self.powerstates[plotPoints[0]].profiles
            zVals = []
            z = ((p.derived["aLne"] - p0.derived["aLne"]) / p0.derived["aLne"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axne_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.powerstates[plotPoints[1]].profiles
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

            p0 = self.powerstates[plotPoints[0]].profiles
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
                p0 = self.powerstates[plotPoints[1]].profiles
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

            p0 = self.powerstates[plotPoints[0]].profiles
            zVals = []
            z = ((dw0dr - p0.derived["dw0dr"]) / p0.derived["dw0dr"]) * 100.0
            for roai in roaVals:
                zVals.append(np.interp(roai, roa, z))
            axw0_g_twin.plot(roaVals, zVals, "--s", c=colors[0], lw=0.5, markersize=4)

            if len(labelAssigned) > 1 and "last" in labelAssigned[1]:
                p0 = self.powerstates[plotPoints[1]].profiles
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
        self.opt_fun.mitim_model,
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
        self.opt_fun.mitim_model,
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
            self.opt_fun.mitim_model,
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
    if axTe is not None:
        ax = axTe
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylabel("Te (keV)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax, n=n)
        # ax.	set_xticklabels([])
    if axTi is not None:
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

    if axTe_g is not None:
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

    if axTi_g is not None:
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

    if axTe_f is not None:
        ax = axTe_f
        ax.set_xlim([0, 1])
        ax.set_ylabel(self.labelsFluxes["te"])
        ax.set_ylim(bottom=0)
        # ax.legend(loc='best',prop={'size':6})
        # ax.set_xticklabels([])
        GRAPHICStools.addDenseAxis(ax, n=n)

    if axTi_f is not None:
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

    if axTe_r is not None:
        ax = axTe_r
        ax.set_xlim([0, 1])
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("Residual " + self.labelsFluxes["te"])
        GRAPHICStools.addDenseAxis(ax, n=n)
        ax.axhline(y=0, lw=0.5, ls="--", c="k")

    if axTi_r is not None:
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
            y_trainreal[self.opt_fun.mitim_model.BOmetrics["overall"]["indBest"], :]
            .detach()
            .cpu()
            .cpu().numpy(),
            self.opt_fun.mitim_model,
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

    # `self.iextra` is initialized as int|None by the analyzer but
    # PORTALSanalyzer_plotMetrics overwrites it with a list (see
    # PORTALSplot.py:37 and define_extra_iterators). Whichever runs first
    # wins. plotSummary only uses one extra index in its third slot, so
    # normalize: list -> first element (or None if empty), scalar -> as-is.
    _iextra = self.iextra
    if isinstance(_iextra, (list, tuple)):
        _iextra = _iextra[0] if _iextra else None

    indecesPlot = [
        self.ibest,
        self.i0,
        _iextra,
    ]

    # -------------------------------------------------------
    # Plot TGYROs
    # -------------------------------------------------------

    power = self.powerstates[indecesPlot[1]]

    if power.model_results is not None:

        power.model_results.plot(
            fn=fn, prelabel=f"({indecesPlot[1]}) MODEL - ", fn_color=fn_color
        )
        
        if indecesPlot[0] < len(self.powerstates):

            power = self.powerstates[indecesPlot[0]]
            if power.model_results is not None:
                power.model_results.plot(
                    fn=fn, prelabel=f"({indecesPlot[0]}) MODEL - ", fn_color=fn_color
                )

    # -------------------------------------------------------
    # Plot PROFILES
    # -------------------------------------------------------

    figs = state_plotting.add_figures(fn,fnlab_pre = "PROFILES - ")

    if indecesPlot[0] < len(self.powerstates):
        _ = state_plotting.plotAll(
            [
                self.powerstates[indecesPlot[1]].profiles,
                self.powerstates[indecesPlot[0]].profiles,
            ],
            figs=figs,
            extralabs=[f"{indecesPlot[1]}", f"{indecesPlot[0]}"],
        )

    # -------------------------------------------------------
    # Plot Comparison
    # -------------------------------------------------------

    profile_original = self.mitim_runs[0]["powerstate"].profiles
    profile_best =  self.mitim_runs[self.ibest]["powerstate"].profiles

    profile_original_unCorrected = self.mitim_runs["profiles_original"]
    profile_original_0 = self.mitim_runs["profiles_modified"]

    fig4 = fn.add_figure(label="PROFILES Comparison", tab_color=fn_color)
    grid = plt.GridSpec(
        2,
        np.max([3, len(self.predicted_channels)]),
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
        axs4.append(fig4.add_subplot(grid[0, cont]))
        axs4.append(fig4.add_subplot(grid[1, cont]))
        cont += 1
    if self.runWithRotation:
        axs4.append(fig4.add_subplot(grid[0, cont]))
        axs4.append(fig4.add_subplot(grid[1, cont]))

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
        profiles.plot_gradients(
            axs4,
            color=colors[i],
            label=label,
            lastRho=self.portals_parameters["solution"]["predicted_rho"][-1],
            alpha=alpha,
            useRoa=True,
            predicted_rhoPlot=self.portals_parameters["solution"]["predicted_rho"],
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
            autoscale=i == 3,
        )

    axs4[0].legend(loc="best")

    # -------------------------------------------------------
    # Plot powerstate
    # -------------------------------------------------------

    fig = fn.add_figure(label="Powerstate", tab_color=fn_color)
    axs, axsM = STATEtools.add_axes_powerstate_plot(fig,num_kp=len(self.predicted_channels))

    for indeces,c in zip(indecesPlot,["g","r","m"]):
        if indeces is not None:
            self.powerstates[indeces].plot(axs, label=f"({indeces})", c=c)

    powers = [self.powerstates[indecesPlot[1]], self.powerstates[indecesPlot[0]]]

    POWERplot.plot_metrics_powerstates(axsM,powers)

    axs[0].legend(loc="best")

def PORTALSanalyzer_plotRanges(self, fig=None):
    if fig is None:
        plt.ion()
        fig = plt.figure()

    #pps = np.max([3, len(self.predicted_channels)])  # Because plotGradients require at least Te, Ti, ne
    pps = 6
    grid = plt.GridSpec(2, pps, hspace=0.3, wspace=0.3)
    axsR = []
    for i in range(pps):
        axsR.append(fig.add_subplot(grid[0, i]))
        axsR.append(fig.add_subplot(grid[1, i]))

    produceInfoRanges(
        self.opt_fun.mitim_model.optimization_object,
        self.opt_fun.mitim_model.bounds_orig,
        axsR=axsR,
        color="k",
        lw=0.2,
        alpha=0.05,
        label="original",
    )
    produceInfoRanges(
        self.opt_fun.mitim_model.optimization_object,
        self.opt_fun.mitim_model.bounds,
        axsR=axsR,
        color="c",
        lw=0.2,
        alpha=0.05,
        label="final",
    )

    ms = 0
    
    p = self.mitim_runs[self.i0]["powerstate"].profiles
    p.plot_gradients(
        axsR,
        color="b",
        lastRho=self.portals_parameters["solution"]["predicted_rho"][-1],
        ms=ms,
        lw=1.0,
        label="Initial (#0)",
        ls="-o" if self.opt_fun.mitim_model.avoidPoints is not None else "--o",
        plotImpurity=self.runWithImpurity,
        plotRotation=self.runWithRotation,
    )

    for ikey in self.mitim_runs:
        if not isinstance(self.mitim_runs[ikey], dict):
            break

        p = self.mitim_runs[ikey]["powerstate"].profiles
        p.plot_gradients(
            axsR,
            color="r",
            lastRho=self.portals_parameters["solution"]["predicted_rho"][-1],
            ms=ms,
            lw=0.3,
            ls="-o" if self.opt_fun.mitim_model.avoidPoints is not None else "-.o",
            plotImpurity=self.runWithImpurity,
            plotRotation=self.runWithRotation,
        )

    p = self.mitim_runs[self.ibest]["powerstate"].profiles
    p.plot_gradients(
        axsR,
        color="g",
        lastRho=self.portals_parameters["solution"]["predicted_rho"][-1],
        ms=ms,
        lw=1.0,
        label=f"Best (#{self.opt_fun.res.best_absolute_index})",
        plotImpurity=self.runWithImpurity,
        plotRotation=self.runWithRotation,
    )

    axsR[0].legend(loc="best")

def PORTALSanalyzer_plotDebug(self, fig=None):
    if fig is None:
        plt.ion()
        fig = plt.figure()
        
    axs = fig.subplot_mosaic(
        [
            ["Te_training", "Ti_training", "ne_training",           "Te_opt", "Ti_opt", "ne_opt"],
            ["aLTe_training", "aLTi_training", "aLne_training",     "aLTe_opt", "aLTi_opt", "aLne_opt"],
            ["Qe_training", "Qi_training", "Ge_training",           "Qe_opt", "Qi_opt", "Ge_opt"],
        ]
    )

    # Plot the evolution of profiles and their gradients during the initial training
    num_training = self.opt_fun.mitim_model.optimization_options['initialization_options']['initial_training']
    num_total = len(self.powerstates)
    roa_pred = self.powerstates[0].plasma['roa'][0,1:].cpu().numpy()
    
    lw = 1
    mm = '-s'
    mm2 = '--o'

    def _plot_evaluations(axs, evals, mm='', mm2='--o', lw=1, roa_pred=None, lab = 'Training'):

        colors, _ = GRAPHICStools.colorTableFade(len(evals), startcolor="b", endcolor="r", alphalims=[1.0, 1.0])

        min_grads = [0,0,0]
        max_grads = [0,0,0]
        for j,i in enumerate(evals):
        
            power = self.powerstates[i]
            p = power.profiles
            
            axs['Te'].plot(p.derived['roa'], p.profiles["te(keV)"], label=f"#{i}", c=colors[j], lw=lw)
            axs['Te'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['te'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)
            
            axs['Ti'].plot(p.derived['roa'], p.profiles["ti(keV)"][:,0], c=colors[j], lw=lw)
            axs['Ti'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['ti'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)
            
            axs['ne'].plot(p.derived['roa'], p.profiles["ne(10^19/m^3)"] * 1e-1, c=colors[j], lw=lw)
            axs['ne'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['ne'][0,1:].cpu().numpy() * 1e-1, mm, c=colors[j], markersize=3)
            
            axs['aLTe'].plot(p.derived['roa'], p.derived["aLTe"], c=colors[j], lw=lw)
            axs['aLTe'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['aLte'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)
            
            axs['aLTi'].plot(p.derived['roa'], p.derived["aLTi"][:,0], c=colors[j], lw=lw)
            axs['aLTi'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['aLti'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)
            
            axs['aLne'].plot(p.derived['roa'], p.derived["aLne"], c=colors[j], lw=lw)
            axs['aLne'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['aLne'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)
            
            axs['Qe'].plot(p.derived['roa'], p.derived["qe_MWm2"], c=colors[j], lw=lw/2, label=f"HR target")
            axs['Qe'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['QeMWm2'][0,1:].cpu().numpy(), mm2, c=colors[j], markersize=3, label=f"target")
            axs['Qe'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['QeMWm2_tr'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3, label=f"transport")
            
            axs['Qi'].plot(p.derived['roa'], p.derived["qi_MWm2"], c=colors[j], lw=lw/2)
            axs['Qi'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['QiMWm2'][0,1:].cpu().numpy(), mm2, c=colors[j], markersize=3)
            axs['Qi'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['QiMWm2_tr'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)

            axs['Ge'].plot(p.derived['roa'], p.derived["ge_10E20m2"], c=colors[j], lw=lw/2)
            axs['Ge'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['Ge1E20m2'][0,1:].cpu().numpy(), mm2, c=colors[j], markersize=3)
            axs['Ge'].plot(power.plasma['roa'][0,1:].cpu().numpy(), power.plasma['Ge1E20m2_tr'][0,1:].cpu().numpy(), mm, c=colors[j], markersize=3)
            
            max_grads[0] = max(max_grads[0], power.plasma['aLte'][0,1:].cpu().numpy().max())
            max_grads[1] = max(max_grads[1], power.plasma['aLti'][0,1:].cpu().numpy().max())
            max_grads[2] = max(max_grads[2], power.plasma['aLne'][0,1:].cpu().numpy().max())
            
            min_grads[0] = min(min_grads[0], power.plasma['aLte'][0,1:].cpu().numpy().min())
            min_grads[1] = min(min_grads[1], power.plasma['aLti'][0,1:].cpu().numpy().min())
            min_grads[2] = min(min_grads[2], power.plasma['aLne'][0,1:].cpu().numpy().min())
            
        for ax in axs.values():
            ax.set_xlabel("$r/a$"); ax.set_xlim([0, 1])
            GRAPHICStools.addDenseAxis(ax)
        
        axs['Te'].legend(loc="best")
        axs['Te'].set_ylabel("Te (keV)"); axs['Te'].set_ylim(bottom=0); axs['Te'].set_title(f'{lab}: Te')
        axs['Ti'].set_ylabel("Ti (keV)"); axs['Ti'].set_ylim(bottom=0); axs['Ti'].set_title(f'{lab}: Ti')
        axs['ne'].set_ylabel("ne ($10^{20}m^{-3}$)"); axs['ne'].set_ylim(bottom=0); axs['ne'].set_title(f'{lab}: ne')
        axs['aLTe'].set_ylabel("$a/L_{Te}$"); axs['aLTe'].set_ylim([min_grads[0], max_grads[0]*1.1])
        axs['aLTi'].set_ylabel("$a/L_{Ti}$"); axs['aLTi'].set_ylim([min_grads[1], max_grads[1]*1.1])
        axs['aLne'].set_ylabel("$a/L_{ne}$"); axs['aLne'].set_ylim([min_grads[2], max_grads[2]*1.1])
        axs['Qe'].set_ylabel("Qe (MW/m$^2$)"); #axs['Qe'].set_ylim(bottom=0)
        axs['Qi'].set_ylabel("Qi (MW/m$^2$)"); #axs['Qi'].set_ylim(bottom=0)
        axs['Ge'].set_ylabel("$\\Gamma_e$ ($10^{20}m^{-2}s^{-1}$)"); #axs['Ge'].set_ylim(bottom=0)
        
        axs['Qe'].legend(loc="best")
            
    # Plot training evaluations     
    evals = np.arange(0,num_training,1)
    _plot_evaluations(
        axs = {
            'Te': axs['Te_training'],
            'Ti': axs['Ti_training'],
            'ne': axs['ne_training'],
            'aLTe': axs['aLTe_training'],
            'aLTi': axs['aLTi_training'],
            'aLne': axs['aLne_training'],
            'Qe': axs['Qe_training'],
            'Qi': axs['Qi_training'],
            'Ge': axs['Ge_training'],
        },
        evals = evals,
        mm = mm,
        mm2 = mm2,
        lw = lw,
        roa_pred = roa_pred,
        lab = 'Training',
    )
    
    # Plot a maximum of 5 evaluations during optimization (from the last one, equidistant going back until the last training one
    evals = np.unique(
        np.concatenate(
            [
                np.arange(num_training, num_total, max(1, (num_total - num_training) // 5)),
                [num_total - 1],
            ]
        )
    )
    _plot_evaluations(
        axs = {
            'Te': axs['Te_opt'],
            'Ti': axs['Ti_opt'],
            'ne': axs['ne_opt'],
            'aLTe': axs['aLTe_opt'],
            'aLTi': axs['aLTi_opt'],
            'aLne': axs['aLne_opt'],
            'Qe': axs['Qe_opt'],
            'Qi': axs['Qi_opt'],
            'Ge': axs['Ge_opt'],
        },
        evals = evals,
        mm = mm,
        mm2 = mm2,
        lw = lw,
        roa_pred = roa_pred,
        lab = 'Optimization',
    )
    
def _linear_regression(x, y):
    '''Least-squares slope/intercept, or (None, None) if the fit is not defined'''
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) == 0.0:
        return None, None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return slope, intercept


def PORTALSanalyzer_plotFluxesVsGradients(self, fig=None, flux_type="turb", normalized=True, plot_errors=True):
    '''
    Scatter of every evaluated flux against every evolved gradient, one panel per
    (flux, gradient) pair and one color per radius. The diagonal panels are the
    critical-gradient views (flux vs its own driving gradient); the off-diagonal
    ones show cross-channel drives. Scatter (not lines) because each point is an
    independent transport-code call at a different plasma state, so the vertical
    spread at fixed gradient is the effect of everything else that moved (Ti/Te,
    nu_ei, beta_e, ...).

    flux_type:  'turb' (default), 'neoc' or 'total'
    normalized: gyro-Bohm normalized fluxes (default) -- the right y for a
                critical-gradient view, since it removes the trivial radial
                scaling of Qgb/Ggb
    plot_errors:1-sigma error bars from the transport model (the '_stds' fields:
                TGLF's assigned relative error, CGYRO's time-trace scatter)
    '''

    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(15, 9))

    channel_info = {
        "te": {"grad": "aLte", "grad_label": "$a/L_{Te}$",
               "flux": "QeMWm2", "gb": "Qgb",
               "label": "$Q_e$ ($MW/m^2$)",       "label_gb": "$Q_e/Q_{GB}$"},
        "ti": {"grad": "aLti", "grad_label": "$a/L_{Ti}$",
               "flux": "QiMWm2", "gb": "Qgb",
               "label": "$Q_i$ ($MW/m^2$)",       "label_gb": "$Q_i/Q_{GB}$"},
        "ne": {"grad": "aLne", "grad_label": "$a/L_{ne}$",
               "flux": "Ge1E20m2", "gb": "Ggb",
               "label": "$\\Gamma_e$ ($10^{20}m^{-2}s^{-1}$)", "label_gb": "$\\Gamma_e/\\Gamma_{GB}$"},
        "nZ": {"grad": "aLnZ", "grad_label": "$a/L_{nZ}$",
               "flux": "GZ1E20m2", "gb": "Ggb",
               "label": "$\\Gamma_Z$ ($10^{20}m^{-2}s^{-1}$)", "label_gb": "$\\Gamma_Z/\\Gamma_{GB}$"},
        # aLw0_n is the c_s-normalized rotation-gradient the surrogates actually see, not a/L_w0
        "w0": {"grad": "aLw0_n", "grad_label": "$-(a/c_s)\\cdot d\\omega_0/dr$",
               "flux": "MtJm2", "gb": "Pgb",
               "label": "$M_T$ ($J/m^2$)",        "label_gb": "$M_T/\\Pi_{GB}$"},
    }

    channels = [c for c in self.predicted_channels if c in channel_info]

    suffix = {"turb": "_tr_turb", "neoc": "_tr_neoc", "total": "_tr"}[flux_type]
    name_flux = {"turb": "turbulent", "neoc": "neoclassical", "total": "turb+neoc"}[flux_type]

    # ------------------------------------------------------------------------
    # Gather all evaluations: (n_evaluations, n_radii) arrays per quantity
    # ------------------------------------------------------------------------

    def _grab(power, key):
        return power.plasma[key][0, 1:].cpu().numpy()

    gradients, fluxes, errors = {}, {}, {}
    for c in channels:
        info = channel_info[c]
        gradients[c] = np.array([_grab(p, info["grad"]) for p in self.powerstates])
        f = np.array([_grab(p, info["flux"] + suffix) for p in self.powerstates])

        # There is no '_tr_stds' field: for the summed flux, add turb and neoc in quadrature
        if flux_type == "total":
            e = np.sqrt(sum(np.array([_grab(p, f"{info['flux']}_tr_{s}_stds") for p in self.powerstates])**2
                            for s in ["turb", "neoc"]))
        else:
            e = np.array([_grab(p, info["flux"] + suffix + "_stds") for p in self.powerstates])

        if normalized:
            gb = np.array([_grab(p, info["gb"]) for p in self.powerstates])
            f, e = f / gb, e / gb

        fluxes[c], errors[c] = f, e

    # ------------------------------------------------------------------------
    # Grid: rows = fluxes, columns = gradients
    # ------------------------------------------------------------------------

    n = len(channels)
    grid = plt.GridSpec(nrows=n, ncols=n, hspace=0.1, wspace=0.1)
    colors = GRAPHICStools.listColors()

    for i, c_flux in enumerate(channels):

        ax_row = None
        for j, c_grad in enumerate(channels):

            ax = fig.add_subplot(grid[i, j], sharey=ax_row)
            if ax_row is None:
                ax_row = ax

            for ir in range(len(self.rhos)):

                x, y = gradients[c_grad][:, ir], fluxes[c_flux][:, ir]

                if plot_errors:
                    ax.errorbar(
                        x, y, yerr=errors[c_flux][:, ir],
                        fmt="none", ecolor=colors[ir], elinewidth=0.8, capsize=2, alpha=0.5, zorder=2,
                    )

                ax.scatter(
                    x, y,
                    s=45,
                    c=colors[ir],
                    alpha=0.6,
                    edgecolors="none",
                    label=f"$r/a$ = {self.roa[ir]:.2f}" if (i == 0 and j == 0) else None,
                )

                slope, intercept = _linear_regression(x, y)
                if slope is not None:
                    xfit = np.array([x.min(), x.max()])
                    ax.plot(xfit, slope * xfit + intercept, "--", c=colors[ir], lw=1.2, alpha=0.9, zorder=3)

            GRAPHICStools.addDenseAxis(ax, n=5)

            # Only the frame panels carry labels, otherwise the matrix is unreadable
            if i == n - 1:
                ax.set_xlabel(channel_info[c_grad]["grad_label"], fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(channel_info[c_flux]["label_gb" if normalized else "label"], fontsize=12)
            else:
                ax.tick_params(labelleft=False)

            if i == j:
                ax.set_facecolor("#f2f2f2")

    fig.suptitle(f"PORTALS transport database: {name_flux} fluxes vs gradients "
                 f"({len(self.powerstates)} evaluations, {len(self.rhos)} radii, "
                 f"{'gyro-Bohm normalized' if normalized else 'real units'}). "
                 f"Shaded diagonal = critical-gradient view")

    fig.axes[0].legend(loc="best", prop={"size": 8})


def PORTALSanalyzer_plotTransportModels(self, fn = None, fn_color=None):
    
    print("- Plotting PORTALS Simulations - Transport models")
    
    colors = GRAPHICStools.listColors()
    
    # Lazy import — avoid dragging CGYROtools into every caller of this module
    # when the transport-models tab isn't being rendered.
    from mitim_tools.gacode_tools import CGYROtools

    k = 0
    for it in self.transport_model_objects:
        turb = self.transport_model_objects[it].get('turbulence')
        neo  = self.transport_model_objects[it].get('neoclassical')
        # Skip iterations with missing halves (e.g. SR CGYRO-only populates
        # this dict with turb=None, or a partial read where one leg failed).
        if turb is None or neo is None:
            continue
        # Skip iterations where turbulence is CGYRO — the transport-models
        # tab renders TGLF-style plots (fn_color / extratitle kwargs) that
        # CGYROtools.CGYRO.plot does not accept, and CGYRO has its own
        # dedicated per-rho / per-channel time-trace tabs below.
        if isinstance(turb, CGYROtools.CGYRO):
            continue
        turb.plot(fn=fn, fn_color=fn_color+k, labels = ['base'], extratitle=f"Turb (#{it}) - ")

        if "distributions" in turb.__dict__:
            distributions = turb.distributions
            k += 1
            fig = fn.add_figure(label=f"Turb (#{it}) - Distributions", tab_color=fn_color+k)
            axs = fig.subplots(ncols=3)
            
            varss = [('Qe', '$Q_e$ (MW/m$^2$)'), ('Qi', '$Q_i$ (MW/m$^2$)'), ('Ge', '$\\Gamma_e$ ($10^{20}m^{-2}s^{-1}$)')]
            for i, (var, label) in enumerate(varss):
                ax = axs[i]
                y = np.array(distributions['y'][var])
                # Plot each distribution case as a light profile
                for jj in range(y.shape[0]):
                    ax.plot(
                        turb.rhos,
                        y[jj, :],
                        marker='o',
                        ms=3,
                        lw=0.8,
                        alpha=0.4,
                        color=colors[jj % len(colors)],
                        label=distributions['x'][jj],
                        zorder=2,
                    )

                # Overlay mean ± 2std with a clear point+errorbar style
                y_mean = y.mean(axis=0)
                y_std = y.std(axis=0)
                ax.errorbar(
                    turb.rhos,
                    y_mean,
                    yerr=2*y_std,
                    fmt='o-',
                    color='k',
                    ms=6,
                    lw=2.0,
                    elinewidth=1.5,
                    capsize=5,
                    capthick=1.5,
                    markerfacecolor='white',
                    markeredgewidth=1.1,
                    label='mean ± 2std',
                    zorder=5,
                )
                        
                ax.set_xlabel("$\\rho_N$")
                ax.set_ylabel(label)
                ax.legend(loc="best",prop={'size': 6})
                GRAPHICStools.addDenseAxis(ax)
                if var in ["Qe", "Qi"]:
                    ax.set_ylim(bottom=0)
        
        neo.plot(fn=fn, fn_color=fn_color+k+1, labels = ['base'], extratitle=f"Neoc (#{it}) - ")
        k += 2

    # CGYRO-specific per-rho time traces: one tab per radius with
    # Qe/Qi/Ge(t) overlaid across every PORTALS iteration, ev0 drawn on
    # top as the baseline. Loaded lazily here (not in
    # read_transport_models) so the TGLF/NEO path doesn't pay the cost,
    # and so missing/failed iterations are skipped cleanly rather than
    # aborting the plot. Actual loading and drawing live in the CGYRO
    # tools layer so PORTALS stays model-agnostic — this block is just
    # the PORTALS-side discovery + namelist lookup glue.
    try:
        from mitim_modules.portals.utils.PORTALSanalysis import _model_highest_fidelity
        turbulence_model = _model_highest_fidelity(
            self.powerstate.transport_options['evaluator_instance_attributes']['turbulence_model']
        )
    except Exception:
        turbulence_model = None
    # Dispatch on the backend code (namelist entry may be a named instance like 'cgyro1'
    # whose options block sets `code: cgyro`). Fall back to the raw string if the
    # options block is missing (e.g. during a pre-evaluate dry run).
    code_for_dispatch = None
    if turbulence_model is not None:
        try:
            opts = self.powerstate.transport_options.get('options', {}).get(turbulence_model, {}) or {}
            code_for_dispatch = str(opts.get('code', turbulence_model)).lower()
        except Exception:
            code_for_dispatch = str(turbulence_model).lower()
    if code_for_dispatch == "cgyro":
        _plot_cgyro_time_traces_dispatch(self, fn, fn_color_start=fn_color + k + 1)


def _iterate_portals_evaluation_folders(root_folder):
    '''
    Yield (iteration_index, transport_simulation_folder_path) pairs for every
    PORTALS evaluation visible on disk, preferring the BO layout
    (Execution/Evaluation.{N}) and falling back to the simple-relax layout
    (Initialization/initialization_simple_relax/portals_sr_ev_{N}) when BO
    hasn't started yet.

    Numeric-suffix sorting uses PORTALSanalysis._extract_trailing_int so
    partial runs (0, 1, 3 with 2 missing) don't truncate at the gap.
    '''
    from pathlib import Path
    from mitim_modules.portals.utils.PORTALSanalysis import _extract_trailing_int

    root = Path(root_folder)

    bo_root = root / "Execution"
    if bo_root.is_dir():
        bo_evs = sorted(
            (d for d in bo_root.glob("Evaluation.*") if d.is_dir() and _extract_trailing_int(d.name) is not None),
            key=lambda d: _extract_trailing_int(d.name),
        )
        if bo_evs:
            for d in bo_evs:
                folder = d / "transport_simulation_folder"
                if folder.is_dir():
                    yield _extract_trailing_int(d.name), folder
            return

    sr_root = root / "Initialization" / "initialization_simple_relax"
    if sr_root.is_dir():
        sr_evs = sorted(
            (d for d in sr_root.glob("portals_sr_ev_*") if d.is_dir() and _extract_trailing_int(d.name) is not None),
            key=lambda d: _extract_trailing_int(d.name),
        )
        for d in sr_evs:
            folder = d / "transport_simulation_folder"
            if folder.is_dir():
                yield _extract_trailing_int(d.name), folder


def _targets_GB_from_powerstate_pkl(pkl_path):
    '''Legacy fallback for runs whose fluxes_neoc.json predates the
    targets_GB block in additional_info: read the targets (GB, as populated
    by calculateTargets) from the per-evaluation powerstate pickle that sits
    next to the transport folder.'''
    if not pkl_path.is_file():
        return None
    try:
        from mitim_modules.powertorch import STATEtools
        p = STATEtools.read_saved_state(pkl_path)
        return {var: p.plasma[var][0, 1:].cpu().numpy().tolist()
                for var in ("QeGB", "QiGB", "GeGB", "GZGB", "MtGB") if var in p.plasma}
    except Exception as e:
        print(f"\t- Could not extract targets from {pkl_path} ({e})", typeMsg='w')
        return None


def _load_turb_targets_for_iterations(iter_folders, predicted_channels):
    '''
    Build {iteration: {"QeGB": np.ndarray, ...}} of turbulence-only targets
    (target_GB - neoc_GB, per predicted rho) for the channels PORTALS is
    flux-matching — consumed by CGYROplot as `targets_per_iter` to mark a
    star at the end of each turbulent time trace. Per iteration, the
    neoclassical fluxes come from fluxes_neoc.json (written at run time);
    the targets come from its additional_info['targets_GB'] block (newer
    runs) or, as legacy fallback, the evaluation's powerstate.pkl.
    Iterations missing either piece contribute no entry — the plotter just
    draws no star for them.
    '''
    import json
    channel_to_gb = {"te": "QeGB", "ti": "QiGB", "ne": "GeGB"}
    gb_keys = [channel_to_gb[ch] for ch in (predicted_channels or []) if ch in channel_to_gb]
    out = {}
    for it, folder in iter_folders:
        neoc_path = folder / "fluxes_neoc.json"
        if not neoc_path.is_file():
            continue
        try:
            with open(neoc_path, "r") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            print(f"\t- fluxes_neoc.json unreadable at {neoc_path} ({e}); no target marker for iter {it}", typeMsg='w')
            continue
        neoc = payload.get("fluxes_mean", {})
        targets = (payload.get("additional_info", {}) or {}).get("targets_GB")
        if not targets:
            targets = _targets_GB_from_powerstate_pkl(folder.parent / "powerstate.pkl")
        if not targets:
            continue
        d = {}
        for key in gb_keys:
            if key in targets and key in neoc:
                d[key] = np.asarray(targets[key], dtype=float) - np.asarray(neoc[key], dtype=float)
        if d:
            out[it] = d
    return out


def _plot_cgyro_time_traces_dispatch(self, fn, fn_color_start):
    '''
    PORTALS-side shim for the CGYRO per-rho time-trace plot. Resolves the
    root folder, discovers iteration folders (BO or SR layout), reads each
    iteration's restart_sources.json (the persisted per-(rho, iter) parent
    map), then delegates both the iteration loading and the drawing to
    CGYROplot so PORTALS stays transport-model-agnostic. Restart-mode
    handling is fully driven by what's on disk: when no restart_sources.json
    is present (older runs, or restart_from_cases=null), the plotter
    renders without time-axis alignment. Both the tool cache and the
    sources cache are memoised on `self` so interactive re-invocations
    don't re-read pickles or JSONs.
    '''
    from mitim_tools.gacode_tools.utils import CGYROplot

    print("\t- Adding per-rho CGYRO time-trace tabs (Qe, Qi, Ge)")

    # Resolve the PORTALS root folder in an attribute-agnostic way:
    # analyzer uses self.opt_fun.folder, initializer just has self.folder.
    opt_fun = getattr(self, "opt_fun", None)
    root_folder = opt_fun.folder if (opt_fun is not None and getattr(opt_fun, "folder", None) is not None) else getattr(self, "folder", None)
    if root_folder is None:
        print("\t- Cannot resolve PORTALS root folder for CGYRO trace plot; skipping", typeMsg='w')
        return

    # Resolve the active CGYRO instance name — defaults to 'cgyro' for single-fidelity,
    # could be 'cgyro1'/'cgyro2'/... under named multi-fidelity. This drives:
    #  (a) which options sub-block we read tmin from, and
    #  (b) the on-disk base_<name> folder the loaders look inside.
    try:
        from mitim_modules.portals.utils.PORTALSanalysis import _model_highest_fidelity
        turb_spec = self.powerstate.transport_options['evaluator_instance_attributes']['turbulence_model']
        cgyro_key = _model_highest_fidelity(turb_spec) or "cgyro"
    except Exception:
        cgyro_key = "cgyro"

    # Read-time config from the namelist — so the raw-fallback re-read
    # inside CGYROplot.load_tool_for_iteration uses exactly the window
    # PORTALS used at simulation time (pickles already carry this baked
    # in).
    try:
        _cgyro_read_cfg = self.powerstate.transport_options['options'][cgyro_key]['read']
        _read_kwargs = {k: v for k, v in _cgyro_read_cfg.items()
                        if k in ("tmin", "tmin_is_rel", "last_tmin_for_linear")}
    except Exception:
        _read_kwargs = {}

    base_subfolder = f"base_{cgyro_key}"

    # Materialize the iter-folder list once so both loaders consume it.
    iter_folders = list(_iterate_portals_evaluation_folders(root_folder))

    # Lazy caches on self so re-invocations don't re-read pickles / JSONs.
    if getattr(self, "_cgyro_traces_cache", None) is None:
        self._cgyro_traces_cache = CGYROplot.load_tools_for_iterations(
            iter_folders,
            self.rhos,
            read_kwargs=_read_kwargs,
            base_subfolder=base_subfolder,
        )
    if getattr(self, "_cgyro_sources_cache", None) is None:
        self._cgyro_sources_cache = CGYROplot.load_restart_sources_for_iterations(
            iter_folders,
            base_subfolder=base_subfolder,
        )
    if getattr(self, "_cgyro_targets_cache", None) is None:
        self._cgyro_targets_cache = _load_turb_targets_for_iterations(
            iter_folders,
            getattr(self.powerstate, "predicted_channels", []) or [],
        )

    CGYROplot.plot_time_traces_per_radius(
        fn,
        fn_color_start,
        self.rhos,
        self._cgyro_traces_cache,
        sources_per_iter=self._cgyro_sources_cache,
        base_iter=0,
        targets_per_iter=self._cgyro_targets_cache,
    )
    # Same data, pivoted: one figure per channel with rhos as rows.
    # Per-radius and per-channel tab groups now each use a single color
    # internally, so the channel group only needs +1 offset to land on a
    # different color from the radius group.
    CGYROplot.plot_time_traces_per_channel(
        fn,
        fn_color_start + 1,
        self.rhos,
        self._cgyro_traces_cache,
        sources_per_iter=self._cgyro_sources_cache,
        base_iter=0,
        targets_per_iter=self._cgyro_targets_cache,
    )


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
        fig = plt.figure(figsize=(15, 6 if len(self.predicted_channels)+int(self.portals_parameters["solution"]["turbulent_exchange_as_surrogate"]) < 4 else 10))

    if axs is None:
        if len(self.predicted_channels)+int(self.portals_parameters["solution"]["turbulent_exchange_as_surrogate"]) < 4:
            axs = fig.subplots(ncols=3)
        else:
            axs = fig.subplots(ncols=3, nrows=2)

        plt.subplots_adjust(wspace=0.25, hspace=0.25)

    axs = axs.flatten()
    cont = 0

    metrics = {}

    # te
    if 'te' in self.predicted_channels:
        quantityX = "QeGB_sim_turb" if UseTGLFfull_x is None else "[TGLF]Qe"
        quantityX_stds = "QeGB_sim_turb_stds" if UseTGLFfull_x is None else None
        quantityY = "QeGB_sim_turb"
        quantityY_stds = "QeGB_sim_turb_stds"
        metrics["Qe"] = plotModelComparison_quantity(
            self,
            axs[cont],
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

        axs[cont].set_xscale("log")
        axs[cont].set_yscale("log")

        cont += 1

    # ti
    if 'ti' in self.predicted_channels:
        quantityX = "QiGBIons_sim_turb_thr" if UseTGLFfull_x is None else "[TGLF]Qi"
        quantityX_stds = "QiGBIons_sim_turb_thr_stds" if UseTGLFfull_x is None else None
        quantityY = "QiGBIons_sim_turb_thr"
        quantityY_stds = "QiGBIons_sim_turb_thr_stds"
        metrics["Qi"] = plotModelComparison_quantity(
            self,
            axs[cont],
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

        axs[cont].set_xscale("log")
        axs[cont].set_yscale("log")

        cont += 1

    # ne
    if 'ne' in self.predicted_channels:
        quantityX = "GeGB_sim_turb" if UseTGLFfull_x is None else "[TGLF]Ge"
        quantityX_stds = "GeGB_sim_turb_stds" if UseTGLFfull_x is None else None
        quantityY = "GeGB_sim_turb"
        quantityY_stds = "GeGB_sim_turb_stds"
        metrics["Ge"] = plotModelComparison_quantity(
            self,
            axs[cont],
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
                    self.tglf_full.results["ev0"]["output"][j].__dict__[
                        quantityX.replace("[TGLF]", "")
                    ]
                    for j in range(len(self.rhos))
                ]
            )

        try:
            thre = 10 ** round(np.log10(np.abs(val_calc).min()))
            axs[cont].set_xscale("symlog", linthresh=thre)
            axs[cont].set_yscale("symlog", linthresh=thre)
            # axs[2].tick_params(axis="both", which="major", labelsize=8)
        except OverflowError:
            pass

        cont += 1


    if "nZ" in self.predicted_channels:

        impurity_search = self.runWithImpurity_transport

        # nZ
        quantityX = "GiGB_sim_turb" if UseTGLFfull_x is None else "[TGLF]GiAll"
        quantityX_stds = "GiGB_sim_turb_stds" if UseTGLFfull_x is None else None
        quantityY = "GiGB_sim_turb"
        quantityY_stds = "GiGB_sim_turb_stds"
        metrics["Gi"] = plotModelComparison_quantity(
            self,
            axs[cont],
            quantityX=quantityX,
            quantityX_stds=quantityX_stds,
            quantityY=quantityY,
            quantityY_stds=quantityY_stds,
            quantity_label="$\\Gamma_Z^{GB}$",
            title="Impurity particle flux (GB)",
            runWithImpurity=impurity_search,
            includeErrors=includeErrors,
            includeMetric=includeMetric,
            includeLeg=includeLegAll,
        )

        if UseTGLFfull_x is None:
            val_calc = (
                self.mitim_runs[0]["powerstate"].model_results
                .__dict__[quantityX][impurity_search, 0, 1:]
            )
        else:
            val_calc = np.array(
                [
                    self.tglf_full.results["ev0"]["output"][j].__dict__[
                        quantityX.replace("[TGLF]", "")
                    ]
                    for j in range(len(self.rhos))
                ]
            )[impurity_search]

        thre = 10 ** round(np.log10(np.abs(val_calc).min()))
        axs[cont].set_xscale("symlog", linthresh=thre)
        axs[cont].set_yscale("symlog", linthresh=thre)
        axs[cont].tick_params(axis="both", which="major", labelsize=8)

        cont += 1

    if "w0" in self.predicted_channels:
        if UseTGLFfull_x is not None:
            raise Exception("Momentum plot not implemented yet")
        # w0
        quantityX = "MtGB_sim_turb"
        quantityX_stds = "MtGB_sim_turb_stds"
        quantityY = "MtGB_sim_turb"
        quantityY_stds = "MtGB_sim_turb_stds"
        metrics["MtJm2"] = plotModelComparison_quantity(
            self,
            axs[cont],
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
        axs[cont].set_xscale("symlog", linthresh=thre)
        axs[cont].set_yscale("symlog", linthresh=thre)
        axs[cont].tick_params(axis="both", which="major", labelsize=8)

        cont += 1

    if self.portals_parameters["solution"]["turbulent_exchange_as_surrogate"]:
        if UseTGLFfull_x is not None:
            raise Exception("Turbulent exchange plot not implemented yet")
        # Sexch
        quantityX = "EXeGB_sim_turb"
        quantityX_stds = "EXeGB_sim_turb_stds"
        quantityY = "EXeGB_sim_turb"
        quantityY_stds = "EXeGB_sim_turb_stds"
        metrics["EX"] = plotModelComparison_quantity(
            self,
            axs[cont],
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
        axs[cont].set_xscale("symlog", linthresh=thre)
        axs[cont].set_yscale("symlog", linthresh=thre)
        axs[cont].tick_params(axis="both", which="major", labelsize=8)

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

    if "cgyro_neo" in self.mitim_runs[0]["powerstate"].model_results.extra_analysis:
        resultsY = "cgyro_neo"
        quantity_label_resultsY = "(CGYRO)"
    else:
        resultsY = resultsX
        quantity_label_resultsY = quantity_label_resultsX

    nr = len(self.rhos)

    X, X_stds = [], []
    Y, Y_stds = [], []
    for i in range(self.ilast + 1):
        """
        Read the fluxes to be plotted in Y from the TGYRO results
        """
        t = self.mitim_runs[i]["powerstate"].model_results.extra_analysis
        Y.append(
            t[resultsY].__dict__[quantityY][
                ... if runWithImpurity is None else runWithImpurity, 0, 1:nr+1
            ]
        )
        Y_stds.append(
            t[resultsY].__dict__[quantityY_stds][
                ... if runWithImpurity is None else runWithImpurity, 0, 1:nr+1
            ]
        )

        """
        Read the fluxes to be plotted in X from...
        """

        # ...from the TGLF full results
        if "[TGLF]" in quantityX:
            X.append(
                [
                    self.tglf_full.results[f"ev{i}"]["output"][j].__dict__[
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
                    (... if runWithImpurity is None else runWithImpurity), 0, 1:nr+1
                ]
            )
            X_stds.append(
                t[resultsX].__dict__[quantityX_stds][
                    ... if runWithImpurity is None else runWithImpurity, 0, 1:nr+1
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


def varToReal(y, mitim_model):

    of, cal, res = mitim_model.optimization_object.scalarized_objective(
        torch.Tensor(y).to(mitim_model.optimization_object.dfT).unsqueeze(0)
    )

    cont = 0
    Qe, Qi, Ge, GZ, Mt = [], [], [], [], []
    Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = [], [], [], [], []
    for prof in mitim_model.optimization_object.portals_parameters["solution"]["predicted_channels"]:
        for rad in mitim_model.optimization_object.portals_parameters["solution"]["predicted_rho"]:
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
    mitim_model,
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
            mitim_model.optimization_object.surrogate_parameters["powerstate"]
            .plasma["roa"][0, 1:]
            .cpu()
            .cpu().numpy()
        ) 

        try:
            Qe, Qi, Ge, GZ, Mt, Qe_tar, Qi_tar, Ge_tar, GZ_tar, Mt_tar = varToReal(
                y[i, :].detach().cpu().numpy(), mitim_model
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
            ) = varToReal(yerr[0][i, :].detach().cpu().numpy(), mitim_model)
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
            ) = varToReal(yerr[1][i, :].detach().cpu().numpy(), mitim_model)

        if axTe_f is not None:
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

        if axTi_f is not None:
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
            if axTe_r is not None:
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
            if axTi_r is not None:
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
    power,
    axTe_f,
    axTi_f,
    axne_f,
    axnZ_f,
    axw0_f,
    force_zero_particle_flux=False,
    runWithImpurity=3,
    labZ="Z",
    includeFirst=True,
    alpha=1.0,
    stds=2,
    col="b",
    lab="",
    msFlux=1,
    maxStore=False,
    plotFlows=True,
    addFlowLegend=True,
    decor=True,
    fontsize_leg=12,
    useRoa=False,
    locLeg="upper left",
):

    r = power.plasma['rho'].cpu().numpy() if not useRoa else power.plasma['roa'].cpu().numpy()

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
            power.plasma['QeMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc'].cpu().numpy()[0][ixF:],
            "-s",
            c=col,
            lw=2,
            markersize=msFlux,
            label="Transport",
            alpha=alpha,
        )

        sigma = power.plasma['QeMWm2_tr_turb_stds'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc_stds'].cpu().numpy()[0][ixF:]

        m_Qe, M_Qe = (power.plasma['QeMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc'].cpu().numpy()[0][ixF:]) - stds * sigma, (
            power.plasma['QeMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc'].cpu().numpy()[0][ixF:]
        ) + stds * sigma
        axTe_f.fill_between(r[0][ixF:], m_Qe, M_Qe, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Ion energy flux
    # -----------------------------------------------------------------------------------------------

    if axTi_f is not None:
        axTi_f.plot(
            r[0][ixF:],
            power.plasma['QiMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QiMWm2_tr_neoc'].cpu().numpy()[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )

        sigma = (
            power.plasma['QiMWm2_tr_turb_stds'].cpu().numpy()[0][ixF:] + power.plasma['QiMWm2_tr_neoc_stds'].cpu().numpy()[0][ixF:]
        )

        m_Qi, M_Qi = (
            power.plasma['QiMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QiMWm2_tr_neoc'].cpu().numpy()[0][ixF:]
        ) - stds * sigma, (
            power.plasma['QiMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QiMWm2_tr_neoc'].cpu().numpy()[0][ixF:]
        ) + stds * sigma
        axTi_f.fill_between(r[0][ixF:], m_Qi, M_Qi, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Electron particle flux
    # -----------------------------------------------------------------------------------------------

    if axne_f is not None:

        Ge = power.plasma['Ge1E20m2_tr_turb'].cpu().numpy() + power.plasma['Ge1E20m2_tr_neoc'].cpu().numpy()

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

        sigma = power.plasma['Ge1E20m2_tr_turb_stds'].cpu().numpy()[0][ixF:] + power.plasma['Ge1E20m2_tr_neoc_stds'].cpu().numpy()[0][ixF:]


        m_Ge, M_Ge = Ge[0][ixF:] - stds * sigma, Ge[0][ixF:] + stds * sigma
        axne_f.fill_between(r[0][ixF:], m_Ge, M_Ge, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Impurity flux
    # -----------------------------------------------------------------------------------------------

    if axnZ_f is not None:
        GZ = power.plasma['GZ1E20m2_tr_turb'].cpu().numpy() + power.plasma['GZ1E20m2_tr_neoc'].cpu().numpy()

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

        sigma = power.plasma['GZ1E20m2_tr_turb_stds'].cpu().numpy()[0][ixF:] + power.plasma['GZ1E20m2_tr_neoc_stds'].cpu().numpy()[0][ixF:]

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
            power.plasma['MtJm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['MtJm2_tr_neoc'].cpu().numpy()[0][ixF:],
            "-s",
            markersize=msFlux,
            c=col,
            lw=2,
            label="Transport",
            alpha=alpha,
        )

        sigma = power.plasma['MtJm2_tr_turb_stds'].cpu().numpy()[0][ixF:] + power.plasma['MtJm2_tr_neoc_stds'].cpu().numpy()[0][ixF:]

        m_Mt, M_Mt = (power.plasma['MtJm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['MtJm2_tr_neoc'].cpu().numpy()[0][ixF:]) - stds * sigma, (
            power.plasma['MtJm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['MtJm2_tr_neoc'].cpu().numpy()[0][ixF:]
        ) + stds * sigma
        axw0_f.fill_between(r[0][ixF:], m_Mt, M_Mt, facecolor=col, alpha=alpha / 3)

    # -----------------------------------------------------------------------------------------------
    # Plot targets
    # -----------------------------------------------------------------------------------------------

    # Retrieve targets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Qe_tar = power.plasma['QeMWm2'].cpu().numpy()[0][ixF:]
    Qi_tar = power.plasma['QiMWm2'].cpu().numpy()[0][ixF:]
    Ge_tar = power.plasma['Ge1E20m2'].cpu().numpy()[0][ixF:] * (1-int(force_zero_particle_flux))
    GZ_tar = power.plasma['GZ1E20m2'].cpu().numpy()[0][ixF:]
    Mt_tar = power.plasma['MtJm2'].cpu().numpy()[0][ixF:]

    # Plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    rad = r[0][ixF:]

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

    tBest = power.profiles
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

                if var == "ge_10E20m2":
                    y *= 1 - int(force_zero_particle_flux)

                ax.plot(
                    (tBest.profiles["rho(-)"] if not useRoa else tBest.derived["roa"]),
                    y,
                    ":",
                    lw=1.0,
                    c=col,
                    alpha=alpha,
                )

    # -----------------------------------------------------------------------------------------------
    # Some decor
    # -----------------------------------------------------------------------------------------------

    # -- for legend
    if axTe_f is not None:
        (l1,) = axTe_f.plot(
            r[0][ixF:],
            power.plasma['QeMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc'].cpu().numpy()[0][ixF:],
            "-",
            c="k",
            lw=2,
            markersize=0,
            label="Transport",
        )
        (l2,) = axTe_f.plot(
            r[0][ixF:], power.plasma['QeMWm2'].cpu().numpy()[0][ixF:], "--*", c="k", lw=2, markersize=0, label="Target"
        )
        l3 = axTe_f.fill_between(
            r[0][ixF:],
            (power.plasma['QeMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc'].cpu().numpy()[0][ixF:]) - stds,
            (power.plasma['QeMWm2_tr_turb'].cpu().numpy()[0][ixF:] + power.plasma['QeMWm2_tr_neoc'].cpu().numpy()[0][ixF:]) + stds,
            facecolor="k",
            alpha=0.3,
        )

        setl = [l1, l3, l2]
        setlab = ["Transport", f"$\\pm{stds}\\sigma$", "Target"]

        if addFlowLegend:
            (l4,) = axTe_f.plot(
                tBest.profiles["rho(-)"] if not useRoa else tBest.derived["roa"],
                tBest.derived["qe_MWm2"],
                ":",
                c="k",
                lw=1.0,
                markersize=0,
            )
            setl.append(l4)
            setlab.append("Target high-res")
        else:
            l4 = l3

        axTe_f.legend(setl, setlab, loc=locLeg, prop={"size": fontsize_leg})
        l1.set_visible(False)
        l2.set_visible(False)
        l3.set_visible(False)
        l4.set_visible(False)
        # ---------------

    if decor:
        if axTe_f is not None:
            ax = axTe_f
            GRAPHICStools.addDenseAxis(ax)
            ax.set_xlabel("$\\rho_N$") if not useRoa else ax.set_xlabel("$r/a$")
            ax.set_ylabel(labelsFluxesF["te"])
            ax.set_xlim([0, 1])

        if axTi_f is not None:
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
            if axTe_f is not None:
                Qmax = QeBest_max
                Qmax += np.abs(Qmax) * 0.5
                Qmin = QeBest_min
                Qmin -= np.abs(Qmin) * 0.5
                axTe_f.set_ylim([0, Qmax])

            if axTi_f is not None:
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
    rhos = np.append([0], self_complete.portals_parameters["solution"]["predicted_rho"])
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

    X = torch.zeros(((len(rhos) - 1) * len(self_complete.portals_parameters["solution"]["predicted_channels"]), 2))
    l = len(rhos) - 1
    
    cont = 0 
    if "te" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        X[(0 + cont) * l : (1 + cont) * l, :] = torch.from_numpy(aLTe[1:, :])
        cont += 1
    if "ti" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        X[(0 + cont) * l : (1 + cont) * l, :] = torch.from_numpy(aLTi[1:, :])
        cont += 1
    if "ne" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        X[(0 + cont) * l : (1 + cont) * l, :] = torch.from_numpy(aLne[1:, :])
        cont += 1
    if "nZ" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        X[(0 + cont) * l : (1 + cont) * l, :] = torch.from_numpy(aLnZ[1:, :])
        cont += 1  
    if "w0" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        X[(0 + cont) * l : (1 + cont) * l, :] = torch.from_numpy(aLw0[1:, :])
        cont += 1

    X = X.transpose(0, 1)

    powerstate = PORTALStools.constructEvaluationProfiles(X, copy.deepcopy(self_complete.surrogate_parameters))

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

    if "nZ" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["nZ"][0] * 0.1,       # in 10^20
            y_up=powerstate.plasma["nZ"][1] * 0.1,  # in 10^20
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

    if "w0" in self_complete.portals_parameters["solution"]["predicted_channels"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["w0"][0]*1E-3,        # in krad/s
            y_up=powerstate.plasma["w0"][1]*1E-3,   # in krad/s
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
