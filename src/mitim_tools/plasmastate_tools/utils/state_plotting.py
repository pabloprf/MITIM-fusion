import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __version__
from IPython import embed


def add_figures(fn, fnlab='', fnlab_pre='', tab_color=None):

    fig1 = fn.add_figure(label= fnlab_pre + "Profiles" + fnlab, tab_color=tab_color)
    fig2 = fn.add_figure(label= fnlab_pre + "Powers" + fnlab, tab_color=tab_color)
    fig3 = fn.add_figure(label= fnlab_pre + "Geometry" + fnlab, tab_color=tab_color)
    fig4 = fn.add_figure(label= fnlab_pre + "Gradients" + fnlab, tab_color=tab_color)
    fig5 = fn.add_figure(label= fnlab_pre + "Flows" + fnlab, tab_color=tab_color)
    fig6 = fn.add_figure(label= fnlab_pre + "Other" + fnlab, tab_color=tab_color)
    fig7 = fn.add_figure(label= fnlab_pre + "Impurities" + fnlab, tab_color=tab_color)
    figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]

    return figs


def add_axes(figs):

    fig1, fig2, fig3, fig4, fig5, fig6, fig7 = figs

    grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    axsProf_1 = [
        fig1.add_subplot(grid[0, 0]),
        fig1.add_subplot(grid[1, 0]),
        fig1.add_subplot(grid[2, 0]),
        fig1.add_subplot(grid[0, 1]),
        fig1.add_subplot(grid[1, 1]),
        fig1.add_subplot(grid[2, 1]),
        fig1.add_subplot(grid[0, 2]),
        fig1.add_subplot(grid[1, 2]),
        fig1.add_subplot(grid[2, 2]),
    ]

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsProf_2 = [
        fig2.add_subplot(grid[0, 0]),
        fig2.add_subplot(grid[0, 1]),
        fig2.add_subplot(grid[1, 0]),
        fig2.add_subplot(grid[1, 1]),
        fig2.add_subplot(grid[0, 2]),
        fig2.add_subplot(grid[1, 2]),
    ]
    grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.3)
    ax00c = fig3.add_subplot(grid[0, 0])
    axsProf_3 = [
        ax00c,
        fig3.add_subplot(grid[1, 0], sharex=ax00c),
        fig3.add_subplot(grid[2, 0]),
        fig3.add_subplot(grid[0, 1]),
        fig3.add_subplot(grid[1, 1]),
        fig3.add_subplot(grid[2, 1]),
        fig3.add_subplot(grid[0, 2]),
        fig3.add_subplot(grid[1, 2]),
        fig3.add_subplot(grid[2, 2]),
        fig3.add_subplot(grid[:, 3]),
    ]

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsProf_4 = [
        fig4.add_subplot(grid[0, 0]),
        fig4.add_subplot(grid[1, 0]),
        fig4.add_subplot(grid[0, 1]),
        fig4.add_subplot(grid[1, 1]),
        fig4.add_subplot(grid[0, 2]),
        fig4.add_subplot(grid[1, 2]),
    ]

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsFlows = [
        fig5.add_subplot(grid[0, 0]),
        fig5.add_subplot(grid[1, 0]),
        fig5.add_subplot(grid[0, 1]),
        fig5.add_subplot(grid[0, 2]),
        fig5.add_subplot(grid[1, 1]),
        fig5.add_subplot(grid[1, 2]),
    ]

    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    axsProf_6 = [
        fig6.add_subplot(grid[0, 0]),
        fig6.add_subplot(grid[0, 1]),
        fig6.add_subplot(grid[0, 2]),
        fig6.add_subplot(grid[1, 0]),
        fig6.add_subplot(grid[1, 1]),
        fig6.add_subplot(grid[1, 2]),
        fig6.add_subplot(grid[0, 3]),
        fig6.add_subplot(grid[1, 3]),
    ]
    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    axsImps = [
        fig7.add_subplot(grid[0, 0]),
        fig7.add_subplot(grid[0, 1]),
        fig7.add_subplot(grid[0, 2]),
        fig7.add_subplot(grid[1, 0]),
        fig7.add_subplot(grid[1, 1]),
        fig7.add_subplot(grid[1, 2]),
    ]

    return axsProf_1, axsProf_2, axsProf_3, axsProf_4, axsFlows, axsProf_6, axsImps 


def plot_profiles(self, axs1, color="b", legYN=True, extralab="", lw=1, fs=6):
    
    [ax00, ax10, ax20, ax01, ax11, ax21, ax02, ax12, ax22] = axs1
    
    rho = self.profiles["rho(-)"]

    lines = GRAPHICStools.listLS()

    ax=ax00
    var = self.profiles["te(keV)"]
    varL = "$T_e$ , $T_i$ (keV)"
    if legYN:
        lab = extralab + "e"
    else:
        lab = ""
    ax.plot(rho, var, lw=lw, ls="-", label=lab, c=color)
    var = self.profiles["ti(keV)"][:, 0]
    if legYN:
        lab = extralab + "i"
    else:
        lab = ""
    ax.plot(rho, var, lw=lw, ls="--", label=lab, c=color)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    GRAPHICStools.autoscale_y(ax, bottomy=0)
    ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)


    ax=ax01
    var = self.profiles["ne(10^19/m^3)"] * 1e-1
    varL = "$n_e$ ($10^{20}/m^3$)"
    if legYN:
        lab = extralab + "e"
    else:
        lab = ""
    ax.plot(rho, var, lw=lw, ls="-", label=lab, c=color)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    GRAPHICStools.autoscale_y(ax, bottomy=0)
    ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)

    ax = ax10
    cont = 0
    for i in range(len(self.Species)):
        if self.Species[i]["S"] == "therm":
            var = self.profiles["ti(keV)"][:, i]
            ax.plot(
                rho,
                var,
                lw=lw,
                ls=lines[cont],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
            cont += 1
    varL = "Thermal $T_i$ (keV)"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    # ax.set_ylim(bottom=0);
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax20
    cont = 0
    for i in range(len(self.Species)):
        if self.Species[i]["S"] == "fast":
            var = self.profiles["ti(keV)"][:, i]
            ax.plot(
                rho,
                var,
                lw=lw,
                ls=lines[cont],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
            cont += 1
    varL = "Fast $T_i$ (keV)"
    ax.plot(
        rho,
        self.profiles["ti(keV)"][:, 0],
        lw=0.5,
        ls="-",
        alpha=0.5,
        c=color,
        label=extralab + "$T_{i,1}$",
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    # ax.set_ylim(bottom=0);
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax11
    cont = 0
    for i in range(len(self.Species)):
        if self.Species[i]["S"] == "therm":
            var = self.profiles["ni(10^19/m^3)"][:, i] * 1e-1
            ax.plot(
                rho,
                var,
                lw=lw,
                ls=lines[cont],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
            cont += 1
    varL = "Thermal $n_i$ ($10^{20}/m^3$)"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    # ax.set_ylim(bottom=0);
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax21
    cont = 0
    for i in range(len(self.Species)):
        if self.Species[i]["S"] == "fast":
            var = self.profiles["ni(10^19/m^3)"][:, i] * 1e-1 * 1e5
            ax.plot(
                rho,
                var,
                lw=lw,
                ls=lines[cont],
                c=color,
                label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
            )
            cont += 1
    varL = "Fast $n_i$ ($10^{15}/m^3$)"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    # ax.set_ylim(bottom=0);
    ax.set_ylabel(varL)
    if legYN and cont>0:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax02
    var = self.profiles["w0(rad/s)"]
    ax.plot(rho, var, lw=lw, ls="-", c=color)
    varL = "$\\omega_{0}$ (rad/s)"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax12
    var = self.profiles["ptot(Pa)"] * 1e-6
    ax.plot(rho, var, lw=lw, ls="-", c=color, label=extralab + "ptot")
    if "ptot_manual" in self.derived:
        ax.plot(
            rho,
            self.derived["ptot_manual"],
            lw=lw,
            ls="--",
            c=color,
            label=extralab + "check",
        )
        ax.plot(
            rho,
            self.derived["pthr_manual"],
            lw=lw,
            ls="-.",
            c=color,
            label=extralab + "check, thrm",
        )

    varL = "$p$ (MPa)"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    # ax.set_ylim(bottom=0)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax22
    var = self.profiles["q(-)"]
    ax.plot(rho, var, lw=lw, ls="-", c=color)
    varL = "$q$ profile"
    ax.axhline(y=1.0, lw=0.5, ls="--", c="k")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)


def plot_powers(self, axs2, legYN=True, extralab="", color="b", lw=1, fs=6):

    [ax00b, ax01b, ax10b, ax11b, ax20b, ax21b] = axs2

    rho = self.profiles["rho(-)"]

    lines = GRAPHICStools.listLS()

    ax = ax00b
    varL = "$MW/m^3$"
    cont = 0
    var = -self.profiles["qei(MW/m^3)"]
    ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "i->e", c=color)
    cont += 1
    if "qrfe(MW/m^3)" in self.profiles:
        var = self.profiles["qrfe(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "rf", c=color)
        cont += 1
    if "qfuse(MW/m^3)" in self.profiles:
        var = self.profiles["qfuse(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "fus", c=color)
        cont += 1
    if "qbeame(MW/m^3)" in self.profiles:
        var = self.profiles["qbeame(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "beam", c=color)
        cont += 1
    if "qione(MW/m^3)" in self.profiles:
        var = self.profiles["qione(MW/m^3)"]
        ax.plot(
            rho, var, lw=lw / 2, ls=lines[cont], label=extralab + "extra", c=color
        )
        cont += 1
    if "qohme(MW/m^3)" in self.profiles:
        var = self.profiles["qohme(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "ohmic", c=color)
        cont += 1

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)
    ax.set_title("Electron Power Density")
    ax.axhline(y=0, lw=0.5, ls="--", c="k")
    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax01b

    ax.plot(rho, self.profiles["qmom(N/m^2)"], lw=lw, ls="-", c=color)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("$N/m^2$, $J/m^3$")
    ax.axhline(y=0, lw=0.5, ls="--", c="k")
    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)
    ax.set_title("Momentum Source Density")

    ax = ax21b
    ax.plot(
        rho, self.derived["qe_MWm2"], lw=lw, ls="-", label=extralab + "qe", c=color
    )
    ax.plot(
        rho, self.derived["qi_MWm2"], lw=lw, ls="--", label=extralab + "qi", c=color
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("Heat Flux ($MW/m^2$)")
    if legYN:
        ax.legend(loc="lower left", fontsize=fs)
    ax.set_title("Flux per unit area (gacode: P/V')")
    ax.axhline(y=0, lw=0.5, ls="--", c="k")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax21b.twinx()
    ax.plot(
        rho,
        self.derived["ge_10E20m2"],
        lw=lw,
        ls="-.",
        label=extralab + "$\\Gamma_e$",
        c=color,
    )
    ax.set_ylabel("Particle Flux ($10^{20}/m^2/s$)")
    if legYN:
        ax.legend(loc="lower right", fontsize=fs)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax20b
    varL = "$Q_{rad}$ ($MW/m^3$)"
    if "qbrem(MW/m^3)" in self.profiles:
        var = self.profiles["qbrem(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls="-", label=extralab + "brem", c=color)
    if "qline(MW/m^3)" in self.profiles:
        var = self.profiles["qline(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls="--", label=extralab + "line", c=color)
    if "qsync(MW/m^3)" in self.profiles:
        var = self.profiles["qsync(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=":", label=extralab + "sync", c=color)

    var = self.derived["qrad"]
    ax.plot(rho, var, lw=lw * 1.5, ls="-", label=extralab + "Total", c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    # ax.set_ylim(bottom=0);
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)
    ax.set_title("Radiation Contributions")
    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax10b
    varL = "$MW/m^3$"
    cont = 0
    var = self.profiles["qei(MW/m^3)"]
    ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "e->i", c=color)
    cont += 1
    if "qrfi(MW/m^3)" in self.profiles:
        var = self.profiles["qrfi(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "rf", c=color)
        cont += 1
    if "qfusi(MW/m^3)" in self.profiles:
        var = self.profiles["qfusi(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "fus", c=color)
        cont += 1
    if "qbeami(MW/m^3)" in self.profiles:
        var = self.profiles["qbeami(MW/m^3)"]
        ax.plot(rho, var, lw=lw, ls=lines[cont], label=extralab + "beam", c=color)
        cont += 1
    if "qioni(MW/m^3)" in self.profiles:
        var = self.profiles["qioni(MW/m^3)"]
        ax.plot(
            rho, var, lw=lw / 2, ls=lines[cont], label=extralab + "extra", c=color
        )
        cont += 1

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)
    ax.set_title("Ion Power Density")
    ax.axhline(y=0, lw=0.5, ls="--", c="k")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax11b
    cont = 0
    var = self.profiles["qpar_beam(1/m^3/s)"] * 1e-20
    ax.plot(rho, var, lw=lw, ls=lines[0], c=color, label=extralab + "beam")
    var = self.profiles["qpar_wall(1/m^3/s)"] * 1e-20
    ax.plot(rho, var, lw=lw, ls=lines[1], c=color, label=extralab + "wall")

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    # ax.set_ylim(bottom=0);
    ax.axhline(y=0, lw=0.5, ls="--", c="k")
    ax.set_ylabel("$10^{20}m^{-3}s^{-1}$")
    ax.set_title("Particle Source Density")
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)


def plot_geometry(self, axs3, color="b", legYN=True, extralab="", lw=1, fs=6):

    [ax00c,ax10c,ax20c,ax01c,ax11c,ax21c,ax02c,ax12c,ax22c,ax13c] = axs3

    rho = self.profiles["rho(-)"]
    lines = GRAPHICStools.listLS()

    ax = ax00c
    varL = "cos Shape Params"
    yl = 0
    cont = 0

    for i, s in enumerate(self.shape_cos):
        if s is not None:
            valmax = np.abs(s).max()
            if valmax > 1e-10:
                lab = f"c{i}"
                ax.plot(rho, s, lw=lw, ls=lines[cont], label=lab, c=color)
                cont += 1

            yl = np.max([yl, valmax])

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)

    

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    if legYN:
        ax.legend(loc="best", fontsize=fs)

    ax = ax01c
    varL = "sin Shape Params"
    cont = 0
    for i, s in enumerate(self.shape_sin):
        if s is not None:
            valmax = np.abs(s).max()
            if valmax > 1e-10:
                lab = f"s{i}"
                ax.plot(rho, s, lw=lw, ls=lines[cont], label=lab, c=color)
                cont += 1

            yl = np.max([yl, valmax])

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax02c
    var = self.profiles["polflux(Wb/radian)"]
    ax.plot(rho, var, lw=lw, ls="-", c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("Poloidal $\\psi$ ($Wb/rad$)")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax10c
    var = self.profiles["delta(-)"]
    ax.plot(rho, var, "-", lw=lw, c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("$\\delta$")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)


    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax11c

    var = self.profiles["rmin(m)"]
    ax.plot(rho, var, "-", lw=lw, c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("$r_{min}$")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = ax20c

    var = self.profiles["rmaj(m)"]
    ax.plot(rho, var, "-", lw=lw, c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("$R_{maj}$")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax21c

    var = self.profiles["zmag(m)"]
    ax.plot(rho, var, "-", lw=lw, c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    yl = np.max([0.1, np.max(np.abs(var))])
    ax.set_ylim([-yl, yl])
    ax.set_ylabel("$Z_{maj}$")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax22c

    var = self.profiles["kappa(-)"]
    ax.plot(rho, var, "-", lw=lw, c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("$\\kappa$")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=1)

    ax = ax12c

    var = self.profiles["zeta(-)"]
    ax.plot(rho, var, "-", lw=lw, c=color)

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("zeta")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = ax13c
    self.plot_state_flux_surfaces(ax=ax, color=color)

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    GRAPHICStools.addDenseAxis(ax)


def plot_gradients(
    self,
    axs4,
    color="b",
    lw=1.0,
    label="",
    ls="-o",
    lastRho=0.89,
    ms=2,
    alpha=1.0,
    useRoa=False,
    RhoLocationsPlot=None,
    plotImpurity=None,
    plotRotation=False,
    autoscale=True,
    ):

    if RhoLocationsPlot is None: RhoLocationsPlot=[]

    if axs4 is None:
        plt.ion()
        fig, axs = plt.subplots(
            ncols=3 + int(plotImpurity is not None) + int(plotRotation),
            nrows=2,
            figsize=(12, 5),
        )

        axs4 = []
        for i in range(axs.shape[-1]):
            axs4.append(axs[0, i])
            axs4.append(axs[1, i])

    ix = np.argmin(np.abs(self.profiles["rho(-)"] - lastRho)) + 1

    xcoord = self.profiles["rho(-)"] if (not useRoa) else self.derived["roa"]
    labelx = "$\\rho$" if (not useRoa) else "$r/a$"

    ax = axs4[0]
    ax.plot(
        xcoord,
        self.profiles["te(keV)"],
        ls,
        c=color,
        lw=lw,
        label=label,
        markersize=ms,
        alpha=alpha,
    )
    ax = axs4[2]
    ax.plot(
        xcoord,
        self.profiles["ti(keV)"][:, 0],
        ls,
        c=color,
        lw=lw,
        markersize=ms,
        alpha=alpha,
    )
    ax = axs4[4]
    ax.plot(
        xcoord,
        self.profiles["ne(10^19/m^3)"] * 1e-1,
        ls,
        c=color,
        lw=lw,
        markersize=ms,
        alpha=alpha,
    )

    if "derived" in self.__dict__:
        ax = axs4[1]
        ax.plot(
            xcoord[:ix],
            self.derived["aLTe"][:ix],
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )
        ax = axs4[3]
        ax.plot(
            xcoord[:ix],
            self.derived["aLTi"][:ix, 0],
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )
        ax = axs4[5]
        ax.plot(
            xcoord[:ix],
            self.derived["aLne"][:ix],
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )

    for ax in axs4:
        ax.set_xlim([0, 1])

    ax = axs4[0]
    ax.set_ylabel("$T_e$ (keV)")
    ax.set_xlabel(labelx)
    if autoscale:
        GRAPHICStools.autoscale_y(ax, bottomy=0)
    ax.legend(loc="best", fontsize=7)
    ax = axs4[2]
    ax.set_ylabel("$T_i$ (keV)")
    ax.set_xlabel(labelx)
    if autoscale:
        GRAPHICStools.autoscale_y(ax, bottomy=0)
    ax = axs4[4]
    ax.set_ylabel("$n_e$ ($10^{20}m^{-3}$)")
    ax.set_xlabel(labelx)
    if autoscale:
        GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axs4[1]
    ax.set_ylabel("$a/L_{Te}$")
    ax.set_xlabel(labelx)
    if autoscale:
        GRAPHICStools.autoscale_y(ax, bottomy=0)
    ax = axs4[3]
    ax.set_ylabel("$a/L_{Ti}$")
    ax.set_xlabel(labelx)
    if autoscale:
        GRAPHICStools.autoscale_y(ax, bottomy=0)
    ax = axs4[5]
    ax.set_ylabel("$a/L_{ne}$")
    ax.axhline(y=0, ls="--", lw=0.5, c="k")
    ax.set_xlabel(labelx)
    if autoscale:
        GRAPHICStools.autoscale_y(ax, bottomy=0)

    cont = 0
    if plotImpurity is not None:
        axs4[6 + cont].plot(
            xcoord,
            self.profiles["ni(10^19/m^3)"][:, plotImpurity] * 1e-1,
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )
        axs4[6 + cont].set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
        axs4[6].set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        if "derived" in self.__dict__:
            axs4[7 + cont].plot(
                xcoord[:ix],
                self.derived["aLni"][:ix, plotImpurity],
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )
        axs4[7 + cont].set_ylabel("$a/L_{nZ}$")
        axs4[7 + cont].axhline(y=0, ls="--", lw=0.5, c="k")
        axs4[7 + cont].set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        cont += 2

    if plotRotation:
        axs4[6 + cont].plot(
            xcoord,
            self.profiles["w0(rad/s)"] * 1e-3,
            ls,
            c=color,
            lw=lw,
            markersize=ms,
            alpha=alpha,
        )
        axs4[6 + cont].set_ylabel("$w_0$ (krad/s)")
        axs4[6 + cont].set_xlabel(labelx)
        if "derived" in self.__dict__:
            axs4[7 + cont].plot(
                xcoord[:ix],
                self.derived["dw0dr"][:ix] * 1e-5,
                ls,
                c=color,
                lw=lw,
                markersize=ms,
                alpha=alpha,
            )
        axs4[7 + cont].set_ylabel("-$d\\omega_0/dr$ (krad/s/cm)")
        axs4[7 + cont].axhline(y=0, ls="--", lw=0.5, c="k")
        axs4[7 + cont].set_xlabel(labelx)
        if autoscale:
            GRAPHICStools.autoscale_y(ax, bottomy=0)
        cont += 2

    for x0 in RhoLocationsPlot:
        ix = np.argmin(np.abs(self.profiles["rho(-)"] - x0))
        for ax in axs4:
            ax.axvline(x=xcoord[ix], ls="--", lw=0.5, c=color)

    for i in range(len(axs4)):
        ax = axs4[i]
        GRAPHICStools.addDenseAxis(ax)

def plot_other(self, axs6, color="b", lw=1.0, extralab="", fs=6):
    
    rho = self.profiles["rho(-)"]
    lines = GRAPHICStools.listLS()
    
    # Others
    ax = axs6[0]
    ax.plot(self.profiles["rho(-)"], self.derived["dw0dr"] * 1e-5, c=color, lw=lw)
    ax.set_ylabel("$-d\\omega_0/dr$ (krad/s/cm)")
    ax.set_xlabel("$\\rho$")
    ax.set_xlim([0, 1])

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)
    ax.axhline(y=0, lw=1.0, c="k", ls="--")

    ax = axs6[2]
    ax.plot(self.profiles["rho(-)"], self.derived["q_fus"], c=color, lw=lw)
    ax.set_ylabel("$q_{fus}$ ($MW/m^3$)")
    ax.set_xlabel("$\\rho$")
    ax.set_xlim([0, 1])

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axs6[3]
    ax.plot(self.profiles["rho(-)"], self.derived["q_fus_MW"], c=color, lw=lw)
    ax.set_ylabel("$P_{fus}$ ($MW$)")
    ax.set_xlim([0, 1])

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axs6[4]
    ax.plot(self.profiles["rho(-)"], self.derived["tite"], c=color, lw=lw)
    ax.set_ylabel("$T_i/T_e$")
    ax.set_xlabel("$\\rho$")
    ax.set_xlim([0, 1])
    ax.axhline(y=1, ls="--", lw=1.0, c="k")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = axs6[5]
    if "MachNum" in self.derived:
        ax.plot(self.profiles["rho(-)"], self.derived["MachNum"], c=color, lw=lw)
    ax.set_ylabel("Mach Number")
    ax.set_xlabel("$\\rho$")
    ax.set_xlim([0, 1])
    ax.axhline(y=0, ls="--", c="k", lw=0.5)
    ax.axhline(y=1, ls="--", c="k", lw=0.5)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    ax = axs6[6]
    safe_division = np.divide(
        self.derived["qi_MWm2"],
        self.derived["qe_MWm2"],
        where=self.derived["qe_MWm2"] != 0,
        out=np.full_like(self.derived["qi_MWm2"], np.nan),
    )
    ax.plot(
        self.profiles["rho(-)"],
        safe_division,
        c=color,
        lw=lw,
        label=extralab + "$Q_i/Q_e$",
    )
    safe_division = np.divide(
        self.derived["qi_aux_MW"],
        self.derived["qe_aux_MW"],
        where=self.derived["qe_aux_MW"] != 0,
        out=np.full_like(self.derived["qi_aux_MW"], np.nan),
    )
    ax.plot(
        self.profiles["rho(-)"],
        safe_division,
        c=color,
        lw=lw,
        ls="--",
        label=extralab + "$P_i/P_e$",
    )
    ax.set_ylabel("Power ratios")
    ax.set_xlabel("$\\rho$")
    ax.set_xlim([0, 1])
    ax.axhline(y=1.0, ls="--", c="k", lw=1.0)
    GRAPHICStools.addDenseAxis(ax)
    # GRAPHICStools.autoscale_y(ax,bottomy=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=fs)

    # Currents
    
    ax = axs6[1]
    
    var = self.profiles["johm(MA/m^2)"]
    ax.plot(rho, var, "-", lw=lw, c=color, label=extralab + "$J_{OH}$")
    var = self.profiles["jbs(MA/m^2)"]
    ax.plot(rho, var, "--", lw=lw, c=color, label=extralab + "$J_{BS,par}$")
    var = self.profiles["jbstor(MA/m^2)"]
    ax.plot(rho, var, "-.", lw=lw, c=color, label=extralab + "$J_{BS,tor}$")

    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("J ($MA/m^2$)")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axs6[7]
    cont = 0
    if "vtor(m/s)" in self.profiles:
        for i in range(len(self.Species)):
            try:  # REMOVE FOR FUTURE
                var = self.profiles["vtor(m/s)"][:, i] * 1e-3
                ax.plot(
                    rho,
                    var,
                    lw=lw,
                    ls=lines[cont],
                    c=color,
                    label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
                )
                cont += 1
            except:
                break
    varL = "$V_{tor}$ (km/s)"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)


def plot_flows(self, axs=None, limits=None, ls="-", leg=True, showtexts=True):
    if axs is None:
        fig1 = plt.figure()
        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        axs = [
            fig1.add_subplot(grid[0, 0]),
            fig1.add_subplot(grid[1, 0]),
            fig1.add_subplot(grid[0, 1]),
            fig1.add_subplot(grid[0, 2]),
            fig1.add_subplot(grid[1, 1]),
            fig1.add_subplot(grid[1, 2]),
        ]

    # Profiles

    ax = axs[0]
    axT = axs[1]
    roa = self.profiles["rmin(m)"] / self.profiles["rmin(m)"][-1]
    Te = self.profiles["te(keV)"]
    ne = self.profiles["ne(10^19/m^3)"] * 1e-1
    ni = self.profiles["ni(10^19/m^3)"] * 1e-1
    niT = np.sum(ni, axis=1)
    Ti = self.profiles["ti(keV)"][:, 0]
    ax.plot(roa, Te, lw=2, c="r", label="$T_e$" if leg else "", ls=ls)
    ax.plot(roa, Ti, lw=2, c="b", label="$T_i$" if leg else "", ls=ls)
    axT.plot(roa, ne, lw=2, c="m", label="$n_e$" if leg else "", ls=ls)
    axT.plot(roa, niT, lw=2, c="c", label="$\\sum n_i$" if leg else "", ls=ls)
    if limits is not None:
        [roa_first, roa_last] = limits
        ax.plot(roa_last, np.interp(roa_last, roa, Te), "s", c="r", markersize=3)
        ax.plot(roa_first, np.interp(roa_first, roa, Te), "s", c="r", markersize=3)
        ax.plot(roa_last, np.interp(roa_last, roa, Ti), "s", c="b", markersize=3)
        ax.plot(roa_first, np.interp(roa_first, roa, Ti), "s", c="b", markersize=3)
        axT.plot(roa_last, np.interp(roa_last, roa, ne), "s", c="m", markersize=3)
        axT.plot(roa_first, np.interp(roa_first, roa, ne), "s", c="m", markersize=3)

    ax.set_xlabel("r/a")
    ax.set_xlim([0, 1])
    axT.set_xlabel("r/a")
    axT.set_xlim([0, 1])
    ax.set_ylabel("$T$ (keV)")
    ax.set_ylim(bottom=0)
    axT.set_ylabel("$n$ ($10^{20}m^{-3}$)")
    axT.set_ylim(bottom=0)
    # axT.set_ylim([0,np.max(ne)*1.5])
    ax.legend()
    axT.legend()
    ax.set_title("Final Temperature profiles")
    axT.set_title("Final Density profiles")

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    GRAPHICStools.addDenseAxis(axT)
    GRAPHICStools.autoscale_y(axT, bottomy=0)

    if showtexts:
        if self.derived["Q"] > 0.005:
            ax.text(
                0.05,
                0.05,
                f"Pfus = {self.derived['Pfus']:.1f}MW, Q = {self.derived['Q']:.2f}",
                color="k",
                fontsize=10,
                fontweight="normal",
                horizontalalignment="left",
                verticalalignment="bottom",
                rotation=0,
                transform=ax.transAxes,
            )

        axT.text(
            0.05,
            0.4,
            "ne_20 = {0:.1f} (fG = {1:.2f}), Zeff = {2:.1f}".format(
                self.derived["ne_vol20"],
                self.derived["fG"],
                self.derived["Zeff_vol"],
            ),
            color="k",
            fontsize=10,
            fontweight="normal",
            horizontalalignment="left",
            verticalalignment="bottom",
            rotation=0,
            transform=axT.transAxes,
        )

    # F
    ax = axs[2]
    P = (
        self.derived["qe_fus_MW"]
        + self.derived["qe_aux_MW"]
        + -self.derived["qe_rad_MW"]
        + -self.derived["qe_exc_MW"]
    )

    ax.plot(
        roa,
        -self.derived["qe_MW"],
        c="g",
        lw=2,
        label="$P_{e}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qe_fus_MW"],
        c="r",
        lw=2,
        label="$P_{fus,e}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qe_aux_MW"],
        c="b",
        lw=2,
        label="$P_{aux,e}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        -self.derived["qe_exc_MW"],
        c="m",
        lw=2,
        label="$P_{exc,e}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        -self.derived["qe_rad_MW"],
        c="c",
        lw=2,
        label="$P_{rad,e}$" if leg else "",
        ls=ls,
    )
    ax.plot(roa, -P, lw=1, c="y", label="sum" if leg else "", ls=ls)

    # Pe = self.profiles['te(keV)']*1E3*e_J*self.profiles['ne(10^19/m^3)']*1E-1*1E20 *1E-6
    # ax.plot(roa,Pe,ls='-',lw=3,alpha=0.1,c='k',label='$W_e$ (MJ/m^3)')

    ax.plot(
        roa,
        -self.derived["ce_MW"],
        c="k",
        lw=1,
        label="($P_{conv,e}$)" if leg else "",
    )

    ax.set_xlabel("r/a")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$P$ (MW)")
    # ax.set_ylim(bottom=0)
    ax.set_title("Electron Thermal Flows")
    ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
    )

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axs[3]
    P = (
        self.derived["qi_fus_MW"]
        + self.derived["qi_aux_MW"]
        + self.derived["qe_exc_MW"]
    )

    ax.plot(
        roa,
        -self.derived["qi_MW"],
        c="g",
        lw=2,
        label="$P_{i}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qi_fus_MW"],
        c="r",
        lw=2,
        label="$P_{fus,i}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qi_aux_MW"],
        c="b",
        lw=2,
        label="$P_{aux,i}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qe_exc_MW"],
        c="m",
        lw=2,
        label="$P_{exc,i}$" if leg else "",
        ls=ls,
    )
    ax.plot(roa, -P, lw=1, c="y", label="sum" if leg else "", ls=ls)

    # Pi = self.profiles['ti(keV)'][:,0]*1E3*e_J*self.profiles['ni(10^19/m^3)'][:,0]*1E-1*1E20 *1E-6
    # ax.plot(roa,Pi,ls='-',lw=3,alpha=0.1,c='k',label='$W_$ (MJ/m^3)')

    ax.set_xlabel("r/a")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$P$ (MW)")
    # ax.set_ylim(bottom=0)
    ax.set_title("Ion Thermal Flows")
    ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
    )

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    # F
    ax = axs[4]

    ax.plot(
        roa,
        self.derived["ge_10E20"],
        c="g",
        lw=2,
        label="$\\Gamma_{e}$" if leg else "",
        ls=ls,
    )
    # ax.plot(roa,self.profiles['ne(10^19/m^3)']*1E-1,lw=3,alpha=0.1,c='k',label='$n_e$ ($10^{20}/m^3$)' if leg else '',ls=ls)

    ax.set_xlabel("r/a")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$\\Gamma$ ($10^{20}/s$)")
    ax.set_title("Particle Flows")
    ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
    )

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    # TOTAL
    ax = axs[5]
    P = (
        self.derived["qOhm_MW"]
        + self.derived["qRF_MW"]
        + self.derived["qFus_MW"]
        + -self.derived["qe_rad_MW"]
        + self.derived["qz_MW"]
        + self.derived["qBEAM_MW"]
    )

    ax.plot(
        roa,
        -self.derived["q_MW"],
        c="g",
        lw=2,
        label="$P$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qOhm_MW"],
        c="k",
        lw=2,
        label="$P_{Oh}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qRF_MW"],
        c="b",
        lw=2,
        label="$P_{RF}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qBEAM_MW"],
        c="pink",
        lw=2,
        label="$P_{NBI}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qFus_MW"],
        c="r",
        lw=2,
        label="$P_{fus}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        -self.derived["qe_rad_MW"],
        c="c",
        lw=2,
        label="$P_{rad}$" if leg else "",
        ls=ls,
    )
    ax.plot(
        roa,
        self.derived["qz_MW"],
        c="orange",
        lw=1,
        label="$P_{ionz.}$" if leg else "",
        ls=ls,
    )

    # P = Pe+Pi
    # ax.plot(roa,P,ls='-',lw=3,alpha=0.1,c='k',label='$W$ (MJ)')

    ax.plot(roa, -P, lw=1, c="y", label="sum" if leg else "", ls=ls)
    ax.set_xlabel("r/a")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$P$ (MW)")
    # ax.set_ylim(bottom=0)
    ax.set_title("Total Thermal Flows")

    GRAPHICStools.addLegendApart(
        ax, ratio=0.9, withleg=True, extraPad=0, size=None, loc="upper left"
    )

    ax.axhline(y=0.0, lw=0.5, ls="--", c="k")
    # GRAPHICStools.drawLineWithTxt(ax,0.0,label='',orientation='vertical',color='k',lw=1,ls='--',alpha=1.0,fontsize=10,fromtop=0.85,fontweight='normal',
    # 				verticalalignment='bottom',horizontalalignment='left',separation=0)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)




def plot_ions(self, axsImps, legYN=True, extralab="", color="b", lw=1, fs=6):

    rho = self.profiles["rho(-)"]
    lines = GRAPHICStools.listLS()

    # Impurities
    ax = axsImps[0]
    for i in range(len(self.Species)):
        var = (
            self.profiles["ni(10^19/m^3)"][:, i]
            / self.profiles["ni(10^19/m^3)"][0, i]
        )
        ax.plot(
            rho,
            var,
            lw=lw,
            ls=lines[i],
            c=color,
            label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
        )
    varL = "$n_i/n_{i,0}$"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axsImps[1]
    for i in range(len(self.Species)):
        var = self.derived["fi"][:, i]
        ax.plot(
            rho,
            var,
            lw=lw,
            ls=lines[i],
            c=color,
            label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
        )
    varL = "$f_i$"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    ax.set_ylim([0, 1])
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axsImps[2]

    lastRho = 0.9

    ix = np.argmin(np.abs(self.profiles["rho(-)"] - lastRho)) + 1
    ax.plot(
        rho[:ix], self.derived["aLne"][:ix], lw=lw * 3, ls="-", c=color, label="e"
    )
    for i in range(len(self.Species)):
        var = self.derived["aLni"][:, i]
        ax.plot(
            rho[:ix],
            var[:ix],
            lw=lw,
            ls=lines[i],
            c=color,
            label=extralab + f"{i + 1} = {self.profiles['name'][i]}",
        )
    varL = "$a/L_{ni}$"
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel(varL)
    if legYN:
        ax.legend(loc="best", fontsize=fs)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax)

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)

    ax = axsImps[5]

    ax = axsImps[3]
    ax.plot(self.profiles["rho(-)"], self.derived["Zeff"], c=color, lw=lw)
    ax.set_ylabel("$Z_{eff}$")
    ax.set_xlabel("$\\rho$")
    ax.set_xlim([0, 1])

    GRAPHICStools.addDenseAxis(ax)
    GRAPHICStools.autoscale_y(ax, bottomy=0)


def plotAll(profiles_list, figs=None, extralabs=None, lastRhoGradients=0.89):
    if figs is not None:
        fn = None
    else:
        from mitim_tools.misc_tools.GUItools import FigureNotebook
        fn = FigureNotebook("Profiles", geometry="1800x900")
        figs = add_figures(fn)

    axsProf_1, axsProf_2, axsProf_3, axsProf_4, axsFlows, axsProf_6, axsImps = add_axes(figs)

    ls = GRAPHICStools.listLS()
    colors = GRAPHICStools.listColors()
    for i, profiles in enumerate(profiles_list):
        if extralabs is None:
            extralab = f"#{i}, "
        else:
            extralab = f"{extralabs[i]}, "
            
        profiles.plot(
            axs1=axsProf_1,axs2=axsProf_2,axs3=axsProf_3,axs4=axsProf_4,axsFlows=axsFlows,axs6=axsProf_6,axsImps=axsImps,
            color=colors[i],legYN=True,extralab=extralab,lsFlows=ls[i],legFlows=i == 0,showtexts=False,lastRhoGradients=lastRhoGradients,
        )

    return fn
