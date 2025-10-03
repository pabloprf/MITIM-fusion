import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

#TODO: add current profiles and flux-surface average fields to megpy and restore plots

def compareGeqdsk(geqdsks, fn=None, extraLabel="", plotAll=True, labelsGs=None):
    
    if fn is None:
        from mitim_tools.misc_tools.GUItools import FigureNotebook
        fn = FigureNotebook("GEQDSK Notebook", geometry="1600x1000")

    if labelsGs is None:
        labelsGs = []
        for i, g in enumerate(geqdsks):
            labelsGs.append(f"#{i + 1}")

    # -----------------------------------------------------------------------------
    # Plot All
    # -----------------------------------------------------------------------------
    if plotAll:
        for i, g in enumerate(geqdsks):
            _ = g.plot(fn=fn, extraLabel=f"{labelsGs[i]} - ")

    # -----------------------------------------------------------------------------
    # Compare in same plot - Surfaces
    # -----------------------------------------------------------------------------
    fig = fn.add_figure(label=extraLabel + "Comp. - Surfaces")

    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[:, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[1, 2])
    axs = [ax1, ax2, ax3, ax4]

    cols = GRAPHICStools.listColors()

    for i, g in enumerate(geqdsks):
        g.plotFS(axs=axs, color=cols[i], label=f"#{i + 1}")

    ax3.legend()

    # -----------------------------------------------------------------------------
    # Compare in same plot - Surfaces
    # -----------------------------------------------------------------------------
    fig = fn.add_figure(label=extraLabel + "Comp. - Plasma")
    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)

    ax_plasma = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[0, 3]),
    ]  # ,
    # fig.add_subplot(grid[1,3])]

    cols = GRAPHICStools.listColors()

    for i, g in enumerate(geqdsks):
        g.plotFS(axs=[ax1, ax2, ax3, ax4], color=cols[i], label=f"{labelsGs[i]} ")
        g.plotPlasma(
            axs=ax_plasma,
            legendYN=False,
            color=cols[i],
            label=f"{labelsGs[i]} ",
        )

    return ax_plasma, fn

# -----------------------------------------------------------------------------
# Plot of GEQ class
# -----------------------------------------------------------------------------

def plot(self, fn=None, extraLabel=""):
    if fn is None:
        wasProvided = False

        from mitim_tools.misc_tools.GUItools import FigureNotebook

        self.fn = FigureNotebook("GEQDSK Notebook", geometry="1600x1000")
    else:
        wasProvided = True
        self.fn = fn
    # -----------------------------------------------------------------------------
    # OMFIT Summary
    # -----------------------------------------------------------------------------
    # fig = self.fn.add_figure(label=extraLabel+'OMFIT Summ')
    # self.g.plot()

    # -----------------------------------------------------------------------------
    # Flux
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "Surfaces")
    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[:, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[1, 2])

    self.plotFS(axs=[ax1, ax2, ax3, ax4])

    # -----------------------------------------------------------------------------
    # Currents
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "Currents")
    grid = plt.GridSpec(3, 5, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[2, 0])
    ax2 = fig.add_subplot(grid[:2, 0])
    ax3 = fig.add_subplot(grid[2, 1])
    ax4 = fig.add_subplot(grid[:2, 1])
    ax5 = fig.add_subplot(grid[2, 2])
    ax6 = fig.add_subplot(grid[:2, 2])
    ax7 = fig.add_subplot(grid[2, 3])
    ax8 = fig.add_subplot(grid[:2, 3])
    ax9 = fig.add_subplot(grid[2, 4])
    ax10 = fig.add_subplot(grid[:2, 4])

    plotCurrents(self,
        axs=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10], zlims_thr=[-1, 1]
    )

    # -----------------------------------------------------------------------------
    # Fields
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "Fields")
    grid = plt.GridSpec(3, 5, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[2, 0])
    ax2 = fig.add_subplot(grid[:2, 0])
    ax3 = fig.add_subplot(grid[2, 1])
    ax4 = fig.add_subplot(grid[:2, 1])
    ax5 = fig.add_subplot(grid[2, 2])
    ax6 = fig.add_subplot(grid[:2, 2])
    ax7 = fig.add_subplot(grid[2, 3])
    ax8 = fig.add_subplot(grid[:2, 3])
    ax9 = fig.add_subplot(grid[2, 4])
    ax10 = fig.add_subplot(grid[:2, 4])

    plotFields(self,
        axs=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10], zlims_thr=[-1, 1]
    )

    # -----------------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "GS Quality")
    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1E = ax1.twinx()
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[:, 1])
    ax4 = fig.add_subplot(grid[:, 2])
    ax5 = fig.add_subplot(grid[:, 3])

    plotChecks(self,axs=[ax1, ax1E, ax2, ax3, ax4, ax5])

    # -----------------------------------------------------------------------------
    # Parameterization
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "Parameteriz.")
    grid = plt.GridSpec(3, 4, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1])
    ax4 = fig.add_subplot(grid[2, 1])
    ax5 = fig.add_subplot(grid[:, 2])

    plotParameterization(self,axs=[ax1, ax2, ax3, ax4, ax5])

    # -----------------------------------------------------------------------------
    # Plasma
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "Plasma")
    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)

    ax_plasma = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 3]),
    ]
    ax_plasma = self.plotPlasma(axs=ax_plasma, legendYN=not wasProvided)

    # -----------------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------------
    fig = self.fn.add_figure(label=extraLabel + "Geometry")
    grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    self.plotGeometry(axs=[ax1, ax2, ax3, ax4])

    return ax_plasma

def plotFS(self, axs=None, color="b", label=""):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=4)

    ax = axs[0]
    self.plotFluxSurfaces(
        ax=ax, fluxes=np.linspace(0, 1, 21), rhoPol=True, sqrt=False, color=color
    )
    ax.plot(self.Rb, self.Yb, lw=1, c="r")
    ax.set_title("Poloidal Flux")
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")

    ax = axs[1]
    self.plotFluxSurfaces(
        ax=ax, fluxes=np.linspace(0, 1, 21), rhoPol=False, sqrt=True, color=color
    )
    ax.plot(self.Rb, self.Yb, lw=1, c="r")
    ax.set_title("Sqrt Toroidal Flux")
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")

    ax = axs[2]
    x = self.psi_pol_norm
    y = self.rho_tor
    ax.plot(x, y, lw=2, ls="-", c=color, label=label)
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.5)

    ax.set_xlabel("$\\Psi_n$ (PSI_NORM)")
    ax.set_ylabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax = axs[3]
    x = self.rho_tor
    y = self.rho_pol
    ax.plot(x, y, lw=2, ls="-", c=color)
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.5)

    ax.set_ylabel("$\\sqrt{\\Psi_n}$ (RHOp)")
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

def plotCurrents(self, axs=None, zlims_thr=[-1, 1]):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=10)

    ax = axs[0]
    x = self.psi_pol_norm
    y = np.zeros(x.shape)
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_ylabel("FSA $\\langle J\\rangle$ ($MA/m^2$)")
    ax.set_xlim([0, 1])

    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    #ax = axs[1]
    #plot2Dquantity(self,
    #    ax=ax, var="Jr", title="Radial Current Jr", zlims=zlims, factor=1e-6
    #)

    ax = axs[2]
    x = self.psi_pol_norm
    y = np.zeros(x.shape)
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    #ax = axs[3]
    #plot2Dquantity(self,
    #    ax=ax, var="Jz", title="Vertical Current Jz", zlims=zlims, factor=1e-6
    #)

    ax = axs[4]
    x = self.psi_pol_norm
    y = self.Jt
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    #ax = axs[5]
    #plot2Dquantity(self,
    #    ax=ax, var="Jt", title="Toroidal Current Jt", zlims=zlims, factor=1e-6
    #)

    ax = axs[6]
    x = self.psi_pol_norm
    y = np.zeros(x.shape)
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    #ax = axs[7]
    #plot2Dquantity(self,
    #    ax=ax, var="Jp", title="Poloidal Current Jp", zlims=zlims, factor=1e-6
    #)

    ax = axs[8]
    x = self.psi_pol_norm
    y = np.zeros(x.shape)
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    #ax = axs[9]
    #plot2Dquantity(self,
    #    ax=ax, var="Jpar", title="Parallel Current Jpar", zlims=zlims, factor=1e-6
    #)

def plotFields(self, axs=None, zlims_thr=[-1, 1]):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=10)

    ax = axs[0]
    x = self.psi_pol_norm
    y = np.zeros(x.shape) # self.g.surfAvg("Br")
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_ylabel("FSA $\\langle B\\rangle$ ($T$)")
    ax.set_xlim([0, 1])

    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    ax = axs[1]
    plot2Dquantity(self,
        ax=ax, var="B_r", title="Radial Field Br", zlims=zlims, titlebar="B ($T$)"
    )

    ax = axs[2]
    x = self.psi_pol_norm
    y = np.zeros(x.shape) # self.g.surfAvg("Bz")
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    ax = axs[3]
    plot2Dquantity(self,
        ax=ax, var="B_z", title="Vertical Field Bz", zlims=zlims, titlebar="B ($T$)"
    )

    ax = axs[4]
    x = self.psi_pol_norm
    y = np.zeros(x.shape) # self.g.surfAvg("Bt")
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [y.min(), y.max()]
    # zlims = [np.min([zlims_thr[0],y.min()]),np.max([zlims_thr[1],y.max()])]
    # zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    #ax = axs[5]
    #plot2Dquantity(self,
    #    ax=ax, var="Jt", title="Toroidal Field Bt", zlims=zlims, titlebar="B ($T$)"
    #)

    ax = axs[6]
    x = self.psi_pol_norm
    y = np.zeros(x.shape) # self.g.surfAvg("Bp")
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
    zlims = GRAPHICStools.aroundZeroLims(zlims)
    ax.set_ylim(zlims)

    ax = axs[7]
    plot2Dquantity(self,
        ax=ax, var="B_pol_rz", title="Poloidal Field Bp", zlims=zlims, titlebar="B ($T$)"
    )

    ax = axs[8]
    x = self.psi_pol_norm
    y = np.zeros(x.shape) # self.g["fluxSurfaces"]["avg"]["Bp**2"]
    ax.plot(x, y, lw=2, ls="-", c="r")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$\\langle B_{\\theta}^2\\rangle$")

    #ax = axs[9]
    #x = self.g["fluxSurfaces"]["midplane"]["R"]
    #y = self.g["fluxSurfaces"]["midplane"]["Bt"]
    #ax.plot(x, y, lw=2, ls="-", c="r", label="$B_{t}$")
    #y = self.g["fluxSurfaces"]["midplane"]["Bp"]
    #ax.plot(x, y, lw=2, ls="-", c="b", label="$B_{p}$")
    #y = self.g["fluxSurfaces"]["midplane"]["Bz"]
    #ax.plot(x, y, lw=2, ls="-", c="g", label="$B_{z}$")
    #y = self.g["fluxSurfaces"]["midplane"]["Br"]
    #ax.plot(x, y, lw=2, ls="-", c="m", label="$B_{r}$")
    #y = self.g["fluxSurfaces"]["geo"]["bunit"]
    #ax.plot(x, y, lw=2, ls="-", c="c", label="$B_{unit}$")
    #ax.set_xlabel("$R$ LF midplane")
    #ax.set_ylabel("$B$ (T)")
    #ax.legend()

def plotChecks(self, axs=None):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=8)

    ax = axs[0]
    x = self.psi_pol_norm
    y1 = self.Jt
    ax.plot(x, np.abs(y1), lw=2, ls="-", c="b", label="$\\langle Jt\\rangle$")
    zmax = y1.max()
    zmin = y1.min()
    y2 = self.Jt_fb
    ax.plot(x, np.abs(y2), lw=2, ls="-", c="g", label="$\\langle Jt_{FB}\\rangle$")

    y3 = self.Jerror
    ax.plot(
        x,
        y3,
        lw=1,
        ls="-",
        c="r",
        label="$|\\langle Jt\\rangle-\\langle Jt_{FB}\\rangle|$",
    )

    ax.set_ylabel("Current Density ($MA/m^2$)")

    axE = axs[1]
    yErr = np.abs(self.Jerror / self.Jt) * 100.0
    axE.plot(x, yErr, lw=0.5, ls="--", c="b")
    axE.set_ylim([0, 50])
    axE.set_ylabel("Relative Error (%)")

    ax.set_title("$|\\langle Jt\\rangle|$")
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\Psi_n$")
    ax.legend()

    ax = axs[2]
    x = self.psi_pol_norm
    y1 = self.g.raw["ffprim"]
    ax.plot(x, y1, lw=2, ls="-", c="r", label="$FF'$")
    y2 = self.g.raw["pprime"] * (4 * np.pi * 1e-7)
    ax.plot(x, y2, lw=2, ls="-", c="b", label="$p'*\\mu_0$")

    ax.set_ylabel("")
    ax.legend()
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])

    #ax = axs[3]
    #plot2Dquantity(self,
    #    ax=ax,
    #    var="Jt",
    #    title="Toroidal Current Jt",
    #    zlims=[zmin, zmax],
    #    cmap="viridis",
    #    factor=1e-6,
    #)

    #ax = axs[4]
    #plot2Dquantity(self,
    #    ax=ax,
    #    var="Jt_fb",
    #    title="Toroidal Current Jt (FB)",
    #    zlims=[zmin, zmax],
    #    cmap="viridis",
    #    factor=1e-6,
    #)

    #ax = axs[5]
    #z = (
    #    np.abs(self.g["AuxQuantities"]["Jt"] - self.g["AuxQuantities"]["Jt_fb"])
    #    * 1e-6
    #)
    #zmaxx = np.max([np.abs(zmax), np.abs(zmin)])
    #plot2Dquantity(self,
    #    ax=ax,
    #    var=z,
    #    title="Absolute Error",
    #    zlims=[0, zmaxx],
    #    cmap="viridis",
    #    direct=True,
    #)

def plotParameterization(self, axs=None):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=5)

    ax = axs[0]
    cs, csA = self.plotFluxSurfaces(
        ax=ax, fluxes=np.linspace(0, 1, 21), rhoPol=True, sqrt=False
    )
    # Boundary, axis and limiter
    ax.plot(self.Rb, self.Yb, lw=1, c="r")
    ax.plot(self.g.raw["rmaxis"], self.g.raw["zmaxis"], "+", markersize=10, c="r")
    ax.plot([self.Rmag], [self.Zmag], "o", markersize=5, c="m")
    ax.plot([self.Rmajor], [self.Zmag], "+", markersize=10, c="k")
    if 'rlim' in self.g.raw and 'zlim' in self.g.raw:
        ax.plot(self.g.raw["rlim"], self.g.raw["zlim"], lw=1, c="k")

        import matplotlib

        path = matplotlib.path.Path(
            np.transpose(np.array([self.g.raw["rlim"], self.g.raw["zlim"]]))
        )
        patch = matplotlib.patches.PathPatch(path, facecolor="none")
        ax.add_patch(patch)
        # for col in cs.collections:
        #     col.set_clip_path(patch)
        # for col in csA.collections:
        #     col.set_clip_path(patch)

    self.plotEnclosingBox(ax=ax)

    ax.set_aspect("equal")
    ax.set_title("Poloidal Flux")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")

    ax = axs[1]
    x = self.psi_pol_norm
    y = self.g.derived["miller_geo"]["kappa"].copy()
    ax.plot(x, y, label="$\\kappa$")
    y = self.g.derived["miller_geo"]["kappa_l"].copy()
    ax.plot(x, y, ls="--", label="$\\kappa_L$")
    y = self.g.derived["miller_geo"]["kappa_u"].copy()
    ax.plot(x, y, ls="--", label="$\\kappa_U$")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("Elongation $\\kappa$")
    ax.legend()

    ax = axs[2]
    x = self.psi_pol_norm
    y = self.g.derived["miller_geo"]["delta"].copy()
    ax.plot(x, y, label="$\\delta$")
    y = self.g.derived["miller_geo"]["delta_l"].copy()
    ax.plot(x, y, ls="--", label="$\\delta_L$")
    y = self.g.derived["miller_geo"]["delta_u"].copy()
    ax.plot(x, y, ls="--", label="$\\delta_U$")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("Triangularity $\\delta$")
    ax.legend()

    ax = axs[3]
    x = self.psi_pol_norm
    y = self.g.derived["miller_geo"]["zeta"].copy()
    ax.plot(x, y, label="$\\zeta$")
    y = self.g.derived["miller_geo"]["zeta_li"].copy()
    ax.plot(x, y, ls="--", label="$\\zeta_{IL}$")
    y = self.g.derived["miller_geo"]["zeta_ui"].copy()
    ax.plot(x, y, ls="--", label="$\\zeta_{IU}$")
    y = self.g.derived["miller_geo"]["zeta_lo"].copy()
    ax.plot(x, y, ls="--", label="$\\zeta_{OL}$")
    y = self.g.derived["miller_geo"]["zeta_uo"].copy()
    ax.plot(x, y, ls="--", label="$\\zeta_{OU}$")
    ax.set_xlabel("$\\Psi_n$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("Squareness $\\zeta$")
    ax.legend()

    ax = axs[4]
    ax.text(
        0.0,
        11.0,
        "Rmajor = {0:.3f}m, Rmag = {1:.3f}m (Zmag = {2:.3f}m)".format(
            self.Rmajor, self.Rmag, self.Zmag
        ),
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        10.0,
        f"a = {self.a:.3f}m, eps = {self.eps:.3f}",
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        9.0,
        "kappa = {0:.3f} (kU = {1:.3f}, kL = {2:.3f})".format(
            self.kappa, self.kappaU, self.kappaL
        ),
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        8.0,
        f"    kappa95 = {self.kappa95:.3f},  kappa995 = {self.kappa995:.3f}",
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        7.0,
        f"    kappa_areal = {self.kappa_a:.3f}",
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        6.0,
        "delta = {0:.3f} (dU = {1:.3f}, dL = {2:.3f})".format(
            self.delta, self.deltaU, self.deltaL
        ),
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        5.0,
        f"    delta95 = {self.delta95:.3f},  delta995 = {self.delta995:.3f}",
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )
    ax.text(
        0.0,
        4.0,
        f"zeta = {self.zeta:.3f}",
        color="k",
        fontsize=10,
        fontweight="normal",
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=0,
    )

    ax.set_ylim([0, 12])
    ax.set_xlim([-1, 1])

    ax.set_axis_off()

def plotPlasma(self, axs=None, legendYN=False, color="r", label=""):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=7)

    ax_plasma = axs

    ax = ax_plasma[0]
    ax.plot(
        self.rho_tor,
        self.g.raw["pres"] * 1e-6,
        "-s",
        c=color,
        lw=2,
        markersize=3,
        label="geqdsk p",
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("pressure (MPa)")

    ax = ax_plasma[1]
    ax.plot(
        self.rho_tor,
        -self.g.raw["pprime"] * 1e-6,
        c=color,
        lw=2,
        ls="-",
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylabel("pressure gradient -p' (MPa/[])")
    ax.axhline(y=0.0, ls="--", lw=0.5, c="k")

    ax = ax_plasma[2]
    ax.plot(self.rho_tor, self.g.raw["fpol"], c=color, lw=2, ls="-")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylabel("$F = RB_{\\phi}$ (T*m)")

    ax = ax_plasma[3]
    ax.plot(self.rho_tor, self.g.raw["ffprim"], c=color, lw=2, ls="-")
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylabel("FF' (T*m/[])")
    ax.axhline(y=0.0, ls="--", lw=0.5, c="k")

    ax = ax_plasma[4]
    ax.plot(
        self.rho_tor,
        np.abs(self.g.raw["qpsi"]),
        "-s",
        c=color,
        lw=2,
        markersize=3,
        label=label + "geqdsk q",
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("safety factor q")
    ax.axhline(y=1.0, ls="--", lw=0.5, c="k")

    ax = ax_plasma[5]
    ax.plot(
        self.rho_tor,
        np.abs(self.Jt),
        "-s",
        c=color,
        lw=2,
        markersize=3,
        label=label + "geqdsk Jt",
    )
    ax.plot(
        self.rho_tor,
        np.abs(self.Jt_fb),
        "--o",
        c=color,
        lw=2,
        markersize=3,
        label=label + "geqdsk Jt(fb)",
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylabel("FSA toroidal current density ($MA/m^2$)")
    ax.axhline(y=0.0, ls="--", lw=0.5, c="k")

    if legendYN:
        ax.legend()

    #ax = ax_plasma[6]
    #ax.plot(
    #    self.g["fluxSurfaces"]["midplane"]["R"],
    #    np.abs(self.g["fluxSurfaces"]["midplane"]["Bt"]),
    #    "-s",
    #    c=color,
    #    lw=2,
    #    markersize=3,
    #    label=label + "geqdsk Bt",
    #)
    #ax.plot(
    #    self.g["fluxSurfaces"]["midplane"]["R"],
    #    np.abs(self.g["fluxSurfaces"]["midplane"]["Bp"]),
    #    "--o",
    #    c=color,
    #    lw=2,
    #    markersize=3,
    #    label=label + "geqdsk Bp",
    #)
    #ax.set_xlabel("R (m) midplane")
    #ax.set_ylabel("Midplane fields (abs())")

    if legendYN:
        ax.legend()

    return ax_plasma

def plotGeometry(self, axs=None, color="r"):
    if axs is None:
        plt.ion()
        fig, axs = plt.subplots(ncols=4)

    ax = axs[0]
    x = self.rho_tor
    y = self.cx_area
    ax.plot(
        x,
        y,
        "-",
        c=color,
        lw=2,
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("CX Area ($m^2$)")

    ax = axs[1]
    x = self.rho_tor
    y = np.zeros(x.shape)
    ax.plot(
        x, # self.rho_tor,
        y, # self.g["fluxSurfaces"]["geo"]["surfArea"],
        "-",
        c=color,
        lw=2,
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Surface Area ($m^2$)")

    ax = axs[2]
    x = self.rho_tor
    y = np.zeros(x.shape)
    ax.plot(
        x, # self.rho_tor,
        y, # self.g["fluxSurfaces"]["geo"]["vol"],
        "-",
        c=color,
        lw=2,
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Volume ($m^3$)")

def plotFluxSurfaces(
    self,
    ax=None,
    fluxes=[1.0],
    color="b",
    alpha=1.0,
    rhoPol=True,
    sqrt=False,
    lw=1,
    lwB=2,
    plot1=True,
    label = '',
):
    x = self.g.derived["R"]
    y = self.g.derived["Z"]

    if rhoPol:
        z = self.g.derived["rhorz_pol"]
    else:
        z = self.g.derived["rhorz_tor"]

    if not sqrt:
        z = z**2

    cs, csA = plotSurfaces(
        x, y, z, fluxes=fluxes, ax=ax, color=color, alpha=alpha, lw=lw, lwB=lwB, plot1=plot1, label = label
    )

    return cs, csA

def plotXpointEnvelope(
    self, ax=None, color="b", alpha=1.0, rhoPol=True, sqrt=False
):
    flx = 0.001
    fluxes = [1.0 - flx, 1.0 - flx / 2, 1.0, 1.0 + flx / 2, 1.0 + flx]

    self.plotFluxSurfaces(
        fluxes=fluxes, ax=ax, color=color, alpha=alpha, rhoPol=rhoPol, sqrt=sqrt
    )

def plot2Dquantity(
    self,
    ax=None,
    var="Jr",
    zlims=None,
    title="Radial Current",
    cmap="seismic",
    direct=False,
    titlebar="J ($MA/m^2$)",
    factor=1.0,
    includeSurfs=True,
):
    if ax is None:
        fig, ax = plt.subplots()

    x = self.g.derived["R"]
    y = self.g.derived["Z"]
    if not direct:
        z = self.g.derived[var] * factor
    else:
        z = var

    if zlims is None:
        am = np.amax(np.abs(z[:, :]))
        ming = -am
        maxg = am
    else:
        ming = zlims[0]
        maxg = zlims[1]
    levels = np.linspace(ming, maxg, 100)
    colticks = np.linspace(ming, maxg, 5)

    cs = ax.contourf(x, y, z, levels=levels, extend="both", cmap=cmap)

    cbar = GRAPHICStools.addColorbarSubplot(
        ax,
        cs,
        barfmt="%3.1f",
        title=titlebar,
        fontsize=10,
        fontsizeTitle=8,
        ylabel="",
        ticks=colticks,
        orientation="bottom",
        drawedges=False,
        padCB="25%",
    )

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")

    if includeSurfs:
        self.plotFluxSurfaces(
            ax=ax,
            fluxes=np.linspace(0, 1, 6),
            rhoPol=False,
            sqrt=True,
            color="k",
            lw=0.5,
        )

    return cs


def plotSurfaces(R, Z, F, fluxes=[1.0], ax=None, color="b", alpha=1.0, lw=1, lwB=2, plot1=True, label = ''):
    if ax is None:
        fig, ax = plt.subplots()

    [Rg, Yg] = np.meshgrid(R, Z)

    if plot1:
        csA = ax.contour(
            Rg, Yg, F, 1000, levels=[1.0], colors=color, alpha=alpha, linewidths=lwB
        )
    else:
        csA = None
    cs = ax.contour(
        Rg, Yg, F, 1000, levels=fluxes, colors=color, alpha=alpha, linewidths=lw, label = label
    )

    return cs, csA

