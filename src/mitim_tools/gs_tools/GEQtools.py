import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, MATHtools
from IPython import embed

"""
Note that this module relies on OMFIT classes (https://omfit.io/classes.html) procedures to intrepret the content of g-eqdsk files.
Modifications are made for nice visualizations and a few extra derivations.
"""


class MITIMgeqdsk:
    def __init__(self, filename, fullLCFS=False, removeCoils=True):
        """
        Read g-eqdsk file using OMFIT classes (dynamic loading)

        Notes:
                I impose FindSeparatrix because I don't trust the g-file one
        """

        if removeCoils:
            print(
                f"\t- If geqdsk is appended with coils, removing them to read {filename}"
            )
            with open(filename, "r") as f:
                lines = f.readlines()
            with open(f"{filename}_noCoils.geqdsk", "w") as f:
                for cont,line in enumerate(lines):
                    if cont>0 and line[:2] == "  ":
                        break
                    f.write(line)
            filename = f"{filename}_noCoils.geqdsk"

        # -------------------------------------------------------------
        import omfit_classes.omfit_eqdsk

        self.g = omfit_classes.omfit_eqdsk.OMFITgeqdsk(
            filename, forceFindSeparatrix=True
        )
        # -------------------------------------------------------------

        # Extra derivations in MITIM
        self.derive(fullLCFS=fullLCFS)

        if removeCoils:
            os.system(f"rm {filename}")

    @classmethod
    def timeslices(cls, filename, **kwargs):
        print("\n...Opening GEQ file with several time slices")

        with open(filename, "rb") as f:
            lines_full = f.readlines()

        resolutions = [int(a) for a in lines_full[0].split()[-3:]]

        lines_files = []
        lines_files0 = []
        for i in range(len(lines_full)):
            line = lines_full[i]
            resols = []
            try:
                resols = [int(a) for a in line.split()[-3:]]
            except:
                pass
            if (resols == resolutions) and (i != 0):
                lines_files.append(lines_files0)
                lines_files0 = []
            lines_files0.append(line)
        lines_files.append(lines_files0)

        # Write files
        gs = []
        for i in range(len(lines_files)):
            with open(f"{filename}_time{i}.geqdsk", "wb") as f:
                f.writelines(lines_files[i])

            gs.append(
                cls(
                    f"{filename}_time{i}.geqdsk",**kwargs,
                )
            )
            os.system(f"rm {filename}_time{i}.geqdsk")

        return gs

    def derive(self, fullLCFS=False):
        self.Jt = self.g.surfAvg("Jt") * 1e-6
        self.Jt_fb = self.g.surfAvg("Jt_fb") * 1e-6

        self.Jerror = np.abs(self.Jt - self.Jt_fb)

        self.Ip = self.g["CURRENT"]

        # Parameterizations of LCFS
        self.kappa = self.g["fluxSurfaces"]["geo"]["kap"][-1]
        self.kappaU = self.g["fluxSurfaces"]["geo"]["kapu"][-1]
        self.kappaL = self.g["fluxSurfaces"]["geo"]["kapl"][-1]

        self.delta = self.g["fluxSurfaces"]["geo"]["delta"][-1]
        self.deltaU = self.g["fluxSurfaces"]["geo"]["dell"][-1]
        self.deltaL = self.g["fluxSurfaces"]["geo"]["dell"][-1]

        self.zeta = self.g["fluxSurfaces"]["geo"]["zeta"][-1]

        self.a = self.g["fluxSurfaces"]["geo"]["a"][-1]
        self.Rmag = self.g["fluxSurfaces"]["geo"]["R"][0]
        self.Zmag = self.g["fluxSurfaces"]["geo"]["Z"][0]
        self.Rmajor = np.mean(
            [
                self.g["fluxSurfaces"]["geo"]["Rmin_centroid"][-1],
                self.g["fluxSurfaces"]["geo"]["Rmax_centroid"][-1],
            ]
        )

        self.Zmajor = self.Zmag

        self.eps = self.a / self.Rmajor

        # Core values
        self.kappa995 = np.interp(
            0.995,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["kap"],
        )
        self.kappa95 = np.interp(
            0.95,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["kap"],
        )
        self.delta995 = np.interp(
            0.995,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["delta"],
        )
        self.delta95 = np.interp(
            0.95,
            self.g["AuxQuantities"]["PSI_NORM"],
            self.g["fluxSurfaces"]["geo"]["delta"],
        )

        # Boundary
        self.determineBoundary(full=fullLCFS)

    def determineBoundary(self, debug=False, full=False):
        """
        Note that the RBBS and ZBBS values in the gfile are often too scattered and do not reproduce the boundary near x-points.
        The shaping parameters calculated using fluxSurfaces are correct though.

        Since at __init__ I forced to find separatrix, I'm using OMFIT here
        """

        self.Rb_gfile, self.Yb_gfile = self.g["RBBBS"], self.g["ZBBBS"]
        self.Rb, self.Yb = self.g["fluxSurfaces"].sep.transpose()

        # Using PRF routines (otherwise IM workflow will run TRANSP with error of curvature... I have to fix it)
        if full:
            self.Rb_prf, self.Yb_prf = MATHtools.findBoundaryMath(
                self.g["AuxQuantities"]["R"],
                self.g["AuxQuantities"]["Z"],
                self.g["AuxQuantities"]["PSIRZ_NORM"],
                0.99999,
                5e3,
                500,
                None,
                False,
                0.1,
            )

        if debug:
            fig, ax = plt.subplots()

            # OMFIT
            ax.plot(self.Rb, self.Yb, "-s", c="r")

            # GFILE
            ax.plot(self.Rb_gfile, self.Yb_gfile, "-s", c="y")

            # Old routines
            if full:
                ax.plot(self.Rb_prf, self.Yb_prf, "-o", c="k")

            # Extras
            self.plotFluxSurfaces(
                ax=ax, fluxes=[0.99999, 1.0], rhoPol=True, sqrt=False, color="m"
            )
            self.plotXpointEnvelope(
                ax=ax, color="c", alpha=1.0, rhoPol=True, sqrt=False
            )
            self.plotEnclosingBox(ax=ax)

            plt.show()

    def plotEnclosingBox(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        plotEnclosed(
            self.Rmajor,
            self.a,
            self.Zmajor,
            self.kappaU,
            self.kappaL,
            self.deltaU,
            self.deltaL,
            ax=ax,
        )
        ax.plot([self.Rmajor, self.Rmag], [self.Zmajor, self.Zmag], "o", markersize=5)

    def paramsLCFS(self):
        rmajor, epsilon, kappa, delta, zeta, z0 = (
            self.Rmajor,
            self.eps,
            self.kappa,
            self.delta,
            self.zeta,
            self.Zmag,
        )

        return self.Rmajor, self.eps, self.kappa, self.delta, self.zeta, self.Zmag

    def getShapingRatios(self, name="", runRemote=False):
        kR = self.kappa995 / self.kappa
        dR = self.delta995 / self.delta

        return round(kR, 3), round(dR, 3)

    def defineMapping(self):
        psi = self.g["PSI_NORM"]
        rho = self.g["RHOVN"]

        return rho, psi

    def changeRZgrid(
        self,
        Rmin=0.5,
        Rmax=1.5,
        Zext=1.5,
        interpol="cubic",
        useOnlyOriginalPoints=False,
    ):
        """
        New grid will be R=[Rmin,Rmax] and Z=[-Zext,Zext]
        """

        # Old grid
        Rold = self.g["AuxQuantities"]["R"]
        Zold = self.g["AuxQuantities"]["Z"]
        PSIRZ = self.g["PSIRZ"]

        if useOnlyOriginalPoints:
            i1 = np.argmin(np.abs(Rold - Rmin))
            i2 = np.argmin(np.abs(Rold - Rmax))
            j1 = np.argmin(np.abs(Zold + Zext))
            j2 = np.argmin(np.abs(Zold - Zext))

            self.g["NW"] = i2 - i1
            Rmin = Rold[i1]
            Rmax = Rold[i2]
            self.g["NH"] = j2 - j1
            Zext = Zold[j2]

        # New grid
        self.g["RDIM"] = Rmax - Rmin
        self.g["RLEFT"] = Rmin
        self.g["ZDIM"] = Zext * 2

        Rnew = np.linspace(0, self.g["RDIM"], self.g["NW"]) + self.g["RLEFT"]
        Znew = (
            np.linspace(0, self.g["ZDIM"], self.g["NH"])
            - self.g["ZDIM"] / 2.0
            + self.g["ZMID"]
        )

        # Interpolate
        self.g["PSIRZ"] = MATHtools.interp2D(
            Rnew, Znew, Rold, Zold, PSIRZ, kind=interpol
        )

        # Re-load stuff
        self.g.addAuxQuantities()
        self.g.addFluxSurfaces(**self.g.OMFITproperties)

    def translateQuantityTo2D(self, rhoTor, z):
        return np.interp(self.g["AuxQuantities"]["RHORZ"], rhoTor, z)

    def write(self, filename=None):
        """
        If filename is None, use the original one
        """

        if filename is not None:
            self.g.filename = filename

        self.g.save()

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------------------------------------------------------------------

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

        self.plotCurrents(
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

        self.plotFields(
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

        self.plotChecks(axs=[ax1, ax1E, ax2, ax3, ax4, ax5])

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

        self.plotParameterization(axs=[ax1, ax2, ax3, ax4, ax5])

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
        ax.set_title("Poloidal Flux")
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")

        ax = axs[1]
        self.plotFluxSurfaces(
            ax=ax, fluxes=np.linspace(0, 1, 21), rhoPol=False, sqrt=True, color=color
        )
        ax.set_title("Sqrt Toroidal Flux")
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")

        ax = axs[2]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g["AuxQuantities"]["RHO"]
        ax.plot(x, y, lw=2, ls="-", c=color, label=label)
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.5)

        ax.set_xlabel("$\\Psi_n$ (PSI_NORM)")
        ax.set_ylabel("$\\sqrt{\\phi_n}$ (RHO)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax = axs[3]
        x = self.g["AuxQuantities"]["RHO"]
        y = self.g["AuxQuantities"]["RHOp"]
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
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Jr") * 1e-6
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_ylabel("FSA $\\langle J\\rangle$ ($MA/m^2$)")
        ax.set_xlim([0, 1])

        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[1]
        self.plot2Dquantity(
            ax=ax, var="Jr", title="Radial Current Jr", zlims=zlims, factor=1e-6
        )

        ax = axs[2]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Jz") * 1e-6
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[3]
        self.plot2Dquantity(
            ax=ax, var="Jz", title="Vertical Current Jz", zlims=zlims, factor=1e-6
        )

        ax = axs[4]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Jt") * 1e-6
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[5]
        self.plot2Dquantity(
            ax=ax, var="Jt", title="Toroidal Current Jt", zlims=zlims, factor=1e-6
        )

        ax = axs[6]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Jp") * 1e-6
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[7]
        self.plot2Dquantity(
            ax=ax, var="Jp", title="Poloidal Current Jp", zlims=zlims, factor=1e-6
        )

        ax = axs[8]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Jpar") * 1e-6
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[9]
        self.plot2Dquantity(
            ax=ax, var="Jpar", title="Parallel Current Jpar", zlims=zlims, factor=1e-6
        )

    def plotFields(self, axs=None, zlims_thr=[-1, 1]):
        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(ncols=10)

        ax = axs[0]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Br")
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_ylabel("FSA $\\langle B\\rangle$ ($T$)")
        ax.set_xlim([0, 1])

        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[1]
        self.plot2Dquantity(
            ax=ax, var="Br", title="Radial Field Br", zlims=zlims, titlebar="B ($T$)"
        )

        ax = axs[2]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Bz")
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[3]
        self.plot2Dquantity(
            ax=ax, var="Bz", title="Vertical Field Bz", zlims=zlims, titlebar="B ($T$)"
        )

        ax = axs[4]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Bt")
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [y.min(), y.max()]
        # zlims = [np.min([zlims_thr[0],y.min()]),np.max([zlims_thr[1],y.max()])]
        # zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[5]
        self.plot2Dquantity(
            ax=ax, var="Jt", title="Toroidal Field Bt", zlims=zlims, titlebar="B ($T$)"
        )

        ax = axs[6]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g.surfAvg("Bp")
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        zlims = [np.min([zlims_thr[0], y.min()]), np.max([zlims_thr[1], y.max()])]
        zlims = GRAPHICStools.aroundZeroLims(zlims)
        ax.set_ylim(zlims)

        ax = axs[7]
        self.plot2Dquantity(
            ax=ax, var="Bp", title="Poloidal Field Bp", zlims=zlims, titlebar="B ($T$)"
        )

        ax = axs[8]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g["fluxSurfaces"]["avg"]["Bp**2"]
        ax.plot(x, y, lw=2, ls="-", c="r")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("$\\langle B_{\\theta}^2\\rangle$")

        ax = axs[9]
        x = self.g["fluxSurfaces"]["midplane"]["R"]
        y = self.g["fluxSurfaces"]["midplane"]["Bt"]
        ax.plot(x, y, lw=2, ls="-", c="r", label="$B_{t}$")
        y = self.g["fluxSurfaces"]["midplane"]["Bp"]
        ax.plot(x, y, lw=2, ls="-", c="b", label="$B_{p}$")
        y = self.g["fluxSurfaces"]["midplane"]["Bz"]
        ax.plot(x, y, lw=2, ls="-", c="g", label="$B_{z}$")
        y = self.g["fluxSurfaces"]["midplane"]["Br"]
        ax.plot(x, y, lw=2, ls="-", c="m", label="$B_{r}$")
        y = self.g["fluxSurfaces"]["geo"]["bunit"]
        ax.plot(x, y, lw=2, ls="-", c="c", label="$B_{unit}$")
        ax.set_xlabel("$R$ LF midplane")
        ax.set_ylabel("$B$ (T)")
        ax.legend()

    def plotChecks(self, axs=None):
        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(ncols=8)

        ax = axs[0]
        x = self.g["AuxQuantities"]["PSI_NORM"]
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
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y1 = self.g["FFPRIM"]
        ax.plot(x, y1, lw=2, ls="-", c="r", label="$FF'$")
        y2 = self.g["PPRIME"] * (4 * np.pi * 1e-7)
        ax.plot(x, y2, lw=2, ls="-", c="b", label="$p'*\\mu_0$")

        ax.set_ylabel("")
        ax.legend()
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])

        ax = axs[3]
        self.plot2Dquantity(
            ax=ax,
            var="Jt",
            title="Toroidal Current Jt",
            zlims=[zmin, zmax],
            cmap="viridis",
            factor=1e-6,
        )

        ax = axs[4]
        self.plot2Dquantity(
            ax=ax,
            var="Jt_fb",
            title="Toroidal Current Jt (FB)",
            zlims=[zmin, zmax],
            cmap="viridis",
            factor=1e-6,
        )

        ax = axs[5]
        z = (
            np.abs(self.g["AuxQuantities"]["Jt"] - self.g["AuxQuantities"]["Jt_fb"])
            * 1e-6
        )
        zmaxx = np.max([np.abs(zmax), np.abs(zmin)])
        self.plot2Dquantity(
            ax=ax,
            var=z,
            title="Absolute Error",
            zlims=[0, zmaxx],
            cmap="viridis",
            direct=True,
        )

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
        ax.plot(self.g["RMAXIS"], self.g["ZMAXIS"], "+", markersize=10, c="r")
        ax.plot([self.Rmag], [self.Zmag], "o", markersize=5, c="m")
        ax.plot([self.Rmajor], [self.Zmag], "+", markersize=10, c="k")
        ax.plot(self.g["RLIM"], self.g["ZLIM"], lw=1, c="k")

        import matplotlib

        path = matplotlib.path.Path(
            np.transpose(np.array([self.g["RLIM"], self.g["ZLIM"]]))
        )
        patch = matplotlib.patches.PathPatch(path, facecolor="none")
        ax.add_patch(patch)
        for col in cs.collections:
            col.set_clip_path(patch)
        for col in csA.collections:
            col.set_clip_path(patch)

        self.plotEnclosingBox(ax=ax)

        ax.set_aspect("equal")
        ax.set_title("Poloidal Flux")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")

        ax = axs[1]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g["fluxSurfaces"]["geo"]["kap"]
        ax.plot(x, y, label="$\\kappa$")
        y = self.g["fluxSurfaces"]["geo"]["kapl"]
        ax.plot(x, y, ls="--", label="$\\kappa_L$")
        y = self.g["fluxSurfaces"]["geo"]["kapu"]
        ax.plot(x, y, ls="--", label="$\\kappa_U$")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Elongation $\\kappa$")
        ax.legend()

        ax = axs[2]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g["fluxSurfaces"]["geo"]["delta"]
        ax.plot(x, y, label="$\\delta$")
        y = self.g["fluxSurfaces"]["geo"]["dell"]
        ax.plot(x, y, ls="--", label="$\\delta_L$")
        y = self.g["fluxSurfaces"]["geo"]["delu"]
        ax.plot(x, y, ls="--", label="$\\delta_U$")
        ax.set_xlabel("$\\Psi_n$")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Triangularity $\\delta$")
        ax.legend()

        ax = axs[3]
        x = self.g["AuxQuantities"]["PSI_NORM"]
        y = self.g["fluxSurfaces"]["geo"]["zeta"]
        ax.plot(x, y, label="$\\zeta$")
        y = self.g["fluxSurfaces"]["geo"]["zetail"]
        ax.plot(x, y, ls="--", label="$\\zeta_{IL}$")
        y = self.g["fluxSurfaces"]["geo"]["zetaiu"]
        ax.plot(x, y, ls="--", label="$\\zeta_{IU}$")
        y = self.g["fluxSurfaces"]["geo"]["zetaol"]
        ax.plot(x, y, ls="--", label="$\\zeta_{OL}$")
        y = self.g["fluxSurfaces"]["geo"]["zetaou"]
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
            6.0,
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
            5.0,
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
            self.g["AuxQuantities"]["RHO"],
            self.g["PRES"] * 1e-6,
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
            self.g["AuxQuantities"]["RHO"],
            -self.g["PPRIME"] * 1e-6,
            c=color,
            lw=2,
            ls="-",
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
        ax.set_ylabel("pressure gradient -p' (MPa/[])")
        ax.axhline(y=0.0, ls="--", lw=0.5, c="k")

        ax = ax_plasma[2]
        ax.plot(self.g["AuxQuantities"]["RHO"], self.g["FPOL"], c=color, lw=2, ls="-")
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
        ax.set_ylabel("$F = RB_{\\phi}$ (T*m)")

        ax = ax_plasma[3]
        ax.plot(self.g["AuxQuantities"]["RHO"], self.g["FFPRIM"], c=color, lw=2, ls="-")
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
        ax.set_ylabel("FF' (T*m/[])")
        ax.axhline(y=0.0, ls="--", lw=0.5, c="k")

        ax = ax_plasma[4]
        ax.plot(
            self.g["AuxQuantities"]["RHO"],
            np.abs(self.g["QPSI"]),
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
            self.g["AuxQuantities"]["RHO"],
            np.abs(self.g.surfAvg("Jt") * 1e-6),
            "-s",
            c=color,
            lw=2,
            markersize=3,
            label=label + "geqdsk Jt",
        )
        ax.plot(
            self.g["AuxQuantities"]["RHO"],
            np.abs(self.g.surfAvg("Jt_fb") * 1e-6),
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

        ax = ax_plasma[6]
        ax.plot(
            self.g["fluxSurfaces"]["midplane"]["R"],
            np.abs(self.g["fluxSurfaces"]["midplane"]["Bt"]),
            "-s",
            c=color,
            lw=2,
            markersize=3,
            label=label + "geqdsk Bt",
        )
        ax.plot(
            self.g["fluxSurfaces"]["midplane"]["R"],
            np.abs(self.g["fluxSurfaces"]["midplane"]["Bp"]),
            "--o",
            c=color,
            lw=2,
            markersize=3,
            label=label + "geqdsk Bp",
        )
        ax.set_xlabel("R (m) midplane")
        ax.set_ylabel("Midplane fields (abs())")

        if legendYN:
            ax.legend()

        return ax_plasma

    def plotGeometry(self, axs=None, color="r"):
        if axs is None:
            plt.ion()
            fig, axs = plt.subplots(ncols=4)

        ax = axs[0]
        ax.plot(
            self.g["AuxQuantities"]["RHO"],
            self.g["fluxSurfaces"]["geo"]["cxArea"],
            "-",
            c=color,
            lw=2,
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("CX Area ($m^2$)")

        ax = axs[1]
        ax.plot(
            self.g["AuxQuantities"]["RHO"],
            self.g["fluxSurfaces"]["geo"]["surfArea"],
            "-",
            c=color,
            lw=2,
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\sqrt{\\phi_n}$ (RHO)")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Surface Area ($m^2$)")

        ax = axs[2]
        ax.plot(
            self.g["AuxQuantities"]["RHO"],
            self.g["fluxSurfaces"]["geo"]["vol"],
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
        plot1=True,
    ):
        x = self.g["AuxQuantities"]["R"]
        y = self.g["AuxQuantities"]["Z"]

        if rhoPol:
            z = self.g["AuxQuantities"]["RHOpRZ"]
        else:
            z = self.g["AuxQuantities"]["RHORZ"]

        if not sqrt:
            z = z**2

        cs, csA = plotSurfaces(
            x, y, z, fluxes=fluxes, ax=ax, color=color, alpha=alpha, lw=lw, plot1=plot1
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

        x = self.g["AuxQuantities"]["R"]
        y = self.g["AuxQuantities"]["Z"]
        if not direct:
            z = self.g["AuxQuantities"][var] * factor
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
    
    def get_MXH_coeff(self, n, n_coeff=6, plot=False): 
        """
        Calculates MXH Coefficients as a function of poloidal flux
        n: number of grid points to interpolate the flux surfaces
           NOT the number of radial points returned. This function
           will return n_coeff sin and cosine coefficients for each
           flux surface in the geqdsk file. Changing n is only nessary
           if the last closed flux surface is not well resolved.
        """
        from scipy.signal import savgol_filter
        start=time()

        # Upsample the poloidal flux grid
        Raux, Zaux = self.g["AuxQuantities"]["R"], self.g["AuxQuantities"]["Z"]
        R = np.linspace(np.min(Raux),np.max(Raux),n)
        Z = np.linspace(np.min(Zaux),np.max(Zaux),n)
        Psi_norm = MATHtools.interp2D(R, Z, Raux, Zaux, self.g["AuxQuantities"]["PSIRZ_NORM"])

        # calculate the LCFS boundary
        # Select only the R and Z values within the LCFS- I psi_norm outside to 2
        # this works fine for fixed-boundary equilibria but needs v. high tolerance
        # for the full equilibrium with x-point.
        R_max_ind = np.argmin(np.abs(R-np.max(self.Rb)))+1 # extra index for tolerance
        Z_max_ind = np.argmin(np.abs(Z-np.max(self.Yb)))+1
        R_min_ind = np.argmin(np.abs(R-np.min(self.Rb)))-1
        Z_min_ind = np.argmin(np.abs(Z-np.min(self.Yb)))-1
        Psi_norm[:,:R_min_ind] = 2
        Psi_norm[:,R_max_ind:] = 2
        Psi_norm[:Z_min_ind,:] = 2
        Psi_norm[Z_max_ind:,:] = 2

        #psis = np.linspace(0.001,0.9999,self.g["AuxQuantities"]["PSI_NORM"].size)
        #psi_reg = self.g["AuxQuantities"]["PSI"]
        rhos = self.g["AuxQuantities"]["RHO"]
        psis = self.g["AuxQuantities"]["PSI_NORM"]
        
        cn, sn, gn = np.zeros((n_coeff,psis.size)), np.zeros((n_coeff,psis.size)), np.zeros((4,psis.size))
        print(f" \t\t--> Finding g-file flux-surfaces")
        for i, psi in enumerate(psis):
            
            if psi == 0:
                psi+=0.0001

            # need to construct level contours for each flux surface 
            Ri, Zi = MATHtools.drawContours(
                R,
                Z,
                Psi_norm,
                n,
                psi,
            )
            Ri, Zi = Ri[0], Zi[0]
                
            # interpolate R,Z contours to have the same dimensions
            Ri = np.interp(np.linspace(0,1,n),np.linspace(0,1,Ri.size),Ri)
            Zi = np.interp(np.linspace(0,1,n),np.linspace(0,1,Zi.size),Zi)
    
            #calculate Miller Extended Harmionic coefficients
            #enforce zero at the innermost flux surface
            
            cn[:,i], sn[:,i], gn[:,i] = get_flux_surface_geometry(Ri, Zi, n_coeff)
            if i == 0:
                cn[:,i]*=0 ; sn[:,i] *=0 # set shaping parameters zero for innermost flux surface near zero

        end=time()

        print(f'\ntotal run time: {end-start} s')
        if plot:
            fig, axes = plt.subplots(2,1)
            for i in np.arange(n_coeff):
                axes[0].plot(psis,cn[i,:],label=f"$c_{i}$")
                axes[1].plot(psis,sn[i,:],label=f"$s_{i}$")
            axes[0].legend() ; axes[1].legend()
            axes[0].set_xlabel("$\\Psi_N$") ; axes[1].set_xlabel("$\\Psi_N$")
            axes[0].grid() ; axes[1].grid()
            axes[0].set_title("MXH Coefficients - Cosine")
            axes[1].set_title("MXH Coefficients - Sine")
            plt.tight_layout()
            plt.show()
        print("Interpolated delta995:", np.interp(0.995,psis, sn[1,:]))
        return cn, sn, gn, psis
        
def get_flux_surface_geometry(R, Z, n_coeff=3):
    """
    Calculates MXH Coefficients for a flux surface
    """
    Z = np.roll(Z, -np.argmax(R))
    R = np.roll(R, -np.argmax(R))
    if Z[1] < Z[0]: # reverses array so that theta increases
        Z = np.flip(Z)
        R = np.flip(R)

    # compute bounding box for each flux surface
    r = 0.5*(np.max(R)-np.min(R))
    kappa = 0.5*(np.max(Z) - np.min(Z))/r
    R0 = 0.5*(np.max(R)+np.min(R))
    Z0 = 0.5*(np.max(Z)+np.min(Z))
    bbox = [R0, r, Z0, kappa]

    # solve for polar angles
    # need to use np.clip to avoid floating-point precision errors
    theta_r = np.arccos(np.clip(((R - R0) / r), -1, 1))
    theta = np.arcsin(np.clip(((Z - Z0) / r / kappa),-1,1))

    # Find the continuation of theta and theta_r to [0,2pi]
    theta_r_cont = np.copy(theta_r) ; theta_cont = np.copy(theta)

    max_theta = np.argmax(theta) ; min_theta = np.argmin(theta)
    max_theta_r = np.argmax(theta_r) ; min_theta_r = np.argmin(theta_r)

    theta_cont[:max_theta] = theta_cont[:max_theta]
    theta_cont[max_theta:max_theta_r] = np.pi-theta[max_theta:max_theta_r]
    theta_cont[max_theta_r:min_theta] = np.pi-theta[max_theta_r:min_theta]
    theta_cont[min_theta:] = 2*np.pi+theta[min_theta:]

    theta_r_cont[:max_theta] = theta_r_cont[:max_theta]
    theta_r_cont[max_theta:max_theta_r] = theta_r[max_theta:max_theta_r]
    theta_r_cont[max_theta_r:min_theta] = 2*np.pi - theta_r[max_theta_r:min_theta]
    theta_r_cont[min_theta:] = 2*np.pi - theta_r[min_theta:]

    theta_r_cont = theta_r_cont - theta_cont ; theta_r_cont[-1] = theta_r_cont[0]
    
    # fourier decompose to find coefficients
    c = np.zeros(n_coeff)
    s = np.zeros(n_coeff)
    f_theta_r = lambda theta: np.interp(theta, theta_cont, theta_r_cont)
    from scipy.integrate import quad
    for i in np.arange(n_coeff):
        integrand_sin = lambda theta: np.sin(i*theta)*(f_theta_r(theta))
        integrand_cos = lambda theta: np.cos(i*theta)*(f_theta_r(theta))

        s[i] = quad(integrand_sin,0,2*np.pi)[0]/np.pi
        c[i] = quad(integrand_cos,0,2*np.pi)[0]/np.pi
        
        
    return c, s, bbox




def plotSurfaces(
    R, Z, F, fluxes=[1.0], ax=None, color="b", alpha=1.0, lw=1, plot1=True
):
    if ax is None:
        fig, ax = plt.subplots()

    [Rg, Yg] = np.meshgrid(R, Z)

    if plot1:
        csA = ax.contour(
            Rg, Yg, F, 1000, levels=[1.0], colors=color, alpha=alpha, linewidths=lw * 2
        )
    else:
        csA = None
    cs = ax.contour(
        Rg, Yg, F, 1000, levels=fluxes, colors=color, alpha=alpha, linewidths=lw
    )

    return cs, csA


def compareGeqdsk(geqdsks, fn=None, extraLabel="", plotAll=True, labelsGs=None):
    if fn is None:
        wasProvided = False

        from mitim_tools.misc_tools.GUItools import FigureNotebook

        fn = FigureNotebook("GEQDSK Notebook", geometry="1600x1000")
    else:
        wasProvided = True

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


def plotEnclosed(Rmajor, a, Zmajor, kappaU, kappaL, deltaU, deltaL, ax=None, c="k"):
    if ax is None:
        fig, ax = plt.subplots()

    ax.axhline(y=Zmajor, ls="--", c=c, lw=0.5)
    ax.axvline(x=Rmajor, ls="--", c=c, lw=0.5)
    ax.axvline(x=Rmajor - a, ls="--", c=c, lw=0.5)
    ax.axvline(x=Rmajor + a, ls="--", c=c, lw=0.5)
    Rtop = Zmajor + a * kappaU
    ax.axhline(y=Rtop, ls="--", c=c, lw=0.5)
    Rbot = Zmajor - a * kappaL
    ax.axhline(y=Rbot, ls="--", c=c, lw=0.5)

    ax.axvline(x=Rmajor - a * deltaU, ls="--", c=c, lw=0.5)
    ax.axvline(x=Rmajor - a * deltaL, ls="--", c=c, lw=0.5)


def create_geo_MXH3(
    Rmaj, rmin, zmag, kappa, delta, zeta, shape_cos, shape_sin, debugPlot=False
):
    """
    R and Z outputs have (dim_flux_surface,dim_theta)
    """

    theta = np.linspace(0, 2 * np.pi, 100)

    # Organize cos/sin
    shape_cos0 = shape_cos[0]
    shape_cos_n = []
    shape_sin_n = [np.arcsin(delta), -zeta]
    for i in range(len(shape_cos) - 1):
        shape_cos_n.append(shape_cos[i + 1])
        if i > 1:
            shape_sin_n.append(shape_sin[i + 1])
    shape_cos_n = np.array(shape_cos_n)
    shape_sin_n = np.array(shape_sin_n)

    R, Z = [], []
    for ir in range(Rmaj.shape[0]):
        c_0 = shape_cos0[ir]
        c_n = shape_cos_n[:, ir]
        s_n = shape_sin_n[:, ir]

        theta_R = []
        for i in range(len(theta)):
            s = theta[i] + c_0
            for m in range(len(c_n)):
                s += c_n[m] * np.cos((m + 1) * theta[i]) + s_n[m] * np.sin(
                    (m + 1) * theta[i]
                )
            theta_R.append(s)
        theta_R = np.array(theta_R)

        R_0 = Rmaj[ir] + rmin[ir] * np.cos(theta_R)
        Z_0 = zmag[ir] + kappa[ir] * rmin[ir] * np.sin(theta)

        R.append(R_0)
        Z.append(Z_0)

    R, Z = np.array(R), np.array(Z)

    if debugPlot:
        fig, ax = plt.subplots()
        for ir in range(R.shape[0]):
            ax.plot(R[ir, :], Z[ir, :])
        plt.show()

    return R, Z
