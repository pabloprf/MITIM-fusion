import math, netCDF4, os, glob, copy, pdb
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    pass
from IPython import embed
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.transp_tools.utils import toric_tools_JCW as toric_tools
from scipy.interpolate import griddata
from mitim_tools.misc_tools.IOtools import printMsg as print


def getTORICfromTRANSP(folderWork, nameRunid):
    print(f"\t- Gathering TORIC data")

    torics = []
    cdf_FI = None

    nameICRF = f"{folderWork}/TORIC_folder/{nameRunid}_ICRF_TAR.GZ1"

    print(f"\t\t- Looking for TORIC_folder/{nameRunid}_ICRF_TAR.GZ1 file...")
    if os.path.exists(nameICRF):
        folder = convertToReadable(
            nameICRF, folderWork=folderWork, checkExtension="ncdf"
        )

        # check maximum of 10 antennas
        for i in range(10):
            t = toricCDF(folder, antenna=f"A{i+1}")
            if t.simulation is not None:
                torics.append(t)
            else:
                break

        print(f"\t\t- TORIC file found, read {len(torics)} antennas")

    else:
        print(f"\t\t- TORIC file not found", typeMsg="w")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~ FI
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nameFI = f"{folderWork}/FI_folder/{nameRunid}_FI_TAR.GZ1"

    print(f"\t\t- Looking for FI_folder/{nameRunid}_FI_TAR.GZ1 file...")
    if os.path.exists(nameFI):
        folder_FI = convertToReadable(
            nameFI, folderWork=folderWork, checkExtension="cdf"
        )

        fileN_FI = folder_FI.replace("\\", "") + f"{nameRunid}_fpp_curstate.cdf"

        cdf_FI = netCDF4.Dataset(fileN_FI).variables

        print(f"\t\t\t- FI file found")

    else:
        print(f"\t\t\t- FI file not found", typeMsg="w")

    return torics, cdf_FI


class toricCDF:
    def __init__(self, folderWorkN, antenna="A1"):
        numTOR = IOtools.findFileByExtension(
            folderWorkN, "_toric.ncdf", ForceFirst=True
        )

        name, _, numTOR = numTOR.split("_")

        fileN = folderWorkN.replace("\\", "") + f"{name}_{antenna}_{numTOR}_toric.ncdf"
        fileN_msg = (
            folderWorkN.replace("\\", "") + f"{name}_{antenna}_{numTOR}_toric5.msgs"
        )

        self.simulation = None
        if os.path.exists(fileN):
            print(f"\t\t- Reading toric file {fileN[np.max([-40,-len(fileN)]):]}")
            self.simulation = toric_tools.toric_analysis(
                fileN, mode="ICRF", layout="paper"
            )

            self.cdf = netCDF4.Dataset(fileN).variables

            self.units_Power = self.simulation.cdf_hdl.variables["PwE"].units.decode(
                "UTF-8"
            )
            self.units_E = self.simulation.cdf_hdl.variables["ReEminus"].units.decode(
                "UTF-8"
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~ What species are there
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            self.spec, self.spec_pos, self.resF1, self.resH1 = findSpecies(fileN_msg)

            # Find those variables with a non-zero power absorbed, with the threshold, for both fundamental and harmonic
            self.thresh = 1e-8

            (
                self.specs_F,
                self.specs_H,
                self.specs_F_pos,
                self.specs_H_pos,
                self.resF,
                self.resH,
            ) = ([], [], [], [], [], [])
            for cont, i in enumerate(self.spec_pos):
                if (
                    np.sum(self.simulation.cdf_hdl.variables["PwIF"][:, i])
                    > self.thresh
                ):
                    self.specs_F.append(self.spec[cont])
                    self.specs_F_pos.append(i)
                    self.resF.append(self.resF1[cont])
                if (
                    np.sum(self.simulation.cdf_hdl.variables["PwIH"][:, i])
                    > self.thresh
                ):
                    self.specs_H.append(self.spec[cont])
                    self.specs_H_pos.append(i)
                    self.resH.append(self.resH1[cont])

    def plotPowerDensity_rho(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.simulation.plotpower(ax, power="PwE", label="e LD")
        for cont, i in enumerate(self.specs_F_pos):
            self.simulation.plotpower(
                ax, power="PwIF", species=i + 1, label=self.specs_F[cont] + " (Fund)"
            )
        for cont, i in enumerate(self.specs_H_pos):
            self.simulation.plotpower(
                ax, power="PwIH", species=i + 1, label=self.specs_F[cont] + " (Harm)"
            )

        ax.legend(loc="best")

        ax.set_ylabel("Power density (" + self.units_Power + ")")

    def plotPowerDensities_x(self, ax=None):
        colors = ["r", "b", "g", "m", "c", "orange", "y", "k"]

        if ax is None:
            fig, ax = plt.subplots()

        XX = self.simulation.cdf_hdl.variables["Xplasma"]
        ZZ = self.simulation.cdf_hdl.variables["Zplasma"]
        pwre = self.simulation.cdf_hdl.variables["TDPwE"]
        shp = pwre.shape

        X1D = self.simulation.cdf_hdl.variables["Ef_abscissa"][:]
        Z1D = np.zeros(len(X1D))

        grid_e = griddata(
            (XX[:, :].ravel(), ZZ[:, :].ravel()),
            pwre[:, :].ravel(),
            (X1D[None, :], Z1D[0, None]),
            method="nearest",
        )
        contcolor = 0
        ax.plot(
            X1D * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
            grid_e.T,
            label="e LD",
            ls="-",
            lw=1,
            alpha=0.5,
            c=colors[contcolor],
        )
        contcolor += 1

        for cont, i in enumerate(self.specs_F_pos):
            pwrHe = self.simulation.cdf_hdl.variables["TDPwIF"][:, :, i]
            grid_He = griddata(
                (XX[:, :].ravel(), ZZ[:, :].ravel()),
                pwrHe[:, :].ravel(),
                (X1D[None, :], Z1D[0, None]),
                method="nearest",
            )
            ax.plot(
                X1D * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                grid_He.T,
                label=self.specs_F[cont] + " (Fund)",
                c=colors[contcolor],
                ls="-",
                lw=3,
            )
            ax.axvline(
                x=self.resF[cont] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                ls="-",
                lw=3,
                alpha=0.5,
                c=colors[contcolor],
            )
            contcolor += 1

        for cont, i in enumerate(self.specs_H_pos):
            pwrHe = self.simulation.cdf_hdl.variables["TdPwIH"][:, :, i]
            grid_He = griddata(
                (XX[:, :].ravel(), ZZ[:, :].ravel()),
                pwrHe[:, :].ravel(),
                (X1D[None, :], Z1D[0, None]),
                method="nearest",
            )
            ax.plot(
                X1D * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                grid_He.T,
                label=self.specs_H[cont] + " (Harm)",
                c=colors[contcolor],
                ls="--",
                lw=3,
            )
            ax.axvline(
                x=self.resH[cont] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                ls="--",
                lw=3,
                alpha=0.5,
                c=colors[contcolor],
            )
            contcolor += 1

        ax.axvline(x=self.cdf["Raxis"][:] * 1e-2, ls="-.", lw=2, c="k")
        ax.set_ylabel("Power density (" + self.units_Power + ")")
        ax.set_xlabel("R (m)")
        ax.legend(loc="best")
        ax.set_ylim(bottom=0)

    def plotPowerDensity_x(
        self,
        ax=None,
        specie="He3 fast",
        harmonic=False,
        color="b",
        multY=1.0,
        plotres=True,
        fill=0.05,
        label=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()

        if multY < 1.0:
            extra = f" /{1/multY:.1f}"
        elif multY > 1.0:
            extra = f" *{multY:.1f}"
        else:
            extra = ""

        plt.rcParams["image.cmap"] = "inferno"

        XX = self.simulation.cdf_hdl.variables["Xplasma"]
        ZZ = self.simulation.cdf_hdl.variables["Zplasma"]
        pwre = self.simulation.cdf_hdl.variables["TDPwE"]
        shp = pwre.shape

        X1D = self.simulation.cdf_hdl.variables["Ef_abscissa"][:]
        Z1D = np.zeros(len(X1D))

        if specie != "e":
            if not harmonic:
                for cont in range(len(self.specs_F)):
                    if specie in self.specs_F[cont]:
                        break
                i = self.specs_F_pos[cont]

                pwrHe = self.simulation.cdf_hdl.variables["TDPwIF"][:, :, i]
                grid_He = griddata(
                    (XX[:, :].ravel(), ZZ[:, :].ravel()),
                    pwrHe[:, :].ravel(),
                    (X1D[None, :], Z1D[0, None]),
                    method="nearest",
                )
                x = X1D * 1e-2 + self.cdf["Raxis"][:] * 1e-2
                y = grid_He.T * multY
                if label is None:
                    label = self.specs_F[cont] + " (Fund)" + extra
                (l,) = ax.plot(x, y, label=label, ls="-", lw=3, color=color)
                if plotres:
                    ax.axvline(
                        x=self.resF[cont] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                        ls="-",
                        lw=5,
                        alpha=0.3,
                        color=color,
                    )

                _ = GRAPHICStools.fillGraph(
                    ax, x, y[:, 0], alpha=fill, color=color, label=""
                )

            else:
                for cont in range(len(self.specs_H)):
                    if specie in self.specs_H[cont]:
                        break
                i = self.specs_H_pos[cont]

                pwrHe = self.simulation.cdf_hdl.variables["TdPwIH"][:, :, i]
                grid_He = griddata(
                    (XX[:, :].ravel(), ZZ[:, :].ravel()),
                    pwrHe[:, :].ravel(),
                    (X1D[None, :], Z1D[0, None]),
                    method="nearest",
                )
                x = X1D * 1e-2 + self.cdf["Raxis"][:] * 1e-2
                y = grid_He.T * multY
                if label is None:
                    label = self.specs_H[cont] + " (Harm)" + extra
                (l,) = ax.plot(x, y, label=label, ls="--", lw=3, color=color)
                if plotres:
                    ax.axvline(
                        x=self.resH[cont] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                        ls="--",
                        lw=5,
                        alpha=0.3,
                        color=color,
                    )

                _ = GRAPHICStools.fillGraph(
                    ax, x, y[:, 0], alpha=fill, color=color, label=""
                )

        else:
            grid_e = griddata(
                (XX[:, :].ravel(), ZZ[:, :].ravel()),
                pwre[:, :].ravel(),
                (X1D[None, :], Z1D[0, None]),
                method="nearest",
            )
            x = X1D * 1e-2 + self.cdf["Raxis"][:] * 1e-2
            y = grid_e.T * multY
            if label is None:
                label = "e LD" + extra
            (l,) = ax.plot(x, y, label=label, ls="-", lw=3, color=color)

            _ = GRAPHICStools.fillGraph(
                ax, x, y[:, 0], alpha=fill, color=color, label=""
            )

        return l

    def plotElectricField_x(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.simulation.plot_1Dfield(
            ax,
            component="ReEminus",
            label="$Re(E_{minus})$",
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
        )
        self.simulation.plot_1Dfield(
            ax,
            component="ReEplus",
            label="$Re(E_{plus})$",
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
        )
        ax.axhline(y=0, lw=1, ls="--", c="k")

        ax.axvline(x=self.cdf["Raxis"][:] * 1e-2, ls="-.", lw=2, c="k")
        ax.set_xlabel("R (m)")
        ax.legend(loc="best")

    def plotElectricFields_2D(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(ncols=4, figsize=(15, 5))

        plt.rcParams["image.cmap"] = "jet"  #'inferno'

        self.simulation.plot_2Dfield(
            axs[0],
            "Re2Eplus",
            logl=20,
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
            title=f"Re2Eplus ({self.units_E})",
        )
        self.simulation.plot_2Dfield(
            axs[1],
            "Re2Eminus",
            logl=20,
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
            title=f"Re2Eminus ({self.units_E})",
        )
        self.simulation.plot_2Dfield(
            axs[2],
            "Im2Eplus",
            logl=20,
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
            title=f"Im2Eplus ({self.units_E})",
        )
        self.simulation.plot_2Dfield(
            axs[3],
            "Im2Eminus",
            logl=20,
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
            title=f"Im2Eminus ({self.units_E})",
        )

        for ax in axs:
            ax.plot(
                self.cdf["Xvessel"][:] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                self.cdf["Zvessel"][:] * 1e-2,
                lw=1,
                c="m",
            )

    def plotPowerDensities_2D(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(
                ncols=np.max([1 + len(self.specs_F), len(self.specs_H)]),
                nrows=2,
                figsize=(15, 7),
            )

        plt.rcParams["image.cmap"] = "inferno"

        self.simulation.plot_2Dfield(
            axs[0, 0],
            "TDPwE",
            logl=20,
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
            title="$P_{e,LD}$ (" + self.units_Power + ")",
        )
        for cont, i in enumerate(self.specs_F_pos):
            self.simulation.plot_2Dfield(
                axs[0, cont + 1],
                "TDPwIF",
                species=i + 1,
                logl=20,
                offsetX=self.cdf["Raxis"][:] * 1e-2,
                multX=1e-2,
                title=f"$P_{{{self.specs_F[cont]},F}}$ ({self.units_Power})",
            )
            axs[0, cont + 1].plot(
                self.cdf["Xvessel"][:] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                self.cdf["Zvessel"][:] * 1e-2,
                lw=1,
                c="m",
            )
        for cont, i in enumerate(self.specs_H_pos):
            self.simulation.plot_2Dfield(
                axs[1, cont],
                "TdPwIH",
                species=i + 1,
                logl=20,
                offsetX=self.cdf["Raxis"][:] * 1e-2,
                multX=1e-2,
                title=f"$P_{{{self.specs_H[cont]},H}}$ ({self.units_Power})",
            )
            axs[0, cont].plot(
                self.cdf["Xvessel"][:] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
                self.cdf["Zvessel"][:] * 1e-2,
                lw=1,
                c="m",
            )

    def plotPowerDensity_2D(self, ax=None, specie="He3 fast", harmonic=False):
        if ax is None:
            fig, ax = plt.subplots()

        plt.rcParams["image.cmap"] = "inferno"

        if specie != "e":
            if not harmonic:
                for cont in range(len(self.specs_F)):
                    if specie in self.specs_F[cont]:
                        break
                i = self.specs_F_pos[cont]
                self.simulation.plot_2Dfield(
                    ax,
                    "TDPwIF",
                    species=i + 1,
                    logl=20,
                    offsetX=self.cdf["Raxis"][:] * 1e-2,
                    multX=1e-2,
                    title=f"$P_{{{self.specs_F[cont]},F}}$ ({self.units_Power})",
                )
            else:
                for cont in range(len(self.specs_H)):
                    if specie in self.specs_H[cont]:
                        break
                i = self.specs_H_pos[cont]
                self.simulation.plot_2Dfield(
                    ax,
                    "TdPwIH",
                    species=i + 1,
                    logl=20,
                    offsetX=self.cdf["Raxis"][:] * 1e-2,
                    multX=1e-2,
                    title=f"$P_{{{self.specs_H[cont]},F}}$ ({self.units_Power})",
                )

        else:
            self.simulation.plot_2Dfield(
                ax,
                "TDPwE",
                logl=20,
                offsetX=self.cdf["Raxis"][:] * 1e-2,
                multX=1e-2,
                title="$P_{e,LD}$ (" + self.units_Power + ")",
            )

        ax.plot(
            self.cdf["Xvessel"][:] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
            self.cdf["Zvessel"][:] * 1e-2,
            lw=1,
            c="m",
        )

    def plotComplete(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        grid = plt.GridSpec(2, 5, hspace=0.2, wspace=0.4)

        ax1 = fig.add_subplot(grid[:, 0])
        plt.rcParams["image.cmap"] = "jet"  #'inferno'
        self.simulation.plot_2Dfield(
            ax1,
            "Re2Eplus",
            logl=20,
            offsetX=self.cdf["Raxis"][:] * 1e-2,
            multX=1e-2,
            title=f"Re2Eplus ({self.units_E})",
        )
        ax1.plot(
            self.cdf["Xvessel"][:] * 1e-2 + self.cdf["Raxis"][:] * 1e-2,
            self.cdf["Zvessel"][:] * 1e-2,
            lw=1,
            c="m",
        )

        # Plot minority absorption
        ax2 = fig.add_subplot(grid[:, 1])
        if "He3 fast" in self.specs_F:
            self.plotPowerDensity_2D(ax=ax2, specie="He3 fast", harmonic=False)
        elif "H fast" in self.specs_F:
            self.plotPowerDensity_2D(ax=ax2, specie="H fast", harmonic=False)
        elif "He3" in self.specs_F:
            self.plotPowerDensity_2D(ax=ax2, specie="He3", harmonic=False)
        elif "H" in self.specs_F:
            self.plotPowerDensity_2D(ax=ax2, specie="H", harmonic=False)
        # ------

        ax2e = fig.add_subplot(grid[:, 2])
        self.plotPowerDensity_2D(ax=ax2e, specie="e", harmonic=False)

        ax3 = fig.add_subplot(grid[0, 3:])
        self.plotPowerDensities_x(ax=ax3)
        GRAPHICStools.addDenseAxis(ax3)

        ax4 = fig.add_subplot(grid[1, 3:])
        self.plotElectricField_x(ax=ax4)
        GRAPHICStools.addDenseAxis(ax4)

        return ax1, ax2, ax2e, ax3, ax4


def findSpecies(toricmessage):
    with open(toricmessage, "r") as f:
        lines = f.readlines()

    contFlag = True
    i = 0
    spec = []
    while contFlag:
        if "Ion Species" in lines[i]:
            spec.append(
                IOtools.OrderedDict(
                    {
                        "Z": int(
                            float(
                                lines[i + 1].split("(atomic units)")[-1].split("\n")[0]
                            )
                        ),
                        "A": int(
                            float(
                                lines[i + 2].split("(atomic units)")[-1].split("\n")[0]
                            )
                        ),
                        "T": float(
                            lines[i + 5].split("=")[-1].split("\n")[0].split("keV")[0]
                        ),
                        "n": float(lines[i + 3].split("=")[-1].split("\n")[0]),
                        "F": float(
                            lines[i + 10].split("=")[-1].split("\n")[0].split("cm")[0]
                        ),
                        "H": float(
                            lines[i + 12].split("=")[-1].split("\n")[0].split("cm")[0]
                        ),
                    }
                )
            )

        if "PLASMA MODEL:" in lines[i]:
            contFlag = False
        i += 1

    spec_n, positions, resF, resH = [], [], [], []
    for i, s in enumerate(spec):
        name = whatIon(s["Z"], s["A"], s["T"], s["n"])
        if name is not None:
            spec_n.append(name)
            positions.append(i)
            resF.append(s["F"])
            resH.append(s["H"])

    return spec_n, positions, resF, resH


def whatIon(Z, A, T, n, thresholdT=50, thresholdn=1e5):
    if n > thresholdn:
        if Z == 1:
            if A == 1:
                st = "H"
            if A == 2:
                st = "D"
            if A == 3:
                st = "T"
        elif Z == 2:
            if A == 3:
                st = "He3"
            if A == 4:
                st = "He4"
        elif Z == 74:
            st = "W"
        elif Z == 42:
            st = "Mo"
        else:
            st = "TOK"

        if T > thresholdT:
            st = st + " fast"
    else:
        st = None

    try:
        return st
    except:
        embed()


def convertToReadable(tarfile, folderWork="~/scratch/", checkExtension="ncdf"):
    tarfile = IOtools.expandPath(tarfile)
    tarfileC = IOtools.expandPath(tarfile, fixSpaces=True)

    foldertar, _ = IOtools.getLocInfo(tarfile)
    foldertarC = IOtools.expandPath(foldertar, fixSpaces=True)

    try:
        ncdf_exists = IOtools.findFileByExtension(foldertar, checkExtension) is not None
    except:
        ncdf_exists = True

    if not ncdf_exists:
        print(
            f"\t\t- There is not a TORIC ncdf in {foldertar}, I need to extract the tar file first"
        )

        folderWork = IOtools.expandPath(folderWork)

        IOtools.askNewFolder(folderWork + "/toricAC/", force=True)

        os.system(f"cp {tarfileC} {folderWork}/toricAC/.")
        tarfile = tarfile.split("/")[-1]
        os.system(f"cd {folderWork}/toricAC/ && tar -xvf {tarfile}")

        os.system(f"cp {folderWork}/toricAC/* {foldertarC}/.")

        print(f"\t\t\t* Extracted! copy files from scratch workspace to {foldertar}")

    return foldertar
