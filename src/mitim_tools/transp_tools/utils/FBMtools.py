import netCDF4, os, scipy.interpolate
import numpy as np
from mitim_tools.misc_tools import IOtools, FARMINGtools, GRAPHICStools
from mitim_tools.misc_tools import CONFIGread

try:
    import matplotlib.pyplot as plt
except:
    pass

from IPython import embed

from mitim_tools.misc_tools.LOGtools import printMsg as print


class fbmCDF:
    def __init__(
        self, file, particle="He4_FUSN", guidingCenter=True, ACalreadyConverted=False
    ):
        print(
            f"\t\t- Gathering FBM data for {particle}. Guiding Center: {guidingCenter}"
        )

        file = IOtools.expandPath(file)
        self.guidingCenter = guidingCenter
        self.particle = particle

        if ACalreadyConverted:
            self.cdf = netCDF4.Dataset(file)
        else:
            if guidingCenter:
                extralab = "_GC"
            else:
                extralab = "_PO"
            self.cdf = convertACtoCDF(
                file, guidingCenter=self.guidingCenter, extralab=extralab
            )

        try:
            self.time = float(self.cdf["TIME"][:][0])
        except:
            self.time = float(self.cdf["TIME"][:].data)

        try:
            self.avtime = float(self.cdf["DT_AVG"][:][0])
        except:
            self.avtime = float(self.cdf["DT_AVG"][:].data)

        # Distribution function, f(L,energy,pitch)
        self.f = (
            0.5 * self.cdf["F_" + self.particle][:] * (1e6 * 1e-20) * (1e3)
        )  # m^-3 * keV^-1 / d(omega/4pi)

        # Coordinates

        self.energy = self.cdf["E_" + self.particle][:] * 1e-3  # keV
        self.pitch = self.cdf["A_" + self.particle][:]  # vpar/v

        """
		Note: The spatial index label of the distribution function has an-artibitrary structure: It moves around radially
		and poloidally. More resolution around the axis.
		"""

        self.r2d = self.cdf["R2D"][:] * 1e-2  # m
        self.z2d = self.cdf["Z2D"][:] * 1e-2  # m
        self.theta = self.cdf["TH2D"][:]

        self.rho = self.cdf["X2D"][:]

        self.processInformation(self.particle)

        # Get birth
        self.birth = None
        birthfile = None

        print("\t- Looking for birth data...")
        if (file.parent / f'{file.name.split("_fi_")[0] + "_birth.cdf1"}').exists():
            birthfile = file.parent / file.name.split("_fi_")[0] + "_birth.cdf1"
        elif (file.parent / f'{file.split(".DATA")[0] + "_birth.cdf1"}').exists():
            birthfile = file.parent / file.name.split(".DATA")[0] + "_birth.cdf1"
        else:
            print("\t\t- NUBEAM birth cdf file could not be found", typeMsg="w")

        if birthfile is not None and self.particle == "D_NBI":
            self.birth = birthCDF(birthfile, particle="D_MCBEAM")

    def processInformation(self, particle):
        # Bins
        self.denergy = np.diff(self.cdf["EB_" + particle][:]) * 1e-3  # keV
        self.dpitch = np.diff(self.cdf["AB_" + particle][:])  # vpar/v
        self.dpitchdenergy = np.outer(self.dpitch, self.denergy)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Total density (m^-3), sum over energies and pitch
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.nFast = np.tensordot(self.f, self.dpitchdenergy, axes=((1, 2), (0, 1)))

        # Interpolate to a regular R,Z, grid

        self.R = np.linspace(np.min(self.r2d), np.max(self.r2d), 51)
        self.Z = np.linspace(np.min(self.z2d), np.max(self.z2d), 51)
        self.nFast_rz = scipy.interpolate.griddata(
            (self.r2d, self.z2d),
            self.nFast,
            (self.R[None, :], self.Z[:, None]),
            method="cubic",
        )

        # Calculate rho surfaces
        self.rhoU = np.unique(self.rho)

        r = np.append(self.rhoU, [self.rhoU[-1] + (self.rhoU[-1] - self.rhoU[-2])])
        rn = []
        for i in range(len(r) - 1):
            rn.append(np.mean([r[i], r[i + 1]]))
        self.rhoUsurf = np.array(rn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~    PLOTTING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def plotTotal_rz(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots()

        cs = ax.contourf(self.R, self.Z, self.nFast_rz, 201)  # len(self.rhoU)+1)
        # ax.contour(self.R,self.Z,self.nFast_rz,len(self.rhoU),colors='black')
        ax.set_aspect("equal")

        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("$n(R,Z) = \\int f dEd\\xi$")

        GRAPHICStools.addColorbarSubplot(
            ax,
            cs,
            fig=fig,
            barfmt="%.1e",
            title="",
            fontsize=10,
            fontsizeTitle=None,
            ylabel="Density $[10^{20}m^{-3}]$",
            ticks=[],
            orientation="vertical",
            drawedges=False,
            force_position=None,
            padCB="10%",
            sizeC="3%",
        )

    def plotF_ep(self, ax=None, R=None, Z=None, fig=None):  # in meters
        if R is None:
            R = self.r2d[0]
        if Z is None:
            Z = self.z2d[0]

        if ax is None:
            fig, ax = plt.subplots()

        idx = np.argmin(np.abs(self.r2d - R) + np.abs(self.z2d - Z))

        maxEnergy = self.energy.max()

        cs = ax.contourf(self.energy, self.pitch, self.f[idx, :, :], 201)
        ax.set_ylim(-1, 1)
        ax.set_xlim([0, maxEnergy])
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Pitch  $\\xi=v_{\\parallel}/v$")
        ax.set_title(f"$f(E,\\xi)$ @ R={self.r2d[idx]:0.3f}m, Z={self.z2d[idx]:0.3f}m")

        GRAPHICStools.addColorbarSubplot(
            ax,
            cs,
            fig=fig,
            barfmt="%.1e",
            title="",
            fontsize=10,
            fontsizeTitle=None,
            ylabel="Density $[10^{20}m^{-3}]$",
            ticks=[],
            orientation="vertical",
            drawedges=False,
            force_position=None,
            padCB="10%",
            sizeC="3%",
        )

        return idx

    def plotTotal_rho(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots()

        cs = ax.scatter(
            self.rho,
            self.nFast,
            50,
            c=self.theta,
            alpha=0.4,
            vmin=-np.pi,
            vmax=np.pi,
            label="FBM",
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, np.max(self.nFast) * 1.2])
        ax.set_xlabel("$\\rho$")
        # ax.legend(loc='best')
        ax.set_ylabel("Density $[10^{20}m^{-3}]$")
        ax.set_title("$n(\\rho_N,\\theta) = \\int f dEd\\xi$")

        GRAPHICStools.addColorbarSubplot(
            ax,
            cs,
            fig=fig,
            barfmt="%.1e",
            title="",
            fontsize=10,
            fontsizeTitle=None,
            ylabel="$\\theta$",
            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            orientation="vertical",
            drawedges=False,
            force_position=None,
            padCB="10%",
            sizeC="3%",
            ticklabels=["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$-\\pi$"],
        )

        GRAPHICStools.addDenseAxis(ax)

    def plotComplete(self, R=[None], Z=[None], fig=None):
        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        grid = plt.GridSpec(2, 1 + len(R), hspace=0.3, wspace=0.4)

        plt.rcParams["image.cmap"] = "inferno"

        ax1 = fig.add_subplot(grid[:, 0])
        self.plotTotal_rz(ax=ax1, fig=fig)

        colors = ["r", "m", "b"]

        axs2 = []
        for i in range(len(R)):
            # print(f' --> Plotting FBM at R = {R[i]:.2f}m and Z = {Z[i]:.2f}m')
            ax2 = fig.add_subplot(grid[0, i + 1])
            idx = self.plotF_ep(ax=ax2, R=R[i], Z=Z[i], fig=fig)
            axs2.append(ax2)
            ax1.plot(self.r2d[idx], self.z2d[idx], marker="o", color="c", markersize=10)

        ax3 = fig.add_subplot(grid[1, 1])
        self.plotTotal_rho(ax=ax3, fig=fig)

        if self.guidingCenter:
            typ = '"guiding center"'
        else:
            typ = '"particle position"'
        fig.suptitle(
            f"{self.particle} {typ} Distribution function, extracted at time = {self.time:.3f}-{self.time+self.avtime:.3f} s"
        )

        ax4 = fig.add_subplot(grid[1, 2])
        rhos_plotted = []
        for rho, c in zip([0.0, 0.25, 0.5, 0.75], ["r", "m", "b", "c"]):
            ix = np.argmin(np.abs(self.rho - rho))
            ax4.plot(
                [self.time + self.avtime / 2],
                [self.nFast[ix]],
                "s",
                color=c,
                label=f"FBM, $\\rho_N$={rho:.2f}",
            )
            rhos_plotted.append(self.rho[ix])

        return rhos_plotted, ax1, axs2, ax3, ax4


def convertACtoCDF(file, guidingCenter=True, copyWhereOriginal=True, extralab="_GC"):
    """
    This converts the .DATA1 file into the _fi_1.cdf
    """

    file = IOtools.expandPath(file)
    folderOrig, fileOrig = IOtools.getLocInfo(file, with_extension=True)

    runid = fileOrig.split(".DATA")[0]
    num = fileOrig.split(".DATA")[1]

    finFile = f"{runid}_fi_{num}.cdf"
    typeF = "c" if guidingCenter else "p"

    commandMain = f"get_fbm {runid} d q {num} t sprc w {typeF} q q"
    finFile2 = f"{runid}_fi_{num}{extralab}.cdf"

    _, fileonly = IOtools.reducePathLevel(file, level=1)
    commandOrder = f"trfbm_order {fileonly}"  # This alias is defined in .bashrc of e.g. mfews: alias trfbm_order='python3 ~/TRANSPhub/05_Utilities/acsort.py'

    runGetFBM(
        folderOrig, commandMain, file, finFile, name=runid, commandOrder=commandOrder
    )

    os.system(f"mv {folderOrig / f'finFile'} {folderOrig / f'finFile2'}")

    cdf = netCDF4.Dataset(folderOrig / f"finFile2")

    return cdf.variables


def runGetFBM(
    folderOrig,
    commandMain,
    file,
    finFile2,
    name="",
    MaxSeconds=60 * 1,
    commandOrder=None,
):
    folder, fileonly = IOtools.reducePathLevel(file, level=1)

    fbm_job = FARMINGtools.mitim_job(folder)

    fbm_job.define_machine(
        "get_fbm",
        f"tmp_fbm_{name}",
        launchSlurm=False,
    )

    print("\t\t\t- Running get_fbm command")
    if commandOrder is not None:
        print("\t\t\t- [First running AC corrector]")

        fbm_job.prep(
            f"{commandOrder} && mv {fileonly} {fileonly}_converted",
            output_files=[f"{fileonly}_converted"],
            input_files=[file],
        )
        fbm_job.run(timeoutSecs=MaxSeconds)

        os.system(f"mv {file} {file}_original")
        os.system(f"mv {file}_converted {file}")

    fbm_job.prep(
        commandMain,
        output_files=[finFile2],
        input_files=[file],
    )
    fbm_job.run(timeoutSecs=MaxSeconds)


class birthCDF:
    def __init__(self, file, particle="D_MCBEAM"):
        print(f" >> Gathering BIRTH data for {particle}")

        file = IOtools.expandPath(file)
        self.cdf = netCDF4.Dataset(file)

        # Major radius (cm)
        self.R = self.cdf[f"bs_r_{particle}"][:].data * 1e-2
        # Elevation (cm)
        self.Z = self.cdf[f"bs_z_{particle}"][:].data * 1e-2
        # Toroidal angle (deg)
        self.zeta = self.cdf[f"bs_zeta_{particle}"][:].data
        # Energy (eV)
        self.energy = self.cdf[f"bs_einj_{particle}"][:].data
        # V||/V (-)
        self.pitch = self.cdf[f"bs_xksid_{particle}"][:].data

        self.zeta = self.zeta
        # Get top-down cumulative population of particles
        self.X, self.Y = self.R * np.cos(np.radians(self.zeta)), self.R * np.sin(
            np.radians(self.zeta)
        )

    def plotTopDown(self, ax=None, color="b"):
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.X, self.Y, alpha=0.25, facecolor=color, edgecolor=color, marker="o"
        )

    def plotPol(self, ax=None, color="b"):
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.R, self.Z, alpha=0.25, facecolor=color, edgecolor=color, marker="o"
        )


def getFBMprocess(folderWork, nameRunid, datanum=1, FBMparticle="He4_FUSN"):
    fbm_He4_gc, fbm_He4_po = None, None
    noHe4 = False

    # Guiding Center
    nameTry = folderWork / "NUBEAM_folder" / f"{nameRunid}_fi_{datanum}_GC.cdf"
    nameTry2 = folderWork / "NUBEAM_folder" / f"{nameRunid}.DATA{datanum}"

    if nameTry.exists():
        ACalreadyConverted, name = True, nameTry
    elif nameTry2.exists():
        ACalreadyConverted, name = False, nameTry2
    else:
        ACalreadyConverted = None

    if ACalreadyConverted is not None:
        try:
            print(
                f"\t\t\t- File ...{IOtools.clipstr(name)} found, running workflow to get FBM"
            )
            fbm_He4_gc = fbmCDF(
                name,
                ACalreadyConverted=ACalreadyConverted,
                particle=FBMparticle,
                guidingCenter=True,
            )
        except:
            noHe4 = True
            print("\t\t- He4 from Fusion could not be found in FBM", typeMsg="w")

    # Particle position
    nameTry = folderWork / "NUBEAM_folder" / f"{nameRunid}_fi_{datanum}_PO.cdf"
    nameTry2 = folderWork / "NUBEAM_folder" / f"{nameRunid}.DATA{datanum}"
    nameTry3 = folderWork / "NUBEAM_folder" / f"{nameRunid}.DATA{datanum}_original"

    if nameTry.exists():
        ACalreadyConverted, name = True, nameTry
    elif nameTry2.exists(): 
        ACalreadyConverted, name = False, nameTry2
    elif nameTry3.exists():
        ACalreadyConverted, name = False, nameTry2
        os.system(f"mv {nameTry3} {nameTry2}")
    else:
        ACalreadyConverted = None

    if ACalreadyConverted is not None:
        if not noHe4:
            try:
                print(
                    f"\t\t\t- File ...{IOtools.clipstr(name)} found, running workflow to get FBM"
                )
                fbm_He4_po = fbmCDF(
                    name,
                    ACalreadyConverted=ACalreadyConverted,
                    particle=FBMparticle,
                    guidingCenter=False,
                )
            except:
                print("\t\t- He4 from Fusion could not be found in FBM", typeMsg="w")
    else:
        print("\t\t\t- FBM files not found", typeMsg="w")

    return fbm_He4_gc, fbm_He4_po
