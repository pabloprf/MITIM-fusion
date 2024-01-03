import os
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools import TGYROtools
from mitim_tools.gacode_tools.aux import GACODEdefaults, GACODErun
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.gacode_tools.aux import GACODEplotting
from mitim_tools.misc_tools.IOtools import printMsg as print
from pygacode.cgyro.data_plot import cgyrodata_plot
from pygacode import gacodefuncs


class CGYRO:
    def __init__(
        self, alreadyRun=None, cdf=None, time=100.0, rhos=[0.4, 0.6], avTime=0.0
    ):
        if alreadyRun is not None:
            # For the case in which I have run TGLF somewhere else, not using to plot and modify the class
            self.__class__ = alreadyRun.__class__
            self.__dict__ = alreadyRun.__dict__
            print("Readying previously-run case")
        else:
            self.ResultsFiles = [
                "bin.cgyro.ky_cflux",
                "input.cgyro.gen",
                "out.cgyro.memory",
                "out.cgyro.startups",
                "bin.cgyro.ky_flux",
                "out.cgyro.version",
                "out.cgyro.egrid",
                "bin.cgyro.geo",
                "out.cgyro.grids",
                "out.cgyro.mpi",
                "out.cgyro.tag",
                "out.cgyro.hosts",
                "out.cgyro.prec",
                "out.cgyro.time",
                "bin.cgyro.kxky_phi",
                "bin.cgyro.restart",
                "out.cgyro.equilibrium",
                "out.cgyro.info",
                "out.cgyro.rotation",
                "out.cgyro.timing",
            ]

            # self.ResultsFiles.append('bin.cgyro.freq')
            self.ResultsFiles.append("out.cgyro.freq")
            self.ResultsFiles.append("bin.cgyro.phib")
            self.ResultsFiles.append("bin.cgyro.aparb")

            self.LocationCDF = cdf
            try:
                self.folderWork, self.nameRunid = IOtools.getLocInfo(self.LocationCDF)
            except:
                self.folderWork, self.nameRunid = None, None

            self.rhos, self.time, self.avTime = rhos, time, avTime

            self.results, self.scans, self.tgyro = {}, {}, None

            self.NormalizationSets = None

    def prep(
        self,
        FolderGACODE,
        restart=False,
        newGACODE=True,
        remove_tmpTGYRO=False,
        onlyThermal_TGYRO=False,
        cdf_open=None,
    ):
        self.norm_select = "PROFILES"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize by preparing a tgyro class and running for -1 iterations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # TGYRO class. It checks existence and creates input.profiles/input.gacode

        self.tgyro = TGYROtools.TGYRO(
            cdf=self.LocationCDF, time=self.time, avTime=self.avTime
        )
        self.tgyro.prep(
            FolderGACODE,
            restart=restart,
            remove_tmp=remove_tmpTGYRO,
            subfolder="tmp_tgyro_prep",
            newGACODE=newGACODE,
        )

        self.FolderGACODE, self.FolderGACODE_tmp = (
            self.tgyro.FolderGACODE,
            self.tgyro.FolderGACODE_tmp,
        )

    def runLinear(
        self,
        subFolderCGYRO="cgyro1/",
        CGYROsettings=1,
        kys=[0.3],
        cores_per_ky=32,
        restart=False,
        gacodeCGYRO="gacode",
        forceIfRestart=False,
        extra_name="",
    ):
        """
        kys = [minky,num_toroidal]
        """

        self.kys = kys
        numcores = int(len(self.kys) * cores_per_ky)

        self.gacodeCGYRO = gacodeCGYRO
        self.FolderCGYRO = IOtools.expandPath(self.FolderGACODE + subFolderCGYRO + "/")
        self.FolderCGYRO_tmp = self.FolderCGYRO + "/tmp_standard/"

        exists = not restart
        for ir in self.rhos:
            for j in self.ResultsFiles:
                exists = exists and os.path.exists(f"{self.FolderCGYRO}{j}_{ir:.2f}")

        if not exists:
            IOtools.askNewFolder(self.FolderCGYRO, force=forceIfRestart)
            IOtools.askNewFolder(self.FolderCGYRO_tmp)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Change this specific run of CGYRO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ky = self.kys[0]
        self.latest_inputsFileCGYRO = changeANDwrite_CGYRO(
            self.rhos, ky, self.FolderCGYRO, CGYROsettings=CGYROsettings
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # input.gacode
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        inputGacode = self.FolderCGYRO_tmp + "/input.gacode"
        self.tgyro.profiles.writeCurrentStatus(file=inputGacode)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run CGYRO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if exists:
            print(
                " --> CGYRO not run b/c results found. Please ensure those were run with same inputs (restart may be good)"
            )
        else:
            GACODErun.runCGYRO(
                self.rhos,
                self.FolderCGYRO,
                self.FolderCGYRO_tmp,
                self.latest_inputsFileCGYRO,
                inputGacode,
                numcores=numcores,
                filesToRetrieve=self.ResultsFiles,
                name=f'cgyro_{self.nameRunid}_{subFolderCGYRO.strip("/")}{extra_name}',
                gacode_compilation=self.gacodeCGYRO,
            )

    def read(self, label="cgyro1", folder=None):
        # ~~~~ If no specified folder, check the last one
        if folder is None:
            folder = self.FolderCGYRO
        if folder[-1] != "/":
            folder += "/"

        # ~~~~ Read using PYGACODE package
        # tmpf = f'{folder}/tmp_pygacode/'
        # os.system(f'mkdir {tmpf}')
        # allfiles = IOtools.findExistingFiles(folder,'')
        # for i in allfiles: os.system(f'cp {folder}/{i} {tmpf}/{i.split("_")[0]}')
        # self.results[label] = cgyrodata_plot(tmpf)
        # #os.system('rm -r {0}'.format(tmpf))

        try:
            self.results[label] = cgyrodata_plot(folder)
        except:
            if (
                True
            ):  # print('- Could not read data, do you want me to try do "cgyro -t" in the folder?',typeMsg='q'):
                os.system(f"cd {folder} && cgyro -t")
            self.results[label] = cgyrodata_plot(folder)

        # Extra postprocessing
        self.results[label].electron_flag = np.where(self.results[label].z == -1)[0][0]
        self.results[label].all_flags = np.arange(0, len(self.results[label].z), 1)
        self.results[label].ions_flags = self.results[label].all_flags[
            self.results[label].all_flags != self.results[label].electron_flag
        ]

        self.results[label].all_names = [
            f"{gacodefuncs.specmap(self.results[label].mass[i],self.results[label].z[i])}({self.results[label].z[i]},{self.results[label].mass[i]:.1f})"
            for i in self.results[label].all_flags
        ]

    def get_flux(self, label="", moment="e", ispec=0, retrieveSpecieFlag=True):
        cgyro = self.results[label]

        # Time
        usec = cgyro.getflux()
        cgyro.getnorm("elec")
        t = cgyro.tnorm

        # Flux
        ys = np.sum(cgyro.ky_flux, axis=(2, 3))
        if moment == "n":
            y = ys[ispec, 0, :]
            mtag = "\Gamma"
        elif moment == "e":
            y = ys[ispec, 1, :] / cgyro.qc
            mtag = "Q"
        elif moment == "v":
            y = ys[ispec, 2, :]
            mtag = "\Pi"
        elif moment == "s":
            y = ys[ispec, 3, :]
            mtag = "S"

        name = gacodefuncs.specmap(cgyro.mass[ispec], cgyro.z[ispec])

        if retrieveSpecieFlag:
            flag = f"${mtag}_{{{name}}}$ (GB)"
        else:
            flag = f"${mtag}$ (GB)"

        return t, y, flag

    def plot_flux(
        self,
        ax=None,
        label="",
        moment="e",
        ispecs=[0],
        labelPlot="",
        c="b",
        lw=1,
        tmax=None,
        dense=True,
        ls="-",
    ):
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots()

        for i, ispec in enumerate(ispecs):
            t, y0, yl = self.get_flux(
                label=label,
                moment=moment,
                ispec=ispec,
                retrieveSpecieFlag=len(ispecs) == 1,
            )

            if i == 0:
                y = y0
            else:
                y += y0

        if tmax is not None:
            it = np.argmin(np.abs(t - tmax))
            t = t[: it + 1]
            y = y[: it + 1]

        ax.plot(t, y, ls=ls, lw=lw, c=c, label=labelPlot)

        ax.set_xlabel("$t$ ($a/c_s$)")
        # ax.set_xlim(left=0)
        ax.set_ylabel(yl)

        if dense:
            GRAPHICStools.addDenseAxis(ax)

    def plot_fluxes(self, axs=None, label="", c="b", lw=1, plotLegend=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
										 ABC
										 DEF
										 """
            )

        ls = GRAPHICStools.listLS()

        # Electron
        ax = axs["A"]
        self.plot_flux(
            ax=ax,
            label=label,
            moment="e",
            ispecs=[self.results[label].electron_flag],
            labelPlot=label,
            c=c,
            lw=lw,
        )
        ax.set_title("Electron energy flux")

        ax = axs["B"]
        self.plot_flux(
            ax=ax,
            label=label,
            moment="e",
            ispecs=self.results[label].ions_flags,
            labelPlot=f"{label}, sum",
            c=c,
            lw=lw,
            ls=ls[0],
        )
        for j, i in enumerate(self.results[label].ions_flags):
            self.plot_flux(
                ax=ax,
                label=label,
                moment="e",
                ispecs=[i],
                labelPlot=f"{label}, {self.results[label].all_names[i]}",
                c=c,
                lw=lw / 2,
                ls=ls[j + 1],
            )
        ax.set_title(f"Ion energy fluxes")

        # Ion
        ax = axs["D"]
        self.plot_flux(
            ax=ax,
            label=label,
            moment="n",
            ispecs=[self.results[label].electron_flag],
            labelPlot=label,
            c=c,
            lw=lw,
        )
        ax.set_title("Electron particle flux")

        ax = axs["E"]
        for j, i in enumerate(self.results[label].ions_flags):
            self.plot_flux(
                ax=ax,
                label=label,
                moment="n",
                ispecs=[i],
                labelPlot=f"{label}, {self.results[label].all_names[i]}",
                c=c,
                lw=lw / 2,
                ls=ls[j + 1],
            )
        self.plot_flux(
            ax=ax,
            label=label,
            moment="n",
            ispecs=self.results[label].ions_flags,
            labelPlot=f"{label}, sum",
            c=c,
            lw=lw,
            ls=ls[0],
        )
        ax.set_title("Ion particle fluxes")

        # Extra
        ax = axs["C"]
        for j, i in enumerate(self.results[label].all_flags):
            self.plot_flux(
                ax=ax,
                label=label,
                moment="v",
                ispecs=[i],
                labelPlot=f"{label}, {self.results[label].all_names[i]}",
                c=c,
                lw=lw / 2,
                ls=ls[j + 1],
            )
        self.plot_flux(
            ax=ax,
            label=label,
            moment="v",
            ispecs=self.results[label].all_flags,
            labelPlot=f"{label}, sum",
            c=c,
            lw=lw,
            ls=ls[0],
        )
        ax.set_title("Momentum flux")

        ax = axs["F"]
        try:
            for j, i in enumerate(self.results[label].all_flags):
                self.plot_flux(
                    ax=ax,
                    label=label,
                    moment="s",
                    ispecs=[self.results[label].electron_flag],
                    labelPlot=f"{label}, {self.results[label].all_names[i]}",
                    c=c,
                    lw=lw,
                )
            worked = True
        except:
            print("Could not plot energy exchange", typeMsg="w")
            worked = False
        ax.set_title("Electron energy exchange")

        plt.subplots_adjust()
        if plotLegend:
            for n in ["B", "E", "C"]:
                GRAPHICStools.addLegendApart(
                    axs[n],
                    ratio=0.7,
                    withleg=True,
                    extraPad=0,
                    size=7,
                    loc="upper left",
                )
            for n in ["F"]:
                GRAPHICStools.addLegendApart(
                    axs[n],
                    ratio=0.7,
                    withleg=worked,
                    extraPad=0,
                    size=7,
                    loc="upper left",
                )
            for n in ["A", "D"]:
                GRAPHICStools.addLegendApart(
                    axs[n],
                    ratio=0.7,
                    withleg=False,
                    extraPad=0,
                    size=7,
                    loc="upper left",
                )

    def plot(self, labels=[""]):
        plt.rcParams["figure.max_open_warning"] = False
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        plt.ioff()
        fn = FigureNotebook(0, "CGYRO Notebook", geometry="1600x1000")

        colors = GRAPHICStools.listColors()

        fig = fn.add_figure(label="Fluxes Time Traces")
        axsFluxes_t = fig.subplot_mosaic(
            """
									 ABC
									 DEF
									 """
        )

        for j in range(len(labels)):
            self.plot_fluxes(
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )

        fn.show()

    def plotLS(self, labels=["cgyro1"], fig=None):
        colors = GRAPHICStools.listColors()

        if fig is None:
            # fig = plt.figure(figsize=(15,9))
            plt.rcParams["figure.max_open_warning"] = False
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            plt.ioff()
            fn = FigureNotebook(
                0,
                f"CGYRO Notebook, run #{self.nameRunid}, time {self.time:3f}s",
                geometry="1600x1000",
            )
            fig1 = fn.add_figure(label="Linear Stability")
            fig2 = fn.add_figure(label="Ballooning")

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        ax00 = fig1.add_subplot(grid[0, 0])
        ax10 = fig1.add_subplot(grid[1, 0], sharex=ax00)
        ax01 = fig1.add_subplot(grid[0, 1])
        ax11 = fig1.add_subplot(grid[1, 1], sharex=ax01)

        K, G, F = [], [], []
        for cont, label in enumerate(self.results):
            c = self.results[label]
            baseColor = colors[cont]
            colorsC, _ = GRAPHICStools.colorTableFade(
                len(c.ky),
                startcolor=baseColor,
                endcolor=baseColor,
                alphalims=[1.0, 0.4],
            )

            ax = ax00
            for ky in range(len(c.ky)):
                ax.plot(
                    c.t,
                    c.freq[1, ky, :],
                    color=colorsC[ky],
                    label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                )

            ax = ax10
            for ky in range(len(c.ky)):
                ax.plot(
                    c.t,
                    c.freq[0, ky, :],
                    color=colorsC[ky],
                    label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                )

            K.append(np.abs(c.ky[0]))
            G.append(c.freq[1, 0, -1])
            F.append(c.freq[0, 0, -1])

        GACODEplotting.plotTGLFspectrum(
            [ax01, ax11],
            K,
            G,
            freq=F,
            coeff=0.0,
            c=colors[0],
            ls="-",
            lw=1,
            label="",
            facecolors=colors[: len(K)],
            markersize=50,
            alpha=1.0,
            titles=["Growth Rate", "Real Frequency"],
            removeLow=1e-4,
            ylabel=True,
        )

        ax = ax00
        ax.set_xlabel("Time $(a/c_s)$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$\\gamma$ $(c_s/a)$")
        ax.set_title("Growth Rate")
        ax.set_xlim(left=0)
        ax.legend()
        ax = ax10
        ax.set_xlabel("Time $(a/c_s)$")
        ax.set_ylabel("$\\omega$ $(c_s/a)$")
        ax.set_title("Real Frequency")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_xlim(left=0)

        ax = ax01
        ax.set_xlim([5e-2, 50.0])

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        ax00 = fig2.add_subplot(grid[0, 0])
        ax01 = fig2.add_subplot(grid[0, 1], sharex=ax00, sharey=ax00)
        ax02 = fig2.add_subplot(grid[0, 2], sharex=ax00, sharey=ax00)
        ax10 = fig2.add_subplot(grid[1, 0], sharex=ax00, sharey=ax00)
        ax11 = fig2.add_subplot(grid[1, 1], sharex=ax01, sharey=ax00)
        ax12 = fig2.add_subplot(grid[1, 2], sharex=ax02, sharey=ax00)

        it = -1

        for cont, label in enumerate(self.results):
            c = self.results[label]
            baseColor = colors[cont]

            colorsC, _ = GRAPHICStools.colorTableFade(
                len(c.ky),
                startcolor=baseColor,
                endcolor=baseColor,
                alphalims=[1.0, 0.4],
            )

            ax = ax00
            for ky in range(len(c.ky)):
                for var, axs, label in zip(
                    ["phib", "aparb", "bparb"],
                    [[ax00, ax10], [ax01, ax11], [ax02, ax12]],
                    ["phi", "abar", "aper"],
                ):
                    try:
                        f = c.__dict__[var][0, :, it] + 1j * c.__dict__[var][1, :, it]
                        y1 = np.real(f)
                        y2 = np.imag(f)
                        x = c.thetab / np.pi

                        ax = axs[0]
                        ax.plot(
                            x,
                            y1,
                            color=colorsC[ky],
                            ls="-",
                            label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                        )
                        ax = axs[1]
                        ax.plot(x, y2, color=colorsC[ky], ls="-")
                    except:
                        pass

        ax = ax00
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Re($\\delta\\phi$)")
        ax.set_title("$\\delta\\phi$")
        ax.legend(loc="best")

        ax.set_xlim([-2 * np.pi, 2 * np.pi])

        ax = ax01
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Re($\\delta A\\parallel$)")
        ax.set_title("$\\delta A\\parallel$")
        ax = ax02
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Re($\\delta B\\parallel$)")
        ax.set_title("$\\delta B\\parallel$")
        ax = ax10
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta\\phi$)")
        ax = ax11
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta A\\parallel$)")
        ax = ax12
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta B\\parallel$)")

        for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
            ax.axvline(x=0, lw=0.5, ls="--", c="k")
            ax.axhline(y=0, lw=0.5, ls="--", c="k")

        fn.show()


def changeANDwrite_CGYRO(rhos, ky, FolderCGYRO, CGYROsettings=1):
    inputFilesCGYRO = {}

    for i in rhos:
        rmin = i
        inputFilesCGYRO[i], CGYROoptions = GACODEdefaults.addCGYROcontrol(
            rmin, ky, CGYROsettings=CGYROsettings
        )

    return inputFilesCGYRO
