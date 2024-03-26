import os
import copy
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools.aux import GACODEdefaults, GACODErun
from mitim_tools.misc_tools import IOtools, GRAPHICStools, FARMINGtools
from mitim_tools.gacode_tools.aux import GACODEplotting
from mitim_tools.misc_tools.IOtools import printMsg as print
from pygacode.cgyro.data_plot import cgyrodata_plot
from pygacode import gacodefuncs
from IPython import embed


class CGYRO:
    def __init__(self):

        self.output_files_test = [
            "out.cgyro.equilibrium",
            "out.cgyro.info",
            "out.cgyro.mpi",
            "input.cgyro.gen",
            "out.cgyro.egrid",
            "out.cgyro.grids",
            "out.cgyro.memory",
            "out.cgyro.rotation",
        ]

        self.output_files = [
            "bin.cgyro.geo",
            "bin.cgyro.kxky_e",
            "bin.cgyro.kxky_n",
            "bin.cgyro.kxky_phi",
            "bin.cgyro.kxky_v",
            "bin.cgyro.ky_cflux",
            "bin.cgyro.ky_flux",
            "bin.cgyro.phib",
            "bin.cgyro.restart",
            "bin.cgyro.restart.old",
            "input.cgyro",
            "input.cgyro.gen",
            "input.gacode",
            "mitim.out",
            "mitim_bash.src",
            "mitim_shell_executor.sh",
            "out.cgyro.egrid",
            "out.cgyro.equilibrium",
            "out.cgyro.freq",
            "out.cgyro.grids",
            "out.cgyro.hosts",
            "out.cgyro.info",
            "out.cgyro.memory",
            "out.cgyro.mpi",
            "out.cgyro.prec",
            "out.cgyro.rotation",
            "out.cgyro.startups",
            "out.cgyro.tag",
            "out.cgyro.time",
            "out.cgyro.timing",
            "out.cgyro.version",
        ]

        self.results = {}

    def prep(self, folder, inputgacode_file):

        # Prepare main folder with input.gacode
        self.folder = IOtools.expandPath(folder)

        if not os.path.exists(self.folder):
            os.system(f"mkdir -p {self.folder}")

        self.inputgacode_file = f"{self.folder}/input.gacode"
        os.system(f"cp {inputgacode_file} {self.inputgacode_file}")

    def run(
        self,
        subFolderCGYRO,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
        test_run=False,
        n=16,
        nomp=1,
    ):

        self.folderCGYRO = f"{self.folder}/{subFolderCGYRO}_{roa:.6f}/"

        if not os.path.exists(self.folderCGYRO):
            os.system(f"mkdir -p {self.folderCGYRO}")

        input_cgyro_file = f"{self.folderCGYRO}/input.cgyro"
        inputCGYRO = CGYROinput(file=input_cgyro_file)

        inputgacode_file_this = f"{self.folderCGYRO}/input.gacode"
        os.system(f"cp {self.inputgacode_file} {inputgacode_file_this}")

        ResultsFiles_new = []
        for i in self.output_files:
            if "mitim.out" not in i:
                ResultsFiles_new.append(i)
        self.output_files = ResultsFiles_new

        inputCGYRO = GACODErun.modifyInputs(
            inputCGYRO,
            Settings=CGYROsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            addControlFunction=GACODEdefaults.addCGYROcontrol,
            rmin=roa,
        )

        inputCGYRO.writeCurrentStatus()

        self.cgyro_job = FARMINGtools.mitim_job(self.folderCGYRO)

        name = f'mitim_cgyro_{subFolderCGYRO}_{roa:.6f}{"_test" if test_run else ""}'

        if test_run:

            self.cgyro_job.define_machine(
                "cgyro",
                name,
                slurm_settings={
                    "name": name,
                    "minutes": 5,
                    "cpuspertask": 1,
                    "ntasks": 1,
                },
            )

            CGYROcommand = "cgyro -t ."

        else:

            self.cgyro_job.define_machine(
                "cgyro",
                name,
                launchSlurm=False,
            )

            if self.cgyro_job.launchSlurm:
                CGYROcommand = f'gacode_qsub -e . -n {n} -nomp {nomp} -repo {self.cgyro_job.machineSettings["slurm"]["account"]} -queue {self.cgyro_job.machineSettings["slurm"]["partition"]} -w 0:10:00 -s'
            else:

                CGYROcommand = f"cgyro -e . -n {n} -nomp {nomp}"

        self.cgyro_job.prep(
            CGYROcommand,
            input_files=[input_cgyro_file, inputgacode_file_this],
            output_files=self.output_files if not test_run else self.output_files_test,
        )

        self.cgyro_job.run(
            waitYN=not self.cgyro_job.launchSlurm
        )  # ,removeScratchFolders=False)

    def check(self, every_n_minutes=5):

        if self.cgyro_job.launchSlurm:
            print("- Checker job status")

            while True:
                self.cgyro_job.check()
                print(
                    f'\t- Current status (as of  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}): {self.cgyro_job.status} ({self.cgyro_job.infoSLURM["STATE"]})'
                )
                if self.cgyro_job.status == 2:
                    break
                else:
                    print(f"\t- Waiting {every_n_minutes} minutes")
                    time.sleep(every_n_minutes * 60)
        else:
            print("- Not checking status because this was run command line (not slurm)")

        print("\t- Job considered finished")

    def get(self):
        """
        For a job that has been submitted but not waited for, once it is done, get the results
        """

        if self.cgyro_job.launchSlurm:
            self.cgyro_job.connect()
            self.cgyro_job.retrieve()
            self.cgyro_job.close()
        else:
            print(
                "- Not retrieving results because this was run command line (not slurm)"
            )

    # ---------------------------------------------------------------------------------------------------------
    # Reading and plotting
    # ---------------------------------------------------------------------------------------------------------

    def read(self, label="cgyro1", folder=None):

        folder = folder or self.folderCGYRO

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

    def plotLS(self, labels=["cgyro1"], fig=None):
        colors = GRAPHICStools.listColors()

        if fig is None:
            # fig = plt.figure(figsize=(15,9))

            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fnLS = FigureNotebook(
                "Linear CGYRO Notebook",
                geometry="1600x1000",
            )
            fig1 = self.fnLS.add_figure(label="Linear Stability")
            fig2 = self.fnLS.add_figure(label="Ballooning")

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
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")

        colors = GRAPHICStools.listColors()

        fig = self.fn.add_figure(label="Fluxes Time Traces")
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


class CGYROinput:
    def __init__(self, file=None):
        self.file = file

        if self.file is not None and os.path.exists(self.file):
            with open(self.file, "r") as f:
                lines = f.readlines()
            self.file_txt = "".join(lines)
        else:
            self.file_txt = ""

        self.controls = GACODErun.buildDictFromInput(self.file_txt)

    def writeCurrentStatus(self, file=None):
        print("\t- Writting CGYRO input file")

        if file is None:
            file = self.file

        with open(file, "w") as f:
            f.write(
                "#-------------------------------------------------------------------------\n"
            )
            f.write(
                "# CGYRO input file modified by MITIM framework (Rodriguez-Fernandez, 2020)\n"
            )
            f.write(
                "#-------------------------------------------------------------------------"
            )

            f.write("\n\n# Control parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.controls:
                var = self.controls[ikey]
                f.write(f"{ikey.ljust(23)} = {var}\n")
