import os
from re import sub
import shutil
import datetime
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools.utils import GACODEdefaults, GACODErun, CGYROutils
from mitim_tools.misc_tools import IOtools, GRAPHICStools, FARMINGtools
from mitim_tools.gacode_tools.utils import GACODEplotting
from mitim_tools.misc_tools.LOGtools import printMsg as print
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

        self.folder.mkdir(parents=True, exist_ok=True)

        self.inputgacode_file = self.folder / "input.gacode"
        shutil.copy2(IOtools.expandPath(inputgacode_file), self.inputgacode_file)

    def _prerun(
        self,
        subFolderCGYRO,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
    ):

        self.folderCGYRO = self.folder / Path(subFolderCGYRO)

        self.folderCGYRO.mkdir(parents=True, exist_ok=True)

        input_cgyro_file = self.folderCGYRO / "input.cgyro"
        inputCGYRO = CGYROinput(file=input_cgyro_file)

        inputgacode_file_this = self.folderCGYRO / "input.gacode"
        shutil.copy2(self.inputgacode_file, inputgacode_file_this)

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
            control_file = 'input.cgyro.controls'
        )

        inputCGYRO.writeCurrentStatus()

        return input_cgyro_file, inputgacode_file_this

    def run_test(
        self,
        subFolderCGYRO,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
        **kwargs
    ):
        
        if 'scan_param' in kwargs:
            print("\t- Cannot run CGYRO tests with scan_param, running just the base",typeMsg="i")
        
        input_cgyro_file, inputgacode_file_this = self._prerun(
            subFolderCGYRO,
            roa=roa,
            CGYROsettings=CGYROsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
        )

        self.cgyro_job = FARMINGtools.mitim_job(self.folderCGYRO)

        name = f'mitim_cgyro_{subFolderCGYRO}_{roa:.6f}_test'

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

        self.cgyro_job.prep(
            CGYROcommand,
            input_files=[input_cgyro_file, inputgacode_file_this],
            output_files=self.output_files_test,
        )

        self.cgyro_job.run()

    def run(self,subFolderCGYRO,test_run=False,**kwargs):

        if test_run:
            self.run_test(subFolderCGYRO,**kwargs)
        else:
            self.run_full(subFolderCGYRO,**kwargs)

    def run_full(
        self,
        subFolderCGYRO,
        roa=0.55,
        CGYROsettings=None,
        extraOptions={},
        multipliers={},
        scan_param = None,      # {'variable': 'KY', 'values': [0.2,0.3,0.4]}
        enforce_equality = None,  # e.g. {'DLNTDR_SCALE_2': 'DLNTDR_SCALE_1', 'DLNTDR_SCALE_3': 'DLNTDR_SCALE_1'}
        minutes = 5,
        n = 16,
        nomp = 1,
        submit_via_qsub=True, #TODO fix this, works only at NERSC? no scans?
        clean_folder_going_in=True, # Make sure the scratch folder is removed before running (unless I want a restart!)
        submit_run=True, # False if I just want to check and fetch the job that was already submitted (e.g. via qsub or slurm)
    ):
        
        input_cgyro_file, inputgacode_file_this = self._prerun(
            subFolderCGYRO,
            roa=roa,
            CGYROsettings=CGYROsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
        )

        self.cgyro_job = FARMINGtools.mitim_job(self.folderCGYRO)

        name = f'mitim_cgyro_{subFolderCGYRO}_{roa:.6f}'

        if scan_param is not None and submit_via_qsub:
            raise Exception(" <MITIM> Cannot use scan_param with submit_via_qsub=True, because it requires a different job for each value of the scan parameter.")

        if submit_via_qsub:

            self.cgyro_job.define_machine(
                "cgyro",
                name,
                launchSlurm=False,
            )

            subfolder = "scan0"
            queue = "-queue " + self.cgyro_job.machineSettings['slurm']['partition'] if "partition" in self.cgyro_job.machineSettings['slurm'] else ""

            CGYROcommand = f'gacode_qsub -e {subfolder} -n {n} -nomp {nomp} {queue} -w 0:{minutes}:00 -s'

            if "account" in self.cgyro_job.machineSettings["slurm"] and self.cgyro_job.machineSettings["slurm"]["account"] is not None:
                CGYROcommand += f" -repo {self.cgyro_job.machineSettings['slurm']['account']}"

            self.slurm_output = "batch.out"

            # ---
            folder_run = self.folderCGYRO / subfolder
            folder_run.mkdir(parents=True, exist_ok=True)

            # Copy the input.cgyro in the subfolder
            input_cgyro_file_this = folder_run / "input.cgyro"
            shutil.copy2(input_cgyro_file, input_cgyro_file_this)

            # Copy the input.gacode file in the subfolder
            inputgacode_file_this = folder_run / "input.gacode"
            shutil.copy2(self.inputgacode_file, inputgacode_file_this)

            # Prepare the input and output folders
            input_folders = [folder_run]
            output_folders = [subfolder]

        else:

            if scan_param is None:
                job_array = None
                folder = 'scan0'
                scan_param = {'variable': None, 'values': [0]}  # Dummy scan parameter to avoid issues with the code below
            else:
                # Array
                job_array = ''
                for i,value in enumerate(scan_param['values']):
                    if job_array != '':
                        job_array += ','
                    job_array += str(i)

                folder = 'scan"$SLURM_ARRAY_TASK_ID"'

            # Machine
            self.cgyro_job.define_machine(
                "cgyro",
                name,
                slurm_settings={
                    "name": name,
                    "minutes": minutes,
                    "ntasks": n,
                    "job_array": job_array,
                },
            )

            if not self.cgyro_job.launchSlurm:
                raise Exception(" <MITIM> Cannot run CGYRO scans without slurm")

            # Command to run cgyro
            CGYROcommand = f'cgyro -e {folder} -n {n} -nomp {nomp} -p {self.cgyro_job.folderExecution}'

            # Scans
            input_folders = []
            output_folders = []
            for i,value in enumerate(scan_param['values']):
                subfolder = f"scan{i}"
                folder_run = self.folderCGYRO / subfolder
                folder_run.mkdir(parents=True, exist_ok=True)

                # Copy the input.cgyro in the subfolder
                input_cgyro_file_this = folder_run / "input.cgyro"
                shutil.copy2(input_cgyro_file, input_cgyro_file_this)

                # Modify the input.cgyro file with the scan parameter
                extraOptions_this = extraOptions.copy()
                if scan_param['variable'] is not None:
                    extraOptions_this[scan_param['variable']] = value


                # If there is an enforce_equality, apply it
                if enforce_equality is not None:
                    for key in enforce_equality:
                        extraOptions_this[key] = extraOptions_this[enforce_equality[key]]

                inputCGYRO = CGYROinput(file=input_cgyro_file_this)
                input_cgyro_file_this = GACODErun.modifyInputs(
                    inputCGYRO,
                    Settings=CGYROsettings,
                    extraOptions=extraOptions_this,
                    multipliers=multipliers,
                    addControlFunction=GACODEdefaults.addCGYROcontrol,
                    rmin=roa,
                    control_file = 'input.cgyro.controls'
                )

                input_cgyro_file_this.writeCurrentStatus()

                # Copy the input.gacode file in the subfolder
                inputgacode_file_this = folder_run / "input.gacode"
                shutil.copy2(self.inputgacode_file, inputgacode_file_this)

                # Prepare the input and output folders
                input_folders.append(folder_run)
                output_folders.append(subfolder)

            self.slurm_output = "slurm_output.dat"

        # First submit the job with gacode_qsub, which will submit the cgyro job via slurm, with name 
        self.cgyro_job.prep(
            CGYROcommand,
            input_folders = input_folders,
            output_folders=output_folders,
            )

        if submit_run:
            self.cgyro_job.run(
                waitYN=False,
                check_if_files_received=False,
                removeScratchFolders=False,
                removeScratchFolders_goingIn=clean_folder_going_in, 
                )

        # Prepare how to search for the job without waiting for it
        name_default_submission_qsub = Path(self.cgyro_job.folderExecution).name

        self.cgyro_job.launchSlurm = True
        self.cgyro_job.slurm_settings['name'] = name_default_submission_qsub


    def check(self, every_n_minutes=5):

        if self.cgyro_job.launchSlurm:
            print("- Checker job status")

            while True:
                self.cgyro_job.check(file_output = self.slurm_output)
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

        print("\n\t* Job considered finished",typeMsg="i")

    def fetch(self):
        """
        For a job that has been submitted but not waited for, once it is done, get the results
        """

        print("\n\n\t- Fetching results")

        if self.cgyro_job.launchSlurm:
            self.cgyro_job.connect()
            self.cgyro_job.retrieve()
            self.cgyro_job.close()
        else:
            print("- Not retrieving results because this was run command line (not slurm)")

    def delete(self):

        print("\n\n\t- Deleting job")

        self.cgyro_job.launchSlurm = False

        self.cgyro_job.prep(
            f"scancel -n {self.cgyro_job.slurm_settings['name']}",
            label_log_files="_finish",
        )

        self.cgyro_job.run()

    # ---------------------------------------------------------------------------------------------------------
    # Reading and plotting
    # ---------------------------------------------------------------------------------------------------------

    def read(self, label="cgyro1", folder=None, tmin = 0.0):

        folder = IOtools.expandPath(folder) if folder is not None else self.folderCGYRO

        folders = sorted(list((folder).glob("scan*")))

        if len(folders) == 0:
            folders = [folder]
            attach_name = False
        else:
            print(f"\t- Found {len(folders)} scan folders in {folder.resolve()}:")
            for f in folders:
                print(f"\t\t- {f.name}")
            attach_name = True

        for folder in folders:

            original_dir = os.getcwd()

            if attach_name:
                label_new = f"{label}_{folder.name}"
            else:
                label_new = label

            try:
                print(f"\t- Reading CGYRO data from {folder.resolve()}")
                self.results[label_new] = cgyrodata_plot(f"{folder.resolve()}{os.sep}")
            except:
                if print('- Could not read data, do you want me to try do "cgyro -t" in the folder?',typeMsg='q'):
                    os.chdir(folder)
                    os.system("cgyro -t")
                self.results[label_new]  = cgyrodata_plot(f"{folder.resolve()}{os.sep}")

            os.chdir(original_dir)

            try:
                self._postprocess(label_new, folder, tmin=tmin)
            except Exception as e:
                print(f"\t- Error during postprocessing: {e}")
                print("\t- Skipping postprocessing, results may be incomplete or linear run")

    def _postprocess(self, label, folder, tmin=0.0):

        self.results[label].tmin = tmin

        # Extra postprocessing
        self.results[label].electron_flag = np.where(self.results[label].z == -1)[0][0]
        self.results[label].all_flags = np.arange(0, len(self.results[label].z), 1)
        self.results[label].ions_flags = self.results[label].all_flags[self.results[label].all_flags != self.results[label].electron_flag]

        self.results[label].all_names = [f"{gacodefuncs.specmap(self.results[label].mass[i],self.results[label].z[i])}({self.results[label].z[i]},{self.results[label].mass[i]:.1f})" for i in self.results[label].all_flags]

        # ************************
        # Calculations
        # ************************

        cgyro = self.results[label]
        cgyro.getflux()
        cgyro.getnorm("elec")

        self.results[label].t = self.results[label].tnorm

        ys = np.sum(cgyro.ky_flux, axis=(2, 3))

        self.results[label].Qe = ys[-1, 1, :] / cgyro.qc
        self.results[label].Qi_all = ys[:-1, 1, :] / cgyro.qc
        self.results[label].Qi = self.results[label].Qi_all.sum(axis=0)
        self.results[label].Ge = ys[-1, 0, :]

        roa,alne,self.results[label].aLTi,alte,self.results[label].Qi_mean,self.results[label].Qi_std,self.results[label].Qe_mean,self.results[label].Qe_std,self.results[label].Ge_mean, self.results[label].Ge_std,m_gimp,std_gimp,m_mo,std_mo,m_tur,std_tur,qgb,ggb,pgb,sgb,tstart,nt = CGYROutils.grab_cgyro_nth(str(folder.resolve()), tmin, False, False)
    
        self.results[label].qgb = qgb
    
        self.results[label].Qi_mean *= qgb
        self.results[label].Qi_std *= qgb
        self.results[label].Qe_mean *= qgb
        self.results[label].Qe_std *= qgb

    
    def derive_statistics(self,x,y,x_min=0.0):

        return y.mean(), y.std()

    def plot(self, labels=[""]):
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")

        colors = GRAPHICStools.listColors()

        fig = self.fn.add_figure(label="Fluxes Time Traces")
        axsFluxes_t = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )

        for j in range(len(labels)):
            self.plot_fluxes(
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )

    def plot_fluxes(self, axs=None, label="", c="b", lw=1, plotLegend=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
				AB
                CD
				"""
            )

        ls = GRAPHICStools.listLS()

        # Electron energy flux
        ax = axs["A"]
        ax.plot(
            self.results[label].t,
            self.results[label].Qe,
            ls=ls[0],
            lw=lw,
            c=c,
            label=f"{label}, electron",
        )
        ax.set_xlabel("$t$ ($a/c_s$)")
        ax.set_ylabel("$Q_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron energy flux')

        # Ion energy fluxes
        ax = axs["B"]
        ax.plot(
            self.results[label].t,
            self.results[label].Qi,
            ls=ls[0],
            lw=lw,
            c=c,
            label=f"{label}",
        )
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes')

        ax = axs["C"]
        for j, i in enumerate(self.results[label].ions_flags):
            ax.plot(
                self.results[label].t,
                self.results[label].Qi_all[j],
                ls=ls[j + 1],
                lw=lw / 2,
                c=c,
                label=f"{label}, {self.results[label].all_names[i]}",
            )
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes (separate species)')
        GRAPHICStools.addLegendApart(ax,ratio=0.95, withleg=True, size = 8)


        # Electron particle flux
        ax = axs["D"]
        ax.plot(
            self.results[label].t,
            self.results[label].Ge,
            ls=ls[0],
            lw=lw,
            c=c,
            label=f"{label}, electron",
        )
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle flux')



    def plotLS(self, labels=["cgyro1"], fig=None):
        colors = GRAPHICStools.listColors()

        if fig is None:
            # fig = plt.figure(figsize=(15,9))

            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fn = FigureNotebook(
                "Linear CGYRO Notebook",
                geometry="1600x1000",
            )
            fig1 = self.fn.add_figure(label="Linear Stability")
            fig2 = self.fn.add_figure(label="Ballooning")

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
        #ax.set_xlim([5e-2, 50.0])

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


class CGYROinput:
    def __init__(self, file=None):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None

        if self.file is not None and self.file.exists():
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
