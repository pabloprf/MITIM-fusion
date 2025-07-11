import os
import shutil
import datetime
import time
from pathlib import Path
from lazy_loader import attach
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools.utils import GACODEdefaults, GACODErun, CGYROutils
from mitim_tools.misc_tools import IOtools, GRAPHICStools, FARMINGtools
from mitim_tools.gacode_tools.utils import GACODEplotting
from mitim_tools.misc_tools.LOGtools import printMsg as print
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
            # "bin.cgyro.kxky_apar", # May not exist?
            # "bin.cgyro.kxky_bpar", # May not exist?
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

            # if not self.cgyro_job.launchSlurm:
            #     raise Exception(" <MITIM> Cannot run CGYRO scans without slurm")

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

        data = {}
        labels = []
        for folder in folders:
            
            if attach_name:
                label1 = f"{label}_{folder.name}"
            else:
                label1 = label

            data[label1] = CGYROutils.CGYROout(folder, tmin=tmin)
            labels.append(label1)

        self.results.update(data)
        
        if attach_name:
            self.results[label] = CGYROutils.CGYROlinear_scan(labels, data)

    def plot(self, labels=[""], include_2D=True):
        

        # If it has scans, we need to correct the labels
        labels_corrected = []
        for i in range(len(labels)):
            if isinstance(self.results[labels[i]], CGYROutils.CGYROlinear_scan):    
                for scan_label in self.results[labels[i]].labels:
                    labels_corrected.append(scan_label)
            else:
                labels_corrected.append(labels[i])
        labels = labels_corrected
        # ------------------------------------------------
        
        
        from mitim_tools.misc_tools.GUItools import FigureNotebook
        self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")

        fig = self.fn.add_figure(label="Fluxes (time)")
        axsFluxes_t = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Fluxes (ky)")
        axsFluxes_ky = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        
        fig = self.fn.add_figure(label="Turbulence")
        axsTurbulence = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
      
        fig = self.fn.add_figure(label="Intensities")
        axsIntensities = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        
        fig = self.fn.add_figure(label="Ballooning")
        axsBallooning = fig.subplot_mosaic(
            """
            135
            246
            """
            )
        
        if include_2D:
            axs2D = []
            for i in range(len(labels)):
                fig = self.fn.add_figure(label="Turbulence (2D), " + labels[i])
                axs2D.append(fig.subplot_mosaic(
                    """
                    123
                    456
                    789
                    """
                ))
        
        fig = self.fn.add_figure(label="Inputs")
        axsInputs = fig.subplot_mosaic(
            """
            A
            """
        )

        
        colors = GRAPHICStools.listColors()

        for j in range(len(labels)):
            
            self.plot_fluxes(
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            self.plot_fluxes_ky(
                axs=axsFluxes_ky,
                label=labels[j],
                c=colors[j],
            )
            self.plot_turbulence(
                axs=axsTurbulence,
                label=labels[j],
                c=colors[j],
            )
            self.plot_intensities(
                axs=axsIntensities,
                label=labels[j],
                c=colors[j],
            )
            if 'phi_ballooning' in self.results[labels[j]].__dict__:
                self.plot_ballooning(
                    axs=axsBallooning,
                    label=labels[j],
                    c=colors[j],
                )
            
            if include_2D:
                
                self.plot_2D(
                    axs=axs2D[j],
                    label=labels[j],
                    c=colors[j],
                )
            
            self.plot_inputs(
                ax=axsInputs["A"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
                normalization_label= labels[0],  # Normalize to the first label
                only_plot_differences=len(labels) > 1,  # Only plot differences if there are multiple labels
            )
            
        axsInputs["A"].axhline(
            1.0,
            color="k",
            ls="--",
            lw=2.0
        )

    def _plot_trace(self, ax, label, variable, c="b", lw=1, ls="-", label_plot='', meanstd=True, var_meanstd= None):
        
        t = self.results[label].t
        
        if not isinstance(variable, str):
            z = variable
            if var_meanstd is not None:
                z_mean = var_meanstd[0]
                z_std = var_meanstd[1]
            
        else:
            z = self.results[label].__dict__[variable]
            if meanstd and (f'{variable}_mean' in self.results[label].__dict__):
                z_mean = self.results[label].__dict__[variable + '_mean']
                z_std = self.results[label].__dict__[variable + '_std']
            else:
                z_mean = None
                z_std = None
        
        ax.plot(
            t,
            z,
            ls=ls,
            lw=lw,
            c=c,
            label=label_plot,
        )
        
        if meanstd and z_std>0.0:
            GRAPHICStools.fillGraph(
                ax,
                t[t>self.results[label].tmin],
                z_mean,
                y_down=z_mean
                - z_std,
                y_up=z_mean
                + z_std,
                alpha=0.1,
                color=c,
                lw=0.5,
                islwOnlyMean=True,
                label=label_plot + f" {z_mean:.2f} ± {z_std:.2f} (1$\\sigma$)",
            )

    def plot_inputs(self, ax = None, label="", c="b", ms = 10, normalization_label=None, only_plot_differences=False):
        
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(18, 9))

        rel_tol = 1e-2

        legadded = False
        for i, ikey in enumerate(self.results[label].params1D):
            
            z = self.results[label].params1D[ikey]
            
            if normalization_label is not None:
                z0 = self.results[normalization_label].params1D[ikey]
                zp = z/z0 if z0 != 0 else 0
                label_plot = f"{label} / {normalization_label}"
            else:
                label_plot = label
                zp = z

            if (not only_plot_differences) or (not np.isclose(z, z0, rtol=rel_tol)):
                ax.plot(ikey,zp,'o',markersize=ms,color=c,label=label_plot if not legadded else '')
                legadded = True

        if normalization_label is not None:
            if only_plot_differences:
                ylabel = f"Parameters (DIFFERENT by {rel_tol*100:.2f}%) relative to {normalization_label}"
            else:
                ylabel = f"Parameters relative to {normalization_label}"
        else:
            ylabel = "Parameters"

        ax.set_xlabel("Parameter")
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel(ylabel)
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best')

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
        self._plot_trace(ax,label,"Qe",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
        self._plot_trace(ax,label,"Qe_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
        
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron energy flux')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Electron particle flux
        ax = axs["B"]
        self._plot_trace(ax,label,"Ge",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
        self._plot_trace(ax,label,"Ge_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
        
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle flux')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Ion energy fluxes
        ax = axs["C"]
        self._plot_trace(ax,label,"Qi",c=c,lw=lw,ls=ls[0],label_plot=f"{label}, Total")
        self._plot_trace(ax,label,"Qi_EM",c=c,lw=lw,ls=ls[1],label_plot=f"{label}, EM ($A_\\parallel$+$A_\\perp$)", meanstd=False)
        
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        # Ion species energy fluxes
        ax = axs["D"]
        for j, i in enumerate(self.results[label].ions_flags):
            self._plot_trace(ax,label,self.results[label].Qi_all[j],c=c,lw=lw,ls=ls[j],label_plot=f"{label}, {self.results[label].all_names[i]}", meanstd=False)
            
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes (separate species)')
        if plotLegend:
            ax.legend(loc='best', prop={'size': 8},)

        plt.tight_layout()

    def plot_fluxes_ky(self, axs=None, label="", c="b", lw=1):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Electron energy flux
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].Qe_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Qe_ky_mean-self.results[label].Qe_ky_std, self.results[label].Qe_ky_mean+self.results[label].Qe_ky_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$Q_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron energy flux vs ky')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Electron particle flux
        ax = axs["B"]
        ax.plot(self.results[label].ky, self.results[label].Ge_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Ge_ky_mean-self.results[label].Ge_ky_std, self.results[label].Ge_ky_mean+self.results[label].Ge_ky_std, color=c, alpha=0.2)
    
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\Gamma_e$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle flux vs ky')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Ion energy flux
        ax = axs["C"]
        ax.plot(self.results[label].ky, self.results[label].Qi_ky_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].Qi_ky_mean-self.results[label].Qi_ky_std, self.results[label].Qi_ky_mean+self.results[label].Qi_ky_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$Q_i$ (GB)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion energy fluxes vs ky')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)


    def plot_turbulence(self, axs = None, label= "cgyro1", c="b", kys = None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Is no kys provided, select just 3: first, last and middle
        if kys is None:
            ikys = [0]
            if len(self.results[label].ky) > 1:
                ikys.append(-1)
            if len(self.results[label].ky) > 2:
                ikys.append(len(self.results[label].ky) // 2)
                
            ikys = np.unique(ikys)            
        else:
            ikys = [self.results[label].ky.index(ky) for ky in kys if ky in self.results[label].ky]    

        # Growth rate as function of time
        ax = axs["A"]
        for i,ky in enumerate(ikys):
            self._plot_trace(
                ax,
                label,
                self.results[label].g[ky, :],
                c=c,
                ls = GRAPHICStools.listLS()[i],
                lw=1,
                label_plot=f"$k_{{\\theta}}\\rho_s={np.abs(self.results[label].ky[ky]):.2f}$",
                var_meanstd = [self.results[label].g_mean[ky], self.results[label].g_std[ky]],
            )
            
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\gamma$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Growth rate vs time')
        ax.legend(loc='best', prop={'size': 8},)

        # Frequency as function of time
        ax = axs["B"]
        for i,ky in enumerate(ikys):
            self._plot_trace(
                ax,
                label,
                self.results[label].f[ky, :],
                c=c,
                ls = GRAPHICStools.listLS()[i],
                lw=1,
                label_plot=f"$k_{{\\theta}}\\rho_s={np.abs(self.results[label].ky[ky]):.2f}$",
                var_meanstd = [self.results[label].f_mean[ky], self.results[label].f_std[ky]],
            )
            
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\omega$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Real Frequency vs time')
        ax.legend(loc='best', prop={'size': 8},)

        # Mean+Std Growth rate as function of ky
        ax = axs["C"]
        ax.errorbar(self.results[label].ky, self.results[label].g_mean, yerr=self.results[label].g_std, fmt='-o', markersize=5, color=c, label=label+' (mean+std)')
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\gamma$ (norm.)")
        ax.set_title('Saturated Growth Rate')
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best', prop={'size': 8},)
        
        # Mean+Std Frequency as function of ky
        ax = axs["D"]
        ax.errorbar(self.results[label].ky, self.results[label].f_mean, yerr=self.results[label].f_std, fmt='-o', markersize=5, color=c, label=label+' (mean+std)')
        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\omega$ (norm.)")
        ax.set_title('Saturated Real Frequency')
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc='best', prop={'size': 8},)
        
        plt.tight_layout()

    def plot_intensities(self, axs = None, label= "cgyro1", c="b"):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )
            
        ax = axs["A"]
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_n0, '-', c=c, lw=1, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_n, '--', c=c, lw=1, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        # ax.set_ylabel("$\\gamma$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Fluctuation intensity - Potential')
        ax.legend(loc='best', prop={'size': 8},)


        ax = axs["B"]
        ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_n0, '-', c=c, lw=1, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_n, '--', c=c, lw=1, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        # ax.set_ylabel("$\\gamma$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Fluctuation intensity - Electron Density')
        ax.legend(loc='best', prop={'size': 8},)


        ax = axs["D"]
        ax.plot(self.results[label].t, self.results[label].Ee_rms_sumnr_n0, '-', c=c, lw=1, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].Ee_rms_sumnr_n, '--', c=c, lw=1, label=f"{label}, $n>0$")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        # ax.set_ylabel("$\\gamma$ (norm.)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Fluctuation intensity - Electron Energy')
        ax.legend(loc='best', prop={'size': 8},)



        plt.tight_layout()


    def plot_ballooning(self, label="cgyro1", c="b", axs=None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                135
                246
                """
            )

        it = -1
        
        colorsC, _ = GRAPHICStools.colorTableFade(
            len(self.results[label].ky),
            startcolor=c,
            endcolor=c,
            alphalims=[1.0, 0.4],
        )

        ax = axs['1']
        for ky in range(len(self.results[label].ky)):
            for var, axsT in zip(
                ["phi_ballooning", "apar_ballooning", "bpar_ballooning"],
                [[axs['1'], axs['2']], [axs['3'], axs['4']], [axs['5'], axs['6']]],
            ):

                f = self.results[label].__dict__[var][:, it]
                y1 = np.real(f)
                y2 = np.imag(f)
                x = self.results[label].theta_ballooning / np.pi

                # Normalize
                y1_max = np.max(np.abs(y1))
                y2_max = np.max(np.abs(y2))
                y1 /= y1_max
                y2 /= y2_max

                ax = axsT[0]
                ax.plot(
                    x,
                    y1,
                    color=colorsC[ky],
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y1_max:.2e})",
                )
                ax = axsT[1]
                ax.plot(
                    x, 
                    y2, 
                    color=colorsC[ky], 
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y2_max:.2e})",
                )


        ax = axs['1']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta\\phi$)")
        ax.set_title("$\\delta\\phi$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([-2 * np.pi, 2 * np.pi])

        ax = axs['3']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta A\\parallel$)")
        ax.set_title("$\\delta A\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['5']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta B\\parallel$)")
        ax.set_title("$\\delta B\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['2']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta\\phi$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['4']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta A\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['6']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta B\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)


        for ax in [axs['1'], axs['3'], axs['5'], axs['2'], axs['4'], axs['6']]:
            ax.axvline(x=0, lw=0.5, ls="--", c="k")
            ax.axhline(y=0, lw=0.5, ls="--", c="k")

    def plot_2D(self, label="cgyro1", c="b", axs=None, times = None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))
            
            axs = fig.subplot_mosaic(
                """
                123
                456
                789
                """
            )
        
        if times is None:
            times = [self.results[label].t[-20], self.results[label].t[-10], self.results[label].t[-1]]

        for time_i, time in enumerate(times):
            
            it = np.argmin(np.abs(self.results[label].t - time))
            
            ax = axs[str(time_i+1)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)

            fa = np.max(np.abs(fp))
            f0, f1 = -fa, +fa
            
            ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(f0,f1,(f1-f0)/256),cmap=plt.get_cmap('jet'))
            
            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta\\phi$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            ax = axs[str(time_i+4)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)

            fa = np.max(np.abs(fp))
            f0, f1 = -fa, +fa
            
            ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(f0,f1,(f1-f0)/256),cmap=plt.get_cmap('jet'))
            
            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta n_e$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            ax = axs[str(time_i+7)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)

            fa = np.max(np.abs(fp))
            f0, f1 = -fa, +fa
            
            ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(f0,f1,(f1-f0)/256),cmap=plt.get_cmap('jet'))
            
            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta E_e$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            plt.tight_layout()
        
        
    def _to_real_space(self, variable = 'kxky_phi', species = None, label="cgyro1", nx = 256, ny = 512, theta_plot = 0, it = -1):
        
        # FFT version
        def maptoreal_fft(nr,nn,nx,ny,c):

            import numpy as np
            import time

            # Storage for numpy inverse real transform (irfft2)
            d = np.zeros([nx,nn],dtype=complex)

            start = time.time()

            for i in range(nr):
                p = i-nr//2
                # k is the "standard FFT index"
                if -p < 0:
                    k = -p+nx
                else:
                    k = -p
                # Use identity f(p,-n) = f(-p,n)*
                d[k,0:nn] = np.conj(c[i,0:nn])

            # 2D inverse real Hermitian transform
            # NOTE: using inverse FFT with convention exp(ipx+iny), so need n -> -n
            # NOTE: need factor of 0.5 to match half-sum method of slow maptoreal()
            f = np.fft.irfft2(d,s=[nx,ny],norm='forward')*0.5

            end = time.time()

            return f,end-start

        # Real space
        nr = self.results[label].cgyrodata.n_radial
        nn = self.results[label].cgyrodata.n_n
        craw = self.results[label].cgyrodata.__dict__[variable]
        
        if species is None:
            c = craw[:,theta_plot,:,it]
        else:
            c = craw[:,theta_plot,species,:,it]
        f,t = maptoreal_fft(nr,nn,nx,ny,c)
        
        x = np.arange(nx)*2*np.pi/nx
        y = np.arange(ny)*2*np.pi/ny
        
        # Physical maxima
        ky1 = self.results[label].cgyrodata.ky[1] if len(self.results[label].cgyrodata.ky) > 1 else self.results[label].cgyrodata.ky[0]
        xmax = self.results[label].cgyrodata.length
        ymax = (2*np.pi)/np.abs(ky1)
        xp = x/(2*np.pi)*xmax
        yp = y/(2*np.pi)*ymax

        # Periodic extensions
        xp = np.append(xp,xmax)
        yp = np.append(yp,ymax)
        fp = np.zeros([nx+1,ny+1])
        fp[0:nx,0:ny] = f[:,:]
        fp[-1,:] = fp[0,:]
        fp[:,-1] = fp[:,0]
        
        return xp, yp, fp
        
    def plot_quick_linear(self, labels=["cgyro1"], fig=None):
        colors = GRAPHICStools.listColors()

        if fig is None:
            fig = plt.figure(figsize=(15,9))

        axs = fig.subplot_mosaic(
            """
            12
            34
            """
        )
            
        def _plot_linear_stability(axs, labels, label_base,col_lin ='b', start_cont=0):

            for cont, label in enumerate(labels):
                c = self.results[label]
                baseColor = colors[cont+start_cont+1]
                colorsC, _ = GRAPHICStools.colorTableFade(
                    len(c.ky),
                    startcolor=baseColor,
                    endcolor=baseColor,
                    alphalims=[1.0, 0.4],
                )

                ax = axs['1']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.g[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                    )

                ax = axs['2']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.f[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$",
                    )

            GACODEplotting.plotTGLFspectrum(
                [axs['3'], axs['4']],
                self.results[label_base].ky,
                self.results[label_base].g_mean,
                freq=self.results[label_base].f_mean,
                coeff=0.0,
                c=col_lin,
                ls="-",
                lw=1,
                label="",
                facecolors=colors,
                markersize=50,
                alpha=1.0,
                titles=["Growth Rate", "Real Frequency"],
                removeLow=1e-4,
                ylabel=True,
            )
            
            return cont

        co = -1
        for i,label0 in enumerate(labels):
            if isinstance(self.results[label0], CGYROutils.CGYROlinear_scan):
                co = _plot_linear_stability(axs, self.results[label0].labels, label0, start_cont=co, col_lin=colors[i])
            else:
                co = _plot_linear_stability(axs, [label0], label0, start_cont=co, col_lin=colors[i])

        ax = axs['1']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$\\gamma$ $(c_s/a)$")
        ax.set_title("Growth Rate")
        ax.set_xlim(left=0)
        ax.legend()
        ax = axs['2']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.set_ylabel("$\\omega$ $(c_s/a)$")
        ax.set_title("Real Frequency")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_xlim(left=0)



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
