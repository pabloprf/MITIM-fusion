import os
import re
import copy
import subprocess
import matplotlib.pyplot as plt
import f90nml
from pathlib import Path
from mitim_tools.misc_tools import FARMINGtools, GRAPHICStools, IOtools, GUItools
import numpy as np
import pandas as pd
import xarray as xr
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


def _to_scalar(x):
    '''
    Safely extract a Python float from a value that may be a Python scalar,
    a numpy scalar, a 0-d array, or a 1-element xarray DataArray / ndarray.
    In numpy>=2, float(da) on a 1-element DataArray raises; this avoids that.
    '''
    try:
        return float(np.asarray(x).ravel()[-1])
    except (IndexError, TypeError, ValueError):
        return float('nan')


class EPED:
    def __init__(
            self,
            folder,
            template_config_file = None,
            eped_repo_files = '$EPED_SOURCE_PATH/template/engaging/eped_run_template',
            ):
        
        self.folder = Path(folder) if folder is not None else None # None for just reading

        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)

        self.results = {}

        self.inputs_potential = ['ip', 'bt', 'r', 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep', 'zeta', 's_three', 's_four']

        if template_config_file:
            self.template_config_file = IOtools.expandPath(template_config_file)
        else:
            self.template_config_file = IOtools.expandPath(__mitimroot__ / "templates" / "eped.config")
            
        self.required_files_folder = eped_repo_files

    def run(
            self,
            subfolder = 'run1',
            input_params = None,    # {'ip': 12.0, 'bt': 12.16, 'r': 1.85, 'a': 0.57, 'kappa': 1.9, 'delta': 0.5, 'neped': 30.0, 'betan': 1.0, 'zeffped': 1.5, 'nesep': 10.0, 'tesep': 100.0, 'zeta': 0, 's_three': 0.0, 's_four': 0.0},
            scan_param = None,      # {'variable': 'neped', 'values': [10.0, 20.0, 30.0]}
            keep_nsep_ratio = None, # Ratio of neped to nesep
            nproc_per_run = 64,
            minutes_slurm = 30,
            cold_start = False,
            forceifcold_start = False,  # If True, when cold_start=True and an output already exists, warn ('w') and
                                        # rerun from scratch instead of asking ('q'). For non-interactive callers.
            job_array_limit = 5,
            removeScratchFolders = True,  #ONLY CHANGE THIS FOR DEBUGGING, if you make this False, your EPED runs will be saved and they are enormous
            eped_params_override = None,
            teped_guess_eV = -1, # if -1, EPED will choose its own guess
            m = 2.5, z = 1, mi = 20, zi = 10, # plasma composition: main-ion mass/charge and impurity mass/charge.
                                              # Defaults (50/50 D-T main ion + neon impurity) match the EPED-NN training;
                                              # callers (e.g. the MAESTRO EPED beat) pass the actual plasma values.
            ):
        '''
        Notes:
            - eped_params_override: dictionary with EPED input parameters to override in the template config file
                e.g. eped_params_override = {
                    'NMODES': [5, 6, 8, 10, 15, 20, 30],
                    'TEPED_BOUND': [0.1, 1.4, 0.01],
                    }
        '''
        # ------------------------------------
        # Prepare job
        # ------------------------------------

        # Prepare folder structure
        self.folder_run = self.folder / subfolder

        # Prepare scan parameters
        scan_param_variable = scan_param['variable'] if scan_param is not None else None
        scan_param_values = scan_param['values'] if scan_param is not None else [None]

        # Prepare job array setup. Only include cases that will actually run:
        # when cold_start=False, an existing output_run<i>.nc is skipped in the
        # loop below, so it must NOT be submitted as an array task either —
        # otherwise every re-run resubmits the full scan even though most cases
        # are already cached. Mirrors the per-case skip check below.
        job_array_indices = []
        for i in range(len(scan_param_values)):
            already_done = (self.folder_run / f'output_run{i+1}.nc').exists()
            if already_done and not cold_start:
                continue
            job_array_indices.append(i + 1)
        job_array = ",".join(str(k) for k in job_array_indices)

        # Initialize Job
        self.eped_job = FARMINGtools.mitim_job(self.folder_run)

        from mitim_tools.misc_tools import SLURMtools
        self.eped_job.define_machine(
            "eped",
            "mitim_eped",
            slurm_settings={
                'job-name': 'mitim_eped',
                'time': SLURMtools.format_time(minutes_slurm),
                'ntasks-per-node': nproc_per_run,
                'array': job_array,
                # The %N concurrency throttle only means something with several
                # array elements; suppress it for a single case so the sbatch
                # does not read confusingly as "--array=1%N"
                'array_limit': job_array_limit if len(job_array_indices) > 1 else None,
            }
        )

        # ------------------------------------
        # Prepare each individual case
        # ------------------------------------

        folder_cases, output_files, shellPreCommands = [], [], []
        for i,value in enumerate(scan_param_values):

            # Folder structure
            subfolder = f'run{i+1}'
            folder_case = self.folder_run / subfolder
            
            # Prepare input parameters
            if scan_param_variable is not None:
                input_params_new = input_params.copy() if input_params is not None else {}
                input_params_new[scan_param_variable] = value
            else:
                input_params_new = input_params

            if keep_nsep_ratio is not None:
                print(f'\t> Setting nesep to {keep_nsep_ratio} * neped')
                input_params_new['nesep'] = keep_nsep_ratio * input_params_new['neped']

            # *******************************
            # Check if the case should be run
            run_case = True
            force_res = False
            if (self.folder_run / f'output_{subfolder}.nc').exists():
                if cold_start:
                    if forceifcold_start:
                        # Non-interactive callers (e.g. MAESTRO, especially a preempted+requeued run
                        # re-running a cold_start beat/creator on top of a leftover output) cannot
                        # answer a prompt: warn and rerun from scratch, since cold_start=True already
                        # means "run fresh". Avoids an InteractiveTerminalError killing the run.
                        print(f'\t> Run {subfolder} already exists but cold_start is set to True: removing and running from scratch (forceifcold_start).', typeMsg='w')
                        res = True
                    else:
                        res = print(f'\t> Run {subfolder} already exists but cold_start is set to True. Running from scratch?', typeMsg='q')
                    if res:
                        IOtools.shutil_rmtree(folder_case)
                        (self.folder_run / f'output_{subfolder}.nc').unlink(missing_ok=True)
                        force_res = True
                    else:
                        run_case = False
                else:
                    print(f'\t> Run {subfolder} already exists and cold_start is set to False. Skipping run.', typeMsg='i')
                    run_case = False

            if not run_case:
                continue
            # *******************************

            # Set up folder
            folder_case.mkdir(parents=True, exist_ok=True)

            # Preparation of the run folder by copying the template files
            eped_input_file = 'eped.input.1'
            eped_config_file = 'eped.config1'
            
            # Write input file to EPED, determining the expected output file
            output_file = self._prep_input_files(
                folder_case,
                input_params=input_params_new,
                eped_input_file=eped_input_file,
                eped_config_file=eped_config_file,
                eped_params_override=eped_params_override,
                teped_guess=teped_guess_eV,
                m=m, z=z, mi=mi, zi=zi,
                )
            
            # Before running, copy the files from EPED source, and copy the input file to the expected name, and the config file
            shellPreCommands.append(
                f'cp {self.required_files_folder}/* {self.eped_job.folderExecution}/{subfolder}/. ' +
                f'&& mv {self.eped_job.folderExecution}/{subfolder}/{eped_input_file} {self.eped_job.folderExecution}/{subfolder}/eped.input ' +
                f'&& cp {self.eped_job.folderExecution}/{subfolder}/{eped_config_file} {self.eped_job.folderExecution}/{subfolder}/eped.config'
                )

            output_files.append(output_file.as_posix())
            folder_cases.append(folder_case)

        # If no cases to run, exit
        if len(folder_cases) == 0:
            return

        # -------------------------------------
        # Execute
        # -------------------------------------
        
        # Submit as a slurm job array
        if self.eped_job.launchSlurm:
            EPEDcommand  = (
                f'cd {self.eped_job.folderExecution}/run"$SLURM_ARRAY_TASK_ID" '                                                    # Change to the run folder
                f'&& export NPROC_EPED={nproc_per_run} '                                                                            # Defines the number of processors for EPED
                f'&& if [ -z "${{SLURM_JOB_TASKS_PER_NODE:-}}" ]; then export SLURM_JOB_TASKS_PER_NODE={nproc_per_run}; fi '        # Ensure SLURM_JOB_TASKS_PER_NODE is defined
                f'&& ips.py --config=eped.config --platform=psfc_cluster.conf'                                                      # Run EPED                
            )
        # Submit locally in parallel
        else:
            EPEDcommand = ""
            for i in job_array.split(','):
                EPEDcommand += (
                    f'cd {self.eped_job.folderExecution}/run{i} '                                                                   # Change to the run folder              
                    f'&& export NPROC_EPED={nproc_per_run} '                                                                        # Defines the number of processors for EPED
                    f'&& if [ -z "${{SLURM_JOB_TASKS_PER_NODE:-}}" ]; then export SLURM_JOB_TASKS_PER_NODE={nproc_per_run}; fi '    # Ensure SLURM_JOB_TASKS_PER_NODE is defined    
                    f'&& ips.py --config=eped.config --platform=psfc_cluster.conf & \n'                                             # Run EPED in background       
                )
            EPEDcommand += 'wait\n'

        # Prepare the job script
        self.eped_job.prep(EPEDcommand,input_folders=folder_cases,output_files=copy.deepcopy(output_files),shellPreCommands=shellPreCommands)

        # Run the job
        self.eped_job.run(removeScratchFolders=removeScratchFolders) 

        # -------------------------------------
        # Postprocessing
        # -------------------------------------

        # Rename each freshly-run output to its scan-indexed name (output_run<i>.nc,
        # matching the skip check above). Only the files for cases actually run this
        # round are touched: previously-completed cases skipped this round keep their
        # existing output and are NOT wiped. The run-folder name (e.g. 'run3') carries
        # the original scan index, so a partial re-run cannot mislabel survivors by
        # compacting list positions.
        for rel in output_files:
            run_name = rel.split('/')[0]  # 'run<i+1>' — the original scan index
            target = self.folder_run / f'output_{run_name}.nc'
            target.unlink(missing_ok=True)
            (self.folder_run / rel).replace(target)

    def _prep_input_files(
            self,
            folder_case,
            input_params = None,
            eped_input_file = 'eped.input.1', # Do not call it directly 'eped.input' as it may be overwritten by the job script template copying commands
            eped_config_file = 'eped.config1',
            eped_params_override = None,
            teped_guess = -1,
            m = 2.5, z = 1, mi = 20, zi = 10,  # plasma composition (see run())
            ):

        # ----------------------------------------
        # EPED input file
        # ----------------------------------------
        
        shot = 0
        timeid = 0

        # Update with fixed parameters
        input_params.update(
            {'num_scan': 1,
             'shot': shot,
             'timeid': timeid,
             'runid': 0,
             'tewid': 0.03,
             'ptotwid': 0.03,
             'teped': teped_guess,
             'ptotped': -1,
            }
        )

        # Plasma composition: main-ion (m, z) and impurity (mi, zi). Defaults match
        # the EPED-NN training; the MAESTRO EPED beat passes the actual plasma values.
        # setdefault (not update) so composition can also be supplied/scanned through
        # input_params (e.g. a SLURM job-array scan over impurity charge); an explicit
        # input_params value then wins over the run() default.
        for _key, _val in (('m', m), ('z', z), ('mi', mi), ('zi', zi)):
            input_params.setdefault(_key, _val)

        eped_input = {'eped_input': input_params}
        nml = f90nml.Namelist(eped_input)
        
        # Write the input file
        f90nml.write(nml, folder_case / eped_input_file, force=True)

        # ----------------------------------------
        # EPED config file
        # ----------------------------------------
        modify_eped_config(self.template_config_file, folder_case / eped_config_file, eped_params_override)
            
        # ----------------------------------------
        # EPED output file
        # ----------------------------------------

        # What's the expected output file?
        output_file = folder_case.relative_to(self.folder_run) / 'eped' / 'SUMMARY' / f'e{shot:06d}.{timeid:05d}'

        return output_file

    def read(
            self,
            subfolder = 'run1',
            print_results = True,
            label = None,
            specific_folder = None,
            ):

        self.results[label if label is not None else subfolder] = {}

        if specific_folder is not None:
            folder = Path(specific_folder)
        else:
            folder = self.folder

        where_is_this = folder / subfolder if folder is not None else Path(subfolder)

        output_files = sorted(list(where_is_this.glob("*.nc")))

        for output_file in output_files:

            with xr.open_dataset(f'{output_file.resolve()}', engine='netcdf4') as ds:
                data = postprocess_eped(ds, 'G', 0.03)

            sublabel = output_file.name.split('_')[-1].split('.')[0]

            self.results[label if label is not None else subfolder][sublabel] = data

            if print_results:
                self.print(label if label is not None else subfolder, sublabel)

    def print(self,label,sublabel):
        
        print(f'\n\t> EPED results {sublabel}:')
        data = self.results[label][sublabel]

        print('\t\t> Inputs:')
        for input_param in self.inputs_potential:
            try:
                print(f'\t\t\t{input_param}: {data[input_param].values[0]}')
            except:
                print(f'\t\t\t{input_param}: Not available',typeMsg='w')

        print('\t\t> Outputs:')
        if 'ptop' in data.data_vars:
            print(f'\t\t\tptop: {data["ptop"].values[0]:.2f} kPa')
            print(f'\t\t\twptop: {data["wptop"].values[0]:.3f} psi_pol')
            if 'n_limiting' in data.data_vars:
                n_lim = int(_to_scalar(data['n_limiting']))
                if n_lim > 0:
                    dome_f = _to_scalar(data['dome_frac']) if 'dome_frac' in data.data_vars else None
                    label_pb = classify_pedestal_limit(n_lim, dome_frac=dome_f)
                    wtxt = f'{dome_f:.0%}' if dome_f is not None else '?'
                    print(f'\t\t\tlimited by: n = {n_lim}, dome width = {wtxt} of T_ped range -> {label_pb}')
        else:
            print('\t\t\tptop: Not available',typeMsg='w')

    def plot(
        self,
        labels = ['run1'],
        scan_params = ['neped'],
        scan_params_labels = ['$n_{e,ped}$ ($10^{19}m^{-3}$)'],
        colors = None,
        fn = None,
        tab_color=0,
        **kwargs_plot_prediction,
    ):
        
        if len(scan_params) != len(labels):
            if len(scan_params) == 1:
                scan_params = scan_params * len(labels)
            else:
                raise ValueError('Length of scan_params must be either 1 or equal to length of labels.')

        
        if fn is None:
            GRAPHICStools.prep_figure_papers(size=14)
            self.fn = GUItools.FigureNotebook("EPED",  geometry="1600x900")
        else:
            self.fn = fn
            
        if colors is None:
            colors = GRAPHICStools.listColors()
            
        # Figure out if labels have the same scan parameter
        if np.unique(scan_params).shape[0] > 1:
            scan_params_label = 'X (see legend for parameter that was scanned)'
            additional_labels = [f' - {label} scan' for label in scan_params_labels]
        else:
            scan_params_label = scan_params_labels[0]
            additional_labels = None
            
        
        fig = self.fn.add_figure(label="Pedestal Top", tab_color=tab_color)
        axs = fig.subplots(2, 1)
        self.plot_prediction(
            labels = labels,
            scan_params = scan_params,
            scan_params_label = scan_params_label,
            additional_labels= additional_labels,
            axs = axs,
            colors = colors,
            **kwargs_plot_prediction
        )
        
        figs_stability = {}
        figs_eped_profile_ptot = {}
        figs_eped_profile_q = {}
        figs_eped_profile_j = {}
        for label in labels:
            figs_stability[label] = self.fn.add_figure(label="EPED Stability (teped) - " + label, tab_color=tab_color)
            figs_eped_profile_ptot[label] = self.fn.add_figure(label="EPED profiles (ptot) - " + label, tab_color=tab_color)
            figs_eped_profile_q[label] = self.fn.add_figure(label="EPED profiles (q) - " + label, tab_color=tab_color)
            figs_eped_profile_j[label] = self.fn.add_figure(label="EPED profiles (J) - " + label, tab_color=tab_color)

        
        for i, label in enumerate(labels):
            self.plot_g_stability(
                label = label,
                fig = figs_stability[label],
                scan_param = scan_params[i],
                color = colors[i],
                variable=['teped_list','$T_{e,ped}$ (keV)', 'tped', 1E-3, 1.0],
            )
            self.plot_eped_profiles(
                    label = label,
                    fig = figs_eped_profile_ptot[label],
                    scan_param = scan_params[i],
                    color = colors[i],
                    variable = ['profile_ptot','$p_{tot}$ (kPa)']
                )
            self.plot_eped_profiles(
                    label = label,
                    fig = figs_eped_profile_q[label],
                    scan_param = scan_params[i],
                    color = colors[i],
                    variable = ['profile_q','$q$']
                )
            self.plot_eped_profiles(
                    label = label,
                    fig = figs_eped_profile_j[label],
                    scan_param = scan_params[i],
                    color = colors[i],
                    variable = ['profile_jtot','$J$ ($A/m^2$)']
                )

    def plot_prediction(
            self,
            labels = ['run1'],
            scan_params =['neped'],
            scan_params_label = '$n_{e,ped}# ($10^{19}m^{-3}$)',
            axs = None,
            plot_labels = None,
            legend_title = None,
            legend_location = 'best',   
            ms = 8,
            colors = None,
            additional_labels = None,
            dome_min_frac = 0.05,       # min fraction of the T_e,ped range a mode must span to count as a ballooning dome
            n_peeling_max = 6,          # n<=this is the pure-peeling branch regardless of width (0 -> width-only)
            annotate_limiting_n = True, # write the limiting mode number + dome width next to each marker
            ):
        
        # --------------------
        # Prepare graphics
        # --------------------
        
        if axs is None:
            GRAPHICStools.prep_figure_papers(size=15)
            self.fn = GUItools.FigureNotebook("EPED",  geometry="900x900")
            fig = self.fn.add_figure(label="Pedestal Top")
            axs = fig.subplots(2, 1)

        if colors is None:
            colors = GRAPHICStools.listColors()

        for i,name in enumerate(labels):

            data = self.results[name]

            # --------------------
            # Graph parameters of this scan
            # --------------------
            
            x, ptop, wtop, nlim, dfrac = [], [], [], [], []
            sublabels = data.keys()
            try:
                sublabels = sorted(sublabels, key=lambda x: int(x.split('run')[1]))
            except:
                print('\t> Warning: sublabels could not be sorted numerically.', typeMsg='w')

            for sublabel in sublabels:

                # Grab scanning parameter
                x.append(_to_scalar(data[sublabel][scan_params[i]]))

                # Grab outputs
                ptop.append(_to_scalar(data[sublabel]['ptop']))
                wtop.append(_to_scalar(data[sublabel]['wptop']))

                # Grab the limiting-mode diagnostics (peeling vs ballooning)
                nlim.append(_to_scalar(data[sublabel].get('n_limiting', -1)))
                dfrac.append(_to_scalar(data[sublabel].get('dome_frac', None)))

            # --------------------
            # Plot results of the scan
            # --------------------

            # Connecting line only; the per-point markers below carry a shape that
            # flags whether each pedestal is peeling- (circle) or ballooning-limited (square)
            line_label = name + additional_labels[i] if additional_labels is not None else name
            axs[0].plot(x, ptop, '-', c=colors[i], label=line_label)
            axs[1].plot(x, wtop, '-', c=colors[i])

            for xj, pj, wj, nj, fj in zip(x, ptop, wtop, nlim, dfrac):
                if np.isnan(pj):
                    continue
                mk = _PB_MARKERS[classify_pedestal_limit(nj, dome_frac=fj, n_peeling_max=n_peeling_max, dome_min_frac=dome_min_frac)]
                axs[0].plot([xj], [pj], mk, c=colors[i], ms=ms, mfc=colors[i], mec=colors[i])
                axs[1].plot([xj], [wj], mk, c=colors[i], ms=ms, mfc=colors[i], mec=colors[i])
                if annotate_limiting_n and np.isfinite(nj) and nj > 0:
                    # annotate the limiting n AND the dome width (% of T_ped range): this
                    # is what separates a peeling spike (e.g. "n=20, 1%") from a ballooning
                    # mountain ("n=30, 29%"), so the marker shape is never read in isolation
                    wtxt = f', {fj:.0%}' if np.isfinite(fj) else ''
                    axs[0].annotate(f'n={int(nj)}{wtxt}', (xj, pj), textcoords='offset points',
                                    xytext=(6, 6), fontsize=7, color=colors[i])

            # Plot those with nans as zero with a red cross
            x_nan = [xj for j,xj in enumerate(x) if np.isnan(ptop[j])]
            if len(x_nan) > 0:
                ptop_nan = [0.0 for j,xj in enumerate(x) if np.isnan(ptop[j])]
                wtop_nan = [0.0 for j,xj in enumerate(x) if np.isnan(wtop[j])]
                axs[0].plot(x_nan,ptop_nan,'x', c = colors[i], ms = ms, mew=3, label = 'Problematic' + additional_labels[i] if additional_labels is not None else 'Problematic')
                axs[1].plot(x_nan,wtop_nan,'x', c = colors[i], ms = ms, mew=3)

        ax = axs[0]
        ax.set_xlabel(scan_params_label)
        ax.set_ylabel('$p_{top}$ (kPa)')
        ax.set_ylim(bottom=0)

        # Legend: case lines + a marker-shape key for the pedestal-limiting mode type
        from matplotlib.lines import Line2D
        shape_key = [
            Line2D([0], [0], marker=_PB_MARKERS['peeling'], color='0.4', ls='none', ms=ms,
                   label='Peeling-limited (low-$n$ or spike)'),
            Line2D([0], [0], marker=_PB_MARKERS['ballooning'], color='0.4', ls='none', ms=ms,
                   label='Ballooning-limited (high-$n$ dome)'),
        ]
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles + shape_key, labels_ + [h.get_label() for h in shape_key],
                  loc=legend_location, title=legend_title)

        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1]

        ax.set_xlabel(scan_params_label)
        ax.set_ylabel('$w_{top}$ ($\\psi_{pol,N}$)')
        ax.set_ylim(bottom=0)
        ax.legend(loc=legend_location, title=legend_title)
        GRAPHICStools.addDenseAxis(ax)

        plt.tight_layout()
        
    def _grab_axis_sublabels(
        self,
        data_master,
        fig = None,
        axs = None,
        ):

        if axs is None:
            
            max_sublabels = len(data_master.keys())

            if fig is None:
                GRAPHICStools.prep_figure_papers(size=14)
                self.fn = GUItools.FigureNotebook("EPED Stability", geometry="1900x1600")
                fig = self.fn.add_figure(label="EPED Stability")

            # Arrange panels in a near-square grid, filled left-to-right, top-to-bottom
            ncols = int(np.ceil(np.sqrt(max_sublabels)))
            nrows = int(np.ceil(max_sublabels / ncols))

            mosaic = []
            k = 1
            for _ in range(nrows):
                row = []
                for _ in range(ncols):
                    if k <= max_sublabels:
                        row.append(f"ax{k}")
                        k += 1
                    else:
                        row.append(".")
                mosaic.append(row)

            axs = fig.subplot_mosaic(mosaic, sharex=False, sharey=False)
            # Extra breathing room between panels
            fig.subplots_adjust(wspace=0.4, hspace=0.9)

        # Plot each sublabel into its panel index; overlay curves from each label
        
        sublabels = data_master.keys()
        try:  
            sublabels = sorted(sublabels, key=lambda x: int(x.split('run')[1]))
        except: 
            print('\t> Warning: sublabels could not be sorted numerically.', typeMsg='w')
        
        return axs, sublabels
        
    def plot_g_stability(
        self,
        label = 'run1',
        scan_param = 'neped',
        fig = None,
        axs = None,
        g_base = 0.03,
        color = 'b',
        variable = ['teped_list','$T_{e,ped}$ (keV)', 'tped', 1E-3, 1.0],
    ):
        
        data_master = self.results[label]
        
        axs, sublabels = self._grab_axis_sublabels(
            data_master,
            fig = fig,
            axs = axs,
        )
        
        for j, sublabel in enumerate(sublabels):
            
            ax = axs[f'ax{j+1}']
            
            data = data_master[sublabel]
        
            n = np.array(data['nmodes'])
            h = np.array(data[variable[0]]) * variable[3]
            
            g = np.array(data['gamma'])
            
            colors = GRAPHICStools.listColors()
            
            for mode in range(n.shape[0]):
                ax.plot(h, g[:, mode], '-', c=colors[mode], lw=0.5, label=f'n = {n[mode]}' if j==0 else None)
            
            if j == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # Plot prediction
            xbase = _to_scalar(data[variable[2]]) * variable[4]
            ax.plot([xbase], [g_base], '-s', c=color, ms=12)

            # Plot criterion
            ax.axhline(g_base, color='k', ls='--', lw=1.0)

            # Plot starting point
            ax.axvline(h[0], color='k', ls='--', lw=0.5)

            ax.set_xlabel(variable[1])
            ax.set_ylabel('$\\gamma/\\omega_A$')
            ax.set_title(f'{scan_param} = {_to_scalar(data[scan_param])}', fontsize=10)
            ax.set_ylim([0,g_base*2.0])
            ax.set_xlim(left=0)
            GRAPHICStools.addDenseAxis(ax)
            
    def plot_eped_profiles(
        self,
        label = 'run1',
        scan_param = 'neped',
        fig = None,
        axs = None,
        color = 'b',
        variable = ['profile_ptot','$p_{tot}$ (kPa)'],
    ):
        
        data_master = self.results[label]
        
        axs, sublabels = self._grab_axis_sublabels(
            data_master,
            fig = fig,
            axs = axs,
        )
        
        for j, sublabel in enumerate(sublabels):
            
            ax = axs[f'ax{j+1}']
            
            data = data_master[sublabel]
            
            psin = np.array(data['profile_psin'])
            p = np.array(data[variable[0]])
            
            teped = np.array(data['teped_list'])* 1E-3
            teped_base = _to_scalar(data['tped'])

            minwidth = 1-_to_scalar(data['wptop'])
            
            for iheight in range(p.shape[0]):
                
                is_it_on_point = abs(teped[iheight] - teped_base) < 0.01
                
                alpha_case = 1.0 if is_it_on_point else (0.3 if teped[iheight] < teped_base else 0.05)
                lw = 2.0 if is_it_on_point else 0.5
                ax.plot(psin[iheight,:], p[iheight,:], '-', c=color, lw=lw, alpha=alpha_case)
            
            if variable[0] == 'profile_ptot':
                ax.plot([minwidth], [data['ptop']], 's', c='k', ms=8)
                ax.plot([1-data['wpped']], [data['pped']], 's', c='k', ms=8)
            
            ax.set_xlabel("$\\psi_N$")
            ax.set_ylabel(variable[1])
            ax.set_title(f'{scan_param} = {_to_scalar(data[scan_param])}', fontsize=10)
            ax.set_xlim([minwidth-0.03,1.0])
            GRAPHICStools.addDenseAxis(ax)
            
# ************************************************************************************************************
# ************************************************************************************************************

def convert_to_dimensional(df):
    #ee = 1.60217663e-19
    mu0 = 1.25663706127e-6
    df['a'] = df['r'] / df['epsilon']
    df['ip'] = 1.0e-6 * (2.0 * np.pi * np.square(df['a']) * df['kappa'] * df['bt']) / (df['qstar'] * df['r'] * mu0)
    df['neped'] = 10.0 * df['fgped'] * df['ip'] / (np.pi * np.square(df['a']))
    df['nesep'] = 0.25 * df['neped']
    #df['teped'] = 2500 * df['bt'] * df['ip'] * df['betan'] / (3 * df['a'] * 1.5 * df['neped'])
    #df['teped'] = df['teped'].clip(upper=8000)
    df['teped'] = df['r'] * 0.0 - 1.0
    return df


def convert_to_dimensionless(df):
    mu0 = 1.25663706127e-6
    df['epsilon'] = df['r'] / df['a']
    df['fgped'] = df['neped'] * np.pi * np.square(df['a']) / (10.0 * df['ip'])
    df['qstar'] = (2.0 * np.pi * np.square(df['a']) * df['kappa'] * df['bt']) / (1.0e6 * df['ip'] * df['r'] * mu0)
    df['nesep'] = 0.25 * df['neped']
    df['teped'] = df['r'] * 0.0 - 1.0
    return df


def setup_eped(output_path, inputs_list, template_path):

    output_path = Path(output_path).resolve()  # Ensure absolute path
    output_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(['cp', str(template_path.resolve() / 'exec_eped.sh'), str(output_path)])
    subprocess.run(['cp', str(template_path.resolve() / 'submit_eped_array_psfc.batch'), str(output_path)])
    subprocess.run(['cp', str(template_path.resolve() / 'postprocessing.py'), str(output_path)])
    rpaths = []

    for run_num, inputs in enumerate(inputs_list):
        run_id = f'run{run_num + 1:03d}'
        rpath = output_path / run_id  # Construct the absolute path for the run directory
        subprocess.run(['cp', '-r', str(template_path.resolve() / 'eped_run_template'), str(rpath)])

        # Edit input file
        input_file = rpath / 'eped.input'
        contents = f90nml.read(str(input_file))
        for param, value in inputs.items():
            contents['eped_input'][param] = value
        contents.write(str(input_file), force=True)

        rpaths.append(rpath)

    #logger.info(f'{len(inputs_list)} Runs created at {output_path}')

    return rpaths


def setup_array_batch(launch_path, rpaths, maxqueue=5):

    # Convert to Path object and ensure absolute path
    launch_path = Path(launch_path).resolve()
    
    s = ''
    for path in rpaths:
        if s:
            s += '\n'
        s += f'"./exec_eped.sh {path.resolve()}"'
    
    # Use proper Path object for file operations
    batch_file = launch_path / 'submit_eped_array_psfc.batch'
    with batch_file.open('r') as f:
        content = f.read()
        new_content = re.sub('<numruns>', str(len(rpaths) - 1), content)
        new_content = re.sub('<maxqueue>', str(maxqueue), new_content)
        new_content = re.sub('<rundir>', str(launch_path), new_content)  # Convert to string for substitution
        new_content = re.sub('#<launchdirs>', s, new_content)
    with batch_file.open('w') as f:
        f.write(new_content)

    #logger.info('Batch array created')

    return batch_file


def modify_eped_config(config_file, file_to_write, parameters_to_change=None):
    """Minimal EPED config editor.

    Replaces lines like "NMODES = ..." with the provided values and writes a new file.
    Preserves indentation and inline "# ..." comments.
    """

    config_file = Path(config_file)
    file_to_write = Path(file_to_write)
    parameters_to_change = {} if parameters_to_change is None else dict(parameters_to_change)

    text = config_file.read_text()

    def fmt(v):
        if v is None:
            return ""
        try:
            if isinstance(v, np.generic):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, (list, tuple, set, np.ndarray)):
            return " ".join(str(x) for x in v)
        if isinstance(v, str):
            s = v.strip()
            if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
                s = s[1:-1].replace(",", " ")
                s = " ".join(s.split())
                return s
            return v
        return str(v)

    for key, value in parameters_to_change.items():
        rhs = fmt(value)
        # Replace all occurrences of KEY = ... (any indentation), keep inline comments
        pattern = re.compile(
            rf"^(?P<indent>\s*){re.escape(str(key))}\s*=\s*(?P<val>.*?)(?P<comment>\s+#.*)?$",
            re.M,
        )

        def _repl(m):
            indent = m.group("indent")
            comment = m.group("comment") or ""
            return f"{indent}{key} = {rhs}{comment}"

        text = pattern.sub(_repl, text)

    file_to_write.parent.mkdir(parents=True, exist_ok=True)
    file_to_write.write_text(text)

# Marker shapes used to flag the pedestal-limiting mode on the "Pedestal Top" plot
_PB_MARKERS = {'peeling': 'o', 'ballooning': 's', 'none': '*'}


def _limiting_dome_frac(stability, teped_axis, step, mode_idx, threshold, foot_frac=0.3):
    '''
    Fraction of the explored T_e,ped range over which the limiting mode stays above
    foot_frac*threshold, measured as the contiguous band of pedestal heights containing
    the crossing `step`.

    This is the "mountain-vs-spike" discriminator, and it is what makes the
    peeling/ballooning call robust rather than a bare cut on the mode number n. A
    coherent ballooning mode grows smoothly over a broad span of pedestal temperature
    -> a wide "mountain" (here ~0.3 of the explored range). An isolated peeling /
    numerical spike is a violent ~1 grid-cell excursion -> a narrow sliver (<~0.02).
    The fraction is used instead of a raw bin count on purpose: it is independent of
    the height-grid resolution (EPED auto-scales the T_e,ped scan, so keV-per-bin
    differs several-fold across a density scan), so the cut survives a grid change.
    '''
    if step < 0:
        return 0.0
    foot = foot_frac * threshold
    col = np.asarray(stability)[:, mode_idx].astype(float)
    above = np.where(np.isnan(col), False, col > foot)
    if not above[step]:
        return 0.0
    lo = hi = step
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    while hi + 1 < above.shape[0] and above[hi + 1]:
        hi += 1
    T = np.asarray(teped_axis).astype(float).ravel()
    total = abs(T[-1] - T[0])
    if total <= 0:
        return 0.0
    return abs(T[hi] - T[lo]) / total


def classify_pedestal_limit(n_limiting, dome_frac=None, n_peeling_max=6, dome_min_frac=0.05):
    '''
    Label the pedestal-limiting MHD mode as 'peeling' or 'ballooning'.

    This is deliberately NOT a bare cut on the toroidal mode number n -- a single n
    test mislabels isolated high-n spikes (which can sit at the operating point in
    low-density / low-collisionality cases) as ballooning. The active discriminator is
    the *width* of the limiting mode's unstable band:

      1. dome_frac   : fraction of the explored T_e,ped range that the limiting mode
                       spans (see _limiting_dome_frac) -- a coherent ballooning
                       "mountain" is broad, an isolated spike is a sliver. PRIMARY test.
      2. n_limiting  : the dominant toroidal mode number at the crossing -- used only as
                       a light physical floor (the pure-peeling branch is the lowest-n,
                       current-driven modes), NOT as the main criterion.

    A case is 'ballooning' only if the limiter is a coherent broad dome
    (dome_frac >= dome_min_frac) AND above the pure-peeling floor (n > n_peeling_max).
    Everything else -- low-n current-driven crossings, and high-n crossings that are
    spikes rather than mountains -- is 'peeling'. Returns 'none' if no unstable crossing.

    CONVENTION (calibrated on the EPED test scan, tunable -- not a hard physics law):
        dome_min_frac = 0.05  -> spikes span ~0.01 of the range, the real dome ~0.29,
                                 so the cut sits in a wide (>5x either side) gap.
        n_peeling_max = 6     -> n<=6 is treated as the pure-peeling branch regardless
                                 of width; set to 0 to make the call purely width-based.
    The limiting n and width are annotated on the plot so any call can be cross-checked
    against the Stability tab.
    '''
    n = _to_scalar(n_limiting)
    if not np.isfinite(n) or n < 0:
        return 'none'
    # dome_frac=None (e.g. legacy results without the metric) -> treat width as ample.
    f = _to_scalar(dome_frac) if dome_frac is not None else np.inf
    is_dome = (not np.isfinite(f)) or (f >= dome_min_frac)
    if (n > n_peeling_max) and is_dome:
        return 'ballooning'
    return 'peeling'


def limiting_mode_from_dataset(data, **classify_kwargs):
    '''
    Extract the pedestal-limiting-mode classification from a single postprocessed EPED
    results dataset (one entry of EPED.results[label][sublabel]).

    Returns a dict {'limiting_mode', 'n_limiting', 'dome_frac'}, or None if the dataset
    predates the metric (older MITIM postprocessing without `n_limiting`). The None is
    the backward-compat signal -- callers store it as-is instead of inventing a label.
    '''
    if 'n_limiting' not in getattr(data, 'data_vars', {}):
        return None
    n_lim = int(_to_scalar(data['n_limiting']))
    if n_lim < 0:
        return {'limiting_mode': 'none', 'n_limiting': n_lim, 'dome_frac': None}
    dome_frac = _to_scalar(data['dome_frac']) if 'dome_frac' in data.data_vars else None
    if dome_frac is not None and not np.isfinite(dome_frac):
        dome_frac = None
    label = classify_pedestal_limit(n_lim, dome_frac=dome_frac, **classify_kwargs)
    return {
        'limiting_mode': label,
        'n_limiting': n_lim,
        'dome_frac': float(dome_frac) if dome_frac is not None else None,
    }


def postprocess_eped(data, diamagnetic_stab_rule, stability_threshold, dome_foot_frac=0.3):
    '''
    Note that this postprocessing uses the diagmanetic stabilization rule to determine stability, may not match EPED
    '''


    coords = {k: data[k].values for k in ['dim_height', 'dim_widths', 'dim_nmodes', 'dim_rho', 'dim_three', 'dim_one']}
    data = data.assign_coords(coords)

    x = data['eq_betanped'].data
    index = np.where(x < 0)[0]
    if diamagnetic_stab_rule == 'G':
        y = data['gamma'].data.copy()
    elif diamagnetic_stab_rule in ['GH', 'HG']:
        y = data['gamma_PB'].data.copy()
        y *= data['gamma'].data.copy()
    elif diamagnetic_stab_rule == 'H':
        y = data['gamma_PB'].data.copy()
    else:
        y = data['gamma'].data.copy()
    y[index, :] = np.nan

    data['stability'] = (('dim_height', 'dim_nmodes'), y)
    y0 = np.nanmax(y, 1)
    y0 = np.where(y0 == None, 0, y0)
    indices = np.where(y0 > stability_threshold)[0]
    if len(indices):
        step = indices[0]
    else:
        step = -1

    dims = ('dim_one')
    data['stability_rule'] = (dims, [diamagnetic_stab_rule])
    data['stability_threshold'] = (dims, np.array([stability_threshold]))
    if step > 0:
        data['stability_index'] = (dims, np.array([step]))
        # Limiting crossing diagnostics (data-driven; the peeling/ballooning label is
        # applied later by classify_pedestal_limit). `n_limiting` is the toroidal mode
        # number with the largest growth rate at the limiting height `step` -- by
        # construction the mode that first crosses the threshold and caps the pedestal.
        # `dome_frac` is how broad that mode's unstable band is, as a fraction of the
        # explored T_e,ped range, distinguishing a coherent ballooning "mountain" (broad)
        # from an isolated spike (sliver).
        n_modes_here = np.asarray(data['nmodes']).ravel()
        mode_idx = int(np.nanargmax(y[step, :]))
        data['n_limiting'] = (dims, np.array([int(n_modes_here[mode_idx])]))
        data['dome_frac'] = (dims, np.array([_limiting_dome_frac(y, data['teped_list'].data, step, mode_idx, stability_threshold, dome_foot_frac)]))
        data['pped'] = (dims, np.array([data['eq_pped'].data[step] * 1.0e3]))
        data['ptop'] = (dims, np.array([data['eq_ptop'].data[step] * 1.0e3]))
        data['tped'] = (dims, np.array([data['eq_tped'].data[step]]))
        data['ttop'] = (dims, np.array([data['eq_ttop'].data[step]]))
        data['wpped'] = (dims, np.array([data['eq_wped_psi'].data[step]]))
        data['wptop'] = (dims, np.array([data['eq_wped_psi'].data[step] * 1.5]))
        data['wrped'] = (dims, np.array([data['eq_wped_rho'].data[step]]))
        if np.any(data['tesep'].data < 0):
            data['tesep'] = (dims, np.array([75.0]))
            data['nesep'] = 0.25 * data['neped']
    else:
        print(f'\t> Warning: No stable solution found in EPED postprocessing using the diamagnetic stabilization rule ({diamagnetic_stab_rule} > {stability_threshold}), proceed with caution', typeMsg='w')
        data['stability_index'] = (dims, np.array([-1]))
        data['n_limiting'] = (dims, np.array([-1]))
        data['dome_frac'] = (dims, np.array([0.0]))
        data['pped'] = (dims, np.array([np.nan]))
        data['tped'] = (dims, np.array([np.nan]))
        data['ptop'] = (dims, np.array([np.nan]))
        data['ttop'] = (dims, np.array([np.nan]))
        data['wpped'] = (dims, np.array([np.nan]))
        data['wptop'] = (dims, np.array([np.nan]))
        data['wrped'] = (dims, np.array([np.nan]))

    return data

def read_eped_file(ipaths):
    invars = ['ip', 'bt', 'r' , 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep']
    data_arrays = []
    for ipath in ipaths:
        dummy_coords = {
            'dim_height': np.empty((0, ), dtype=int),
            'dim_nmodes': np.empty((0, ), dtype=int),
            'dim_widths': np.empty((0, ), dtype=int),
            'dim_rho': np.empty((0, ), dtype=int),
            'dim_three': np.empty((0, ), dtype=int),
            'dim_one': np.arange(1),
        }
        set_inputs = f90nml.read(str(ipath.parent.parent.parent / 'eped.input'))
        dummy_vars = {k: (['dim_one'], [v]) for k, v in set_inputs['eped_input'].items() if k in invars}
        data = xr.Dataset(coords=dummy_coords, data_vars=dummy_vars)
        if ipath.is_file():
            with xr.open_dataset(f'{ipath.resolve()}', engine='netcdf4') as ds:
                data = ds.load()
            data = postprocess_eped(data, 'G', 0.03)
        data_arrays.append(data.expand_dims({'filename': [ipath.parent.parent.parent.name]}))

    dataset = xr.merge(data_arrays, join='outer', fill_value=np.nan).sortby('filename')
    return dataset

def launch_eped_slurm(input_params, scan_params, nscan, output_path, template_path, run_tag, wait=False): 
    ivars = ['ip', 'bt', 'r', 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep', 'teped']
    input_params.update(scan_params)
    data = {}
    for var, val in input_params.items():
        if isinstance(val, (tuple, list, np.ndarray)) and len(val) > 1:
            data[var] = np.linspace(val[0], val[1], nscan)
        else:
            data[var] = np.zeros((nscan, )) + val
    #if scan_var == 'qstar': # Use for ip scan
    #    data['fgped'] = (0.5 / 3.5) * data['qstar']
    inp = pd.DataFrame(data=data, index=pd.RangeIndex(nscan))
    #inp = convert_to_dimensional(inp)
    inputs  = [{ivar: inp[ivar].iloc[i] for ivar in ivars} for i in range(len(inp))]
    run_paths = setup_eped(output_path, inputs, template_path)
    spath = setup_array_batch(output_path, run_paths)
    inp.to_hdf(output_path / f'{output_path.name}.h5', key='/data')
    command = ['sbatch']
    if wait:
        command.append('--wait')
    command.append(f'{spath.resolve()}')
    subprocess.run(command)

    return run_paths

def main():

    rootdir = Path(os.environ.get('PIXI_PROJECT_ROOT', './'))
    run_tag = 'mitim_eped_test'
    base_input_path = Path('./') / 'eped.input'
    scan_params = {
    #    'tesep': [50.0, 300.0],
    }
    nscan = 1
    output_path = Path('./') / f'eped_{run_tag}'
    template_path = rootdir / 'ips-eped-master' / 'template' / 'engaging'
    wait = False

    input_params = f90nml.read(str(base_input_path)).todict().get('eped_input', {})

    launch_eped_slurm(input_params, scan_params, nscan, output_path, template_path, run_tag, wait=wait)


if __name__ == '__main__':
    main()
