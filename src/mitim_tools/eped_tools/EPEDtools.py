import os
import re
import copy
import subprocess
import matplotlib.pyplot as plt
import f90nml
from pathlib import Path
from mitim_tools.misc_tools import FARMINGtools, GRAPHICStools, IOtools, GUItools
from mitim_tools.gacode_tools import PROFILEStools
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
            clean_intermediate_files = False, # If True, EPED deletes each per-height TOQ/ELITE work dir (toq.log, raw peddata,
                                              # eigenfunctions) as soon as its stage is collected (CLEAN_AFTER=1 in all config
                                              # sections). Default keeps them in the remote scratch so failures are diagnosable
                                              # post-mortem (the scratch itself still goes away unless removeScratchFolders=False).
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

        # CLEAN_AFTER appears in every port section; modify_eped_config replaces all occurrences
        if clean_intermediate_files:
            eped_params_override = {**(eped_params_override or {}), 'CLEAN_AFTER': 1}

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

        # Refuse (interactively) to submit a case that would overflow the EPED runner's silent
        # job-table limit (see check_runner_job_limit) — the failure mode is gamma = -1
        # everywhere with exit code 0, i.e. undetectable until read() masks every height
        if len(job_array_indices) > 0:
            check_runner_job_limit(self.template_config_file, eped_params_override)

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
            diamagnetic_stab_rule = 'G',    # 'G'/'H'/'GH': flat cut; 'W': EPED1 gamma > C*omega_*i(n)/2
            stability_threshold = 0.03,     # flat rules: cut on gamma/omega_A; 'W': calibration factor C
            gacode_state = None,            # companion plasma state (path or gacode_state), required by 'W'
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
                data = postprocess_eped(ds, diamagnetic_stab_rule, stability_threshold, gacode_state=gacode_state)

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
        tab_color = None,   # FigureNotebook tab color (int index into GRAPHICStools.convert_to_hex_soft,
                            # or one of its color keys). None: give each label its own color, so the tabs
                            # of a label are visually grouped; pass a value to force it on every tab
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
            
        
        # Tab color of each label's own set of figures; the shared "Pedestal Top" tab keeps index 0
        tab_colors = [tab_color if tab_color is not None else i for i in range(len(labels))]

        fig = self.fn.add_figure(label="Pedestal Top", tab_color=tab_color if tab_color is not None else 0)
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
        for i, label in enumerate(labels):
            figs_stability[label] = self.fn.add_figure(label="EPED Stability (teped) - " + label, tab_color=tab_colors[i])
            figs_eped_profile_ptot[label] = self.fn.add_figure(label="EPED profiles (ptot) - " + label, tab_color=tab_colors[i])
            figs_eped_profile_q[label] = self.fn.add_figure(label="EPED profiles (q) - " + label, tab_color=tab_colors[i])
            figs_eped_profile_j[label] = self.fn.add_figure(label="EPED profiles (J) - " + label, tab_color=tab_colors[i])

        
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
        # This panel mirrors the one above without labels (the legend lives there), so only
        # legend it when something here actually carries one
        if ax.get_legend_handles_labels()[0]:
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
            
            # Plot prediction. Under the 'W' rule the crossing happens on the limiting mode's
            # own threshold curve, not at the flat g_base -- place the marker there
            xbase = _to_scalar(data[variable[2]]) * variable[4]
            ybase = g_base
            if 'stability_threshold_n' in data.data_vars:
                step = int(_to_scalar(data['stability_index']))
                n_lim = int(_to_scalar(data['n_limiting']))
                if step >= 0 and n_lim > 0:
                    ybase = np.array(data['stability_threshold_n'])[step, int(np.where(n == n_lim)[0][0])]
            ax.plot([xbase], [ybase], '-s', c=color, ms=7, zorder=10)

            # Plot criterion: flat cut, or the per-mode threshold of the 'W' (omega_*) rule.
            # Logarithmic y-window sized so nothing decision-relevant is clipped (every
            # threshold curve and the selection marker -- the 'W' thresholds span ~n_max/n_min
            # by construction, unshowable on a linear axis) while staying robust to the huge
            # garbage gamma of unconverged-ELITE "forest" regions
            g_pos = g[np.isfinite(g) & (g > 0)]
            if 'stability_threshold_n' in data.data_vars:
                thr = np.array(data['stability_threshold_n'])
                for mode in range(n.shape[0]):
                    ax.plot(h, thr[:, mode], '--', c=colors[mode], lw=0.5)
                # thr is all-nan only if every equilibrium of the scan was degenerate
                thr_pos = thr[np.isfinite(thr) & (thr > 0)]
                ytop = 2.0 * thr_pos.max() if thr_pos.size else g_base * 2.0
                ybot = 0.5 * thr_pos.min() if thr_pos.size else g_base / 10.0
            else:
                ax.axhline(g_base, color='k', ls='--', lw=1.0)
                # show the gamma domes above the flat cut without letting forest garbage set the scale
                ytop = min(2.0 * g_pos.max(), 20.0 * g_base) if g_pos.size else g_base * 2.0
                ybot = g_base / 10.0
            if np.isfinite(ybase) and ybase > 0:
                ytop = max(ytop, 1.5 * ybase)

            # Plot starting point
            ax.axvline(h[0], color='k', ls='--', lw=0.5)

            ax.set_xlabel(variable[1])
            ax.set_ylabel('$\\gamma/\\omega_A$')
            ax.set_title(f'{scan_param} = {_to_scalar(data[scan_param])}', fontsize=10)
            ax.set_yscale('log')
            ax.set_ylim([ybot, ytop])
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

            # When postprocess_eped found no stable solution in the scanned window, the SELECTED
            # pedestal is undefined (tped/ptop/wptop = nan) but every per-height profile exists.
            # Draw the full stack anyway, over the widest pedestal window in the scan.
            has_selection = np.isfinite(teped_base)

            minwidth = 1-_to_scalar(data['wptop'])
            if not np.isfinite(minwidth):
                wped_all = np.array(data['eq_wped_psi'])
                wped_all = wped_all[np.isfinite(wped_all) & (wped_all > 0)]
                minwidth = 1 - 1.5 * wped_all.max() if wped_all.size else 0.85

            for iheight in range(p.shape[0]):

                is_it_on_point = has_selection and abs(teped[iheight] - teped_base) < 0.01

                # Without a selected pedestal no height is privileged, so do not fade any out
                alpha_case = (1.0 if is_it_on_point else (0.3 if teped[iheight] < teped_base else 0.05)) if has_selection else 0.3
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


# Hardcoded job-table size of the EPED driver's runner (MAXJOB in run_parallel.cpp): one ELITE
# job is dispatched per (pedestal height, mode number) pair and there is NO bounds check — the
# excess beyond this limit is silently never run (exit code still 0), leaving gamma = -1 for
# ALL pairs in the output netCDF.
_EPED_RUNNER_MAXJOB = 1024


def check_runner_job_limit(template_config_file, eped_params_override=None):
    '''
    Estimate num_heights x num_modes of the case about to be submitted (override wins over the
    template config file) and ask before proceeding if it exceeds the runner's silent job-table
    limit. In non-interactive (batch) contexts the question raises, which is the desired loud
    failure instead of the silent gamma = -1 one.
    '''
    override = eped_params_override or {}

    def _effective(key):
        if key in override:
            v = override[key]
            if isinstance(v, str):
                # accept the same brace format modify_eped_config does: '{0.1, 1.4, 0.01}'
                v = v.replace('{', ' ').replace('}', ' ').replace(',', ' ').split()
            return [float(x) for x in v]
        m = re.search(rf"^\s*{key}\s*=\s*([^#\n]+)", Path(template_config_file).read_text(), re.M)
        return [float(x) for x in m.group(1).split()]

    tmin, tmax, tstep = _effective('TEPED_BOUND')
    num_heights = int(round((tmax - tmin) / tstep)) + 1  # endpoints included
    num_modes = len(_effective('NMODES'))

    njobs = num_heights * num_modes
    if njobs > _EPED_RUNNER_MAXJOB:
        if not print(
            f'\t> TEPED_BOUND gives {num_heights} heights x {num_modes} modes = {njobs} ELITE jobs, over the '
            f'runner job limit ({_EPED_RUNNER_MAXJOB}, hardcoded MAXJOB with no bounds check): ELITE would '
            f'silently never run and gamma = -1 would be written for ALL (height, mode) pairs. '
            f'Reduce NMODES or the TEPED_BOUND window. Proceed anyway?',
            typeMsg='q',
        ):
            raise Exception('[MITIM] EPED launch aborted: num_heights x num_modes exceeds the runner job limit')


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
    the crossing `step`. `threshold` is a scalar for the flat rules, or the limiting mode's
    per-height threshold column for the 'W' (omega_*) rule.

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


# Constants for the omega_* (diamagnetic) stability rule, SI
_E_C = 1.602176634e-19      # elementary charge [C]
_MU0 = 4.0e-7 * np.pi       # vacuum permeability [H/m]
_AMU_KG = 1.66053907e-27    # atomic mass unit [kg]

# Barrier window over which omega_*i is maximized: psi_N in [1 - _BARRIER_WIDTHS*w_ped_psi, 1]
_BARRIER_WIDTHS = 2.0


def _eped_profiles_at_height(data, ih):
    '''
    Per-pedestal-height EPED profiles in SI, reordered to ascending psi_N.

    UNITS in the netCDF: profile_ne [1e19 m^-3], profile_Te/Ti [keV], profile_ptot [kPa];
    converted here to [m^-3], [eV], [Pa]. The last stored point is a psi_N=0 padding point
    (q = ne = Te = 0) and is dropped.
    '''
    out = {}
    for key, conversion in (('psin', 1.0), ('rho', 1.0), ('q', 1.0), ('ne', 1e19),
                            ('Te', 1e3), ('Ti', 1e3), ('ptot', 1e3)):
        out[key] = np.array(data[f'profile_{key}'])[ih][:-1][::-1] * conversion
    return out


def _omega_star_threshold(data, gacode_state, calibration_factor):
    '''
    Per-(height, toroidal mode number) EPED1 diamagnetic-stabilization threshold on gamma/omega_A:

        threshold(h,n) = C * 0.5 * max_barrier[ omega_*i(h,n) ] / omega_A(h)

    so that the stability rule reads  gamma > C * omega_*i(n)/2  , with
    omega_*i = (n / (Z_i e n_i)) dp_i/dpsi  (EPED1 criterion, Snyder et al., Phys. Plasmas 16,
    056118 (2009); omega_* convention as in Saarelma et al., Nucl. Fusion 52, 103020 (2012)).

    Unlike the flat 'G' cut, the threshold grows ~linearly with n, so high-n modes are
    progressively harder to declare limiting.

    `calibration_factor` C is the O(1) calibration knob (nominal 1.0): the ABSOLUTE
    normalization of omega_*/omega_A against ELITE's internal Alfven normalization is
    uncertain to an O(1) factor -- the robust content of this rule is the ~n scaling.
    '''
    state = gacode_state if hasattr(gacode_state, 'profiles') else PROFILEStools.gacode_state(gacode_state)
    torfluxa = float(np.asarray(state.profiles['torfluxa(Wb/radian)']).ravel()[0])  # [Wb/rad]

    n_modes = np.array(data['nmodes'])
    z, zi, zeff = _to_scalar(data['z']), _to_scalar(data['zi']), _to_scalar(data['zeffped'])
    m, mi = _to_scalar(data['m']), _to_scalar(data['mi'])                # [amu]
    B, R = _to_scalar(data['bt']), _to_scalar(data['r'])                 # [T], [m]
    wped_psi = np.array(data['eq_wped_psi'])

    threshold = np.full(np.array(data['gamma']).shape, np.nan)
    for ih in range(threshold.shape[0]):

        pr = _eped_profiles_at_height(data, ih)

        # TOQ equilibria at very high T_e,ped come back degenerate (q=0, repeated psi_N), and
        # failed heights carry fill values (eq_wped_psi <= 0, which would empty the barrier
        # window below). Leaving their threshold at NaN keeps them from ever setting the
        # pedestal limit, consistent with the eq_betanped < 0 masking of the growth rates.
        if not (np.all(pr['q'] > 0) and np.all(np.diff(pr['psin']) > 0) and wped_psi[ih] > 0):
            continue

        # Main-ion and impurity densities from quasineutrality with EPED's own z, zi, zeffped
        # (p_i = n_i T_i, NOT the ptot/2 shortcut, which is ~18% off on the validation case)
        n_imp = pr['ne'] * (zeff - z) / (zi * (zi - z))
        n_i = (pr['ne'] - zi * n_imp) / z
        p_i = n_i * pr['Ti'] * _E_C                                      # [m^-3] * [eV] * [C] -> [Pa]

        # psi_N is linear in psi, so d/dpsi = (d/dpsi_N)/delta_psi. The netCDF carries no
        # dimensional psi and no torfluxa, so delta_psi = psi_edge - psi_axis is rebuilt from
        # the netCDF q profile plus the companion input.gacode torfluxa, via
        # dPsi = dPhi/q with Phi = torfluxa*rho^2. CONVENTION: this runs ~7% below the
        # input.gacode polflux(edge)-polflux(axis) on the validation case; omega_* ~ 1/delta_psi.
        delta_psi = float(np.trapezoid(2.0 * torfluxa * pr['rho'] / pr['q'], pr['rho']))

        omega_star_per_n = np.abs(np.gradient(p_i, pr['psin']) / delta_psi) / (z * _E_C * n_i)  # [rad/s]

        rho_m = (n_i * m + n_imp * mi) * _AMU_KG                         # [amu] -> [kg/m^3]
        omega_A = B / np.sqrt(_MU0 * rho_m) / R                          # v_A/R0 [rad/s], local

        # EPED1 evaluates omega_*i at its maximum across the pedestal barrier
        barrier = pr['psin'] >= 1.0 - _BARRIER_WIDTHS * wped_psi[ih]
        j = np.where(barrier)[0][int(np.argmax(omega_star_per_n[barrier]))]

        # omega_A taken with rho_m AT THE BARRIER PEAK; the alternative on-axis rho_m
        # would scale every threshold by ~1.44x on the validation case
        threshold[ih, :] = 0.5 * n_modes * omega_star_per_n[j] / omega_A[j]

    return calibration_factor * threshold


def postprocess_eped(data, diamagnetic_stab_rule, stability_threshold, dome_foot_frac=0.3, gacode_state=None):
    '''
    Note that this postprocessing uses the diagmanetic stabilization rule to determine stability, may not match EPED

    Rules 'G', 'H', 'GH'/'HG' apply a FLAT cut (max-over-n of the chosen growth-rate metric
    against `stability_threshold`). Rule 'W' applies the EPED1 diamagnetic criterion
    gamma > C*omega_*i(n)/2, in which case `stability_threshold` plays the role of the O(1)
    calibration factor C and `gacode_state` (path or gacode_state) must be the companion
    plasma state (see _omega_star_threshold).
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
    elif diamagnetic_stab_rule == 'W':
        if gacode_state is None:
            raise ValueError("[MITIM] The 'W' (omega_*) stability rule needs a companion gacode_state (path or gacode_state) to get the dimensional psi and mass density")
        y = data['gamma'].data.copy()
    else:
        y = data['gamma'].data.copy()
    y[index, :] = np.nan

    data['stability'] = (('dim_height', 'dim_nmodes'), y)

    if diamagnetic_stab_rule == 'W':
        # Per-(height, n) threshold -> the crossing test must be elementwise, not on max-over-n
        threshold = _omega_star_threshold(data, gacode_state, stability_threshold)
        data['stability_threshold_n'] = (('dim_height', 'dim_nmodes'), threshold)
        excess = y - threshold
        excess = np.where(np.isnan(excess), -np.inf, excess)
        indices = np.where(np.max(excess, axis=1) > 0)[0]
    else:
        threshold = stability_threshold
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
        # With a per-(height,n) threshold the limiting mode is the largest EXCESS over the
        # threshold, not the largest growth rate (for a flat cut the two are identical).
        n_modes_here = np.asarray(data['nmodes']).ravel()
        if diamagnetic_stab_rule == 'W':
            mode_idx = int(np.argmax(excess[step, :]))
            threshold_dome = threshold[:, mode_idx]
        else:
            mode_idx = int(np.nanargmax(y[step, :]))
            threshold_dome = stability_threshold
        data['n_limiting'] = (dims, np.array([int(n_modes_here[mode_idx])]))
        data['dome_frac'] = (dims, np.array([_limiting_dome_frac(y, data['teped_list'].data, step, mode_idx, threshold_dome, dome_foot_frac)]))
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
        if len(index) == x.shape[0]:
            # Every height was masked upstream (eq_* = -1): the EPED driver failed to parse
            # TOQ's pedestal-top summary on ALL heights. Known cause: TOQ's fixed-width
            # peddata output glues adjacent fields when a value fills its column (neped >= 100
            # in 1e19 units, or very large nu* at cold pedestals) and the driver's
            # read_peddata() whitespace-splits those lines. This is a bookkeeping failure,
            # NOT a stability result -- the equilibria and growth rates are typically fine.
            print(f'\t> Warning: EVERY pedestal height has failed pedestal characterization (eq_* = -1 from EPED/TOQ). Growth rates exist but no height can be reported. Known trigger: peddata fixed-width fields gluing at high neped (>=100e19) or high nu* -- a parse bug in the EPED driver (toq_io.read_peddata), not a physics failure.', typeMsg='w')
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

def read_eped_file(ipaths, diamagnetic_stab_rule = 'G', stability_threshold = 0.03, gacode_state = None):
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
            data = postprocess_eped(data, diamagnetic_stab_rule, stability_threshold, gacode_state=gacode_state)
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
