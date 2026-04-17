import shutil
import datetime
import json
import time
import os
import copy
import numpy as np
import dill as pickle_dill
from pathlib import Path
from mitim_tools import __version__ as mitim_version
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import GACODEdefaults, NORMtools
from mitim_tools.misc_tools import FARMINGtools, IOtools, LOGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

_RUN_TYPE_ALIASES = {'run': 'normal'}

def _normalize_run_type(run_type):
    return _RUN_TYPE_ALIASES.get(run_type, run_type)

class mitim_simulation:
    '''
    Main class for running GACODE simulations.
    '''

    # Subclasses that want `_run(run_type='submit')` to persist slurm-job metadata
    # (so a later process can re-attach instead of resubmitting) set this to a
    # filename (e.g. "cgyro_submission.json"). Leave as None to opt out.
    _submission_metadata_filename = None

    def __init__(
        self,
        rhos=[None],  # rho locations of interest, e.g. [0.4,0.6,0.8]
    ):
        self.rhos = np.array(rhos) if rhos is not None else None

        # A simulation may have multiple ways to run (e.g. linear, nonlinear, etc) with different outputs, or not desirable to bring everything locally
        self.output_files_simulation = {
            'complete': [],
            'minimal': [],
        } 
        
        self.nameRunid = "0"
        
        self.results, self.scans = {}, {}
        
        self.run_specifications = None
        
        self.NormalizationSets = {'SELECTED': None}

    def prep(
        self,
        mitim_state,                # A MITIM state class
        FolderGACODE,               # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,           # If True, do not use what it potentially inside the folder, run again
        forceIfcold_start=False,    # Extra flag
        ):
        '''
        This method prepares the GACODE run from a MITIM state class by setting up the necessary input files and directories.
        '''

        print("> Preparation run from MITIM state class (direct conversion)")

        if self.run_specifications is None:
            raise Exception("[MITIM] Simulation child class did not define run specifications")

        state_converter = self.run_specifications['state_converter']    # e.g. to_tglf
        input_class     = self.run_specifications['input_class']        # e.g. TGLFinput
        input_file      = self.run_specifications['input_file']         # e.g. input.tglf

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)
        
        if cold_start or not self.FolderGACODE.exists():
            IOtools.askNewFolder(self.FolderGACODE, force=forceIfcold_start)
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if isinstance(mitim_state, str) or isinstance(mitim_state, Path):
            # If a string, assume it's a path to input.gacode
            self.profiles = PROFILEStools.gacode_state(mitim_state)
        else:
            self.profiles = mitim_state
            
        # Keep a copy of the file
        self.profiles.write_state(file=self.FolderGACODE / "input.gacode_torun")

        self.profiles.derive_quantities(mi_ref=md_u)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize from state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Call the method dynamically based on state_converter
        conversion_method = getattr(self.profiles, state_converter)
        self.inputs_files = conversion_method(r=self.rhos, r_is_rho=True)

        for rho in self.inputs_files:
            
            # Initialize class
            self.inputs_files[rho] = input_class.initialize_in_memory(self.inputs_files[rho])
                
            # Write input.tglf file
            self.inputs_files[rho].file = self.FolderGACODE / f'{input_file}_{rho:.4f}'
            self.inputs_files[rho].write_state()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Definining normalizations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print("> Setting up normalizations")
        self.NormalizationSets, cdf = NORMtools.normalizations(self.profiles)

        return cdf
    
    def run(
        self,
        subfolder,  # 'neo1/',
        code_settings=None,
        extraOptions={},
        multipliers={},
        minimum_delta_abs={},
        ApplyCorrections=True,  # Removing ions with too low density and that are fast species
        Quasineutral=False,  # Ensures quasineutrality. By default is False because I may want to run the file directly
        launchSlurm=True,
        cold_start=False,
        forceIfcold_start=False,
        extra_name="exe",
        allocation=None,   # {'resources_per_call': int, 'minutes': int, 'mem': str|None}
        attempts_execution=1,
        only_minimal_files=False,
        run_type = 'normal', # 'normal': send, submit and wait; 'submit': send and submit and do not wait; 'send': send and do not submit; 'prep': do not submit
        additional_files_to_send = None, # Dict (rho keys) of files to send along with the run (e.g. for restart)
        helper_lostconnection=False, # If True, it means that the connection to the remote machine was lost, but the files are there, so I just want to retrieve them not execute the commands
    ):

        run_type = _normalize_run_type(run_type)

        if allocation is None:
            from mitim_tools.misc_tools import SLURMtools
            allocation = {
                "resources_per_call": SLURMtools.CODE_HINTS.get(
                    self.run_specifications.get('code', ''),
                    {}).get("default_resources_per_call", 1),
                "minutes": 10,
            }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare inputs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        code_executor, code_executor_full = self._run_prepare(
            #
            subfolder,
            code_executor={},
            code_executor_full={},
            #
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            #
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            only_minimal_files=only_minimal_files,
            #
            launchSlurm=launchSlurm,
            allocation=allocation,
            #
            additional_files_to_send=additional_files_to_send,
            #
            ApplyCorrections=ApplyCorrections,
            minimum_delta_abs=minimum_delta_abs,
            Quasineutral=Quasineutral,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run NEO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self._run(
            code_executor,
            code_executor_full=code_executor_full,
            code_settings=code_settings,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            allocation=allocation,
            only_minimal_files=only_minimal_files,
            attempts_execution=attempts_execution,
            run_type=run_type,
            helper_lostconnection=helper_lostconnection,
            base_subfolder=subfolder,
        )

        return code_executor_full

    def _run_prepare(
        self,
        # ********************************
        # Required options
        # ********************************
        subfolder_simulation,
        code_executor=None,
        code_executor_full=None,
        # ********************************
        # Run settings
        # ********************************
        code_settings=None,
        extraOptions={},
        multipliers={},
        # ********************************
        # IO settings
        # ********************************
        cold_start=False,
        forceIfcold_start=False,
        only_minimal_files=False,
        # ********************************
        # Slurm settings (for warnings)
        # ********************************
        launchSlurm=True,
        allocation=None,
        # ********************************
        # Additional files to send (e.g. restarts). Must be a dictionary with rho keys
        # ********************************
        additional_files_to_send = None,
        # ********************************
        # Additional settings to correct/modify inputs
        # ********************************
        **kwargs_control
        ):

        if allocation is None:
            from mitim_tools.misc_tools import SLURMtools
            allocation = {
                "resources_per_call": SLURMtools.CODE_HINTS.get(
                    self.run_specifications.get('code', ''),
                    {}).get("default_resources_per_call", 1),
                "minutes": 5,
            }

        if self.run_specifications is None:
            raise Exception("[MITIM] Simulation child class did not define run specifications")

        # Because of historical relevance, I allow both TGLFsettings and code_settings #TODO #TOREMOVE
        if "TGLFsettings" in kwargs_control:
            if code_settings is not None:
                raise Exception('[MITIM] Cannot use both TGLFsettings and code_settings')
            else:
                code_settings = kwargs_control["TGLFsettings"]
                del kwargs_control["TGLFsettings"]
        # ------------------------------------------------------------------------------------

        if code_executor is None:
            code_executor = {}
        if code_executor_full is None:
            code_executor_full = {}

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare for run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        rhos = self.rhos

        inputs = copy.deepcopy(self.inputs_files)
        Folder_sim = self.FolderGACODE / subfolder_simulation

        output_files_new = []
        for i in self.output_files_simulation["complete"]:
            if "mitim.out" not in i:
                output_files_new.append(i)
        self.output_files_simulation["complete"] = output_files_new

        # ------------------------------------------------
        # Selection of files to retrieve
        # ------------------------------------------------
        
        if only_minimal_files:
            filesToRetrieve = self.output_files_simulation["minimal"]
        else:
            filesToRetrieve = self.output_files_simulation["complete"]

        # Do I need to run all radii?
        rhosEvaluate = cold_start_checker(
            rhos,
            filesToRetrieve,
            Folder_sim,
            cold_start=cold_start,
        )

        if len(rhosEvaluate) == len(rhos):
            # All radii need to be evaluated
            IOtools.askNewFolder(Folder_sim, force=forceIfcold_start)
            
        # Once created, expand here
        Folder_sim = IOtools.expandPath(Folder_sim)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Change this specific run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        latest_inputsFile, latest_inputsFileDict = change_and_write_code(
            rhos,
            inputs,
            Folder_sim,
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            addControlFunction=self.run_specifications['control_function'],
            controls_file=self.run_specifications['controls_file'],
            allocation=allocation,
            **kwargs_control
        )
        
        code_executor_full[subfolder_simulation] = {}
        code_executor[subfolder_simulation] = {}
        for irho in self.rhos:
            code_executor_full[subfolder_simulation][irho] = {
                "folder": Folder_sim,
                "dictionary": latest_inputsFileDict[irho],
                "inputs": latest_inputsFile[irho],
                "extraOptions": extraOptions,
                "multipliers": multipliers,
                "additional_files_to_send": additional_files_to_send[irho] if additional_files_to_send is not None else None
            }
            if irho in rhosEvaluate:
                code_executor[subfolder_simulation][irho] = code_executor_full[subfolder_simulation][irho]

        # Check input file problems
        for irho in latest_inputsFileDict:
            latest_inputsFileDict[irho].anticipate_problems()

        # Check cores problem
        # if launchSlurm:
        #     self._check_cores(rhosEvaluate, allocation)

        self.FolderSimLast = Folder_sim
            
        return code_executor, code_executor_full

    def _check_cores(self, rhosEvaluate, allocation, warning = 32 * 2):
        expected_allocated_cores = int(len(rhosEvaluate) * allocation["cores"])

        print(f'\t- Slurm job will be submitted with {expected_allocated_cores} cores ({len(rhosEvaluate)} radii x {allocation["cores"]} cores/radius)',
            typeMsg="" if expected_allocated_cores < warning else "q",)

    def _run(
        self,
        code_executor,
        run_type = 'normal', # 'normal': submit and wait; 'submit': submit and do not wait; 'prep': do not submit
        **kwargs_run
    ):
        """
        extraOptions and multipliers are not being grabbed from kwargs_NEOrun, but from code_executor for WF
        """
        
        if kwargs_run.get("only_minimal_files", False):
            filesToRetrieve = self.output_files_simulation["minimal"]
        else:
            filesToRetrieve = self.output_files_simulation["complete"]

        # Best-effort retrievals: tarred if present on the remote, but their
        # absence only emits a warning (no 60s retry, no cold-start trigger).
        # Populated by subclasses via `self.output_files_simulation["optional"]`
        # (e.g. CGYRO's bin.cgyro.restart / .restart.flag / out.cgyro.tag).
        optional_files_to_retrieve = list(self.output_files_simulation.get("optional", []))

        c = 0
        for subfolder_simulation in code_executor:
            c += len(code_executor[subfolder_simulation])

        if c == 0:
            
            print(f"\t- {self.run_specifications['code'].upper()} not run because all results files found (please ensure consistency!)",typeMsg="i")
        
            self.simulation_job = None
        
        else:
            
            print(f"\t- {self.run_specifications['code'].upper()} needs to run because not all results files found",typeMsg="i")
            
            # ----------------------------------------------------------------------------------------------------------------
            # Run simulation
            # ----------------------------------------------------------------------------------------------------------------
            """
            launchSlurm = True -> Launch as a batch job in the machine chosen, if partition specified
            launchSlurm = False -> Launch locally as a bash script
            """

            # Get code info
            code = self.run_specifications.get('code', 'tglf')
            input_file = self.run_specifications.get('input_file', 'input.tglf')
            code_call = self.run_specifications.get('code_call', None)

            # Get execution info
            allocation = kwargs_run.get("allocation") or {}
            minutes = allocation.get("minutes", 5)
            from mitim_tools.misc_tools import SLURMtools as _SLURM
            _default_rpc = _SLURM.CODE_HINTS.get(
                self.run_specifications.get('code', ''), {}
            ).get("default_resources_per_call", 1)
            resources_per_call = allocation.get("resources_per_call", _default_rpc)
            launchSlurm = kwargs_run.get("launchSlurm", True)
            # Optional per-call YAML overrides:
            #   submission_type: 'slurm_array' | 'slurm_standard' | 'bash' to override
            #                    the resolver heuristic / per-code default
            #   exclusive:       True/False to force/disable --exclusive (handy
            #                    for array elements on clusters without strict
            #                    per-job GPU isolation — guarantees whole-node)
            user_submission_type = allocation.get("submission_type")
            user_exclusive = allocation.get("exclusive")
            
            extraFlag = kwargs_run.get('extra_name', '')
            name = f"{self.run_specifications['code']}_{self.nameRunid}{extraFlag}"
            
            attempts_execution = kwargs_run.get("attempts_execution", 1)
            
            tmpFolder = self.FolderGACODE / f"tmp_{code}"
            IOtools.askNewFolder(tmpFolder, force=True)
            
            kkeys = [str(keys).replace('/','') for keys in code_executor.keys()]
            log_simulation_file=self.FolderGACODE / f"mitim_simulation_{kkeys[0]}.log" # Refer with the first folder
            self.simulation_job = FARMINGtools.mitim_job(tmpFolder, log_simulation_file=log_simulation_file)

            self.simulation_job.define_machine_quick(code,f"mitim_{name}")

            folders, folders_red = [], []
            for subfolder_sim in code_executor:

                rhos = list(code_executor[subfolder_sim].keys())

                # ---------------------------------------------
                # Prepare files and folders
                # ---------------------------------------------

                for i, rho in enumerate(rhos):
                    print(f"\t- Preparing {code.upper()} execution ({subfolder_sim}) at rho={rho:.4f}")

                    folder_sim_this = tmpFolder / subfolder_sim / f"rho_{rho:.4f}"
                    folders.append(folder_sim_this)

                    folder_sim_this_rel = folder_sim_this.relative_to(tmpFolder)
                    folders_red.append(folder_sim_this_rel.as_posix() if self.simulation_job.machineSettings['machine'] != 'local' else str(folder_sim_this_rel))

                    folder_sim_this.mkdir(parents=True, exist_ok=True)

                    input_file_sim = folder_sim_this / input_file
                    with open(input_file_sim, "w") as f:
                        f.write(code_executor[subfolder_sim][rho]["inputs"])
                        
                    # Copy potential additional files to send
                    if code_executor[subfolder_sim][rho]["additional_files_to_send"] is not None:
                        for file in code_executor[subfolder_sim][rho]["additional_files_to_send"]:
                            shutil.copy(file, folder_sim_this / Path(file).name)

            # ---------------------------------------------
            # Prepare command
            # ---------------------------------------------

            # Grab machine local limits -------------------------------------------------
            from mitim_tools.misc_tools import SLURMtools
            machineSettings = FARMINGtools.mitim_job.grab_machine_settings(code)

            # For local-bash, also consider env-provided SLURM limits (nested SLURM jobs)
            # and the machine's own cpu_count as an upper bound on concurrency.
            if (machineSettings["machine"] == "local") and \
                not (launchSlurm and ("partition" in self.simulation_job.machineSettings["slurm"])):
                _slurm_ntasks = os.environ.get('SLURM_NTASKS')
                _slurm_cpt = os.environ.get('SLURM_CPUS_PER_TASK')
                if (_slurm_cpt is not None) and (_slurm_ntasks is not None):
                    _env_cores = int(_slurm_cpt) * int(_slurm_ntasks)
                elif _slurm_cpt is not None:
                    _env_cores = int(_slurm_cpt)
                elif _slurm_ntasks is not None:
                    _env_cores = int(_slurm_ntasks)
                else:
                    _env_cores = int(os.cpu_count() or 16)
                cpn_for_resolver = machineSettings.get("cores_per_node")
                if cpn_for_resolver is None or _env_cores < cpn_for_resolver:
                    print(f"\t- Local execution capped at {_env_cores} cores (machine config says {cpn_for_resolver})", typeMsg="i")
                    machineSettings = dict(machineSettings)
                    machineSettings["cores_per_node"] = _env_cores

            total_simulation_executions = len(rhos) * len(code_executor)
            total_cores_required = int(resources_per_call) * total_simulation_executions

            # ---- Resolve allocation once (submission_type + sbatch dict + mpi + concurrency)
            # YAML override (allocation.submission_type) takes precedence over the
            # per-code default in run_specifications (e.g. CGYRO defaults to
            # 'slurm_array' on GPU machines).
            array_list_preview = [str(i) for i in range(total_simulation_executions)]
            forced_submission_type = user_submission_type or self.run_specifications.get('force_submission_type')
            resolved = SLURMtools.resolve(
                code=code,
                allocation={"resources_per_call": resources_per_call, "minutes": minutes,
                            "mem": allocation.get("mem")},
                n_rhos=len(rhos), n_subfolders=len(code_executor),
                machine_settings=machineSettings,
                launch_slurm=launchSlurm,
                force_submission_type=forced_submission_type,
                job_name=code + '_sim',
                array_list=array_list_preview,
                exclusive=user_exclusive,
            )
            type_of_submission = resolved.submission_type
            max_cores_per_node = machineSettings.get("cores_per_node") or 16

            shellPreCommands, shellPostCommands = None, None

            # Simply bash, no slurm
            if type_of_submission == "bash":

                max_parallel_execution = max(1, resolved.concurrency)

                print(f"\t- {code.upper()} will be executed as bash script (total cores: {total_cores_required},  cores per simulation: {resources_per_call}). MITIM will launch {total_simulation_executions // max_parallel_execution+1} sequential executions",typeMsg="i")

                # Build the bash script with job control enabled and a loop to limit parallel jobs
                GACODEcommand = "#!/usr/bin/env bash\n"
                GACODEcommand += "set -m\n"  # Enable job control even in non-interactive mode
                GACODEcommand += f"max_parallel_execution={max_parallel_execution}\n\n"  # Set the maximum number of parallel processes

                # Create a bash array of folders
                GACODEcommand += "folders=(\n"
                for folder in folders_red:
                    GACODEcommand += f'    "{folder}"\n'
                GACODEcommand += ")\n\n"

                # Loop over each folder and launch code, waiting if we've reached max_parallel_execution
                GACODEcommand += "for folder in \"${folders[@]}\"; do\n"
                folder_str = '"$folder"'  # literal double quotes around $folder
                GACODEcommand += f'    {code_call(folder=folder_str, n=resources_per_call, p=self.simulation_job.folderExecution)} &\n'
                GACODEcommand += "    while (( $(jobs -r | wc -l) >= max_parallel_execution )); do sleep 1; done\n"
                GACODEcommand += "done\n\n"
                GACODEcommand += "wait\n"

            # Standard job
            elif type_of_submission == "slurm_standard":

                print(f"\t- {code.upper()} will be executed in SLURM as standard job (cpus: {total_cores_required})",typeMsg="i")

                # Code launches
                GACODEcommand = ""
                for folder in folders_red:
                    GACODEcommand += f'    {code_call(folder = folder, n = resources_per_call, p = self.simulation_job.folderExecution)}  &\n'
                GACODEcommand += "\nwait"  # This is needed so that the script doesn't end before each job

            # Job array
            elif type_of_submission == "slurm_array":

                print(f"\t- {code.upper()} will be executed in SLURM as job array due to its size (cpus: {total_cores_required})",typeMsg="i")

                folders_list = "FOLDERS=( "
                array_list = []
                for i, folder in enumerate(folders_red):
                    array_list.append(f"{i}")
                    folders_list += f"{folder} "
                folders_list += ")"

                # Code launches
                GACODEcommand = folders_list + "\n\n"

                indexed_folder = "${FOLDERS[$SLURM_ARRAY_TASK_ID]}"
                GACODEcommand += code_call(
                    folder = indexed_folder,
                    n = resources_per_call,
                    p = self.simulation_job.folderExecution,
                    additional_command = f'1> {self.simulation_job.folderExecution}/{indexed_folder}/slurm_output.dat 2> {self.simulation_job.folderExecution}/{indexed_folder}/slurm_error.dat\n')

            # ---------------------------------------------
            # Execute
            # ---------------------------------------------

            # Re-resolve with the final array_list (folder indices) now that
            # we know them, then either use the resolver's sbatch dict directly
            # or fall back to a code-specific `code_slurm_settings` hook for
            # codes that have not been registered in SLURMtools.CODE_HINTS.
            if type_of_submission == "slurm_array":
                resolved = SLURMtools.resolve(
                    code=code,
                    allocation={"resources_per_call": resources_per_call, "minutes": minutes,
                                "mem": allocation.get("mem")},
                    n_rhos=len(rhos), n_subfolders=len(code_executor),
                    machine_settings=machineSettings,
                    launch_slurm=launchSlurm,
                    force_submission_type=forced_submission_type,
                    job_name=code + '_sim',
                    array_list=array_list,
                    exclusive=user_exclusive,
                )

            if code not in SLURMtools.CODE_HINTS:
                raise Exception(
                    f"[MITIM] Code '{code}' is not registered in SLURMtools.CODE_HINTS. "
                    f"Add a hints entry ({{'default_resources_per_call', 'uses_gpu', ...}})."
                )
            slurm_settings = resolved.sbatch

            self.simulation_job.define_machine(
                code,
                f"mitim_{name}",
                launchSlurm=launchSlurm,
                slurm_settings=slurm_settings,
            )
            
            # Mandatory vs best-effort retrieval lists. `files_we_must_check` is
            # what check_all_received flags as missing (triggers the retry). The
            # tarball (`files_we_want_to_tar`) additionally includes optional
            # files so they come down if present on the remote.
            files_we_must_check = {}
            files_we_want_to_tar = {}
            for folder in folders_red:
                files_we_must_check[folder] = list(filesToRetrieve)
                files_we_want_to_tar[folder] = list(filesToRetrieve) + list(optional_files_to_retrieve)
            # ---------------------------------------------

            self.simulation_job.prep(
                GACODEcommand,
                input_folders=folders,
                output_folders=folders_red,
                output_folders_selective=files_we_want_to_tar,
                check_files_in_folder=files_we_must_check,
                shellPreCommands=shellPreCommands,
                shellPostCommands=shellPostCommands,
            )

            # Submit run and wait
            if run_type in ['normal', 'send']:
            
                run_status_int = 0
                while run_status_int < 2:
                    
                    try:
                        self.simulation_job.run(
                            removeScratchFolders=True,
                            attempts_execution=attempts_execution,
                            helper_lostconnection=kwargs_run.get("helper_lostconnection", False),
                            execute_case_flag = run_type == 'normal'
                            )
                        run_status_int = 2
                    except LOGtools.InteractiveTerminalError:
                        print('\n\t Run wanted to crash because interactive terminal is not allowed in this bash job, but repeating once to see if error was random')
                        run_status_int += 1
                    
                self._organize_results(code_executor, tmpFolder, filesToRetrieve, optional_files_to_retrieve=optional_files_to_retrieve)

            # Submit run but do not wait; the user should do checks and fetch results
            elif run_type == 'submit':

                self.simulation_job.run(
                    waitYN=False,
                    check_if_files_received=False,
                    removeScratchFolders=False,
                    removeScratchFolders_goingIn=kwargs_run.get("cold_start", False),
                )

                self.kwargs_organize = {
                    "code_executor": code_executor,
                    "tmpFolder": tmpFolder,
                    "filesToRetrieve": filesToRetrieve,
                    "optional_files_to_retrieve": optional_files_to_retrieve,
                }

                self.slurm_output = "slurm_output.dat"

                # Prepare how to search for the job without waiting for it
                self.simulation_job.launchSlurm = True
                self.simulation_job.slurm_settings['name'] = Path(self.simulation_job.folderExecution).name

                # Persist submission metadata so a future process can re-attach
                # to this in-flight job instead of resubmitting. Opt-in via
                # subclass `_submission_metadata_filename` (e.g. CGYRO).
                if self._submission_metadata_filename is not None:
                    self._write_submission_metadata(kwargs_run.get("base_subfolder"))

    def check(self, every_n_minutes=None, skip_first_iteration_squeue=False, max_completing_polls=2, custom_checker=None):
        '''
        Poll slurm until the job leaves the queue (state "NOT FOUND" / status=2).

        skip_first_iteration_squeue: when True, the first loop iteration does NOT
        run a fresh `squeue`; instead it reuses `self.simulation_job.status`
        already populated by an earlier call (e.g. the re-attach liveness probe
        in transport_cgyro.py) so the poller does not immediately redo a squeue
        that we just ran seconds ago.

        max_completing_polls: safety valve for jobs stuck in slurm state
        "COMPLETING" (usually a node/IO/epilog hang after the compute already
        finished). When squeue reports COMPLETING for this many consecutive
        polls, exit the loop with a warning so `fetch()` can pull whatever is
        on disk rather than sleeping forever on a dead node. Set to None to
        disable the heuristic.

        custom_checker: optional callable invoked as `custom_checker(self)`
        after every squeue poll. Intended for subclass-specific inspection
        (e.g. CGYRO's per-(subfolder,rho) out.cgyro.info / out.cgyro.timing
        walk) — the generic check() stays code-agnostic. Exceptions are
        caught and logged so a flaky remote query does not abort the poll.
        '''

        if self.simulation_job is None:
            print("- Not checking status because simulation job is not defined (not run)", typeMsg="i")
            return

        if self.simulation_job.launchSlurm:
            print("- Checker job status")

            first = True
            completing_streak = 0
            while True:
                if first and skip_first_iteration_squeue:
                    print(f"\t- Reusing status from the earlier liveness probe (skipping redundant squeue)", typeMsg='i')
                else:
                    self.simulation_job.check(file_output = self.slurm_output)
                first = False
                state = self.simulation_job.infoSLURM.get("STATE")
                print(f'\t- Current status (as of  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}): {self.simulation_job.status} ({state})')

                if custom_checker is not None:
                    try:
                        custom_checker(self)
                    except Exception as _e:
                        print(f"\t- custom_checker raised: {_e}; continuing poll", typeMsg='w')

                if self.simulation_job.status == 2:
                    print("\n\t* Job considered finished (please do .fetch() to retrieve results)",typeMsg="i")
                    break

                # Track consecutive COMPLETING polls — slurm's "COMPLETING"
                # state normally lasts seconds; when it persists it is almost
                # always an epilog/filesystem hang on the node rather than the
                # user code still running. Give up after max_completing_polls
                # so fetch() can run against whatever has been flushed.
                if state == "COMPLETING":
                    completing_streak += 1
                    if max_completing_polls is not None and completing_streak >= max_completing_polls:
                        print(f"\n\t* Slurm state has been COMPLETING for {completing_streak} consecutive polls — likely node/IO/epilog hang; giving up on the poll loop and letting fetch() pull whatever is on the remote", typeMsg='w')
                        break
                else:
                    completing_streak = 0

                if every_n_minutes is None:
                    print("\n\t* Job not finished yet")
                    break
                else:
                    print(f"\n\t* Waiting {every_n_minutes} minutes")
                    time.sleep(every_n_minutes * 60)
        else:
            print("- Not checking status because this was run command line (not slurm)")

    def fetch(self):
        """
        For a job that has been submitted but not waited for, once it is done, get the results
        """

        if self.simulation_job is None:
            print("- Not fetching because simulation job is not defined (not run)", typeMsg="i")
            return

        print("\n\n\t- Fetching results")

        if self.simulation_job.launchSlurm:
            self.simulation_job.connect()
            self.simulation_job.retrieve()
            self.simulation_job.close()

            self._organize_results(**self.kwargs_organize)

        else:
            print("- Not retrieving results because this was run command line (not slurm)")

    def delete(self):

        print("\n\n\t- Deleting job")

        self.simulation_job.launchSlurm = False

        self.simulation_job.prep(
            f"scancel -n {self.simulation_job.slurm_settings['name']}",
            label_log_files="_finish",
        )

        self.simulation_job.run()

    def _organize_results(self, code_executor, tmpFolder, filesToRetrieve, optional_files_to_retrieve=None):

        # ---------------------------------------------
        # Organize
        # ---------------------------------------------

        optional_files_to_retrieve = list(optional_files_to_retrieve) if optional_files_to_retrieve else []

        print("\t- Retrieving files and changing names for storing")
        fineall = True
        missing_optional = []
        for subfolder_sim in code_executor:

            for rho in code_executor[subfolder_sim].keys():
                for file in filesToRetrieve:
                    original_file = f"{file}_{rho:.4f}"
                    final_destination = code_executor[subfolder_sim][rho]['folder'] / f"{original_file}"

                    final_destination.unlink(missing_ok=True)

                    temp_file = tmpFolder / subfolder_sim / f"rho_{rho:.4f}" / f"{file}"

                    if not temp_file.exists():
                        print(f"\t!! file {file} ({original_file}) could not be retrieved", typeMsg="w")
                        continue

                    temp_file.replace(final_destination)

                    fineall = fineall and final_destination.exists()

                    if not final_destination.exists():
                        print(f"\t!! file {file} ({original_file}) could not be retrived",typeMsg="w",)

                # Optional retrievals — move if present, silently skip if not;
                # we aggregate the misses and emit one summary warning so the
                # log is not flooded with "restart not found" noise per rho.
                for file in optional_files_to_retrieve:
                    original_file = f"{file}_{rho:.4f}"
                    final_destination = code_executor[subfolder_sim][rho]['folder'] / f"{original_file}"
                    final_destination.unlink(missing_ok=True)
                    temp_file = tmpFolder / subfolder_sim / f"rho_{rho:.4f}" / f"{file}"
                    if not temp_file.exists():
                        missing_optional.append((subfolder_sim, float(rho), file))
                        continue
                    temp_file.replace(final_destination)

        if missing_optional:
            distinct = sorted({f for _, _, f in missing_optional})
            print(f"\t- Optional file(s) not present on the remote (ok, proceeding): {distinct}   [{len(missing_optional)} (subfolder, rho, file) tuples]", typeMsg="w")

        if fineall:
            print("\t\t- All files were successfully retrieved")

            # Remove temporary folder
            IOtools.shutil_rmtree(tmpFolder)

        else:
            print("\t\t- Some files were not retrieved", typeMsg="w")

    # ------------------------------------------------------------------
    # Submission-state persistence: lets a later process re-attach to an
    # already-submitted (run_type='submit') slurm job instead of resubmitting.
    # Write is triggered from `_run()`'s submit branch when the subclass opts in
    # via `_submission_metadata_filename`. Load is invoked by the caller of
    # `run()` (e.g. transport_cgyro.py) before `check()` / `fetch()` / `read()`.
    # ------------------------------------------------------------------

    def _submission_metadata_path(self, base_subfolder):
        if self._submission_metadata_filename is None or base_subfolder is None:
            return None
        return self.FolderGACODE / base_subfolder / self._submission_metadata_filename

    def _write_submission_metadata(self, base_subfolder):
        path = self._submission_metadata_path(base_subfolder)
        if path is None:
            return

        job = self.simulation_job
        # Only the fields _organize_results actually reads from code_executor
        # (folder per (subfolder, rho)) — everything else is derivable from the
        # staged input files on disk and re-prep on rehydration.
        code_executor_serial = {}
        for sub, rhos in self.kwargs_organize["code_executor"].items():
            code_executor_serial[sub] = {
                f"{float(rho):.6f}": {"folder": str(v["folder"])}
                for rho, v in rhos.items()
            }

        results_per_plasma_serial = None
        if getattr(self, "results_per_plasma", None):
            results_per_plasma_serial = {
                str(int(p)): {"subfolder": info["subfolder"], "folder": str(info["folder"])}
                for p, info in self.results_per_plasma.items()
            }

        payload = {
            "schema_version": 1,
            "mode": "batched" if results_per_plasma_serial is not None else "single",
            "code": self.run_specifications.get("code"),
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "base_subfolder": base_subfolder,
            "slurm_output": self.slurm_output,
            "kwargs_organize": {
                "tmpFolder": str(self.kwargs_organize["tmpFolder"]),
                "filesToRetrieve": list(self.kwargs_organize["filesToRetrieve"]),
                "optional_files_to_retrieve": list(self.kwargs_organize.get("optional_files_to_retrieve", [])),
                "code_executor": code_executor_serial,
            },
            "results_per_plasma": results_per_plasma_serial,
            "job": {
                "folder_local": str(job.folder_local),
                "folderExecution": str(job.folderExecution),
                "jobid": job.jobid,
                "launchSlurm": bool(job.launchSlurm),
                "slurm_settings": job.slurm_settings,
                "machineSettings": job.machineSettings,
                "output_files": [str(f) for f in getattr(job, "output_files", [])],
                "output_folders": [str(f) for f in getattr(job, "output_folders", [])],
                "check_files_in_folder": getattr(job, "check_files_in_folder", {}),
                "output_folders_selective": getattr(job, "output_folders_selective", {}),
                "log_simulation_file": str(job.log_simulation_file) if job.log_simulation_file else None,
                "run_in_place": bool(getattr(job, "run_in_place", False)),
            },
        }

        # TODO(multi-process-safety): concurrent PORTALS drivers writing to the
        # same folder could race here — add fcntl.flock if that becomes real.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\t- Submission metadata written to {path}", typeMsg="i")

    def load_submission_state(self, path):
        '''
        Rehydrate `self.simulation_job`, `self.kwargs_organize`, `self.slurm_output`,
        and (for batched mode) `self.results_per_plasma` from a JSON file written
        by `_write_submission_metadata`, so `check()` / `fetch()` can talk to the
        already-running slurm job without resubmitting.
        '''
        with open(path, "r") as f:
            data = json.load(f)

        job = FARMINGtools.mitim_job(
            data["job"]["folder_local"],
            log_simulation_file=data["job"]["log_simulation_file"],
        )
        job.folderExecution = data["job"]["folderExecution"]
        job.jobid = data["job"]["jobid"]
        job.launchSlurm = data["job"]["launchSlurm"]
        job.slurm_settings = data["job"]["slurm_settings"]
        job.machineSettings = data["job"]["machineSettings"]
        job.output_files = list(data["job"]["output_files"])
        job.output_folders = list(data["job"]["output_folders"])
        job.check_files_in_folder = data["job"]["check_files_in_folder"]
        job.output_folders_selective = data["job"]["output_folders_selective"]
        job.run_in_place = data["job"].get("run_in_place", False)

        self.simulation_job = job
        self.slurm_output = data["slurm_output"]

        code_executor_rehydrated = {}
        for sub, rhos in data["kwargs_organize"]["code_executor"].items():
            code_executor_rehydrated[sub] = {
                float(rho): {"folder": Path(v["folder"])} for rho, v in rhos.items()
            }
        self.kwargs_organize = {
            "code_executor": code_executor_rehydrated,
            "tmpFolder": Path(data["kwargs_organize"]["tmpFolder"]),
            "filesToRetrieve": list(data["kwargs_organize"]["filesToRetrieve"]),
            "optional_files_to_retrieve": list(data["kwargs_organize"].get("optional_files_to_retrieve", [])),
        }

        # Note: results_per_plasma is rebuilt in memory by the caller via
        # `_prepare_plasmas_state`, which also restores `profiles` /
        # `inputs_files` / `NormalizationSets` that `read_plasma` needs.
        # We do not overwrite it here.

        return data

    def _local_results_complete(self):
        '''
        True when every expected result file (per rho per subfolder) from
        `kwargs_organize` already exists on disk — used by the re-attach path to
        skip check()+fetch() when the prior job finished while no PORTALS
        process was watching.
        '''
        if not getattr(self, "kwargs_organize", None):
            return False
        code_executor = self.kwargs_organize["code_executor"]
        files_to_retrieve = self.kwargs_organize["filesToRetrieve"]
        for sub, rhos in code_executor.items():
            for rho, v in rhos.items():
                folder = Path(v["folder"])
                for fname in files_to_retrieve:
                    if not (folder / f"{fname}_{float(rho):.4f}").exists():
                        return False
        return True

    def run_over_plasmas(
        self,
        list_of_states,           # List of mitim_state / input.gacode path / gacode_state objects, one per plasma
        base_subfolder,           # 'base' -> produces subfolder labels base_plasma0, base_plasma1, ...
        cold_start=False,
        forceIfcold_start=False,
        code_settings=None,
        extraOptions=None,
        multipliers=None,
        minimum_delta_abs=None,
        ApplyCorrections=True,
        Quasineutral=False,
        launchSlurm=True,
        extra_name="exe",
        allocation=None,
        attempts_execution=1,
        only_minimal_files=False,
        run_type='normal',
        additional_files_to_send=None,
        helper_lostconnection=False,
    ):
        '''
        Phase-1 multi-plasma runner. Runs the same simulation configuration (same rhos,
        same code_settings, same non-scan multipliers, ...) for N independent plasma
        states in a single parallel submission by reusing the subfolder_simulation
        axis that scans already rely on.

        Each plasma p ends up under subfolder_simulation = f"{base_subfolder}_plasma{p}"
        and its prep-time state (profiles, inputs_files, normalizations, FolderSim) is
        cached on self.results_per_plasma[p] so the caller can read each plasma's
        results afterwards via self.read_plasma(p).

        Notes:
            - Requires self.FolderGACODE to be set. Call prep(...) once before this
              helper, or rely on the first iteration's prep call to create it.
            - The i-th prep(...) call overwrites the single staging input.gacode_torun
              under self.FolderGACODE, then this helper copies it to
              input.gacode_torun_plasma{p} for traceability. The in-memory
              self.inputs_files is snapshotted into the code_executor before the next
              plasma's prep touches it, so per-plasma inputs do not leak across.
        '''

        run_type = _normalize_run_type(run_type)

        if extraOptions is None:
            extraOptions = {}
        if multipliers is None:
            multipliers = {}
        if minimum_delta_abs is None:
            minimum_delta_abs = {}

        if allocation is None:
            from mitim_tools.misc_tools import SLURMtools
            allocation = {
                "resources_per_call": SLURMtools.CODE_HINTS.get(
                    self.run_specifications.get('code', ''),
                    {}).get("default_resources_per_call", 1),
                "minutes": 10,
            }

        if not hasattr(self, 'FolderGACODE') or self.FolderGACODE is None:
            raise Exception(
                "[MITIM] run_over_plasmas requires FolderGACODE to be set. Either call "
                "prep(...) once before run_over_plasmas or pass FolderGACODE via a "
                "preceding prep call."
            )

        code_executor, code_executor_full, plasma_labels = self._prepare_plasmas_state(
            list_of_states,
            base_subfolder,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            only_minimal_files=only_minimal_files,
            launchSlurm=launchSlurm,
            allocation=allocation,
            additional_files_to_send=additional_files_to_send,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            announce=True,
        )

        # Parallel dispatch of every (plasma, rho) work unit via the existing _run path.
        self._run(
            code_executor,
            code_executor_full=code_executor_full,
            code_settings=code_settings,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            allocation=allocation,
            only_minimal_files=only_minimal_files,
            attempts_execution=attempts_execution,
            run_type=run_type,
            helper_lostconnection=helper_lostconnection,
            base_subfolder=base_subfolder,
        )

        return plasma_labels

    def _prepare_plasmas_state(
        self,
        list_of_states,
        base_subfolder,
        cold_start=False,
        forceIfcold_start=False,
        code_settings=None,
        extraOptions=None,
        multipliers=None,
        minimum_delta_abs=None,
        only_minimal_files=False,
        launchSlurm=True,
        allocation=None,
        additional_files_to_send=None,
        ApplyCorrections=True,
        Quasineutral=False,
        announce=False,
    ):
        '''
        Build per-plasma `code_executor` and populate `self.results_per_plasma`
        in memory without submitting anything. Extracted from `run_over_plasmas`
        so the re-attach path (transport_cgyro.py) can rebuild the per-plasma
        state that `read_plasma` needs after a prior process submitted the job.
        '''
        if extraOptions is None:
            extraOptions = {}
        if multipliers is None:
            multipliers = {}
        if minimum_delta_abs is None:
            minimum_delta_abs = {}

        code_executor, code_executor_full = {}, {}
        self.results_per_plasma = {}
        plasma_labels = {}

        for p, state in enumerate(list_of_states):
            subfolder_simulation = f"{base_subfolder}_plasma{p}"
            plasma_labels[p] = subfolder_simulation

            if announce:
                print(f"\n=============================================================")
                print(f"  run_over_plasmas: preparing plasma {p} -> {subfolder_simulation}")
                print(f"=============================================================")

            # Populate self.profiles / self.inputs_files / self.NormalizationSets for plasma p.
            # Only the first prep may need to create FolderGACODE; subsequent plasmas reuse it.
            self.prep(
                state,
                self.FolderGACODE,
                cold_start=cold_start if p == 0 else False,
                forceIfcold_start=forceIfcold_start,
            )

            # Keep a per-plasma copy of the input.gacode alongside the shared staging one
            # (subprocess path writes this file; in-process prep uses synthetic in-memory
            # paths that never touch disk, so the file may not exist — skip gracefully).
            src_gacode = self.FolderGACODE / "input.gacode_torun"
            if src_gacode.exists():
                shutil.copy2(src_gacode, self.FolderGACODE / f"input.gacode_torun_plasma{p}")

            # _run_prepare snapshots self.inputs_files into code_executor[subfolder][rho]['inputs']
            # so plasma p's inputs are frozen here; subsequent per-plasma prep calls do not
            # corrupt already-queued work.
            self._run_prepare(
                subfolder_simulation,
                code_executor=code_executor,
                code_executor_full=code_executor_full,
                code_settings=code_settings,
                extraOptions=extraOptions,
                multipliers=multipliers,
                cold_start=cold_start,
                forceIfcold_start=forceIfcold_start,
                only_minimal_files=only_minimal_files,
                launchSlurm=launchSlurm,
                allocation=allocation,
                additional_files_to_send=additional_files_to_send,
                ApplyCorrections=ApplyCorrections,
                minimum_delta_abs=minimum_delta_abs,
                Quasineutral=Quasineutral,
            )

            self.results_per_plasma[p] = {
                'subfolder': subfolder_simulation,
                'folder': self.FolderSimLast,
                'profiles': self.profiles,
                'inputs_files': copy.deepcopy(self.inputs_files),
                'NormalizationSets': self.NormalizationSets,
            }

        return code_executor, code_executor_full, plasma_labels

    def read_plasma(self, plasma_idx, label=None, **read_kwargs):
        '''
        Read results for a specific plasma produced by run_over_plasmas.

        Temporarily activates that plasma's in-memory prep state (profiles /
        inputs_files / NormalizationSets / FolderSimLast) so that the existing
        read(...) path can reuse per-plasma profiles and normalizations without
        being aware of the plasma axis.
        '''
        if not hasattr(self, 'results_per_plasma') or plasma_idx not in self.results_per_plasma:
            raise KeyError(
                f"No cached plasma run for index {plasma_idx}. Call run_over_plasmas(...) first."
            )

        info = self.results_per_plasma[plasma_idx]
        effective_label = label if label is not None else info['subfolder']

        saved_profiles = self.profiles
        saved_inputs_files = self.inputs_files
        saved_NormalizationSets = self.NormalizationSets
        saved_FolderSimLast = getattr(self, 'FolderSimLast', None)

        self.profiles = info['profiles']
        self.inputs_files = info['inputs_files']
        self.NormalizationSets = info['NormalizationSets']
        self.FolderSimLast = info['folder']

        try:
            self.read(label=effective_label, folder=info['folder'], **read_kwargs)
        finally:
            self.profiles = saved_profiles
            self.inputs_files = saved_inputs_files
            self.NormalizationSets = saved_NormalizationSets
            if saved_FolderSimLast is not None:
                self.FolderSimLast = saved_FolderSimLast

        return effective_label

    def run_scan(
        self,
        subfolder,  # 'scan1',
        multipliers={},
        minimum_delta_abs={},
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        variables_scanTogether=[],
        relativeChanges=True,
        **kwargs_run,
    ):

        # -------------------------------------
        # Add baseline
        # -------------------------------------
        if (1.0 not in varUpDown) and relativeChanges:
            print("\n* Since variations vector did not include base case, I am adding it",typeMsg="i",)
            varUpDown_new = []
            added = False
            for i in varUpDown:
                if i > 1.0 and not added:
                    varUpDown_new.append(1.0)
                    added = True
                varUpDown_new.append(i)
        else:
            varUpDown_new = varUpDown


        code_executor, code_executor_full, folders, varUpDown_new = self._prepare_scan(
            subfolder,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            variable=variable,
            varUpDown=varUpDown_new,
            variables_scanTogether=variables_scanTogether,
            relativeChanges=relativeChanges,
            **kwargs_run,
        )

        # Run them all
        self._run(
            code_executor,
            code_executor_full=code_executor_full,
            **kwargs_run,
        )
        
        # Read results
        for cont_mult, mult in enumerate(varUpDown_new):
            name = f"{variable}_{mult}"
            self.read(
                label=f"{self.subfolder_scan}_{name}",
                folder=folders[cont_mult],
                cold_startWF = False,
                require_all_files=not kwargs_run.get("only_minimal_files",False),
            )

        return code_executor_full

    def _prepare_scan(
        self,
        subfolder,  # 'scan1',
        multipliers=None,
        minimum_delta_abs=None,
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        variables_scanTogether=None,
        relativeChanges=True,
        **kwargs_run,
    ):
        """
        Multipliers will be modified by adding the scaning variables, but I don't want to modify the original
        multipliers, as they may be passed to the next scan

        Set relativeChanges=False if varUpDown contains the exact values to change, not multipleiers
        """
        
        if multipliers is None:
            multipliers = {}
        if minimum_delta_abs is None:
            minimum_delta_abs = {}
        if variables_scanTogether is None:
            variables_scanTogether = []
        
        completeVariation = self.run_specifications['complete_variation']
        
        multipliers_mod = copy.deepcopy(multipliers)

        self.subfolder_scan = subfolder

        if relativeChanges:
            for i in range(len(varUpDown)):
                varUpDown[i] = round(varUpDown[i], 6)

        print(f"\n- Proceeding to scan {variable}{' together with '+', '.join(variables_scanTogether) if len(variables_scanTogether)>0 else ''}:")

        code_executor = {}
        code_executor_full = {}
        folders = []
        for cont_mult, mult in enumerate(varUpDown):
            mult = round(mult, 6)

            if relativeChanges:
                print(f"\n + Multiplier: {mult} -----------------------------------------------------------------------------------------------------------")
            else:
                print(f"\n + Value: {mult} ----------------------------------------------------------------------------------------------------------------")

            # If multipliers already had the variable, make sure I account for that variation as well
            if variable in multipliers:
                base_mult = multipliers[variable]
                mult = round(base_mult * mult, 6)
            
            multipliers_mod[variable] = mult

            for variable_scanTogether in variables_scanTogether:
                multipliers_mod[variable_scanTogether] = mult

            name = f"{variable}_{mult}"

            species = self.inputs_files[self.rhos[0]]  # Any rho will do

            if completeVariation is not None:
                multipliers_mod = completeVariation(multipliers_mod, species)

            if not relativeChanges:
                for ikey in multipliers_mod:
                    kwargs_run["extraOptions"][ikey] = multipliers_mod[ikey]
                multipliers_mod = {}

            # Force ensure quasineutrality if the
            if variable in ["AS_3", "AS_4", "AS_5", "AS_6"]:
                kwargs_run["Quasineutral"] = True

            # Only ask the cold_start in the first round
            kwargs_run["forceIfcold_start"] = cont_mult > 0 or ("forceIfcold_start" in kwargs_run and kwargs_run["forceIfcold_start"])

            code_executor, code_executor_full = self._run_prepare(
                f"{self.subfolder_scan}_{name}",
                code_executor=code_executor,
                code_executor_full=code_executor_full,
                multipliers=multipliers_mod,
                minimum_delta_abs=minimum_delta_abs,
                **kwargs_run,
            )

            folders.append(copy.deepcopy(self.FolderSimLast))

        return code_executor, code_executor_full, folders, varUpDown

    def read(
        self,
        label="run1",
        folder=None,  # If None, search in the previously run folder
        suffix=None,  # If None, search with my standard _0.55 suffixes corresponding to rho of this TGLF class
        input_gacode=None,  # If provided, will try to get normalizations from it if they are not already populated in the class
        **kwargs_to_class_output
    ):
        print("> Reading simulation results")

        class_output = self.run_specifications['output_class']

        # If no specified folder, check the last one
        if folder is None:
            folder = self.FolderSimLast
            
        self.results[label] = {
            'output':[],
            'parsed': [],
            "x": np.array(self.rhos),
            }

        # Try get normalizations if they weren't populated (e.g. this is just a "read" of an already run folder)
        if self.NormalizationSets["SELECTED"] is None and input_gacode is not None:
            print("\t- Getting normalizations from input.gacode provided in the read function")
            from mitim_tools.gacode_tools import PROFILEStools
            self.NormalizationSets, _ = NORMtools.normalizations(PROFILEStools.gacode_state(input_gacode))

        for rho in self.rhos:

            SIMout = class_output(
                folder,
                suffix=(f"_{rho:.4f}" if rho is not None else "") if suffix is None else suffix,
                **kwargs_to_class_output
            )
            
            # Unnormalize
            if 'NormalizationSets' in self.__dict__:
                SIMout.unnormalize(
                    self.NormalizationSets["SELECTED"],
                    rho=rho,
                )
            else:
                print("No normalization sets found.")

            self.results[label]['output'].append(SIMout)

            self.results[label]['parsed'].append(buildDictFromInput(SIMout.inputFile) if SIMout.inputFile else None)

    def read_scan(        
        self,
        label="scan1",
        subfolder=None,
        variable="RLTS_1",
        ion_OI_position_in_total_padded_list=2,
        variable_mapping=None,
        variable_mapping_unn=None
    ):
        '''
        ion_OI_position_in_total_padded_list is the index in the input.tglf file... so if you want for ion RLNS_5, ion_OI_position_in_total_padded_list=5
        The name comes from the fact that in input.tglf files electrons are 1, first ion is 2, etc.
        '''

        if subfolder is None:
            subfolder = self.subfolder_scan
            
        if variable_mapping is None:
            variable_mapping = {}
        if variable_mapping_unn is None:
            variable_mapping_unn = {}

        self.scans[label] = {}
        self.scans[label]["variable"] = variable
        self.scans[label]["positionBase"] = None
        self.scans[label]["unnormalization_successful"] = True
        self.scans[label]["results_tags"] = []

        self.ion_OI_position_in_total_padded_list_scan = ion_OI_position_in_total_padded_list

        # ----
        
        scan = {}
        for ikey in variable_mapping | variable_mapping_unn:
            scan[ikey] = []

        cont = 0
        for ikey in self.results:
            
            isThisTheRightReadResults = (subfolder in ikey) and (variable== "_".join(ikey.split("_")[:-1]).split(subfolder + "_")[-1])

            if isThisTheRightReadResults:

                self.scans[label]["results_tags"].append(ikey)
                
                # Initialize lists
                scan0 = {}
                for ikey2 in variable_mapping | variable_mapping_unn:
                    scan0[ikey2] = []

                # Loop over radii
                for irho_cont in range(len(self.rhos)):
                    irho = np.where(self.results[ikey]["x"] == self.rhos[irho_cont])[0][0]

                    for ikey2 in variable_mapping:
                        
                        obj = self.results[ikey][variable_mapping[ikey2][0]][irho]
                        if not hasattr(obj, '__dict__'):
                            obj_dict = obj
                        else:
                            obj_dict = obj.__dict__
                        var0 = obj_dict[variable_mapping[ikey2][1]]
                        scan0[ikey2].append(var0 if variable_mapping[ikey2][2] is None else var0[variable_mapping[ikey2][2]])

                    # Unnormalized
                    self.scans[label]["unnormalization_successful"] = True
                    for ikey2 in variable_mapping_unn:
                        obj = self.results[ikey][variable_mapping_unn[ikey2][0]][irho]
                        if not hasattr(obj, '__dict__'):
                            obj_dict = obj
                        else:
                            obj_dict = obj.__dict__
                            
                        if variable_mapping_unn[ikey2][1] not in obj_dict:
                            self.scans[label]["unnormalization_successful"] = False
                            break
                        var0 = obj_dict[variable_mapping_unn[ikey2][1]]
                        scan0[ikey2].append(var0 if variable_mapping_unn[ikey2][2] is None else var0[variable_mapping_unn[ikey2][2]])
                
                for ikey2 in variable_mapping | variable_mapping_unn:
                    scan[ikey2].append(scan0[ikey2])

                if float(ikey.split('_')[-1]) == 1.0:
                    self.scans[label]["positionBase"] = cont
                cont += 1

        self.scans[label]["x"] = np.array(self.rhos)

        for ikey2 in variable_mapping | variable_mapping_unn:
            self.scans[label][ikey2] = np.atleast_2d(np.transpose(scan[ikey2]))

    def prepare_for_save(self, class_to_store = None):
        """
        Remove potential unpickleable objects
        """

        if class_to_store is None:
            class_to_store = self

        if 'fn' in class_to_store.__dict__:
            print('\t- Removing Qt object before pickling')
            del class_to_store.fn

        return class_to_store

    def save_pickle(self, file, class_to_store = None):
        
        print('...Pickling simulation class...')
                
        class_to_store = self.prepare_for_save(class_to_store=class_to_store)

        with open(file, "wb") as handle:
            pickle_dill.dump(class_to_store, handle, protocol=4)
            
def restore_class_pickle(file):
    
    print('...Restoring pickled simulation class...')
    
    return IOtools.unpickle_mitim(file)

def change_and_write_code(
    rhos,
    inputs0,
    Folder_sim,
    code_settings=None,
    extraOptions={},
    multipliers={},
    minimum_delta_abs={},
    ApplyCorrections=True,
    Quasineutral=False,
    addControlFunction=None,
    controls_file='input.tglf.controls',
    **kwargs
):
    """
    Received inputs classes and gives text.
    ApplyCorrections refer to removing ions with too low density and that are fast species
    """

    inputs = copy.deepcopy(inputs0)

    mod_input_file = {}
    ns_max = []
    for i, rho in enumerate(rhos):
        print(f"\t- Changing input file for rho={rho:.4f}")
        input_sim_rho = modifyInputs(
            inputs[rho],
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            position_change=i,
            addControlFunction=addControlFunction,
            controls_file=controls_file,
            NS=inputs[rho].num_recorded,
            allocation=kwargs.get("allocation", None),
        )

        input_file = input_sim_rho.file.name.split('_')[0]

        newfile = Folder_sim / f"{input_file}_{rho:.4f}"

        if code_settings is not None:
            # Apply corrections
            if ApplyCorrections:
                print("\t- Applying corrections")
                input_sim_rho.removeLowDensitySpecie()
                input_sim_rho.remove_fast()

            # Ensure that plasma to run is quasineutral
            if Quasineutral:
                input_sim_rho.ensureQuasineutrality()
        else:
            print('\t- Not applying corrections because settings is None')

        input_sim_rho.write_state(file=newfile)

        mod_input_file[rho] = input_sim_rho

        ns_max.append(inputs[rho].num_recorded)
        
    # Convert back to a string because that's how the run operates
    inputFile = inputToVariable(Folder_sim, rhos, file=input_file)

    if (np.diff(ns_max) > 0).any():
        print("> Each radial location has its own number of species... probably because of removal of fast or low density...",typeMsg="w")
        print("\t * Reading of simulation results will fail... consider doing something before launching run",typeMsg="q")

    return inputFile, mod_input_file

def inputToVariable(folder, rhos, file='input.tglf'):
    """
    Entire text file to variable
    """

    inputFilesTGLF = {}
    for rho in rhos:
        fileN = folder / f"{file}_{rho:.4f}"

        with open(fileN, "r") as f:
            lines = f.readlines()
        inputFilesTGLF[rho] = "".join(lines)

    return inputFilesTGLF

def cold_start_checker(
    rhos,
    output_files_simulation_select,
    Folder_sim,
    cold_start=False,
    print_each_time=False,
):
    """
    This function checks if the TGLF inputs are already in the folder. If they are, it returns True
    """
    cont_each = 0
    if cold_start:
        rhosEvaluate = rhos
    else:
        rhosEvaluate = []
        for ir in rhos:
            existsRho = True
            for j in output_files_simulation_select:
                ffi = Folder_sim / f"{j}_{ir:.4f}"
                existsThis = ffi.exists()
                existsRho = existsRho and existsThis
                if not existsThis:
                    if print_each_time:
                        print(f"\t* {ffi} does not exist")
                    else:
                        cont_each += 1
            if not existsRho:
                rhosEvaluate.append(ir)

    if not print_each_time and cont_each > 0:
        print(f'\t* {cont_each} files from expected set are missing')

    if len(rhosEvaluate) < len(rhos) and len(rhosEvaluate) > 0:
        print("~ Not all radii are found, but not removing folder and running only those that are needed",typeMsg="i",)

    return rhosEvaluate

def modifyInputs(
    input_class,
    code_settings=None,
    extraOptions=None,
    multipliers=None,
    minimum_delta_abs=None,
    position_change=0,
    addControlFunction=None,
    controls_file = 'input.tglf.controls',
    **kwargs_to_function,
):

    if extraOptions is None:
        extraOptions = {}
    if multipliers is None:
        multipliers = {}
    if minimum_delta_abs is None:
        minimum_delta_abs = {}

    # Check that those are valid flags
    GACODEdefaults.review_controls(extraOptions, control = controls_file)
    GACODEdefaults.review_controls(multipliers, control = controls_file)
    # -------------------------------------------

    if code_settings is not None:
        CodeOptions = addControlFunction(code_settings, extraOptions=extraOptions,**kwargs_to_function)

        # ~~~~~~~~~~ Change with presets
        print(f" \t- Using presets code_settings = {code_settings}", typeMsg="i")
        input_class.controls = CodeOptions

    else:
        print("\t- Input file was not modified by code_settings, using what was there before",typeMsg="i")

    # Make all upper case
    #extraOptions = {ikey.upper(): value for ikey, value in extraOptions.items()}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Change with external options -> Input directly, not as multiplier
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if len(extraOptions) > 0:
        print("\t- External options:")
    for ikey in extraOptions:
        if isinstance(extraOptions[ikey], (list, np.ndarray)):
            value_to_change_to = extraOptions[ikey][position_change]
        else:
            value_to_change_to = extraOptions[ikey]
            
        try:
            isspecie = ikey.split("_")[0] in input_class.species[1]
        except:
            isspecie = False

        # is a species parameter?
        if isspecie:
            specie = int(ikey.split("_")[-1])
            varK = "_".join(ikey.split("_")[:-1])
            var_orig = input_class.species[specie][varK]
            var_new = value_to_change_to
            input_class.species[specie][varK] = var_new
        # is a another parameter?
        else:
            if ikey in input_class.controls:
                var_orig = input_class.controls[ikey]
                var_new = value_to_change_to
                input_class.controls[ikey] = var_new
            elif ikey in input_class.plasma:
                var_orig = input_class.plasma[ikey]
                var_new = value_to_change_to
                input_class.plasma[ikey] = var_new
            else:
                # If the variable in extraOptions wasn't in there, consider it a control param
                print(f"\t\t- Variable {ikey} to change did not exist previously, creating now",typeMsg="i")
                var_orig = None
                var_new = value_to_change_to
                input_class.controls[ikey] = var_new

        print(f"\t\t- Changing {ikey} from {var_orig} to {var_new}",typeMsg="i",)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Change with multipliers -> Input directly, not as multiplier
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if len(multipliers) > 0:
        print("\t\t- Variables change:")
    for ikey in multipliers:
    
        if isinstance(multipliers[ikey], (list, np.ndarray)):
            value_to_change_to = multipliers[ikey][position_change]
        else:
            value_to_change_to = multipliers[ikey]
    
        # is a specie one?
        if "species" in input_class.__dict__.keys() and ikey.split("_")[0] in input_class.species[1]:
            specie = int(ikey.split("_")[-1])
            varK = "_".join(ikey.split("_")[:-1])
            var_orig = input_class.species[specie][varK]
            var_new = multiplier_input(var_orig, value_to_change_to, minimum_delta_abs = minimum_delta_abs.get(ikey,None))
            input_class.species[specie][varK] = var_new
        else:
            if ikey in input_class.controls:
                var_orig = input_class.controls[ikey]
                var_new = multiplier_input(var_orig, value_to_change_to, minimum_delta_abs = minimum_delta_abs.get(ikey,None))
                input_class.controls[ikey] = var_new
            
            elif ikey in input_class.plasma:
                var_orig = input_class.plasma[ikey]
                var_new = multiplier_input(var_orig, value_to_change_to, minimum_delta_abs = minimum_delta_abs.get(ikey,None))
                input_class.plasma[ikey] = var_new
            
            else:
                print("\t- Variable to scan did not exist in original file, add it as extraOptions first",typeMsg="w",)

        print(f"\t\t\t- Changing {ikey} from {var_orig} to {var_new} (x{value_to_change_to})")

    return input_class

def multiplier_input(var_orig, multiplier, minimum_delta_abs = None):

    delta = var_orig * (multiplier - 1.0)

    if minimum_delta_abs is not None:
        if (multiplier != 1.0) and abs(delta) < minimum_delta_abs:
            print(f"\t\t\t- delta = {delta} is smaller than minimum_delta_abs = {minimum_delta_abs}, enforcing",typeMsg="i")
            delta = np.sign(delta) * minimum_delta_abs

    return var_orig + delta

def buildDictFromInput(inputFile):
    parsed = {}

    lines = inputFile.split("\n")
    for line in lines:
        if "=" in line:
            splits = [i.split()[0] for i in line.split("=")]
            if ("." in splits[1]) and (splits[1][0].split()[0] != "."):
                try:
                    parsed[splits[0].split()[0]] = float(splits[1].split()[0])
                    continue
                except:
                    pass
                    
            try:
                parsed[splits[0].split()[0]] = int(splits[1].split()[0])
            except:
                parsed[splits[0].split()[0]] = splits[1].split()[0]

    for i in parsed:
        if isinstance(parsed[i], str):
            if (
                parsed[i].lower() == "t"
                or parsed[i].lower() == "true"
                or parsed[i].lower() == ".true."
            ):
                parsed[i] = True
            elif (
                parsed[i].lower() == "f"
                or parsed[i].lower() == "false"
                or parsed[i].lower() == ".false."
            ):
                parsed[i] = False

    return parsed

class GACODEoutput:
    def __init__(self, *args, **kwargs):
        self.inputFile = None

    def unnormalize(self, *args, **kwargs):
        print("No unnormalization implemented.")

class GACODEinput:
    def __init__(self, file=None, controls_file=None, code='', n_species=None):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None
        
        self.controls_file = controls_file
        self.code = code
        self.n_species = n_species
        
        self.num_recorded = 100

        if self.file is not None and self.file.exists():
            with open(self.file, "r") as f:
                lines = f.readlines()
            file_txt = "".join(lines)
        else:
            file_txt = ""
        input_dict = buildDictFromInput(file_txt)

        self.process(input_dict)

    @classmethod
    def initialize_in_memory(cls, input_dict):
        instance = cls()
        instance.process(input_dict)
        return instance

    def process(self, input_dict):

        if self.controls_file is not None:
            options_check = [key for key in IOtools.generateMITIMNamelist(self.controls_file, caseInsensitive=False).keys()]
        else:
            options_check = []

        self.controls, self.plasma = {}, {}
        for key in input_dict.keys():
            if key in options_check:
                self.controls[key] = input_dict[key]
            else:
                self.plasma[key] = input_dict[key]

        # Get number of recorded species
        if self.n_species is not None and self.n_species in input_dict:
            self.num_recorded = int(input_dict[self.n_species])

    def write_state(self, file=None):
        
        if file is None:
            file = self.file

        # Local formatter: floats -> 6 significant figures in exponential (uppercase),
        # ints stay as ints, bools as 0/1, sequences space-separated with same rule.
        def _fmt_num(x):
            import numpy as _np
            if isinstance(x, (bool, _np.bool_)):
                return "True" if x else "False"
            if isinstance(x, (_np.floating, float)):
                # 6 significant figures in exponential => 5 digits after decimal
                return f"{float(x):.5E}"
            if isinstance(x, (_np.integer, int)):
                return f"{int(x)}"
            return str(x)

        def _fmt_value(val):
            import numpy as _np
            if isinstance(val, (list, tuple, _np.ndarray)):
                # Flatten numpy arrays but keep ordering; join with spaces
                if isinstance(val, _np.ndarray):
                    flat = val.flatten().tolist()
                else:
                    flat = list(val)
                return " ".join(_fmt_num(v) for v in flat)
            return _fmt_num(val)

        with open(file, "w") as f:
            f.write("#-------------------------------------------------------------------------\n")
            f.write(f"# {self.code} input file modified by MITIM {mitim_version}\n")
            f.write("#-------------------------------------------------------------------------\n")

            f.write("\n\n# Control parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.controls:
                var = self.controls[ikey]
                f.write(f"{ikey.ljust(23)} = {_fmt_value(var)}\n")

            f.write("\n\n# Plasma/Geometry parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.plasma:
                var = self.plasma[ikey]
                f.write(f"{ikey.ljust(23)} = {_fmt_value(var)}\n")

    def anticipate_problems(self):
        pass

    def remove_fast(self):
        pass

    def removeLowDensitySpecie(self, *args):
        pass
    