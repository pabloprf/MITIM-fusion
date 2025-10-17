import shutil
import datetime
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

class mitim_simulation:
    '''
    Main class for running GACODE simulations.
    '''
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
        slurm_setup=None,  # Cores per call (so, when running nR radii -> nR*4)
        attempts_execution=1,
        only_minimal_files=False,
        run_type = 'normal', # 'normal': send, submit and wait; 'submit': send and submit and do not wait; 'send': send and do not submit; 'prep': do not submit
        additional_files_to_send = None, # Dict (rho keys) of files to send along with the run (e.g. for restart)
        helper_lostconnection=False, # If True, it means that the connection to the remote machine was lost, but the files are there, so I just want to retrieve them not execute the commands
    ):
        
        if slurm_setup is None:
            slurm_setup = {"cores": self.run_specifications['default_cores'], "minutes": 10}

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
            slurm_setup=slurm_setup,
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
            slurm_setup=slurm_setup,
            only_minimal_files=only_minimal_files,
            attempts_execution=attempts_execution,
            run_type=run_type,
            helper_lostconnection=helper_lostconnection,
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
        slurm_setup=None, 
        # ********************************
        # Additional files to send (e.g. restarts). Must be a dictionary with rho keys
        # ********************************
        additional_files_to_send = None,
        # ********************************
        # Additional settings to correct/modify inputs
        # ********************************
        **kwargs_control
        ):

        if slurm_setup is None:
            slurm_setup = {"cores": self.run_specifications['default_cores'], "minutes": 5}

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
            slurm_setup=slurm_setup,
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
        #     self._check_cores(rhosEvaluate, slurm_setup)

        self.FolderSimLast = Folder_sim
            
        return code_executor, code_executor_full

    def _check_cores(self, rhosEvaluate, slurm_setup, warning = 32 * 2):
        expected_allocated_cores = int(len(rhosEvaluate) * slurm_setup["cores"])
        
        print(f'\t- Slurm job will be submitted with {expected_allocated_cores} cores ({len(rhosEvaluate)} radii x {slurm_setup["cores"]} cores/radius)',
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

        c = 0
        for subfolder_simulation in code_executor:
            c += len(code_executor[subfolder_simulation])

        if c == 0:
            
            print(f"\t- {self.run_specifications['code'].upper()} not run because all results files found (please ensure consistency!)",typeMsg="i")
        
            self.simulation_job = None
        
        else:
            
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
            code_slurm_settings = self.run_specifications.get('code_slurm_settings', None)  
        
            # Get execution info
            minutes = kwargs_run.get("slurm_setup", {}).get("minutes", 5)
            cores_per_code_call = kwargs_run.get("slurm_setup", {}).get("cores", self.run_specifications['default_cores'])
            launchSlurm = kwargs_run.get("launchSlurm", True)
            
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
            machineSettings = FARMINGtools.mitim_job.grab_machine_settings(code)
            max_cores_per_node = machineSettings["cores_per_node"]

            # If the run is local and not slurm, let's check the number of cores
            if (machineSettings["machine"] == "local") and \
                not (launchSlurm and ("partition" in self.simulation_job.machineSettings["slurm"])):
                
                cores_in_machine = int(os.cpu_count())
                cores_allocated = int(os.environ.get('SLURM_CPUS_PER_TASK')) if os.environ.get('SLURM_CPUS_PER_TASK') is not None else None

                if cores_allocated is not None:
                    if max_cores_per_node is None or (cores_allocated < max_cores_per_node):
                        print(f"\t- Detected {cores_allocated} cores allocated by SLURM, using this value as maximum for local execution (vs {max_cores_per_node} specified as available)",typeMsg="i")
                        max_cores_per_node = cores_allocated
                elif cores_in_machine is not None:
                    if max_cores_per_node is None or (cores_in_machine < max_cores_per_node):
                        print(f"\t- Detected {cores_in_machine} cores in machine, using this value as maximum for local execution (vs {max_cores_per_node} specified as available)",typeMsg="i")
                        max_cores_per_node = cores_in_machine
                else:
                    # Default to just 16 just in case
                    if max_cores_per_node is None: 
                        max_cores_per_node = 16
            else:
                # For remote execution, default to just 16 just in case
                if max_cores_per_node is None: 
                    max_cores_per_node = 16
            # ---------------------------------------------------------------------------

            # Grab the total number of cores of this job --------------------------------
            total_simulation_executions = len(rhos) * len(code_executor)
            total_cores_required = int(cores_per_code_call) * total_simulation_executions
            # ---------------------------------------------------------------------------

            # If it's GPUS enable machine, do the comparison based on it
            if machineSettings['gpus_per_node'] == 0:
                max_cores_per_node_compare = max_cores_per_node
            else:
                print(f"\t- Detected {machineSettings['gpus_per_node']} GPUs in machine, using this value as maximum for non-array execution (vs {max_cores_per_node} specified as available)",typeMsg="i")
                max_cores_per_node_compare = machineSettings['gpus_per_node']

            if not (launchSlurm and ("partition" in self.simulation_job.machineSettings["slurm"])):
                type_of_submission = "bash"
            elif total_cores_required < max_cores_per_node_compare:
                type_of_submission = "slurm_standard"
            elif total_cores_required >= max_cores_per_node_compare:
                type_of_submission = "slurm_array"

            shellPreCommands, shellPostCommands = None, None

            # Simply bash, no slurm
            if type_of_submission == "bash":

                if cores_per_code_call > max_cores_per_node:
                    print(f"\t- Detected {cores_per_code_call} cores required, using this value as maximum for local execution (vs {max_cores_per_node} specified as available)",typeMsg="i")
                    max_cores_per_node = cores_per_code_call
                
                max_parallel_execution = max_cores_per_node // cores_per_code_call # Make sure we don't overload the machine when running locally (assuming no farming trans-node)

                print(f"\t- {code.upper()} will be executed as bash script (total cores: {total_cores_required},  cores per simulation: {cores_per_code_call}). MITIM will launch {total_simulation_executions // max_parallel_execution+1} sequential executions",typeMsg="i")

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
                GACODEcommand += f'    {code_call(folder=folder_str, n=cores_per_code_call, p=self.simulation_job.folderExecution)} &\n'
                GACODEcommand += "    while (( $(jobs -r | wc -l) >= max_parallel_execution )); do sleep 1; done\n"
                GACODEcommand += "done\n\n"
                GACODEcommand += "wait\n"

            # Standard job
            elif type_of_submission == "slurm_standard":

                print(f"\t- {code.upper()} will be executed in SLURM as standard job (cpus: {total_cores_required})",typeMsg="i")

                # Code launches
                GACODEcommand = ""
                for folder in folders_red:
                    GACODEcommand += f'    {code_call(folder = folder, n = cores_per_code_call, p = self.simulation_job.folderExecution)}  &\n'
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
                    n = cores_per_code_call,
                    p = self.simulation_job.folderExecution,
                    additional_command = f'1> {self.simulation_job.folderExecution}/{indexed_folder}/slurm_output.dat 2> {self.simulation_job.folderExecution}/{indexed_folder}/slurm_error.dat\n')

            # ---------------------------------------------
            # Execute
            # ---------------------------------------------

            slurm_settings = code_slurm_settings(
                name=code+'_sim',
                minutes=minutes,
                total_cores_required=total_cores_required,
                cores_per_code_call=cores_per_code_call,
                type_of_submission=type_of_submission,
                array_list=array_list if type_of_submission == "slurm_array" else None,
                raise_warning= run_type == 'normal'
            )

            self.simulation_job.define_machine(
                code,
                f"mitim_{name}",
                launchSlurm=launchSlurm,
                slurm_settings=slurm_settings,
            )
            
            # I would like the mitim_job to check if the retrieved folders were complete
            files_we_care_about = {}
            for folder in folders_red:
                files_we_care_about[folder] = filesToRetrieve
            # ---------------------------------------------

            self.simulation_job.prep(
                GACODEcommand,
                input_folders=folders,
                output_folders=folders_red,
                output_folders_selective=files_we_care_about,
                check_files_in_folder=files_we_care_about,
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
                    
                self._organize_results(code_executor, tmpFolder, filesToRetrieve)

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
                    "filesToRetrieve": filesToRetrieve
                }

                self.slurm_output = "slurm_output.dat"

                # Prepare how to search for the job without waiting for it
                self.simulation_job.launchSlurm = True
                self.simulation_job.slurm_settings['name'] = Path(self.simulation_job.folderExecution).name

    def check(self, every_n_minutes=None):

        if self.simulation_job is None:
            print("- Not checking status because simulation job is not defined (not run)", typeMsg="i")
            return

        if self.simulation_job.launchSlurm:
            print("- Checker job status")

            while True:
                self.simulation_job.check(file_output = self.slurm_output)
                print(f'\t- Current status (as of  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}): {self.simulation_job.status} ({self.simulation_job.infoSLURM["STATE"]})')
                if self.simulation_job.status == 2:
                    print("\n\t* Job considered finished (please do .fetch() to retrieve results)",typeMsg="i")
                    break
                elif every_n_minutes is None:
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

    def _organize_results(self, code_executor, tmpFolder, filesToRetrieve):
    
        # ---------------------------------------------
        # Organize
        # ---------------------------------------------

        print("\t- Retrieving files and changing names for storing")
        fineall = True
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

        if fineall:
            print("\t\t- All files were successfully retrieved")

            # Remove temporary folder
            IOtools.shutil_rmtree(tmpFolder)

        else:
            print("\t\t- Some files were not retrieved", typeMsg="w")

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
        multipliers={},
        minimum_delta_abs={},
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        variables_scanTogether=[],
        relativeChanges=True,
        **kwargs_run,
    ):
        """
        Multipliers will be modified by adding the scaning variables, but I don't want to modify the original
        multipliers, as they may be passed to the next scan

        Set relativeChanges=False if varUpDown contains the exact values to change, not multipleiers
        """
        
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
        positionIon=2,
        variable_mapping=None,
        variable_mapping_unn=None
    ):
        '''
        positionIon is the index in the input.tglf file... so if you want for ion RLNS_5, positionIon=5
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

        self.positionIon_scan = positionIon

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
            slurm_setup=kwargs.get("slurm_setup", None),
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
    