import shutil
import datetime
import time
import os
import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mitim_tools import __version__ as mitim_version
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import GACODEdefaults, NORMtools
from mitim_tools.transp_tools.utils import NTCCtools
from mitim_tools.misc_tools import FARMINGtools, IOtools, MATHtools, GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

class gacode_simulation:
    '''
    Main class for running GACODE simulations.
    '''
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        self.rhos = np.array(rhos) if rhos is not None else None

        self.ResultsFiles = []
        self.ResultsFiles_minimal = []
        
        self.nameRunid = "0"
        
        self.results, self.scans = {}, {}
        
        self.run_specifications = None

    def prep(
        self,
        mitim_state,    # A MITIM state class
        FolderGACODE,  # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,  # If True, do not use what it potentially inside the folder, run again
        forceIfcold_start=False,  # Extra flag
        ):
        '''
        This method prepares the GACODE run from a MITIM state class by setting up the necessary input files and directories.
        '''

        print("> Preparation run from input.gacode (direct conversion)")

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
        self.profiles.write_state(file=self.FolderGACODE / "input.gacode")

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
        run_type = 'normal', # 'normal': submit and wait; 'submit': submit and do not wait; 'prep': do not submit
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
            run_type=run_type
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

        ResultsFiles_new = []
        for i in self.ResultsFiles:
            if "mitim.out" not in i:
                ResultsFiles_new.append(i)
        self.ResultsFiles = ResultsFiles_new

        if only_minimal_files:
            filesToRetrieve = self.ResultsFiles_minimal
        else:
            filesToRetrieve = self.ResultsFiles

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
            filesToRetrieve = self.ResultsFiles_minimal
        else:
            filesToRetrieve = self.ResultsFiles

        c = 0
        for subfolder_simulation in code_executor:
            c += len(code_executor[subfolder_simulation])

        if c == 0:
            
            print(f"\t- {self.run_specifications['code'].upper()} not run because all results files found (please ensure consistency!)",typeMsg="i")
        
        else:
            
            # ----------------------------------------------------------------------------------------------------------------
            # Run simulation
            # ----------------------------------------------------------------------------------------------------------------
            """
            launchSlurm = True -> Launch as a batch job in the machine chosen
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

            self.simulation_job = FARMINGtools.mitim_job(tmpFolder)

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
                        print(f"\t - Detected {cores_allocated} cores allocated by SLURM, using this value as maximum for local execution (vs {max_cores_per_node} specified)",typeMsg="i")
                        max_cores_per_node = cores_allocated
                elif cores_in_machine is not None:
                    if max_cores_per_node is None or (cores_in_machine < max_cores_per_node):
                        print(f"\t - Detected {cores_in_machine} cores in machine, using this value as maximum for local execution (vs {max_cores_per_node} specified)",typeMsg="i")
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
                print(f"\t - Detected {machineSettings['gpus_per_node']} GPUs in machine, using this value as maximum for non-arra execution (vs {max_cores_per_node} specified)",typeMsg="i")
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
                    print(f"\t- Detected {cores_per_code_call} cores required, using this value as maximum for local execution (vs {max_cores_per_node} specified)",typeMsg="i")
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
                GACODEcommand += f'    {code_call(folder = '\"$folder\"', n = cores_per_code_call, p = self.simulation_job.folderExecution)}\n'
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

                # As a pre-command, organize all folders in a simpler way
                shellPreCommands = []
                shellPostCommands = []
                array_list = []
                for i, folder in enumerate(folders_red):
                    array_list.append(f"{i}")
                    folder_temp_array = f"run{i}"
                    folder_actual = folder
                    shellPreCommands.append(f"mkdir {self.simulation_job.folderExecution}/{folder_temp_array}; cp {self.simulation_job.folderExecution}/{folder_actual}/*  {self.simulation_job.folderExecution}/{folder_temp_array}/.")
                    shellPostCommands.append(f"cp {self.simulation_job.folderExecution}/{folder_temp_array}/* {self.simulation_job.folderExecution}/{folder_actual}/.; rm -r {self.simulation_job.folderExecution}/{folder_temp_array}")

                # Code launches
                indexed_folder = 'run"$SLURM_ARRAY_TASK_ID"'
                GACODEcommand = code_call(
                    folder = indexed_folder,
                    n = cores_per_code_call,
                    p = self.simulation_job.folderExecution,
                    additional_command = f'1> {self.simulation_job.folderExecution}/{indexed_folder}/slurm_output.dat 2> {self.simulation_job.folderExecution}/{indexed_folder}/slurm_error.dat\n')

            # ---------------------------------------------
            # Execute
            # ---------------------------------------------

            slurm_settings = code_slurm_settings(
                name=code,
                minutes=minutes,
                total_cores_required=total_cores_required,
                cores_per_code_call=cores_per_code_call,
                type_of_submission=type_of_submission,
                array_list=array_list if type_of_submission == "slurm_array" else None
            )

            self.simulation_job.define_machine(
                code,
                f"mitim_{name}",
                launchSlurm=launchSlurm,
                slurm_settings=slurm_settings,
            )
            
            # I would like the mitim_job to check if the retrieved folders were complete
            check_files_in_folder = {}
            for folder in folders_red:
                check_files_in_folder[folder] = filesToRetrieve
            # ---------------------------------------------

            self.simulation_job.prep(
                GACODEcommand,
                input_folders=folders,
                output_folders=folders_red,
                check_files_in_folder=check_files_in_folder,
                shellPreCommands=shellPreCommands,
                shellPostCommands=shellPostCommands,
            )

            # Submit run and wait
            if run_type == 'normal':
            
                self.simulation_job.run(
                    removeScratchFolders=True,
                    attempts_execution=attempts_execution
                    )
                    
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

            for i, rho in enumerate(code_executor[subfolder_sim].keys()):
                for file in filesToRetrieve:
                    original_file = f"{file}_{rho:.4f}"
                    final_destination = (
                        code_executor[subfolder_sim][rho]['folder'] / f"{original_file}"
                    )
                    final_destination.unlink(missing_ok=True)

                    temp_file = tmpFolder / subfolder_sim / f"rho_{rho:.4f}" / f"{file}"
                    temp_file.replace(final_destination)

                    fineall = fineall and final_destination.exists()

                    if not final_destination.exists():
                        print(f"\t!! file {file} ({original_file}) could not be retrived",typeMsg="w",)

        if fineall:
            print("\t\t- All files were successfully retrieved")

            # Remove temporary folder
            shutil.rmtree(tmpFolder)

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

        class_output = [self.run_specifications['output_class'], self.run_specifications['output_store']]

        # If no specified folder, check the last one
        if folder is None:
            folder = self.FolderSimLast
            
        self.results[label] = {
            class_output[1]:[],
            'parsed': [],
            "x": np.array(self.rhos),
            }
        for rho in self.rhos:

            SIMout = class_output[0](
                folder,
                suffix=f"_{rho:.4f}" if suffix is None else suffix,
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

            self.results[label][class_output[1]].append(SIMout)

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
    ResultsFiles,
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
            for j in ResultsFiles:
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


def runTGYRO(
    folderWork,
    outputFiles=None,
    nameRunid="",
    nameJob="tgyro_mitim",
    nparallel=8,
    minutes=30,
    inputFiles=None,
    launchSlurm=True,
):
    """
    This is the auxiliary function that TGYROtools will call to run TGYRO. It must be standalone.
    ------------------------------------------------------------------------------------------------
    launchSlurm = True 	-> Launch as a batch job in the machine chosen
    launchSlurm = False -> Launch locally as a bash script
    """

    if outputFiles is None:
        outputFiles = []
    
    if inputFiles is None:
        inputFiles = []

    # This routine assumes that folderWork contains input.profiles and input.tgyro already

    tgyro_job = FARMINGtools.mitim_job(folderWork)

    tgyro_job.define_machine(
        "tgyro",
        f"mitim_{nameRunid}",
        launchSlurm=launchSlurm,
        slurm_settings={
            "minutes": minutes,
            "ntasks": 1,
            "name": nameJob,
            "cpuspertask": nparallel,
        },
    )

    # ------ Run TGYRO

    inputFiles.append(folderWork / "input.tglf")
    inputFiles.append(folderWork / "input.tgyro")
    inputFiles.append(folderWork / "input.gacode")

    # ---------------
    # Execution command
    # ---------------

    folderExecution = tgyro_job.machineSettings["folderWork"]

    TGYROcommand = f"tgyro -e . -n {nparallel} -p {folderExecution}"

    # Before calling tgyro, create TGLF folders and place input.tglfs there
    shellPreCommands = [
        f"for i in `seq 1 {nparallel}`;",
        "do",
        "	mkdir TGLF$i",
        "	cp input.tglf TGLF$i/input.tglf",
        "done",
    ]

    # After calling tgyro, move the out.tglf.localdump files
    shellPostCommands = [
        f"for i in `seq 1 {nparallel}`;",
        "do",
        "	cp TGLF$i/out.tglf.localdump input.tglf.new$i",
        "done",
    ]

    # ---------------------------------------------
    # Execute
    # ---------------------------------------------

    tgyro_job.prep(
        TGYROcommand,
        input_files=inputFiles,
        output_files=outputFiles,
        shellPreCommands=shellPreCommands,
        shellPostCommands=shellPostCommands,
    )

    tgyro_job.run()


def modifyInputs(
    input_class,
    code_settings=None,
    extraOptions={},
    multipliers={},
    minimum_delta_abs={},
    position_change=0,
    addControlFunction=None,
    controls_file = 'input.tglf.controls',
    **kwargs_to_function,
):

    # Check that those are valid flags
    GACODEdefaults.review_controls(extraOptions, control = controls_file)
    GACODEdefaults.review_controls(multipliers, control = controls_file)
    # -------------------------------------------

    if code_settings is not None:
        CodeOptions = addControlFunction(code_settings, **kwargs_to_function)

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
        # is a specie one?
        if "species" in input_class.__dict__.keys() and ikey.split("_")[0] in input_class.species[1]:
            specie = int(ikey.split("_")[-1])
            varK = "_".join(ikey.split("_")[:-1])
            var_orig = input_class.species[specie][varK]
            var_new = multiplier_input(var_orig, multipliers[ikey], minimum_delta_abs = minimum_delta_abs.get(ikey,None))
            input_class.species[specie][varK] = var_new
        else:
            if ikey in input_class.controls:
                var_orig = input_class.controls[ikey]
                var_new = multiplier_input(var_orig, multipliers[ikey], minimum_delta_abs = minimum_delta_abs.get(ikey,None))
                input_class.controls[ikey] = var_new
            
            elif ikey in input_class.plasma:
                var_orig = input_class.plasma[ikey]
                var_new = multiplier_input(var_orig, multipliers[ikey], minimum_delta_abs = minimum_delta_abs.get(ikey,None))
                input_class.plasma[ikey] = var_new
            
            else:
                print("\t- Variable to scan did not exist in original file, add it as extraOptions first",typeMsg="w",)

        print(f"\t\t\t- Changing {ikey} from {var_orig} to {var_new} (x{multipliers[ikey]})")

    return input_class

def multiplier_input(var_orig, multiplier, minimum_delta_abs = None):

    delta = var_orig * (multiplier - 1.0)

    if minimum_delta_abs is not None:
        if (multiplier != 1.0) and abs(delta) < minimum_delta_abs:
            print(f"\t\t\t- delta = {delta} is smaller than minimum_delta_abs = {minimum_delta_abs}, enforcing",typeMsg="i")
            delta = np.sign(delta) * minimum_delta_abs

    return var_orig + delta

def findNamelist(LocationCDF, folderWork=None, nameRunid="10000", ForceFirst=True):
    # -----------------------------------------------------------
    # Find namelist
    # -----------------------------------------------------------

    LocationCDF = IOtools.expandPath(LocationCDF)
    Folder = LocationCDF.parent
    print(f"\t- Looking for namelist in folder ...{IOtools.clipstr(Folder)}")

    LocationNML = IOtools.findFileByExtension(Folder, "TR.DAT", ForceFirst=ForceFirst)

    # -----------------------------------------------------------
    # Copy to folder or create dummy if it has not been found
    # -----------------------------------------------------------

    if LocationNML is None:
        LocationNML = folderWork / f"{nameRunid}TR.DAT"
        print("\t\t- Creating dummy namelist because it was not found in folder",typeMsg="i",)
        with open(LocationNML, "w") as f:
            f.write(f"nshot = {nameRunid}")
        dummy = True
    else:
        dummy = False

    return LocationNML, dummy


def prepareTGYRO(
    LocationCDF,
    LocationNML,
    timeRun,
    avTime=0.0,
    BtIp_dirs=[0, 0],
    fixPlasmaState=True,
    gridsTRXPL=[151, 101, 101],
    folderWork="~/scratchFolder/",
    StateGenerated=False,
    sendState=True,
    nameRunid="",
    includeGEQ=True,
):
    nameWork = "10001"
    folderWork = IOtools.expandPath(folderWork)

    if not StateGenerated:
        print("\t- Running TRXPL to extract g-file and plasmastate")
        CDFtoTRXPLoutput(
            LocationCDF,
            LocationNML,
            timeRun,
            avTime=avTime,
            BtIp_dirs=BtIp_dirs,
            nameOutputs=nameWork,
            folderWork=folderWork,
            grids=gridsTRXPL,
            sendState=sendState,
        )

    print("\t- Running PROFILES_GEN to generate input.profiles/input.gacode files")
    runPROFILES_GEN(
        folderWork,
        nameFiles=nameWork,
        UseMITIMmodification=fixPlasmaState,
        includeGEQ=includeGEQ,
    )


def CDFtoTRXPLoutput(
    LocationCDF,
    LocationNML,
    timeRun,
    avTime=0.0,
    BtIp_dirs=[0, 0],
    nameOutputs="10001",
    folderWork="~/scratchFolder/",
    grids=[151, 101, 101],
    sendState=True,
):
    nameFiles, fail_attempts = "10000", 2
    folderWork = IOtools.expandPath(folderWork)

    folderWork.mkdir(parents=True, exist_ok=True)
    if sendState:
        cdffile = folderWork / f'{nameFiles}.CDF'
        shutil.copy2(LocationCDF, cdffile)
    trfile = folderWork / f'{nameFiles}TR.DAT'
    LocationNML.replace(trfile)

    runTRXPL(
        folderWork,
        timeRun,
        BtDir=BtIp_dirs[0],
        IpDir=BtIp_dirs[1],
        avTime=avTime,
        nameFiles=nameFiles,
        sendState=sendState,
        nameOutputs=nameOutputs,
        grids=grids,
    )

    # Retry for random error
    cont = 1
    while (
        (not (folderWork / f"{nameOutputs}.cdf").exists())
        or (not (folderWork / f"{nameOutputs}.geq").exists())
    ) and cont < fail_attempts:
        print("\t\t- Re-running to see if it was a random error", typeMsg="i")
        cont += 1
        runTRXPL(
            folderWork,
            timeRun,
            BtDir=BtIp_dirs[0],
            IpDir=BtIp_dirs[1],
            avTime=avTime,
            nameFiles=nameFiles,
            sendState=sendState,
            nameOutputs=nameOutputs,
            grids=grids,
        )

def runTRXPL(
    FolderTRXPL,
    timeRun,
    BtDir="1",
    IpDir="1",
    avTime=0.0,
    nameFiles="10000",
    nameOutputs="10001",
    sendState=True,
    grids=[151, 101, 101],
):
    """
    trxpl_path:  #theta pts for 2d splines:
    trxpl_path:  #R pts for cartesian overlay grid:
    trxpl_path:  #Z pts for cartesian overlay grid:

    TRXPL asks for direction:
            trxpl_path:  enter "1" for if Btoroidal is ccw:
              "ccw" means "counter-clockwise looking down from above".
              ...enter "-1" for clockwise orientation.
              ...enter "0" to read orientation from TRANSP data archive.
            trxpl_path:  enter "1" for if Btoroidal is ccw:
    """

    commandTRXPL = f"P\n10000\nA\n{timeRun}\n{avTime}\n{grids[0]}\n{grids[1]}\n{grids[2]}\n{BtDir}\n{IpDir}\nY\nX\nH\nW\n10001\nQ\nQ\nQ"
    with open(FolderTRXPL / "trxpl.in", "w") as f:
        f.write(commandTRXPL)

    if sendState:
        inputFiles = [
            FolderTRXPL / "trxpl.in",
            FolderTRXPL / f"{nameFiles}TR.DAT",
            FolderTRXPL / f"{nameFiles}.CDF",
        ]
    else:
        inputFiles = [
            FolderTRXPL / "trxpl.in",
            FolderTRXPL / f"{nameFiles}TR.DAT",
        ]

    if grids[0] > 301:
        raise Exception("~~~~ Max grid for TRXPL is 301")

    print(f"\t\t- testProceeding to run TRXPL with: {' '.join(commandTRXPL.splitlines())}", typeMsg="i")


    trxpl_job = FARMINGtools.mitim_job(FolderTRXPL)
    trxpl_job.define_machine(
        "trxpl",
        f"mitim_trxpl_{nameOutputs}",
    )

    command = "trxpl < trxpl.in"

    trxpl_job.prep(
        command,
        input_files=inputFiles,
        output_files=[f"{nameOutputs}.cdf", f"{nameOutputs}.geq"],
    )
    trxpl_job.run()


def runPROFILES_GEN(
    FolderTGLF,
    nameFiles="10001",
    UseMITIMmodification=False,
    includeGEQ=True,
):
    runWithoutEqIfFail = False  # If profiles_gen fails, try without the "-g" option

    if UseMITIMmodification:
        print("\t\t- Running modifyPlasmaState")
        shutil.copy2(FolderTGLF / f"{nameFiles}.cdf", FolderTGLF / f"{nameFiles}.cdf_old")
        pls = NTCCtools.Plasmastate(FolderTGLF / f"{nameFiles}.cdf_old")
        pls.modify_default(FolderTGLF / f"{nameFiles}.cdf")

    inputFiles = [
        FolderTGLF / "profiles_gen.sh",
        FolderTGLF / f"{nameFiles}.cdf",
    ]

    if includeGEQ:
        inputFiles.append(FolderTGLF / f"{nameFiles}.geq")

    # **** Write command
    txt = f"profiles_gen -i {nameFiles}.cdf"
    if includeGEQ:
        txt += f" -g {nameFiles}.geq\n"
    else:
        txt += "\n"
    with open(FolderTGLF / "profiles_gen.sh", "w") as f:
        f.write(txt)
    # ******************

    print(f"\t\t- Proceeding to run PROFILES_GEN with: {txt}")

    pgen_job = FARMINGtools.mitim_job(FolderTGLF)
    pgen_job.define_machine(
        "profiles_gen",
        f"mitim_profiles_gen_{nameFiles}",
    )

    pgen_job.prep(
        "bash profiles_gen.sh",
        input_files=inputFiles,
        output_files=["input.gacode"],
    )
    pgen_job.run()

    if (
        runWithoutEqIfFail
        and (not (FolderTGLF / "input.gacode").exists())
        and (includeGEQ)
    ):
        print(
            "\t\t- PROFILE_GEN failed, running without the geqdsk file option to see if that works...",
            typeMsg="w",
        )

        # **** Write command
        txt = f"profiles_gen -i {nameFiles}.cdf\n"
        with open(FolderTGLF / "profiles_gen.sh", "w") as f:
            f.write(txt)
        # ******************

        print(f"\t\t- Proceeding to run PROFILES_GEN with: {txt}")
        pgen_job.run()


def runVGEN(
    workingFolder,
    numcores=32,
    minutes=60,
    vgenOptions={},
    name_run="vgen1",
):
    """
    Driver for the vgen (velocity-generation) capability of NEO.
    This will write a new input.gacode with NEO-computed electric field and/or velocities.

    **** Options
            -er:  Method to compute Er
                            1 = Computing omega0 (Er) from force balance
                            2 = Computing omega0 (Er) from NEO (weak rotation limit)
                            3 = ?????NEO (strong rot)?????
                            4 = Returning given omega0 (Er)
            -vel: Method to compute velocities
                            1 = Computing velocities from NEO (weak rotation limit)
                            2 = Computing velocities from NEO (strong rotation limit)
                            3 = ?????Return given?????
            -in:  Number of ion species (uses default neo template. Otherwise, input.neo must exist)
            -ix:  Which ion to match velocities? Index of ion species to match NEO and given velocities
            -nth: Minimum and maximum theta resolutions (e.g. 17,39)
    """

    workingFolder = IOtools.expandPath(workingFolder)
    vgen_job = FARMINGtools.mitim_job(workingFolder)

    vgen_job.define_machine(
        "profiles_gen",
        f"mitim_vgen_{name_run}",
        slurm_settings={
            "minutes": minutes,
            "ntasks": numcores,
            "name": f"neo_vgen_{name_run}",
        },
    )

    print(
        f"\t- Running NEO (with {vgenOptions['numspecies']} species) to populate w0(rad/s) in input.gacode file"
    )
    print(f"\t\t> Matching ion {vgenOptions['matched_ion']} Vtor")

    options = f"-er {vgenOptions['er']} -vel {vgenOptions['vel']} -in {vgenOptions['numspecies']} -ix {vgenOptions['matched_ion']} -nth {vgenOptions['nth']}"

    # ***********************************

    print(
        f"\t\t- Proceeding to generate Er from NEO run using profiles_gen -vgen ({options})"
    )

    inputgacode_file = workingFolder / f"input.gacode"

    _, nameFile = IOtools.reducePathLevel(inputgacode_file, level=1, isItFile=True)

    command = f"cd {vgen_job.machineSettings['folderWork']} && bash profiles_vgen.sh"
    with open(workingFolder / f"profiles_vgen.sh", "w") as f:
        f.write(f"profiles_gen -vgen -i {nameFile} {options} -n {numcores}")

    # ---------------
    # Execute
    # ---------------

    vgen_job.prep(
        command,
        input_files=[inputgacode_file, workingFolder / f"profiles_vgen.sh"],
        output_files=["slurm_output.dat", "slurm_error.dat"],
    )

    vgen_job.run()

    file_new = workingFolder / "vgen" / "input.gacode"

    return file_new


def buildDictFromInput(inputFile):
    parsed = {}

    lines = inputFile.split("\n")
    for line in lines:
        if "=" in line:
            splits = [i.split()[0] for i in line.split("=")]
            if ("." in splits[1]) and (splits[1][0].split()[0] != "."):
                parsed[splits[0].split()[0]] = float(splits[1].split()[0])
            else:
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


# ----------------------------------------------------------------------
# 						Reading/Writing routines
# ----------------------------------------------------------------------


def obtainFluctuationLevel(
    ky,
    Amplitude,
    rhos,
    a,
    convolution_fun_fluct=None,
    rho_eval=0.6,
    factorTot_to_Perp=1.0,
    printYN=True,
):
    """
    Amplitude must be AMPLITUDE (see my notes), not intensity
    factorTot_to_Perp is applied if I want to convert TOTAL to Perpendicular

    """
    ky_integr = [0, 5]
    integrand = np.array(Amplitude) ** 2
    fluctSim = (
        rhos
        / a
        * np.sqrt(
            integrateSpectrum(
                ky,
                integrand,
                ky_integr[0],
                ky_integr[1],
                convolution_fun_fluct=convolution_fun_fluct,
                rhos=rhos,
                rho_eval=rho_eval,
            )
        )
    )

    if factorTot_to_Perp != 1.0 and printYN:
        print(
            f'\t\t- Fluctuations x{factorTot_to_Perp} to account for "total-to-perp." conversion',
            typeMsg="i",
        )

    return fluctSim * 100.0 * factorTot_to_Perp


def obtainNTphase(
    ky,
    nTphase,
    rhos,
    a,
    convolution_fun_fluct=None,
    rho_eval=0.6,
    factorTot_to_Perp=1.0,
):
    ky_integr = [0, 5]
    x, y = defineNewGrid(ky, nTphase, ky_integr[0], ky_integr[1], kind="linear")
    if convolution_fun_fluct is not None:
        gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
    else:
        gaussW = np.ones(len(x))

    neTe = np.sum(y * gaussW) / np.sum(gaussW)

    return neTe


def integrateSpectrum(
    xOriginal,
    yOriginal,
    xmin,
    xmax,
    convolution_fun_fluct=None,
    rhos=None,
    debug=False,
    rho_eval=0.6,
):
    x, y = defineNewGrid(xOriginal, yOriginal, xmin, xmax, kind="linear")

    if convolution_fun_fluct is not None:
        gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
    else:
        gaussW = np.ones(len(x))

    integ = MATHtools.integrate_definite(x, y * gaussW)

    if debug:
        fig = plt.figure(figsize=(17, 8))
        grid = plt.GridSpec(2, 3, hspace=0.2, wspace=0.35)

        ax = fig.add_subplot(grid[0, 0])
        ax.set_title("Option 1")
        ax1 = ax.twinx()
        ax.plot(x, y, "-o", c="r", markersize=2, label="interpolation")
        ax.plot(xOriginal, yOriginal, "o", c="b", markersize=4, label="TGLF output")
        ax1.plot(x, gaussW, ls="--", lw=0.5, c="k")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fluctuation Intensity ($A^2$)")
        ax1.set_ylabel("Convolution (C)")
        ax1.set_ylim([0, 1])
        ax = fig.add_subplot(grid[1, 0])
        gaussWO = np.interp(xOriginal, x, gaussW)
        ax.plot(x, y * gaussW, "-o", c="r", markersize=2)
        ax.plot(xOriginal, yOriginal * gaussWO, "o", c="b", markersize=4)
        GRAPHICStools.fillGraph(
            ax, x, y * gaussW, y_down=np.zeros(len(x)), y_up=None, alpha=0.2, color="g"
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A^2\\cdot C$")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")

        ax.text(
            0.5,
            0.5,
            f"I = {integ:.2f}",
            horizontalalignment="center",
            transform=ax.transAxes,
        )

        x, y = defineNewGrid(xOriginal, yOriginal, xmin, xmax)
        if convolution_fun_fluct is not None:
            gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
        else:
            gaussW = np.ones(len(x))
        integ = MATHtools.integrate_definite(x, y * gaussW)
        ax = fig.add_subplot(grid[0, 1])
        ax.set_title("Option 2")
        ax1 = ax.twinx()
        ax.plot(x, y, "-o", c="r", markersize=2, label="interpolation")
        ax.plot(xOriginal, yOriginal, "o", c="b", markersize=4, label="TGLF output")
        ax1.plot(x, gaussW, ls="--", lw=0.5, c="k")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fluctuation Intensity ($A^2$)")
        ax1.set_ylabel("Convolution (C)")
        ax1.set_ylim([0, 1])
        ax = fig.add_subplot(grid[1, 1])
        gaussWO = np.interp(xOriginal, x, gaussW)
        ax.plot(x, y * gaussW, "-o", c="r", markersize=2)
        ax.plot(xOriginal, yOriginal * gaussWO, "o", c="b", markersize=4)
        GRAPHICStools.fillGraph(
            ax, x, y * gaussW, y_down=np.zeros(len(x)), y_up=None, alpha=0.2, color="g"
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A^2\\cdot C$")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")

        ax.text(
            0.5,
            0.5,
            f"I = {integ:.2f}",
            horizontalalignment="center",
            transform=ax.transAxes,
        )

        x, y = defineNewGrid(xOriginal, yOriginal, xmin, xmax)
        if convolution_fun_fluct is not None:
            gaussW = convolution_fun_fluct(x, rho_s=rhos * 100, rho_eval=rho_eval)
        else:
            gaussW = np.ones(len(x))
        gaussWO = np.interp(xOriginal, x, gaussW)

        ax = fig.add_subplot(grid[0, 2])
        ax.set_title("Option 3")
        ax1 = ax.twinx()
        # ax.plot(x,y,'-o',c='r',markersize=2,label='interpolation')
        ax.plot(xOriginal, yOriginal, "o", c="b", markersize=4, label="TGLF output")
        ax1.plot(x, gaussW, ls="--", lw=0.5, c="k")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Fluctuation Intensity ($A^2$)")
        ax1.set_ylabel("Convolution (C)")
        ax1.set_ylim([0, 1])
        ax = fig.add_subplot(grid[1, 2])

        x, ys = defineNewGrid(xOriginal, yOriginal * gaussWO, xmin, xmax)
        integ = MATHtools.integrate_definite(x, ys)
        ax.plot(x, ys, "-o", c="r", markersize=2)
        ax.plot(xOriginal, yOriginal * gaussWO, "o", c="b", markersize=4)
        GRAPHICStools.fillGraph(
            ax, x, ys, y_down=np.zeros(len(x)), y_up=None, alpha=0.2, color="g"
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A^2\\cdot C$")
        ax.set_xlim([0, xmax])
        ax.set_xlabel("ky")

        ax.text(
            0.5,
            0.5,
            f"I = {integ:.2f}",
            horizontalalignment="center",
            transform=ax.transAxes,
        )

        plt.show()

    return integ


def defineNewGrid(
    xOriginal1,
    yOriginal1,
    xmin,
    xmax,
    debug=False,
    createZero=True,
    interpolateY=True,
    kind="linear",
):
    """
    if createZero, then it adds a point at x=0
    if createZero and interpolateY, the point to be added is y_new[0] = y_old[0], i.e. extrapolation of first value
    """

    if createZero:
        xOriginal1 = np.insert(xOriginal1, 0, 0, axis=0)
        if interpolateY:
            yOriginal1 = np.insert(yOriginal1, 0, yOriginal1[0], axis=0)
        else:
            yOriginal1 = np.insert(yOriginal1, 0, 0, axis=0)

        # Making sure that xOriginal is monotonically increasing
    xOriginal, yOriginal = [], []
    prev = 0.0
    for i in range(len(xOriginal1)):
        if xOriginal1[i] >= prev:
            xOriginal.append(xOriginal1[i])
            yOriginal.append(yOriginal1[i])
            prev = xOriginal1[i]
        else:
            break

    if xOriginal[0] > xmax:
        print(
            "Wavenumber spectrum is too coarse for fluctuations analysis, using the minimum value",
            typeMsg="w",
        )
        xmax = xOriginal[0] + 1e-4

    f = interp1d(xOriginal, yOriginal, kind=kind)
    x = np.linspace(xOriginal[0], max(xOriginal), int(1e4))
    y = f(x)

    imin = np.argmin(np.abs(x - xmin))  # [i for i, j in enumerate(x) if j>=xmin][0]
    imax = np.argmin(np.abs(x - xmax))  # [i for i, j in enumerate(x) if j>=xmax][0]

    if debug:
        fn = plt.figure()
        ax = fn.add_subplot(111)
        ax.plot(x, y)
        ax.scatter(x, y, label="New points")
        ax.scatter(xOriginal, yOriginal, 100, label="Original points")
        xli = np.array(x[imin:imax])
        yli2 = np.array(y[imin:imax])
        ax.fill_between(xli, yli2, facecolor="r", alpha=0.5)
        ax.set_xlim([0, 1.0])
        ax.set_xlabel("ky")
        ax.set_ylabel("T_fluct")
        ax.legend()

        plt.show()

    return x[imin:imax], y[imin:imax]

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
    