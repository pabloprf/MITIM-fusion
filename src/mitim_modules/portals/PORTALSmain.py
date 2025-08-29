import shutil
import torch
import copy
import numpy as np
import dill as pickle_dill
from functools import partial
from collections import OrderedDict
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.portals import PORTALStools
from mitim_modules.portals.utils import (
    PORTALSinit,
    PORTALSoptimization,
    PORTALSanalysis,
)
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


"""
Reading analysis for PORTALS has more options than standard:
--------------------------------------------------------------------------------------------------------
	Standard:
	**************************************
	  -1:  Only improvement
		0:  Only optimization_results
		1:  0 + Pickle
		2:  1 + Final redone in this machine
		
	PORTALS-specific:
	**************************************
		3:  1 + PORTALSplot metrics  (only works if optimization_extra is provided or Execution exists)
		4:  3 + PORTALSplot expected (only works if optimization_extra is provided or Execution exists)
		5:  2 + 4                    (only works if optimization_extra is provided or Execution exists)

		>2 will also plot profiles & gradients comparison (original, initial, best)
"""

def default_namelist(optimization_options):
    """
    This is to be used after reading the namelist, so self.optimization_options should be completed with main defaults.
    """

    # Initialization
    optimization_options["initialization_options"]["initial_training"] = 5
    optimization_options["initialization_options"]["initialization_fun"] = PORTALSoptimization.initialization_simple_relax

    # Strategy for stopping
    optimization_options["convergence_options"]["maximum_iterations"] = 50
    optimization_options['convergence_options']['stopping_criteria'] = PORTALStools.stopping_criteria_portals
    optimization_options['convergence_options']['stopping_criteria_parameters'] =  {
                "maximum_value": 5e-3,                  # Reducing residual by 200x is enough
                "maximum_value_is_rel": True,
                "minimum_dvs_variation": [10, 5, 0.1],  # After iteration 10, Check if 5 consecutive DVs are varying less than 0.1% from the rest that has been evaluated
                "ricci_value": 0.05,
                "ricci_d0": 2.0,
                "ricci_lambda": 0.5,
            }

    optimization_options['acquisition_options']['relative_improvement_for_stopping'] = 1e-2

    # Surrogate
    optimization_options["surrogate_options"]["surrogate_selection"] = partial(PORTALStools.surrogate_selection_portals)

    # if CGYROrun:
    #     # CGYRO runs should prioritize accuracy
    #     optimization_options["acquisition_options"]["type"] = "posterior_mean"
    #     optimization_options["acquisition_options"]["optimizers"] = ["root", "botorch", "ga"]
    # else:
    #     # TGLF runs should prioritize speed
    optimization_options["acquisition_options"]["type"] = "posterior_mean" # "noisy_logei_mc"
    optimization_options["acquisition_options"]["optimizers"] = ["sr", "root"] #, "botorch"]   

    return optimization_options


class portals(STRATEGYtools.opt_evaluator):
    def __init__(
        self, 
        folder,                             # Folder where the PORTALS workflow will be run
        transport_models = ['tglf', 'neo'],
        namelist=None,                      # If None, default namelist will be used. If not None, it will be read and used
        tensor_options = {
            "dtype": torch.double,
            "device": torch.device("cpu"),
        },
        portals_transformation_variables = None,          # If None, use defaults for both main and trace
        portals_transformation_variables_trace = None,
        additional_params_in_surrogate = [] # Additional parameters to be used in the surrogate (e.g. ['q'])
        ):
        '''
        Note that additional_params_in_surrogate They must exist in the plasma dictionary of the powerstate object
        '''
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t PORTALS class module")
        print("-----------------------------------------------------------------------------------------\n")

        # Store folder, namelist. Read namelist

        super().__init__(
            folder,
            namelist=namelist,
            tensor_options=tensor_options,
            default_namelist_function=(
                partial(default_namelist)
                if (namelist is None)
                else None
            ),
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Default (please change to your desire after instancing the object)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.portals_parameters = {}

        """
		Physics-informed parameters to fit surrogates
		---------------------------------------------
		"""

        (
            portals_transformation_variables,
            portals_transformation_variables_trace,
        ) = PORTALStools.default_portals_transformation_variables(additional_params = additional_params_in_surrogate)

        """
		PORTALS-specific parameters (i.e. on the transport solver level, not model)
		----------------------------------------------------------------------------
		"""

        self.portals_parameters["solution"] = {
            
            # Specification of radial locations (roa wins over rho, if provided)
            "predicted_roa": [0.35, 0.55, 0.75, 0.875, 0.9],
            "predicted_rho": [0.3,  0.45,  0.6,  0.75, 0.9],
            
            # Channels to be predicted
            "predicted_channels": ["te", "ti", "ne"],  # ['nZ','w0']
            
            "trace_impurity": None,  # Impurity to do flux-matching for if nZ enabled (name of first impurity instance AFTER postprocessing), e.g. "W"
            
            # PORTALS
            "portals_transformation_variables": portals_transformation_variables,               # Physics-informed parameters to fit surrogates
            "portals_transformation_variables_trace": portals_transformation_variables_trace,   # Physics-informed parameters to fit surrogates for trace impurities
            "turbulent_exchange_as_surrogate": False,                                           # Run turbulent exchange as surrogate?
            "Pseudo_multipliers": [1.0]*5,                                                      # [Qe,Qi,Ge] multipliers to calculate pseudo
            "impurity_trick": True,                                                             # If True, fit model to GZ/nZ, valid on the trace limit
            "fZ0_as_weight": 1.0,                                                               # If not None, using fZ0_as_weight/fZ_0 as scaling factor for GZ, where fZ_0 is the original impurity concentration on axis
            "fImp_orig": 1.0,
            "additional_params_in_surrogate": additional_params_in_surrogate,
            "keep_full_model_folder": True,                                                     # If False, remove full model folder after evaluation, to avoid large folders (e.g. in MAESTRO runs)
        }

        '''
        model_parameters
        ----------------
        These parameters are communicated to the powertorch object.
        '''
        
        from mitim_modules.powertorch.utils.TRANSPORTtools import portals_transport_model as transport_evaluator

        # -------------------------
        # Transport model settings 
        # -------------------------
        self.portals_parameters["transport"] = {

            # Transport model class
            
            "evaluator": transport_evaluator,

            "evaluator_instance_attributes": {
                "turbulence_model": transport_models[0],
                "neoclassical_model": transport_models[1],
            },

            # Simulation kwargs to be passed directly to run and read commands
            
            "options": {
                
                # Defaults for TGLF simulation
                "tglf": {
                    "run": {
                        "code_settings": 6,
                        "extraOptions": {},
                    },
                    "read": {},
                    "use_scan_trick_for_stds": 0.02,  # If not None, use TGLF scan trick to calculate TGLF errors with this maximum delta
                    "keep_files": "base", # minimal: only retrieve minimal files always; base: retrieve all files when running TGLF base; none: always minimal files
                    "cores_per_tglf_instance": 1,  # Number of cores to use per TGLF instance
                    "launchEvaluationsAsSlurmJobs": True, # Launch each evaluation as a batch job (vs just comand line)
                    "percent_error": 5, # (%) Error (std, in percent) of model evaluation TGLF if not scan trick
                    "Qi_includes_fast": False,  # If True, and fast ions have been included, sum fast. This only occurs if the specie is considered fast by TGLF (it could be fast in input.gacode but thermal for TGLF)                
                },
                
                # Defaults for NEO simulation
                "neo": {
                    "run": {},
                    "read": {},
                    "percent_error": 10 # (%) Error (std, in percent) of model evaluation
                    },
                
                # Defaults for CGYRO simulation
                "cgyro": {
                    "run": {
                        "code_settings": 1,
                        "extraOptions": {},
                        "run_type": "normal",   # 'normal': submit and wait; 'submit': submit and do not wait; 'prep': do not submit
                    },
                    "read": {
                        "tmin": 0.0
                    },
                    "Qi_stable_criterion": 0.01,  # For CGYRO runs, MW/m^2 of Qi below which the case is considered stable
                    "Qi_stable_percent_error": 5.0,  # (%) For CGYRO runs, minimum error based on target if case is considered stable
                },
                # Defaults for GX simulation
                "gx": {
                    "run": {
                        "code_settings": 1,
                        "extraOptions": {},
                        "run_type": "normal",   # 'normal': submit and wait; 'submit': submit and do not wait; 'prep': do not submit
                    },
                    "read": {
                        "tmin": 0.0
                    },
                },
            },
            
            # Corrections to be applied to each iteration input.gacode file
            
            "profiles_postprocessing_fun": None, # Function to post-process input.gacode only BEFORE passing to transport codes
            
            "applyCorrections": {
                "Ti_thermals": True,        # Keep all thermal ion temperatures equal to the main Ti
                "ni_thermals": True,        # Adjust for quasineutrality by modifying the thermal ion densities together with ne
                "recalculate_ptot": True,   # Recompute PTOT to insert in input file each time
                "Tfast_ratio": False,       # Keep the ratio of Tfast/Te constant throughout the Te evolution
                "ensureMachNumber": None,   # Change w0 to match this Mach number when Ti varies
            },
        }
            
        # -------------------------
        # Target model settings 
        # -------------------------
        
        from mitim_modules.powertorch.physics_models.targets_analytic import analytical_model as target_evaluator
        
        self.portals_parameters["target"] = {
            "evaluator": target_evaluator,
            "options": {
                "targets_evolve": ["qie", "qrad", "qfus"],
                "target_evaluator_method": "powerstate",  # Method to calculate targets (tgyro or powerstate)
                "force_zero_particle_flux": False,  # If True, ignore particle flux profile and assume zero for all radii
                "targets_resolution": 20,  # If not None, calculate targets with this radial resolution (defaults target_evaluator_method to powerstate)
                "percent_error": 1 # (%) Error (std, in percent) of model evaluation 
            }
        }

        # Grab all the flags here in a way that, after changing the dictionary extenrally, I make sure it's the same flags as PORTALS expects
        
        def _grab_flags_dictionary(d):
            keys = {}
            for key in d.keys():
                keys[key] = _grab_flags_dictionary(d[key]) if isinstance(d[key], dict) else None
            return keys

        self.potential_flags = _grab_flags_dictionary(self.portals_parameters)

    def prep(
        self,
        fileGACODE,
        cold_start=False,
        ymax_rel=1.0,
        ymin_rel=1.0,
        limitsAreRelative=True,
        dvs_fixed=None,
        hardGradientLimits=None,
        enforceFiniteTemperatureGradients=None,
        define_ranges_from_profiles=None,
        start_from_folder=None,
        reevaluateTargets=0,
        seedInitial=None,
        askQuestions=True,
        transport_evaluator_options=None,
    ):
        """
        Notes:
            - ymax_rel (and ymin_rel) can be float (common for all radii, channels) or the dictionary directly, e.g.:
                    ymax_rel = {
                        'te': [1.0, 0.5, 0.5, 0.5],
                        'ti': [0.5, 0.5, 0.5, 0.5],
                        'ne': [1.0, 0.5, 0.5, 0.5]
                    }
            - enforceFiniteTemperatureGradients is used to be able to select ymin_rel = 2.0 for ne but ensure that te, ti is at, e.g., enforceFiniteTemperatureGradients = 0.95
            - start_from_folder is a folder from which to grab optimization_data and optimization_extra
                (if used with reevaluateTargets>0, change targets by reevaluating with different parameters)
            - seedInitial can be optionally give a seed to randomize the starting profile (useful for developing, paper writing)
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make sure that options that are required by good behavior of PORTALS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        key_rhos = self.check_flags()

        # TO BE REMOVED IN FUTURE
        if not isinstance(cold_start, bool):
            raise Exception("cold_start must be a boolean")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if IOtools.isfloat(ymax_rel):
            ymax_rel0 = copy.deepcopy(ymax_rel)

            ymax_rel = {}
            for prof in self.portals_parameters["solution"]["predicted_channels"]:
                ymax_rel[prof] = np.array( [ymax_rel0] * len(self.portals_parameters["solution"][key_rhos]) )
        
        if IOtools.isfloat(ymin_rel):
            ymin_rel0 = copy.deepcopy(ymin_rel)

            ymin_rel = {}
            for prof in self.portals_parameters["solution"]["predicted_channels"]:
                ymin_rel[prof] = np.array( [ymin_rel0] * len(self.portals_parameters["solution"][key_rhos]) )

        if enforceFiniteTemperatureGradients is not None:
            for prof in ['te', 'ti']:
                if prof in ymin_rel:
                    ymin_rel[prof] = np.array(ymin_rel[prof]).clip(min=None,max=enforceFiniteTemperatureGradients)

        # Initialize
        print(">> PORTALS initalization module (START)", typeMsg="i")
        PORTALSinit.initializeProblem(
            self,
            self.folder,
            fileGACODE,
            ymax_rel,
            ymin_rel,
            start_from_folder=start_from_folder,
            define_ranges_from_profiles=define_ranges_from_profiles,
            dvs_fixed=dvs_fixed,
            limitsAreRelative=limitsAreRelative,
            cold_start=cold_start,
            hardGradientLimits=hardGradientLimits,
            tensor_options = self.tensor_options,
            seedInitial=seedInitial,
            checkForSpecies=askQuestions,
        )
        print(">> PORTALS initalization module (END)", typeMsg="i")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Option to cold_start
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if start_from_folder is not None:
            self.reuseTrainingTabular(
                start_from_folder, self.folder, reevaluateTargets=reevaluateTargets
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ignore targets in surrogate_data.csv
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if 'extrapointsModels' not in self.optimization_options['surrogate_options'] or \
            self.optimization_options['surrogate_options']['extrapointsModels'] is None or \
            len(self.optimization_options['surrogate_options']['extrapointsModels'])==0:

            self._define_reuse_models()

        else:
            print("\t- extrapointsModels already defined, not changing")

    def _define_reuse_models(self):
        '''
        The user can define a list of strings to avoid reusing surrogates.
        e.g. 
            '_tar' to avoid reusing targets
            '_5' to avoid reusing position 5
        '''

        self.optimization_options['surrogate_options']['extrapointsModels'] = []

        # Define avoiders
        if self.optimization_options['surrogate_options']['extrapointsModelsAvoidContent'] is None:
            self.optimization_options['surrogate_options']['extrapointsModelsAvoidContent'] = ['_tar']

        # Define extrapointsModels
        for key in self.surrogate_parameters['surrogate_transformation_variables_lasttime'].keys():
            add_key = True
            for avoid in self.optimization_options['surrogate_options']['extrapointsModelsAvoidContent']:
                if avoid in key:
                    add_key = False
                    break
            if add_key:
                self.optimization_options['surrogate_options']['extrapointsModels'].append(key)

    def run(self, paramsfile, resultsfile):
        # Read what PORTALS sends
        FolderEvaluation, numPORTALS, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        a, b = IOtools.reducePathLevel(self.folder, level=1)
        name = f"portals_{b}_ev{numPORTALS}"  # e.g. portals_jet37_ev0

        # Run
        powerstate, dictOFs = runModelEvaluator(
            self,
            FolderEvaluation,
            dictDVs,
            name,
            numPORTALS=numPORTALS,
            dictOFs=dictOFs,
            remove_folder_upon_completion=not self.portals_parameters["solution"]["keep_full_model_folder"],
        )

        # Write results
        self.write(dictOFs, resultsfile)

        """
		************************************************************************************************************************************
		Extra operations
		************************************************************************************************************************************
		"""

        # Extra operations: Store data that will be useful to store and interpret in a machine were this was not run

        if self.optimization_extra is not None:
            dictStore = IOtools.unpickle_mitim(self.optimization_extra)                           #TODO: This will fail in future versions of torch
            dictStore[int(numPORTALS)] = {"powerstate": powerstate}
            dictStore["profiles_modified"] = PROFILEStools.gacode_state(
                self.folder / "Initialization" / "input.gacode_modified"
            )
            dictStore["profiles_original"] = PROFILEStools.gacode_state(
                self.folder / "Initialization" / "input.gacode_original"
            )
            with open(self.optimization_extra, "wb") as handle:
                pickle_dill.dump(dictStore, handle, protocol=4)

    def scalarized_objective(self, Y):
        """
        Notes
        -----
                - Y is the multi-output evaluation of the model in the shape of (dim1...N, num_ofs), i.e. this function should not care
                  about number of dimensions
        """

        ofs_ordered_names = np.array(self.optimization_options["problem_options"]["ofs"])

        """
		-------------------------------------------------------------------------
		Prepare transport dictionary
		-------------------------------------------------------------------------
			Note: var_dict['Qe_tr_turb'] must have shape (dim1...N, num_radii)
		"""

        var_dict = {}
        for of in ofs_ordered_names:

            var = '_'.join(of.split("_")[:-1])
            if var not in var_dict:
                var_dict[var] = torch.Tensor().to(Y)
            var_dict[var] = torch.cat((var_dict[var], Y[..., ofs_ordered_names == of]), dim=-1)

        """
		-------------------------------------------------------------------------
		Calculate quantities
		-------------------------------------------------------------------------
			Note: of and cal must have shape (dim1...N, num_radii*num_channels)
				  res must have shape (dim1...N)
		"""

        of, cal, _, res = PORTALStools.calculate_residuals(self.powerstate, self.portals_parameters,specific_vars=var_dict)

        return of, cal, res

    def analyze_results(self, plotYN=True, fn=None, cold_start=False, analysis_level=2):
        """
        analysis_level = 2: Standard as other classes. Run best case, plot transport model analysis
        analysis_level = 3: Read from Execution and also calculate metrics (4: full metrics)
        """
        return analyze_results(
            self, plotYN=plotYN, fn=fn, cold_start=cold_start, analysis_level=analysis_level
        )

    def check_flags(self):

        print(">> PORTALS flags pre-check")

        # Check that I haven't added a deprecated variable that I expect some behavior from
        
        def _check_flags_dictionary(d, d_check, avoid = ["run", "read"]):
            for key in d.keys():
                if key not in d_check:
                    print(f"\t- {key} is an unexpected variable, prone to errors or misinterpretation",typeMsg="q")
                elif not isinstance(d[key], dict):
                    continue
                elif key in avoid:
                    continue
                else:
                    _check_flags_dictionary(d[key], d_check[key])

        _check_flags_dictionary(self.portals_parameters, self.potential_flags)

        key_rhos = "predicted_roa" if self.portals_parameters["solution"]["predicted_roa"] is not None else "predicted_rho"

        return key_rhos

    def reuseTrainingTabular(
        self, folderRead, folderNew, reevaluateTargets=0, cold_startIfExists=False):
        """
        reevaluateTargets:
                0: No
                1: Quick targets from powerstate with no transport calculation
                2: Full original model (either with transport model targets or powerstate targets, but also calculate transport)
        """

        (folderNew / "Outputs").mkdir(parents=True, exist_ok=True)

        shutil.copy2(folderRead / "Outputs" / "optimization_data.csv", folderNew / "Outputs")
        shutil.copy2(folderRead / "Outputs" / "optimization_extra.pkl", folderNew / "Outputs")

        optimization_data = BOgraphics.optimization_data(
            self.optimization_options["problem_options"]["dvs"],
            self.optimization_options["problem_options"]["ofs"],
            file=folderNew / "Outputs" / "optimization_data.csv",
        )

        self.optimization_options["initialization_options"]["initial_training"] = len(optimization_data.data)
        self.optimization_options["initialization_options"]["read_initial_training_from_csv"] = True
        self.optimization_options["initialization_options"]["initialization_fun"] = None

        print(
            f'- Reusing the training set ({self.optimization_options["initialization_options"]["initial_training"]} points) from optimization_data in {folderRead}',
            typeMsg="i",
        )

        if reevaluateTargets > 0:
            print("- Re-evaluate targets", typeMsg="i")

            (folderNew / "TargetsRecalculate").mkdir(parents=True, exist_ok=True)

            for numPORTALS in range(len(optimization_data.data)):

                FolderEvaluation = folderNew / "TargetsRecalculate" / f"Evaluation.{numPORTALS}"

                FolderEvaluation.mkdir(parents=True, exist_ok=True)

                # ------------------------------------------------------------------------------------
                # Produce design variables
                # ------------------------------------------------------------------------------------
                dictDVs = OrderedDict()
                for i in self.optimization_options["problem_options"]["dvs"]:
                    dictDVs[i] = {"value": np.nan}
                dictOFs = OrderedDict()
                for i in self.optimization_options["problem_options"]["ofs"]:
                    dictOFs[i] = {"value": np.nan, "error": np.nan}

                for i in dictDVs:
                    dictDVs[i]["value"] = optimization_data.data[i].to_numpy()[numPORTALS]

                # ------------------------------------------------------------------------------------
                # Run to evaluate new targets
                # ------------------------------------------------------------------------------------

                a, b = IOtools.reducePathLevel(self.folder, level=1)
                name = f"portals_{b}_targets_ev{numPORTALS}"

                self_copy = copy.deepcopy(self)
                if reevaluateTargets == 1:
                    self_copy.powerstate.transport_options["evaluator"] = None
                    self_copy.powerstate.target_options["options"]["targets_evolve"] = "target_evaluator_method"

                _, dictOFs = runModelEvaluator(
                    self_copy,
                    FolderEvaluation,
                    dictDVs,
                    name,
                    cold_start=cold_startIfExists,
                    dictOFs=dictOFs,
                )

                # ------------------------------------------------------------------------------------
                # From the optimization_data, change ONLY the targets, since I still want CGYRO!
                # ------------------------------------------------------------------------------------

                for i in dictOFs:
                    if "_tar" in i:
                        print(f"Changing {i} in file")

                        optimization_data.data[i].iloc[numPORTALS] = dictOFs[i]["value"].cpu().numpy().item()
                        optimization_data.data[i + "_std"].iloc[numPORTALS] = dictOFs[i]["error"].cpu().numpy().item()

            # ------------------------------------------------------------------------------------
            # Update new Tabulars
            # ------------------------------------------------------------------------------------

            optimization_data.data.to_csv(optimization_data.file, index=False)

def runModelEvaluator(
    self,
    FolderEvaluation,
    dictDVs,
    name,
    cold_start=False,
    numPORTALS=0,
    dictOFs=None,
    remove_folder_upon_completion=False,
    ):

    # Copy powerstate (that was initialized) but will be different per call to the evaluator
    powerstate = copy.deepcopy(self.powerstate)

    # ---------------------------------------------------------------------------------------------------
    # Prep run
    # ---------------------------------------------------------------------------------------------------

    folder_model = FolderEvaluation / "transport_simulation_folder"
    folder_model.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------------------------
    # Prepare evaluating vector X
    # ---------------------------------------------------------------------------------------------------

    X = torch.zeros(len(powerstate.predicted_channels) * (powerstate.plasma["rho"].shape[1] - 1)).to(powerstate.dfT)
    cont = 0
    for ikey in powerstate.predicted_channels:
        for ix in range(powerstate.plasma["rho"].shape[1] - 1):
            X[cont] = dictDVs[f"aL{ikey}_{ix+1}"]["value"]
            cont += 1
    X = X.unsqueeze(0)

    # Ensure that the powerstate has the right dimensions
    powerstate._repeat_tensors(batch_size=X.shape[0])

    # ---------------------------------------------------------------------------------------------------
    # Run model through powerstate
    # ---------------------------------------------------------------------------------------------------

    # In certain cases, I want to cold_start the model directly from the PORTALS call instead of powerstate
    powerstate.transport_options["cold_start"] = cold_start

    # Evaluate X (DVs) through powerstate.calculate(). This will populate .plasma with the results
    powerstate.calculate(X, nameRun=name, folder=folder_model, evaluation_number=numPORTALS)

    # ---------------------------------------------------------------------------------------------------
    # Produce dictOFs
    # ---------------------------------------------------------------------------------------------------

    if dictOFs is not None:
        dictOFs = map_powerstate_to_portals(powerstate, dictOFs)

    # ---------------------------------------------------------------------------------------------------
    # Remove folder
    # ---------------------------------------------------------------------------------------------------
    if remove_folder_upon_completion:
        print(f"\t- To avoid exceedingly large PORTALS runs, removing ...{IOtools.clipstr(folder_model)}")
        IOtools.shutil_rmtree(folder_model)

    return powerstate, dictOFs

def map_powerstate_to_portals(powerstate, dictOFs):

    for var in powerstate.predicted_channels:
        # Write in OFs
        for i in range(powerstate.plasma["rho"].shape[1] - 1): # Ignore position 0, which is rho=0
            if var == "te":
                var0, var1 = "Qe", "QeMWm2"
            elif var == "ti":
                var0, var1 = "Qi", "QiMWm2"
            elif var == "ne":
                var0, var1 = "Ge", "Ce"
            elif var == "nZ":
                var0, var1 = "GZ", "CZ"
            elif var == "w0":
                var0, var1 = "Mt", "MtJm2"

            """
            TRANSPORT calculation
            ---------------------
            """

            dictOFs[f"{var0}_tr_turb_{i+1}"]["value"] = powerstate.plasma[f"{var1}_tr_turb"][0, i+1]
            dictOFs[f"{var0}_tr_turb_{i+1}"]["error"] = powerstate.plasma[f"{var1}_tr_turb_stds"][0, i+1]

            dictOFs[f"{var0}_tr_neoc_{i+1}"]["value"] = powerstate.plasma[f"{var1}_tr_neoc"][0, i+1]
            dictOFs[f"{var0}_tr_neoc_{i+1}"]["error"] = powerstate.plasma[f"{var1}_tr_neoc_stds"][0, i+1]

            """
            TARGET calculation
            ---------------------
                If that radius & profile position has target, evaluate
            """

            dictOFs[f"{var0}_tar_{i+1}"]["value"] = powerstate.plasma[f"{var1}"][0, i+1]
            dictOFs[f"{var0}_tar_{i+1}"]["error"] = powerstate.plasma[f"{var1}_stds"][0, i+1]

    """
    Turbulent Exchange
    ------------------
    """
    if 'Qie_tr_turb_1' in dictOFs:
        for i in range(powerstate.plasma["rho"].shape[1] - 1):
            dictOFs[f"Qie_tr_turb_{i+1}"]["value"] = powerstate.plasma["QieMWm3_tr_turb"][0, i+1]
            dictOFs[f"Qie_tr_turb_{i+1}"]["error"] = powerstate.plasma["QieMWm3_tr_turb_stds"][0, i+1]

    return dictOFs

def analyze_results(
    self, plotYN=True, fn=None, cold_start=False, analysis_level=2, onlyBest=False, tabs_colors=0,
    ):
    if plotYN:
        print("\n *****************************************************")
        print("* MITIM plotting module - PORTALS")
        print("*****************************************************\n")

    # ----------------------------------------------------------------------------------------------------------------
    # Interpret stuff
    # ----------------------------------------------------------------------------------------------------------------

    portals_full = PORTALSanalysis.PORTALSanalyzer(self)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot full information from analysis class
    # ----------------------------------------------------------------------------------------------------------------

    if plotYN:
        portals_full.fn = fn
        portals_full.plotPORTALS(tabs_colors_common=tabs_colors)

    # ----------------------------------------------------------------------------------------------------------------
    # Running cases: Original and Best
    # ----------------------------------------------------------------------------------------------------------------

    if analysis_level in [2, 5]:
        portals_full.runCases(onlyBest=onlyBest, cold_start=cold_start, fn=fn)

    return portals_full.opt_fun.mitim_model.optimization_object
