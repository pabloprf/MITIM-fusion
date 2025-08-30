import shutil
import torch
import copy
from collections import OrderedDict
import numpy as np
import dill as pickle_dill
from collections import OrderedDict
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.portals import PORTALStools
from mitim_modules.portals.utils import (
    PORTALSinit,
    PORTALSanalysis,
)
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed


class portals(STRATEGYtools.opt_evaluator):
    def __init__(
        self, 
        folder,                             # Folder where the PORTALS workflow will be run
        portals_namelist = None,
        tensor_options = {
            "dtype": torch.double,
            "device": torch.device("cpu"),
        },
        ):

        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t PORTALS class module")
        print("-----------------------------------------------------------------------------------------\n")

        super().__init__(
            folder,
            tensor_options=tensor_options
            )

        # Read PORTALS namelist (if not provided, use default)
        if portals_namelist is None:
            self.portals_namelist = __mitimroot__ / "templates" / "namelist.portals.yaml"
            print(f"\t- No PORTALS namelist provided, using default in {IOtools.clipstr(self.portals_namelist)}")
        else:
            self.portals_namelist = portals_namelist
            print(f"\t- Using provided PORTALS namelist in {IOtools.clipstr(self.portals_namelist)}")
        self.portals_parameters = IOtools.read_mitim_yaml(self.portals_namelist)

        # Read optimization namelist (always the default, the values to be modified are in the portals one)
        if self.portals_parameters["optimization_namelist"] is not None:
            self.optimization_namelist = self.portals_parameters["optimization_namelist"]
        else:
            self.optimization_namelist = __mitimroot__ / "templates" / "namelist.optimization.yaml"
        self.optimization_options = IOtools.read_mitim_yaml(self.optimization_namelist)

        # Apply the optimization options to the proper namelist and drop it from portals_parameters
        if 'optimization_options' in self.portals_parameters:
            self.optimization_options = IOtools.deep_dict_update(self.optimization_options, self.portals_parameters['optimization_options'])
            del self.portals_parameters['optimization_options']

        # Grab all the flags here in a way that, after changing the dictionary extenrally, I make sure it's the same flags as PORTALS expects
        self.potential_flags = IOtools.deep_grab_flags_dict(self.portals_parameters)

    def prep(
        self,
        mitim_state,
        cold_start=False,
        seedInitial=None,
        askQuestions=True,
    ):
        """
        Notes:
            - ymax_rel (and ymin_rel) can be float (common for all radii, channels) or the dictionary directly, e.g.:
                    ymax_rel = {
                        'te': [1.0, 0.5, 0.5, 0.5],
                        'ti': [0.5, 0.5, 0.5, 0.5],
                        'ne': [1.0, 0.5, 0.5, 0.5]
                    }
            - enforce_finite_aLT is used to be able to select ymin_rel = 2.0 for ne but ensure that te, ti is at, e.g., enforce_finite_aLT = 0.95
            - start_from_folder is a folder from which to grab optimization_data and optimization_extra
                (if used with reevaluate_targets>0, change targets by reevaluating with different parameters)
            - seedInitial can be optionally give a seed to randomize the starting profile (useful for developing, paper writing)
        """

        ymax_rel = self.portals_parameters["solution"]["exploration_ranges"]["ymax_rel"]
        ymin_rel = self.portals_parameters["solution"]["exploration_ranges"]["ymin_rel"]
        limits_are_relative = self.portals_parameters["solution"]["exploration_ranges"]["limits_are_relative"]
        fixed_gradients = self.portals_parameters["solution"]["exploration_ranges"]["fixed_gradients"]
        yminymax_atleast = self.portals_parameters["solution"]["exploration_ranges"]["yminymax_atleast"]
        enforce_finite_aLT = self.portals_parameters["solution"]["exploration_ranges"]["enforce_finite_aLT"]
        define_ranges_from_profiles = self.portals_parameters["solution"]["exploration_ranges"]["define_ranges_from_profiles"]
        start_from_folder = self.portals_parameters["solution"]["exploration_ranges"]["start_from_folder"]
        reevaluate_targets = self.portals_parameters["solution"]["exploration_ranges"]["reevaluate_targets"]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make sure that options that are required by good behavior of PORTALS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(">> PORTALS flags pre-check")

        # Check that I haven't added a deprecated variable that I expect some behavior from
        IOtools.check_flags_mitim_namelist(self.portals_parameters, self.potential_flags, avoid = ["run", "read"], askQuestions=askQuestions)

        key_rhos = "predicted_roa" if self.portals_parameters["solution"]["predicted_roa"] is not None else "predicted_rho"

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

        if enforce_finite_aLT is not None:
            for prof in ['te', 'ti']:
                if prof in ymin_rel:
                    ymin_rel[prof] = np.array(ymin_rel[prof]).clip(min=None,max=enforce_finite_aLT)

        # Initialize
        print(">> PORTALS initalization module (START)", typeMsg="i")
        PORTALSinit.initializeProblem(
            self,
            self.folder,
            mitim_state,
            ymax_rel,
            ymin_rel,
            start_from_folder=start_from_folder,
            define_ranges_from_profiles=define_ranges_from_profiles,
            fixed_gradients=fixed_gradients,
            limits_are_relative=limits_are_relative,
            cold_start=cold_start,
            yminymax_atleast=yminymax_atleast,
            tensor_options = self.tensor_options,
            seedInitial=seedInitial,
            checkForSpecies=askQuestions,
        )
        print(">> PORTALS initalization module (END)", typeMsg="i")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Option to cold_start
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if start_from_folder is not None:
            self.reuseTrainingTabular(start_from_folder, self.folder, reevaluate_targets=reevaluate_targets)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ignore targets in surrogate_data.csv
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if 'extrapointsModels' not in self.optimization_options['surrogate_options'] or \
            self.optimization_options['surrogate_options']['extrapointsModels'] is None or \
            len(self.optimization_options['surrogate_options']['extrapointsModels'])==0:

            self._define_reuse_models()

        else:
            print("\t- extrapointsModels already defined, not changing")

        # Make a copy of the namelist that was imported to the folder
        shutil.copy(self.portals_namelist, self.folder / "portals.namelist_original.yaml")

        # Write the parameters (after script modification) to a yaml namelist for tracking purposes
        IOtools.write_mitim_yaml(self.portals_parameters, self.folder / "namelist.portals.yaml")

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

    def reuseTrainingTabular(
        self, folderRead, folderNew, reevaluate_targets=0, cold_startIfExists=False):
        """
        reevaluate_targets:
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

        if reevaluate_targets > 0:
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
                if reevaluate_targets == 1:
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
