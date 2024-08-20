import os
import torch
import copy
import numpy as np
import dill as pickle_dill
from functools import partial
from collections import OrderedDict
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import PORTALSinteraction
from mitim_modules.portals import PORTALStools
from mitim_modules.portals.utils import (
    PORTALSinit,
    PORTALSoptimization,
    PORTALSanalysis,
)
from mitim_modules.powertorch.physics import TRANSPORTtools, TARGETStools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
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
		5:  2 + 4                  (only works if optimization_extra is provided or Execution exists)

		>2 will also plot profiles & gradients comparison (original, initial, best)
"""

def default_namelist(optimization_options, CGYROrun=False):
    """
    This is to be used after reading the namelist, so self.optimization_options should be completed with main defaults.
    """

    # Initialization
    optimization_options["initial_training"] = 5
    optimization_options["initialization_fun"] = PORTALSoptimization.initialization_simple_relax

    # Strategy
    optimization_options["BO_iterations"] = 50
    optimization_options["parallel_evaluations"] = 1
    optimization_options["minimum_dvs_variation"] = [
        10,
        3,
        1e-1,
    ]  # After iteration 10, Check if 3 consecutive DVs are varying less than 0.1% from the rest I have! (stiff behavior?)
    optimization_options["maximum_value_is_rel"]  = True
    optimization_options["maximum_value"]       = 5e-3  # Reducing residual by 200x is enough

    if CGYROrun:
        # Do not allow excursions for CGYRO, at least by default
        optimization_options["StrategyOptions"]["AllowedExcursions"] = [0.0, 0.0]
        optimization_options["optimizers"] = "root_5-botorch-ga"  # Added root which is not a default bc it needs dimX=dimY
    else:
        # Allow excursions for TGLF
        optimization_options["StrategyOptions"]["AllowedExcursions"] = [
            0.05,
            0.05,
        ]  # This would be 10% if [-100,100]
        optimization_options["optimizers"] = "botorch"  # TGLF runs should prioritize speed, and botorch is robust enough

    # Surrogate
    optimization_options["surrogateOptions"]["selectSurrogate"] = partial(
        PORTALStools.selectSurrogate, CGYROrun=CGYROrun
    )
    # optimization_options['surrogateOptions']['MinimumRelativeNoise']   = 1E-3  # Minimum error bar (std) of 0.1% of maximum value of each output (untransformed! so careful with far away initial condition)

    optimization_options["surrogateOptions"]["ensure_within_bounds"] = True

    # Acquisition
    optimization_options["acquisition_type"] = "posterior_mean"

    return optimization_options


class portals(STRATEGYtools.opt_evaluator):
    def __init__(
        self, 
        folder,                             # Folder where the PORTALS workflow will be run
        namelist=None,                      # If None, default namelist will be used. If not None, it will be read and used
        TensorsType=torch.double,           # Type of tensors to be used (torch.float, torch.double)
        CGYROrun=False,                     # If True, use CGYRO defaults for best optimization practices
        portals_transformation_variables = None,          # If None, use defaults for both main and trace
        portals_transformation_variables_trace = None,
        additional_params_in_surrogate = [] # Additional parameters to be used in the surrogate (e.g. ['q'])
        ):
        '''
        Note that additional_params_in_surrogate They must exist in the plasma dictionary of the powerstate object
        '''
        
        print(
            "\n-----------------------------------------------------------------------------------------"
        )
        print("\t\t\t PORTALS class module")
        print(
            "-----------------------------------------------------------------------------------------\n"
        )

        # Store folder, namelist. Read namelist

        super().__init__(
            folder,
            namelist=namelist,
            TensorsType=TensorsType,
            default_namelist_function=(
                partial(default_namelist, CGYROrun=CGYROrun)
                if (namelist is None)
                else None
            ),
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Default (please change to your desire after instancing the object)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.potential_flags = {'INITparameters': [], 'MODELparameters': [], 'PORTALSparameters': []}

        """
		Parameters to initialize files
		------------------------------
			These parameters are used to initialize the input.gacode to work with, before any PORTALS workflow
			( passed directly to profiles.correct() )
            Bear in mind that this is not necessary, you provide an already ready-to-go input.gacode without the need
            to run these corrections.
		"""

        self.INITparameters = {
            "recompute_ptot": True,  # Recompute PTOT to match kinetic profiles (after removals)
            "quasineutrality": False,  # Make sure things are quasineutral by changing the *MAIN* ion (D,T or both)  (after removals)
            "removeIons": [],  # Remove this ion from the input.gacode (if D,T,Z, eliminate T with [2])
            "removeFast": False,  # Automatically detect which are fast ions and remove them
            "FastIsThermal": False,  # Do not remove fast, keep their diluiton effect but make them thermal
            "sameDensityGradients": False,  # Make all ion density gradients equal to electrons
            "groupQIONE": False,
            "ensurePostiveGamma": False,
            "ensureMachNumber": None,
        }

        for key in self.INITparameters.keys():
            self.potential_flags['INITparameters'].append(key)

        """
		Parameters to run the model
		---------------------------
			The corrections are applied prior to each evaluation, so that things are consistent.
			Here, do not include things that are not specific for a given iteration. Otherwise if they are general
			changes to input.gacode, then that should go into INITparameters.

			if MODELparameters contains RoaLocations, use that instead of RhoLocations
		"""

        self.MODELparameters = {
            "RhoLocations": [0.3, 0.45, 0.6, 0.75, 0.9],
            "RoaLocations": None,
            "ProfilesPredicted": ["te", "ti", "ne"],  # ['nZ','w0']
            "Physics_options": {
                "TypeTarget": 3,
                "TurbulentExchange": 0,  # In PORTALS TGYRO evaluations, let's always calculate turbulent exchange, but NOT include it in targets!
                "PtotType": 1,  # In PORTALS TGYRO evaluations, let's always use the PTOT column (so control of that comes from the ap)
                "GradientsType": 0,  # In PORTALS TGYRO evaluations, we need to not recompute gradients
                "InputType": 1,  # In PORTALS TGYRO evaluations, we need to use exact profiles
            },
            "applyCorrections": {
                "Ti_thermals": True,  # Keep all thermal ion temperatures equal to the main Ti
                "ni_thermals": True,  # Adjust for quasineutrality by modifying the thermal ion densities together with ne
                "recompute_ptot": True,  # Recompute PTOT to insert in input file each time
                "Tfast_ratio": False,  # Keep the ratio of Tfast/Te constant throughout the Te evolution
                "ensureMachNumber": None,  # Change w0 to match this Mach number when Ti varies
            },
            "transport_model": {"turbulence":'TGLF',"TGLFsettings": 6, "extraOptionsTGLF": {}}
        }

        for key in self.MODELparameters.keys():
            self.potential_flags['MODELparameters'].append(key)

        """
		Physics-informed parameters to fit surrogates
		---------------------------------------------
		"""

        

        (
            portals_transformation_variables,
            portals_transformation_variables_trace,
        ) = PORTALStools.default_portals_transformation_variables(additional_params = additional_params_in_surrogate)

        """
		Parameters to run PORTALS
		-----------------------
		"""

        # Selection of model
        transport_evaluator = TRANSPORTtools.tgyro_model
        targets_evaluator = TARGETStools.analytical_model

        self.PORTALSparameters = {
            "percentError": [
                10,
                10,
                1,
            ],  # (%) Error (std, in percent) of model evaluation [TGLF, NEO, TARGET]
            "transport_evaluator": transport_evaluator,
            "targets_evaluator": targets_evaluator,
            "TargetCalc": "powerstate",  # Method to calculate targets (tgyro or powerstate)
            "launchEvaluationsAsSlurmJobs": True,  # Launch each evaluation as a batch job (vs just comand line)
            "useConvectiveFluxes": True,  # If True, then convective flux for final metric (not fitting). If False, particle flux
            "includeFastInQi": False,  # If True, and fast ions have been included, in seprateNEO, sum fast
            "useDiffusivities": False,  # If True, use [chi_e,chi_i,D] instead of [Qe,Qi,Gamma]
            "useFluxRatios": False,  # If True, fit to [Qi,Qe/Qi,Ge/Qi]
            "portals_transformation_variables": portals_transformation_variables,  # Physics-informed parameters to fit surrogates
            "portals_transformation_variables_trace": portals_transformation_variables_trace,  # Physics-informed parameters to fit surrogates for trace impurities
            "Qi_criterion_stable": 0.01,  # For CGYRO runs, MW/m^2 of Qi below which the case is considered stable
            "percentError_stable": 5.0,  # (%) For CGYRO runs, minimum error based on target if case is considered stable
            "forceZeroParticleFlux": False,  # If True, ignore particle flux profile and assume zero for all radii
            "surrogateForTurbExch": False,  # Run turbulent exchange as surrogate?
            "profiles_postprocessing_fun": None,  # Function to post-process input.gacode only BEFORE passing to transport codes (only CGYRO so far)
            "Pseudo_multipliers": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],  # [Qe,Qi,Ge] multipliers to calculate pseudo
            "ImpurityOfInterest": 1,  # Position in ions vector of the impurity to do flux matching
            "applyImpurityGammaTrick": True,  # If True, fit model to GZ/nZ, valid on the trace limit
            "UseOriginalImpurityConcentrationAsWeight": True,  # If True, using original nZ/ne as scaling factor for GZ
            "fineTargetsResolution": 20,  # If not None, calculate targets with this radial resolution (defaults TargetCalc to powerstate)
            "hardCodedCGYRO": None,  # If not None, use this hard-coded CGYRO evaluation
            "additional_params_in_surrogate": additional_params_in_surrogate,
        }

        for key in self.PORTALSparameters.keys():
            self.potential_flags['PORTALSparameters'].append(key)

    def prep(
        self,
        fileGACODE,
        folderWork,
        restartYN=False,
        ymax_rel=1.0,
        ymin_rel=1.0,
        dvs_fixed=None,
        limitsAreRelative=True,
        hardGradientLimits=None,
        define_ranges_from_profiles=None,
        start_from_folder=None,
        reevaluateTargets=0,
        seedInitial=None,
        askQuestions=True,
        ModelOptions=None,
    ):
        """
        start_from_folder is a folder from which to grab optimization_data and optimization_extra
                (if used with reevaluateTargets>0, change targets by reevaluating with different parameters)

        ymax_rel (and ymin_rel) can be float (common for all radii, channels) or the array directly, e.g.:
                ymax_rel = np.array([   [1.0, 0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5, 0.5],
                                        [1.0, 0.5, 0.5, 0.5]    ])

        seedInitial can be optionally give a seed to randomize the starting profile (useful for developing, paper writing)
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make sure that options that are required by good behavior of PORTALS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        key_rhos = self.check_flags()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if IOtools.isfloat(ymax_rel):
            ymax_rel = np.array(
                [ymax_rel * np.ones(len(self.MODELparameters[key_rhos]))]
                * len(self.MODELparameters["ProfilesPredicted"])
            )
        if IOtools.isfloat(ymin_rel):
            ymin_rel = np.array(
                [ymin_rel * np.ones(len(self.MODELparameters[key_rhos]))]
                * len(self.MODELparameters["ProfilesPredicted"])
            )

        # Initialize
        print(">> PORTALS initalization module (START)", typeMsg="i")
        PORTALSinit.initializeProblem(
            self,
            self.folder,
            fileGACODE,
            self.INITparameters,
            ymax_rel,
            ymin_rel,
            start_from_folder=start_from_folder,
            define_ranges_from_profiles=define_ranges_from_profiles,
            dvs_fixed=dvs_fixed,
            limitsAreRelative=limitsAreRelative,
            restartYN=restartYN,
            hardGradientLimits=hardGradientLimits,
            dfT=self.dfT,
            seedInitial=seedInitial,
            checkForSpecies=askQuestions,
            ModelOptions=ModelOptions,
        )
        print(">> PORTALS initalization module (END)", typeMsg="i")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Option to restart
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if start_from_folder is not None:
            self.reuseTrainingTabular(
                start_from_folder, folderWork, reevaluateTargets=reevaluateTargets
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ignore targets in surrogate_data.csv
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.optimization_options['surrogateOptions']['extrapointsModels'] = []
        for key in self.surrogate_parameters['surrogate_transformation_variables_lasttime'].keys():
            if 'Tar' not in key:
                self.optimization_options['surrogateOptions']['extrapointsModels'].append(key)

    def run(self, paramsfile, resultsfile):
        # Read what PORTALS sends
        FolderEvaluation, numPORTALS, dictDVs, dictOFs = self.read(
            paramsfile, resultsfile
        )

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
            with open(self.optimization_extra, "rb") as handle:
                dictStore = pickle_dill.load(handle)
            dictStore[int(numPORTALS)] = {"powerstate": powerstate}
            dictStore["profiles_modified"] = PROFILEStools.PROFILES_GACODE(
                f"{self.folder}/Initialization/input.gacode_modified"
            )
            dictStore["profiles_original"] = PROFILEStools.PROFILES_GACODE(
                f"{self.folder}/Initialization/input.gacode_original"
            )
            with open(self.optimization_extra, "wb") as handle:
                pickle_dill.dump(dictStore, handle)

    def scalarized_objective(self, Y):
        """
        Notes
        -----
                - Y is the multi-output evaluation of the model in the shape of (dim1...N, num_ofs), i.e. this function should not care
                  about number of dimensions
        """

        ofs_ordered_names = np.array(self.optimization_options["ofs"])

        """
		-------------------------------------------------------------------------
		Prepare transport dictionary
		-------------------------------------------------------------------------
			Note: var_dict['QeTurb'] must have shape (dim1...N, num_radii)
		"""

        var_dict = {}
        for of in ofs_ordered_names:
            var, _ = of.split("_")
            if var not in var_dict:
                var_dict[var] = torch.Tensor().to(Y)
            var_dict[var] = torch.cat(
                (var_dict[var], Y[..., ofs_ordered_names == of]), dim=-1
            )

        """
		-------------------------------------------------------------------------
		Calculate quantities
		-------------------------------------------------------------------------
			Note: of and cal must have shape (dim1...N, num_radii*num_channels)
				  res must have shape (dim1...N)
		"""

        of, cal, _, res = PORTALSinteraction.calculatePseudos(self.powerstate, self.PORTALSparameters,specific_vars=var_dict)

        return of, cal, res

    def analyze_results(self, plotYN=True, fn=None, restart=False, analysis_level=2):
        """
        analysis_level = 2: Standard as other classes. Run best case, plot transport model analysis
        analysis_level = 3: Read from Execution and also calculate metrics (4: full metrics)
        """
        return analyze_results(
            self, plotYN=plotYN, fn=fn, restart=restart, analysis_level=analysis_level
        )

    def check_flags(self):

        print(">> PORTALS flags pre-check")

        # Check that I haven't added a deprecated variable that I expect some behavior from
        for key in self.potential_flags.keys():
            for flag in self.__dict__[key]:
                if flag not in self.potential_flags[key]:
                    print(
                        f"\t- {key}['{flag}'] is an unexpected variable, prone to errors or misinterpretation",
                        typeMsg="q",
                    )
        # ----------------------------------------------------------------------------------

        if self.PORTALSparameters["fineTargetsResolution"] is not None:
            if self.PORTALSparameters["TargetCalc"] != "powerstate":
                print(
                    "\t- Requested fineTargetsResolution, so running powerstate target calculations",
                    typeMsg="w",
                )
                self.PORTALSparameters["TargetCalc"] = "powerstate"

        if (self.PORTALSparameters["transport_evaluator"] != TRANSPORTtools.tgyro_model) and (self.PORTALSparameters["TargetCalc"] == "tgyro"):
            print(
                "\t- Requested TGYRO targets, but transport evaluator is not tgyro, so changing to powerstate",
                typeMsg="w",
            )
            self.PORTALSparameters["TargetCalc"] = "powerstate"

        if (
            "InputType" not in self.MODELparameters["Physics_options"]
        ) or self.MODELparameters["Physics_options"]["InputType"] != 1:
            print(
                "\t- In PORTALS TGYRO evaluations, we need to use exact profiles (InputType=1)",
                typeMsg="i",
            )
            self.MODELparameters["Physics_options"]["InputType"] = 1

        if (
            "GradientsType" not in self.MODELparameters["Physics_options"]
        ) or self.MODELparameters["Physics_options"]["GradientsType"] != 0:
            print(
                "\t- In PORTALS TGYRO evaluations, we need to not recompute gradients (GradientsType=0)",
                typeMsg="i",
            )
            self.MODELparameters["Physics_options"]["GradientsType"] = 0

        if 'TargetType' in self.MODELparameters["Physics_options"]:
            raise Exception("\t- TargetType is not used in PORTALS anymore, removing")


        key_rhos = (
            "RoaLocations" if self.MODELparameters["RoaLocations"] is not None else "RhoLocations"
        )

        return key_rhos

    def reuseTrainingTabular(
        self, folderRead, folderNew, reevaluateTargets=0, restartIfExists=False):
        """
        reevaluateTargets:
                0: No
                1: Quick targets from powerstate with no transport calculation
                2: Full original model (either with transport model targets or powerstate targets, but also calculate transport)
        """

        if not os.path.exists(folderNew):
            os.system(f"mkdir {folderNew}")
        if not os.path.exists(f"{folderNew}/Outputs"):
            os.system(f"mkdir {folderNew}/Outputs")

        os.system(f"cp {folderRead}/Outputs/optimization_data.csv {folderNew}/Outputs/.")
        os.system(f"cp {folderRead}/Outputs/optimization_extra.pkl {folderNew}/Outputs/.")

        optimization_data = BOgraphics.optimization_data(
            self.optimization_options["dvs"],
            self.optimization_options["ofs"],
            file=f"{folderNew}/Outputs/optimization_data.csv",
        )

        self.optimization_options["initial_training"] = len(optimization_data.data)
        self.optimization_options["read_initial_training_from_csv"] = True
        self.optimization_options["initialization_fun"] = None

        print(
            f'- Reusing the training set ({self.optimization_options["initial_training"]} points) from optimization_data in {folderRead}',
            typeMsg="i",
        )

        if reevaluateTargets > 0:
            print("- Re-evaluate targets", typeMsg="i")

            os.system(f"mkdir {folderNew}/TargetsRecalculate/")

            for numPORTALS in range(len(optimization_data.data)):

                FolderEvaluation = (
                    f"{folderNew}/TargetsRecalculate/Evaluation.{numPORTALS}"
                )
                if not os.path.exists(FolderEvaluation):
                    os.system(f"mkdir {FolderEvaluation}")

                # ------------------------------------------------------------------------------------
                # Produce design variables
                # ------------------------------------------------------------------------------------
                dictDVs = OrderedDict()
                for i in self.optimization_options["dvs"]:
                    dictDVs[i] = {"value": np.nan}
                dictOFs = OrderedDict()
                for i in self.optimization_options["ofs"]:
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
                    self_copy.powerstate.TransportOptions["transport_evaluator"] = None
                    self_copy.powerstate.TargetOptions["ModelOptions"]["TypeTarget"] = "powerstate"
                else:
                    self_copy.powerstate.TransportOptions["transport_evaluator"] = TRANSPORTtools.tgyro_model

                _, dictOFs = runModelEvaluator(
                    self_copy,
                    FolderEvaluation,
                    dictDVs,
                    name,
                    restart=restartIfExists,
                    dictOFs=dictOFs,
                )

                # ------------------------------------------------------------------------------------
                # From the optimization_data, change ONLY the targets, since I still want CGYRO!
                # ------------------------------------------------------------------------------------

                for i in dictOFs:
                    if "Tar" in i:
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
    restart=False,
    numPORTALS=0,
    dictOFs=None,
    ):
    # Copy powerstate (that was initialized) but will be different per call to the evaluator
    powerstate = copy.deepcopy(self.powerstate)

    # ---------------------------------------------------------------------------------------------------
    # Prep run
    # ---------------------------------------------------------------------------------------------------

    folder_model = FolderEvaluation + "/model_complete/"
    os.system(f"mkdir {folder_model}")

    # ---------------------------------------------------------------------------------------------------
    # Prepare evaluating vector X
    # ---------------------------------------------------------------------------------------------------

    X = torch.zeros(
        len(powerstate.ProfilesPredicted) * (powerstate.plasma["rho"].shape[1] - 1)
    ).to(powerstate.dfT)
    cont = 0
    for ikey in powerstate.ProfilesPredicted:
        for ix in range(powerstate.plasma["rho"].shape[1] - 1):
            X[cont] = dictDVs[f"aL{ikey}_{ix+1}"]["value"]
            cont += 1
    X = X.unsqueeze(0)

    # ---------------------------------------------------------------------------------------------------
    # Run model through powerstate
    # ---------------------------------------------------------------------------------------------------

    # In certain cases, I want to restart the model directly from the PORTALS call instead of powerstate
    powerstate.TransportOptions["ModelOptions"]["restart"] = restart

    # Evaluate X (DVs) through powerstate.calculate(). This will populate .plasma with the results
    powerstate.calculate(
        X, nameRun=name, folder=folder_model, evaluation_number=numPORTALS
    )

    # ---------------------------------------------------------------------------------------------------
    # Produce dictOFs
    # ---------------------------------------------------------------------------------------------------

    if dictOFs is not None:
        dictOFs = map_powerstate_to_portals(powerstate, dictOFs)

    return powerstate, dictOFs

def map_powerstate_to_portals(powerstate, dictOFs):
    """
    """

    for var in powerstate.ProfilesPredicted:
        # Write in OFs
        for i in range(powerstate.plasma["rho"].shape[1] - 1): # Ignore position 0, which is rho=0
            if var == "te":
                var0, var1 = "Qe", "Pe"
            elif var == "ti":
                var0, var1 = "Qi", "Pi"
            elif var == "ne":
                var0, var1 = "Ge", "Ce"
            elif var == "nZ":
                var0, var1 = "GZ", "CZ"
            elif var == "w0":
                var0, var1 = "Mt", "Mt"

            """
            TRANSPORT calculation
            ---------------------
            """

            dictOFs[f"{var0}Turb_{i+1}"]["value"] = powerstate.plasma[
                f"{var1}_tr_turb"
            ][0, i+1]
            dictOFs[f"{var0}Turb_{i+1}"]["error"] = powerstate.plasma[
                f"{var1}_tr_turb_stds"
            ][0, i+1]

            dictOFs[f"{var0}Neo_{i+1}"]["value"] = powerstate.plasma[
                f"{var1}_tr_neo"
            ][0, i+1]
            dictOFs[f"{var0}Neo_{i+1}"]["error"] = powerstate.plasma[
                f"{var1}_tr_neo_stds"
            ][0, i+1]

            """
            TARGET calculation
            ---------------------
                If that radius & profile position has target, evaluate
            """

            dictOFs[f"{var0}Tar_{i+1}"]["value"] = powerstate.plasma[f"{var1}"][
                0, i+1
            ]
            dictOFs[f"{var0}Tar_{i+1}"]["error"] = powerstate.plasma[
                f"{var1}_stds"
            ][0, i+1]

    """
    Turbulent Exchange
    ------------------
    """
    if 'PexchTurb_1' in dictOFs:
        for i in range(powerstate.plasma["rho"].shape[1] - 1):
            dictOFs[f"PexchTurb_{i+1}"]["value"] = powerstate.plasma["PexchTurb"][
                0, i+1
            ]
            dictOFs[f"PexchTurb_{i+1}"]["error"] = powerstate.plasma[
                "PexchTurb_stds"
            ][0, i+1]

    return dictOFs

def analyze_results(
    self, plotYN=True, fn=None, restart=False, analysis_level=2, onlyBest=False
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
        portals_full.plotPORTALS()

    # ----------------------------------------------------------------------------------------------------------------
    # Running cases: Original and Best
    # ----------------------------------------------------------------------------------------------------------------

    if analysis_level in [2, 5]:
        portals_full.runCases(onlyBest=onlyBest, restart=restart, fn=fn)

    return portals_full.opt_fun.prfs_model.optimization_object
