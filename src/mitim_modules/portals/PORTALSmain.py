import os, torch, copy
import numpy as np
import dill as pickle_dill
import matplotlib.pyplot as plt
from IPython import embed
from collections import OrderedDict
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.aux import PORTALSinteraction
from mitim_modules.portals import PORTALStools
from mitim_modules.portals.aux import (
    PORTALSinit,
    PORTALSplot,
    PORTALSoptimization,
)
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.aux import BOgraphics
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

"""
Reading analysis for PORTALS has more options than standard:
--------------------------------------------------------------------------------------------------------
	Standard:
	**************************************
	  -1:  Only improvement
		0:  Only ResultsOptimization
		1:  0 + Pickle
		2:  1 + Final redone in this machine
		
	PORTALS-specific:
	**************************************
		3:  1 + PORTALSplot metrics  (only works if MITIMextra is provided or Execution exists)
		4:  3 + PORTALSplot expected (only works if MITIMextra is provided or Execution exists)
		5:  2 + 4                  (only works if MITIMextra is provided or Execution exists)

		>2 will also plot profiles & gradients comparison (original, initial, best)
"""


def default_namelist(Optim):
    """
    This is to be used after reading the namelist, so self.Optim should be completed with main defaults.
    """

    # Initialization
    Optim["initialPoints"] = 5
    Optim["initializationFun"] = PORTALSoptimization.initialization_simple_relax

    # Strategy
    Optim["BOiterations"] = 50
    Optim["parallelCalls"] = 1
    Optim["minimumDVvariation"] = [
        10,
        3,
        1e-1,
    ]  # After iteration 10, Check if 3 consecutive DVs are varying less than 0.1% from the rest I have! (stiff behavior?)
    Optim["minimumResidual"] = -5e-3  # Reducing residual by 200x is enough
    Optim["StrategyOptions"]["AllowedExcursions"] = [
        0.05,
        0.05,
    ]  # This would be 10% if [-100,100]

    # Surrogate
    Optim["surrogateOptions"]["selectSurrogate"] = PORTALStools.selectSurrogate
    # Optim['surrogateOptions']['MinimumRelativeNoise']   = 1E-3  # Minimum error bar (std) of 0.1% of maximum value of each output (untransformed! so careful with far away initial condition)

    Optim["surrogateOptions"]["ensureTrainingBounds"] = True

    # Acquisition
    Optim[
        "optimizers"
    ] = "root_5-botorch-ga"  # Added root which is not a default bc it needs dimX=dimY
    Optim["acquisitionType"] = "posterior_mean"

    return Optim


class evaluatePORTALS(STRATEGYtools.FUNmain):
    def __init__(self, folder, namelist=None, TensorsType=torch.double):
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
            default_namelist_function=default_namelist if (namelist is None) else None,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Default (please change to your desire after instancing the object)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        """
		Parameters to initialize files
		------------------------------
			These parameters are used to initialize the input.gacode to work with, before any PORTALS workflow
			( passed directly to profiles.correct() )
		"""

        self.INITparameters = {
            "recompute_ptot": True,  # Recompute PTOT to match kinetic profiles (after removals)
            "quasineutrality": True,  # Make sure things are quasineutral by changing the *MAIN* ion (D,T or both)  (after removals)
            "removeIons": [],  # Remove this ion from the input.gacode (if D,T,Z, eliminate T with [2])
            "removeFast": False,  # Automatically detect which are fast ions and remove them
            "FastIsThermal": True,  # Do not remove fast, keep their diluiton effect but make them thermal
            "sameDensityGradients": False,  # Make all ion density gradients equal to electrons
            "groupQIONE": False,
            "ensurePostiveGamma": False,
            "ensureMachNumber": None,
        }

        """
		Parameters to run TGYRO
		-----------------------
			The corrections are applied prior to each evaluation, so that things are consistent.
			Here, do not include things that are not specific for a given iteration. Otherwise if they are general
			changes to input.gacode, then that should go into INITparameters.

			if TGYROparameters contains RoaLocations, use that instead of RhoLocations
		"""

        self.TGYROparameters = {
            "RhoLocations": [0.25, 0.45, 0.65, 0.85],
            "ProfilesPredicted": ["te", "ti", "ne"],  # ['nZ','w0']
            "TGYRO_physics_options": {
                "TargetType": 3,
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
        }

        """
		Parameters to run TGLF
		-----------------------
		"""

        self.TGLFparameters = {"TGLFsettings": 5, "extraOptionsTGLF": {}}

        """
		Physics-informed parameters to fit surrogates
		---------------------------------------------
			Note: Dict value indicates what variables need to change at this location to add this one (only one of them is needed)
			Note 2: index key indicates when to transition to next (in terms of number of individuals available for fitting)
			Things to add:
					'aLte': ['aLte'],   'aLti': ['aLti'],      'aLne': ['aLne'],
					'nuei': ['te','ne'],'tite': ['te','ti'],   'c_s': ['te'],      'w0_n': ['w0'],
					'beta_e':  ['te','ne']

			transition_evaluations is the number of points to be fitted that require a parameter transition.
				Note that this ignores ExtraData or ExtraPoints.
					- transition_evaluations[0]: max to only consider gradients
					- transition_evaluations[1]: no beta_e
					- transition_evaluations[2]: full
		"""

        transition_evaluations = [10, 30, 100]
        physicsBasedParams = {
            transition_evaluations[0]: OrderedDict(
                {
                    "aLte": ["aLte"],
                    "aLti": ["aLti"],
                    "aLne": ["aLne"],
                    "aLw0_n": ["aLw0"],
                }
            ),
            transition_evaluations[1]: OrderedDict(
                {
                    "aLte": ["aLte"],
                    "aLti": ["aLti"],
                    "aLne": ["aLne"],
                    "aLw0_n": ["aLw0"],
                    "nuei": ["te", "ne"],
                    "tite": ["te", "ti"],
                    "w0_n": ["w0"],
                }
            ),
            transition_evaluations[2]: OrderedDict(
                {
                    "aLte": ["aLte"],
                    "aLti": ["aLti"],
                    "aLne": ["aLne"],
                    "aLw0_n": ["aLw0"],
                    "nuei": ["te", "ne"],
                    "tite": ["te", "ti"],
                    "w0_n": ["w0"],
                    "beta_e": ["te", "ne"],
                }
            ),
        }

        # If doing trace impurities, alnZ only affects that channel, but the rest of turbulent state depends on the rest of parameters
        physicsBasedParams_trace = copy.deepcopy(physicsBasedParams)
        physicsBasedParams_trace[transition_evaluations[0]]["aLnZ"] = ["aLnZ"]
        physicsBasedParams_trace[transition_evaluations[1]]["aLnZ"] = ["aLnZ"]
        physicsBasedParams_trace[transition_evaluations[2]]["aLnZ"] = ["aLnZ"]

        """
		Parameters to run PORTALS
		-----------------------
		"""

        self.PORTALSparameters = {
            "percentError": [
                10,
                10,
                1,
            ],  # (%) Error (std, in percent) of model evaluation [TGLF, NEO, TARGET]
            "model_used": "tglf_neo-tgyro",  # Options: 'tglf_neo-tgyro', 'cgyro_neo-tgyro'
            "TargetCalc": "tgyro",  # Method to calculate targets (tgyro or powerstate)
            "launchEvaluationsAsSlurmJobs": True,  # Launch each evaluation as a batch job (vs just comand line)
            "useConvectiveFluxes": True,  # If True, then convective flux for final metric (not fitting). If False, particle flux
            "includeFastInQi": False,  # If True, and fast ions have been included, in seprateNEO, sum fast
            "useDiffusivities": False,  # If True, use [chi_e,chi_i,D] instead of [Qe,Qi,Gamma]
            "useFluxRatios": False,  # If True, fit to [Qi,Qe/Qi,Ge/Qi]
            "physicsBasedParams": physicsBasedParams,  # Physics-informed parameters to fit surrogates
            "physicsBasedParams_trace": physicsBasedParams_trace,  # Physics-informed parameters to fit surrogates for trace impurities
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
            "fineTargetsResolution": None,  # If not None, calculate targets with this radial resolution
        }

        self.storeDataSurrogates = [["Turb", "Neo"], ["Tar"]]

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
        profileForBase=None,
        grabFrom=None,
        reevaluateTargets=0,
        seedInitial=None,
    ):
        """
        grabFrom is a folder from which to grab TabularData and MITIMextra
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

        print(">> PORTALS flags pre-check")

        if self.PORTALSparameters["fineTargetsResolution"] is not None:
            if self.PORTALSparameters["TargetCalc"] != "powerstate":
                print(
                    "\t- Requested fineTargetsResolution, so running powerstate target calculations",
                    typeMsg="w",
                )
                self.PORTALSparameters["TargetCalc"] = "powerstate"

        if (
            "InputType" not in self.TGYROparameters["TGYRO_physics_options"]
        ) or self.TGYROparameters["TGYRO_physics_options"]["InputType"] != 1:
            print(
                "\t- In PORTALS TGYRO evaluations, we need to use exact profiles (InputType=1)",
                typeMsg="i",
            )
            self.TGYROparameters["TGYRO_physics_options"]["InputType"] = 1

        if (
            "GradientsType" not in self.TGYROparameters["TGYRO_physics_options"]
        ) or self.TGYROparameters["TGYRO_physics_options"]["GradientsType"] != 0:
            print(
                "\t- In PORTALS TGYRO evaluations, we need to not recompute gradients (GradientsType=0)",
                typeMsg="i",
            )
            self.TGYROparameters["TGYRO_physics_options"]["GradientsType"] = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        keycheck = (
            "RoaLocations" if "RoaLocations" in self.TGYROparameters else "RhoLocations"
        )
        if IOtools.isfloat(ymax_rel):
            ymax_rel = np.array(
                [ymax_rel * np.ones(len(self.TGYROparameters[keycheck]))]
                * len(self.TGYROparameters["ProfilesPredicted"])
            )
        if IOtools.isfloat(ymin_rel):
            ymin_rel = np.array(
                [ymin_rel * np.ones(len(self.TGYROparameters[keycheck]))]
                * len(self.TGYROparameters["ProfilesPredicted"])
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
            grabFrom=grabFrom,
            profileForBase=profileForBase,
            dvs_fixed=dvs_fixed,
            limitsAreRelative=limitsAreRelative,
            restartYN=restartYN,
            hardGradientLimits=hardGradientLimits,
            dfT=self.dfT,
            seedInitial=seedInitial,
        )
        print(">> PORTALS initalization module (END)", typeMsg="i")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Option to restart
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if grabFrom is not None:
            self.reuseTrainingTabular(
                grabFrom, folderWork, reevaluateTargets=reevaluateTargets
            )

        self.extra_params = {
            "PORTALSparameters": self.PORTALSparameters,
            "folder": self.folder,
        }

    def run(self, paramsfile, resultsfile):
        # Read what PORTALS sends
        FolderEvaluation, numPORTALS, dictDVs, dictOFs = self.read(
            paramsfile, resultsfile
        )

        a, b = IOtools.reducePathLevel(self.folder, level=1)
        name = f"portals_{b}_ev{numPORTALS}".format(
            b, numPORTALS
        )  # e.g. portals_jet37_ev0

        # Specify the number of PORTALS evaluation. Copy in case of parallel run
        extra_params_model = copy.deepcopy(self.extra_params)
        extra_params_model["numPORTALS"] = numPORTALS

        # Run
        _, tgyro, powerstate, dictOFs = runModelEvaluator(
            self,
            FolderEvaluation,
            numPORTALS,
            dictDVs,
            name,
            extra_params_model=extra_params_model,
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

        if self.MITIMextra is not None:
            with open(self.MITIMextra, "rb") as handle:
                dictStore = pickle_dill.load(handle)
            dictStore[int(numPORTALS)] = {"tgyro": tgyro}
            dictStore["profiles_original"] = PROFILEStools.PROFILES_GACODE(
                f"{self.folder}/Initialization/input.gacode_original"
            )
            dictStore["profiles_original_un"] = PROFILEStools.PROFILES_GACODE(
                f"{self.folder}/Initialization/input.gacode_original_originalResol_uncorrected"
            )
            with open(self.MITIMextra, "wb") as handle:
                pickle_dill.dump(dictStore, handle)

    def scalarized_objective(self, Y):
        """
        Notes
        -----
                - Y is the multi-output evaluation of the model in the shape of (dim1...N, num_ofs), i.e. this function should not care
                  about number of dimensions
        """

        ofs_ordered_names = np.array(self.Optim["ofs"])

        """
		-------------------------------------------------------------------------
		Prepare transport dictionary
		-------------------------------------------------------------------------
			Note: var_dict['QeTurb'] must have shape (dim1...N, num_radii)
		"""

        var_dict = {}
        for of in ofs_ordered_names:
            var, pos = of.split("_")
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

        of, cal, _, res = PORTALSinteraction.calculatePseudos(
            var_dict, self.PORTALSparameters, self.TGYROparameters, self.powerstate
        )

        return of, cal, res

    def analyze_results(self, plotYN=True, fn=None, restart=False, analysis_level=2):
        self_complete = analyze_results(
            self, plotYN=plotYN, fn=fn, restart=restart, analysis_level=analysis_level
        )

        return self_complete

    def plotPORTALS_metrics(
        self,
        self_parent,
        MITIMextra_dict=None,
        plotAllFluxes=False,
        stds=2,
        plotExpected=True,
    ):
        indecesPlot = [
            self_parent.res.best_absolute_index,
            0,
            -1,
        ]  # [self_parent.res.best_absolute_index,0,None]

        print("- Proceeding to read PORTALS results")
        self.portals_plot = PORTALSplot.PORTALSresults(
            self.folder,
            self_parent.prfs_model,
            self_parent.res,
            MITIMextra_dict=MITIMextra_dict,
            indecesPlot=indecesPlot,
        )

        # It may have changed
        indecesPlot = [
            self.portals_plot.numBest,
            self.portals_plot.numOrig,
            self.portals_plot.numExtra,
        ]

        indexToReferMaxValues = (
            self.portals_plot.numBest
        )  # self.portals_plot.numOrig #self.portals_plot.numBest

        print(f"- Proceeding to plot PORTALS results ({stds}sigma models)")
        self.portals_plot.tgyros[indecesPlot[1]].plot(
            fn=self_parent.fn, prelabel=f"({indecesPlot[1]}) TGYRO - "
        )
        if indecesPlot[0] < len(self.portals_plot.tgyros):
            self.portals_plot.tgyros[indecesPlot[0]].plot(
                fn=self_parent.fn, prelabel=f"({indecesPlot[0]}) TGYRO - "
            )

        figs = [
            self_parent.fn.add_figure(label="PROFILES - Profiles"),
            self_parent.fn.add_figure(label="PROFILES - Powers"),
            self_parent.fn.add_figure(label="PROFILES - Geometry"),
            self_parent.fn.add_figure(label="PROFILES - Gradients"),
            self_parent.fn.add_figure(label="PROFILES - Flows"),
            self_parent.fn.add_figure(label="PROFILES - Other"),
            self_parent.fn.add_figure(label="PROFILES - Impurities"),
        ]

        if indecesPlot[0] < len(self.portals_plot.profiles):
            PROFILEStools.plotAll(
                [
                    self.portals_plot.profiles[indecesPlot[1]],
                    self.portals_plot.profiles[indecesPlot[0]],
                ],
                figs=figs,
                extralabs=[f"{indecesPlot[1]}", f"{indecesPlot[0]}"],
            )

        fig = self_parent.fn.add_figure(label="PORTALS Metrics")
        PORTALSplot.plotConvergencePORTALS(
            self.portals_plot,
            fig=fig,
            plotAllFluxes=plotAllFluxes,
            indexToMaximize=indexToReferMaxValues,
            stds=stds,
        )

        # ----------------------------------------------------------------------------------------------------------------
        # Next analysis
        # ----------------------------------------------------------------------------------------------------------------

        if plotExpected:
            self.plotPORTALS_expected(
                self_parent,
                labelsFluxes=self.portals_plot.labelsFluxes,
                MITIMextra_dict=MITIMextra_dict,
                stds=stds,
            )

    def plotPORTALS_expected(
        self,
        self_parent,
        labelsFluxes={},
        max_plot_points=4,
        plotNext=True,
        MITIMextra_dict=None,
        stds=2,
    ):
        print(
            f"- Proceeding to plot PORTALS expected next variations ({stds}sigma models)"
        )

        trained_points = self_parent.prfs_model.steps[-1].train_X.shape[0]
        indexBest = self_parent.res.best_absolute_index

        # Best point
        plotPoints = [indexBest]
        labelAssigned = [f"#{indexBest} (best)"]

        # Last point
        if (trained_points - 1) != indexBest:
            plotPoints.append(trained_points - 1)
            labelAssigned.append(f"#{trained_points-1} (last)")

        # Last ones
        i = 0
        while len(plotPoints) < max_plot_points:
            if (trained_points - 2 - i) < 1:
                break
            if (trained_points - 2 - i) != indexBest:
                plotPoints.append(trained_points - 2 - i)
                labelAssigned.append(f"#{trained_points-2-i}")
            i += 1

        # First point
        if 0 not in plotPoints:
            if len(plotPoints) == max_plot_points:
                plotPoints[-1] = 0
                labelAssigned[-1] = "#0 (base)"
            else:
                plotPoints.append(0)
                labelAssigned.append("#0 (base)")

        PORTALSplot.plotExpected(
            self_parent.prfs_model,
            self.folder,
            self_parent.fn,
            plotPoints=plotPoints,
            plotNext=plotNext,
            labelAssigned=labelAssigned,
            labelsFluxes=labelsFluxes,
            MITIMextra_dict=MITIMextra_dict,
            stds=stds,
        )

    def reuseTrainingTabular(
        self, folderRead, folderNew, reevaluateTargets=0, restartIfExists=False
    ):
        """
        reevaluateTargets:
                0: No
                1: Quick targets from powerstate with no transport calculation
                2: Full original model (either with tgyro targets or powerstate targets, but also calculate transport)
        """

        if not os.path.exists(folderNew):
            os.system(f"mkdir {folderNew}")
        if not os.path.exists(f"{folderNew}/Outputs"):
            os.system(f"mkdir {folderNew}/Outputs")

        os.system(f"cp {folderRead}/Outputs/TabularData* {folderNew}/Outputs/.")
        os.system(f"cp {folderRead}/Outputs/MITIMextra.pkl {folderNew}/Outputs/.")

        Tabular_Read = BOgraphics.TabularData(
            self.Optim["dvs"],
            self.Optim["ofs"],
            file=f"{folderNew}/Outputs/TabularData.dat",
        )
        TabularErrors_Read = BOgraphics.TabularData(
            self.Optim["dvs"],
            self.Optim["ofs"],
            file=f"{folderNew}/Outputs/TabularDataStds.dat",
        )

        self.Optim["initialPoints"] = len(Tabular_Read.data)
        self.Optim["readInitialTabular"] = True
        self.Optim["initializationFun"] = None

        print(
            f'- Reusing the training set ({self.Optim["initialPoints"]} points )from {folderRead}',
            typeMsg="i",
        )

        if reevaluateTargets > 0:
            print("- Re-evaluate targets", typeMsg="i")

            os.system(f"mkdir {folderNew}/TargetsRecalculate/")

            for numPORTALS in Tabular_Read.data:
                FolderEvaluation = (
                    f"{folderNew}/TargetsRecalculate/Evaluation.{numPORTALS}"
                )
                if not os.path.exists(FolderEvaluation):
                    os.system(f"mkdir {FolderEvaluation}")

                # ------------------------------------------------------------------------------------
                # Produce design variables
                # ------------------------------------------------------------------------------------
                dictDVs = OrderedDict()
                for i in self.Optim["dvs"]:
                    dictDVs[i] = {"value": np.nan}
                dictOFs = OrderedDict()
                for i in self.Optim["ofs"]:
                    dictOFs[i] = {"value": np.nan, "error": np.nan}

                for i in dictDVs:
                    dictDVs[i]["value"] = Tabular_Read.data[numPORTALS][i]

                # ------------------------------------------------------------------------------------
                # Run to evaluate new targets
                # ------------------------------------------------------------------------------------

                a, b = IOtools.reducePathLevel(self.folder, level=1)
                name = f"portals_{b}_targets_ev{numPORTALS}"

                self_copy = copy.deepcopy(self)
                if reevaluateTargets == 1:
                    self_copy.powerstate.TransportOptions["TypeTransport"] = None
                    self_copy.powerstate.TransportOptions[
                        "TypeTarget"
                    ] = self_copy.powerstate.TargetCalc = "powerstate"
                else:
                    self_copy.powerstate.TransportOptions[
                        "TypeTransport"
                    ] = "tglf_neo_tgyro"

                results, tgyro, powerstate, dictOFs = runModelEvaluator(
                    self_copy,
                    FolderEvaluation,
                    numPORTALS,
                    dictDVs,
                    name,
                    restart=restartIfExists,
                    dictOFs=dictOFs,
                )

                # ------------------------------------------------------------------------------------
                # From the TabularData, change ONLY the targets, since I still want CGYRO!
                # ------------------------------------------------------------------------------------

                for i in dictOFs:
                    if "Tar" in i:
                        print(f"Changing {i} in file")
                        Tabular_Read.data[numPORTALS][i] = (
                            dictOFs[i]["value"].cpu().numpy().item()
                        )
                        try:
                            TabularErrors_Read.data[numPORTALS][i] = (
                                dictOFs[i]["error"].cpu().numpy().item()
                            )
                        except:
                            TabularErrors_Read.data[numPORTALS][i] = dictOFs[i]["error"]

            # ------------------------------------------------------------------------------------
            # Update new Tabulars
            # ------------------------------------------------------------------------------------

            Tabular_Read.updateFile()
            TabularErrors_Read.updateFile()


def analyze_results(
    self,
    plotYN=True,
    fn=None,
    restart=False,
    onlyBest=False,
    analysis_level=2,
    useMITIMextra=True,
):
    """
    analysis_level = 2: Standard as other classes. Run best case, plot TGYRO
    analysis_level = 3: Read from Execution and also calculate metrics (4: full metrics)
    """

    if plotYN:
        print(
            "\n ***************************************************************************"
        )
        print("* PORTALS plotting module - PORTALS")
        print(
            "***************************************************************************\n"
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Interpret stuff
    # ----------------------------------------------------------------------------------------------------------------

    (
        variations_original,
        variations_best,
        self_complete,
    ) = self.analyze_optimization_results()

    # ----------------------------------------------------------------------------------------------------------------
    # Running cases: Original and Best
    # ----------------------------------------------------------------------------------------------------------------

    if analysis_level in [2, 5]:
        if not onlyBest:
            print("\t- Running original case")
            FolderEvaluation = f"{self.folder}/Outputs/final_analysis_original/"
            if not os.path.exists(FolderEvaluation):
                IOtools.askNewFolder(FolderEvaluation, force=True)

            dictDVs = {}
            for i in variations_best:
                dictDVs[i] = {"value": variations_original[i]}

            # Run
            a, b = IOtools.reducePathLevel(self.folder, level=1)
            name0 = f"portals_{b}_ev{0}"  # e.g. portals_jet37_ev0

            resultsO, tgyroO, powerstateO, _ = runModelEvaluator(
                self_complete, FolderEvaluation, 0, dictDVs, name0, restart=restart
            )

            # Run TGLF
            powerstateO.tgyro_current.nameRuns_default = name0
            powerstateO.tgyro_current.runTGLF(
                fromlabel=name0,
                rhos=self_complete.TGYROparameters["RhoLocations"],
                restart=restart,
            )

        print(f"\t- Running best case #{self.res.best_absolute_index}")
        FolderEvaluation = f"{self.folder}/Outputs/final_analysis_best/"
        if not os.path.exists(FolderEvaluation):
            IOtools.askNewFolder(FolderEvaluation, force=True)

        dictDVs = {}
        for i in variations_best:
            dictDVs[i] = {"value": variations_best[i]}

        # Run
        a, b = IOtools.reducePathLevel(self.folder, level=1)
        name = f"portals_{b}_ev{self.res.best_absolute_index}"  # e.g. portals_jet37_ev0
        resultsB, tgyroB, powerstateB, _ = runModelEvaluator(
            self_complete,
            FolderEvaluation,
            self.res.best_absolute_index,
            dictDVs,
            name,
            restart=restart,
        )

        # Run TGLF
        powerstateB.tgyro_current.nameRuns_default = name
        powerstateB.tgyro_current.runTGLF(
            fromlabel=name,
            rhos=self_complete.TGYROparameters["RhoLocations"],
            restart=restart,
        )

        # ----------------------------------------------------------------------------------------------------------------
        # Plotting
        # ----------------------------------------------------------------------------------------------------------------

        # Plot
        if plotYN:
            if not onlyBest:
                tgyroO.plotRun(fn=fn, labels=[name0])
            tgyroB.plotRun(fn=fn, labels=[name])

    # ----------------------------------------------------------------------------------------------------------------
    # Ranges of variation
    # ----------------------------------------------------------------------------------------------------------------

    if plotYN:
        figR = fn.add_figure(label="PROFILES Ranges")
        pps = np.max(
            [3, len(self_complete.TGYROparameters["ProfilesPredicted"])]
        )  # Because plotGradients require at least Te, Ti, ne
        grid = plt.GridSpec(2, pps, hspace=0.3, wspace=0.3)
        axsR = []
        for i in range(pps):
            axsR.append(figR.add_subplot(grid[0, i]))
            axsR.append(figR.add_subplot(grid[1, i]))

        # Ranges ---------------------------------------------------------------------------------------------------------
        produceInfoRanges(
            self_complete,
            self.prfs_model.bounds_orig,
            axsR=axsR,
            color="k",
            lw=0.2,
            alpha=0.05,
            label="original",
        )
        produceInfoRanges(
            self_complete,
            self.prfs_model.bounds,
            axsR=axsR,
            color="c",
            lw=0.2,
            alpha=0.05,
            label="final",
        )
        # ----------------------------------------------------------------------------------------------------------------

        if analysis_level > 2:
            if useMITIMextra:
                with open(self_complete.MITIMextra, "rb") as handle:
                    MITIMextra_dict = pickle_dill.load(handle)
            else:
                MITIMextra_dict = None

            # ----------------------------------------------------------------------------------------------------------------
            # Metrics analysis
            # ----------------------------------------------------------------------------------------------------------------
            self_complete.plotPORTALS_metrics(
                self,
                MITIMextra_dict=MITIMextra_dict,
                plotAllFluxes=True,
                plotExpected=analysis_level > 3,
            )

            # ----------------------------------------------------------------------------------------------------------------
            # Ranges of variation
            # ----------------------------------------------------------------------------------------------------------------
            if useMITIMextra:
                plotImpurity = (
                    self_complete.PORTALSparameters["ImpurityOfInterest"]
                    if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]
                    else None
                )
                plotRotation = (
                    "w0" in self_complete.TGYROparameters["ProfilesPredicted"]
                )

                try:
                    avoidPoints = self.prfs_model.avoidPoints
                except:
                    avoidPoints = []

                profiles = (
                    MITIMextra_dict[0]["tgyro"]
                    .results[
                        [ik for ik in MITIMextra_dict[0]["tgyro"].results.keys()][0]
                    ]
                    .profiles
                )
                profiles.plotGradients(
                    axsR,
                    color="b",
                    lastRho=self_complete.TGYROparameters["RhoLocations"][-1],
                    ms=0,
                    lw=1.0,
                    label="#0",
                    ls="-o" if avoidPoints else "--o",
                    plotImpurity=plotImpurity,
                    plotRotation=plotRotation,
                )

                for i in range(100):
                    try:
                        profiles = (
                            MITIMextra_dict[i]["tgyro"]
                            .results[
                                [
                                    ik
                                    for ik in MITIMextra_dict[i]["tgyro"].results.keys()
                                ][0]
                            ]
                            .profiles
                        )
                    except:
                        break
                    profiles.plotGradients(
                        axsR,
                        color="r",
                        lastRho=self_complete.TGYROparameters["RhoLocations"][-1],
                        ms=0,
                        lw=0.3,
                        ls="-o" if avoidPoints else "-.o",
                        plotImpurity=plotImpurity,
                        plotRotation=plotRotation,
                    )

                profiles.plotGradients(
                    axsR,
                    color="g",
                    lastRho=self_complete.TGYROparameters["RhoLocations"][-1],
                    ms=0,
                    lw=1.0,
                    label=f"#{self.res.best_absolute_index} (best)",
                    plotImpurity=plotImpurity,
                    plotRotation=plotRotation,
                )

                axsR[0].legend(loc="best")

            # ----------------------------------------------------------------------------------------------------------------
            # Compare profiles
            # ----------------------------------------------------------------------------------------------------------------
            if useMITIMextra:
                profile_original = (
                    MITIMextra_dict[0]["tgyro"]
                    .results[
                        [ik for ik in MITIMextra_dict[0]["tgyro"].results.keys()][0]
                    ]
                    .profiles
                )
                profile_best = (
                    MITIMextra_dict[self.res.best_absolute_index]["tgyro"]
                    .results[
                        [
                            ik
                            for ik in MITIMextra_dict[self.res.best_absolute_index][
                                "tgyro"
                            ].results.keys()
                        ][0]
                    ]
                    .profiles
                )
                profile_original_unCorrected = MITIMextra_dict["profiles_original_un"]
                profile_original_0 = MITIMextra_dict["profiles_original"]

                fig4 = fn.add_figure(label="PROFILES Comparison")
                grid = plt.GridSpec(
                    2,
                    np.max(
                        [3, len(self_complete.TGYROparameters["ProfilesPredicted"])]
                    ),
                    hspace=0.3,
                    wspace=0.3,
                )
                axs4 = [
                    fig4.add_subplot(grid[0, 0]),
                    fig4.add_subplot(grid[1, 0]),
                    fig4.add_subplot(grid[0, 1]),
                    fig4.add_subplot(grid[1, 1]),
                    fig4.add_subplot(grid[0, 2]),
                    fig4.add_subplot(grid[1, 2]),
                ]

                cont = 1
                if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]:
                    axs4.append(fig4.add_subplot(grid[0, 2 + cont]))
                    axs4.append(fig4.add_subplot(grid[1, 2 + cont]))
                    cont += 1
                if "w0" in self_complete.TGYROparameters["ProfilesPredicted"]:
                    axs4.append(fig4.add_subplot(grid[0, 2 + cont]))
                    axs4.append(fig4.add_subplot(grid[1, 2 + cont]))

                colors = GRAPHICStools.listColors()

                plotImpurity = (
                    self_complete.PORTALSparameters["ImpurityOfInterest"]
                    if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]
                    else None
                )
                plotRotation = (
                    "w0" in self_complete.TGYROparameters["ProfilesPredicted"]
                )

                for i, (profiles, label, alpha) in enumerate(
                    zip(
                        [
                            profile_original_unCorrected,
                            profile_original_0,
                            profile_original,
                            profile_best,
                        ],
                        ["Original", "Corrected", "Initial", "Final"],
                        [0.2, 1.0, 1.0, 1.0],
                    )
                ):
                    profiles.plotGradients(
                        axs4,
                        color=colors[i],
                        label=label,
                        lastRho=self_complete.TGYROparameters["RhoLocations"][-1],
                        alpha=alpha,
                        useRoa=True,
                        RhoLocationsPlot=self_complete.TGYROparameters["RhoLocations"],
                        plotImpurity=plotImpurity,
                        plotRotation=plotRotation,
                    )

                axs4[0].legend(loc="best")

        # self.fn.show()

    return self_complete


def runModelEvaluator(
    self,
    FolderEvaluation,
    numPORTALS,
    dictDVs,
    name,
    restart=False,
    extra_params_model={},
    dictOFs=None,
):
    powerstate = copy.deepcopy(self.powerstate)

    # ---------------------------------------------------------------------------------------------------
    # Prep run
    # ---------------------------------------------------------------------------------------------------

    FolderEvaluation_TGYRO = FolderEvaluation + "/model_complete/"
    os.system(f"mkdir {FolderEvaluation_TGYRO}")

    # Better to write/read each time than passing the class in self.portals_parameters, because self.portals_parameters will be used during the surrogate, which can be expensive

    readFile = f"{FolderEvaluation}/input.gacode_copy_initialization"
    with open(readFile, "w") as f:
        f.writelines(self.file_in_lines_initial_input_gacode)

    # ---------------------------------------------------------------------------------------------------
    # Prepare evaluating vector
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

    powerstate.profiles = PROFILEStools.PROFILES_GACODE(
        readFile, calculateDerived=False
    )

    powerstate.TransportOptions["ModelOptions"]["restart"] = restart

    powerstate.calculate(
        X, nameRun=name, folder=FolderEvaluation_TGYRO, extra_params=extra_params_model
    )

    tgyro_current_results = (
        powerstate.tgyro_current.results["use"]
        if "tgyro_current" in powerstate.__dict__
        else None
    )
    tgyro_current = (
        powerstate.tgyro_current if "tgyro_current" in powerstate.__dict__ else None
    )

    # ---------------------------------------------------------------------------------------------------
    # Produce dictOFs is asked for
    # ---------------------------------------------------------------------------------------------------

    if dictOFs is not None:
        for var in powerstate.ProfilesPredicted:
            # Write in OFs
            for i in range(powerstate.plasma["rho"].shape[1] - 1):
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
                ][0, i]
                dictOFs[f"{var0}Turb_{i+1}"]["error"] = powerstate.plasma[
                    f"{var1}_tr_turb_stds"
                ][0, i]

                dictOFs[f"{var0}Neo_{i+1}"]["value"] = powerstate.plasma[
                    f"{var1}_tr_neo"
                ][0, i]
                dictOFs[f"{var0}Neo_{i+1}"]["error"] = powerstate.plasma[
                    f"{var1}_tr_neo_stds"
                ][0, i]

                """
				TARGET calculation
				---------------------
					If that radius & profile position has target, evaluate
				"""

                dictOFs[f"{var0}Tar_{i+1}"]["value"] = powerstate.plasma[f"{var1}"][
                    0, i
                ]
                dictOFs[f"{var0}Tar_{i+1}"]["error"] = powerstate.plasma[
                    f"{var1}_stds"
                ][0, i]

        """
		Turbulent Exchange
		------------------
		"""
        if extra_params_model["PORTALSparameters"]["surrogateForTurbExch"]:
            for i in range(powerstate.plasma["rho"].shape[1] - 1):
                dictOFs[f"PexchTurb_{i+1}"]["value"] = powerstate.plasma["PexchTurb"][
                    0, i
                ]
                dictOFs[f"PexchTurb_{i+1}"]["error"] = powerstate.plasma[
                    "PexchTurb_stds"
                ][0, i]

    return tgyro_current_results, tgyro_current, powerstate, dictOFs


def produceInfoRanges(
    self_complete, bounds, axsR, label="", color="k", lw=0.2, alpha=0.05
):
    rhos = np.append([0], self_complete.TGYROparameters["RhoLocations"])
    aLTe, aLTi, aLne, aLnZ, aLw0 = (
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
        np.zeros((len(rhos), 2)),
    )
    for i in range(len(rhos) - 1):
        if f"aLte_{i+1}" in bounds:
            aLTe[i + 1, :] = bounds[f"aLte_{i+1}"]
        if f"aLti_{i+1}" in bounds:
            aLTi[i + 1, :] = bounds[f"aLti_{i+1}"]
        if f"aLne_{i+1}" in bounds:
            aLne[i + 1, :] = bounds[f"aLne_{i+1}"]
        if f"aLnZ_{i+1}" in bounds:
            aLnZ[i + 1, :] = bounds[f"aLnZ_{i+1}"]
        if f"aLw0_{i+1}" in bounds:
            aLw0[i + 1, :] = bounds[f"aLw0_{i+1}"]

    X = torch.zeros(
        ((len(rhos) - 1) * len(self_complete.TGYROparameters["ProfilesPredicted"]), 2)
    )
    l = len(rhos) - 1
    X[0:l, :] = torch.from_numpy(aLTe[1:, :])
    X[l : 2 * l, :] = torch.from_numpy(aLTi[1:, :])

    cont = 0
    if "ne" in self_complete.TGYROparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLne[1:, :])
        cont += 1
    if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLnZ[1:, :])
        cont += 1
    if "w0" in self_complete.TGYROparameters["ProfilesPredicted"]:
        X[(2 + cont) * l : (3 + cont) * l, :] = torch.from_numpy(aLw0[1:, :])
        cont += 1

    X = X.transpose(0, 1)

    powerstate = PORTALStools.constructEvaluationProfiles(
        X, self_complete.surrogate_parameters, recalculateTargets=False
    )

    GRAPHICStools.fillGraph(
        axsR[0],
        powerstate.plasma["rho"][0],
        powerstate.plasma["te"][0],
        y_up=powerstate.plasma["te"][1],
        alpha=alpha,
        color=color,
        lw=lw,
        label=label,
    )
    GRAPHICStools.fillGraph(
        axsR[1],
        rhos,
        aLTe[:, 0],
        y_up=aLTe[:, 1],
        alpha=alpha,
        color=color,
        label=label,
        lw=lw,
    )

    GRAPHICStools.fillGraph(
        axsR[2],
        powerstate.plasma["rho"][0],
        powerstate.plasma["ti"][0],
        y_up=powerstate.plasma["ti"][1],
        alpha=alpha,
        color=color,
        label=label,
        lw=lw,
    )
    GRAPHICStools.fillGraph(
        axsR[3],
        rhos,
        aLTi[:, 0],
        y_up=aLTi[:, 1],
        alpha=alpha,
        color=color,
        label=label,
        lw=lw,
    )

    cont = 0
    if "ne" in self_complete.TGYROparameters["ProfilesPredicted"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["ne"][0] * 0.1,
            y_up=powerstate.plasma["ne"][1] * 0.1,
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 2],
            rhos,
            aLne[:, 0],
            y_up=aLne[:, 1],
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        cont += 2

    if "nZ" in self_complete.TGYROparameters["ProfilesPredicted"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["nZ"][0] * 0.1,
            y_up=powerstate.plasma["nZ"][1] * 0.1,
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 2],
            rhos,
            aLnZ[:, 0],
            y_up=aLnZ[:, 1],
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        cont += 2

    if "w0" in self_complete.TGYROparameters["ProfilesPredicted"]:
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 1],
            powerstate.plasma["rho"][0],
            powerstate.plasma["w0"][0] * 1e-3,
            y_up=powerstate.plasma["w0"][1] * 1e-3,
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        GRAPHICStools.fillGraph(
            axsR[3 + cont + 2],
            rhos,
            aLw0[:, 0],
            y_up=aLw0[:, 1],
            alpha=alpha,
            color=color,
            label=label,
            lw=lw,
        )
        cont += 2
