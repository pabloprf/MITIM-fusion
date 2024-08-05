import os
import copy
import datetime
import array
import traceback
import torch
from collections import OrderedDict
from IPython import embed
import dill as pickle_dill
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import OPTtools, STEPtools
from mitim_tools.opt_tools.utils import (
    BOgraphics,
    SBOcorrections,
    TESTtools,
    EVALUATORtools,
    SAMPLINGtools,
)
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print

UseCUDAifAvailable = True

"""
Example usage (see tutorials for actual examples and parameter definitions):

	# Define function to optimize

		class optimization_object(opt_evaluator):

			def __init__(self,folder,namelist=None,function_parameters={}):

				super().__init__(folder,namelist=namelist)

				self.function_parameters = function_parameters

			def run(self,paramsfile,resultsfile):

				# Read stuff
				FolderEvaluation,numEval,dictDVs,dictOFs = self.read(paramsfile,resultsfile)

				# Operations
				...

				# Write stuff
				self.write(dictOFs,resultsfile)

			def analyze_results(self,plotYN=True,fn = None):

				# Things to do when looking at final results [OPTIONAL]

	# Run Workflow

	PRF_BO = STRATEGYtools.PRF_BO(optimization_object)
	PRF_BO.run()

Notes:
	- A "nan" in the evaluator output (e.g. written in optimization_data) means that it has not been evaluated, so this is prone to be tried again.
		This is especially useful when I only want to read from a previous optimization workflow the x values, but I want to evaluate.
	- An "inf" in the evaluator output (e.g. written in optimization_data) means that the evaluation failed and won't be re-tried again. That individual
		will just not be considered during surrogate fitting.
"""


# Parent optimization function
class opt_evaluator:
    def __init__(
        self,
        folder,
        namelist=None,
        TensorsType=torch.double,
        default_namelist_function=None,
    ):
        """
        Namelist file can be provided and will be copied to the folder
        """

        print("- Parent opt_evaluator function initialized")

        self.folder = folder

        print(f"\t- Folder: {self.folder}")

        if self.folder is not None:

            if self.folder[-1] != "/":
                self.folder += "/"

            self.folder = IOtools.expandPath(self.folder)
            if not os.path.exists(self.folder):
                IOtools.askNewFolder(self.folder)
            if not os.path.exists(self.folder + "/Outputs/"):
                IOtools.askNewFolder(self.folder + "/Outputs/")

        if namelist is not None:
            print(f"\t- Namelist provided: {namelist}", typeMsg="i")

            self.optimization_options = IOtools.read_mitim_nml(namelist)

        elif default_namelist_function is not None:
            print(
                "\t- Namelist not provided, using MITIM default for this optimization sub-module",
                typeMsg="i",
            )

            namelist = IOtools.expandPath("$MITIM_PATH/templates/main.namelist.json")
            self.optimization_options = IOtools.read_mitim_nml(namelist)

            self.optimization_options = default_namelist_function(self.optimization_options)

        else:
            print(
                "\t- No namelist provided (likely b/c for reading/plotting purposes)",
                typeMsg="i",
            )
            self.optimization_options = None

        self.surrogate_parameters = {
            "parameters_combined": {},
            "physicsInformedParams_dict": None,
            "physicsInformedParamsComplete": None,
            "transformationInputs": STEPtools.identity,  # Transformation of inputs
            "transformationOutputs": STEPtools.identityOutputs,  # Transformation of outputs
        }

        # Determine type of tensors to work with
        torch.set_default_dtype(
            TensorsType
        )  # In case I forgot to specify a type explicitly, use as default (https://github.com/pytorch/botorch/discussions/1444)
        self.dfT = torch.randn(
            (2, 2),
            dtype=TensorsType,
            device=torch.device(
                "cpu"
                if ((not UseCUDAifAvailable) or (not torch.cuda.is_available()))
                else "cuda"
            ),
        )

        # Name of calibrated objectives (e.g. QiRes1 to represent the objective from Qi1-QiT1)
        self.name_objectives = None

        # Name of transformed functions (e.g. Qi1_GB to represent the transformation of Qi1)
        self.name_transformed_ofs = None

        # Variables in the class not to save (e.g. COMSOL model)
        self.doNotSaveVariables = []

    def read(self, paramsfile, resultsfile):
        # Read stuff
        (
            FolderEvaluation,
            numEval,
            inputFilePath,
            outputFilePath,
        ) = IOtools.obtainGeneralParams(paramsfile, resultsfile)
        MITIMparams = IOtools.generateDictionaries(inputFilePath)
        dictDVs = MITIMparams["dictDVs"]
        dictOFs = MITIMparams["dictOFs"]

        # Do not store as part of the class not to confuse parallel evals
        return FolderEvaluation, numEval, dictDVs, dictOFs

    def write(self, dictOFs, resultsfile):
        IOtools.writeOFs(resultsfile, dictOFs)

    """
	**********************************************************************************************************************************************************
	Methods that need to be re-defined by children classes for specific applications
	**********************************************************************************************************************************************************
	"""

    def run(self, paramsfile, resultsfile):
        # Read stuff
        FolderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        # Operations (please, modify as needed when this class is used as parent)
        pass

        # Write stuff
        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        """
        * Receives Y as (batch1...N,dimY)
        * Must produce OF (batch1...N,dimYof), CAL (batch1...N,dimYof) and residual (batch1...N)
        * Notes:
                - Residual must be ready for maximization. It's used by the residual tracker (best) and some optimization algorithms
                - The reason why OF and CAL must be provided is because of visualization purposes of matching conditions and for optimization algorithms such as ROOT
                - Works with tensors
                - Here is where I control weights, relatives, etc
        """
        pass

    # **********************************************************************************************************************************************************

    def read_optimization_results(
        self,
        plotYN=False,
        folderRemote=None,
        analysis_level=0,
        pointsEvaluateEachGPdimension=50,
    ):
        with np.errstate(all="ignore"):
            CONFIGread.ignoreWarnings()
            (
                self.fn,
                self.res,
                self.prfs_model,
                self.log,
                self.data,
            ) = BOgraphics.retrieveResults(
                self.folder,
                analysis_level=analysis_level,
                doNotShow=True,
                plotYN=plotYN,
                folderRemote=folderRemote,
                pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
            )

        # Make folders local
        try:
            self.prfs_model.folderOutputs = self.prfs_model.folderOutputs.replace(
                self.prfs_model.folderExecution, self.folder
            )
            self.prfs_model.optimization_extra = self.prfs_model.optimization_object.optimization_extra = (
                self.prfs_model.optimization_extra.replace(
                    self.prfs_model.folderExecution, self.folder
                )
            )
            self.prfs_model.folderExecution = self.prfs_model.optimization_object.folder = (
                self.folder
            )
        except:
            pass

    def analyze_optimization_results(self):
        print("- Analyzing MITIM BO results")

        # ----------------------------------------------------------------------------------------------------------------
        # Interpret stuff
        # ----------------------------------------------------------------------------------------------------------------

        if "res" not in self.__dict__.keys():
            self.read_optimization_results()
        variations_best = self.res.best_absolute_full["x"]
        variations_original = self.res.evaluations[0]["x"]

        print(
            f"\t- Best case in MITIM was achieved at evaluation #{self.res.best_absolute_index}:"
        )
        for ikey in variations_best:
            print(f"\t\t* {ikey} = {variations_best[ikey]}")

        try:
            self_complete = self.prfs_model.optimization_object
        except:
            self_complete = None
            print("\t- Problem retrieving function", typeMsg="w")

        return variations_original, variations_best, self_complete

    def plot_optimization_results(
        self,
        analysis_level=0,
        folderRemote=None,
        retrieval_level=None,
        plotYN=True,
        pointsEvaluateEachGPdimension=50,
        save_folder=None,
    ):
        time1 = datetime.datetime.now()

        if analysis_level < 0:
            print("\t- Only read optimization_results.out")
        if analysis_level == 0:
            print("\t- Only plot optimization_results.out")
        if analysis_level == 1:
            print("\t- Read optimization_results.out and pickle")
        if analysis_level == 2:
            print("\t- Perform full analysis")
        if analysis_level > 2:
            print(
                f"\t- Perform extra analysis for this sub-module (analysis level {analysis_level})"
            )

        self.read_optimization_results(
            plotYN=plotYN and (analysis_level >= 0),
            folderRemote=folderRemote,
            analysis_level=(
                retrieval_level if (retrieval_level is not None) else analysis_level
            ),
            pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
        )

        self_complete = None
        if analysis_level > 1:
            """
            If the analyze_results exists, I'm in a child class, so just proceed to analyze.
            Otherwise, let's grab the method from the pickled
            """
            if hasattr(self, "analyze_results"):
                self_complete = self.analyze_results(
                    plotYN=plotYN, fn=self.fn, analysis_level=analysis_level
                )

            else:
                # What function is it?
                class_name = str(self.prfs_model.optimization_object).split()[0].split(".")[-1]
                print(
                    f'\t- Retrieving "analyze_results" method from class "{class_name}"',
                    typeMsg="i",
                )

                if class_name == "freegsu":
                    from mitim_modules.freegsu.FREEGSUmain import analyze_results
                elif class_name == "vitals":
                    from mitim_modules.vitals.VITALSmain import analyze_results
                elif class_name == "portals":
                    from mitim_modules.portals.PORTALSmain import analyze_results
                else:
                    analyze_results = None

                if analyze_results is not None:
                    self_complete = analyze_results(
                        self, plotYN=plotYN, fn=self.fn, analysis_level=analysis_level
                    )
                else:
                    print(
                        '\t- No "analyze_results" method found for this function class',
                        typeMsg="w",
                    )

        if plotYN and (analysis_level >= 0):
            print(f"\n- Plotting took {IOtools.getTimeDifference(time1)}")

            if save_folder is not None:
                self.fn.save(save_folder)

        return self_complete


# Main BO class that performs optimization
class PRF_BO:
    def __init__(
        self,
        optimization_object,
        restartYN=False,
        storeClass=True,
        onlyInitialize=False,
        seed=0,
        askQuestions=True,
    ):
        """
        Inputs:
                - optimization_object   :  Function that is executed,
                        with .optimization_options in it (Dictionary with optimization parameters (must be obtained using namelist and read_mitim_nml))
                        and .folder (Where the function runs)
                        and surrogate_parameters: Parameters to pass to surrogate (e.g. for transformed function), It can be different from function_parameters because of making evaluations fast.
                - restartYN 	 :  If False, try to find the values from Outputs/optimization_data.csv
                - storeClass 	 :  If True, write a class pickle for well-behaved restarting
                - askQuestions 	 :  To avoid that a SLURM job gets stop becuase something is asked, set to False
        """

        self.optimization_object = optimization_object
        self.restartYN = restartYN
        self.storeClass = storeClass
        self.askQuestions = askQuestions
        self.seed = seed
        self.avoidPoints = []

        if (not self.restartYN) and askQuestions:
            if not print(
                f"\t* Because {restartYN = }, MITIM will try to read existing results from folder",
                typeMsg="q",
            ):
                raise Exception("[MITIM] - User requested to stop")

        if self.optimization_object.name_objectives is None:
            self.optimization_object.name_objectives = "y"

        # Folders and Logger
        self.folderExecution = (
            IOtools.expandPath(self.optimization_object.folder)
            if (self.optimization_object.folder is not None)
            else ""
        )

        self.folderOutputs = self.folderExecution + "/Outputs/"

        if optimization_object.optimization_options is not None:
            if not os.path.exists(self.folderOutputs):
                IOtools.askNewFolder(self.folderOutputs, force=True)

            """
			Prepare class where I will store some extra data
			---
			Do not carry out this dictionary through the workflow, just read and write
			"""

            self.optimization_extra = f"{self.folderOutputs}/optimization_extra.pkl"

            # Read if exists
            exists = False
            if os.path.exists(self.optimization_extra):
                try:
                    with open(self.optimization_extra, "rb") as handle:
                        dictStore = pickle_dill.load(handle)
                    exists = True
                except (ModuleNotFoundError,EOFError):
                    exists = False
                    print('Problem loading "optimization_extra.pkl"',typeMsg="w")
            
            # nans if not
            if not exists:
                dictStore = {}
                for i in range(200):
                    dictStore[i] = np.nan

            # Write
            with open(self.optimization_extra, "wb") as handle:
                pickle_dill.dump(dictStore, handle)

            # Write the class into the optimization_object
            optimization_object.optimization_extra = self.optimization_extra

        # Function to execute
        self.surrogate_parameters = self.optimization_object.surrogate_parameters
        self.optimization_options = self.optimization_object.optimization_options

        # Check if the variables are expected
        if self.optimization_options is not None:
            namelist = IOtools.expandPath("$MITIM_PATH/templates/main.namelist.json")
            Optim_potential = IOtools.read_mitim_nml(namelist)
            for ikey in self.optimization_options:
                if ikey not in Optim_potential:
                    print(
                        f"\t- optimization_options['{ikey}'] is an unexpected variable, prone to errors or misinterpretation",
                        typeMsg="q",
                    )
        # -----------------------------------

        if not onlyInitialize:
            print(
                "\n-----------------------------------------------------------------------------------------"
            )
            print("\t\t\t BO class module")
            print(
                "-----------------------------------------------------------------------------------------\n"
            )

            """
			------------------------------------------------------------------------------
			Grab variables
			------------------------------------------------------------------------------
			"""

            # Logger
            self.logFile = BOgraphics.LogFile(self.folderOutputs + "optimization_log.txt")
            self.logFile.activate()

            # Meta
            self.numIterations = self.optimization_options["BO_iterations"]
            self.StrategyOptions = self.optimization_options["StrategyOptions"]
            self.parallel_evaluations = self.optimization_options["parallel_evaluations"]
            self.dfT = self.optimization_object.dfT

            """
			Notes about the "avoidPoints" variables
			---------------------------------------
				The avoidPoints_failed list is updated in the following instances:
					- When evaluating initial batch, result has failed for this simulation
					- When updating the set, result has failed for this simulation
				The avoidPoints_outside list is updated in the following instances:
					- When reducing the trust region, DV fall outside of new bounds
				After each iteration, avoidPoints is re-constructed with the failed points and the outside points.
				This logic is because when expanding the TR, points may fall again into the TR, so I don't
				want to keep the same track iteration after iteration
			"""
            self.avoidPoints_failed, self.avoidPoints_outside = [], []

            # Initialize the metrics as a dictionary with fixed size of number of iterations that I will be updating
            self.hard_finish = False
            self.numEval = 0
            self.keys_metrics = [
                "BOratio",
                "xBest_track",
                "yBest_track",
                "yVarBest_track",
                "BOmetric",
                "TRoperation",
                "BoundsStorage",
                "iteration",
                "BOmetric_it",
            ]
            self.BOmetrics = {"overall": {}}
            for ikey in self.keys_metrics:
                self.BOmetrics[ikey] = {}
                for i in range(self.optimization_options["BO_iterations"] + 1):
                    self.BOmetrics[ikey][i] = np.nan

            """
			------------------------------------------------------------------------------
			Prepare Desgin variables (DVs) with minimum and maximum values
			------------------------------------------------------------------------------
			"""

            self.bounds, self.boundsInitialization = OrderedDict(), []
            for cont, i in enumerate(self.optimization_options["dvs"]):
                self.bounds[i] = np.array(
                    [self.optimization_options["dvs_min"][cont], self.optimization_options["dvs_max"][cont]]
                )
                self.boundsInitialization.append(
                    np.array([self.optimization_options["dvs_min"][cont], self.optimization_options["dvs_max"][cont]])
                )

            self.boundsInitialization = np.transpose(self.boundsInitialization)

            # Bounds may change during the workflow (corrections, TURBO)
            self.bounds_orig = copy.deepcopy(self.bounds)

            """
			----------------------------------------------------------------------------------------------------------------------------
			Prepare Objective functions (OFs) with minimum and maximum values
			----------------------------------------------------------------------------------------------------------------------------
			"""

            # Objective functions (OFs)
            self.outputs = self.surrogate_parameters["outputs"] = self.optimization_options["ofs"]

            # How many points each iteration will produce?
            self.best_points_sequence = self.optimization_options["newPoints"]
            self.best_points = int(np.sum(self.best_points_sequence))

            """
			------------------------------------------------------------------------------
			Prepare Initialization
			------------------------------------------------------------------------------
			"""

            if (
                (self.optimization_options["type_initialization"] == 1)
                and (os.path.exists(self.folderExecution + "Execution/Evaluation.1/"))
                and (self.restartYN)
            ):
                print(
                    "\t--> Random initialization has been requested",
                    typeMsg="q" if self.askQuestions else "qa",
                )

            self.type_initialization = self.optimization_options["type_initialization"]
            self.initial_training = self.optimization_options["initial_training"]

            """
			------------------------------------------------------------------------------
			Initialize Output files
			------------------------------------------------------------------------------
			"""

            if (self.type_initialization == 3) and (self.restartYN):
                print(
                    "\t* Initialization based on Tabular, yet restart has been requested. I am NOT removing the previous optimization_data",
                    typeMsg="w",
                )
                if self.askQuestions:
                    flagger = print(
                        "\t\t* Are you sure this was your intention?", typeMsg="q"
                    )
                    if not flagger:
                        embed()
                forceNewTabulars = False
            else:
                forceNewTabulars = self.restartYN

            inputs = [i for i in self.bounds]

            self.lambdaSingleObjective = self.optimization_object.scalarized_objective

            self.optimization_data = BOgraphics.optimization_data(
                inputs,
                self.outputs,
                file=self.folderOutputs + "/optimization_data.csv",
                forceNew=forceNewTabulars,
            )

            res_file = self.folderOutputs + "/optimization_results.out"

            """
			------------------------------------------------------------------------------
			Parameters that will be needed at each step (unchanged)
			------------------------------------------------------------------------------
			"""

            self.stepSettings = {
                "optimization_options": self.optimization_options,
                "dfT": self.dfT,
                "bounds_orig": self.bounds_orig,
                "best_points_sequence": self.best_points_sequence,
                "folderOutputs": self.folderOutputs,
                "fileOutputs": res_file,
                "name_objectives": self.optimization_object.name_objectives,
                "name_transformed_ofs": self.optimization_object.name_transformed_ofs,
                "outputs": self.outputs,
            }

            self.optimization_results = BOgraphics.optimization_results(file=res_file)

            self.optimization_results.initialize(self)

    def run(self):
        """
        Notes:
                - self.train_X,self.train_Y are still provided in absolute units, not normalized
                - train_Ystd is in standard deviations (square root of the variance), not normalized and not relative
        """

        timeBeginning = datetime.datetime.now()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.initializeOptimization()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Iterative workflow
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.StrategyOptions_use = self.StrategyOptions

        self.steps, self.resultsSet = [], []
        for self.currentIteration in range(self.numIterations + 1):
            timeBeginningThis = datetime.datetime.now()

            print("\n------------------------------------------------------------")
            print(
                f'\tMITIM Step {self.currentIteration} ({timeBeginningThis.strftime("%Y-%m-%d %H:%M:%S")})'
            )
            print("------------------------------------------------------------")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~ Update training population with next points
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if self.currentIteration > 0:
                print(
                    f"--> Proceeding to updating set (which currently has {len(self.train_X)} points)"
                )

                # *NOTE*: self.x_next has been updated earlier, either from a restart or from the full workflow
                yN, yNstd = self.updateSet(self.StrategyOptions_use)

                # Stored in previous step
                self.steps[-1].y_next = yN
                self.steps[-1].ystd_next = yNstd

            # After evaluating metrics inside updateSet, I may have requested a hard finish
            if self.hard_finish:
                print("- Hard finish has been requested", typeMsg="i")

                # Removing those spaces in the metrics that were not filled up
                for ikey in self.keys_metrics:
                    for i in range(
                        self.currentIteration + 1, self.optimization_options["BO_iterations"] + 1
                    ):
                        del self.BOmetrics[ikey][i]
                # ------------------------------------------------------------------------------------------

                break

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~ Perform BO step
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Does the tabular file include at least as many rows as requested to be run in this step?
            pointsTabular = len(
                self.optimization_data.data
            )  # Number of points that the Tabular contains
            pointsExpected = len(self.train_X) + self.best_points
            if not self.restartYN:
                if not pointsTabular >= pointsExpected:
                    print(
                        f"--> Because points are not all in Tabular ({pointsTabular}/{pointsExpected}), disabling restarting-from-previous from this point on",
                        typeMsg="w",
                    )
                    self.restartYN = True
                else:
                    print(
                        f"--> Tabular contains at least as many points as expected at this stage ({pointsTabular}/{pointsExpected})",
                        typeMsg="f",
                    )

            # In the case of starting from previous, do not run BO process.
            if not self.restartYN:
                """
                Philosophy of restarting is:
                        - Restarting requires that settings are the same (e.g. best_points).
                        - Restarting requires Tabular data, since I will be grabbing from it.
                        - The pkl file in restarting is only used to store the "step" data with opt info, but not
                                required to continue. But here I enforce it anyway. I want to know what I have done.
                        - Later I pass the x_next from the pickle, so I could just trust the pickle if I wanted.
                """

                # Read step from pkl
                current_step = self.read()

                if current_step is None:
                    print(
                        "\t* Because reading pkl step had problems, disabling restarting-from-previous from this point on",
                        typeMsg="w",
                    )
                    print(
                        "\t* Are you aware of the consequences of continuing?",
                        typeMsg="q",
                    )

                    self.restartYN = True

            if not self.restartYN:
                # Read next from Tabular
                self.x_next, _, _ = self.optimization_data.extract_points(
                    points=np.arange(
                        len(self.train_X), len(self.train_X) + self.best_points
                    )
                )
                self.x_next = torch.from_numpy(self.x_next).to(self.dfT)

                # Re-write x_next from the pkl... reason for this is that if optimization is heuristic, I may prefer what was in Tabular
                if current_step is not None:
                    current_step.x_next = self.x_next

                # If there is any Nan, assume that I cannot restart this step
                if IOtools.isAnyNan(self.x_next.cpu()):
                    print(
                        "\t* Because x_next points have NaNs, disabling restarting-from-previous from this point on",
                        typeMsg="w",
                    )
                    self.restartYN = True

                # Step is valid, append to this current one
                if not self.restartYN:
                    self.steps.append(current_step)

                # When restarting, make sure that the strategy options are preserved (like correction, bounds and TURBO)
                self.StrategyOptions_use = current_step.StrategyOptions_use

                print("\t* Step successfully restarted from pkl file", typeMsg="f")

            # Standard (i.e. start from beginning, not read values)
            if self.restartYN:
                # For standard use, use the actual strategyOptions launched
                self.StrategyOptions_use = self.StrategyOptions

                # Remove from tabular next points in case they were there. Since I'm not restarting, I don't care about what has come next
                self.optimization_data.removePointsAfter(len(self.train_X) - 1)

                """
				---------------------------------------------------------------------------------------
				BOstep is in charge to fit models and optimize objective function
							(inputs and returns are unnormalized)
				---------------------------------------------------------------------------------------
				"""

                train_Ystd = (
                    self.train_Ystd
                    if (self.optimization_options["train_Ystd"] is None)
                    else self.optimization_options["train_Ystd"]
                )

                # Making copy because it changes per step ---- ---- ---- ---- ----
                surrogate_parameters = copy.deepcopy(self.surrogate_parameters)
                # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

                current_step = STEPtools.OPTstep(
                    self.train_X,
                    self.train_Y,
                    train_Ystd,
                    bounds=self.bounds,
                    stepSettings=self.stepSettings,
                    currentIteration=self.currentIteration,
                    StrategyOptions=self.StrategyOptions_use,
                    BOmetrics=self.BOmetrics,
                    surrogate_parameters=surrogate_parameters,
                )

                # Incorporate strategy_options for later retrieving
                current_step.StrategyOptions_use = copy.deepcopy(
                    self.StrategyOptions_use
                )

                self.steps.append(current_step)

                # Avoid points
                avoidPoints = np.append(
                    self.avoidPoints_failed, self.avoidPoints_outside
                )
                self.avoidPoints = np.unique([int(j) for j in avoidPoints])

                # ***** Fit
                self.steps[-1].fit_step(avoidPoints=self.avoidPoints)

                # Store class with the model fitted
                if self.storeClass:
                    self.save()

                # ***** Optimize
                self.steps[-1].optimize(
                    self.lambdaSingleObjective,
                    position_best_so_far=self.BOmetrics["overall"]["indBest"],
                    seed=self.seed,
                )

            # Pass the information about next step
            self.x_next = self.steps[-1].x_next

            # ~~~~~~~~ Store class now with the next points found (after optimization)
            if self.storeClass and self.restartYN:
                self.save()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Run last point with actual model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if not self.hard_finish:
            print("\n\n------------------------------------------------------------")
            print(" Final evaluation of optima predicted at last MITIM step")
            print("------------------------------------------------------------\n\n")

            _, _ = self.updateSet(self.StrategyOptions, ForceNotApplyCorrections=True)

        self.save()

        print(
            f"- Complete MITIM workflow took {IOtools.getTimeDifference(timeBeginning)} ~~"
        )
        print("********************************************************\n")

    def prepare_for_save_PRFBO(self, copyClass):
        """
        Downselect what elements to store
        """

        # -------------------------------------------------------------------------------------------------
        # To avoid circularity when restarting, do not store the class in the optimization_results sub-class
        # -------------------------------------------------------------------------------------------------

        del copyClass.optimization_results.PRF_BO

        # -------------------------------------------------------------------------------------------------
        # Saving state files with functions is very expensive (deprecated maybe when I had lambdas?) [TO REMOVE]
        # -------------------------------------------------------------------------------------------------

        del copyClass.lambdaSingleObjective

        for i in range(len(self.steps)):
            if "functions" in copyClass.steps[i].__dict__:
                del copyClass.steps[i].functions
            if "evaluators" in copyClass.steps[i].__dict__:
                del copyClass.steps[i].evaluators

        # -------------------------------------------------------------------------------------------------
        # Add time stamp
        # -------------------------------------------------------------------------------------------------

        copyClass.timeStamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return copyClass

    def prepare_for_read_PRFBO(self, copyClass):
        """
        Repair the downselection
        """

        copyClass.optimization_results.PRF_BO = copy.deepcopy(self)

        copyClass.lambdaSingleObjective = copyClass.optimization_object.scalarized_objective

        for i in range(len(copyClass.steps)):
            copyClass.steps[i].defineFunctions(copyClass.lambdaSingleObjective)

        return copyClass

    def save(self, name="optimization_object.pkl"):
        print("* Proceeding to save new MITIM state pickle file")
        stateFile = f"{self.folderOutputs}/{name}"
        stateFile_tmp = f"{self.folderOutputs}/{name}_tmp"

        # Do not store certain variables (that cannot even be copied, that's why I do it here)
        saver = {}
        for ikey in self.optimization_object.doNotSaveVariables:
            saver[ikey] = self.optimization_object.__dict__[ikey]
            del self.optimization_object.__dict__[ikey]
        # -----------------------------------------------------------------------------------

        copyClass = self.prepare_for_save_PRFBO(copy.deepcopy(self))

        with open(stateFile_tmp, "wb") as handle:
            try:
                pickle_dill.dump(copyClass, handle)
            except:
                print("problem saving")

        # Get variables back ----------------------------------------------------------------
        for ikey in saver:
            self.optimization_object.__dict__[ikey] = saver[ikey]
        # -----------------------------------------------------------------------------------

        os.system(
            f"mv {stateFile_tmp} {stateFile}"
        )  # This way I reduce the risk of getting a mid-creation file

        print(
            f"\t- MITIM state file {stateFile} generated, containing the PRF_BO class"
        )

    def read(
        self, name="optimization_object.pkl", iteration=None, file=None, provideFullClass=False
    ):
        iteration = iteration or self.currentIteration

        print("- Reading pickle file with optimization_object class")
        stateFile = file if (file is not None) else f"{self.folderOutputs}/{name}"

        try:
            # If I don't create an Individual attribute I cannot unpickle GA information
            try:
                import deap

                deap.creator.create("Individual", array.array)
            except:
                pass
            with open(stateFile, "rb") as f:
                try:
                    aux = pickle_dill.load(f)
                except:
                    print(
                        "Pickled file could not be opened, likely because of GPU-based tensors, going with custom unpickler..."
                    )
                    f.seek(0)
                    aux = CPU_Unpickler(f).load()

            aux = self.prepare_for_read_PRFBO(aux)

            step = aux.steps[iteration]
            print(
                f"\t* Read {IOtools.clipstr(stateFile)} state file, grabbed step #{iteration}",
                typeMsg="f",
            )
        except FileNotFoundError:
            print(f"\t- State file {stateFile} not found", typeMsg="w")
            step, aux = None, None

        return aux if provideFullClass else step

    def updateSet(
        self, StrategyOptions_use, isThisCorrected=False, ForceNotApplyCorrections=False
    ):
        # ~~~~~~~~~~~~~~~~~~
        # What's the expected value of the next points?
        # ~~~~~~~~~~~~~~~~~~

        y, u, l, _ = self.steps[-1].GP["combined_model"].predict(self.x_next)
        self.y_next_pred = y.detach()
        self.y_next_pred_u = u.detach()
        self.y_next_pred_l = l.detach()

        # ~~~~~~~~~~~~~~~~~~
        # What's the actual value of the next points? Insert them in the database
        # ~~~~~~~~~~~~~~~~~~

        # Update the train_X
        self.train_X = np.append(self.train_X, self.x_next.cpu(), axis=0)

        # Update optimization_data with nans
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.numpy())

        # Update optimization_results only as "predicted"
        if not isThisCorrected:
            self.optimization_results.addPoints(
                includePoints=[
                    len(self.train_X) - len(self.x_next),
                    len(self.train_X) - len(self.x_next) + 1,
                ],
                executed=False,
                predicted=True,
                Best=True,
            )
            self.optimization_results.addPoints(
                includePoints=[len(self.train_X) - len(self.x_next), len(self.train_X)],
                executed=False,
                predicted=True,
                Name=f"Evaluating points from iteration {self.currentIteration}, comprised of {len(self.x_next)} points",
            )

        # --- Evaluation
        time1 = datetime.datetime.now()
        y_next, ystd_next, self.numEval = EVALUATORtools.fun(
            self.optimization_object,
            self.x_next,
            self.folderExecution,
            self.bounds,
            self.outputs,
            self.optimization_data,
            parallel=self.parallel_evaluations,
            restartYN=self.restartYN,
            numEval=self.numEval,
        )
        txt_time = IOtools.getTimeDifference(time1)
        print(f"\t- Complete model update took {txt_time}")
        # ------------------

        # Update the train_Y
        self.train_Y = np.append(self.train_Y, y_next, axis=0)
        self.train_Ystd = np.append(self.train_Ystd, ystd_next, axis=0)

        # --- If problem in evaluation don't use this point -------------------------------------------------------------------
        for i in range(self.train_Y.shape[0]):
            boole = (np.isinf(self.train_Y[i]).any()) and (
                i not in self.avoidPoints_failed
            )
            if boole:
                self.avoidPoints_failed.append(i)
        if len(self.avoidPoints_failed) > 0:
            print(
                f"\t- Points {self.avoidPoints_failed} are avoided b/c at least one of the OFs could not be computed"
            )
        # ---------------------------------------------------------------------------------------------------------------------

        # Update Tabular data with the actual evaluations
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.numpy())

        # Update optimization_results with the actual evaluations
        if not isThisCorrected:
            txt = f"Evaluating points from iteration {self.currentIteration}, comprised of {len(self.x_next)} points"
            predicted, forceWrite, addheader = True, True, True
        else:
            txt = f"Evaluating further points after trust region operation... batch comprised of {len(self.x_next)} points"
            predicted, forceWrite, addheader = False, False, False
        self.optimization_results.addPoints(
            includePoints=[len(self.train_X) - len(self.x_next), len(self.train_X)],
            executed=True,
            predicted=predicted,
            Name=txt,
            forceWrite=forceWrite,
            addheader=addheader,
            timingString=txt_time,
        )

        """
		~~~~~~~~~~~~~~~~~~
		If the optimization step has allowed out-of-bounds points, I should here upgrade my original bounds if the point chosen was out.
		This is not really an option, it must always happen. Otherwise it doesn't make sense to allow extrapolations
		~~~~~~~~~~~~~~~~~~
		"""
        print("\n~~~~~~~~~~~~~~~ Entering bounds upgrade module ~~~~~~~~~~~~~~~~~~~")
        print("(if extrapolations were allowed during optimization)")
        self.bounds = SBOcorrections.upgradeBounds(
            self.bounds, self.train_X, self.avoidPoints_outside
        )
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        )

        # ~~~~~~~~~~~~~~~~~~
        # Possible corrections to modeled & optimization region
        # ~~~~~~~~~~~~~~~~~~

        if not isThisCorrected:
            SBOcorrections.updateMetrics(
                self,
                evaluatedPoints=self.x_next.shape[0],
                position=self.currentIteration,
            )

            changesMade = 0
            # Apply TURBO
            if StrategyOptions_use["TURBO"]:
                changesMade = SBOcorrections.TURBOupdate(
                    self,
                    StrategyOptions_use,
                    position=self.currentIteration,
                    seed=self.seed,
                )
            # Apply some corrections
            if StrategyOptions_use["applyCorrections"] and not ForceNotApplyCorrections:
                changesMade = SBOcorrections.correctionsSet(self, StrategyOptions_use)

            if changesMade > 0:
                print(
                    f"\t~ {changesMade} correction strategies implemented, requesting a new evaluation of metrics in this new region"
                )

                # Get the metrics because I may have changed the index of which one of the trained is the best
                SBOcorrections.updateMetrics(
                    self,
                    IsThisAFreshIteration=False,
                    evaluatedPoints=self.x_next.shape[0],
                    position=self.currentIteration,
                )

        # ~~~~~~~~~~~~~~~~~~
        # Stopping criteria
        # ~~~~~~~~~~~~~~~~~~

        self.evaluateIfConverged()

        return y_next, ystd_next

    def evaluateIfConverged(self):
        hard_finish_residual, hard_finish_variation = False, False

        # ~~~~~~~~~~~~~~~~~~
        # Stopping criterion based on the minimum residual indicated in namelist
        # ~~~~~~~~~~~~~~~~~~

        if self.optimization_options["maximum_value"] is not None:
            print("- Checking residual...")
            _, _, maximization_value = self.lambdaSingleObjective(
                torch.from_numpy(self.train_Y).to(self.dfT)
            )

            best_value_so_far = np.nanmax(maximization_value.cpu().numpy())

            print(
                f'\t- Best scalar function so far (to maximize): {best_value_so_far:.3e} (absolute stopping criterion: {self.optimization_options["maximum_value"]:.3e}), based on original: {self.original_maximum_value:.3e}'
            )

            if best_value_so_far > self.optimization_options["maximum_value"]:
                hard_finish_residual = True

        # ~~~~~~~~~~~~~~~~~~
        # Stopping criterion based on the minimum variation indicated in namelist
        # ~~~~~~~~~~~~~~~~~~

        if self.optimization_options["minimum_dvs_variation"] is not None:
            print("- Checking DV variations...")

            _, yG_max = TESTtools.DVdistanceMetric(self.train_X)

            hard_finish_variation = (
                self.currentIteration
                >= self.optimization_options["minimum_dvs_variation"][0]
                + self.optimization_options["minimum_dvs_variation"][1]
            )
            for i in range(int(self.optimization_options["minimum_dvs_variation"][1])):
                hard_finish_variation = hard_finish_variation and (
                    yG_max[-1 - i] < self.optimization_options["minimum_dvs_variation"][2]
                )

            if hard_finish_variation:
                print(
                    f"\t- DVs varied by less than {self.optimization_options['minimum_dvs_variation'][2]}% compared to the rest of individuals for the past {int(self.optimization_options['minimum_dvs_variation'][1])} iterations"
                )

        # ~~~~~~~~~~~~~~~~~~
        # Final convergence
        # ~~~~~~~~~~~~~~~~~~

        if hard_finish_residual or hard_finish_variation:
            self.hard_finish = self.hard_finish or True
            print("- * Optimization considered converged *", typeMsg="w")

    def initializeOptimization(self):
        print("\n")
        print("------------------------------------------------------------")
        print(" Problem initialization")
        print("------------------------------------------------------------\n")

        # Print Optimization Settings

        print(
            "\n=============================================================================="
        )
        print(f"  {IOtools.getStringFromTime()}, Starting MITIM Optimization")
        print(
            "=============================================================================="
        )

        print(f"* Folder: {self.folderExecution}")
        print("* Optimization Settings:")
        for i in self.optimization_options:
            strs = f"\t\t{i:25}:"
            if i in ["StrategyOptions", "surrogateOptions"]:
                print(strs)
                for j in self.optimization_options[i]:
                    print(f"\t\t\t{j:25}: {self.optimization_options[i][j]}")
            else:
                print(f"{strs} {self.optimization_options[i]}")

        print("* Main Function Parameters:")
        par = self.optimization_object.__dict__
        for i in par:
            if i not in ["optimization_options"]:
                strs = f"\t\t{i:25}:"
                if "file_in_lines_" not in i:
                    print(f"{strs} {par[i]}")
                else:
                    print(f"{strs} NOT PRINTED")

        # --------------------------------------------------------------------------------------------------

        self.Originalinitial_training = copy.deepcopy(self.initial_training)

        # -----------------------------------------------------------------
        # Force certain optimizations depending on existence of folders
        # -----------------------------------------------------------------

        if (not self.restartYN) and (self.optimization_data is not None):
            self.type_initialization = 3
            print(
                "--> Since restart from a previous MITIM has been requested, forcing initialization type to 3 (read from optimization_data)",
                typeMsg="i",
            )

        if self.type_initialization == 3:
            print("--> Initialization by reading tabular data...")

            try:
                tabExists = len(self.optimization_data.data) >= self.initial_training
                print(
                    f"\t- optimization_data file has {len(self.optimization_data.data)} elements, and initial_training were {self.initial_training}"
                )
            except:
                tabExists = False
                print("\n\nCould not read Tabular, because:", typeMsg="w")
                print(traceback.format_exc())

            if not tabExists:
                print(
                    "--> type_initialization 3 requires optimization_data but something failed. Assigning type_initialization=1 and restarting from scratch",
                    typeMsg="i",
                )
                if self.askQuestions:
                    flagger = print("Are you sure?", typeMsg="q")
                    if not flagger:
                        embed()

                self.type_initialization = 1
                self.restartYN = True

        # -----------------------------------------------------------------
        # Initialization
        # -----------------------------------------------------------------

        readCasesFromTabular = (not self.restartYN) and self.optimization_options[
            "read_initial_training_from_csv"
        ]  # Read when starting from previous or forced it

        # Restarted run from previous. Grab DVs of initial set
        if readCasesFromTabular:
            try:
                self.train_X, self.train_Y, self.train_Ystd = self.optimization_data.extract_points(
                    points=np.arange(self.initial_training)
                )

                # It could be the case that those points in Tabular are outside the bounds that I want to apply to this optimization, remove outside points?
                
                if self.optimization_options["ensure_within_bounds"]:
                    for i in range(self.train_X.shape[0]):
                        insideBounds = TESTtools.checkSolutionIsWithinBounds(
                            torch.from_numpy(self.train_X[i, :]).to(self.dfT),
                            torch.from_numpy(np.array(list(self.bounds.values())).T),
                        )
                        if not insideBounds.item():
                            self.avoidPoints_outside.append(i)

            except:
                flagger = print(
                    "Error reading Tabular. Do you want to continue without restart and do standard initialization instead?",
                    typeMsg="q",
                )

                self.type_initialization = 1
                self.restartYN = True
                readCasesFromTabular = False

            if readCasesFromTabular and IOtools.isAnyNan(self.train_X):
                flagger = print(
                    " --> Restart requires non-nan DVs, doing normal initialization",
                    typeMsg="q",
                )
                if not flagger:
                    embed()

                self.type_initialization = 1
                self.restartYN = True
                readCasesFromTabular = False

        # Standard - RUN

        if not readCasesFromTabular:
            if self.type_initialization == 1 and self.optimization_options["dvs_base"] is not None:
                self.initial_training = self.initial_training - 1
                print(
                    f"--> Baseline point has been requested with LHS initialization, reducing requested initial random set to {self.initial_training}",
                    typeMsg="i",
                )

            """
			Initialization
			--------------
			"""

            if self.optimization_options["initialization_fun"] is None:
                if self.type_initialization == 1:
                    if self.initial_training == 0:
                        self.train_X = np.atleast_2d(
                            [i for i in self.optimization_options["dvs_base"]]
                        )
                    else:
                        self.train_X = SAMPLINGtools.LHS(
                            self.initial_training,
                            self.boundsInitialization,
                            seed=self.seed,
                        )
                        self.train_X = self.train_X.cpu().numpy().astype("float")

                        # if (self.optimization_options['dvs_base'] is not None):
                        # 	self.train_X = np.append(np.atleast_2d([i for i in self.optimization_options['dvs_base']]),self.train_X,axis=0)

                elif self.type_initialization == 2:
                    raise Exception("Option not implemented yet")
                elif self.type_initialization == 3:
                    self.train_X = SAMPLINGtools.readInitializationFile(
                        f"{self.folderExecution}/Outputs/optimization_data.csv",
                        self.initial_training,
                        self.stepSettings["optimization_options"]["dvs"],
                    )
                elif self.type_initialization == 4:
                    self.train_X = IOtools.readExecutionParams(
                        self.folderExecution, nums=[0, self.initial_training - 1]
                    )

                if (
                    (self.type_initialization == 1)
                    and (self.optimization_options["dvs_base"] is not None)
                    and (self.initial_training > 0)
                ):
                    self.train_X = np.append(
                        np.atleast_2d([i for i in self.optimization_options["dvs_base"]]),
                        self.train_X,
                        axis=0,
                    )

            else:
                print("- Initialization function has been selected", typeMsg="i")
                self.train_X = self.optimization_options["initialization_fun"](self)
                readCasesFromTabular = True

            # Initialize train_Y as nan until evaluated
            self.train_Y = (
                np.ones((self.Originalinitial_training, len(self.outputs))) * np.nan
            )
            self.train_Ystd = (
                np.ones((self.Originalinitial_training, len(self.outputs))) * np.nan
            )

        # -----------------------------------------------------------------
        # Write prior to evaluation
        # -----------------------------------------------------------------

        # Write initialization in Tabular
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.numpy())

        # Write optimization_results
        self.optimization_results.addPoints(
            includePoints=[0, self.Originalinitial_training],
            executed=False,
            predicted=False,
            Name=f"Initial trust region, comprised of {self.Originalinitial_training} points",
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Evaluate initial training set
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        time1 = datetime.datetime.now()

        self.train_Y, self.train_Ystd, self.numEval = EVALUATORtools.fun(
            self.optimization_object,
            self.train_X,
            self.folderExecution,
            self.bounds,
            self.outputs,
            self.optimization_data,
            parallel=self.parallel_evaluations,
            restartYN=(not readCasesFromTabular),
            numEval=self.numEval,
        )

        txt_time = IOtools.getTimeDifference(time1)
        print(f"\t- Complete model initial training took: {txt_time}\n")
        # ------------------

        # --- If nan for important outputs don't use this point
        for i in range(self.train_Y.shape[0]):
            boole = (np.isinf(self.train_Y[i]).any()) and (
                i not in self.avoidPoints_failed
            )
            if boole:
                self.avoidPoints_failed.append(i)
        if len(self.avoidPoints_failed) > 0:
            print(
                f"\t- Points {self.avoidPoints_failed} are avoided b/c at least one of the OFs could not be computed"
            )
        # ------------------
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.numpy())
        self.optimization_results.addPoints(
            includePoints=[0, self.Originalinitial_training],
            executed=True,
            predicted=False,
            timingString=txt_time,
            Name=f"Initial trust region, comprised of {self.Originalinitial_training} points",
        )

        # Get some metrics about this iteration
        SBOcorrections.updateMetrics(
            self, evaluatedPoints=self.Originalinitial_training, position=0
        )

        # Make sure is 2D
        self.train_X = np.atleast_2d(self.train_X)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Determine relative residuals
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.original_maximum_value = copy.deepcopy(self.optimization_options["maximum_value"])

        if self.optimization_options["maximum_value"] is not None:

            res_base = -self.BOmetrics["overall"]["Residual"][0].item()

            if self.optimization_options["maximum_value_is_rel"]:

                res_new = self.optimization_options["maximum_value"] * res_base
                print(
                    f'\n* Maximum value for MITIM optimization convergence provided as relative value of {self.optimization_options["maximum_value"]} from base {res_base:.3e} --> {res_new:.3e}',
                    typeMsg="i",
                )

                self.stepSettings["optimization_options"]["maximum_value"] = self.optimization_options[
                    "maximum_value"
                ] = res_new
            else:
                print(
                    f'\n* Maximum value for MITIM optimization convergence: {self.optimization_options["maximum_value"]} (starting case has {res_base:.3e})',
                    typeMsg="i",
                )

        """
		Some initialization strategies may create points outside of the original bounds, but I may want to include them!
		"""
        if self.optimization_options["expand_bounds"]:
            for i, ikey in enumerate(self.bounds):
                self.bounds[ikey][0] = np.min(
                    [self.bounds[ikey][0], self.train_X.min(axis=0)[i]]
                )
                self.bounds[ikey][1] = np.max(
                    [self.bounds[ikey][1], self.train_X.max(axis=0)[i]]
                )

    def plot(
        self,
        fn=None,
        plotoptimization_results=True,
        doNotShow=False,
        number_of_models_per_tab=5,
        stds=2,
        pointsEvaluateEachGPdimension=50,
    ):
        print(
            "\n ***************************************************************************"
        )
        print(f"* MITIM plotting module - Generic ({stds}sigma models)")
        print(
            "***************************************************************************\n"
        )

        GPs = self.steps

        if doNotShow:
            plt.ioff()

        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            geometry = (
                "1200x1000" if len(GPs[0].GP["individual_models"]) == 1 else "1700x1000"
            )
            fn = FigureNotebook("MITIM BO Strategy", geometry=geometry)

        """
		****************************************************************
		Model Stuff
		****************************************************************
		"""

        number_of_models_per_tab = np.min([number_of_models_per_tab, len(self.outputs)])
        Tabs_needed = int(np.ceil(len(self.outputs) / number_of_models_per_tab))

        if len(GPs) == 1:
            rangePlot = [0]
        elif len(GPs) == 2:
            rangePlot = range(len(GPs))
        else:
            rangePlot = [len(GPs) - 2, len(GPs) - 1]

        for ck, k in enumerate(rangePlot):
            print(
                f"- Plotting MITIM step #{k} information ({ck+1}/{len(rangePlot)})... {len(self.outputs)} GP models need to be plotted ({pointsEvaluateEachGPdimension} points/dim)..."
            )

            tab_color = ck + 5

            figsFund, figs, figsFundTrain = [], [], []
            for i in range(Tabs_needed):
                figsFund.append(
                    fn.add_figure(
                        label=f"#{k} Fundamental Surr. ({i+1}/{Tabs_needed})",
                        tab_color=tab_color,
                    )
                )
            for i in range(Tabs_needed):
                figsFundTrain.append(
                    fn.add_figure(
                        label=f"#{k} Fundamental Train ({i+1}/{Tabs_needed})",
                        tab_color=tab_color,
                    )
                )
            for i in range(Tabs_needed):
                figs.append(
                    fn.add_figure(
                        label=f"#{k} Surrogate ({i+1}/{Tabs_needed})",
                        tab_color=tab_color,
                    )
                )

            grid = plt.GridSpec(
                nrows=5, ncols=number_of_models_per_tab, hspace=0.5, wspace=0.3
            )
            gridTrain = plt.GridSpec(
                nrows=3, ncols=number_of_models_per_tab, hspace=0.3, wspace=0.3
            )

            for i in range(len(self.outputs)):
                GP = GPs[k].GP["individual_models"][i]

                figIndex = i // number_of_models_per_tab
                figIndex_inner = i % number_of_models_per_tab

                fig = figs[figIndex]
                figFund = figsFund[figIndex]
                figFundTrain = figsFundTrain[figIndex]

                x_next = None
                if "x_next" in GPs[k].__dict__.keys():
                    x_next = GPs[k].x_next
                x_best = self.BOmetrics["xBest_track"][k].unsqueeze(0)

                y_best = self.BOmetrics["yBest_track"][k]
                yVar_best = self.BOmetrics["yVarBest_track"][k]

                # --------------------------------------------------------------------
                # Plotting of models
                # --------------------------------------------------------------------

                # Fundamental Surrogates

                plotFundamental = True

                dimX = GP.gpmodel.ard_num_dims if plotFundamental else len(GP.bounds)
                if dimX == 1:
                    ax0 = figFund.add_subplot(grid[0:4, figIndex_inner])
                    ax1 = None
                    axL = figFund.add_subplot(grid[4, figIndex_inner])
                elif dimX == 2:
                    ax0 = figFund.add_subplot(grid[:2, figIndex_inner])
                    ax1 = figFund.add_subplot(grid[2:4, figIndex_inner], sharex=ax0)
                    axL = figFund.add_subplot(grid[4, figIndex_inner])
                else:
                    ax0 = figFund.add_subplot(grid[:2, figIndex_inner])
                    ax1 = figFund.add_subplot(grid[2:4, figIndex_inner])
                    axL = figFund.add_subplot(grid[4:, figIndex_inner])

                if y_best is not None:
                    y_best1, yVar_best1 = y_best[i], yVar_best[i]
                else:
                    y_best1, yVar_best1 = None, None
                GP.plot(
                    axs=[ax0, ax1, axL],
                    x_next=x_next,
                    x_best=x_best,
                    y_best=y_best1,
                    yVar_best=yVar_best1,
                    plotFundamental=plotFundamental,
                    stds=stds,
                    pointsEvaluate=pointsEvaluateEachGPdimension,
                )

                # Fundamental Surrogates - Training

                relative_to = -1

                ax0 = figFundTrain.add_subplot(gridTrain[0, figIndex_inner])
                ax1 = figFundTrain.add_subplot(gridTrain[1, figIndex_inner], sharex=ax0)
                ax2 = figFundTrain.add_subplot(gridTrain[2, figIndex_inner], sharex=ax0)

                GP.plotTraining(
                    axs=[ax0, ax1, ax2],
                    relative_to=relative_to,
                    figIndex_inner=figIndex_inner,
                    stds=stds,
                )

                # Optimization-relevant variables

                plotFundamental = False

                dimX = GP.dimX if plotFundamental else len(GP.bounds)
                if dimX == 1:
                    ax0 = fig.add_subplot(grid[0:4, figIndex_inner])
                    ax1 = None
                    axL = fig.add_subplot(grid[4, figIndex_inner])
                elif dimX == 2:
                    ax0 = fig.add_subplot(grid[:2, figIndex_inner])
                    ax1 = fig.add_subplot(grid[2:4, figIndex_inner], sharex=ax0)
                    axL = fig.add_subplot(grid[4, figIndex_inner])
                else:
                    ax0 = fig.add_subplot(grid[:2, figIndex_inner])
                    ax1 = fig.add_subplot(grid[2:4, figIndex_inner])
                    axL = fig.add_subplot(grid[4:, figIndex_inner])

                GP.plot(
                    axs=[ax0, ax1, axL],
                    x_next=x_next,
                    x_best=x_best,
                    y_best=y_best1,
                    yVar_best=yVar_best1,
                    plotFundamental=plotFundamental,
                    stds=stds,
                    pointsEvaluate=pointsEvaluateEachGPdimension,
                )

            # Plot model specifics from last model
            self.plotModelStatus(boStep=k, fn=fn, stds=stds, tab_color=tab_color)

        print("- Finished plotting of step models")

        """
		****************************************************************
		Optimization Stuff
		****************************************************************
		"""

        tab_color = ck + 5 + 1

        # ---- Trust region ----------------------------------------------------------
        figTR = fn.add_figure(label="Trust Region", tab_color=tab_color)
        try:
            SBOcorrections.plotTrustRegionInformation(self, fig=figTR)
        except:
            print("\t- Problem plotting trust region", typeMsg="w")

        # ---- optimization_results ---------------------------------------------------
        if plotoptimization_results:
            # Most current state of the optimization_results.out
            self.optimization_results.read()
            if "logFile" in self.__dict__.keys():
                logFile = self.logFile
            else:
                logFile = None
            self.optimization_results.plot(
                fn=fn, doNotShow=True, log=logFile, tab_color=tab_color
            )

        return fn

    def plotModelStatus(
        self, fn=None, boStep=-1, plotsPerFigure=20, stds=2, tab_color=None
    ):
        step = self.steps[boStep]

        GP = step.GP["combined_model"]

        # ---- Jacobian -------------------------------------------------------
        fig = fn.add_figure(label=f"#{boStep}: Jacobian", tab_color=tab_color)
        maxPoints = 1  # 4
        xExplore = []
        if "x_next" in step.__dict__.keys():
            for i in range(np.min([step.x_next.shape[0], maxPoints])):
                xExplore.append(step.x_next[i].cpu().numpy())
        else:
            xExplore.append(step.train_X[0])

        axs = GRAPHICStools.producePlotsGrid(
            len(xExplore), fig=fig, hspace=0.3, wspace=0.3
        )

        for i in range(len(xExplore)):
            GP.localBehavior(
                torch.from_numpy(xExplore[i]),
                prefix=f"Next #{i+1}\n",
                outputs=self.outputs,
                ax=axs[i],
            )

        # ---- Training quality -------------------------------------------------------
        x_next = step.x_next if "x_next" in step.__dict__.keys() else None
        y_next, ystd_next = (
            (step.y_next, step.ystd_next)
            if "y_next" in step.__dict__.keys()
            else (None, None)
        )

        numGPs = GP.train_Y.shape[1]
        numfigs = int(np.ceil(numGPs / plotsPerFigure))

        figsQuality = []
        for i in range(numfigs):
            figsQuality.append(
                fn.add_figure(
                    label=f"#{boStep}: Quality {i+1}/{numfigs}", tab_color=tab_color
                )
            )

        axs = GP.testTraining(
            plotYN=True,
            figs=figsQuality,
            x_next=x_next,
            y_next=y_next,
            ystd_next=ystd_next,
            plotsPerFigure=plotsPerFigure,
            ylabels=self.outputs,
            stds=stds,
        )

        if axs is not None:
            for i in range(len(self.outputs)):
                # axs[i].set_title(self.outputs[i])
                GRAPHICStools.addDenseAxis(axs[i])

        # ---- Optimization Performance ---------------------------------------
        if "InfoOptimization" not in step.__dict__.keys():
            return

        figOPT1 = fn.add_figure(label=f"#{boStep}: Optim. Perfom.", tab_color=tab_color)
        figOPT2 = fn.add_figure(label=f"#{boStep}: Optim. Ranges", tab_color=tab_color)
        self.plotSurrogateOptimization(fig1=figOPT1, fig2=figOPT2, boStep=boStep)
        # ---------------------------------------------------------------------

    def plotSurrogateOptimization(self, fig1=None, fig2=None, boStep=-1):
        # ----------------------------------------------------------------------
        # Select information
        # ----------------------------------------------------------------------

        step = self.steps[boStep]
        info, boundsRaw = step.InfoOptimization, step.bounds

        bounds = torch.Tensor([boundsRaw[b] for b in boundsRaw])
        boundsThis = (
            info[0]["bounds"].numpy().transpose(1, 0) if "bounds" in info[0] else None
        )

        # ----------------------------------------------------------------------
        # Prep figures
        # ----------------------------------------------------------------------

        colors = GRAPHICStools.listColors()

        if fig1 is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            fn = FigureNotebook("PRF BO Strategy", geometry="1700x1000")
            fig2 = fn.add_figure(label=f"#{boStep}: optimization_options Ranges")
            fig1 = fn.add_figure(label=f"#{boStep}: optimization_options Perfom.")

        grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.2, wspace=0.2)
        ax0_r = fig2.add_subplot(grid[:, 0])
        ax1_r = fig2.add_subplot(grid[0, 1])
        ax2_r = fig2.add_subplot(grid[1, 1])

        # Get dimensions and prepare figures
        num_x = step.InfoOptimization[0]["info"]["x_start"].shape[1]
        num_y = step.InfoOptimization[0]["info"]["yFun_start"].shape[1]

        num_axes_x, num_axes_y, num_axes_res = (
            int(np.ceil(num_x / 2)),
            int(np.ceil(num_y / 2)),
            1,
        )

        if num_axes_x == 0:
            num_axes_x += 1
        if num_axes_y == 0:
            num_axes_y += 1

        num_plots = num_axes_x + num_axes_y + num_axes_res
        axs = GRAPHICStools.producePlotsGrid(
            num_plots, fig=fig1, hspace=0.4, wspace=0.4
        )

        axsDVs = axs[:num_axes_x]
        axsOFs = axs[num_axes_x:-1]
        axR = [axs[-1]]

        axislabels = [i for i in boundsRaw]

        # ----------------------------------------------------------------------
        # Plot DVs and OFs - Training
        # ----------------------------------------------------------------------

        it_start = 0

        iinfo = info[0]["info"]
        it_start, xypair = OPTtools.plotInfo(
            iinfo,
            label="Training",
            plotStart=True,
            xypair=[],
            axTraj=ax0_r,
            axDVs_r=ax1_r,
            axOFs_r=ax2_r,
            axDVs=axsDVs,
            axOFs=axsOFs,
            axR=axR,
            bounds=bounds,
            boundsThis=boundsThis,
            axislabels_x=axislabels,
            axislabels_y=self.optimization_object.name_objectives,
            color="k",
            ms=10,
            alpha=0.5,
            it_start=it_start,
        )

        # Loop over posterior steps
        for ipost in range(len(info) - 1):
            iinfo = info[ipost]["info"]
            it_start, xypair = OPTtools.plotInfo(
                iinfo,
                label=info[ipost]["method"],
                plotStart=False,
                xypair=xypair,
                axTraj=ax0_r,
                axDVs_r=ax1_r,
                axOFs_r=ax2_r,
                axDVs=axsDVs,
                axOFs=axsOFs,
                axR=axR,
                axislabels_x=axislabels,
                axislabels_y=self.optimization_object.name_objectives,
                color=colors[ipost],
                ms=8 - ipost * 1.5,
                alpha=0.5,
                it_start=it_start,
            )

        xypair = np.array(xypair)

        loga = True if xypair[:, 1].min() > 0 else False

        axsDVs[0].legend(prop={"size": 5})
        if loga:
            for p in range(len(axsOFs)):
                axsOFs[p].set_xscale("log")
                axsOFs[p].set_yscale("log")

        ax1_r.set_ylabel("DV values")
        GRAPHICStools.addDenseAxis(ax1_r)
        GRAPHICStools.autoscale_y(ax1_r)

        ax2_r.set_ylabel("Residual values")
        if loga:
            ax2_r.set_yscale("log")
        GRAPHICStools.addDenseAxis(ax2_r)
        GRAPHICStools.autoscale_y(ax2_r)

        ax0_r.plot(xypair[:, 0], xypair[:, 1], "-s", markersize=5, lw=2.0, c="k")

        iinfo = info[-1]["info"]
        for i, y in enumerate(iinfo["y_res"]):
            ax0_r.axhline(
                y=-y,
                c=colors[ipost + 1],
                ls="--",
                lw=2,
                label=info[-1]["method"] if i == 0 else "",
            )
        iinfo = info[0]["info"]
        ax0_r.axhline(y=-iinfo["y_res_start"][0], c="k", ls="--", lw=2)

        ax0_r.set_xlabel("Optimization iterations")
        ax0_r.set_ylabel("$-f_{acq}$")
        GRAPHICStools.addDenseAxis(ax0_r)
        if loga:
            ax0_r.set_yscale("log")
        ax0_r.legend(loc="best", prop={"size": 8})
        ax0_r.set_title("Evolution of acquisition in optimization stages")

        for i in range(len(axs)):
            GRAPHICStools.addDenseAxis(axs[i])


def read_from_scratch(file):
    """
    This reads a pickle file for the entire class
    """

    optimization_object = opt_evaluator(None)
    prf = PRF_BO(optimization_object, onlyInitialize=True, askQuestions=False)
    prf = prf.read(file=file, iteration=-1, provideFullClass=True)

    return prf


def avoidClassInitialization(folderWork):
    print(
        "It was requested that I try read the class before I initialize and select parameters...",
        typeMsg="i",
    )

    try:
        with open(
            f"{IOtools.expandPath(folderWork)}/Outputs/optimization_object.pkl", "rb"
        ) as handle:
            aux = pickle_dill.load(handle)
        opt_fun = aux.optimization_object
        restart = False
        print("\t- Restart was successful", typeMsg="i")
    except:
        opt_fun = None
        restart = True
        flagger = print("\t- Restart was requested but it didnt work (c)", typeMsg="q")
        if not flagger:
            embed()

    return opt_fun, restart


"""
To load pickled GPU-cuda classes on a CPU machine
From:
	https://github.com/pytorch/pytorch/issues/16797
	https://stackoverflow.com/questions/35879096/pickle-unpicklingerror-could-not-find-mark
"""


class CPU_Unpickler(pickle_dill.Unpickler):
    def find_class(self, module, name):
        import io

        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
