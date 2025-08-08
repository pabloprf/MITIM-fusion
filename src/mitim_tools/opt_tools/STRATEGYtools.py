import sys
import copy
import datetime
import array
import traceback
import torch
from pathlib import Path
from collections import OrderedDict
from IPython import embed
import dill as pickle_dill
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools, LOGtools
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.opt_tools import OPTtools, STEPtools
from mitim_tools.opt_tools.utils import (
    BOgraphics,
    SBOcorrections,
    TESTtools,
    EVALUATORtools,
    SAMPLINGtools,
)
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__

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

	MITIM_BO = STRATEGYtools.MITIM_BO(optimization_object)
	MITIM_BO.run()

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
        default_namelist_function=None,
        tensor_options = {
            "dtype": torch.double,
            "device": torch.device("cpu"),
        }
    ):
        """
        Namelist file can be provided and will be copied to the folder
        """

        self.tensor_options = tensor_options

        print("- Parent opt_evaluator function initialized")

        self.folder = folder

        print(f"\t- Folder: {self.folder}")

        if self.folder is not None:

            self.folder = IOtools.expandPath(self.folder)
            if not self.folder.exists():
                IOtools.askNewFolder(self.folder)
            if not (self.folder / "Outputs").exists():
                IOtools.askNewFolder(self.folder / "Outputs")

        if namelist is not None:
            print(f"\t- Namelist provided: {namelist}", typeMsg="i")

            self.optimization_options = IOtools.read_mitim_nml(namelist)

        elif default_namelist_function is not None:
            print("\t- Namelist not provided, using MITIM default for this optimization sub-module", typeMsg="i")

            namelist = __mitimroot__ / "templates" / "main.namelist.json"
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
            "surrogate_transformation_variables_alltimes": None,
            "surrogate_transformation_variables_lasttime": None,
            "transformationInputs": STEPtools.identity,  # Transformation of inputs
            "transformationOutputs": STEPtools.identityOutputs,  # Transformation of outputs
        }

        # Determine type of tensors to work with
        torch.set_default_dtype(self.tensor_options["dtype"])  # In case I forgot to specify a type explicitly, use as default (https://github.com/pytorch/botorch/discussions/1444)
        self.dfT = torch.randn( (2, 2), **tensor_options)

        # Name of calibrated objectives (e.g. QiRes1 to represent the objective from Qi1-QiT1)
        self.name_objectives = None

        # Name of transformed functions (e.g. Qi1_GB to represent the transformation of Qi1)
        self.name_transformed_ofs = None

        # Variables in the class not to save (e.g. COMSOL model)
        self.doNotSaveVariables = []

    def read(self, paramsfile, resultsfile):
        # Read stuff
        FolderEvaluation,numEval,inputFilePath,outputFilePath = IOtools.obtainGeneralParams(paramsfile, resultsfile)
        
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
        plotFN=None,
        folderRemote=None,
        analysis_level=0,
        pointsEvaluateEachGPdimension=50,
        rangePlot=None,
    ):
        with np.errstate(all="ignore"):
            LOGtools.ignoreWarnings()
            (
                self.fn,
                self.res,
                self.mitim_model,
                self.data,
            ) = BOgraphics.retrieveResults(
                self.folder,
                analysis_level=analysis_level,
                doNotShow=True,
                plotFN=plotFN,
                folderRemote=folderRemote,
                pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
                rangePlot=rangePlot,
            )

        # Make folders local
        try:
            self.mitim_model.folderOutputs = Path(str(self.mitim_model.folderOutputs).replace(
                str(self.mitim_model.folderExecution), str(self.folder)
            ))
            self.mitim_model.optimization_extra = self.mitim_model.optimization_object.optimization_extra = (
                Path(str(self.mitim_model.optimization_extra).replace(
                    str(self.mitim_model.folderExecution), str(self.folder)
                ))
            )
            self.mitim_model.folderExecution = self.mitim_model.optimization_object.folder = (
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
            self_complete = self.mitim_model.optimization_object
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
        rangesPlot=None,
        save_folder=None,
        tabs_colors=0,
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

        if plotYN and (analysis_level >= 0):
            if "fn" not in self.__dict__:
                self.fn = GUItools.FigureNotebook("MITIM Optimization Results")
            
        self.read_optimization_results(
            plotFN=self.fn if (plotYN and (analysis_level >= 0)) else None,
            folderRemote=folderRemote,
            analysis_level= retrieval_level if (retrieval_level is not None) else analysis_level,
            pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
            rangePlot=rangesPlot,
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
                class_name = str(self.mitim_model.optimization_object).split()[0].split(".")[-1]
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
class MITIM_BO:
    def __init__(
        self,
        optimization_object,
        cold_start=False,
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
                - cold_start 	 :  If False, try to find the values from Outputs/optimization_data.csv
                - storeClass 	 :  If True, write a class pickle for well-behaved cold_starting
                - askQuestions 	 :  To avoid that a SLURM job gets stop becuase something is asked, set to False
        """

        self.optimization_object = optimization_object
        self.cold_start = cold_start
        self.storeClass = storeClass
        self.askQuestions = askQuestions
        self.seed = seed
        self.avoidPoints = []

        if self.optimization_object.name_objectives is None:
            self.optimization_object.name_objectives = "y"

        # Folders and Logger
        self.folderExecution = (
            IOtools.expandPath(self.optimization_object.folder)
            if (self.optimization_object.folder is not None)
            else Path("")
        )

        self.folderOutputs = self.folderExecution / "Outputs"

        if (not self.cold_start) and askQuestions:
            
            # Check if Outputs folder is empty (if it's empty, do not ask the user, just continue)
            if self.folderOutputs.exists() and (len(list(self.folderOutputs.iterdir())) > 0):
                if not print(f"\t* Because {cold_start = }, MITIM will try to read existing results from folder",typeMsg="q"):
                    raise Exception("[MITIM] - User requested to stop")

        if optimization_object.optimization_options is not None:
            if not self.folderOutputs.exists():
                IOtools.askNewFolder(self.folderOutputs, force=True)

            """
			Prepare class where I will store some extra data
			---
			Do not carry out this dictionary through the workflow, just read and write
			"""

            self.optimization_extra = self.folderOutputs / "optimization_extra.pkl"

            # Read if exists
            exists = False
            if self.optimization_extra.exists():
                try:
                    dictStore = IOtools.unpickle_mitim(self.optimization_extra)
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
                pickle_dill.dump(dictStore, handle, protocol=4)

            # Write the class into the optimization_object
            optimization_object.optimization_extra = self.optimization_extra

        # Function to execute
        self.surrogate_parameters = self.optimization_object.surrogate_parameters
        self.optimization_options = self.optimization_object.optimization_options

        # Curate namelist ---------------------------------------------------------------------------------
        if self.optimization_options is not None:
            self.optimization_options = IOtools.curate_mitim_nml(
                self.optimization_options,
                stopping_criteria_default = stopping_criteria_default
                )
        # -------------------------------------------------------------------------------------------------

        if not onlyInitialize:
            print("\n-----------------------------------------------------------------------------------------")
            print("\t\t\t BO class module")
            print("-----------------------------------------------------------------------------------------\n")

            """
			------------------------------------------------------------------------------
			Grab variables
			------------------------------------------------------------------------------
			"""
   
            self.timings_file = self.folderOutputs / "timing.jsonl"

            # Logger
            sys.stdout = LOGtools.Logger(logFile=self.folderOutputs / "optimization_log.txt", writeAlsoTerminal=True)

            # Meta
            self.numIterations = self.optimization_options["convergence_options"]["maximum_iterations"]
            self.strategy_options = self.optimization_options["strategy_options"]
            self.parallel_evaluations = self.optimization_options["evaluation_options"]["parallel_evaluations"]
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
                for i in range(self.optimization_options["convergence_options"]["maximum_iterations"] + 1):
                    self.BOmetrics[ikey][i] = np.nan

            """
			------------------------------------------------------------------------------
			Prepare Desgin variables (DVs) with minimum and maximum values
			------------------------------------------------------------------------------
			"""

            self.bounds, self.boundsInitialization = OrderedDict(), []
            for cont, i in enumerate(self.optimization_options["problem_options"]["dvs"]):
                self.bounds[i] = np.array([self.optimization_options["problem_options"]["dvs_min"][cont], self.optimization_options["problem_options"]["dvs_max"][cont]])
                self.boundsInitialization.append(np.array([self.optimization_options["problem_options"]["dvs_min"][cont], self.optimization_options["problem_options"]["dvs_max"][cont]]))

            self.boundsInitialization = np.transpose(self.boundsInitialization)

            # Bounds may change during the workflow (corrections, TURBO)
            self.bounds_orig = copy.deepcopy(self.bounds)

            """
			----------------------------------------------------------------------------------------------------------------------------
			Prepare Objective functions (OFs) with minimum and maximum values
			----------------------------------------------------------------------------------------------------------------------------
			"""

            # Objective functions (OFs)
            self.outputs = self.surrogate_parameters["outputs"] = self.optimization_options["problem_options"]["ofs"]

            # How many points each iteration will produce?
            self.best_points_sequence = self.optimization_options["acquisition_options"]["points_per_step"]
            self.best_points = int(np.sum(self.best_points_sequence))

            """
			------------------------------------------------------------------------------
			Prepare Initialization
			------------------------------------------------------------------------------
			"""

            if (
                (self.optimization_options["initialization_options"]["type_initialization"] == 1)
                and ((self.folderExecution / "Execution" / "Evaluation.1").exists())
                and (self.cold_start)
            ):
                print("\t--> Random initialization has been requested",typeMsg="q" if self.askQuestions else "qa")

            self.type_initialization = self.optimization_options["initialization_options"]["type_initialization"]
            self.initial_training = self.optimization_options["initialization_options"]["initial_training"]

            """
			------------------------------------------------------------------------------
			Initialize Output files
			------------------------------------------------------------------------------
			"""

            if (self.type_initialization == 3) and (self.cold_start):
                print("\t* Initialization based on Tabular, yet cold_start has been requested. I am NOT removing the previous optimization_data",typeMsg="w")
                if self.askQuestions:
                    flagger = print("\t\t* Are you sure this was your intention?", typeMsg="q")
                    if not flagger:
                        embed()
                forceNewTabulars = False
            else:
                forceNewTabulars = self.cold_start

            inputs = [i for i in self.bounds]

            self.scalarized_objective = self.optimization_object.scalarized_objective

            self.optimization_data = BOgraphics.optimization_data(
                inputs,
                self.outputs,
                file=self.folderOutputs / "optimization_data.csv",
                forceNew=forceNewTabulars,
            )

            # If the file turned out to be empty, I will force it to be new
            if forceNewTabulars and (len(self.optimization_data.data) == 0):
                print("\t* Tabular file is empty, forcing new, to avoid radii/channel specifications from dummy sims",typeMsg="w")
                self.optimization_data = BOgraphics.optimization_data(
                    inputs,
                    self.outputs,
                    file=self.folderOutputs / "optimization_data.csv",
                    forceNew=True,
                )

            res_file = self.folderOutputs / "optimization_results.out"

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

        self.currentIteration = -1

        # Has the problem reached convergence in the training?
        converged,_ = self.optimization_options['convergence_options']['stopping_criteria'](self, parameters = self.optimization_options['convergence_options']['stopping_criteria_parameters'])
        if converged:
            print("- Optimization has converged in training!",typeMsg="i")
            self.numIterations = 0

        # If no iterations are requested, just run the training step
        if self.numIterations == 0:
            print("- No BO iterations requested, workflow will stop after running a training step (to enable reading later)",typeMsg="i")
            self.numIterations = 1
            self.hard_finish = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Iterative workflow
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.strategy_options_use = self.strategy_options

        self.steps, self.resultsSet = [], []
        for self.currentIteration in range(self.numIterations+1):
            timeBeginningThis = datetime.datetime.now()

            print("\n------------------------------------------------------------")
            print(f'\tMITIM Step {self.currentIteration} ({timeBeginningThis.strftime("%Y-%m-%d %H:%M:%S")})')
            print("------------------------------------------------------------")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~ Update training population with next points
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if self.currentIteration > 0:
                print(f"--> Proceeding to updating set (which currently has {len(self.train_X)} points)")

                # *NOTE*: self.x_next has been updated earlier, either from a cold_start or from the full workflow
                yN, yNstd = self.updateSet(self.strategy_options_use)

                # Stored in previous step
                self.steps[-1].y_next = yN
                self.steps[-1].ystd_next = yNstd

                # Determine here when to stop the loop
                if self.currentIteration > self.numIterations - 1:
                    print("- Last iteration has been reached",typeMsg="i")
                    self.hard_finish = True

            # After evaluating metrics inside updateSet, I may have requested a hard finish
            if self.hard_finish:
                print("- Hard finish has been requested", typeMsg="i")

                # Removing those spaces in the metrics that were not filled up
                for ikey in self.keys_metrics:
                    for i in range(self.currentIteration + 1, self.optimization_options["convergence_options"]["maximum_iterations"] + 1):
                        del self.BOmetrics[ikey][i]
                # ------------------------------------------------------------------------------------------

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~ Perform BO step
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Does the tabular file include at least as many rows as requested to be run in this step?
            pointsTabular = len(self.optimization_data.data)  # Number of points that the Tabular contains
            pointsExpected = len(self.train_X) + self.best_points
            if not self.cold_start:
                if not pointsTabular >= pointsExpected:
                    print(f"--> CSV file does not contain information for all points ({pointsTabular}/{pointsExpected}), disabling cold_starting-from-previous from this point on",typeMsg="w", )
                    self.cold_start = True
                else:
                    print(f"--> CSV file contains at least as many points as expected at this stage ({pointsTabular}/{pointsExpected})",typeMsg="i",)

            # In the case of starting from previous, do not run BO process.
            if not self.cold_start:
                """
                Philosophy of cold_starting is:
                        - cold_starting requires that settings are the same (e.g. best_points).
                        - cold_starting requires Tabular data, since I will be grabbing from it.
                        - The pkl file in cold_starting is only used to store the "step" data with opt info, but not
                                required to continue. But here I enforce it anyway. I want to know what I have done.
                        - Later I pass the x_next from the pickle, so I could just trust the pickle if I wanted.
                """

                # Read step from pkl
                current_step = self.read()

                if current_step is None:
                    print("\t* Because reading pkl step had problems, disabling cold_starting-from-previous from this point on",typeMsg="w")
                    print("\t* Are you aware of the consequences of continuing?",typeMsg="q")

                    self.cold_start = True

            if not self.cold_start:
                # Read next from Tabular
                self.x_next, _, _ = self.optimization_data.extract_points(points=np.arange(len(self.train_X), len(self.train_X) + self.best_points))
                self.x_next = torch.from_numpy(self.x_next).to(self.dfT)

                # Re-write x_next from the pkl... reason for this is that if optimization is heuristic, I may prefer what was in Tabular
                if current_step is not None:
                    current_step.x_next = self.x_next

                # If there is any Nan, assume that I cannot cold_start this step
                if IOtools.isAnyNan(self.x_next.cpu()):
                    print("\t* Because x_next points have NaNs, disabling cold_starting-from-previous from this point on",typeMsg="w")
                    self.cold_start = True

                # Step is valid, append to this current one
                if not self.cold_start:
                    self.steps.append(current_step)

                # When cold_starting, make sure that the strategy options are preserved (like correction, bounds and TURBO)
                self.strategy_options_use = current_step.strategy_options_use

                print("\t* Step successfully restarted from pkl file", typeMsg="i")

            # Standard (i.e. start from beginning, not read values)
            if self.cold_start:
                # For standard use, use the actual strategy_options launched
                self.strategy_options_use = self.strategy_options

                # Remove from tabular next points in case they were there. Since I'm not cold_starting, I don't care about what has come next
                self.optimization_data.removePointsAfter(len(self.train_X) - 1)

                """
				---------------------------------------------------------------------------------------
				BOstep is in charge to fit models and optimize objective function
							(inputs and returns are unnormalized)
				---------------------------------------------------------------------------------------
				"""

                self._step()


            # Pass the information about next step
            self.x_next = self.steps[-1].x_next

            # ~~~~~~~~ Store class now with the next points found (after optimization)
            if self.storeClass and self.cold_start:
                self.save()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.hard_finish:
                break

        self.save()

        print(f"- Complete MITIM workflow took {IOtools.getTimeDifference(timeBeginning)} ~~")
        print("********************************************************\n")

    def prepare_for_save_MITIMBO(self, copyClass):
        """
        Downselect what elements to store
        """

        # -------------------------------------------------------------------------------------------------
        # To avoid circularity when cold_starting, do not store the class in the optimization_results sub-class
        # -------------------------------------------------------------------------------------------------

        del copyClass.optimization_results.MITIM_BO

        # -------------------------------------------------------------------------------------------------
        # Saving state files with functions is very expensive (deprecated maybe when I had lambdas?) [TODO: Remove]
        # -------------------------------------------------------------------------------------------------

        del copyClass.scalarized_objective

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

    def prepare_for_read_MITIMBO(self, copyClass):
        """
        Repair the downselection
        """

        copyClass.optimization_results.MITIM_BO = copy.deepcopy(self)

        copyClass.scalarized_objective = copyClass.optimization_object.scalarized_objective

        for i in range(len(copyClass.steps)):
            copyClass.steps[i].defineFunctions(copyClass.scalarized_objective)

        return copyClass

    def save(self, name="optimization_object.pkl"):
        print("* Proceeding to save new MITIM state pickle file")
        stateFile = self.folderOutputs / f"{name}"
        stateFile_tmp = self.folderOutputs / f"{name}_tmp"

        # Do not store certain variables (that cannot even be copied, that's why I do it here)
        saver = {}
        for ikey in self.optimization_object.doNotSaveVariables:
            saver[ikey] = self.optimization_object.__dict__[ikey]
            del self.optimization_object.__dict__[ikey]
        # -----------------------------------------------------------------------------------

        copyClass = self.prepare_for_save_MITIMBO(copy.deepcopy(self))

        with open(stateFile_tmp, "wb") as handle:
            try:
                pickle_dill.dump(copyClass, handle, protocol=4)
            except:
                print(f"\t* Problem saving {name}, trying without the optimization_object, but that will lead to limiting applications. I recommend you populate self.optimization_object.doNotSaveVariables = ['variable1', 'variable2'] with the variables you think cannot be pickled", typeMsg="w")
                del copyClass.optimization_object
                pickle_dill.dump(copyClass, handle, protocol=4)

        # Get variables back ----------------------------------------------------------------
        for ikey in saver:
            self.optimization_object.__dict__[ikey] = saver[ikey]
        # -----------------------------------------------------------------------------------

        stateFile_tmp.replace(stateFile)  # This way I reduce the risk of getting a mid-creation file

        print(f"\t- MITIM state file {IOtools.clipstr(stateFile)} generated, containing the MITIM_BO class")

    def read(self, name="optimization_object.pkl", iteration=None, file=None, provideFullClass=False):
        iteration = iteration or self.currentIteration

        print("- Reading pickle file with optimization_object class")
        stateFile = file if (file is not None) else self.folderOutputs / f"{name}"

        try:
            # If I don't create an Individual attribute I cannot unpickle GA information
            try:
                import deap
                deap.creator.create("Individual", array.array)
            except:
                pass

            aux = IOtools.unpickle_mitim(stateFile)

            aux = self.prepare_for_read_MITIMBO(aux)

            step = aux.steps[iteration]
            print(f"\t* Read {IOtools.clipstr(stateFile)} state file, grabbed step #{iteration}",typeMsg="i")
        
        except FileNotFoundError:
            print(f"\t- State file {IOtools.clipstr(stateFile)} not found", typeMsg="w")
            step, aux = None, None
        
        except IndexError:
            print(f"\t- State file {IOtools.clipstr(stateFile)} does not have all iterations required to continue from it", typeMsg="w")
            step = None

        return aux if provideFullClass else step

    # Convenient helper methods to track timings of components

    @mitim_timer(lambda self: f'Eval @ {self.currentIteration}', log_file=lambda self: self.timings_file)
    def _evaluate(self):
        
        y_next, ystd_next, self.numEval = EVALUATORtools.fun(
            self.optimization_object,
            self.x_next,
            self.folderExecution,
            self.bounds,
            self.outputs,
            self.optimization_data,
            parallel=self.parallel_evaluations,
            cold_start=self.cold_start,
            numEval=self.numEval,
        )
        
        return y_next, ystd_next
    
    @mitim_timer(lambda self: f'Surr @ {self.currentIteration}', log_file=lambda self: self.timings_file)
    def _step(self):
        
        train_Ystd = self.train_Ystd if (self.optimization_options["evaluation_options"]["train_Ystd"] is None) else self.optimization_options["evaluation_options"]["train_Ystd"]
        
        current_step = STEPtools.OPTstep(
            self.train_X,
            self.train_Y,
            train_Ystd,
            bounds=self.bounds,
            stepSettings=self.stepSettings,
            currentIteration=self.currentIteration,
            strategy_options=self.strategy_options_use,
            BOmetrics=self.BOmetrics,
            surrogate_parameters=self.surrogate_parameters,
        )

        # Incorporate strategy_options for later retrieving
        current_step.strategy_options_use = copy.deepcopy(self.strategy_options_use)

        self.steps.append(current_step)

        # Avoid points
        avoidPoints = np.append(self.avoidPoints_failed, self.avoidPoints_outside)
        self.avoidPoints = np.unique([int(j) for j in avoidPoints])

        # ***** Fit
        self.steps[-1].fit_step(avoidPoints=self.avoidPoints)

        # ***** Define evaluators
        self.steps[-1].defineFunctions(self.scalarized_objective)

        # Store class with the model fitted and evaluators defined
        if self.storeClass:
            self.save()

        # ***** Optimize
        if not self.hard_finish:
            self.steps[-1].optimize(
                position_best_so_far=self.BOmetrics["overall"]["indBest"],
                seed=self.seed,
            )
        else:
            self.steps[-1].x_next = None
        
    # ---------------------------------------------------------------------------------


    def updateSet(
        self, strategy_options_use, isThisCorrected=False, ForceNotApplyCorrections=False
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
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())

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
        y_next, ystd_next = self._evaluate()
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
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())

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
        self.bounds = SBOcorrections.upgradeBounds(self.bounds, self.train_X, self.avoidPoints_outside)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

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
            if strategy_options_use["TURBO_options"]["apply"]:
                changesMade = SBOcorrections.TURBOupdate(
                    self,
                    strategy_options_use,
                    position=self.currentIteration,
                    seed=self.seed,
                )
            # Apply some corrections
            if strategy_options_use["applyCorrections"] and not ForceNotApplyCorrections:
                changesMade = SBOcorrections.correctionsSet(self, strategy_options_use)

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

        converged,_ = self.optimization_options['convergence_options']['stopping_criteria'](self, parameters = self.optimization_options['convergence_options']['stopping_criteria_parameters'])

        if converged:
            self.hard_finish = self.hard_finish or True
            print("- * Optimization considered converged *", typeMsg="w")

        return y_next, ystd_next

    @mitim_timer(lambda self: f'Init', log_file=lambda self: self.timings_file)
    def initializeOptimization(self):
        print("\n")
        print("------------------------------------------------------------")
        print(" Problem initialization")
        print("------------------------------------------------------------\n")

        # Print Optimization Settings

        print("\n==============================================================================")
        print(f"  {IOtools.getStringFromTime()}, Starting MITIM Optimization")
        print("==============================================================================")

        print(f"* Folder: {self.folderExecution}")
        print("* Optimization Settings:")
        for i in self.optimization_options:
            strs = f"\t\t{i:25}:"
            if i in ["strategy_options", "surrogate_options"]:
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

        if (not self.cold_start) and (self.optimization_data is not None):
            self.type_initialization = 3
            print("--> Since restart from a previous MITIM has been requested, forcing initialization type to 3 (read from optimization_data)",typeMsg="i",)

        if self.type_initialization == 3:
            print("--> Initialization by reading tabular data...")

            try:
                tabExists = len(self.optimization_data.data) >= self.initial_training
                print(f"\t- optimization_data file has {len(self.optimization_data.data)} elements, and initial_training were {self.initial_training}")
            except:
                tabExists = False
                print("\n\nCould not read Tabular, because:", typeMsg="w")
                print(traceback.format_exc())

            if not tabExists:
                print("--> type_initialization 3 requires optimization_data but something failed. Assigning type_initialization=1 and cold_starting from scratch",typeMsg="i",)
                if self.askQuestions:
                    flagger = print("Are you sure?", typeMsg="q")
                    if not flagger:
                        embed()

                self.type_initialization = 1
                self.cold_start = True

        # -----------------------------------------------------------------
        # Initialization
        # -----------------------------------------------------------------

        readCasesFromTabular = (not self.cold_start) or self.optimization_options["initialization_options"]["read_initial_training_from_csv"]  # Read when starting from previous or forced it

        # cold_started run from previous. Grab DVs of initial set
        if readCasesFromTabular:
            try:
                self.train_X, self.train_Y, self.train_Ystd = self.optimization_data.extract_points(points=np.arange(self.initial_training))

                # It could be the case that those points in Tabular are outside the bounds that I want to apply to this optimization, remove outside points?
                
                if self.optimization_options["initialization_options"]["ensure_within_bounds"]:
                    for i in range(self.train_X.shape[0]):
                        insideBounds = TESTtools.checkSolutionIsWithinBounds(
                            torch.from_numpy(self.train_X[i, :]).to(self.dfT),
                            torch.from_numpy(np.array(list(self.bounds.values())).T),
                        )
                        if not insideBounds.item():
                            self.avoidPoints_outside.append(i)

            except:
                flagger = print("Error reading Tabular. Do you want to continue without cold_start and do standard initialization instead?",typeMsg="q",)

                self.type_initialization = 1
                self.cold_start = True
                readCasesFromTabular = False

            if readCasesFromTabular and IOtools.isAnyNan(self.train_X):
                flagger = print(" --> cold_start requires non-nan DVs, doing normal initialization",typeMsg="q",)
                if not flagger:
                    embed()

                self.type_initialization = 1
                self.cold_start = True
                readCasesFromTabular = False

        # Standard - RUN

        if not readCasesFromTabular:
            if self.type_initialization == 1 and self.optimization_options["problem_options"]["dvs_base"] is not None:
                self.initial_training = self.initial_training - 1
                print(f"--> Baseline point has been requested with LHS initialization, reducing requested initial random set to {self.initial_training}",typeMsg="i",)

            """
			Initialization
			--------------
			"""

            if self.optimization_options["initialization_options"]["initialization_fun"] is None:
                if self.type_initialization == 1:
                    if self.initial_training == 0:
                        self.train_X = np.atleast_2d(
                            [i for i in self.optimization_options["problem_options"]["dvs_base"]]
                        )
                    else:
                        self.train_X = SAMPLINGtools.LHS(
                            self.initial_training,
                            self.boundsInitialization,
                            seed=self.seed,
                        )
                        self.train_X = self.train_X.cpu().numpy().astype("float")

                        # if (self.optimization_options['problem_options']['dvs_base'] is not None):
                        # 	self.train_X = np.append(np.atleast_2d([i for i in self.optimization_options['problem_options']['dvs_base']]),self.train_X,axis=0)

                elif self.type_initialization == 2:
                    raise Exception("Option not implemented yet")
                elif self.type_initialization == 3:
                    self.train_X = SAMPLINGtools.readInitializationFile(
                        self.folderExecution / "Outputs" / "optimization_data.csv",
                        self.initial_training,
                        self.stepSettings["optimization_options"]["problem_options"]["dvs"],
                    )
                elif self.type_initialization == 4:
                    self.train_X = IOtools.readExecutionParams(
                        self.folderExecution, nums=[0, self.initial_training - 1]
                    )

                if (
                    (self.type_initialization == 1)
                    and (self.optimization_options["problem_options"]["dvs_base"] is not None)
                    and (self.initial_training > 0)
                ):
                    self.train_X = np.append(
                        np.atleast_2d([i for i in self.optimization_options["problem_options"]["dvs_base"]]),
                        self.train_X,
                        axis=0,
                    )

            else:
                print("- Initialization function has been selected", typeMsg="i")
                self.train_X = self.optimization_options["initialization_options"]["initialization_fun"](self)
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
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())

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
            cold_start=(not readCasesFromTabular),
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
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())
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

        """
		Some initialization strategies may create points outside of the original bounds, but I may want to include them!
		"""
        if self.optimization_options["initialization_options"]["expand_bounds"]:
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
        rangePlot_force=None,
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

        if rangePlot_force is not None:
            rangePlot = rangePlot_force[:len(GPs)]
        else:
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
            self.optimization_results.plot(
                fn=fn, doNotShow=True, log=self.timings_file, tab_color=tab_color
            )

        """
		****************************************************************
		Acquisition
		****************************************************************
		"""
        try:    
            self.plotAcquisitionOptimizationSummary(fn=fn)
        except: 
            print('\t- Problem plotting acquisition optimization summary', typeMsg='w')

        return fn


    def plotAcquisitionOptimizationSummary(self, fn=None, step_from=0, step_to=-1):

        if step_to == -1:
            step_to = len(self.steps)

        step_to = np.min([step_to, len(self.steps)])

        step_num = np.arange(step_from, step_to)

        fig = fn.add_figure(label='Acquisition Convergence')

        axs = GRAPHICStools.producePlotsGrid(len(step_num), fig=fig, hspace=0.6, wspace=0.3)
        colors = GRAPHICStools.listColors()

        for step in step_num:

            ax = axs[step]

            if 'InfoOptimization' not in self.steps[step].__dict__: break

            # Grab info from optimization
            infoOPT = self.steps[step].InfoOptimization
            acq = self.steps[step].evaluators['acq_function']

            acq_trained = np.zeros(self.steps[step].train_X.shape[0])
            for ix in range(self.steps[step].train_X.shape[0]):
                acq_trained[ix] = acq(torch.Tensor(self.steps[step].train_X[ix,:]).unsqueeze(0)).item()

            # Plot trained acquisition
            ax.axhline(y=acq_trained.max(), c='k', ls='--', lw=1.0, label='max of trained')

            # Plot acquisition evolution 
            for i in range(len(infoOPT)-1): #no cleanup stage
                y_acq = infoOPT[i]['info']['acq_evaluated'].cpu().numpy()
                
                if len(y_acq.shape)>1:
                    for j in range(y_acq.shape[1]):
                        ax.plot(y_acq[:,j],'-o', c=colors[i], markersize=0.5, lw = 0.3, label=f'{infoOPT[i]["method"]} (candidate #{j})')
                else:
                    ax.plot(y_acq,'-o', c=colors[i], markersize=1, lw = 0.5, label=f'{infoOPT[i]["method"]}')
                
                # Plot max of guesses
                if len(y_acq)>0:
                    ax.axhline(y=y_acq.max(axis=1)[0], c=colors[i], ls='--', lw=1.0, label=f'{infoOPT[i]["method"]} (max of guesses)')

            ax.set_title(f'BO Step #{step}')
            ax.set_ylabel('$f_{acq}$ (to max)')
            ax.set_xlabel('Evaluations')
            if step == step_num[0]:
                ax.legend(loc='best', fontsize=6)

            GRAPHICStools.addDenseAxis(ax)


    def plotModelStatus(
        self, fn=None, boStep=-1, plotsPerFigure=20, stds=2, tab_color=None
    ):
        step = self.steps[boStep]

        GP = step.GP["combined_model"]

        # ---- Jacobian -------------------------------------------------------
        fig = fn.add_figure(label=f"#{boStep}: Jacobian", tab_color=tab_color)
        maxPoints = 1  # 4
        xExplore = []
        if "x_next" in step.__dict__.keys() and step.x_next is not None:
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
        boundsThis = info[0]["bounds"].cpu().numpy().transpose(1, 0) if "bounds" in info[0] else None

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
        axs = GRAPHICStools.producePlotsGrid(num_plots, fig=fig1, hspace=0.4, wspace=0.4)

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
            try:
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
            except KeyError as e:
                print(f"\t- Problem plotting {info[ipost]['method']}: ",e, typeMsg="w")

        xypair = np.array(xypair)

        axsDVs[0].legend(prop={"size": 5})
        ax1_r.set_ylabel("DV values")
        GRAPHICStools.addDenseAxis(ax1_r)
        GRAPHICStools.autoscale_y(ax1_r)

        ax2_r.set_ylabel("Acquisition values")
        GRAPHICStools.addDenseAxis(ax2_r)
        GRAPHICStools.autoscale_y(ax2_r)

        ax0_r.plot(xypair[:, 0], xypair[:, 1], "-s", markersize=5, lw=2.0, c="k")

        iinfo = info[-1]["info"]
        for i, y in enumerate(iinfo["y_res"]):
            ax0_r.axhline(
                y=y,
                c=colors[ipost + 1],
                ls="--",
                lw=2,
                label=info[-1]["method"] if i == 0 else "",
            )
        iinfo = info[0]["info"]
        ax0_r.axhline(y=iinfo["y_res_start"][0], c="k", ls="--", lw=2)

        ax0_r.set_xlabel("Optimization iterations")
        ax0_r.set_ylabel("$f_{acq}$")
        GRAPHICStools.addDenseAxis(ax0_r)
        ax0_r.legend(loc="best", prop={"size": 8})
        ax0_r.set_title("Evolution of acquisition in optimization stages")

        for i in range(len(axs)):
            GRAPHICStools.addDenseAxis(axs[i])

# ----------------------------------------------------------------------
# Stopping criteria
# ----------------------------------------------------------------------

def max_val(maximum_value_orig, maximum_value_is_rel, res_base):
    if maximum_value_is_rel:
        maximum_value = maximum_value_orig * res_base
        print(f'\t* Maximum value for convergence provided as relative value of {maximum_value_orig} from base {res_base:.3e} --> {maximum_value:.3e}')
    else:
        maximum_value = maximum_value_orig
        print(f'\t* Maximum value for convergence: {maximum_value} (starting case has {res_base:.3e})' )

    return maximum_value

def stopping_criteria_default(mitim_bo, parameters = {}):

    # ------------------------------------------------------------------------------------
    # Determine the stopping criteria
    # ------------------------------------------------------------------------------------

    maximum_value_is_rel    = parameters["maximum_value_is_rel"]
    maximum_value_orig      = parameters["maximum_value"]
    minimum_dvs_variation   = parameters["minimum_dvs_variation"]

    res_base = -mitim_bo.BOmetrics["overall"]["Residual"][0].item()

    maximum_value = max_val(maximum_value_orig, maximum_value_is_rel, res_base)

    # ------------------------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------------------------

    if minimum_dvs_variation is not None:
        converged_by_dvs, yvals = stopping_criteria_by_dvs(mitim_bo, minimum_dvs_variation)
    else:
        converged_by_dvs = False
        yvals = None

    if maximum_value is not None:
        converged_by_value, yvals = stopping_criteria_by_value(mitim_bo, maximum_value)
    else:
        converged_by_value = False
        yvals = None

    converged = converged_by_value or converged_by_dvs

    return converged, yvals

def stopping_criteria_by_value(mitim_bo, maximum_value):

    # Grab scalarized objectives for each case
    print("\t- Checking maximum value so far...")
    _, _, maximization_value = mitim_bo.scalarized_objective(torch.from_numpy(mitim_bo.train_Y).to(mitim_bo.dfT))
    yvals = maximization_value.cpu().numpy()

    # Best case (maximization)
    best_value_so_far = np.nanmax(yvals)

    # Converged?
    print(f'\t\t* Best scalar function so far (to maximize): {best_value_so_far:.3e} (threshold: {maximum_value:.3e})')
    criterion_is_met = best_value_so_far > maximum_value

    return criterion_is_met, -yvals

def stopping_criteria_by_dvs(mitim_bo, minimum_dvs_variation):

    print("\t- Checking DV variations...")
    _, yG_max = TESTtools.DVdistanceMetric(mitim_bo.train_X)

    criterion_is_met = (
        mitim_bo.currentIteration
        >= minimum_dvs_variation[0]
        + minimum_dvs_variation[1]
    )
    for i in range(int(minimum_dvs_variation[1])):
        criterion_is_met = criterion_is_met and (
            yG_max[-1 - i] < minimum_dvs_variation[2]
        )

    if criterion_is_met:
        print(
            f"\t\t* DVs varied by less than {minimum_dvs_variation[2]}% compared to the rest of individuals for the past {int(minimum_dvs_variation[1])} iterations"
        )
    else:
        print(
            f"\t\t* DVs have varied by more than {minimum_dvs_variation[2]}% compared to the rest of individuals for the past {int(minimum_dvs_variation[1])} iterations"
        )

    return criterion_is_met, yG_max

def read_from_scratch(file):
    """
    This reads a pickle file for the entire class
    """

    optimization_object = opt_evaluator(None)
    mitim = MITIM_BO(optimization_object, onlyInitialize=True, askQuestions=False)
    mitim = mitim.read(file=file, iteration=-1, provideFullClass=True)

    return mitim

def avoidClassInitialization(folderWork):
    print("It was requested that I try read the class before I initialize and select parameters...",typeMsg="i")

    try:
        aux = IOtools.unpickle_mitim(folderWork / "Outputs" / "optimization_object.pkl")
        opt_fun = aux.optimization_object
        cold_start = False
        print("\t- cold_start was successful", typeMsg="i")
    except:
        opt_fun = None
        cold_start = True
        flagger = print("\t- cold_start was requested but it didnt work (c)", typeMsg="q")
        if not flagger:
            embed()

    return opt_fun, cold_start

def clean_state(folder):
    '''
    This function cleans the a read pickle file to avoid problems with reading cases run in a different machine
    '''        

    print(">><<>><< Cleaning state of the class...", typeMsg="i")

    aux = read_from_scratch(folder / "Outputs" / "optimization_object.pkl")

    if aux is not None:
        
        from mitim_modules.portals import PORTALStools, PORTALSmain

        if isinstance(aux.optimization_object, PORTALSmain.portals):
            aux.optimization_options['convergence_options']['stopping_criteria'] = PORTALStools.stopping_criteria_portals

        aux.folderOutputs = folder / "Outputs"
        aux.timings_file = aux.folderOutputs / "timing.jsonl"

        aux.save()

    print(">><<>><< Cleaning state of the class... Done", typeMsg="i")
