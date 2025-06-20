import copy
import datetime
import torch
import botorch
import numpy as np
from mitim_tools.misc_tools import IOtools, MATHtools
from mitim_tools.opt_tools import SURROGATEtools, OPTtools, BOTORCHtools
from mitim_tools.opt_tools.utils import TESTtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


def identity(X, *args):
    return X, {}


def identityOutputs(X, *args):
    return torch.ones(X.shape[:-1]).unsqueeze(-1)


class OPTstep:
    def __init__(
        self,
        train_X,
        train_Y,
        train_Ystd,
        bounds,
        stepSettings={},
        surrogate_parameters={},
        strategy_options={},
        BOmetrics=None,
        currentIteration=1,
    ):
        """
        train_Ystd is in standard deviations (square root of the variance), absolute magnitude
        Rule: X_Y are provided in absolute units. Normalization has to happen inside each surrogate_model,
                and de-normalized before giving results to the outside of the function
        """

        self.train_X, self.train_Y, self.train_Ystd = train_X, train_Y, train_Ystd

        """
		Check dimensions
			- train_X should be (num_train,dimX)
			- train_Y should be (num_train,dimY)
			- train_Ystd should be (num_train,dimY) or just one float representing all values
		"""

        if len(self.train_X.shape) < 2:
            print(
                "--> train x only had 1 dimension, assuming that it has only 1 dimension"
            )
            self.train_X = np.transpose(np.atleast_2d(self.train_X))

        if len(self.train_Y.shape) < 2:
            print(
                "--> train y only had 1 dimension, assuming that it has only 1 dimension"
            )
            self.train_Y = np.transpose(np.atleast_2d(self.train_Y))

        if (
            isinstance(self.train_Ystd, float)
            or isinstance(self.train_Ystd, int)
            or len(self.train_Ystd.shape) < 2
        ):
            print(
                "--> train y noise only had 1 value only, assuming constant (std dev) for all samples in absolute terms"
            )
            if self.train_Ystd > 0:
                print(
                    "--> train y noise only had 1 value only, assuming constant (std dev) for all samples in absolute terms"
                )
                self.train_Ystd = self.train_Y * 0.0 + self.train_Ystd
            else:
                print(
                    "--> train y noise only had 1 value only, assuming constant (std dev) for all samples in relative terms"
                )
                self.train_Ystd = self.train_Y * np.abs(self.train_Ystd)

        if len(self.train_Ystd.shape) < 2:
            print(
                "--> train y noise only had 1 dimension, assuming that it has only 1 dimension"
            )
            self.train_Ystd = np.transpose(np.atleast_2d(self.train_Ystd))

        # **** Get argumnets into this class

        self.bounds = bounds
        self.stepSettings = stepSettings
        self.BOmetrics = BOmetrics
        self.currentIteration = currentIteration
        self.strategy_options = strategy_options

        # **** Step settings
        self.surrogate_options = self.stepSettings["optimization_options"]["surrogate_options"]
        self.acquisition_type = self.stepSettings["optimization_options"]["acquisition_options"]["type"]
        self.acquisition_params = self.stepSettings["optimization_options"]["acquisition_options"]["parameters"]
        self.favor_proximity_type = self.stepSettings["optimization_options"]["acquisition_options"]["favor_proximity_type"]
        self.optimizers = {}
        for optimizer in self.stepSettings["optimization_options"]["acquisition_options"]["optimizers"]:
            self.optimizers[optimizer] = self.stepSettings["optimization_options"]["acquisition_options"]["optimizer_options"][optimizer]     
        self.outputs = self.stepSettings["outputs"]
        self.dfT = self.stepSettings["dfT"]
        self.best_points_sequence = self.stepSettings["best_points_sequence"]
        self.fileOutputs = self.stepSettings["fileOutputs"]
        self.surrogate_parameters = surrogate_parameters

        # **** From standard deviation to variance
        self.train_Yvar = self.train_Ystd**2

    def fit_step(self, avoidPoints=None, fitWithTrainingDataIfContains=None):
        """
        Notes:
            - Note that fitWithTrainingDataIfContains = 'Tar' would only use the train_X,Y,Yvar tensors
                    to fit those surrogate variables that contain 'Tar' in their names. This is useful when in
                    PORTALS I want to simply use the training in a file and not directly from train_X,Y,Yvar for
                    the fluxes but I do want *new* target calculation
        """

        if avoidPoints is None:
            avoidPoints = []

        """
		*********************************************************************************************************************
			Preparing for fit
		*********************************************************************************************************************
		"""

        # Prepare case information. Copy because I'll be removing outliers
        self.x, self.y, self.yvar = (
            copy.deepcopy(self.train_X),
            copy.deepcopy(self.train_Y),
            copy.deepcopy(self.train_Yvar),
        )

        # Add outliers to avoid points (it cannot happen inside of SURROGATEtools or it will fail at combining)
        self.avoidPoints = copy.deepcopy(avoidPoints)
        self.curate_outliers()

        if self.fileOutputs is not None:
            with open(self.fileOutputs, "a") as f:
                f.write("\n\n-----------------------------------------------------")
                f.write("\n * Fitting GP models to training data...")
        print(
            f"\n~~~~~~~ Performing fitting with {len(self.train_X)-len(self.avoidPoints)} training points ({len(self.avoidPoints)} avoided from {len(self.train_X)} total) ~~~~~~~~~~\n"
        )

        """
		*********************************************************************************************************************
			Performing Fit
		*********************************************************************************************************************
		"""

        self.GP = {"individual_models": [None] * self.y.shape[-1]}
        fileTraining = IOtools.expandPath(self.stepSettings['folderOutputs']) / "surrogate_data.csv"
        fileBackup = fileTraining.parent / "surrogate_data.csv.bak"
        if fileTraining.exists():
            fileTraining.replace(fileBackup)

        print("--> Fitting multiple single-output models and creating composite model")
        time1 = datetime.datetime.now()

        for i in range(self.y.shape[-1]):
            outi = self.outputs[i] if (self.outputs is not None) else None

            # ----------------- specialTreatment is applied when I only want to use training data from a file, not from train_X
            specialTreatment = (
                (outi is not None)
                and (fitWithTrainingDataIfContains is not None)
                and (fitWithTrainingDataIfContains not in outi)
            )
            # -----------------------------------------------------------------------------------------------------------------------------------

            outi_transformed = (
                self.stepSettings["name_transformed_ofs"][i]
                if (self.stepSettings["name_transformed_ofs"] is not None)
                else outi
            )

            # ---------------------------------------------------------------------------------------------------
            # Define model-specific functions for this output
            # ---------------------------------------------------------------------------------------------------

            surrogate_options = copy.deepcopy(self.surrogate_options)

            # Then, depending on application (e.g. targets in mitim are fitted differently)
            if (
                "selectSurrogate" in surrogate_options
                and surrogate_options["selectSurrogate"] is not None
            ):
                surrogate_options = surrogate_options["selectSurrogate"](
                    outi, surrogate_options
                )

            # ---------------------------------------------------------------------------------------------------
            # To avoid problems with fixed values (e.g. calibration terms that are fixed)
            # ---------------------------------------------------------------------------------------------------

            threshold_to_consider_fixed = 1e-6
            MaxRelativeDifference = np.abs(self.y.max() - self.y.min()) / np.abs(
                self.y.mean()
            )

            if (
                np.isnan(MaxRelativeDifference)
                or (
                    (self.y.shape[0] > 1)
                    and ((MaxRelativeDifference < threshold_to_consider_fixed).all())
                )
            ) and (not specialTreatment):
                print(
                    f"\t- Identified that outputs did not change, utilizing constant kernel for {outi}",
                    typeMsg="w",
                )
                FixedValue = True
                surrogate_options["TypeMean"] = 0
                surrogate_options["TypeKernel"] = 6  # Constant kernel

            else:
                FixedValue = False

            # ---------------------------------------------------------------------------------------------------
            # Fit individual output
            # ---------------------------------------------------------------------------------------------------

            # Data to train the surrogate
            x = self.x
            y = np.expand_dims(self.y[:, i], axis=1)
            yvar = np.expand_dims(self.yvar[:, i], axis=1)

            if specialTreatment:
                x, y, yvar = (
                    np.empty((0, x.shape[-1])),
                    np.empty((0, y.shape[-1])),
                    np.empty((0, y.shape[-1])),
                )

            # Surrogate

            print(f"~ Model for output: {outi}")

            GP = SURROGATEtools.surrogate_model(
                x,
                y,
                yvar,
                self.surrogate_parameters,
                bounds=self.bounds,
                output=outi,
                output_transformed=outi_transformed,
                avoidPoints=self.avoidPoints,
                dfT=self.dfT,
                surrogate_options=surrogate_options,
                FixedValue=FixedValue,
                fileTraining=fileTraining,
            )

            # Fitting
            GP.fit()

            self.GP["individual_models"][i] = GP

        fileBackup.unlink(missing_ok=True)

        # ------------------------------------------------------------------------------------------------------
        # Combine them in a ModelListGP (create one single with MV but do not fit)
        # ------------------------------------------------------------------------------------------------------

        print("~ MV model to initialize combination")

        self.GP["combined_model"] = SURROGATEtools.surrogate_model(
            self.x,
            self.y,
            self.yvar,
            self.surrogate_parameters,
            avoidPoints=self.avoidPoints,
            bounds=self.bounds,
            dfT=self.dfT,
            surrogate_options=self.surrogate_options,
        )

        models = ()
        for GP in self.GP["individual_models"]:
            models += (GP.gpmodel,)
        self.GP["combined_model"].gpmodel = BOTORCHtools.ModifiedModelListGP(*models)

        # ------------------------------------------------------------------------------------------------------
        # Make sure each model has the right surrogate_transformation_variables inside the combined model
        # ------------------------------------------------------------------------------------------------------
        if self.GP["combined_model"].surrogate_transformation_variables is not None:
            for i in range(self.y.shape[-1]):

                outi = self.outputs[i] if (self.outputs is not None) else None

                if outi is not None:
                    self.GP["combined_model"].surrogate_transformation_variables[outi] = self.GP["individual_models"][i].surrogate_transformation_variables[outi]

        print(f"--> Fitting of all models took {IOtools.getTimeDifference(time1)}")

        """
		*********************************************************************************************************************
			Postprocessing
		*********************************************************************************************************************
		"""

        # Test (if test could not be launched is likely because a singular matrix for Choleski decomposition)
        print("--> Launching tests to assure batch evaluation accuracy")
        TESTtools.testBatchCapabilities(self.GP["combined_model"])
        print("--> Launching tests to assure model combination accuracy")
        TESTtools.testCombinationCapabilities(
            self.GP["individual_models"], self.GP["combined_model"]
        )
        print("--> Launching tests evaluate accuracy on training set (absolute units)")
        self.GP["combined_model"].testTraining()

        txt_time = IOtools.getTimeDifference(time1)

        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        )

        if self.fileOutputs is not None:
            with open(self.fileOutputs, "a") as f:
                f.write(f" (took total of {txt_time})")

    def defineFunctions(self, scalarized_objective):
        """
        I create this so that, upon reading a pickle, I re-call it. Otherwise, it is very heavy to store lambdas
        """

        self.evaluators = {"GP": self.GP["combined_model"]}

        # **************************************************************************************************
        # Objective (multi-objective model -> single objective residual)
        # **************************************************************************************************

        # Build function to pass to acquisition
        def residual(Y, X = None):
            return scalarized_objective(Y)[2]

        self.evaluators["objective"] = botorch.acquisition.objective.GenericMCObjective(residual)

        # **************************************************************************************************
        # Quick function to return components (I need this for ROOT too, since I need the components)
        # **************************************************************************************************

        def residual_function(x, outputComponents=False):
            mean, _, _, _ = self.evaluators["GP"].predict(x) #TODO: make the predict method simply the callable of my GP
            yOut_fun, yOut_cal, yOut = scalarized_objective(mean)

            return (yOut, yOut_fun, yOut_cal, mean) if outputComponents else yOut

        self.evaluators["residual_function"] = residual_function

        # **************************************************************************************************
        # Acquisition functions (following BoTorch assumption of maximization)
        # **************************************************************************************************

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Analytic acquisition functions
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if self.acquisition_type == "posterior_mean":
            print('\t* Chosen analytic posterior_mean acquisition, objective nonlinearity not considered', typeMsg="i")
            self.evaluators["acq_function"] = BOTORCHtools.PosteriorMean(
                self.evaluators["GP"].gpmodel,
                objective=self.evaluators["objective"]
            )

        elif self.acquisition_type == "logei":
            print("\t* Chosen analytic logei acquisition, igoring objective", typeMsg="w")
            self.evaluators["acq_function"] = (
                botorch.acquisition.analytic.LogExpectedImprovement(
                    self.evaluators["GP"].gpmodel,
                    best_f=self.evaluators["objective"](self.evaluators["GP"].train_Y.unsqueeze(1)).max()
                )
            )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Monte Carlo acquisition functions
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        sampler = botorch.sampling.normal.SobolQMCNormalSampler(torch.Size([self.acquisition_params["mc_samples"]]))

        if self.acquisition_type == "simple_regret_mc": # Former posterior_mean_mc
            self.evaluators["acq_function"] = (
                botorch.acquisition.monte_carlo.qSimpleRegret(
                    self.evaluators["GP"].gpmodel,
                    objective=self.evaluators["objective"],
                    sampler=sampler
                )
            )

        elif self.acquisition_type == "ei_mc":
            self.evaluators["acq_function"] = (
                botorch.acquisition.monte_carlo.qExpectedImprovement(
                    self.evaluators["GP"].gpmodel,
                    objective=self.evaluators["objective"],
                    best_f=self.evaluators["objective"](self.evaluators["GP"].train_Y.unsqueeze(1)).max(),
                    sampler=sampler
                )
            )

        elif self.acquisition_type == "logei_mc":
            self.evaluators["acq_function"] = (
                botorch.acquisition.logei.qLogExpectedImprovement(
                    self.evaluators["GP"].gpmodel,
                    objective=self.evaluators["objective"],
                    best_f=self.evaluators["objective"](self.evaluators["GP"].train_Y.unsqueeze(1)).max(),
                    sampler=sampler
                )
            )

        elif self.acquisition_type == "noisy_logei_mc":
            self.evaluators["acq_function"] = (
                botorch.acquisition.logei.qLogNoisyExpectedImprovement(
                    self.evaluators["GP"].gpmodel,
                    objective=self.evaluators["objective"],
                    X_baseline=self.evaluators["GP"].train_X,
                    sampler=sampler
                )
            )

        # Add this because of the way train_X is defined within the gpmodel, which is fundamental, but the acquisition for sample
        # around best, needs the raw one! (for noisy it is automatic)
        self.evaluators["acq_function"]._X_baseline = self.evaluators["GP"].train_X #TOFIX

        # **************************************************************************************************
        # Selector (Takes x and residuals of optimized points, and provides the indices for organization)
        # **************************************************************************************************

        self.evaluators["lambdaSelect"] = self.lambda_select

    def lambda_select(self, x, res):
        return correctResidualForProximity(
            x,
            res,
            self.train_X[self.BOmetrics["overall"]["indBest"]],
            self.BOmetrics["overall"]["Residual"][self.BOmetrics["overall"]["indBest"]],
            self.favor_proximity_type,
        )

    def optimize(
        self,
        position_best_so_far=-1,
        seed=0,
        forceAllPointsInBounds=False,
    ):

        """
		***********************************************
		Peform optimization
		***********************************************
		"""

        if self.fileOutputs is not None:
            with open(self.fileOutputs, "a") as f:
                f.write("\n\n * Running optimization workflows to find next points...")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Evaluate Adquisition
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~ Running optimization methods")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print(f'\n~~ Maximization of "{self.acquisition_type}" acquisition using "{self.optimizers.keys()}" methods to find {self.best_points_sequence} points\n')

        self.x_next, self.InfoOptimization = OPTtools.acquire_next_points(
            stepSettings=self.stepSettings,
            evaluators=self.evaluators,
            strategy_options=self.strategy_options,
            best_points=int(self.best_points_sequence),
            optimizers=self.optimizers,
            it_number=self.currentIteration,
            position_best_so_far=position_best_so_far,
            seed=seed,
            forceAllPointsInBounds=forceAllPointsInBounds,
        )

        print(f"\n~~ Complete acquisition workflows found {self.x_next.shape[0]} points")

    def curate_outliers(self):
        # Remove outliers
        self.outliers = removeOutliers(
            self.y,
            stds_outside=self.surrogate_options["stds_outside"],
            stds_outside_checker=self.surrogate_options["stds_outside_checker"],
            alreadyAvoided=self.avoidPoints,
        )

        # Info
        if len(self.outliers) > 0:
            print(f"\t* OUTLIERS in positions: {self.outliers}. Adding to avoid points")

        try:
            self.avoidPoints.extend(self.outliers)
        except:
            self.avoidPoints = [
                int(i) for i in np.append(self.avoidPoints, self.outliers)
            ]

        if len(self.avoidPoints) > 0:
            print(f"\t ~~ Avoiding {len(self.avoidPoints)} points: ", self.avoidPoints)


def removeOutliers(y, stds_outside=5, stds_outside_checker=1, alreadyAvoided=[]):
    """
    This routine finds outliers to be removed
    """

    if stds_outside is not None:
        print(
            f"\t Checking outliers by +-{stds_outside}sigma from the rest (min number of {stds_outside_checker})"
        )

        avoidPoints = []
        for i in range(y.shape[0]):
            outlier = False
            for j in range(y.shape[1]):
                outlier_this = TESTtools.isOutlier(
                    y[i, j],
                    np.delete(y[:, j], [i], axis=0),
                    stds_outside=stds_outside,
                    stds_outside_checker=stds_outside_checker,
                )
                outlier = outlier or outlier_this

                if outlier_this:
                    print(f"\t Point #{i} is an outlier in position {j}: {y[i,j]:.5f}")

            if outlier and i not in alreadyAvoided:
                avoidPoints.append(i)

    else:
        avoidPoints = []

    return avoidPoints


def correctResidualForProximity(x, res, xBest, resBest, favor_proximity_type):
    what_is_already_good_improvement = (
        1e-2  # Improvement of 100x is already good enough
    )

    indeces_raw = torch.argsort(res, dim=0, descending=True)

    # Raw organized
    if favor_proximity_type == 0:
        indeces = indeces_raw

    # Special treatment
    if favor_proximity_type == 1:
        # Improvement in residual
        resn = res / resBest

        # If improvement in residual is better than what_is_already_good_improvement, clip it
        resn = resn.clip(what_is_already_good_improvement)

        # Normalized distance
        dn = MATHtools.calculateDistance(xBest, x) / (
            MATHtools.calculateDistance(xBest, x * 0.0)
        )

        # Add the distance just as a super small, for organizing
        resn -= dn * 1e-6

        indeces = torch.argsort(resn, dim=0)

    # Provide info
    if indeces[0] != indeces_raw[0]:
        print("\t* Selection of best point has accounted for proximity")

    return indeces
