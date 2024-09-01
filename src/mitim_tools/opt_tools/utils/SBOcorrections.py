import copy, torch
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.opt_tools.utils import SAMPLINGtools, TESTtools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed


def upgradeBounds(bounds_orig, train_X, avoidPoints):
    """
    make sure that bounds encompass all points
    """
    bounds = copy.deepcopy(bounds_orig)

    for i, ikey in enumerate(bounds):
        for j in range(train_X.shape[0]):
            if j not in avoidPoints:
                if bounds[ikey][0] > train_X[j, i]:
                    print(
                        f"\t~ Lower bound in variable {ikey} decreased to encompass training point #{j} ({bounds[ikey][0]:.3f} to {train_X[j,i]:.3f})"
                    )
                    bounds[ikey][0] = train_X[j, i]
                if bounds[ikey][1] < train_X[j, i]:
                    print(
                        f"\t~ Upper bound in variable {ikey} increased to encompass training point #{j} ({bounds[ikey][1]:.3f} to {train_X[j,i]:.3f})"
                    )
                    bounds[ikey][1] = train_X[j, i]

    return bounds


def modifyTrustRegion(
    self, BoundsReduction=0.75, addYN=True, pointsToAdd=None, seed=0, allow_extrap=False
):
    """
    This function modifies the next point to evaluate, the training set and the bounds
    if pointsToAdd is None, then assign the same number was was lost by reducing bounds
    """

    if BoundsReduction < 1.0:
        txt_extra2 = "reduction"
    else:
        txt_extra2 = "increase"

    index = self.BOmetrics["overall"]["indBest"]
    train_X_Best = self.BOmetrics["overall"]["xBest"]
    train_Y_Best = self.BOmetrics["overall"]["yBest"]

    print(
        f"\t--> Running {txt_extra2} of trust region to {BoundsReduction*100-100}% of previous size by checking best point, #{index}:"
    )
    print("\t\t", train_X_Best)

    # Reduce bounds centered at best point so far

    #TODO: Remove when fixed self.bounds
    # to tensor
    boundsTensor = []
    for i in self.bounds:
        boundsTensor.append(self.bounds[i])
    boundsTensor = np.transpose(boundsTensor)

    bounds, minimum_reached, atleastone = factorBounds(
        center=train_X_Best,
        bounds=boundsTensor,
        BoundsReduction=BoundsReduction,
        bounds_lim=self.bounds_orig,
        allow_extrap=allow_extrap,
    )

    self.hard_finish = self.hard_finish or minimum_reached

    if self.hard_finish:
        return False

    # Check how many of the old training points fall outside (will not be used in the training)
    avoided_previous = len(self.avoidPoints_outside)
    self.avoidPoints_outside = []
    for i in range(self.train_X.shape[0]):
        if not TESTtools.checkSolutionIsWithinBounds(self.train_X[i], bounds):
            self.avoidPoints_outside.append(i)
            print(f"\t~ Adding point #{i} to the avoidPoints_outside tensor:")
            print("\t\t", self.train_X[i])
    avoided_new = len(self.avoidPoints_outside)

    if avoided_new > 0:
        print(f"\t* Points {self.avoidPoints_outside} were outside of bounds")

    eliminated = avoided_new - avoided_previous

    print(f"\t This stage is removing {eliminated} points")

    if pointsToAdd is None:
        pointsToAdd = eliminated

    # If I have eliminated points, add points randomly now
    if pointsToAdd > 0 and addYN:
        eliminated0 = np.min([pointsToAdd, self.initial_training])
        print(
            f"\t\t--> Removal of {eliminated} points, filling region with {eliminated0} new points"
        )
        newnext = SAMPLINGtools.LHS(eliminated0, bounds, seed=seed)
        self.x_next = newnext

    #TODO: Remove when fixed self.bounds
    # from tensor
    for j, i in enumerate(self.bounds):
        self.bounds[i][0] = bounds[0, j]
        self.bounds[i][1] = bounds[1, j]

    self.stepSettings["bounds"] = self.bounds

    return atleastone


def TURBOupdate(self, StrategyOptions_use, position=0, seed=0):
    """
    Remember. Explanation of metrics:

            +-0.2 - Very bad accuracy
            +-1 - Bad accuracy
            +-2 - Good accuracy
            +-3 - Excellent accuracy

            >0 - Prediction got better
            <0 - Prediction got worse

    """

    addPoints = StrategyOptions_use["TURBO_addPoints"]
    changes = StrategyOptions_use["TURBO_changeBounds"]
    metrics = StrategyOptions_use["TURBO_metricsRow"]

    print("\n~~~~~~~~~~~~~~~~~~~~~~~ Performing TURBO update ~~~~~~~~~~~~~~~~~~~~~~~")

    MetricThreshold = 0
    Nbad = int(metrics[0])  # 1
    Ngood = int(metrics[1])  # 2
    BoundsReduction = float(changes[0])  # 0.75
    BoundsIncrease = float(changes[1])  # 1/0.75

    reduceSize, increaseSize = False, False

    seed += self.train_X.shape[0]  # Seed for random adds during TURBO

    # ------------------------------------
    # Understanding behavior
    # ------------------------------------

    metrics = np.array(
        [self.BOmetrics["BOmetric"][i] for i in self.BOmetrics["BOmetric_it"]][1:]
    )

    # When was the last operation? Count from them

    lim = -1
    for i in range(len(metrics)):
        if np.isnan(metrics[i]):
            lim = i
            break
    metrics = metrics[:lim]

    # If N metrics bad in a row, reduce TR
    failure_counter = 0
    if Nbad <= len(metrics):
        HaveRunEnough = True
        for i in range(Nbad):
            HaveRunEnough = HaveRunEnough and (
                (self.BOmetrics["TRoperation"][position - 1 - i] == 1.0)
                or np.isnan(self.BOmetrics["TRoperation"][position - 1 - i])
            )
        if HaveRunEnough:
            for i in range(Nbad):
                if metrics[-1 - i] < MetricThreshold:
                    failure_counter += 1
                else:
                    break

        if failure_counter > 0:
            print(f"\t- Failures in a row: {failure_counter}")

        if failure_counter == Nbad:
            print(f"\t- Equal or more than {Nbad} failures in a row")
            reduceSize = True

    # If N metrics good in a row, expand TR
    success_counter = 0
    if Ngood <= len(metrics):
        HaveRunEnough = True
        for i in range(Nbad):
            HaveRunEnough = HaveRunEnough and (
                (self.BOmetrics["TRoperation"][position - 1 - i] == 1.0)
                or np.isnan(self.BOmetrics["TRoperation"][position - 1 - i])
            )

        if HaveRunEnough:
            for i in range(Ngood):
                if metrics[-1 - i] >= MetricThreshold:
                    success_counter += 1
                else:
                    break

        if success_counter > 0:
            print(f"\t- Successes in a row: {success_counter}")

        if success_counter == Ngood:
            print(f"\t- Equal or more than {Ngood} successes in a row")
            increaseSize = True

    # ------------------------------------
    # Apply corrections
    # ------------------------------------

    if reduceSize:
        atleastone = modifyTrustRegion(
            self, BoundsReduction=BoundsReduction, pointsToAdd=0, seed=seed
        )
        self.BOmetrics["TRoperation"][position] = BoundsReduction

    elif increaseSize:
        atleastone = modifyTrustRegion(
            self, BoundsReduction=BoundsIncrease, pointsToAdd=0, seed=seed
        )
        self.BOmetrics["TRoperation"][position] = BoundsIncrease
    else:
        self.BOmetrics["TRoperation"][
            position
        ] = 1.0  # self.BOmetrics['TRoperation'][-1]*1.0)
        atleastone = False

    if atleastone:
        if addPoints > 0:
            print(f"* Requested to add {addPoints} points to new region")
            self.x_next = torch.from_numpy(
                SAMPLINGtools.LHS(addPoints, self.bounds, seed=seed + 1)
            ).to(self.dfT)
            yE, yE_next = self.updateSet(StrategyOptions_use, isThisCorrected=True)
    else:
        print(
            "\t Nothing done. Next iteration will go on over the same trust region without adding points."
        )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    return int(reduceSize) + int(increaseSize)


def hitbounds(self, hitbounds_var, changesMade=0):
    """
    If x_next (best pointsf from OPT) have reached bounds, extend those bounds according to namelist
    """

    for cont, ikey in enumerate(self.bounds):
        minVal = copy.deepcopy(self.bounds[ikey][0])
        maxVal = copy.deepcopy(self.bounds[ikey][1])

        dim = maxVal - minVal

        factor = 0.05

        variationLow = (hitbounds_var[0] - 1) * dim
        variationUp = (hitbounds_var[1] - 1) * dim

        checker = self.x_next

        # ---- Check if any has gone outside of bounds

        isAtLowerBounds = False
        isAtUpperBounds = False
        for j in range(checker.shape[0]):
            item = checker[j][cont]
            if (item - minVal) < dim * factor:
                isAtLowerBounds = True
            if (item - maxVal) > -dim * factor:
                isAtUpperBounds = True

        # ----

        # If any has reached bounds, lower
        if isAtLowerBounds:
            if variationLow > 0:
                print(
                    "\t-> Lower bound for {0} reached, lowering by {1}% ({2:.3f} --> {3:.3f})".format(
                        ikey,
                        hitbounds_var[0] - 1,
                        self.bounds[ikey][0],
                        self.bounds[ikey][0] - variationLow,
                    )
                )
                self.bounds[ikey][0] -= variationLow
                changesMade += 1
            else:
                print(f"\t-> Lower bound for {ikey} reached, but no action taken")

        # If any has reached bounds, increase
        if isAtUpperBounds:
            if variationUp > 0:
                print(
                    "\t-> Upper bound for {0} reached, increasing by {1}% ({2:.3f} --> {3:.3f})".format(
                        ikey,
                        hitbounds_var[1] - 1,
                        self.bounds[ikey][1],
                        self.bounds[ikey][1] + variationUp,
                    )
                )
                self.bounds[ikey][1] += variationUp
                changesMade += 1
            else:
                print(f"\t-> Upper bound for {ikey} reached, but no action taken")

    return changesMade


def correctionsSet(self, StrategyOptions_use):
    print(
        "\n~~~~~~~~~~~~~~~~~~~~~~~ Entering correction module ~~~~~~~~~~~~~~~~~~~~~~~"
    )

    seed = 0  # Change this

    StrategyOptions = copy.deepcopy(StrategyOptions_use)

    if "MaxTrainedPoints" not in StrategyOptions:
        StrategyOptions["MaxTrainedPoints"] = None
    if "HitBoundsIncrease" not in StrategyOptions:
        StrategyOptions["HitBoundsIncrease"] = [1.0, 1.0]
    if "SwitchIterationsReduction" not in StrategyOptions:
        StrategyOptions["SwitchIterationsReduction"] = [
            None,
            0.75,
        ]  # [np.arange(100,1000,1), 0.75],
    if "BadMetricsReduction" not in StrategyOptions:
        StrategyOptions["BadMetricsReduction"] = [None, 0.25]  # [100, 0.25],

    changesMade = 0

    """
	---------------------------------------------------------------------------------------------------
	Increase dimensions if one of the best solutions has hit the bounds
	---------------------------------------------------------------------------------------------------
	"""
    changesMade = hitbounds(self, StrategyOptions["HitBoundsIncrease"])

    """
	After a given mitim iteration, reduce bounds
	"""
    # [SwitchIterations, BoundsReduction]  = StrategyOptions['SwitchIterationsReduction']
    # if SwitchIterations is not None:
    #     if self.currentIteration in SwitchIterations:
    #         changesMade += 1
    #         print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #         print(' Evaluations: {0}. So I am restarting entire process by reducing bounds by {1}%'.format(self.train_X.shape[0],BoundsReduction*100))
    #         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    #         atleastone  = modifyTrustRegion(self,BoundsReduction=BoundsReduction,pointsToAdd=None,seed=seed)
    #         self.updateSet(isThisCorrected=True)

    """
	Maximum number of training points
	"""
    # PRF: Fix this with the new outside+failed

    # if StrategyOptions['MaxTrainedPoints'] is not None:
    #     if self.train_X.shape[0]-len(self.avoidPoints) > StrategyOptions['MaxTrainedPoints']:
    #         changesMade += 1
    #         x,y = np.delete(self.train_X,self.avoidPoints,axis=0), np.delete(self.train_Y,self.avoidPoints,axis=0)
    #         resAn,resMn = gatherActualAndModel(x,y,self.GPmodels[-1]['stepSettings'])
    #         worstNew    = resAn.argmin()
    #         resA,resM   = gatherActualAndModel(self.train_X,self.train_Y,self.GPmodels[-1]['stepSettings'])
    #         worst       = np.where(resA==resAn[worstNew])[0][0]
    #         self.avoidPoints.append(worst)

    #         print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #         print(' Evaluations: {0}. Adding point #{1} to list of avoidable points: {2}'.format(self.train_X.shape[0],worst,self.avoidPoints))
    #         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    """
	If it has bad metrics in a row
	"""
    # [howmany, BoundsReduction] = StrategyOptions['BadMetricsReduction']

    # if howmany is not None:
    #     if len(self.BOmetrics['BOmetric']) >= howmany:
    #         condition = True
    #         for i in range(howmany): condition = condition and ( np.abs(self.BOmetrics['BOmetric'][-1-i]) <= 1 )
    #     else: condition = False

    #     if condition:
    #         changesMade += 1
    #         print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #         print(' Because of bad metrics, reducing bounds to {0}%'.format(BoundsReduction))
    #         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    #         atleastone  = modifyTrustRegion(self,BoundsReduction=BoundsReduction,pointsToAdd=5,seed=seed)
    #         self.updateSet(isThisCorrected=True)

    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )

    return changesMade


def factorBounds(
    center,
    bounds,
    BoundsReduction=0.75,
    printYN=True,
    bounds_lim=None,
    min_dim_rel=0.005,
    allow_extrap=False,
):
    """
    Notes:
            - The reduction can be stronger if the center point if one of the dimensions is clamped. E.g. if the center is at bounds[0], then
                    a BoundsReduction of 0.5 would mean really a 0.25x reduction. This is the extreme case

            - bounds_lim is still a dictionary, rest tensors
    """

    bounds_new = copy.deepcopy(bounds)
    atleastonemodification = False

    minimum_dimension = []
    for idim, namedim in enumerate(bounds_lim):
        if printYN:
            print(f"\t\t--> Variable {namedim}:")

        # Initial (it#0) bounds, for reference
        dimesize_init = bounds_lim[namedim][1] - bounds_lim[namedim][0]

        # Bounds before this action
        dimesize_orig = bounds[1, idim] - bounds[0, idim]

        # Reduce overall size
        dimesize_new = dimesize_orig * BoundsReduction

        minimum_dimension.append(dimesize_new < min_dim_rel * dimesize_init)

        # If it has reached the minimum of this dimension, do not reduce further
        if minimum_dimension[-1]:
            if printYN:
                print("\t\t\t--> Minimum bounds reached, do not reduce further")
            bounds_new[:, idim] = bounds[:, idim]
            continue
        else:
            if BoundsReduction > 1.0:
                # When increasing, the center is the center of the original region
                center0 = bounds[0, idim] + dimesize_orig * 0.5
            else:
                # When decreasing, the center is the point I indicated
                center0 = center[idim]

            # Define bounds that are centered on the center
            lowerbound = center0 - dimesize_new * 0.5
            upperbound = center0 + dimesize_new * 0.5

            # No not go more than the original bounds
            if bounds_lim is not None:
                if not allow_extrap and lowerbound < bounds_lim[namedim][0]:
                    lowerbound = bounds_lim[namedim][0]
                    print("\t\t\t(Lower bounds clamped to original)")
                else:
                    atleastonemodification = True

                if not allow_extrap and upperbound > bounds_lim[namedim][1]:
                    upperbound = bounds_lim[namedim][1]
                    print("\t\t\t(Upper bounds clamped to original)")
                else:
                    atleastonemodification = True

            bounds_new[0, idim] = lowerbound
            bounds_new[1, idim] = upperbound

            if printYN:
                print(
                    f"\t\t\t* Original bound:\t [{bounds[0,idim]:.3f},{bounds[1,idim]:.3f}]"
                )
                print(
                    f"\t\t\t* Updated  bound:\t [{bounds_new[0,idim]:.3f},{bounds_new[1,idim]:.3f}]"
                )

    minimum_reached = all(minimum_dimension)
    if printYN and minimum_reached:
        print("All dimensions reduced to minimum value, hard break.")

    return bounds_new, minimum_reached, atleastonemodification


def correctInfs(train_X, train_Y):
    train_Xn, train_Yn = [], []
    for i in range(len(train_X)):
        if not np.isinf(np.abs(train_Y[i])).any():
            train_Xn.append(train_X[i])
            train_Yn.append(train_Y[i])
    print(
        f"--> {len(train_X)} points passed to BOstep, {len(train_Xn)} actually used (rest were infinity)"
    )
    train_X = np.array(train_Xn)
    train_Y = np.array(train_Yn)

    return train_X, train_Y


def updateMetrics(self, evaluatedPoints=1, IsThisAFreshIteration=True, position=0):
    """
    The logic here is that the residuals are for minimization

    Bring from GPU to CPU
    """

    # ------------------------------------------------------------------------------------------------------------
    # Update the immediate best residual tracking
    # ------------------------------------------------------------------------------------------------------------

    X = torch.from_numpy(self.train_X).to(self.dfT)
    Y = torch.from_numpy(self.train_Y).to(self.dfT)
    Yvar = torch.from_numpy(self.train_Ystd).to(self.dfT) ** 2

    if "steps" in self.__dict__:
        self.BOmetrics["overall"]["Residual"] = (
            -self.steps[-1].evaluators["objective"](Y).detach().cpu().numpy()
        )
        self.BOmetrics["overall"]["ResidualModeledLast"] = (
            -self.steps[-1].evaluators["residual_function"](X).detach().cpu().numpy()
        )

    else:
        print(
            "\t~ Cannot perform prediction at this iteration step, returning objective evaluation as acquisition",
            typeMsg="w",
        )
        self.BOmetrics["overall"]["Residual"] = (
            -self.lambdaSingleObjective(Y)[2].detach().cpu().numpy()
        )
        self.BOmetrics["overall"]["ResidualModeledLast"] = self.BOmetrics["overall"][
            "Residual"
        ]

    resi = self.BOmetrics["overall"]["Residual"]
    resiM = self.BOmetrics["overall"]["ResidualModeledLast"]

    # Absolute best
    zA_abs = np.nanmin(resi, axis=0)
    self.BOmetrics["overall"]["indBest"] = np.nanargmin(resi, axis=0)
    self.BOmetrics["overall"]["indBestModel"] = np.nanargmin(resiM, axis=0)

    self.BOmetrics["overall"]["xBest"] = (
        X[self.BOmetrics["overall"]["indBest"], :].detach().cpu()
    )
    self.BOmetrics["overall"]["yBest"] = (
        Y[self.BOmetrics["overall"]["indBest"], :].detach().cpu()
    )
    self.BOmetrics["overall"]["yVarBest"] = (
        Yvar[self.BOmetrics["overall"]["indBest"], :].detach().cpu()
    )

    # Best from last iteration
    zA = np.nanmin(resi[-evaluatedPoints:], axis=0)
    self.BOmetrics["overall"]["indBestLast"] = np.nanargmin(
        resi[-evaluatedPoints:], axis=0
    )
    zM = np.nanmin(resiM[-evaluatedPoints:], axis=0)
    self.BOmetrics["overall"]["indBestModelLast"] = np.nanargmin(
        resiM[-evaluatedPoints:], axis=0
    )

    self.BOmetrics["overall"]["xBestLast"] = X[
        self.BOmetrics["overall"]["indBestLast"], :
    ]
    self.BOmetrics["overall"]["yBestLast"] = Y[
        self.BOmetrics["overall"]["indBestLast"], :
    ]

    # Metric tracking of previous iterations

    if (
        len(self.BOmetrics["overall"]["ResidualModeledLast"])
        == self.Originalinitial_training
    ):
        ratio, metric = np.inf, 0.0
        label = "\t(Initial batch only)"
    else:
        zA_prev = np.nanmin(resi[:-evaluatedPoints], axis=0)
        self.BOmetrics["overall"]["indBestExceptLast"] = np.nanargmin(
            resi[:-evaluatedPoints], axis=0
        )
        zM_prev = np.nanmin(resiM[:-evaluatedPoints], axis=0)
        self.BOmetrics["overall"]["indBestModelExceptLast"] = np.nanargmin(
            resiM[:-evaluatedPoints], axis=0
        )

        ratio, metric, label = constructMetricsTR(zA, zM, zA_prev, zM_prev)

    # ------------------------------------------------------------------------------------------------------------
    # Print for convenience
    # ------------------------------------------------------------------------------------------------------------

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METRICS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if zA_abs == zA:
        print("- This is the best iteration so far (yay!)")
    else:
        print("- There was a better evaluation earlier (yikes!)")

    if "Initial batch only" in label:
        print(label)
    elif IsThisAFreshIteration:
        print(f"- BO fitness ratio = {ratio:.3f}, metric = {metric} ({label})")
    else:
        print(
            "- Note that this one was not fresh (e.g. post-correction), do not track metrics"
        )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # ------------------------------------------------------------------------------------------------------------
    # If this is a fresh iteration, update the metric
    # ------------------------------------------------------------------------------------------------------------

    if IsThisAFreshIteration:
        # Store current bounds
        self.BOmetrics["BoundsStorage"][position] = copy.deepcopy(self.bounds)

        # Store current ratio and metric
        self.BOmetrics["BOratio"][position] = ratio
        self.BOmetrics["BOmetric"][position] = metric

        # Store the x,y too, because of the problem with avoid points, when passing to surrogate, the indeces may be messed up
        self.BOmetrics["xBest_track"][position] = self.BOmetrics["overall"]["xBest"]

        self.BOmetrics["yBest_track"][position] = self.BOmetrics["overall"]["yBest"]
        self.BOmetrics["yVarBest_track"][position] = self.BOmetrics["overall"][
            "yVarBest"
        ]

        self.BOmetrics["BOmetric_it"][position] = (
            X.shape[0] - 1
        )  # Evaluation position until now


def constructMetricsTR(zA, zM, zA_prev, zM_prev):
    """
    zA          - High-fidelity best residual at this iteration
    zM          - Modeled best residual at this iteration
    zA_prev     - High-fidelity best residual at previous iteration
    zM_prev     - Model best residual at previous iteration (should be the same as zA_prev since it's trained there)

    Explanation of metrics:

            +-0.2 - Very bad accuracy
            +-1 - Bad accuracy
            +-2 - Good accuracy
            +-3 - Excellent accuracy

            >0 - Prediction got better
            <0 - Prediction got worse

    Note that the zA and zM do not necesarily reflect the actual residual because of weights

    """

    label = "Evaluated previous = {2:.2e} --> Predicted new = {1:.2e} --> Evaluated new = {0:.2e}; ".format(
        zA, zM, zA_prev, zM_prev
    )

    # Relative improvements (Positive if it has gotten better = lower residual)
    zA = (zA_prev - zA) / np.abs(zA_prev)
    zM = (zA_prev - zM) / np.abs(zM_prev)

    ratio = zA / zM

    # High fidelity got better
    if zA > 0.0:
        if ratio > 1.0:
            # Improvement was better than predicted by the model
            if ratio > 2.0:
                label += "we were lucky b/c we predicted less improvement than what it ended up happening (>x2 more)"
                metric = 1
            elif ratio > 1.5:
                label += "good accuracy and iteration improved more (x1.50 - x2) than expected"
                metric = 2
            else:
                label += "excellent accuracy and iteration improved somewhat more (x1.0 - x1.5) than expected"
                metric = 3
        elif ratio > 0.0:
            # Improvement was worse but in the same direction
            if ratio > 0.5:
                label += "excellent accuracy but iteration improved somewhat less (x0.5 - x1.0) than expected"
                metric = 3
            else:
                label += "somewhat good accuracy but iteration improved less (x0.0 - x0.5) than expected"
                metric = 2
        else:
            # The model followed the opposite trend, improved, but the model wanted to reduce
            label += "we had luck b/c it improved but model wanted to go worse"
            metric = 0.2

    # High fidelity got worse
    else:
        if ratio > 1.0:
            # Worsening was strongger than predicted by the model
            if ratio > 2.0:
                label += "model wanted to reduce improvement and got even worse"
                metric = -1
            elif ratio > 1.5:
                label += "good accuracy but getting worse as predicted"
                metric = -2
            else:
                label += "excellent accuracy but getting worse as predicted"
                metric = -3
        elif ratio > 0.0:
            # Improvement was worse but in the same direction
            if ratio > 0.5:
                label += "excellent accuracy but getting worse as predicted"
                metric = -3
            else:
                label += "good accuracy but getting worse as predicted"
                metric = -2
        else:
            # The model followed the opposite trend, got worse, but the model wanted to get better
            label += (
                "very bad accuracy b/c model wanted to improve and that did not happen"
            )
            metric = -0.2

    return ratio, metric, label


def plotTrustRegionInformation(self, fig=None):
    if fig is None:
        fig = plt.figure()

    grid = plt.GridSpec(nrows=3, ncols=2, hspace=0.4, wspace=0.4)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[0, 1], sharex=ax0)
    ax3 = fig.add_subplot(grid[1, 1], sharex=ax0)
    ax4 = fig.add_subplot(grid[2, 1], sharex=ax0)
    ax5 = fig.add_subplot(grid[2, 0], sharex=ax0)

    resA = self.BOmetrics["overall"]["Residual"]
    resM = self.BOmetrics["overall"]["ResidualModeledLast"]
    ratio = np.array(
        [self.BOmetrics["BOratio"][i] for i in self.BOmetrics["BOmetric_it"]]
    )  # self.BOmetrics['BOratio']#[1:]

    lim = -1
    for i in range(len(ratio)):
        if np.isnan(ratio[i]):
            lim = i
            break

    ratio = ratio[:lim]

    metrics = np.array(
        [self.BOmetrics["BOmetric"][i] for i in self.BOmetrics["BOmetric_it"]]
    )[
        :lim
    ]  # self.BOmetrics['BOmetric']#[1:]
    boundsX1 = np.array(
        [self.BOmetrics["BoundsStorage"][i] for i in self.BOmetrics["BOmetric_it"]]
    )[
        :lim
    ]  # self.BOmetrics['BoundsStorage']
    operat = np.array(
        [self.BOmetrics["TRoperation"][i] for i in self.BOmetrics["BOmetric_it"]]
    )[
        :lim
    ]  # self.BOmetrics['TRoperation'] #[1:]
    evaluations = np.array(
        [self.BOmetrics["BOmetric_it"][i] for i in self.BOmetrics["BOmetric_it"]]
    )[
        :lim
    ]  # [1:]

    size, center = [], []
    va = list(boundsX1[0].keys())[0]
    for i in range(len(boundsX1)):
        [m, M] = boundsX1[i][va]
        size.append(M - m)
        center.append(m + 0.5 * (M - m))
    size, center = np.array(size), np.array(center)

    ms = 3

    # ---------------------------------------------------------------
    # Plot evaluated residuals and the predicted by last model
    # ---------------------------------------------------------------

    x1 = np.arange(0, len(resA), 1)

    ax = ax0
    ax.plot(x1, resA, "-s", color="r", label="individual_models", markersize=ms)
    ax.plot(x1, resM, "-*", color="b", label="Predicted by last model", markersize=ms)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Residuals")
    ax.set_xlabel("Evaluations")
    ax.legend()
    ax.set_xlim([0, evaluations[-1] + 2])

    ax = ax1
    ax.plot(x1, resA, "-s", color="r", label="individual_models", markersize=ms)
    ax.plot(x1, resM, "-*", color="b", label="Predicted by last model", markersize=ms)
    ax.set_yscale("log")
    ax.set_ylabel("Residuals")
    ax.set_xlabel("Evaluations")
    ax.legend()

    # ---------------------------------------------------------------
    # Plot ratios and metrics
    # ---------------------------------------------------------------

    ax = ax2
    ax.plot(evaluations, ratio, "-s", color="r", label="Ratio", markersize=ms)
    ax.set_ylabel("Ratio")
    GRAPHICStools.drawLineWithTxt(
        ax,
        0,
        verticalalignment="bottom",
        label="as model",
        fromtop=0.1,
        orientation="horizontal",
        color="k",
        lw=1,
        ls="--",
        alpha=1.0,
        fontsize=10,
        fontweight="normal",
    )
    GRAPHICStools.drawLineWithTxt(
        ax,
        0,
        verticalalignment="top",
        label="opposite",
        fromtop=0.1,
        orientation="horizontal",
        color="k",
        lw=1,
        ls="--",
        alpha=1.0,
        fontsize=10,
        fontweight="normal",
    )

    ax = ax3
    ax.plot(evaluations, metrics, "-s", color="r", label="Metric", markersize=ms)
    ax.set_ylabel("Metric")
    ax.set_xlabel("Iterations")
    ax.set_ylim([-4, 4])

    alpha, extrathr = 0.1, 0.5

    GRAPHICStools.drawLineWithTxt(
        ax,
        0,
        verticalalignment="bottom",
        label="improved",
        fromtop=0.1,
        orientation="horizontal",
        color="b",
        lw=1,
        ls="--",
        alpha=1.0,
        fontsize=10,
        fontweight="normal",
    )
    GRAPHICStools.drawLineWithTxt(
        ax,
        0,
        verticalalignment="top",
        label="worsened",
        fromtop=0.1,
        orientation="horizontal",
        color="b",
        lw=1,
        ls="--",
        alpha=1.0,
        fontsize=10,
        fontweight="normal",
    )

    GRAPHICStools.gradientSPAN(
        ax,
        0,
        1 - extrathr,
        color="r",
        startingalpha=alpha,
        endingalpha=alpha,
        orientation="horizontal",
    )
    GRAPHICStools.gradientSPAN(
        ax,
        1 - extrathr,
        2 - extrathr,
        color="orange",
        startingalpha=alpha,
        endingalpha=alpha,
        orientation="horizontal",
    )
    GRAPHICStools.gradientSPAN(
        ax,
        2 - extrathr,
        4,
        color="green",
        startingalpha=alpha,
        endingalpha=alpha,
        orientation="horizontal",
    )

    GRAPHICStools.gradientSPAN(
        ax,
        -1 + extrathr,
        0,
        color="r",
        startingalpha=alpha,
        endingalpha=alpha,
        orientation="horizontal",
    )
    GRAPHICStools.gradientSPAN(
        ax,
        -2 + extrathr,
        -1 + extrathr,
        color="orange",
        startingalpha=alpha,
        endingalpha=alpha,
        orientation="horizontal",
    )
    GRAPHICStools.gradientSPAN(
        ax,
        -4,
        -2 + extrathr,
        color="green",
        startingalpha=alpha,
        endingalpha=alpha,
        orientation="horizontal",
    )

    ax = ax4
    ax.plot(evaluations, operat, "-s", color="r", markersize=ms)
    ax.set_title("TR operation to apply next")
    ax.set_ylabel("Factor")
    ax.set_xlabel("Iterations")
    ax.axhline(y=1.0, lw=0.5, ls="--", c="k")

    ax = ax5
    ax.plot(evaluations, size, "-s", color="r", markersize=ms)
    ax.set_ylabel(f'"{va}" bounds size')
    ax.yaxis.label.set_color("r")
    ax.set_title("Bounds used to find this optimum")

    axx = ax5.twinx()
    axx.plot(evaluations, center, "--s", color="b", markersize=ms)
    axx.set_ylabel(f'"{va}" bounds center')
    axx.yaxis.label.set_color("b")

    extr = 0.25
    for ax in [ax0, ax1, ax2, ax4, ax5]:
        for i in range(len(evaluations) - 2):
            GRAPHICStools.gradientSPAN(
                ax,
                evaluations[i + 1] - 0.5 + extr,
                evaluations[i + 2] - 0.5 - extr,
                color="b",
                startingalpha=0.2,
                endingalpha=0.2,
                orientation="vertical",
            )
        ax.axvline(x=evaluations[0] + 0.5, ls="--", c="m", lw=1)
    ax = ax3
    for i in evaluations:
        ax.axvline(x=i, ls="--", lw=0.2, c="b")
    ax.axvline(x=evaluations[0] + 0.5, ls="--", c="m", lw=1)
