import torch
import datetime
import copy
import botorch
import numpy as np
from mitim_tools.opt_tools.utils import SBOcorrections, TESTtools, SAMPLINGtools
from mitim_tools.misc_tools import IOtools, MATHtools, GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class fun_optimization:
    def __init__(self, stepSettings, evaluators, StrategyOptions):
        self.stepSettings = stepSettings
        self.evaluators = evaluators
        self.StrategyOptions = StrategyOptions

        self.dimOFs = 1  # len(self.stepSettings['name_objectives'])
        self.dimDVs = self.evaluators["GP"].train_X.shape[-1]

        # Pass the original bounds of the problem already to the fun class (they may be modified by boundsRefine)

        self.bounds = torch.zeros((2, len(self.evaluators["GP"].bounds))).to(
            self.evaluators["GP"].train_X
        )
        for i, ikey in enumerate(self.evaluators["GP"].bounds):
            self.bounds[0, i] = copy.deepcopy(self.evaluators["GP"].bounds[ikey][0])
            self.bounds[1, i] = copy.deepcopy(self.evaluators["GP"].bounds[ikey][1])

        # Define bounds_mod to search optimizer

        self.bounds_mod = self.bounds.clone()
        for i in range(self.bounds_mod.shape[-1]):
            tot = abs(self.bounds_mod[1, i] - self.bounds_mod[0, i])
            self.bounds_mod[0, i] -= self.StrategyOptions["AllowedExcursions"][0] * tot
            self.bounds_mod[1, i] += self.StrategyOptions["AllowedExcursions"][1] * tot

    def prep(self, number_optimized_points=1, xGuesses=None, seed=0):
        self.number_optimized_points = number_optimized_points
        self.xGuesses = xGuesses
        self.seed = seed

    def changeBounds(
        self, it_number, position_best_so_far, forceAllPointsInBounds=False
    ):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Do I want to optimize in different bounds than the original training box?
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if forceAllPointsInBounds:
            print(
                "\t- Optimization will be performed by extending bounds to emcompass all training points",
                typeMsg="i",
            )

            bounds = self.bounds.clone()
            for k in range(bounds.shape[-1]):
                bounds[0, k] = torch.min(
                    self.bounds[0, k], self.evaluators["GP"].train_X[0, k].min()
                )
                bounds[1, k] = torch.max(
                    self.bounds[1, k], self.evaluators["GP"].train_X[1, k].max()
                )

            self.bounds = bounds

        if (self.StrategyOptions["boundsRefine"] is not None) and (
            it_number >= self.StrategyOptions["boundsRefine"][0]
        ):
            relativeVariation = self.StrategyOptions["boundsRefine"][1]
            basePoint = self.StrategyOptions["boundsRefine"][2]

            if basePoint is None:
                basePoint = position_best_so_far

            print(
                f"\t- Optimization will be performed around {relativeVariation*100.0:.1f}% of the training point in position {basePoint}\n",
                typeMsg="i",
            )

            x_best = self.evaluators["GP"].train_X[basePoint]

            bounds = self.bounds.clone()
            for k in range(bounds.shape[-1]):
                if relativeVariation is not None:
                    magnitude = abs(x_best[k] * relativeVariation)

                    bounds[0, k], bounds[1, k] = (
                        x_best[k] - magnitude,
                        x_best[k] + magnitude,
                    )

            self.bounds = bounds

        # Define bounds_mod to search optimizer

        self.bounds_mod = self.bounds.clone()
        for i in range(self.bounds_mod.shape[-1]):
            tot = abs(self.bounds_mod[1, i] - self.bounds_mod[0, i])
            self.bounds_mod[0, i] -= self.StrategyOptions["AllowedExcursions"][0] * tot
            self.bounds_mod[1, i] += self.StrategyOptions["AllowedExcursions"][1] * tot

    def optimize(
        self,
        method_for_optimization,
        previous_solutions=None,
        best_performance_previous_iteration=None,
    ):
        """
        Possible Methods
        ----------------
        As output of this, x_opt is complete set of unnormalized DVs, while y_opt_residual is a pseudo-metric
        of goodness (residual, to minimize)

        Functions are to be maximized

        Guesses must be unnormalized. I normalize them here for specific optimizers that need normalized inputs
        """

        # ** OPTIMIZE **
        x_opt2, y_opt_residual2, z_opt2, acq_evaluated = method_for_optimization(
            self, writeTrajectory=True
        )
        # **********************************************************************

        info = storeInfo(x_opt2, acq_evaluated, self)

        # ----------------------------------------------------------------
        # Concatenate to previous solutions, check within bounds and order
        # ----------------------------------------------------------------

        if previous_solutions is not None:
            x_opt, y_opt_residual, z_opt = previous_solutions

            x_opt, y_opt_residual, z_opt = pointSelection(
                x_opt2,
                y_opt_residual2,
                z_opt2,
                self,
                x_opt,
                y_opt_residual,
                z_opt,
                maxExtrapolation=self.StrategyOptions["AllowedExcursions"],
                ToleranceNiche=self.StrategyOptions["ToleranceNiche"],
            )

        else:
            x_opt, y_opt_residual, z_opt = (
                x_opt2,
                y_opt_residual2,
                z_opt2,
            )

        return x_opt, y_opt_residual, z_opt, info


def optAcq(
    stepSettings={},
    evaluators={},
    StrategyOptions={},
    best_points=5,
    optimization_sequence=["ga"],
    it_number=1,
    seed=0,
    position_best_so_far=-1,
    forceAllPointsInBounds=False,
):
    """
    Everything returned here is unnormalized

    This function will launch a number of optimization strategies, coupled to each other
    (they send best solutions). As starting point, we use the training set (which means
    that it contains the best of the previous iteration) and the previous GA fronts (with
    points that were not necesarily added to the optimal solution).

    The philosophy is that I always carry out a number of solutions (starting from training)
    that each optimizer finds, and I order from best to worse in residue. At the end, I remove
    the training and provide as solution, where I add random points to fulfill the number of
    points requirement.

    The functions provided in evaluators must be for maximization
    """

    time1 = datetime.datetime.now()
    print(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )
    print(
        f" Posterior Optimization (GPs trained with {evaluators['GP'].train_X.shape[0]}/{evaluators['GP'].train_X.shape[0]+len(evaluators['GP'].avoidPoints)} points), {time1.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instance fun
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fun = fun_optimization(stepSettings, evaluators, StrategyOptions)

    fun.changeBounds(
        it_number, position_best_so_far, forceAllPointsInBounds=forceAllPointsInBounds
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find some initial conditions
    #   Complete initial guess with training and remove if outside bounds of this iteration
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    x_opt, y_opt_residual, z_opt = prepFirstStage(
        fun, checkBounds=True, seed=it_number + seed
    )
    best_performance_previous_iteration = y_opt_residual[0].item()
    x_initial = x_opt.clone()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Optimize function (accounting for maximum posterior only)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    infoOptimization = []

    for i, optimizers in enumerate(optimization_sequence):
        time2 = datetime.datetime.now()
        print(
            f'\n\n~~~~~~~~~~~ Optimization stage {i+1}: {optimization_sequence[i]} ({time2.strftime("%Y-%m-%d %H:%M:%S")}) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n'
        )

        # Prepare (run more now to find more solutions, more diversity, even if later best_points is 1)

        if optimizers == "ga":
            from mitim_tools.opt_tools.optimizers.GAtools import findOptima

            number_optimized_points = np.max([best_points, 32])
        elif optimizers == "botorch":
            from mitim_tools.opt_tools.optimizers.BOTORCHoptim import findOptima

            number_optimized_points = best_points
        elif "root" in optimizers:
            from mitim_tools.opt_tools.optimizers.ROOTtools import findOptima

            number_optimized_points = int(optimizers.split("_")[1])

        fun.prep(
            number_optimized_points=number_optimized_points,
            xGuesses=x_opt,
            seed=it_number + seed,
        )

        # *********** Optimize
        x_opt, y_opt_residual, z_opt, info = fun.optimize(
            findOptima,
            previous_solutions=[x_opt, y_opt_residual, z_opt],
            best_performance_previous_iteration=best_performance_previous_iteration,
        )
        # ****************************************************************************************

        infoOptimization.append(
            {
                "method": optimization_sequence[i],
                "info": info,
                "elapsed_seconds": IOtools.getTimeDifference(time2, niceText=False),
                "bounds": fun.bounds,
            }
        )

    # ~~~~ Clean-up set and complete with actual OFs

    x_opt, y_opt, y_opt_residual, z_opt = cleanupCandidateSet(
        x_opt,
        y_opt_residual,
        z_opt,
        fun,
        x_initial,
        best_points=best_points,
        ToleranceNiche=StrategyOptions["ToleranceNiche"],
        RandomRangeBounds=StrategyOptions["RandomRangeBounds"],
        it_number=it_number,
        seed=seed,
    )

    infoOptimization.append(
        {"method": "cleanup", "info": storeInfo(x_opt, torch.Tensor([]), fun)}
    )

    print(
        f"\t~~~~~~~ Optimization workflow took {IOtools.getTimeDifference(time1)}, and it passes {x_opt.shape[0]} optima to next MITIM iteration"
    )

    return x_opt, infoOptimization


def pointSelection(
    x_opt,
    y_res,
    z_opt,
    fun,
    x_opt_previous,
    y_res_previous,
    z_opt_previous,
    maxExtrapolation=[0.0, 0.0],
    ToleranceNiche=None,
):
    # Remove points if they are outside of bounds by more than margin
    x_opt, y_res, z_opt = pointsOperation_bounds(
        x_opt, y_res, z_opt, fun, maxExtrapolation=maxExtrapolation
    )

    # Concatenate to list of possible best (previous + new)
    x_opt, y_res, z_opt = pointsOperation_concat(
        x_opt, y_res, z_opt, x_opt_previous, y_res_previous, z_opt_previous
    )

    # Order by best
    x_opt, y_res, z_opt, _ = pointsOperation_order(x_opt, y_res, z_opt, fun)

    # Apply niche
    x_opt, y_res, z_opt = pointsOperation_niche(
        x_opt, y_res, z_opt, fun, ToleranceNiche=ToleranceNiche
    )

    # Summarize
    method = TESTtools.identifyType(z_opt[0].item())
    print(
        f"\t- New candidate set has an acquisition spanning from {-y_res[0]:.5f} to {-y_res[-1]:.5f}. Best found by method = {method}"
    )
    print("\t- " + TESTtools.summaryTypes(z_opt))

    return x_opt, y_res, z_opt


def pointsOperation_concat(
    x_opt2, y_opt_residual2, z_opt2, x_opt_previous, y_opt_previous, z_opt_previous
):
    x_opt = torch.cat((x_opt_previous, x_opt2)).to(x_opt2)
    y_opt_residual = torch.cat((y_opt_previous, y_opt_residual2)).to(x_opt2)
    z_opt = torch.cat((z_opt_previous, z_opt2)).to(x_opt2)

    print(
        f"\t- Previous solution set had {x_opt_previous.shape[0]} points, this optimization step adds {x_opt2.shape[0]} new points. Optimization so far has found {x_opt.shape[0]} candidate optima"
    )

    return x_opt, y_opt_residual, z_opt


def pointsOperation_order(x_opt, y_opt_residual, z_opt, fun):
    if (x_opt.shape[0] > 0) and ("lambdaSelect" in fun.evaluators):
        indeces = fun.evaluators["lambdaSelect"](x_opt, y_opt_residual)

        x_opt = x_opt[indeces]
        y_opt_residual = y_opt_residual[indeces]
        if z_opt is not None:
            z_opt = z_opt[indeces]
    else:
        indeces = torch.Tensor([0])

    return x_opt, y_opt_residual, z_opt, indeces


def pointsOperation_niche(x_opt1, y_opt_residual1, z_opt1, fun, ToleranceNiche=None):
    x_opt, y_opt_residual, z_opt = (
        x_opt1.clone(),
        y_opt_residual1.clone(),
        z_opt1.clone(),
    )

    # Normalizations now occur inside model, extract such function ************************
    normalizeVar = botorch.models.transforms.input.Normalize(
        x_opt.shape[-1], bounds=None
    )
    denormalizeVar = normalizeVar._untransform
    # *************************************************************************************

    if ToleranceNiche is not None:
        x_opt_Norm = normalizeVar(x_opt)

        # Niches
        _, z_opt = MATHtools.applyNiche(
            x_opt_Norm.cpu(), z_opt.unsqueeze(1).cpu(), tol=ToleranceNiche
        )
        x_opt_Norm, y_opt_residual = MATHtools.applyNiche(
            x_opt_Norm.cpu(), y_opt_residual.cpu(), tol=ToleranceNiche
        )
        # ------
        x_opt_Norm, y_opt_residual, z_opt = (
            x_opt_Norm.to(fun.stepSettings["dfT"]),
            y_opt_residual.to(fun.stepSettings["dfT"]),
            z_opt.to(fun.stepSettings["dfT"]),
        )

        z_opt = z_opt[:, 0]
        removedNum = x_opt.shape[0] - x_opt_Norm.shape[0]
        x_opt = denormalizeVar(x_opt_Norm)

        if removedNum > 0:
            print(
                f"\t- Removed {removedNum} points because I applied a niching tolerance in relative [0,1] bounds of {ToleranceNiche*100:.1f}%"
            )

    return x_opt, y_opt_residual, z_opt


def pointsOperation_bounds(
    x_opt, y_opt_residual, z_opt, fun, maxExtrapolation=[0.0, 0.0]
):
    """
    maxExtrapolation = None: Allow any point
    """

    bounds = fun.bounds

    if y_opt_residual is None:
        y_opt_residual = x_opt.clone()
    if z_opt is None:
        z_opt = x_opt.clone()[:, 0]

    x_opt_inbounds, y_opt_inbounds, z_opt_inbounds = (
        torch.Tensor().to(fun.stepSettings["dfT"]),
        torch.Tensor().to(fun.stepSettings["dfT"]),
        torch.Tensor().to(fun.stepSettings["dfT"]),
    )
    x_removeds = torch.Tensor().to(fun.stepSettings["dfT"])
    for i in range(x_opt.shape[0]):
        if maxExtrapolation is not None:
            insideBounds = TESTtools.checkSolutionIsWithinBounds(
                x_opt[i], bounds, maxExtrapolation=maxExtrapolation
            )
        else:
            insideBounds = TESTtools.checkSolutionIsWithinBounds(
                x_opt[i], bounds, maxExtrapolation=[0.0, 0.0]
            )
            if not insideBounds:
                print(
                    f"\t- Point #{i} is not inside bounds, but I am allowing it to exists"
                )
            insideBounds = True

        if insideBounds:
            x_opt_inbounds = torch.cat(
                (x_opt_inbounds, x_opt[i].unsqueeze(0)), axis=0
            ).to(fun.stepSettings["dfT"])
            y_opt_inbounds = torch.cat(
                (y_opt_inbounds, y_opt_residual[i].unsqueeze(0)), axis=0
            ).to(fun.stepSettings["dfT"])
            z_opt_inbounds = torch.cat(
                (z_opt_inbounds, z_opt[i].unsqueeze(0)), axis=0
            ).to(fun.stepSettings["dfT"])
        else:
            x_removeds = torch.cat((x_removeds, x_opt[i].unsqueeze(0)), axis=0).to(
                fun.stepSettings["dfT"]
            )

    txt = (
        f" (allowed exploration of [{maxExtrapolation[0]*100.0:.1f}%,{maxExtrapolation[1]*100.0:.1f}%] outside bounds)"
        if ((maxExtrapolation is not None) and (maxExtrapolation[1] > 0))
        else ""
    )

    numRemoved = x_removeds.shape[0]
    if numRemoved > 0:
        print(
            f"\t- Postprocessing removed {numRemoved}/{x_opt.shape[0]} points b/c they went outside bounds{txt}"
        )
        IOtools.printPoints(x_removeds, numtabs=2)
    else:
        print(
            f"\t- No points removed b/c they are inside bounds or they were allowed{txt}"
        )

    return x_opt_inbounds, y_opt_inbounds, z_opt_inbounds


def pointsOperation_common(x_opt, y_opt_residual, z_opt, fun, best_points=None):
    evaluators = fun.evaluators
    stepSettings = fun.stepSettings

    xopt_new, yopt_new, zopt_new = (
        torch.Tensor().to(x_opt),
        torch.Tensor().to(y_opt_residual),
        torch.Tensor().to(z_opt),
    )

    for i in range(z_opt.shape[0]):
        if z_opt[i] != 1.0:
            xopt_new = torch.cat((xopt_new, x_opt[i].unsqueeze(0)), axis=0).to(
                stepSettings["dfT"]
            )
            yopt_new = torch.cat((yopt_new, y_opt_residual[i].unsqueeze(0)), axis=0).to(
                stepSettings["dfT"]
            )
            zopt_new = torch.cat((zopt_new, z_opt[i].unsqueeze(0)), axis=0).to(
                stepSettings["dfT"]
            )

    print(
        f"\t- Removed {x_opt.shape[0]-xopt_new.shape[0]} points from final candidate set because they belonged to training set (already evaluated), now candidate set has {xopt_new.shape[0]} points"
    )
    x_opt, y_opt_residual, z_opt = xopt_new, yopt_new, zopt_new

    # Even if they weren't training, the opt procedures may have generated identical ones
    xopt_new, yopt_new, zopt_new = (
        torch.Tensor().to(x_opt),
        torch.Tensor().to(y_opt_residual),
        torch.Tensor().to(z_opt),
    )
    for i in range(x_opt.shape[0]):
        equal = False
        for j in range(evaluators["GP"].train_X.shape[0]):
            equal = equal or MATHtools.arePointsEqual(
                x_opt[i], evaluators["GP"].train_X[j]
            )
        if not equal:
            xopt_new = torch.cat((xopt_new, x_opt[i].unsqueeze(0)), axis=0).to(
                stepSettings["dfT"]
            )
            yopt_new = torch.cat((yopt_new, y_opt_residual[i].unsqueeze(0)), axis=0).to(
                stepSettings["dfT"]
            )
            zopt_new = torch.cat((zopt_new, z_opt[i].unsqueeze(0)), axis=0).to(
                stepSettings["dfT"]
            )

    if x_opt.shape[0] - xopt_new.shape[0] > 0:
        print(
            f"\t- Removed {x_opt.shape[0]-xopt_new.shape[0]} points from final candidate set because they were identical to training set, now candidate set has {xopt_new.shape[0]} points"
        )
    x_opt, y_opt_residual, z_opt = xopt_new, yopt_new, zopt_new

    if best_points is not None:
        if x_opt.shape[0] > best_points:
            removedNum = x_opt.shape[0] - best_points
            x_opt, y_opt_residual, z_opt = (
                x_opt[:-removedNum],
                y_opt_residual[:-removedNum],
                z_opt[:-removedNum],
            )
            print(
                f"\t- Removed {removedNum} points because they were more than I wanted: {x_opt.shape[0]}"
            )

    if len(x_opt) == 0:
        x_opt = y_opt_residual = z_opt = torch.Tensor([[]]).to(evaluators["GP"].train_X)

    return x_opt, y_opt_residual, z_opt


def pointsOperation_random(
    x_opt,
    y_opt_residual,
    z_opt,
    fun,
    best_points=1,
    RandomRangeBounds=0,
    it_number=1,
    seed=0,
):
    evaluators = fun.evaluators
    stepSettings = fun.stepSettings
    randomSeed = seed + it_number

    if x_opt.nelement() == 0:
        print(
            f"\t- Filling space with {best_points} random (LHS) points becaue optimization method found none"
        )
        draw_bounds = fun.bounds
        x_opt = SAMPLINGtools.LHS(best_points, draw_bounds, seed=randomSeed)
        y_opt_residual = evaluators["acq_function"](x_opt.unsqueeze(1)).detach()
        z_opt = torch.ones(x_opt.shape[0]) * 2

    elif RandomRangeBounds > 0:
        x_optRandom, y_optRandom, z_optRandom = (
            x_opt.clone(),
            y_opt_residual.clone(),
            z_opt.clone(),
        )
        ib = 0  # Around the best, which is the first one since I have ordered them

        if (x_optRandom.shape[0] < best_points) and stepSettings["optimization_options"][
            "ensure_new_points"
        ]:
            print(
                f"\n\t ~~~~ Completing set with {best_points-x_optRandom.shape[0]} extra points around ({RandomRangeBounds*100}%) the best predicted point"
            )
            draw_bounds, _, _ = SBOcorrections.factorBounds(
                center=x_optRandom[ib],
                bounds=fun.bounds,
                BoundsReduction=RandomRangeBounds,
                printYN=False,
                bounds_lim=stepSettings["bounds_orig"],
            )

            new_opt = SAMPLINGtools.LHS(
                best_points - x_optRandom.shape[0], draw_bounds, seed=randomSeed
            )
            new_y = evaluators["acq_function"](new_opt.unsqueeze(1)).detach()
            x_optRandom = torch.cat((x_optRandom, new_opt)).to(stepSettings["dfT"])
            y_optRandom = torch.cat((y_optRandom, new_y)).to(stepSettings["dfT"])
            new_z = torch.ones(x_optRandom.shape[0]).to(stepSettings["dfT"]) * 2
            z_optRandom = torch.cat((z_optRandom, new_z), axis=0).to(
                stepSettings["dfT"]
            )

        x_opt, y_opt_residual, z_opt = x_optRandom, y_optRandom, z_optRandom

    return x_opt, y_opt_residual, z_opt


def cleanupCandidateSet(
    x_opt,
    y_opt_residual,
    z_opt,
    fun,
    x_initial,
    best_points=1,
    ToleranceNiche=None,
    RandomRangeBounds=0,
    it_number=1,
    seed=0,
):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("- Optimization clean-up phase")

    # Remove training data from optimization solution

    x_opt, y_opt_residual, z_opt = pointsOperation_common(
        x_opt, y_opt_residual, z_opt, fun, best_points=best_points
    )

    if x_opt.nelement() > 0:
        method = TESTtools.identifyType(z_opt[0].item())
        print(
            f"\t- This semi-final candidate ({y_opt_residual.shape[0]} points) set has an acquisition spanning from {-y_opt_residual[0]:.5f} to {-y_opt_residual[-1]:.5f}. Best found by method = {method}"
        )
        print("\t- " + TESTtools.summaryTypes(z_opt))

    # ~~~~  Complete with random AROUND the best)
    x_opt, y_opt_residual, z_opt = pointsOperation_random(
        x_opt,
        y_opt_residual,
        z_opt,
        fun,
        best_points=best_points,
        RandomRangeBounds=RandomRangeBounds,
        it_number=it_number,
        seed=seed,
    )

    # ~~~~ Order array
    x_opt, y_opt_residual, z_opt, _ = pointsOperation_order(
        x_opt, y_opt_residual, z_opt, fun
    )

    # ~~~~ Summarize Optimization
    bestPseudo_y = summarizeSituation(x_initial, fun, x_opt)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Completion
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~ Complete y_opt with non-optimizable objectives (pure predictive) and update calibrations

    print(" --> Completing non-optimized objectives")

    with torch.no_grad():
        _, _, _, y_opt = fun.evaluators["residual_function"](
            x_opt, outputComponents=True
        )

    # ~~~~~~~~

    method = TESTtools.identifyType(z_opt[0].item())
    print(
        f"\t- This semi-final candidate ({y_opt_residual.shape[0]} points) set has an acquisition spanning from {-y_opt_residual[0]:.5f} to {-y_opt_residual[-1]:.5f}. Best found by method = {method}"
    )

    print("\n\t- Going out from the clean-up phase")

    return x_opt, y_opt, y_opt_residual, z_opt


def storeInfo(x_opt, acq_evaluated, fun):
    """
    x:      DVs
    y_res:  Residue used in optimization

    yFun:   Objective function
    yCal:   Calibration

    y:      Individual residues

    """

    infoOPT = {}

    x_ini = fun.xGuesses

    # Start
    y_ini_res = summarizeSituation(x_ini, fun, printYN=False)
    y, y1, y2, _ = fun.evaluators["residual_function"](x_ini, outputComponents=True)

    infoOPT["x_start"] = copy.deepcopy(x_ini.cpu().numpy())
    infoOPT["y_res_start"] = copy.deepcopy(y_ini_res.cpu().numpy())
    infoOPT["yFun_start"] = copy.deepcopy(y1.detach().cpu().numpy())
    infoOPT["yCal_start"] = copy.deepcopy(y2.detach().cpu().numpy())
    infoOPT["y_start"] = copy.deepcopy(y.detach().cpu().numpy())

    # End
    if x_opt.shape[0] > 0:
        y_opt_res = summarizeSituation(x_opt, fun, printYN=False)
        y, y1, y2, _ = fun.evaluators["residual_function"](x_opt, outputComponents=True)

        infoOPT["x"] = copy.deepcopy(x_opt.cpu().numpy())
        infoOPT["y_res"] = copy.deepcopy(y_opt_res.cpu().numpy())
        infoOPT["yFun"] = copy.deepcopy(y1.detach().cpu().numpy())
        infoOPT["yCal"] = copy.deepcopy(y2.detach().cpu().numpy())
        infoOPT["y"] = copy.deepcopy(y.detach().cpu().numpy())

    infoOPT["acq_evaluated"] = acq_evaluated

    return infoOPT


def plotInfo(
    infoOPT,
    axTraj=None,
    axDVs=None,
    axOFs=None,
    axR=None,
    label="",
    color="b",
    ms=5,
    axislabels_x=None,
    axislabels_y=None,
    plotStart=False,
    bounds=None,
    alpha=1.0,
    axDVs_r=None,
    axOFs_r=None,
    boundsThis=None,
    it_start=0,
    xypair=None,
):

    if xypair is None:
        xypair = []

    # Ranges ----------
    if plotStart:
        if bounds is not None:
            axDVs_r.plot(
                axislabels_x, np.array(bounds)[:, 0], "o-", lw=0.5, c="g", markersize=0
            )
            axDVs_r.plot(
                axislabels_x, np.array(bounds)[:, 1], "o-", lw=0.5, c="g", markersize=0
            )
            axDVs_r.fill_between(
                axislabels_x,
                np.array(bounds)[:, 0],
                np.array(bounds)[:, 1],
                facecolor="g",
                alpha=0.05,
                label="Total Range",
            )
        if boundsThis is not None:
            axDVs_r.plot(
                axislabels_x, boundsThis[:, 0], "o-", lw=0.5, c="orange", markersize=0
            )
            axDVs_r.plot(
                axislabels_x, boundsThis[:, 1], "o-", lw=0.5, c="orange", markersize=0
            )
            axDVs_r.fill_between(
                axislabels_x,
                boundsThis[:, 0],
                boundsThis[:, 1],
                facecolor="orange",
                alpha=0.05,
                label="Reduced Range",
            )

        for i in range(infoOPT["x_start"].shape[0]):
            axDVs_r.plot(
                axislabels_x,
                infoOPT["x_start"][i, :],
                "s-",
                lw=0.5,
                c=color,
                markersize=3,
                label=label if i == 0 else "",
            )
        yr = np.abs(infoOPT["yFun_start"] - infoOPT["yCal_start"])
        for i in range(yr.shape[0]):
            axOFs_r.plot(axislabels_y, yr[i, :], "s-", lw=0.5, c=color, markersize=3)
        axOFs_r.axhline(y=infoOPT["y_res_start"][0], ls="--", lw=2, c=color)

    else:
        for i in range(infoOPT["x"].shape[0]):
            axDVs_r.plot(
                axislabels_x,
                infoOPT["x"][i, :],
                "s-",
                lw=0.5,
                c=color,
                markersize=1,
                label=label if i == 0 else "",
            )
        yr = np.abs(infoOPT["yFun"] - infoOPT["yCal"])
        for i in range(yr.shape[0]):
            axOFs_r.plot(axislabels_y, yr[i, :], "s-", lw=0.5, c=color, markersize=3)
        axOFs_r.axhline(y=infoOPT["y_res"][i], ls="--", lw=2, c=color)

    # ----------------

    if plotStart:
        # ---------- Plot DVs
        GRAPHICStools.plotMultiVariate(
            infoOPT["x_start"],
            axs=axDVs,
            marker="s",
            markersize=ms,
            color=color,
            label=label,
            axislabels=axislabels_x,
            bounds=bounds,
            alpha=alpha,
            boundsThis=boundsThis,
        )
        # ---------- Plot Residue
        GRAPHICStools.plotMultiVariate(
            np.transpose(np.atleast_2d(infoOPT["y_res_start"])),
            axs=axR,
            marker="s",
            markersize=ms,
            color=color,
            label=label,
            axislabels=["acquisition"],
            alpha=alpha,
        )
        # ---------- Plot Calibration errors
        GRAPHICStools.plotMultiVariate(
            np.abs(infoOPT["yFun_start"] - infoOPT["yCal_start"]),
            axs=axOFs,
            marker="s",
            markersize=ms,
            color=color,
            label=label,
            axislabels=axislabels_y,
            alpha=alpha,
        )

    else:
        # ---------- Plot DVs
        GRAPHICStools.plotMultiVariate(
            infoOPT["x"],
            axs=axDVs,
            marker="s",
            markersize=ms,
            color=color,
            label=label,
            axislabels=axislabels_x,
            alpha=alpha,
        )
        # ---------- Plot Residue
        GRAPHICStools.plotMultiVariate(
            np.transpose(np.atleast_2d(infoOPT["y_res"])),
            axs=axR,
            marker="s",
            markersize=ms,
            color=color,
            label=label,
            axislabels=["acquisition"],
            alpha=alpha,
        )
        # ---------- Plot Calibration errors
        GRAPHICStools.plotMultiVariate(
            np.abs(infoOPT["yFun"] - infoOPT["yCal"]),
            axs=axOFs,
            marker="s",
            markersize=ms,
            color=color,
            label=label,
            axislabels=axislabels_y,
            alpha=alpha,
        )

    # Traj ----------

    if not plotStart:
        y = (
            -infoOPT["acq_evaluated"]
            if "acq_evaluated" in infoOPT
            else -infoOPT["y_res"]
        )
    else:
        y = -infoOPT["y_res_start"]

    if not plotStart:
        yo = -infoOPT["y_res"][0]
    else:
        yo = -infoOPT["y_res_start"][0]

    x_origin = 0 + it_start
    x_last = len(y) - 1 + it_start

    x = np.linspace(x_origin, x_last, len(y))
    axTraj.plot(x, y, "-o", c=color, markersize=1, lw=0.5, alpha=0.5, label=label)

    xo, summ = (x_last, len(y)) if len(y) > 0 else (x_last, 1)

    xypair.append([xo, yo])
    it_start += summ

    return it_start, xypair


def prepFirstStage(fun, previousGA=None, numMax=None, checkBounds=False, seed=0):
    """
    x_opt is unnormalized
    Output is unnormalized

    if numMax is a number (not None), fill up with these random points

    z_opt indicates:
            0 - From previous iteration
            1 - Trained
            2 - Random
    """

    # ~~~~~~~~~ Guessed Population (from previous GA) ~~~~~~~~~

    if previousGA is not None and "Paretos_x_unnormalized" in previousGA:
        x_opt = previousGA["Paretos_x_unnormalized"]
    else:
        x_opt = []

    # --------------------------------------------------
    # Add to the previous optimum, all the trained points
    # --------------------------------------------------

    # Consider also the training points
    x_train = fun.evaluators["GP"].train_X
    z_train = torch.ones(x_train.shape[0]).to(x_train)

    # Concatenate together
    if len(x_opt) == 0:
        x_opt = torch.Tensor([]).to(x_train)
        z_opt = torch.Tensor([]).to(x_train)
    if type(x_opt) == np.ndarray:
        x_opt = torch.from_numpy(x_opt).to(fun.stepSettings["dfT"])
        z_opt = torch.zeros(x_opt.shape[0]).to(fun.stepSettings["dfT"])
    xGuesses = torch.cat((x_opt, x_train), axis=0).to(fun.stepSettings["dfT"])
    z_opt = torch.cat((z_opt, z_train), axis=0).to(fun.stepSettings["dfT"])

    if numMax is not None:
        howmany = np.max([0, numMax - xGuesses.shape[0]])
        txt = f", getting {howmany} more randomnly"
    else:
        howmany = 0
        txt = ""
    print(
        f"\t- {xGuesses.shape[0]} guesses already ({x_train.shape[0]} trained, {x_opt.shape[0]} predicted by previous MITIM iteration){txt}"
    )

    # --------------------------------------------------
    # Make sure that they are in between bounds (because in this step, the bounds may have changed if I processed it
    #                                            or if I have allowed ROOT to go outside in the previous iteration)
    # --------------------------------------------------
    if checkBounds:
        xGuesses_new = torch.Tensor().to(fun.stepSettings["dfT"])
        z_opt_new = torch.Tensor().to(fun.stepSettings["dfT"])
        for i in range(xGuesses.shape[0]):
            if TESTtools.checkSolutionIsWithinBounds(xGuesses[i], fun.bounds):
                xGuesses_new = torch.cat(
                    (xGuesses_new, xGuesses[i].unsqueeze(0)), axis=0
                ).to(fun.stepSettings["dfT"])
                z_opt_new = torch.cat((z_opt_new, z_opt[i].unsqueeze(0)), axis=0).to(
                    fun.stepSettings["dfT"]
                )

        print(
            f"\t~~ Keeping (inside bounds) {xGuesses_new.shape[0]} points from the total of {xGuesses.shape[0]}"
        )

        xGuesses = xGuesses_new
        z_opt = z_opt_new

    # --------------------------------------------------
    # Add random until filling up max number requested
    # --------------------------------------------------

    if howmany > 0:
        LHSdraw = torch.from_numpy(
            SAMPLINGtools.LHS(howmany, fun.bounds, seed=seed)
        ).to(fun.stepSettings["dfT"])
        xGuesses = torch.cat((xGuesses, LHSdraw), axis=0).to(fun.stepSettings["dfT"])

        z_opt_draw = torch.ones(LHSdraw.shape[0]).to(fun.stepSettings["dfT"]) * 2
        z_opt = torch.cat((z_opt, z_opt_draw), axis=0).to(fun.stepSettings["dfT"])

    if len(xGuesses) == 0:
        print("* Initial points equal to zero, will likely fail", typeMsg="q")

    print(
        "********************** Status update after prep phase **********************************"
    )
    y_opt_residual = summarizeSituation(xGuesses, fun)
    print(
        "****************************************************************************************"
    )

    # Order by best
    xGuesses, y_opt_residual, z_opt, _ = pointsOperation_order(
        xGuesses, y_opt_residual, z_opt, fun
    )

    return xGuesses, y_opt_residual, z_opt


def summarizeSituation(previous_x, fun, new_x=None, printYN=True):
    evaluators, stepSettings = fun.evaluators, fun.stepSettings

    # ------------------------------------------------------------------------
    # Previous iteration
    # ------------------------------------------------------------------------

    previous_y = -evaluators["residual_function"](previous_x).detach()
    previous_y_acq = evaluators["acq_function"](previous_x.unsqueeze(1)).detach()
    previous_yReal = -evaluators["objective"](evaluators["GP"].train_Y).detach()

    best_y = previous_y.min(axis=0)[0]
    best_yReal = previous_yReal.min(axis=0)[0]
    best_y_acq = previous_y_acq.max(axis=0)[0]

    # Print Info

    if printYN:
        print(
            f"\t- Previous iteration had a best minimization-based objective (residue) of {best_y:.4e} (note that real trained value is {best_yReal:.4e})"
        )
        print(
            f"\t- Previous iteration had a best maximization-based acquisition of {best_y_acq:.4e} (remember it may be MC, some randomness)"
        )

    # ------------------------------------------------------------------------
    # New iteration
    # ------------------------------------------------------------------------

    if new_x is not None:
        # Objective
        new_y = -evaluators["residual_function"](new_x).detach()
        new_y_acq = evaluators["acq_function"](new_x.unsqueeze(1)).detach()
        new_yReal = -evaluators["objective"](evaluators["GP"].train_Y).detach()

        new_best_y = new_y.min(axis=0)[0]
        new_best_yReal = new_yReal.min(axis=0)[0]
        new_best_y_acq = new_y_acq.max(axis=0)[0]

        # Print Info

        impr2 = new_best_y_acq / best_y_acq
        impr = (new_best_y_acq - best_y_acq) / best_y_acq.abs().cpu() * 100

        if printYN:
            print(
                f"\t- Previous best acquisition: {best_y_acq:.4e}, predicted new acquisition: {new_best_y_acq:.4e}. (predicted {impr:.1f}% improvement, {impr2:.1e} residual factor)",
                typeMsg="i",
            )

        # Return value of acquisition, which is what I'm optimizing here

        return new_y_acq

    else:
        if printYN:
            print(
                f"\t- Best acquisition so far: {best_y_acq:.4e}. The following procedure will aim at maximizing this value"
            )

        # Return value of acquisition, which is what I'm optimizing here

        return previous_y_acq


def untransformation_loop(X_transformed, input_transform, x0):
    def evaluator_losses(x):
        X = torch.from_numpy(x).to(X_transformed).unsqueeze(0).requires_grad_(True)
        loss = (input_transform(X) - X_transformed).square().mean()
        V = torch.autograd.grad(loss, X)[0]

        return loss.detach().numpy(), V.detach().numpy()

    from scipy.optimize import minimize

    sol = minimize(
        evaluator_losses,
        x0.numpy()[0, :],
        method="L-BFGS-B",
        jac=True,
        options={"disp": 1, "gtol": 1e-15, "ftol": 1e-15},
    )

    x_new = torch.Tensor(sol.x).to(X_transformed).unsqueeze(0)

    print("Losses per dimension:")
    print(X_transformed - input_transform(x_new))
    maxRel = (
        ((X_transformed - input_transform(x_new)).abs() / X_transformed.abs())
        .max()
        .item()
    )
    print(
        f"Max relative loss of untransformation: {maxRel*100:.1e}%",
        typeMsg="q" if (maxRel * 100 > 1) else "i",
    )

    return x_new
