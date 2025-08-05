import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.opt_tools.optimizers import optim
from mitim_tools.opt_tools.utils import TESTtools
from mitim_tools.misc_tools import GRAPHICStools
from IPython import embed

def optimize_function(fun, optimization_params = {}, writeTrajectory=False, method = 'scipy_root', debugYN=False):
    
    np.random.seed(fun.seed)

    # --------------------------------------------------------------------------------------------------------
    # Solver options
    # --------------------------------------------------------------------------------------------------------

    num_restarts = optimization_params.get("num_restarts",1)
    bounds = fun.bounds_mod

    if method == 'scipy_root':

        print("\t- Implementation of SCIPY.ROOT multi-variate root finding method")

        solver_options = {
            'algorithm_options': {
                "maxiter": optimization_params.get("maxiter",None),
                "ftol": optimization_params.get("relative_improvement_for_stopping",1e-8),
                },
            'solver': optimization_params.get("solver","lm"),
            'write_trajectory': writeTrajectory
        }
        solver_fun = optim.scipy_root
        numZ = 5
    
    elif method == "sr":

        print("\t- Implementation of simple relaxation method")
        
        solver_options = {
            "tol_rel": optimization_params.get("relative_improvement_for_stopping",1e-4),
            "maxiter": optimization_params.get("maxiter",1000),
            "relax": optimization_params.get("relax",0.1),     
            "relax_dyn": optimization_params.get("relax_dyn",True),
            "print_each": optimization_params.get("maxiter",1000)//20,
        }
        solver_fun = optim.simple_relaxation
        numZ = 6
    
    # --------------------------------------------------------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------------------------------------------------------

    def flux_residual_evaluator(X, y_history=None, x_history=None, metric_history=None):

        # Evaluate source term
        yOut, y1, y2, _ = fun.evaluators["residual_function"](X, outputComponents=True)

        # -----------------------------------------
        # Post-process
        # -----------------------------------------

        # Best in batch
        best_candidate = yOut.argmax().item()
        # Only pass the best candidate
        yRes = (y2-y1)[best_candidate, :].detach()
        yMetric = yOut[best_candidate].detach()
        Xpass = X[best_candidate, :].detach()

        # Store values
        if metric_history is not None:
            metric_history.append(yMetric)
        if x_history is not None:
            x_history.append(Xpass)
        if y_history is not None:
            y_history.append(yRes)

        return y1, y2, yOut

    # --------------------------------------------------------------------------------------------------------
    # Preparation of guesses
    # --------------------------------------------------------------------------------------------------------

    print("\t- Preparing starting points")

    xGuesses = copy.deepcopy(fun.xGuesses)

    num_random = int(np.ceil(num_restarts/2)) # Half of the restarts will be random, the other half will be the best guesses

    # Take the best num_restarts-num_random points
    xGuesses = xGuesses[:num_restarts-num_random, :] if xGuesses.shape[0] > num_restarts-num_random else xGuesses
    
    # Add random points (to avoid local minima and getting stuck as much as possible) 
    cases_to_choose_from = fun.xGuesses.shape[0]-xGuesses.shape[0]
    random_choice = xGuesses.shape[0]+np.random.choice(cases_to_choose_from, np.min([cases_to_choose_from,num_random]), replace=False)
    xGuesses = torch.cat((xGuesses, fun.xGuesses[random_choice, :]), axis=0) 

    print(f"\t\t- From training set, taking the best {num_restarts-num_random} points and adding {num_random} random points (ordered positions {random_choice})")

    print(f'\t\t- Running for {len(xGuesses)} starting points , as a an augmented optimization problem')

    # --------------------------------------------------------------------------------------------------------
    # Solver
    # --------------------------------------------------------------------------------------------------------

    print("************************************************************************************************")
    x_res, y_history, x_history, acq_evaluated = solver_fun(flux_residual_evaluator,xGuesses,solver_options=solver_options,bounds=bounds)
    print("************************************************************************************************")

    if debugYN:
        fig, axs = plt.subplots(nrows= 2, figsize=(6, 10))
        ax = axs[0]
        ax.plot(np.array(x_history))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("X")
        GRAPHICStools.addDenseAxis(ax)
        
        ax = axs[1]
        ax.plot(np.abs(np.array(y_history)))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|Y|")
        ax.set_yscale("log")
        GRAPHICStools.addDenseAxis(ax)

        plt.show()
        embed()

    # --------------------------------------------------------------------------------------------------------
    # Post-process
    # --------------------------------------------------------------------------------------------------------

    bb = TESTtools.checkSolutionIsWithinBounds(x_res, fun.bounds).item()
    if not bb:
        print(f"\t- Is this solution inside bounds? {bb}")
        print(f"\t\t- with allowed extrapolations? {TESTtools.checkSolutionIsWithinBounds(x_res,fun.bounds_mod).item()}")

    from mitim_tools.opt_tools.OPTtools import summarizeSituation, pointsOperation_bounds, pointsOperation_order

    # I apply the bounds correction BEFORE the summary because of possibility of crazy values (problems with GP)
    x_opt, _, _ = pointsOperation_bounds(
        x_res,
        None,
        None,
        fun,
        maxExtrapolation=fun.strategy_options["AllowedExcursions"],
    )

    # Summary
    y_opt_residual = summarizeSituation(fun.xGuesses, fun, x_opt) if (len(x_opt) > 0) else torch.Tensor([]).to(fun.stepSettings["dfT"])

    # Order points them
    x_opt, y_opt_residual, _, indeces = pointsOperation_order(x_opt, y_opt_residual, None, fun)

    print(f"\t- Points ordered: {indeces.cpu().numpy()}")

    # Get out
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    return x_opt, y_opt_residual, z_opt, acq_evaluated
