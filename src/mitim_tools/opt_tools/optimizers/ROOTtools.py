import torch
import copy
import datetime
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.opt_tools.optimizers import optim
from mitim_tools.opt_tools.utils import TESTtools


def findOptima(fun, writeTrajectory=False, **kwargs):
    print("\t- Implementation of SCIPY.ROOT multi-variate root finding method")
    np.random.seed(fun.seed)

    # Options
    numCases = fun.number_optimized_points
    runCasesInParallelAsBatch = True
    solver = "lm"
    algorithmOptions = {"maxiter": 1000}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    bounds = fun.bounds_mod

    # transform from unbounded to bounded
    bound_transform = logistic(l=bounds[0, :], u=bounds[1, :])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Define evaluator
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    acq_evaluated = []

    if writeTrajectory:

        def channel_residual_evaluator(
            x, dimX=fun.xGuesses.shape[-1], fun=fun, bound_transform=bound_transform
        ):
            """
            Notes:
                    - x comes extended, batch*dim
                    - y must be returned extended as well, batch*dim
            """

            X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

            # Transform from infinite bounds
            X = bound_transform.transform(X)

            # Evaluate residuals
            yOut, y1, y2, _ = fun.evaluators["residual_function"](
                X, outputComponents=True
            )
            y = y1 - y2

            acq_evaluated.append(
                -yOut.abs().min().item()
            )  # yOut has [batch] dimensions, so look at the best

            # Root requires that len(x)==len(y)
            y = fixDimensions_ROOT(X, y)

            # Compress again  [batch,dim]->[batch*dim]
            y = y.view(x.shape)

            return y

    else:

        def channel_residual_evaluator(
            x, dimX=fun.xGuesses.shape[-1], fun=fun, bound_transform=bound_transform
        ):
            """
            Notes:
                    - x comes extended, batch*dim
                    - y must be returned extended as well, batch*dim
            """

            X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

            # Transform from infinite bounds
            X = bound_transform.transform(X)

            # Evaluate residuals
            _, y1, y2, _ = fun.evaluators["residual_function"](X, outputComponents=True)
            y = y1 - y2

            # Root requires that len(x)==len(y)
            y = fixDimensions_ROOT(X, y)

            # Compress again  [batch,dim]->[batch*dim]
            y = y.view(x.shape)

            return y

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Guesses
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    xGuesses = copy.deepcopy(fun.xGuesses)

    # Limit the guesses the the number of cases I want to run from
    xGuesses = xGuesses[:numCases, :] if xGuesses.shape[0] > numCases else xGuesses

    # Untransform guesses
    xGuesses = bound_transform.untransform(xGuesses)

    print(
        f'\t\t- Running for {len(xGuesses)} starting points{", as a big 1D tensor" if runCasesInParallelAsBatch else ""}'
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Process
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Convert to 1D
    x0 = xGuesses.view(-1).unsqueeze(0) if runCasesInParallelAsBatch else xGuesses

    time1 = datetime.datetime.now()

    x_res = torch.Tensor().to(fun.stepSettings["dfT"])
    for i in range(len(x0)):
        if len(x0) > 1:
            print(
                "\n",
                f"\t\t- ROOT from guessed point {i+1}/{x0.shape[0]}",
            )
        x_res0 = optim.powell(
            channel_residual_evaluator,
            x0[i, :],
            fun,
            writeTrajectory=writeTrajectory,
            algorithmOptions=algorithmOptions,
            solver=solver,
        )
        x_res = torch.cat((x_res, x_res0.unsqueeze(0)), axis=0)

    acq_evaluated = torch.Tensor(acq_evaluated)

    print(
        f"\t\t- Optimization took {IOtools.getTimeDifference(time1)}, and it found {x_res.shape[0]} optima"
    )

    if runCasesInParallelAsBatch:
        x_res = x_res.view(
            (x_res.shape[1] // fun.xGuesses.shape[1], fun.xGuesses.shape[1])
        )

    x_res = bound_transform.transform(x_res)

    bb = TESTtools.checkSolutionIsWithinBounds(x_res, fun.bounds).item()
    print(f"\t- Is this solution inside bounds? {bb}")
    if not bb:
        print(
            f"\t\t- with allowed extrapolations? {TESTtools.checkSolutionIsWithinBounds(x_res,fun.bounds_mod).item()}"
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Post-process
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mitim_tools.opt_tools.OPTtools import (
        summarizeSituation,
        pointsOperation_bounds,
        pointsOperation_order,
    )

    # I apply the bounds correction BEFORE the summary because of possibility of crazy values (problems with GP)
    x_opt, _, _ = pointsOperation_bounds(
        x_res,
        None,
        None,
        fun,
        maxExtrapolation=fun.StrategyOptions["AllowedExcursions"],
    )

    # Summary
    if len(x_opt) > 0:
        y_opt_residual = summarizeSituation(fun.xGuesses, fun, x_opt)
    else:
        y_opt_residual = torch.Tensor([]).to(fun.stepSettings["dfT"])

    # Order points them
    x_opt, y_opt_residual, _, indeces = pointsOperation_order(
        x_opt, y_opt_residual, None, fun
    )

    print(f"\t~ Order of ROOT points: {indeces.cpu().numpy()}")

    # Get out
    numZ = 5
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    return x_opt, y_opt_residual, z_opt, acq_evaluated


def fixDimensions_ROOT(x, y):
    # ------------------------------------------------------------
    # Root requires that len(x)==len(y)
    # ------------------------------------------------------------

    # If dim_x larger than dim_y, completing now with repeating objectives
    i = 0
    while x.shape[-1] > y.shape[-1]:
        y = torch.cat((y, y[:, i].unsqueeze(1)), axis=1)
        i += 1

    # If dim_y larger than dim_x, building the last y as the means
    if x.shape[-1] < y.shape[-1]:
        yn = y[:, : x.shape[-1] - 1]
        yn = torch.cat((yn, y[:, x.shape[1] - 1 :].mean(axis=1).unsqueeze(0)), axis=1)
        y = yn

    return y


class logistic:
    """
    To transform from bounds to unbound
    """

    def __init__(self, l=0.0, u=1.0, k=0.5, x0=0.0):
        self.l, self.u, self.k, self.x0 = l, u, k, x0

    def transform(self, x):
        # return self.l + (self.u-self.l)*(1/(1+torch.exp(-self.k*(x-self.x0))))
        # Proposed by chatGPT3.5 to solve the exponential overflow (torch autograd failed for large x):
        return self.l + 0.5 * (torch.tanh(self.k * (x - self.x0)) + 1) * (
            self.u - self.l
        )

    def untransform(self, y):
        # return self.x0-1/self.k * torch.log( (self.u-self.l)/(y-self.l)-1 )
        # Proposed by chatGPT3.5 to solve the exponential overflow (torch autograd failed for large x):
        return self.x0 + (1 / self.k) * torch.atanh(
            2 * (y - self.l) / (self.u - self.l) - 1
        )
