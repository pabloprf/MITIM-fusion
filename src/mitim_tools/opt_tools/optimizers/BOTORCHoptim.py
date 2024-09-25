import torch
import datetime
import botorch
import random
from mitim_tools.opt_tools import OPTtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from IPython import embed

def findOptima(fun, writeTrajectory=False):
    print("\t--> BOTORCH optimization techniques used to maximize acquisition")

    # Seeds
    random.seed(fun.seed)
    torch.manual_seed(seed=fun.seed)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Preparation: Optimizer Options
    ------------------------------
    The way botorch.optim.optimize_acqf works is as follows:
        - To start the workflow it needs initial conditions [num_restarts,q,dim] that can be provided via:
            * batch_initial_conditions (directly by the user)
            * raw_samples. This is used to randomly explore the parameter space and select the best.
        - It will optimize to look for a q-number of optimized points from num_restart initial conditions
    Options are:
        - q, number of candidates to produce
        - raw_samples, number of random points to evaluate the acquisition function initially, to select
            the best points ("num_restarts" points) to initialize the scipy optimization.
        - num_restarts number of starting points for multistart acquisition function optimization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    raw_samples = 10_000  # Note: Only evaluated once, it's fine that it's a large number
    num_restarts = 16
    maxiter = 1000

    q = fun.number_optimized_points
    options = {
        "maxiter": maxiter,
        "sample_around_best": True,
        "disp": 50 if read_verbose_level() == 5 else False,
        "seed": fun.seed,
    }

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Optimization
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    acq_evaluated = []
    if writeTrajectory:
        def fun_opt(x, v=acq_evaluated):
            f = fun.evaluators["acq_function"](x)
            v.append(f.max().item())
            return f
    else:
        fun_opt = fun.evaluators["acq_function"]

    time1 = datetime.datetime.now()
    print(f'\t\t- Time: {time1.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"\t\t- Optimizing to find {q} point(s) with {num_restarts =} from {raw_samples =}, {maxiter =}\n")

    x_opt, _ = botorch.optim.optimize_acqf(
        acq_function=fun_opt,
        bounds=fun.bounds_mod,
        raw_samples=raw_samples,
        q=q,
        num_restarts=num_restarts,
        options=options,
    )

    acq_evaluated = torch.Tensor(acq_evaluated)

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Post-processing
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    x_opt = x_opt.flatten(start_dim=0, end_dim=-2) if len(x_opt.shape) > 2 else ( x_opt.unsqueeze(0) if len(x_opt.shape) == 1 else x_opt )

    print(
        f"\n\t- Optimization took {IOtools.getTimeDifference(time1)}, and it found {x_opt.shape[0]} optima"
    )

    # Summarize
    y_res = OPTtools.summarizeSituation(fun.xGuesses, fun, x_opt)

    # Order points them
    x_opt, y_res, _, indeces = OPTtools.pointsOperation_order(x_opt, y_res, None, fun)

    # Provide numZ index to track where this solution came from
    numZ = 3
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    return x_opt, y_res, z_opt, acq_evaluated
