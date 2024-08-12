import torch, copy, datetime, botorch, random, socket
import numpy as np
from IPython import embed
from mitim_tools.opt_tools import BOTORCHtools, OPTtools
from mitim_tools.opt_tools.utils import SAMPLINGtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level




def findOptima(fun, writeTrajectory=False):
    print("\t--> BOTORCH optimization techniques used to maximize acquisition")

    # Seeds
    random.seed(fun.seed)
    torch.manual_seed(seed=fun.seed)

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Preparation: Bounds
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    bounds = fun.bounds_mod

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Preparation: Optimizer Options
	------------------------------
	The way botorch.optim.optimize_acqf works is as follows:
		- To start the workflow it needs initial conditions [num_restarts,q,dim] that can be provided via:
			* batch_initial_conditions (directly by the user)
			* raw_samples. This is used to randomly explore the parameter space and select the best.
		- It will optimize to look for a q-number of optimized points from num_restart initial conditions
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    """
	Initialize from botorch tools. Number of random points to evaluate acquisition initially, to select
	the best points ("num_restarts" points) to initialize the scipy optimization.
	"""
    raw_samples = 2**10  # 1024, see my tests

    """
	Optimizer options
	-----------------
		- q, number of candidates to produce
		- num_restarts number of starting points for multistart acquisition function optimization
	"""
    q = 1  # fun.number_optimized_points
    num_restarts = 2**6  # 64, see my tests

    iterations = 1000

    timeout_sec = None
    options = {
        "maxiter": iterations,
        "sample_around_best": True,
        "disp": 50 if read_verbose_level() in [4, 5] else False,
        "seed": fun.seed,
    }  # , "nonnegative" : True}

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
    print(
        f"\t\t- Running optimization to find {q} candidate(s) with {num_restarts} restarts from {raw_samples} raw samples ({iterations} iterations)\n"
    )

    x_opt, _ = botorch.optim.optimize_acqf(
        acq_function=fun_opt,
        bounds=bounds,
        options=options,
        raw_samples=raw_samples,
        batch_initial_conditions=None,
        ic_generator=BOTORCHtools.ic_generator_wrapper(fun.xGuesses),
        q=q,
        num_restarts=num_restarts,
        return_best_only=True,
        timeout_sec=timeout_sec,
    )

    acq_evaluated = torch.Tensor(acq_evaluated)

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Post-processing
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    if len(x_opt.shape) > 2:
        x_opt = x_opt.flatten(start_dim=0, end_dim=-2)
    elif len(x_opt.shape) == 1:
        x_opt = x_opt.unsqueeze(0)

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
