import torch
import botorch
import random
from mitim_tools.opt_tools import OPTtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from IPython import embed

def optimize_function(fun, optimization_params = {}, writeTrajectory=False):
    print("\t--> BOTORCH optimization techniques used to maximize acquisition")

    # Seeds
    random.seed(fun.seed)
    torch.manual_seed(seed=fun.seed)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Preparation: Optimizer Options
    ------------------------------
    Options are:
        - q, number of candidates to produce
        - raw_samples, number of random points to evaluate the acquisition function initially, to select
            the best points ("num_restarts" points) to initialize the scipy optimization.
            Note: Only evaluated once, it's fine that it's a large number
        - num_restarts number of starting points for multistart acquisition function optimization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    raw_samples = optimization_params.get("raw_samples",100)
    num_restarts = optimization_params.get("num_restarts",10)
   
    q = optimization_params.get("keep_best",1)
    sequential_q = True # Not really relevant for q=1, but recommendation from BoTorch team for q>1
    options = {
        "sample_around_best": True,
        "disp": 50 if read_verbose_level() == 5 else False,
        "seed": fun.seed,
    }

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Optimization
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    fun_opt = fun.evaluators["acq_function"]

    acq_evaluated = []
    if writeTrajectory:
        class CustomFunctionWrapper:
            def __init__(self, func, eval_list):
                self.func = func
                self.eval_list = eval_list

            def __call__(self, x, *args, **kwargs):
                f = self.func(x, *args, **kwargs)
                self.eval_list.append(f.max().item())
                return f

        fun_opt = CustomFunctionWrapper(fun_opt, acq_evaluated)

    seq_message = f'({"sequential" if sequential_q else "joint"}) ' if q>1 else ''
    print(f"\t\t- Optimizing using optimize_acqf: {q = } {seq_message}, {num_restarts = }, {raw_samples = }")

    with IOtools.timer(name = "\n\t- Optimization", name_timer = '\t\t- Time: '):
        x_opt, _ = botorch.optim.optimize_acqf(
            acq_function=fun_opt,
            bounds=fun.bounds_mod,
            raw_samples=raw_samples,
            q=q,
            sequential=sequential_q,
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

    # Summarize
    y_res = OPTtools.summarizeSituation(fun.xGuesses, fun, x_opt)

    # Order points them
    x_opt, y_res, _, indeces = OPTtools.pointsOperation_order(x_opt, y_res, None, fun)

    # Provide numZ index to track where this solution came from
    numZ = 3
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    return x_opt, y_res, z_opt, acq_evaluated
