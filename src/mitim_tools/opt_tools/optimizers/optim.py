import torch
import copy
import numpy as np
from scipy.optimize import root
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: MV
# --------------------------------------------------------------------------------------------------------

def scipy_root(flux_residual_evaluator, x_initial, bounds=None, solver_options=None):
    """
    Inputs:
        - x_initial is the initial guesses and must be a tensor of (batches,dimX)
        - flux_residual_evaluator is a function that:
            - Takes X (batches,dimX)
            - Provides Y1: transport (batches,dimY), Y2: target (batches,dimY) and M: maximization metric (batches,1)
            It must also take optional arguments, to capture the best in the batch:
                x_history
                y_history () 
                metric_history (to maximize, similar to acquisition definition, must be 1D, best in batch)
    Outputs:
        - Optium vector x_sol with (batches,dimX) and the trajectory of the acquisition function evaluations (best per batch)
    Notes:
        - tol in root is the same as xtol for LM
        - ftol in LM will define the relative reduction in the sum of squares of the residuals between one iteration and another
    """

    # --------------------------------------------------------------------------------------------------------
    # Solver options
    # --------------------------------------------------------------------------------------------------------

    if solver_options is None:
        solver_options = {}

    solver = solver_options.get("solver", "lm")
    jac_ad = solver_options.get("jac_ad", True)
    tol = solver_options.get("tol", None)
    jacobian_numerical_filter = solver_options.get("jacobian_numerical_filter", 1e-10)
    write_trajectory = solver_options.get("write_trajectory", True)
    algorithm_options = solver_options.get("algorithm_options", {})

    # Forced parameters based on implementation here
    algorithm_options['col_deriv'] = True # Faster in scipy to avoid transposing the Jacobian. I can optimize it in pytorch instead, see above transpose

    # --------------------------------------------------------------------------------------------------------
    # Bounds treatment
    # --------------------------------------------------------------------------------------------------------

    bound_transform = logistic(l=bounds[0, :], u=bounds[1, :]) if bounds is not None else no_bounds()

    # --------------------------------------------------------------------------------------------------------
    # Curation of function to be optimized: tensorization, reshaping and jacobian
    # --------------------------------------------------------------------------------------------------------

    x_history, y_history, metric_history = [], [], []
    def function_for_optimizer_prep(x, dimX=x_initial.shape[-1], flux_residual_evaluator=flux_residual_evaluator, bound_transform=bound_transform):
        """
        Notes:
            - x comes extended, batch*dim
            - y must be returned extended as well, batch*dim
        """

        X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

        # Transform from infinite bounds
        X = bound_transform.transform(X)

        # Evaluate residuals
        y1, y2, _ = flux_residual_evaluator(
            X,
            y_history = y_history if write_trajectory else None,
            x_history = x_history if write_trajectory else None,
            metric_history = metric_history if write_trajectory else None
            )
        y = y2-y1

        # Root requires that len(x)==len(y)
        y = equal_dimensions(X, y)

        # Compress again  [batch,dim]->[batch*dim]
        y = y.view(x.shape)

        return y

    def function_for_optimizer(x, dfT1=torch.zeros(1).to(x_initial)):

        # Root will work with arrays, convert to tensor with AD
        X = torch.tensor(x, requires_grad=True).to(dfT1)

        # Evaluate value and local jacobian
        QhatD, JD = mitim_jacobian(function_for_optimizer_prep, X, vectorize=True)  # vectorize: Fast calculation of the jacobian (much faster, but experimental)

        # Avoid numerical artifacts for off-block-diagonal elements that should be zero but numerically are not
        JD[JD.abs() <= jacobian_numerical_filter] = 0.0

        # Back to arrays
        if jac_ad:  
            return QhatD.detach().cpu().numpy(), JD.transpose(0,1).cpu().numpy()    # Transpose here so that I can use col_deriv=True
        else:       
            return QhatD.detach().cpu().numpy()

    # --------------------------------------------------------------------------------------------------------
    # Preparation of the initial guess
    # --------------------------------------------------------------------------------------------------------

    # Untransform guesses (from bounds to infinite)
    x_initial0 = bound_transform.untransform(x_initial)

    # Convert to 1D ([batch,dim]->[batch*dim])
    x_initial0 = x_initial0.view(-1)

    # To numpy
    x_initial0 = x_initial0.cpu().numpy()

    # --------------------------------------------------------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------------------------------------------------------

    f0 = function_for_optimizer(x_initial0)
    if jac_ad: f0 = f0[0]
    print(f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f0)):.3e} of {f0.shape[0]} channels):\n\t{f0}\n")

    # --------------------------------------------------------------------------------------------------------
    # Perform optimization
    # --------------------------------------------------------------------------------------------------------

    with IOtools.timer(name="\t- SCIPY.ROOT multi-variate root finding method"):
        sol = root(function_for_optimizer, x_initial0, jac=jac_ad, method=solver, tol=tol, options=algorithm_options)

    # --------------------------------------------------------------------------------------------------------
    # Evaluate final case to compare
    # --------------------------------------------------------------------------------------------------------

    f = function_for_optimizer(sol.x)
    if jac_ad: f = f[0]
    print(f"\t\n|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f)):.3e} of {f.shape[0]} channels):\n\t{f}")

    print("\t- Results from scipy solver:", sol)

    if write_trajectory:
        try:
            y_history = torch.stack(y_history)
        except (TypeError,RuntimeError):
            y_history = torch.Tensor(y_history)
        try:
            x_history = torch.stack(x_history)
        except (TypeError,RuntimeError):
            x_history = torch.Tensor(x_history)
        try:
            metric_history = torch.stack(metric_history)
        except (TypeError,RuntimeError):
            metric_history = torch.Tensor(metric_history)
    else:
        y_history, x_history, metric_history = torch.Tensor(), torch.Tensor(), torch.Tensor()

    # --------------------------------------------------------------------------------------------------------
    # Preparation of the final solution
    # --------------------------------------------------------------------------------------------------------

    # Convert to tensor
    x_best = torch.tensor(sol.x).to(x_initial)

    # Reshape to original shape
    x_best = x_best.view( (x_best.shape[0] // x_initial.shape[1], x_initial.shape[1]) )

    # Transform to bounded
    x_best = bound_transform.transform(x_best)

    return x_best, y_history, x_history, metric_history


# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: Simple Relax
# --------------------------------------------------------------------------------------------------------

def simple_relaxation( flux_residual_evaluator, x_initial, bounds=None, solver_options=None ):
    """
    See scipy_root for the inputs and outputs
    """

    tol = solver_options.get("tol", -1e-6)                      # Tolerance for the residual (negative because I want to maximize)
    tol_rel = solver_options.get("tol_rel", None)               # Relative tolerance for the residual (superseeds tol)
    
    maxiter = solver_options.get("maxiter", 1e5)
    relax = solver_options.get("relax", 0.1)                    # Defines relationship between flux_residual_evaluator and gradient
    dx_max = solver_options.get("dx_max", 0.1)                  # Maximum step size in gradient, relative (e.g. a/Lx can only increase by 10% each time)
    dx_max_abs = solver_options.get("dx_max_abs", None)         # Maximum step size in gradient, absolute (e.g. a/Lx can only increase by 0.1 each time)
    dx_min_abs = solver_options.get("dx_min_abs", None)         # Minimum step size in gradient, absolute (e.g. a/Lx must at least increase by 0.01 each time)
    
    relax_dyn = solver_options.get("relax_dyn", False)                 # Dynamic relax, decreases relax if residual is not decreasing
    relax_dyn_decrease = solver_options.get("relax_dyn_decrease", 5)   # Decrease relax by this factor
    relax_dyn_num = solver_options.get("relax_dyn_num", 100)           # Number of iterations to average over
    relax_dyn_tol = solver_options.get("relax_dyn_tol", 1e-4)          # Tolerance to consider that the residual is not decreasing

    print_each = solver_options.get("print_each", 1e2)
    
    write_trajectory = solver_options.get("write_trajectory", True)
    x_history, y_history, metric_history = [], [], []

    thr_bounds = 1e-4 # To avoid being exactly in the bounds (relative -> 0.01%)

    x = copy.deepcopy(x_initial)
    Q, QT, M = flux_residual_evaluator(
        x,
        y_history = y_history if write_trajectory else None,
        x_history = x_history if write_trajectory else None,
        metric_history = metric_history if write_trajectory else None
        )
    print(f"\t* Starting residual: {(Q-QT).abs().mean(axis=1)[0].item():.4e}, will run {int(maxiter)-1} more evaluations, printing every {print_each} iteration:",typeMsg="i")

    if tol_rel is not None:
        tol = tol_rel * M.max().item()
        print(f"\t* Relative tolerance of {tol_rel:.1e} will be used, resulting in an absolute tolerance of {tol:.1e}")


    print(f"\t* Flux-grad relationship of {relax*100.0:.1f}% and maximum gradient jump of {dx_max*100.0:.1f}%,{f' to achieve residual of {tol:.1e}' if tol is not None else ''} in maximum of {maxiter:.0f} iterations")

    its_since_last_dyn_relax = 0
    for i in range(int(maxiter) - 1):
        # --------------------------------------------------------------------------------------------------------
        # Iterative Strategy
        # --------------------------------------------------------------------------------------------------------

        x_new = _simple_relax_iteration(x, Q, QT, relax, dx_max, dx_max_abs = dx_max_abs, dx_min_abs = dx_min_abs)

        # Clamp to bounds
        if bounds is not None:
            bb = bounds[1,:]-bounds[0,:]
            x_new = x_new.clamp(min=bounds[0,:]+thr_bounds*bb, max=bounds[1,:]-thr_bounds*bb)

        x = x_new.clone()

        # --------------------------------------------------------------------------------------------------------
        Q, QT, M = flux_residual_evaluator(
            x,
            y_history = y_history if write_trajectory else None,
            x_history = x_history if write_trajectory else None,
            metric_history = metric_history
            )

        # Best metric of the batch
        metric_best = M.max(axis=-1)[0].item()

        if (i + 1) % int(print_each) == 0:
            print(f"\t\t- Metric (to maximize) @{i+1}: {metric_best:.2e}")

        # Stopping based on the best of the batch based on the metric
        if tol is not None and M.max().item() > tol:
            print(f"\t\t- Metric (to maximize) @{i+1}: {metric_best:.2e}",typeMsg="i")
            print(f"\t* Converged in {i+1} iterations with metric of {metric_best:.2e} > {tol:.2e}",typeMsg="i")
            break

        if relax_dyn and (i-its_since_last_dyn_relax > relax_dyn_num):
            relax, changed, hardbreak = _dynamic_relaxation(relax, relax_dyn_decrease, metric_history, relax_dyn_num, relax_dyn_tol,i+1)
            if changed:
                its_since_last_dyn_relax = i
            if hardbreak:
                break

    if i == int(maxiter) - 2:
        print(f"\t* Did not converge in {maxiter} iterations",typeMsg="i")

    if write_trajectory:
        try:
            y_history = torch.stack(y_history)
        except (TypeError,RuntimeError):
            y_history = torch.Tensor(y_history)
        try:
            x_history = torch.stack(x_history)
        except (TypeError,RuntimeError):
            x_history = torch.Tensor(x_history)
        try:
            metric_history = torch.stack(metric_history)
        except(TypeError,RuntimeError):
            metric_history = torch.Tensor(metric_history)
    else:
        y_history, x_history, metric_history = torch.Tensor(), torch.Tensor(), torch.Tensor()

    index_best = metric_history.argmax()
    print(f"\t* Best metric: {metric_history[index_best].mean().item():.2e} at iteration {index_best}",typeMsg="i")
    
    # The best candidate, regardless of the restarts
    x_best = x_history[index_best,:].unsqueeze(0)

    return x_best, y_history, x_history, metric_history


def _dynamic_relaxation(relax, relax_dyn_decrease, metric_history, relax_dyn_num, relax_dyn_tol, it, min_relax=1e-6):
    '''
    Logic:  If the metric is not improving enough, decrease the relax parameter. To determine
            if the metric is improving enough, I will fit a line to the last relax_dyn_num points and
            check if the slope is small enough. If it is, I will decrease the relax parameter.
    '''

    metric_history_considered = torch.Tensor(metric_history[-relax_dyn_num:])

    # Linear fit to the time series
    x = np.arange(len(metric_history_considered))
    y = metric_history_considered
    slope, intercept = np.polyfit(x, y, 1)
    metric0 = intercept
    metric1 = slope * len(metric_history_considered) + intercept
    change_in_metric = abs(metric1 - metric0)

    if (change_in_metric < relax_dyn_tol):
        if relax > min_relax:
            print(f"\t\t\t<> Metric not improving enough (@{it}), decreasing relax from {relax:.1e} to {relax/relax_dyn_decrease:.1e}")
            relax = relax / relax_dyn_decrease
            return relax, True, False
        else:
            print(f"\t\t\t<> Metric not improving enough (@{it}), relax already at minimum of {min_relax:.1e}, not worth continuing", typeMsg="i")
            return relax, False, True
    else:
        return relax, False, False

def _simple_relax_iteration(x, Q, QT, relax, dx_max, dx_max_abs = None, dx_min_abs = None, threshold_zero_flux_issue=1e-10):
    # Calculate step in gradient (if target > transport, dx>0 because I want to increase gradients)
    dx = relax * (QT - Q) / (Q**2 + QT**2).clamp(min=threshold_zero_flux_issue) ** 0.5

    # Prevent big steps - Clamp to the max step (with the right sign)
    ix = dx.abs() > dx_max
    dx[ix] = dx_max * (dx[ix] / dx[ix].abs())

    # Define absolute step (Note for PRF: abs() was added by me, I think it performs better that way!)
    x_step = dx * x.abs()

    # Absolute steps limits
    if dx_max_abs is not None:
        ix = x_step.abs() > dx_max_abs
        direction = torch.nan_to_num(x_step[ix] / x_step[ix].abs(), nan=1.0)
        x_step[ix] = dx_max_abs * direction
    if dx_min_abs is not None:
        ix = x_step.abs() < dx_min_abs
        direction = torch.nan_to_num(x_step[ix] / x_step[ix].abs(), nan=1.0)
        x_step[ix] = dx_min_abs * direction

    # Update
    x_new = x + x_step

    return x_new

'''
********************************************************************************************************************************** 
The original implementation of torch.autograd.functional.jacobian runs the function once and then computes the jacobian.
This implementation simply copies what the original does, but returns the outputs so that I don't need to calculate them again.
**********************************************************************************************************************************
'''

from torch.autograd.functional import _autograd_grad, _construct_standard_basis_for, _grad_postprocess, _grad_preprocess, _tuple_postprocess, _as_tuple, _check_requires_grad

def mitim_jacobian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    strategy="reverse-mode",
    ):

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jacobian"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if vectorize:

            # Step 1: Construct grad_outputs by splitting the standard basis
            output_numels = tuple(output.numel() for output in outputs)
            grad_outputs = _construct_standard_basis_for(outputs, output_numels)
            flat_outputs = tuple(output.reshape(-1) for output in outputs)

            # Step 2: Call vmap + autograd.grad
            def vjp(grad_output):
                vj = list(
                    _autograd_grad(
                        flat_outputs,
                        inputs,
                        grad_output,
                        create_graph=create_graph,
                        is_grads_batched=True,
                    )
                )
                for el_idx, vj_el in enumerate(vj):
                    if vj_el is not None:
                        continue
                    vj[el_idx] = torch.zeros_like(inputs[el_idx]).expand(
                        (sum(output_numels),) + inputs[el_idx].shape
                    )
                return tuple(vj)

            jacobians_of_flat_output = vjp(grad_outputs)

            # Step 3: The returned jacobian is one big tensor per input. In this step,
            # we split each Tensor by output.
            jacobian_input_output = []
            for jac_input_i, input_i in zip(jacobians_of_flat_output, inputs):
                jacobian_input_i_output = []
                for jac, output_j in zip(
                    jac_input_i.split(output_numels, dim=0), outputs
                ):
                    jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                    jacobian_input_i_output.append(jacobian_input_i_output_j)
                jacobian_input_output.append(jacobian_input_i_output)

            # Step 4: Right now, `jacobian` is a List[List[Tensor]].
            # The outer List corresponds to the number of inputs,
            # the inner List corresponds to the number of outputs.
            # We need to exchange the order of these and convert to tuples
            # before returning.
            jacobian_output_input = tuple(zip(*jacobian_input_output))

            jacobian_output_input = _grad_postprocess(
                jacobian_output_input, create_graph
            )
            return outputs[0],_tuple_postprocess(
                jacobian_output_input, (is_outputs_tuple, is_inputs_tuple)
            )

class logistic:
    """
    To transform from bounds to unbound
    """

    def __init__(self, l=0.0, u=1.0, k=0.5, x0=0.0):
        self.l, self.u, self.k, self.x0 = l, u, k, x0

    def transform(self, x):
        # return self.l + (self.u-self.l)*(1/(1+torch.exp(-self.k*(x-self.x0))))
        # Proposed by chatGPT3.5 to solve the exponential overflow (torch autograd failed for large x):
        return self.l + 0.5 * (torch.tanh(self.k * (x - self.x0)) + 1) * (self.u - self.l)

    def untransform(self, y):
        # return self.x0-1/self.k * torch.log( (self.u-self.l)/(y-self.l)-1 )
        # Proposed by chatGPT3.5 to solve the exponential overflow (torch autograd failed for large x):
        return self.x0 + (1 / self.k) * torch.atanh(2 * (y - self.l) / (self.u - self.l) - 1)

class no_bounds:
    def __init__(self, *args, **kwargs):
        pass

    def transform(self, x):
        return x

    def untransform(self, y):
        return y

def equal_dimensions(x, y):
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
