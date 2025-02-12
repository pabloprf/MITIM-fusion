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

def powell(flux_residual_evaluator, x_initial, bounds=None, solver_options=None):
    """
    Inputs:
        - x_initial is the initial guesses and must be a tensor of (batches,dimX)
        - flux_residual_evaluator is a function that must take X (batches,dimX) and provide the source term (batches,dimY).
            It must also take optional arguments x_history,y_history that will be a list of the evaluations of the residual
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

    x_history, y_history = [], []
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
        y = flux_residual_evaluator(X, y_history = y_history if write_trajectory else None, x_history = x_history if write_trajectory else None)

        # Root requires that len(x)==len(y)
        y = fixDimensions_ROOT(X, y)

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
    print(f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f0)):.3e} of {f0.shape[0]} channels):\n\t{f0}")

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
    print(f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f)):.3e} of {f.shape[0]} channels):\n\t{f}")

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
    else:
        y_history, x_history = torch.Tensor(), torch.Tensor()

    # --------------------------------------------------------------------------------------------------------
    # Preparation of the final solution
    # --------------------------------------------------------------------------------------------------------

    # Convert to tensor
    x_best = torch.tensor(sol.x).to(x_initial)

    # Reshape to original shape
    x_best = x_best.view( (x_best.shape[0] // x_initial.shape[1], x_initial.shape[1]) )

    # Transform to bounded
    x_best = bound_transform.transform(x_best)

    return x_best, y_history, x_history


# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: Simple Relax
# --------------------------------------------------------------------------------------------------------

def simple_relaxation(
    flux,
    x_initial,
    bounds=None,
    solver_options=None
    ):
    """
    Inputs:
            - flux is a function that must take X (dimX) and provide Q and QT as tensors of dimensions (1,dimY) each
    """

    tol = solver_options.get("tol", 1e-6)
    maxiter = solver_options.get("maxiter", 1e5)
    relax = solver_options.get("relax", 0.1)
    dx_max = solver_options.get("dx_max", 0.05)
    dx_max_abs = solver_options.get("dx_max_abs", None)
    dx_min_abs = solver_options.get("dx_min_abs", None)
    print_each = solver_options.get("print_each", 1e2)
    storeValues = solver_options.get("storeValues", False)

    print(f"* Flux-grad relationship of {relax} and maximum gradient jump of {dx_max*100.0:.1f}%, to achieve residual of {tol:.1e} in {maxiter:.0f} iterations")

    x = copy.deepcopy(x_initial)
    Q, QT = flux(x, cont=0)
    print(f"* Starting residual: {(Q-QT).abs().mean(axis=1)[0].item():.4e}, will run {int(maxiter)-1} more evaluations",typeMsg="i",)

    store_x = x.clone()
    store_Q = (Q - QT).abs()
    for i in range(int(maxiter) - 1):
        # --------------------------------------------------------------------------------------------------------
        # Iterative Strategy
        # --------------------------------------------------------------------------------------------------------

        x_new = simple_relax_iteration(x, Q, QT, relax, dx_max, dx_max_abs = dx_max_abs, dx_min_abs = dx_min_abs)

        # Clamp to bounds
        if bounds is not None:
            x_new = x_new.clamp(min=bounds[0,:], max=bounds[1,:])

        x = x_new.clone()

        # --------------------------------------------------------------------------------------------------------
        Q, QT = flux(x, cont=i + 1)

        if (i + 1) % int(print_each) == 0:
            print(
                f"\t- Residual @ #{i+1}: {(Q-QT).abs().mean(axis=1)[0].item():.2e}",
                typeMsg="i",
            )

        if (Q - QT).abs().mean(axis=1)[0].item() < tol:
            break

        if storeValues:
            store_x = torch.cat((store_x, x.detach()), axis=0)
            store_Q = torch.cat((store_Q, (Q - QT).abs().detach()), axis=0)

    return store_x, store_Q

def simple_relax_iteration(x, Q, QT, relax, dx_max, dx_max_abs = None, dx_min_abs = None):
    # Calculate step in gradient (if target > transport, dx>0 because I want to increase gradients)
    dx = relax * (QT - Q) / (Q**2 + QT**2) ** 0.5

    # Prevent big steps - Clamp to the max step (with the right sign)
    ix = dx.abs() > dx_max
    dx[ix] = dx_max * (dx[ix] / dx[ix].abs())

    # Define absolute step (Note for PRF: abs() was added by me, I think it performs better that way!)
    x_step = x.abs() * dx

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
