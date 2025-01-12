import torch
import copy
import numpy as np
from scipy.optimize import root
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: MV
# --------------------------------------------------------------------------------------------------------


def powell(flux, xGuess, optim_fun, writeTrajectory=False, algorithm_options={}, solver="lm"):
    """
    Inputs:
            - xGuess is the initial guess and must be a tensor of (1,dimX) or (dimX). It will be transformed to dimX.
            - optim_fun is a function that must take X (dimX) and provide Q and QT as tensors of dimensions (1,dimY) each
    Outputs:
            - Optium vector x with (dimX)
    Notes:
            - The porblem must be: dimX = dimY
            - Must all be tensors that allow Jacobian calculation
            - tol in root is the same as xtol for LM
            - ftol in LM will define the relative reduction in the sum of squares of the residuals between one iteration and another
    """

    def func(x, dfT1=torch.zeros(1).to(xGuess)):
        # Root will work with arrays, convert to tensor with AD
        X = torch.tensor(x, requires_grad=True).to(dfT1)

        # Evaluate value and local jacobian
        QhatD, JD = mitim_jacobian(flux, X, vectorize=True)  # vectorize: Fast calculation of the jacobian (much faster, but experimental)

        # Avoid numerical artifacts for off-block-diagonal elements that should be zero but numerically are not
        JD[JD.abs() <= 1e-10] = 0.0

        # Back to arrays
        return QhatD.detach().cpu().numpy(), JD.transpose(0,1).cpu().numpy()    # Transpose here so that I can use col_deriv=True

    # No batching is allowed in ROOT. If you want to run batching flux matching you need to concatenate the vector in one dim
    xGuess0 = xGuess.squeeze(0).cpu().numpy() if xGuess.dim() > 1 else xGuess.cpu().numpy()

    # ************
    # Root process
    # ************
    f0,_ = func(xGuess0)
    print(f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f0)):.3e} of {f0.shape[0]} channels):\n\t{f0}")

    algorithm_options['col_deriv'] = True # Faster in scipy to avoid transposing the Jacobian. I can optimize it in pytorch instead, see above transpose

    with IOtools.timer(name="\t- SCIPY.ROOT multi-variate root finding method"):
        sol = root(func, xGuess0, jac=True, method=solver, tol=None, options=algorithm_options)

    f,_ = func(sol.x)
    print(f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f)):.3e} of {f.shape[0]} channels):\n\t{f}")

    # ************

    print("\t- Results from scipy solver:", sol)

    x_best = torch.tensor(sol.x).to(xGuess)

    return x_best


# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: Picard
# --------------------------------------------------------------------------------------------------------


def picard(fun, xGuess, tol=1e-6, max_it=1e3, relax_param=1.0):
    """
    Inputs:
            - xGuess is the initial guess and must be a tensor of (batch,dimX). It will be converted to batch.
            - fun is a function that must take X (batch,dimX) and provide the source term (batch,dimY)

    Outputs:
            - Optium vector x with (batch,dimX)

    """

    # xGuess = xGuess[0,:].unsqueeze(0)

    def func(X):
        S = fun(X)
        # Residual (batch)
        res = S.abs().mean(axis=1)
        # Average and Max residual between batches
        return S, res, res.mean().item(), res.max().item()

    # ******************************************************************************************************
    # Evaluate initial condition
    # ******************************************************************************************************

    # If no batch dimension add it
    x = xGuess.unsqueeze(0).clone() if xGuess.dim() == 1 else xGuess.clone()

    S, res, resMean, resMax = func(x)
    print(f"* Residual at it#0: {resMean:.2e} (max = {resMax:.2e})")
    if res.shape[0] > 1 and res.shape[0] < 100:
        str_txt = ""
        for i in range(res.shape[0]):
            str_txt += f"{res[i]:.2e}, "
        print("* Per individual:", str_txt[:-2])

    # ******************************************************************************************************
    # Peform loop
    # ******************************************************************************************************

    cont = 0
    while (resMean > tol) and (cont < max_it):
        # ---------------------------------------------------------------------------
        # Multiplier of source term for faster/slower steps towards flux matching
        # ---------------------------------------------------------------------------

        factor_motion = relax_param  # / S.abs().mean(axis=1).unsqueeze(1)

        # ---------------------------------------------------------------------------
        # Make step
        # ---------------------------------------------------------------------------

        x += S * factor_motion

        # ---------------------------------------------------------------------------
        # Evaluate and update counter
        # ---------------------------------------------------------------------------

        S, res, resMean, resMax = func(x)
        cont += 1

        if True:  # cont%25 == 0:
            print(f"\t- Residual at it#{cont}: {resMean:.2e} (max = {resMax:.2e})")

    # ---------------------------------------------------------------------------
    # Final residual
    # ---------------------------------------------------------------------------

    print(f"* Residual at it#{cont}: {resMean:.1e} (max = {resMax:.2e})")
    if res.shape[0] > 1 and res.shape[0] < 100:
        str_txt = ""
        for i in range(res.shape[0]):
            str_txt += f"{res[i]:.2e}, "
        print("* Per individual:", str_txt[:-2])

    return x


# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: Simple Relax
# --------------------------------------------------------------------------------------------------------


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


def relax(
    flux,
    xGuess,
    bounds=None,
    tol=None,
    max_it=1e5,
    relax=0.1,
    dx_max=0.05,
    dx_max_abs = None,
    dx_min_abs = None,
    print_each=1e2,
    storeValues=False,
):
    """
    Inputs:
            - flux is a function that must take X (dimX) and provide Q and QT as tensors of dimensions (1,dimY) each
    """

    print(
        f"* Flux-grad relationship of {relax} and maximum gradient jump of {dx_max*100.0:.1f}%, to achieve residual of {tol:.1e} in {max_it:.0f} iterations"
    )

    x = copy.deepcopy(xGuess)
    Q, QT = flux(x, cont=0)
    print(
        f"* Starting residual: {(Q-QT).abs().mean(axis=1)[0].item():.4e}, will run {int(max_it)-1} more evaluations",
        typeMsg="i",
    )

    store_x = x.clone()
    store_Q = (Q - QT).abs()
    for i in range(int(max_it) - 1):
        # --------------------------------------------------------------------------------------------------------
        # Iterative Strategy
        # --------------------------------------------------------------------------------------------------------

        x_new = simple_relax_iteration(x, Q, QT, relax, dx_max, dx_max_abs = dx_max_abs, dx_min_abs = dx_min_abs)

        # Clamp to bounds
        if bounds is not None:
            x_new = x_new.clamp(min=bounds[:, 0], max=bounds[:, 1])

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
