import torch
import copy
import numpy as np
from scipy.optimize import root
from IPython import embed
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level



# --------------------------------------------------------------------------------------------------------
#  Ready to go optimization tool: MV
# --------------------------------------------------------------------------------------------------------


def powell(
    flux, xGuess, optim_fun, writeTrajectory=False, algorithmOptions={}, solver="lm"
):
    """
    Inputs:
            - xGuess is the initial guess and must be a tensor of (1,dimX) or (dimX). It will be transformed to dimX.
            - optim_fun is a function that must take X (dimX) and provide Q and QT as tensors of dimensions (1,dimY) each
    Outputs:
            - Optium vector x with (dimX)
    Notes:
            - The porblem must be: dimX = dimY
            - Must all be tensors that allow Jacobian calculation
    """

    # torch.autograd.set_detect_anomaly(True)

    vectorize = True  # Fast calculation of the jacobian (much faster, but experimental)

    def func(x, dfT1=torch.zeros(1).to(xGuess)):
        # Root will work with arrays, convert to tensor with AD
        X = torch.tensor(x, requires_grad=True).to(dfT1)

        # Evaluate value and local jacobian
        QhatD = flux(X)
        JD = torch.autograd.functional.jacobian(
            flux, X, strict=False, vectorize=vectorize
        )

        # Back to arrays
        return QhatD.detach().cpu().numpy(), JD.detach().cpu().numpy()

    # No batching is allowed in ROOT. If you want to run batching flux matching you need to concatenate the vector in one dim
    xGuess0 = (
        xGuess.squeeze(0).cpu().numpy() if xGuess.dim() > 1 else xGuess.cpu().numpy()
    )

    # ************
    # Root process
    # ************
    f0,_ = func(xGuess0)
    print(
        f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f0)):.3e} of {f0.shape[0]} channels):\n\t{f0}",
        verbose=read_verbose_level(),
    )

    sol = root(
        func, xGuess0, jac=True, method=solver, tol=None, options=algorithmOptions
    )

    f,_ = func(sol.x)
    print(
        f"\t|f-fT|*w (mean (over batched members) = {np.mean(np.abs(f)):.3e} of {f.shape[0]} channels):\n\t{f}",
        verbose=read_verbose_level(),
    )
    # ************

    if read_verbose_level() in [4, 5]:
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


def simple_relax_iteration(x, Q, QT, relax, dx_max):
    # Calculate step (if target is larger than transport, dx is positive because I want to increase gradients)
    dx = relax * (QT - Q) / (Q**2 + QT**2) ** 0.5

    # Prevent big steps - Clamp to the max step (with the right sign)
    ix = dx.abs() > dx_max
    dx[ix] = dx_max * (dx[ix] / dx[ix].abs())

    # Update (Note for PRF: abs() was added by me, I think it performs better that way!)
    x_new = x + x.abs() * dx

    return x_new


def relax(
    flux,
    xGuess,
    bounds=None,
    tol=None,
    max_it=1e5,
    relax=0.1,
    dx_max=0.05,
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

        x_new = simple_relax_iteration(x, Q, QT, relax, dx_max)

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
