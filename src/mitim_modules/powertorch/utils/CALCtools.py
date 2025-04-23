import torch
import numpy as np
from mitim_tools.misc_tools import MATHtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ********************************************************************************************************************
# Normalized logaritmic gradient calculations
# ********************************************************************************************************************

def integrateGradient(x, z, z0_bound):
    """
    inputs as 
    (batch,dim)
    From tgyro_profile_functions.f90
    x is r
    z is 1/LT = =-1/T*dT/dr

    """

    # Calculate profile
    b = torch.exp(0.5 * (z[..., :-1] + z[..., 1:]) * (x[..., 1:] - x[..., :-1]))
    f1 = b / torch.cumprod(b, 1) * torch.prod(b, 1, keepdims=True)

    # Add the extra point of bounday condition
    f = torch.cat((f1, torch.ones(z.shape[0], 1).to(f1)), dim=1) * z0_bound

    return f
    

def produceGradient(r, p):
    """
    Produces -1/p * dp/dr
    or if r is roa: a/Lp
    """

    # This is the same as it happens in expro_util.f90, bound_deriv
    z = MATHtools.deriv(r, -torch.log(p), array=False)

    # # COMMENTED because this should happen at the coarse grid
    # z = tgyro_math_zfind(r,p,z=z)

    return z  # .nan_to_num(0.0) # Added this so that, when evaluating things like rotation shear, it doesn't blow

# ********************************************************************************************************************
# Linear gradient calculations
# ********************************************************************************************************************


def integrateGradient_lin(x, z, z0_bound):
    """
    (batch,dim)
    From tgyro_profile_functions.f90
    x is r
    z is -dT/dr

    """

    # Calculate profile
    b = 0.5 * (z[..., :-1] + z[..., 1:]) * (x[..., 1:] - x[..., :-1])
    f1 = b - torch.cumsum(b, 1) + torch.sum(b, 1, keepdims=True)

    # Add the extra point of bounday condition
    f = torch.cat((f1, torch.zeros(z.shape[0], 1).to(f1)), dim=1) + z0_bound

    return f


def produceGradient_lin(r, p):
    """
    Produces -dp/dr
    """

    # This is the same as it happens in expro_util.f90, bound_deriv
    z = MATHtools.deriv(r, -p, array=False)

    return z



def integrateQuadPoly(r, s, p=None):
    """
    (batch,dim)

    Computes int(s*dr), so if s is s*dV/dr, then int(s*dV), which is the full integral

    From tgyro_volume_int.f90
    r - minor raidus
    s - s*volp

    (Modified to avoid if statements and for loops)

    """

    if p is None:
        p = torch.zeros((r.shape[0], r.shape[1])).to(r)

    # First point

    x1, x2, x3 = r[..., 0], r[..., 1], r[..., 2]
    f1, f2, f3 = s[..., 0], s[..., 1], s[..., 2]

    p[..., 1] = (x2 - x1) * (
        (3 * x3 - x2 - 2 * x1) * f1 / 6 / (x3 - x1)
        + (3 * x3 - 2 * x2 - x1) * f2 / 6 / (x3 - x2)
        - (x2 - x1) ** 2 * f3 / 6 / (x3 - x1) / (x3 - x2)
    )

    # Next points
    x1, x2, x3 = r[..., :-2], r[..., 1:-1], r[..., 2:]
    f1, f2, f3 = s[..., :-2], s[..., 1:-1], s[..., 2:]

    p[..., 2:] = (
        (x3 - x2)
        / (x3 - x1)
        / 6
        * (
            (2 * x3 + x2 - 3 * x1) * f3
            + (x3 + 2 * x2 - 3 * x1) * f2 * (x3 - x1) / (x2 - x1)
            - (x3 - x2) ** 2 * f1 / (x2 - x1)
        )
    )

    try:
        p = torch.cumsum(p, 1)
    except:
        p = np.cumsum(p, 1)

    return p


def integrateFS(P, r, volp):
    """
    Based on the idea that volp = dV/dr, whatever r is

    Ptot = int_V P*dV = int_r P*V'*dr

    """

    I = integrateQuadPoly(
        np.atleast_2d(r), np.atleast_2d(P * volp), p=np.zeros((1, P.shape[0]))
    )[0, :]

    return I


"""
----------------------------------------------------------------------------------------------------------------
https://github.com/aliutkus/torchinterp1d
----------------------------------------------------------------------------------------------------------------
"""

import torch
import contextlib


class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
                A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
                A 1-D or 2-D tensor of real values. The length of `y` along its
                last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
                A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
                _both_ `x` and `y` are 1-D. Otherwise, its length along the first
                dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
                Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
            assert len(vec.shape) <= 2, "interp1d: all inputs must be " "at most 2-D."
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, "All parameters must be on the same device."
        device = device[0]

        # Checking for the dimensions
        assert v["x"].shape[1] == v["y"].shape[1] and (
            v["x"].shape[0] == v["y"].shape[0]
            or v["x"].shape[0] == 1
            or v["y"].shape[0] == 1
        ), (
            "x and y must have the same number of columns, and either "
            "the same number of row or one of them having only one "
            "row."
        )

        reshaped_xnew = False
        if (
            (v["x"].shape[0] == 1)
            and (v["y"].shape[0] == 1)
            and (v["xnew"].shape[0] > 1)
        ):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v["xnew"].shape
            v["xnew"] = v["xnew"].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v["x"].shape[0], v["xnew"].shape[0])
        shape_ynew = (D, v["xnew"].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v["xnew"].shape[0] == 1:
            v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

        torch.searchsorted(v["x"].contiguous(), v["xnew"].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ["x", "y", "xnew"]:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [
                    None,
                ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat["slopes"] = is_flat["x"]
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
                eps + (v["x"][:, 1:] - v["x"][:, :-1])
            )

            # now build the linear interpolation
            ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
            ctx.saved_tensors[0],
            [i for i in inputs if i is not None],
            grad_out,
            retain_graph=True,
        )
        result = [
            None,
        ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)
