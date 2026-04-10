import copy
import torch
import numpy as np
from mitim_modules.powertorch.utils import CALCtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

def piecewise_linear(
    x_coord,
    y_coord_raw,
    x_coarse_tensor,
    parameterize_in_aLx=True,
    multiplier_quantity=1.0,
    smooth_around_coarsing=True,
    ):
    """
    Notes:
        - x_coarse_tensor must be torch
    """

    # **********************************************************************************************************
    # Define the integrator and derivator functions (based on whether I want to parameterize in aLx or in gradX)
    # **********************************************************************************************************

    if parameterize_in_aLx:
        # 1/Lx = -1/X*dX/dr
        integrator_function, derivator_function = (
            CALCtools.integration_Lx,
            CALCtools.derivation_into_Lx,
        )
    else:
        # -dX/dr
        integrator_function, derivator_function = (
            CALCtools.integration_dxdr,
            CALCtools.derivation_into_dxdr,
        )

    y_coord = torch.from_numpy(y_coord_raw).to(x_coarse_tensor) * multiplier_quantity

    ygrad_coord = derivator_function( torch.from_numpy(x_coord).to(x_coarse_tensor), y_coord )

    # **********************************************************************************************************
    # Get control points
    # **********************************************************************************************************

    x_coarse = x_coarse_tensor[1:].cpu().numpy()

    # Index of the last control point on the fine grid
    ir_end = np.argmin(np.abs(x_coord - x_coarse[-1]))

    # Trailing edge: everything beyond the last control point
    x_trail = torch.from_numpy(x_coord[ir_end:]).to(x_coarse_tensor)
    y_trail = y_coord[ir_end:]
    x_notrail = torch.from_numpy(x_coord[: ir_end + 1]).to(x_coarse_tensor)

    # Gradient of the original profile at the trailing edge start (for Hermite blending)
    ygrad_trail_start = ygrad_coord[ir_end]

    # Produce control points, including a zero at the beginning
    aLy_coarse = [[0.0, 0.0]]
    for cont, i in enumerate(x_coarse):
        yValue = ygrad_coord[np.argmin(np.abs(x_coord - i))]
        aLy_coarse.append([i, yValue.cpu().item()])

    aLy_coarse = torch.from_numpy(np.array(aLy_coarse)).to(ygrad_coord)

    # Boundary condition at the last control point
    y_bc_real = torch.from_numpy(interpolation_function([x_coarse[-1]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # **********************************************************************************************************
    # Define profile_constructor functions
    # **********************************************************************************************************

    def profile_constructor_coarse(x, y, multiplier=multiplier_quantity):
        """
        Construct curve in a coarse grid
        ----------------------------------------------------------------------------------------------------
        This constructs a curve in any grid, with any batch given in y=y.
        Useful for surrogate evaluations. Fast in a coarse grid. For HF evaluations,
        I need to do in a finer grid so that it is consistent with TGYRO.
        x, y must be (batch, radii),	y_bc must be (1)
        """
        return x, integrator_function(x, y, y_bc_real) / multiplier

    def profile_constructor_middle(x, y, multiplier=multiplier_quantity):
        """
        Deparamterizes a finer profile based on the values in the coarse.
        Reason why something like this is not used for the full profile is because derivative of this will not be as original,
                which is needed to match TGYRO
        """
        yCPs = CALCtools.Interp1d_torch()(aLy_coarse[:, 0].repeat((y.shape[0], 1)), y, x)
        return x, integrator_function(x, yCPs, y_bc_real) / multiplier

    def profile_constructor_fine(x, y, multiplier=multiplier_quantity):
        """
        Notes:
            - x is a 1D array, but y can be a 2D array for a batch of individuals: (batch,x)
            - I am assuming it is 1/LT for parameterization, but gives T
        """

        y = torch.atleast_2d(y)
        x = x[0, :] if x.dim() == 2 else x

        # Piecewise-linear interpolation of gradients from coarse to fine grid
        yBS = CALCtools.Interp1d_torch()(x.repeat(y.shape[0], 1), y, x_notrail.repeat(y.shape[0], 1))

        # smoothAroundCoarsing: flatten gradient at neighbors of each control point so that
        # any derivative stencil (including higher-order) recovers the correct gradient value.
        if smooth_around_coarsing:
            num_around = 1
            for i in range(x.shape[0] - 1):
                ir = torch.argmin(torch.abs(x[i + 1] - x_notrail))
                for k in range(-num_around, num_around + 1, 1):
                    if 0 <= ir + k < yBS.shape[1]:
                        yBS[:, ir + k] = yBS[:, ir]

        # Integrate with BC directly at the last control point (no extra point, no rescaling)
        yBS = integrator_function(x_notrail.repeat(yBS.shape[0], 1), yBS.clone(), y_bc_real)

        # Smooth Hermite blending into the trailing edge
        # -----------------------------------------------
        # At the junction (last control point), the reconstructed profile has value y_bc_real
        # and gradient from the last coarse a/LT. The original profile continues beyond.
        # A cubic Hermite blend over a small region ensures C1 continuity (no kink).

        n_trail = x_trail.shape[0]
        if n_trail > 1:
            # Blend width: a few grid points into the trailing edge
            n_blend = min(max(3, n_trail // 3), n_trail)
            x_blend = x_trail[:n_blend]
            blend_width = x_blend[-1] - x_blend[0]

            if blend_width > 0:
                # Smoothstep: 0 at junction, 1 at end of blend region
                t = (x_blend - x_blend[0]) / blend_width
                w = 3 * t**2 - 2 * t**3  # C1 Hermite smoothstep

                # Reconstructed value at junction = yBS[:, -1] (the BC point)
                y_junction = yBS[:, -1:]  # (batch, 1)

                # Original trailing edge values
                y_orig_blend = y_trail[:n_blend].unsqueeze(0).repeat(yBS.shape[0], 1)
                y_orig_rest = y_trail[n_blend:].unsqueeze(0).repeat(yBS.shape[0], 1) if n_blend < n_trail else torch.empty(yBS.shape[0], 0).to(yBS)

                # Blend: (1-w)*junction_extrapolated + w*original
                # For the extrapolation from the junction, use the last gradient to extend
                last_grad = yBS[:, -1:] - yBS[:, -2:-1]  # finite diff of profile at junction
                dx_from_junction = (x_blend - x_blend[0]).unsqueeze(0)
                y_extrap = y_junction + last_grad / (x_notrail[-1] - x_notrail[-2]) * dx_from_junction

                y_blended = (1 - w).unsqueeze(0) * y_extrap + w.unsqueeze(0) * y_orig_blend

                # Assemble: notrail[:-1] + blended + rest of original
                x_full = torch.cat((x_notrail[:-1], x_trail), dim=0)
                yBS_full = torch.cat((yBS[:, :-1], y_blended, y_orig_rest), dim=1)
            else:
                # Blend width is zero (single trailing point), just concat
                x_full = torch.cat((x_notrail[:-1], x_trail), dim=0)
                y_trailnew = y_trail.unsqueeze(0).repeat(yBS.shape[0], 1)
                yBS_full = torch.cat((yBS[:, :-1], y_trailnew), dim=1)
        else:
            # No trailing edge points, return as-is
            x_full = x_notrail
            yBS_full = yBS

        return x_full, yBS_full / multiplier

    # **********************************************************************************************************

    return (
        aLy_coarse,
        profile_constructor_fine,
        profile_constructor_coarse,
        profile_constructor_middle,
    )