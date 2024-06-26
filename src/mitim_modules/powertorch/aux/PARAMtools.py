import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function


def parameterize_curve(
    x_coord,
    y_coord,
    x_coarse_tensor,
    parameterize_in_aLx=True,
    preSmoothing=False,
    PreventNegative=False,
    ):
    """
    Notes:
        - x_coarse_tensor must be torch
    """

    # **********************************************************************************************************
    # Define the integrator and derivator functions (based on whether I want to parameterize in aLx or in gradX)
    # **********************************************************************************************************

    if parameterize_in_aLx:
        integrator_function, derivator_function = (
            CALCtools.integrateGradient,
            CALCtools.produceGradient,
        )
    else:
        integrator_function, derivator_function = (
            CALCtools.integrateGradient_lin,
            CALCtools.produceGradient_lin,
        )

    ygrad_coord = derivator_function(
        torch.from_numpy(x_coord).to(x_coarse_tensor), torch.from_numpy(y_coord).to(x_coarse_tensor)
    )

    # **********************************************************************************************************
    # Get control points
    # **********************************************************************************************************

    x_coarse = x_coarse_tensor[1:].cpu().numpy()

    # Clip to zero if I want to prevent negative values
    if PreventNegative:
        ygrad_coord = ygrad_coord.clip(0)

    # Perform smoothing to grab from when smoothing option is active
    if preSmoothing:
        from scipy.signal import savgol_filter

        filterlen = int(int(len(x_coord) / 20 / 2) * 10) + 1  # 651
        yV_smth = torch.from_numpy(savgol_filter(ygrad_coord, filterlen, 2)).to(ygrad_coord)
        points_untouched = 5

    """
    Define region to get control points from
    ------------------------------------------------------------
	Trick: Addition of extra point
		This is important because if I don't, when I combine the trailing edge and the new
		modified profile, there's going to be a discontinuity in the gradient.
	"""
    
    ir_end = np.argmin(np.abs(x_coord - x_coarse[-1]))

    if ir_end < len(x_coord) - 1:
        ir = ir_end + 2  # To prevent that TGYRO does a 2nd order derivative
        x_coarse = np.append(x_coarse, [x_coord[ir]])
    else:
        ir = ir_end

	# Definition of trailing edge. Any point after, and including, the extra point
    x_trail = torch.from_numpy(x_coord[ir:])
    y_trail = y_coord[ir:]
    x_notrail = x_coord[: ir + 1]

    # Produce control points, including a zero at the beginning
    aLy_coarse = [[0.0, 0.0]]
    for cont, i in enumerate(x_coarse):
        if (
            preSmoothing
            and (cont < len(x_coarse) - 1 - points_untouched)
            and (cont > 0)
        ):
            """
            Perform some radial averaging if points are not the last ones or the first
            """
            yValue = yV_smth[np.argmin(np.abs(x_coord - i))]
        else:
            """
            Simply grab the values
            """
            yValue = ygrad_coord[np.argmin(np.abs(x_coord - i))]

        aLy_coarse.append([i, yValue.cpu().item()])

    aLy_coarse = torch.from_numpy(np.array(aLy_coarse)).to(ygrad_coord)

    # Since the last one is an extra point very close, I'm making it the same
    aLy_coarse[-1, 1] = aLy_coarse[-2, 1]

    # Boundary condition at point moved by gridPointsAllowed
    y_bc = torch.from_numpy(interpolation_function([x_coarse[-1]], x_coord, y_coord)).to(
        ygrad_coord
    )

    # Boundary condition at point (ACTUAL THAT I WANT to keep fixed, i.e. rho=0.8)
    y_bc_real = torch.from_numpy(interpolation_function([x_coarse[-2]], x_coord, y_coord)).to(
        ygrad_coord
    )

    # **********************************************************************************************************
    # Define deparametrizer functions
    # **********************************************************************************************************

    def deparametrizer(x, y, printMessages=False):
        return constructCurve(
            x,
            y,
            y_bc,
            y_bc_real,
            x_trail,
            y_trail,
            integrator_function,
            derivator_function,
            xBS=x_notrail,
            xCoord=x_coord,
            Te=y_coord,
            extraP=aLy_coarse[-1],
            printMessages=printMessages,
        )

    """
	Construct curve in a coarse grid
	----------------------------------------------------------------------------------------------------
	This constructs a curve in any grid, with any batch given in y=yCPs.
	Useful for surrogate evaluations. Fast in a coarse grid. For HF evaluations,
	I need to do in a finer grid so that it is consistent with TGYRO.
	x=xCPs, y=yCPs must be (batch, radii),	BC must be (1)
	"""

    def deparametrizer_coarse(x, y, **kwargs):
        return (
            x,
            integrator_function(x, y, y_bc_real),
        )

    def deparametrizer_coarse_middle(x, y, **kwargs):
        """
        Deparamterizes a finer profile based on the values in the coarse.
        Reason why something like this is not used for the full profile is because derivative of this will not be as original,
                which is needed to match TGYRO
        Notes:
            - x is in the new grid,
            - yOrig is the values evaluated in xOrig
        """
        yCPs = CALCtools.Interp1d()( aLy_coarse[:, 0][:-1].repeat(( aLy_coarse[:, 0][:-1].shape[0], 1)), y, x)
        return x, integrator_function(x, yCPs, y_bc_real)

    return (
        aLy_coarse,
        deparametrizer,
        deparametrizer_coarse,
        deparametrizer_coarse_middle,
    )

def constructCurve(
    xCPs,
    yCPs,
    BC,
    BC_real,
    x_trail,
    y_trail,
    integrator_function,
    derivator_function,
    xBS=None,
    plotYN=False,
    xCoord=None,
    Te=None,
    extraP=None,
    printMessages=True,
    ):
    """
        - xCPs is a 1D array, but yCPs can be a 2D array for a batch of individuals: (batch,x)
        - I am assuming it is 1/LT for parameterization, but gives T
    """

    yCPs = torch.atleast_2d(yCPs)

    #x_trail = torch.from_numpy(x_trail).to(xCPs)
    y_trail = torch.from_numpy(y_trail).to(xCPs)

    if xCPs.dim() == 2:
        xCPs = xCPs[0, :]

    # Add the extra trick point
    if extraP is not None:
        extraP_x, extraP_y = extraP
        xCPs = torch.cat((xCPs, extraP_x.repeat((1))))
        yCPs = torch.cat((yCPs, extraP_y.repeat((yCPs.shape[0], 1))), dim=1)

        # force extra point to the same as previous. I think this avoids problems
        # yCPs = torch.cat((yCPs,yCPs[:,-1].repeat(yCPs.shape[0],1)),dim=1)

    # ----------------------------------------------------------------------
    # 	Profile generator
    # ----------------------------------------------------------------------

    # Model curve (basically, what happens in between points)
    xBS = torch.from_numpy(xBS).to(xCPs)
    yBS = CALCtools.Interp1d()(
        xCPs.repeat(yCPs.shape[0], 1), yCPs, xBS.repeat(yCPs.shape[0], 1)
    )

    """
	---------------------------------------------------------------------------------------------------------
		Trick 1: smoothAroundCoarsing
			TGYRO will use a 2nd order scheme to obtain gradients out of the profile, so a piecewise linear
			will simply not give the right derivatives.
			Here, this rough trick is to modify the points in gradient space around the coarse grid with the
			same value of gradient, so in principle it doesn't matter the order of the derivative.
	"""
    num_around = 1
    try:
        for i in range(xCPs.shape[0] - 2):
            ir = torch.argmin(torch.abs(xCPs[i + 1] - xBS))
            for k in range(-num_around, num_around + 1, 1):
                yBS[:, ir + k] = yBS[:, ir]
    except:
        if printMessages:
            print(
                "\t- Could not perform smoothing around the coarse points", typeMsg="w"
            )
    # --------------------------------------------------------------------------------------------------------

    xBS_deriv, yBS_deriv = xBS.clone(), yBS.clone()

    yBS = integrator_function(xBS.repeat(yBS.shape[0], 1), yBS_deriv, BC)

    """
	Trick 2: Correct BC
		The BC for the profile integration started at gridPointsAllowed, but that's not the real
		BC. I want the temperature fixed at my first point that I actually care for.
		Here, I multiply the profile to get that.
		Multiplication works because:
			1/LT = 1/T * dT/dr
			1/LT' = 1/(T*m) * d(T*m)/dr = 1/T * dT/dr = 1/LT
		Same logarithmic gradient, but with the right boundary condition

	"""
    ir = torch.argmin(torch.abs(xBS - xCPs[-2]))
    yBS = yBS * torch.transpose((BC_real / yBS[:, ir]).repeat(yBS.shape[1], 1), 0, 1)

    """
	Add trailing edge
	"""

    y_trailnew = copy.deepcopy(y_trail).repeat(yBS.shape[0], 1)

    xBS = torch.cat((xBS[:-1], x_trail), dim=0)
    yBS = torch.cat((yBS[:, :-1], y_trailnew), dim=1)

    # ----------------------------------------------------------------------
    # 	Plot / Debug
    # ----------------------------------------------------------------------

    if printMessages:
        batch_pos_plot = 0

        deriv0 = derivator_function(
            torch.from_numpy(xCoord).to(xCPs), torch.from_numpy(Te).to(xCPs)
        )
        deriv1 = derivator_function(xBS, yBS[batch_pos_plot, :])

        v0 = interpolation_function(xCPs.cpu(), xCoord, deriv0.cpu())
        per0 = (v0 - yCPs.cpu().numpy()) / v0
        if per0[0, 1:-1].max() * 100 > 0.1:
            print(
                f"\t\t- [MAX] Error bw aLT(original) and CPs: {per0[0,1:-1].max()*100:.3f}%",
                typeMsg="w",
            )

        v0 = interpolation_function(xCPs.cpu(), xCoord, deriv1.cpu())
        per1 = (v0 - yCPs.cpu().numpy()) / v0
        if per1[0, 1:-1].max() * 100 > 0.1:
            print(
                f"\t\t- [MAX] Error bw aLT(Integr(interpolation)) and CPs (max): {per1[0,1:-1].max()*100:.3f}%",
                typeMsg="w",
            )

        v0 = interpolation_function(xCPs[-2].cpu(), xCoord, Te)
        v1 = interpolation_function(xCPs[-2].cpu(), xBS.cpu(), yBS[batch_pos_plot].cpu())
        per2 = (v0 - v1) / v0
        if per2 * 100 > 0.1:
            print(f"\t\t- Error bw BCs: {per2*100:.3f}", typeMsg="w")

    if plotYN:
        fig, ax = plt.subplots(nrows=2, sharex=True)

        # ---- Original
        ax[0].plot(xCoord, Te, "-s", color="k", label="original", lw=0.5, markersize=1)
        ax[1].plot(
            xCoord, deriv0, "-s", color="k", label="aLT(original)", lw=0.5, markersize=1
        )

        # ---- CPs
        ax[1].plot(xCPs, yCPs[0, :], "s", color="m", label="CPs")
        for i in range(xCPs.shape[0]):
            ax[0].axvline(x=xCPs[i], ls="--", lw=0.5, c="m")
            ax[1].axvline(x=xCPs[i], ls="--", lw=0.5, c="m")

        # ---- Fit

        ax[1].plot(
            xBS_deriv,
            yBS_deriv[batch_pos_plot, :],
            "-ob",
            label="interpolation",
            markersize=2,
        )
        ax[0].plot(
            xBS,
            yBS[batch_pos_plot, :],
            "-ob",
            label="Integr(interpolation) - Passed",
            markersize=2,
        )
        ax[1].plot(
            xBS, deriv1, "--or", label="aLT(Integr(interpolation))", markersize=2
        )

        # ---- Checks
        ycheck = integrator_function(
            xCPs[:-1].repeat(yBS.shape[0], 1),
            yCPs[:, :-1],
            BC_real.repeat(yBS.shape[0], 1),
        )

        ax[0].plot(
            xCPs[:-1],
            ycheck[0, :],
            "-s",
            c="m",
            label="aLT(Integr(CPs))",
            markersize=1,
            lw=0.5,
        )

        # ax[0].set_xlim([0,1]);
        ax[0].set_ylim(bottom=0)
        ax[1].legend()
        ax[0].legend()

        plt.show()

        embed()

    return xBS, yBS
