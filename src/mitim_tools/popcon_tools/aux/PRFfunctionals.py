"""	
____________________________________________________________________
																	
Realistic functional forms for T and n for POPCON analysis	(v.0.3)
																
							   		by	P. Rodriguez-Fernandez 2020

Description:
	This functional form imposes:
		1) tanh pedestal for T and n.
		2) Linear aLT profile from 0 at rho=0 to X at rho=x_a,
			where X is the specified core aLT value (default 2.0)
			and x_a is calculated by matching specified nu_T (peaking)
		3) Flat aLT profile from rho=x_a to rho=1-width_ped, where
			width_ped is the pedestal width (default 0.05).
		4) T and n share the same x_a, and aLn is calculated by matching
			specified nu_n (peaking). Density profile also has a linear
			aLn from 0 to rho=x_a and a flat one until rho=1-width_ped
		5) Pedestal is rescaled to match specified volume averages for
			T and n.

Notes:
	- Not all combinations of aLT and nu_T are valid. If aLT is too low,
		nu_T cannot be excessively high and viceversa. The code will not
		crash, but will give profiles that do not match the specified
		temperature	peaking.
		e.g. aLT = 2.0 requires nu_T to be within [1.5,3.0]

	- It is not recommended to change width_ped from the default value,
		since the Look-Up-Table hard-coded was computed using
		width_ped=0.05

	- If rho-grid is passed as argument, it is recommended to have equally
		spaced 100 points.

Example use:

	T_avol = 7.6
	n_avol = 3.1
	nu_T   = 2.5
	nu_n   = 1.4

	x, T, n = nTprofiles( T_avol, n_avol, nu_T, nu_n, aLT = 2.0 )

	Optionally:
		- rho-grid can be passed (100 points recommended)
		- Pedestal width can be passed (0.05 recommended)

____________________________________________________________________
"""

import numpy as np
from scipy import interpolate


def nTprofiles(T_avol, n_avol, nu_T, nu_n, aLT=2.0, width_ped=0.05, rho=None):
    # ---- Find parameters consistent with peaking
    x_a = find_xa_from_nu(aLT, nu_T, width_ped=width_ped)
    aLn = find_aLT_from_nu(x_a, nu_n, width_ped=width_ped)

    # ---- Evaluate profiles
    x, T, _ = EvaluateProfile(
        T_avol, width_ped=width_ped, aLT_core=aLT, width_axis=x_a, rho=rho
    )
    x, n, _ = EvaluateProfile(
        n_avol, width_ped=width_ped, aLT_core=aLn, width_axis=x_a, rho=rho
    )

    return x, T, n


def EvaluateProfile(Tavol, aLT_core, width_axis, width_ped=0.05, rho=None):
    """

    This function generates a profile from <T>, aLT and x_a

    Example:
            x,T,nu_T = EvaluateProfile(7.6,2.0,0.2)

    """

    # ~~~~ Grid

    if rho is None:
        x = np.linspace(0, 1, 100)
    else:
        x = rho
    ix_c = np.argmin(np.abs(x - (1 - width_ped)))  # Extend of core
    ix_a = np.min([ix_c, np.argmin(np.abs(x - width_axis))])  # Extend of axis

    # ~~~~ aLT must be different from zero, adding non-rational small offset
    aLT_core = aLT_core + np.pi * 1e-8

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Functional Forms (normalized to pedestal temperature)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    """
	# ~~~~ Pedestal

	Notes:
		- Because width_ped and Teped in my function represent the top values, I need to rescale them
		- The tanh does not result exactly in the top value (since it's an asymptote), so I need to correct for it

	"""
    wped_tanh = (
        width_ped / 1.5
    )  # The pedestal width in the tanh formula is 50% inside the pedestal-top width
    Tedge_aux = 1 / 2 * (1 + np.tanh((1 - x - (wped_tanh / 2)) / (wped_tanh / 2)))
    Tedge = Tedge_aux[ix_c:] / Tedge_aux[ix_c]

    # ~~~~ Core
    Tcore_aux = np.e ** (aLT_core * (1 - width_ped - x))
    Tcore = Tcore_aux[ix_a:ix_c]

    # ~~~~ Axis
    Taxis_aux = np.e ** (
        aLT_core * (-1 / 2 * x**2 / width_axis - 1 / 2 * width_axis + 1 - width_ped)
    )
    Taxis = Taxis_aux[:ix_a]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analytical Integral ("pre-factor")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Pedestal contribution (solved with Matematica)
    I1 = -0.0277778 * width_ped * (-23.3473 + 14.6132 * width_ped)

    # Core and axis contributions
    I23 = (
        1
        / aLT_core**2
        * (
            (width_axis * aLT_core * np.e ** (width_axis * aLT_core / 2) + 1)
            * np.e ** (-aLT_core * (width_axis + width_ped - 1))
            + aLT_core * width_ped
            - aLT_core
            - 1
        )
    )

    # Total (factor that relates Teped to Tavol)
    I = 2 * (I1 + I23)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluation of the profile
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Teped = Tavol / I
    T = Teped * np.hstack((Taxis, Tcore, Tedge)).ravel()
    peaking = T[0] / Tavol

    return x, T, peaking


def find_xa_from_nu(aLT_core, peaking, width_ped=0.05):
    aLTs = [
        1.0,
        1.33333333,
        1.66666667,
        2.0,
        2.33333333,
        2.66666667,
        3.0,
        3.33333333,
        3.66666667,
        4.0,
    ]
    peakings = [
        1.5,
        1.66666667,
        1.83333333,
        2.0,
        2.16666667,
        2.33333333,
        2.5,
        2.66666667,
        2.83333333,
        3.0,
    ]
    widths = [
        [
            0.55525657,
            0.78965428,
            0.94114763,
            1.04129876,
            1.10808831,
            1.15259499,
            1.18196076,
            1.20093812,
            1.2127608,
            1.21966273,
        ],
        [
            0.30947926,
            0.57192988,
            0.75135627,
            0.87761883,
            0.96722366,
            1.03104933,
            1.07655831,
            1.10896602,
            1.13196415,
            1.14819024,
        ],
        [
            0.11214453,
            0.39602688,
            0.59122468,
            0.73461171,
            0.84095725,
            0.920017,
            0.9788843,
            1.02278864,
            1.05559384,
            1.08015795,
        ],
        [
            -0.06643852,
            0.25158105,
            0.45634084,
            0.61014671,
            0.72816489,
            0.81886394,
            0.88856167,
            0.9421717,
            0.98349881,
            1.01546521,
        ],
        [
            -0.2559608,
            0.12822814,
            0.34229274,
            0.50209313,
            0.62772236,
            0.72695611,
            0.80521335,
            0.86688094,
            0.91552797,
            0.95401137,
        ],
        [
            -0.4861132,
            0.01560391,
            0.24466838,
            0.40832025,
            0.53850546,
            0.64365945,
            0.72846229,
            0.79668207,
            0.85153022,
            0.89569578,
        ],
        [
            -0.78658663,
            -0.09665591,
            0.15905574,
            0.3266974,
            0.45938999,
            0.56833991,
            0.65793142,
            0.73134083,
            0.7913545,
            0.8404178,
        ],
        [
            -1.18707197,
            -0.21891553,
            0.08104281,
            0.25509385,
            0.38925174,
            0.50036343,
            0.59324367,
            0.67062294,
            0.7348497,
            0.78807679,
        ],
        [
            -1.71726012,
            -0.36153923,
            0.00621758,
            0.19137892,
            0.32696652,
            0.43909596,
            0.53402199,
            0.61429414,
            0.68186476,
            0.73857209,
        ],
        [
            -2.406842,
            -0.53489122,
            -0.06983196,
            0.1334219,
            0.27141011,
            0.38390344,
            0.47988931,
            0.56212015,
            0.63224858,
            0.69180306,
        ],
    ]

    func = interpolate.interp2d(aLTs, peakings, widths, kind="cubic")
    result = func(aLT_core, peaking)[0]

    return result


def find_aLT_from_nu(width_axis, peaking, width_ped=0.05):
    peakings = [
        1.0,
        1.22222222,
        1.44444444,
        1.66666667,
        1.88888889,
        2.11111111,
        2.33333333,
        2.55555556,
        2.77777778,
        3.0,
    ]
    widths = [
        0.0,
        0.10555556,
        0.21111111,
        0.31666667,
        0.42222222,
        0.52777778,
        0.63333333,
        0.73888889,
        0.84444444,
        0.95,
    ]
    aLTs = [
        [
            -0.05069091,
            -0.05582754,
            -0.06176364,
            -0.06864038,
            -0.07662657,
            -0.08590871,
            -0.09666655,
            -0.1090216,
            -0.12294055,
            -0.13807588,
        ],
        [
            0.25014517,
            0.27424909,
            0.30241125,
            0.33536809,
            0.37396822,
            0.41910617,
            0.47157803,
            0.5318111,
            0.59941914,
            0.67257332,
        ],
        [
            0.51647618,
            0.56574693,
            0.62368044,
            0.69191175,
            0.772276,
            0.86662501,
            0.9764702,
            1.10238582,
            1.243156,
            1.39477072,
        ],
        [
            0.75227265,
            0.82363737,
            0.90825389,
            1.0087155,
            1.12785707,
            1.26841447,
            1.43241685,
            1.6202727,
            1.82964466,
            2.05439189,
        ],
        [
            0.96150508,
            1.05289181,
            1.16234156,
            1.29350424,
            1.45027173,
            1.63624119,
            1.8538249,
            2.1030419,
            2.38025976,
            2.67731241,
        ],
        [
            1.14814398,
            1.25848166,
            1.39215342,
            1.55400289,
            1.74908029,
            1.98187183,
            2.25510124,
            2.56826356,
            2.91637592,
            3.28940788,
        ],
        [
            1.31615985,
            1.44537832,
            1.60389941,
            1.79793635,
            2.03384305,
            2.31707304,
            2.65065277,
            3.03350782,
            3.45936777,
            3.91655387,
        ],
        [
            1.46952321,
            1.61855318,
            1.8037895,
            2.03302952,
            2.31412033,
            2.65361146,
            3.05488639,
            3.51634484,
            4.03060994,
            4.58462596,
        ],
        [
            1.61220457,
            1.78297764,
            1.99803364,
            2.26700733,
            2.59947242,
            3.00325374,
            3.482209,
            4.03434476,
            4.65147707,
            5.31949974,
        ],
        [
            1.74817444,
            1.94362311,
            2.19284179,
            2.50759466,
            2.89945963,
            3.37776654,
            3.94702751,
            4.60507774,
            5.34334378,
            6.1470508,
        ],
    ]

    func = interpolate.interp2d(widths, peakings, aLTs, kind="cubic")
    result = func(width_axis, peaking)[0]

    return result
