import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.popcon_tools.aux import PRFfunctionals, FUNCTIONALScalc
from mitim_tools.misc_tools import MATHtools
from mitim_tools.misc_tools.IOtools import printMsg as print

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARABOLIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def parabolic(Tbar=5, nu=2.5, rho=None, Tedge=0):
    if rho is None:
        rho = np.linspace(0, 1, 100)

    nu_mod = (nu - Tedge / Tbar) / (1 - Tedge / Tbar)

    T = (Tbar - Tedge) * nu_mod * ((1.0 - rho**2.0) ** (nu_mod - 1.0)) + Tedge

    T[-1] += 1e-5  # To avoid zero problems

    return rho, T


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# H-mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def PRFfunctionals_Hmode(T_avol, n_avol, nu_T, nu_n, aLT=2.0, width_ped=0.05, rho=None):
    return PRFfunctionals.nTprofiles(
        T_avol, n_avol, nu_T, nu_n, aLT=aLT, width_ped=width_ped, rho=rho
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# L-mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def PRFfunctionals_Lmode(T_avol, n_avol, nu_n, rho=None, debug=False):
    """
    Double linear

    Method: Cubic spline to search for value of edge gradient that provides vol avg
    """

    points_search = 50

    if rho is None:
        x = np.linspace(0, 1, 100)
    else:
        x = rho

    x = np.repeat(np.atleast_2d(x), points_search, axis=0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Temperature
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    g2_T = 3
    g1Range_T = [1, 50]
    xtransition_T = 0.8
    T_roa1 = 0.25

    gs, Tvol = np.linspace(g1Range_T[0], g1Range_T[1], points_search), []

    T = FUNCTIONALScalc.doubleLinear_aLT(x, gs, g2_T, xtransition_T, T_roa1)
    Tvol = FUNCTIONALScalc.calculate_simplified_volavg(x, T)
    g1 = MATHtools.extrapolateCubicSpline(T_avol, Tvol, gs)
    T = FUNCTIONALScalc.doubleLinear_aLT(
        np.atleast_2d(x[0]), g1, g2_T, xtransition_T, T_roa1
    )[0]

    if g1 < g1Range_T[0] or g1 > g1Range_T[1]:
        print(f">> Gradient aLT outside of search range ({g1})", typeMsg="w")

    if debug:
        fig, ax = plt.subplots()
        ax.plot(gs, Tvol, "-s")
        ax.axhline(y=T_avol)
        ax.axvline(x=g1)
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Density
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    n_roa1 = n_avol / 3

    _, n = parabolic(Tbar=n_avol, nu=nu_n, rho=x[0], Tedge=n_roa1)

    # g2_n 		= 1
    # g1Range_n 	= [0.1,10]
    # xtransition_n 		= 0.8

    # gs,nvol = np.linspace(g1Range_n[0],g1Range_n[1],points_search), []

    # n 		= FUNCTIONALScalc.doubleLinear_aLT(x,gs,g2_n,xtransition_n,n_roa1)
    # nvol 	= FUNCTIONALScalc.calculate_simplified_volavg(x,n)
    # g1 		= MATHtools.extrapolateCubicSpline(n_avol,nvol,gs)
    # n 		= FUNCTIONALScalc.doubleLinear_aLT(np.atleast_2d(x[0]),g1,g2_n,xtransition_n,n_roa1)[0]

    # if g1<g1Range_n[0] or g1>g1Range_n[1]: print(f'>> Gradient aLn outside of search range ({g1})',typeMsg='w')

    return x[0], T, n
