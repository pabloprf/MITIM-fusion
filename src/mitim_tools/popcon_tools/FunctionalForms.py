import pdb
import torch
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.popcon_tools.utils import PRFfunctionals, FUNCTIONALScalc
from mitim_tools.misc_tools import MATHtools, IOtools, FARMINGtools, GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.powertorch.physics import CALCtools


def parabolic(Tbar=5, nu=2.5, rho=None, Tedge=0):
    if rho is None:
        rho = np.linspace(0, 1, 100)

    nu_mod = (nu - Tedge / Tbar) / (1 - Tedge / Tbar)

    T = (Tbar - Tedge) * nu_mod * ((1.0 - rho**2.0) ** (nu_mod - 1.0)) + Tedge

    T[-1] += 1e-5  # To avoid zero problems

    return rho, T


def PRFfunctionals_Hmode(T_avol, n_avol, nu_T, nu_n, aLT=2.0, width_ped=0.05, rho=None):
    return PRFfunctionals.nTprofiles(
        T_avol, n_avol, nu_T, nu_n, aLT=aLT, width_ped=width_ped, rho=rho
    )


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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pedestal tanh fit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pedestal_tanh(Y_top, Y_sep, width_top, x=None):

    from scipy.optimize import curve_fit

    # if fitting this to an EPED pedestal, remember that width_eped
    # is defined in terms of normalized poloidal flux and not rho.
    # Be sure to interpolate accordingly 

    if x is None:
        x = np.linspace(0, 1, 101)

    x_top = 1-width_top
    x_ped = 1-2*width_top/3
    x_mid = 1-width_top/2

    # Fit
    density_func = lambda x, a, b: Y_sep + b * (
        np.tanh(2*(1-x_mid)/width_top/a)-np.tanh(2*(x-x_mid)/width_top/a)
        )
    
    n0, pcov = curve_fit(density_func, [x_top,x_ped,1.0], [Y_top,Y_top/1.08, Y_sep])

    Y = density_func(x, n0[0], n0[1])

    return x, Y

def MITIMfunctional_aLyTanh(
        x_top,
        y_top,
        y_sep,
        aLy, # aLy is a derivative wrt rho, not truly with respect to roa, 1/Ly_x
        x_a = 0.3,
        x = None,
        nx = 201,   # If x grid not provided, create one with this number of points
        plotYN = False,
        ):
    ''' 
    Create a profile with a pedestal tanh and a core with a 1/Ly profile
    '''

    if x is None:
        x = np.linspace(0, 1, nx)

    # Create pedestal profile
    x, Yped = pedestal_tanh(y_top, y_sep, 1-x_top, x=x)

    # Find where the core starts
    bc_index = np.argmin(np.abs(x-x_top))
    xcore = x[:bc_index]

    # Create core 1/Ly profile
    aLy_profile = np.zeros_like(xcore)
    linear_region = xcore <= x_a
    aLy_profile[linear_region] = (aLy / x_a) * xcore[linear_region]
    aLy_profile[~linear_region] = aLy 

    # Create core profile
    Ycore = CALCtools.integrateGradient(torch.from_numpy(xcore).unsqueeze(0),
                                        torch.from_numpy(aLy_profile).unsqueeze(0),
                                        y_top
                                    ).numpy()[0]

    # Merge
    y = np.concatenate([Ycore, Yped[bc_index:]])

    if plotYN:
        fig, axs = plt.subplots(nrows=2, figsize=(6, 8))

        ax = axs[0]
        ax.plot(x, y, '-o', c='b', markersize=3, lw=1.0)

        ax.axvline(x_top, color='g', ls='--')
        ax.axhline(y_top, color='g', ls='--')
        ax.axhline(y_sep, color='k', ls='--')
        ax.axvline(x_a, color='r', ls='--')

        ax.set_xlabel(r'$x$')
        ax.set_xlim([0,1])
        ax.set_ylabel(r'$y$')
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1]
        aLy_reconstructed = CALCtools.produceGradient(torch.from_numpy(x), torch.from_numpy(y)).numpy()

        ax.plot(x, aLy_reconstructed, '-o', c='b', markersize=3, lw=1.0)
        ax.plot(xcore, aLy_profile, '-*', c='r', markersize=1, lw=1.0) 
        ax.axvline(x_a, color='r', ls='--')

        ax.axvline(x_top, color='g', ls='--')
        ax.axvline(x_a, color='r', ls='--')

        ax.set_xlabel(r'$x$')
        ax.set_xlim([0,1])
        ax.set_ylabel(r'$1/Ly$')
        ax.set_ylim([0,2*aLy])
        GRAPHICStools.addDenseAxis(ax)

        plt.show()

    return x, y
