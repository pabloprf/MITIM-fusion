"""
Plasma physics models/info/approximations
"""

import scipy, torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.misc_tools import MATHtools
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.IOtools import printMsg as print

"""
Acknowledgement:
	Various definitions of plasma physics quantities are inspired or extracted from the
	TGYRO code:

		J. Candy, C. Hollan2, R. E. Waltz, M. R. Fahey, and E. Belli, "Tokamak profile prediction using
		direct gyrokinetic and neoclassical simulation", Physics of Plasmas 16, 060704 (2009)

"""

# ---------------------------
# CONSTANTS
# ---------------------------

pi = 3.14152

# SI
e_J = 1.60218e-19  # C
u = 1.66054e-27  # kg
eps0 = 8.85419e-12  # F/m

# cgs (for TGYRO-inspired calculations)
me_g = 9.1094e-28  # g
k = 1.6022e-12  # erg/eV
e = 4.8032e-10  # statcoul
c_light = 2.9979e10  # cm/s
md = 3.34358e-24
me_u = 5.4488741e-04  # as in input.gacode


def magneticshear(q, rmin, R0):
    """
    [Rice PRL 2013]
            shat defined as r/q * dq/dr, where r is the minor radius coordinate
            Ls defined as Ls = R0 * q / shat, where R0 is the major radius
    """

    dqdr = MATHtools.deriv(rmin, q)

    shat = rmin / q * dqdr

    Ls = R0 * q / shat

    return shat, Ls


def parabolicProfile(Tbar=5, nu=2.5, rho=None, Tedge=0):
    return FunctionalForms.parabolic(Tbar=Tbar, nu=nu, rho=rho, Tedge=Tedge)


def constructVtorFromMach(Mach, Ti_keV, mi_u):
    vTi = (2 * (Ti_keV * 1e3 * e_J) / (mi_u * u)) ** 0.5  # m/s
    Vtor = Mach * vTi  # m/s

    return Vtor


def RicciMetric(y1, y2, y1_std, y2_std, h=None, d0=2.0, l=1.0):
    """
    From Molina PoP 2023
    Notes:
            - Arrays must have [batch,dim] to give [batch]
    From Ricci PoP 2011
            - d0 represents the transition from agreement to disagreement.
              The value of d0 should then correspond to a discrepancy between experiment and simulation
              that is comparable to their uncertainties.
              PRF: d0=2.0 means that I'm ok with 2sigma?
            - The parameter l instead denotes the sharpness of the transition from Rj =0 to Rj = 1

    """

    if h is None:
        h = np.ones(y1.shape[1])
    H = 1 / h

    d = np.zeros((y1.shape[0], y1.shape[1]))
    R = np.zeros((y1.shape[0], y1.shape[1]))
    S = np.zeros((y1.shape[0], y1.shape[1]))

    for i in range(d.shape[1]):
        d[:, i] = (
            (y1[:, i] - y2[:, i]) ** 2 / (y1_std[:, i] ** 2 + y2_std[:, i] ** 2)
        ) ** 0.5
        R[:, i] = 0.5 * (np.tanh((d[:, i] - d0) / l) + 1)
        S[:, i] = np.exp(
            -(y1_std[:, i] + y2_std[:, i]) / (abs(y1[:, i]) + abs(y2[:, i]))
        )  # Quantification of measurement precision (0-1)

    QR = (H * S).sum(axis=1)  # Quality of comparison

    chiR = (R * H * S).sum(
        axis=1
    ) / QR  # Agreement between measurement and simulation (0-1)

    return QR, chiR


def LHthreshold_nmin(Ip, Bt, a, Rmajor):
    nLH_min = (
        0.07 * Ip**0.34 * Bt**0.62 * a ** (-0.95) * (Rmajor / a) ** 0.4
    )  # in 1E20 m^-3

    return nLH_min


def LHthreshold_Martin1(n, Bt, S, nmin=0):
    return 0.0488 * n**0.717 * Bt**0.803 * S**0.941 * nminfactor(nmin, n)


def LHthreshold_Martin2(n, Bt, a, Rmajor, nmin=0):
    return (
        2.15
        * n**0.782
        * Bt**0.772
        * a**0.975
        * nminfactor(nmin, n)
        * Rmajor**0.999
    )


def LHthreshold_Schmid1(n, Bt, S, nmin=0):
    return 0.0029 * (n * 10) ** 1.05 * Bt**0.68 * S**0.93 * nminfactor(nmin, n)


def LHthreshold_Schmid2(n, Bt, S, nmin=0):
    return 0.0021 * (n * 10) ** 1.07 * Bt**0.76 * S * nminfactor(nmin, n)


def LHthreshold_Martin1_low(n, Bt, S, nmin=0):
    return (
        0.0488
        * np.exp(-0.057)
        * n ** (0.717 - 0.035)
        * Bt ** (0.803 - 0.032)
        * S ** (0.941 - 0.019)
        * nminfactor(nmin, n)
    )


def LHthreshold_Martin1_up(n, Bt, S, nmin=0):
    return (
        0.0488
        * np.exp(+0.057)
        * n ** (0.717 + 0.035)
        * Bt ** (0.803 + 0.032)
        * S ** (0.941 + 0.019)
        * nminfactor(nmin, n)
    )


def nminfactor(nmin, n):
    try:
        l = len(n)
    except:
        l = 0
    if l == 0:
        n = [n]

    try:
        l2 = len(nmin)
    except:
        l2 = 0
    if l2 == 0:
        nmin = [nmin]

    nminfact = []
    for i in range(len(n)):
        if n[i] < nmin[i]:
            nminfact.append((nmin[i] / n[i]) ** 2)
        else:
            nminfact.append(1.0)

    if l == 0:
        return nminfact[0]
    else:
        return np.array(nminfact)


def convective_flux(Te, Gamma_e, factor=3 / 2):
    # keV and 1E20 m^-3/s (or /m^2)

    Te_J = Te * 1e3 * e_J
    Gamma = Gamma_e * 1e20

    Qe_conv = factor * Te_J * Gamma * 1e-6  # MW (or /m^2)

    return Qe_conv


def Greenwald_density(Ip_MA, a_m):
    nG = Ip_MA / (pi * a_m**2)

    return nG  # In 20


def Bunit(phi_wb, rmin, array=True):
    B = MATHtools.deriv(0.5 * rmin**2, phi_wb / (2 * np.pi), array=array)

    return B


def vThermal(T_keV, mi_kg):
    vT = np.sqrt((T_keV * 1e3 * e_J) / mi_kg)

    return vT  # m/s


def c_s(Te_keV, mref_u):
    """
    Definition:
            cs = SQRT( k*Te / mi )
    """

    precomputed_factor = 310621.12  # (1E3 * e_J / u )**(0.5)

    cs = precomputed_factor * (Te_keV / mref_u) ** (0.5)

    return cs  # m/s


def rho_s(Te_keV, mi_u, B_T):
    """
    Definition:
            rho_s = SQRT( mi*k*Te ) / ( e*B )
    """

    precomputed_factor = 0.0032194  # ( (u) * 1E3 * e_J  )**(0.5) / ( e_J )

    rho_s = precomputed_factor * (mi_u * Te_keV) ** (0.5) / B_T

    return rho_s  # m/s?


def betae(Te_keV, ne_20, B_T):
    """
    Definition:
            beta_e = 100 * ( 4. * np.pi * 1E-7 * ( 2. * self.ne*1E20 * self.Te*self.e_J*1E3 ) / self.TGLF_Bunit**2. )
    """

    precomputed_factor = (
        4.026717  # 100* (4. * np.pi * 1E-7 * 2 * 1E20 * 1.60218E-19 * 1E3)
    )

    beta_e = precomputed_factor * ne_20 * Te_keV / B_T**2

    return beta_e


def calculatePressure(Te, Ti, ne, ni):
    """
    T in keV
    n in 1E20m^-3

    p in MPa

    vectors have dimensions of (iteration, radius) or (iteration,ion,radius)
    Output will have dimensions of (iteration)

    Notes:
            - It assumes all thermal ions.
            - It only works if the vectors contain the entire plasma (i.e. roa[-1]=1.0), otherwise it will miss that contribution.
    """

    p, peT, piT = [], [], []
    for it in range(Te.shape[0]):
        pe = (Te[it, :] * 1e3 * e_J) * (ne[it, :] * 1e20) * 1e-6  # MPa
        pi = np.zeros(Te.shape[1])
        for i in range(ni.shape[1]):
            pi += (Ti[it, i, :] * 1e3 * e_J) * (ni[it, i, :] * 1e20) * 1e-6  # MPa

        peT.append(pe)
        piT.append(pi)

        # Total pressure
        press = pe + pi
        p.append(press)

    p = np.array(p)
    pe = np.array(peT)
    pi = np.array(piT)

    return p, pe, pi


def calculateVolumeAverage(rmin, var, dVdr):
    W, vals = [], []
    for it in range(rmin.shape[0]):
        W.append(CALCtools.integrateFS(var[it, :], rmin[it, :], dVdr[it, :])[-1])
        vals.append(
            CALCtools.integrateFS(np.ones(rmin.shape[1]), rmin[it, :], dVdr[it, :])[-1]
        )

    W = np.array(W) / np.array(vals)

    return W


def calculateContent(rmin, Te, Ti, ne, ni, dVdr):
    """
    T in keV
    n in 1E20m^-3
    V in m^3
    rmin in m

    Provides:
            Number of electrons (in 1E20)
            Number of ions (in 1E20)
            Electron Energy (MJ)
            Ion Energy (MJ)

    vectors have dimensions of (iteration, radius) or (iteration,ion,radius)
    Output will have dimensions of (iteration)

    Notes:
            - It assumes all thermal ions.
            - It only works if the vectors contain the entire plasma (i.e. roa[-1]=1.0), otherwise it will miss that contribution.

    """

    p, pe, pi = calculatePressure(Te, Ti, ne, ni)

    We, Wi, Ne, Ni = [], [], [], []
    for it in range(rmin.shape[0]):
        # Number of electrons
        Ne.append(
            CALCtools.integrateFS(ne[it, :], rmin[it, :], dVdr[it, :])[-1]
        )  # Number of particles total

        # Number of ions
        ni0 = np.zeros(Te.shape[1])
        for i in range(ni.shape[1]):
            ni0 += ni[it, i, :]
        Ni.append(
            CALCtools.integrateFS(ni0, rmin[it, :], dVdr[it, :])[-1]
        )  # Number of particles total

        # Electron stored energy
        Wx = 3 / 2 * pe[it, :]
        We.append(CALCtools.integrateFS(Wx, rmin[it, :], dVdr[it, :])[-1])  # MJ
        Wx = 3 / 2 * pi[it, :]
        Wi.append(CALCtools.integrateFS(Wx, rmin[it, :], dVdr[it, :])[-1])  # MJ

    We = np.array(We)
    Wi = np.array(Wi)
    Ne = np.array(Ne)
    Ni = np.array(Ni)

    return We, Wi, Ne, Ni


def gyrobohmUnits(Te_keV, ne_20, mref_u, Bunit, a):
    """
    I send Bunit because of X to XB transformations happen outside
    """

    precomputed_factor = 0.0160218  # e_J *1E17 = 0.0160218

    commonFactor = ne_20 * (rho_s(Te_keV, mref_u, Bunit) / a) ** 2

    # Particle flux
    Ggb = commonFactor * c_s(Te_keV, mref_u)  # in 1E20/s/m^2
    # Heat flux
    Qgb = Ggb * Te_keV * precomputed_factor  # in MW/m^2
    # Momentum flux
    Pgb = commonFactor * Te_keV * precomputed_factor * a * 1e6  # J/m^2
    # Exchange source
    Sgb = Qgb / a

    return Qgb, Ggb, Pgb, Sgb


def conduction(n19, TkeV, chi, aLT, a):
    t = TkeV * 1e3 * e_J
    n = n19 * 1e19
    gradT = -aLT * t / a  # dT/dr [J/m]
    Q = -n * gradT * chi * 1e-6  # MW/m^2

    return Q


# --------------------------------------------------------------------------------------------------------------------------------
# Collisions
# --------------------------------------------------------------------------------------------------------------------------------


def loglam(Te_keV, ne_20):
    precomputed_factor = 9.21  # torch.log( (1E20*1E-6)**0.5/1E3 )

    l = 24.0 - torch.log(ne_20**0.5 / Te_keV) - precomputed_factor

    return l


def nue(Te_keV, ne_20):
    # From tgyro_profile_functions.f90

    precomputed_factor = 12216.80  # (4*pi*e**4 / (2*2**0.5*me_g**0.5)) / k**1.5 / (1E3)**1.5  * 1E20*1E-6

    nu = precomputed_factor * loglam(Te_keV, ne_20) * ne_20 / Te_keV**1.5

    return nu  # in 1/s


def xnue(Te_keV, ne_20, a_m, mref_u):
    # From tgyro_tglf_map.f90

    xnu = nue(Te_keV, ne_20) / (c_s(Te_keV, mref_u) / a_m)

    return xnu


def energy_exchange(Te_keV, Ti_keV, ne_20, ni_20, mi_u, Zi):
    # From tgyro_auxiliary_routines.f90:

    mime = mi_u / me_u

    c_exch_20 = 441.74  # 2.0*(4.0/3)*np.sqrt(2.0*pi)*e**4/k**1.5 *1E20  *1.5 * 1E4 / (1E3)**1.5 / (me_u * 0.5)**0.5 * (k/md**0.5, where md = 3.34358e-24)

    nu_exch = (
        ni_20[:, :, 0]
        * Zi[0] ** 2
        * mime[0] ** 0.5
        / (Ti_keV + mime[0] * Te_keV) ** 1.5
    )
    for i in range(ni_20.shape[2] - 1):
        nu_exch += (
            ni_20[:, :, i + 1]
            * Zi[i + 1] ** 2
            * mime[i + 1] ** 0.5
            / (Ti_keV + mime[i + 1] * Te_keV) ** 1.5
        )

    s_exch = c_exch_20 * nu_exch * (Te_keV - Ti_keV) * ne_20 * loglam(Te_keV, ne_20)

    return s_exch


def calculateCollisionalities(
    ne, Te, Zeff, R, q, epsilon, ne_avol, Te_avol, R0, Zeff_avol, mi_u=1.0
):  # TO FIX
    """
    Notes:
            ne in m^-3
            Te in keV
            R is the flux surface major radius
    """

    # Factor to multiply collisionalities (PRF OneNotes)
    factorIsotopeCs = mi_u**0.5

    # Ion-electron collision frequency
    wpe = calculatePlasmaFrequency(ne)  # in s^-1
    L, _ = calculateCoulombLogarithm(Te, ne)
    nu_ei = (
        Zeff
        * wpe**2
        * e_J**2
        * me_g
        * 1e-3**0.5
        * L
        / (4 * np.pi * eps0 * (2 * Te * 1e3 * e_J) ** 1.5)
    )

    # self.nu_ei = 1.0 / ( 1.09E16 * (self.Te**1.5) / ( self.nmain*1E20*self.Lambda_e ) )

    # Effective dimensionless collisionality (As in Angioni 2005 PoP)
    nu_eff = Zeff * (ne * 1e-20) * R * Te ** (-2) * factorIsotopeCs

    # Normalized collisionality (As in Conway 2006 NF)
    nu_star = 0.0118 * q * R * Zeff * (ne * 1e-20) / (Te**2 * epsilon**1.5)

    # Dimensionless collisionality (as in GACODE, and Angioni 2005 PoP)
    nu_norm = 0.89 * epsilon * nu_eff * factorIsotopeCs

    # Vol av from Angioni NF 2007
    nu_eff_Angioni = coll_Angioni07(ne_avol * 1e-19, Te_avol, R0, Zeff=Zeff_avol)

    return nu_ei, nu_eff, nu_star, nu_norm, nu_eff_Angioni


def coll_Angioni07(ne19, TekeV, Rgeo, Zeff=2.0):
    """
    As explained in Martin07, Angioni assumed Zeff=2.0 for the entire database, that's why it has a factor of 0.2
    """

    return 0.1 * Zeff * ne19 * Rgeo * TekeV ** (-2)


def predictPeaking(nu, p, Bt, Gstar_NBI, logFun=np.log):  # TO FIX Gstar
    """
    p is total pressure in MPa (e.g. p.derived['pthr_manual_vol'])
    nu is nu_effective (e.g. p.derived['nu_eff'] * 2/p.derived['Zeff_vol'])
    Bt is in T

    # Gstar_NBI as defined in Angioni NF 2007:  Gstar_NBI = R * G_NBI / (n * chi) = R * G_NBI / ()

    # Gstar_NBI = 2 * T * G_NBI/Q_TOT * | R/T * dT/dr |
    #     | R/T * dT/dr | ~ T2/<T> − 0.37
    #     G_NBI/Q_TOT at r/a=0.5

    #     chi / D ~ 1.5
    """

    press = p * 1e6 * 1e-3 * 1e-19 / e_J
    Bet = 4.02e-3 * press / Bt**2

    ne_peak_empirical_l = (
        1.347
        - 0.014
        + (0.117 - 0.005) * (-logFun(nu))
        - (4.03 + 0.81) * Bet
        + (1.331 - 0.117) * Gstar_NBI
    )
    ne_peak_empirical = 1.347 + 0.117 * (-logFun(nu)) - 4.03 * Bet + 1.331 * Gstar_NBI
    ne_peak_empirical_u = (
        1.347
        + 0.014
        + (0.117 + 0.005) * (-logFun(nu))
        - (4.03 - 0.81) * Bet
        + (1.331 + 0.117) * Gstar_NBI
    )

    return ne_peak_empirical_l, ne_peak_empirical, ne_peak_empirical_u


def calculateCoulombLogarithm(Te, ne, Z=None, ni=None, Ti=None):  # TO FIX
    """
    Notes:
            Te in keV
            ne in m^-3
    """

    # Freidberg p. 194
    L = 4.9e7 * Te**1.5 / (ne * 1e-20) ** 0.5
    logL = np.log(L)

    # Lee = 30.0 - np.log( ne**0.5 * (Te*1E-3)**-1.5 )
    # Lc = 24.0 - np.log( np.sqrt( ne/Te ) )
    # Coulomb logarithm (Fitzpatrick)
    # self.LambdaCoul = 6.6 - 0.5*np.log(self.ne)+1.5*np.log(self.Te*1E3)

    # logLe = 31.3 - np.log( ne**0.5 * (Te*1E3)**-1 )
    # if Z is not None:   logLi = 30.0 - np.log( Z**3*ni**0.5 * (Ti*1E3)**-1 )
    # else:               logLi = logLe

    # # NRL Plasma formulary
    # logLe  = 23.5 - np.log( ne**0.5 * (Te*1E3)**-(5/4) ) - ( 1E-5 + ( np.log(Te*1E3)-2 )^2 /16 )**0.5
    # logLei = 24.0 - np.log( ne**0.5 * (Te*1E3)**-(1) )
    # #logLi  = 24.0 - np.log( ne**0.5 * (Te*1E3)**-(1) )

    # P.A. Schneider thesis 2012
    Ti = Te
    logLii = 17.3 - 0.5 * np.log((ne * 1e-20) * Ti ** -(3 / 2))  # Valid for Te<20keV
    logLei = 15.2 - 0.5 * np.log((ne * 1e-20) * Te ** -(1))  # Valid for Te>10eV

    return logLei, logLii


# --------------------------------------------------------------------------------------------------------------------------------
# Radiation
# --------------------------------------------------------------------------------------------------------------------------------


def synchrotron(Te_keV, ne20, B_ref, aspect_rat, r_min, r_coeff=0.8):
    # From TGYRO

    c1 = 22.6049  # 1/( k*1E3/(me_g*c_light**2) )**0.5
    c2 = 4.1533  # c1**2.5 * 20/pi*  e**3.5*me_g*1E16/(me_g*c_light)**3 *(4*pi)**0.5
    f = c2 * ((1.0 - r_coeff) * (1 + c1 / aspect_rat / Te_keV**0.5) / r_min) ** 0.5
    qsync = f * B_ref**2.5 * Te_keV**2.5 * ne20**0.5

    return qsync * 1e-7  # from erg


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------


def chi_inc(aLTe, Qe_MWm2, Te_keV, a_m, ne_20, aLTe_base, order=2):
    Te_J = Te_keV * 1e3 * e_J
    ne = ne_20 * 1e20

    Chi_inc_norm, Chi_inc_norm_x, x_grid, y_grid, fit_params = MATHtools.deriv_fit(
        aLTe, Qe_MWm2, aLTe_base, order=order
    )

    # Chi incremental at the base location
    Chi_inc = Chi_inc_norm * (1.0e6 / (Te_J * ne / a_m))  # m^2/s
    # Chi incremental array
    Chi_inc_x = Chi_inc_norm_x * (1.0e6 / (Te_J * ne / a_m))  # m^2/s

    gradTe = -x_grid * (Te_J / a_m)
    gradTe_base = -aLTe_base * (Te_J / a_m)
    Qe_base = Qe_MWm2[np.argmin(np.abs(aLTe - aLTe_base))]
    Chi_eff = -Qe_base / gradTe_base * (1.0e6 / ne)

    # Calculate power balance chi
    Chi_pb_x = (1 / gradTe) * scipy.integrate.cumtrapz(Chi_inc_x, gradTe, initial=0)
    Chi_pb = np.interp(aLTe_base, x_grid, Chi_pb_x)

    QeCond = -ne * Chi_pb * gradTe_base  # W/m^2
    QeConv = Qe_base * 1e6 - QeCond  # W/m^2
    Vpinch = QeConv / (Te_J * ne)  # m/s

    return Chi_inc, x_grid, y_grid, Chi_pb, Vpinch, Chi_eff


def DVanalysis(aLNZ, Ge_1020m2s, a_m, ni_20):
    fact = a_m / ni_20

    _, _, x_grid, y_grid, fit_params = MATHtools.deriv_fit(
        aLNZ, Ge_1020m2s * fact, 1.0, order=1
    )

    D = fit_params[0]
    V = fit_params[1] / a_m

    return D, V, x_grid, y_grid / fact


def fitTANHPedestal(
    TtopN=1,
    Tbottom=0,
    xsym=1.0,
    w=0.1,
    xgrid=np.linspace(0, 1, 100),
    perc=[0.001, 0.001],
    debug=False,
):
    def f(x, a1=-1, a2=10, a3=-10, a4=1):
        formule = a1 * np.tanh(a2 * x + a3) + a4
        return formule

    # generate points used to plot
    xp = [0, xsym - w, xsym - w / 2, 2]
    yp = [
        (1 + perc[0]),
        1.0,
        (1.0 + Tbottom / TtopN) / 2,
        Tbottom / TtopN * (1 - perc[1]),
    ]

    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(f, xp, yp, p0=[-1, 10, -10, 1])

    y = f(xgrid, *popt) * TtopN

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------

    if debug:
        fig, ax = plt.subplots()
        ax.plot(xgrid, y, "-s", lw=2, c="b")
        ax.set_xlim([xsym - 2 * w, xsym + w])
        ax.axvline(x=1.0, ls="--", c="k")
        ax.axhline(y=TtopN, ls="--", c="g")
        ax.axhline(y=0.0, ls="--", c="k")
        ax.axhline(y=Tbottom, ls="--", c="g")
        ax.axvline(x=1.0 - w, ls="--", c="r")
        plt.show()

    return y


def fitTANH_Hughes(TtopN=1, Tbottom=0, xsym=1.0, w=0.1, xgrid=np.linspace(0, 1, 100)):
    """
    function tanh_multi, c, x, derzero=derzero

    ;adapted from Osborne
    ; tanh function with cubic or quartic inner and linear
    ; to quadratic outer extensions and

    ;INPUTS:
    ;c = vector of coefficients defined as such:
    ;      c0 = SYMMETRY POINT
    ;      c1 = FULL WIDTH
    ;      c2 = HEIGHT
    ;      c3 = OFFSET
    ;      c4 = SLOPE OR QUADRATIC (IF ZERO DER) INNER
    ;      c5 = QUADRADIC OR CUBIC (IF ZERO DER) INNER
    ;      c6 = CUBIC OR QUARTIC (IF ZERO DER) INNER
    ;      c7 = SLOPE OUTER
    ;      c8 = QUADRATIC OUTER
    ;x = x-axis

    ;KEYWORD
    ;derzero: if set, force derivative=0 at derzero

    """

    # Convert my setup to Hughes' convention:
    w = w / 2
    param = None
    c = [xsym - w / 2, w, TtopN, Tbottom * 0, 0.05, 0, 0, 0, 0]
    x = xgrid

    z = 2.0 * (c[0] - x) / c[1]

    if param is None:
        pz1 = 1.0 + c[4] * z + c[5] * z * z + c[6] * z * z * z

    else:
        z0 = 2.0 * (c[0] - param) / c[1]
        cder = -1.0 * (
            2.0 * c[4] * z0 + 3.0 * c[5] * z0 * z0 + 4.0 * c[6] * z0 * z0 * z0
        )
        pz1 = 1.0 + cder * z + c[4] * z * z + c[5] * z * z * z + c[6] * z * z * z * z

    pz2 = 1 + c[7] * z + c[8] * z * z

    y = 0.5 * (c[2] - c[3]) * (pz1 * np.exp(z) - pz2 * np.exp(-z)) / (
        np.exp(z) + np.exp(-z)
    ) + 0.5 * (c[2] + c[3])

    return y


def getOmegaBC():
    omegaEdge = 1.0e4  # rad/s?
    whereOmega = 1.0

    return omegaEdge, whereOmega


def implementProfileBoost(x, y, y_new_ix, ix=90):
    """
    Assume initial y(x) profile, with a pedestal at index position ix.
    Now, new pedestal y_new_ix, adjust core profile
    """

    # ~~~~ Simple scale up/down the entire core profile according to how much the pedestal has changed
    # This has a big caveat, if the width is very different, this scales up/down the core accounting for a
    # position that was not the pedestal position before.

    y_new = y_new_ix / y[ix] * y

    return y_new


def findPedestalDensity(dictParams, mitimNML):
    """
    The value "nPRF" in the namelist is the Greenwald fraction for neped density (which is different from ne_top)
    """

    if mitimNML.PedestalType in ["lut", "vals"]:
        embed()
        dictParams["neped"] = 0.0

    elif mitimNML.PedestalType in ["nn", "lmode"]:
        Rmajor = dictParams["rmajor"]
        Ip = dictParams["Ip"]
        epsilon = dictParams["epsilon"]

        nPRF = mitimNML.nPRF

        dictParams["neped"] = (
            Ip / (np.pi * (Rmajor * epsilon) ** 2) * nPRF * 10.0
        )  # in 1E19

        print(
            " >>>>>>>>>>> neped has been enforced from nPRF constrain, value: {0:.2f}".format(
                dictParams["neped"]
            )
        )

    else:
        raise Exception(
            "No pedestal specified but namelist indicates to replace Ip with LuT"
        )

    return dictParams


def estimateDensityProfile(mitimNML):
    x = np.linspace(0, 1.01, 2001)

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Pedestal
    # ~~~~~~~~~~~~~~~~~~~~~~~

    netop = mitimNML.MITIMparams["ne_height"]
    newidth = mitimNML.MITIMparams["ne_width"]

    ne = fitTANHPedestal(TtopN=netop, Tbottom=netop * 0.35, w=newidth, xgrid=x)

    ix = np.argmin(np.abs(x - (1 - newidth)))

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Core
    # ~~~~~~~~~~~~~~~~~~~~~~~

    ne0 = netop * estimatePeaking(mitimNML)

    ne = MATHtools.fitCoreFunction(ne, x, ne0, netop, ix, coeff=3)

    return x, ne


def estimatePeaking(mitimNML):
    ne0_neTop = 1.49

    return ne0_neTop


def BaxisBcoil(a, b, rmajor, Bt=None, Bcoil=None):
    epsilonB = (a + b) / rmajor

    if Bt is None:
        Bt = Bcoil * (1 - epsilonB)
    elif Bcoil is None:
        Bcoil = Bt / (1 - epsilonB)

    return Bt, Bcoil


def FrequencyOnAxis(Bt):
    return Bt * 10.0


def getLowZimpurity(Fmain, Zeff, Zmini, Fmini, ZimpH, FimpH=0, Zother=1, Fother=0):
    Delta_Zeff = Zeff - Fmini * Zmini**2 - FimpH * ZimpH**2 - Fother * Zother**2
    Delta_quas = 1 - Fmini * Zmini - FimpH * ZimpH - Fother * Zother

    ZimpL = (Delta_Zeff - Fmain) / (Delta_quas - Fmain)
    ZimpL_mod = int(round(ZimpL))

    FimpL = (Delta_quas - Fmain) / ZimpL_mod
    Rimp = FimpH / FimpL

    print(
        f"\t- Low-Z impurity must have Z_low = {ZimpL:.2f}~{ZimpL_mod}, with f_low = {FimpL:.2e} --> f_high/f_low = {Rimp:.2e}"
    )

    return ZimpL_mod, Rimp


def getICHefficiency(dictParams):
    eta_ICH = 0.6

    return eta_ICH


def findResonance(Fich_MHz, Bt, Rmaj, qm):
    B_D = (Fich_MHz * 1e6 * 2 * np.pi) / qm
    R = []
    for it in range(len(Fich_MHz)):
        # Resonant field within the plasma
        if (B_D[it] > Bt[it, -1]) and (B_D[it] < Bt[it, 0]):
            R.append(Rmaj[it, np.argmin(np.abs(Bt[it] - B_D[it]))])
        # Resonant field outside the plasma
        else:
            R.append(np.nan)

    return np.array(R)


def evaluate_qstar(
    varInput,
    Rmajor,
    kappa,
    Bt,
    epsilon,
    delta,
    isInputIp=False,
    ITERcorrection=False,
    includeShaping=True,
):
    """
    Notes:
            Ip in MA
    """

    Constant = 5.0  # 2*np.pi/(4*np.pi*1E-7) * 1E-6
    mainformulation = (Rmajor * epsilon) ** 2 * Bt / Rmajor
    shapingFactor = 1.0

    if includeShaping:
        shapingFactor = (
            shapingFactor
            * (1 + kappa**2 * (1 + 2 * delta**2 - 1.2 * delta**3))
            / 2.0
        )  # Uckan 1990 (used by MARS)

    if ITERcorrection:
        shapingFactor = (
            shapingFactor * (1.17 - 0.65 * epsilon) / (1 - epsilon**2) ** 2
        )

    qstar = Constant * mainformulation * shapingFactor / varInput
    Ip = Constant * mainformulation * shapingFactor / varInput

    if isInputIp:
        return qstar
    else:
        return Ip


def tau98y2(Ip, Rmajor, kappa, ne, epsilon, Bt, mbg, ptot, tauE=None):
    """
    As specified in the 1999 ITER confinement physics basis:

            Ip:         total plasma current in MA
            Rmajor:     geometric major radius in m
            kappa:      aerial elongation derived as SurfXS / pi*a^2
            epsilon:    standard inverse aspect ratio
            mbg:        isotope mass (AMU)
            ptot:       total power (MW), with radiation NOT substracted
            ne:         line averaged density (20)

    Notes:
            - ITER and SPARC 0D empirical predictions do substract radiation...
            - This give the THERMAL energy confinement time, so fast ions should be excluded
    """

    tau_scaling = (
        0.0562
        * Ip ** (0.93)
        * Rmajor ** (1.97)
        * kappa ** (0.78)
        * (ne * 10) ** (0.41)
        * epsilon ** (0.58)
        * Bt ** (0.15)
        * mbg ** (0.19)
        * ptot ** (-0.69)
    )

    return tau_scaling, tauE / tau_scaling if tauE is not None else None


def tau89p(Ip, Rmajor, kappa, ne, epsilon, Bt, mbg, ptot, tauE=None):
    """
    As specified in Yushmanov NF 1990:

            Ip:         total plasma current in MA
            Rmajor:     geometric major radius in m
            kappa:      ????
            epsilon:    standard inverse aspect ratio
            mbg:        isotope mass (AMU)
            ptot:       total power (MW), with radiation NOT substracted
            ne:         line averaged density in 1E20   (I think Rice 2011 has a wrong -0.1 in density?)
    """

    tau_scaling = (
        0.048
        * Ip ** (0.85)
        * Rmajor ** (1.50)
        * kappa ** (0.50)
        * ne ** (0.10)
        * epsilon ** (0.30)
        * Bt ** (0.20)
        * mbg ** (0.50)
        * ptot ** (-0.50)
    )

    return tau_scaling, tauE / tau_scaling if tauE is not None else None


def tau97L(Ip, Rmajor, kappa, ne, epsilon, Bt, mbg, ptot, tauE=None):
    """
    As specified in Kaye NF 1997:

            Ip:         total plasma current in MA
            Rmajor:     geometric major radius in m
            kappa:      ????
            epsilon:    standard inverse aspect ratio
            mbg:        isotope mass (AMU)
            ptot:       total power (MW), with radiation NOT substracted
            ne:         line averaged density in 1E20
    """

    tau_scaling = (
        0.023
        * Ip ** (0.96)
        * Rmajor ** (1.83)
        * kappa ** (0.64)
        * (ne * 10) ** (0.40)
        * epsilon ** (0.06)
        * Bt ** (0.03)
        * mbg ** (0.20)
        * ptot ** (-0.73)
    )

    return tau_scaling, tauE / tau_scaling if tauE is not None else None


def tauNA(Rmajor, kappa, ne_l, a, mbg, Bt, Ip, delta=0.0, tauE=None):
    """
    Notes:
            - As specified in Rice 2011 and IPB 1999 Chapter 2: ne_l in 1E20m^-3
            - mbg correction, as in IPB99
    """

    # Evaluated with cylindrical definition
    q = evaluate_qstar(Ip, Rmajor, kappa, Bt, a / Rmajor, delta, isInputIp=True)

    # Small machine???
    try:
        smallMachine = (Rmajor < 1.0).any()
    except:
        smallMachine = Rmajor < 1.0

    if smallMachine:
        alpha_mbg = 0.5
    else:
        alpha_mbg = 0.2

    tau_scaling = (
        0.07 * ne_l * q * np.sqrt(kappa) * a * Rmajor**2.0 * mbg**alpha_mbg
    )

    return tau_scaling, tauE / tau_scaling if tauE is not None else None


def ncrit_LOCSOC(Rmajor, Bt, mbg, q):
    """
    Notes:
            -As specified in in IPB 1999 Chapter 2 and Rice 2012 PoP:   density in 1E20m^-3
    """

    return 0.65 * mbg**0.5 * Bt / (q * Rmajor)


def calculateDebyeLength(Te, ne):
    """
    Notes:
            ne in m^-3
            Te in keV
    """

    lD = 2.35e-5 * np.sqrt(
        Te / (ne * 1e-20)
    )  # np.sqrt(self.Eps0*self.Te_J/(self.e_J**2*self.ne*1E20))

    return lD


def calculatePlasmaFrequency(ne):
    """
    Notes:
            ne in m^-3
    """

    wpe = (ne * e_J**2) / (me_g * 1e-3 * eps0) ** 0.5

    return wpe  # rad/s


def calculateKappaLimit(epsilon, delta, inductance, betap, feedback=2.25, wallrad=0.1):
    k0, k1 = 0, 0

    if delta < 0.25:
        k0 = 1.0 + 0.54 * (inductance ** (-0.68)) * (feedback) ** 0.62 * (
            1 + wallrad
        ) ** (-3.52)
        k1 = (
            0.04
            * (inductance) ** (-6.98)
            * (betap) ** (-2.67)
            * (feedback) ** (-1.47)
            * (1 + wallrad) ** (1.84)
        )

    if delta >= 0.25 and delta < 0.415:
        k0 = 1.0 + 0.54 * (inductance ** (-0.47)) * (feedback) ** 0.71 * (
            1 + wallrad
        ) ** (-4.0)
        k1 = (
            0.35
            * (inductance) ** (-1.42)
            * (betap) ** (-0.04)
            * (feedback) ** (-0.27)
            * (1 + wallrad) ** (0.42)
        )

    if delta >= 0.415 and delta < 0.6:
        k0 = 1.0 + 0.55 * (inductance ** (-0.08)) * (feedback) ** 0.82 * (
            1 + wallrad
        ) ** (-4.74)
        k1 = (
            0.41
            * (inductance) ** (-1.21)
            * (betap) ** (-0.06)
            * (feedback) ** (-0.18)
            * (1 + wallrad) ** (0.68)
        )

    if delta >= 0.6:
        k0 = 1.0 + 0.63 * (inductance ** (1.2)) * (feedback) ** 1.14 * (
            1 + wallrad
        ) ** (-6.67)
        k1 = (
            0.52
            * (inductance) ** (-2.00)
            * (betap) ** (0.17)
            * (feedback) ** (-0.5)
            * (1 + wallrad) ** (2.32)
        )

    kmax = k0 + k1 * ((2 * epsilon) / (1 + epsilon**2)) ** 2

    return kmax


def calculateHeatFluxWidth_Brunner(press_atm):
    return 0.91 * (press_atm) ** (-0.48)  # mm


def calculateHeatFluxWidth_Eich(Bp_OMP, Psol, R0, epsilon):
    Lambda_q_Eich14 = 0.63 * Bp_OMP ** (-1.19)
    Lambda_q_Eich15 = (
        1.35
        * (Psol * 1e6) ** (-0.02)
        * Bp_OMP ** (-0.92)
        * R0**0.04
        * epsilon**0.42
    )

    return Lambda_q_Eich14, Lambda_q_Eich15


def calculateUpstreamTemperature(Lambda_q, q95, ne, P_LCFS, R, Bp, Bt):
    """
    Notes:
            mm, [], E20 (vol_av), MW, m (OMP), T (OMP), T (OMP)
    """

    # ---------------------------------------------------------
    # Surface area for the parallel heat flux (evaluated with outer midplane)
    # ---------------------------------------------------------

    Aqpar = 4 * np.pi * R * Lambda_q * 1e-3 * Bp / Bt

    # ---------------------------------------------------------
    # Connection length
    # ---------------------------------------------------------

    L = np.pi * R * q95  # check R0 or OMP

    # ---------------------------------------------------------
    # Iteration loop
    # ---------------------------------------------------------

    max_evals, thr, x, T_u, cont = 100, 5.0, 10.0, 100.0, 0
    while np.abs(x - T_u) > thr and cont < max_evals:
        cont += 1
        x = T_u
        T_u = evaluateTE_u(ne, x, Aqpar, L, P_LCFS)

    Te_u = T_u * 1e-3
    Ti_u = 2 * T_u * 1e-3  # typical approximation

    return Te_u, Ti_u  # in keV


def evaluateTE_u(ne, Te, Aqpar, L, P_LCFS, PowerFractionToElectrons=0.5):
    """
    Notes:
            E20 (vol_av), eV, m (OMP), T (OMP), T (OMP)
    """

    ne_u = ne * 0.6
    Psol = P_LCFS * 1e6 * PowerFractionToElectrons

    # ---------------------------------------------------------
    # Sptizer-Harm conductivity (from NRL formulary)
    # ---------------------------------------------------------

    lnC = 24 - np.log((ne_u * 1e20 * 1e-6) ** 0.5 / Te)  # low temperature approximation
    k0e = (3.2 * 3.44e5 * e_J**2) / (me_g * 1e-3 * lnC) * 1e6

    # ---------------------------------------------------------
    # Upstream temperature (LCFS value)
    # ---------------------------------------------------------

    Te_u = (7.0 / 2.0 * (Psol / Aqpar * L) / k0e) ** (2.0 / 7.0)

    return Te_u


def evaluateLCFS_Lmode(
    n20, pressure_atm=5.0, Psol_MW=10.0, R=1.85, Bp=1.0, Bt=12.2, q95=3.2
):
    """
    Assume that n20 is the volume average in E20
    """

    # ~~~~~~~~ Evaluation of the temperature

    Lambda_q = calculateHeatFluxWidth_Brunner(pressure_atm)
    Te, _ = calculateUpstreamTemperature(Lambda_q, q95, n20, Psol_MW, R, Bp, Bt)
    Te_LCFS = Te * 1e3

    # ~~~~~~~~ Evaluation of the density

    ne_LCFS = n20 * 1e20 * 0.6

    return ne_LCFS, Te_LCFS, Lambda_q


def calculatePowerDrawn(t, I, M):
    """
    t = time
    I (coil,time) coil terminal currents in A
    M (coil,coil) inductance matrix in H (V*s/A)

    Output:
            dIdt  (coil,time)
            V     (coil,time) in V
            Power (coil,time) in V*A   (W)
    """

    V = np.zeros((I.shape[0], I.shape[1]))
    Power = np.zeros((I.shape[0], I.shape[1]))
    dIdt = np.zeros((I.shape[0], I.shape[1]))

    for i in range(M.shape[0]):
        dIdt[i, :] = MATHtools.deriv(t, I[i, :])

        # ------------------------------------------------------------
        # Voltage calculation:  [V] = [M] * [dI/dt]     in V
        # ------------------------------------------------------------
        for j in range(M.shape[1]):
            V[i, :] += M[i, j] * MATHtools.deriv(t, I[j, :])

        # ------------------------------------------------------------
        # Power calculation:    [P] = [I] * [V]         in V*A
        # ------------------------------------------------------------
        Power[i, :] = I[i] * V[i, :]

    return V, Power, dIdt


def calculateFirstStrikePoint(
    divR,
    divZ,
    psiRZfunction,
    psi_separatrix,
    psiMargin=1e-5,
    resol_div=1e-5,
    debug=False,
    minZ=-np.inf,
    maxZ=np.inf,
):
    """
    Function inspired from FreeGS wrapper by A.J. Creely
    """

    # --------------------------------------------------
    # Increase resolution, to avoid having regions with very few points where intersection is wrong
    # --------------------------------------------------

    rd, zd = np.diff(divR), np.diff(divZ)
    dist = np.sqrt(rd**2 + zd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0], u])

    t = np.arange(0, u.max(), resol_div)
    divRinterp, divZinterp = np.interp(t, u, divR), np.interp(t, u, divZ)

    # --------------------------------------------------
    # Strike point calculation. Grabs the first that coincides
    # --------------------------------------------------

    psi_div_points = psiRZfunction(divRinterp, divZinterp)

    divIndexS = np.where(
        [
            i and j and k
            for i, j, k in zip(
                (divZinterp > minZ),
                (divZinterp < maxZ),
                (np.abs(psi_div_points - psi_separatrix) < psiMargin),
            )
        ]
    )[0]

    # of those that interect, get the one with the highest Z
    divIndex = divIndexS[divZinterp[divIndexS].argmax()]
    # ----

    RstrikeInner, ZstrikeInner = divRinterp[divIndex], divZinterp[divIndex]

    divSpec = (divRinterp, divZinterp, divIndex)

    if debug:
        fig, axs = plt.subplots(nrows=2)
        ax = axs[0]
        ax.plot(divR, divZ, "-*", c="r")
        ax.plot(divR[0], divZ[0], "-*", c="b")
        ax.plot(divRinterp, divZinterp, "-o", c="g", markersize=3)
        ax.plot(RstrikeInner, ZstrikeInner, "-o", c="c", markersize=4)

        ax = axs[1]
        ax.plot(
            np.arange(len(psi_div_points)), psi_div_points, "-o", c="k", markersize=3
        )
        ax.axhline(y=psi_separatrix)
        ax.axhline(y=psi_separatrix + psiMargin)
        ax.axhline(y=psi_separatrix - psiMargin)

        ax.plot(
            np.arange(len(psi_div_points))[divIndex],
            psi_div_points[divIndex],
            "-o",
            c="c",
            markersize=4,
        )

        plt.show()
        embed()

    return RstrikeInner, ZstrikeInner, divSpec


def calculateStrikeIncidenceAngle(Rstrike, Zstrike, BR, Br_fun, Bz_fun, divSpec):
    """
    Function inspired from FreeGS wrapper by A.J. Creely
    """

    (divertor_Rarray, divertor_Zarray, divertor_StrikeIndex) = divSpec

    if Rstrike is None:
        Astrike = None

    BtStrike = BR / Rstrike
    BrStrike = Br_fun(R=Rstrike, Z=Zstrike)
    BzStrike = Bz_fun(R=Rstrike, Z=Zstrike)

    if (
        divertor_Rarray[divertor_StrikeIndex + 1]
        - divertor_Rarray[divertor_StrikeIndex]
    ) == 0:
        divNormR, divNormZ = 1.0, 0.0
    else:
        dZ = (
            divertor_Zarray[divertor_StrikeIndex + 1]
            - divertor_Zarray[divertor_StrikeIndex]
        )
        dR = (
            divertor_Rarray[divertor_StrikeIndex + 1]
            - divertor_Rarray[divertor_StrikeIndex]
        )

        divSlope = dZ / dR

        divNormR, divNormZ = -divSlope, 1.0

    divNormTor = 0.0
    divNormMag = np.sqrt(divNormR**2 + divNormZ**2 + divNormTor**2)
    Bmag = np.sqrt(BtStrike**2 + BrStrike**2 + BzStrike**2)

    Astrike = np.abs(
        90.0
        - (180 / np.pi)
        * np.arccos(
            (divNormR * BrStrike + divNormZ * BzStrike + divNormTor * BtStrike)
            / (divNormMag * Bmag)
        )
    )

    return Astrike
