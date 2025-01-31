import copy
import numpy as np
from mitim_modules.powertorch import STATEtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_modules.maestro.utils.EPEDbeat import eped_postprocessing,eped_profiler
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def modify_profile(nn, aLT, aLn, TiTe, p_base, R=None, a=None, Bt=None, Ip=None, kappa_sep=None, delta_sep=None, neped=None, Zeff=None, tesep_eV=75, nesep_ratio=0.3):

    p = copy.deepcopy(p_base)

    # Change major radius
    p.profiles['rcentr(m)'][0] = R
    p.profiles['rmaj(m)'] *= R/p_base.profiles['rmaj(m)'][0]

    # Change minor radius
    p.profiles['rmin(m)'] *= a/p_base.profiles['rmin(m)'][-1]

    # Change elongation
    p.profiles['kappa(-)'] *= kappa_sep/p_base.profiles['kappa(-)'][-1]

    # Change triangularity
    p.profiles['delta(-)'] *= delta_sep/p_base.profiles['delta(-)'][-1]

    # Change magnetic field
    p.profiles['bcentr(T)'][0] = Bt
    p.profiles['torfluxa(Wb/radian)'] *= Bt/p.profiles['bcentr(T)'][0]

    # Change plasma current
    p.profiles['current(MA)'][0] = Ip
    p.profiles['polflux(Wb/radian)'] *= Ip/p.profiles['current(MA)'][0]

    # Gradient-based profiles
    rhotop_assume = 0.9
    Ttop_assume = 4.0
    ntop_assume = 1.0

    roatop = np.interp(rhotop_assume, p.profiles['rho(-)'], p.derived['roa'])
    roa, Te = FunctionalForms.MITIMfunctional_aLyTanh(roatop, Ttop_assume, tesep_eV*1E-3, aLT)
    p.profiles['te(keV)'] = np.interp(p.derived['roa'], roa, Te)
    p.profiles['ti(keV)'] = np.repeat(np.transpose(np.atleast_2d(p.profiles['te(keV)']*TiTe)), p.profiles['ti(keV)'].shape[-1],axis=-1)

    roa, ne = FunctionalForms.MITIMfunctional_aLyTanh(roatop, ntop_assume*10, nesep_ratio*neped*10, aLn)
    p.profiles['ne(10^19/m^3)'] = np.interp(p.derived['roa'], roa, ne)
    p.profiles['ni(10^19/m^3)'] = p_base.profiles['ni(10^19/m^3)'] * np.transpose(np.atleast_2d((p.profiles['ne(10^19/m^3)']/p_base.profiles['ne(10^19/m^3)'])))

    p.deriveQuantities()

    # Change Zeff
    p.changeZeff(Zeff, ion_pos=3)

    def pedestal(p):

        # Calculate new pedestal
        eped_evaluation = p.to_eped()

        eped_evaluation["neped"] = neped*10
        eped_evaluation["nesep_ratio"] = nesep_ratio
        eped_evaluation["tesep"] = tesep_eV

        ptop_kPa, wtop_psipol = nn(**eped_evaluation)

        rhotop, netop_20, Ttop_keV, rhoped = eped_postprocessing(eped_evaluation["neped"]*0.1, eped_evaluation["nesep_ratio"]*eped_evaluation["neped"]*0.1, ptop_kPa, wtop_psipol, p)

        p = eped_profiler(p, rhotop_assume, rhotop, Ttop_keV, netop_20)
        
        # Derive quantities
        p.deriveQuantities(rederiveGeometry=False)

        error_betaN = np.abs(p.derived['BetaN'] - eped_evaluation["betan"])/p.derived['BetaN']
        print(f'BetaN evaluated: {eped_evaluation["betan"]} vs new profiles betaN: {p.derived['BetaN']} ({error_betaN*100:.1f}%)',typeMsg = 'i')

        return p, ptop_kPa, error_betaN, eped_evaluation

    # Loop for better beta definition
    for i in range(5):
        p, ptop_kPa, error_betaN, eped_evaluation = pedestal(p)
        if error_betaN < 0.05:
            break

    if error_betaN > 0.05:
        raise Exception('BetaN error too high')

    # Power
    power = STATEtools.powerstate(p,EvolutionOptions={"rhoPredicted": np.linspace(0.0, 0.9, 50)})

    power.calculate(None, folder='~/scratch/power/')
    profiles_new = power.to_gacode(insert_highres_powers=True)

    return ptop_kPa,profiles_new, eped_evaluation
