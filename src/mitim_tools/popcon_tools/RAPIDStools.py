import copy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, PLASMAtools
from mitim_modules.powertorch import STATEtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_modules.maestro.utils.EPEDbeat import eped_postprocessing,eped_profiler
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed
'''
        RAPIDS (Rapid Assessment of Pedestal Integrity for Device Scenarios)
'''

def rapids_evaluator(nn, aLT, aLn, TiTe, p_base, R=None, a=None, Bt=None, Ip=None, kappa_sep=None, delta_sep=None, kappa995=None, delta995=None,neped=None, Zeff=None, tesep_eV=75, nesep_ratio=0.3, TioverTe=1.0, Paux = 0.0, BetaN_multiplier=1.0,thr_beta=0.02):

    p = copy.deepcopy(p_base)

    # -------------------------------------------------------
    # Main quantities
    # -------------------------------------------------------

    # Change major radius
    p.profiles['rcentr(m)'][0] = R
    p.profiles['rmaj(m)'] *= R/p_base.profiles['rmaj(m)'][0]

    # Change minor radius
    p.profiles['rmin(m)'] *= a/p_base.profiles['rmin(m)'][-1]

    # Change elongation
    if kappa995 is not None:
        # If 995 available, use that
        mutilier_kappa = kappa995/p_base.derived['kappa995']
    else:
        # Otherwise, use the separatrix value
        mutilier_kappa = kappa_sep/p_base.profiles['kappa(-)'][-1]
    p.profiles['kappa(-)'] *= mutilier_kappa

    # Change triangularity
    if delta995 is not None:
        # If 995 available, use that
        mutilier_delta = delta995/p_base.derived['delta995']
    else:
        # Otherwise, use the separatrix value
        mutilier_delta = delta_sep/p_base.profiles['delta(-)'][-1]
    p.profiles['delta(-)'] *= mutilier_delta

    # Change magnetic field
    p.profiles['bcentr(T)'][0] = Bt
    
    # Change plasma current
    p.profiles['current(MA)'][0] = Ip

    # -------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------

    kappa_sep = p.profiles['kappa(-)'][-1]
    delta_sep = p.profiles['delta(-)'][-1]

    # Approximate XS area
    area_new = np.pi * a**2 * kappa_sep * (1-delta_sep**2/2)
    area_old = np.pi * p_base.profiles['rmin(m)'][-1]**2 * p_base.profiles['kappa(-)'][-1] * (1-p_base.profiles['delta(-)'][-1]**2/2)

    # Make sure that q95 is roughly consistent
    p.profiles['q(-)'] *= p_base.profiles['current(MA)'][0] / Ip

    # Make sure that toroidal flux is roughly consistent
    p.profiles['torfluxa(Wb/radian)'] *= ( Bt / p_base.profiles['bcentr(T)'][0] ) * ( area_new / area_old )
    p.profiles['polflux(Wb/radian)'] *= ( Ip / p_base.profiles['current(MA)'][0] )

    # -------------------------------------------------------
    # Others
    # -------------------------------------------------------

    # Change auxiliary power
    p.changeRFpower(PrfMW=Paux)
    for i in ["qohme(MW/m^3)"]:
        p.profiles[i] *= 0.0

    # -------------------------------------------------------
    # Gradient-based profiles
    # -------------------------------------------------------

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

    p.derive_quantities()

    # Change Zeff
    p.changeZeff(Zeff, ion_pos=3)

    def pedestal(p):

        # Calculate new pedestal
        eped_evaluation = p.to_eped()

        eped_evaluation["betan"] *= BetaN_multiplier
        eped_evaluation["neped"] = neped*10
        eped_evaluation["nesep_ratio"] = nesep_ratio
        eped_evaluation["tesep"] = tesep_eV

        ptop_kPa, wtop_psipol = nn(**eped_evaluation)

        rhotop, netop_20, Tetop_keV, Titop_keV, rhoped = eped_postprocessing(eped_evaluation["neped"]*0.1, eped_evaluation["nesep_ratio"]*eped_evaluation["neped"]*0.1, ptop_kPa, TioverTe, wtop_psipol, p)

        p = eped_profiler(p, rhotop_assume, rhotop, Tetop_keV, Titop_keV, netop_20)
        
        # Derive quantities
        p.derive_quantities(rederiveGeometry=False)

        BetaN_used = p.derived['BetaN_engineering'] * BetaN_multiplier

        error_betaN = np.abs(BetaN_used - eped_evaluation["betan"])/BetaN_used
        print(f'BetaN evaluated: {eped_evaluation["betan"]} vs new profiles betaN: {BetaN_used} ({error_betaN*100:.1f}%)',typeMsg = 'i')

        return p, ptop_kPa, error_betaN, eped_evaluation

    # Loop for better beta definition
    for i in range(10):
        p, ptop_kPa, error_betaN, eped_evaluation = pedestal(p)
        if error_betaN < thr_beta:
            break

    if error_betaN > thr_beta:
        raise Exception('BetaN error too high')

    # Power
    power = STATEtools.powerstate(p,evolution_options={"rhoPredicted": np.linspace(0.0, 0.9, 50)[1:]})

    power.calculate(None, folder='~/scratch/power/')
    profiles_new = power.from_powerstate(insert_highres_powers=True)

    return ptop_kPa,profiles_new, eped_evaluation

def plot_cases(axs, results, xlabel = '$n_{e,ped}$', leg='',c='b'):


    ax = axs[0,0]
    ax.plot(results['x'], results['Ptop'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$p_{top}$ (kPa)')

    ax = axs[1,0]
    ax.plot(results['x'], results['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)

    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$P_{fus}$ (MW)')

    ax = axs[0,1]
    ax.plot(results['fG'], results['Ptop'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel('$<f_G>$')
    ax.set_ylabel('$p_{top}$ (kPa)')

    ax = axs[1,1]
    ax.plot(results['fG'], results['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)

    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel('$<f_G>$')
    ax.set_ylabel('$P_{fus}$ (MW)')

    ax = axs[0,2]
    ax.plot(results['qstar_ITER'], results['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel('$q^*$ ITER')
    ax.set_ylabel('$P_{fus}$ (MW)')
    ax.set_xlim(2.8, 4.5)

    ax = axs[1,2]
    ax.plot(results['vol'], results['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel('$V$ ($m^3$)')
    ax.set_ylabel('$P_{fus}$ (MW)')

    ax = axs[0,3]
    ax.plot(results['x'], results['H98'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$H_{98y2}$')
    ax.set_ylim(0.5, 1.5)
    ax.axhline(y=1.0,ls='-.',lw=1.0,c='k')

    ax = axs[1,3]
    ax.plot(results['x'], results['betaN'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\beta_N$ (w/ $B_0$)')

def scan_parameter(nn,p_base, xparam, x, nominal_parameters, core, xparamlab='', axs=None, relative=False,c='b', leg='', goal_pfusion=1_100, Paux = 0.0, vertical_at_nominal=True):
    
    if axs is None:
        plt.ion(); fig, axs = plt.subplots(nrows=2,ncols=4,figsize=(20,10))

    values = copy.deepcopy(nominal_parameters)

    results1 = {
        'x' : x if not relative else x*nominal_parameters[xparam],
        'profs' : [],'eped_inputs': [],'Ptop' : [],
        'fG': [],'Pfus' : [], 'vol': [], 'qstar_ITER': [], 'H98': [], 'betaN': []
        }
    for x in results1['x']:
        values[xparam] = x
        ptop_kPa,profiles_new, eped_evaluation = rapids_evaluator(nn, core['aLT'], core['aLn'], core['TiTe'], p_base, BetaN_multiplier=core['BetaN_multiplier'],Paux=Paux, **values)
        results1['profs'].append(profiles_new)
        results1['Ptop'].append(ptop_kPa)
        results1['eped_inputs'].append(eped_evaluation)

        # Specific outputs of profiles
        results1['fG'].append(profiles_new.derived["fG"])
        results1['Pfus'].append(profiles_new.derived['Pfus'])
        results1['vol'].append(profiles_new.derived['volume'])
        results1['qstar_ITER'].append(profiles_new.derived['qstar_ITER'])
        results1['H98'].append(profiles_new.derived['H98'])
        results1['betaN'].append(profiles_new.derived['BetaN_engineering']*core['BetaN_multiplier'])

    plot_cases(axs, results1, xlabel = xparamlab, leg=leg,c=c)
    if vertical_at_nominal:
        axs[0,0].axvline(x=nominal_parameters[xparam],ls='-.',lw=1.0,c=c)
        axs[1,0].axvline(x=nominal_parameters[xparam],ls='-.',lw=1.0,c=c)

    axs[0,1].axvspan(1.0, 1.5, facecolor="k", alpha=0.1, edgecolor="none")
    axs[1,1].axvspan(1.0, 1.5, facecolor="k", alpha=0.1, edgecolor="none")

    axs[0,1].set_xlim(0.5, 1.2)
    axs[1,1].set_xlim(0.5, 1.2)

    # axs[0,0].set_ylim(bottom=0)
    # axs[0,1].set_ylim(bottom=0)

    axs[1,0].axhspan(goal_pfusion, goal_pfusion*1.5, facecolor="g", alpha=0.1, edgecolor="none")
    axs[1,1].axhspan(goal_pfusion, goal_pfusion*1.5, facecolor="g", alpha=0.1, edgecolor="none")
   
    axs[1,0].set_ylim(0, goal_pfusion*1.5)
    axs[1,1].set_ylim(0, goal_pfusion*1.5)

    axs[0,3].axhspan(0.85, 1.15, facecolor="g", alpha=0.1, edgecolor="none")
   
    return results1


def scan_density_additional(nn, p_base, nominal_parameters, core, r, param, paramlabel,x0=1.0,xf=3.0,num=20,fig=None, keep_qstar=False, keep_eps=False, Paux=0.0):

    if fig is None:
        fig = plt.figure(figsize=(14,10))
    axsL = fig.subplot_mosaic(
        """
        ABFHE
        CDGIE
        """
    )

    extr = ''
    if keep_qstar:
        extr += ' (fixed $q^*$)'
    if keep_eps:
        extr += ' (fixed $\\epsilon$)'

    axs = np.array([[axsL['A'], axsL['B'], axsL['F'], axsL['H']], [axsL['C'], axsL['D'], axsL['G'], axsL['I']]])

    resultsS = []
    for varrel,c,leg in zip(
            [1.0-r,1.0,1.0+r],
            ['r','b','g'],
            [f'$-{r*100:.1f}\\%$'+extr,f"{paramlabel} = {nominal_parameters[param]:.3f}",f'$+{r*100:.1f}\\%$'+extr]
            ):
        parameters = copy.deepcopy(nominal_parameters)
        parameters[param] *= varrel

        if keep_eps:
            parameters['a'] = parameters['R'] * nominal_parameters['a']/nominal_parameters['R']
            print(f"\t-> Keeping aspect ratio constant, hence changing minor radius from {nominal_parameters['a']} to {parameters['a']}")

        if keep_qstar:
            qstar_orig = PLASMAtools.evaluate_qstar(
                nominal_parameters['Ip'],
                nominal_parameters['R'],
                nominal_parameters['kappa_sep'] * (p_base.derived['kappa95']/p_base.profiles['kappa(-)'][-1]),
                nominal_parameters['Bt'],
                nominal_parameters['a']/nominal_parameters['R'],
                nominal_parameters['delta_sep'] * (p_base.derived['delta95']/p_base.profiles['delta(-)'][-1]),
                isInputIp=False,ITERcorrection=False,includeShaping=True,)

            qstar_new = PLASMAtools.evaluate_qstar(
                parameters['Ip'],
                parameters['R'],
                parameters['kappa_sep'] * p_base.derived['kappa95']/p_base.profiles['kappa(-)'][-1],
                parameters['Bt'],
                parameters['a']/parameters['R'],
                parameters['delta_sep'] * p_base.derived['delta95']/p_base.profiles['delta(-)'][-1],
                isInputIp=False,ITERcorrection=False,includeShaping=True,)

            parameters['Ip'] *= qstar_new/qstar_orig

            print(f"\t-> Keeping qstar constant, hence changing current from {nominal_parameters['Ip']} to {parameters['Ip']}")

        results = scan_parameter(nn, p_base, 'neped',  np.linspace(x0,xf,num), parameters, core, xparamlab='$n_{e,ped}$ ($10^{20}/m^3$)', axs=axs, c=c, leg=leg, Paux=Paux)
        resultsS.append(results)


    axs[0,0].legend(prop={'size': 10})

    ax = axsL['E']
    for results,c in zip(
            resultsS,
            ['r','b','g'],
            ):
        results['profs'][0].plot_state_flux_surfaces(ax=ax, surfaces_rho=[1.0], color=c)

    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
