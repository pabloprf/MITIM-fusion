import copy
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from mitim_tools.misc_tools import GRAPHICStools, PLASMAtools, LOGtools, IOtools
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.utils import TRANSFORMtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_modules.maestro.utils.EPEDbeat import eped_postprocessing,eped_profiler
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

'''
    RAPIDS (Rapid Assessment of Pedestal Integrity for Device Scenarios)
'''

def prepare_profiles(
    p_base,
    core,
    R=None, a=None, Bt=None, Ip=None,
    kappa_sep=None, delta_sep=None, kappa995=None, delta995=None,
    Zeff=None,
    tesep_eV=75, nesep19=1.0,
    Paux = 0.0,
    scale_zeta=False,   # Trick for now to fix negative jacobians when moving triangularity too much
    fDT=0.85,           # If not None: If Zeff is not None: fDT to mtaintain 
    ion_position=3,     # If Zeff is not None: if (T,D,Z,...), change Z to match Zeff choice
    roatop = 0.9,
    Ttop_keV = 4.0,
    ntop_20 = 1.0,
    force_fixed_geometry=False, # If True, do not change geometry (R,a,...) and only change profiles; this is useful to analyze the effect of the profiles alone without changing the geometry,
    **kwargs_rederive_geometry
    ):
    
    p = copy.deepcopy(p_base)

    if not force_fixed_geometry:

        # -------------------------------------------------------
        # Main quantities
        # -------------------------------------------------------

        # Change major radius
        p.profiles['rcentr(m)'][0] = R
        p.profiles['rmaj(m)'] *= R / p_base.profiles['rmaj(m)'][-1]

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
        
        # Squareness: for now reduce its magnitude proportionally to triangularity change
        if scale_zeta and mutilier_delta > 1.0:
            if np.sign(p.profiles['zeta(-)'][-1]) < 0:
                p.profiles['zeta(-)'] /= mutilier_delta
            else:
                p.profiles['zeta(-)'] *= mutilier_delta
        
        # Change magnetic field
        p.profiles['bcentr(T)'][0] = Bt
        
        # Change plasma current
        p.profiles['current(MA)'][0] = Ip

        # ---------------------------------------------------
        # Derived quantities
        # ---------------------------------------------------

        kappa_sep = p.profiles['kappa(-)'][-1]
        delta_sep = p.profiles['delta(-)'][-1]

        # Approximate XS area
        area_new = np.pi * a**2 * kappa_sep * (1-delta_sep**2/2)
        area_old = np.pi * p_base.profiles['rmin(m)'][-1]**2 * p_base.profiles['kappa(-)'][-1] * (1-p_base.profiles['delta(-)'][-1]**2/2)

        # Make sure that q95 is roughly consistent, scale based on the same as qstar_ITER
        if kappa995 is None:
            factor_sep_to_95_kappa = p_base.derived['kappa95']/p_base.profiles['kappa(-)'][-1]
            kappa95 = kappa_sep * factor_sep_to_95_kappa
        else:
            factor_995_to_95_kappa = p_base.derived['kappa95']/p_base.derived['kappa995']
            kappa95 = kappa995 * factor_995_to_95_kappa
        
        if delta995 is None:
            factor_sep_to_95_delta = p_base.derived['delta95']/p_base.profiles['delta(-)'][-1]
            delta95 = delta_sep * factor_sep_to_95_delta
        else:
            factor_995_to_95_delta = p_base.derived['delta95']/p_base.derived['delta995']
            delta95 = delta995 * factor_995_to_95_delta
        
        qstar = PLASMAtools.evaluate_qstar(
            Ip,
            R,
            kappa95,
            Bt,
            a/R,
            delta95,
            isInputIp=True,
            ITERcorrection=True,
            includeShaping=True,
        )
        
        p.profiles['q(-)'] = PLASMAtools.q_profile_scale(p.derived['psi_pol_n'], p.profiles['q(-)'], qstar / p_base.derived['qstar_ITER'])

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
    
    # Option for core specification: aLT, aLn, TiTe
    if 'TiTe' in core:
    
        # Te profile based on aLT
        roa, Te = FunctionalForms.MITIMfunctional_aLyTanh(roatop, Ttop_keV, tesep_eV*1E-3, core['aLT'])
        p.profiles['te(keV)'] = np.interp(p.derived['roa'], roa, Te)
        
        # Ti profile based on TiTe ratio
        p.profiles['ti(keV)'] = np.repeat(np.transpose(np.atleast_2d(p.profiles['te(keV)']*core['TiTe'])), p.profiles['ti(keV)'].shape[-1],axis=-1)

    # Option for core specification: aLTe, aLTi, aLn
    elif 'aLTe' in core:
        
        # Te profile based on aLTe
        roa, Te = FunctionalForms.MITIMfunctional_aLyTanh(roatop, Ttop_keV, tesep_eV*1E-3, core['aLTe'])
        p.profiles['te(keV)'] = np.interp(p.derived['roa'], roa, Te)
        
        # Ti profile based on aLTi (thermal ones)
        roa, Ti = FunctionalForms.MITIMfunctional_aLyTanh(roatop, Ttop_keV, tesep_eV*1E-3, core['aLTi'])

        for i in range(len(p.Species)):
            if p.Species[i]['S'] == 'therm':
                p.profiles['ti(keV)'][:,i] = np.interp(p.derived['roa'], roa, Ti)

    else:
        raise Exception('Core specification not recognized, provide either TiTe or aLTe, aLTi')

    # ne profile based on aLn
    roa, ne = FunctionalForms.MITIMfunctional_aLyTanh(roatop, ntop_20*10, nesep19, core['aLn'])
    p.profiles['ne(10^19/m^3)'] = np.interp(p.derived['roa'], roa, ne)
    p.profiles['ni(10^19/m^3)'] = p_base.profiles['ni(10^19/m^3)'] * np.transpose(np.atleast_2d((p.profiles['ne(10^19/m^3)']/p_base.profiles['ne(10^19/m^3)'])))
    
    p.derive_quantities(**kwargs_rederive_geometry)

    # Change Zeff
    if Zeff is not None:
        p.changeZeff(Zeff, ion_pos=ion_position, keep_fmain=fDT is not None, fmain_force=fDT)
    
    return p

def rapids_evaluator(nn, core, p_base_orig,
                     R=None, a=None, Bt=None, Ip=None, kappa_sep=None, delta_sep=None, kappa995=None, delta995=None,neped=None, Zeff=None, tesep_eV=75, nesep_ratio=0.3,
                     Paux = 0.0,
                     fDT=0.85,
                     thr_beta=0.025,
                     ion_position=3, # if (T,D,Z,...), change Z to match Zeff choice
                     hide_prints=True,  # -> If True, only print warnings and the case flag
                     optional_flag="RAPIDS case ",
                     analyze_distance_to_pb = False,
                     scale_zeta=False, # Trick for now to fix negative jacobians when moving triangularity too much
                     state_resol=None, # If not None, change resolution of the profiles for the state calculation
                     initial_betan=1.0, # Starting guess for the BetaN loop; warm-starting from a nearby case reduces iterations
                     **kwargs_rederive_geometry):
    '''
    neped in this evaluator is in 1E20 m^-3
    '''
    
    p_base = copy.deepcopy(p_base_orig)
    if state_resol is not None:
        p_base.changeResolution(n=state_resol)

    rhotop_start = 0.9

    #with IOtools.nullcontext(): # To allow debugging and printing
    with LOGtools.HiddenPrints(show_if_contains=["[*WARNING*]", f"Evaluating {optional_flag}"] if hide_prints else ""):
        
        print(f'\t\t Evaluating {optional_flag}')
        
        '''
        ---------------------------------------------------------------------------------------------------------------------
        Prepare profiles
        ---------------------------------------------------------------------------------------------------------------------
        '''
        
        Ttop_start_keV = np.max([4.0, (tesep_eV*1E-3) * 1.5])              # To avoid too low Ttop that creates hollowing later (but not too high to break the betan loop)
        ntop_start_20 = np.max([1.0, (nesep_ratio*neped) * 1.5])       # To avoid too low ntop that creates hollowing later (but not too high to break the betan loop)
                
        roatop = np.interp(rhotop_start, p_base.profiles['rho(-)'], p_base.derived['roa'])
        
        p = prepare_profiles(
            p_base,
            core,
            R=R, a=a, Bt=Bt, Ip=Ip, kappa_sep=kappa_sep, delta_sep=delta_sep, kappa995=kappa995, delta995=delta995,
            Zeff=Zeff,
            tesep_eV=tesep_eV,
            nesep19 = nesep_ratio*neped*10,
            Paux = Paux,
            fDT=fDT,
            ion_position=ion_position,
            scale_zeta=scale_zeta,
            roatop = roatop,
            Ttop_keV = Ttop_start_keV,
            ntop_20 = ntop_start_20,
            **kwargs_rederive_geometry
        )

        # Option for BetaN: provide multiplier
        if 'BetaN_multiplier' in core:
            BetaN_multiplier = core['BetaN_multiplier']
        # Option for BetaN: use same fraction as original
        else:
            BetaN_multiplier = 1 + p_base.derived['pfast_fraction']

        # For EPED runs, scale 
        TiTe_ped = core['TiTe'] if 'TiTe' in core else 1.0

        '''
        ---------------------------------------------------------------------------------------------------------------------
        Function to add a pedestal to the profiles object based on the EPED-NN evaluation and the current BetaN
        ---------------------------------------------------------------------------------------------------------------------
        '''
        def pedestal(p, force_within_range=None, force_betan=None):
            
            # Calculate new pedestal
            eped_evaluation = p.to_eped(beta_pass = "BetaNthr_engineering")

            if force_betan is not None:
                eped_evaluation["betan"] = force_betan
            else:
                eped_evaluation["betan"] *= BetaN_multiplier
            eped_evaluation["neped"] = neped*10             # the EPED-NN expects in 10e19m^-3
            eped_evaluation["nesep_ratio"] = nesep_ratio
            eped_evaluation["tesep"] = tesep_eV

            nn.force_within_range = force_within_range
            ptop_kPa, wtop_psipol = nn(**eped_evaluation)

            rhotop, netop_20, Tetop_keV, Titop_keV, _ = eped_postprocessing(
                eped_evaluation["neped"]*0.1,
                eped_evaluation["nesep_ratio"]*eped_evaluation["neped"]*0.1,
                ptop_kPa, TiTe_ped, wtop_psipol, p)

            # Unphysical values check
            if (rhotop<0 or rhotop>1.0) or (Tetop_keV<0.0 or netop_20<0.0):
                print(f'Pedestal evaluation returned unphysical values, assume pedestal does not exist', typeMsg='w')
                failed_case = True
            # Unrealistic pedestal (SEP>PED)
            elif (Tetop_keV<tesep_eV*1E-3 or netop_20<neped*nesep_ratio):
                print(f'Pedestal evaluation returned unrealistic values (SEP>PED), assume pedestal does not exist', typeMsg='w')
                failed_case = True
            else:
                failed_case = False
            
            # Note that I cannot simply make them equal to zero because the profiler will give weird resultrs
            if failed_case:
                rhotop = 0.9
                Tetop_keV = Titop_keV = tesep_eV*1E-3
                netop_20 = neped*nesep_ratio
            
            p = eped_profiler(p, rhotop_start, rhotop, Tetop_keV, Titop_keV, netop_20, print_msgs=False)
            
            # Derive quantities, but not the geometry again because this is only changing the profiles
            p.derive_quantities(rederiveGeometry=False)

            BetaN_used = p.derived["BetaNthr_engineering"] * BetaN_multiplier

            return p, ptop_kPa, wtop_psipol, eped_evaluation, BetaN_used, eped_evaluation["betan"], failed_case
        
        '''
        ---------------------------------------------------------------------------------------------------------------------
        Loop to adjust the pedestal to be consistent with the BetaN, if needed
        ---------------------------------------------------------------------------------------------------------------------
        '''
        
        Beta_EPED0 = initial_betan # Starting guess; warm-starting from a nearby converged case reduces BetaN loop iterations
        minimum_its = 2  # To make sure that at least one iteration of adjustment is done, even if the guessed Beta_EPED0 is close enough
        
        profs, Beta, Beta_EPED, fails = [], [], [], []
        #with IOtools.speeder('profiler.prof'):
        for i in range(100):
            
            print(f'\n- Iteration {i+1} for the BetaN loop: "previous" BetaN = {Beta_EPED0}\n', typeMsg='i')

            # Force to start with a reasonable betaN such that the effect of the initial condition is negligible
            p, ptop_kPa, wtop_psipol, eped_evaluation, Beta0, Beta_EPED0, failed_case = pedestal(p, force_betan=Beta_EPED0 if i==0 else None)

            # Store stuff for debugging
            profs.append(copy.deepcopy(p)) # Store a copy of the profiles for debugging
            Beta.append(Beta0)
            Beta_EPED.append(Beta_EPED0)
            fails.append(failed_case)
            
            # Decide if getting out of the loop
            error_betaN = np.abs(Beta0 - Beta_EPED0)/Beta0
            print(f'BetaN evaluated: {Beta_EPED0} vs new profiles betaN: {Beta0} ({error_betaN*100:.3f}%)',typeMsg = 'i')
        
            # If the error is small enough and it's not a failed case, get out of the loop
            if (error_betaN < thr_beta) and (not failed_case) and (i+1) > minimum_its:
                print(f'BetaN within {thr_beta*100:.2f}% after {i+1} iterations, get out of the loop', typeMsg='i')
                break
            
            # If many failed cases, assumed it is in a loop of fail-nofail and get out
            if np.sum(fails)>3:
                print(f'Many failed cases in a row, assume it is in a loop of fail-nofail and get out of the loop', typeMsg='w')
                break
        
        # # TO HELP DEBUGGING
        # fig, ax = plt.subplots()
        # ax.plot(Beta, '-o', label='From profiles')
        # ax.plot(Beta_EPED, '-o', label='From EPED evaluation')
        # for i in range(len(fails)):
        #     if fails[i]:
        #         ax.axvline(x=i, color='r', ls='--', lw=5.0)
        # ax.set_xlabel('Iteration')
        # ax.set_ylabel('$\\beta_N$')
        # ax.legend()
        # plt.show()
        # from mitim_tools.plasmastate_tools.utils import state_plotting
        # fn = state_plotting.plotAll(profs)
        # fn.show()
        # embed()
        
        # Run again the last point but with warning prints
        p_to_run = profs[-2] # The last one is the one that broke the loop, so take the previous one that is consistent with the BetaN
        p, ptop_kPa, wtop_psipol, eped_evaluation, Beta0, Beta_EPED0, failed_case = pedestal(p_to_run, force_within_range=False)

        error_betaN = np.abs(Beta0 - Beta_EPED0)/Beta0

        if error_betaN > thr_beta or failed_case:
            # # TO HELP DEBUGGING
            # plt.ioff()
            # fig, ax = plt.subplots()
            # ax.plot(Beta, '-o', label='From profiles')
            # ax.plot(Beta_EPED, '-o', label='From EPED evaluation')
            # for i in range(len(fails)):
            #     if fails[i]:
            #         ax.axvline(x=i, color='r', ls='--', lw=5.0)
            # ax.set_xlabel('Iteration')
            # ax.set_ylabel('$\\beta_N$')
            # ax.legend()
            # plt.show()
            raise Exception(f'Failed case or BetaN relative error too high ({error_betaN} vs {thr_beta}), for parameters: {eped_evaluation}')
        else:
            print(f"\t\t- Evaluating {optional_flag} required {i+1} iterations for parameters: {eped_evaluation}")

        # Calculate targets
        power = STATEtools.powerstate(p,evolution_options={"rhoPredicted": np.linspace(0.0, 0.9, 20)[1:]}, increase_profile_resol=False)
        power.calculateProfileFunctions()
        power.calculateTargets()
        profiles_new = copy.deepcopy(p)
        TRANSFORMtools.powerstate_to_gacode_powers(power, profiles_new, rederive_at_high_res=False)
        
        profiles_new.derive_quantities(rederiveGeometry=False)
        profiles_new.selfconsistentPTOT()

        neped_transition_estimate = None
        if analyze_distance_to_pb:
            neped_transition_estimate_abs = estimate_neped_transition(nn, eped_evaluation)
            
            neped_transition_estimate = neped_transition_estimate_abs / eped_evaluation['neped']
        
    return ptop_kPa,wtop_psipol,profiles_new,eped_evaluation, neped_transition_estimate

def estimate_neped_transition(nn, eped_evaluation, plotYN=False):
    # Analyze distance to Peeling balooning transition 
    
    neped_base = eped_evaluation['neped']
    
    min_rel_search = 0.2
    max_rel_search = 3.0
    num = 50
    
    # Scan around neped to find transition
    neped, ptop = [], []
    for factor in np.linspace(min_rel_search, max_rel_search, num):
        neped_test = neped_base * factor
        eped_evaluation_test = copy.deepcopy(eped_evaluation)
        eped_evaluation_test['neped'] = neped_test
        ptop_kPa, wtop_psipol = nn(**eped_evaluation_test)
        neped.append(neped_test)
        ptop.append(ptop_kPa)
        
    neped = np.array(neped)
    ptop = np.array(ptop)
    
    # Calculate derivative
    dptop_dneped = np.gradient(ptop, neped)
    
    # Find where three points in a row have negative derivative
    transition_index = None
    for i in range(1, len(dptop_dneped)-1):
        if dptop_dneped[i-1]<0 and dptop_dneped[i]<0 and dptop_dneped[i+1]<0:
            transition_index = i
            break
    
    ne_trans = neped[transition_index] if transition_index is not None else 0.0
        
    if plotYN:
        fig, axs = plt.subplots(nrows=2, figsize=(10,8))
        ax = axs[0]
        ax.plot(neped, ptop, '-o')
        ax.set_xlabel('$n_{e,ped}$ (10$^{19}$ m$^{-3}$)')
        ax.set_ylabel('$p_{top}$ (kPa)')
        GRAPHICStools.addDenseAxis(ax)
        
        ax.axvline(x=neped[transition_index], ls='--', lw=1.0, c='r', label='Estimated transition')
        
        ax = axs[1]
        ax.plot(neped, dptop_dneped, '-o')
        ax.set_xlabel('$n_{e,ped}$ (10$^{19}$ m$^{-3}$)')
        ax.set_ylabel('$dp_{top}/dn_{e,ped}$ (kPa/(10$^{19}$ m$^{-3}$))')
        GRAPHICStools.addDenseAxis(ax)
        
        ax.axvline(x=neped[transition_index], ls='--', lw=1.0, c='r', label='Estimated transition')
        
        plt.show()
        embed()
    
    return ne_trans

def scan_parameter(
    nn, p_base_orig, xparam, x, nominal_parameters, core,
    xparamlab='',
    relative=False,
    c='b',
    leg='',
    goal_pfusion=1_100,
    Paux = 0.0,
    vertical_at_nominal=True,
    type_plot='full',
    axs=None,
    state_resol=None,
    n_jobs=1,   # >1 uses ThreadPoolExecutor (real speedup when NN inference dominates)
    ):
    '''
    axs must be a list of 8 cases if full plot
    '''
    
    if state_resol is not None:
        p_base = copy.deepcopy(p_base_orig)
        p_base.changeResolution(n=state_resol)
    else:
        p_base = p_base_orig

    values = copy.deepcopy(nominal_parameters)

    results1 = {
        'x' : x if not relative else x*nominal_parameters[xparam],
        'profs' : [],'eped_inputs': [],'Ptop' : [],
        'fG': [],'Pfus' : [], 'vol': [], 'qstar_ITER': [], 'H98': [], 'betaN': []
        }
    
    # Option for BetaN: provide multiplier
    if 'BetaN_multiplier' in core:
        BetaN_multiplier = core['BetaN_multiplier']
    # Option for BetaN: use same fraction as original
    else:
        BetaN_multiplier = 1+p_base.derived['pfast_fraction']
    
    xs_scan = results1['x']
    n_scan   = len(xs_scan)

    def _evaluate_one(i, x_val, initial_betan):
        vals = dict(values)
        vals[xparam] = x_val
        return rapids_evaluator(
            nn, core, p_base,
            Paux=Paux,
            **vals,
            n_theta_geo=101,
            optional_flag=f'RAPIDS case {i+1}/{n_scan}: {xparam}={x_val:.3f}',
            initial_betan=initial_betan,
        )

    if n_jobs == 1:
        # Sequential: carry over converged betan as warm start for the next point
        eval_results = []
        next_betan = 1.0
        for i, x_val in enumerate(xs_scan):
            res = _evaluate_one(i, x_val, initial_betan=next_betan)
            eval_results.append(res)
            next_betan = res[3].get('betan', 1.0)  # eped_evaluation['betan'] from converged point
    else:
        # Parallel: all points run concurrently; no betan carry-over (points are independent)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_evaluate_one, i, x_val, 1.0) for i, x_val in enumerate(xs_scan)]
            eval_results = [f.result() for f in futures]

    for ptop_kPa, wtop_psipol, profiles_new, eped_evaluation, _ in eval_results:
        results1['profs'].append(profiles_new)
        results1['Ptop'].append(ptop_kPa)
        results1['wtop_psipol'] = wtop_psipol
        results1['eped_inputs'].append(eped_evaluation)

        # Specific outputs of profiles
        results1['fG'].append(profiles_new.derived["fG"])
        results1['Pfus'].append(profiles_new.derived['Pfus'])
        results1['vol'].append(profiles_new.derived['volume'])
        results1['qstar_ITER'].append(profiles_new.derived['qstar_ITER'])
        results1['H98'].append(profiles_new.derived['H98'])
        results1['betaN'].append(profiles_new.derived['BetaNthr_engineering']*BetaN_multiplier)

    if axs is None:
        plt.ion()
        fig = plt.figure(figsize=(16,9))
        
        if type_plot=='full':
            axsL = fig.subplot_mosaic(
                """
                ABFH
                CDGI
                """
            )
            axs = [axsL['A'], axsL['C'], axsL['B'], axsL['D'], axsL['F'], axsL['G'], axsL['H'], axsL['I']]
        elif type_plot=='simple':
            axsL = fig.subplot_mosaic(
                """
                1
                2
                3
                """
            )
            axs = [axsL['1'], axsL['2'], axsL['3']]


    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------

    fG_nominal = results1['fG'][np.argmin(np.abs(results1['x']- (nominal_parameters[xparam] if not relative else nominal_parameters[xparam])))]

    ax = axs[0]
    ax.plot(results1['x'], results1['Ptop'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel(xparamlab)
    ax.set_ylabel('$p_{top}$ (kPa)')
    ax.set_title('Senstivity to scan parameter')

    if vertical_at_nominal:
        axs[0].axvline(x=nominal_parameters[xparam],ls='-.',lw=1.0,c=c)

    ax = axs[1]
    ax.plot(results1['x'], results1['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)

    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel(xparamlab)
    ax.set_ylabel('$P_{fus}$ (MW)')

    axs[1].axhspan(goal_pfusion, goal_pfusion*1.5, facecolor="g", alpha=0.1, edgecolor="none")
    axs[1].set_ylim(0, goal_pfusion*1.5)

    if vertical_at_nominal:
        axs[1].axvline(x=nominal_parameters[xparam],ls='-.',lw=1.0,c=c)

    ax = axs[2]
    ax.plot(results1['fG'], results1['Ptop'], '-s', color= c, lw=1.0, markersize=5, label =leg)
    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel('$<f_G>$')
    ax.set_ylabel('$p_{top}$ (kPa)')
    ax.set_title('Senstivity to $<f_G>$')

    axs[2].axvspan(1.0, 1.5, facecolor="k", alpha=0.1, edgecolor="none")
    axs[2].set_xlim(0.5, 1.2)
    
    if vertical_at_nominal:
        axs[2].axvline(x=fG_nominal,ls='-.',lw=1.0,c=c)
    
    ax = axs[3]
    ax.plot(results1['fG'], results1['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)

    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel('$<f_G>$')
    ax.set_ylabel('$P_{fus}$ (MW)')

    axs[3].axvspan(1.0, 1.5, facecolor="k", alpha=0.1, edgecolor="none")
    axs[3].set_xlim(0.5, 1.2)

    axs[3].axhspan(goal_pfusion, goal_pfusion*1.5, facecolor="g", alpha=0.1, edgecolor="none")
    axs[3].set_ylim(0, goal_pfusion*1.5)

    if vertical_at_nominal:
        axs[3].axvline(x=fG_nominal,ls='-.',lw=1.0,c=c)
    
    if type_plot=='full':

        ax = axs[4]
        ax.plot(results1['qstar_ITER'], results1['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel('$q^*$ ITER')
        ax.set_ylabel('$P_{fus}$ (MW)')
        ax.set_xlim(2.8, 4.5)

        ax = axs[5]
        ax.plot(results1['vol'], results1['Pfus'], '-s', color= c, lw=1.0, markersize=5, label =leg)
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel('$V$ ($m^3$)')
        ax.set_ylabel('$P_{fus}$ (MW)')

        ax = axs[6]
        ax.plot(results1['x'], results1['H98'], '-s', color= c, lw=1.0, markersize=5, label =leg)
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel(xparamlab)
        ax.set_ylabel('$H_{98y2}$')
        ax.set_ylim(0.5, 1.5)
        ax.axhline(y=1.0,ls='-.',lw=1.0,c='k')

        axs[6].axhspan(0.85, 1.15, facecolor="g", alpha=0.1, edgecolor="none")

        ax = axs[7]
        ax.plot(results1['x'], results1['betaN'], '-s', color= c, lw=1.0, markersize=5, label =leg)
        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlabel(xparamlab)
        ax.set_ylabel('$\\beta_N$ (w/ $B_0$)')

    plt.tight_layout()
   
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
    axs = [axsL['A'], axsL['B'], axsL['F'], axsL['H'], axsL['C'], axsL['D'], axsL['G'], axsL['I']]

    extr = ''
    if keep_qstar:
        extr += ' (fixed $q^*$)'
    if keep_eps:
        extr += ' (fixed $\\epsilon$)'

    

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


    axs[0].legend(prop={'size': 10})

    ax = axsL['E']
    for results,c in zip(
            resultsS,
            ['r','b','g'],
            ):
        results1['profs'][0].plot_state_flux_surfaces(ax=ax, surfaces_rho=[1.0], color=c)

    GRAPHICStools.addDenseAxis(ax)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
