import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.utils import TRANSFORMtools
from scipy.optimize import minimize
from IPython import embed

def calculate_new(aLTe, aLn, aLTi, p, roatop = 0.95):

    Ttop = np.interp(roatop, p.derived['roa'], p.profiles['te(keV)'])  # keV
    ntop = np.interp(roatop, p.derived['roa'], p.profiles['ne(10^19/m^3)'])  # 10^19/m^3
    Tsep = p.profiles['te(keV)'][-1]  # keV
    nsep = p.profiles['ne(10^19/m^3)'][-1]  # 10^19/m^3

    p_mod = copy.deepcopy(p)

    roa, Te = FunctionalForms.MITIMfunctional_aLyTanh(roatop, Ttop, Tsep, aLTe)
    p_mod.profiles['te(keV)'] = np.interp(p_mod.derived['roa'], roa, Te)
    
    # Change only thermal ion temperature
    roa, Ti = FunctionalForms.MITIMfunctional_aLyTanh(roatop, Ttop, Tsep, aLTi)
    for i in range(len(p_mod.Species)):
        if p_mod.Species[i]['S'] == 'therm':
            p_mod.profiles['ti(keV)'][:,i] = np.interp(p_mod.derived['roa'], roa, Ti)

    roa, ne = FunctionalForms.MITIMfunctional_aLyTanh(roatop, ntop, nsep, aLn)
    p_mod.profiles['ne(10^19/m^3)'] = np.interp(p_mod.derived['roa'], roa, ne)
    p_mod.profiles['ni(10^19/m^3)'] = p.profiles['ni(10^19/m^3)'] * np.transpose(np.atleast_2d((p_mod.profiles['ne(10^19/m^3)']/p.profiles['ne(10^19/m^3)'])))

    resolution_targets = 10 # For Pfus

    if resolution_targets is not None:
        
        p_mod.derive_quantities(rederiveGeometry=False)

        p_mod.recompute_targets()

    p_mod.derive_quantities(rederiveGeometry=False)
    p_mod.selfconsistentPTOT()

    return p_mod

# --------------------------------------------------------
# Minimize functions
# --------------------------------------------------------

def minimize_nevol(aLn, aLTe, aLTi, ps, roatop = 0.95):
    
    p_mods = [calculate_new(aLTe, aLn, aLTi, p, roatop) for p in ps]
    
    residuals = 0.0
    for i in range(len(p_mods)):
        p_mod = p_mods[i]
        p = ps[i]
        residuals += ( (p_mod.derived['ne_vol20'] - p.derived['ne_vol20'])/p.derived['ne_vol20'])**2

    return residuals

def minimize_fusion_and_beta(varS, aLn, ps, roatop = 0.95):
    
    aLTe, aLTi = varS
    
    p_mods = [calculate_new(aLTe, aLn, aLTi, p, roatop) for p in ps]
    
    residuals = 0.0
    for i in range(len(p_mods)):
        p_mod = p_mods[i]
        p = ps[i]
        residuals += ( (p_mod.derived['Pfus'] - p.derived['Pfus'])/p.derived['Pfus'])**2
        residuals += ( (p_mod.derived['BetaN'] - p.derived['BetaN'])/p.derived['BetaN'])**2

    return residuals

def find_core_parameters_RAPIDS(profiles_list, provideBetaN_multiplier=True, roatop = 0.95, plotYN=False, search_ranges=0.5):

    # --------------------------------------------------------
    # Prepare case
    # --------------------------------------------------------
    ps = []
    for p in profiles_list:
         if isinstance(p, str):
            p = PROFILEStools.gacode_state(p)
         ps.append(p)

    p = ps[0]  # Use first profile as reference for initial guesses
    iroa = np.argmin( np.abs(p.derived['roa'] - roatop) )
    BetaN_multiplier = p.derived['ptot_manual_vol'] / p.derived['pthr_manual_vol']

    # Guesses
    aLTe = p.derived['aLTe'][:iroa].mean()
    aLn = p.derived['aLne'][:iroa].mean()
    aLTi = p.derived['aLTi'][:iroa].mean()

    lims_aLTe = [aLTe*(1-search_ranges),aLTe*(1+search_ranges)]
    lims_aLn = [aLn*(1-search_ranges),aLn*(1+search_ranges)]
    lims_aLTi = [aLTi*(1-search_ranges),aLTi*(1+search_ranges)]

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    # Find optimal density gradient to match ne
    res = minimize(minimize_nevol, [aLn],
                args=(aLTe, aLTi, ps, roatop),
                method='Nelder-Mead',
                tol=1e-5,
                bounds=[tuple(lims_aLn)]
                )
    aLn = res.x[0]

    # Find optimal temperature gradient and ratio to match Pfus and BetaN
    res = minimize(minimize_fusion_and_beta, [aLTe, aLTi],
                args=(aLn, ps, roatop),
                method='Nelder-Mead',
                tol=1e-3,
                bounds=[
                    tuple(lims_aLTe),
                    tuple(lims_aLTi)
                    ]
                )
    aLTe = res.x[0]
    aLTi = res.x[1]

    # Calculate final modified profiles
    p_mods = [calculate_new(aLTe, aLn, aLTi, p, roatop=roatop) for p in ps]

    # --------------------------------------------------------
    # Show results
    # --------------------------------------------------------

    print(res)
    print('Optimization results:')
    print(f'\t- aLTe: {aLTe:.2f}')
    print(f'\t- aLn: {aLn:.2f}')
    print(f'\t- aLTi: {aLTi:.2f}')
    print(f'\t- BetaN_multiplier: {BetaN_multiplier:.2f}')
    print('Optimization quality:')
    for i in range(len(p_mods)):
        p_mod = p_mods[i]
        p = ps[i]
        print(f'\t* Profile: {i+1} ---')
        print(f'\t\t- ne_vol20: {p_mod.derived["ne_vol20"]:.2f} (target={p.derived["ne_vol20"]:.2f}) -> rel error = {(p_mod.derived["ne_vol20"] - p.derived["ne_vol20"])/p.derived["ne_vol20"]*100.0:.1f}%')
        print(f'\t\t- Pfus: {p_mod.derived["Pfus"]:.2f} (target={p.derived["Pfus"]:.2f}) -> rel error = {(p_mod.derived["Pfus"] - p.derived["Pfus"])/p.derived["Pfus"]*100.0:.4f}%')
        print(f'\t\t- BetaN: {p_mod.derived["BetaN"]:.2f} (target={p.derived["BetaN"]:.2f}) -> rel error = {(p_mod.derived["BetaN"] - p.derived["BetaN"])/p.derived["BetaN"]*100.0:.4f}%')
        print(f'\t\t- tite_vol: {p_mod.derived["tite_vol"]:.2f} (initial={p.derived["tite_vol"]:.2f}) -> rel change = {(p_mod.derived["tite_vol"] - p.derived["tite_vol"])/p.derived["tite_vol"]*100.0:.1f}%')

    # Store errors to pass as information
    rel_errors = []
    for i in range(len(p_mods)):
        p_mod = p_mods[i]
        p = ps[i]
        rel_errors.append(abs((p_mod.derived["ne_vol20"] - p.derived["ne_vol20"]) / p.derived["ne_vol20"] * 100.0))
        rel_errors.append(abs((p_mod.derived["Pfus"] - p.derived["Pfus"]) / p.derived["Pfus"] * 100.0))
        rel_errors.append(abs((p_mod.derived["BetaN"] - p.derived["BetaN"]) / p.derived["BetaN"] * 100.0))

    mean_error = float(np.mean(rel_errors)) if len(rel_errors) > 0 else np.nan
    
    if plotYN:
        fn = state_plotting.plotAll(ps+p_mods)

        fn.show()
        
    if provideBetaN_multiplier:
        return {'aLTe': aLTe, 'aLn': aLn, 'aLTi': aLTi, 'BetaN_multiplier': BetaN_multiplier}, p_mods, mean_error
    else:
        return {'aLTe': aLTe, 'aLn': aLn, 'aLTi': aLTi}, p_mods, mean_error