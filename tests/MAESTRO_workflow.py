import os
from mitim_modules.maestro import MAESTROmain
from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools import __mitimroot__

cold_start = True

folder = __mitimroot__ / "tests" / "scratch" / "maestro_test"

if cold_start and os.path.exists(folder):
    os.system(f"rm -r {folder}")

folder.mkdir(parents=True, exist_ok=True)

params = {'Ip_MA': 1.15, 'B_T': 2.1, 'Zeff': 2.0, 'PichT_MW': 2.2}
geometry = {'R': 1.675, 'a': 0.67, 'kappa_sep': 1.8, 'delta_sep': 0.237, 'zeta_sep': 0.0, 'z0': 0.0}
params_bc = {'rhotop': 0.9, 'Ttop_keV': 0.6, 'netop_20': 0.3, 'Tsep_keV': 0.1, 'nesep_20': 0.1}
params_init = {'BetaN': 1.0}
def P_aux(rhotor): return PLASMAtools.parabolicProfile(Tbar=1.0,nu=5.0,rho=rhotor,Tedge=0.0)[-1]

transp_namelist = { 'flattop_window': 0.1, 'zlump' :[ [6.0, 12.0, 0.1] ],
                    'dtEquilMax_ms': 10.0, 'dtHeating_ms' : 5.0, 'dtOut_ms' : 10.0, 'dtIn_ms' : 10.0,
                    'nzones' : 60, 'nzones_energetic' : 20, 'nzones_distfun' : 10,      
                    'Pich': False, 'useNUBEAMforAlphas':False, 'DTplasma': False }

portals_namelist = { "optimization_options": {"BO_iterations": 2 },
                     "exploration_ranges": {'ymax_rel': 1.0,'ymin_rel': 0.9,'hardGradientLimits': [None,2]} }

@mitim_timer('\t\t* MAESTRO')
def run_maestro():

    m = MAESTROmain.maestro(folder, master_cold_start = cold_start, terminal_outputs = True)

    m.define_beat('transp', initializer='freegs')
    m.define_creator('parameterization', **params_bc, **params_init)
    m.initialize(**geometry, **params, **params_init)
    m.prepare(**transp_namelist)
    m.run(force_auxiliary_heating_at_output = { 'Pe': [P_aux, params['PichT_MW']*0.5], 'Pi': [P_aux, params['PichT_MW']*0.5]})

    m.define_beat('portals')
    m.prepare(**portals_namelist)
    m.run()

    m.define_beat('transp')
    m.prepare(**transp_namelist)
    m.run(force_auxiliary_heating_at_output = { 'Pe': [P_aux, params['PichT_MW']*0.5], 'Pi': [P_aux, params['PichT_MW']*0.5]})

    m.define_beat('portals')
    m.prepare(**portals_namelist)
    m.run()

    m.finalize()

    return m

m = run_maestro()

m.plot(num_beats = 4)