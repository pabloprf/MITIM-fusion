from mitim_modules.maestro.MAESTROmain import maestro

mfe_im_path = '/Users/pablorf/MFE-IM'
folder = '/Users/pablorf/PROJECTS/project_2024_MITIMsurrogates/maestro_development/tests12/arc8'

# -----------------------------------------------------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------------------------------------------------

parameters  = {'Ip_MA': 10.95, 'B_T': 10.8, 'Zeff': 1.5, 'PichT_MW': 18.0, 'neped_20' : 2.0 , 'Tesep_keV': 0.1, 'nesep_20': 2.0/3.0}
parameters_mix = {'DTplasma': True, 'lowZ_impurity': 9.0, 'impurity_ratio_WtoZ': 0.00286*0.5, 'minority': [1,1,0.05]}

#geometry    = {'R': 4.24, 'a': 1.17, 'kappa_sep': 1.77, 'delta_sep': 0.58, 'zeta_sep': 0.0, 'z0': 0.0}
geometry = {'geqdsk_file': f'{mfe_im_path}/private_data/ARCV2B.geqdsk', 'coeffs_MXH' : 5}

BetaN_initialization = 1.5

# -----------------------------------------------------------------------------------------------------------------------
# Namelists
# -----------------------------------------------------------------------------------------------------------------------

# To see what values this namelist can take: mitim_tools/transp_tools/NMLtools.py: _default_params()
transp_namelist = {
    'flattop_window': 0.2,       # <--- To allow stationarity
    'extractAC': True,           # <--- To extract TORIC and NUBEAM extra files
    'dtEquilMax_ms': 10.0,       # Default
    'dtHeating_ms' : 5.0,        # Default
    'dtOut_ms' : 10.0,
    'dtIn_ms' : 10.0,
    'nzones' : 60,
    'nzones_energetic' : 20,    # Default but lower than what I used to use
    'nzones_distfun' : 10,      # Default but lower than what I used to use    
    'MCparticles' : 1e4,
    'toric_ntheta' : 64,        # Default values of TORIC, but lower than what I used to use
    'toric_nrho' : 128,         # Default values of TORIC, but lower than what I used to use
    'Pich': parameters['PichT_MW']>0.0,
    'DTplasma': parameters_mix['DTplasma'],
    'Minorities': parameters_mix['minority'],
    "zlump" :[  [74.0, 184.0, 0.1*parameters_mix['impurity_ratio_WtoZ']],
                [parameters_mix['lowZ_impurity'], parameters_mix['lowZ_impurity']*2, 0.1] ],
    }

# To see what values this namelist can take: mitim_modules/portals/PORTALSmain.py: __init__()
portals_namelist = {    "PORTALSparameters": {"launchEvaluationsAsSlurmJobs": True,"forceZeroParticleFlux": True, 'use_tglf_scan_trick': 0.02},
                        "MODELparameters": { "RoaLocations": [0.35,0.55,0.75,0.875,0.9],
                                            "ProfilesPredicted": ["te", "ti", "ne"],
                                            "Physics_options": {"TypeTarget": 3},
                                             "transport_model": {"turbulence":'TGLF',"TGLFsettings": 6, "extraOptionsTGLF": {'USE_BPER':True}}},
                        "INITparameters": {"FastIsThermal": True, "removeIons": [5,6], "quasineutrality": True},
                        "optimization_options": {"BO_iterations": 5,"maximum_value": 1e-2,"maximum_value_is_rel": True, "StrategyOptions": {"AllowedExcursions":[0.0, 0.0]} } }

# To see what values this namelist can take: mitim_modules/maestro/utils/EPEDbeat.py: prepare()
eped_parameters = { 'nn_location': f'{mfe_im_path}/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5',
                    'norm_location': f'{mfe_im_path}/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt' }

# -----------------------------------------------------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------------------------------------------------

m = maestro(folder, terminal_outputs = True)

m.define_beat('transp', initializer='geqdsk')
m.define_creator('eped', BetaN = BetaN_initialization, **eped_parameters,**parameters)
m.initialize(**geometry, **parameters)
m.prepare(**transp_namelist)
m.run(checkMin=3, retrieveAC=transp_namelist['extractAC'])

m.define_beat('eped')
m.prepare(**eped_parameters)
m.run()

m.define_beat('portals')
m.prepare(**portals_namelist)
m.run()

m.define_beat('transp')
m.prepare(**transp_namelist)
m.run(checkMin=3, retrieveAC=transp_namelist['extractAC'])

m.define_beat('eped')
m.prepare(**eped_parameters)
m.run()

m.define_beat('portals')
m.prepare(**portals_namelist)
m.run()

m.finalize()

# # # PORTALS beat only evolving the temperature profiles and fixed targets (5 iterations)
# # import copy
# # portals_namelist_beat = copy.deepcopy(portals_namelist)
# # portals_namelist_beat["MODELparameters"]['ProfilesPredicted'] = ["te", "ti"]
# # portals_namelist_beat["MODELparameters"]['Physics_options']["TypeTarget"] = 1
# # portals_namelist_beat["optimization_options"]["BO_iterations"] = 5
# # portals_namelist_beat["additional_params_in_surrogate"] = ['aLne'] # Such that I can reuse surrogate data in next PORTALS
# # exploration_ranges = { 'ymax_rel': 1.0, 'ymin_rel': 0.8, 'hardGradientLimits': [0.1,2]}

# # m.define_beat('portals')
# # m.prepare(exploration_ranges = exploration_ranges, **portals_namelist_beat)
# # m.run()

# # # PORTALS beat evolving density too but still with fixed targets (5 iterations)
# # portals_namelist_beat["MODELparameters"]['ProfilesPredicted'] = ["te", "ti", "ne"]
# # portals_namelist_beat["additional_params_in_surrogate"] = []
# # exploration_ranges = { 'ymax_rel': 1.0, 'ymin_rel': 0.8, 'hardGradientLimits': [0.1,2]}

# # m.define_beat('portals')
# # m.prepare(exploration_ranges = exploration_ranges, use_previous_surrogate_data=True,**portals_namelist_beat)
# # m.run()

# # PORTALS beat, full
# exploration_ranges = { 'ymax_rel': 1.0, 'ymin_rel': 0.75, 'hardGradientLimits': [0.2,2]}
# m.define_beat('portals')
# m.prepare(exploration_ranges = exploration_ranges, use_previous_surrogate_data=True,**portals_namelist)
# m.run()

# # PORTALS beat, full
# exploration_ranges = { 'ymax_rel': 1.0, 'ymin_rel': 0.75, 'hardGradientLimits': [0.2,2]}
# m.define_beat('portals')
# m.prepare(exploration_ranges = exploration_ranges, use_previous_surrogate_data=True,**portals_namelist)
# m.run()

# 
