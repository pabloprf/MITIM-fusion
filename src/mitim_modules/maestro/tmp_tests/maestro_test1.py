from tensorflow._api.v2 import compat 
from mitim_modules.maestro.MAESTROmain import maestro

mfe_im_path = '/Users/pablorf/MFE-IM'
folder = '/Users/pablorf/PROJECTS/project_2024_ARCim/maestro_runs/runs_v2/arcV2B_run15/'

# -----------------------------------------------------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------------------------------------------------

parameters  = {'Ip_MA': 10.95, 'B_T': 10.8, 'Zeff': 1.5, 'PichT_MW': 18.0, 'neped_20' : 1.8 , 'Tesep_keV': 0.1, 'nesep_20': 2.0/3.0}
parameters_mix = {'DTplasma': True, 'lowZ_impurity': 9.0, 'impurity_ratio_WtoZ': 0.00286*0.5, 'minority': [1,1,0.02]}

#initializer, geometry    = 'freegs', {'R': 4.25, 'a': 1.17, 'kappa_sep': 1.77, 'delta_sep': 0.58, 'zeta_sep': 0.0, 'z0': 0.0}
initializer, geometry = 'geqdsk', {'geqdsk_file': f'{mfe_im_path}/private_data/ARCV2B.geqdsk', 'coeffs_MXH' : 7}

BetaN_initialization = 1.5

# -----------------------------------------------------------------------------------------------------------------------
# Namelists
# -----------------------------------------------------------------------------------------------------------------------

# To see what values this namelist can take: mitim_tools/transp_tools/NMLtools.py: _default_params()
transp_namelist = {
    'flattop_window': 1.0,       # <--- To allow stationarity
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
                        "optimization_options": {
                            "convergence_options": {
                                "maximum_iterations": 50,
                                "stopping_criteria_parameters": {
                                    "maximum_value": 1e-3,
                                    "maximum_value_is_rel": True,
                                    },
                                },
                            "strategy_options": {
                                "AllowedExcursions":[0.0, 0.0]
                                 },
                            },
                        "exploration_ranges": {
                            'ymax_rel': 1.0,
                            'ymin_rel': 0.9,
                            'hardGradientLimits': [None,2]
                        }
                        }

# To see what values this namelist can take: mitim_modules/maestro/utils/EPEDbeat.py: prepare()
eped_parameters = { 'nn_location': f'{mfe_im_path}/private_code_mitim/NN_DATA/EPED-NN-ARC/new-EPED-NN-MODEL-ARC.keras',
                    'norm_location': f'{mfe_im_path}/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt'}

# -----------------------------------------------------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------------------------------------------------

from mitim_tools.misc_tools.IOtools import mitim_timer

@mitim_timer('\t\t* MAESTRO')
def run_maestro():
    m = maestro(folder, terminal_outputs = False)

    # TRANSP with only current diffusion
    transp_namelist['flattop_window'] = 10.0
    transp_namelist['dtEquilMax_ms'] = 50.0 # Let the equilibrium evolve with long steps
    transp_namelist['useNUBEAMforAlphas'] = False
    transp_namelist['Pich'] = False

    m.define_beat('transp', initializer=initializer)
    m.define_creator('eped', BetaN = BetaN_initialization, **eped_parameters,**parameters)
    m.initialize(**geometry, **parameters)
    m.prepare(**transp_namelist)
    m.run()

    # TRANSP for toric and nubeam
    transp_namelist['flattop_window'] = 0.5
    transp_namelist['dtEquilMax_ms'] = 10.0
    transp_namelist['useNUBEAMforAlphas'] = True
    transp_namelist['Pich'] = True
    
    m.define_beat('transp')
    m.prepare(**transp_namelist)
    m.run()

    # EPED
    m.define_beat('eped')
    m.prepare(**eped_parameters)
    m.run()

    # PORTALS
    m.define_beat('portals')
    m.prepare(**portals_namelist, change_last_radial_call = True)
    m.run()

    # TRANSP
    m.define_beat('transp')
    m.prepare(**transp_namelist)
    m.run()

    for i in range(9):
        # EPED
        m.define_beat('eped')
        m.prepare(**eped_parameters)
        m.run()

        # PORTALS
        m.define_beat('portals')
        m.prepare(**portals_namelist,use_previous_surrogate_data=i>0, change_last_radial_call = True) # Reuse the surrogate data if I'm not coming from a TRANSP run
        m.run()

    m.finalize()

    return m

m = run_maestro()