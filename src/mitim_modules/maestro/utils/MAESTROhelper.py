from mitim_tools.misc_tools import CONFIGread
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_modules.maestro.MAESTROmain import maestro
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

@mitim_timer('\t- MAESTRO')
def simple_maestro_workflow(
    folder,
    geometry,
    parameters,
    Tbc_keV,
    nbc_20,
    TGLFsettings = 6,
    DTplasma = True,
    terminal_outputs = False,
    full_loops = 2, # By default, do 2 loops of TRANSP-PORTALS
    quality = {
        'maximum_value': 1e-2,  # x100 better residual
        'BO_iterations': 20,
        'flattop_window': 0.15, # s
        }
    ):

    m = maestro(folder, terminal_outputs = terminal_outputs)
    
    # ---------------------------------------------------------
    # beat 0: Define info
    # ---------------------------------------------------------

    # Simple profiles
    parameters_profiles = {'rhotop' : 0.95, 'Ttop' : Tbc_keV, 'netop' : nbc_20}

    # Faster TRANSP (different than defaults)
    transp_namelist = {
        'Pich'   : True,
        'dtEquilMax_ms': 1.0,       # Higher resolution than default (10.0) to avoid quval error
        'dtHeating_ms' : 5.0,       # Default
        'dtOut_ms' : 10.0,
        'dtIn_ms' : 10.0,
        'nzones' : 60,
        'nzones_energetic' : 20,    # Default but lower than what I used to use
        'nzones_distfun' : 10,      # Default but lower than what I used to use    
        'MCparticles' : 1e4,
        'toric_ntheta' : 64,        # Default values of TORIC, but lower than what I used to use
        'toric_nrho' : 128,         # Default values of TORIC, but lower than what I used to use
        'DTplasma': DTplasma
    }

    # Simple PORTALS

    portals_namelist = {
        "PORTALSparameters": {
            "launchEvaluationsAsSlurmJobs": not CONFIGread.isThisEngaging(),
            "forceZeroParticleFlux": True
        },
        "MODELparameters": {
            "RoaLocations": [0.35,0.55,0.75,0.875,0.9],
            "transport_model": {"turbulence":'TGLF',"TGLFsettings": TGLFsettings, "extraOptionsTGLF": {}}
        },
        "INITparameters": {
            "FastIsThermal": True
        },
        "optimization_options": {
            "BO_iterations": quality.get('BO_iterations', 20),
            "maximum_value": quality.get('maximum_value', 1e-2),
            "maximum_value_is_rel": True,
        }
    }

    # ------------------------------------------------------------
    # beat N: TRANSP from FreeGS and freeze engineering parameters
    # ------------------------------------------------------------

    m.define_beat('transp', initializer='geqdsk' if 'geqdsk_file' in geometry else 'freegs')
    m.define_creator('parameterization', **parameters_profiles)
    m.initialize(**geometry, **parameters)
    m.prepare(flattop_window = quality.get('flattop_window', 0.15), **transp_namelist)
    # m.run()

    # # ---------------------------------------------------------
    # # beat N+1: PORTALS from TRANSP
    # # ---------------------------------------------------------

    # m.define_beat('portals')
    # m.prepare(**portals_namelist)
    # m.run()

    # for i in range(full_loops-1):

    #     # ---------------------------------------------------------
    #     # beat N: TRANSP from PORTALS
    #     # ---------------------------------------------------------

    #     m.define_beat('transp')
    #     m.prepare(flattop_window = quality.get('flattop_window', 0.15), **transp_namelist)
    #     m.run()

    #     # ---------------------------------------------------------
    #     # beat N+1: PORTALS from TRANSP
    #     # ---------------------------------------------------------

    #     m.define_beat('portals')
    #     m.prepare(**portals_namelist)
    #     m.run()

    # # ---------------------------------------------------------
    # # Finalize
    # # ---------------------------------------------------------

    # m.finalize()

    return m