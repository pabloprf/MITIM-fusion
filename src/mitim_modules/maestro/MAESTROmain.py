import torch
import os
import numpy as np
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.misc_tools import IOtools, GUItools, CONFIGread
from mitim_modules.maestro.utils import MAESTROplot
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools import __version__
from IPython import embed

from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat

'''
MAESTRO:
    Modular and Accelerated Engine for Simulation of Transport and Reactor Optimization
 (If MAESTRO is the orchestrator, then BEAT is each of the beats (steps) that MAESTRO orchestrates)

Description of capabilities:

- If initialized with... # TO DO

'''

# --------------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------------

class maestro:

    def __init__(self, folder, terminal_outputs = False, master_restart = False):
        '''
        Inputs:
            - folder: Main folder where all the beats will be saved
            - terminal_outputs: If True, all outputs will be printed to terminal. If False, they will be saved to a log file per beat step
        '''

        self.folder = IOtools.expandPath(folder)
        
        self.folder_output = f'{self.folder}/Outputs/'
        self.folder_logs = f'{self.folder_output}/Logs/'
        os.makedirs(self.folder_logs, exist_ok=True)

        self.folder_beats = f'{self.folder}/Beats/'
        os.makedirs(self.folder_beats, exist_ok=True)

        self.save_file = f'{self.folder_output}/maestro_save.pkl'
        self.terminal_outputs = terminal_outputs

        self.beats = {}
        self.counter = 0

        self.master_restart = master_restart # If True, all next beats will be restarted

        self.parameters_trans_beat = {} # I can save parameters that can be useful for future beats
        self.engineering_parameters = {}

        print('\n -----------------------------------------------------------------------------------')
        print(f'MAESTRO run (MITIM version {__version__})')
        print('-----------------------------------------------------------------------------------')
        print(f'folder: {self.folder}')

    def define_beat(self, beat, initializer = None, restart = False):

        self.counter += 1
        if beat == 'transp':
            print(f'\n- Beat {self.counter}: TRANSP ********************************************************************')
            self.beats[self.counter] = transp_beat(self)
        elif beat == 'portals':
            print(f'\n- Beat {self.counter}: PORTALS ********************************************************************')
            self.beats[self.counter] = portals_beat(self)

        # Access current beat easily
        self.beat = self.beats[self.counter]

        # Define initializer
        self.beat.define_initializer(initializer)

        # Check here if the beat has already been performed
        self._check(restart = restart or self.master_restart )

    def freeze_parameters(self, Ip, Bt, Zeff, PichT_MW, RZsep):
        '''
        During MAESTRO, the separatrix and main engineering parameters do not change, so I need 
        to freeze them upon initialization, otherwise we'll have a leak of power or geometry quantities
        if beat by beat it's, for whatever reason, lower.
        In other words, e.g., it's best to not pass the ICH power from input.gacode PORTALS to TRANSP, but
        rather take the original intended ICH power
        '''

        self.engineering_parameters = {
            'Ip_MA': Ip,
            'B_T': Bt,
            'Zeff': Zeff,
            'PichT_MW': PichT_MW,
            'RZsep': RZsep
        }

    # --------------------------------------------------------------------------------------------
    # Beat operations
    # --------------------------------------------------------------------------------------------
    
    @mitim_timer('\t\t* Checker')
    def _check(self, restart = False, **kwargs):
        '''
        Note:
            After each beat, the results are passed to an output folder.
            If the required files are already there, the beat will not be run again.
            It is also assumed that the results were correct if they were put there, so
            the checks should happen in the finalize() method of each beat.
        '''

        print('\t- Checking...')
        log_file = f'{self.folder_logs}/beat_{self.counter}_check.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            exists = self.beat.check(restart = restart, **kwargs)
            self.beat.run_flag = not exists

        # If this beat is restarted, all next beats will be restarted
        if self.beat.run_flag:
            if not self.master_restart:
                print('\t\t- Since this step needs to start from scratch, all next ones will too', typeMsg = 'i')
            self.master_restart = True
            
    @mitim_timer('\t\t* Initializer')
    def initialize(self, *args, **kwargs):

        print('\t- Initializing...')
        if self.beat.run_flag:
            log_file = f'{self.folder_logs}/beat_{self.counter}_ini.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.initialize(*args, **kwargs)
        else:
            print('\t\t- Skipping beat initialization because this beat was already run', typeMsg = 'i')

    @mitim_timer('\t\t* Preparation')
    def prepare(self, *args, **kwargs):

        print('\t- Preparing...')
        if self.beat.run_flag:
            log_file = f'{self.folder_logs}/beat_{self.counter}_prep.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.prepare(*args, **kwargs)
        else:
            print('\t\t- Skipping beat preparation because this beat was already run', typeMsg = 'i')

    @mitim_timer('\t\t* Run')
    def run(self, **kwargs):

        # Run 
        print('\t- Running...')
        if self.beat.run_flag:
            log_file = f'{self.folder_logs}/beat_{self.counter}_run.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.run(**kwargs)

                # Finalize
                self.beat.finalize()

        else:
            print('\t\t- Skipping beat run because this beat was already run', typeMsg = 'i')

        # Inform next beats
        log_file = f'{self.folder_logs}/beat_{self.counter}_inform.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file):
            self.beat.inform_save()

    @mitim_timer('\t\t* Finalizing')
    def finalize(self):

        print('\t- Finalizing MAESTRO run...')
        
        log_file = f'{self.folder_output}/beat_final' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            
            self.beat.finalize_maestro()

    # --------------------------------------------------------------------------------------------
    # Plotting operations
    # --------------------------------------------------------------------------------------------
    
    @mitim_timer('\t\t* Plotting')
    def plot(self, fn = None, num_beats = 2, only_beats = None, full_plot = True):

        print('*** Plotting MAESTRO ******************************************************************** ')

        if fn is None:
            wasProvided = False
            self.fn = GUItools.FigureNotebook("MAESTRO")
        else:
            wasProvided = True
            self.fn = fn

        self._plot_beats(self.fn, num_beats = num_beats, only_beats = only_beats, full_plot = full_plot)
        self._plot_results(self.fn)

        if not wasProvided:
            self.fn.show()

    def _plot_beats(self, fn, num_beats = 2, only_beats = None, full_plot = True):

        beats_keys = sorted(sorted(list(self.beats.keys()),reverse=True)[:num_beats])
        for i,counter in enumerate(beats_keys):
            beat = self.beats[counter]
            if only_beats is None or only_beats == beat.name:

                print(f'\t- Plotting beat #{counter}...')
                log_file = f'{self.folder_logs}/plot_{counter}.log' if (not self.terminal_outputs) else None
                with IOtools.conditional_log_to_file(log_file=log_file):
                    msg = beat.plot(fn = self.fn, counter = i, full_plot = full_plot)
                print(msg)

    def _plot_results(self, fn):

        print('\t- Plotting MAESTRO results...')

        MAESTROplot.plot_results(self, fn)

# --------------------------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------------------------

def procreate(y_top = 2.0, y_sep = 0.1, w_top = 0.07, aLy = 2.0, w_a = 0.3):
    
    roa = np.linspace(0.0, 1-w_top, 100)
    aL_profile = np.zeros_like(roa)
    linear_region = roa <= w_a
    aL_profile[linear_region] = (aLy / w_a) * roa[linear_region]
    aL_profile[~linear_region] = aLy
    y = CALCtools.integrateGradient(torch.from_numpy(roa).unsqueeze(0), torch.from_numpy(aL_profile).unsqueeze(0), y_top).numpy()
    roa = np.append( roa, 1.0)
    y = np.append(y, y_sep)

    return roa, y

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

    w_top, w_a = 0.05, 0.3
    rho, Te = procreate(y_top = Tbc_keV, y_sep = 0.1, w_top = w_top, aLy = 1.7, w_a = w_a)
    rho, Ti = procreate(y_top = Tbc_keV, y_sep = 0.1, w_top = w_top, aLy = 1.5, w_a = w_a)
    rho, ne = procreate(y_top = nbc_20, y_sep = nbc_20/3.0, w_top = w_top, aLy = 0.2, w_a = w_a)
    profiles = {'Te': [rho, Te],'Ti': [rho, Ti],'ne': [rho, ne]}

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

    m.define_beat('transp', initializer='freegs')
    m.initialize(**geometry,**parameters, profiles = profiles, flattop_window = quality.get('flattop_window', 0.15), freeze_parameters = True)
    m.prepare(**transp_namelist)
    m.run()

    # ---------------------------------------------------------
    # beat N+1: PORTALS from TRANSP
    # ---------------------------------------------------------

    m.define_beat('portals', initializer='transp')
    m.initialize()
    m.prepare(**portals_namelist)
    m.run()

    for i in range(full_loops-1):
        # ---------------------------------------------------------
        # beat N: TRANSP from PORTALS
        # ---------------------------------------------------------

        m.define_beat('transp', initializer='portals')
        m.initialize(flattop_window = quality.get('flattop_window', 0.15))
        m.prepare(**transp_namelist)
        m.run()

        # ---------------------------------------------------------
        # beat N+1: PORTALS from TRANSP
        # ---------------------------------------------------------

        m.define_beat('portals', initializer='transp')
        m.initialize()
        m.prepare(**portals_namelist)
        m.run()

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------

    m.finalize()

    return m

