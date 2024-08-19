import torch
import os
# import dill
import copy
import hashlib
import numpy as np
from mitim_tools.gs_tools import FREEGStools
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.opt_tools 	 	import STRATEGYtools
from mitim_modules.portals 		import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools 	import CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools import __version__
from IPython import embed

'''
MAESTRO:
    Modular and Accelerated Engine for Simulation of Transport and Reactor Optimization

If MAESTRO is the orchestrator, then BEAT is each of the beats (steps) that MAESTRO orchestrates
'''

# --------------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------------

class maestro:

    def __init__(self, folder, terminal_outputs = False):
        '''
        Inputs:
            - folder: Main folder where all the beats will be saved
            - terminal_outputs: If True, all outputs will be printed to terminal. If False, they will be saved to a log file per beat step
        '''

        self.folder = IOtools.expandPath(folder)
        self.folder_output = f'{self.folder}/Outputs/'
        os.makedirs(self.folder_output, exist_ok=True)

        self.save_file = f'{self.folder_output}/maestro_save.pkl'
        self.terminal_outputs = terminal_outputs

        self.beats = {}
        self.counter = 0

        print('\n -----------------------------------------------------------------------------------')
        print(f'MAESTRO run (MITIM version {__version__})')
        print('-----------------------------------------------------------------------------------')
        print(f'folder: {self.folder}')

    def define_beat(self, beat, initializer):

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

    @mitim_timer('\t\t* Initializer')
    def initialize(self, *args, **kwargs):

        print('\t- Initializing...')
        log_file = f'{self.folder_output}/beat_{self.counter}_ini.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            self.beat.initialize(*args, **kwargs)

    @mitim_timer('\t\t* Preparation')
    def prepare(self, *args, **kwargs):

        print('\t- Preparing...')
        log_file = f'{self.folder_output}/beat_{self.counter}_prep.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            self.beat.prepare(*args, **kwargs)

    def run(self, **kwargs):

        # Save status
        self.save()

        # Run
        self._run(**kwargs)

        # Save status
        self.save()

    @mitim_timer('\t\t* Run')
    def _run(self, **kwargs):

        restart = kwargs.get('restart', False)

        # Check if output file already exists
        log_file = f'{self.folder_output}/beat_{self.counter}_check.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            exists = self.beat.check(restart=restart)

        # Run 
        print('\t- Running...')
        if not exists:
            log_file = f'{self.folder_output}/beat_{self.counter}_run.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.run(**kwargs)
        else:
            print('\t\t- Skipping beat because output file was found', typeMsg = 'i')

    @mitim_timer('\t\t* Saving')
    def save(self):

        print('\t- Saving...')
        # with open(self.save_file, 'wb') as f:
        #     dill.dump(self, f)
        print('\t\t* Cannot save yet',typeMsg = 'w')
        #embed()

    @mitim_timer('\t\t* Finalizing')
    def finalize(self):

        print('\t- Finalizing MAESTRO run...')
        
        # Find final input.gacode
        if isinstance(self.beat, portals_beat):
            portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.beat.folder_portals)
            self.final_p = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles

        elif isinstance(self.beat, transp_beat):
            cdf = CDFtools.transp_output(f"{self.beat.folder}/{self.beat.transp.shot}{self.beat.transp.runid}.CDF")
            self.final_p = cdf.to_profiles()
        
        # Save final input.gacode
        final_file = f'{self.folder_output}/input.gacode_final'
        self.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

# --------------------------------------------------------------------------------------------
# beatper: TRANSP
# --------------------------------------------------------------------------------------------

class transp_beat:

    def __init__(self, maestro_instance):
            
        self.maestro_instance = maestro_instance
        self.folder = f'{self.maestro_instance.folder}/beat_{self.maestro_instance.counter}/transp/'

        os.makedirs(self.folder, exist_ok=True)

        # Hardcoded for now how long I want each phase to be
        transition_window       = 0.1    # s
        currentheating_window   = 0.001  # s
        flattop_window          = 0.15   # s

        # Define timings
        self.time_init = 0.0                                                # Start with D3D equilibrium
        self.time_transition = self.time_init+ transition_window            # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End

    # --------------------------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------------------------
    def define_initializer(self, initializer):

        if initializer == 'freegs':
            self.initializer = transp_initializer_from_freegs(self)
        elif initializer == 'portals':
            self.initializer = transp_initializer_from_portals(self)
        else:
            raise ValueError(f'Initializer "{initializer}" not recognized')

    def initialize(self,*args,  **kwargs):

        self.initializer(*args,**kwargs)

    def prepare(self, letter = None, shot = None,transp_namelist = {}, **kwargs_initializer):

        '''
        Using some smart defaults to avoid repeating TRANSP runid
            shot will be 5 digits that depend on the last subfolder
                e.g. run_cmod1 -> '94351', run_cmod2 -> '94352', run_d3d1 -> '72821', etc
            letter will depend on username in this machine, if it can be found
                e.g. pablorf -> 'P"
        '''
        if shot is None:
            folder_last = os.path.basename(os.path.normpath(self.maestro_instance.folder))
            shot = string_to_sequential_5_digit_number(folder_last)

        if letter is None:
            username = IOtools.expandPath('$USER')
            letter = username[0].upper()
            if letter == '$':
                letter = 'A'
        # ---------------------------------------------------------

        # Define run parameters
        self.shot = shot
        self.runid = letter + str(self.maestro_instance.counter).zfill(2)

        # Use initializer to prepare beat
        self.initializer.prepare_to_beat(**kwargs_initializer)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generatic TRANSP operation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        transp_namelist_mod = copy.deepcopy(transp_namelist)

        if 'timings' in transp_namelist_mod:
            raise ValueError('Cannot define timings in MAESTRO transp_namelist')
        else:
            transp_namelist_mod['timings'] = {
                "time_start": self.time_init,
                "time_current_diffusion": self.time_diffusion,
                "time_end": self.time_end
            }

        if 'Ufiles' in transp_namelist_mod:
            raise ValueError('Cannot define UFILES in MAESTRO transp_namelist')
        else:
            transp_namelist_mod['Ufiles'] = ["qpr","mry","cur","vsf","ter","ti2","ner","rbz","lim","zf2"]

        # Write namelist
        self.transp.write_namelist(**transp_namelist_mod)

        # Trick needed to avoid quval error when starting from D3D
        self.transp.populate_time.from_freegs(0.0, 1.67, 0.6, 1.75, 0.38, 0.0, 0.0, 0.074, 1.6, 2.0)
        
        # ICRF on
        Frequency_He3 = self.initializer.B_T * (2*np.pi /(2/3))  # ~10
        self.transp.icrf_on_time(self.time_diffusion, power_MW = self.initializer.PichT_MW, freq_MHz = Frequency_He3)
        
        # Write Ufiles
        self.transp.write_ufiles()

    # --------------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------------

    def check(self, restart = False):
        '''
        Check if output file already exists so that I don't need to run this beat again
        '''

        output_file = f'{self.folder}/{self.shot}{self.runid}.CDF'

        if os.path.exists(output_file) and (not restart):
            self.c = CDFtools.transp_output(output_file)
            print('Output file already exists, not running beat', typeMsg = 'i')
            return True
        else:
            return False

    def run(self, **kwargs):

        self.transp.run('D3D', mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*12, case=self.transp.runid, checkMin=5)
        self.c = self.transp.c

    # --------------------------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------------------------
    
    def plot(self):

        self.c.plot()

class transp_initializer_from_freegs:

    def __init__(self, beat_instance):
            
        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder}/initializer_freegs/'

        os.makedirs(self.folder, exist_ok=True)

    def __call__(
        self,
        R,
        a,
        kappa_sep,
        delta_sep,
        zeta_sep,
        z0,
        Ip_MA,
        B_T,
        Zeff,
        PichT_MW,
        p0_MPa = 2.5,
        ne0_20 = 3.0,
        profiles = {}):

        self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0 = R, a, kappa_sep, delta_sep, zeta_sep, z0
        self.p0_MPa, self.Ip_MA, self.B_T = p0_MPa, Ip_MA, B_T
        self.ne0_20, self.Zeff, self.PichT_MW = ne0_20, Zeff, PichT_MW
        self.profiles = profiles

        # FreeGS
        self.f = FREEGStools.freegs_millerized(self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0)
        self.f.prep(self.p0_MPa, self.Ip_MA, self.B_T)
        self.f.solve()
        self.f.derive()
        self.f.write(f'{self.folder}/freegs.geqdsk')

    def prepare_to_beat(self):

        times = [self.beat_instance.time_transition,self.beat_instance.time_end+1.0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FreeGS to TRANSP
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.transp = self.f.to_transp(
            folder = self.beat_instance.folder,
            shot = self.beat_instance.shot, runid = self.beat_instance.runid, times = times,
            ne0_20 = self.ne0_20, Zeff = self.Zeff, PichT_MW = self.PichT_MW)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add profiles  # ROA VA RHO???, psi, etc
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if 'Te' in self.profiles:
            for time in times:
                self.transp.add_variable_time(time, self.profiles['Te'][0], self.profiles['Te'][1]*1E3, variable='TEL')
        if 'Ti' in self.profiles:
            for time in times:
                self.transp.add_variable_time(time, self.profiles['Ti'][0], self.profiles['Ti'][1]*1E3, variable='TIO')
        if 'ne' in self.profiles:
            for time in times:
                self.transp.add_variable_time(time, self.profiles['ne'][0], self.profiles['ne'][1]*1E20*1E-6, variable='NEL')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pass to main class' beat
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.beat_instance.transp = self.transp

    def plot(self):

        self.f.plot()

class transp_initializer_from_portals:

    def __init__(self, beat_instance):
            
        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder}/initializer_portals/'

        os.makedirs(self.folder, exist_ok=True)

    def __call__(
        self,
        R,
        a,
        kappa_sep,
        delta_sep,
        zeta_sep,
        z0,
        Ip_MA,
        B_T,
        Zeff,
        PichT_MW,
        p0_MPa = 2.5,
        ne0_20 = 3.0):

        # Load params
        self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0 = R, a, kappa_sep, delta_sep, zeta_sep, z0
        self.p0_MPa, self.Ip_MA, self.B_T = p0_MPa, Ip_MA, B_T
        self.ne0_20, self.Zeff, self.PichT_MW = ne0_20, Zeff, PichT_MW

        # From previous PORTALS
        folder_portals =  self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1].folder_portals

        self.portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folder_portals)
        
        self.p = self.portals_output.mitim_runs[self.portals_output.ibest]['powerstate'].profiles

        self.p.writeCurrentStatus(file=self.folder+'/input.gacode' )

    def prepare_to_beat(self):

        times = [self.beat_instance.time_transition,self.beat_instance.time_end+1.0]

        self.transp = self.p.to_transp(
            folder = self.beat_instance.folder,
            shot = self.beat_instance.shot, runid = self.beat_instance.runid, times = times)

        # Pass to main class' beat
        self.beat_instance.transp = self.transp

    def plot(self):
        
        self.portals_output.plot()

# --------------------------------------------------------------------------------------------
# beatper: PORTALS
# --------------------------------------------------------------------------------------------

class portals_beat:

    def __init__(self, maestro_instance):
            
        self.maestro_instance = maestro_instance
        self.folder = f'{self.maestro_instance.folder}/beat_{self.maestro_instance.counter}/portals/'

        os.makedirs(self.folder, exist_ok=True)

    # --------------------------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------------------------
    def define_initializer(self, initializer):

        if initializer == 'transp':
            self.initializer = portals_initializer_from_transp(self)

    def initialize(self,*args,  **kwargs):

        self.initializer(*args,**kwargs)

    def prepare(self, PORTALSparameters = {}, MODELparameters = {}, optimization_options = {}, INITparameters = {}):

        self.folder_portals = f"{self.folder}/run_portals/"
        self.fileGACODE = self.initializer.file_gacode

        self.PORTALSparameters = PORTALSparameters
        self.MODELparameters = MODELparameters
        self.optimization_options = optimization_options
        self.INITparameters = INITparameters

    # --------------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------------

    def check(self, restart = False):
        '''
        Check if output file already exists so that I don't need to run this beat again
        '''

        output_file = f'{self.folder_portals}/Outputs/optimization_object.pkl'

        if os.path.exists(output_file) and (not restart):
            self.portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_portals)
            print('Output file already exists, not running beat', typeMsg = 'i')
            return True
        else:
            return False

    def run(self, **kwargs):

        restart = kwargs.get('restart', False)

        portals_fun  = PORTALSmain.portals(self.folder_portals)

        for key in self.PORTALSparameters:
            portals_fun.PORTALSparameters[key] = self.PORTALSparameters[key]
        for key in self.MODELparameters:
            portals_fun.MODELparameters[key] = self.MODELparameters[key]
        for key in self.optimization_options:
            portals_fun.optimization_options[key] = self.optimization_options[key]
        for key in self.INITparameters:
            portals_fun.INITparameters[key] = self.INITparameters[key]

        portals_fun.prep(self.fileGACODE,self.folder_portals,hardGradientLimits = [0,2])

        self.prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN = restart, askQuestions = False)

        self.prf_bo.run()

    # --------------------------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------------------------
    def plot(self):

        embed()

class portals_initializer_from_transp:

    def __init__(self, beat_instance):
            
        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder}/initializer_transp/'

        os.makedirs(self.folder, exist_ok=True)

    def __call__(self, time_extraction = None):

        # From previous TRANSP (memory)
        #self.cdf =  self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1].c
        
        # From previous TRANSP (read)
        prev_beat = self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1]
        self.cdf = CDFtools.transp_output(f"{prev_beat.folder}/{prev_beat.transp.shot}{prev_beat.transp.runid}.CDF")

        self.p = self.cdf.to_profiles(time_extraction=time_extraction)

        self.file_gacode = f"{self.folder}/input.gacode"
        self.p.writeCurrentStatus(file=self.file_gacode)

    def plot(self):
        self.cdf.plot()


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

# --------------------------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------------------------

@mitim_timer('\t- MAESTRO')
def simple_maestro_workflow(
    folder,
    geometry,
    parameters,
    Tbc_keV,
    nbc_20,
    TGLFsettings = 6,
    DTplasma = True,
    terminal_outputs = False
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
            "BO_iterations": 30,
            "maximum_value": 1e-2, # x100 better residual
            "maximum_value_is_rel": True,
            "optimizers": "botorch"
        }
    }

    # ---------------------------------------------------------
    # beat 1: TRANSP from FreeGS
    # ---------------------------------------------------------

    m.define_beat('transp', initializer='freegs')
    m.initialize(**geometry,**parameters,profiles = profiles)
    m.prepare(transp_namelist = transp_namelist)
    m.run()

    # ---------------------------------------------------------
    # beat 2: PORTALS from TRANSP
    # ---------------------------------------------------------

    m.define_beat('portals', initializer='transp')
    m.initialize()
    m.prepare(**portals_namelist)
    m.run()

    # ---------------------------------------------------------
    # beat 3: TRANSP from PORTALS
    # ---------------------------------------------------------

    m.define_beat('transp', initializer='portals')
    m.initialize(**geometry,**parameters)
    m.prepare(transp_namelist = transp_namelist)
    m.run()

    # ---------------------------------------------------------
    # beat 4: PORTALS from TRANSP
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

# chatGPT 4o (08/18/2024)
def string_to_sequential_5_digit_number(input_string):
    # Split the input string into the base and the numeric suffix
    base_part = input_string[:-1]
    sequence_digit = int(input_string[-1])

    # Create a hash of the base part using SHA-256
    hash_object = hashlib.sha256(base_part.encode())
    
    # Convert the hash to an integer
    hash_int = int(hash_object.hexdigest(), 16)
    
    # Take the hash modulo 10,000 to get a 4-digit number
    four_digit_number = hash_int % 10000
    
    # Combine the 4-digit hash with the sequence digit to get a 5-digit number
    five_digit_number = (four_digit_number * 10) + sequence_digit
    
    # Ensure it's always 5 digits by adding leading zeros if necessary
    return f'{five_digit_number:05d}'
