import os
import copy
import hashlib
import numpy as np
from mitim_tools.gs_tools import GEQtools
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

from mitim_modules.maestro.utils.MAESTRObeat import beat, beat_initializer

class transp_beat(beat):

    def __init__(self, maestro_instance):            
        super().__init__(maestro_instance, beat_name = 'transp')

    # --------------------------------------------------------------------------------------------
    # Checker
    # --------------------------------------------------------------------------------------------
    def check(self, restart = False):
        return super().check(restart=restart, folder_search = self.folder_output, suffix = '.CDF')

    # --------------------------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------------------------
    def define_initializer(self, initializer):

        if initializer is None:
            self.initializer = beat_initializer(self)
        elif initializer == 'freegs':
            self.initializer = transp_initializer_from_freegs(self)
        elif initializer == 'geqdsk':
            self.initializer = transp_initializer_from_geqdsk(self)
        elif initializer == 'portals':
            self.initializer = transp_initializer_from_portals(self)
        elif initializer == 'profiles':
            self.initializer = transp_initializer_from_profiles(self)
        else:
            raise ValueError(f'Initializer "{initializer}" not recognized')

    def initialize(
        self,
        flattop_window      = 0.15,  # To allow for steady-state in heating and current diffusion
        transition_window   = 0.10,  # To prevent equilibrium crashes
        **kwargs_initializer
        ):


        # Define timings
        currentheating_window = 0.001
        self.time_init = 0.0                                                # Start with a TRANSP machine equilibrium
        self.time_transition = self.time_init+ transition_window            # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End

        # Initialize
        self.initializer(**kwargs_initializer)

    def freeze_parameters(self):

        print(f'\t\t- Freezing engineering parameters from MAESTRO: {IOtools.clipstr(self.initializer.folder+"/input.gacode_frozen")}')
        self.maestro_instance.profiles_with_engineering_parameters = self.initializer.p
        self.maestro_instance.profiles_with_engineering_parameters.writeCurrentStatus(file=self.initializer.folder+'/input.gacode_frozen' )

    def retrieve_frozen_parameters_when_skipping(self):

        print(f'\t\t- Retrieving frozen engineering parameters from MAESTRO: {IOtools.clipstr(self.initializer.folder+"/input.gacode_frozen")}')
        self.initializer.p = PROFILEStools.PROFILES_GACODE(self.initializer.folder+'/input.gacode_frozen')

    def prepare(self, letter = None, shot = None, **transp_namelist):
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
        self.initializer.prepare_to_beat()

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

        # Additional operations
        self._additional_operations_add_initialization()
        self._additional_operations_add_engineering_parameters()
        self._additional_operations_add_ICH()

        # Write Ufiles
        self.transp.write_ufiles()

    def _additional_operations_add_initialization(self, machine_initialization = 'CMOD'):
        '''
        ----------------------------------------------------------------------------------------------------------------------
        TRANSP must be initialized with a specific machine, so here I use the trick of changing the equilibrium and parameters
        in time, to make a smooth transition and avoid equilibrium crashes (e.g. quval error)
        ----------------------------------------------------------------------------------------------------------------------
        '''
        self.machine_run = machine_initialization

        if self.machine_run == 'D3D':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 1.67, 0.6, 1.75, 0.38, 0.0, 0.0, 0.074, 1.6, 2.0, 1.0
        elif self.machine_run == 'CMOD':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 0.68, 0.22, 1.5, 0.46, 0.0, 0.0, 0.3, 1.0, 5.4, 1.0
        
        self.transp.populate_time.from_freegs(self.time_init, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = ne0_20)

    def _additional_operations_add_engineering_parameters(self):
        '''
        --------------------------------------------------------------------------------------------
        Throughout MAESTRO, the engineering parameters must not change. This avoids leaks.
        --------------------------------------------------------------------------------------------
        '''

        print('\t\t- Using engineering parameters from MAESTRO')

        p = self.maestro_instance.profiles_with_engineering_parameters

        # Engineering parameters ---------------------------------
        Ip_MA = p.profiles['current(MA)'][0]
        B_T = p.profiles['bcentr(T)'][0]
        Zeff = p.derived['Zeff_vol']
        PichT_MW = p.derived['qRF_MWmiller'][-1]
        R0 = p.profiles['rcentr(m)'][0] #(RZsep.max(axis=0)[0]+RZsep.min(axis=0)[0])/2
        RZsep =  np.array([p.derived['R_surface'][-1,:], p.derived['Z_surface'][-1,:]]).T
        # ---------------------------------------------------------

        for time in self.transp.variables:
            if time > self.time_init:
                if 'CUR' in self.transp.variables[time]:
                    self.transp.add_variable_time(time, None, Ip_MA*1E6, variable='CUR')
                if 'RBZ' in self.transp.variables[time]:
                    self.transp.add_variable_time(time, None, R0*B_T*1E2, variable='RBZ')
                if 'ZF2' in self.transp.variables[time]:
                    z = self.transp.variables[time]['ZF2']['z']/self.transp.variables[time]['ZF2']['z'] * Zeff
                    self.transp.add_variable_time(time, self.transp.variables[time]['ZF2']['x'], z, variable='ZF2')
            
        for time in self.transp.geometry:
            if time > self.time_init:
                self.transp._add_separatrix_time(time, RZsep[:,0], RZsep[:,1])

    def _additional_operations_add_ICH(self, qm_minority = 2/3):

        p = self.maestro_instance.profiles_with_engineering_parameters
        PichT_MW = p.derived['qRF_MWmiller'][-1]
        B_T = p.profiles['bcentr(T)'][0]

        # ICRF on
        Frequency_He3 = B_T * (2*np.pi/qm_minority)
        self.transp.icrf_on_time(self.time_diffusion, power_MW = PichT_MW, freq_MHz = Frequency_He3)
        
    # --------------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------------
    def run(self, **kwargs):

        hours_allocation = 8 # 12

        self.transp.run(self.machine_run, mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*hours_allocation, case=self.transp.runid, checkMin=5)
        self.c = self.transp.c

    # --------------------------------------------------------------------------------------------
    # Finalize and plot
    # --------------------------------------------------------------------------------------------

    def finalize(self):

        # Copy to outputs
        os.system(f'cp {self.folder}/{self.shot}{self.runid}TR.DAT {self.folder_output}/.')
        os.system(f'cp {self.folder}/{self.shot}{self.runid}.CDF {self.folder_output}/.')
        os.system(f'cp {self.folder}/{self.shot}{self.runid}tr.log {self.folder_output}/.')

    def plot(self,  fn = None, counter = 0, **kwargs):

        isitfinished = self.check()

        if isitfinished:
            c = CDFtools.transp_output(self.folder_output)
        else:
            # Trying to see if there's an intermediate CDF in folder
            print('\t\t- Searching for intermediate CDF in folder')
            try:
                c = CDFtools.transp_output(self.folder)
            except ValueError:
                return '\t\t- Cannot plot because the TRANSP beat has not finished yet'

        c.plot(fn = fn, counter = counter) 

        return '\t\t- Plotting of TRANSP beat done'

    # --------------------------------------------------------------------------------------------
    # Finalize in case this is the last beat
    # --------------------------------------------------------------------------------------------

    def finalize_maestro(self):

        cdf = CDFtools.transp_output(f"{self.folder}/{self.transp.shot}{self.transp.runid}.CDF")
        self.maestro_instance.final_p = cdf.to_profiles()
        
        final_file = f'{self.maestro_instance.folder_output}/input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

# --------------------------------------------------------------------------------------------
# Initializer: from PROFILES
# --------------------------------------------------------------------------------------------

class transp_initializer_from_profiles(beat_initializer):
    def __init__(self, beat_instance, label = 'profiles'):
        super().__init__(beat_instance, label = label)

    def __call__(self, profiles_file=None, profiles = {}, Vsurf = 0.0):

        # Load profiles
        self.p = PROFILEStools.PROFILES_GACODE(profiles_file)

        # Vsurf is a quantity that isn't in the profiles, so I add it here
        self.p.Vsurf = Vsurf

        # Insert new profiles
        if 'Te' in profiles:
            self.p.profiles['te(keV)'] = np.interp(self.p.profiles['rho(-)'], profiles['Te'][0], profiles['Te'][1])
        if 'Ti' in profiles:
            self.p.profiles['ti(keV)'][:,0] = np.interp(self.p.profiles['rho(-)'], profiles['Ti'][0], profiles['Ti'][1])
            self.p.makeAllThermalIonsHaveSameTemp()
        if 'ne' in profiles:
            old_density = copy.deepcopy(self.p.profiles['ne(10^19/m^3'])
            self.p.profiles['ne(10^19/m^3'] = np.interp(self.p.profiles['rho(-)'], profiles['ne'][0], profiles['ne'][1]*10.0)
            self.p.profiles['ni(10^19/m^3'] = self.p.profiles['ni(10^19/m^3'] * (self.p.profiles['ne(10^19/m^3']/old_density)

        # Write it to initialization folder
        self.p.writeCurrentStatus(file=self.folder+'/input.gacode' )

    def prepare_to_beat(self):

        # Write TRANSP from profiles
        times = [self.beat_instance.time_transition,self.beat_instance.time_end+1.0]
        self.transp = self.p.to_transp(
            folder = self.beat_instance.folder,
            shot = self.beat_instance.shot, runid = self.beat_instance.runid, times = times,
            Vsurf = self.p.Vsurf)

        # Pass to main class' beat
        self.beat_instance.transp = self.transp

# --------------------------------------------------------------------------------------------
# Initializer: from GEQDSK
# --------------------------------------------------------------------------------------------

class transp_initializer_from_geqdsk(transp_initializer_from_profiles):
    '''
    Idea is to write geqdsk to profile and then call the profiles initializer
    '''
    def __init__(self, beat_instance, label = 'geqdsk'):
        super().__init__(beat_instance, label = label)

    def __call__(
        self,
        geqdsk_file = None,
        PichT_MW = 1.0,
        Zeff = 1.5,
        ne0_20 = 1.0,
        **kwargs_profiles
        ):
        
        # Read geqdsk
        self.f = GEQtools.MITIMgeqdsk(geqdsk_file, fullLCFS=True)

        # Convert to profiles
        p = self.f.to_profiles(ne0_20 = ne0_20, Zeff = Zeff, PichT = PichT_MW)

        # Write it to initialization folder
        p.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Copy original geqdsk for reference use
        os.system(f'cp {geqdsk_file} {self.folder}/input.geqdsk')

        # Call the profiles initializer
        super().__call__(self.folder+'/input.gacode', **kwargs_profiles)

# --------------------------------------------------------------------------------------------
# Initializer: from freegs
# --------------------------------------------------------------------------------------------

class transp_initializer_from_freegs(transp_initializer_from_geqdsk):
    '''
    Idea is to write geqdsk and then call the geqdsk initializer
    '''
    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'freegs')
            
    def __call__(self,
        R,
        a,
        kappa_sep,
        delta_sep,
        zeta_sep,
        z0,
        p0_MPa = 1.0,
        Ip_MA = 1.0,
        B_T = 5.4,
        **kwargs_geqdsk
        ):
        
        # If profiles exist, substitute the pressure and density guesses by something better (not perfect though, no ions)
        if 'ne' in kwargs_geqdsk.get('profiles',{}):
            print('\t- Using ne profile instead of the ne0 guess')
            ne0_20 = kwargs_geqdsk['profiles']['ne'][1][0]
        if 'Te' in kwargs_geqdsk.get('profiles',{}):
            print('\t- Using Te profile for a better estimation of pressure, instead of the p0 guess')
            Te0_keV = kwargs_geqdsk['profiles']['Te'][1][0]
            p0_MPa = 2 * (Te0_keV*1E3) * 1.602176634E-19 * (ne0_20 * 1E20) * 1E-6 #MPa
            
        # Run freegs to generate equilibrium
        f = GEQtools.freegs_millerized(R, a, kappa_sep, delta_sep, zeta_sep, z0)
        f.prep(p0_MPa, Ip_MA, B_T)
        f.solve()
        f.derive()

        # Convert to geqdsk and write it to initialization folder
        f.write(f'{self.folder}/freegs.geqdsk')

        # Call the geqdsk initializer
        super().__call__(geqdsk_file = f'{self.folder}/freegs.geqdsk',**kwargs_geqdsk)

# --------------------------------------------------------------------------------------------
# Initializer: from PORTALS
# --------------------------------------------------------------------------------------------

class transp_initializer_from_portals(transp_initializer_from_profiles):
    '''
    Idea is to find the profiles with best residual and then call the profiles initializer
    '''
    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'portals')

    def __call__(self):

        # Load PORTALS from previous beat: profiles with best residual
        folder =  self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1].folder_output
        self.portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
        self.p = self.portals_output.mitim_runs[self.portals_output.ibest]['powerstate'].profiles

        # Write it to initialization folder
        self.p.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Call the profiles initializer
        super().__call__(self.folder+'/input.gacode')

# --------------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------------

# chatGPT 4o (08/18/2024)
def string_to_sequential_5_digit_number(input_string):
    # Split the input string into the base and the numeric suffix
    base_part = input_string[:-1]
    try:
        sequence_digit = int(input_string[-1])
    except ValueError:
        sequence_digit = 0

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
