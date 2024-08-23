import os
import copy
import hashlib
import numpy as np
from mitim_tools.gs_tools import FREEGStools, GEQtools
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

from mitim_modules.maestro.utils.MAESTRObeat import beat, beat_initializer

# Generic initializer

class transp_initializer(beat_initializer):

    def __init__(self, beat_instance, label = ''):
        super().__init__(beat_instance, label = label)

    def prepare_to_beat(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def _add_profiles(self, times):
        pass

# --------------------------------------------------------------------------------------------
# Beat: TRANSP
# --------------------------------------------------------------------------------------------

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
        else:
            raise ValueError(f'Initializer "{initializer}" not recognized')

    def initialize(
        self,
        flattop_window      = 0.15,  # To allow for steady-state in heating and current diffusion
        transition_window   = 0.10,  # To prevent equilibrium crashes
        freeze_parameters = False, 
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

        # Freeze parameters
        if freeze_parameters:
            self._freeze_parameters()

    def _freeze_parameters(self):

        quantities = {
            'Ip_MA': self.initializer.Ip_MA,
            'B_T': self.initializer.B_T,
            'Zeff': self.initializer.Zeff,
            'PichT_MW': self.initializer.PichT_MW,
            'RZsep': self.initializer.RZsep
        }

        # Freeze parameters to maestro class to avoid leaks when going from beat to beat
        self.maestro_instance.freeze_parameters(*quantities.values())

        # Save numpy arrays to folder for easy reading
        np.savez(f'{self.initializer.folder}/frozen_quantities.npz', **quantities)

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
        self._additional_operations()

        # Write Ufiles
        self.transp.write_ufiles()

    def _additional_operations(self, machine_initialization = 'CMOD'):

        '''
        ----------------------------------------------------------------------------------------------------------------------
        TRANSP must be initialized with a specific machine, so here I use the trick of changing the equilibrium and parameters
        in time, to make a smooth transition and avoid equilibrium crashes (e.g. quval error)
        ----------------------------------------------------------------------------------------------------------------------
        '''
        self.machine_run = machine_initialization

        if self.machine_run == 'D3D':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T = 1.67, 0.6, 1.75, 0.38, 0.0, 0.0, 0.074, 1.6, 2.0
        elif self.machine_run == 'CMOD':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T = 0.68, 0.22, 1.5, 0.46, 0.0, 0.0, 0.3, 1.0, 5.4
        
        self.transp.populate_time.from_freegs(self.time_init, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T)
        
        '''
        --------------------------------------------------------------------------------------------
        Throughout MAESTRO, the engineering parameters must not change. This avoids leaks.
        --------------------------------------------------------------------------------------------
        '''
        if self.maestro_instance.engineering_parameters is not None:
            print('\t\t- Using engineering parameters from MAESTRO')

            Ip_MA = self.maestro_instance.engineering_parameters['Ip_MA']
            B_T = self.maestro_instance.engineering_parameters['B_T']
            Zeff = self.maestro_instance.engineering_parameters['Zeff']
            PichT_MW = self.maestro_instance.engineering_parameters['PichT_MW']
            RZsep = self.maestro_instance.engineering_parameters['RZsep']
            R0 = (RZsep.max(axis=0)[0]+RZsep.min(axis=0)[0])/2

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

        else:
            # Get from initializer for the operations below
            PichT_MW = self.initializer.PichT_MW
            B_T = self.initializer.B_T

        # ICRF on
        qm_He3 = 2/3
        Frequency_He3 = B_T * (2*np.pi/qm_He3)
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
# Initializers
# --------------------------------------------------------------------------------------------

class transp_initializer_from_eq(transp_initializer):
    def __init__(self, beat_instance, label = ''):
        super().__init__(beat_instance, label = label)
            
    def __call__(
        self,
        Ip_MA = 1.0,
        B_T = 5.4,
        Zeff = 1.5,
        PichT_MW = 10.0,
        p0_MPa = 1.0,
        ne0_20 = 1.0,
        profiles = {},
        **kwargs_f):

        self.p0_MPa, self.Ip_MA, self.B_T = p0_MPa, Ip_MA, B_T
        self.ne0_20, self.Zeff, self.PichT_MW = ne0_20, Zeff, PichT_MW
        self.profiles = profiles

        # If profiles exist, substitute the pressure and density guesses by something better (not perfect though, no ions)
        if 'ne' in profiles:
            print('\t- Using ne profile instead of the ne0 guess')
            self.ne0_20 = profiles['ne'][1][0]
        if 'Te' in profiles:
            print('\t- Using Te profile for a better estimation of pressure, instead of the p0 guess')
            Te0_keV = profiles['Te'][1][0]
            self.p0_MPa = 2 * (Te0_keV*1E3) * 1.602176634E-19 * (self.ne0_20 * 1E20) * 1E-6 #MPa
            
        # FreeGS or other, that produces f that has a separatrix, engineering parameters and a to_transp()
        self._produce_f(**kwargs_f)

    def _produce_f(self, **kwargs):
        pass 

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
        # Add profiles
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._add_profiles(times)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pass to main class' beat
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.beat_instance.transp = self.transp

    def _add_profiles(self, times):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add profiles  # TO FIX ROA VA RHO???, psi, etc
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

class transp_initializer_from_freegs(transp_initializer_from_eq):

    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'freegs')
            
    def _produce_f(self,
        R,
        a,
        kappa_sep,
        delta_sep,
        zeta_sep,
        z0):

        # Load parameters
        self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0 = R, a, kappa_sep, delta_sep, zeta_sep, z0

        # FreeGS
        self.f = FREEGStools.freegs_millerized(self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0)
        self.f.prep(self.p0_MPa, self.Ip_MA, self.B_T)
        self.f.solve()
        self.f.derive()
        self.f.write(f'{self.folder}/freegs.geqdsk')

        self.RZsep = self.f.eq.separatrix(npoints= 100)

class transp_initializer_from_geqdsk(transp_initializer_from_eq):
    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'geqdsk')

    def _produce_f(self, geqdsk_file = None):
        
        # Generic geqdsk
        self.f = FREEGStools.geqdsk_reader(geqdsk_file)

        # Load parameters
        self.R = self.f.g.Rmajor
        self.a = self.f.g.a
        self.kappa_sep = self.f.g.kappa
        self.delta_sep = self.f.g.delta
        self.zeta_sep = self.f.g.zeta
        self.z0 = self.f.g.Zmajor

        os.system(f'cp {geqdsk_file} {self.folder}/geqdsk')

        # Separatrix
        self.RZsep = np.array([self.f.g.Rb_prf,self.f.g.Yb_prf]).T
        
class transp_initializer_from_portals(transp_initializer):

    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'portals')

    def __call__(self):

        # Load PORTALS from previous beat: profiles with best residual
        folder =  self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1].folder_output

        self.portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
        
        self.p = self.portals_output.mitim_runs[self.portals_output.ibest]['powerstate'].profiles

        self.p.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Parameters needed for later
        self.PichT_MW = self.p.derived['qRF_MWmiller'][-1]
        self.B_T = self.p.profiles['bcentr(T)'][0]

    def prepare_to_beat(self):

        times = [self.beat_instance.time_transition,self.beat_instance.time_end+1.0]

        self.transp = self.p.to_transp(
            folder = self.beat_instance.folder,
            shot = self.beat_instance.shot, runid = self.beat_instance.runid, times = times)

        # Pass to main class' beat
        self.beat_instance.transp = self.transp


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
