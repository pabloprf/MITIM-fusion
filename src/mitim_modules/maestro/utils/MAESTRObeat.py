import os
import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

# --------------------------------------------------------------------------------------------
# Generic beat class with required methods
# --------------------------------------------------------------------------------------------

class beat:

    def __init__(self, maestro_instance, beat_name = 'generic'):

        self.maestro_instance = maestro_instance
        self.folder_beat = f'{self.maestro_instance.folder_beats}/Beat_{self.maestro_instance.counter}/'

        # Where to run it
        self.name = beat_name
        self.folder = f'{self.folder_beat}/run_{self.name}/'
        os.makedirs(self.folder, exist_ok=True)

        # Where to save the results
        self.folder_output = f'{self.folder_beat}/beat_results/'
        os.makedirs(self.folder_output, exist_ok=True)

        self.initialize_called = False

    def define_initializer(self, initializer):

        if initializer is None:
            self.initialize = initializer_from_previous(self)
        elif initializer == 'freegs':
            self.initialize = initializer_from_freegs(self)
        elif initializer == 'geqdsk':
            self.initialize = initializer_from_geqdsk(self)
        elif initializer == 'profiles':
            self.initialize = beat_initializer(self)
        else:
            raise ValueError(f'Initializer "{initializer}" not recognized')

    def prepare(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def merge_parameters(self, profiles_current_is_from_beat = None):
        # self.maestro_instance.profiles_with_engineering_parameters
        # self.profiles_output
        pass

    def _inform_save(self, *args, **kwargs):
        pass

    def _inform(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def finalize_maestro(self, *args, **kwargs):
        pass

    def grab_output(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        return ''

# --------------------------------------------------------------------------------------------
# [Generic] Initializer from profiles: just load profiles and write them to the initialization folder
# --------------------------------------------------------------------------------------------

class beat_initializer:
    def __init__(self, beat_instance, label = 'profiles'):

        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder_beat}/initializer_{label}/'

        if len(label) > 0:
            os.makedirs(self.folder, exist_ok=True)

    def __call__(self, profiles_file = None, profiles = {}, Vsurf = None,  **kwargs_beat):

        # Load profiles
        self.profiles_current = PROFILEStools.PROFILES_GACODE(profiles_file)

        # --------------------------------------------------------------------------------------------
        # Operations
        # --------------------------------------------------------------------------------------------
        
        # Vsurf is a quantity that isn't in the profiles, so I add it here
        if Vsurf is not None:
            # Add if provided
            self.profiles_current.Vsurf = Vsurf
        elif 'Vsurf' not in self.profiles_current.profiles.__dict__:
            # Add default if not there
            self.profiles_current.Vsurf = 0.0
        
        # Insert the provided profiles into the profiles_current object
        self._add_provided_profiles(profiles=profiles)

        # --------------------------------------------------------------------------------------------

        # Write it to initialization folder
        self.profiles_current.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Pass the profiles to the beat instance
        self.beat_instance.profiles_current = self.profiles_current

        # Initializer has been called
        self.beat_instance.initialize_called = True

    def _add_provided_profiles(self, profiles = {}):

        if 'rho' in profiles:
            self.profiles_current.changeResolution(rho_new = profiles['rho'])

        if 'Te' in profiles:
            self.profiles_current.profiles['te(keV)'] = profiles['Te']
        if 'Ti' in profiles:
            self.profiles_current.profiles['ti(keV)'][:,0] = profiles['Ti']
            self.profiles_current.makeAllThermalIonsHaveSameTemp()
        if 'ne' in profiles:
            old_density = copy.deepcopy(self.profiles_current.profiles['ne(10^19/m^3)'])
            self.profiles_current.profiles['ne(10^19/m^3)'] = profiles['ne']*10.0
            self.profiles_current.profiles['ni(10^19/m^3)'] = self.profiles_current.profiles['ni(10^19/m^3)'] * (self.profiles_current.profiles['ne(10^19/m^3)']/old_density)[:,np.newaxis]

# --------------------------------------------------------------------------------------------
# Initializer from previous beat: load the profiles and call the profiles initializer
# --------------------------------------------------------------------------------------------

class initializer_from_previous(beat_initializer):
    
    def __init__(self, beat_instance, label = 'previous_beat'):
        super().__init__(beat_instance, label = label)

    def __call__(self, *args, **kwargs):
        '''
        The call method should produce a self.beat.profiles_current object with the input.gacode profiles
        '''

        print("\t- Initializing profiles from previous beat's result", typeMsg = 'i')
        
        beat_num = self.beat_instance.maestro_instance.counter-1
        profiles_file = f"{self.beat_instance.maestro_instance.beats[beat_num].folder_output}/input.gacode"

        super().__call__(profiles_file)

# --------------------------------------------------------------------------------------------
# Initializer from GEQDSK: load the geqdsk and call the profiles initializer
# --------------------------------------------------------------------------------------------

class initializer_from_geqdsk(beat_initializer):
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
        coeffs_MXH = 7,
        **kwargs_profiles
        ):
        '''
        coeffs_MXH indicated the parameterization used to translate the equilibrium. 
        If too fine, TRANSP might complain about kinks and curvature.
        If too coarse, geometry won't be well represented.
        '''
        
        # Read geqdsk
        f = GEQtools.MITIMgeqdsk(geqdsk_file)

        # Convert to profiles
        p = f.to_profiles(ne0_20 = ne0_20, Zeff = Zeff, PichT = PichT_MW, coeffs_MXH = coeffs_MXH)

        # Write it to initialization folder
        p.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Copy original geqdsk for reference use
        os.system(f'cp {geqdsk_file} {self.folder}/input.geqdsk')

        # Call the profiles initializer
        super().__call__(self.folder+'/input.gacode', **kwargs_profiles)

# --------------------------------------------------------------------------------------------
# Initializer from FreeGS: load the equilibrium and call the geqdsk initializer
# --------------------------------------------------------------------------------------------

class initializer_from_freegs(initializer_from_geqdsk):
    '''
    Idea is to write geqdsk and then call the geqdsk initializer
    '''
    def __init__(self, beat_instance, label = 'freegs'):
        super().__init__(beat_instance, label = label)
            
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

