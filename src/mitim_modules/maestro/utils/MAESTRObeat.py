import os
import copy
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.transp_tools import CDFtools
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

    def check(self, restart = False, folder_search = None, suffix = ''):
        '''
        Check if output file already exists so that I don't need to run this beat again
        '''

        if folder_search is None:
            folder_search = self.folder_output

        output_file = None
        if not restart:
            output_file = IOtools.findFileByExtension(folder_search, suffix, agnostic_to_case=True, provide_full_path=True)

            if output_file is not None:
                print('\t\t- Output file already exists, not running beat', typeMsg = 'i')

        else:
            print('\t\t- Forced restarting of beat', typeMsg = 'i')

        return output_file is not None

    def freeze_parameters(self):
        pass

    def retrieve_frozen_parameters_when_skipping(self):
        pass

    def define_initializer(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def finalize_maestro(self, *args, **kwargs):
        pass

    def grab_output(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        return ''

    def inform_save(self, *args, **kwargs):
        pass

    def _inform(self, *args, **kwargs):
        pass

# --------------------------------------------------------------------------------------------
# Generic initializer classes with required methods
# --------------------------------------------------------------------------------------------

class beat_initializer:
    
    def __init__(self, beat_instance, label = ''):
            
        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder_beat}/initializer_{label}/'

        if len(label) > 0:
            os.makedirs(self.folder, exist_ok=True)

    def __call__(self, *args, **kwargs):
        '''
        The call method should produce a self.beat.profiles_current object with the input.gacode profiles
        '''
        pass

# --------------------------------------------------------------------------------------------
# Initializer from profiles: just load profiles and write them to the initialization folder
# --------------------------------------------------------------------------------------------

class initializer_from_profiles(beat_initializer):
    def __init__(self, beat_instance, label = 'profiles'):
        super().__init__(beat_instance, label = label)

    def __call__(self, profiles_file = None, profiles = {}, Vsurf = 0.0, use_previous_portals_profiles = True,  **kwargs_beat):

        # Load profiles
        self.profiles_current = PROFILEStools.PROFILES_GACODE(profiles_file)

        # Vsurf is a quantity that isn't in the profiles, so I add it here
        self.profiles_current.Vsurf = Vsurf

        # --------------------------------------------------------------------------------------------
        # Insert new profiles
        # --------------------------------------------------------------------------------------------
        if 'Te' in profiles:
            self.profiles_current.profiles['te(keV)'] = np.interp(self.profiles_current.profiles['rho(-)'], profiles['Te'][0], profiles['Te'][1])
        if 'Ti' in profiles:
            self.profiles_current.profiles['ti(keV)'][:,0] = np.interp(self.profiles_current.profiles['rho(-)'], profiles['Ti'][0], profiles['Ti'][1])
            self.profiles_current.makeAllThermalIonsHaveSameTemp()
        if 'ne' in profiles:
            old_density = copy.deepcopy(self.profiles_current.profiles['ne(10^19/m^3)'])
            self.profiles_current.profiles['ne(10^19/m^3)'] = np.interp(self.profiles_current.profiles['rho(-)'], profiles['ne'][0], profiles['ne'][1]*10.0)
            self.profiles_current.profiles['ni(10^19/m^3)'] = self.profiles_current.profiles['ni(10^19/m^3)'] * (self.profiles_current.profiles['ne(10^19/m^3)']/old_density)[:,np.newaxis]

        # --------------------------------------------------------------------------------------------
        # Use previous PORTALS profiles
        # --------------------------------------------------------------------------------------------
        if use_previous_portals_profiles and ('portals_profiles' in self.beat_instance.maestro_instance.parameters_trans_beat):
            print('\t- Using previous PORTALS thermal kinetic profiles instead of the TRANSP profiles')
            p_prev = self.beat_instance.maestro_instance.parameters_trans_beat['portals_profiles']

            self.profiles_current.changeResolution(rho_new=p_prev.profiles['rho(-)'])
            
            self.profiles_current.profiles['te(keV)'] = p_prev.profiles['te(keV)']
            self.profiles_current.profiles['ne(10^19/m^3)'] = p_prev.profiles['ne(10^19/m^3)']
            for i,sp in enumerate(self.profiles_current.Species):
                if sp['S'] == 'therm':
                    self.profiles_current.profiles['ti(keV)'][:,i] = p_prev.profiles['ti(keV)'][:,i]
                    self.profiles_current.profiles['ni(10^19/m^3)'][:,i] = p_prev.profiles['ni(10^19/m^3)'][:,i]

        # Write it to initialization folder
        self.profiles_current.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Pass the profiles to the beat instance
        self.beat_instance.profiles_current = self.profiles_current

# --------------------------------------------------------------------------------------------
# Initializer from PORTALS results: load the profiles with best residual and call the profiles initializer
# --------------------------------------------------------------------------------------------

class initializer_from_portals(initializer_from_profiles):
    '''
    Idea is to find the profiles with best residual and then call the profiles initializer
    '''
    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'portals')

    def __call__(self, **kwargs_profiles):

        # Load PORTALS from previous beat: profiles with best residual
        folder =  self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1].folder_output
        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
        p = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles

        # Write it to initialization folder
        p.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Call the profiles initializer
        super().__call__(self.folder+'/input.gacode', **kwargs_profiles)

# --------------------------------------------------------------------------------------------
# Initializer from TRANSP: load the profiles at a given time and call the profiles initializer
# --------------------------------------------------------------------------------------------

class initializer_from_transp(initializer_from_profiles):

    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'transp')

    def __call__(self, time_extraction = None, **kwargs_profiles):

        # Load TRANSP results from previous beat
        beat_num = self.beat_instance.maestro_instance.counter-1
        cdf = CDFtools.transp_output(self.beat_instance.maestro_instance.beats[beat_num].folder_output)

        # Extract profiles at time_extraction
        time_extraction = cdf.t[cdf.ind_saw -1] # Since the time is coarse in MAESTRO TRANSP runs, make I'm not extracting with profiles sawtoothing
        p = cdf.to_profiles(time_extraction=time_extraction)

        file_gacode = f"{self.folder}/input.gacode"
        p.writeCurrentStatus(file=file_gacode)

        super().__call__(file_gacode, **kwargs_profiles)

# --------------------------------------------------------------------------------------------
# Initializer from GEQDSK: load the geqdsk and call the profiles initializer
# --------------------------------------------------------------------------------------------

class initializer_from_geqdsk(initializer_from_profiles):
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

