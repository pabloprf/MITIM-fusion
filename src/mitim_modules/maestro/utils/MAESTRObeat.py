import os
import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.IOtools import printMsg as print
from scipy.optimize import minimize
from IPython import embed

# --------------------------------------------------------------------------------------------
# Generic beat class with required methods
# --------------------------------------------------------------------------------------------
0
class beat:

    def __init__(self, maestro_instance, beat_name = 'generic', folder_name = None):

        self.maestro_instance = maestro_instance

        if folder_name is None:
            folder_name = f'{self.maestro_instance.folder_beats}/Beat_{self.maestro_instance.counter_current}'
        
        self.folder_beat = f'{folder_name}/'

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
        
        # Call a potential profile creator -----------------------------------------------------------
        if hasattr(self, 'profile_creator'):
            self.profile_creator()
        # --------------------------------------------------------------------------------------------

        # Write it to initialization folder
        self.profiles_current.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Pass the profiles to the beat instance
        self.beat_instance.profiles_current = self.profiles_current

        # Initializer has been called
        self.beat_instance.initialize_called = True

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
        
        beat_num = self.beat_instance.maestro_instance.counter_current-1
        profiles_file = f"{self.beat_instance.maestro_instance.beats[beat_num].folder_output}/input.gacode"

        super().__call__(profiles_file)

# --------------------------------------------------------------------------------------------
# Initializer from GEQDSK: load the geqdsk, convert to profiles and call the profiles initializer
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
        netop_20 = 1.0,
        coeffs_MXH = 7,
        **kwargs_profiles
        ):
        '''
        coeffs_MXH indicated the parameterization used to translate the equilibrium. 
        If too fine, TRANSP might complain about kinks and curvature.
        If too coarse, geometry won't be well represented.
        '''
        
        # Read geqdsk
        self.f = GEQtools.MITIMgeqdsk(geqdsk_file)

        # Potentially save variables
        self._inform_save()

        # Convert to profiles
        p = self.f.to_profiles(ne0_20 = netop_20, Zeff = Zeff, PichT = PichT_MW, coeffs_MXH = coeffs_MXH)

        # Write it to initialization folder
        p.writeCurrentStatus(file=self.folder+'/input.gacode.geqdsk')

        # Copy original geqdsk for reference use
        os.system(f'cp {geqdsk_file} {self.folder}/input.geqdsk')

        # Call the profiles initializer
        super().__call__(self.folder+'/input.gacode.geqdsk', **kwargs_profiles)

    def _inform_save(self):

        self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = self.f.kappa995
        self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = self.f.delta995

        print('\t\t- 0.995 flux surface kappa and delta saved for future beats')

# --------------------------------------------------------------------------------------------
# Initializer from FreeGS: load the equilibrium, convert to geqdsk and call the geqdsk initializer
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

# --------------------------------------------------------------------------------------------
# [Generic] Profile creator: Insert profiles
# --------------------------------------------------------------------------------------------

class creator:
    
        def __init__(self, initialize_instance, profiles_insert = {}, label = 'generic'):
    
            self.initialize_instance = initialize_instance
            self.folder = f'{self.initialize_instance.folder}/creator_{label}/'
    
            if len(label) > 0:
                os.makedirs(self.folder, exist_ok=True)
    
            self.profiles_insert = profiles_insert

        def __call__(self):

            rho, Te, Ti, ne = self.profiles_insert['rho'], self.profiles_insert['Te'], self.profiles_insert['Ti'], self.profiles_insert['ne']
            
            # Update profiles
            self.initialize_instance.profiles_current.changeResolution(rho_new = rho)

            self.initialize_instance.profiles_current.profiles['te(keV)'] = Te

            self.initialize_instance.profiles_current.profiles['ti(keV)'][:,0] = Ti
            self.initialize_instance.profiles_current.makeAllThermalIonsHaveSameTemp()

            old_density = copy.deepcopy(self.initialize_instance.profiles_current.profiles['ne(10^19/m^3)'])
            self.initialize_instance.profiles_current.profiles['ne(10^19/m^3)'] = ne*10.0
            self.initialize_instance.profiles_current.profiles['ni(10^19/m^3)'] = self.initialize_instance.profiles_current.profiles['ni(10^19/m^3)'] * (self.initialize_instance.profiles_current.profiles['ne(10^19/m^3)']/old_density)[:,np.newaxis]

            # Update derived
            self.initialize_instance.profiles_current.deriveQuantities()

# --------------------------------------------------------------------------------------------
# Profile creator from parameterization: Create profiles from a parameterization
# --------------------------------------------------------------------------------------------

class creator_from_parameterization(creator):
    
        def __init__(self, initialize_instance, rhotop = None, Ttop_keV = None, netop_20 = None, Tsep_keV = None, nesep_20 = None, BetaN = None, label = 'parameterization'):
            super().__init__(initialize_instance, label = label)

            self.rhotop = rhotop
            self.Ttop_keV = Ttop_keV
            self.netop_20 = netop_20
            self.BetaN = BetaN
            self.Tsep_keV = Tsep_keV
            self.nesep_20 = nesep_20

        def _return_profile_betan_residual(self, x, x_a, betan):
            aLT, aLn = x
            # returns the residual of the betaN to match the profile to the EPED guess

            rho, Te = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a = x_a)
            rho, Ti = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a = x_a)
            rho, ne = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.netop_20, self.nesep_20, aLn, x_a = x_a)

            # Call the generic creator
            self.profiles_insert = {'rho': rho, 'Te': Te, 'Ti': Ti, 'ne': ne}
            super().__call__()

            return ((self.initialize_instance.profiles_current.derived['BetaN'] - self.BetaN) / self.BetaN) ** 2
    
        def __call__(self):

            # Function to wrap the parameterization ------------------------------------------------------
            aLT = 2.0
            aLn = 0.2
            x_a = 0.3

            x0 = [aLT, aLn]
            bounds = [(1.0,3.0), (0.1, 0.3)] # in the future, fix aLT/aLn ?
            print('\n\t -Optimizing aLT and aLn to match BetaN')
            res = minimize(self._return_profile_betan_residual, x0, args=(x_a, self.BetaN), method='Nelder-Mead', tol=1e-2, bounds=bounds)
            print(f'\n\t - Gradients: aLT = {res.x[0]:.2f}, aLn = {res.x[1]:.2f}')
            aLT, aLn = res.x

            rho, Te = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a=x_a)
            rho, Ti = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a=x_a)
            rho, ne = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.netop_20, self.nesep_20, aLn, x_a=x_a)

            # Call the generic creator
            self.profiles_insert = {'rho': rho, 'Te': Te, 'Ti': Ti, 'ne': ne}
            super().__call__()

            # --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Profile creator from EPED: Create parameterization using EPED
# --------------------------------------------------------------------------------------------

class creator_from_eped(creator_from_parameterization):

    def __init__(self, initialize_instance, parameters = None, label = 'eped'):
        super().__init__(initialize_instance, label = label)

        self.parameters = parameters

    def __call__(self):

        # Create a beat within here
        from mitim_modules.maestro.utils.EPEDbeat import eped_beat
        beat_eped = eped_beat(self.initialize_instance.beat_instance.maestro_instance, folder_name = self.folder)
        beat_eped.prepare(**self.parameters)

        # Work with this profile
        beat_eped.profiles_current = self.initialize_instance.profiles_current
        
        # Run EPED
        eped_results = beat_eped._run()

        # Potentially save variables
        np.save(f'{beat_eped.folder_output}/eped_results.npy', eped_results)
        beat_eped._inform_save(eped_results)

        # Call the profiles creator
        self.rhotop = eped_results['rhotop']
        self.Ttop_keV = eped_results['Ttop_keV']
        self.netop_20 = eped_results['netop_20']        
        self.Tsep_keV = eped_results['Tesep_keV']
        self.nesep_20 = eped_results['nesep_20']
        self.BetaN = beat_eped.BetaN
        super().__call__()

        # Save
        np.save(f'{self.folder}/eped_results.npy', eped_results)
