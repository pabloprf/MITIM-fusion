import shutil
import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.LOGtools import printMsg as print
from scipy.optimize import minimize
from IPython import embed

# --------------------------------------------------------------------------------------------
# Generic beat class with required methods
# --------------------------------------------------------------------------------------------

class beat:

    def __init__(self, maestro_instance, beat_name = 'generic', folder_name = None):

        self.maestro_instance = maestro_instance

        if folder_name is None:
            folder_name = self.maestro_instance.folder_beats / f'Beat_{self.maestro_instance.counter_current}'
        
        self.folder_beat = folder_name

        # Where to run it
        self.name = beat_name
        self.folder = self.folder_beat / f'run_{self.name}'
        self.folder.mkdir(parents=True, exist_ok=True)

        # Where to save the results
        self.folder_output = self.folder_beat / 'beat_results'
        self.folder_output.mkdir(parents=True, exist_ok=True)

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
        self.folder = self.beat_instance.folder_beat / f'initializer_{label}'

        if len(label) > 0:
            self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, profiles_file = None, Vsurf = None,   **kwargs_beat):

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
        self.profiles_current.writeCurrentStatus(file=self.folder / 'input.gacode')

        # Pass the profiles to the beat instance
        self.beat_instance.profiles_current = self.profiles_current

        # Initializer has been called
        self.beat_instance.initialize_called = True

    def _inform_save(self):
        pass

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
        profiles_file = self.beat_instance.maestro_instance.beats[beat_num].folder_output / 'input.gacode'

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
        coeffs_MXH = 5,
        **kwargs_profiles
        ):
        '''
        coeffs_MXH indicated the parameterization used to translate the equilibrium. 
        If too fine, TRANSP might complain about kinks and curvature.
        If too coarse, geometry won't be well represented.
        '''
        
        # Read geqdsk
        self.f = GEQtools.MITIMgeqdsk(geqdsk_file)

        # Convert to profiles
        print(f'\t- Converting geqdsk to profiles, using {coeffs_MXH = }')
        p = self.f.to_profiles(ne0_20 = netop_20, Zeff = Zeff, PichT = PichT_MW, coeffs_MXH = coeffs_MXH)

        # Sometimes I may want to change Ip and Bt
        if 'Ip_MA' in kwargs_profiles and kwargs_profiles['Ip_MA'] is not None:
            Ip_in_geqdsk = p.profiles['current(MA)'][0]
            if Ip_in_geqdsk != kwargs_profiles['Ip_MA']:
                print(f'\t- Requested to ignore geqdsk current and use user-specified one, changing Ip from {Ip_in_geqdsk} to {kwargs_profiles["Ip_MA"]}', typeMsg = 'w')
                p.profiles['current(MA)'][0] = kwargs_profiles['Ip_MA']
                print(f'\t\t* Scaling poloidal flux by same factor as Ip, {kwargs_profiles["Ip_MA"] / Ip_in_geqdsk:.2f}')
                p.profiles['polflux(Wb/radian)'] *= kwargs_profiles['Ip_MA'] / Ip_in_geqdsk
                print(f'\t\t* Scaling q-profile by same factor as Ip, {kwargs_profiles["Ip_MA"] / Ip_in_geqdsk:.2f}')
                p.profiles['q(-)'] *= 1/(kwargs_profiles['Ip_MA'] / Ip_in_geqdsk)

        if 'B_T' in kwargs_profiles and kwargs_profiles['B_T'] is not None:
            Bt_in_geqdsk = p.profiles['bcentr(T)'][0]
            if Bt_in_geqdsk != kwargs_profiles['B_T']:
                print(f'\t- Requested to ignore geqdsk B and use user-specified one, changing Bt from {Bt_in_geqdsk} to {kwargs_profiles["B_T"]}', typeMsg = 'w')
                p.profiles['bcentr(T)'][0] = kwargs_profiles['B_T']
                print(f'\t\t* Scaling toroidal flux by same factor as Bt, {kwargs_profiles["B_T"] / Bt_in_geqdsk:.2f}')
                p.profiles['torfluxa(Wb/radian)'] *= kwargs_profiles['B_T'] / Bt_in_geqdsk
                print(f'\t\t* Scaling q-profile by same factor as Bt, {kwargs_profiles["B_T"] / Bt_in_geqdsk:.2f}')
                p.profiles['q(-)'] *= kwargs_profiles['B_T'] / Bt_in_geqdsk

        # Write it to initialization folder
        p.writeCurrentStatus(file=self.folder / 'input.geqdsk.gacode')

        # Copy original geqdsk for reference use
        shutil.copy2(geqdsk_file, self.folder / "input.geqdsk")

        # Save parameters also here in case they are needed already at this beat (e.g. for EPED)
        self._inform_save()

        # Call the profiles initializer
        super().__call__(self.folder / 'input.geqdsk.gacode', **kwargs_profiles)

    def _inform_save(self):

        f = GEQtools.MITIMgeqdsk(self.folder / 'input.geqdsk')

        self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.kappa995
        self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.delta995

        print('\t\t- 0.995 flux surface kappa and delta saved for future beats -> ', f.kappa995, f.delta995)

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
        if ('ne' in kwargs_geqdsk.get('profiles_insert',{})) and ('Te' in kwargs_geqdsk.get('profiles_insert',{})):
            print('\t- Using ne profile instead of the ne0 guess')
            ne0_20 = kwargs_geqdsk['profiles_insert']['ne'][1][0]
            print('\t- Using Te profile for a better estimation of pressure, instead of the p0 guess')
            Te0_keV = kwargs_geqdsk['profiles_insert']['Te'][1][0]
            p0_MPa = 2 * (Te0_keV*1E3) * 1.602176634E-19 * (ne0_20 * 1E20) * 1E-6 #MPa
        # If betaN provided, use it to estimate the pressure
        elif 'BetaN' in kwargs_geqdsk:
            print('\t- Using BetaN for a better estimation of pressure, instead of the p0 guess')
            pvol_MPa = ( Ip_MA / (a * B_T) ) * (B_T ** 2 / (2 * 4 * np.pi * 1e-7)) / 1e6 * kwargs_geqdsk['BetaN'] * 1E-2
            p0_MPa = pvol_MPa * 3.0


        # Run freegs to generate equilibrium
        f = GEQtools.freegs_millerized(R, a, kappa_sep, delta_sep, zeta_sep, z0)
        f.prep(p0_MPa, Ip_MA, B_T)
        f.solve()
        f.derive()

        # Convert to geqdsk and write it to initialization folder
        f.write(self.folder / 'freegs.geqdsk')

        # Call the geqdsk initializer
        super().__call__(geqdsk_file = self.folder / 'freegs.geqdsk',**kwargs_geqdsk)

# --------------------------------------------------------------------------------------------
# [Generic] Profile creator: Insert profiles
# --------------------------------------------------------------------------------------------

class creator:
    
        def __init__(self, initialize_instance, profiles_insert = {}, label = 'generic'):
    
            self.initialize_instance = initialize_instance
            self.folder = self.initialize_instance.folder / f'creator_{label}'
    
            if len(label) > 0:
                self.folder.mkdir(parents=True, exist_ok=True)
    
            self.profiles_insert = profiles_insert

        def __call__(self):

            if 'roa' in self.profiles_insert:
                if 'rho' in self.profiles_insert:
                    print('\t- Both r/a and rho provided to insert profiles, using roa',typeMsg = 'w')
                self.profiles_insert['rho'] = np.interp(self.profiles_insert['roa'], self.initialize_instance.profiles_current.derived['roa'], self.initialize_instance.profiles_current.profiles['rho(-)'])
            if 'psin' in self.profiles_insert:
                if 'rho' in self.profiles_insert:
                    print('\t- Both psin and rho provided to insert profiles, using psin',typeMsg = 'w')
                self.profiles_insert['rho'] = np.interp(self.profiles_insert['psin'], self.initialize_instance.profiles_current.derived['psi_pol_n'], self.initialize_instance.profiles_current.profiles['rho(-)'])

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
            self.initialize_instance.profiles_current.derive_quantities()

        def _inform_save(self, **kwargs):
            pass

# --------------------------------------------------------------------------------------------
# Profile creator from parameterization: Create profiles from a parameterization
# --------------------------------------------------------------------------------------------

class creator_from_parameterization(creator):
    
        def __init__(
            self,
            initialize_instance,
            rhotop = None,
            Ttop_keV = None,
            netop_20 = None,
            Tsep_keV = None,
            nesep_20 = None,
            BetaN = None,
            nu_ne = None,
            aLn = None,
            aLT = None,
            label = 'parameterization',
            nresol = 501
            ):
            super().__init__(initialize_instance, label = label)

            self.rhotop = rhotop
            self.Ttop_keV = Ttop_keV
            self.netop_20 = netop_20
            self.Tsep_keV = Tsep_keV
            self.nesep_20 = nesep_20

            # Initialization parameters
            self.BetaN = BetaN
            self.nu_ne = nu_ne

            self.aLn_guess = aLn
            self.aLT_guess = aLT

            self.nresol = nresol

        def _return_profile_peaking_residual(self, aLn, x_a):

            # returns the residual of the betaN to match the profile to the EPED guess

            rho, ne = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.netop_20, self.nesep_20, aLn, x_a = x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'rho': rho, 'Te': ne, 'Ti': ne, 'ne': ne}
            super().__call__()

            return ((self.initialize_instance.profiles_current.derived['ne_peaking0.2'] - self.nu_ne) / self.nu_ne) ** 2

        def _return_profile_betan_residual(self, aLT, x_a, aLn):

            # returns the residual of the betaN to match the profile to the EPED guess

            rho, Te = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a = x_a,nx = self.nresol)
            rho, Ti = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a = x_a,nx = self.nresol)
            rho, ne = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.netop_20, self.nesep_20, aLn, x_a = x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'rho': rho, 'Te': Te, 'Ti': Ti, 'ne': ne}
            super().__call__()

            return ((self.initialize_instance.profiles_current.derived['BetaN_engineering'] - self.BetaN) / self.BetaN) ** 2
    
        def __call__(self):

            x_a = 0.3

            if (self.aLn_guess is not None) or (self.nu_ne is None):
                aLn = self.aLn_guess if self.aLn_guess is not None else 0.2
                print(f'\n\t - Using aLn = {aLn}')
            else:
                aLn_guess = 0.2
                # Find the density gradient that matches the peaking
                print('\n\t -Optimizing aLn to match ne peaking')
                bounds = [(0.0,3.0)]
                res = minimize(self._return_profile_peaking_residual, [aLn_guess], args=(x_a), method='Nelder-Mead', tol=1e-3, bounds=bounds)
                aLn = res.x[0]
                print(f'\n\t - Gradient: aLn = {aLn:.2f}')
                print(f'\t - ne peaking: {self.initialize_instance.profiles_current.derived["ne_peaking0.2"]:.5f} (target: {self.nu_ne:.5f})')

            # Find the temperature gradient that matches the BetaN
            if (self.aLT_guess is not None) or (self.BetaN is None):
                aLT = self.aLT_guess if self.aLT_guess is not None else 2.0
                print(f'\n\t - Using aLT = {aLT}')
            else:
                aLT_guess = 2.0
                # Find the temperature gradient that matches the BetaN
                print('\n\t -Optimizing aLT to match BetaN')
                bounds = [(0.5,3.0)]
                res = minimize(self._return_profile_betan_residual, [aLT_guess], args=(x_a, aLn), method='Nelder-Mead', tol=1e-3, bounds=bounds)
                aLT = res.x[0]
                print(f'\n\t - Gradient: aLT = {aLT:.2f}')
                print(f'\t - BetaN: {self.initialize_instance.profiles_current.derived["BetaN_engineering"]:.5f} (target: {self.BetaN:.5f})')

            # Create profiles

            rho, Te = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a=x_a,nx = self.nresol)
            rho, Ti = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.Ttop_keV, self.Tsep_keV, aLT, x_a=x_a,nx = self.nresol)
            rho, ne = FunctionalForms.MITIMfunctional_aLyTanh(self.rhotop, self.netop_20, self.nesep_20, aLn, x_a=x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'rho': rho, 'Te': Te, 'Ti': Ti, 'ne': ne}
            super().__call__()


# --------------------------------------------------------------------------------------------
# Profile creator from EPED: Create parameterization using EPED
# --------------------------------------------------------------------------------------------

class creator_from_eped(creator_from_parameterization):

    def __init__(
        self,
        initialize_instance,
        label = 'eped',
        BetaN = None,
        nu_ne = None,
        aLT = None,
        aLn = None,
        nresol = 501,
        **kwargs_eped
        ):
        super().__init__(initialize_instance, label = label)

        self.BetaN = BetaN
        self.nu_ne = nu_ne
        self.aLT_guess = aLT
        self.aLn_guess = aLn
        self.parameters = kwargs_eped
        self.nresol = nresol
        if self.BetaN is None:
            raise ValueError('[MITIM] BetaN must be provided in the current implementation of EPED creator')

    def __call__(self):

        # Create a beat within here
        from mitim_modules.maestro.utils.EPEDbeat import eped_beat
        self.beat_eped = eped_beat(self.initialize_instance.beat_instance.maestro_instance, folder_name = self.folder)
        self.beat_eped.prepare(BetaN = self.BetaN, **self.parameters)

        # Work with this profile
        self.beat_eped.profiles_current = self.initialize_instance.profiles_current
        
        # Run EPED
        nproc_per_run = 64 #TODO: make it a parameter to be received from MAESTRO namelist
        eped_results = self.beat_eped._run(loopBetaN = 1, nproc_per_run=nproc_per_run, cold_start=True) # Assume always cold start for a creator

        # Potentially save variables
        np.save(self.beat_eped.folder_output / 'eped_results.npy', eped_results)
        self._inform_save(eped_results)

        # Call the profiles creator
        self.rhotop = eped_results['rhotop']
        self.Ttop_keV = eped_results['Tetop_keV']
        self.netop_20 = eped_results['netop_20']        
        self.Tsep_keV = eped_results['Tesep_keV']
        self.nesep_20 = eped_results['nesep_20']
        self.BetaN = self.beat_eped.BetaN
        super().__call__()

        # Save
        np.save(self.folder / 'eped_results.npy', eped_results)

    def _inform_save(self, eped_results = None):

        from mitim_modules.maestro.utils.EPEDbeat import eped_beat
        beat_eped_for_save = eped_beat(self.initialize_instance.beat_instance.maestro_instance, folder_name = self.folder)

        if eped_results is None:
            eped_results =  np.load(beat_eped_for_save.folder_output / 'eped_results.npy', allow_pickle=True).item()

        beat_eped_for_save._inform_save(eped_results)