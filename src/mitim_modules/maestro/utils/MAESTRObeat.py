import shutil
import copy
import csv
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.LOGtools import printMsg as print
from pyro import factor
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
        
        self.cold_start = False

    def restart(self):
        '''
        If the restart has been called (e.g. cold_start=True for this beat), empty the run and results folders
        This is to avoid conflicting information and files
        '''

        if self.folder.exists() or self.folder_output.exists():
            print('\t- Restarting beat: clearing run and output folders', typeMsg = 'i')

            if self.folder.exists():
                shutil.rmtree(self.folder, ignore_errors=True)
                self.folder.mkdir(parents=True, exist_ok=True)
                
            if self.folder_output.exists():
                shutil.rmtree(self.folder_output, ignore_errors=True)   
                self.folder_output.mkdir(parents=True, exist_ok=True)

    def define_initializer(self, initializer):

        if initializer is None:
            self.initialize = initializer_from_previous(self)
        elif initializer == 'freegs':
            self.initialize = initializer_from_freegs(self)
        elif initializer == 'fibe':
            self.initialize = initializer_from_fibe(self)
        elif initializer == 'geqdsk':
            self.initialize = initializer_from_geqdsk(self)
        elif initializer == 'separatrix':
            self.initialize = initializer_from_separatrix(self)
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

    def _minimal_call(self, *args, **kwargs):
        '''
        This function should be used to pre-define parameters before calling the main __call__
        because if I'm skipping some execution upon restart, I still may want some variables
        '''
        pass

    def __call__(self, profiles_file = None, Vsurf = None,   **kwargs_beat):

        # Load profiles
        self.profiles_current = PROFILEStools.gacode_state(profiles_file)

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
        self.profiles_current.write_state(file=self.folder / 'input.gacode')

        # Pass the profiles to the beat instance
        self.beat_instance.profiles_current = self.profiles_current

        # Initializer has been called
        self.beat_instance.initialize_called = True

    def _inform_save(self):
        pass

    # Useful for some child classes
    def _produce_p0guess(self, kwargs_geqdsk, Ip_MA = 1.0, a = 0.5, B_T = 5.4):
        
        # If profiles exist, substitute the pressure and density guesses by something better (not perfect though, no ions)
        if ('ne' in kwargs_geqdsk.get('profiles_insert',{})) and ('Te' in kwargs_geqdsk.get('profiles_insert',{})):
            print('\t- Using ne profile instead of the ne0 guess')
            ne0_20 = kwargs_geqdsk['profiles_insert']['ne'][0]
            print('\t- Using Te profile for a better estimation of pressure, instead of the p0 guess')
            Te0_keV = kwargs_geqdsk['profiles_insert']['Te'][0]
            p0_MPa = 2 * (Te0_keV*1E3) * 1.602176634E-19 * (ne0_20 * 1E20) * 1E-6 #MPa
        # If betaN provided, use it to estimate the pressure
        elif 'BetaN' in kwargs_geqdsk:
            print('\t- Using BetaN for a better estimation of pressure, instead of the p0 guess')
            pvol_MPa = ( Ip_MA / (a * B_T) ) * (B_T ** 2 / (2 * 4 * np.pi * 1e-7)) / 1e6 * kwargs_geqdsk['BetaN'] * 1E-2
            p0_MPa = pvol_MPa * 3.0
            
        return p0_MPa
            
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

    def _minimal_call(self, *args, **kwargs):
     
        if 'extract_995_from' in kwargs:
            self.extract_995_from = kwargs['extract_995_from']

    def __call__(
        self,
        geqdsk_file = None,
        PichT_MW = 1.0,
        Zeff = 1.5,
        netop_20 = 1.0,
        coeffs_MXH = 5,
        extract_995_from='geo',
        **kwargs_profiles
        ):
        '''
        coeffs_MXH indicated the parameterization used to translate the equilibrium. 
        If too fine, TRANSP might complain about kinks and curvature.
        If too coarse, geometry won't be well represented.
        '''
        
        # Read geqdsk
        self.f = GEQtools.MITIMgeqdsk(geqdsk_file)
        
        self._minimal_call(extract_995_from=extract_995_from)

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
                p.profiles['q(-)'] = PLASMAtools.q_profile_scale(p.derived['psi_pol_n'], p.profiles['q(-)'], 1/(kwargs_profiles['Ip_MA'] / Ip_in_geqdsk) )

        if 'B_T' in kwargs_profiles and kwargs_profiles['B_T'] is not None:
            Bt_in_geqdsk = p.profiles['bcentr(T)'][0]
            if Bt_in_geqdsk != kwargs_profiles['B_T']:
                print(f'\t- Requested to ignore geqdsk B and use user-specified one, changing Bt from {Bt_in_geqdsk} to {kwargs_profiles["B_T"]}', typeMsg = 'w')
                p.profiles['bcentr(T)'][0] = kwargs_profiles['B_T']
                print(f'\t\t* Scaling toroidal flux by same factor as Bt, {kwargs_profiles["B_T"] / Bt_in_geqdsk:.2f}')
                p.profiles['torfluxa(Wb/radian)'] *= kwargs_profiles['B_T'] / Bt_in_geqdsk
                print(f'\t\t* Scaling q-profile by same factor as Bt, {kwargs_profiles["B_T"] / Bt_in_geqdsk:.2f}')
                p.profiles['q(-)'] = PLASMAtools.q_profile_scale(p.derived['psi_pol_n'], p.profiles['q(-)'], kwargs_profiles['B_T'] / Bt_in_geqdsk)

        # Write it to initialization folder
        p.write_state(file=self.folder / 'input.geqdsk.gacode')

        # Copy original geqdsk for reference use
        shutil.copy2(geqdsk_file, self.folder / "input.geqdsk")

        # Save parameters also here in case they are needed already at this beat (e.g. for EPED)
        self._inform_save()

        # Call the profiles initializer
        kwargs_profiles["profiles_file"] = self.folder / 'input.geqdsk.gacode'
        super().__call__(**kwargs_profiles)

    def _inform_save(self):
        
        if self.extract_995_from is None:
            return
        
        f = GEQtools.MITIMgeqdsk(self.folder / 'input.geqdsk')

        if self.extract_995_from == 'geo':
            print('\t- Extracting 0.995 flux surface parameters from "geo"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["geo"]["kappa_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["geo"]["delta_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["turnbull"]["zeta_995"] #TODO
        elif self.extract_995_from == 'turnbull':
            print('\t- Extracting 0.995 flux surface parameters from "turnbull"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["turnbull"]["kappa_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["turnbull"]["delta_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["turnbull"]["zeta_995"]
        elif self.extract_995_from == 'mxh':
            print('\t- Extracting 0.995 flux surface parameters from "mxh"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["mxh"]["kappa_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["mxh"]["delta_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["mxh"]["zeta_995"]
            self.beat_instance.maestro_instance.parameters_trans_beat['s_three995'] = f.geometric_parameters["mxh"]["shape_sin_995"][2]
            self.beat_instance.maestro_instance.parameters_trans_beat['s_four995'] = f.geometric_parameters["mxh"]["shape_sin_995"][3]

        print('\t\t- 0.995 flux surface kappa, delta, and zeta saved for future beats -> ', 
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'], 
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'],   
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] )
        if self.extract_995_from == 'mxh':
            print('\t\t- 0.995 flux surface s_three and s_four saved for future beats -> ', 
                self.beat_instance.maestro_instance.parameters_trans_beat['s_three995'],
                self.beat_instance.maestro_instance.parameters_trans_beat['s_four995'] )

# --------------------------------------------------------------------------------------------
# Initializer from separatrix + guesses: convert to profiles and call the profiles initializer
# --------------------------------------------------------------------------------------------

class initializer_from_separatrix(beat_initializer):
    '''
    Idea is to write geqdsk to profile and then call the profiles initializer
    '''
    def __init__(self, beat_instance, label = 'separatrix'):
        super().__init__(beat_instance, label = label)

    def _minimal_call(self, *args, **kwargs):
     
        if 'extract_995_from' in kwargs:
            self.extract_995_from = kwargs['extract_995_from']

    def __call__(
        self,
        PichT_MW = 1.0,
        Zeff = 1.5,
        netop_20 = 1.0,
        coeffs_MXH = 5,
        extract_995_from='geo',
        **kwargs
        ):
        
        self._minimal_call(extract_995_from=extract_995_from)
        
        if 'rz_boundary_file' in kwargs and kwargs['rz_boundary_file'] is not None:
            print('\t- Using rz_boundary_file to define the boundary parameters', typeMsg = 'i')
            boundary_parameters = {'file': kwargs['rz_boundary_file'], 'B_T': kwargs['B_T'], 'Ip_MA': kwargs['Ip_MA'], 'coeffs_MXH': coeffs_MXH}
            separatrix_parameters = None
        else:
            print('\t- Using separatrix parameters to define the equilibrium', typeMsg = 'i')
            boundary_parameters = None
            separatrix_parameters = {
                'BT': kwargs['B_T'],
                'Ip': kwargs['Ip_MA'],
                'R0': kwargs['R'],
                'a': kwargs['a'],
                'z0': 0.0,
                'kappa_sep': kwargs['kappa_sep'],
                'delta_sep': kwargs['delta_sep'],
                'zeta_sep': kwargs['zeta_sep']
            }
            
        internal_flux_file = kwargs["internal_flux_file"] if "internal_flux_file" in kwargs else None
            
        self.extract_995_from = extract_995_from
            
        # From separatrix parameters to guess of profiles
        B0, Ip, R0, rho, rmin, rmaj, z0, kappa, delta, zeta, sn, cn, torfluxa, psi, q, pressure = separatrix_to_equilibrium(
            boundary_parameters=boundary_parameters,
            separatrix_parameters=separatrix_parameters,
            internal_flux_file=internal_flux_file,
            )
        
        # Write to profiles
        self.p = GEQtools.equilibrium_to_profiles(
            rho, psi, q, pressure, torfluxa, R0, B0, Ip,
            kappa, delta, zeta, rmin, rmaj, z0, sn[:,:coeffs_MXH], cn[:,:coeffs_MXH],
            ne0_20 = netop_20,
            Zeff = Zeff,
            Z = 9,
            PichT = PichT_MW
        )
        
        # [Optional] Use the freegs to correct the profiles (keeping the shaping)
        try:
            self._correct_profiles_withfreegs(PichT_MW = PichT_MW, Zeff = Zeff, netop_20 = netop_20, coeffs_MXH = coeffs_MXH, **kwargs)
        except:
            print('\t- Could not run freegs to correct the profiles, proceeding with uncorrected ones', typeMsg = 'w')
        
        # Write it to initialization folder
        self.p.write_state(file=self.folder / 'input.separatrix.gacode')

        # Save parameters also here in case they are needed already at this beat (e.g. for EPED)
        self._inform_save()

        # Call the profiles initializer
        kwargs["profiles_file"] = self.folder / 'input.separatrix.gacode'
        super().__call__(**kwargs)

    def _correct_profiles_withfreegs(self,
            PichT_MW = 1.0, Zeff = 1.5, netop_20 = 1.0, coeffs_MXH = 5,
            Ip_MA = 1.0, a = 0.5, B_T = 5.4, R = 1.5, kappa_sep = 1.7, delta_sep = 0.3, zeta_sep = 0.0, z0 = 0.0, **kwargs):
        '''
        This runs freegs to copy all but the shapings
        '''
        
        p0_MPa = self._produce_p0guess(kwargs, Ip_MA, a, B_T)

        # Run freegs to generate equilibrium
        f = GEQtools.freegs_millerized(R, a, kappa_sep, delta_sep, zeta_sep if zeta_sep is not None else 0.0, z0)
        f.prep(p0_MPa, Ip_MA, B_T)
        f.solve()
        f.derive()
        
        f.write(self.folder / 'freegs.geqdsk.helper')

        f = GEQtools.MITIMgeqdsk(self.folder / 'freegs.geqdsk.helper')
        
        # Use old shaping
        
        p_old = copy.deepcopy(self.p)
        self.p = f.to_profiles(ne0_20 = netop_20, Zeff = Zeff, PichT = PichT_MW, coeffs_MXH = coeffs_MXH)

        for i in ['kappa(-)', 'delta(-)', 'zeta(-)', 'rmin(m)', 'rmaj(m)', 'zmag(m)']:
            self.p.profiles[i] = np.interp(self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles[i])
        
        for i in ['rcentr(m)']:
            self.p.profiles[i] = p_old.profiles[i]
        
        for i in range(coeffs_MXH):
            self.p.profiles[f'shape_cos{i}(-)'] = np.interp(self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles[f'shape_cos{i}(-)'])
        for i in range(coeffs_MXH-3):
            self.p.profiles[f'shape_sin{i+3}(-)'] = np.interp(self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles[f'shape_sin{i+3}(-)'])
        
    def _inform_save(self):
        
        if self.extract_995_from is None:
            return
        
        if "p" not in dir(self):
            self.p = PROFILEStools.gacode_state(self.folder / 'input.separatrix.gacode')
        
        kappa995, delta995, zeta995 = self.p.derived["kappa995"], self.p.derived["delta995"], self.p.derived["zeta995"]

        self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = kappa995
        self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = delta995
        self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = zeta995

        print('\t\t- 0.995 flux surface kappa, delta, and zeta saved for future beats -> ', kappa995, delta995, zeta995)

def separatrix_to_equilibrium(boundary_parameters=None,separatrix_parameters=None, internal_flux_file=None):

    if ( (separatrix_parameters is None) and (boundary_parameters is None) or (separatrix_parameters is not None and boundary_parameters is not None) ):
        raise ValueError('Either separatrix_parameters or boundary_parameters must be provided')

    if boundary_parameters is not None:
        # Load boundary from file
        print('\t- Loading separatrix parameters from file:', boundary_parameters['file'])
        separatrix_parameters = load_separatrix_from_file(boundary_parameters)
    else:
        print('\t- Using provided separatrix parameters dictionary')
    
    B0 = separatrix_parameters['BT']
    Ip = separatrix_parameters['Ip']
    
    R0 = separatrix_parameters['R0']
    a = separatrix_parameters['a']
    z0 = separatrix_parameters['z0']
    
    kappa_sep = separatrix_parameters['kappa_sep']
    delta_sep = separatrix_parameters['delta_sep']
    zeta_sep = separatrix_parameters['zeta_sep']
    
    # Other parameters (not important, not passed to TRANSP beat and/or updated later)
    torflux_total = 1.0
    polflux_total = 1.0
    p0 = 1.0 #separatrix_parameters['p0_MPa']
    
    qstar_sep = PLASMAtools.evaluate_qstar(Ip, R0, kappa_sep, B0, a / R0, delta_sep, isInputIp=True) 
    factor_qstar = 1.4
    qstar = qstar_sep * factor_qstar
    
    # ---------------------------------------------------------------------------------
    # Internal equilibrium 
    # ---------------------------------------------------------------------------------
    
    if internal_flux_file is None:
        
        resol = 501
        
        # Internal equilibrium guess
        print('\t- Internal flux surfaces will be guessed')
        
        rho = np.linspace(0, 1, resol)
        
        rmin = np.linspace(0, a, resol)
        rmaj = R0*np.ones_like(rmin) # Assuming no Shafranov shift for the guess
        z0 = z0*np.ones_like(rmin)
        kappa = np.linspace(1, kappa_sep, resol)
        delta = np.linspace(0, delta_sep, resol)
        zeta = np.linspace(0, zeta_sep, resol)
        
        coeffs_MXH = 7
        sn = np.zeros((resol, coeffs_MXH))
        cn = np.zeros((resol, coeffs_MXH))
        
        torfluxa = torflux_total
        psi = np.linspace(0, polflux_total, resol)
        
        pressure = guess_pressure_profile(rho, p0)
        q = guess_q_profile(rho, qstar)
        
    else:
        
        print('\t- Internal flux surfaces will be loaded from file:', internal_flux_file)
        
        # Read inputgacode
        p = PROFILEStools.gacode_state(internal_flux_file)
        
        rho = p.profiles['rho(-)']
        
        rmin = p.profiles['rmin(m)'] * ( a/p.profiles['rmin(m)'][-1] )
        rmaj = p.profiles['rmaj(m)'] * ( R0/p.profiles['rcentr(m)'][0] ) # Scale from center, assuming then same Shafranov shift (relative) # This is equivalent to ( R0/p.profiles['rmaj(m)'][-1] )
        
        z0 = p.profiles['zmag(m)'] * z0/p.profiles['zmag(m)'][-1]
        kappa = p.profiles['kappa(-)'] * kappa_sep/p.profiles['kappa(-)'][-1]
        delta = p.profiles['delta(-)'] * delta_sep/p.profiles['delta(-)'][-1]
        if zeta_sep is not None:
            zeta = p.profiles['zeta(-)'] * zeta_sep/p.profiles['zeta(-)'][-1]  
        else:
            zeta = p.profiles['zeta(-)']
        
        sn, cn = [], []
        for i in range(len(p.shape_cos)):
            cn.append( p.profiles[f'shape_cos{i}(-)'])
            if i > 2:
                sn.append( p.profiles[f'shape_sin{i}(-)'])
            else:
                sn.append( np.zeros_like(rho) )
                
        sn = np.array(sn).T
        cn = np.array(cn).T
                
        torfluxa = p.profiles['torfluxa(Wb/radian)'][-1] * B0 / p.profiles['bcentr(T)'][-1]
        psi = p.profiles['polflux(Wb/radian)'] * Ip / p.profiles['current(MA)'][-1]
        pressure = p.profiles['ptot(Pa)'] * p0 / p.profiles['ptot(Pa)'][0]
        
        factor_sep_to_95_kappa = p.derived['kappa95'] / p.profiles['kappa(-)'][-1]
        factor_sep_to_95_delta = p.derived['delta95'] / p.profiles['delta(-)'][-1]
        
        qstar = PLASMAtools.evaluate_qstar(Ip, R0, kappa_sep*factor_sep_to_95_kappa, B0, a / R0, delta_sep*factor_sep_to_95_delta, isInputIp=True, ITERcorrection=True,includeShaping=True)
        q = PLASMAtools.q_profile_scale(p.derived['psi_pol_n'], p.profiles['q(-)'], qstar / p.derived['qstar_ITER'])
    
    return B0, Ip, R0, rho, rmin, rmaj, z0, kappa, delta, zeta, sn, cn, torfluxa, psi, q, pressure
    
def guess_q_profile(rho, qstar, q0 = 1.0):
    
    nu_q = 2.0
    
    _, iota = PLASMAtools.parabolicProfile( q0/nu_q, nu_q, rho, 1/qstar)
    q = 1/iota
    
    return q    

def guess_pressure_profile(rho, p0, pedge = 0.0):
    
    nu_p = 2.5
    
    _, pressure = PLASMAtools.parabolicProfile( p0/nu_p, nu_p, rho, pedge)
    
    return pressure

def load_separatrix_from_file(boundary_parameters):
    '''
    Load separatrix parameters from a file
    '''
    
    # Read R, Z from CSV
    R, Z = [], []
    with open(boundary_parameters['file'], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            R.append(float(row['R']))
            Z.append(float(row['Z']))

    R = np.array(R)
    Z = np.array(Z)
    
    # Get MXH coefficients
    surfaces = GEQtools.mitim_flux_surfaces()
    surfaces.reconstruct_from_RZ(R,Z)
    surfaces._to_mxh(n_coeff=boundary_parameters['coeffs_MXH'])

    
    separatrix_parameters = {
        'BT': boundary_parameters['B_T'],
        'Ip': boundary_parameters['Ip_MA'],
        'R0': surfaces.R0[0],
        'a': surfaces.a[0],
        'z0': surfaces.Z0[0],
        'kappa_sep': surfaces.kappa[0],
        'delta_sep': surfaces.delta[0],
        'zeta_sep': surfaces.zeta[0]
    }
    
    return separatrix_parameters

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
        
        p0_MPa = self._produce_p0guess(kwargs_geqdsk, Ip_MA, a, B_T)

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
# Initializer from FiBE: create the equilibrium, convert to geqdsk and call the geqdsk initializer
# --------------------------------------------------------------------------------------------

class initializer_from_fibe(initializer_from_geqdsk):
    '''
    Idea is to write geqdsk and then call the geqdsk initializer
    '''
    def __init__(self, beat_instance, label = 'fibe'):
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

        p0 = p0_MPa * 1.0e6
        Ip = Ip_MA * 1.0e6
        # If profiles exist, substitute the pressure and density guesses by something better (not perfect though, no ions)
        if ('ne' in kwargs_geqdsk.get('profiles_insert',{})) and ('Te' in kwargs_geqdsk.get('profiles_insert',{})):
            print('\t- Using ne profile instead of the ne0 guess')
            ne0_20 = kwargs_geqdsk['profiles_insert']['ne'][0]
            print('\t- Using Te profile for a better estimation of pressure, instead of the p0 guess')
            Te0_keV = kwargs_geqdsk['profiles_insert']['Te'][0]
            p0 = 2 * (Te0_keV*1E3) * 1.602176634E-19 * (ne0_20 * 1E20)
        # If betaN provided, use it to estimate the pressure
        elif 'BetaN' in kwargs_geqdsk:
            print('\t- Using BetaN for a better estimation of pressure, instead of the p0 guess')
            pvol_MPa = ( Ip_MA / (a * B_T) ) * (B_T ** 2 / (2 * 4 * np.pi * 1e-7)) / 1e6 * kwargs_geqdsk['BetaN'] * 1E-2
            p0 = pvol_MPa * 3.0 * 1.0e6

        # Run FiBE to generate equilibrium
        from fibe import FixedBoundaryEquilibrium
        eq = FixedBoundaryEquilibrium()
        eq.define_grid_and_boundary_with_mxh(
            nr=129,
            nz=129,
            rgeo=R,
            zgeo=z0,
            rminor=a,
            kappa=kappa_sep,
            cos_coeffs=[0.0, 0.0, 0.0],
            sin_coeffs=[0.0, np.arcsin(delta_sep), -zeta_sep])
        eq.initialize_profiles_with_minimal_input(p0, Ip, B_T)
        eq.initialize_psi()
        eq.solve_psi()

        # Convert to geqdsk and write it to initialization folder
        eq.to_geqdsk(str(self.folder / 'fibe.geqdsk'))

        # Call the geqdsk initializer
        super().__call__(geqdsk_file = self.folder / 'fibe.geqdsk',**kwargs_geqdsk)

# --------------------------------------------------------------------------------------------
# [Generic] Profile creator: Insert profiles
# --------------------------------------------------------------------------------------------

class creator:
    
        def __init__(self, initialize_instance, profiles_insert = {}, label = 'generic', **kwargs):
    
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
            label = 'parameterization',
            # Standard parameters
            BetaN = None,
            nu_ne = None,
            aLn = None,
            aLT = None,
            aLTe_to_aLTi_ratio = 1.0, # aLTe = aLTe_to_aLTi_ratio * aLTi for the BetaN optimization
            nresol = 501,
            # From a pedestal model
            rhotop = None,
            Ttop_keV = None,
            netop_20 = None,
            Tsep_keV = None,
            nesep_20 = None,
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
                        
            self.aLTe_to_aLTi_ratio = aLTe_to_aLTi_ratio

        def _return_profile_peaking_residual(self, aLn, x_a, x_top=None):

            # returns the residual of the betaN to match the profile to the EPED guess

            x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a = x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'roa': x, 'Te': ne, 'Ti': ne, 'ne': ne}
            super().__call__()

            return ((self.initialize_instance.profiles_current.derived['ne_peaking0.2'] - self.nu_ne) / self.nu_ne) ** 2

        def _return_profile_betan_residual(self, aLTi, x_a, aLn, x_top=None):

            # returns the residual of the betaN to match the profile to the EPED guess
            
            x, Te = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ttop_keV, self.Tsep_keV, aLTi*self.aLTe_to_aLTi_ratio, x_a = x_a,nx = self.nresol)
            x, Ti = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ttop_keV, self.Tsep_keV, aLTi, x_a = x_a,nx = self.nresol)
            x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a = x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'roa': x, 'Te': Te, 'Ti': Ti, 'ne': ne}
            super().__call__()

            return ((self.initialize_instance.profiles_current.derived['BetaN_engineering'] - self.BetaN) / self.BetaN) ** 2
    
        def __call__(self):

            # Gradients must use r/a coordinate but rhotop is in rho
            x_top = np.interp(self.rhotop, self.initialize_instance.profiles_current.profiles['rho(-)'], self.initialize_instance.profiles_current.derived['roa'])
            
            x_a = 0.3

            if (self.aLn_guess is not None) or (self.nu_ne is None):
                aLn = self.aLn_guess if self.aLn_guess is not None else 0.2
                print(f'\n\t - Using aLn = {aLn}')
            else:
                aLn_guess = 0.2
                # Find the density gradient that matches the peaking
                print(f'\n\t- Optimizing aLn to match ne peaking = {self.nu_ne}')
                bounds = [(0.0,3.0)]
                res = minimize(self._return_profile_peaking_residual, [aLn_guess], args=(x_a, x_top), method='Nelder-Mead', tol=1e-3, bounds=bounds)
                aLn = res.x[0]
                print(f'\n\t- Gradient: aLn = {aLn:.2f}')
                print(f'\t- ne peaking: {self.initialize_instance.profiles_current.derived["ne_peaking0.2"]:.5f} (target: {self.nu_ne:.5f})')

            # Find the temperature gradient that matches the BetaN
            if (self.aLT_guess is not None) or (self.BetaN is None):
                aLT = self.aLT_guess if self.aLT_guess is not None else 2.0
                print(f'\n\t- Using aLT = {aLT}')
            else:
                aLT_guess = 2.0
                # Find the temperature gradient that matches the BetaN
                print(f'\n\t- Optimizing aLTi to match BetaN = {self.BetaN}, with aLTe/aLTi = {self.aLTe_to_aLTi_ratio}')
                bounds = [(0.5,3.0)]
                res = minimize(self._return_profile_betan_residual, [aLT_guess], args=(x_a, aLn, x_top), method='Nelder-Mead', tol=1e-3, bounds=bounds)
                aLT = res.x[0]
                print(f'\n\t- Gradient: aLTi = {aLT:.2f}, aLTe = {aLT*self.aLTe_to_aLTi_ratio:.2f}')
                print(f'\t- BetaN: {self.initialize_instance.profiles_current.derived["BetaN_engineering"]:.5f} (target: {self.BetaN:.5f})')

            # Create profiles

            x, Te = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ttop_keV, self.Tsep_keV, aLT*self.aLTe_to_aLTi_ratio, x_a=x_a,nx = self.nresol)
            x, Ti = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ttop_keV, self.Tsep_keV, aLT, x_a=x_a,nx = self.nresol)
            x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a=x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'roa': x, 'Te': Te, 'Ti': Ti, 'ne': ne}
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
        aLTe_to_aLTi_ratio = 1.0,
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
        self.aLTe_to_aLTi_ratio = aLTe_to_aLTi_ratio
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
