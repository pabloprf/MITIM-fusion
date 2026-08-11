import shutil
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools.misc_tools.LOGtools import printMsg as print
from pyro import factor
from scipy.optimize import brentq
from IPython import embed

# --------------------------------------------------------------------------------------------
# Pruning levels (maestro.prune_level, per-beat override maestro.<beat>.prune_level)
# --------------------------------------------------------------------------------------------
# 0 PRUNE_NOTHING : keep everything                                    (legacy keep_all_files: true)
# 1 PRUNE_SCRATCH : drop execution scratch nothing reads back; every plot tab still works
# 2 PRUNE_RUN     : 1 + wipe run_<name>/ entirely; _persist moves instead of copying
# 3 PRUNE_OUTPUTS : 2 + prune persisted outputs and initializers       (legacy keep_all_files: false)
#
# `beat_results/` is NEVER touched by any level here -- it carries the sole idempotence key
# (beat_results/input.gacode) and the small sidecars the next beat reads. The only pruning that
# reaches into beat_results is PORTALS' end-of-run pass (portals_beat.optional_postprocessing).
PRUNE_NOTHING, PRUNE_SCRATCH, PRUNE_RUN, PRUNE_OUTPUTS = 0, 1, 2, 3
PRUNE_LEVELS = (PRUNE_NOTHING, PRUNE_SCRATCH, PRUNE_RUN, PRUNE_OUTPUTS)

# Initializer artifacts that are written and consumed within the same call and never read back.
# Everything else in initializer_*/ is load-bearing: input.gacode is re-read by MAESTRO's
# engineering-parameter freeze on EVERY invocation, input.geqdsk by mitim_plot_maestro, and
# initializer_eped/beat_results/ by the EPED creator's _inform_save on a restart.
_INITIALIZER_SCRATCH = ['freegs.geqdsk', 'freegs.geqdsk.helper', 'input.geqdsk.gacode']


def _prune_paths(paths):
    '''
    Delete the given files/folders, returning the bytes freed. Never raises: a failure to
    remove one item is reported and the rest still go, since pruning is opportunistic.
    '''

    freed = 0
    for path in paths:
        if not path.exists():
            continue
        size = IOtools.path_size_bytes(path)
        try:
            IOtools.shutil_rmtree(path) if path.is_dir() else path.unlink()
            freed += size
        except Exception as e:
            print(f'\t\t- Could not prune {IOtools.clipstr(path)}: {type(e).__name__}: {e}', typeMsg='w')
    return freed


# --------------------------------------------------------------------------------------------
# Generic beat class with required methods
# --------------------------------------------------------------------------------------------

class beat:

    # Level-1 prune targets inside run_<name>/, as glob patterns relative to it. Only artifacts
    # that nothing reads back after the beat completes belong here -- a level-1 run must still
    # replot in full. Overridden per beat; the generic beat drops nothing.
    scratch_patterns = []

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

        # Per-beat prune level from maestro.<beat>.prune_level (None -> inherit maestro.prune_level)
        self.prune_level_override = None

    @property
    def prune_level(self):
        '''Effective prune level for this beat: the per-beat namelist override, else the global one'''
        if self.prune_level_override is not None:
            return self.prune_level_override
        return self.maestro_instance.prune_level

    def _scratch_to_drop(self):
        '''
        Level-1 targets inside run_<name>/, resolved from `scratch_patterns`. Beats override
        this when the selection needs logic rather than a glob (see eped_beat).
        '''
        paths = []
        for pattern in self.scratch_patterns:
            paths += sorted(self.folder.glob(pattern))
        return paths

    def prune_run_folder(self):
        '''
        Post-beat pruning of this beat's run_<name>/, dispatched on the effective prune level.
        Called by MAESTRO after finalize/merge/inform, so everything a downstream beat or a
        replot needs has already been persisted into beat_results/ (never touched here).
        '''

        level = self.prune_level

        if level >= PRUNE_RUN:
            targets = sorted(self.folder.iterdir()) if self.folder.exists() else []
            what = f'run_{self.name}/ contents'
        elif level == PRUNE_SCRATCH:
            targets = self._scratch_to_drop()
            what = f'run_{self.name}/ execution scratch'
        else:
            return

        if not targets:
            return

        freed = _prune_paths(targets)
        print(f'\t\t- Pruning (level {level}): freed {IOtools.human_readable_size(freed)} of {what}')

    def prune_initializer(self):
        '''
        Level-3 pruning inside this beat's initializer_*/ folders. Drops the throwaway geqdsk
        intermediates and, when the initializer hosts a nested beat (eped_initializer builds a
        real eped_beat rooted there), prunes that beat's run folder with the same level.

        NEVER removes the initializer folder itself, nor input.gacode / input.geqdsk /
        beat_results -- all of those are read back on a re-invocation or by mitim_plot_maestro.
        '''

        if self.prune_level < PRUNE_OUTPUTS:
            return

        targets = []
        for initializer_folder in sorted(self.folder_beat.glob('initializer_*')):
            targets += [initializer_folder / name for name in _INITIALIZER_SCRATCH]
            # The nested beat's run folder (e.g. initializer_eped/run_eped/, which holds a full
            # per-height TOQ/ELITE tree); its beat_results/ sidecar is deliberately left alone
            for nested_run in sorted(initializer_folder.glob('run_*')):
                targets += sorted(nested_run.iterdir()) if nested_run.is_dir() else []

        targets = [t for t in targets if t.exists()]
        if not targets:
            return

        freed = _prune_paths(targets)
        print(f'\t\t- Pruning (level {self.prune_level}): freed {IOtools.human_readable_size(freed)} of initializer scratch')

    def incoming_profiles(self):
        '''
        The input.gacode this beat received, for the "before" trace in plots. run_<name>/input.gacode
        is gone from prune level 2 on, so fall back to the initializer copy (the very same state the
        beat ran on, which pruning never removes). Returns None when neither survives, so callers
        skip that trace instead of raising -- mitim_plot_maestro must never fail on a pruned run.
        '''

        for f in [self.folder / 'input.gacode'] + sorted(self.folder_beat.glob('initializer_*/input.gacode')):
            if f.exists():
                return PROFILEStools.gacode_state(f)

        print(f'\t\t- Skipping the "before" profiles of beat {self.name}: input.gacode not available (pruned)', typeMsg='w')
        return None

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

    def _persist(self, src, dst):
        '''
        Copy src to dst, or move when this beat's prune level is about to wipe the run folder
        anyway (>= PRUNE_RUN). shutil.move reduces to os.rename on the same filesystem (folder
        and folder_output share parent folder_beat), so the move path is essentially free
        regardless of file size.
        '''
        if self.prune_level < PRUNE_RUN:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    def prepare(self, *args, **kwargs):
        pass

    def prepare_minimal(self, *args, **kwargs):
        '''
        Skip-path counterpart of prepare(): called by MAESTROmain.prepare() with the
        same namelist kwargs when the beat is already complete and prepare() is skipped.
        Beats override this to stash the parameters that finalize()/merge_parameters()
        still need on a re-invocation (those always run, even for completed beats).
        '''
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

    def grab_output(self, *args, **kwargs):
        pass

    def optional_postprocessing(self):
        # Hook for beat-specific end-of-run postprocessing (e.g. space-saving cleanup).
        # Called once per beat by MAESTRO.finalize() after all beats have run. No-op by
        # default; override in a beat subclass (see portals_beat).
        pass

    def plot(self, *args, **kwargs):
        return ''

    def summary(self, output_dir, counter = None, wall_time_s = None):
        '''
        Best-effort generation of a markdown section describing this beat's
        final state. Writes any figures into `output_dir` and returns a
        markdown string (or None if nothing meaningful can be reported).
        Subclasses override; base returns None.

        Args:
            output_dir: Path where figures should be written (relative links
                in the returned markdown will resolve against this).
            counter: Integer beat counter (1-based) as registered in maestro.beats.
            wall_time_s: Wall-clock seconds for this beat's run+finalize, as
                read from Outputs/Performance/timing.jsonl by the orchestrator.
        '''
        return None


def _format_seconds(seconds):
    '''Format a duration in seconds as h:mm:ss (or m:ss if under an hour).'''
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f'{h}:{m:02d}:{sec:02d}'
    return f'{m}:{sec:02d}'

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
        elif kwargs_geqdsk.get('BetaN') is not None:
            print('\t- Using BetaN for a better estimation of pressure, instead of the p0 guess')
            pvol_MPa = ( Ip_MA / (a * B_T) ) * (B_T ** 2 / (2 * 4 * np.pi * 1e-7)) / 1e6 * kwargs_geqdsk['BetaN'] * 1E-2
            p0_MPa = pvol_MPa * 3.0
        # Otherwise, fall back to a fixed guess
        else:
            print('\t- No profiles or BetaN available, using default p0 guess of 1.0 MPa', typeMsg='w')
            p0_MPa = 1.0

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
        Paux_MW = 1.0,
        Zeff = 1.5,
        netop_20 = 1.0,
        coeffs_MXH = 5,
        extract_995_from="analytic_interpolation",
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
        
        type_heating = kwargs_profiles.get('type_heating', 'ICRH')
        if type_heating == 'ICRH':
            aux_channels = {'e': 'qrfe(MW/m^3)', 'i': 'qrfi(MW/m^3)', 'total': 'qRF_MW'}
        elif type_heating == 'NBI':
            aux_channels = {'e': 'qbeame(MW/m^3)', 'i': 'qbeami(MW/m^3)', 'total': 'qBEAM_MW'}
        else:
            aux_channels = None
        
        p = self.f.to_profiles(ne0_20 = netop_20, Zeff = Zeff, Paux = Paux_MW, coeffs_MXH = coeffs_MXH, aux_channels = aux_channels)

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

        try:
            shaping_psin = self.beat_instance.maestro_instance.maestro_namelist['plasma']['parameters']['separatrix'].get('shaping_extraction_psin', 0.995)
        except (KeyError, AttributeError):
            shaping_psin = 0.995
        f = GEQtools.MITIMgeqdsk(self.folder / 'input.geqdsk', shaping_psin=shaping_psin)

        if self.extract_995_from == "analytic_interpolation":
            print('\t- Extracting 0.995 flux surface parameters from "analytic_interpolation"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["analytic_interpolation"]["psin995"]["kappa"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["analytic_interpolation"]["psin995"]["delta"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["analytic_interpolation"]["psin995"]["zeta"]
        elif self.extract_995_from == 'analytic':
            print('\t- Extracting 0.995 flux surface parameters from "analytic"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["analytic"]["psin995"]["kappa"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["analytic"]["psin995"]["delta"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["analytic"]["psin995"]["zeta"]
        elif self.extract_995_from == 'turnbull':
            print('\t- Extracting 0.995 flux surface parameters from "turnbull"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["turnbull"]["psin995"]["kappa"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["turnbull"]["psin995"]["delta"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["turnbull"]["psin995"]["zeta"]
        elif self.extract_995_from == 'mxh':
            print('\t- Extracting 0.995 flux surface parameters from "mxh"')
            self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["mxh"]["psin995"]["kappa"]
            self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["mxh"]["psin995"]["delta"]
            self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["mxh"]["psin995"]["zeta"]
            self.beat_instance.maestro_instance.parameters_trans_beat['s_three995'] = f.geometric_parameters["mxh"]["psin995"]["shape_sin"][2]
            self.beat_instance.maestro_instance.parameters_trans_beat['s_four995'] = f.geometric_parameters["mxh"]["psin995"]["shape_sin"][3]
        elif self.extract_995_from == 'miller':
            raise Exception('[MITIM] Miller extraction not available yet')
            # print('\t- Extracting 0.995 flux surface parameters from "miller"')
            # self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = f.geometric_parameters["miller"]["psin995"]["kappa"]
            # self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = f.geometric_parameters["miller"]["psin995"]["delta"]
            # self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = f.geometric_parameters["miller"]["psin995"]["zeta"] # Should be zero

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
        Paux_MW = 1.0,
        Zeff = 1.5,
        netop_20 = 1.0,
        coeffs_MXH = 5,
        extract_995_from="analytic_interpolation",
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
        
        type_heating = kwargs.get('type_heating', 'ICRH')
        if type_heating == 'ICRH':
            aux_channels = {'e': 'qrfe(MW/m^3)', 'i': 'qrfi(MW/m^3)', 'total': 'qRF_MW'}
        elif type_heating == 'NBI':
            aux_channels = {'e': 'qbeame(MW/m^3)', 'i': 'qbeami(MW/m^3)', 'total': 'qBEAM_MW'}
        else:
            aux_channels = None

        self.p = GEQtools.equilibrium_to_profiles(
            rho, psi, q, pressure, torfluxa, R0, B0, Ip,
            kappa, delta, zeta, rmin, rmaj, z0, sn[:,:coeffs_MXH], cn[:,:coeffs_MXH],
            ne0_20 = netop_20,
            Zeff = Zeff,
            Z = 9,
            Paux = Paux_MW,
            aux_channels = aux_channels,
        )

        # [Optional] Use the freegs to correct the profiles (keeping the shaping)
        try:
            self._correct_profiles_withfreegs(Paux_MW = Paux_MW, Zeff = Zeff, netop_20 = netop_20, coeffs_MXH = coeffs_MXH, **kwargs)
        except Exception as e:
            print(f'\t- Could not run freegs to correct the profiles ({type(e).__name__}: {e}), proceeding with uncorrected ones', typeMsg = 'w')
        
        # Write it to initialization folder
        self.p.write_state(file=self.folder / 'input.separatrix.gacode')

        # Save parameters also here in case they are needed already at this beat (e.g. for EPED)
        self._inform_save()

        # Call the profiles initializer
        kwargs["profiles_file"] = self.folder / 'input.separatrix.gacode'
        super().__call__(**kwargs)

    def _correct_profiles_withfreegs(self,
            Paux_MW = 1.0, Zeff = 1.5, netop_20 = 1.0, coeffs_MXH = 5,
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
        
        type_heating = kwargs.get('type_heating', 'ICRH')
        if type_heating == 'ICRH':
            aux_channels = {'e': 'qrfe(MW/m^3)', 'i': 'qrfi(MW/m^3)', 'total': 'qRF_MW'}
        elif type_heating == 'NBI':
            aux_channels = {'e': 'qbeame(MW/m^3)', 'i': 'qbeami(MW/m^3)', 'total': 'qBEAM_MW'}
        else:
            aux_channels = None
        
        self.p = f.to_profiles(ne0_20 = netop_20, Zeff = Zeff, Paux = Paux_MW, coeffs_MXH = coeffs_MXH, aux_channels = aux_channels)

        for i in ['kappa(-)', 'delta(-)', 'zeta(-)', 'rmin(m)', 'rmaj(m)', 'zmag(m)']:
            self.p.profiles[i] = np.interp(self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles[i])
        
        for i in ['rcentr(m)']:
            self.p.profiles[i] = p_old.profiles[i]

        # When a real equilibrium was supplied via internal_flux_file, preserve ITS poloidal-flux
        # mapping too -- not just the shaping. Otherwise the freegs psi (less edge-compressed than a
        # real equilibrium) is kept, so boundary_surface_psin extracts a too-near-separatrix (over-
        # squared) surface and the realistic radial decay is partly wasted. Without a file, keep
        # freegs's self-consistent psi (still better than the linear-ramp guess).
        if kwargs.get('internal_flux_file') is not None:
            self.p.profiles['polflux(Wb/radian)'] = np.interp(
                self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles['polflux(Wb/radian)'])

        for i in range(coeffs_MXH):
            self.p.profiles[f'shape_cos{i}(-)'] = np.interp(self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles[f'shape_cos{i}(-)'])
        for i in range(coeffs_MXH-3):
            self.p.profiles[f'shape_sin{i+3}(-)'] = np.interp(self.p.profiles['rho(-)'], p_old.profiles['rho(-)'], p_old.profiles[f'shape_sin{i+3}(-)'])

        # The shaping overwrite above replaces the geometry that to_profiles normalized the
        # auxiliary power against (the solved freegs flux surfaces) with the analytic guess
        # (r = a*rho, kappa ramping from 1). dV/dr changes by up to ~15% in the core at high
        # elongation, so the volume integral of the aux channels no longer returns Paux_MW.
        # Renormalize against the geometry actually written.
        if aux_channels is None:
            # mirrors the fallback inside GEQtools.equilibrium_to_profiles
            aux_channels = {'e': 'qrfe(MW/m^3)', 'i': 'qrfi(MW/m^3)', 'total': 'qRF_MW'}
        self.p.derive_quantities()
        P_now = self.p.derived[aux_channels['total']][-1]
        if P_now > 0.0:
            factor = Paux_MW / P_now
            print(f'\t- Renormalizing auxiliary power after the shaping overwrite '
                  f'({P_now:.4f} -> {Paux_MW:.4f} MW, factor {factor:.4f})', typeMsg='i')
            # non-in-place on purpose: if the two channels alias the same array
            # (equilibrium_to_profiles used to), in-place *= would apply factor twice
            self.p.profiles[aux_channels['e']] = self.p.profiles[aux_channels['e']] * factor
            self.p.profiles[aux_channels['i']] = self.p.profiles[aux_channels['i']] * factor
            self.p.derive_quantities()

    def _inform_save(self):
        
        if self.extract_995_from is None:
            return
        
        try:
            shaping_psin = self.beat_instance.maestro_instance.maestro_namelist['plasma']['parameters']['separatrix'].get('shaping_extraction_psin', 0.995)
        except (KeyError, AttributeError):
            shaping_psin = 0.995
        if "p" not in dir(self):
            self.p = PROFILEStools.gacode_state(self.folder / 'input.separatrix.gacode')
        # __call__ overwrites the shaping profiles after their last derive-at-0.995, so re-derive
        # here to freeze the shaping at the chosen extraction surface (default 0.995).
        self.p.derive_quantities(shaping_psin=shaping_psin)
        kappa995, delta995, zeta995 = self.p.derived["kappa995"], self.p.derived["delta995"], self.p.derived["zeta995"]

        self.beat_instance.maestro_instance.parameters_trans_beat['kappa995'] = kappa995
        self.beat_instance.maestro_instance.parameters_trans_beat['delta995'] = delta995
        self.beat_instance.maestro_instance.parameters_trans_beat['zeta995'] = zeta995

        print('\t\t- 0.995 flux surface kappa, delta, and zeta saved for future beats -> ', kappa995, delta995, zeta995)

def _scale_profile_to_sep(prof, target_sep):
    '''
    Scale the file's internal profile shape so its separatrix value matches the
    requested one. A ~zero edge value (e.g. zmag of an up-down-symmetric file —
    the separatrix initializer's own default output — or delta/zeta of a
    circular one) makes the ratio target/edge a NaN/inf that silently poisons
    the whole profile; shift instead, which preserves the internal structure
    and still hits the target at the edge.
    '''
    edge = prof[-1]
    if abs(edge) < 1e-10:
        return prof + target_sep
    return prof * (target_sep / edge)


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
    
    # Assumed q0
    q0_assume = 1.5
    
    # Guess qstar from separatrix parameters
    kappa95 = kappa_sep*0.95
    delta95 = delta_sep*0.95
    qstar = PLASMAtools.evaluate_qstar(Ip, R0, kappa95, B0, a / R0, delta95, isInputIp=True, ITERcorrection=True,includeShaping=True)
    factor_qstar = 1.4
    qstar_sep = qstar * factor_qstar
    
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
        zeta = np.linspace(0, zeta_sep if zeta_sep is not None else 0, resol)
        
        coeffs_MXH = 7
        sn = np.zeros((resol, coeffs_MXH))
        cn = np.zeros((resol, coeffs_MXH))
        
        torfluxa = torflux_total
        psi = np.linspace(0, polflux_total, resol)
        
        pressure = guess_pressure_profile(rho, p0)
        q = guess_q_profile(rho, qstar_sep, q0 = q0_assume)
        
    else:
        
        print('\t- Internal flux surfaces will be loaded from file:', internal_flux_file)
        
        # Read inputgacode
        p = PROFILEStools.gacode_state(internal_flux_file)
        
        rho = p.profiles['rho(-)']

        rmin = p.profiles['rmin(m)'] * ( a/p.profiles['rmin(m)'][-1] )
        rmaj = p.profiles['rmaj(m)'] * ( R0/p.profiles['rcentr(m)'][0] ) # Scale from center, assuming then same Shafranov shift (relative) # This is equivalent to ( R0/p.profiles['rmaj(m)'][-1] )

        z0 = _scale_profile_to_sep(p.profiles['zmag(m)'], z0)
        kappa = p.profiles['kappa(-)'] * kappa_sep/p.profiles['kappa(-)'][-1]
        delta = _scale_profile_to_sep(p.profiles['delta(-)'], delta_sep)
        if zeta_sep is not None:
            zeta = _scale_profile_to_sep(p.profiles['zeta(-)'], zeta_sep)
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
    
def guess_q_profile(rho, qstar, q0 = 1.0, debug = False):
    
    nu_q = 2.0
    
    _, iota = PLASMAtools.parabolicProfile( q0/nu_q, nu_q, rho, 1/qstar)
    q = 1/iota
    
    if debug:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
        ax = axs[0]
        ax.plot(rho, q, label='q profile')
        ax.axhline(q0, color='k', ls='--', label='q0')
        ax.axhline(qstar, color='r', ls='--', label='qstar')
        ax.set_xlabel('rho')
        ax.set_ylabel('q')
        ax.legend()
        
        ax = axs[1]
        ax.plot(rho, 1/q, label='iota profile')
        ax.axhline(1/q0, color='k', ls='--', label='iota0')
        ax.axhline(1/qstar, color='r', ls='--', label='iotastar')
        ax.set_xlabel('rho')
        ax.set_ylabel('iota')
        ax.legend()
        
        plt.show()
        embed()
    
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

        Ip = Ip_MA * 1.0e6
        p0 = self._produce_p0guess(kwargs_geqdsk, Ip_MA, a, B_T) * 1.0e6

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

            # Optional rotation (read_fixed_profiles maps 'w0_rads' -> 'w0')
            if 'w0' in self.profiles_insert:
                self.initialize_instance.profiles_current.profiles['w0(rad/s)'] = self.profiles_insert['w0']

            # Update derived
            self.initialize_instance.profiles_current.derive_quantities()

        def _inform_save(self, **kwargs):
            pass

# --------------------------------------------------------------------------------------------
# Profile creator from parameterization: Create profiles from a parameterization
# --------------------------------------------------------------------------------------------

def _match_gradient_to_target(mismatch_fun, bounds, label, xtol=1e-4):
    '''
    Solve mismatch_fun(x) = 0 for a signed, monotonic relative mismatch (e.g. BetaN or
    ne-peaking vs their targets as a function of a single a/L knob) by bracketed root
    finding. If the target is unreachable within bounds, saturate at the closest bound
    and warn (Nelder-Mead used to stall silently against the bound here).
    '''
    f_lo, f_hi = mismatch_fun(bounds[0]), mismatch_fun(bounds[1])

    if f_lo * f_hi > 0:
        x = bounds[0] if abs(f_lo) < abs(f_hi) else bounds[1]
        mismatch = f_lo if x == bounds[0] else f_hi
        print(f"\t- {label} target unreachable within bounds {bounds}, saturating at {x} (relative mismatch: {mismatch:+.2%})", typeMsg='w')
        return x

    return brentq(mismatch_fun, bounds[0], bounds[1], xtol=xtol)

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

        def _return_profile_peaking_mismatch(self, aLn, x_a, x_top=None):

            # returns the signed relative mismatch of the ne peaking (monotonic in aLn)

            x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a = x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'roa': x, 'Te': ne, 'Ti': ne, 'ne': ne}
            super().__call__()

            return (self.initialize_instance.profiles_current.derived['ne_peaking0.2'] - self.nu_ne) / self.nu_ne

        def _return_profile_betan_mismatch(self, aLTi, x_a, aLn, x_top=None):

            # returns the signed relative mismatch of the BetaN (monotonic in aLTi)

            x, Te = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ttop_keV, self.Tsep_keV, aLTi*self.aLTe_to_aLTi_ratio, x_a = x_a,nx = self.nresol)
            x, Ti = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ttop_keV, self.Tsep_keV, aLTi, x_a = x_a,nx = self.nresol)
            x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a = x_a,nx = self.nresol)

            # Call the generic creator
            self.profiles_insert = {'roa': x, 'Te': Te, 'Ti': Ti, 'ne': ne}
            super().__call__()

            return (self.initialize_instance.profiles_current.derived['BetaN_engineering'] - self.BetaN) / self.BetaN
    
        def __call__(self):

            # Gradients must use r/a coordinate but rhotop is in rho
            x_top = np.interp(self.rhotop, self.initialize_instance.profiles_current.profiles['rho(-)'], self.initialize_instance.profiles_current.derived['roa'])
            
            x_a = 0.3

            if (self.aLn_guess is not None) or (self.nu_ne is None):
                aLn = self.aLn_guess if self.aLn_guess is not None else 0.2
                print(f'\n\t - Using aLn = {aLn}')
            else:
                # Find the density gradient that matches the peaking (bracketed root find; monotonic)
                print(f'\n\t- Optimizing aLn to match ne peaking = {self.nu_ne}')
                aLn = _match_gradient_to_target(lambda a: self._return_profile_peaking_mismatch(a, x_a, x_top=x_top), (0.0, 3.0), 'ne peaking')
                self._return_profile_peaking_mismatch(aLn, x_a, x_top=x_top)
                print(f'\n\t- Gradient: aLn = {aLn:.4f}')
                print(f'\t- ne peaking: {self.initialize_instance.profiles_current.derived["ne_peaking0.2"]:.5f} (target: {self.nu_ne:.5f})')

            # Find the temperature gradient that matches the BetaN
            if (self.aLT_guess is not None) or (self.BetaN is None):
                aLT = self.aLT_guess if self.aLT_guess is not None else 2.0
                print(f'\n\t- Using aLT = {aLT}')
            else:
                # Find the temperature gradient that matches the BetaN (bracketed root find; monotonic).
                # Guard: TRANSP's TRDAT rejects Te/Ti data above 100 keV (CKDRNG), so if this seed
                # would exceed T0_cap_keV on axis (BetaN unreachable at low density -> aLT saturates
                # high), lower the BetaN target 25% and re-solve until it fits.
                T0_cap_keV = 95.0
                for _ in range(10):
                    print(f'\n\t- Optimizing aLTi to match BetaN = {self.BetaN}, with aLTe/aLTi = {self.aLTe_to_aLTi_ratio}')
                    aLT = _match_gradient_to_target(lambda a: self._return_profile_betan_mismatch(a, x_a, aLn, x_top=x_top), (0.5, 3.0), 'BetaN')
                    self._return_profile_betan_mismatch(aLT, x_a, aLn, x_top=x_top)
                    print(f'\n\t- Gradient: aLTi = {aLT:.4f}, aLTe = {aLT*self.aLTe_to_aLTi_ratio:.4f}')
                    print(f'\t- BetaN: {self.initialize_instance.profiles_current.derived["BetaN_engineering"]:.5f} (target: {self.BetaN:.5f})')
                    T0 = max(float(self.initialize_instance.profiles_current.profiles['te(keV)'][0]),
                             float(self.initialize_instance.profiles_current.profiles['ti(keV)'][0, 0]))
                    if T0 <= T0_cap_keV:
                        break
                    self.BetaN *= 0.75
                    print(f'\t- On-axis T = {T0:.1f} keV exceeds the TRANSP-safe {T0_cap_keV:.0f} keV cap, '
                          f'lowering initialization BetaN to {self.BetaN:.3f} and re-solving', typeMsg='w')
                else:
                    raise Exception(f'[MITIM] Initialization on-axis T still above {T0_cap_keV} keV '
                                    f'after lowering BetaN to {self.BetaN:.3f}')

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
        cpus_master = self.beat_eped.maestro_instance.maestro_namelist['maestro']['master_cpus']
        cpus_eped = self.beat_eped.maestro_instance.maestro_namelist['maestro']['eped']['preprocess_prepare_parameters']['cpus']
        
        nproc_per_run = cpus_eped if cpus_eped is not None else cpus_master
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

# --------------------------------------------------------------------------------------------
# Profile creator from fixed boundary conditions: Create profiles from user-specified BC values
# --------------------------------------------------------------------------------------------

class creator_from_fixed_bc(creator_from_parameterization):

    def __init__(
        self,
        initialize_instance,
        label = 'fixed_bc',
        x_bc = None,                # BC location value in the coordinate given by bc_coordinate
        bc_coordinate = 'rho',      # coordinate for x_bc: 'rho' (rho_tor), 'roa' (r/a), or 'psin'
        Te_bc = None,               # Te at x_bc (keV)
        Ti_bc = None,               # Ti at x_bc (keV); if None, uses Te_bc
        neped_20 = None,            # ne at x_bc (10^20 m^-3)
        Tesep_keV = None,           # Te at separatrix (keV); if None, read from current profiles
        nesep_20 = None,            # ne at separatrix (10^20 m^-3); if None, read from current profiles
        BetaN = None,
        nu_ne = None,
        aLn = None,
        aLT = None,
        aLTe_to_aLTi_ratio = 1.0,
        nresol = 501,
        **kwargs,                   # Absorb extra engineering parameters passed from the namelist
        ):
        netop_20 = neped_20

        super().__init__(
            initialize_instance,
            label = label,
            BetaN = BetaN,
            nu_ne = nu_ne,
            aLn = aLn,
            aLT = aLT,
            aLTe_to_aLTi_ratio = aLTe_to_aLTi_ratio,
            nresol = nresol,
            rhotop = x_bc,          # stored as-is; converted to rho_tor in __call__
            Ttop_keV = Te_bc,
            netop_20 = netop_20,
            Tsep_keV = Tesep_keV,
            nesep_20 = nesep_20,
            )

        self.x_bc = x_bc
        self.bc_coordinate = bc_coordinate
        self.Te_bc = Te_bc
        self.Ti_bc = Ti_bc if Ti_bc is not None else Te_bc

    def _return_profile_betan_mismatch(self, aLTi, x_a, aLn, x_top=None):

        x, Te = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Te_bc, self.Tsep_keV, aLTi*self.aLTe_to_aLTi_ratio, x_a=x_a, nx=self.nresol)
        x, Ti = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ti_bc, self.Tsep_keV, aLTi, x_a=x_a, nx=self.nresol)
        x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a=x_a, nx=self.nresol)

        self.profiles_insert = {'roa': x, 'Te': Te, 'Ti': Ti, 'ne': ne}
        creator.__call__(self)

        return (self.initialize_instance.profiles_current.derived['BetaN_engineering'] - self.BetaN) / self.BetaN

    def __call__(self):

        # Convert x_bc from bc_coordinate to rho_tor now that profiles are available
        _profs = self.initialize_instance.profiles_current
        _rho   = _profs.profiles['rho(-)']
        _roa   = _profs.derived['roa']
        _psin  = _profs.derived['psi_pol_n']
        if self.bc_coordinate == 'rho':
            self.rhotop = float(self.x_bc)
        elif self.bc_coordinate == 'roa':
            self.rhotop = float(np.interp(self.x_bc, _roa, _rho))
        elif self.bc_coordinate == 'psin':
            self.rhotop = float(np.interp(self.x_bc, _psin, _rho))
        else:
            raise ValueError(f"bc_coordinate must be 'rho', 'roa', or 'psin', got '{self.bc_coordinate}'")

        print('\n\t--------------------------------')
        print('\t  fixed_bc profile creator')
        print('\t--------------------------------')
        print(f'\t  Boundary condition location:  x_bc    = {self.x_bc} ({self.bc_coordinate})  ->  rho_tor = {self.rhotop:.4f}')
        print(f'\t  Boundary condition values:    Te_bc   = {self.Te_bc:.4f} keV')
        print(f'\t                                Ti_bc   = {self.Ti_bc:.4f} keV')
        print(f'\t                                ne_bc   = {self.netop_20:.4f} 10^20/m^3')
        print(f'\t  Optimization targets:         BetaN   = {self.BetaN}')
        print(f'\t                                nu_ne   = {self.nu_ne}')
        print(f'\t  aLTe/aLTi ratio:              {self.aLTe_to_aLTi_ratio}')

        # Populate separatrix values from current profiles if not provided by user or engineering parameters
        if self.Tsep_keV is None:
            self.Tsep_keV = self.initialize_instance.profiles_current.profiles['te(keV)'][-1]
            print(f'\t  Tsep_keV not provided, read from current profiles: {self.Tsep_keV:.4f} keV')
        else:
            print(f'\t  Separatrix values:            Tsep    = {self.Tsep_keV:.4f} keV')
        if self.nesep_20 is None:
            self.nesep_20 = self.initialize_instance.profiles_current.profiles['ne(10^19/m^3)'][-1] / 10.0
            print(f'\t  nesep_20 not provided, read from current profiles: {self.nesep_20:.4f} 10^20/m^3')
        else:
            print(f'\t                                nesep   = {self.nesep_20:.4f} 10^20/m^3')
        print('\t--------------------------------\n')

        # Gradients use r/a coordinate; rhotop is now in rho_tor
        x_top = np.interp(self.rhotop, self.initialize_instance.profiles_current.profiles['rho(-)'], self.initialize_instance.profiles_current.derived['roa'])
        print(f'\t- x_bc = {self.x_bc} ({self.bc_coordinate}) -> rho_tor = {self.rhotop:.4f} -> r/a = {x_top:.4f}')

        x_a = 0.3

        # Optimize aLn to match nu_ne (density peaking)
        if (self.aLn_guess is not None) or (self.nu_ne is None):
            aLn = self.aLn_guess if self.aLn_guess is not None else 0.2
            print(f'\n\t- Using fixed aLn = {aLn:.4f} (no nu_ne optimization)')
        else:
            print(f'\n\t- Optimizing aLn to match nu_ne = {self.nu_ne:.4f}')
            aLn = _match_gradient_to_target(lambda a: self._return_profile_peaking_mismatch(a, x_a, x_top=x_top), (0.0, 3.0), 'ne peaking')
            self._return_profile_peaking_mismatch(aLn, x_a, x_top=x_top)
            print(f'\t  --> aLn = {aLn:.4f}')
            print(f'\t  --> ne peaking achieved: {self.initialize_instance.profiles_current.derived["ne_peaking0.2"]:.5f} (target: {self.nu_ne:.5f})')

        # Optimize aLT to match BetaN
        if (self.aLT_guess is not None) or (self.BetaN is None):
            aLT = self.aLT_guess if self.aLT_guess is not None else 2.0
            print(f'\n\t- Using fixed aLT = {aLT:.4f} (no BetaN optimization)')
        else:
            print(f'\n\t- Optimizing aLTi to match BetaN = {self.BetaN:.4f} (aLTe/aLTi = {self.aLTe_to_aLTi_ratio:.4f})')
            aLT = _match_gradient_to_target(lambda a: self._return_profile_betan_mismatch(a, x_a, aLn, x_top=x_top), (0.5, 3.0), 'BetaN')
            self._return_profile_betan_mismatch(aLT, x_a, aLn, x_top=x_top)
            print(f'\t  --> aLTi = {aLT:.4f}, aLTe = {aLT*self.aLTe_to_aLTi_ratio:.4f}')
            print(f'\t  --> BetaN achieved: {self.initialize_instance.profiles_current.derived["BetaN_engineering"]:.5f} (target: {self.BetaN:.5f})')

        print(f'\n\t- Final gradients: aLn = {aLn:.4f}, aLTi = {aLT:.4f}, aLTe = {aLT*self.aLTe_to_aLTi_ratio:.4f}')

        # Create profiles using the user-specified Te_bc and Ti_bc as separate boundary conditions
        x, Te = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Te_bc, self.Tsep_keV, aLT*self.aLTe_to_aLTi_ratio, x_a=x_a, nx=self.nresol)
        x, Ti = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.Ti_bc, self.Tsep_keV, aLT, x_a=x_a, nx=self.nresol)
        x, ne = FunctionalForms.MITIMfunctional_aLyTanh(x_top, self.netop_20, self.nesep_20, aLn, x_a=x_a, nx=self.nresol)

        self.profiles_insert = {'roa': x, 'Te': Te, 'Ti': Ti, 'ne': ne}
        creator.__call__(self)

        print(f'\n\t- Profiles inserted. Final derived quantities:')
        print(f'\t  --> BetaN   = {self.initialize_instance.profiles_current.derived["BetaN_engineering"]:.5f}')
        print(f'\t  --> nu_ne   = {self.initialize_instance.profiles_current.derived["ne_peaking0.2"]:.5f}')
        print(f'\t  --> Te0     = {self.initialize_instance.profiles_current.profiles["te(keV)"][0]:.4f} keV')
        print(f'\t  --> Ti0     = {self.initialize_instance.profiles_current.profiles["ti(keV)"][0,0]:.4f} keV')
        print(f'\t  --> ne0     = {self.initialize_instance.profiles_current.profiles["ne(10^19/m^3)"][0]/10.0:.4f} 10^20/m^3')
