import os
import re
import shutil
import copy
import numpy as np
from typing import OrderedDict
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat, _format_seconds
from IPython import embed


def _extraction_index(cdf_results, extract_at):
    """Map an `extract_at` selector to a CDF time-slice index.

        'saw'  / 'saw-N'  -> cdf.ind_saw - N   (the last sawtooth, or N coarse slices before it)
        'last' / 'last-N' -> last slice,        or N coarse slices before it

    `ind_saw` is CDFtools' last-sawtooth reference (findLastSawtoothIndex). The small step-back
    matters because MAESTRO's coarse output grid can otherwise land on the sawtoothing (crash)
    profiles. Out-of-range selections are clamped with a warning.
    """
    s = str(extract_at).strip().lower()
    if s.startswith("saw"):
        it = cdf_results.ind_saw + int(s[3:] or 0)          # 'saw-2' -> ind_saw - 2
    elif s.startswith("last"):
        it = (len(cdf_results.t) - 1) + int(s[4:] or 0)     # 'last-2' -> last - 2
    else:
        raise ValueError(f"[MITIM] Unrecognized extract_at '{extract_at}' (use 'saw', 'saw-N', 'last' or 'last-N')")
    nt = len(cdf_results.t)
    if not (0 <= it < nt):
        print(f"\t- extract_at '{extract_at}' -> slice {it} out of range for {nt} time slices; clamping", typeMsg='w')
        it = min(max(it, 0), nt - 1)
    return it


def _apply_flattop_floor(cdf_results, it_extract, time_diffusion, time_end, min_flattop_fraction):
    """Do not extract before a minimum fraction of the flattop window.

    A plasma whose ONLY sawtooth fires early (then never crashes again) would otherwise be
    sampled by 'saw'/'saw-N' before the heating and current diffusion have settled. If the
    selected slice sits earlier than
        time_diffusion + min_flattop_fraction * (time_end - time_diffusion)
    it is moved forward to the first slice at/after that floor. Returns the (possibly updated)
    index. A None fraction -- or missing timing on a skipped-prepare restart -- is a no-op.
    """
    if min_flattop_fraction is None or time_diffusion is None or time_end is None:
        return it_extract
    t_floor = time_diffusion + min_flattop_fraction * (time_end - time_diffusion)
    if cdf_results.t[it_extract] >= t_floor:
        return it_extract
    it_floor = min(int(np.searchsorted(cdf_results.t, t_floor)), len(cdf_results.t) - 1)
    print(f"\t- extraction slice t={cdf_results.t[it_extract]:.4f}s is before the flattop floor "
          f"({min_flattop_fraction:.0%} of flattop -> t={t_floor:.4f}s; likely only an early sawtooth); "
          f"extracting at t={cdf_results.t[it_floor]:.4f}s instead", typeMsg='w')
    return it_floor


class transp_beat(beat):

    def __init__(
        self,
        maestro_instance,
        letter              = None,
        shot                = None,
        extract_at          = "saw-1",              # Which CDF time slice feeds the next beat (see prepare()/finalize())
        min_extraction_flattop_fraction = 0.5,      # Floor the extraction at this fraction of the flattop window (see finalize()); None disables
        ):

        super().__init__(maestro_instance, beat_name = 'transp')

        # Decide now the shot and runid and how to extract (need to do this now and not in prepare because of restart options, that do not run prepare)

        if shot is None:
            folder_last = self.maestro_instance.folder.resolve().name
            shot = IOtools.string_to_sequential_number(folder_last, num_digits=5)

        if letter is None:
            username = os.environ['USER']
            letter = username[0].upper()
            if letter == '$':
                letter = 'A'

        self.shot = shot
        self.runid = letter + str(self.maestro_instance.counter_current).zfill(2)

        # Which CDF time slice is handed to the next beat (see finalize()); the namelist sets it
        # via parameters_prepare.extract_at. Set here too so a warm restart (prepare() skipped)
        # still has the value.
        self.extract_at = extract_at
        self.min_extraction_flattop_fraction = min_extraction_flattop_fraction

    def prepare(
        self,
        flattop_window      = 0.20,                 # To allow for steady-state in heating and current diffusion
        min_sawtooth_period_ms = 1.0,               # Adaptive Porcelli min-period floor [ms]: sets c_sawtooth(2)=floor/PM_period so crashes are >= this far apart. None disables (raw NMLtools default). Defaults to 1 ms (~historical behavior) so old namelists are unchanged
        ensure_sawtooths    = None,                 # If not None, ensure at least this many sawtooths in the simulation (if previous TRANSP beat provided this information)
        freq_ICH            = None,                 # Frequency of ICRF heating (if None, find optimal)
        extractAC           = False,                # To extract AC quantities
        transition_window   = 0.1,                  # Transition (in seconds) to move from guess TRANSP equilibrium to actual. To prevent equilibrium crashes
        currentheating_window = 0.001,
        time_before_end     = 0.001,
        machine_initialization = 'CMOD',
        machine_initialization_match_target = False,
        mxh_coeffs_smooth_sep = None,
        extract_at          = "saw-1",              # Which CDF time slice feeds the next beat: 'saw[-N]' (N slices before the last sawtooth) or 'last[-N]'
        min_extraction_flattop_fraction = 0.5,      # Floor the extraction at this fraction (0-1) of the flattop window; guards against an only-early-sawtooth plasma being sampled too soon. None disables
        rotation_source = 'echo',                   # How this beat sets w0 (the E×B rotation -c dPhi/dpsi) for the next
                                                    # input.gacode: 'echo' = feed the seed rotation into TRANSP and write
                                                    # the resulting E×B rotation (carries the seed via the ERVTOR term);
                                                    # 'neoclassical' = ship a zero omg U-File and write the weak-rotation
                                                    # (pure neoclassical) E×B rotation, ignoring any seed; 'off' = no omg
                                                    # in and zero w0 out, even if the seed carries a rotation.
        **transp_namelist
        ):
        '''
        - For letter and shot:
            Using some smart defaults to avoid repeating TRANSP runid
                shot will be 5 digits that depend on the last subfolder
                    e.g. run_cmod1 -> '94351', run_cmod2 -> '94352', run_d3d1 -> '72821', etc
                letter will depend on username in this machine, if it can be found
                    e.g. pablorf -> 'P"
        - transp_namelist is a dictionary with the keys that I want to be different from the defaults
            (mitim_tools/transp_tools/NMLtools.py: _default_params())
        '''

        # Back-compat: the legacy 'write_rotation' (bool/None) namelist key maps to
        # rotation_source ('echo'/'off'); None/True -> 'echo', False -> 'off'.
        if 'write_rotation' in transp_namelist:
            legacy = transp_namelist.pop('write_rotation')
            rotation_source = 'off' if legacy is False else 'echo'
            print(f"\t- [deprecation] 'write_rotation' is deprecated; use rotation_source. Mapped to '{rotation_source}'", typeMsg='w')
        if rotation_source not in ('echo', 'off', 'neoclassical'):
            raise ValueError(f"[MITIM] rotation_source='{rotation_source}' not recognized (use 'echo', 'off' or 'neoclassical')")

        # Grab structures
        tokamak_structures = transp_namelist.get('tokamak_structures', None)
        is_machine_fixed = tokamak_structures is not None

        # Define timings
        # Namelist documents "If null, no transition performed" — same effect as 0.0
        # (the > 0.0 guard below then skips the initialization transition)
        if transition_window is None:
            transition_window = 0.0
        self.transition_window     = transition_window
        self.rotation_source = rotation_source   # how this beat handles rotation: in (omg U-File) and out (w0)
        self.time_init = 0.0                                                # Start with a TRANSP machine equilibrium
        self.time_transition = self.time_init+ self.transition_window       # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End
        
        self._inform(ensure_sawtooths=ensure_sawtooths)
        
        self.timeAC = self.time_end - time_before_end if extractAC else None          # Time to extract TORIC and NUBEAM files

        # Which CDF time slice feeds the next beat, and an optional floor on how early we sample
        # (see finalize()).
        self.extract_at = extract_at
        self.min_extraction_flattop_fraction = min_extraction_flattop_fraction

        if mxh_coeffs_smooth_sep is None:
            print('\t- No MXH coefficients for smoothing separatrix provided', typeMsg='i')
            try:
                mxh_coeffs_smooth_sep = self.maestro_instance.maestro_namelist['plasma']['parameters']['separatrix']['n_mxh']
                print(f'\t\t- Grabbing MXH coefficients for smoothing from MAESTRO separatrix: n = {mxh_coeffs_smooth_sep}', typeMsg='i')
            except KeyError:
                print('\t\t- Proceeding without smoothing', typeMsg='i')
                pass
        else:
            print(f'\t- Using provided MXH coefficients for smoothing separatrix: n = {mxh_coeffs_smooth_sep}', typeMsg='i')
        
        # Initialize TRANSP object and profiles from input.gacode
        times = [self.time_transition,self.time_end+1.0]
        self.transp = self.profiles_current.to_transp(
            folder = self.folder,
            shot = self.shot, runid = self.runid, times = times,
            Vsurf = self.profiles_current.Vsurf,
            mxh_coeffs_smooth = mxh_coeffs_smooth_sep,
            rotation_source = rotation_source,
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generic TRANSP operation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        transp_namelist_mod = copy.deepcopy(transp_namelist)

        if 'timings' in transp_namelist_mod:
            raise ValueError('[MITIM] You cannot define timings in a MAESTRO transp_namelist!')
        else:
            transp_namelist_mod['timings'] = {
                "time_start": self.time_init,
                "time_current_diffusion": self.time_diffusion,
                "time_end": self.time_end,
                "time_extraction": self.timeAC,
            }

        if 'Ufiles' in transp_namelist_mod:
            raise ValueError('[MITIM] You cannot define UFILES in a MAESTRO transp_namelist')
        else:
            transp_namelist_mod['Ufiles'] = ["qpr","cur","vsf","ter","ti2","ner","rbz","lim","zf2", "rfs", "zfs"]

        if is_machine_fixed:
            # Remove antenna geometry that may have been written from GACODE
            for var in ['rmjicha', 'rmnicha', 'thicha']:
                del self.transp.namelist_variables[var]

        # Adaptive Porcelli sawtooth minimum-period floor. The Porcelli trigger only enforces a
        # floor relative to the Park-Monticello period (z_period_min = c_sawtooth(2) x tau_PM), and
        # tau_PM ~ R^2 Te0^1.5 / Zeff is sub-ms for compact, hot plasmas -> crashes every ~timestep
        # and a multi-GB output CDF. Pick c_sawtooth(2) so the floor lands at min_sawtooth_period_ms
        # for this plasma; large machines (tau_PM already above the floor) are left effectively
        # untouched. Set None to bypass and use the raw NMLtools c_sawtooth(2) default.
        if min_sawtooth_period_ms is not None and 'c_sawtooth_2' not in transp_namelist_mod:
            p = self.profiles_current
            tau_PM = PLASMAtools.park_monticello_sawtooth_period(
                p.profiles["rmaj(m)"][0], p.profiles["te(keV)"][0], p.profiles["z_eff(-)"][0])
            transp_namelist_mod['c_sawtooth_2'] = (min_sawtooth_period_ms * 1e-3) / tau_PM
            print(f'\t- Sawtooth floor: min_sawtooth_period_ms={min_sawtooth_period_ms} -> '
                  f'c_sawtooth(2)={transp_namelist_mod["c_sawtooth_2"]:.3g} (tau_PM={tau_PM*1e3:.3g} ms)', typeMsg='i')

        # Write namelist
        self.transp.write_namelist(**transp_namelist_mod)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Additional operations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if machine_initialization_match_target:
            modify_Ip_to_match_qstar = self.profiles_current.derived['qstar_ITER']
            modify_p_to_match_pB2 = self.profiles_current.derived['pthr_manual'][0]/self.profiles_current.derived["B0"]**2
        else:
            modify_Ip_to_match_qstar = None
            modify_p_to_match_pB2 = None
            
        self.machine_run = machine_initialization
        
        if transition_window > 0.0:
            self._additional_operations_add_initialization(
                machine_initialization = self.machine_run,
                modify_Ip_to_match_qstar=modify_Ip_to_match_qstar,
                modify_p_to_match_pB2=modify_p_to_match_pB2,
                )
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ICRF power
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if transp_namelist.get('Pich', False):

            Paux_MW    = self.profiles_current.derived['qRF_MW'][-1]

            # gaussian_sources + fmini > 0: Pich was forced on so the H minority
            # is carried as a TRANSP species AND TORIC runs at the full P_icrh
            # specified in the maestro namelist, which lets TRANSP evolve the
            # minority density and pressure self-consistently. The auxiliary
            # heating is replaced with the gaussian Pe/Pi at TRANSP output
            # (see preprocess_run_transp / force_auxiliary_heating_at_output),
            # so only the minority's profiles persist into downstream beats.
            heating_block = self.maestro_instance.maestro_namelist['plasma']['heating']
            if heating_block['type'] == 'gaussian_sources' \
               and heating_block['parameters'].get('fmini', 0.0) > 0.0:
                P_icrh_nml = heating_block['parameters'].get('P_icrh', 0.0)
                print(f'\t- gaussian_sources + fmini>0: running TORIC at P_icrh={P_icrh_nml:.3f} MW '
                      f'from namelist (qRF_MW in profiles was {Paux_MW:.3e}); '
                      f'aux heating reinserted as gaussian Pe/Pi at TRANSP output.',
                      typeMsg='i')
                Paux_MW = P_icrh_nml

            # Antennas
            nicha = self._number_of_antennas(tokamak_structures)
            
            if is_machine_fixed:
                '''
                For realistic antenna geometry, I need to use the predefined frequencies that come with the antenna definitions
                '''
                if freq_ICH is not None:
                    print(f'[MITIM] Warning: freq_ICH is defined but will be ignored since tokamak_structures = {tokamak_structures} is not None', typeMsg='w')
                    
                freq_ICH = None

            else:
                '''
                For simple antenna geometry, I can choose the frequency to match the best resonance condition for minority ions.
                If freq_ICH is not provided, I estimate it based on the on-axis (vacuum) magnetic field and the minority species.
                '''
                if freq_ICH is None:

                    B_T         = self.profiles_current.profiles['bcentr(T)'][0]

                    '''
                    Best resonance condition for minority ions
                    ------------------------------------------
                    B = (Fich * 2 * np.pi) / qm 
                    Fich_MHz = B * qm / (2 * np.pi) * 1e-6
                    qm ~ q/m * 1E8
                    Fich_MHz = B * q/m * 1E8  / (2 * np.pi) * 1e-6 ~ B * q/m * 15.0
                        e.g. He3 in SPARC: F = 12 * 2/3 * 15 = 120 MHz
                    '''

                    qm_minority = self.transp.nml_object.Minorities[0]/self.transp.nml_object.Minorities[1]
                    factor_to_account_for_Bplasma = 1.0 #1.05
                    freq_ICH = B_T * qm_minority * 15.0 * factor_to_account_for_Bplasma

            self.transp.icrf_on_time(self.time_diffusion, power_MW = Paux_MW, freq_MHz = freq_ICH, nicha = nicha)
        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NBI power
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if transp_namelist.get('Pnbi', False):

            Paux_MW    = self.profiles_current.derived['qBEAM_MW'][-1]
            
            # Beams
            nbeams, active_beams = self._number_of_beams(tokamak_structures)
            
            nbeams_active = len(active_beams)
            print(f'\t- NBI heating requested with Paux_MW = {Paux_MW:.2f} MW. Total number of beams is nbeams = {nbeams}, with active_beams = {active_beams}', typeMsg='i')

            self.transp.nbi_on_time(self.time_diffusion, power_MW = Paux_MW, nbeams = nbeams, nbeams_active = nbeams_active)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write Ufiles
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.transp.write_ufiles(mxh_coeffs_smooth = mxh_coeffs_smooth_sep, is_machine_fixed=is_machine_fixed)

    def _number_of_antennas(self, tokamak_structures):
        # Determine the number of antennas
    
        if tokamak_structures is None:
            from mitim_tools.experiment_tools.TOKtools import ICRFantennas
        elif tokamak_structures == "SPARC":
            from mitim_tools.experiment_tools.SPARCtools import ICRFantennas
        elif tokamak_structures == "CMOD":
            from mitim_tools.experiment_tools.CMODtools import ICRFantennas
        else:
            raise ValueError(f"[MITIM] tokamak_structures = {tokamak_structures} not recognized for ICRF-only mode. Supported options are: None, 'SPARC', 'CMOD'")
        transp_antenna_setup = ICRFantennas()
        
        nicha = int(re.search(r'^\s*nicha\s*=\s*(\d+)\b', transp_antenna_setup, re.M).group(1))
        
        return nicha


    def _number_of_beams(self, tokamak_structures):
        # Determine the number of beams and which are active (PINJA > 0)

        if tokamak_structures is None:
            from mitim_tools.experiment_tools.TOKtools import NBIbeams
        elif tokamak_structures == "D3D":
            from mitim_tools.experiment_tools.DIIIDtools import NBIbeams
        else:
            raise ValueError(f"[MITIM] tokamak_structures = {tokamak_structures} not recognized for NBI-only mode. Supported options are: None, D3D'")
        transp_beam_setup = NBIbeams()

        nbeam = int(re.search(r'^\s*nbeam\s*=\s*(\d+)\b', transp_beam_setup, re.M).group(1))

        pinja_matches = re.findall(r'^\s*PINJA\s*\(\s*(\d+)\s*\)\s*=\s*([\d.eE+\-]+)', transp_beam_setup, re.M | re.I)
        active_beams = [int(i) for i, p in pinja_matches if float(p) > 0]

        return nbeam, active_beams

    def run(self, **kwargs):

        mpi_settings = kwargs.get("mpisettings",{"trmpi": 32, "toricmpi": 32, "ptrmpi": 1})
        
        print('\t\t- Running TRANSP beat with MPI settings: ',mpi_settings)

        self.transp.run(
            self.machine_run,
            mpisettings = mpi_settings,
            minutesAllocation = 60*kwargs.get("hours_allocation",8),
            case = self.transp.runid,
            tokamak_name = kwargs.get("tokamak_name",None),
            checkMin = kwargs.get("checkMin",3),
            retrieveAC = self.timeAC is not None,
            )
        
        # Check if run went as expected. Sometimes it may come back "just fine" but in reality it
        # didn't run until the specified time
        print('\t\t- Checking TRANSP run completeness')
        cdf_results = CDFtools.transp_output(self.folder / f"{self.shot}{self.runid}.CDF")
        last_time_simulated = cdf_results.t[-1]
        cdf_results.close()  # release the (multi-GB) CDF fd immediately; nothing below needs it
        seconds_check = 0.1
        if last_time_simulated < self.time_end - seconds_check:   
            raise RuntimeError(f"[MITIM] TRANSP run did not complete until the expected time_end = {self.time_end:.4f} s. Last time simulated was {last_time_simulated:.4f} s.")
        else:
            print(f'\t\t- TRANSP run completed until expected time_end = {self.time_end:.4f} s. Last time simulated was {last_time_simulated:.4f} s.')

        # Release the heavy CDF handle the TRANSP wrapper cached during the run (self.transp.c, set in
        # transp_run.run -> checkUntilFinished -> storeCDF, plus any self.transp.t.cdfs). It is held for
        # the WHOLE MAESTRO run and fork-inherited by the downstream PORTALS workers, so the wiped CDF
        # lingers as a multi-GB NFS silly-rename (.nfsXXXX) until the case ends -- the dominant during-run
        # scratch hog (NOT the finalize()/completeness-check readers, which are short-lived).
        self._release_transp_cdf_handles()

    def _release_transp_cdf_handles(self):
        tr = getattr(self, "transp", None)
        if tr is None:
            return
        for obj in [getattr(tr, "c", None)] + list(getattr(getattr(tr, "t", None), "cdfs", {}).values()):
            if obj is not None and hasattr(obj, "close"):
                obj.close()

    def finalize(self, force_auxiliary_heating_at_output = None, **kwargs):

        # The multi-GB CDF is NOT copied to beat_results/. Instead, we extract the
        # small subset downstream beats need (sawtooth_times, impurity ordering)
        # into `transp_results.npy`; the CDF and AC subfolders stay in self.folder
        # and are kept or wiped according to `maestro.keep_all_files`. On a
        # re-invocation after a prior keep_all_files: false cleanup, _locate_cdf
        # returns None and folder_output already holds input.gacode and
        # transp_results.npy from the prior run.
        cdf_file = self._locate_cdf()
        if cdf_file is None:
            self.profiles_output = PROFILEStools.gacode_state(self.folder_output / 'input.gacode')
            return

        cdf_results = CDFtools.transp_output(cdf_file)

        # Which CDF time slice feeds the next beat's input.gacode (grammar in _extraction_index),
        # floored so an only-early-sawtooth plasma is not sampled before the flattop settles.
        it_extract = _extraction_index(cdf_results, self.extract_at)
        it_extract = _apply_flattop_floor(cdf_results, it_extract,
                                          getattr(self, 'time_diffusion', None),
                                          getattr(self, 'time_end', None),
                                          self.min_extraction_flattop_fraction)
        time_extraction = cdf_results.t[it_extract]
        print(f'\t\t- Extracting profiles at extract_at={self.extract_at} -> t={time_extraction:.4f}s (slice {it_extract} of {len(cdf_results.t)-1})', typeMsg='i')
        self.profiles_output = cdf_results.to_profiles(time_extraction=time_extraction)

        # Rotation OUT to the next beat. to_profiles writes w0 = TGLF_w0_exb, the E×B/potential
        # rotation (-c dPhi/dpsi, the GACODE w0 convention) -- the SAME quantity for 'echo' and
        # 'neoclassical', since both just want the E×B rotation. The two modes differ only on the
        # INPUT (handled in to_transp): 'echo' feeds the seed rotation (so ERTOT, hence the E×B
        # rotation, includes it via the ERVTOR term), 'neoclassical' ships a zero omg U-File (so it
        # is the weak-rotation / pure-neoclassical E×B). Only 'off' overrides here, zeroing w0 to
        # keep the chain rotation-free.
        _rot = getattr(self, 'rotation_source', 'echo')
        if _rot == 'off':
            self.profiles_output.profiles['w0(rad/s)'] = self.profiles_output.profiles['w0(rad/s)'] * 0.0
            self.profiles_output.derive_quantities(rederiveGeometry=False)

        # Potentially force auxiliary
        self._add_heating_profiles(force_auxiliary_heating_at_output)

        # Write profiles
        self.profiles_output.write_state(file=self.folder_output / "input.gacode")

        # Sidecar with the CDF-derived state that downstream beats need.
        profiles_species = [sp['N'] for sp in self.profiles_output.Species]
        impurity_order = OrderedDict()
        for z in cdf_results.nZs.keys():
            for i, spec in enumerate(profiles_species):
                if spec == z:
                    impurity_order[spec] = i
                    break
        np.save(self.folder_output / 'transp_results.npy', {
            'impurity_order': impurity_order,
            'sawtooth_times': np.array(cdf_results.tlastsawU),
        })

        # Close the (multi-GB) TRANSP CDF now, before the downstream PORTALS beats fork their
        # workers -- otherwise the open fd is inherited and the wiped CDF stays pinned on disk
        # (NFS silly-rename) for the whole run. See transp_output.close().
        cdf_results.close()

    def _locate_cdf(self):
        '''
        Locate the TRANSP CDF for finalize. Returns Path or None
        (None means `maestro.keep_all_files: false` cleanup ran in a prior
        invocation and folder_output already holds the summary artifacts).
        '''
        if not self.folder.exists():
            return None
        expected = self.folder / f"{self.shot}{self.runid}.CDF"
        if expected.exists():
            return expected
        # Rare: extending a prior run with different shot/runid — accept any CDF in self.folder
        for f in self.folder.iterdir():
            if f.is_file() and f.suffix.lower() == ".cdf" and not f.name.lower().endswith("ph.cdf"):
                print(f'\t\t- Using TRANSP CDF with non-matching prefix: {f.name}', typeMsg='w')
                return f
        return None

    def _add_heating_profiles(self, force_auxiliary_heating_at_output = None):
        '''
        force_auxiliary_heating_at_output['Pe'] has the shaping function (takes rho) and the integrated value
        '''
        if force_auxiliary_heating_at_output is None:
            force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None}


        for key, pkey, ikey in zip(['Pe','Pi'], ['qrfe(MW/m^3)', 'qrfi(MW/m^3)'], ['qRFe_MW', 'qRFi_MW']):

            if force_auxiliary_heating_at_output[key] is not None:
                print(f'\t\t- Adding {key} = {force_auxiliary_heating_at_output[key][1]} MW of power')
                self.profiles_output.profiles[pkey] = force_auxiliary_heating_at_output[key][0](self.profiles_output.profiles['rho(-)'])
                self.profiles_output.derive_quantities()
                self.profiles_output.profiles[pkey] = self.profiles_output.profiles[pkey] *  force_auxiliary_heating_at_output[key][1]/self.profiles_output.derived[ikey][-1]
            else:
                print(f'\t\t- Keeping auxiliary power from TRANSP output')

    def _rescale_aux_to_frozen(self, p_frozen, ekey, ikey, totkey, prescribed_at_output=False):
        '''
        Renormalize an auxiliary-power channel pair (electron ekey + ion ikey) so its volume
        integral equals the frozen engineering total (totkey). This preserves the SHAPE that
        TRANSP/TORIC produced while pinning the MAGNITUDE to the engineering spec (Pin), which
        MAESTRO treats as a fixed input carried unchanged from beat to beat. See the block comment
        in merge_parameters for the ICRH/NBI vs gaussian_sources rationale.

        When the TRANSP output has essentially no power in this channel (|total| <= 5e-6, or
        negative) we CANNOT rescale: dividing by ~0 gives 0/0 or 0*inf = NaN. Two behaviors:

          - ICRH/NBI (prescribed_at_output=False, default): the frozen engineering power must
            survive, but a soft TRANSP beat (Pich/Pnbi off) -- or a restart that reloaded a
            previously-zeroed input.gacode (qrfe == 0 exactly) -- leaves nothing to scale up. So
            insert the frozen deposition profile directly. (self.profiles_output was already
            re-gridded to p_frozen's resolution earlier in merge_parameters, so shapes match.)

          - gaussian_sources (prescribed_at_output=True): the Pe/Pi gaussians are written straight
            into qrfe/qrfi at TRANSP output, so ~0 genuinely means "no power here" -> zero it
            (Kaitlyn's 5b02314b guard).

        Note: the totals are read once on entry; `derived` is not recomputed mid-call, so both
        channels scale by the same factor (the new integral is recomputed later by the caller's
        derive/write path).
        '''
        out_total = self.profiles_output.derived[totkey][-1]

        if out_total < 0 or abs(out_total) <= 5e-6:
            if prescribed_at_output:
                # gaussian_sources: ~0 genuinely means no prescribed power in this channel -> zero it.
                self.profiles_output.profiles[ekey] = np.zeros_like(self.profiles_output.profiles[ekey])
                self.profiles_output.profiles[ikey] = np.zeros_like(self.profiles_output.profiles[ikey])
            else:
                # ICRH/NBI: a zero output cannot be scaled up, but the frozen engineering power must
                # survive. Insert the frozen deposition shape, then renormalize it to the frozen total
                # over THIS beat's geometry, so the engineering power is preserved exactly and does not
                # drift across repeated soft beats. The frozen shape is nonzero, so this divide is safe.
                self.profiles_output.profiles[ekey] = p_frozen.profiles[ekey].copy()
                self.profiles_output.profiles[ikey] = p_frozen.profiles[ikey].copy()
                self.profiles_output.derive_quantities()
                inserted_total = self.profiles_output.derived[totkey][-1]
                if abs(inserted_total) > 5e-6:
                    r = p_frozen.derived[totkey][-1] / inserted_total
                    self.profiles_output.profiles[ekey] *= r
                    self.profiles_output.profiles[ikey] *= r
            return

        ratio = p_frozen.derived[totkey][-1] / out_total
        self.profiles_output.profiles[ekey] *= ratio
        self.profiles_output.profiles[ikey] *= ratio

    def merge_parameters(self):
        '''
        The goal of the TRANSP beat is to produce:
            - Internal GS equilibrium
            - q-profile
            - Power deposition profiles of high quality (auxiliary heating, but also dynamic targets)
            - Species and fast ions
        However, TRANSP is not modifying the kinetic profiles and therefore I should use the profiles that were frozen before, to
        avoid "grid leaks", i.e. from beat to beat, the coarse grid interpolates to point to point.
        So, this merge:
            - Brings back the resolution of the frozen profiles
            - Inserts kinetic profiles from frozen
            - Inserts engineering parameters (Ip, Bt) from frozen
            - Scales power deposition profiles to match the frozen power deposition which I treat as an engineering parameter (Pin)
        '''

        # Write the pre-merge input.gacode before modifying it
        profiles_output_pre_merge = copy.deepcopy(self.profiles_output)
        profiles_output_pre_merge.write_state(file=self.folder_output / 'input.gacode_pre_merge')

        # First, bring back to the resolution of the frozen
        p_frozen = self.maestro_instance.profiles_with_engineering_parameters
        print('\t\t\t* Bringing resolution of frozen plasma state to new plasma state')
        self.profiles_output.changeResolution(rho_new = p_frozen.profiles['rho(-)'])

        # --------------------------------------------------------------------------------------------
        # Insert relevant quantities
        # --------------------------------------------------------------------------------------------

        # Insert kinetic profiles from frozen
        print('\t\t\t* Bringing kinetic profiles of frozen plasma state to new plasma state')
        self.profiles_output.profiles['ne(10^19/m^3)'] = p_frozen.profiles['ne(10^19/m^3)']
        self.profiles_output.profiles['te(keV)'] = p_frozen.profiles['te(keV)']
        self.profiles_output.profiles['ti(keV)'][:,0] = p_frozen.profiles['ti(keV)'][:,0]

        self.profiles_output.makeAllThermalIonsHaveSameTemp()
        profiles_output_pre_merge.changeResolution(rho_new = p_frozen.profiles['rho(-)'])
        self.profiles_output.scaleAllThermalDensities(scaleFactor = self.profiles_output.profiles['ne(10^19/m^3)']/profiles_output_pre_merge.profiles['ne(10^19/m^3)'])

        # Insert engineering parameters (except shape)
        print('\t\t\t* Bringing Bt and Ip of frozen plasma state to new plasma state')
        for key in ['current(MA)', 'bcentr(T)']:
            self.profiles_output.profiles[key] = p_frozen.profiles[key]

        # ------------------------------------------------------------------------------------------------
        # Power scale
        # ------------------------------------------------------------------------------------------------
        # The TOTAL auxiliary power is treated as a frozen *engineering* input (Pin), on the same footing
        # as R, a, Bt, Ip. TRANSP/TORIC produces a deposition whose integral can drift from the requested
        # value (numerics, grid resolution, or a "soft" beat that ran no heating at all), but MAESTRO wants
        # the power that flows into the next beat to be EXACTLY the engineering spec. So we keep the SHAPE
        # TRANSP produced and renormalize its INTEGRAL to the frozen total (see _rescale_aux_to_frozen).
        #
        # This matters beyond the current beat: right after this merge, MAESTROmain._freeze_parameters
        # re-snapshots the whole profile, so whatever total survives here becomes the frozen reference the
        # NEXT beat reads (a TRANSP beat sets its antenna/beam power from the incoming qRF/qBEAM, line ~158).
        # A wrong total therefore propagates forward, not just into this beat.
        #
        # ICRH/NBI and gaussian_sources need OPPOSITE handling on the first beat, which is why we branch:
        #
        #   ICRH/NBI: the creator deposits Paux=P_icrh into qrfe/qrfi (or qbeam*) at INITIALIZATION
        #       (aux_channels set), so frozen.qRF/qBEAM already holds the engineering value before any beat.
        #       => rescale to frozen on EVERY beat. A soft beat (Pich/Pnbi off) leaves only a tiny nonzero
        #       residual; rescaling lifts that residual's integral back to the engineering value. The shape
        #       is then meaningless, but the only thing the next beat consumes is the scalar total, and the
        #       next real TRANSP beat re-runs TORIC to recover a physical shape. Skipping the rescale here is
        #       what caused "no ICRF in later transp beats": the ~0 got re-frozen and never recovered.
        #
        #   gaussian_sources: the creator uses aux_channels=None, so frozen.qRF STARTS AT 0; the prescribed
        #       Pe/Pi gaussians are injected only at TRANSP OUTPUT (_add_heating_profiles overwrites qrfe/qrfi
        #       with a parabola normalized to Pe/Pi). On the first beat, rescaling output(Pe+Pi) to frozen(0)
        #       would WIPE the gaussian -> so we must NOT rescale the first beat; it becomes the frozen
        #       reference (Pe+Pi) for subsequent beats, where the rescale ratio is ~1. zero_if_small guards
        #       the genuine 0/0 case. (This is Kaitlyn's 5b02314b fix, now scoped to gaussian_sources only.)
        # ------------------------------------------------------------------------------------------------
        heating_type = self.maestro_instance.maestro_namelist['plasma']['heating']['type']

        if heating_type == 'gaussian_sources':
            if self.maestro_instance.counter_current == 1:  # TODO: key on the first TRANSP beat that actually heated, not just the first beat
                print('\t\t\t* gaussian_sources, first beat: NOT rescaling auxiliary power to frozen (it defines the reference)')
            else:
                print('\t\t\t* gaussian_sources: rescaling auxiliary power to frozen')
                self._rescale_aux_to_frozen(p_frozen, 'qrfe(MW/m^3)',   'qrfi(MW/m^3)',   'qRF_MW',   prescribed_at_output=True)
                self._rescale_aux_to_frozen(p_frozen, 'qbeame(MW/m^3)', 'qbeami(MW/m^3)', 'qBEAM_MW', prescribed_at_output=True)
        else:
            # ICRH / NBI
            print('\t\t\t* Bringing total power of frozen plasma state to new plasma state (scaling the profile)')
            self._rescale_aux_to_frozen(p_frozen, 'qrfe(MW/m^3)',   'qrfi(MW/m^3)',   'qRF_MW')
            self._rescale_aux_to_frozen(p_frozen, 'qbeame(MW/m^3)', 'qbeami(MW/m^3)', 'qBEAM_MW')

        # --------------------------------------------------------------------------------------------

        # Write to final input.gacode
        self.profiles_output.derive_quantities()
        self.profiles_output.write_state(file=self.folder_output / 'input.gacode')

    def grab_output(self, read_ac=False, **kwargs):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:
            # CDF lives in self.folder when `keep_all_files: true`; wiped under false.
            # Pre-change MAESTRO folders may still have it in folder_output. Try both;
            # return c=None if neither has it (TRANSPbeat.plot already guards on that).
            if (self.folder / f"{self.shot}{self.runid}.CDF").exists():
                c = CDFtools.transp_output(self.folder, readFBM=read_ac, readTORIC=read_ac)
            elif (self.folder_output / f"{self.shot}{self.runid}.CDF").exists():
                c = CDFtools.transp_output(self.folder_output, readFBM=read_ac, readTORIC=read_ac)
            else:
                c = None
            profiles = PROFILEStools.gacode_state(self.folder_output / 'input.gacode') if (self.folder_output / 'input.gacode').exists() else None
        else:
            # Trying to see if there's an intermediate CDF in folder
            print('\t\t- Searching for intermediate CDF in folder')
            try:
                c = CDFtools.transp_output(self.folder)
            except ValueError:
                c = None
            profiles = None

        return c, profiles

    def plot(self,  fn = None, counter = 0, **kwargs):

        c, _ = self.grab_output(read_ac=True)

        if c is None:
            return '\t\t- Cannot plot because the TRANSP beat has not finished yet'

        c.plot(fn = fn, tab_color = counter)

        return '\t\t- Plotting of TRANSP beat done'

    def summary(self, output_dir, counter = None, wall_time_s = None):
        '''
        Markdown section for the last TRANSP beat: q-profile + power balance
        + sawtooth count + wall-time + a small profile snapshot figure.
        Power balance prefers the CDF if still on disk; otherwise falls back
        to input.gacode integrals.
        '''
        import matplotlib.pyplot as plt

        gacode_file = self.folder_output / 'input.gacode'
        if not gacode_file.exists():
            return f'## TRANSP\n*(input.gacode missing in {gacode_file.parent}; no summary available)*\n'

        profiles = PROFILEStools.gacode_state(gacode_file)
        profiles.derive_quantities()
        derived = profiles.derived

        rho = profiles.profiles['rho(-)']
        q = profiles.profiles['q(-)']

        # q-scalars: prefer derived (q0, q95 computed against psi_pol, the canonical
        # MITIM convention). qmin isn't in derived, so compute it locally.
        q0 = float(derived['q0'])
        q95 = float(derived['q95'])
        qmin = float(np.min(q))

        # Power balance from derived quantities (volume integrals at extraction time)
        def _last_or_nan(key):
            arr = derived.get(key)
            if arr is None:
                return float('nan')
            try:
                return float(np.asarray(arr).flat[-1])
            except (IndexError, TypeError, ValueError):
                try:
                    return float(arr)
                except (TypeError, ValueError):
                    return float('nan')

        def _scalar_or_nan(key):
            val = derived.get(key)
            if val is None:
                return float('nan')
            try:
                return float(val)
            except (TypeError, ValueError):
                return float('nan')

        P_fus = _scalar_or_nan('Pfus')
        P_RF  = _last_or_nan('qRF_MW')
        P_NBI = _last_or_nan('qBEAM_MW')
        P_aux = (0.0 if np.isnan(P_RF) else P_RF) + (0.0 if np.isnan(P_NBI) else P_NBI)
        if np.isnan(P_RF) and np.isnan(P_NBI):
            P_aux = float('nan')
        P_OH  = _last_or_nan('qOhm_MW')
        P_rad = _scalar_or_nan('Prad')

        # Sawtooth count from sidecar
        sawtooth_count = None
        sidecar = self.folder_output / 'transp_results.npy'
        if sidecar.exists():
            try:
                d = np.load(sidecar, allow_pickle=True).item()
                st = d.get('sawtooth_times')
                if st is not None:
                    sawtooth_count = int(np.size(st))
            except Exception:
                pass

        # Profile snapshot figure: three panels side by side —
        #   (1) kinetic profiles Te+Ti+ne (twin y-axis for ne)
        #   (2) power sources (all on a single MW/m^3 axis)
        #   (3) q-profile alone
        png_name = 'transp_profiles.png'
        png_path = output_dir / png_name
        try:
            fig, (ax_kin, ax_pow, ax_q) = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

            # ---- Panel 1: kinetic profiles (Te + Ti on keV axis, ne on twin)
            te = profiles.profiles['te(keV)']
            ti = profiles.profiles['ti(keV)'][:, 0]
            ne20 = profiles.profiles['ne(10^19/m^3)'] * 0.1

            l1, = ax_kin.plot(rho, te, color='tab:red',    lw=2,         label=r'$T_e$ [keV]')
            l2, = ax_kin.plot(rho, ti, color='tab:orange', lw=2, ls='--', label=r'$T_i$ [keV]')
            ax_kin.set_xlabel(r'$\rho$')
            ax_kin.set_ylabel(r'$T$ [keV]', color='tab:red')
            ax_kin.tick_params(axis='y', colors='tab:red')

            ax_ne = ax_kin.twinx()
            l3, = ax_ne.plot(rho, ne20, color='tab:green', lw=2, label=r'$n_e$ [$10^{20}\,m^{-3}$]')
            ax_ne.set_ylabel(r'$n_e$ [$10^{20}\,m^{-3}$]', color='tab:green')
            ax_ne.tick_params(axis='y', colors='tab:green')

            ax_kin.legend(handles=[l1, l2, l3], loc='upper right', fontsize=9, framealpha=0.9)
            ax_kin.set_xlim(0, 1)
            ax_kin.grid(True, alpha=0.3)
            ax_kin.set_title('Kinetic profiles')

            # ---- Panel 2: power densities (all on the same MW/m^3 axis)
            POWER_CHANNELS = [
                ('qrfe(MW/m^3)',   r'$q_{RF,e}$',     'tab:blue',   '-'),
                ('qrfi(MW/m^3)',   r'$q_{RF,i}$',     'tab:cyan',   '--'),
                ('qbeame(MW/m^3)', r'$q_{NBI,e}$',    'tab:purple', '-'),
                ('qbeami(MW/m^3)', r'$q_{NBI,i}$',    'tab:pink',   '--'),
                ('qfuse(MW/m^3)',  r'$q_{\alpha,e}$', 'tab:olive',  '-'),
                ('qfusi(MW/m^3)',  r'$q_{\alpha,i}$', 'tab:brown',  '--'),
                ('qohme(MW/m^3)',  r'$q_{Ohm}$',      'gray',       ':'),
            ]
            any_power = False
            for key, label, color, ls in POWER_CHANNELS:
                if key not in profiles.profiles:
                    continue
                arr = np.asarray(profiles.profiles[key])
                if not np.any(np.abs(arr) > 1e-6):
                    continue
                ax_pow.plot(rho, arr, color=color, lw=1.5, ls=ls, label=label)
                any_power = True
            ax_pow.set_xlabel(r'$\rho$')
            ax_pow.set_ylabel(r'Power density [MW/$m^3$]')
            ax_pow.set_xlim(0, 1)
            ax_pow.grid(True, alpha=0.3)
            ax_pow.set_title('Source profiles')
            if any_power:
                ax_pow.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
            else:
                ax_pow.text(0.5, 0.5, '(no non-zero power channels)',
                            transform=ax_pow.transAxes, ha='center', va='center', color='gray')

            # ---- Panel 3: q-profile alone
            ax_q.plot(rho, q, color='k', lw=2)
            ax_q.axhline(1.0, color='gray', ls='--', lw=0.8, label=r'$q=1$')
            ax_q.set_xlabel(r'$\rho$')
            ax_q.set_ylabel(r'$q$')
            ax_q.set_xlim(0, 1)
            ax_q.grid(True, alpha=0.3)
            ax_q.set_title(r'$q$-profile')
            ax_q.legend(loc='best', fontsize=9)

            fig.tight_layout()
            fig.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            fig_md = f'\n![TRANSP profiles]({png_name})\n'
        except Exception as e:
            fig_md = f'\n*(profile snapshot unavailable: {e})*\n'

        # Compose markdown
        header_extra = f' (Beat {counter})' if counter is not None else ''
        lines = [f'## TRANSP{header_extra}', '']
        lines.append('### q-profile')
        lines.append('')
        lines.append('| Quantity | Value |')
        lines.append('|---|---|')
        lines.append(f'| q0 | {q0:.3f} |')
        lines.append(f'| qmin | {qmin:.3f} |')
        lines.append(f'| q95 | {q95:.3f} |')
        lines.append('')
        lines.append('### Power balance (input.gacode volume integrals at extraction time)')
        lines.append('')
        lines.append('| Quantity | Value [MW] |')
        lines.append('|---|---|')
        for label, val in [('P_fus', P_fus), ('P_aux', P_aux), ('P_OH', P_OH), ('P_rad', P_rad)]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                lines.append(f'| {label} | n/a |')
            else:
                lines.append(f'| {label} | {val:.3f} |')
        lines.append('')
        lines.append('### Sawteeth & timing')
        lines.append('')
        lines.append('| Quantity | Value |')
        lines.append('|---|---|')
        lines.append(f'| Sawtooth count | {sawtooth_count if sawtooth_count is not None else "n/a"} |')
        lines.append(f'| Run wall-time | {_format_seconds(wall_time_s) if wall_time_s is not None else "n/a"} |')
        lines.append('')
        lines.append(fig_md)
        return '\n'.join(lines)

    # --------------------------------------------------------------------------------------------
    # Additional TRANSP utilities
    # --------------------------------------------------------------------------------------------

    def _additional_operations_add_initialization(self, machine_initialization = 'CMOD', modify_Ip_to_match_qstar = None, modify_p_to_match_pB2=None):
        '''
        ----------------------------------------------------------------------------------------------------------------------
        TRANSP must be initialized with a specific machine, so here I use the trick of changing the equilibrium and parameters
        in time, to make a smooth transition and avoid equilibrium crashes (e.g. quval error)
        ----------------------------------------------------------------------------------------------------------------------
        '''

        if machine_initialization == 'D3D':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 1.67, 0.6, 1.75, 0.38, 0.0, 0.0, 0.074, 1.6, 2.0, 1.0
        elif machine_initialization == 'CMOD':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 0.68, 0.22, 1.5, 0.46, 0.0, 0.0, 0.3, 1.0, 5.4, 1.0
        elif machine_initialization == 'NSTX':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 0.89, 0.61, 2.5, 0.46, 0.0, 0.0, 0.4, 1.0, 0.5, 1.0
            # says it has no psi-bndry

        if modify_Ip_to_match_qstar is not None:
            qstar_now = PLASMAtools.evaluate_qstar(
                Ip_MA,
                R,
                kappa_sep*0.95,
                B_T,
                a/R,
                delta_sep*0.95,
                isInputIp=True,
                ITERcorrection=True,
                includeShaping=True,
            )
            factor_Ip = qstar_now / modify_Ip_to_match_qstar
            
            print(f'\t\t- Modifying Ip of initialization machine from {Ip_MA:.3f} MA to {Ip_MA*factor_Ip:.3f} MA to match target qstar = {modify_Ip_to_match_qstar:.3f} (original qstar was {qstar_now:.3f})')
            Ip_MA = Ip_MA * factor_Ip
            
        if modify_p_to_match_pB2 is not None:
            beta_now = p0_MPa / B_T**2
            factor_p = beta_now / modify_p_to_match_pB2
            
            print(f'\t\t- Modifying p0 of initialization machine from {p0_MPa:.3f} MPa to {p0_MPa/factor_p:.3f} MPa to match target p/B^2 = {modify_p_to_match_pB2:.3f} (original p/B^2 was {beta_now:.3f})')
            p0_MPa = p0_MPa / factor_p

        self.transp.populate_time.from_freegs(self.time_init, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = ne0_20)

    # -----------------------------------------------------------------------------------------------------------------------
    # MAESTRO interface
    # -----------------------------------------------------------------------------------------------------------------------
    def _inform_save(self, *args, **kwargs):

        summary_file = self.folder_output / 'transp_results.npy'

        if summary_file.exists():
            transp_results = np.load(summary_file, allow_pickle=True).item()
            impurity_order = transp_results['impurity_order']
            sawtooth_times = transp_results['sawtooth_times']
        else:
            # Backwards-compat: MAESTRO folders that pre-date the sidecar still have
            # the CDF in folder_output (legacy copy) or in self.folder (keep_all_files: true).
            c, _ = self.grab_output()
            profiles_species = [sp['N'] for sp in self.profiles_output.Species]
            impurity_order = OrderedDict()
            for z in c.nZs.keys():
                for i, spec in enumerate(profiles_species):
                    if spec == z:
                        impurity_order[spec] = i
                        break
            sawtooth_times = np.array(c.tlastsawU)

        self.maestro_instance.parameters_trans_beat['impurity_order_transp'] = impurity_order
        # If I have run TRANSP, I cannot reuse surrogate data #TODO: Maybe not always true?
        self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] = None
        self.maestro_instance.parameters_trans_beat['sawtooth_times'] = sawtooth_times
        
    def _inform(self, ensure_sawtooths=None):
        
        time_end_minimum = 0.0
        if "sawtooth_times" in self.maestro_instance.parameters_trans_beat and ensure_sawtooths is not None:
            
            time_end_minimum = self._determine_minimum_time(ensure_sawtooths=ensure_sawtooths)
                    
            if self.time_end < time_end_minimum:
                print(f'\t- Extending TRANSP simulation time_end from {self.time_end:.4f} s to {time_end_minimum:.4f} s to ensure at least {ensure_sawtooths} sawtooths (estimate)', typeMsg='i')
                    
        self.time_end = max(self.time_end, time_end_minimum)
             
        if 'lowZ_impurity' in self.maestro_instance.parameters_trans_beat:
            print(f'\t- Lengyel provided low-Z impurity Z={self.maestro_instance.parameters_trans_beat["lowZ_impurity"]} to TRANSP beat, but MAESTRO is not ready to accept that yet, it is recommended to run a Lengyel beat after a TRANSP beat', typeMsg='w')
             
    def _determine_minimum_time(self, ensure_sawtooths=None):
            
        # ---------------------------------------------------------------------------------------
        # Sawtooth handling
        # ---------------------------------------------------------------------------------------
            
        sawtooth_times = self.maestro_instance.parameters_trans_beat['sawtooth_times']

        # No sawteeth in the previous run (e.g. q0 > 1): there is no period to
        # estimate from, so impose no minimum (the caller keeps its time_end)
        if len(sawtooth_times) == 0:
            print('\t- Previous TRANSP run had no sawtooth crashes; cannot estimate a minimum time from the sawtooth period', typeMsg='w')
            return 0.0

        # If simulation already has enough sawtooths, ensure it has at least that duration
        if len(sawtooth_times) >= ensure_sawtooths:
            time_end_minimum = sawtooth_times[-1]
        else:
            howmany_missing = ensure_sawtooths - len(sawtooth_times)
            # If the simulation had more than one sawtooth, estimate the period of the last
            if len(sawtooth_times) >= 2:
                last_period = sawtooth_times[-1] - sawtooth_times[-2]
                time_end_minimum = sawtooth_times[-1] + howmany_missing * last_period * 1.1 # Overestimation factor of 1.1
            # If the simulation only had one sawtooth, estimate the period assuming the first "sawtooth" was at t=0
            else:
                last_period = sawtooth_times[-1] - 0.0
                time_end_minimum = sawtooth_times[-1] + howmany_missing * last_period * 1.5 # Overestimation factor of 1.5

        # ---------------------------------------------------------------------------------------
        # Fast particle transport
        # ---------------------------------------------------------------------------------------
        
        #TODO
        
        return time_end_minimum
             
# -----------------------------------------------------------------------------------------------------------------------
# Defaults to help MAESTRO
# -----------------------------------------------------------------------------------------------------------------------

def preprocess_prepare_transp(transp_namelist,maestro_namelist, preprocess_prepare_parameters):
    
    print('\t- Preprocessing settings for TRANSP beat')
    
    # Minority
    Zmini = maestro_namelist["plasma"]["heating"]["parameters"]["minority"][0]
    Amini = maestro_namelist["plasma"]["heating"]["parameters"]["minority"][1]
    fmini = maestro_namelist["plasma"]["heating"]["parameters"]["fmini"]

    # Only correct Pich from the maestro namelist if it's not already False
    # ICRH is on when heating.type == 'ICRH' AND P_icrh > 0. As a special case,
    # heating.type == 'gaussian_sources' with fmini > 0 also forces Pich on so
    # the H minority is added as a TRANSP species for correct dilution accounting;
    # the run-time ICRF power is floored to a negligible value in that branch.
    heating_type = maestro_namelist["plasma"]['heating']['type']
    icrh_active = heating_type == 'ICRH' and \
                  maestro_namelist["plasma"]['heating']['parameters']['P_icrh'] > 0.0
    minority_only = heating_type == 'gaussian_sources' and fmini > 0.0
    if transp_namelist.get('Pich', False):
        transp_namelist['Pich'] = icrh_active or minority_only
    
    if transp_namelist.get('Pnbi', False):
        transp_namelist['Pnbi'] =   maestro_namelist["plasma"]['heating']['type'] == 'NBI' and \
                                    maestro_namelist["plasma"]['heating']['parameters']['P_nbi'] > 0.0
    
    if transp_namelist['Pich']:
        transp_namelist['Minorities'] = [ Zmini, Amini, fmini ]
        transp_namelist['freq_ICH'] = maestro_namelist["plasma"]['heating']['parameters']['freq_ICH']

    # Grab Z and A of high-Z
    import periodictable as pt
    e = pt.elements.symbol(maestro_namelist["plasma"]["species"]["mix"]["highZ"])
    highZ = e.number
    highA = e.mass    
    # ------ 

    LowZ, Wratio = PLASMAtools.estimateLowZ(
        maestro_namelist["plasma"]["species"]["mix"]["fmain"],
        maestro_namelist["plasma"]["species"]["Zeff"],
        Zmini,
        fmini,
        maestro_namelist["plasma"]["species"]["mix"]["CShighZ_estimate"],
        maestro_namelist["plasma"]["species"]["mix"]["fhighZ"] )
    
    lowA = 2*LowZ   # Approximation
    
    transp_namelist["zlump"] =[  [ highZ, highA, 0.1*Wratio ],
                                 [  LowZ,  lowA, 0.1        ] ]

    transp_namelist['DTplasma'] = maestro_namelist["plasma"]["species"]['fuel'] == ['D', 'T']   #TODO: generalize TRANSP module
    
    return transp_namelist

def preprocess_run_transp(run_namelist, maestro_namelist, cpus, cold_start):
    
    toric = maestro_namelist["maestro"]["transp"]["parameters_prepare"]["Pich"]
    nubeam = maestro_namelist["maestro"]["transp"]["parameters_prepare"]["useNUBEAMforAlphas"]
    cpus_toric = maestro_namelist["maestro"]["transp"]["preprocess_prepare_parameters"]["cpus_toric"]
    cpus_nubeam = maestro_namelist["maestro"]["transp"]["preprocess_prepare_parameters"]["cpus_nubeam"]
    
    if toric:
        toricmpi = cpus_toric if cpus_toric is not None else cpus
    else:
        toricmpi = 1
        
    if nubeam:
        trmpi = cpus_nubeam if cpus_nubeam is not None else cpus
    else:
        trmpi = 1
    
    # Force auxiliary heating at output
    if maestro_namelist["plasma"]["heating"]["type"] == 'gaussian_sources':

        print('\t- Gaussian sources specified, adding to run_namelist of TRANSP beat')
        
        Pe = maestro_namelist["plasma"]["heating"]["parameters"]["Pe"]
        Pi = maestro_namelist["plasma"]["heating"]["parameters"]["Pi"]
        nu_source = maestro_namelist["plasma"]["heating"]["parameters"]["nu_source"]

        def P_auxiliary(rhotor):
            _, y = PLASMAtools.parabolicProfile(Tbar=1.0,nu=nu_source,rho=rhotor,Tedge=0.0)
            return y
    
        force_auxiliary_heating_at_output = {
            'Pe': [P_auxiliary, Pe],
            'Pi': [P_auxiliary, Pi],
            }
        
    else:
        force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None}
        
    run_namelist['mpisettings'] = {
        "trmpi": trmpi, 
        "toricmpi": toricmpi,
        "ptrmpi": 1
        }

    run_namelist['force_auxiliary_heating_at_output'] = force_auxiliary_heating_at_output
    
    return run_namelist