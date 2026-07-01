import shutil
import copy
import pandas as pd
from functools import partial
import numpy as np
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis, PORTALSoptimization
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat, _format_seconds
from IPython import embed
from mitim_tools import __mitimroot__

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

class portals_beat(beat):

    def __init__(self, maestro_instance):
        super().__init__(maestro_instance, beat_name = 'portals')
        self.initialization_parameters = {}

    def prepare_minimal(self, *args, initialization_parameters = None, **kwargs):
        # Skip-path stash: merge_parameters() re-applies zero_source_blocks from
        # initialization_parameters on every invocation, but prepare() (which sets it)
        # is skipped for a completed beat — leaving {} and silently un-zeroing the
        # sources in the re-merged outputs.
        self.initialization_parameters = initialization_parameters if initialization_parameters is not None else {}

    def prepare(self,
            use_previous_residual = True,
            use_previous_surrogate_data = True,
            use_previous_ranges = True,
            try_flux_match_only_for_first_point = True,
            change_last_radial_call = True,
            portals_namelist_location = None,
            portals_parameters = None,
            initialization_parameters = None,
            enforce_impurity_radiation_existence = True,
            ):
        
        if portals_parameters is None:
            portals_parameters = {}
        if initialization_parameters is None:
            initialization_parameters = {}


        self.fileGACODE = self.initialize.folder / 'input.gacode'

        if enforce_impurity_radiation_existence:

            profiles = self.profiles_current
            for i in range(len(profiles.Species)):
                data_df = pd.read_csv(__mitimroot__ / "src" / "mitim_modules" / "powertorch" / "physics_models" / "radiation_chebyshev.csv")
                if not (data_df['Ion'].str.lower()==profiles.Species[i]["N"].lower()).any():
                    print(f"\t\t- '{profiles.Species[i]['N']}' species not found in radiation table, looking for closest Z (+- 5) using the Z specified in the input.gacode (fully stripped assumption)",typeMsg='w')
                    # Find closest Z
                    Z = data_df['Z'].to_numpy()
                    iZ = np.argmin(abs(Z - profiles.Species[i]["Z"]))

                    if abs(Z[iZ] - profiles.Species[i]["Z"]) > 5:
                        print(f"\t\t- {profiles.Species[i]['N']} not found in radiation table, closest Z is {Z[iZ]} but not close enough",typeMsg='q')

                    new_name = data_df['Ion'][iZ]

                    print(f'\t\t\t- Changing name of ion from {profiles.Species[i]["N"]} ({profiles.Species[i]["Z"]}) to {new_name} ({Z[iZ]})')

                    profiles.profiles['name'][i] = profiles.Species[i]["N"] = new_name

            self.profiles_current = profiles


        self.profiles_current.write_state(file = self.fileGACODE)

        self.portals_parameters = portals_parameters
        self.portals_namelist_location = portals_namelist_location
        self.initialization_parameters = initialization_parameters

        self.use_previous_residual = use_previous_residual
        self.use_previous_surrogate_data = use_previous_surrogate_data
        self.change_last_radial_call = change_last_radial_call
        self.use_previous_ranges = use_previous_ranges

        self.try_flux_match_only_for_first_point = try_flux_match_only_for_first_point


        # Initializat optimization options to empty, but may be filled in _inform, from previous beats information
        self.optimization_options_additional = {}

        self._inform(use_previous_residual = self.use_previous_residual, 
                     use_previous_surrogate_data = self.use_previous_surrogate_data,
                     change_last_radial_call = self.change_last_radial_call,
                     use_previous_ranges = self.use_previous_ranges
                     )

    def run(self, **kwargs):

        cold_start = kwargs.get('cold_start', False)
        ENABLE_EMBED = kwargs.get('ENABLE_EMBED', False)

        # Read the namelist if explicitly given in the MAESTRO namelist (variable: portals_namelist_location)
        portals_fun  = PORTALSmain.portals(self.folder, portals_namelist = self.portals_namelist_location)

        # Update the namelist with the parameters in the MAESTRO namelist (variable: portals_parameters)
        portals_fun.portals_parameters = IOtools.deep_dict_update(portals_fun.portals_parameters, self.portals_parameters)
        if 'optimization_options' in self.portals_parameters:
            portals_fun.portals_parameters['optimization_options'] = portals_fun.optimization_options = IOtools.deep_dict_update(portals_fun.optimization_options, self.portals_parameters['optimization_options'])
        
        # MAESTRO beat may receive optimization options changes from previous beats (via _inform() inside prepare), so allow that too
        portals_fun.portals_parameters['optimization_options'] = portals_fun.optimization_options = IOtools.deep_dict_update(portals_fun.optimization_options, self.optimization_options_additional)

        # Initialization now happens by the user
        from mitim_tools.gacode_tools.PROFILEStools import gacode_state
        p = gacode_state(self.fileGACODE)
        p.correct(options=self.initialization_parameters)

        portals_fun.prep(p,askQuestions=False)

        self.mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, seed = self.maestro_instance.master_seed, cold_start = cold_start, askQuestions = False, ENABLE_EMBED=ENABLE_EMBED)

        if self.use_previous_surrogate_data and \
            self.try_flux_match_only_for_first_point and \
            self.folder_starting_point is not None and \
            ('portals_surrogate_data_file' in self.maestro_instance.parameters_trans_beat) and \
            self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] is not None:

            # Warm-start: seed the run with a single flux-matched point (against the previous beat's
            # surrogate) plus one training point. If a seed already exists (a restart of THIS beat)
            # keep it; otherwise try to produce one. _flux_match_for_first_point() returns False when
            # the anchor surrogate is slim/pruned (keep_all_files: false -- e.g. appending beats onto a
            # finished MAESTRO, whose last beat was slimmed): then fall back to the normal
            # initialization (surrogate DATA reuse via extrapointsFile = surrogate_data.csv is
            # independent and still applies), so an appended beat never crashes on a pruned anchor.
            have_seed = len(self.mitim_bo.optimization_data.data) > 0
            if not have_seed:
                have_seed = self._flux_match_for_first_point()

            if have_seed:
                # PORTALS with just one extra training point on top of the flux-matched seed
                portals_fun.optimization_options['initialization_options']['initial_training'] = 1

                portals_fun.prep(p,askQuestions=False)

                self.mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, seed=self.maestro_instance.master_seed,cold_start = cold_start, askQuestions = False)

        self.mitim_bo.run()

    def _flux_match_for_first_point(self):
        '''Seed this beat's first evaluation by flux-matching against the previous PORTALS beat's
        converged surrogate (self.folder_starting_point). Returns True if a flux-matched point was
        produced, False if the anchor surrogate is unavailable.

        This needs the anchor beat's fitted GP (portals.step). Under maestro.keep_all_files: false the
        anchor pickle can be slim/pruned (no surrogate steps) -- e.g. a re-run that APPENDS beats onto
        a finished MAESTRO, whose last beat was slimmed to save space. Then there is no GP to
        flux-match against: warn and return False so the caller keeps the normal initialization.
        (Surrogate DATA reuse via extrapointsFile = surrogate_data.csv is independent and still applies.)'''

        print('\n\t- Running flux match for first point')

        portals = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_starting_point)

        if getattr(portals, 'step', None) is None:
            print('\t\t- Previous-beat surrogate unavailable (slim/pruned optimization_object.pkl); '
                  'skipping first-point flux match, keeping the normal initialization', typeMsg='w')
            return False

        # Flux-match first
        folder_fm = self.folder / 'flux_match'
        folder_fm.mkdir(parents=True, exist_ok=True)

        p = portals.powerstates[portals.ibest].profiles
        _ = PORTALSoptimization.flux_match_surrogate(
            portals.step,
            p,
            target_options_use = self.mitim_bo.optimization_object.powerstate.target_options,   # Use the target_options of the new run, not the old one (which may be with fixed targets if soft)
            file_write_csv=folder_fm / 'optimization_data.csv'
            )

        # Move files
        (self.folder / 'Outputs').mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder_fm / 'optimization_data.csv', self.folder / 'Outputs')
        return True

    def finalize(self, **kwargs):

        # Refresh folder_output from self.folder only when the PORTALS run in self.folder actually
        # COMPLETED, signalled by Outputs/optimization_object.pkl (MITIM_BO.save() runs unconditionally
        # at the end of run(), after the step loop, for both full-BO and converged-in-training cases).
        # Keying off mere existence of Outputs/ is unsafe: a run killed mid-loop leaves an empty/incomplete
        # Outputs/, and the wipe-then-persist below would then destroy a good folder_output and replace it
        # with an unreadable one (which subsequently crashes from_folder / merge_parameters).
        # On a re-invocation after `maestro.keep_all_files: false` wiped self.folder, optimization_object.pkl
        # is gone too, so we skip and read the authoritative content already in folder_output.
        portals_completed = (self.folder / 'Outputs' / 'optimization_object.pkl').exists()

        if portals_completed:
            for item in self.folder_output.glob('*'):
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    IOtools.shutil_rmtree(item)

            self._persist(self.folder / 'Outputs', self.folder_output / 'Outputs')

        # --------------------------------------------------------------------------------------------
        # Prepare final beat's input.gacode
        # --------------------------------------------------------------------------------------------
        # Completion is keyed off beat_results/input.gacode (the merged result, written below /
        # by merge_parameters), which survives even when optimization_object.pkl was pruned for space
        # (keep_all_files: false, an intermediate PORTALS beat). The pickle is only needed to
        # RECONSTRUCT the result; if it is gone but input.gacode is present, the beat is finished and we
        # load it directly so a re-run of a finished MAESTRO stays idempotent without the pickle.
        self._finalized_from_pruned = False
        pkl_present = (self.folder_output / 'Outputs' / 'optimization_object.pkl').exists()
        final_gacode = self.folder_output / 'input.gacode'

        if pkl_present:
            portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)
            # Standard PORTALS output
            try:
                self.profiles_output = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
            # Converged in training case
            except AttributeError:
                print('\t\t- PORTALS probably converged in training, so analyzing a bit differently')
                self.profiles_output = portals_output.profiles[portals_output.opt_fun_full.res.best_absolute_index]
            self.profiles_output.write_state(file=self.folder_output / 'input.gacode')

        elif final_gacode.exists():
            # Finished PORTALS beat whose pickle was pruned for space: beat_results/input.gacode already
            # holds the merged result. Load it and let merge_parameters() short-circuit (its full
            # re-derivation needs the pruned optimization_object/optimization_extra pickles).
            print('\t\t- PORTALS pickle pruned for space; loading finished beat from beat_results/input.gacode', typeMsg='i')
            self.profiles_output = PROFILEStools.gacode_state(final_gacode)
            self.profiles_output.derive_quantities()
            self._finalized_from_pruned = True

        else:
            # Neither a freshly-completed run (self.folder) nor a prior persisted result (folder_output)
            # exists: the beat genuinely did not finish. Fail loudly and actionably instead of crashing
            # cryptically in from_folder (or silently producing an empty beat_results).
            raise RuntimeError(
                f"[MAESTRO][PORTALSbeat] PORTALS run in '{IOtools.clipstr(self.folder)}' did not complete "
                f"(no Outputs/optimization_object.pkl) and no prior finalized result exists in "
                f"'{IOtools.clipstr(self.folder_output)}'. Re-run this beat (cold-start it) before finalizing."
            )

    def optional_postprocessing(self):
        '''Space-saving (keep_all_files: false), run once per beat at the END of the MAESTRO run
        (MAESTRO.finalize, AFTER all beats and generate_summary). Safe to slim/drop here because
        nothing reads a PORTALS beat's GP surrogates any more: the in-run consumers
        (the next beat's _flux_match_for_first_point, and summary()) have already happened.
          - LAST portals beat: keep it (so the final core solution still replots) but SLIM it
            (drop the `steps` GP, the bulk of optimization_object.pkl).
          - intermediate portals beat: drop everything heavy it no longer needs --
            optimization_object/optimization_extra pickles, the per-iteration profile snapshots
            (portals_profiles/), optimization_log.txt, and its MAESTRO per-phase stdout logs
            (Outputs/Logs/beat_<n>_*.log); chaining keeps surrogate_data.csv and
            beat_results/input.gacode, and warnings are already in warnings.log.'''
        if self.maestro_instance.keep_all_files:
            return
        out = getattr(self, 'folder_output', None)
        if out is None:
            return
        beats = self.maestro_instance.beats
        portals_counters = [c for c, b in beats.items() if getattr(b, 'name', None) == 'portals']
        my_counter = next((c for c, b in beats.items() if b is self), None)
        pkl = out / 'Outputs' / 'optimization_object.pkl'

        if my_counter is None or my_counter == max(portals_counters):
            # last (or only) portals beat: slim in place (load full, re-save without steps)
            if pkl.exists():
                m = STRATEGYtools.read_from_scratch(pkl)
                m.folderOutputs = out / 'Outputs'
                m.save(lean=True)
                print(f'\t\t- Space-saving: slimmed the final PORTALS beat {my_counter} pickle (dropped GP steps)')
            return

        # intermediate portals beat: drop everything heavy that nothing downstream needs --
        # the pickles, the per-iteration profile snapshots (portals_profiles/) and the log.
        # Safe because: restart reads the pickle (gone, but an intermediate beat is never the
        # resume point), plotMetrics uses the powerstates in optimization_extra (this beat is
        # not replotted), chaining uses surrogate_data.csv + beat_results/input.gacode (both
        # KEPT). portals_profiles is only read by the initializer-fallback / offline read_portals,
        # and optimization_log.txt is never read back.
        removed = 0
        for fname in ('optimization_object.pkl', 'optimization_extra.pkl', 'optimization_log.txt'):
            f = out / 'Outputs' / fname
            if f.exists():
                try:
                    f.unlink(); removed += 1
                except Exception as e:
                    print(f'\t\t- Could not remove {IOtools.clipstr(f)}: {e}', typeMsg='w')
        pdir = out / 'Outputs' / 'portals_profiles'
        if pdir.is_dir():
            try:
                IOtools.shutil_rmtree(pdir); removed += 1
            except Exception as e:
                print(f'\t\t- Could not remove {IOtools.clipstr(pdir)}: {e}', typeMsg='w')
        # ...and this beat's MAESTRO per-phase stdout logs (Outputs/Logs/beat_<n>_*.log) -- the
        # PORTALS run-phase log is the bulk (~7 MB/beat). Safe: interpret() runs after every beat
        # (before finalize) and has already collected warnings into warnings.log, and nothing reads
        # these per-beat logs afterwards. Only this (intermediate) beat's logs are removed.
        logs_dir = getattr(self.maestro_instance, 'folder_logs', None)
        if logs_dir is not None and logs_dir.is_dir():
            for lf in logs_dir.glob(f'beat_{my_counter}_*.log'):
                try:
                    lf.unlink(); removed += 1
                except Exception as e:
                    print(f'\t\t- Could not remove {IOtools.clipstr(lf)}: {e}', typeMsg='w')
        if removed:
            print(f'\t\t- Space-saving: pruned {removed} heavy item(s) from intermediate PORTALS beat {my_counter} (pickles + portals_profiles + logs)')

    def merge_parameters(self):
        '''
        The goal of the PORTALS beat is to produce:
            - Kinetic profiles
            - Dynamics targets that gave rise to the kinetic profiles
        However, the PORTALS run makes the existing fast ion profiles thermal,
        so this merge needs to bring back the fast ion species from the last TRANSP beat
        So, this merge:
            - Frozen profiles are converted to PORTALS output resolution (opposite to usual, but keeps gradients)
            - Inserts kinetic profiles
            - Inserts dynamic targets (only those that were evolved)
            - Restore fast ion profiles
        '''

        if getattr(self, '_finalized_from_pruned', False):
            # Finished PORTALS beat whose pickle was pruned: beat_results/input.gacode already holds the
            # merged result (loaded into profiles_output by finalize), and the re-derivation below needs
            # the pruned optimization_object/optimization_extra pickles. Skip it; profiles_output (the
            # merged state) still feeds the chain via _freeze_parameters.
            print('\t\t- Skipping PORTALS merge re-derivation (pickle pruned; using saved merged input.gacode)', typeMsg='i')
            return

        # Write the pre-merge input.gacode before modifying it
        self.profiles_output.write_state(file=self.folder_output / 'input.gacode_pre_merge')

        # First, bring back to the resolution of the frozen
        p_frozen = self.maestro_instance.profiles_with_engineering_parameters
        # self.profiles_output.changeResolution(rho_new = p_frozen.profiles['rho(-)'])

        # In PORTALS it is more convenient to bring frozen to portals resolution instead (keeps gradients from beat to beat)
        p_frozen.changeResolution(rho_new = self.profiles_output.profiles['rho(-)'])

        # --------------------------------------------------------------------------------------------
        # Re-define baseline
        # --------------------------------------------------------------------------------------------

        profiles_portals_out = copy.deepcopy(self.profiles_output)

        # Baseline is frozen, I'll modify things from here
        self.profiles_output = p_frozen

        # --------------------------------------------------------------------------------------------
        # Insert relevant quantities
        # --------------------------------------------------------------------------------------------

        # Merge Te and ne:
        self.profiles_output.profiles['te(keV)'] = profiles_portals_out.profiles['te(keV)']
        self.profiles_output.profiles['ne(10^19/m^3)'] = profiles_portals_out.profiles['ne(10^19/m^3)']

        # Insert Ti and ni (but check for species in case portals has removed them, e.g. fast ions)
        for i,sp in enumerate(profiles_portals_out.Species):
            for j,sp1 in enumerate(self.profiles_output.Species):
                if (sp['Z'] == sp1['Z']) and (sp['A'] == sp1['A']): 
                    self.profiles_output.profiles['ni(10^19/m^3)'][:,j] = profiles_portals_out.profiles['ni(10^19/m^3)'][:,i]
                    if sp1["S"] == "fast" and sp["S"] == "therm": 
                        # make all fast ions fast again
                        self.profiles_output.Species[j]["S"] = "fast"
                        # leave FI profile unchanged
                        self.profiles_output.profiles['ti(keV)'][:,j] = self.profiles_output.profiles['ti(keV)'][:,i] 
                    else:
                        # update thermal ion profiles from PORTALS
                        self.profiles_output.profiles['ti(keV)'][:,j] = profiles_portals_out.profiles['ti(keV)'][:,i]

        # Enforce quasineutrality because now I have all the ions
        self.profiles_output.enforce_quasineutrality()

        # Make sure the pressure is consistent with the new profiles
        self.profiles_output.selfconsistentPTOT()

        # Insert powers
        # Read from folder_output, NOT self.folder: merge_parameters runs after finalize(), which has
        # already persisted the completed PORTALS run into folder_output (moved out of self.folder under
        # keep_all_files: false, copied otherwise). So folder_output holds the full Outputs and from_folder
        # returns a PORTALSanalyzer with portals_parameters. Reading self.folder would instead hit the now-
        # emptied run folder and fall back to a PORTALSinitializer whose powerstates can be empty (e.g. a
        # try_flux_match_only_for_first_point beat never writes initialization_simple_relax/), which used to
        # blow up at powerstates[-1] with IndexError.
        opt_fun = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)

        try:
            target_options = opt_fun.portals_parameters['target']['options']
        except AttributeError:
            # Fallback for an SR-step/initializer read (no portals_parameters): recover from the last powerstate.
            # Note the extra ['options']: portals_parameters['target']['options'] is already the inner dict, but
            # the powerstate stores target_options as the outer {'evaluator':..., 'options': {...}}, so dive one
            # more level to keep target_options['targets_evolve'] valid below.
            if not opt_fun.powerstates:
                raise RuntimeError(
                    f"[MAESTRO][PORTALSbeat] Could not read PORTALS target options from "
                    f"'{IOtools.clipstr(self.folder_output)}': no portals_parameters and no powerstates. The "
                    f"PORTALS beat likely did not produce a complete result; re-run it (cold-start) before merging."
                )
            target_options = opt_fun.powerstates[-1].target_options['options']
        
        if 'qie' in target_options['targets_evolve']:
            self.profiles_output.profiles['qei(MW/m^3)'] = profiles_portals_out.profiles['qei(MW/m^3)']
        if 'qrad' in target_options['targets_evolve']:
            for key in ['qbrem(MW/m^3)', 'qsync(MW/m^3)', 'qline(MW/m^3)']:
                self.profiles_output.profiles[key] = profiles_portals_out.profiles[key]
        if 'qfus' in target_options['targets_evolve']:
            for key in ['qfuse(MW/m^3)', 'qfusi(MW/m^3)']:
                self.profiles_output.profiles[key] = profiles_portals_out.profiles[key]

        # Re-apply zero_source_blocks: the merge above rebased profiles_output on
        # p_frozen (the pre-PORTALS engineering snapshot), which still carries the
        # original source columns from the upstream beat. Channels in targets_evolve
        # were just pulled back from PORTALS (zero) above; channels NOT in
        # targets_evolve must be re-zeroed here so the zeros survive into
        # input.gacode_final and the next beat's seed.
        zero_blocks = getattr(self, 'initialization_parameters', {}).get('zero_source_blocks', [])
        if zero_blocks:
            self.profiles_output.correct(options={'zero_source_blocks': zero_blocks, 'recalculate_ptot': False})
        # --------------------------------------------------------------------------------------------

        # Write to final input.gacode
        self.profiles_output.derive_quantities()
        self.profiles_output.write_state(file=self.folder_output / 'input.gacode')

    def grab_output(self, full = False, **kwargs):

        isitfinished = self.maestro_instance.check(beat_check=self)

        folder = self.folder_output if isitfinished else self.folder

        opt_fun = STRATEGYtools.opt_evaluator(folder) if full else PORTALSanalysis.PORTALSanalyzer.from_folder(folder)

        profiles = PROFILEStools.gacode_state(self.folder_output / 'input.gacode') if isitfinished else None
        
        return opt_fun, profiles

    def summary(self, output_dir, counter = None, wall_time_s = None):
        '''
        Markdown section for the last PORTALS beat: convergence scalars + Metrics figure.
        '''
        import matplotlib.pyplot as plt

        analyzer = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)

        # Iteration count and best-iteration index
        try:
            n_iters = len(analyzer.powerstates)
        except Exception:
            n_iters = None
        ibest = getattr(analyzer, 'ibest', None)

        # Residual at iter 0 and at best iter
        residual0 = residual_best = None
        try:
            residual_arr = analyzer.step.BOmetrics["overall"]["Residual"]
            residual0 = -residual_arr[0].item()
            if ibest is not None and ibest < len(residual_arr):
                residual_best = -residual_arr[ibest].item()
        except Exception:
            pass

        # Number of points each GP surrogate was actually fitted to. This is per-output
        # because previous-beat data can be appended via file (MAESTRO's
        # use_previous_surrogate_data), so different channels can carry different counts;
        # train_X_usedToTrain includes those file-added points. On success this is a list
        # of (output_name, n_points); on failure the reason is surfaced in the table
        # rather than silently dropped.
        surrogate_points = None
        try:
            gps = analyzer.step.GP["individual_models"]
            surrogate_points = [(gp.output, int(gp.gpmodel.train_X_usedToTrain.shape[0])) for gp in gps]
        except Exception as e:
            surrogate_points = ('__error__', type(e).__name__)

        # Time per iteration from this beat's timing.jsonl (Eval @ N entries)
        time_per_iter_s = None
        timing_file = self.folder_output / 'Outputs' / 'timing.jsonl'
        if timing_file.exists():
            import json, re
            iter_pat = re.compile(r'@\s*\d+\s*$')
            times = []
            with open(timing_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if 'duration_s' not in d:
                        continue
                    script = d.get('script', '')
                    if not iter_pat.search(script):
                        continue
                    try:
                        times.append(float(d['duration_s']))
                    except (TypeError, ValueError):
                        pass
            if times:
                time_per_iter_s = sum(times) / len(times)

        # Generate Metrics figure
        png_name = 'portals_metrics.png'
        png_path = output_dir / png_name
        try:
            if n_iters is not None and n_iters > 0:
                fig = plt.figure(figsize=(12, 7))
                analyzer.plotMetrics(fig=fig)
                fig.savefig(png_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
                fig_md = f'\n![PORTALS metrics]({png_name})\n'
            else:
                fig_md = '\n*(PORTALS has not run enough iterations to plot metrics)*\n'
        except Exception as e:
            fig_md = f'\n*(PORTALS metrics figure unavailable: {e})*\n'

        # Compose markdown
        header_extra = f' (Beat {counter})' if counter is not None else ''
        lines = [f'## PORTALS{header_extra}', '']
        lines.append('| Quantity | Value |')
        lines.append('|---|---|')
        lines.append(f'| Iterations | {n_iters if n_iters is not None else "n/a"} |')
        lines.append(f'| Best iteration | {ibest if ibest is not None else "n/a"} |')
        if isinstance(surrogate_points, list) and surrogate_points:
            counts = [n for _, n in surrogate_points]
            m = len(counts)
            if len(set(counts)) == 1:
                lines.append(f'| Training points / surrogate | {counts[0]} (× {m} surrogates) |')
            else:
                lines.append(f'| Training points / surrogate | {min(counts)}–{max(counts)} (× {m} surrogates) |')
        elif isinstance(surrogate_points, tuple) and surrogate_points[0] == '__error__':
            lines.append(f'| Training points / surrogate | n/a ({surrogate_points[1]}) |')
        if residual0 is not None:
            lines.append(f'| Residual at iter 0 | {residual0:.4g} |')
        if residual_best is not None:
            lines.append(f'| Residual at best iter | {residual_best:.4g} |')
        if residual0 is not None and residual_best is not None and residual_best != 0:
            lines.append(f'| Residual reduction | {residual0 / residual_best:.2f}x |')
        if time_per_iter_s is not None:
            lines.append(f'| Mean time per iteration | {time_per_iter_s:.1f} s |')
        if wall_time_s is not None:
            lines.append(f'| Beat wall-time | {_format_seconds(wall_time_s)} |')
        lines.append('')
        # Per-surrogate breakdown only when the counts actually differ (otherwise the
        # single table row above already says it). Keeps the common uniform case clean.
        if isinstance(surrogate_points, list) and len({n for _, n in surrogate_points}) > 1:
            lines.append('### Surrogate training points')
            lines.append('')
            lines.append('| Surrogate | Points |')
            lines.append('|---|---|')
            for name, n in surrogate_points:
                lines.append(f'| {name} | {n} |')
            lines.append('')
        lines.append(fig_md)
        return '\n'.join(lines)

    def plot(self,  fn = None, counter = 0, full_plot = True):

        opt_fun, _ = self.grab_output(full = full_plot)

        if full_plot:
            opt_fun.fn = fn
            opt_fun.plot_optimization_results(analysis_level=4, tabs_colors=counter)
        else:
            if len(opt_fun.powerstates)>0:
                fig = fn.add_figure(label="PORTALS Metrics", tab_color=counter)
                opt_fun.plotMetrics(fig=fig)
            else:
                print('\t\t- PORTALS has not run enough to plot anything', typeMsg='w')

        msg = '\t\t- Plotting of PORTALS beat done'

        return msg

    # --------------------------------------------------------------------------------------------
    # Additional PORTALS utilities
    # --------------------------------------------------------------------------------------------
    def _inform(
        self,
        use_previous_residual = True,
        use_previous_surrogate_data = True,
        change_last_radial_call = False,
        minimum_relative_change_in_x = 0.005,
        use_previous_ranges = True,
        ):
        '''
        Prepare next PORTALS runs accounting for what previous PORTALS runs have done
        '''

        # The user's portals_parameters is a PARTIAL overlay over the PORTALS template
        # ("only specify the keys you want to override"), so resolve the knobs read
        # below against the template defaults instead of KeyError-ing on a lean
        # namelist (which would only bite at the SECOND portals beat, hours in).
        _portals_template = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.portals.yaml")
        _stopping = {
            **_portals_template['optimization_options']['convergence_options']['stopping_criteria_parameters'],
            **self.portals_parameters.get('optimization_options', {}).get('convergence_options', {}).get('stopping_criteria_parameters', {}),
        }

        # ----------------------------------------------------------------------------------------------
        # Use previous residual goal if available from previous PORTALS beat (added in _inform_save)
        # ----------------------------------------------------------------------------------------------

        if use_previous_residual and \
            ('original_residual' in self.maestro_instance.parameters_trans_beat) and \
            (_stopping['maximum_value_is_rel']):
            
            if 'convergence_options' not in self.optimization_options_additional:
                self.optimization_options_additional['convergence_options'] = {}
            if 'stopping_criteria_parameters' not in self.optimization_options_additional['convergence_options']:
                self.optimization_options_additional['convergence_options']['stopping_criteria_parameters'] = {}

            original_residual = self.maestro_instance.parameters_trans_beat['original_residual']
            rel_val = _stopping['maximum_value']

            # Make it absolute from now on
            self.optimization_options_additional['convergence_options']['stopping_criteria_parameters']['maximum_value_is_rel'] = False
            
            # Set the absolute value based on the residual
            self.optimization_options_additional['convergence_options']['stopping_criteria_parameters']['maximum_value'] = original_residual*rel_val
            
            print(f"\t\t- Using previous residual goal as maximum value for optimization (not relative): {self.optimization_options_additional['convergence_options']['stopping_criteria_parameters']['maximum_value']}")

        # ----------------------------------------------------------------------------------------------
        # Use previous surrogate data if available
        # ----------------------------------------------------------------------------------------------
        
        reusing_surrogate_data = False
        self.folder_starting_point = None
        if use_previous_surrogate_data and \
            ('portals_surrogate_data_file' in self.maestro_instance.parameters_trans_beat) and \
            ('portals_last_run_folder' in self.maestro_instance.parameters_trans_beat):
                    
            if 'surrogate_options' not in self.optimization_options_additional:
                self.optimization_options_additional['surrogate_options'] = {}
            self.optimization_options_additional['surrogate_options']["extrapointsFile"] = self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file']

            self.folder_starting_point = self.maestro_instance.parameters_trans_beat['portals_last_run_folder']

            print(f"\t\t- Using previous surrogate data for optimization: {IOtools.clipstr(self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'])}")

            reusing_surrogate_data = True
            
        # ----------------------------------------------------------------------------------------------
        # Change last radial location if requested
        # ----------------------------------------------------------------------------------------------
        
        last_radial_location_moved = False
        if change_last_radial_call and ('rhotop' in self.maestro_instance.parameters_trans_beat):

            solution_overlay = self.portals_parameters.setdefault('solution', {})

            # Value-aware check (overlays may carry predicted_roa: null alongside predicted_rho)
            if solution_overlay.get('predicted_roa') is not None:

                print('\t\t- Using EPED pedestal top rho to select last radial location of PORTALS (in r/a)')

                # interpolate the correct roa location from the EPED pedestal top, if it is defined
                roatop = interpolation_function(self.maestro_instance.parameters_trans_beat['rhotop'], 
                                self.profiles_current.profiles['rho(-)'], 
                                self.profiles_current.derived['roa']).item()
                
                #roatop = roatop.round(3)
                
                # set the last value of the radial locations to the interpolated value
                roatop_old = copy.deepcopy(self.portals_parameters['solution']["predicted_roa"][-1])
                self.portals_parameters['solution']["predicted_roa"][-1] = roatop
                print(f'\t\t\t* Last radial location moved from r/a = {roatop_old} to {self.portals_parameters["solution"]["predicted_roa"][-1]}')
                print(f'\t\t\t* predicted_roa: {self.portals_parameters["solution"]["predicted_roa"]}')

                strKeys = 'predicted_roa'

            else:

                print('\t\t- Using EPED pedestal top rho to select last radial location of PORTALS (in rho)')

                if 'predicted_rho' not in solution_overlay:
                    # Lean overlay: seed from the template default before moving its last point
                    solution_overlay['predicted_rho'] = copy.deepcopy(_portals_template['solution']['predicted_rho'])

                # set the last value of the radial locations to the interpolated value
                rhotop_old = copy.deepcopy(self.portals_parameters['solution']['predicted_rho'][-1])
                self.portals_parameters['solution']['predicted_rho'][-1] = self.maestro_instance.parameters_trans_beat['rhotop']
                print(f'\t\t\t* Last radial location moved from rho = {rhotop_old} to {self.portals_parameters["solution"]["predicted_rho"][-1]}')

                strKeys = 'predicted_rho'

            last_radial_location_moved = True

            # Check if I changed it previously and it hasn't moved
            # (value-aware: old runs may have stored predicted_roa=None in the trans-beat
            #  parameters, and a missing/None entry must not skip this no-move check)
            if self.maestro_instance.parameters_trans_beat.get(strKeys) is not None:
                print(f'\t\t\t* {strKeys} in previous PORTALS beat: {self.maestro_instance.parameters_trans_beat[strKeys]}')
                print(f'\t\t\t* {strKeys} in current PORTALS beat: {self.portals_parameters["solution"][strKeys]}')

                if abs(self.portals_parameters['solution'][strKeys][-1]-self.maestro_instance.parameters_trans_beat[strKeys][-1]) / self.maestro_instance.parameters_trans_beat[strKeys][-1] < minimum_relative_change_in_x:
                    print('\t\t\t* Last radial location was not moved because the change is minimal')
                    last_radial_location_moved = False
                    self.portals_parameters['solution'][strKeys][-1] = self.maestro_instance.parameters_trans_beat[strKeys][-1]

        # In the situation where the last radial location moves, I cannot reuse that surrogate data
        if last_radial_location_moved and reusing_surrogate_data:
            print('\t\t- Last radial location was moved, so surrogate data will not be reused for that specific location')
            self.optimization_options_additional['surrogate_options']["extrapointsModelsAvoidContent"] = ['_tar',f"_{len(self.portals_parameters['solution'][strKeys])}"]
            self.try_flux_match_only_for_first_point = False

        # ----------------------------------------------------------------------------------------------
        # Change ranges
        # ----------------------------------------------------------------------------------------------
        if use_previous_ranges and 'portals_ymin' in self.maestro_instance.parameters_trans_beat:
            print('\t\t- Freezing original ranges for PORTALS optimization from previous beat')

            solution = {
                'exploration_ranges': {
                    'limits_are_relative': False,
                    'ymin': self.maestro_instance.parameters_trans_beat['portals_ymin'],
                    'ymax': self.maestro_instance.parameters_trans_beat['portals_ymax'],
                }
            }
            
            if 'solution' not in self.optimization_options_additional:
                self.optimization_options_additional['solution'] = solution
            else:
                self.optimization_options_additional['solution'] = IOtools.deep_dict_update(self.optimization_options_additional['solution'], solution)

    def _inform_save(self):

        print('\t- Saving PORTALS beat parameters for future beats')

        # Save the residual goal to use in the next PORTALS beat
        portals_output, _ = self.grab_output()

        # Standard PORTALS output
        try:
            stepSettings = portals_output.step.stepSettings
            portals_parameters = portals_output.portals_parameters
        # Converged in training case
        except AttributeError:
            stepSettings = portals_output.opt_fun_full.mitim_model.stepSettings
            portals_parameters = portals_output.opt_fun_full.mitim_model.optimization_object.portals_parameters

        '''
        -------------------------------------------------------------------------------------------
        Store residual for convergence
        -------------------------------------------------------------------------------------------
        '''
        
        # Get maximum value of negative residual (absolute)
        original_residual = -portals_output.step.BOmetrics["overall"]["Residual"][0].item()
        self.maestro_instance.parameters_trans_beat['original_residual'] = original_residual
        print(f'\t\t* Original value of negative residual (absolute) saved for future beats: {original_residual}')

        '''
        -------------------------------------------------------------------------------------------
        Store surrogate data to be reused
        -------------------------------------------------------------------------------------------
        '''
        
        fileTraining = self.folder_output / 'Outputs' / 'surrogate_data.csv'

        self.maestro_instance.parameters_trans_beat['portals_last_run_folder'] = self.folder_output
        self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] = fileTraining
        print(f'\t\t* Surrogate data saved for future beats: {IOtools.clipstr(fileTraining)}')

        '''
        -------------------------------------------------------------------------------------------
        Store locations to be predicted
        -------------------------------------------------------------------------------------------
        '''

        # Value-aware checks: the PORTALS template carries BOTH keys, with the unused
        # one set to null (predicted_roa wins only when actually provided). Checking
        # key presence alone stored predicted_roa=None here, so the next beat's
        # _inform never found its predicted_rho in the trans-beat parameters, skipped
        # the no-move check, declared a bogus move ("from 0.9 to 0.9"), and silently
        # disabled the flux-match-first warm start and the last-location surrogate reuse.
        if portals_parameters['solution'].get('predicted_roa') is not None:
            self.maestro_instance.parameters_trans_beat['predicted_roa'] = portals_parameters['solution']['predicted_roa']
            print(f'\t\t* predicted_roa saved for future beats: {portals_parameters["solution"]["predicted_roa"]}')
        elif portals_parameters['solution'].get('predicted_rho') is not None:
            self.maestro_instance.parameters_trans_beat['predicted_rho'] = portals_parameters['solution']['predicted_rho']
            print(f'\t\t* predicted_rho saved for future beats: {portals_parameters["solution"]["predicted_rho"]}')

        '''
        -------------------------------------------------------------------------------------------
        Store ranges
        -------------------------------------------------------------------------------------------
        '''
        ymin, ymax = {}, {}
        cont = 0
        for channel in portals_parameters['solution']['predicted_channels']:
            ymin0 = []
            ymax0 = []
            for rho in portals_parameters['solution']['predicted_rho']:
                ymin0.append(stepSettings['optimization_options']['problem_options']['dvs_min'][cont])
                ymax0.append(stepSettings['optimization_options']['problem_options']['dvs_max'][cont])
                cont += 1
            ymin[channel] = ymin0
            ymax[channel] = ymax0
            
        self.maestro_instance.parameters_trans_beat['portals_ymin'] = ymin
        self.maestro_instance.parameters_trans_beat['portals_ymax'] = ymax
        print(f'\t\t* ymin saved for future beats: {ymin}')
        print(f'\t\t* ymax saved for future beats: {ymax}')
        # NB: do NOT slim the pickle here. A PORTALS beat's surrogates are still consumed
        # later in the run -- the NEXT PORTALS beat warm-starts from them in
        # _flux_match_for_first_point (folder_starting_point.step.GP) -- and by summary().
        # All slimming/pruning happens once at the very end, in optional_postprocessing.

# -----------------------------------------------------------------------------------------------------------------------
# Defaults to help MAESTRO
# -----------------------------------------------------------------------------------------------------------------------

def profiles_postprocessing_fun(file_profs, lumpImpurities = True, enforce_same_density_gradients = True):
    p = PROFILEStools.gacode_state(file_profs)
    if lumpImpurities:
        p.lumpImpurities()
    if enforce_same_density_gradients:
        p.enforce_same_density_gradients()
    p.enforceQuasineutrality()
    p.write_state(file=file_profs)
    return p

def preprocess_prepare_portals(beat_namelist,maestro_namelist, preprocess_prepare_parameters):
    
    lumpImpurities = preprocess_prepare_parameters["lumpImpurities"]
    enforce_same_density_gradients = preprocess_prepare_parameters["enforce_same_density_gradients"]

    # add postprocessing function (tolerate a lean overlay without these sections)
    beat_namelist.setdefault('portals_parameters', {}).setdefault('transport', {})['profiles_postprocessing_fun'] = partial(profiles_postprocessing_fun, lumpImpurities=lumpImpurities, enforce_same_density_gradients=enforce_same_density_gradients)

    return beat_namelist