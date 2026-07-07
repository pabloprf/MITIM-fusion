import copy
import shutil
import json
import numpy as np
from pathlib import Path
import sys
import re
import datetime
from typing import OrderedDict
from mitim_tools.misc_tools import IOtools, GUItools, LOGtools
from mitim_modules.maestro.utils import MAESTROplot
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __version__, __mitimroot__
from IPython import embed

from mitim_modules.maestro.utils.EPEDbeat import eped_beat
from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat
from mitim_modules.maestro.utils.LENGYELbeat import lengyel_beat
from mitim_modules.maestro.utils.SHARPNESSbeat import sharpness_beat
from mitim_modules.maestro.utils.CONFINEMENTbeat import confinement_beat
from mitim_modules.maestro.utils.MAESTRObeat import creator_from_eped, creator_from_parameterization, creator_from_fixed_bc, creator
from mitim_modules.maestro.utils.MAESTRObeat import beat as beat_generic

'''
MAESTRO:
    Modular and Accelerated Engine for Simulation of Transport and Reactor Optimization
 (If MAESTRO is the orchestrator, then BEAT is each of the beats (steps) that MAESTRO orchestrates)
'''

ENABLE_EMBED = False # If True, will enable IPython embed, useful for debugging (but won't write maestro.log or Logs/ files... so only use for debugging a run)

class maestro:

    def __init__(
            self,
            folder,
            terminal_outputs = False,
            master_cold_start = False,
            overall_log_file = True,
            keep_all_files = True,
            master_seed = 0,
            maestro_namelist = {}
            ):
        '''
        Inputs:
            - folder: Main folder where all the beats will be saved
            - terminal_outputs: If True, all outputs will be printed to terminal. If False, they will be saved to a log file per beat step
        '''

        self.terminal_outputs = terminal_outputs
        self.master_cold_start = master_cold_start        # If True, all beats will be cold_started
        self.keep_all_files = keep_all_files              # If True, all files will be kept, if False, only the final output files will be kept
        self.master_seed = master_seed

        self.maestro_namelist = maestro_namelist

        # --------------------------------------------------------------------------------------------
        # Prepare folders
        # --------------------------------------------------------------------------------------------

        self.folder = IOtools.expandPath(folder)
        
        self.folder_output = self.folder / "Outputs"
        self.folder_logs = self.folder_output / "Logs"
        self.folder_performance = self.folder_output / "Performance"
        self.folder_beats = self.folder / "Beats"

        self.folder_logs.mkdir(parents=True, exist_ok=True)
        self.folder_beats.mkdir(parents=True, exist_ok=True)
        self.folder_performance.mkdir(parents=True, exist_ok=True)

        # If terminal outputs, I also want to keep track of what has happened in a log file
        if terminal_outputs and overall_log_file and not ENABLE_EMBED:
            self.master_log_file = self.folder_output / "maestro.log"
            sys.stdout = LOGtools.Logger(logFile=self.master_log_file, writeAlsoTerminal=True)
        else:
            self.master_log_file = None
            
        self.warnings_log_file = self.folder_output / "warnings.log"
        self.warnings_log_file_new = True

        branch, commit_hash = IOtools.get_git_info(__mitimroot__)
        print('\n ---------------------------------------------------------------------------------------------------')
        print(f'MAESTRO run (MITIM version {__version__}, branch {branch}, commit {commit_hash})')
        print('---------------------------------------------------------------------------------------------------')
        print(f'folder: {self.folder}')

        # --------------------------------------------------------------------------------------------
        # Prepare variables
        # --------------------------------------------------------------------------------------------
    
        self.beats = {}             # Where all the beats will be stored
        self.counter_current = 0    # Counter of current beat

        '''
        Engineering parameters performed during "freezing"
        --------------------------------------------------------------------------------------------------------------------
        During MAESTRO, the separatrix and main engineering parameters do not change, so I need to freeze them upon
        initialization, otherwise we'll have a leak of power or geometry quantities if from beat to beat it's, for whatever
        reason, lower.  In other words, e.g., it's best to just pass the relevant outputs from PORTALS or TRANSP to a base
        profiles object that is frozen and with the resolutions I want to keep for the rest of the MAESTRO run.
        '''
        self.profiles_with_engineering_parameters = None # Start with None, but will be populated at first initialization

        '''
        Parameters that can be passed from beat to beat (e.g. PORTALS residual or geqdsk 0.995 flux surface or rho_top EPED)
        --------------------------------------------------------------------------------------------------------------------
        '''
        self.parameters_trans_beat = {}

        # When to freeze the 99.5% shaping (kappa995/delta995/zeta995) fed to EPED
        # (see templates/namelist.maestro.yaml -> maestro.refreeze_995_after_beat):
        #   0    : keep the value extracted at initialization (default, old behavior)
        #   N>0  : re-extract ONCE from beat N's evolved equilibrium (e.g. after TRANSP)
        #   null : never freeze -- each EPED beat recomputes from its own current equilibrium
        #          (the null case is enforced in the EPED beat's _inform)
        self.refreeze_995_after_beat = self.maestro_namelist.get('maestro', {}).get('refreeze_995_after_beat', 0)

        # Whether this instance has already stashed a previous run's finalization
        # artifacts (done automatically at the first beat run())
        self._unfinalize_done = False

    def define_beat(self, beat, initializer = None, cold_start = False):

        timeBeginning = datetime.datetime.now()

        self.counter_current += 1
        if beat is None:
            print(f'\n- Beat {self.counter_current}: EMPTY ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = beat_generic(self)
        elif beat == 'transp':
            print(f'\n- Beat {self.counter_current}: TRANSP ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = transp_beat(self)
        elif beat == 'portals':
            print(f'\n- Beat {self.counter_current}: PORTALS ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = portals_beat(self)
        elif beat == 'eped':
            print(f'\n- Beat {self.counter_current}: EPED ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = eped_beat(self)
        elif beat == 'lengyel':
            print(f'\n- Beat {self.counter_current}: LENGYEL ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = lengyel_beat(self)
        elif beat == 'sharpness':
            print(f'\n- Beat {self.counter_current}: SHARPNESS ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = sharpness_beat(self)
        elif beat == 'confinement':
            print(f'\n- Beat {self.counter_current}: CONFINEMENT ******************************* {timeBeginning.strftime("%Y-%m-%d %H:%M:%S")}')
            self.beats[self.counter_current] = confinement_beat(self)

        # Access current beat easily
        self.beat = self.beats[self.counter_current]

        # Define initializer
        self.beat.define_initializer(initializer)
        
        # Restart or check
        self._restart_or_check(cold_start=cold_start)

    def _restart_or_check(self, cold_start = False):
        '''
        ------------------------------------------------------------------
        Checker of existence and cold_start handling
        ------------------------------------------------------------------
        '''
        
        self.beat.cold_start = cold_start or self.master_cold_start
        
        # If the beat needs to be cold started, remove folders (to avoid confusions) and proceed to restart
        if self.beat.cold_start:
            
            self.beat.restart()
            self.beat.run_flag = True
            
        # If restart is not imposed manually, check if beat needs to run and inform future ones
        else:
            
            # Check if beat results contain the expected output files
            self.check()
        
            '''
            If a beat needs to run, all the rest of the beats will need to run from scratch.
            Inform that such that I call restart() in the next beats via the master_cold_start flag.
            '''
            
            # Print some info if that's the case. If not already cold started, inform that all next beats will need to be cold started
            if self.beat.run_flag and not self.master_cold_start:
                print('\t\t- Since this step needs to run, all next ones will need to be cold started', typeMsg = 'i')
            
            # Pass the info
            self.master_cold_start = self.master_cold_start or self.beat.run_flag
          
    def define_creator(self, method, **kwargs_creator):
        '''
        To initialize some profile functional form
        '''
        if method in ['eped', 'eped_initializer']:
            self.beat.initialize.profile_creator = creator_from_eped(self.beat.initialize,**kwargs_creator)
        elif method == 'parameterization':
            self.beat.initialize.profile_creator = creator_from_parameterization(self.beat.initialize,**kwargs_creator)
        elif method in ['profiles', "fixed_profiles"]:
            self.beat.initialize.profile_creator = creator(self.beat.initialize,**kwargs_creator)
        elif method == 'fixed_bc':
            self.beat.initialize.profile_creator = creator_from_fixed_bc(self.beat.initialize,**kwargs_creator)
        else:
            raise ValueError(f'[MITIM] Creator method {method} not recognized')

    # --------------------------------------------------------------------------------------------
    # Beat operations
    # --------------------------------------------------------------------------------------------
    
    @mitim_timer(lambda self: f'Beat #{self.counter_current} ({self.beat.name}) - Checker')
    def check(self, beat_check = None, **kwargs):
        '''
        Note:
            After each beat, the results are passed to an output folder.
            If the required files are already there, the beat will not be run again.
            It is also assumed that the results were correct if they were put there, so
            the checks should happen in the finalize() method of each beat.
        '''

        if beat_check is None:
            beat_check = self.beat

        print('\t- Checking...')
        log_file = self.folder_logs / f'beat_{self.counter_current}_check.log' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):

            # Does the output file already exist? That will inform whether the beat needs to run
            output_file = IOtools.findFileByExtension(beat_check.folder_output, 'input.gacode', agnostic_to_case=True)
            
            if output_file is not None:
                print(f'\t\t- Output file {IOtools.clipstr(output_file)} already exists, not running beat', typeMsg = 'i')
            else:
                print(f'\t\t- Output file {IOtools.clipstr(output_file)} not found, beat will be run')

            # The beat needs to run if output_file is None
            self.beat.run_flag = output_file is None
        
        isitfinished = not self.beat.run_flag
            
        return isitfinished

    @mitim_timer(lambda self: f'Beat #{self.counter_current} ({self.beat.name}) - Initializer',
        log_file = lambda self: self.folder_performance / "timing.jsonl")
    def initialize(self, *args, **kwargs):

        print('\t- Initializing...')
        if self.beat.run_flag:
            log_file = self.folder_logs / f'beat_{self.counter_current}_ini.log' if (not self.terminal_outputs) else None
            with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                # Initialize: produce self.profiles_current
                self.beat.initialize(*args, **kwargs)

        else:
            print('\t\t- Skipping beat initialization because this beat was already run', typeMsg = 'i')
            self.beat.initialize._minimal_call(*args, **kwargs)

        log_file = self.folder_logs / f'beat_{self.counter_current}_inform.log' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            # Initializer can also save important parameters
            self.beat.initialize._inform_save()

            # Creator can also save important parameters
            if ("profile_creator" in self.beat.initialize.__dict__) and (self.beat.initialize.profile_creator is not None):
                self.beat.initialize.profile_creator._inform_save()

            if self.profiles_with_engineering_parameters is None:
                # First initialization, freeze engineering parameters
                self._freeze_parameters(profiles = PROFILEStools.gacode_state(self.beat.initialize.folder / 'input.gacode'))

    @mitim_timer(lambda self: f'Beat #{self.counter_current} ({self.beat.name}) - Preparation',
        log_file = lambda self: self.folder_performance / "timing.jsonl")    
    def prepare(self, *args, **kwargs):

        print('\t- Preparing...')
        if self.beat.run_flag:
            log_file = self.folder_logs / f'beat_{self.counter_current}_prep.log' if (not self.terminal_outputs) else None
            with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                
                # Initialize if necessary
                if not self.beat.initialize_called:
                    print('\t\t- Initializing beat before preparing...')
                    self.beat.initialize()
                # -----------------------------

                self.beat.profiles_current.derive_quantities()
                
                self.beat.prepare(*args, **kwargs)
        else:
            print('\t\t- Skipping beat preparation because this beat was already run', typeMsg = 'i')
            # Still hand the beat the namelist parameters that merge_parameters() needs
            # (finalize+merge re-run even for completed beats): e.g. without
            # zero_source_blocks a re-invocation would silently un-zero sources
            # in beat_results/input.gacode and input.gacode_final.
            self.beat.prepare_minimal(*args, **kwargs)

    @mitim_timer(lambda self: f'Beat #{self.counter_current} ({self.beat.name}) - Run + Finalization',
        log_file = lambda self: self.folder_performance / "timing.jsonl")
    def run(self, **kwargs):

        # First beat execution of this instance: stash any previous run's finalization
        # artifacts so the restarted run regenerates them at its own finalize().
        # Hooked here (and not in __init__) because plot-only consumers (grabMAESTRO)
        # also construct maestro and call define_beat()/check(), but never run() —
        # plotting a finished case must not touch its finalization.
        if not self._unfinalize_done:
            self.unfinalize()

        # Pass ENABLE_EMBED to the beat run
        kwargs.update({'ENABLE_EMBED': ENABLE_EMBED})

        # Run 
        print('\t- Running...')
        if self.beat.run_flag:
            log_file = self.folder_logs / f'beat_{self.counter_current}_run.log' if (not self.terminal_outputs) else None
            with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.run(**kwargs)
        else:
            print('\t\t- Skipping beat run because this beat was already run', typeMsg = 'i')

        # Finalize, merging and freezing should occur even if the run has not been performed because the results are already there
        print('\t- Finalizing beat...')
        log_file = self.folder_logs / f'beat_{self.counter_current}_finalize.log' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):

            # Finalize
            self.beat.finalize(**kwargs)

            # Merge parameters, from self.profiles_current take what's needed and merge with the self.profiles_with_engineering_parameters
            print('\t\t- Merging engineering parameters from MAESTRO')
            self.beat.merge_parameters()

            # Produce a new self.profiles_with_engineering_parameters from this merged object
            self._freeze_parameters()

        # Inform next beats. Persist a JSON snapshot of parameters_trans_beat per beat so a re-run
        # of a finished MAESTRO (and the space-saving pickle prune) can restore the cross-beat state
        # without recomputing it from the heavy PORTALS/TRANSP artifacts. A skipped beat restores its
        # snapshot instead of calling _inform_save(); a run without a snapshot (legacy) recomputes.
        log_file = self.folder_logs / f'beat_{self.counter_current}_inform.log' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file):
            if self.beat.run_flag:
                self.beat._inform_save()
                self._save_trans_beat_parameters()
            elif not self._restore_trans_beat_parameters():
                self.beat._inform_save()

        # Optionally re-freeze the 99.5% shaping from this beat's evolved equilibrium (runs on both the
        # run and skip paths, right after the snapshot is written/restored, so it stays restart-safe)
        self._maybe_refreeze_995()

        # To save space, we can remove the contents of the run_ folder, as everything needed is in the output folder
        if not self.keep_all_files:
            for item in self.beat.folder .iterdir():
                IOtools.shutil_rmtree(item) if item.is_dir() else item.unlink()

    # --------------------------------------------------------------------------------------------
    # Cross-beat parameters (parameters_trans_beat) persistence
    # --------------------------------------------------------------------------------------------
    # A per-beat JSON snapshot of parameters_trans_beat under Outputs/trans_beat_parameters/. It lets
    # a re-run (and the keep_all_files: false pickle prune) restore the cross-beat state of a finished
    # beat without recomputing it from the heavy PORTALS/TRANSP artifacts (which may be pruned). Path
    # values are stored relative to the MAESTRO root so a snapshot survives the run folder being moved.

    def _maybe_refreeze_995(self):
        '''
        If maestro.refreeze_995_after_beat == N (a positive int) and this is beat N, overwrite the
        frozen 99.5% shaping (kappa995/delta995/zeta995) in parameters_trans_beat with the values
        derived from THIS beat's evolved equilibrium, so later EPED beats use a real (e.g.
        post-TRANSP) equilibrium instead of the initialization guess. The snapshot is re-saved so a
        restart restores the updated values. (0 = keep the init value; null = never freeze, which is
        handled in the EPED beat's _inform by simply not reusing the stored value.)
        '''
        target = self.refreeze_995_after_beat
        if not (isinstance(target, int) and not isinstance(target, bool) and target > 0 and self.counter_current == target):
            return

        p = PROFILEStools.gacode_state(self.beat.folder_output / 'input.gacode')
        p.derive_quantities()
        for key in ('kappa995', 'delta995', 'zeta995'):
            if key in p.derived:
                self.parameters_trans_beat[key] = float(p.derived[key])
        self._save_trans_beat_parameters()
        print(f'\t\t- Re-froze 99.5% shaping from beat {target} equilibrium -> '
              f'kappa995={self.parameters_trans_beat["kappa995"]:.3f}, '
              f'delta995={self.parameters_trans_beat["delta995"]:.3f}, '
              f'zeta995={self.parameters_trans_beat["zeta995"]:.3f}', typeMsg='i')

    def _trans_beat_parameters_file(self, counter):
        return self.folder_output / 'trans_beat_parameters' / f'beat_{counter}.json'

    def _encode_trans_beat_value(self, v):
        if isinstance(v, Path):
            try:
                return {'__maestro_relpath__': str(v.relative_to(self.folder))}
            except ValueError:
                return {'__abspath__': str(v)}
        if isinstance(v, dict):
            return {k: self._encode_trans_beat_value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._encode_trans_beat_value(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def _decode_trans_beat_value(self, v):
        if isinstance(v, dict):
            if '__maestro_relpath__' in v:
                return self.folder / v['__maestro_relpath__']
            if '__abspath__' in v:
                return Path(v['__abspath__'])
            return {k: self._decode_trans_beat_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._decode_trans_beat_value(x) for x in v]
        return v

    def _save_trans_beat_parameters(self):
        f = self._trans_beat_parameters_file(self.counter_current)
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, 'w') as fh:
            json.dump(self._encode_trans_beat_value(self.parameters_trans_beat), fh, indent=2)
        print(f'\t\t- Saved cross-beat parameters snapshot to {IOtools.clipstr(f)}', typeMsg='i')

    def _restore_trans_beat_parameters(self):
        f = self._trans_beat_parameters_file(self.counter_current)
        if not f.exists():
            return False
        with open(f, 'r') as fh:
            self.parameters_trans_beat = self._decode_trans_beat_value(json.load(fh))
        print(f'\t\t- Restored cross-beat parameters from {IOtools.clipstr(f)} (skipped beat; artifacts not needed)', typeMsg='i')
        return True


    def interpret(self):
        
        self.warnings_dict = OrderedDict()
        
        # Once each beat is finished, collect all "Warnings" that were in the logs into a single file
        print('\t- Collecting warnings...')
        
        # If master logger exists (run was done with terminal outputs enabled), read from it
        if self.master_log_file is not None:
            
            read_warning(self.master_log_file, self.warnings_dict, 'master')
            
        # If not, check all logs
        else:
                    
            order_flags = ["check", "ini", "prep", "run", "inform", "finalize"]
            order_index = {flag: i for i, flag in enumerate(order_flags)}

            files = [item.name for item in self.folder_logs.glob('*')]

            pattern = re.compile(r"beat_(\d+)_(\w+)\.log")

            def sort_key(name):
                m = pattern.match(name)
                if not m:
                    return (float('inf'), float('inf'))  # unknown pattern → sort last

                beat = int(m.group(1))
                flag = m.group(2)

                return (beat, order_index.get(flag, float('inf')))
                
            sorted_files = sorted(files, key=sort_key)
            
            # Read all in order
            for file in sorted_files:
                
                read_warning(self.folder_logs / Path(file), self.warnings_dict, file)  
            
        # Organize per group
        log_group = {}
        for key in self.warnings_dict:
            group = key.split('_$')[0]
            line = key.split('_$')[1]
            if group not in log_group:
                log_group[group] = {}
            log_group[group][line] = self.warnings_dict[key]
            
        # If file exist, make sure I keep its contents by appending it to the beginning
        self.previous_contents = ''
        if self.warnings_log_file_new and self.warnings_log_file.exists():
            with open(self.warnings_log_file, 'r') as f:
                self.previous_contents = f.read()
            
        self.warnings_log_file_new = False
            
        # Write file
        with open(self.warnings_log_file, 'w') as f:
            
            f.write('\n')
            f.write(f'   Writing warnings @ time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} from previous runs')
            f.write('\n')
            
            f.write(self.previous_contents)
            
            f.write('\n')
            f.write(f'   Writing warnings @ time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            f.write('\n')
            
            for group in log_group:
                f.write(f'\n------------------------------------------------------------\n')
                f.write(f'   Warnings in section: {group}\n')
                f.write(f'------------------------------------------------------------\n')
                for line in log_group[group]:
                    f.write(log_group[group][line]+'\n')

    def _freeze_parameters(self, profiles = None):

        if profiles is None:
            profiles = PROFILEStools.gacode_state(self.beat.folder_output / 'input.gacode')

        print('\t\t- Freezing engineering parameters from MAESTRO')
        self.profiles_with_engineering_parameters = copy.deepcopy(profiles)
        self.profiles_with_engineering_parameters.write_state(file= (self.folder_output / 'input.gacode_frozen'))

    def unfinalize(self):
        '''
        Move the finalization artifacts of a previously completed MAESTRO run
        (final input.gacode, summary report, beat-flow diagram, saved figures)
        to a timestamped backup folder under Outputs. Called automatically at
        the first beat run() when MAESTRO executes on a folder that already
        contains a run: the finalization is regenerated when the restarted run
        reaches its own finalize() — immediately if all beats are already
        complete, or after any newly added beats have run. No-op if no
        finalization artifacts exist.
        '''

        self._unfinalize_done = True

        artifacts = [
            self.folder_output / 'input.gacode_final',
            self.folder_output / 'maestro_summary.md',
            self.folder_output / 'beat_flow.png',
            self.folder_output / 'maestro_special.png',
            self.folder_output / 'maestro_timing.png',
            self.folder_output / 'beat_final',      # log file of the finalize step
            self.folder / 'maestro_plots',          # figures from mitim_run_maestro --save
        ]
        artifacts = [item for item in artifacts if item.exists()]

        if len(artifacts) == 0:
            return

        backup_folder = self.folder_output / f'finalization_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        backup_folder.mkdir(parents=True, exist_ok=True)

        print(f'\t- Folder contains a previously finalized MAESTRO run, moving its finalization artifacts to {IOtools.clipstr(backup_folder)}', typeMsg='i')
        for item in artifacts:
            shutil.move(str(item), str(backup_folder / item.name))
            print(f'\t\t- {item.name}')

    @mitim_timer(lambda self: f'Beat #{self.counter_current} ({self.beat.name}) - Finalizing',
        log_file = lambda self: self.folder_performance / "timing.jsonl")
    def finalize(self):

        print(f'- MAESTRO finalizing ******************************* {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        log_file = self.folder_output / 'beat_final' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):

            final_file= (self.folder_output / 'input.gacode_final')

            self.beat.profiles_output.write_state(file= final_file)

            print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

            # End-of-run human-readable summary report. MUST run BEFORE optional_postprocessing:
            # the summary reads the last PORTALS beat's full surrogates (step.GP / BOmetrics), which
            # postprocessing then slims.
            try:
                self.generate_summary()
            except Exception as e:
                print(f'\t\t- Could not generate maestro_summary.md: {e}', typeMsg='w')

            # Beat-specific end-of-run postprocessing (PORTALS space-saving: slim the last beat,
            # drop intermediates). Runs LAST -- after every beat's run (so all next-beat flux-match
            # warm-starts already consumed the prior beats' surrogates) and after the summary -- so
            # nothing still needs the full GP surrogates. No-op for beats that don't override it.
            for beat_obj in self.beats.values():
                beat_obj.optional_postprocessing()

    # --------------------------------------------------------------------------------------------
    # Summary report
    # --------------------------------------------------------------------------------------------

    def generate_summary(self):
        '''
        Build Outputs/maestro_summary.md at end of run.
        - Rendered (PNG) flowchart of all beats with type + wall-time labels.
        - printInfo() dump of the final plasma state (input.gacode_final).
        - One detailed section per beat type {portals, transp, eped},
          using only the last beat of each type.
        - Exceptions inside any beat.summary() are caught and recorded;
          summary generation itself never raises.
        '''
        from mitim_modules.maestro.utils.MAESTRObeat import _format_seconds

        print('\t- Generating MAESTRO summary report...')

        # Wall-times per beat counter, parsed from timing.jsonl
        beat_wall_times = _parse_beat_wall_times(self.folder_performance / 'timing.jsonl')

        # Find the last beat of each tracked type (insertion order in self.beats)
        TRACKED_TYPES = ['transp', 'portals', 'eped']
        last_by_type = {}
        for counter, beat_obj in self.beats.items():
            if beat_obj.name in TRACKED_TYPES:
                last_by_type[beat_obj.name] = (counter, beat_obj)

        # Compose final markdown
        total_wall_time = sum(v for v in beat_wall_times.values() if v is not None) if beat_wall_times else None

        md = []
        md.append('# MAESTRO summary')
        md.append('')
        md.append(f'- **Root:** `{self.folder}`')
        md.append(f'- **Beats run:** {len(self.beats)}')
        if total_wall_time is not None:
            md.append(f'- **Total wall-time (sum of beat timings):** {_format_seconds(total_wall_time)}')
        md.append(f'- **Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        md.append('')

        # Rendered flowchart PNG (matplotlib boxes + arrows)
        md.append('## Beat flow')
        md.append('')
        try:
            flow_png = _render_beat_flow_png(self.beats, beat_wall_times,
                                             self.folder_output / 'beat_flow.png')
            md.append(f'![Beat flow]({flow_png.name})')
        except Exception as e:
            md.append(f'*(beat-flow diagram unavailable: {e})*')
        md.append('')

        # Special-quantities figure: per-beat evolution of the key 0D quantities
        # (the same 'MAESTRO special' view produced when plotting the case).
        md.append('## Special quantities')
        md.append('')
        try:
            special_png = _render_special_png(self, self.folder_output / 'maestro_special.png')
            md.append(f'![Special quantities]({special_png.name})')
        except Exception as e:
            md.append(f'*(special-quantities figure unavailable: {e})*')
        md.append('')

        # Timing figure: per-beat cumulative wall-time (the 'MAESTRO timings' view)
        timing_jsonl = self.folder_performance / 'timing.jsonl'
        if timing_jsonl.exists():
            md.append('## Timing')
            md.append('')
            try:
                timing_png = _render_timing_png(timing_jsonl, self.folder_output / 'maestro_timing.png')
                md.append(f'![Timing]({timing_png.name})')
            except Exception as e:
                md.append(f'*(timing figure unavailable: {e})*')
            md.append('')

        # Final plasma state: printInfo() output for input.gacode_final
        final_file = self.folder_output / 'input.gacode_final'
        if final_file.exists():
            md.append('## Final plasma state (`input.gacode_final`)')
            md.append('')
            try:
                info_text = _capture_print_info(final_file)
                if info_text.strip():
                    md.append('```text')
                    md.append(info_text)
                    md.append('```')
                else:
                    md.append('*(printInfo produced no output — verbose level may be 0)*')
            except Exception as e:
                md.append(f'*(could not capture printInfo: {e})*')
            md.append('')

        # Warnings: link rather than embed (file can be long)
        if self.warnings_log_file.exists():
            try:
                size = self.warnings_log_file.stat().st_size
                if size > 0:
                    md.append('## Warnings')
                    md.append('')
                    md.append(f'See [`warnings.log`](warnings.log) ({size} bytes).')
                    md.append('')
            except Exception:
                pass

        # Detailed sections per tracked type (last beat only)
        for beat_type in TRACKED_TYPES:
            if beat_type not in last_by_type:
                continue
            counter, beat_obj = last_by_type[beat_type]
            wt = beat_wall_times.get(counter)
            try:
                section = beat_obj.summary(self.folder_output, counter=counter, wall_time_s=wt)
            except Exception as e:
                section = f'## {beat_type.upper()} (Beat {counter})\n*(summary generation failed: {e})*\n'
            if section is not None:
                md.append(section)
                md.append('')

        summary_path = self.folder_output / 'maestro_summary.md'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(md))
        print(f'\t\t- Summary written to {IOtools.clipstr(summary_path)}')

    # --------------------------------------------------------------------------------------------
    # Plotting operations
    # --------------------------------------------------------------------------------------------
    
    @mitim_timer(lambda self: f'Beat #{self.counter_current} ({self.beat.name}) - Plotting')
    def plot(self, fn = None, num_beats = 2, only_beats = None, full_plot = True, summary_only = False):

        print('*** Plotting MAESTRO ******************************************************************** ')

        if fn is None:
            wasProvided = False
            self.fn = GUItools.FigureNotebook("MAESTRO")
        else:
            wasProvided = True
            self.fn = fn

        # summary_only -> only the cross-beat 'special' + 'timings' tabs (no per-beat tabs)
        if num_beats>0 and not summary_only:
            self._plot_beats(self.fn, num_beats = num_beats, only_beats = only_beats, full_plot = full_plot)
        ps, ps_lab = self._plot_results(self.fn, summary_only = summary_only)

        if not wasProvided:
            self.fn.show()

        return ps, ps_lab

    def _plot_beats(self, fn, num_beats = 2, only_beats = None, full_plot = True):

        beats_keys = sorted(sorted(list(self.beats.keys()),reverse=True)[:num_beats])
        for i,counter in enumerate(beats_keys):
            beat = self.beats[counter]
            if only_beats is None or only_beats == beat.name:

                print(f'\t- Plotting beat #{counter}...')
                try:
                    log_file = self.folder_logs / f'plot_{counter}.log' if (not self.terminal_outputs) else None
                    with LOGtools.conditional_log_to_file(write_log=not ENABLE_EMBED,log_file=log_file):
                        msg = beat.plot(fn = self.fn, counter = i, full_plot = full_plot)
                    print(msg)
                except FileNotFoundError:
                    print(f'\t\t- Could not plot beat #{counter} because some files are missing', typeMsg = 'w')
                except Exception as e:
                    print(f'\t\t- Could not plot beat #{counter} because of an error: {e}', typeMsg = 'w')

    def _plot_results(self, fn, summary_only = False):

        print('\t- Plotting MAESTRO results...')

        return MAESTROplot.plot_results(self, fn, summary_only = summary_only)


def read_warning(file, d, label):

    # Read contents
    with open(file, 'r') as f:
        log_lines = f.readlines()

    for i in range(len(log_lines)):
        if '*WARNING*' in log_lines[i]:
            d[f'{label}_${i}']= log_lines[i].replace('\t','').replace('\n','').replace('[*WARNING*]','')

    return d


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHFJ]')


def _capture_print_info(gacode_file):
    '''
    Run gacode_state.printInfo() and return its stdout with ANSI color codes
    stripped. Temporarily forces verbose level >= 3 so printMsg actually emits.
    '''
    import io
    import contextlib
    from mitim_tools.gacode_tools import PROFILEStools
    from mitim_tools.misc_tools import CONFIGread

    p = PROFILEStools.gacode_state(gacode_file)
    p.derive_quantities()

    buf = io.StringIO()
    original = CONFIGread.read_verbose_level
    CONFIGread.read_verbose_level = lambda: 5
    try:
        with contextlib.redirect_stdout(buf):
            p.printInfo(label='input.gacode_final')
    finally:
        CONFIGread.read_verbose_level = original

    return _ANSI_RE.sub('', buf.getvalue())


def _render_special_png(maestro, out_path):
    '''
    Render the per-beat "special quantities" evolution (BetaN, Pfus, Q, density,
    current, confinement, ...) to a PNG, reusing the exact same figure mosaic and
    MAESTROplot.plot_special_quantities used by the interactive 'MAESTRO special'
    tab. Returns the output Path.
    '''
    import matplotlib.pyplot as plt
    from mitim_modules.maestro.utils import MAESTROplot

    _, ps, ps_lab = MAESTROplot.collect_beat_states(maestro)
    for p in ps:
        p.derive_quantities()

    fig = plt.figure(figsize=(16, 9))
    axs = fig.subplot_mosaic(
        """
        ABGIK
        ABGIK
        AEGIK
        DEHJL
        DFHJL
        DFHJL
        """,
        gridspec_kw={"wspace": 0.55},
    )
    MAESTROplot.plot_special_quantities(ps, ps_lab, axs)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _render_timing_png(timing_jsonl, out_path):
    '''
    Render the per-beat cumulative wall-time (the 'MAESTRO timings' view) to a PNG,
    reusing IOtools.plot_timings. Returns the output Path.
    '''
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.35)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[1, 0], sharex=ax_A)
    ax_C = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[1, 1])
    IOtools.plot_timings(timing_jsonl, axs=[ax_A, ax_B], ax_summary=ax_C, ax_total=ax_D, log=False)
    ax_C.set_title("Time per Beat", fontsize=9)
    ax_D.set_title("Total by type", fontsize=9)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _render_beat_flow_png(beats, wall_times, out_path):
    '''
    Render the beat-flow diagram as a PNG (boxes + arrows). Wraps to multiple
    rows when the chain is long. Returns the output Path.
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from mitim_modules.maestro.utils.MAESTRObeat import _format_seconds

    COLORS = {
        'transp':    '#f4a896',  # salmon
        'portals':   '#9ec5fe',  # sky-blue
        'eped':      '#b7e4c7',  # light green
        'lengyel':   '#fff3b0',  # pale yellow
        'sharpness': '#e0c3fc',  # lavender
        'confinement': '#a8dadc',  # light teal
    }
    DEFAULT_COLOR = '#d0d0d0'

    items = list(beats.items())  # [(counter, beat_obj), ...] insertion order
    n = len(items)
    if n == 0:
        # Empty diagram — still emit something
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, '(no beats)', ha='center', va='center')
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return out_path

    # Wrap layout: up to PER_ROW per row
    PER_ROW = 6
    rows = [items[i:i + PER_ROW] for i in range(0, n, PER_ROW)]
    nrows = len(rows)
    ncols = max(len(r) for r in rows)

    # Box geometry (in axis data units)
    box_w, box_h = 2.0, 1.4
    gap_x, gap_y = 0.7, 1.0

    fig_w = ncols * (box_w + gap_x) + 1.0
    fig_h = nrows * (box_h + gap_y) + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Positions: row 0 at top, row index increases downward
    positions = {}  # counter -> (x_center, y_center)
    for r, row in enumerate(rows):
        y = (nrows - 1 - r) * (box_h + gap_y) + box_h / 2
        # Even rows L->R, odd rows R->L (serpentine flow, looks more natural)
        order = row if r % 2 == 0 else list(reversed(row))
        for c, (counter, beat_obj) in enumerate(order):
            x = c * (box_w + gap_x) + box_w / 2
            positions[counter] = (x, y, r, c)

    # Draw boxes
    for counter, beat_obj in items:
        x, y, r, c = positions[counter]
        color = COLORS.get(beat_obj.name, DEFAULT_COLOR)
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.05,rounding_size=0.15',
            facecolor=color, edgecolor='black', linewidth=1.5,
        )
        ax.add_patch(box)
        wt = wall_times.get(counter)
        wt_str = _format_seconds(wt) if wt is not None else '–'
        ax.text(x, y + 0.25, f'Beat {counter}',
                ha='center', va='center', fontsize=10, color='black')
        ax.text(x, y - 0.05, beat_obj.name.upper(),
                ha='center', va='center', fontsize=13, color='black', fontweight='bold')
        ax.text(x, y - 0.4, wt_str,
                ha='center', va='center', fontsize=9, color='#333333', style='italic')

    # Draw arrows between consecutive beats
    ordered = [it[0] for it in items]
    for prev_counter, next_counter in zip(ordered, ordered[1:]):
        x0, y0, r0, c0 = positions[prev_counter]
        x1, y1, r1, c1 = positions[next_counter]
        if r0 == r1:
            # Same row: horizontal arrow from edge to edge
            start = (x0 + box_w / 2, y0)
            end = (x1 - box_w / 2, y1) if c1 > c0 else (x1 + box_w / 2, y1)
            if c1 < c0:  # serpentine reversed row
                start = (x0 - box_w / 2, y0)
            arrow = FancyArrowPatch(
                start, end, arrowstyle='-|>', mutation_scale=18,
                color='#333', linewidth=1.5, shrinkA=0, shrinkB=0,
            )
        else:
            # Row wrap: drop straight down from the last box of previous row
            start = (x0, y0 - box_h / 2)
            end = (x1, y1 + box_h / 2)
            arrow = FancyArrowPatch(
                start, end, arrowstyle='-|>', mutation_scale=18,
                color='#333', linewidth=1.5,
                connectionstyle='arc3,rad=0.0',
                shrinkA=0, shrinkB=0,
            )
        ax.add_patch(arrow)

    ax.set_xlim(-0.3, ncols * (box_w + gap_x) + 0.3)
    ax.set_ylim(-0.3, nrows * (box_h + gap_y) + 0.3)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _parse_beat_wall_times(timing_file):
    '''
    Parse Outputs/Performance/timing.jsonl produced by `mitim_timer` and return
    {beat_counter: total_seconds} aggregating all phases (Initializer, Preparation,
    Run + Finalization, Finalizing) per beat. The mitim_timer log labels carry the
    beat number — match it via regex. Returns {} on any parse failure.
    '''
    import json

    if not timing_file.exists():
        return {}

    pattern = re.compile(r'Beat\s*#\s*(\d+)')
    totals = {}
    try:
        with open(timing_file, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                label = d.get('script', '')
                if 'duration_s' not in d:
                    continue
                try:
                    seconds = float(d['duration_s'])
                except (TypeError, ValueError):
                    continue
                m = pattern.search(label)
                if not m:
                    continue
                counter = int(m.group(1))
                totals[counter] = totals.get(counter, 0.0) + seconds
    except Exception:
        return {}
    return totals