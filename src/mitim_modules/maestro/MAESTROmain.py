import copy
import sys
import datetime
import shutil
from mitim_tools.misc_tools import IOtools, GUItools, LOGtools
from mitim_modules.maestro.utils import MAESTROplot
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __version__, __mitimroot__
from IPython import embed

from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat
from mitim_modules.maestro.utils.EPEDbeat import eped_beat
from mitim_modules.maestro.utils.MAESTRObeat import creator_from_eped, creator_from_parameterization, creator
from mitim_modules.maestro.utils.MAESTRObeat import beat as beat_generic

'''
MAESTRO:
    Modular and Accelerated Engine for Simulation of Transport and Reactor Optimization
 (If MAESTRO is the orchestrator, then BEAT is each of the beats (steps) that MAESTRO orchestrates)

'''

class maestro:

    def __init__(
            self,
            folder,
            terminal_outputs = False,
            master_cold_start = False,
            overall_log_file = True,
            keep_all_files = True
            ):
        '''
        Inputs:
            - folder: Main folder where all the beats will be saved
            - terminal_outputs: If True, all outputs will be printed to terminal. If False, they will be saved to a log file per beat step
        '''

        self.terminal_outputs = terminal_outputs
        self.master_cold_start = master_cold_start        # If True, all beats will be cold_started
        self.keep_all_files = keep_all_files             # If True, all files will be kept, if False, only the final output files will be kept

        # --------------------------------------------------------------------------------------------
        # Prepare folders
        # --------------------------------------------------------------------------------------------

        self.folder = IOtools.expandPath(folder)
        
        self.folder_output = self.folder / "Outputs"
        self.folder_logs = self.folder_output / "Logs"
        self.folder_beats = self.folder / "Beats"

        self.folder_logs.mkdir(parents=True, exist_ok=True)
        self.folder_beats.mkdir(parents=True, exist_ok=True)

        # If terminal outputs, I also want to keep track of what has happened in a log file
        if terminal_outputs and overall_log_file:
            sys.stdout = LOGtools.Logger(logFile=self.folder_output / "maestro.log", writeAlsoTerminal=True)

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

        # Access current beat easily
        self.beat = self.beats[self.counter_current]

        # Define initializer
        self.beat.define_initializer(initializer)

        # Check here if the beat has already been performed
        self.check(cold_start = cold_start or self.master_cold_start )

    def define_creator(self, method, **kwargs_creator):
        '''
        To initialize some profile functional form
        '''
        if method == 'eped':
            self.beat.initialize.profile_creator = creator_from_eped(self.beat.initialize,**kwargs_creator)
        elif method == 'parameterization':
            self.beat.initialize.profile_creator = creator_from_parameterization(self.beat.initialize,**kwargs_creator)
        elif method == 'profiles':
            self.beat.initialize.profile_creator = creator(self.beat.initialize,**kwargs_creator)
        else:
            raise ValueError(f'[MITIM] Creator method {method} not recognized')

    # --------------------------------------------------------------------------------------------
    # Beat operations
    # --------------------------------------------------------------------------------------------
    
    @mitim_timer('\t\t* Checker', name_timer=None)
    def check(self, beat_check = None, cold_start = False, **kwargs):
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
        with LOGtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):

            output_file = None
            if not cold_start:
                output_file = IOtools.findFileByExtension(beat_check.folder_output, 'input.gacode', agnostic_to_case=True)
                if output_file is not None:
                    print('\t\t- Output file already exists, not running beat', typeMsg = 'i')
            else:
                print('\t\t- Forced cold_starting of beat', typeMsg = 'i')

            self.beat.run_flag = output_file is None

        # If this beat is cold_started, all next beats will be cold_started
        if self.beat.run_flag:
            if not self.master_cold_start:
                print('\t\t- Since this step needs to start from scratch, all next ones will too', typeMsg = 'i')
            self.master_cold_start = True

        return output_file is not None

    @mitim_timer('\t\t* Initializer', name_timer=None)
    def initialize(self, *args, **kwargs):

        print('\t- Initializing...')
        if self.beat.run_flag:
            log_file = self.folder_logs / f'beat_{self.counter_current}_ini.log' if (not self.terminal_outputs) else None
            with LOGtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                # Initialize: produce self.profiles_current
                self.beat.initialize(*args, **kwargs)

        else:
            print('\t\t- Skipping beat initialization because this beat was already run', typeMsg = 'i')

        log_file = self.folder_logs / f'beat_{self.counter_current}_inform.log' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            # Initializer can also save important parameters
            self.beat.initialize._inform_save()

            if self.profiles_with_engineering_parameters is None:
                # First initialization, freeze engineering parameters
                self._freeze_parameters(profiles = PROFILEStools.PROFILES_GACODE(self.beat.initialize.folder / 'input.gacode'))

    @mitim_timer('\t\t* Preparation', name_timer=None)
    def prepare(self, *args, **kwargs):

        print('\t- Preparing...')
        if self.beat.run_flag:
            log_file = self.folder_logs / f'beat_{self.counter_current}_prep.log' if (not self.terminal_outputs) else None
            with LOGtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                
                # Initialize if necessary
                if not self.beat.initialize_called:
                    print('\t\t- Initializing beat before preparing...')
                    self.beat.initialize()
                # -----------------------------

                self.beat.profiles_current.deriveQuantities()
                
                self.beat.prepare(*args, **kwargs)
        else:
            print('\t\t- Skipping beat preparation because this beat was already run', typeMsg = 'i')

    @mitim_timer('\t\t* Run', name_timer=None)
    def run(self, **kwargs):

        # Run 
        print('\t- Running...')
        if self.beat.run_flag:
            log_file = self.folder_logs / f'beat_{self.counter_current}_run.log' if (not self.terminal_outputs) else None
            with LOGtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.run(**kwargs)

                # Finalize
                self.beat.finalize(**kwargs)

                # Merge parameters, from self.profiles_current take what's needed and merge with the self.profiles_with_engineering_parameters
                print('\t\t- Merging engineering parameters from MAESTRO')
                self.beat.merge_parameters()

        else:
            print('\t\t- Skipping beat run because this beat was already run', typeMsg = 'i')

        # Produce a new self.profiles_with_engineering_parameters from this merged object
        self._freeze_parameters()

        # Inform next beats
        log_file = self.folder_logs / f'beat_{self.counter_current}_inform.log' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(log_file=log_file):
            self.beat._inform_save()

        # To save space, we can remove the contents of the run_ folder, as everything needed is in the output folder
        if not self.keep_all_files:
            for item in self.beat.folder .iterdir():
                IOtools.shutil_rmtree(item) if item.is_dir() else item.unlink()

    def _freeze_parameters(self, profiles = None):

        if profiles is None:
            profiles = PROFILEStools.PROFILES_GACODE(self.beat.folder_output / 'input.gacode')

        print('\t\t- Freezing engineering parameters from MAESTRO')
        self.profiles_with_engineering_parameters = copy.deepcopy(profiles)
        self.profiles_with_engineering_parameters.writeCurrentStatus(file= (self.folder_output / 'input.gacode_frozen'))

    @mitim_timer('\t\t* Finalizing', name_timer=None)
    def finalize(self):

        print('\t- Finalizing MAESTRO run...')
        
        log_file = self.folder_output / 'beat_final' if (not self.terminal_outputs) else None
        with LOGtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            self.beat.finalize_maestro()

    # --------------------------------------------------------------------------------------------
    # Plotting operations
    # --------------------------------------------------------------------------------------------
    
    @mitim_timer('\t\t* Plotting', name_timer=None)
    def plot(self, fn = None, num_beats = 2, only_beats = None, full_plot = True):

        print('*** Plotting MAESTRO ******************************************************************** ')

        if fn is None:
            wasProvided = False
            self.fn = GUItools.FigureNotebook("MAESTRO")
        else:
            wasProvided = True
            self.fn = fn

        self._plot_beats(self.fn, num_beats = num_beats, only_beats = only_beats, full_plot = full_plot)
        self._plot_results(self.fn)

        if not wasProvided:
            self.fn.show()

    def _plot_beats(self, fn, num_beats = 2, only_beats = None, full_plot = True):

        beats_keys = sorted(sorted(list(self.beats.keys()),reverse=True)[:num_beats])
        for i,counter in enumerate(beats_keys):
            beat = self.beats[counter]
            if only_beats is None or only_beats == beat.name:

                print(f'\t- Plotting beat #{counter}...')
                log_file = self.folder_logs / f'plot_{counter}.log' if (not self.terminal_outputs) else None
                with LOGtools.conditional_log_to_file(log_file=log_file):
                    msg = beat.plot(fn = self.fn, counter = i, full_plot = full_plot)
                print(msg)

    def _plot_results(self, fn):

        print('\t- Plotting MAESTRO results...')

        MAESTROplot.plot_results(self, fn)



