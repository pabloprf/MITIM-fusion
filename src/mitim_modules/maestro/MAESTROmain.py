import torch
import os
import copy
import hashlib
import numpy as np
from mitim_tools.gs_tools import FREEGStools
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.opt_tools 	 	import STRATEGYtools
from mitim_modules.portals 		import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools, GUItools, CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools import __version__
from IPython import embed

'''
MAESTRO:
    Modular and Accelerated Engine for Simulation of Transport and Reactor Optimization

If MAESTRO is the orchestrator, then BEAT is each of the beats (steps) that MAESTRO orchestrates
'''

# --------------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------------

class maestro:

    def __init__(self, folder, terminal_outputs = False, master_restart = False):
        '''
        Inputs:
            - folder: Main folder where all the beats will be saved
            - terminal_outputs: If True, all outputs will be printed to terminal. If False, they will be saved to a log file per beat step
        '''

        self.folder = IOtools.expandPath(folder)
        
        self.folder_output = f'{self.folder}/Outputs/'
        self.folder_logs = f'{self.folder_output}/Logs/'
        os.makedirs(self.folder_logs, exist_ok=True)

        self.folder_beats = f'{self.folder}/Beats/'
        os.makedirs(self.folder_beats, exist_ok=True)

        self.save_file = f'{self.folder_output}/maestro_save.pkl'
        self.terminal_outputs = terminal_outputs

        self.beats = {}
        self.counter = 0

        self.master_restart = master_restart # If True, all next beats will be restarted

        self.parameters_trans_beat = {} # I can save parameters that can be useful for future beats

        print('\n -----------------------------------------------------------------------------------')
        print(f'MAESTRO run (MITIM version {__version__})')
        print('-----------------------------------------------------------------------------------')
        print(f'folder: {self.folder}')

    def define_beat(self, beat, initializer = None, restart = False):

        self.counter += 1
        if beat == 'transp':
            print(f'\n- Beat {self.counter}: TRANSP ********************************************************************')
            self.beats[self.counter] = transp_beat(self)
        elif beat == 'portals':
            print(f'\n- Beat {self.counter}: PORTALS ********************************************************************')
            self.beats[self.counter] = portals_beat(self)

        # Access current beat easily
        self.beat = self.beats[self.counter]

        # Define initializer
        self.beat.define_initializer(initializer)

        # Check here if the beat has already been performed
        self._check(restart = restart or self.master_restart )

    @mitim_timer('\t\t* Checker')
    def _check(self, restart = False, **kwargs):
        '''
        Note:
            After each beat, the results are passed to an output folder.
            If the required files are already there, the beat will not be run again.
            It is also assumed that the results were correct if they were put there, so
            the checks should happen in the finalize() method of each beat.
        '''

        print('\t- Checking...')
        log_file = f'{self.folder_logs}/beat_{self.counter}_check.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            exists = self.beat.check(restart = restart, **kwargs)
            self.beat.run_flag = not exists

        # If this beat is restarted, all next beats will be restarted
        if self.beat.run_flag:
            if not self.master_restart:
                print('\t\t- Since this step needs to start from scratch, all next ones will too', typeMsg = 'i')
            self.master_restart = True
            
    @mitim_timer('\t\t* Initializer')
    def initialize(self, *args, **kwargs):

        print('\t- Initializing...')
        if self.beat.run_flag:
            log_file = f'{self.folder_logs}/beat_{self.counter}_ini.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.initialize(*args, **kwargs)
        else:
            print('\t\t- Skipping beat initialization because this beat was already run', typeMsg = 'i')

    @mitim_timer('\t\t* Preparation')
    def prepare(self, *args, **kwargs):

        print('\t- Preparing...')
        if self.beat.run_flag:
            log_file = f'{self.folder_logs}/beat_{self.counter}_prep.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.prepare(*args, **kwargs)
        else:
            print('\t\t- Skipping beat preparation because this beat was already run', typeMsg = 'i')

    @mitim_timer('\t\t* Run')
    def run(self, **kwargs):

        # Run 
        print('\t- Running...')
        if self.beat.run_flag:
            log_file = f'{self.folder_logs}/beat_{self.counter}_run.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
                self.beat.run(**kwargs)

                # Finalize
                self.beat.finalize()

        else:
            print('\t\t- Skipping beat run because this beat was already run', typeMsg = 'i')

        # Inform next beats
        log_file = f'{self.folder_logs}/beat_{self.counter}_inform.log' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file):
            self.beat.inform_save()

    @mitim_timer('\t\t* Finalizing')
    def finalize(self):

        print('\t- Finalizing MAESTRO run...')
        
        log_file = f'{self.folder_output}/beat_final' if (not self.terminal_outputs) else None
        with IOtools.conditional_log_to_file(log_file=log_file, msg = f'\t\t* Log info being saved to {IOtools.clipstr(log_file)}'):
            
            self.beat.finalize_maestro()

    @mitim_timer('\t\t* Plotting')
    def plot(self, fn = None, num_beats = 2):

        print('*** Plotting MAESTRO ******************************************************************** ')

        if fn is None:
            wasProvided = False
            self.fn = GUItools.FigureNotebook("MAESTRO")
        else:
            wasProvided = True
            self.fn = fn

        self._plot_beats(self.fn, num_beats = num_beats)

        if not wasProvided:
            self.fn.show()

    def _plot_beats(self, fn, num_beats = 2):

        beats_keys = sorted(sorted(list(self.beats.keys()),reverse=True)[:num_beats])
        for i,counter in enumerate(beats_keys):
            beat = self.beats[counter]
            print(f'\t- Plotting beat #{counter}...')
            log_file = f'{self.folder_logs}/plot_{counter}.log' if (not self.terminal_outputs) else None
            with IOtools.conditional_log_to_file(log_file=log_file):
                msg = beat.plot(fn = self.fn, counter = i)
            print(msg)


# --------------------------------------------------------------------------------------------
# Generic beat class with required methods
# --------------------------------------------------------------------------------------------

class beat:

    def __init__(self, maestro_instance, beat_name = 'generic'):

        self.maestro_instance = maestro_instance
        self.folder_beat = f'{self.maestro_instance.folder_beats}/Beat_{self.maestro_instance.counter}/'

        # Where to run it
        self.folder = f'{self.folder_beat}/run_{beat_name}/'
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

    def plot(self, *args, **kwargs):
        return ''

    def inform_save(self, *args, **kwargs):
        pass

    def _inform(self, *args, **kwargs):
        pass

# --------------------------------------------------------------------------------------------
# Generic initializer class with required methods
# --------------------------------------------------------------------------------------------

class beat_initializer:
    
    def __init__(self, beat_instance, label = ''):
            
        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder_beat}/initializer_{label}/'

        if len(label) > 0:
            os.makedirs(self.folder, exist_ok=True)

    def __call__(self, *args, **kwargs):
        pass

# --------------------------------------------------------------------------------------------
# Beat: TRANSP
# --------------------------------------------------------------------------------------------

class transp_beat(beat):

    def __init__(self, maestro_instance):            
        super().__init__(maestro_instance, beat_name = 'transp')

    # --------------------------------------------------------------------------------------------
    # Checker
    # --------------------------------------------------------------------------------------------
    def check(self, restart = False):
        return super().check(restart=restart, folder_search = self.folder_output, suffix = '.CDF')

    # --------------------------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------------------------
    def define_initializer(self, initializer):

        if initializer is None:
            self.initializer = beat_initializer(self)
        elif initializer == 'freegs':
            self.initializer = transp_initializer_from_freegs(self)
        elif initializer == 'portals':
            self.initializer = transp_initializer_from_portals(self)
        else:
            raise ValueError(f'Initializer "{initializer}" not recognized')

    def initialize(self, flattop_window = 0.05,*args,  **kwargs):

        transition_window       = 0.1    # s
        currentheating_window   = 0.001  # s

        # Define timings
        self.time_init = 0.0                                                # Start with D3D equilibrium
        self.time_transition = self.time_init+ transition_window            # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End

        # Initialize
        self.initializer(*args,**kwargs)

    def prepare(self, letter = None, shot = None, **transp_namelist):

        '''
        Using some smart defaults to avoid repeating TRANSP runid
            shot will be 5 digits that depend on the last subfolder
                e.g. run_cmod1 -> '94351', run_cmod2 -> '94352', run_d3d1 -> '72821', etc
            letter will depend on username in this machine, if it can be found
                e.g. pablorf -> 'P"
        '''
        if shot is None:
            folder_last = os.path.basename(os.path.normpath(self.maestro_instance.folder))
            shot = string_to_sequential_5_digit_number(folder_last)

        if letter is None:
            username = IOtools.expandPath('$USER')
            letter = username[0].upper()
            if letter == '$':
                letter = 'A'
        # ---------------------------------------------------------

        # Define run parameters
        self.shot = shot
        self.runid = letter + str(self.maestro_instance.counter).zfill(2)

        # Use initializer to prepare beat
        self.initializer.prepare_to_beat()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generatic TRANSP operation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        transp_namelist_mod = copy.deepcopy(transp_namelist)

        if 'timings' in transp_namelist_mod:
            raise ValueError('Cannot define timings in MAESTRO transp_namelist')
        else:
            transp_namelist_mod['timings'] = {
                "time_start": self.time_init,
                "time_current_diffusion": self.time_diffusion,
                "time_end": self.time_end
            }

        if 'Ufiles' in transp_namelist_mod:
            raise ValueError('Cannot define UFILES in MAESTRO transp_namelist')
        else:
            transp_namelist_mod['Ufiles'] = ["qpr","mry","cur","vsf","ter","ti2","ner","rbz","lim","zf2"]

        # Write namelist
        self.transp.write_namelist(**transp_namelist_mod)

        # Trick needed to avoid quval error when starting from D3D
        self.transp.populate_time.from_freegs(0.0, 1.67, 0.6, 1.75, 0.38, 0.0, 0.0, 0.074, 1.6, 2.0)
        
        # ICRF on
        qm_He3 = 2/3
        Frequency_He3 = self.initializer.B_T * (2*np.pi/qm_He3)
        self.transp.icrf_on_time(self.time_diffusion, power_MW = self.initializer.PichT_MW, freq_MHz = Frequency_He3)
        
        # Write Ufiles
        self.transp.write_ufiles()

    # --------------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------------
    def run(self, **kwargs):

        hours_allocation = 8 # 12

        self.transp.run('D3D', mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*hours_allocation, case=self.transp.runid, checkMin=5)
        self.c = self.transp.c

    # --------------------------------------------------------------------------------------------
    # Finalize and plot
    # --------------------------------------------------------------------------------------------

    def finalize(self):

        # Copy to outputs
        os.system(f'cp {self.folder}/{self.shot}{self.runid}.CDF {self.folder_output}/.')
        os.system(f'cp {self.folder}/{self.shot}{self.runid}tr.log {self.folder_output}/.')

    def plot(self,  fn = None, counter = 0):

        isitfinished = self.check()

        if isitfinished:
            c = CDFtools.transp_output(self.folder_output)
            c.plot(fn = fn, counter = counter) 
            msg = '\t\t- Plotting of TRANSP beat done'
        else:
            msg = '\t\t- Cannot plot because the TRANSP beat has not finished yet'

        return msg
        
    # --------------------------------------------------------------------------------------------
    # Finalize in case this is the last beat
    # --------------------------------------------------------------------------------------------

    def finalize_maestro(self):

        cdf = CDFtools.transp_output(f"{self.folder}/{self.transp.shot}{self.transp.runid}.CDF")
        self.maestro_instance.final_p = cdf.to_profiles()
        
        final_file = f'{self.maestro_instance.folder_output}/input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

# TRANSP initializers ************************************************************************

class transp_initializer(beat_initializer):

    def __init__(self, beat_instance, label = ''):
        super().__init__(beat_instance, label = label)

class transp_initializer_from_freegs(transp_initializer):

    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'freegs')
            
    def __call__(
        self,
        R,
        a,
        kappa_sep,
        delta_sep,
        zeta_sep,
        z0,
        Ip_MA,
        B_T,
        Zeff,
        PichT_MW,
        p0_MPa = 1.0,
        ne0_20 = 1.0,
        profiles = {}):

        self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0 = R, a, kappa_sep, delta_sep, zeta_sep, z0
        self.p0_MPa, self.Ip_MA, self.B_T = p0_MPa, Ip_MA, B_T
        self.ne0_20, self.Zeff, self.PichT_MW = ne0_20, Zeff, PichT_MW
        self.profiles = profiles

        # If profiles exist, substitute the pressure and density guesses by something better (not perfect though, no ions)
        if 'ne' in profiles:
            print('\t- Using ne profile instead of the ne0 guess')
            self.ne0_20 = profiles['ne'][1][0]
        if 'Te' in profiles:
            print('\t- Using Te profile for a better estimation of pressure, instead of the p0 guess')
            Te0_keV = profiles['Te'][1][0]
            self.p0_MPa = 2 * (Te0_keV*1E3) * 1.602176634E-19 * (self.ne0_20 * 1E20) * 1E-6 #MPa
            
        # FreeGS
        self.f = FREEGStools.freegs_millerized(self.R, self.a, self.kappa_sep, self.delta_sep, self.zeta_sep, self.z0)
        self.f.prep(self.p0_MPa, self.Ip_MA, self.B_T)
        self.f.solve()
        self.f.derive()
        self.f.write(f'{self.folder}/freegs.geqdsk')

    def prepare_to_beat(self):

        times = [self.beat_instance.time_transition,self.beat_instance.time_end+1.0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FreeGS to TRANSP
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.transp = self.f.to_transp(
            folder = self.beat_instance.folder,
            shot = self.beat_instance.shot, runid = self.beat_instance.runid, times = times,
            ne0_20 = self.ne0_20, Zeff = self.Zeff, PichT_MW = self.PichT_MW)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add profiles
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._add_profiles(times)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pass to main class' beat
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.beat_instance.transp = self.transp

    def _add_profiles(self, times):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add profiles  # TO FIX ROA VA RHO???, psi, etc
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if 'Te' in self.profiles:
            for time in times:
                self.transp.add_variable_time(time, self.profiles['Te'][0], self.profiles['Te'][1]*1E3, variable='TEL')
        if 'Ti' in self.profiles:
            for time in times:
                self.transp.add_variable_time(time, self.profiles['Ti'][0], self.profiles['Ti'][1]*1E3, variable='TIO')
        if 'ne' in self.profiles:
            for time in times:
                self.transp.add_variable_time(time, self.profiles['ne'][0], self.profiles['ne'][1]*1E20*1E-6, variable='NEL')

class transp_initializer_from_geqdsk(transp_initializer):
    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'geqdsk')

    # TO DO

class transp_initializer_from_portals(transp_initializer):

    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'portals')

    def __call__(self):

        # Load PORTALS from previous beat: profiles with best residual
        folder =  self.beat_instance.maestro_instance.beats[self.beat_instance.maestro_instance.counter-1].folder_output

        self.portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
        
        self.p = self.portals_output.mitim_runs[self.portals_output.ibest]['powerstate'].profiles

        self.p.writeCurrentStatus(file=self.folder+'/input.gacode' )

        # Parameters needed for later
        self.PichT_MW = self.p.derived['qRF_MWmiller'][-1]
        self.B_T = self.p.profiles['bcentr(T)'][0]

    def prepare_to_beat(self):

        times = [self.beat_instance.time_transition,self.beat_instance.time_end+1.0]

        self.transp = self.p.to_transp(
            folder = self.beat_instance.folder,
            shot = self.beat_instance.shot, runid = self.beat_instance.runid, times = times)

        # Pass to main class' beat
        self.beat_instance.transp = self.transp


# --------------------------------------------------------------------------------------------
# Beat: PORTALS
# --------------------------------------------------------------------------------------------

class portals_beat(beat):

    def __init__(self, maestro_instance):
        super().__init__(maestro_instance, beat_name = 'portals')

    # --------------------------------------------------------------------------------------------
    # Checker
    # --------------------------------------------------------------------------------------------

    def check(self, restart = False):
        return super().check(restart=restart, folder_search = self.folder_output+'/Outputs', suffix = '_object.pkl')

    # --------------------------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------------------------
    def define_initializer(self, initializer):

        if initializer is None:
            self.initializer = beat_initializer(self)
        elif initializer == 'transp':
            self.initializer = portals_initializer_from_transp(self)

    def initialize(self,*args,  **kwargs):

        self.initializer(*args,**kwargs)

    def prepare(self, use_previous_residual = True, PORTALSparameters = {}, MODELparameters = {}, optimization_options = {}, INITparameters = {}):

        self.fileGACODE = self.initializer.file_gacode

        self.PORTALSparameters = PORTALSparameters
        self.MODELparameters = MODELparameters
        self.optimization_options = optimization_options
        self.INITparameters = INITparameters

        self._inform(use_previous_residual = use_previous_residual)

    def _inform(self, use_previous_residual = True):
        '''
        Prepare next PORTALS runs accounting for what previous PORTALS runs have done
        '''
        if use_previous_residual and ('portals_neg_residual_obj' in self.maestro_instance.parameters_trans_beat):
            self.optimization_options['maximum_value'] = self.maestro_instance.parameters_trans_beat['portals_neg_residual_obj']
            self.optimization_options['maximum_value_is_rel'] = False

            print(f"\t\t- Using previous residual goal as maximum value for optimization: {self.optimization_options['maximum_value']}")

    # --------------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------------
    def run(self, **kwargs):

        restart = kwargs.get('restart', False)

        portals_fun  = PORTALSmain.portals(self.folder)

        for key in self.PORTALSparameters:
            portals_fun.PORTALSparameters[key] = self.PORTALSparameters[key]
        for key in self.MODELparameters:
            portals_fun.MODELparameters[key] = self.MODELparameters[key]
        for key in self.optimization_options:
            portals_fun.optimization_options[key] = self.optimization_options[key]
        for key in self.INITparameters:
            portals_fun.INITparameters[key] = self.INITparameters[key]

        portals_fun.prep(self.fileGACODE,self.folder,hardGradientLimits = [0,2])

        self.prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN = restart, askQuestions = False)

        self.prf_bo.run()

    # --------------------------------------------------------------------------------------------
    # Finalize and plot
    # --------------------------------------------------------------------------------------------
    def finalize(self):

        # Remove output folders
        os.system(f'rm -r {self.folder_output}/*')

        # Copy to outputs
        os.system(f'cp -r {self.folder}/Outputs {self.folder_output}/Outputs')

        # Add final input.gacode
        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)
        p = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
        p.writeCurrentStatus(file=f'{self.folder_output}/input.gacode' )

    def inform_save(self):

        # Save the residual goal to use in the next PORTALS beat
        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)
        max_value_neg_residual = portals_output.step.stepSettings['optimization_options']['maximum_value']
        self.maestro_instance.parameters_trans_beat['portals_neg_residual_obj'] = max_value_neg_residual
        print(f'\t\t- Maximum value of negative residual saved for future beats: {max_value_neg_residual}')

        # Save the best profiles to use in the next PORTALS beat to avoid issues with TRANSP coarse grid
        self.maestro_instance.parameters_trans_beat['portals_profiles'] = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles

    def plot(self,  fn = None, counter = 0, full_opt = True):

        isitfinished = self.check()

        if isitfinished:
            folder = self.folder_output
        else:
            folder = self.folder
        
        if full_opt:
            opt_fun = STRATEGYtools.opt_evaluator(folder)
            opt_fun.fn = fn
            opt_fun.plot_optimization_results(analysis_level=4)
        else:
            portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
            portals_output.fn = fn
            portals_output.plotPORTALS()
            
        msg = '\t\t- Plotting of PORTALS beat done'

        return msg

    # --------------------------------------------------------------------------------------------
    # Finalize in case this is the last beat
    # --------------------------------------------------------------------------------------------

    def finalize_maestro(self):

        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder)
        self.maestro_instance.final_p = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
        
        final_file = f'{self.maestro_instance.folder_output}/input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

# PORTALS initializers ************************************************************************

class portals_initializer(beat_initializer):

    def __init__(self, beat_instance, label = ''):
        super().__init__(beat_instance, label = label)

class portals_initializer_from_transp(portals_initializer):

    def __init__(self, beat_instance):
        super().__init__(beat_instance, label = 'transp')

    def __call__(self, time_extraction = None, use_previous_portals_profiles = True):

        # Load TRANSP results from previous beat
        beat_num = self.beat_instance.maestro_instance.counter-1
        self.cdf = CDFtools.transp_output(self.beat_instance.maestro_instance.beats[beat_num].folder_output)

        # Extract profiles at time_extraction
        time_extraction = self.cdf.t[self.cdf.ind_saw -1] # Since the time is coarse in MAESTRO TRANSP runs, make I'm not extracting with profiles sawtoothing
        self.p = self.cdf.to_profiles(time_extraction=time_extraction)

        if use_previous_portals_profiles and ('portals_profiles' in self.beat_instance.maestro_instance.parameters_trans_beat):
            print('\t- Using previous PORTALS thermal kinetic profiles instead of the TRANSP profiles')
            p_prev = self.beat_instance.maestro_instance.parameters_trans_beat['portals_profiles']

            self.p.changeResolution(rho_new=p_prev.profiles['rho(-)'])
            
            self.p.profiles['te(keV)'] = p_prev.profiles['te(keV)']
            self.p.profiles['ne(10^19/m^3)'] = p_prev.profiles['ne(10^19/m^3)']
            for i,sp in enumerate(self.p.Species):
                if sp['S'] == 'therm':
                    self.p.profiles['ti(keV)'][:,i] = p_prev.profiles['ti(keV)'][:,i]
                    self.p.profiles['ni(10^19/m^3)'][:,i] = p_prev.profiles['ni(10^19/m^3)'][:,i]

        self.file_gacode = f"{self.folder}/input.gacode"
        self.p.writeCurrentStatus(file=self.file_gacode)

# --------------------------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------------------------

def procreate(y_top = 2.0, y_sep = 0.1, w_top = 0.07, aLy = 2.0, w_a = 0.3):
    
    roa = np.linspace(0.0, 1-w_top, 100)
    aL_profile = np.zeros_like(roa)
    linear_region = roa <= w_a
    aL_profile[linear_region] = (aLy / w_a) * roa[linear_region]
    aL_profile[~linear_region] = aLy
    y = CALCtools.integrateGradient(torch.from_numpy(roa).unsqueeze(0), torch.from_numpy(aL_profile).unsqueeze(0), y_top).numpy()
    roa = np.append( roa, 1.0)
    y = np.append(y, y_sep)

    return roa, y

@mitim_timer('\t- MAESTRO')
def simple_maestro_workflow(
    folder,
    geometry,
    parameters,
    Tbc_keV,
    nbc_20,
    TGLFsettings = 6,
    DTplasma = True,
    terminal_outputs = False,
    full_loops = 2, # By default, do 2 loops of TRANSP-PORTALS
    quality = {
        'maximum_value': 1e-2,  # x100 better residual
        'BO_iterations': 20,
        'flattop_window': 0.15, # s
        }
    ):

    m = maestro(folder, terminal_outputs = terminal_outputs)
    
    # ---------------------------------------------------------
    # beat 0: Define info
    # ---------------------------------------------------------

    # Simple profiles

    w_top, w_a = 0.05, 0.3
    rho, Te = procreate(y_top = Tbc_keV, y_sep = 0.1, w_top = w_top, aLy = 1.7, w_a = w_a)
    rho, Ti = procreate(y_top = Tbc_keV, y_sep = 0.1, w_top = w_top, aLy = 1.5, w_a = w_a)
    rho, ne = procreate(y_top = nbc_20, y_sep = nbc_20/3.0, w_top = w_top, aLy = 0.2, w_a = w_a)
    profiles = {'Te': [rho, Te],'Ti': [rho, Ti],'ne': [rho, ne]}

    # Faster TRANSP (different than defaults)
    transp_namelist = {
        'Pich'   : True,
        'dtEquilMax_ms': 1.0,       # Higher resolution than default (10.0) to avoid quval error
        'dtHeating_ms' : 5.0,       # Default
        'dtOut_ms' : 10.0,
        'dtIn_ms' : 10.0,
        'nzones' : 60,
        'nzones_energetic' : 20,    # Default but lower than what I used to use
        'nzones_distfun' : 10,      # Default but lower than what I used to use    
        'MCparticles' : 1e4,
        'toric_ntheta' : 64,        # Default values of TORIC, but lower than what I used to use
        'toric_nrho' : 128,         # Default values of TORIC, but lower than what I used to use
        'DTplasma': DTplasma
    }

    # Simple PORTALS

    portals_namelist = {
        "PORTALSparameters": {
            "launchEvaluationsAsSlurmJobs": not CONFIGread.isThisEngaging(),
            "forceZeroParticleFlux": True
        },
        "MODELparameters": {
            "RoaLocations": [0.35,0.55,0.75,0.875,0.9],
            "transport_model": {"turbulence":'TGLF',"TGLFsettings": TGLFsettings, "extraOptionsTGLF": {}}
        },
        "INITparameters": {
            "FastIsThermal": True
        },
        "optimization_options": {
            "BO_iterations": quality.get('BO_iterations', 20),
            "maximum_value": quality.get('maximum_value', 1e-2),
            "maximum_value_is_rel": True,
        }
    }

    # ---------------------------------------------------------
    # beat N: TRANSP from FreeGS
    # ---------------------------------------------------------

    m.define_beat('transp', initializer='freegs')
    m.initialize(**geometry,**parameters, profiles = profiles, flattop_window = quality.get('flattop_window', 0.15))
    m.prepare(**transp_namelist)
    m.run()

    # ---------------------------------------------------------
    # beat N+1: PORTALS from TRANSP
    # ---------------------------------------------------------

    m.define_beat('portals', initializer='transp')
    m.initialize()
    m.prepare(**portals_namelist)
    m.run()

    for i in range(full_loops-1):
        # ---------------------------------------------------------
        # beat N: TRANSP from PORTALS
        # ---------------------------------------------------------

        m.define_beat('transp', initializer='portals')
        m.initialize(flattop_window = quality.get('flattop_window', 0.15))
        m.prepare(**transp_namelist)
        m.run()

        # ---------------------------------------------------------
        # beat N+1: PORTALS from TRANSP
        # ---------------------------------------------------------

        m.define_beat('portals', initializer='transp')
        m.initialize()
        m.prepare(**portals_namelist)
        m.run()

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------

    m.finalize()

    return m

# chatGPT 4o (08/18/2024)
def string_to_sequential_5_digit_number(input_string):
    # Split the input string into the base and the numeric suffix
    base_part = input_string[:-1]
    sequence_digit = int(input_string[-1])

    # Create a hash of the base part using SHA-256
    hash_object = hashlib.sha256(base_part.encode())
    
    # Convert the hash to an integer
    hash_int = int(hash_object.hexdigest(), 16)
    
    # Take the hash modulo 10,000 to get a 4-digit number
    four_digit_number = hash_int % 10000
    
    # Combine the 4-digit hash with the sequence digit to get a 5-digit number
    five_digit_number = (four_digit_number * 10) + sequence_digit
    
    # Ensure it's always 5 digits by adding leading zeros if necessary
    return f'{five_digit_number:05d}'
