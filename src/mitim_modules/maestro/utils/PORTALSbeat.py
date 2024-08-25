import os
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import (
    beat, 
    beat_initializer, 
    initializer_from_profiles, 
    initializer_from_portals, 
    initializer_from_transp
    )
from IPython import embed

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
            self.initializer = initializer_from_transp(self)
        elif initializer == 'profiles':
            self.initializer = initializer_from_profiles(self)
        elif initializer == 'portals':
            self.initializer = initializer_from_portals(self)
        else:
            raise ValueError(f'Initializer "{initializer}" not recognized')

    def initialize(self,*args,  **kwargs):

        self.initializer(*args,**kwargs)

    def prepare(self, use_previous_residual = True, PORTALSparameters = {}, MODELparameters = {}, optimization_options = {}, INITparameters = {}):

        self.fileGACODE = f"{self.folder}/input.gacode"
        self.profiles_current.writeCurrentStatus(file = self.fileGACODE)

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

    def grab_output(self, full = False):

        isitfinished = self.check()

        if isitfinished:
            folder = self.folder_output
        else:
            folder = self.folder

        if full:
            opt_fun = STRATEGYtools.opt_evaluator(folder)
        else:
            opt_fun = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)

        return opt_fun

    def plot(self,  fn = None, counter = 0, full_plot = True):

        opt_fun = self.grab_output(full = full_plot)

        if full_plot:
            opt_fun.fn = fn
            opt_fun.plot_optimization_results(analysis_level=4)
        else:
            fig = fn.add_figure(label="PORTALS Metrics", tab_color=2)
            opt_fun.plotMetrics(fig=fig)

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
