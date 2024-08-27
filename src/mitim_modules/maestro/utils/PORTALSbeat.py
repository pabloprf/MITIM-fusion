import os
import copy
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

class portals_beat(beat):

    def __init__(self, maestro_instance):
        super().__init__(maestro_instance, beat_name = 'portals')

    def prepare(self, use_previous_residual = True, PORTALSparameters = {}, MODELparameters = {}, optimization_options = {}, INITparameters = {}):

        self.fileGACODE = f"{self.initialize.folder}/input.gacode"
        self.profiles_current.writeCurrentStatus(file = self.fileGACODE)

        self.PORTALSparameters = PORTALSparameters
        self.MODELparameters = MODELparameters
        self.optimization_options = optimization_options
        self.INITparameters = INITparameters

        self._inform(use_previous_residual = use_previous_residual)

    def run(self, **kwargs):

        restart = kwargs.get('restart', False)

        portals_fun  = PORTALSmain.portals(self.folder)

        for key in self.PORTALSparameters:
            if not isinstance(portals_fun.PORTALSparameters[key], dict):
                portals_fun.PORTALSparameters[key] = self.PORTALSparameters[key]
            else:
                for subkey in self.PORTALSparameters[key]:
                    portals_fun.PORTALSparameters[key][subkey] = self.PORTALSparameters[key][subkey]
        for key in self.MODELparameters:
            if not isinstance(portals_fun.MODELparameters[key], dict):
                portals_fun.MODELparameters[key] = self.MODELparameters[key]
            else:
                for subkey in self.MODELparameters[key]:
                    portals_fun.MODELparameters[key][subkey] = self.MODELparameters[key][subkey]
        for key in self.optimization_options:
            if not isinstance(portals_fun.optimization_options[key], dict):
                portals_fun.optimization_options[key] = self.optimization_options[key]
            else:
                for subkey in self.optimization_options[key]:
                    portals_fun.optimization_options[key][subkey] = self.optimization_options[key][subkey]
        for key in self.INITparameters:
            if not isinstance(portals_fun.INITparameters[key], dict):
                portals_fun.INITparameters[key] = self.INITparameters[key]
            else:
                for subkey in self.INITparameters[key]:
                    portals_fun.INITparameters[key][subkey] = self.INITparameters[key][subkey]

        portals_fun.prep(self.fileGACODE,self.folder,hardGradientLimits = [0,2])

        self.prf_bo = STRATEGYtools.PRF_BO(portals_fun, restartYN = restart, askQuestions = False)

        self.prf_bo.run()

    def finalize(self):

        # Remove output folders
        os.system(f'rm -r {self.folder_output}/*')

        # Copy to outputs
        os.system(f'cp -r {self.folder}/Outputs {self.folder_output}/Outputs')

        # Prepare final beat's input.gacode
        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)
        self.profiles_output = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
        self.profiles_output.writeCurrentStatus(file=f'{self.folder_output}/input.gacode' )

    def merge_parameters(self):
        '''
        The goal of the PORTALS beat is to produce:
            - Kinetic profiles
            - Dynamics targets that gave rise to the kinetic profiles
        So, this merge:
            - Brings back the resolution of the frozen profiles
            - Inserts kinetic profiles
            - Inserts dynamic targets
        '''

        profiles_output_pre_merge = copy.deepcopy(self.profiles_output)
        profiles_output_pre_merge.writeCurrentStatus(file=f"{self.folder_output}/input.gacode_pre_merge")

        p = self.maestro_instance.profiles_with_engineering_parameters

        # First, bring back to the resolution of the frozen
        self.profiles_output.changeResolution(rho_new = p.profiles['rho(-)'])

        # Insert everything but kinetic profiles and dynamic targets from frozen
        for key in ['ne(10^19/m^3)', 'te(keV)', 'ti(keV)', 'qei(MW/m^3)', 'qbrem(MW/m^3)', 'qsync(MW/m^3)', 'qline(MW/m^3)', 'qfuse(MW/m^3)', 'qfusi(MW/m^3)']:
            self.profiles_output.profiles[key] = p.profiles[key]

        # Write to final input.gacode
        self.profiles_output.deriveQuantities()
        self.profiles_output.writeCurrentStatus(file=f"{self.folder_output}/input.gacode")

    def grab_output(self, full = False):

        isitfinished = self.maestro_instance.check(beat_check=self)

        folder = self.folder_output if isitfinished else self.folder

        opt_fun = STRATEGYtools.opt_evaluator(folder) if full else PORTALSanalysis.PORTALSanalyzer.from_folder(folder)

        profiles = PROFILEStools.PROFILES_GACODE(f'{self.folder_output}/input.gacode') if isitfinished else None
        
        return opt_fun, profiles

    def plot(self,  fn = None, counter = 0, full_plot = True):

        opt_fun, _ = self.grab_output(full = full_plot)

        if full_plot:
            opt_fun.fn = fn
            opt_fun.plot_optimization_results(analysis_level=4)
        else:
            fig = fn.add_figure(label="PORTALS Metrics", tab_color=2)
            opt_fun.plotMetrics(fig=fig)

        msg = '\t\t- Plotting of PORTALS beat done'

        return msg

    def finalize_maestro(self):

        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder)
        self.maestro_instance.final_p = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
        
        final_file = f'{self.maestro_instance.folder_output}/input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

    # --------------------------------------------------------------------------------------------
    # Additional PORTALS utilities
    # --------------------------------------------------------------------------------------------
    def _inform(self, use_previous_residual = True):
        '''
        Prepare next PORTALS runs accounting for what previous PORTALS runs have done
        '''
        if use_previous_residual and ('portals_neg_residual_obj' in self.maestro_instance.parameters_trans_beat):
            self.optimization_options['maximum_value'] = self.maestro_instance.parameters_trans_beat['portals_neg_residual_obj']
            self.optimization_options['maximum_value_is_rel'] = False

            print(f"\t\t- Using previous residual goal as maximum value for optimization: {self.optimization_options['maximum_value']}")

    def _inform_save(self):

        # Save the residual goal to use in the next PORTALS beat
        portals_output, _ = self.grab_output()
        max_value_neg_residual = portals_output.step.stepSettings['optimization_options']['maximum_value']
        self.maestro_instance.parameters_trans_beat['portals_neg_residual_obj'] = max_value_neg_residual
        print(f'\t\t- Maximum value of negative residual saved for future beats: {max_value_neg_residual}')
