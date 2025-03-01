import shutil
import copy
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals import PORTALStools
from mitim_modules.portals.utils import PORTALSanalysis, PORTALSoptimization
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

class portals_beat(beat):

    def __init__(self, maestro_instance):
        super().__init__(maestro_instance, beat_name = 'portals')

    def prepare(self,
            use_previous_residual = True,
            use_previous_surrogate_data = False,
            try_flux_match_only_for_first_point = True,
            change_last_radial_call = False,
            additional_params_in_surrogate = [],
            exploration_ranges = {
                'ymax_rel': 1.0,
                'ymin_rel': 1.0,
                'hardGradientLimits': [0,2]
            },
            PORTALSparameters = {},
            MODELparameters = {},
            optimization_options = {},
            INITparameters = {}
            ):

        self.fileGACODE = self.initialize.folder / 'input.gacode'
        self.profiles_current.writeCurrentStatus(file = self.fileGACODE)

        self.PORTALSparameters = PORTALSparameters
        self.MODELparameters = MODELparameters
        self.optimization_options = optimization_options
        self.INITparameters = INITparameters

        self.additional_params_in_surrogate = additional_params_in_surrogate

        self.exploration_ranges = exploration_ranges
        self.use_previous_surrogate_data = use_previous_surrogate_data
        self.change_last_radial_call = change_last_radial_call

        self.try_flux_match_only_for_first_point = try_flux_match_only_for_first_point

        self._inform(use_previous_residual = use_previous_residual, 
                     use_previous_surrogate_data = self.use_previous_surrogate_data,
                     change_last_radial_call = self.change_last_radial_call
                     )

    def run(self, **kwargs):

        cold_start = kwargs.get('cold_start', False)

        portals_fun  = PORTALSmain.portals(self.folder, additional_params_in_surrogate = self.additional_params_in_surrogate)

        modify_dictionary(portals_fun.PORTALSparameters, self.PORTALSparameters)
        modify_dictionary(portals_fun.MODELparameters, self.MODELparameters)
        modify_dictionary(portals_fun.optimization_options, self.optimization_options)
        modify_dictionary(portals_fun.INITparameters, self.INITparameters)

        portals_fun.prep(self.fileGACODE,askQuestions=False,**self.exploration_ranges)

        self.mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start = cold_start, askQuestions = False)

        if self.use_previous_surrogate_data and self.try_flux_match_only_for_first_point and self.folder_starting_point is not None:

            # PORTALS just with one point
            portals_fun.optimization_options['initialization_options']['initial_training'] = 1

            # If the point is not evaluated (for example, this was not a restart of this portals beat), then flux-match it
            if len(self.mitim_bo.optimization_data.data) == 0:
                self._flux_match_for_first_point()

            portals_fun.prep(self.fileGACODE,askQuestions=False,**self.exploration_ranges)

            self.mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start = cold_start, askQuestions = False)

        self.mitim_bo.run()

    def _flux_match_for_first_point(self):

        print('\t- Running flux match for first point')

        # Flux-match first
        folder_fm = self.folder / 'flux_match'
        folder_fm.mkdir(parents=True, exist_ok=True)

        portals = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_starting_point)
        p = portals.powerstates[portals.ibest].profiles
        _ = PORTALSoptimization.flux_match_surrogate(portals.step,p,file_write_csv=folder_fm / 'optimization_data.csv')

        # Move files
        (self.folder / 'Outputs').mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder_fm / 'optimization_data.csv', self.folder / 'Outputs')

    def finalize(self, **kwargs):

        # Remove output folders
        for item in self.folder_output.glob('*'):
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                IOtools.shutil_rmtree(item)

        # Copy to outputs
        shutil.copytree(self.folder / 'Outputs', self.folder_output / 'Outputs')

        # --------------------------------------------------------------------------------------------
        # Prepare final beat's input.gacode
        # --------------------------------------------------------------------------------------------

        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)

        # Standard PORTALS output
        try:
            self.profiles_output = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
        # Converged in training case
        except AttributeError:
            print('\t\t- PORTALS probably converged in training, so analyzing a bit differently')
            self.profiles_output = portals_output.profiles[portals_output.opt_fun_full.res.best_absolute_index]

        self.profiles_output.writeCurrentStatus(file=self.folder_output / 'input.gacode')

    def merge_parameters(self):
        '''
        The goal of the PORTALS beat is to produce:
            - Kinetic profiles
            - Dynamics targets that gave rise to the kinetic profiles
        So, this merge:
            - Frozen profiles are converted to PORTALS output resolution (opposite to usual, but keeps gradients)
            - Inserts kinetic profiles
            - Inserts dynamic targets (only those that were evolved)
        '''

        # Write the pre-merge input.gacode before modifying it
        self.profiles_output.writeCurrentStatus(file=self.folder_output / 'input.gacode_pre_merge')

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
                    self.profiles_output.profiles['ti(keV)'][:,j] = profiles_portals_out.profiles['ti(keV)'][:,i]

        # Enforce quasineutrality because now I have all the ions
        self.profiles_output.enforceQuasineutrality()

        # Insert powers
        if self.mitim_bo.optimization_object.MODELparameters['Physics_options']["TypeTarget"] > 1:
            # Insert exchange
            self.profiles_output.profiles['qei(MW/m^3)'] = profiles_portals_out.profiles['qei(MW/m^3)']
            if self.mitim_bo.optimization_object.MODELparameters['Physics_options']["TypeTarget"] > 2:
                # Insert radiation and fusion
                for key in ['qbrem(MW/m^3)', 'qsync(MW/m^3)', 'qline(MW/m^3)', 'qfuse(MW/m^3)', 'qfusi(MW/m^3)']:
                    self.profiles_output.profiles[key] = profiles_portals_out.profiles[key]
        # --------------------------------------------------------------------------------------------

        # Write to final input.gacode
        self.profiles_output.deriveQuantities()
        self.profiles_output.writeCurrentStatus(file=self.folder_output / 'input.gacode')

    def grab_output(self, full = False):

        isitfinished = self.maestro_instance.check(beat_check=self)

        folder = self.folder_output if isitfinished else self.folder

        opt_fun = STRATEGYtools.opt_evaluator(folder) if full else PORTALSanalysis.PORTALSanalyzer.from_folder(folder)

        profiles = PROFILEStools.PROFILES_GACODE(self.folder_output / 'input.gacode') if isitfinished else None
        
        return opt_fun, profiles

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
                print('\t\t- PORTALS has not run enough to plot anything')

        msg = '\t\t- Plotting of PORTALS beat done'

        return msg

    def finalize_maestro(self):

        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(self.folder_output)

        # Standard PORTALS output
        try:
            self.maestro_instance.final_p = portals_output.mitim_runs[portals_output.ibest]['powerstate'].profiles
        # Converged in training case
        except AttributeError as e:
            print('\t\t- PORTALS probably converged in training, so analyzing a bit differently, error:', e)
            self.maestro_instance.final_p = portals_output.profiles[portals_output.opt_fun_full.res.best_absolute_index]
        
        final_file = self.maestro_instance.folder_output / 'input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

    # --------------------------------------------------------------------------------------------
    # Additional PORTALS utilities
    # --------------------------------------------------------------------------------------------
    def _inform(self, use_previous_residual = True, use_previous_surrogate_data = True, change_last_radial_call = False, minimum_relative_change_in_x = 0.005):
        '''
        Prepare next PORTALS runs accounting for what previous PORTALS runs have done
        '''
        if use_previous_residual and ('portals_neg_residual_obj' in self.maestro_instance.parameters_trans_beat):
            
            if 'convergence_options' not in self.optimization_options:
                self.optimization_options['convergence_options'] = {}
            if 'stopping_criteria_parameters' not in self.optimization_options['convergence_options']:
                self.optimization_options['convergence_options']['stopping_criteria_parameters'] = {}

            self.optimization_options['convergence_options']['stopping_criteria_parameters']['maximum_value'] = self.maestro_instance.parameters_trans_beat['portals_neg_residual_obj']
            self.optimization_options['convergence_options']['stopping_criteria_parameters']['maximum_value_is_rel'] = False

            print(f"\t\t- Using previous residual goal as maximum value for optimization: {self.optimization_options['convergence_options']['stopping_criteria_parameters']['maximum_value']}")

        reusing_surrogate_data = False
        self.folder_starting_point = None
        if use_previous_surrogate_data and ('portals_surrogate_data_file' in self.maestro_instance.parameters_trans_beat):
            if 'surrogate_options' not in self.optimization_options:
                self.optimization_options['surrogate_options'] = {}
            self.optimization_options['surrogate_options']["extrapointsFile"] = self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file']

            self.folder_starting_point = self.maestro_instance.parameters_trans_beat['portals_last_run_folder']

            print(f"\t\t- Using previous surrogate data for optimization: {IOtools.clipstr(self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'])}")

            reusing_surrogate_data = True
            

        last_radial_location_moved = False
        if change_last_radial_call and ('rhotop' in self.maestro_instance.parameters_trans_beat):

            if 'RoaLocations' in self.MODELparameters:

                print('\t\t- Using EPED pedestal top rho to select last radial location of PORTALS (in r/a)')

                # interpolate the correct roa location from the EPED pedestal top, if it is defined
                roatop = interpolation_function(self.maestro_instance.parameters_trans_beat['rhotop'], 
                                self.profiles_current.profiles['rho(-)'], 
                                self.profiles_current.derived['roa']).item()
                
                #roatop = roatop.round(3)
                
                # set the last value of the radial locations to the interpolated value
                roatop_old = copy.deepcopy(self.MODELparameters["RoaLocations"][-1])
                self.MODELparameters["RoaLocations"][-1] = roatop
                print(f'\t\t\t* Last radial location moved from r/a = {roatop_old} to {self.MODELparameters["RoaLocations"][-1]}')
                print(f'\t\t\t* RoaLocations: {self.MODELparameters["RoaLocations"]}')

                strKeys = 'RoaLocations'

            else:

                print('\t\t- Using EPED pedestal top rho to select last radial location of PORTALS (in rho)')

                # set the last value of the radial locations to the interpolated value
                rhotop_old = copy.deepcopy(self.MODELparameters["RhoLocations"][-1])
                self.MODELparameters["RhoLocations"][-1] = self.maestro_instance.parameters_trans_beat['rhotop']
                print(f'\t\t\t* Last radial location moved from rho = {rhotop_old} to {self.MODELparameters["RhoLocations"][-1]}')

                strKeys = 'RhoLocations'

            last_radial_location_moved = True

            # Check if I changed it previously and it hasn't moved
            if strKeys in self.maestro_instance.parameters_trans_beat:
                print(f'\t\t\t* {strKeys} in previous PORTALS beat: {self.maestro_instance.parameters_trans_beat[strKeys]}')
                print(f'\t\t\t* {strKeys} in current PORTALS beat: {self.MODELparameters[strKeys]}')

                if abs(self.MODELparameters[strKeys][-1]-self.maestro_instance.parameters_trans_beat[strKeys][-1]) / self.maestro_instance.parameters_trans_beat[strKeys][-1] < minimum_relative_change_in_x:
                    print('\t\t\t* Last radial location was not moved')
                    last_radial_location_moved = False
                    self.MODELparameters[strKeys][-1] = self.maestro_instance.parameters_trans_beat[strKeys][-1]

        # In the situation where the last radial location moves, I cannot reuse that surrogate data
        if last_radial_location_moved and reusing_surrogate_data:
            print('\t\t- Last radial location was moved, so surrogate data will not be reused for that specific location')
            self.optimization_options['surrogate_options']["extrapointsModelsAvoidContent"] = ['Tar',f'_{len(self.MODELparameters[strKeys])}']
            self.try_flux_match_only_for_first_point = False

    def _inform_save(self):

        print('\t- Saving PORTALS beat parameters for future beats')

        # Save the residual goal to use in the next PORTALS beat
        portals_output, _ = self.grab_output()

        # Standard PORTALS output
        try:
            stepSettings = portals_output.step.stepSettings
            MODELparameters = portals_output.MODELparameters
        # Converged in training case
        except AttributeError:
            stepSettings = portals_output.opt_fun_full.mitim_model.stepSettings
            MODELparameters =portals_output.opt_fun_full.mitim_model.optimization_object.MODELparameters

        max_value_neg_residual = stepSettings['optimization_options']['convergence_options']['stopping_criteria_parameters']['maximum_value']
        self.maestro_instance.parameters_trans_beat['portals_neg_residual_obj'] = max_value_neg_residual
        print(f'\t\t* Maximum value of negative residual saved for future beats: {max_value_neg_residual}')

        fileTraining = stepSettings['folderOutputs'] / 'surrogate_data.csv'
        
        self.maestro_instance.parameters_trans_beat['portals_last_run_folder'] = self.folder
        self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] = fileTraining
        print(f'\t\t* Surrogate data saved for future beats: {IOtools.clipstr(fileTraining)}')

        if 'RoaLocations' in MODELparameters:
            self.maestro_instance.parameters_trans_beat['RoaLocations'] = MODELparameters['RoaLocations']
            print(f'\t\t* RoaLocations saved for future beats: {MODELparameters["RoaLocations"]}')
        elif 'RhoLocations' in MODELparameters:
            self.maestro_instance.parameters_trans_beat['RhoLocations'] = MODELparameters['RhoLocations']
            print(f'\t\t* RhoLocations saved for future beats: {MODELparameters["RhoLocations"]}')


def modify_dictionary(original, new):
    for key in new:
        # If something on the new dictionary is not in the original, add it
        if key not in original:
            original[key] = new[key]
        # If it is a dictionary, go deeper
        elif isinstance(new[key], dict):
                modify_dictionary(original[key], new[key])
        # If it is not a dictionary, just replace the value
        else:
            original[key] = new[key]

# -----------------------------------------------------------------------------------------------------------------------
# Defaults to help MAESTRO
# -----------------------------------------------------------------------------------------------------------------------

def portals_beat_soft_criteria(portals_namelist):

    portals_namelist_soft = copy.deepcopy(portals_namelist)

    if 'optimization_options' not in portals_namelist_soft:
        portals_namelist_soft['optimization_options'] = {}

    # Relaxation of stopping criteria
    portals_namelist_soft['optimization_options']['convergence_options']["maximum_iterations"] = 15
    portals_namelist_soft['optimization_options']['convergence_options']["stopping_criteria_parameters"]["maximum_value"] = 10e-3
    portals_namelist_soft['optimization_options']['convergence_options']["stopping_criteria_parameters"]["minimum_dvs_variation"] = [10, 3, 1.0]
    portals_namelist_soft['optimization_options']['convergence_options']["stopping_criteria_parameters"]["ricci_value"] = 0.15

    if 'MODELparameters' not in portals_namelist_soft:
        portals_namelist_soft['MODELparameters'] = {}

    portals_namelist_soft["MODELparameters"]["Physics_options"] = {"TypeTarget": 2}

    return portals_namelist_soft
