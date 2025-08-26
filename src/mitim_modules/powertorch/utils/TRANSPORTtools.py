import json
from matplotlib.pylab import f
import torch
import numpy as np
from functools import partial
import copy
import shutil
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def write_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
    '''
    For tracking and reproducibility (e.g. external runs), we want to write a json file
    containing the simulation results. JSON should look like:
    
    {
        'r/a': ...
        'fluxes_mean': 
            {
                'QeMWm2': ...
                'QiMWm2': ...
                'Ge1E20m2': ...
                'GZ1E20m2': ...
                'MtJm2': ...
                'QieMWm3': ...
            },
        'fluxes_stds': 
            {
                'QeMWm2': ...
                'QiMWm2': ...
                'Ge1E20m2': ...
                'GZ1E20m2': ...
                'MtJm2': ...
                'QieMWm3': ...
            },
        'additional': ...
    }
    '''
    
    with open(self.folder / file_name, 'w') as f:

        rho = self.powerstate.plasma["rho"][0, 1:].cpu().numpy()

        fluxes_mean = {}
        fluxes_stds = {}

        for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
            fluxes_mean[var] = self.__dict__[f"{var}_tr_{suffix}"].tolist()
            fluxes_stds[var] = self.__dict__[f"{var}_tr_{suffix}_stds"].tolist()

        json_dict = {
            'r/a': rho.tolist(),
            'fluxes_mean': fluxes_mean,
            'fluxes_stds': fluxes_stds,
            'additional': {
            }
        }

        json.dump(json_dict, f, indent=4)

    print(f"\t* Written JSON with {suffix} information to {self.folder / file_name}")

class power_transport:
    '''
    Default class for power transport models, change "evaluate" method to implement a new model and produce_profiles if the model requires written input.gacode written

    Notes:
        - After evaluation, the self.model_results attribute will contain the results of the model, which can be used for plotting and analysis
        - model results can have .plot() method that can grab kwargs or be similar to TGYRO plot

    '''
    def __init__(self, powerstate, name = "test", folder = "~/scratch/", evaluation_number = 0):

        self.name = name
        self.folder = IOtools.expandPath(folder)
        self.evaluation_number = evaluation_number
        self.powerstate = powerstate

        # Allowed fluxes in powerstate so far
        self.quantities = ['QeMWm2', 'QiMWm2', 'Ce', 'CZ', 'MtJm2']

        # Each flux has a turbulent and neoclassical component
        self.variables = [f'{i}_tr_turb' for i in self.quantities] + [f'{i}_tr_neoc' for i in self.quantities]

        # Each flux component has a standard deviation
        self.variables += [f'{i}_stds' for i in self.variables]

        # There is also turbulent exchange
        self.variables += ['QieMWm3_tr_turb', 'QieMWm3_tr_turb_stds']

        # And total transport flux
        self.variables += [f'{i}_tr' for i in self.quantities]

        # Model results is None by default, but can be assigned in evaluate
        self.model_results = None

        # Assign zeros to transport ones if not evaluated
        for i in self.variables:
            self.powerstate.plasma[i] = self.powerstate.plasma["te"] * 0.0

        # There is also target components
        self.variables += [f'{i}' for i in self.quantities] + [f'{i}_stds' for i in self.quantities]

        # ----------------------------------------------------------------------------------------
        # labels for plotting
        # ----------------------------------------------------------------------------------------

        self.powerstate.labelsFluxes = {
            "te": "$Q_e$ ($MW/m^2$)",
            "ti": "$Q_i$ ($MW/m^2$)",
            "ne": "$Q_{conv}$ ($MW/m^2$)",
            "nZ": "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)",
            "w0": "$M_T$ ($J/m^2$)",
        }

    def produce_profiles(self):
        # Only add self._produce_profiles() if it's needed (e.g. full TGLF), otherwise this is somewhat expensive
        # (e.g. for flux matching of analytical models)
        pass

    def _produce_profiles(self,derive_quantities=True):

        self.applyCorrections = self.powerstate.transport_options["transport_evaluator_options"].get("MODELparameters", {}).get("applyCorrections", {})

        # Write this updated profiles class (with parameterized profiles and target powers)
        self.file_profs = self.folder / "input.gacode"

        powerstate_detached = self.powerstate.copy_state()

        self.powerstate.profiles = powerstate_detached.from_powerstate(
            write_input_gacode=self.file_profs,
            postprocess_input_gacode=self.applyCorrections,
            rederive_profiles = derive_quantities,        # Derive quantities so that it's ready for analysis and plotting later
            insert_highres_powers = derive_quantities,    # Insert powers so that Q, Pfus and all that it's consistent when read later
        )

        self.powerstate.profiles_transport = copy.deepcopy(self.powerstate.profiles)

        self._modify_profiles()

    def _modify_profiles(self):
        '''
        Modify the profiles (e.g. lumping) before running the transport model 
        '''

        # After producing the profiles, copy for future modifications
        self.file_profs_unmod = self.file_profs.parent / f"{self.file_profs.name}_unmodified"
        shutil.copy2(self.file_profs, self.file_profs_unmod)

        profiles_postprocessing_fun = self.powerstate.transport_options["transport_evaluator_options"].get("profiles_postprocessing_fun", None)

        if profiles_postprocessing_fun is not None:
            print(f"\t- Modifying input.gacode to run transport calculations based on {profiles_postprocessing_fun}",typeMsg="i")
            self.powerstate.profiles_transport = profiles_postprocessing_fun(self.file_profs)

        # Position of impurity ion may have changed
        p_old = PROFILEStools.gacode_state(self.file_profs_unmod)
        p_new = PROFILEStools.gacode_state(self.file_profs)

        impurity_of_interest = p_old.Species[self.powerstate.impurityPosition]

        try:
            impurityPosition_new = p_new.Species.index(impurity_of_interest)

        except ValueError:
            print(f"\t- Impurity {impurity_of_interest} not found in new profiles, keeping position {self.powerstate.impurityPosition}",typeMsg="w")
            impurityPosition_new = self.powerstate.impurityPosition

        if impurityPosition_new != self.powerstate.impurityPosition:
            print(f"\t- Impurity position has changed from {self.powerstate.impurityPosition} to {impurityPosition_new}",typeMsg="i")
            self.powerstate.impurityPosition_transport = p_new.Species.index(impurity_of_interest)

    def evaluate(self):

        '''
        ******************************************************************************************************
        Evaluate neoclassical and turbulent transport. 
        These functions use a hook to write the .json files to communicate the results to powerstate.plasma
        ******************************************************************************************************
        '''
        neoclassical = self.evaluate_neoclassical()
        turbulence = self.evaluate_turbulence()
        
        '''
        ******************************************************************************************************
        From the json to powerstate.plasma
        ******************************************************************************************************
        '''
        self._populate_from_json(file_name = 'fluxes_turb.json', suffix= 'turb')
        self._populate_from_json(file_name = 'fluxes_neoc.json', suffix= 'neoc')

        '''
        ******************************************************************************************************
        Post-process the data: add turb and neoc, tensorize and transformations
        ******************************************************************************************************
        '''
        self._postprocess()

    def _postprocess(self):

        OriginalFimp =  self.powerstate.transport_options["transport_evaluator_options"].get("OriginalFimp", 1.0)

        # ------------------------------------------------------------------------------------------------------------------------
        # Curate information for the powerstate (e.g. add models, add batch dimension, rho=0.0, and tensorize)
        # ------------------------------------------------------------------------------------------------------------------------
        
        variables = ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']

        for variable in variables:
            for suffix in ['_tr_turb', '_tr_turb_stds', '_tr_neoc', '_tr_neoc_stds']:

                # Make them tensors and add a batch dimension
                self.powerstate.plasma[f"{variable}{suffix}"] = torch.Tensor(self.powerstate.plasma[f"{variable}{suffix}"]).to(self.powerstate.dfT).unsqueeze(0)
 
                # Pad with zeros at rho=0.0
                self.powerstate.plasma[f"{variable}{suffix}"] = torch.cat((
                    torch.zeros((1, 1)),
                    self.powerstate.plasma[f"{variable}{suffix}"],
                ), dim=1)

        # -----------------------------------------------------------
        # Sum the turbulent and neoclassical contributions
        # -----------------------------------------------------------
        
        variables = ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2']
        
        for variable in variables:
            self.powerstate.plasma[f"{variable}_tr"] = self.powerstate.plasma[f"{variable}_tr_turb"] + self.powerstate.plasma[f"{variable}_tr_neoc"]

        # -----------------------------------------------------------
        # Convective fluxes (& Re-scale the GZ flux by the original impurity concentration)
        # -----------------------------------------------------------
        
        mapper_convective = {
            'Ce': 'Ge1E20m2',
            'CZ': 'GZ1E20m2',
        }
        
        for key in mapper_convective.keys():
            for tt in ['','_turb', '_turb_stds', '_neoc', '_neoc_stds']:
                
                mult = 1.0 if key == 'Ce' else 1/OriginalFimp
                
                self.powerstate.plasma[f"{key}_tr{tt}"] = PLASMAtools.convective_flux(
                    self.powerstate.plasma["te"],
                    self.powerstate.plasma[f"{mapper_convective[key]}_tr{tt}"]
                ) * mult

    def _populate_from_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
        
        '''
        Populate the powerstate.plasma with the results from the json file
        '''

        with open(self.folder / file_name, 'r') as f:
            json_dict = json.load(f)

        for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
            self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
            self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        print(f"\t* Populated powerstate.plasma with JSON data from {self.folder / file_name}")
        
    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self):
        '''
        This needs to populate the following np.arrays in self., with dimensions of rho:
            - QeMWm2_tr_turb
            - QiMWm2_tr_turb
            - Ge1E20m2_tr_turb
            - GZ1E20m2_tr_turb
            - MtJm2_tr_turb
            - QieMWm3_tr_turb (turbulence exchange)
        and their respective standard deviations, e.g. QeMWm2_tr_turb_stds
        '''

        print(">> No turbulent fluxes to evaluate", typeMsg="w")

        dim = self.powerstate.plasma['rho'].shape[-1]-1
        
        for var in [
            'QeMWm2',
            'QiMWm2',
            'Ge1E20m2',
            'GZ1E20m2',
            'MtJm2',
            'QieMWm3'
        ]:

            self.__dict__[f"{var}_tr_turb"] = np.zeros(dim)
            self.__dict__[f"{var}_tr_turb_stds"] = np.zeros(dim)
    
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))    
    def evaluate_neoclassical(self):
        '''
        This needs to populate the following np.arrays in self.:
            - QeMWm2_tr_neoc
            - QiMWm2_tr_neoc
            - Ge1E20m2_tr_neoc
            - GZ1E20m2_tr_neoc
            - MtJm2_tr_neoc
        and their respective standard deviations, e.g. QeMWm2_tr_neoc_stds
        '''

        print(">> No neoclassical fluxes to evaluate", typeMsg="w")
        
        dim = self.powerstate.plasma['rho'].shape[-1]-1
        
        for var in [
            'QeMWm2',
            'QiMWm2',
            'Ge1E20m2',
            'GZ1E20m2',
            'MtJm2',
        ]:

            self.__dict__[f"{var}_tr_neoc"] = np.zeros(dim)
            self.__dict__[f"{var}_tr_neoc_stds"] = np.zeros(dim)