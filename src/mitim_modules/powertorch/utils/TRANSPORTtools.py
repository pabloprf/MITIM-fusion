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
        'fluxes_mean': 
            {
                'QeGB': ...
                'QiGB': ...
                'GeGB': ...
                'GZGB': ...
                'MtGB': ...
                'QieGB': ...
            },
        'fluxes_stds': 
            {
                'QeGB': ...
                'QiGB': ...
                'GeGB': ...
                'GZGB': ...
                'MtGB': ...
                'QieGB': ...
            },
        'additional_info': {
                'rho': rho.tolist(),
            }
    }
    '''
    
    write_json_from_variables = self._write_json_from_variables_turb if suffix == 'turb' else self._write_json_from_variables_neoc
    
    if self.folder.exists() and write_json_from_variables:
        
        with open(self.folder / file_name, 'w') as f:

            fluxes_mean = {}
            fluxes_stds = {}

            for var in ['QeGB', 'QiGB', 'GeGB', 'GZGB', 'MtGB']:
                fluxes_mean[var] = self.__dict__[f"{var}_{suffix}"].tolist()
                fluxes_stds[var] = self.__dict__[f"{var}_{suffix}_stds"].tolist()

            try:
                var = 'QieGB'
                fluxes_mean[var] = self.__dict__[f"{var}_{suffix}"].tolist()
                fluxes_stds[var] = self.__dict__[f"{var}_{suffix}_stds"].tolist()
            except KeyError:
                # NEO file may not have it
                pass

            json_dict = {
                'fluxes_mean': fluxes_mean,
                'fluxes_stds': fluxes_stds,
                'additional_info': {
                    'rho': self.powerstate.plasma["rho"][0, 1:].cpu().numpy().tolist(),
                    'roa': self.powerstate.plasma["roa"][0, 1:].cpu().numpy().tolist(),
                    'Qgb': self.powerstate.plasma["Qgb"][0, 1:].cpu().numpy().tolist(),
                    'aLte': self.powerstate.plasma["aLte"][0, 1:].cpu().numpy().tolist(),
                    'aLti': self.powerstate.plasma["aLti"][0, 1:].cpu().numpy().tolist(),
                    'aLne': self.powerstate.plasma["aLne"][0, 1:].cpu().numpy().tolist(),
                }
            }

            json.dump(json_dict, f, indent=4)

        print(f"\t* Written JSON with {suffix} information to {self.folder / file_name}")
        
    else:
        
        print(f"\t* Folder {self.folder} does not exist, cannot write {file_name}", typeMsg='w')

class power_transport:

    def __init__(self, powerstate, name = "test", folder = "~/scratch/", evaluation_number = 0):

        self.name = name
        self.folder = IOtools.expandPath(folder)
        self.evaluation_number = evaluation_number
        self.powerstate = powerstate

        self.transport_evaluator_options  = self.powerstate.transport_options["options"]
        self.cold_start                   = self.powerstate.transport_options["cold_start"]

        # Model results is None by default, but can be assigned in evaluate
        self.model_results = None
        
        # By default, write the json files after evaluating the variables (will be changed in gyrokinetic "prep" run mode)
        self._write_json_from_variables_turb = True
        self._write_json_from_variables_neoc = True

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

    def evaluate(self):

        # Copy the input.gacode files to the output folder
        self._profiles_to_store()

        '''
        ******************************************************************************************************
        Evaluate neoclassical and turbulent transport (*in GB units*). 
        These functions use a hook to write the .json files to communicate the results to powerstate.plasma
        ******************************************************************************************************
        '''
        
        # Initialize them as zeros
        for var in ['QeGB','QiGB','GeGB','GZGB','MtGB','QieGB']:
            for suffix in ['turb', 'neoc']:
                for suffix0 in ['', '_stds']:
                    self.__dict__[f"{var}_{suffix}{suffix0}"] = torch.zeros(self.powerstate.plasma['rho'].shape[-1]-1)
        
        neoclassical = self.evaluate_neoclassical()
        turbulence = self.evaluate_turbulence()
        
        '''
        ******************************************************************************************************
        From the json to powerstate.plasma and GB to real units transformation
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
        '''
        Curate information for the powerstate (e.g. add models, add batch dimension, rho=0.0, and tensorize)
        Before calling this function, the powerstate.plasma should have the following variables:
            'QeMWm2_tr_X', 'QiMWm2_tr_X', 'Ge1E20m2_tr_X', 'GZ1E20m2_tr_X', 'MtJm2_tr_X', 'QieMWm3_tr_X'
        where X = 'turb' or 'neoc'
        and also the corresponding _stds versions
        '''

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

        # ---------------------------------------------------------------------------------
        # Convective fluxes (& Re-scale the GZ flux by the original impurity concentration)
        # ---------------------------------------------------------------------------------
        
        mapper_convective = {
            'Ce': 'Ge1E20m2',
            'CZ': 'GZ1E20m2',
        }
        
        for key in mapper_convective.keys():
            for tt in ['','_turb', '_turb_stds', '_neoc', '_neoc_stds']:
                
                mult = 1/self.powerstate.fImp_orig if key == 'CZ' else 1.0
                
                self.powerstate.plasma[f"{key}_tr{tt}"] = PLASMAtools.convective_flux(
                    self.powerstate.plasma["te"],
                    self.powerstate.plasma[f"{mapper_convective[key]}_tr{tt}"]
                ) * mult
           
    def produce_profiles(self):
        # Only add self._produce_profiles() if it's needed (e.g. full TGLF), otherwise this is somewhat expensive
        # (e.g. for flux matching of analytical models)
        pass

    def _produce_profiles(self,derive_quantities=True):

        self.applyCorrections = self.powerstate.transport_options["applyCorrections"]

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

        profiles_postprocessing_fun = self.powerstate.transport_options["profiles_postprocessing_fun"]

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

    def _profiles_to_store(self):

        if "folder" in self.powerstate.transport_options:
            whereFolder = IOtools.expandPath(self.powerstate.transport_options["folder"] / "Outputs" / "portals_profiles")
            if not whereFolder.exists():
                IOtools.askNewFolder(whereFolder)

            fil = whereFolder / f"input.gacode.{self.evaluation_number}"
            shutil.copy2(self.file_profs, fil)
            shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
            print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
        else:
            print("\t- Could not move files", typeMsg="w")
                
    def _populate_from_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
        '''
        Populate the powerstate.plasma with the results from the json file
        '''
        
        mapper = {
            'QeGB': ['Qgb', 'QeMWm2'],
            'QiGB': ['Qgb', 'QiMWm2'],
            'GeGB': ['Ggb', 'Ge1E20m2'],
            'GZGB': ['Ggb', 'GZ1E20m2'],
            'MtGB': ['Pgb', 'MtJm2'],
            'QieGB': ['Sgb', 'QieMWm3']
        }
        
        '''
        **********************************************************************************************
        If no population file exists, I only convert from GB to real units and return
        **********************************************************************************************
        '''
        if not (self.folder / file_name).exists():
            print(f"\t* File {self.folder / file_name} does not exist, cannot populate powerstate.plasma", typeMsg='w')
            print(f"\t- Tranforming from GB to real units:")
            
            for var in mapper:
                self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"] = self.__dict__[f"{var}_{suffix}"] * self.powerstate.plasma[f"{mapper[var][0]}"][0,1:]
                self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}_stds"] = self.__dict__[f"{var}_{suffix}_stds"] * self.powerstate.plasma[f"{mapper[var][0]}"][0,1:]

            return
        
        '''
        **********************************************************************************************
        Populate the powerstate.plasma from the json file
        **********************************************************************************************
        '''
        print(f"\t* Populating powerstate.plasma with JSON data from {self.folder / file_name}")

        with open(self.folder / file_name, 'r') as f:
            json_dict = json.load(f)
        
        # See if the file has GB or real units
        units_GB, units_real = False, False
        if 'QeGB' in json_dict['fluxes_mean']:
            units_GB = True
        if 'QeMWm2' in json_dict['fluxes_mean']:
            units_real = True

        units = 'both' if (units_GB and units_real) else 'GB' if units_GB else 'real' if units_real else 'none'

        if units == 'real':
            
            print("\t\t- File has fluxes in real units... populating powerstate directly")

            for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
                self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
                self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        elif units == 'GB' or units == 'both':

            dum = {}
            for var in mapper:
                gb = self.powerstate.plasma[f"{mapper[var][0]}"][0,1:].cpu().numpy()
                dum[f"{mapper[var][1]}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var]) * gb
                dum[f"{mapper[var][1]}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var]) * gb

            if units == 'GB':
                
                print("\t\t- File has fluxes in GB units... using GB units from powerstate to convert to real units")

                for var in mapper:
                    self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"] = dum[f"{mapper[var][1]}_tr_{suffix}"]
                    self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}_stds"] = dum[f"{mapper[var][1]}_tr_{suffix}_stds"]

            elif units == 'both':
                
                print("\t\t- File has fluxes in both GB and real units... using real units and checking consistency")

                for var in mapper:
                    if not np.allclose(self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"], dum[f"{mapper[var][1]}_tr_{suffix}"]):
                        print(f"\t\t\t- Inconsistent values found for {mapper[var][1]}_tr_{suffix}")

                for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
                    self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
                    self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        else:
            raise ValueError("[MITIM] Unknown units in JSON file")

    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self, **kwargs):
        '''
        This needs to populate the following np.arrays in self., with dimensions of rho:
            - QeGB_turb
            - QiGB_turb
            - GeGB_turb
            - GZGB_turb
            - MtGB_turb
            - QieGB_turb (turbulence exchange)
        and their respective standard deviations, e.g. QeGB_turb_stds
        '''

        print(">> No turbulent fluxes to evaluate", typeMsg="w")
    
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))    
    def evaluate_neoclassical(self):
        '''
        This needs to populate the following np.arrays in self.:
            - QeGB_neoc
            - QiGB_neoc
            - GeGB_neoc
            - GZGB_neoc
            - MtGB_neoc
            - QieGB_neoc (zero)
        and their respective standard deviations, e.g. QeGB_neoc_stds
        '''

        print(">> No neoclassical fluxes to evaluate", typeMsg="w")
        
        
# *******************************************************************************************
# Combinations
# *******************************************************************************************

from mitim_modules.powertorch.physics_models.transport_tglf import tglf_model
from mitim_modules.powertorch.physics_models.transport_neo import neo_model
from mitim_modules.powertorch.physics_models.transport_cgyro import cgyro_model
from mitim_modules.powertorch.physics_models.transport_gx import gx_model

class portals_transport_model(power_transport, tglf_model, neo_model, cgyro_model, gx_model):

    def __init__(self, powerstate, fidelity_level=0, **kwargs):
        super().__init__(powerstate, **kwargs)

        self.fidelity_level = fidelity_level

        # Defaults
        self.turbulence_model = 'tglf'
        self.neoclassical_model = 'neo'

    def produce_profiles(self):
        self._produce_profiles()
        
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self, **kwargs):
        
        turbulence_model = self.turbulence_model[int(np.round(self.fidelity_level))] if isinstance(self.turbulence_model, list) else self.turbulence_model
        
        if turbulence_model.split('_')[0] == "tglf":
            return tglf_model.evaluate_turbulence(self, label_options = turbulence_model)
        elif turbulence_model.split('_')[0] == 'cgyro':
            return cgyro_model.evaluate_turbulence(self, label_options = turbulence_model)
        elif turbulence_model.split('_')[0] == 'gx':
            return gx_model.evaluate_turbulence(self, label_options = turbulence_model)
        else:
            raise Exception(f"Unknown turbulence model {turbulence_model}")

    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))
    def evaluate_neoclassical(self):
        if self.neoclassical_model == 'neo':
            return neo_model.evaluate_neoclassical(self)
        else:
            raise Exception(f"Unknown neoclassical model {self.neoclassical_model}")
