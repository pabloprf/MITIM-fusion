import json
from mitim_tools.misc_tools import IOtools
import numpy as np
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class gyrokinetic_model:

    def _evaluate_gyrokinetic_model(self, code = 'cgyro', gk_object = None):
        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options
        # ------------------------------------------------------------------------------------------------------------------------

        simulation_options = self.transport_evaluator_options[code]
        cold_start = self.cold_start

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]        
        run_type = simulation_options["run"]["run_type"]    
        keep_gk_files = simulation_options.get("keep_files", 'all')

        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare object
        # ------------------------------------------------------------------------------------------------------------------------
               
        subfolder_name = f"base_{code}"
            
        # <><><><><><>
        # If the way to store data is in pickle, try first to read the stored pickled in the folder (e.g. for SR stage)
        # <><><><><><>
        gk_object_unpickled = False
        if keep_gk_files in ['pickle']:
            try:
                pickle_file = self.folder / f"{subfolder_name}" / "gk_object.pkl"
                gk_object = SIMtools.restore_class_pickle(pickle_file)
                gk_object_unpickled = True
                print('\t- Pickle file with GK object information has been restored successfully', typeMsg='i')
            except Exception as e:
                gk_object_unpickled = False
                print('\t- Pickle file could not be read, with error:', typeMsg='w')
                print(e)
                
        # <><><><><><>
        # Standard run
        # <><><><><><>
        if not gk_object_unpickled:
            gk_object = gk_object(rhos=rho_locations)

            _ = gk_object.prep(
                self.powerstate.profiles_transport,
                self.folder,
                )

            _ = gk_object.run(
                subfolder_name,
                cold_start=cold_start,
                forceIfcold_start=True,
                only_minimal_files=keep_gk_files in ['none', 'pickle'],
                **simulation_options["run"]
                )
        
        if run_type in ['normal', 'submit', 'send']:
            
            if not gk_object_unpickled:
                
                if run_type in ['submit']:
                    gk_object.check(every_n_minutes=10)
                    gk_object.fetch()

                gk_object.read(
                    label=subfolder_name,
                    **simulation_options["read"]
                    )
                
                # Special case to keep only the pickle file but remove everything else
                if keep_gk_files in ['pickle']:
                    
                    # Remove the contents of subfolder_name
                    IOtools.shutil_rmtree(self.folder / f"{subfolder_name}")
                    
                    # Create the folder again
                    (self.folder / f"{subfolder_name}").mkdir(parents=True, exist_ok=True)
                    
                    # Save the gk_object as pickle
                    gk_object.save_pickle(pickle_file)
        
            # ------------------------------------------------------------------------------------------------------------------------
            # Pass the information to what power_transport expects
            # ------------------------------------------------------------------------------------------------------------------------

            self.QeGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Qe_mean for i in range(len(rho_locations))])
            self.QeGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Qe_std for i in range(len(rho_locations))])
                    
            self.QiGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Qi_mean for i in range(len(rho_locations))])
            self.QiGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Qi_std for i in range(len(rho_locations))])
                    
            self.GeGB_turb = np.array([gk_object.results[subfolder_name]['output'][i].Ge_mean for i in range(len(rho_locations))])
            self.GeGB_turb_stds = np.array([gk_object.results[subfolder_name]['output'][i].Ge_std for i in range(len(rho_locations))]) 
            
            self.GZGB_turb = self.QeGB_turb*0.0 #TODO     
            self.GZGB_turb_stds = self.QeGB_turb*0.0 #TODO          

            self.MtGB_turb = self.QeGB_turb*0.0 #TODO     
            self.MtGB_turb_stds = self.QeGB_turb*0.0 #TODO     

            self.QieGB_turb = self.QeGB_turb*0.0 #TODO     
            self.QieGB_turb_stds = self.QeGB_turb*0.0 #TODO     

        elif run_type == 'prep':
            
            # Prevent writing the json file from variables, as we will wait for the user to run CGYRO externally and provide the json themselves
            self._write_json_from_variables_turb = False
            
            # Wait until the user has placed the json file in the right folder
            
            self.powerstate.profiles_transport.write_state(self.folder / subfolder_name / "input.gacode")
            
            pre_checks(self)

            file_path = self.folder / 'fluxes_turb.json'

            attempts = 0
            all_good = post_checks(self) if file_path.exists() else False
            while (file_path.exists() is False) or (not all_good):
                if attempts > 0:
                    print(f"\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f" MITIM could not find the file... looping back", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                logic_to_wait(self.folder, self.folder / subfolder_name)
                attempts += 1

                if file_path.exists():
                    all_good = post_checks(self)

        if 'Qi_stable_criterion' in simulation_options:
            self._stable_correction(simulation_options)

    def _stable_correction(self, simulation_options):

        Qi_stable_criterion = simulation_options["Qi_stable_criterion"]
        Qi_stable_percent_error = simulation_options["Qi_stable_percent_error"]
        
        # Check if Qi in MW/m2 < Qi_stable_criterion
        QiMWm2 = self.QiGB_turb * self.powerstate.plasma['Qgb'][0,1:].cpu().numpy()
        QiGB_target = self.powerstate.plasma['QiGB'][0,1:].cpu().numpy()
        
        radii_stable = QiMWm2 < Qi_stable_criterion
        
        for i in range(len(radii_stable)):
            if radii_stable[i]:
                print(f"\t- Qi considered stable at radius #{i}, ({QiMWm2[i]:.2f} < {Qi_stable_criterion:.2f})", typeMsg='i')
                Qi_std = QiGB_target[i] * Qi_stable_percent_error / 100
                print(f"\t\t- Assigning {Qi_stable_percent_error:.1f}% from target as standard deviation: {Qi_std:.2f} instead of {self.QiGB_turb_stds[i]}", typeMsg='i')
                self.QiGB_turb_stds[i] = Qi_std


class cgyro_model(gyrokinetic_model):

    def evaluate_turbulence(self):

        if self.transport_evaluator_options["cgyro"].get("run_base_tglf", True):
            # Run base TGLF, to keep track of discrepancies! ---------------------------------------------
            simulation_options_tglf = self.transport_evaluator_options["tglf"]
            simulation_options_tglf["use_scan_trick_for_stds"] = None
            self._evaluate_tglf(pass_info = False)
            # --------------------------------------------------------------------------------------------

        self._evaluate_gyrokinetic_model(code = 'cgyro', gk_object = CGYROtools.CGYRO)


def pre_checks(self):
    
    plasma = self.powerstate.plasma

    txt = "\nFluxes to be matched by turbulence ( Target - Neoclassical ):"

    # Print gradients
    for var, varn in zip(
        ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
        ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
    ):
        txt += f"\n{var}   = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]:.6f}   "

    # Print target fluxes
    for var, varn in zip(
        ["Qe (GB)", "Qi (GB)", "Ge (GB)", "GZ (GB)", "Mt (GB)"],
        ["QeGB", "QiGB", "GeGB", "GZGB", "MtGB"],
    ):
        txt += f"\n{var}  = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]-self.__dict__[f'{varn}_neoc'][j]:.4e}   "

    print(txt)

def logic_to_wait(folder, subfolder):
    print(f"\n**** Simulation inputs prepared. Please, run it from the simulation setup in folder:\n", typeMsg='i')
    print(f"\t {subfolder}\n", typeMsg='i')
    print(f"**** When finished, the fluxes_turb.json file should be placed in:\n", typeMsg='i')
    print(f"\t {folder}/fluxes_turb.json\n", typeMsg='i')
    while not print(f"**** When you have done that, please say yes", typeMsg='q'):
        pass

def post_checks(self, rtol = 1e-3):
    
    with open(self.folder / 'fluxes_turb.json', 'r') as f:
        json_dict = json.load(f)
        
    additional_info_from_json = json_dict.get('additional_info', {})
    
    all_good = True
    
    if len(additional_info_from_json) == 0:
        print(f"\t- No additional info found in fluxes_turb.json to be compared with", typeMsg='i')
        
    else:
        print(f"\t- Additional info found in fluxes_turb.json:", typeMsg='i')
        for k, v in additional_info_from_json.items():
            vP = self.powerstate.plasma[k].cpu().numpy()[0,1:]
            print(f"\t   {k} from JSON      : {[round(i,4) for i in v]}", typeMsg='i')
            print(f"\t   {k} from POWERSTATE: {[round(i,4) for i in vP]}", typeMsg='i')

            if not np.allclose(v, vP, rtol=rtol):
                all_good = print(f"{k} does not match with a relative tolerance of {rtol*100.0:.2f}%:", typeMsg='q')

    return all_good

def write_json_CGYRO(roa, fluxes_mean, fluxes_stds, additional_info = None, file = 'fluxes_turb.json'):
    '''
    *********************
    Helper to write JSON
    *********************
        roa
            Must be an array: [0.25, 0.35, ...]
        fluxes_mean
            Must be a dictionary with the fields and arrays:
                'QeMWm2': [0.1, 0.2, ...],
                'QiMWm2': ...,
                'Ge1E20m2': ...,
                'GZ1E20m2': ...,
                'MtJm2': ...,
                'QieMWm3': ..
            or, alternatively (or complementary), in GB units:
                'QeGB': [0.1, 0.2, ...],
                'QiGB': ...,
                'GeGB': ...,
                'GZGB': ...,
                'MtGB': ...,
                'QieGB': ..
        fluxes_stds
            Exact same structure as fluxes_mean
        additional_info
            A dictionary with any additional information to include in the JSON and compare to powerstate,
            for example (and recommended):
                'aLte': [0.2, 0.5, ...],
                'aLti': [0.3, 0.6, ...],
                'aLne': [0.3, 0.6, ...],
                'Qgb': [0.4, 0.7, ...],
                'rho': [0.2, 0.5, ...],
    '''
    
    if additional_info is None:
        additional_info = {}

    with open(file, 'w') as f:

        additional_info_extended = additional_info | {'roa': roa.tolist() if not isinstance(roa, list) else roa}

        json_dict = {
            'fluxes_mean': fluxes_mean,
            'fluxes_stds': fluxes_stds,
            'additional_info': additional_info_extended
        }

        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.generic,)):
                return obj.item()
            else:
                return obj
            
        json.dump(convert_numpy(json_dict), f, indent=4)
