import json
import numpy as np
from mitim_tools.gacode_tools import CGYROtools
from mitim_modules.powertorch.physics_models import transport_tglfneo
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# Inherit from transport_tglfneo.tglfneo_model so that I have the NEO evaluator
class cgyroneo_model(transport_tglfneo.tglfneo_model):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)
        
    # Do not hook here
    def evaluate_turbulence(self):

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        transport_evaluator_options = self.powerstate.transport_options["transport_evaluator_options"]
        
        cold_start = transport_evaluator_options.get("cold_start", False)
        
        run_type = 'prep'

        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare CGYRO object
        # ------------------------------------------------------------------------------------------------------------------------
        
        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        cgyro = CGYROtools.CGYRO(rhos=rho_locations)

        _ = cgyro.prep(
            self.powerstate.profiles_transport,
            self.folder,
            )

        cgyro = CGYROtools.CGYRO(
            rhos = rho_locations
        )

        cgyro.prep(
            self.powerstate.profiles_transport.files[0],
            self.folder,
            )

        if run_type in ['normal', 'submit']:
            raise Exception("[MITIM] Automatic submission or full run not implemented")

            # cgyro.read(
            #     label='base_cgyro'
            #     )
            
            # TRANSPORTtools.write_json(self, file_name = 'fluxes_turb.json', suffix= 'turb')
            
        elif run_type == 'prep':

            _ = cgyro.run(
                'base_cgyro',
                run_type = run_type,
                code_settings=1,
                cold_start=cold_start,
                forceIfcold_start=True,
                )
    
            # Wait until the user has placed the json file in the right folder
            
            pre_checks(self)

            file_path = self.folder / 'fluxes_turb.json'

            attempts = 0
            all_good = False
            while (file_path.exists() is False) or (not all_good):
                if attempts > 0:
                    print(f"\n\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"\tMITIM could not find the file", typeMsg='i')
                    print(f" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n\n", typeMsg='i')
                logic_to_wait(self.folder)
                attempts += 1

                if file_path.exists():
                    all_good = post_checks(self)

def pre_checks(self):
    
    plasma = self.powerstate.plasma

    txt = "\nFluxes to be matched by CGYRO ( TARGETS - NEO ):"

    # Print gradients
    for var, varn in zip(
        ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
        ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
    ):
        txt += f"\n{var}        = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]:.6f}   "

    # Print target fluxes
    for var, varn in zip(
        ["Qe (MW/m^2)", "Qi (MW/m^2)", "Ge (1E20m2/s)", "GZ (1E20m2/s)", "Mt (J/m^2) "],
        ["QeMWm2", "QiMWm2", "Ge1E20m2", "GZ1E20m2", "MtJm2"],
    ):
        txt += f"\n{var}  = "
        for j in range(plasma["rho"].shape[1] - 1):
            txt += f"{plasma[varn][0,j+1]-self.__dict__[f'{varn}_tr_neoc'][j]:.4e}   "

    print(txt)

def logic_to_wait(folder):
    print(f"\n**** CGYRO prepared. Please, run CGYRO from the simulation setup in folder: ", typeMsg='i')
    print(f"\t {folder}/base_cgyro\n", typeMsg='i')
    print(f" **** When finished, the fluxes_turb.json file should be placed in:", typeMsg='i')
    print(f"\t {folder}/fluxes_turb.json\n", typeMsg='i')
    print(f" **** When you have done that, please write 'exit' and click enter (for continuing and reading that file)\n", typeMsg='i')


def post_checks(self, rtol = 1e-2):
    
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
                all_good = print(f"{k} does not match with a relative tolerance of {rtol}:", typeMsg='q')

    return all_good

def write_json_CGYRO(roa, fluxes_mean, fluxes_stds, additional_info, file = 'fluxes_turb.json'):
    '''
    Helper to write JSON
        roa must be an array: [0.25, 0.35, ...]
        fluxes_mean must be a dictionary with the fields and arrays:
            'QeMWm2': [0.1, 0.2, ...],
            'QiMWm2': ...,
            'Ge1E20m2': ...,
            'GZ1E20m2': ...,
            'MtJm2': ...,
            'QieMWm3': ...
        same for fluxes_stds
        additional_info must be a dictionary with any additional information to include in the JSON and compare to powerstate,
        for example:
            'aLte': [0.2, 0.5, ...],
            'aLti': [0.3, 0.6, ...],
            'aLne': [0.3, 0.6, ...],
            'Qgb': [0.4, 0.7, ...]
    '''
    
    
    with open(file, 'w') as f:

        fluxes_mean = {}
        fluxes_stds = {}
        additional_info_extended = additional_info | {'roa': roa.tolist()}

        json_dict = {
            'fluxes_mean': fluxes_mean,
            'fluxes_stds': fluxes_stds,
            'additional_info': additional_info_extended
        }

        json.dump(json_dict, f, indent=4)

