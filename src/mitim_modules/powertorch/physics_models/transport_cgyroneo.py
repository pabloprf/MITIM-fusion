from mitim_tools.misc_tools import IOtools
from functools import partial
from mitim_tools.gacode_tools import CGYROtools
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_modules.powertorch.physics_models import transport_tglfneo
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# Inherit from transport_tglfneo.tglfneo_model so that I have the NEO evaluator
class cgyroneo_model(transport_tglfneo.tglfneo_model):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)
        
    def evaluate_turbulence(self):

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        transport_evaluator_options = self.powerstate.transport_options["transport_evaluator_options"]
        
        cold_start = transport_evaluator_options.get("cold_start", False)
        
        automatic_run = False

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
        
        if automatic_run:
            raise Exception("[MITIM] automatic_run not implemented yet")
        
            # cgyro.read(
            #     label='base_cgyro'
            #     )
            
            # TRANSPORTtools.write_json(self, file_name = 'fluxes_turb.json', suffix= 'turb')
        
        else:
            
            # Just prepare everything but do not submit (full_submission=False)
            _ = cgyro.run(
                'base_cgyro',
                full_submission=False,
                code_settings=1,
                cold_start=cold_start,
                forceIfcold_start=True,
                )
    
            # Wait until the user has placed the json file in the right folder
            
            attempts = 0
            while (self.folder / 'fluxes_turb.json').exists() is False:
                if attempts > 0:
                    print(f"\n\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", typeMsg='i')
                    print(f"\tMITIM could not find the file", typeMsg='i')
                    print(f" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n\n", typeMsg='i')
                logic_to_wait(self.folder)
                attempts += 1

def logic_to_wait(folder):
    print(f" **** CGYRO prepared. Please, run CGYRO from the simulation setup in folder: ", typeMsg='i')
    print(f"\t {folder}/base_cgyro\n", typeMsg='i')
    print(f" **** When finished, the fluxes_turb.json file should be placed in:", typeMsg='i')
    print(f"\t {folder}/fluxes_turb.json\n", typeMsg='i')
    print(f" **** When you have done that, please write an 'exit' and enter (for continuing)\n", typeMsg='i')
    
    embed()
    
