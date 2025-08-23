from mitim_tools.gacode_tools import CGYROtools
from mitim_modules.powertorch.physics_models import transport_tglfneo
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class cgyroneo_model(transport_tglfneo.tglfneo_model):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)
        
    def evaluate_turbulence(self):

        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        transport_evaluator_options = self.powerstate.transport_options["transport_evaluator_options"]
        
        cold_start = transport_evaluator_options.get("cold_start", False)

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
        
        _ = cgyro.run(
            'base_cgyro',
            full_submission=False,
            code_settings=1,
            cold_start=cold_start,
            forceIfcold_start=True,
            )
    
        # cgyro.read(
        #     label='base_cgyro'
        #     )
        
        embed()