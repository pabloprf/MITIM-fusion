from mitim_tools.gacode_tools import CGYROtools
from mitim_modules.powertorch.physics_models import transport_tglfneo
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class cgyroneo_model(transport_tglfneo.tglfneo_model):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)
        
    def evaluate_turbulence(self):

        pass
        # # ------------------------------------------------------------------------------------------------------------------------
        # # Prepare CGYRO object
        # # ------------------------------------------------------------------------------------------------------------------------
        
        # rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        # tglf = TGLFtools.CGYRO(rhos=rho_locations)

        # _ = tglf.prep(
        #     self.powerstate.profiles_transport,
        #     self.folder,
        #     cold_start = cold_start,
        #     )


        # cgyro = CGYROtools.CGYRO()

        # cgyro.prep(
        #     self.folder,
        #     self.powerstate.profiles_transport.files[0],
        #     )
        
        # cgyro.run(
        #     'base_cgyro',
        #     roa = 0.55,
        #     CGYROsettings=0,
        #     submit_run=False
        #     )
    
        # embed()
        # #cgyro.read(label='base')
        