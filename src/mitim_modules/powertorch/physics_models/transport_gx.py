from mitim_modules.powertorch.physics_models.transport_cgyro import gyrokinetic_model
from mitim_tools.simulation_tools.physics import GXtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class gx_model(gyrokinetic_model):
    def evaluate_turbulence(self):
        gx_key = getattr(self, "_active_turb_options_key", None) or "gx"
        self._evaluate_gyrokinetic_model(code=gx_key, gk_object=GXtools.GX)
