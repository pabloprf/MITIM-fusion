from IPython import embed
from mitim_tools.gacode_tools.aux import GACODEdefaults


def profiles_to_tglf(self, rhos=[0.5], TGLFsettings=5):
    TGLFinput, TGLFoptions, label = GACODEdefaults.addTGLFcontrol(
        TGLFsettings
    )

    # self.controls, self.plasma, self.geom

    embed()
