import os
import copy
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

class eped_beat(beat):

    def __init__(self, maestro_instance):
        super().__init__(maestro_instance, beat_name = 'eped')

    def prepare(self, nn_location, norm_location):

        # Initialize if necessary
        if not self.initialize_called:
            self.initialize()
        # -----------------------------

        self.nn = NNtools.eped_nn(type='tf')
        nn_location = IOtools.expandPath(nn_location)
        norm_location = IOtools.expandPath(norm_location)

        self.nn.load(nn_location, norm=norm_location)

    def run(self, **kwargs):

        # ---------------------------------
        # Grab inputs from profiles_current
        # ---------------------------------

        embed()
        Ip = self.profiles_current.profiles['current'][0]
        Bt = self.profiles_current.profiles['current'][1]
        R = self.profiles_current.profiles['geometry'][0]
        a = self.profiles_current.profiles['geometry'][1]
        kappa995 = self.profiles_current.profiles['geometry'][2]
        delta995 = self.profiles_current.profiles['geometry'][3]
        neped = self.profiles_current.profiles['profiles']['ne'][1]
        betan = self.profiles_current.profiles['profiles']['beta_n'][1]
        zeff = self.profiles_current.profiles['profiles']['zeff'][1]
        tesep = self.profiles_current.profiles['profiles']['te'][0]
        nsep_ratio = self.profiles_current.profiles['profiles']['ne'][0]
        
        # ---------------------------------
        # Run the NN
        # ---------------------------------
        ptop, wtop = self.nn(Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff, tesep=tesep, nesep_ratio=nsep_ratio)
       
        # ---------------------------------
        # Modify profiles
        # ---------------------------------
        pass

    def finalize(self):
        pass

    def merge_parameters(self):
        pass

    def grab_output(self, full = False):
        pass

    def plot(self,  fn = None, counter = 0, full_plot = True):
        pass

    def finalize_maestro(self):
        pass
