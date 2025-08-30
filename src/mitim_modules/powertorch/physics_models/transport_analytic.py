import torch
from mitim_tools.misc_tools import PLASMAtools
from mitim_modules.powertorch.utils import TRANSPORTtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ------------------------------------------------------------------
# SIMPLE Diffusion (#TODO: implement with particle flux and the raw)
# ------------------------------------------------------------------

class diffusion_model(TRANSPORTtools.power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

        # Ensure that the provided diffusivities include the zero location
        self.chi_e = self.powerstate.transport_options["options"]["chi_e"]
        self.chi_i = self.powerstate.transport_options["options"]["chi_i"]

        if self.chi_e.shape[0] < self.powerstate.plasma['rho'].shape[-1]:
            self.chi_e = torch.cat((torch.zeros(1), self.chi_e))

        if self.chi_i.shape[0] < self.powerstate.plasma['rho'].shape[-1]:
            self.chi_i = torch.cat((torch.zeros(1), self.chi_i))

    def produce_profiles(self):
        pass

    def evaluate(self):

        # Make sure the chis are applied to all the points in the batch
        Pe_tr = PLASMAtools.conduction(
            self.powerstate.plasma["ne"],
            self.powerstate.plasma["te"],
            self.chi_e.repeat(self.powerstate.plasma['rho'].shape[0],1),
            self.powerstate.plasma["aLte"],
            self.powerstate.plasma["a"].unsqueeze(-1),
        )
        Pi_tr = PLASMAtools.conduction(
            self.powerstate.plasma["ni"].sum(axis=-1),
            self.powerstate.plasma["ti"],
            self.chi_i.repeat(self.powerstate.plasma['rho'].shape[0],1),
            self.powerstate.plasma["aLti"],
            self.powerstate.plasma["a"].unsqueeze(-1),
        )

        self.QeMWm2_tr_turb = Pe_tr * 2 / 3
        self.QiMWm2_tr_turb = Pi_tr * 2 / 3

        self.QeMWm2_tr_neoc = Pe_tr * 1 / 3
        self.QiMWm2_tr_neoc = Pi_tr * 1 / 3

        self.QeMWm2_tr = self.QeMWm2_tr_turb + self.QeMWm2_tr_neoc
        self.QiMWm2_tr = self.QiMWm2_tr_turb + self.QiMWm2_tr_neoc

# ------------------------------------------------------------------
# SURROGATE
# ------------------------------------------------------------------

class surrogate(TRANSPORTtools.power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        pass

    def evaluate(self):

        """
        flux_fun as given in transport_evaluator_options must produce Q and Qtargets in order of te,ti,ne
        """

        X = torch.Tensor()
        for prof in self.powerstate.predicted_channels:
            X = torch.cat((X,self.powerstate.plasma['aL'+prof][:,1:]),axis=1)

        _, Q, _, _ = self.powerstate.transport_options["options"]["flux_fun"](X)

        numeach = self.powerstate.plasma["rho"].shape[1] - 1

        quantities = {
            "te": "QeMWm2",
            "ti": "QiMWm2",
            "ne": "Ce",
            "nZ": "CZ",
            "w0": "MtJm2",
        }

        for c, i in enumerate(self.powerstate.predicted_channels):
            self.powerstate.plasma[f"{quantities[i]}_tr"] = torch.cat((torch.tensor([[0.0]]),Q[:, numeach * c : numeach * (c + 1)]),dim=1)

