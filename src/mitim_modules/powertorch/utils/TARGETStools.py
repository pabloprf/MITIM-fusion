import torch
from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class power_targets:
    '''
    Default class for power target models, change "evaluate" method to implement a new model
    '''

    def evaluate(self):
        print("No model implemented for power targets", typeMsg="w")

    def __init__(self,powerstate):
        self.powerstate = powerstate

        # Make sub-targets equal to zero
        variables_to_zero = ["qfuse", "qfusi", "qie", "qrad", "qrad_bremms", "qrad_line", "qrad_sync"]
        for i in variables_to_zero:
            self.powerstate.plasma[i] = self.powerstate.plasma["te"] * 0.0

        # ----------------------------------------------------
        # Fixed Targets (targets without a model)
        # ----------------------------------------------------

        if self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] == 1:
            self.Pe_orig, self.Pi_orig = (
                self.powerstate.plasma["QeMWm2_orig_fusradexch"],
                self.powerstate.plasma["QiMWm2_orig_fusradexch"],
            )  # Original integrated from input.gacode
        elif self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] == 2:
            self.Pe_orig, self.Pi_orig = (
                self.powerstate.plasma["QeMWm2_orig_fusrad"],
                self.powerstate.plasma["QiMWm2_orig_fusrad"],
            )
        elif self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] == 3:
            self.Pe_orig, self.Pi_orig = self.powerstate.plasma["te"] * 0.0, self.powerstate.plasma["te"] * 0.0

        # For the moment, I don't have a model for these, so I just grab the original from input.gacode
        self.CextraE = self.powerstate.plasma["Gaux_e"]     # 1E20/s/m^2
        self.CextraZ = self.powerstate.plasma["Gaux_Z"]     # 1E20/s/m^2
        self.Mextra = self.powerstate.plasma["Maux"]        # J/m^2

    def fine_grid(self):

        """
        Make all quantities needed on the fine resolution
        -------------------------------------------------
            In the powerstate creation, the plasma variables are stored in two different resolutions, one for the coarse grid and one for the fine grid,
            if the option is activated.

            Here, at calculation stage I use some precalculated quantities in the fine grid and then integrate the gradients into that resolution

            Note that the set ['te','ti','ne','nZ','w0','ni'] will automatically be substituted during the update_var() that comes next, so
            it's ok that I lose the torch leaf here. However, I must do this copy here because if any of those variables are not updated in
            update_var() then it would fail. But first store them for later use.
        """

        self.plasma_original = {}

        # Bring to fine grid
        variables_to_fine = ["B_unit", "B_ref", "volp", "rmin", "roa", "rho", "ni"]
        for variable in variables_to_fine:
            self.plasma_original[variable] = self.powerstate.plasma[variable].clone()
            self.powerstate.plasma[variable] = self.powerstate.plasma_fine[variable]

        # Bring also the gradients and kinetic variables
        for variable in self.powerstate.profile_map.keys():

            # Kinetic variables (te,ti,ne,nZ,w0,ni)
            self.plasma_original[variable] = self.powerstate.plasma[variable].clone()
            self.powerstate.plasma[variable] = self.powerstate.plasma_fine[variable]

            # Bring also the gradients that are part of the torch trees, so that the derivative is not lost
            self.plasma_original[f'aL{variable}'] = self.powerstate.plasma[f'aL{variable}'].clone()

        # ----------------------------------------------------
        # Integrate through fine de-parameterization
        # ----------------------------------------------------
        for i in self.powerstate.ProfilesPredicted:
            _ = self.powerstate.update_var(i,specific_profile_constructor=self.powerstate.profile_constructors_coarse_middle)

    def flux_integrate(self):
        """
		**************************************************************************************************
		Calculate integral of all targets, and then sum aux.
		Reason why I do it this convoluted way is to make it faster in mitim, not to run integrateQuadPoly all the time.
		Run once for all the batch and also for electrons and ions
		(in MW/m^2)
		**************************************************************************************************
		"""

        qe = self.powerstate.plasma["te"]*0.0
        qi = self.powerstate.plasma["te"]*0.0
        
        if self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] >= 2:
            qe += -self.powerstate.plasma["qie"]
            qi +=  self.powerstate.plasma["qie"]
        
        if self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] == 3:
            qe +=  self.powerstate.plasma["qfuse"] - self.powerstate.plasma["qrad"]
            qi +=  self.powerstate.plasma["qfusi"]

        q = torch.cat((qe, qi)).to(qe)
        self.P = self.powerstate.volume_integrate(q, force_dim=q.shape[0])

    def coarse_grid(self):

        # **************************************************************************************************
        # Come back to original grid for targets
        # **************************************************************************************************

        # Interpolate results from fine to coarse (i.e. whole point is that it is better than integrate interpolated values)
        if self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] >= 2:
            for i in ["qie"]:
                self.powerstate.plasma[i] = self.powerstate.plasma[i][:, self.powerstate.positions_targets]
        
        if self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] == 3:
            for i in [
                "qfuse",
                "qfusi",
                "qrad",
                "qrad_bremms",
                "qrad_line",
                "qrad_sync",
            ]:
                self.powerstate.plasma[i] = self.powerstate.plasma[i][:, self.powerstate.positions_targets]
       
        self.P = self.P[:, self.powerstate.positions_targets]

        # Recover variables calculated prior to the fine-targets method
        for i in self.plasma_original:
            self.powerstate.plasma[i] = self.plasma_original[i]

    def postprocessing(self, useConvectiveFluxes=False, forceZeroParticleFlux=False, assumedPercentError=1.0):

        # **************************************************************************************************
        # Plug-in Targets
        # **************************************************************************************************

        self.powerstate.plasma["QeMWm2"] = (
            self.powerstate.plasma["Paux_e"] + self.P[: self.P.shape[0]//2, :] + self.Pe_orig
        )  # MW/m^2
        self.powerstate.plasma["QiMWm2"] = (
            self.powerstate.plasma["Paux_i"] + self.P[self.P.shape[0]//2 :, :] + self.Pi_orig
        )  # MW/m^2
        self.powerstate.plasma["Ce_raw"] = self.CextraE
        self.powerstate.plasma["CZ_raw"] = self.CextraZ
        self.powerstate.plasma["Mt"] = self.Mextra

        # Merge convective fluxes

        if useConvectiveFluxes:
            self.powerstate.plasma["Ce"] = PLASMAtools.convective_flux(
                self.powerstate.plasma["te"], self.powerstate.plasma["Ce_raw"]
            )  # MW/m^2
            self.powerstate.plasma["CZ"] = PLASMAtools.convective_flux(
                self.powerstate.plasma["te"], self.powerstate.plasma["CZ_raw"]
            )  # MW/m^2
        else:
            self.powerstate.plasma["Ce"] = self.powerstate.plasma["Ce_raw"]
            self.powerstate.plasma["CZ"] = self.powerstate.plasma["CZ_raw"]

        if forceZeroParticleFlux:
            self.powerstate.plasma["Ce"] = self.powerstate.plasma["Ce"] * 0
            self.powerstate.plasma["Ce_raw"] = self.powerstate.plasma["Ce_raw"] * 0

        # **************************************************************************************************
        # Error
        # **************************************************************************************************

        variables_to_error = ["QeMWm2", "QiMWm2", "Ce", "CZ", "Mt", "Ce_raw", "CZ_raw"]

        for i in variables_to_error:
            self.powerstate.plasma[i + "_stds"] = abs(self.powerstate.plasma[i]) * assumedPercentError / 100 

        """
		**************************************************************************************************
		GB Normalized
		**************************************************************************************************
			Note: This is useful for mitim surrogate variables of targets
		"""

        self.powerstate.plasma["QeGB"] = self.powerstate.plasma["QeMWm2"] / self.powerstate.plasma["Qgb"]
        self.powerstate.plasma["QiGB"] = self.powerstate.plasma["QiMWm2"] / self.powerstate.plasma["Qgb"]
        self.powerstate.plasma["CeGB"] = self.powerstate.plasma["Ce"] / self.powerstate.plasma["Qgb" if useConvectiveFluxes else "Ggb"]
        self.powerstate.plasma["CZGB"] = self.powerstate.plasma["CZ"] / self.powerstate.plasma["Qgb" if useConvectiveFluxes else "Ggb"]
        self.powerstate.plasma["MtGB"] = self.powerstate.plasma["Mt"] / self.powerstate.plasma["Pgb"]
