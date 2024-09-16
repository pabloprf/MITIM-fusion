import torch
from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ------------------------------------------------------------------
# Main classes
# ------------------------------------------------------------------

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
                self.powerstate.plasma["Pe_orig_fusradexch"],
                self.powerstate.plasma["Pi_orig_fusradexch"],
            )  # Original integrated from input.gacode
        elif self.powerstate.TargetOptions['ModelOptions']['TypeTarget'] == 2:
            self.Pe_orig, self.Pi_orig = (
                self.powerstate.plasma["Pe_orig_fusrad"],
                self.powerstate.plasma["Pi_orig_fusrad"],
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
            _ = self.powerstate.update_var(
                i,
                specific_deparametrizer=self.powerstate.deparametrizers_coarse_middle,
            )

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

        self.powerstate.plasma["Pe"] = (
            self.powerstate.plasma["Paux_e"] + self.P[: self.P.shape[0]//2, :] + self.Pe_orig
        )  # MW/m^2
        self.powerstate.plasma["Pi"] = (
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

        variables_to_error = ["Pe", "Pi", "Ce", "CZ", "Mt", "Ce_raw", "CZ_raw"]

        for i in variables_to_error:
            self.powerstate.plasma[i + "_stds"] = self.powerstate.plasma[i] * assumedPercentError / 100 

        """
		**************************************************************************************************
		GB Normalized
		**************************************************************************************************
			Note: This is useful for mitim surrogate variables of targets
		"""

        gb_mapping = {
            "Pe": "Qgb",
            "Pi": "Qgb",
            "Ce": "Qgb" if useConvectiveFluxes else "Ggb",
            "CZ": "Qgb" if useConvectiveFluxes else "Ggb",
            "Mt": "Pgb",
        }

        for i in gb_mapping.keys():
            self.powerstate.plasma[f"{i}GB"] = self.powerstate.plasma[i] / self.powerstate.plasma[gb_mapping[i]]

# ----------------------------------------------------------------------------------------------------
# Full analytical models taken from TGYRO
# ----------------------------------------------------------------------------------------------------

# Global physical constants

u = 1.66054e-24  # g
Ae = 9.1094e-28 / u  # Electron mass in atomic units # 9.1094E-28/u
Aalpha = 2 * (3.34358e-24) / u  # Alpha mass in atomic units
Ealpha = 3.5e6  # eV
e = 1.60218e-19
pi = 3.14159265

# For Bosh XS
c1, c2, c3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
c4, c5, c6, c7 = 4.60643e-3, 1.3500e-2, -1.06750e-4, 1.36600e-5
bg, er = 34.3827, 1.124656e6

class analytical_model(power_targets):
    def __init__(self,powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def evaluate(self):

        if self.powerstate.TargetOptions["ModelOptions"]["TypeTarget"] >= 2:
            self._evaluate_energy_exchange()

        if self.powerstate.TargetOptions["ModelOptions"]["TypeTarget"] == 3:
            self._evaluate_alpha_heating()
            self._evaluate_radiation()

    def _evaluate_energy_exchange(self):
        '''
        ----------------------------------------------------
        Classical energy exchange
        ----------------------------------------------------
        '''

        self.powerstate.plasma["qie"] = PLASMAtools.energy_exchange(
            self.powerstate.plasma["te"],
            self.powerstate.plasma["ti"],
            self.powerstate.plasma["ne"] * 1e-1,
            self.powerstate.plasma["ni"] * 1e-1,
            self.powerstate.plasma["ions_set_mi"],
            self.powerstate.plasma["ions_set_Zi"],
        )

    def _evaluate_alpha_heating(self):
        '''
        ----------------------------------------------------
        Alpha heating
        ----------------------------------------------------

        This script calculates the power density profile (W/cm^3) that goes to ions and to electrons from
        fusion alphas, using kinetic profiles as inputs.

        This method follows the same methodology as in TGYRO [Candy et al. PoP 2009] and all the credits
        are due to the authors of TGYRO. From the source code, this function follows the same procedures
        as in tgyro_auxiliary_routines.f90.

        '''

        # -----------------------------------------------------------
        # Obtain the Deuterium and Tritium densities,
        # otherwise there is no alpha power and zeros are returned
        # -----------------------------------------------------------

        if (not self.powerstate.plasma["ions_set_Dion"][0].isnan()) and (not self.powerstate.plasma["ions_set_Tion"][0].isnan()):
            n_d = self.powerstate.plasma["ni"][..., self.powerstate.plasma["ions_set_Dion"][0]] * 1e19
            n_t = self.powerstate.plasma["ni"][..., self.powerstate.plasma["ions_set_Tion"][0]] * 1e19  # m^-3
        else:
            self.powerstate.plasma["qfusi"] = self.powerstate.plasma["te"] * 0.0
            self.powerstate.plasma["qfuse"] = self.powerstate.plasma["te"] * 0.0
            return

        # -----------------------------------------------------------
        # Alpha energy birth rate
        # -----------------------------------------------------------

        sigv = sigv_fun(self.powerstate.plasma["ti"])
        s_alpha_he = sigv * (n_d * 1e-6) * (n_t * 1e-6)  # Reactions/cm^3/s
        p_alpha_he = s_alpha_he * Ealpha * e  # W/cm^3

        # -----------------------------------------------------------
        # Partition between electrons and ions
        # 	from [Stix, Plasma Phys. 14 (1972) 367], Eqs. 15 and 17
        # -----------------------------------------------------------

        c_a = self.powerstate.plasma["te"] * 0.0
        for i in range(self.powerstate.plasma["ni"].shape[2]):
            c_a += (self.powerstate.plasma["ni"][..., i] / self.powerstate.plasma["ne"]) * self.powerstate.plasma["ions_set_Zi"][:,i].unsqueeze(-1) ** 2 * (Aalpha / self.powerstate.plasma["ions_set_mi"][:,i].unsqueeze(-1))

        W_crit = (self.powerstate.plasma["te"] * 1e3) * (4 * (Ae / Aalpha) ** 0.5 / (3 * pi**0.5 * c_a)) ** (
            -2.0 / 3.0
        )  # in eV

        frac_ai = sivukhin(Ealpha / W_crit)  # This solves Eq 17 of Stix

        # -----------------------------------------------------------
        # Return power density profile
        # -----------------------------------------------------------
        self.powerstate.plasma["qfusi"] = p_alpha_he * frac_ai
        self.powerstate.plasma["qfuse"] = p_alpha_he * (1 - frac_ai)

    def _evaluate_radiation(self):

        """
        ----------------------------------------------------
        Radiation
        ----------------------------------------------------

        This script calculates the radiated power density profile (W/cm^3) from synchrotron,
        Bremsstralung and line radiation.
        Note that the ADAS data embeded in the Chebyshev polynomial coefficients already includes
        Bremsstralung and therefore to separate in between the two, it must be estimated somehow else.

        It follows the methodology in TGYRO [Candy et al. PoP 2009]. All the credits are due to
        the authors of TGYRO
        """

        Te_keV = self.powerstate.plasma["te"]
        ne20 = self.powerstate.plasma["ne"] * 1e-1
        b_ref = self.powerstate.plasma["B_ref"]
        aspect_rat = 1/self.powerstate.plasma["eps"]
        r_min = self.powerstate.plasma["a"]
        ni20 = self.powerstate.plasma["ni"] * 1e-1
        c_rad = self.powerstate.plasma["ions_set_c_rad"]
        Zi = self.powerstate.plasma["ions_set_Zi"]

        # ----------------------------------------------------
        # Bremsstrahlung + Line
        # ----------------------------------------------------

        # Calling chevychev polys only once for all the species at the same time, for speed
        Adas = adas_aurora(Te_keV, c_rad)
        Pcool = ne20 * (Adas * ni20.permute(2, 0, 1)).sum(dim=0) # Sum over species

        # ----------------------------------------------------
        # Bremsstrahlung
        # ----------------------------------------------------

        f = 0.005344  # 1.69e-32*(1E20*1E-6)**2*(1E3)**0.5
        self.powerstate.plasma["qrad_bremms"] = f * ne20 * (ni20 * Zi.unsqueeze(1)**2).sum(dim=-1) * Te_keV**0.5

        # ----------------------------------------------------
        # Line
        # ----------------------------------------------------

        # TGYRO "Trick": Calculate bremmstrahlung separate and substract to Pcool to get the actual line
        self.powerstate.plasma["qrad_line"] = Pcool - self.powerstate.plasma["qrad_bremms"]

        # ----------------------------------------------------
        # Synchrotron
        # ----------------------------------------------------
        self.powerstate.plasma["qrad_sync"] = PLASMAtools.synchrotron(Te_keV, ne20, b_ref, aspect_rat.unsqueeze(-1), r_min.unsqueeze(-1))

        # ----------------------------------------------------
        # Total radiation
        # ----------------------------------------------------

        self.powerstate.plasma["qrad"] = (
            self.powerstate.plasma["qrad_sync"] + self.powerstate.plasma["qrad_line"] + self.powerstate.plasma["qrad_bremms"]
        )

def adas_aurora(Te, c):
    """
    - This script calculates the cooling reate from ADAS data of impurity ions (erg cm^3/s), using Te (keV).
    - It follows the methodology in TGYRO [Candy et al. PoP 2009]. All the credits are due to the authors of TGYRO
    - Improvements have been made to make it faster, by taking into account array operations within pytorch rather than loops

    - Input comes as Te[batch,nR] and c[batch,nZ,nR]
    - Output comes as lz[nZ,batch,nR]
    """
    
    # Define Chebyshev grid
    precomputed_factor = 0.28953  # 2/torch.log(t1_adas/t0_adas), 50.0/0.05
    x = (-1.0 + precomputed_factor * torch.log(Te / 0.05)).clip(min=-1, max=1)

    # Chebyshev series ( T_k(x) = cos[k*arccos(x)] )
    precomputed_factor = 48.3542  # log( (1E20*1E-6)**2 * 1E-7 )
    iCoeff = torch.linspace(0, 11, 12).to(Te)

    lz = torch.exp(
        precomputed_factor
        + (
            c.permute(2, 1, 0)[...,None]
            * torch.cos(iCoeff[:, None, None, None] * torch.acos(x))
        ).sum(dim=0)
    )

    return lz

def sigv_fun(ti):
    """
    This script calculates the DT fusion reaction rate coefficient (cm^3/s) from ti (keV), following
    [H.-S. Bosch and G.M. Hale, Nucl. Fusion 32 (1992) 611]

    This method follows the same methodology as in TGYRO [Candy et al. PoP 2009] and all the credits
    are due to the authors of TGYRO. From the source code, this function follows the same procedures
    as in tgyro_auxiliary_routines.f90.
    """

    r0 = ti * (c2 + ti * (c4 + ti * c6)) / (1.0 + ti * (c3 + ti * (c5 + ti * c7)))
    theta = ti / (1.0 - r0)
    xi = (bg**2 / (4.0 * theta)) ** (1.0 / 3.0)

    sigv = c1 * theta * (xi / (er * ti**3)) ** 0.5 * torch.exp(-3.0 * xi)

    return sigv


def sivukhin(x, n=12):
    """
    This script implements the TGYRO's sivukhin algorithm.
    This method follows the same methodology as in TGYRO [Candy et al. PoP 2009] and all the credits
    are due to the authors of TGYRO.

    Improvements have been made to make it faster, by taking into account
    array operations within pytorch rather than loops
    """

    # --------------
    # Asymptotes
    # --------------

    v = 0.866025  # sin(2*pi/3)
    f = (2 * pi / 3) / v - 2.0 / x**0.5 + 0.5 / (x * x)
    sivukhin1 = f / x

    sivukhin3 = 1.0 - 0.4 * x**1.5

    # --------------
    # Numerical (middle)
    # --------------

    dy = x / (n - 1)
    f = 0.0
    for i in range(n):
        yi = i * dy
        if i == 0 or i == n - 1:
            f = f + 0.5 / (1.0 + yi**1.5)
        else:
            f = f + 1.0 / (1.0 + yi**1.5)
    f = f * dy

    sivukhin2 = f / x

    # --------------
    # Construct
    # --------------

    sivukhin = (
        (x > 4.0) * sivukhin1
        + (x < 4.0) * (x > 0.1) * sivukhin2
        + (x < 0.1) * sivukhin3
    )

    return sivukhin
