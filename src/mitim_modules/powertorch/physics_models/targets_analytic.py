import torch
from mitim_tools.misc_tools import PLASMAtools
from mitim_modules.powertorch.utils import TARGETStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

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

class analytical_model(TARGETStools.power_targets):
    def __init__(self,powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def evaluate(self):

        if self.powerstate.target_options["target_evaluator_options"]["TypeTarget"] >= 2:
            self._evaluate_energy_exchange()

        if self.powerstate.target_options["target_evaluator_options"]["TypeTarget"] == 3:
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
