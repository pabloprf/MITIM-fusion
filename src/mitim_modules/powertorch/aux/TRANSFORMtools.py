import copy, torch
import numpy as np
from IPython import embed
from mitim_modules.powertorch.aux import PARAMtools
from mitim_modules.powertorch.physics import TARGETStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpFunction

"""
--------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS THAT DEAL WITH THE CONNECTION BETWEEN POWERSTATE AND PROFILES_GACODE (READING AND WRITING)
--------------------------------------------------------------------------------------------------------------------------------------------
"""


def fromPowerToGacode(
    self,
    profiles_evaluate,
    PositionInBatch=0,
    options={},
    insertPowers=True,
    rederive=True,
    ProfilesPredicted=["te", "ti", "ne"],
    impurityPosition=1,
):
    """
    Notes:
            - This function assumes that "profiles" is the PROFILES_GACODE that everything started with.
            - We assume that what changes is only the kinetic profiles allowed to vary.
            - This only works for a single profile, in PositionInBatch
            - rederive is expensive, so I'm not re-deriving the geometry which is the most expensive
    """

    profiles = copy.deepcopy(profiles_evaluate)

    # ------------------------------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------------------------------
    Tfast_ratio = options.get("Tfast_ratio", True)
    Ti_thermals = options.get("Ti_thermals", True)
    ni_thermals = options.get("ni_thermals", True)
    recompute_ptot = options.get("recompute_ptot", True)
    ensureMachNumber = options.get("ensureMachNumber", None)

    # ------------------------------------------------------------------------------------------
    # Insert Te
    # ------------------------------------------------------------------------------------------

    roa = profiles.profiles["rmin(m)"] / profiles.profiles["rmin(m)"][-1]

    if "te" in ProfilesPredicted:
        print("\t- Changing Te")
        x, y = self.deparametrizers["te"](
            self.plasma["roa"][PositionInBatch, :],
            self.plasma["aLte"][PositionInBatch, :],
        )
        y_interpolated = interpFunction(roa, x.cpu(), y[0, :].cpu())
        te = copy.deepcopy(profiles.profiles["te(keV)"])
        profiles.profiles["te(keV)"] = y_interpolated

        if Tfast_ratio:
            print(
                "\t\t* If any fast species, changing Tfast to ensure fixed Tfast/Te ratio",
                typeMsg="i",
            )
            for sp in range(len(profiles.Species)):
                if profiles.Species[sp]["S"] == "fast":
                    print(f"\t\t\t- Changing temperature of species #{sp}")
                    profiles.profiles["ti(keV)"][:, sp] = profiles.profiles["ti(keV)"][
                        :, sp
                    ] * (profiles.profiles["te(keV)"] / te)

    # ------------------------------------------------------------------------------------------
    # Insert Ti
    # ------------------------------------------------------------------------------------------

    if "ti" in ProfilesPredicted:
        print("\t- Changing Ti")
        x, y = self.deparametrizers["ti"](
            self.plasma["roa"][PositionInBatch, :],
            self.plasma["aLti"][PositionInBatch, :],
        )
        y_interpolated = interpFunction(roa, x.cpu(), y[0, :].cpu())
        profiles.profiles["ti(keV)"][:, 0] = y_interpolated

        if Ti_thermals:
            print("\t\t* Ensuring Ti is equal for all thermal ions", typeMsg="i")
            profiles.makeAllThermalIonsHaveSameTemp()

    # ------------------------------------------------------------------------------------------
    # Insert ne
    # ------------------------------------------------------------------------------------------

    if "ne" in ProfilesPredicted:
        print("\t- Changing ne")
        x, y = self.deparametrizers["ne"](
            self.plasma["roa"][PositionInBatch, :],
            self.plasma["aLne"][PositionInBatch, :],
        )
        y_interpolated = interpFunction(roa, x.cpu(), y[0, :].cpu())
        ne = copy.deepcopy(profiles.profiles["ne(10^19/m^3)"])
        profiles.profiles["ne(10^19/m^3)"] = y_interpolated

        if ni_thermals:
            scaleFactor = y_interpolated / ne
            print("\t\t* Adjusting ni of thermal ions", typeMsg="i")
            profiles.scaleAllThermalDensities(scaleFactor=scaleFactor)

    # ------------------------------------------------------------------------------------------
    # Insert nZ (after scaling rest of ni)
    # ------------------------------------------------------------------------------------------

    if "nZ" in ProfilesPredicted:
        print(f"\t- Changing ni{self.impurityPosition}")
        x, y = self.deparametrizers["nZ"](
            self.plasma["roa"][PositionInBatch, :],
            self.plasma["aLnZ"][PositionInBatch, :],
        )
        y_interpolated = interpFunction(roa, x.cpu(), y[0, :].cpu())
        profiles.profiles["ni(10^19/m^3)"][
            :, self.impurityPosition - 1
        ] = y_interpolated

    # ------------------------------------------------------------------------------------------
    # Insert w0
    # ------------------------------------------------------------------------------------------

    if "w0" in ProfilesPredicted:
        print(f"\t- Changing w0")
        factor_mult = 1 / factorMult_w0(self)
        x, y = self.deparametrizers["w0"](
            self.plasma["roa"][PositionInBatch, :],
            self.plasma["aLw0"][PositionInBatch, :],
        )
        y_interpolated = interpFunction(roa, x.cpu(), y[0, :].cpu())
        profiles.profiles["w0(rad/s)"] = factor_mult * y_interpolated

    # ------------------------------------------------------------------------------------------
    # Insert Powers
    # ------------------------------------------------------------------------------------------

    if insertPowers:
        insertPowersNew(profiles, state=self)

    # ------------------------------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------------------------------

    if (ensureMachNumber is not None) and ("w0" not in ProfilesPredicted):
        profiles.introduceRotationProfile(Mach_LF=ensureMachNumber)

    # ------------------------------------------------------------------------------------------
    # Recalculate
    # ------------------------------------------------------------------------------------------

    if rederive or recompute_ptot:
        profiles.deriveQuantities(rederiveGeometry=False)

    # ------------------------------------------------------------------------------------------
    # Change ptot to make it consistent?
    # ------------------------------------------------------------------------------------------
    if recompute_ptot:
        profiles.selfconsistentPTOT()

    return profiles


def insertPowersNew(profiles, state=None):
    if state is None:
        from mitim_modules.powertorch.STATEtools import powerstate

        state = powerstate(profiles, np.array([0.5, 0.8]))

    print("\t- Insering powers")

    # Modify power flows by tricking the powerstate into a fine grid (same as does TGYRO)
    extra_points = 2  # If I don't allow this, it will fail
    state_temp = copy.deepcopy(state)
    rhoy = profiles.profiles["rho(-)"][:-extra_points]
    with IOtools.HiddenPrints():
        state_temp.__init__(profiles, rhoy)
    state_temp.calculateProfileFunctions()
    state_temp.TargetCalc = "powerstate"
    state_temp.calculateTargets()
    state_temp.unrepeat()
    conversions = {
        "qie": "qei(MW/m^3)",
        "qrad_bremms": "qbrem(MW/m^3)",
        "qrad_sync": "qsync(MW/m^3)",
        "qrad_line": "qline(MW/m^3)",
        "qfuse": "qfuse(MW/m^3)",
        "qfusi": "qfusi(MW/m^3)",
    }
    for ikey in conversions:
        if conversions[ikey] in profiles.profiles:
            profiles.profiles[conversions[ikey]][:-extra_points] = (
                state_temp.plasma[ikey].cpu().numpy()
            )
        else:
            profiles.profiles[conversions[ikey]] = np.zeros(
                len(profiles.profiles["qei(MW/m^3)"])
            )
            profiles.profiles[conversions[ikey]][:-extra_points] = (
                state_temp.plasma[ikey].cpu().numpy()
            )


# --------------------------------------------------------------------------------------------------------------------------------------------


def fromGacodeToPower(self, input_gacode, rho_vec):
    """
    This function is used to convert from the fine input.gacode grid to one that is used to run
    TGYRO or PowerTorch. In particular, those quantities useful for mitim are used here.

    Goal is to avoid these calculations within the transport solver calculation, so have them only once.

    rho_vec is rho, but I will work from now on with roa for easy intergrations

    """

    # What is the radial coordinate of rho_vec?
    rho_use = input_gacode.profiles["rho(-)"]

    dfT = rho_vec.clone()

    rho_vec = rho_vec.cpu()  # Because of the numpy operations

    # Radial positions ----------------------------------------------------------------------------
    roa_array = interpFunction(rho_vec, rho_use, input_gacode.derived["roa"])
    rho_array = interpFunction(rho_vec, rho_use, input_gacode.profiles["rho(-)"])
    print("\t- Producing powerstate")
    if len(rho_array) < 10:
        print(f"\t\t@ rho = {[round(i,6) for i in rho_array]}")
        print(f"\t\t@ r/a = {[round(i,6) for i in roa_array]}")
    else:
        print(f"\t\t@ {len(rho_array)} rho points")
    # ---------------------------------------------------------------------------------------------

    self.plasma["roa"] = torch.from_numpy(roa_array).to(dfT)
    self.plasma["rho"] = torch.from_numpy(rho_array).to(dfT)

    # In case rho_vec is different than rho_vec_evaluate -------------------------------------------------------------------
    self.indexes_simulation = []
    for rho in self.plasma["rho"]:
        self.indexes_simulation.append(
            np.argmin(np.abs(self.plasma["rho"].cpu() - rho.cpu())).item()
        )
    # ------------------------------------------------------------------------------------------------------------------

    self.plasma["volp"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.derived["volp_miller"])
    ).to(dfT)
    self.plasma["rmin"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.profiles["rmin(m)"])
    ).to(dfT)
    self.plasma["te"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.profiles["te(keV)"])
    ).to(dfT)
    self.plasma["ti"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.profiles["ti(keV)"][:, 0])
    ).to(dfT)
    self.plasma["ne"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.profiles["ne(10^19/m^3)"])
    ).to(dfT)
    self.plasma["nZ"] = torch.from_numpy(
        interpFunction(
            rho_vec,
            rho_use,
            input_gacode.profiles["ni(10^19/m^3)"][:, self.impurityPosition - 1],
        )
    ).to(dfT)
    self.plasma["w0"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.profiles["w0(rad/s)"])
    ).to(dfT)
    self.plasma["B_unit"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.derived["B_unit"])
    ).to(dfT)
    self.plasma["mi_u"] = input_gacode.mi_first
    self.plasma["a"] = input_gacode.derived["a"]
    self.plasma["eps"] = input_gacode.derived["eps"]
    self.plasma["B_ref"] = torch.from_numpy(
        interpFunction(rho_vec, rho_use, input_gacode.derived["B_ref"])
    ).to(dfT)

    """
	Auxiliary powers that are not varied within TGYRO
	--------------------------------------------------------------------------------------------------
	Notes:
		- This is the TGYRO definition, with qione summed, giving the intented flows)
		- Same as in TGYRO: The interpolation of those quantities that do not vary throughout the workflow
							needs to happen AFTER integration (otherwise it would be too coarse to calculate auxiliary deposition)
		- Because TGYRO doesn't care about convective fluxes, I need to convert it AFTER I have the ge_Miller integrated and interpolated, so this happens
							at each powerstate evaluation
	"""
    self.plasma["PauxE"] = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["qe_aux_MWmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )
    self.plasma["PauxI"] = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["qi_aux_MWmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )
    self.plasma["GauxE"] = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["ge_10E20miller"])
        ).to(dfT)
        / self.plasma["volp"]
    )
    self.plasma["GauxZ"] = self.plasma["GauxE"] * 0.0

    # Momentum flux is J/m^2. Momentum source is given in input.gacode a N/m^2 or J/m^3. Integrated in volume is J or N*m
    self.plasma["MauxT"] = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["mt_Jmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )

    # Also include radiation, alpha and exchange, in case I want to run powerstate with fixed those
    Prad = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["qrad_MWmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )
    PfusE = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["qe_fus_MWmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )
    PfusI = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["qi_fus_MWmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )
    Pei = (
        torch.from_numpy(
            interpFunction(rho_vec, rho_use, input_gacode.derived["qe_exc_MWmiller"])
        ).to(dfT)
        / self.plasma["volp"]
    )

    self.plasma["PextraE_Target1"] = PfusE - Prad - Pei
    self.plasma["PextraI_Target1"] = PfusI + Pei
    self.plasma["PextraE_Target2"] = PfusE - Prad
    self.plasma["PextraI_Target2"] = PfusI

    # Ions --------
    defineIons(self, input_gacode, rho_vec, dfT)

    """
	# ------------------------------------------------------------------------------------------------------------------------
	# Define deparametrizer functions for the varying profiles and gradients from here
	# ------------------------------------------------------------------------------------------------------------------------
	I define the gradients like that to impose a zero at the parameterization stage
	"""
    defineMovingGradients(self, input_gacode)


def defineIons(self, input_gacode, rho_vec, dfT):
    """
    Store as part of powerstate the thermal ions densities (ni) and the information about how to interpret them (ions_set)
    Notes:
            - I'm not including here the fast species because they are not changing during the powerstate process. I don't have enough physics
              to understand how they vary. It's better to keep them fixed.
            - Note that this means that it won't calculate radiation based on fast species (e.g. 5% He3 minority won't radiate)
    """

    # What is the radial coordinate of rho_vec?
    rho_use = input_gacode.profiles["rho(-)"]

    # If I'm grabbing original axis, include as part of powerstate all the grid points near axis
    rho_vec = rho_vec.clone()
    rho_vec = rho_vec.cpu()

    # ** Store the information about the thermal ions, including the cooling coefficients
    self.plasma["ni"], mi, Zi, c_rad = [], [], [], []
    for i in range(len(input_gacode.profiles["mass"])):
        if input_gacode.profiles["type"][i] == "[therm]":
            self.plasma["ni"].append(
                interpFunction(
                    rho_vec, rho_use, input_gacode.profiles["ni(10^19/m^3)"][:, i]
                )
            )
            mi.append(input_gacode.profiles["mass"][i])
            Zi.append(input_gacode.profiles["z"][i])

            c = TARGETStools.get_chebyshev_coeffs(input_gacode.profiles["name"][i])

            c_rad.append(c)

    self.plasma["ni"] = torch.from_numpy(np.transpose(self.plasma["ni"])).to(dfT)
    mi = torch.from_numpy(np.array(mi)).to(dfT)
    Zi = torch.from_numpy(np.array(Zi)).to(dfT)
    c_rad = torch.from_numpy(np.array(c_rad)).to(dfT)

    # ** Positions of DT ions in the ions array, which will be used to calculate alpha power
    Dion, Tion = input_gacode.Dion, input_gacode.Tion

    # Only store as part of ions_set those IMMUTABLE parameters (NOTE: ni MAY change so that's why it's not here)
    self.plasma["ions_set"] = [mi, Zi, Dion, Tion, c_rad]


def defineMovingGradients(self, profiles):
    (
        self.deparametrizers,
        self.deparametrizers_coarse,
        self.deparametrizers_coarse_middle,
    ) = ({}, {}, {})

    # ----------------------------------------------------------------------------------------------------
    # Te
    # ----------------------------------------------------------------------------------------------------
    (
        aLy_coarse,
        self.deparametrizers["te"],
        self.deparametrizers_coarse["te"],
        self.deparametrizers_coarse_middle["te"],
    ) = PARAMtools.performCurveRegression(
        profiles.derived["roa"], profiles.profiles["te(keV)"], self.plasma["roa"]
    )
    self.plasma["aLte"] = aLy_coarse[:-1, 1]

    # ----------------------------------------------------------------------------------------------------
    # Ti
    # ----------------------------------------------------------------------------------------------------
    (
        aLy_coarse,
        self.deparametrizers["ti"],
        self.deparametrizers_coarse["ti"],
        self.deparametrizers_coarse_middle["ti"],
    ) = PARAMtools.performCurveRegression(
        profiles.derived["roa"], profiles.profiles["ti(keV)"][:, 0], self.plasma["roa"]
    )
    self.plasma["aLti"] = aLy_coarse[:-1, 1]

    # ----------------------------------------------------------------------------------------------------
    # ne
    # ----------------------------------------------------------------------------------------------------
    (
        aLy_coarse,
        self.deparametrizers["ne"],
        self.deparametrizers_coarse["ne"],
        self.deparametrizers_coarse_middle["ne"],
    ) = PARAMtools.performCurveRegression(
        profiles.derived["roa"], profiles.profiles["ne(10^19/m^3)"], self.plasma["roa"]
    )
    self.plasma["aLne"] = aLy_coarse[:-1, 1]

    # ----------------------------------------------------------------------------------------------------
    # nZ
    # ----------------------------------------------------------------------------------------------------
    (
        aLy_coarse,
        self.deparametrizers["nZ"],
        self.deparametrizers_coarse["nZ"],
        self.deparametrizers_coarse_middle["nZ"],
    ) = PARAMtools.performCurveRegression(
        profiles.derived["roa"],
        profiles.profiles["ni(10^19/m^3)"][:, self.impurityPosition - 1],
        self.plasma["roa"],
    )
    self.plasma["aLnZ"] = aLy_coarse[:-1, 1]

    # ----------------------------------------------------------------------------------------------------
    # w0 (SPECIAL, it's not a/Lw0 but -dw0/dr, so some things change here)
    # ----------------------------------------------------------------------------------------------------

    aLT_use, factor_mult = False, factorMult_w0(self)

    (
        aLy_coarse,
        self.deparametrizers["w0"],
        self.deparametrizers_coarse["w0"],
        self.deparametrizers_coarse_middle["w0"],
    ) = PARAMtools.performCurveRegression(
        profiles.derived["roa"],
        profiles.profiles["w0(rad/s)"] * factor_mult,
        self.plasma["roa"],
        aLT=aLT_use,
    )
    self.plasma["aLw0"] = aLy_coarse[:-1, 1]


def factorMult_w0(self):
    """
    w0 is a variable for which the normalized gradient a/Lw0 is ill defined. Consequently, the variable that is truly used as a free parameter
    is dw0/dr. However, my routines are good for dealing with normalized x-coordinate, i.e. dw0/d(r/a)=a*dw0/dr.
    Therefore, right before parametrizing and deparametrizing I divide by a.
    I also divide by 1E5 because rad/s/m tends to be too high, to krad/s/cm may be closer to unity.
    """

    return 1e-5 / self.plasma["a"]
