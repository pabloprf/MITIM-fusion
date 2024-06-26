import copy
import torch
import numpy as np
from mitim_modules.powertorch.aux import PARAMtools
from mitim_modules.powertorch.physics import TARGETStools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpFunction
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

verbose_level = read_verbose_level()

def fromPowerToGacode(
    self,
    profiles_base,
    options={},
    position_in_powerstate_batch=0,
    insert_highres_powers=True,
    rederive=True,
    ):
    """
    Notes:
        - This function assumes that "profiles" is the PROFILES_GACODE that everything started with.
        - We assume that what changes is only the kinetic profiles allowed to vary.
        - This only works for a single profile, in position_in_powerstate_batch
        - rederive is expensive, so I'm not re-deriving the geometry which is the most expensive
    """

    profiles = copy.deepcopy(profiles_base)

    Tfast_ratio = options.get("Tfast_ratio", True)
    Ti_thermals = options.get("Ti_thermals", True)
    ni_thermals = options.get("ni_thermals", True)
    recompute_ptot = options.get("recompute_ptot", True)
    ensureMachNumber = options.get("ensureMachNumber", None)

    # ------------------------------------------------------------------------------------------
    # Insert profiles
    # ------------------------------------------------------------------------------------------

    roa = profiles.profiles["rmin(m)"] / profiles.profiles["rmin(m)"][-1]

    quantities = [
        ["te", "te(keV)", None],
        ["ti", "ti(keV)", 0],
        ["ne", "ne(10^19/m^3)", None],
        ["nZ", "ni(10^19/m^3)", self.impurityPosition - 1],
        ["w0", "w0(rad/s)", None],
    ]

    for key in quantities:
        print(f"\t- Changing {key[0]}")
        if key[0] in self.ProfilesPredicted:
            x, y = self.deparametrizers[key[0]](
                self.plasma["roa"][position_in_powerstate_batch, :],
                self.plasma[f"aL{key[0]}"][position_in_powerstate_batch, :],
            )
            y_interpolated = interpFunction(roa, x.cpu(), y[0, :].cpu())
            if key[2] is None:
                Y_copy = copy.deepcopy(profiles.profiles[key[1]])
                profiles.profiles[key[1]] = y_interpolated
            else:
                Y_copy = copy.deepcopy(profiles.profiles[key[1]][:, key[2]])
                profiles.profiles[key[1]][:, key[2]] = y_interpolated

            # ------------------------------------------------------------------------------------------
            # Special treatments
            # ------------------------------------------------------------------------------------------

            if key[0] == "te" and Tfast_ratio:
                print(
                    "\t\t* If any fast species, changing Tfast to ensure fixed Tfast/Te ratio",
                    typeMsg="i",
                )
                for sp in range(len(profiles.Species)):
                    if profiles.Species[sp]["S"] == "fast":
                        print(f"\t\t\t- Changing temperature of species #{sp}")
                        profiles.profiles["ti(keV)"][:, sp] = profiles.profiles["ti(keV)"][:, sp] * (
                            profiles.profiles["te(keV)"] / Y_copy
                        )

            if key[0] == "ti" and Ti_thermals:
                print("\t\t* Ensuring Ti is equal for all thermal ions", typeMsg="i")
                profiles.makeAllThermalIonsHaveSameTemp()

            if key[0] == "ne" and ni_thermals:
                scaleFactor = y_interpolated / Y_copy
                print("\t\t* Adjusting ni of thermal ions", typeMsg="i")
                profiles.scaleAllThermalDensities(scaleFactor=scaleFactor)

    if "w0" not in self.ProfilesPredicted and ensureMachNumber is not None:
        # Rotation fixed to ensure Mach number
        profiles.introduceRotationProfile(Mach_LF=ensureMachNumber)

    # ------------------------------------------------------------------------------------------
    # Insert Powers
    # ------------------------------------------------------------------------------------------

    if insert_highres_powers:
        insert_powers_to_gacode(self, profiles, position_in_powerstate_batch)

    # ------------------------------------------------------------------------------------------
    # Recalculate and change ptot to make it consistent?
    # ------------------------------------------------------------------------------------------

    if rederive or recompute_ptot:
        profiles.deriveQuantities(rederiveGeometry=False)

    if recompute_ptot:
        profiles.selfconsistentPTOT()

    return profiles

def insert_powers_to_gacode(self, profiles, position_in_powerstate_batch=0):

    profiles.deriveQuantities(rederiveGeometry=False)

    print("\t- Insering powers")

    state_temp = self.copy_state()

    # ------------------------------------------------------------------------------------------
    # Recalculate powers with powerstate on the gacode-original fine grid
    # ------------------------------------------------------------------------------------------

    # Modify power flows by tricking the powerstate into a fine grid (same as does TGYRO)
    extra_points = 2  # If I don't allow this, it will fail
    rhoy = profiles.profiles["rho(-)"][1:-extra_points]
    with IOtools.HiddenPrints():
        state_temp.__init__(profiles, EvolutionOptions={"rhoPredicted": rhoy})
    state_temp.calculateProfileFunctions()
    state_temp.TargetOptions["ModelOptions"]["TargetCalc"] = "powerstate"
    state_temp.calculateTargets()
    # ------------------------------------------------------------------------------------------

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
                state_temp.plasma[ikey][position_in_powerstate_batch,:].cpu().numpy()
            )
        else:
            profiles.profiles[conversions[ikey]] = np.zeros(
                len(profiles.profiles["qei(MW/m^3)"])
            )
            profiles.profiles[conversions[ikey]][:-extra_points] = (
                state_temp.plasma[ikey][position_in_powerstate_batch,:].cpu().numpy()
            )

def fromGacodeToPower(self, input_gacode, rho_vec):
    """
    This function converts from the fine input.gacode grid to a powertorch object and grid.
    Notes:
        - rho_vec is rho, but I will work from now on with roa for easy intergrations
        - Flows:
            - They are in the TGYRO definition, with qione summed, giving the intented flows)
            - Same as in TGYRO: The interpolation of those quantities that do not vary throughout the workflow
                            needs to happen AFTER integration (otherwise it would be too coarse to calculate auxiliary deposition)
            - Because TGYRO doesn't care about convective fluxes, I need to convert it AFTER I have the ge_Miller integrated and interpolated, so this happens
                            at each powerstate evaluation
            - Momentum flux is J/m^2. Momentum source is given in input.gacode a N/m^2 or J/m^3. Integrated in volume is J or N*m
        - PextraE_Target2 and so on are to also include radiation, alpha and exchange, in case I want to run powerstate with fixed those
    """

    print("\t- Producing powerstate")

    # *********************************************************************************************
    # Radial grid
    # *********************************************************************************************

    rho_use = input_gacode.profiles["rho(-)"]

    roa_array = interpFunction(rho_vec.cpu(), rho_use, input_gacode.derived["roa"])
    rho_array = interpFunction(rho_vec.cpu(), rho_use, input_gacode.profiles["rho(-)"])
    
    if len(rho_array) < 10:
        print(f"\t\t@ rho = {[round(i,6) for i in rho_array]}")
        print(f"\t\t@ r/a = {[round(i,6) for i in roa_array]}")
    else:
        print(f"\t\t@ {len(rho_array)} rho points")

    # In case rho_vec is different than rho_vec_evaluate (DEPCRECATED, but I keep it for now)
    self.indexes_simulation = []
    for rho in rho_array:   self.indexes_simulation.append(np.argmin(np.abs(rho_array - rho)).item())

    # *********************************************************************************************
    # Quantities to interpolate and convert
    # *********************************************************************************************

    self.plasma["roa"] = torch.from_numpy(roa_array).to(rho_vec)
    self.plasma["rho"] = torch.from_numpy(rho_array).to(rho_vec)
    self.plasma["mi_u"] = torch.tensor(input_gacode.mi_first).to(rho_vec)
    self.plasma["a"] = torch.tensor(input_gacode.derived["a"]).to(rho_vec)
    self.plasma["eps"] = torch.tensor(input_gacode.derived["eps"]).to(rho_vec)

    quantities_to_interpolate = [
        ["rho", "rho(-)", None, True, False],
        ["roa", "roa", None, True, True],
        ["volp", "volp_miller", None, True, True],
        ["rmin", "rmin(m)", None, True, False],
        ["te", "te(keV)", None, True, False],
        ["ti", "ti(keV)", 0, True, False],
        ["ne", "ne(10^19/m^3)", None, True, False],
        ["nZ", "ni(10^19/m^3)", self.impurityPosition - 1, True, False],
        ["w0", "w0(rad/s)", None, True, False],
        ["B_unit", "B_unit", None, True, True],
        ["B_ref", "B_ref", None, True, True],
    ]

    for key in quantities_to_interpolate:
        quant = input_gacode.derived[key[1]] if key[4] else input_gacode.profiles[key[1]]
        self.plasma[key[0]] = torch.from_numpy(
            interpFunction(rho_vec.cpu(), rho_use, quant if key[2] is None else quant[:, key[2]]) if key[3] else quant
            ).to(rho_vec)

    quantities_to_interpolate_and_volp = [
        ["Paux_e", "qe_aux_MWmiller"],
        ["Paux_i", "qi_aux_MWmiller"],
        ["Gaux_e", "ge_10E20miller"],
        ["Maux", "mt_Jmiller"],
    ]

    for key in quantities_to_interpolate_and_volp:
        self.plasma[key[0]] = torch.from_numpy(
            interpFunction(rho_vec.cpu(), rho_use, input_gacode.derived[key[1]])
        ).to(rho_vec) / self.plasma["volp"]

    self.plasma["Gaux_Z"] = self.plasma["Gaux_e"] * 0.0

    quantitites = {}
    quantitites["PextraE_Target2"] = input_gacode.derived["qe_fus_MWmiller"] - input_gacode.derived["qrad_MWmiller"]
    quantitites["PextraI_Target2"] = input_gacode.derived["qi_fus_MWmiller"]
    quantitites["PextraE_Target1"] = quantitites["PextraE_Target2"] - input_gacode.derived["qe_exc_MWmiller"]
    quantitites["PextraI_Target1"] = quantitites["PextraI_Target2"] + input_gacode.derived["qe_exc_MWmiller"]

    for key in quantitites:
        self.plasma[key] = torch.from_numpy(
            interpFunction(rho_vec.cpu(), rho_use, quantitites[key])
        ).to(rho_vec)

    # *********************************************************************************************
    # Ion species need special treatment
    # *********************************************************************************************

    defineIons(self, input_gacode, rho_vec, rho_vec)

    # *********************************************************************************************
	# Define deparametrizer functions for the varying profiles and gradients from here
    # *********************************************************************************************

    (
        self.deparametrizers,
        self.deparametrizers_coarse,
        self.deparametrizers_coarse_middle,
    ) = ({}, {}, {})

    aLT_use_w0, factor_mult_w0 = False, factorMult_w0(self)

    cases_to_parameterize = [
        ["te", "te(keV)", None, 1.0, True],
        ["ti", "ti(keV)", 0, 1.0, True],
        ["ne", "ne(10^19/m^3)", None, 1.0, True],
        ["nZ", "ni(10^19/m^3)", self.impurityPosition - 1, 1.0, True],
        ["w0", "w0(rad/s)", None, factor_mult_w0.item(), aLT_use_w0], 
    ]

    for key in cases_to_parameterize:
        quant = input_gacode.profiles[key[1]] if key[2] is None else input_gacode.profiles[key[1]][:, key[2]]
        (
            aLy_coarse,
            self.deparametrizers[key[0]],
            self.deparametrizers_coarse[key[0]],
            self.deparametrizers_coarse_middle[key[0]],
        ) = PARAMtools.performCurveRegression(
            input_gacode.derived["roa"],
            quant * key[3],
            self.plasma["roa"],
            aLT=key[4],
        )
        self.plasma[f"aL{key[0]}"] = aLy_coarse[:-1, 1]

        # Check that it's not completely zero
        if key[0] in self.ProfilesPredicted:
            if self.plasma[f"aL{key[0]}"].sum() == 0.0:
                addT = 1e-15
                print(
                    f"\t- All values of {key[0]} detected to be zero, to avoid NaNs, inserting {addT} at the edge",
                    typeMsg="w",
                )
                self.plasma[f"aL{key[0]}"][..., -1] += addT

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
    Dion = torch.tensor(input_gacode.Dion) if input_gacode.Dion is not None else torch.tensor(np.nan)
    Tion = torch.tensor(input_gacode.Tion) if input_gacode.Tion is not None else torch.tensor(np.nan)

    # Only store as part of ions_set those IMMUTABLE parameters (NOTE: ni MAY change so that's why it's not here)
    self.plasma["ions_set_mi"] = mi
    self.plasma["ions_set_Zi"] = Zi
    self.plasma["ions_set_Dion"] = Dion
    self.plasma["ions_set_Tion"] = Tion
    self.plasma["ions_set_c_rad"] = c_rad

def factorMult_w0(self):
    """
    w0 is a variable for which the normalized gradient a/Lw0 is ill defined. Consequently, the variable that is truly used as a free parameter
    is dw0/dr. However, my routines are good for dealing with normalized x-coordinate, i.e. dw0/d(r/a)=a*dw0/dr.
    Therefore, right before parametrizing and deparametrizing I divide by a.
    I also divide by 1E5 because rad/s/m tends to be too high, to krad/s/cm may be closer to unity.
    """

    return 1e-5 / self.plasma["a"]
