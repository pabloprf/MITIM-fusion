import copy
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.misc_tools import LOGtools, IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch.physics import TARGETStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

def gacode_to_powerstate(self, increase_profile_resol=False):
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
        - Pe_orig_fusrad and so on are to also include radiation, alpha and exchange, in case I want to run powerstate with fixed those
    """

    print("\t- Producing powerstate object from input.gacode")

    input_gacode    = self.profiles
    rho_vec         = self.plasma["rho"]

    # Resolution of input.gacode
    if increase_profile_resol:
        improve_resolution_profiles(input_gacode, rho_vec)

    # *********************************************************************************************
    # Radial grid
    # *********************************************************************************************

    rho_use = input_gacode.profiles["rho(-)"]

    roa_array = interpolation_function(rho_vec.cpu(), rho_use, input_gacode.derived["roa"])
    rho_array = interpolation_function(rho_vec.cpu(), rho_use, input_gacode.profiles["rho(-)"])
    
    if len(rho_array) < 10:
        print(f"\t\t@ rho = {[round(i,6) for i in rho_array]}")
        print(f"\t\t@ r/a = {[round(i,6) for i in roa_array]}")
    else:
        print(f"\t\t@ {len(rho_array)} rho points")

    # In case rho_vec is different than rho_vec_evaluate
    self.indexes_simulation = []
    for rho in rho_array:
        self.indexes_simulation.append(np.argmin(np.abs(rho_array - rho)).item())

    # *********************************************************************************************
    # Quantities to interpolate and convert
    # *********************************************************************************************

    self.plasma["roa"] = torch.from_numpy(roa_array).to(rho_vec)
    self.plasma["rho"] = torch.from_numpy(rho_array).to(rho_vec)
    self.plasma["mi_u"] = torch.tensor(input_gacode.mi_first).to(rho_vec)
    self.plasma["a"] = torch.tensor(input_gacode.derived["a"]).to(rho_vec)
    self.plasma["eps"] = torch.tensor(input_gacode.derived["eps"]).to(rho_vec)

    # Add more stuff: name, original name, index, unit conversion, is derived
    quantities_to_interpolate = [
        ["rho", "rho(-)", None, True, False],
        ["roa", "roa", None, True, True],
        ["Rmajoa", "Rmajoa", None, True, True],
        ["volp", "volp_miller", None, True, True],
        ["rmin", "rmin(m)", None, True, False],
        ["te", "te(keV)", None, True, False],
        ["ti", "ti(keV)", 0, True, False],
        ["ne", "ne(10^19/m^3)", None, True, False],
        ["nZ", "ni(10^19/m^3)", self.impurityPosition, True, False],
        ["w0", "w0(rad/s)", None, True, False],
    ]

    # Quantities that do not necessarily need to be used in this powerstate call
    additional_quantities_for_potential_use = [
        ["B_unit", "B_unit", None, True, True],
        ["B_ref", "B_ref", None, True, True],
        ["q", "q(-)", None, True, False],
        ["s_q", "s_q", None, True, True],
        ["aLte", "aLTe", None, True, True],
        ["aLti", "aLTi", 0, True, True],
        ["aLne", "aLne", None, True, True],
        ["aLni", "aLni", 0, True, True],
        ["Zeff", "Zeff", None, True, True],
        ["kappa", "kappa(-)", None, True, False],
        ["delta", "delta(-)", None, True, False],
        ["betae", "betae", None, True, True],
    ]
    # ---------------------------------------------------------------------------

    for quant in additional_quantities_for_potential_use:
        quantities_to_interpolate.append(quant)

    for key in quantities_to_interpolate:
        quant = input_gacode.derived[key[1]] if key[4] else input_gacode.profiles[key[1]]

        # *********************************************************************************************
        # Extract the quantity via interpolation and tensorization
        # *********************************************************************************************
        self.plasma[key[0]] = torch.from_numpy(
            interpolation_function(rho_vec.cpu(), rho_use, quant if key[2] is None else quant[:, key[2]]) if key[3] else quant
            ).to(rho_vec)
        # *********************************************************************************************

    quantities_to_interpolate_and_volp = [
        ["Paux_e", "qe_aux_MWmiller"],
        ["Paux_i", "qi_aux_MWmiller"],
        ["Gaux_e", "ge_10E20miller"],
        ["Maux", "mt_Jmiller"],
    ]

    for key in quantities_to_interpolate_and_volp:

        # *********************************************************************************************
        # Extract the quantity via interpolation and tensorization
        # *********************************************************************************************
        self.plasma[key[0]] = torch.from_numpy(
            interpolation_function(rho_vec.cpu(), rho_use, input_gacode.derived[key[1]])
        ).to(rho_vec) / self.plasma["volp"]
        # *********************************************************************************************

    self.plasma["Gaux_Z"] = self.plasma["Gaux_e"] * 0.0

    quantitites = {}
    quantitites["Pe_orig_fusrad"] = input_gacode.derived["qe_fus_MWmiller"] - input_gacode.derived["qrad_MWmiller"]
    quantitites["Pi_orig_fusrad"] = input_gacode.derived["qi_fus_MWmiller"]
    quantitites["Pe_orig_fusradexch"] = quantitites["Pe_orig_fusrad"] - input_gacode.derived["qe_exc_MWmiller"]
    quantitites["Pi_orig_fusradexch"] = quantitites["Pi_orig_fusrad"] + input_gacode.derived["qe_exc_MWmiller"]

    for key in quantitites:
        
        # *********************************************************************************************
        # Extract the quantity via interpolation and tensorization
        # *********************************************************************************************
        self.plasma[key] = torch.from_numpy(
            interpolation_function(rho_vec.cpu(), rho_use, quantitites[key])
        ).to(rho_vec) / self.plasma["volp"]
        # *********************************************************************************************

    # *********************************************************************************************
    # Ion species need special treatment
    # *********************************************************************************************

    defineIons(self, input_gacode, rho_vec, rho_vec)

    # *********************************************************************************************
    # Treatment of rotation gradient
    # *********************************************************************************************
    """
    w0 is a variable for which the normalized gradient a/Lw0 is ill defined. Consequently, the variable that is truly used as a free parameter
    is dw0/dr. However, my routines are good for dealing with normalized x-coordinate, i.e. dw0/d(r/a)=a*dw0/dr.
    Therefore, right before parametrizing and deparametrizing I divide by a.
    I also divide by 1E5 because rad/s/m tends to be too high, to krad/s/cm may be closer to unity.
    """

    self.plasma["kradcm"] = 1e-5 / self.plasma["a"]

    # *********************************************************************************************
	# Define profile_constructor functions for the varying profiles and gradients from here
    # *********************************************************************************************

    # [quantiy in powerstate, quantity in input.gacode, index of the ion, multiplier, parameterize_in_aLx]
    cases_to_parameterize = [
        ["te", "te(keV)", None, 1.0, True],
        ["ti", "ti(keV)", 0, 1.0, True],
        ["ne", "ne(10^19/m^3)", None, 1.0, True],
        ["nZ", "ni(10^19/m^3)", self.impurityPosition, 1.0, True],
        ["w0", "w0(rad/s)", None, self.plasma["kradcm"], False], 
    ]

    # Add all ions
    for i in range(input_gacode.profiles['ni(10^19/m^3)'].shape[1]):
        cases_to_parameterize.append([f"ni{i}", "ni(10^19/m^3)", i, 1.0, True])

    self.deparametrizers_fine, self.deparametrizers_coarse, self.deparametrizers_coarse_middle = {}, {}, {}
    for key in cases_to_parameterize:
        quant = input_gacode.profiles[key[1]] if key[2] is None else input_gacode.profiles[key[1]][:, key[2]]

        (
            aLy_coarse,
            self.deparametrizers_fine[key[0]],
            self.deparametrizers_coarse[key[0]],
            self.deparametrizers_coarse_middle[key[0]],
        ) = parameterize_curve(
            input_gacode.derived["roa"],
            quant,
            self.plasma["roa"],
            parameterize_in_aLx=key[4],
            multiplier_quantity=key[3],
        )

        self.plasma[f"aL{key[0]}"] = aLy_coarse[:-1, 1]

        # Check that it's not completely zero
        if key[0] in self.ProfilesPredicted:
            if self.plasma[f"aL{key[0]}"].sum() == 0.0:
                addT = 1e-15
                print(f"\t- All values of {key[0]} detected to be zero, to avoid NaNs, inserting {addT} at the edge",typeMsg="w")
                self.plasma[f"aL{key[0]}"][..., -1] += addT

    def to_gacode(
        self,
        write_input_gacode=None,
        position_in_powerstate_batch=0,
        postprocess_input_gacode={},
        insert_highres_powers=False,
        rederive_profiles=True,
    ):
        '''
        Notes:
            - insert_highres_powers: whether to insert high resolution powers (will calculate them with powerstate targets object, not other custom ones)
        '''
        print(">> Inserting powerstate into input.gacode")

        profiles = powerstate_to_gacode(
            self,
            position_in_powerstate_batch=position_in_powerstate_batch,
            postprocess_input_gacode=postprocess_input_gacode,
            insert_highres_powers=insert_highres_powers,
            rederive=rederive_profiles,
        )

        # Write input.gacode
        if write_input_gacode is not None:
            write_input_gacode = Path(write_input_gacode)
            print(f"\t- Writing input.gacode file: {IOtools.clipstr(write_input_gacode)}")
            write_input_gacode.parent.mkdir(parents=True, exist_ok=True)
            profiles.writeCurrentStatus(file=write_input_gacode)

        # If corrections modify the ions set... it's better to re-read, otherwise powerstate will be confused
        if rederive_profiles:
            defineIons(self, profiles, self.plasma["rho"][position_in_powerstate_batch, :], self.dfT)
            # Repeat, that's how it's done earlier
            self._repeat_tensors(batch_size=self.plasma["rho"].shape[0],
                specific_keys=["ni","ions_set_mi","ions_set_Zi","ions_set_Dion","ions_set_Tion","ions_set_c_rad"],
                positionToUnrepeat=None)

        return profiles

def powerstate_to_gacode(
    self,
    postprocess_input_gacode={},
    position_in_powerstate_batch=0,
    insert_highres_powers=True,
    rederive=True,
    debugPlot=False,
    ):
    """
    Notes:
        - This function assumes that "profiles" is the PROFILES_GACODE that everything started with.
        - We assume that what changes is only the kinetic profiles allowed to vary.
        - This only works for a single profile, in position_in_powerstate_batch
        - rederive is expensive, so I'm not re-deriving the geometry which is the most expensive
    """

    # Default options for postprocessing
    
    Tfast_ratio = postprocess_input_gacode.get("Tfast_ratio", True)
    Ti_thermals = postprocess_input_gacode.get("Ti_thermals", True)
    ni_thermals = postprocess_input_gacode.get("ni_thermals", True)
    recompute_ptot = postprocess_input_gacode.get("recompute_ptot", True)
    ensureMachNumber = postprocess_input_gacode.get("ensureMachNumber", None)

    # ------------------------------------------------------------------------------------------
    # Insert profiles
    # ------------------------------------------------------------------------------------------

    # Start from the originally stored (at initialization) in this powerstate object
    profiles = copy.deepcopy(self.profiles)

    quantities = [
        ["te", "te(keV)", None],
        ["ti", "ti(keV)", 0],
        ["ne", "ne(10^19/m^3)", None],
        ["nZ", "ni(10^19/m^3)", self.impurityPosition],
        ["w0", "w0(rad/s)", None],
    ]

    for key in quantities:
        if key[0] in self.ProfilesPredicted:
            print(f"\t- Inserting {key[0]} into input.gacode profiles")

            # *********************************************************************************************
            # From a/Lx to x via fine deparametrizer
            # *********************************************************************************************
            x, y = self.deparametrizers_fine[key[0]](
                self.plasma["roa"][position_in_powerstate_batch, :],
                self.plasma[f"aL{key[0]}"][position_in_powerstate_batch, :],
            )
            # *********************************************************************************************

            y_new = y[0, :].cpu().numpy()
                
            if key[2] is None:
                Y_copy = copy.deepcopy(profiles.profiles[key[1]])
                profiles.profiles[key[1]] = y_new
            else:
                Y_copy = copy.deepcopy(profiles.profiles[key[1]][:, key[2]])
                profiles.profiles[key[1]][:, key[2]] = y_new

            # ------------------------------------------------------------------------------------------
            # Special treatments
            # ------------------------------------------------------------------------------------------

            if key[0] == "te" and Tfast_ratio:
                print("\t\t* If any fast species, changing Tfast to ensure fixed Tfast/Te ratio",typeMsg="i",)
                for sp in range(len(profiles.Species)):
                    if profiles.Species[sp]["S"] == "fast":
                        print(f"\t\t\t- Modifying temperature of species #{sp}")
                        profiles.profiles["ti(keV)"][:, sp] = profiles.profiles["ti(keV)"][:, sp] * (profiles.profiles["te(keV)"] / Y_copy)

            if key[0] == "ti" and Ti_thermals:
                print("\t\t* Ensuring Ti is equal for all thermal ions", typeMsg="i")
                profiles.makeAllThermalIonsHaveSameTemp()

            if key[0] == "ne" and ni_thermals:
                scaleFactor = y_new / Y_copy
                print("\t\t* Adjusting ni of thermal ions", typeMsg="i")
                profiles.scaleAllThermalDensities(scaleFactor=scaleFactor)

    if "w0" not in self.ProfilesPredicted and ensureMachNumber is not None:
        # Rotation fixed to ensure Mach number
        profiles.introduceRotationProfile(Mach_LF=ensureMachNumber)

    # ------------------------------------------------------------------------------------------
    # Insert Powers
    # ------------------------------------------------------------------------------------------

    if insert_highres_powers:
        powerstate_to_gacode_powers(self, profiles, position_in_powerstate_batch)

    # ------------------------------------------------------------------------------------------
    # Recalculate and change ptot to make it consistent?
    # ------------------------------------------------------------------------------------------

    if rederive or recompute_ptot:
        profiles.deriveQuantities(rederiveGeometry=False)

    if recompute_ptot:
        profiles.selfconsistentPTOT()

    if debugPlot:
        debug_transformation(self.profiles,profiles,self)

    return profiles

def powerstate_to_gacode_powers(self, profiles, position_in_powerstate_batch=0):

    profiles.deriveQuantities(rederiveGeometry=False)

    print("\t- Insering powers")

    state_temp = self.copy_state()

    # ------------------------------------------------------------------------------------------
    # Recalculate powers with powerstate on the gacode-original fine grid
    # ------------------------------------------------------------------------------------------

    # Modify power flows by tricking the powerstate into a fine grid (same as does TGYRO)
    extra_points = 2  # If I don't allow this, it will fail
    rhoy = profiles.profiles["rho(-)"][1:-extra_points]
    with LOGtools.HiddenPrints():
        state_temp.__init__(
            profiles,
            EvolutionOptions={"rhoPredicted": rhoy},
            TargetOptions={
                "targets_evaluator": TARGETStools.analytical_model,
                "ModelOptions": {
                    "TypeTarget": self.TargetOptions["ModelOptions"]["TypeTarget"], # Important to keep the same as in the original
                    "TargetCalc": "powerstate",
                    }
                },
            increase_profile_resol = False
            )
    state_temp.calculateProfileFunctions()
    state_temp.TargetOptions["ModelOptions"]["TargetCalc"] = "powerstate"
    state_temp.calculateTargets()
    # ------------------------------------------------------------------------------------------

    conversions = {}

    if self.TargetOptions["ModelOptions"]["TypeTarget"] > 1:
        conversions['qie'] = "qei(MW/m^3)"
    if self.TargetOptions["ModelOptions"]["TypeTarget"] > 2:
        conversions['qrad_bremms'] = "qbrem(MW/m^3)"
        conversions['qrad_sync'] = "qsync(MW/m^3)"
        conversions['qrad_line'] = "qline(MW/m^3)"
        conversions['qfuse'] = "qfuse(MW/m^3)"
        conversions['qfusi'] = "qfusi(MW/m^3)"

    for ikey in conversions:
        if conversions[ikey] in profiles.profiles:
            profiles.profiles[conversions[ikey]][:-extra_points] = state_temp.plasma[ikey][position_in_powerstate_batch,:].cpu().numpy()
        else:
            profiles.profiles[conversions[ikey]] = np.zeros(len(profiles.profiles["qei(MW/m^3)"]))
            profiles.profiles[conversions[ikey]][:-extra_points] = state_temp.plasma[ikey][position_in_powerstate_batch,:].cpu().numpy()

    
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
    rho_vec = rho_vec.clone().cpu()

    # ** Store the information about the thermal ions, including the cooling coefficients
    self.plasma["ni"], mi, Zi, c_rad = [], [], [], []
    for i in range(len(input_gacode.profiles["mass"])):
        if input_gacode.profiles["type"][i] == "[therm]":
            self.plasma["ni"].append(
                interpolation_function(rho_vec, rho_use, input_gacode.profiles["ni(10^19/m^3)"][:, i])
            )
            mi.append(input_gacode.profiles["mass"][i])
            Zi.append(input_gacode.profiles["z"][i])

            # Grab chebyshev coefficients from file
            data_df = pd.read_csv(__mitimroot__ / "src" / "mitim_modules" / "powertorch" / "physics" / "radiation_chebyshev.csv")
            try:
                c = data_df[data_df['Ion'].str.lower()==input_gacode.profiles["name"][i].lower()].to_numpy()[0,2:].astype(float)
            except IndexError:
                print(f'\t- Specie {input_gacode.profiles["name"][i]} not found in ADAS database, assuming zero radiation from it',typeMsg="w")
                c = [-1e10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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

def parameterize_curve(
    x_coord,
    y_coord_raw,
    x_coarse_tensor,
    parameterize_in_aLx=True,
    multiplier_quantity=1.0,
    ):
    """
    Notes:
        - x_coarse_tensor must be torch
    """

    # **********************************************************************************************************
    # Define the integrator and derivator functions (based on whether I want to parameterize in aLx or in gradX)
    # **********************************************************************************************************

    if parameterize_in_aLx:
        # 1/Lx = -1/X*dX/dr
        integrator_function, derivator_function = (
            CALCtools.integrateGradient,
            CALCtools.produceGradient,
        )
    else:
        # -dX/dr
        integrator_function, derivator_function = (
            CALCtools.integrateGradient_lin,
            CALCtools.produceGradient_lin,
        )

    y_coord = torch.from_numpy(y_coord_raw).to(x_coarse_tensor) * multiplier_quantity

    ygrad_coord = derivator_function( torch.from_numpy(x_coord).to(x_coarse_tensor), y_coord )

    # **********************************************************************************************************
    # Get control points
    # **********************************************************************************************************

    x_coarse = x_coarse_tensor[1:].cpu().numpy()

    """
    Define region to get control points from
    ------------------------------------------------------------
	Trick: Addition of extra point
		This is important because if I don't, when I combine the trailing edge and the new
		modified profile, there's going to be a discontinuity in the gradient.
	"""
    
    ir_end = np.argmin(np.abs(x_coord - x_coarse[-1]))

    if ir_end < len(x_coord) - 1:
        ir = ir_end + 2  # To prevent that TGYRO does a 2nd order derivative
        x_coarse = np.append(x_coarse, [x_coord[ir]])
    else:
        ir = ir_end

	# Definition of trailing edge. Any point after, and including, the extra point
    x_trail = torch.from_numpy(x_coord[ir:]).to(x_coarse_tensor)
    y_trail = y_coord[ir:]
    x_notrail = torch.from_numpy(x_coord[: ir + 1]).to(x_coarse_tensor)

    # Produce control points, including a zero at the beginning
    aLy_coarse = [[0.0, 0.0]]
    for cont, i in enumerate(x_coarse):
        yValue = ygrad_coord[np.argmin(np.abs(x_coord - i))]
        aLy_coarse.append([i, yValue.cpu().item()])

    aLy_coarse = torch.from_numpy(np.array(aLy_coarse)).to(ygrad_coord)

    # Since the last one is an extra point very close, I'm making it the same
    aLy_coarse[-1, 1] = aLy_coarse[-2, 1]

    # Boundary condition at point moved by gridPointsAllowed
    y_bc = torch.from_numpy(interpolation_function([x_coarse[-1]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # Boundary condition at point (ACTUAL THAT I WANT to keep fixed, i.e. rho=0.8)
    y_bc_real = torch.from_numpy(interpolation_function([x_coarse[-2]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # **********************************************************************************************************
    # Define deparametrizer functions
    # **********************************************************************************************************

    def deparametrizer_coarse(x, y, multiplier=multiplier_quantity):
        """
        Construct curve in a coarse grid
        ----------------------------------------------------------------------------------------------------
        This constructs a curve in any grid, with any batch given in y=y.
        Useful for surrogate evaluations. Fast in a coarse grid. For HF evaluations,
        I need to do in a finer grid so that it is consistent with TGYRO.
        x, y must be (batch, radii),	y_bc must be (1)
        """
        return x, integrator_function(x, y, y_bc_real) / multiplier

    def deparametrizer_coarse_middle(x, y, multiplier=multiplier_quantity):
        """
        Deparamterizes a finer profile based on the values in the coarse.
        Reason why something like this is not used for the full profile is because derivative of this will not be as original,
                which is needed to match TGYRO
        """
        yCPs = CALCtools.Interp1d()(aLy_coarse[:, 0][:-1].repeat((y.shape[0], 1)), y, x)
        return x, integrator_function(x, yCPs, y_bc_real) / multiplier

    def deparametrizer_fine(x, y, multiplier=multiplier_quantity):
        """
        Notes:
            - x is a 1D array, but y can be a 2D array for a batch of individuals: (batch,x)
            - I am assuming it is 1/LT for parameterization, but gives T
        """

        y = torch.atleast_2d(y)
        x = x[0, :] if x.dim() == 2 else x

        # Add the extra trick point
        x = torch.cat((x, aLy_coarse[-1][0].repeat((1))))
        y = torch.cat((y, aLy_coarse[-1][-1].repeat((y.shape[0], 1))), dim=1)

        # Model curve (basically, what happens in between points)
        yBS = CALCtools.Interp1d()(x.repeat(y.shape[0], 1), y, x_notrail.repeat(y.shape[0], 1))

        """
        ---------------------------------------------------------------------------------------------------------
            Trick 1: smoothAroundCoarsing
                TGYRO will use a 2nd order scheme to obtain gradients out of the profile, so a piecewise linear
                will simply not give the right derivatives.
                Here, this rough trick is to modify the points in gradient space around the coarse grid with the
                same value of gradient, so in principle it doesn't matter the order of the derivative.
        """
        num_around = 1
        for i in range(x.shape[0] - 2):
            ir = torch.argmin(torch.abs(x[i + 1] - x_notrail))
            for k in range(-num_around, num_around + 1, 1):
                yBS[:, ir + k] = yBS[:, ir]
        # --------------------------------------------------------------------------------------------------------

        yBS = integrator_function(x_notrail.repeat(yBS.shape[0], 1), yBS.clone(), y_bc)

        """
        Trick 2: Correct y_bc
            The y_bc for the profile integration started at gridPointsAllowed, but that's not the real
            y_bc. I want the temperature fixed at my first point that I actually care for.
            Here, I multiply the profile to get that.
            Multiplication works because:
                1/LT = 1/T * dT/dr
                1/LT' = 1/(T*m) * d(T*m)/dr = 1/T * dT/dr = 1/LT
            Same logarithmic gradient, but with the right boundary condition

        """
        ir = torch.argmin(torch.abs(x_notrail - x[-2]))
        yBS = yBS * torch.transpose((y_bc_real / yBS[:, ir]).repeat(yBS.shape[1], 1), 0, 1)

        # Add trailing edge
        y_trailnew = copy.deepcopy(y_trail).repeat(yBS.shape[0], 1)

        x_notrail_t = torch.cat((x_notrail[:-1], x_trail), dim=0)
        yBS = torch.cat((yBS[:, :-1], y_trailnew), dim=1)

        return x_notrail_t, yBS / multiplier

    # **********************************************************************************************************

    return (
        aLy_coarse,
        deparametrizer_fine,
        deparametrizer_coarse,
        deparametrizer_coarse_middle,
    )

def improve_resolution_profiles(profiles, rhoMODEL):
    """
    Resolution of input.gacode
    **************************
    - Change resolution to a fine grid in which doing the flattening around coarse points has a small effect on the profile.
    - It is recommended that it goes through the points, with more points around the trailing edge transition.
    - Also, avoid adding too points near axis. (NOT NOW?)
    """

    # ----------------------------------------
    # Parameters
    # ----------------------------------------

    total_points = 100
    d_spacing_coarse = 1e-3
    points_updown = 2

    # ----------------------------------------------------------------------------------
    # 1. Fill up spaces in between the points until the total is total_points
    # ----------------------------------------------------------------------------------

    num_points_rest = int(np.max([3, total_points / (len(rhoMODEL) + 1)]))

    # Correction: If I do a very fine grid, but with a first point away from 0.0, it'll	have a piecewise behavior from 0 to the first points, so ensure a few more
    num_points_0 = int(np.max([num_points_rest, 10]))
    # *******

    rho_new0 = np.append(np.append([0], rhoMODEL), [1])
    rho_new = np.array([])
    for i in range(rho_new0.shape[0] - 1):
        num_points = num_points_rest if i > 0 else num_points_0
        rho_new = np.append(
            rho_new, np.linspace(rho_new0[i], rho_new0[i + 1], num_points)
        )

    # ----------------------------------------------------------------------------------
    # 2. Add extra resolution around the modelled (e.g. TGYRO) points
    # ----------------------------------------------------------------------------------

    for i in range(points_updown):
        rho_new = np.append(
            np.append(rho_new, rhoMODEL + d_spacing_coarse * (i + 1)),
            rhoMODEL - d_spacing_coarse * (i + 1),
        )

    # ----------------------------------------------------------------------------------
    # Change resolution
    # ----------------------------------------------------------------------------------
    profiles.changeResolution(rho_new=rho_new)


def debug_transformation(p, p_new, s):

    rho = s.plasma['rho'][0][1:]

    vars_compare = {
        'aLTe': 'aLte',
        'aLTi': 'aLti',
        'aLne': 'aLne',
        'te(keV)': 'te',
        'ti(keV)': 'ti',
        'ne(10^19/m^3)': 'ne'
    }

    # Error in gradients
    err_grad = []
    err_prof = []
    for var in vars_compare:
        txt = '\t\t'
        for ix_p,r in enumerate(torch.cat((torch.zeros(1),rho))):
            i = np.argmin(np.abs(p.profiles['rho(-)'] - r.item()))
            ix_new = np.argmin(np.abs(p_new.profiles['rho(-)'] - r.item()))

            if var in ['aLTe', 'aLTi', 'aLne']:
                dict_old = p.derived
                dict_new = p_new.derived
            else:
                dict_old = p.profiles
                dict_new = p_new.profiles

            var_orig = dict_old[var][i].item() if var not in ['aLTi', 'ti(keV)'] else dict_old[var][i,0].item()
            var_new = dict_new[var][ix_new].item() if var not in ['aLTi', 'ti(keV)'] else dict_new[var][ix_new,0].item()
            
            var_power = s.plasma[vars_compare[var]][0,ix_p].item()

            # error = abs ( (var_orig - var_new) / var_orig) * 100.0
            error = abs ( (var_power - var_new) / var_power) * 100.0 if var_power != 0.0 else torch.inf

            txt += f'{ error:.1e}%  '

            # Only append when it's not the zero gradient
            if var in ['aLTe', 'aLTi', 'aLne']:
                if r.item() > 0.0:
                    err_grad.append(error)
            else:
                err_prof.append(error)

        print(f'\t{var} error between powerstate and generated input.gacode:\n{txt}')
    print(f'Profile mean error: {np.mean(err_prof):.2f}%', typeMsg='i' if np.mean(err_prof) < 1e-0 else 'w')
    print(f'Gradient mean error (ignoring 0.0): {np.mean(err_grad):.2f}%', typeMsg='i' if np.mean(err_grad) < 1e-0 else 'w')

    fn = PROFILEStools.plotAll([p,p_new],extralabs=['Original','New'],lastRhoGradients=rho[-1].item()+0.01)

    axs = fn.figure_handles[3].figure.axes

    axs[0].plot(s.plasma['rho'][0],s.plasma['te'][0],'--o',c='k',markersize=5)
    axs[1].plot(s.plasma['rho'][0],s.plasma['aLte'][0],'--o',c='k',markersize=5)
    axs[2].plot(s.plasma['rho'][0],s.plasma['ti'][0],'--o',c='k',markersize=5)
    axs[3].plot(s.plasma['rho'][0],s.plasma['aLti'][0],'--o',c='k',markersize=5)
    axs[4].plot(s.plasma['rho'][0],s.plasma['ne'][0]*0.1,'--o',c='k',markersize=5)
    axs[5].plot(s.plasma['rho'][0],s.plasma['aLne'][0],'--o',c='k',markersize=5)

    fn.show()
