import copy
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from mitim_tools.misc_tools import LOGtools, IOtools, PLASMAtools
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_modules.powertorch.physics_models import targets_analytic, parameterizers
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed

# <> Function to interpolate a curve <> 
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

def gacode_to_powerstate(self, rho_vec=None):
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

    input_gacode = self.profiles
    if rho_vec is None:
        rho_vec = self.plasma["rho"]

    # *********************************************************************************************
    # Radial grid
    # *********************************************************************************************

    rho_use = input_gacode.profiles["rho(-)"]

    # *********************************************************************************************
    # plasma_io propagation (see plasma_io_migration_plan.md, Part 2 Phase 2): when the source
    # mitim_state was built from a fusio plasma_io object (via scratch()), prefer reading
    # SI-native quantities straight from it instead of the GACODE-unit dict, one key at a time,
    # each gated by a parity check against the dict-derived value it would otherwise replace.
    # *********************************************************************************************
    plasma_io_overrides = {}
    plasma_io_obj = getattr(input_gacode, "plasma_io", None)
    if plasma_io_obj is not None and "r_minor" in plasma_io_obj.output:
        # Note: input_gacode.profiles may have already been resampled onto a different radial
        # grid than plasma_io's own 'radius' coordinate (e.g. by improve_resolution_profiles,
        # which only touches the dict, not the companion plasma_io object) — interpolate onto
        # rho_use (the grid actually in use here) before comparing/substituting.
        rho_plasma_io = plasma_io_obj.output["radius"].to_numpy()
        rmin_plasma_io_native = plasma_io_obj.output["r_minor"].isel(time=-1).to_numpy()
        rmin_from_plasma_io = interpolation_function(rho_use, rho_plasma_io, rmin_plasma_io_native)
        rmin_from_dict = input_gacode.profiles["rmin(m)"]
        # rmin_from_dict was itself resampled onto rho_use by improve_resolution_profiles()
        # using a different interpolant (pchip) than the cubic spline used here, so a small
        # (sub-mm, sub-percent) discrepancy near sharp-curvature regions (e.g. the axis) is
        # expected even when both sources agree physically; this tolerance is loose enough to
        # absorb that while still catching a real unit/shape/species mismatch.
        if not np.allclose(rmin_from_plasma_io, rmin_from_dict, rtol=1e-2, atol=1e-3):
            raise ValueError(
                "[MITIM] plasma_io-derived 'rmin' (r_minor) does not match the gacode-dict-derived "
                "'rmin(m)' value; refusing to substitute (parity check failed)."
            )
        plasma_io_overrides["rmin"] = rmin_from_plasma_io

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
        ["volp", "volp_geo", None, True, True],
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
        if key[0] in plasma_io_overrides:
            quant = plasma_io_overrides[key[0]]
        else:
            quant = input_gacode.derived[key[1]] if key[4] else input_gacode.profiles[key[1]]

        # *********************************************************************************************
        # Extract the quantity via interpolation and tensorization
        # *********************************************************************************************
        self.plasma[key[0]] = torch.from_numpy(
            interpolation_function(rho_vec.cpu(), rho_use, quant if key[2] is None else quant[:, key[2]]) if key[3] else quant
            ).to(rho_vec)
        # *********************************************************************************************

    # *********************************************************************************************
    # Fixed targets
    # *********************************************************************************************

    quantitites = {}
    quantitites["QeMWm2_fixedtargets"] = input_gacode.derived["qe_aux_MW"]
    quantitites["QiMWm2_fixedtargets"] = input_gacode.derived["qi_aux_MW"]
    quantitites["Ge_fixedtargets"] = input_gacode.derived["ge_10E20"]
    quantitites["GZ_fixedtargets"] = input_gacode.derived["ge_10E20"] * 0.0
    quantitites["MtJm2_fixedtargets"] = input_gacode.derived["mt_J"]

    if 'qfus' not in self.target_options["options"]["targets_evolve"]:
        # Fusion fixed
        quantitites["QeMWm2_fixedtargets"] += input_gacode.derived["qe_fus_MW"]
        quantitites["QiMWm2_fixedtargets"] += input_gacode.derived["qi_fus_MW"]
   
    if 'qrad' not in self.target_options["options"]["targets_evolve"]:
        # Fusion fixed
        quantitites["QeMWm2_fixedtargets"] -= input_gacode.derived["qrad_MW"]
    
    if 'qie' not in self.target_options["options"]["targets_evolve"]:
        # Exchange fixed if 1
        quantitites["QeMWm2_fixedtargets"] -= input_gacode.derived["qe_exc_MW"]
        quantitites["QiMWm2_fixedtargets"] += input_gacode.derived["qe_exc_MW"]

    for key in quantitites:
        
        # *********************************************************************************************
        # Extract the quantity via interpolation and tensorization
        # *********************************************************************************************
        self.plasma[key] = torch.from_numpy(interpolation_function(rho_vec.cpu(), rho_use, quantitites[key])).to(rho_vec) / self.plasma["volp"]
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

    smooth_around_coarsing = self.transport_options.get("flatten_gradients_at_control_points", True)

    self.profile_constructors_fine, self.profile_constructors_coarse, self.profile_constructors_coarse_middle = {}, {}, {}
    for key in cases_to_parameterize:
        quant = input_gacode.profiles[key[1]] if key[2] is None else input_gacode.profiles[key[1]][:, key[2]]

        (
            aLy_coarse,
            self.profile_constructors_fine[key[0]],
            self.profile_constructors_coarse[key[0]],
            self.profile_constructors_coarse_middle[key[0]],
        ) = parameterizers.piecewise_linear(
            input_gacode.derived["roa"],
            quant,
            self.plasma["roa"],
            parameterize_in_aLx=key[4],
            multiplier_quantity=key[3],
            smooth_around_coarsing=smooth_around_coarsing,
        )

        self.plasma[f"aL{key[0]}"] = aLy_coarse[:, 1]

        # Check that it's not completely zero
        if key[0] in self.predicted_channels:
            if self.plasma[f"aL{key[0]}"].sum() == 0.0:
                addT = 1e-15
                print(f"\t- All values of {key[0]} detected to be zero, to avoid NaNs, inserting {addT} at the edge",typeMsg="w")
                self.plasma[f"aL{key[0]}"][..., -1] += addT

def plasma_io_to_powerstate(self, rho_vec=None):
    """
    Equivalent of gacode_to_powerstate(), but sources data from a fusio plasma_io
    object's own SI-native fields (via plasma_io.to_dict()) instead of the
    GACODE-unit profiles/derived dict.

    Requires:
        - self.profiles.plasma_io is set (i.e. the source mitim_state was built via
          scratch() from a plasma_io object).
        - plasma_io.compute_derived_quantities() has already been run on it -- not
          called here, since it's expensive (MXH shape decomposition) and callers
          should control when it reruns rather than paying that cost on every
          powerstate construction.

    Scope (see plasma_io_migration_plan.md discussion):
        - Sourced natively from plasma_io: radial grid, geometry (roa, Rmajoa,
          volp, rmin, a, eps), kinetic profiles (te, ti, ne, nZ, w0), ion
          densities/species bookkeeping, the gradient (aLx) parameterization
          source arrays, the "additional_quantities_for_potential_use" tier
          (B_unit, B_ref, q, s_q, Zeff, kappa, delta, betae, bare aLni), and the
          "fixed targets" block (QeMWm2/QiMWm2/Ge/GZ/MtJm2_fixedtargets), which
          are built from plasma_io's source-resolved heat/particle/momentum
          integrals with MITIM's own sign/summation convention applied
          explicitly (see mismatch #3 -- plasma_io stores loss-type sources
          like ionization/radiation as negative, MITIM's dict convention adds
          qione/qioni positively and defines qrad as a positive quantity).
        - Note: aLte/aLti/aLne (unindexed, from the additional tier) are
          intentionally NOT computed here even though gacode_to_powerstate
          computes them -- they get immediately overwritten by the
          cases_to_parameterize block below in both functions, so computing
          them would be dead code. Only bare aLni (index-0 "main ion" gradient,
          which is never overwritten since cases_to_parameterize only produces
          per-species aLni{i}) is genuinely used.
        - Bookkeeping attributes that scratch() already derives identically
          either way (mi_first, Dion, Tion, Species) are reused from
          self.profiles rather than re-derived from plasma_io, since they're
          the same computation (readSpecies()) regardless of source.
        - No attempt is made to replicate improve_resolution_profiles()'s
          fine-grid resampling (mismatch #1, explicitly skipped per decision):
          quantities are interpolated directly from plasma_io's native radial
          grid onto rho_vec.
    """

    print("\t- Producing powerstate object from plasma_io")

    input_gacode = self.profiles
    plasma_io_obj = getattr(input_gacode, "plasma_io", None)
    if plasma_io_obj is None:
        raise ValueError("[MITIM] plasma_io_to_powerstate requires self.profiles.plasma_io to be set")

    pdict = plasma_io_obj.to_dict(side="output")

    if rho_vec is None:
        rho_vec = self.plasma["rho"]

    # *********************************************************************************************
    # Radial grid (plasma_io's own native, un-resampled grid; see mismatch #1)
    # *********************************************************************************************

    rho_use = pdict["radius"]

    roa_array = interpolation_function(rho_vec.cpu(), rho_use, pdict["r_minor_norm"])
    rho_array = interpolation_function(rho_vec.cpu(), rho_use, pdict["radius"])

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
    self.plasma["a"] = torch.tensor(pdict["r_minor_lcfs"]).to(rho_vec)
    self.plasma["eps"] = torch.tensor(pdict["epsilon_lcfs"]).to(rho_vec)

    # Composite quantities that need pre-computation on the native grid before interpolation
    # (mirrors how MITIM's derive_quantities_full computes s_hat/s_q/betae once on the full
    # profile grid, ahead of gacode_to_powerstate's later interpolation step).
    s_q_native = (pdict["safety_factor"] / pdict["r_minor_norm"]) ** 2 * pdict["magnetic_shear"]
    s_q_native[0] = 0.0  # infinite in first location, matching MITIMstate.py's explicit override
    betae_native = PLASMAtools.betae(
        pdict["temperature_e"] * 1.0e-3,   # eV -> keV
        pdict["density_e"] * 1.0e-20,      # 1/m^3 -> 10^20/m^3
        pdict["field_unit"],
    )
    # B_ref is NOT the same quantity as B_unit (a common confusion): MITIM defines it as
    # |B_unit * bt_geo|, a genuine radius-varying profile that needs a proper flux-surface
    # geometric factor (bt_geo) to compute correctly. plasma_io's mxh_field_inner stores
    # [bt_inner, bp_inner] (direction order ['toroidal', 'poloidal'], dimensionless B/field_unit)
    # at the inboard point of each flux surface, so its toroidal component scaled by field_unit
    # gives a real per-radius profile, matching the gacode_to_powerstate branch's convention.
    # Previously used field_axis (bcentr) broadcast as a flat placeholder instead, since
    # plasma_io's only candidate at the time (mxh_field_ref) was numerically unstable near the
    # axis (blew up to ~1e5) due to a Jacobian-sign bug in plasma_io's MXH synthesis step, since
    # fixed -- mxh_field_inner comes from the same (now-fixed) synthesis and is bounded and
    # radially smooth.
    B_ref_native = np.abs(pdict["field_unit"] * pdict["mxh_field_inner"][:, 0])
    precomputed = {"s_q": s_q_native, "betae": betae_native, "B_ref": B_ref_native}

    # [powerstate key, plasma_io native key, ion index (None if not per-ion), unit-conversion factor]
    quantities_to_interpolate = [
        ["rho", "radius", None, 1.0],
        ["roa", "r_minor_norm", None, 1.0],
        ["Rmajoa", "r_geometric_norm", None, 1.0],
        ["volp", "dvolume_dr", None, 1.0],
        ["rmin", "r_minor", None, 1.0],
        ["te", "temperature_e", None, 1.0e-3],          # eV -> keV
        ["ti", "temperature_i", 0, 1.0e-3],              # eV -> keV, main-ion index 0
        ["ne", "density_e", None, 1.0e-19],              # 1/m^3 -> 10^19/m^3
        ["nZ", "density_i", self.impurityPosition, 1.0e-19],  # 1/m^3 -> 10^19/m^3
        ["w0", "rotation_frequency_sonic", None, 1.0],
        # "additional_quantities_for_potential_use" tier
        ["B_unit", "field_unit", None, 1.0],
        ["B_ref", None, None, 1.0],      # precomputed placeholder -- see note above
        ["q", "safety_factor", None, 1.0],
        ["s_q", None, None, 1.0],       # precomputed
        ["aLni", "grad_density_i_norm", 0, 1.0],
        ["Zeff", "effective_charge", None, 1.0],
        ["kappa", "mxh_kappa", None, 1.0],
        ["delta", "mxh_delta", None, 1.0],
        ["betae", None, None, 1.0],     # precomputed
    ]

    for key in quantities_to_interpolate:
        native = precomputed[key[0]] if key[1] is None else pdict[key[1]]
        quant = (native if key[2] is None else native[:, key[2]]) * key[3]

        # *********************************************************************************************
        # Extract the quantity via interpolation and tensorization
        # *********************************************************************************************
        self.plasma[key[0]] = torch.from_numpy(
            interpolation_function(rho_vec.cpu(), rho_use, quant)
            ).to(rho_vec)
        # *********************************************************************************************

    # *********************************************************************************************
    # Fixed targets
    # *********************************************************************************************
    # Built from plasma_io's already volume-integrated, source-resolved fields
    # (_compute_integrated_quantities), applying MITIM's own sign/summation convention
    # explicitly rather than reusing plasma_io's own aggregate fields, since plasma_io stores
    # loss-type sources (ionization, radiation) as negative contributions while MITIM's dict
    # convention adds qione/qioni positively and defines qrad as a positive quantity (mismatch #3).

    source_names = list(pdict["source"])
    def src(name):
        return source_names.index(name)

    heat_source_e_vol = pdict["heat_source_e_vol"]        # (radius, source), W
    heat_source_i_vol = pdict["heat_source_i_vol"]        # (radius, ion, source), W
    particle_source_e_vol = pdict["particle_source_e_vol"]  # (radius, source), particles/s
    momentum_source_i_vol = pdict["momentum_source_i_vol"]  # (radius, ion, direction, source), N*m
    direction_names = list(pdict["direction"])
    tor = direction_names.index("toroidal")

    # qe_aux = qrfe + qohme + qbeame + qione (all added positively in MITIM's convention;
    # plasma_io stores ionization as a negative/loss source, hence the minus sign here)
    qe_aux_native = (
        heat_source_e_vol[:, src("ohmic")]
        + heat_source_e_vol[:, src("neutral_beam")]
        + heat_source_e_vol[:, src("ion_cyclotron")]
        + heat_source_e_vol[:, src("electron_cyclotron")]
        - heat_source_e_vol[:, src("ionization")]
    ) * 1.0e-6  # W -> MW

    # qi_aux = qrfi + qbeami + qioni (same ionization sign handling as qe_aux)
    qi_aux_native = (
        heat_source_i_vol[:, :, src("neutral_beam")].sum(axis=1)
        + heat_source_i_vol[:, :, src("ion_cyclotron")].sum(axis=1)
        + heat_source_i_vol[:, :, src("electron_cyclotron")].sum(axis=1)
        - heat_source_i_vol[:, :, src("ionization")].sum(axis=1)
    ) * 1.0e-6

    qe_fus_native = pdict["fusion_heat_source_e_vol"] * 1.0e-6
    qi_fus_native = pdict["fusion_heat_source_i_vol"].sum(axis=1) * 1.0e-6
    # qrad is a positive quantity in MITIM's convention; plasma_io stores radiation as a
    # negative (loss) source, hence the sign flip
    qrad_native = -pdict["radiation_heat_source_vol"] * 1.0e-6
    qe_exc_native = pdict["heat_exchange_ei_vol"] * 1.0e-6

    ge_native = (
        particle_source_e_vol[:, src("neutral_beam")]
        + particle_source_e_vol[:, src("ionization")]
        + particle_source_e_vol[:, src("charge_exchange")]
    ) * 1.0e-20

    mt_native = (
        momentum_source_i_vol[:, 0, tor, src("neutral_beam")]
        + momentum_source_i_vol[:, 0, tor, src("ionization")]
        + momentum_source_i_vol[:, 0, tor, src("charge_exchange")]
    )

    quantitites = {}
    quantitites["QeMWm2_fixedtargets"] = qe_aux_native
    quantitites["QiMWm2_fixedtargets"] = qi_aux_native
    quantitites["Ge_fixedtargets"] = ge_native
    quantitites["GZ_fixedtargets"] = ge_native * 0.0
    quantitites["MtJm2_fixedtargets"] = mt_native

    if 'qfus' not in self.target_options["options"]["targets_evolve"]:
        quantitites["QeMWm2_fixedtargets"] = quantitites["QeMWm2_fixedtargets"] + qe_fus_native
        quantitites["QiMWm2_fixedtargets"] = quantitites["QiMWm2_fixedtargets"] + qi_fus_native

    if 'qrad' not in self.target_options["options"]["targets_evolve"]:
        quantitites["QeMWm2_fixedtargets"] = quantitites["QeMWm2_fixedtargets"] - qrad_native

    if 'qie' not in self.target_options["options"]["targets_evolve"]:
        quantitites["QeMWm2_fixedtargets"] = quantitites["QeMWm2_fixedtargets"] - qe_exc_native
        quantitites["QiMWm2_fixedtargets"] = quantitites["QiMWm2_fixedtargets"] + qe_exc_native

    # volp genuinely vanishes at the magnetic axis (it's a volume element); guard the division
    # there rather than propagating a NaN/inf, matching similar safe-division guards elsewhere
    # in this codebase (e.g. inv_j_r in fusio's plasma.py).
    volp_safe = torch.where(torch.isclose(self.plasma["volp"], torch.zeros_like(self.plasma["volp"])), torch.ones_like(self.plasma["volp"]), self.plasma["volp"])
    for key in quantitites:
        self.plasma[key] = torch.from_numpy(interpolation_function(rho_vec.cpu(), rho_use, quantitites[key])).to(rho_vec) / volp_safe

    # *********************************************************************************************
    # Ion species need special treatment
    # *********************************************************************************************

    defineIonsFromPlasmaIO(self, input_gacode, pdict, rho_vec, rho_vec)

    # *********************************************************************************************
    # Treatment of rotation gradient
    # *********************************************************************************************

    self.plasma["kradcm"] = 1e-5 / self.plasma["a"]

    # *********************************************************************************************
    # Define profile_constructor functions for the varying profiles and gradients from here
    # *********************************************************************************************

    cases_to_parameterize = [
        ["te", "temperature_e", None, 1.0e-3, 1.0, True],
        ["ti", "temperature_i", 0, 1.0e-3, 1.0, True],
        ["ne", "density_e", None, 1.0e-19, 1.0, True],
        ["nZ", "density_i", self.impurityPosition, 1.0e-19, 1.0, True],
        ["w0", "rotation_frequency_sonic", None, 1.0, self.plasma["kradcm"], False],
    ]

    n_ions = pdict["density_i"].shape[1]
    for i in range(n_ions):
        cases_to_parameterize.append([f"ni{i}", "density_i", i, 1.0e-19, 1.0, True])

    smooth_around_coarsing = self.transport_options.get("flatten_gradients_at_control_points", True)

    # profile_constructors_fine's fine grid must match input_gacode.profiles's own resolution
    # (not plasma_io's native grid): powerstate_to_gacode() -- shared with the dict-based path,
    # not aware of which dispatch built these constructors -- reconstructs full profiles from
    # them and slice-assigns the result directly into input_gacode.profiles[key], which is fixed
    # at whatever resolution improve_resolution_profiles() left it at. So even though
    # plasma_io_to_powerstate otherwise reads straight from plasma_io's native grid (mismatch #1),
    # this one piece needs plasma_io's quantities resampled onto input_gacode's fine grid first.
    roa_use_dict = input_gacode.derived["roa"]
    rho_use_dict = input_gacode.profiles["rho(-)"]

    self.profile_constructors_fine, self.profile_constructors_coarse, self.profile_constructors_coarse_middle = {}, {}, {}
    for key in cases_to_parameterize:
        native = pdict[key[1]]
        quant_native = (native if key[2] is None else native[:, key[2]]) * key[3]
        quant = interpolation_function(rho_use_dict, rho_use, quant_native)

        (
            aLy_coarse,
            self.profile_constructors_fine[key[0]],
            self.profile_constructors_coarse[key[0]],
            self.profile_constructors_coarse_middle[key[0]],
        ) = parameterizers.piecewise_linear(
            roa_use_dict,
            quant,
            self.plasma["roa"],
            parameterize_in_aLx=key[5],
            multiplier_quantity=key[4],
            smooth_around_coarsing=smooth_around_coarsing,
        )

        self.plasma[f"aL{key[0]}"] = aLy_coarse[:, 1]

        # Check that it's not completely zero
        if key[0] in self.predicted_channels:
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
    debugPlot=False,
):
    '''
    Notes:
        - insert_highres_powers: whether to insert high resolution powers (will calculate them with powerstate targets object, not other custom ones)
        - Does NOT sync/refresh self.profiles.plasma_io: to_gacode()/from_powerstate() is
          called from TRANSPORTtools.py's _produce_profiles() on *every* transport
          evaluation, including every discarded intermediate flux-match sub-iteration
          (STATEtools.flux_match()'s solver loop calls self.calculate() -> ... ->
          _produce_profiles() many times per solve, most of which get immediately
          superseded by the next sub-iteration), so syncing here would redo the
          deepcopy+interpolation+postprocessing work for data nobody ever reads. The
          returned profiles.plasma_io is therefore just whatever self.profiles.plasma_io
          already was (stale hand-me-down, via powerstate_to_gacode's copy.deepcopy).
          plasma_io should only track the result of a real BO step (newly-proposed
          gradients, evaluated and fed back) -- see PORTALSmain.runModelEvaluator()'s
          explicit powerstate.sync_plasma_io() call after its single
          powerstate.calculate(X, ...), and plasma_io_migration_plan.md for why
          flux_match()'s callers (SR initialization, surrogate flux-matching) are
          deliberately excluded even though they also reach a "converged" point.
    '''
    print(">> Inserting powerstate into input.gacode")

    profiles = powerstate_to_gacode(
        self,
        position_in_powerstate_batch=position_in_powerstate_batch,
        postprocess_input_gacode=postprocess_input_gacode,
        insert_highres_powers=insert_highres_powers,
        rederive=rederive_profiles,
        debugPlot=debugPlot,
    )

    # Write input.gacode
    if write_input_gacode is not None:
        write_input_gacode = Path(write_input_gacode)
        print(f"\t- Writing input.gacode file: {IOtools.clipstr(write_input_gacode)}")
        write_input_gacode.parent.mkdir(parents=True, exist_ok=True)
        profiles.write_state(file=write_input_gacode)

    # If corrections modify the ions set... it's better to re-read, otherwise powerstate will be confused
    if rederive_profiles:
        defineIons(self, profiles, self.plasma["rho"][position_in_powerstate_batch, :], self.dfT)
        # Repeat, that's how it's done earlier
        self._repeat_tensors(batch_size=self.plasma["rho"].shape[0],
            specific_keys=["ni","ions_set_mi","ions_set_Zi","ions_set_Dion","ions_set_Tion","ions_set_c_rad"],
            positionToUnrepeat=None)

    return profiles

def to_plasma_io(
    self,
    write_plasma_io=None,
    position_in_powerstate_batch=0,
    time_index=-1,
    postprocess_plasma_io={},
    recompute_derived=True,
):
    '''
    plasma_io-native sibling of to_gacode(): writes powerstate's currently-predicted
    te/ti/ne/nZ/w0 profiles back into a deep copy of self.profiles.plasma_io and returns
    the updated plasma_io instance. Does NOT replace to_gacode()/from_powerstate -- callers
    that need input.gacode text output or gacode_state's species metadata/.derived dict for
    TGYRO/downstream MITIM logic must still use to_gacode().
    '''
    print(">> Inserting powerstate into plasma_io")

    plasma_io_obj = powerstate_to_plasma_io(
        self,
        position_in_powerstate_batch=position_in_powerstate_batch,
        time_index=time_index,
        postprocess_plasma_io=postprocess_plasma_io,
        recompute_derived=recompute_derived,
    )

    if write_plasma_io is not None:
        write_plasma_io = Path(write_plasma_io)
        print(f"\t- Dumping plasma_io file: {IOtools.clipstr(write_plasma_io)}")
        write_plasma_io.parent.mkdir(parents=True, exist_ok=True)
        plasma_io_obj.dump(write_plasma_io, overwrite=True)

    return plasma_io_obj

def sync_plasma_io(self, position_in_powerstate_batch=0):
    '''
    Refresh self.profiles.plasma_io IN PLACE from the powerstate's current predicted
    profiles (cheap path -- powerstate_to_plasma_io with recompute_derived=False, no MXH
    geometry recompute). No-op if self.profiles.plasma_io is not set.

    Call this once per real BO step -- newly-proposed gradients, evaluated and fed back into
    the gacode representation -- i.e. right after a single powerstate.calculate(X, ...) call
    in PORTALSmain's runModelEvaluator(). Deliberately NOT called from inside to_gacode()'s
    hot path (see its docstring), nor from STATEtools.flux_match() (its callers -- SR
    initialization's N parallel trajectories, and the post-hoc surrogate flux-match -- widen
    the initial training set or run diagnostics, and are not "the" proposed-gradients result;
    see plasma_io_migration_plan.md).
    '''
    if getattr(self.profiles, "plasma_io", None) is not None:
        self.profiles.plasma_io = powerstate_to_plasma_io(
            self,
            position_in_powerstate_batch=position_in_powerstate_batch,
            recompute_derived=False,
        )

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
        - This function assumes that "profiles" is the gacode_state that everything started with.
        - We assume that what changes is only the kinetic profiles allowed to vary.
        - This only works for a single profile, in position_in_powerstate_batch
        - rederive is expensive, so I'm not re-deriving the geometry which is the most expensive
    """

    # Default options for postprocessing
    
    Tfast_ratio = postprocess_input_gacode.get("Tfast_ratio", False)  # Fallback matches the template default (namelist.portals.yaml applyCorrections)
    Ti_thermals = postprocess_input_gacode.get("Ti_thermals", True)
    ni_thermals = postprocess_input_gacode.get("ni_thermals", True)
    recalculate_ptot = postprocess_input_gacode.get("recalculate_ptot", True)
    force_mach = postprocess_input_gacode.get("force_mach", None)

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
        if key[0] in self.predicted_channels:
            print(f"\t- Inserting {key[0]} into input.gacode profiles")

            # *********************************************************************************************
            # From a/Lx to x via fine profile_constructor
            # *********************************************************************************************
            x, y = self.profile_constructors_fine[key[0]](
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

    if "w0" not in self.predicted_channels and force_mach is not None:
        # Rotation fixed to ensure Mach number
        profiles.introduceRotationProfile(Mach_LF=force_mach)

    # ------------------------------------------------------------------------------------------
    # Insert Powers
    # ------------------------------------------------------------------------------------------

    if insert_highres_powers:
        powerstate_to_gacode_powers(self, profiles)

    # ------------------------------------------------------------------------------------------
    # Recalculate and change ptot to make it consistent?
    # ------------------------------------------------------------------------------------------

    if rederive or recalculate_ptot:
        profiles.derive_quantities(rederiveGeometry=False)

    if recalculate_ptot:
        profiles.selfconsistentPTOT()

    if debugPlot:
        debug_transformation(self.profiles,profiles,self)

    return profiles

def powerstate_to_plasma_io(
    self,
    position_in_powerstate_batch=0,
    time_index=-1,
    postprocess_plasma_io={},
    recompute_derived=True,
):
    """
    plasma_io-native sibling of powerstate_to_gacode(): writes powerstate's currently
    predicted te/ti/ne/nZ/w0 profiles into a deep copy of self.profiles.plasma_io's own
    SI-native DataTree, instead of the GACODE-unit dict.

    Notes:
        - Requires self.profiles.plasma_io is set (same precondition as
          plasma_io_to_powerstate; if powerstate was constructed via that path,
          self.profile_constructors_fine is already populated).
        - Reciprocal of plasma_io_to_powerstate's quantities_to_interpolate unit factors:
          te/ti keV->eV (x1e3), ne/nZ 10^19/m^3 -> 1/m^3 (x1e19), w0 unchanged.
        - Only self.predicted_channels are written directly; everything else in the
          returned plasma_io is left exactly as in the source snapshot, except for
          the Ti_thermals/ni_thermals postprocessing below (which, like
          powerstate_to_gacode, touches OTHER ion columns as a side effect).
        - Ti_thermals/ni_thermals postprocessing (postprocess_plasma_io dict, both
          default True, same keys/defaults as powerstate_to_gacode's
          postprocess_input_gacode): mirrors powerstate_to_gacode's use of
          gacode_state.makeAllThermalIonsHaveSameTemp()/scaleAllThermalDensities(),
          via the plasma_io-native plasma_io.equalize_thermal_ion_temperatures()/
          scale_thermal_ion_densities(). Applied in the same order/logic as the
          gacode path: Ti_thermals only fires when "ti" was just written (leaves
          ref_ion=0, the just-written channel, untouched); ni_thermals only fires
          when "ne" was just written, and excludes the impurity ion index whenever
          "nZ" is ALSO a predicted channel (so nZ's own direct write is the final
          value there, not further perturbed by the ne-derived scale factor) --
          exactly matching powerstate_to_gacode's insertion order (nZ processed
          after ne, so its direct assignment overwrites whatever
          scaleAllThermalDensities did to that index).
        - No Tfast_ratio/force_mach equivalent (fast-ion and w0-from-Mach handling)
          -- narrower scope than powerstate_to_gacode's postprocessing, since
          plasma_io's fast-ion bookkeeping wasn't part of this pass.
        - recompute_derived (default True): reruns plasma_io_copy.compute_derived_quantities()
          after all writes/postprocessing. This is plasma_io's equivalent of
          gacode_state.selfconsistentPTOT() -- unlike GACODE's separately-stored,
          independently-writable ptot(Pa), plasma_io has no standalone total-pressure
          field; pressure_thermal_total/pressure_total etc. are always derived fresh
          from the current kinetic profiles, so making them self-consistent with what
          was just written IS rerunning the derived-quantity pipeline (there is no
          separate "ptot" correction to make on top of that). Expensive (redoes the
          full MXH geometry decomposition every call even though geometry itself did
          not change here) -- set recompute_derived=False to skip if only the raw
          kinetic-profile fields are needed downstream.
        - Does not insert powers (heat/particle/momentum sources); TGYRO/TGLF power
          consumption remains served by to_gacode()'s input.gacode, unaffected.
        - Batches all direct channel writes into a single update_output_data_vars() call
          (one dataset copy on read via .output, one on write) rather than one call per
          channel, since both .output and update_output_data_vars deep-copy the full
          output Dataset. The Ti_thermals/ni_thermals postprocessing calls that follow
          each do their own additional read-modify-write pass.
    """

    Ti_thermals = postprocess_plasma_io.get("Ti_thermals", True)
    ni_thermals = postprocess_plasma_io.get("ni_thermals", True)

    input_gacode = self.profiles
    plasma_io_obj = getattr(input_gacode, "plasma_io", None)
    if plasma_io_obj is None:
        raise ValueError("[MITIM] powerstate_to_plasma_io requires self.profiles.plasma_io to be set")

    plasma_io_copy = copy.deepcopy(plasma_io_obj)

    output_ds = plasma_io_copy.output  # single deep-copy read buffer
    n_ions = output_ds.sizes["ion"]

    # [powerstate key, plasma_io native var, ion index (None if not per-ion), reciprocal unit factor]
    quantities = [
        ["te", "temperature_e", None, 1.0e3],
        ["ti", "temperature_i", 0, 1.0e3],
        ["ne", "density_e", None, 1.0e19],
        ["nZ", "density_i", self.impurityPosition, 1.0e19],
        ["w0", "rotation_frequency_sonic", None, 1.0],
    ]

    newvars = {}
    ne_old_native, ne_new_native = None, None
    for key in quantities:
        if key[0] not in self.predicted_channels:
            continue

        ion_idx = key[2]
        if ion_idx is not None and ion_idx >= n_ions:
            raise ValueError(
                f"[MITIM] powerstate_to_plasma_io: ion index {ion_idx} for '{key[0]}' "
                f"out of range for plasma_io's {n_ions}-ion 'ion' dimension"
            )

        print(f"\t- Inserting {key[0]} into plasma_io output")

        # *********************************************************************************************
        # From a/Lx to x,y via the fixed-grid fine profile_constructor
        # *********************************************************************************************
        x_fine, y_fine = self.profile_constructors_fine[key[0]](
            self.plasma["roa"][position_in_powerstate_batch, :],
            self.plasma[f"aL{key[0]}"][position_in_powerstate_batch, :],
        )
        # *********************************************************************************************

        x_fine = x_fine.cpu().numpy()
        y_fine = y_fine[0, :].cpu().numpy()

        # Interpolate from the dict-equivalent fixed grid onto plasma_io's own native grid
        roa_native = output_ds["r_minor_norm"].isel(time=time_index).to_numpy()
        y_native = interpolation_function(roa_native, x_fine, y_fine) * key[3]

        var = key[1]
        full_array = output_ds[var].to_numpy().copy()
        if ion_idx is None:
            if key[0] == "ne":
                # Snapshot the pre-write value here (still the original, un-mutated
                # full_array) so ni_thermals can derive its scale factor below.
                ne_old_native = full_array[time_index, :].copy()
                ne_new_native = y_native
            full_array[time_index, :] = y_native
        else:
            full_array[time_index, :, ion_idx] = y_native

        newvars[var] = (list(output_ds[var].dims), full_array)

    if newvars:
        plasma_io_copy.update_output_data_vars(newvars)

    # ------------------------------------------------------------------------------------------
    # Postprocessing: keep thermal-ion Ti/ni bookkeeping consistent with what was just written
    # ------------------------------------------------------------------------------------------

    if "ti" in self.predicted_channels and Ti_thermals:
        print("\t- Ensuring Ti is equal for all thermal ions")
        plasma_io_copy.equalize_thermal_ion_temperatures(ref_ion=0, side='output')

    if "ne" in self.predicted_channels and ni_thermals:
        print("\t- Adjusting ni of thermal ions")
        scale_factor = ne_new_native / ne_old_native
        exclude_ion_indices = [self.impurityPosition] if "nZ" in self.predicted_channels else []
        plasma_io_copy.scale_thermal_ion_densities(scale_factor, side='output', exclude_ion_indices=exclude_ion_indices)

    # ------------------------------------------------------------------------------------------
    # Recalculate derived quantities to make them consistent (see recompute_derived note above)
    # ------------------------------------------------------------------------------------------

    if recompute_derived:
        plasma_io_copy.compute_derived_quantities(side='output')

    return plasma_io_copy

def powerstate_to_gacode_powers(self, profiles, rederive_at_high_res=True):

    profiles.derive_quantities(rederiveGeometry=False)

    print("\t- Inserting powers")

    state_temp = self.copy_state()

    if rederive_at_high_res:
        # ------------------------------------------------------------------------------------------
        # Recalculate powers with powerstate on the gacode-original fine grid
        # ------------------------------------------------------------------------------------------

        # Modify power flows by tricking the powerstate into a fine grid (same as does TGYRO)
        extra_points = 2  # If I don't allow this, it will fail
        rhoy = profiles.profiles["rho(-)"][1:-extra_points]
        with LOGtools.HiddenPrints():
            state_temp.__init__(
                profiles,
                evolution_options={"rhoPredicted": rhoy},
                target_options={
                    "evaluator": targets_analytic.analytical_model,
                    "options": {
                        "targets_evolve": self.target_options["options"]["targets_evolve"], # Important to keep the same as in the original
                        "target_evaluator_method": "powerstate",
                        "force_zero_particle_flux": self.target_options["options"]["force_zero_particle_flux"],
                        "percent_error": self.target_options["options"]["percent_error"]
                        }
                    },
                increase_profile_resol = False
                )
        state_temp.calculateProfileFunctions()
        state_temp.target_options["options"]["target_evaluator_method"] = "powerstate"
        state_temp.calculateTargets()
        # ------------------------------------------------------------------------------------------

        def state_to_profiles(state, profiles, state_key, profiles_key,position_in_powerstate_batch, **kwargs):
            profiles.profiles[conversions[ikey]][:-extra_points] = state.plasma[ikey][position_in_powerstate_batch,:].cpu().numpy()

    else:
        # Just interpolate from powerstate to gacode grid
        def state_to_profiles(state, profiles, state_key, profiles_key,position_in_powerstate_batch, **kwargs):
            profiles.profiles[conversions[ikey]] = interpolation_function(
                profiles.profiles["rho(-)"],
                state.plasma["rho"][position_in_powerstate_batch,:].cpu().numpy(),
                state.plasma[ikey][position_in_powerstate_batch,:].cpu().numpy()
            )


    conversions = {}

    if 'qie' in self.target_options["options"]["targets_evolve"]:
        conversions['qie'] = "qei(MW/m^3)"
    if 'qrad' in self.target_options["options"]["targets_evolve"]:
        conversions['qrad_bremms'] = "qbrem(MW/m^3)"
        conversions['qrad_sync'] = "qsync(MW/m^3)"
        conversions['qrad_line'] = "qline(MW/m^3)"
    if 'qfus' in self.target_options["options"]["targets_evolve"]:
        conversions['qfuse'] = "qfuse(MW/m^3)"
        conversions['qfusi'] = "qfusi(MW/m^3)"

    position_in_powerstate_batch = 0

    for ikey in conversions:
        if conversions[ikey] in profiles.profiles:
            state_to_profiles(state_temp, profiles, ikey, conversions[ikey], position_in_powerstate_batch)
        else:
            profiles.profiles[conversions[ikey]] = np.zeros(len(profiles.profiles["qei(MW/m^3)"]))
            state_to_profiles(state_temp, profiles, ikey, conversions[ikey], position_in_powerstate_batch)
    
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
            self.plasma["ni"].append(interpolation_function(rho_vec, rho_use, input_gacode.profiles["ni(10^19/m^3)"][:, i]))
            mi.append(input_gacode.profiles["mass"][i])
            Zi.append(input_gacode.profiles["z"][i])

            # Grab chebyshev coefficients from file
            data_df = pd.read_csv(__mitimroot__ / "src" / "mitim_modules" / "powertorch" / "physics_models" / "radiation_chebyshev.csv")
            try:
                c = data_df[data_df['Ion'].str.lower()==input_gacode.profiles["name"][i].lower()].to_numpy()[0,2:].astype(float)
            except IndexError:
                print(f'\t- Specie {input_gacode.profiles["name"][i]} not found in radiation database, assuming zero radiation from it',typeMsg="w")
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

def defineIonsFromPlasmaIO(self, input_gacode, pdict, rho_vec, dfT):
    """
    plasma_io-native equivalent of defineIons(): stores as part of powerstate the
    thermal ions densities (ni) and the information about how to interpret them
    (ions_set), sourced from plasma_io's own type_i/mass_i/charge_i/ion fields
    instead of the GACODE-unit dict's type/mass/z/name columns.

    Notes:
        - Species ordering is assumed identical between plasma_io's 'ion'
          coordinate and the dict's 'name' column, since both originate from the
          same scratch() conversion and are never reordered independently.
        - Dion/Tion (positions of D/T for alpha power) and Species are reused
          from input_gacode -- see plasma_io_to_powerstate()'s docstring.
    """

    rho_use = pdict["radius"]
    rho_vec = rho_vec.clone().cpu()

    ion_names = list(pdict["ion"])
    thermal_mask = pdict["type_i"] == "thermal"

    self.plasma["ni"], mi, Zi, c_rad = [], [], [], []
    for i in range(len(ion_names)):
        if thermal_mask[i]:
            self.plasma["ni"].append(interpolation_function(rho_vec, rho_use, pdict["density_i"][:, i] * 1.0e-19))
            mi.append(pdict["mass_i"][i])
            Zi.append(pdict["charge_i"][:, i].mean())

            data_df = pd.read_csv(__mitimroot__ / "src" / "mitim_modules" / "powertorch" / "physics_models" / "radiation_chebyshev.csv")
            try:
                c = data_df[data_df['Ion'].str.lower() == ion_names[i].lower()].to_numpy()[0, 2:].astype(float)
            except IndexError:
                print(f'\t- Specie {ion_names[i]} not found in radiation database, assuming zero radiation from it', typeMsg="w")
                c = [-1e10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            c_rad.append(c)

    self.plasma["ni"] = torch.from_numpy(np.transpose(self.plasma["ni"])).to(dfT)
    mi = torch.from_numpy(np.array(mi)).to(dfT)
    Zi = torch.from_numpy(np.array(Zi)).to(dfT)
    c_rad = torch.from_numpy(np.array(c_rad)).to(dfT)

    Dion = torch.tensor(input_gacode.Dion) if input_gacode.Dion is not None else torch.tensor(np.nan)
    Tion = torch.tensor(input_gacode.Tion) if input_gacode.Tion is not None else torch.tensor(np.nan)

    self.plasma["ions_set_mi"] = mi
    self.plasma["ions_set_Zi"] = Zi
    self.plasma["ions_set_Dion"] = Dion
    self.plasma["ions_set_Tion"] = Tion
    self.plasma["ions_set_c_rad"] = c_rad

def improve_resolution_profiles(profiles, rhoMODEL, smooth_around_coarsing=True):
    """
    Resolution of input.gacode
    **************************
    - Change resolution to a fine grid in which doing the flattening around coarse points has a small effect on the profile.
    - It is recommended that it goes through the points, with more points around the trailing edge transition.
    - Also, avoid adding too points near axis. (NOT NOW?)
    """

    # rhoMODEL may arrive as a CUDA tensor from the powerstate initialisation path;
    # all operations below are numpy-based, so coerce to a plain numpy array first.
    if isinstance(rhoMODEL, torch.Tensor):
        rhoMODEL = rhoMODEL.detach().cpu().numpy()

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
    #    (only needed when smoothAroundCoarsing is enabled, to provide the
    #    neighboring fine-grid points that get flattened)
    # ----------------------------------------------------------------------------------

    if smooth_around_coarsing:
        for i in range(points_updown):
            rho_new = np.append(
                np.append(rho_new, rhoMODEL + d_spacing_coarse * (i + 1)),
                rhoMODEL - d_spacing_coarse * (i + 1),
            )

    # ----------------------------------------------------------------------------------
    # Change resolution
    #   Shape-preserving (PCHIP) interpolation: the default cubic spline rings when
    #   interpolating across a slope discontinuity (e.g. a pedestal top sitting at
    #   the last control point), leaving non-monotonic wiggles in the inserted
    #   fine-grid points right next to the control radius — which show up as
    #   artificial a/L dips for nearly-flat unpredicted channels like ne.
    # ----------------------------------------------------------------------------------
    from mitim_tools.misc_tools import MATHtools
    profiles.changeResolution(rho_new=rho_new, interpolation_function=MATHtools.extrapolatePchip)

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

    fn = state_plotting.plotAll([p,p_new],extralabs=['Original','New'],lastRhoGradients=rho[-1].item()+0.01)

    axs = fn.figure_handles[3].figure.axes

    axs[0].plot(s.plasma['rho'][0],s.plasma['te'][0],'--o',c='k',markersize=5)
    axs[1].plot(s.plasma['rho'][0],s.plasma['aLte'][0],'--o',c='k',markersize=5)
    axs[2].plot(s.plasma['rho'][0],s.plasma['ti'][0],'--o',c='k',markersize=5)
    axs[3].plot(s.plasma['rho'][0],s.plasma['aLti'][0],'--o',c='k',markersize=5)
    axs[4].plot(s.plasma['rho'][0],s.plasma['ne'][0]*0.1,'--o',c='k',markersize=5)
    axs[5].plot(s.plasma['rho'][0],s.plasma['aLne'][0],'--o',c='k',markersize=5)

    fn.show()
