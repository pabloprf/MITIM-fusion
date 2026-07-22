#### this requires you to have CFSPOPCON in your environment, if you have Lengyel installed this requirement is satisfied 


from logging import warning

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cfspopcon
from cfspopcon import named_options
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import Quantity, magnitude_in_units, ureg

from IPython import embed


class Machine:
    def __init__(p, B0, R0, a, delta, kappa, Ip):
        p.magnetic_field_on_axis = Quantity(B0, ureg.T)
        p.major_radius           = Quantity(R0, ureg.m)
        p.minor_radius           = Quantity(a, ureg.m)
        p.triangularity_psi95    = Quantity(delta, ureg.dimensionless)
        p.inverse_aspect_ratio   = a / R0
        p.areal_elongation       = Quantity(kappa, ureg.dimensionless)
        p.plasma_current         = Quantity(Ip, ureg.MA)

def calc_cs_flux(R, 
            a, 
            cs_change_in_field,
            inboard_to_CS_distance = 2.0 * ureg.meters, 
            double_flux_swing = False): 
    ''' inboard_to_CS_distance includes the VV, blanket, and TF on the inboard side (everything between the inboard plasma edge and the CS)'''
    if inboard_to_CS_distance.units != ureg.meters:
        raise ValueError(f"inboard_to_CS_distance must be in meters ( * ureg.meters), got {inboard_to_CS_distance.units}")
    if cs_change_in_field.units != ureg.T:
        raise ValueError(f"cs_change_in_field must be in tesla ( * ureg.T), got {cs_change_in_field.units}")

    R_cs = R - a - inboard_to_CS_distance

    if double_flux_swing:
        return 2 * (np.pi * R_cs**2 * cs_change_in_field) 
    else:
        return np.pi * R_cs**2 * cs_change_in_field 




def calc_flattop_time(p, 
                      overwrite_flux = None, #input unitless for ease and then convert to ureg.Wb in the function
                      cs_change_in_field = None, #input unitless for ease and then convert to ureg.T in the function
                      inboard_to_CS_distance = None, #input unitless for ease and then convert to ureg.meters in the function
                      ejima_coefficient = 0.6, 
                      double_flux_swing = False): 
    if cs_change_in_field is not None:
        if type(cs_change_in_field) is not Quantity:
            cs_change_in_field = cs_change_in_field * ureg.T
        elif cs_change_in_field.units != ureg.T:
            cs_change_in_field = cs_change_in_field.to(ureg.T)
    if overwrite_flux is not None: 
        if type(overwrite_flux) is not Quantity:
            overwrite_flux = overwrite_flux * ureg.Wb
        elif overwrite_flux.units != ureg.Wb:
            overwrite_flux = overwrite_flux.to(ureg.Wb)
    if inboard_to_CS_distance is not None: 
        if type(inboard_to_CS_distance) is not Quantity:
            inboard_to_CS_distance = inboard_to_CS_distance * ureg.meters
        elif inboard_to_CS_distance.units != ureg.meters:
            inboard_to_CS_distance = inboard_to_CS_distance.to(ureg.meters)
    
    if (cs_change_in_field is None or inboard_to_CS_distance is None) and overwrite_flux is None:
        raise ValueError("Either cs_change_in_field and inboard_to_CS_distance must be provided, or overwrite_flux must be provided. If overwrite_flux is provided, the other two parameters will be ignored.")

    irho_95 = np.argmin(np.abs(p.profiles['rho(-)'] - 0.95))

    machine = Machine(B0 = p.derived['B0'],
                       R0 = p.derived['Rgeo'],
                       a = p.derived['a'], 
                       delta = p.profiles['delta(-)'][irho_95], # 95% flux surface triangularity
                       kappa = p.profiles['kappa(-)'][-1], # areal elongation, equivalent to separatrix
                       Ip = float(p.profiles['current(MA)'][0]),
    ) 
    algorithms = [
    "calc_minor_radius_from_inverse_aspect_ratio",
    "calc_plasma_poloidal_circumference",
    "calc_plasma_volume",
    "calc_elongation_at_psi95_from_areal_elongation",
    "calc_average_ion_temp_from_temperature_ratio",
    "calc_f_shaping_for_qstar",
    "calc_q_star_from_plasma_current",
    "calc_beta_toroidal",
    "calc_beta_poloidal",
    "calc_effective_collisionality",
    "calc_ion_density_peaking",
    "calc_electron_density_peaking",
    "calc_bootstrap_fraction",
    "calc_inductive_plasma_current",
    "calc_Spitzer_loop_resistivity",
    "calc_resistivity_trapped_enhancement",
    "calc_neoclassical_loop_resistivity",
    "calc_loop_voltage",
    "calc_cylindrical_edge_safety_factor",
    "calc_internal_inductivity",
    "calc_internal_inductance_for_noncylindrical",
    "calc_external_inductance",
    "calc_vertical_field_mutual_inductance",
    "calc_invmu_0_dLedR",
    "calc_vertical_magnetic_field",
    "calc_internal_flux",
    "calc_external_flux",
    "calc_resistive_flux",
    "calc_poloidal_field_flux",
    "calc_flux_needed_from_solenoid_over_rampup",
    "calc_max_flattop_duration",
    "calc_breakdown_flux_consumption",
    ]

    algs = []

    for key in algorithms:
        algs.append(Algorithm.get_algorithm(key))

    calc_flux_and_inductance_dependencies = CompositeAlgorithm(algs)

    if overwrite_flux is not None:
        total_flux_available_from_CS = overwrite_flux
        warning(f"Overwriting total flux available from CS with {overwrite_flux:.2f} Wb. The cs_change_in_field and inboard_to_CS_distance parameters will be ignored.")
    else:
        if cs_change_in_field is None or inboard_to_CS_distance is None:
            raise ValueError("Either cs_change_in_field and inboard_to_CS_distance must be provided, or overwrite_flux must be provided. If overwrite_flux is provided, the other two parameters will be ignored.")
        total_flux_available_from_CS = calc_cs_flux(R=machine.major_radius, a=machine.minor_radius, cs_change_in_field=cs_change_in_field, inboard_to_CS_distance=inboard_to_CS_distance, double_flux_swing=double_flux_swing)

    dataset = calc_flux_and_inductance_dependencies.run(
        major_radius = machine.major_radius,
        areal_elongation = machine.areal_elongation,
        triangularity_psi95 = machine.triangularity_psi95,
        magnetic_field_on_axis = machine.magnetic_field_on_axis,
        plasma_current = machine.plasma_current,
        inverse_aspect_ratio = machine.inverse_aspect_ratio,
        elongation_ratio_areal_to_psi95 = 1.025, # hardcoded, could be updated to variable in the future 
        average_electron_density = p.derived['ne_vol20'] * ureg.n20,
        average_electron_temp = p.derived["Te_vol"] * ureg.keV, 
        ion_to_electron_temp_ratio = p.derived["tite_vol"],
        surface_inductance_coefficients = named_options.SurfaceInductanceCoeffs.Barr,
        total_flux_available_from_CS = total_flux_available_from_CS,
        ejima_coefficient = ejima_coefficient,
        z_effective = p.derived["Zeff_vol"],
        electron_density_peaking_offset = p.derived["ne_peaking0.2"] - p.derived["ne_peaking_empirical_source_free"],
        ion_density_peaking_offset = p.derived["ni_peaking0.2"] - p.derived["ne_peaking_empirical_source_free"],
        temperature_peaking = p.derived["Te_peaking"], 
        dilution = p.derived["fmain"] 
    ) 

    dataset["max_flux_for_flattop"] = (
        dataset["total_flux_available_from_CS"] - dataset["flux_needed_from_CS_over_rampup"]
    )

    dataset["total_flux_consumed_over_rampup"] = (
        dataset["internal_flux"] + dataset["external_flux"] + dataset["resistive_flux"]
    )
    dataset["rule_of_thumb_flux_consumed_over_rampup"] = (
        magnitude_in_units(2.0 * dataset["major_radius"] * dataset["plasma_current"], ureg.m * ureg.MA) * ureg.Wb
    )

    report_lines = [
        f"Internal flux = {dataset['internal_flux'].data.to(ureg.Wb):.2f}",
        f"External flux = {dataset['external_flux'].data.to(ureg.Wb):.2f}",
        rf"Resistive flux =  Ejima Coefficient * $\mu_0$ * $I_p$ * R0 = {dataset['resistive_flux'].data.to(ureg.Wb):.2f}",
        "-------------------------------------------------------------------------------------",
        f"Total flux for ramp up = Internal flux + External flux + Resistive flux = {dataset['total_flux_consumed_over_rampup'].data.to(ureg.Wb):.2f}",
        f"Poloidal field flux = {dataset['poloidal_field_flux'].data.to(ureg.Wb):.2f}",
        f"Flux needed from CS over rampup = Total flux for ramp up - Poloidal field flux = {dataset['flux_needed_from_CS_over_rampup'].data.to(ureg.Wb):.2f}",
        "-------------------------------------------------------------------------------------",
        f"Total flux available from CS = {total_flux_available_from_CS:.2f} Wb",
        f"Max flux for flattop = Total flux available from CS - Flux needed from CS over rampup = {dataset['max_flux_for_flattop'].data.to(ureg.Wb):.2f}",
        "-------------------------------------------------------------------------------------",
        f"Spitzer resistivity = {dataset['spitzer_resistivity'].data.to(ureg.ohm * ureg.m):.2e}",
        f"Trapped particle fraction = {dataset['trapped_particle_fraction'].data:.2f}",
        f"Neoclassical resistivity = Spitzer resistivity * Zeff * 0.9 * trapped_particle_fraction = {dataset['neoclassical_loop_resistivity'].data.to(ureg.ohm * ureg.m):.2e}",
        f"Inductive plasma current = {dataset['inductive_plasma_current'].data.to(ureg.MA):.2f}",
        f"Loop voltage = Inductive plasma current * neoclassical resistivity = {dataset['loop_voltage'].data.to(ureg.V):.2f}",
        "-------------------------------------------------------------------------------------",
        f"Max flattop duration = {dataset['max_flattop_duration'].data.to(ureg.s):.2f}",
    ]

    print("\n".join(report_lines))

    return dataset['max_flattop_duration'].data.to(ureg.s), dataset