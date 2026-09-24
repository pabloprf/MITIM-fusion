import numpy as np
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

MU0 = 4e-7 * np.pi
E_J = 1.60218e-19


def write_far3d_profiles(profiles, file, alpha_on=0, beam_index=None, alpha_index=None, main_index=None,
                         impurity_index=None, shape_at="95", R_rotation="rmaj", contaminant=None):
    """
    Write a FAR3D external-profiles file (ext_prof = 1, read by ae_profiles in
    equilibrium.f90) from an input.gacode state (PROFILEStools.gacode_state).

    Columns (free format, one row per gacode radial point):
        alpha_on = 0 (14 columns):
            rho, q, n_beam, n_i, n_e, n_imp, T_beam, T_i, T_e, p_beam, p_thermal, p_equil, v_tor, v_pol
        alpha_on = 1 (16 columns):
            rho, q, n_beam, n_i, n_e, n_alpha, n_imp, T_beam, T_i, T_e, T_alpha, p_beam, p_thermal, p_equil, v_tor, v_pol

    Units: n in 1E20 m^-3, T in keV, p in kPa, v in km/s. rho is sqrt of the normalized toroidal flux.

    Species (indices into the gacode ion list; None = automatic choice):
        main_index      main thermal ion (default: first thermal ion)
        impurity_index  impurity (default: first thermal ion other than the main ion; zeros if there is none)
        beam_index      "Beam" columns = FAR3D EP species 1.
                        Default: first fast ion if alpha_on = 0; second fast ion if alpha_on = 1 (zeros if there is none)
        alpha_index     "Alpha" columns = FAR3D EP species 2 (alpha_on = 1 only). Default: first fast ion
        Fast-ion temperatures are the gacode effective temperatures (p_fast / n_fast).

    Pressures:
        p_beam     n_beam * T_beam
        p_thermal  electrons + all thermal ions (derived["pthr_manual"])
        p_equil    ptot(Pa), i.e. the pressure the equilibrium (e.g. VMEC from gacode_to_vmec_input) was built with

    Other options:
        shape_at    "95" (derived kappa95/delta95), "a" (volume-averaged kappa_a, delta95) or "sep" (last gacode point).
                    Header only: FAR3D does not use kappa and delta when it reads the file.
        R_rotation  radius used to convert w0(rad/s) to v_tor: "rmaj" (flux-surface center, rmaj(m)) or "LF" (low field side, derived["R_LF"])
        contaminant name for the "Main Contaminant Species" header line (default: impurity name from gacode)
    """

    p = profiles.profiles
    types = [str(t) for t in p["type"]]
    fast = [i for i, t in enumerate(types) if "fast" in t]
    therm = [i for i, t in enumerate(types) if "therm" in t]

    if main_index is None:
        main_index = therm[0]
    if impurity_index is None:
        others = [i for i in therm if i != main_index]
        impurity_index = others[0] if len(others) > 0 else None
    if alpha_on == 0:
        if alpha_index is not None:
            raise ValueError("[FAR3D] alpha_index requires alpha_on = 1")
        if beam_index is None:
            beam_index = fast[0] if len(fast) > 0 else None
    elif alpha_on == 1:
        if alpha_index is None:
            alpha_index = fast[0] if len(fast) > 0 else None
        if beam_index is None:
            rest = [i for i in fast if i != alpha_index]
            beam_index = rest[0] if len(rest) > 0 else None
    else:
        raise ValueError(f"[FAR3D] alpha_on must be 0 or 1, got {alpha_on}")

    nr = p["rho(-)"].shape[0]
    zero = np.zeros(nr)

    def n20(i):
        return zero if i is None else p["ni(10^19/m^3)"][:, i] * 0.1

    def TkeV(i):
        return zero if i is None else p["ti(keV)"][:, i]

    # Profiles
    rho, q = p["rho(-)"], p["q(-)"]
    ne, Te = p["ne(10^19/m^3)"] * 0.1, p["te(keV)"]
    n_i, T_i = n20(main_index), TkeV(main_index)
    n_imp = n20(impurity_index)
    n_b, T_b = n20(beam_index), TkeV(beam_index)
    n_a, T_a = n20(alpha_index), TkeV(alpha_index)

    p_beam = (n_b * 1e20) * (T_b * 1e3 * E_J) * 1e-3  # kPa
    p_thermal = profiles.derived["pthr_manual"] * 1e3  # MPa -> kPa
    p_equil = p["ptot(Pa)"] * 1e-3

    if R_rotation == "rmaj":
        R = p["rmaj(m)"]
    elif R_rotation == "LF":
        R = profiles.derived["R_LF"]
    else:
        raise ValueError(f"[FAR3D] R_rotation must be 'rmaj' or 'LF', got {R_rotation}")
    v_tor = p["w0(rad/s)"] * R * 1e-3
    v_pol = zero

    if alpha_on == 0:
        columns = [rho, q, n_b, n_i, ne, n_imp, T_b, T_i, Te, p_beam, p_thermal, p_equil, v_tor, v_pol]
        labels = ["Beam Ion Density(10^20 m^-3)", "Ion Density(10^20 m^-3)", "Elec Density(10^20 m^-3)",
                  "Impurity Density(10^20 m^-3)", "Beam Ion Effective Temp(keV)", "Ion Temp(keV)", "Electron Temp(keV)"]
    else:
        columns = [rho, q, n_b, n_i, ne, n_a, n_imp, T_b, T_i, Te, T_a, p_beam, p_thermal, p_equil, v_tor, v_pol]
        labels = ["Beam Ion Density(10^20 m^-3)", "Ion Density(10^20 m^-3)", "Elec Density(10^20 m^-3)",
                  "Alpha Density(10^20 m^-3)", "Impurity Density(10^20 m^-3)", "Beam Ion Effective Temp(keV)",
                  "Ion Temp(keV)", "Electron Temp(keV)", "Alpha Temp(keV)"]
    labels = ["Rho(norml. sqrt. toroid. flux)", "q"] + labels + [
        "Beam Pressure(kPa)", "Thermal Pressure(kPa)", "Equil.Pressure(kPa)", "Tor Rot(km/s)", "Pol Rot(km/s)"]

    # Header quantities
    B0 = np.abs(float(p["bcentr(T)"][-1]))
    R0 = float(p["rcentr(m)"][-1])
    a = float(p["rmin(m)"][-1])
    if shape_at == "95":
        kappa, delta = profiles.derived["kappa95"], profiles.derived["delta95"]
    elif shape_at == "a":
        kappa, delta = profiles.derived["kappa_a"], profiles.derived["delta95"]
    elif shape_at == "sep":
        kappa, delta = p["kappa(-)"][-1], p["delta(-)"][-1]
    else:
        raise ValueError(f"[FAR3D] shape_at must be '95', 'a' or 'sep', got {shape_at}")
    if contaminant is None:
        contaminant = "none" if impurity_index is None else str(p["name"][impurity_index])
    mass_main = float(p["mass"][main_index])
    beta0 = 2 * MU0 * p_equil[0] * 1e3 / B0**2

    # ae_profiles skips every label line and the three lines after the ion mass
    # (beta line, blank, column names), then reads rows list-directed until EOF
    lines = [
        "PLASMA GEOMETRY",
        f"Vacuum Toroidal magnetic field at R={R0:.2f}m [Tesla]",
        f"    {B0:.6f}",
        "Geometric Center Major radius [m]",
        f"    {R0:.6f}",
        "Minor radius [m]",
        f"    {a:.6f}",
        "Avg. Elongation",
        f"    {kappa:.6f}",
        "Avg. Top/Bottom Triangularity",
        f"    {delta:.6f}",
        "Main Contaminant Species",
        f"    {contaminant}",
        "Main Ion Species mass/proton mass",
        f"    {mass_main:.6f}",
        f"TRYING TO GET TO BETA(0)={beta0:.4f} , Rmax={R0:.2f}",
        "",
        ", ".join(labels),
    ]
    data = np.column_stack(columns)
    with open(file, "w") as f:
        f.write("\n".join(lines) + "\n")
        np.savetxt(f, data, fmt="%.10e")

    print(f"\t- FAR3D external profiles written to {file} ({nr} rows, alpha_on = {alpha_on}; set ext_prof_len = {nr})")
    print(f"\t\t* main ion: {p['name'][main_index]}, impurity: {'-' if impurity_index is None else p['name'][impurity_index]}, "
          f"beam (EP 1): {'-' if beam_index is None else p['name'][beam_index]}"
          + (f", alpha (EP 2): {'-' if alpha_index is None else p['name'][alpha_index]}" if alpha_on == 1 else ""))

    return data
