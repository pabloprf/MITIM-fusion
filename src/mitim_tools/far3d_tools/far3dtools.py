import re
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

MU0 = 4e-7 * np.pi
E_J = 1.60218e-19


def write_far3d_inputs(profiles, folder, n=None, ext="input_vmec", profiles_file="far3d_profiles.txt", vmec_options={},
                       booz_options={}, profiles_options={}, input_model_options={}, far3d_root=None, plot=True):
    """
    input.gacode -> complete FAR3D run folder:
        1. VMEC++ fixed-boundary equilibrium   wout_<ext>.nc            (gacode_to_vmec)
        2. Boozer equilibrium                  woutb, booz/             (run_booz_xform)
        3. External profiles (ext_prof = 1)    <profiles_file>          (write_far3d_profiles)
        4. FAR3D namelist                      Input_Model              (write_far3d_input_model)
    Run FAR3D in folder afterwards (far3d_executable("far3d")).

    Inputs:
        profiles:               PROFILEStools.gacode_state or path to input.gacode
        folder:                 output folder (created if needed)
        n:                      toroidal mode number for the Input_Model mode lists (None keeps the template's)
        ext:                    VMEC extension, i.e. wout_<ext>.nc and in_booz.<ext>
        vmec_options:           kwargs to gacode_to_vmec
        booz_options:           kwargs to run_booz_xform
        profiles_options:       kwargs to write_far3d_profiles (the species choices also go to the Input_Model)
        input_model_options:    kwargs to write_far3d_input_model (e.g. mj, template, mode_options, params)
        far3d_root:             FAR3D source tree, default the FAR3D_ROOT environment variable
    """

    profiles = _as_gacode_state(profiles)
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)

    wout = gacode_to_vmec(profiles, folder, ext=ext, plot=plot, **vmec_options)
    run_booz_xform(folder, ext=ext, far3d_root=far3d_root, **booz_options)
    write_far3d_profiles(profiles, folder / profiles_file, **profiles_options)

    species = {k: v for k, v in profiles_options.items() if k in ("alpha_on", "beam_index", "alpha_index", "main_index")}
    write_far3d_input_model(profiles, folder / "Input_Model", profiles_file=profiles_file, n=n, wout=wout,
                            far3d_root=far3d_root, **species, **input_model_options)

    return wout


# Executables relative to the FAR3D source tree (FAR3D_ROOT), as built by its CMake projects
FAR3D_EXECUTABLES = {
    "xbooz_xform": "BOOZ_XFORM/build/xbooz_xform",
    "far3d": "build-release/far3d.x",
}


def far3d_executable(name, far3d_root=None):
    """
    Path to a FAR3D executable ("xbooz_xform" or "far3d"), found at FAR3D_EXECUTABLES[name] inside
    far3d_root, or the FAR3D_ROOT environment variable when far3d_root is None.
    """
    import os

    far3d_root = far3d_root if far3d_root is not None else os.environ.get("FAR3D_ROOT")
    if far3d_root is None:
        raise FileNotFoundError(
            f"[FAR3D] Cannot locate '{name}': set the FAR3D_ROOT environment variable to the FAR3D source tree "
            "(the folder containing BOOZ_XFORM/ and build-release/), or pass far3d_root")

    path = Path(far3d_root).expanduser() / FAR3D_EXECUTABLES[name]
    if not path.is_file():
        raise FileNotFoundError(f"[FAR3D] '{name}' not found at {path}, build it first (FAR3D_ROOT = {far3d_root})")

    return path


def _as_gacode_state(profiles):
    if isinstance(profiles, (str, Path)):
        from mitim_tools.gacode_tools import PROFILEStools
        profiles = PROFILEStools.gacode_state(Path(profiles).expanduser())
    return profiles


# ************************************************************************************************************************************************
# input.gacode -> VMEC++ equilibrium
# ************************************************************************************************************************************************

def gacode_to_vmec(profiles, folder, ext="input_vmec", mpol=16, ns_array=(25, 51, 101, 201), ftol_array=None, niter=20000,
                   lasym=True, max_threads=1, verbose=None, plot=True):
    """
    Run VMEC++ (fixed boundary, ncurr = 0) on the input built by VMECtools.gacode_to_vmec_input
    (boundary from the rho = 1 MXH surface, ptot and 1/q on s = rho^2, phiedge = 2*pi*torfluxa).

    Writes to folder:
        <ext>.json              VMEC++ input (rerun with "python -m vmecpp <ext>.json")
        wout_<ext>.nc           VMEC++ output
        wout_<ext>_compare.png  (plot = True) VMEC vs gacode surfaces, q, p (plot_vmec_vs_gacode)

    max_threads = 1 by default: torch (imported by mitim_tools) and vmecpp ship their own libomp, and a
    threaded vmecpp.run segfaults once torch's is loaded. For this axisymmetric problem one thread is also
    the fastest (~35 s for ns up to 201; threading only adds overhead).

    verbose: VMEC++ terminal output (vmecpp.OutputMode): 0 silent, 1 legacy iteration table, 2 animated progress bar,
             3 progress bar for non-TTY outputs (e.g. Jupyter). None = 2 in a terminal, 1 otherwise (log files).

    Returns the vmecpp.VmecWOut.
    """

    import vmecpp
    from mitim_tools.plasmastate_tools.utils import VMECtools

    profiles = _as_gacode_state(profiles)
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)

    inp = VMECtools.gacode_to_vmec_input(profiles, mpol=mpol, ns_array=ns_array, ftol_array=ftol_array, niter=niter, lasym=lasym)
    inp.save(folder / f"{ext}.json")

    if verbose is None:
        import sys
        verbose = 2 if sys.stdout.isatty() else 1

    wout = vmecpp.run(inp, max_threads=max_threads, verbose=verbose).wout
    wout.save(folder / f"wout_{ext}.nc")

    print(f"\t- VMEC++ finished: ier_flag = {wout.ier_flag}, fsqr/fsqz/fsql = {wout.fsqr:.1e} / {wout.fsqz:.1e} / {wout.fsql:.1e}")
    if wout.ier_flag != 0 or max(wout.fsqr, wout.fsqz, wout.fsql) > inp.ftol_array[-1]:
        print("\t- VMEC++ did not reach the requested force tolerance, check the equilibrium", typeMsg='w')
    print(f"\t- VMEC++ equilibrium written to {folder / f'wout_{ext}.nc'}")

    if plot:
        plot_vmec_vs_gacode(profiles, wout, file=folder / f"wout_{ext}_compare.png")

    return wout


def gacode_surface_RZ(profiles, rho, thetas):
    """R, Z of the gacode MXH surface at rho(-) = rho (moments PCHIP-interpolated in rho)."""
    from mitim_tools.gs_tools import GEQtools
    from mitim_tools.plasmastate_tools.utils import VMECtools

    x = profiles.profiles["rho(-)"]
    R0, a, kappa, Z0, cn, sn = VMECtools._mxh_moments(profiles)
    at = lambda f: np.atleast_1d(PchipInterpolator(x, f, axis=0)(rho))
    R, Z = GEQtools.from_mxh_to_RZ(at(R0), at(a), at(kappa), at(Z0), at(cn)[None, :], at(sn)[None, :], thetas=thetas)
    return R[0], Z[0]


def vmec_surface_RZ(wout, s, thetas):
    """R, Z of the VMEC surface at normalized toroidal flux s, phi = 0 (linear in s between
    full-grid surfaces, exact on a grid point). Includes the lasym terms."""
    x = s * (wout.ns - 1)
    j = min(int(np.floor(x)), wout.ns - 2)
    w = x - j
    coef = lambda A: (1 - w) * A[:, j] + w * A[:, j + 1]
    ang = np.outer(thetas, wout.xm)  # axisymmetric: phase m*u
    R = np.cos(ang) @ coef(wout.rmnc)
    Z = np.sin(ang) @ coef(wout.zmns)
    if wout.lasym:
        R += np.sin(ang) @ coef(wout.rmns)
        Z += np.cos(ang) @ coef(wout.zmnc)
    return R, Z


def _curve_distance(Ra, Za, Rb, Zb, k=8):
    """Max over curve a of the distance to the closed polygon b (densified k-fold)."""
    dens = lambda v: np.concatenate([np.linspace(v[i], v[(i + 1) % len(v)], k, endpoint=False) for i in range(len(v))])
    Rb2, Zb2 = dens(Rb), dens(Zb)
    d = np.hypot(Ra[:, None] - Rb2[None, :], Za[:, None] - Zb2[None, :])
    return float(np.max(np.min(d, axis=1)))


def plot_vmec_vs_gacode(profiles, wout, file=None, rho_plot=np.arange(0.1, 1.01, 0.1), n_theta=512):
    """
    Check a VMEC equilibrium against the gacode state it was built from:
        * flux surfaces: gacode MXH surfaces vs VMEC surfaces at the same s = rho^2. VMEC solves the
          interior shapes itself, so this is a real force-balance test.
        * q, pressure: iota and p are VMEC inputs, so these check the transfer. q is compared on VMEC's
          half grid, where iota is defined; the full-grid axis value iotaf[0] is an extrapolation and
          cannot follow structure inside the first radial cell (rho < 1/sqrt(ns-1)).
        * plasma current: VMEC's ctor vs current(MA), an independent check of q + shape + torfluxa
          (sign follows VMEC's convention).
    Returns the max surface distance [m] at each rho_plot.
    """
    import matplotlib.pyplot as plt
    from mitim_tools.misc_tools import GRAPHICStools

    P = profiles.profiles
    rho_g, q_g = P["rho(-)"], np.abs(P["q(-)"])
    thetas = 2 * np.pi * np.arange(n_theta) / n_theta
    closed = lambda v: np.append(v, v[0])

    rho_full = np.sqrt(np.linspace(0.0, 1.0, wout.ns))
    s_half = (np.arange(1, wout.ns) - 0.5) / (wout.ns - 1)
    q_v_full = 1.0 / np.abs(wout.iotaf)
    q_v_half = 1.0 / np.abs(wout.iotas[1:])
    q_g_half = PchipInterpolator(rho_g, q_g)(np.sqrt(s_half))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw=dict(width_ratios=[1.1, 1]))

    a = ax[0, 0]
    dev = []
    for i, rho in enumerate(rho_plot):
        Rg, Zg = gacode_surface_RZ(profiles, rho, thetas)
        Rv, Zv = vmec_surface_RZ(wout, rho**2, thetas)
        dev.append(_curve_distance(Rv, Zv, Rg, Zg))
        a.plot(closed(Rg), closed(Zg), "k-", lw=1.2, label="input.gacode (MXH)" if i == 0 else None)
        a.plot(closed(Rv), closed(Zv), "r--", lw=1.2, label="VMEC++" if i == 0 else None)
    a.plot([P["rmaj(m)"][0]], [P.get("zmag(m)", [0.0])[0]], "k+", ms=10, label="gacode axis")
    a.set_aspect("equal")
    a.set_xlabel("R [m]")
    a.set_ylabel("Z [m]")
    a.set_title(r"flux surfaces at $\rho_{tor}$ = " + f"{rho_plot[0]:.1f} .. {rho_plot[-1]:.1f}")
    a.legend(loc="upper right", fontsize=9)

    a = ax[0, 1]
    a.semilogy(rho_plot, np.maximum(np.array(dev) * 100, 1e-6), "o-")
    a.set_xlabel(r"$\rho_{tor}$")
    a.set_ylabel("max distance VMEC -> gacode surface [cm]")
    a.set_title("surface shape difference")
    GRAPHICStools.addDenseAxis(a)

    a = ax[1, 0]
    a.plot(rho_g, q_g, "k-", lw=2, label="input.gacode")
    a.plot(rho_full, q_v_full, "r--", lw=1.5, label="VMEC++ 1/iotaf (full grid)")
    a.plot(np.sqrt(s_half), q_v_half, "r.", ms=3, label="VMEC++ 1/iotas (half grid)")
    a.set_xlabel(r"$\rho_{tor} = \sqrt{s}$")
    a.set_ylabel("q")
    a.set_title("safety factor")
    a.legend()
    GRAPHICStools.addDenseAxis(a)

    a = ax[1, 1]
    a.plot(rho_g, P["ptot(Pa)"] / 1e6, "k-", lw=2, label="input.gacode ptot")
    a.plot(rho_full, wout.presf / 1e6, "r--", lw=1.5, label="VMEC++ presf")
    a.set_xlabel(r"$\rho_{tor} = \sqrt{s}$")
    a.set_ylabel("p [MPa]")
    a.set_title("pressure")
    a.legend()
    GRAPHICStools.addDenseAxis(a)

    ip_f = float(np.atleast_1d(P.get("current(MA)", [np.nan]))[0])
    fig.suptitle(
        f"VMEC++ vs input.gacode   |   Ip: VMEC {wout.ctor / 1e6:+.3f} MA, file {ip_f:+.3f} MA"
        f"   |   ns={wout.ns}, mpol={wout.mpol}, fsqr={wout.fsqr:.1e}",
        fontsize=11,
    )
    fig.tight_layout()
    if file is not None:
        fig.savefig(file, dpi=130)
        plt.close(fig)
        print(f"\t- VMEC vs gacode comparison written to {file}")

    print(f"\t- Max surface distance VMEC -> gacode: " + ", ".join(f"{d * 100:.3f}" for d in dev)
          + f" cm at rho = " + ", ".join(f"{r:.1f}" for r in rho_plot))
    print(f"\t- q on the VMEC half grid (rho >= {np.sqrt(s_half[0]):.3f}): max rel diff "
          f"{np.max(np.abs(q_v_half - q_g_half) / q_g_half):.2e}; on axis VMEC {q_v_full[0]:.4f} (extrapolated) vs file {q_g[0]:.4f}")
    print(f"\t- Ip: VMEC {wout.ctor / 1e6:+.4f} MA vs file {ip_f:+.4f} MA")

    return dev


# ************************************************************************************************************************************************
# input.gacode -> FAR3D external profiles
# ************************************************************************************************************************************************

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
    main_index, impurity_index, beam_index, alpha_index = _far3d_species(
        profiles, alpha_on, beam_index=beam_index, alpha_index=alpha_index, main_index=main_index, impurity_index=impurity_index)

    nr = p["rho(-)"].shape[0]
    if nr > FAR3D_MAX_PROFILE_ROWS:
        raise ValueError(f"[FAR3D] ae_profiles reads at most {FAR3D_MAX_PROFILE_ROWS} rows, the gacode state has {nr}")
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

    print(f"\t- FAR3D external profiles written to {file} ({nr} rows, alpha_on = {alpha_on})")
    print(f"\t\t* main ion: {p['name'][main_index]}, impurity: {'-' if impurity_index is None else p['name'][impurity_index]}, "
          f"beam (EP 1): {'-' if beam_index is None else p['name'][beam_index]}"
          + (f", alpha (EP 2): {'-' if alpha_index is None else p['name'][alpha_index]}" if alpha_on == 1 else ""))

    return data


def _far3d_species(profiles, alpha_on, beam_index=None, alpha_index=None, main_index=None, impurity_index=None):
    """Indices (main, impurity, beam = EP 1, alpha = EP 2) into the gacode ion list, None where absent (see write_far3d_profiles)."""

    types = [str(t) for t in profiles.profiles["type"]]
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

    return main_index, impurity_index, beam_index, alpha_index


# ************************************************************************************************************************************************
# VMEC -> Boozer coordinates (woutb)
# ************************************************************************************************************************************************

def run_booz_xform(folder, ext="input_vmec", mboz=0, nboz=0, surfaces=None, far3d_root=None):
    """
    Run FAR3D's xbooz_xform (BOOZ_XFORM with Luis Garcia's "far" extension) on folder/wout_<ext>.nc.
    It runs inside folder/booz/ (wout_<ext>.nc linked in), so everything it produces stays there:
        in_booz.<ext>           BOOZ_XFORM input
        booz.log                xbooz_xform in_booz.<ext> far  output
        boozmn.<ext>, prfeq_vmec.txt, *_booz.txt, R_Z_*.txt, ...   diagnostics
    except woutb, the Boozer equilibrium FAR3D reads (eq_name), which is moved to folder.

    mboz, nboz: Boozer poloidal / toroidal modes; xbooz_xform raises them to at least 6*mpol and 2*ntor-1,
                so 0 means its default.
    surfaces:   VMEC full-grid surfaces to transform (1-based); None = all.
    The executable is found with far3d_executable("xbooz_xform", far3d_root), i.e. FAR3D_ROOT by default.
    """
    import shutil
    import subprocess

    folder = Path(folder).expanduser()
    wout = folder / f"wout_{ext}.nc"
    if not wout.exists():
        raise FileNotFoundError(f"[FAR3D] {wout} not found, run gacode_to_vmec first")
    exe = far3d_executable("xbooz_xform", far3d_root=far3d_root)

    booz = folder / "booz"
    if booz.exists():
        shutil.rmtree(booz)
    booz.mkdir()
    (booz / wout.name).symlink_to(Path("..") / wout.name)

    lines = [f"{mboz} {nboz}", f"'{ext}'"]
    if surfaces is not None:
        lines += [" ".join(str(int(js)) for js in surfaces[i:i + 10]) for i in range(0, len(surfaces), 10)]
    (booz / f"in_booz.{ext}").write_text("\n".join(lines) + "\n")

    woutb = folder / "woutb"
    woutb.unlink(missing_ok=True)
    with open(booz / "booz.log", "w") as log:
        result = subprocess.run([str(exe), f"in_booz.{ext}", "far"], cwd=booz, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not (booz / "woutb").exists():
        raise RuntimeError(f"[FAR3D] xbooz_xform failed (exit code {result.returncode}), see {booz / 'booz.log'}")
    (booz / "woutb").replace(woutb)

    print(f"\t- Boozer equilibrium written to {woutb} (xbooz_xform input, log and diagnostics in {booz})")

    return woutb


# ************************************************************************************************************************************************
# FAR3D Input_Model
#   Read strictly in order: every value line follows one "!!!!!!!!!!! key: description" comment line. The labels
#   are only comments to the Fortran reader, so values are located by label and lines are never added, dropped or reordered.
# ************************************************************************************************************************************************

FAR3D_MAX_PROFILE_ROWS = 500  # ae_profiles stops if the external profile file has more rows

# Same constants as ae_profiles (equilibrium.f90), so the normalized inputs match what FAR3D prints
QE_FAR3D, MP_FAR3D = 1.602e-19, 1.672e-27

_INPUT_MODEL_LABEL = re.compile(r"^!+\s*([A-Za-z0-9_]+)(\([^)]*\))?\s*:")

# Defaults on top of the template. Numerics follow a working FAR3D ARC case (linear, packed grid around rho = 0.625,
# VMEC data extrapolated over the outer 10%, dt0 = 2 for 1000 steps); every one can be overridden with params.
INPUT_MODEL_DEFAULTS = {
    "nstres": 0, "numrun": "00 00", "numruno": "00 00 z", "eq_name": "woutb",
    "maxstp": 1000, "dt0": 2.0, "nprint": 100, "ndump": 1000, "ndiag": 1000, "lplots": 8, "itime": 2,
    "nonlin": 0, "Auto_grid_on": 0, "delta": 0.25, "rc": 0.625,
    "stdifp": "1.E-7", "stdifu": "1.E-7", "stdifv": "1.E-7", "stdifnf": "1.E-7", "stdifvf": "1.E-7",
    "s": "5.e6", "gamma": 0.0, "ipert": 1, "ietaeq": 1, "betath_factor": 1.0,
    "ext_prof": 1, "DIIID_u": 0, "trapped_on": 0,
    "LcA0": 2.718, "LcA1": -1.311, "LcA2": 0.889, "LcA3": 1.077,
    "epflr_on": 0, "iflr_on": 0, "ieldamp_on": 0, "twofl_on": 0, "omegar": 0.4,
    "EP_dens_on": 0, "EP_vel_on": 0, "Alpha_dens_on": 0, "Alpha_vel_on": 0,
    "q_prof_on": 0, "Eq_vel_on": 0, "Eq_velp_on": 0, "Eq_Presseq_on": 0, "Eq_Presstot_on": 0,
    "deltaq": 0.0, "deltaiota": 0.0, "Edge_on": 1,
    "matrix_out": ".false.",
}


def _input_model_line(lines, key):
    hits = [i for i, ln in enumerate(lines) if (m := _INPUT_MODEL_LABEL.match(ln)) and m.group(1).lower() == key.lower()]
    if len(hits) != 1:
        raise KeyError(f"[FAR3D] Input_Model key '{key}' found {len(hits)} times (expected 1)")
    return hits[0] + 1


def input_model_get(text, key):
    """Value of key in Input_Model text."""
    lines = text.split("\n")
    return lines[_input_model_line(lines, key)].strip()


def input_model_set(text, params):
    """Input_Model text with the value line of each key in params replaced."""
    lines = text.split("\n")
    for key, value in params.items():
        lines[_input_model_line(lines, key)] = str(value)
    return "\n".join(lines)


def fortran_list(values):
    """Comma-separated list with Fortran repeat counts (e.g. 4*1) for runs of equal values."""
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        out.append(f"{j - i + 1}*{values[i]}" if j > i else f"{values[i]}")
        i = j + 1
    return ",".join(out)


def far3d_normalizations(profiles, alpha_on=0, beam_index=None, alpha_index=None, main_index=None):
    """
    On-axis normalized FAR3D inputs from a gacode state, mirroring ae_profiles. FAR3D does NOT take omcy / bet0_f
    from the profile file (it only prints its own omgcya / betalf), so they must go into Input_Model.
    Time is normalized to tau_A0 = R0 / v_A0, with v_A0 from the on-axis main-ion mass density.

    For the EP species (beam = EP 1; alpha = EP 2 with alpha_on = 1), T is the effective temperature p_fast / n_fast:
        omcy, omcyalp           Z e B0 / m_EP * tau_A0
        bet0_f, bet0_alp        2 mu0 n T / B0^2 on axis (bet0_f = FAR3D's betalf)
        r_epflr, r_epflralp     m_EP v_th / (Z e B0) / a,  v_th = sqrt(T / m_EP)
        iflr                    main-ion Larmor radius / a
    Also returns va0 [m/s], tauA [s] and f_kHz (code frequency -> kHz).
    """
    p = profiles.profiles
    main_index, _, beam_index, alpha_index = _far3d_species(profiles, alpha_on, beam_index=beam_index, alpha_index=alpha_index, main_index=main_index)

    B0, R0, a = np.abs(float(p["bcentr(T)"][-1])), float(p["rcentr(m)"][-1]), float(p["rmin(m)"][-1])
    m_i = float(p["mass"][main_index]) * MP_FAR3D
    n_i0 = p["ni(10^19/m^3)"][0, main_index] * 1e19
    T_i0 = p["ti(keV)"][0, main_index] * 1e3 * QE_FAR3D

    va0 = B0 / np.sqrt(MU0 * m_i * n_i0)
    tauA = R0 / va0
    out = dict(va0=va0, tauA=tauA, f_kHz=va0 / (2e3 * np.pi * R0), iflr=m_i * np.sqrt(T_i0 / m_i) / (QE_FAR3D * B0) / a)

    def ep(i):
        if i is None:
            return 0.0, 0.0, 0.0, 1
        m, Z = float(p["mass"][i]) * MP_FAR3D, float(p["z"][i])
        n0, T0 = p["ni(10^19/m^3)"][0, i] * 1e19, p["ti(keV)"][0, i] * 1e3 * QE_FAR3D
        return Z * QE_FAR3D * B0 / m * tauA, 2 * MU0 * n0 * T0 / B0**2, np.sqrt(T0 * m) / (Z * QE_FAR3D * B0) / a, round(float(p["mass"][i]))

    out["omcy"], out["bet0_f"], out["r_epflr"], out["spe1"] = ep(beam_index)
    if alpha_on == 1:
        out["omcyalp"], out["bet0_alp"], out["r_epflralp"], out["spe2"] = ep(alpha_index)

    return out


def far3d_mode_lists(n, q, rho, rho_max=0.9, dm=2, meq=20, m_range=None):
    """
    FAR3D mode lists for a single toroidal mode number n. Poloidal harmonics cover the resonant surfaces m ~ n q(rho)
    for rho <= rho_max:  m in [max(1, floor(n q_min) - dm), ceil(n q_max) + dm]. Following the v2.0 convention:
        mm = +m (nn = +n) ..., -m (nn = -n) ..., then the n = 0 family -meq..meq
        mmeq = -meq..meq with nneq = 0 (meq must stay below the Boozer mode count in woutb)
    m_range = (m_lo, m_hi) sets an explicit poloidal window instead (q, rho, rho_max and dm are then unused).
    Returns the Input_Model params lmax, leqmax, mm, nn, mmeq, nneq, widthi, gammai.
    """
    import math

    if m_range is None:
        sel = rho <= rho_max
        qmin, qmax = np.min(q[sel]), np.max(q[sel])
        m_lo, m_hi = max(1, math.floor(n * qmin) - dm), math.ceil(n * qmax) + dm
        origin = f"q = {qmin:.3f} .. {qmax:.3f} for rho <= {rho_max}"
    else:
        m_lo, m_hi = m_range
        origin = "explicit window"
    ms = list(range(m_hi, m_lo - 1, -1))
    eq = list(range(-meq, meq + 1))
    mm = ms + [-m for m in ms] + eq
    nn = [n] * len(ms) + [-n] * len(ms) + [0] * len(eq)

    print(f"\t- n = {n}: {origin} -> m = {m_lo} .. {m_hi}, lmax = {len(mm)}")

    return {
        "lmax": len(mm), "leqmax": len(eq),
        "mm": ",".join(map(str, mm)), "nn": fortran_list(nn),
        "mmeq": ",".join(map(str, eq)), "nneq": f"{len(eq)}*0",
        "widthi": f"{len(mm)}*1.e-140",  # seed every mode
        "gammai": f"{len(mm)}*0.0",
    }


def write_far3d_input_model(profiles, file, profiles_file="far3d_profiles.txt", n=None, wout=None, template=None, mj=1000,
                            alpha_on=0, beam_index=None, alpha_index=None, main_index=None, mode_options={}, params={},
                            far3d_root=None):
    """
    Write a FAR3D Input_Model consistent with the external profiles file (write_far3d_profiles) and woutb (run_booz_xform).

    Starting from template (default: FAR3D_ROOT/Models/DIIID/Input_Model, which only provides the v2.0 line layout), sets
        INPUT_MODEL_DEFAULTS                    numerics of a working linear ARC case
        mj and the grid ni, nis, ne, edge_p     mj//2 - 1, mj//4 + 1, mj//4, 0.9*mj
        ext_prof_name, alpha_on                 the profiles file and its layout
        spe1, omcy, bet0_f, r_epflr, iflr       (+ spe2, omcyalp, bet0_alp, r_epflralp with alpha_on = 1) from far3d_normalizations
        lmax, leqmax, mm, nn, mmeq, ...         for toroidal mode n (far3d_mode_lists, kwargs in mode_options) with q from
                                                VMEC's half grid when wout is given (what FAR3D reads from woutb), else from gacode.
                                                With n = None the template mode lists are kept.
    and finally params (any Input_Model key), which overrides all of the above.
    The species kwargs must match those given to write_far3d_profiles.
    """
    import os

    profiles = _as_gacode_state(profiles)
    if template is None:
        far3d_root = far3d_root if far3d_root is not None else os.environ.get("FAR3D_ROOT")
        if far3d_root is None:
            raise FileNotFoundError("[FAR3D] No Input_Model template: pass template, or set FAR3D_ROOT (uses Models/DIIID/Input_Model)")
        template = Path(far3d_root).expanduser() / "Models" / "DIIID" / "Input_Model"

    norm = far3d_normalizations(profiles, alpha_on=alpha_on, beam_index=beam_index, alpha_index=alpha_index, main_index=main_index)

    values = dict(INPUT_MODEL_DEFAULTS)
    values.update({
        "mj": mj, "ni": mj // 2 - 1, "nis": mj // 4 + 1, "ne": mj // 4, "edge_p": int(0.9 * mj),
        "ext_prof_name": Path(profiles_file).name, "alpha_on": alpha_on,
        "spe1": norm["spe1"], "omcy": f"{norm['omcy']:.4f}", "bet0_f": f"{norm['bet0_f']:.5e}",
        "r_epflr": f"{norm['r_epflr']:.5e}", "iflr": f"{norm['iflr']:.5e}",
    })
    if alpha_on == 1:
        values.update({"spe2": norm["spe2"], "omcyalp": f"{norm['omcyalp']:.4f}", "bet0_alp": f"{norm['bet0_alp']:.5e}",
                       "r_epflralp": f"{norm['r_epflralp']:.5e}"})

    if n is not None:
        if wout is not None:
            s_half = (np.arange(1, wout.ns) - 0.5) / (wout.ns - 1)
            q, rho = 1.0 / np.abs(wout.iotas[1:]), np.sqrt(s_half)
        else:
            q, rho = np.abs(profiles.profiles["q(-)"]), profiles.profiles["rho(-)"]
        values.update({"numrun": f"{n // 100:02d} {n % 100:02d}", **far3d_mode_lists(n, q, rho, **mode_options)})

    values.update(params)

    text = input_model_set(Path(template).expanduser().read_text(), values)
    Path(file).write_text(text)

    print(f"\t- FAR3D Input_Model written to {file} ({'template mode lists' if n is None else f'n = {n}'}, mj = {values['mj']})")
    print(f"\t\t* v_A0 = {norm['va0']:.4e} m/s, tau_A0 = {norm['tauA']:.4e} s (code frequency -> kHz x {norm['f_kHz']:.2f})")
    print(f"\t\t* omcy = {norm['omcy']:.3f}, bet0_f = {norm['bet0_f']:.4e}, r_epflr = {norm['r_epflr']:.4e}, iflr = {norm['iflr']:.4e}"
          + (f"; omcyalp = {norm['omcyalp']:.3f}, bet0_alp = {norm['bet0_alp']:.4e}" if alpha_on == 1 else ""))

    return text
