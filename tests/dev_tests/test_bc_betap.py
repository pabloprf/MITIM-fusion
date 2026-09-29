"""
test_bc_betap.py
================
Exercises the bc beat's 'betap' method (closed-form Te_bc from a prescribed edge
poloidal-beta gradient, BCbeat._run_betap). Three parts:

1) Analytic self-consistency: run the solver at the _run() level on a real
   input.gacode (mocked maestro instance), then recompute betap' from the WRITTEN
   state with an INDEPENDENT implementation of the formula
       beta_p = 2*mu0*p_th/Bpa^2,  Bpa = mu0*Ip/L_pol,
       betap'  = [beta_p(bc) - beta_p(sep)]/(1 - psin_bc)
   and assert it matches the prescribed target under BOTH density treatments
   ('bc' and 'keep'), both on the in-memory output state (machine precision) and
   on the state re-read from the written file (file-precision tolerance).

2) Validation: betap_prime at the parameters_prepare top level raises with a
   pointer to betap_parameters; an unknown key inside betap_parameters raises.

3) Chain smoke: minimal real MAESTRO run, init (FreeGS + fixed BC) -> bc (betap).
   Run folder KEPT for inspection: mitim_plot_maestro tests/scratch/dev_bc_betap

Usage
-----
    ./run_with_env.sh python tests/dev_tests/test_bc_betap.py
"""

import sys
import copy
from types import SimpleNamespace
import numpy as np

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.maestro.utils.BCbeat import bc_beat

GACODE = __mitimroot__ / "tests" / "data" / "input.gacode"
folder_scratch = __mitimroot__ / "tests" / "scratch" / "dev_bc_betap_unit"
if folder_scratch.exists():
    IOtools.shutil_rmtree(folder_scratch)
folder_scratch.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Independent implementation of the betap' measurement (deliberately separate
# from BCbeat's helpers)
# ---------------------------------------------------------------------------

def measure_betap_prime(p, rho_bc_rho):
    """Two-point betap' of state *p* between the grid point nearest rho_bc_rho and the
    separatrix, engineering norm Bpa = mu0*Ip/L_pol. Thermal pressure only."""
    e_J, mu0 = 1.602176634e-19, 4.0e-7 * np.pi

    # thermal pressure [Pa]: n(1e19)*T(keV) -> Pa
    p_th = p.profiles["ne(10^19/m^3)"] * p.profiles["te(keV)"]
    for sp in range(len(p.Species)):
        if p.Species[sp]["S"] != "fast":
            p_th = p_th + p.profiles["ni(10^19/m^3)"][:, sp] * p.profiles["ti(keV)"][:, sp]
    p_th = p_th * 1e19 * 1e3 * e_J

    Ip_A = abs(float(p.profiles["current(MA)"][-1])) * 1e6
    R = np.asarray(p.derived["R_surface"][0][-1])
    Z = np.asarray(p.derived["Z_surface"][0][-1])
    dR, dZ = np.diff(np.append(R, R[0])), np.diff(np.append(Z, Z[0]))
    L_pol = float(np.sum(np.sqrt(dR**2 + dZ**2)))
    Bpa = mu0 * Ip_A / L_pol

    beta_p = 2.0 * mu0 * p_th / Bpa**2
    ibc = int(np.argmin(np.abs(p.profiles["rho(-)"] - rho_bc_rho)))
    psin = p.derived["psi_pol_n"]
    return (float(beta_p[ibc]) - float(beta_p[-1])) / (1.0 - float(psin[ibc]))


def measure_edge_slopes(p, rho_bc_rho):
    """Local finite-difference |d(beta_p)/dpsin| between consecutive grid points on the
    rewritten edge interior (from i_edge+1 = ibc+2 to the separatrix): with the
    pressure-linear edge, every one of these slopes must equal the two-point betap'.
    (The single cell i_edge -> i_edge+1 deliberately carries the kink and is excluded.)"""
    e_J, mu0 = 1.602176634e-19, 4.0e-7 * np.pi
    p_th = p.profiles["ne(10^19/m^3)"] * p.profiles["te(keV)"]
    for sp in range(len(p.Species)):
        if p.Species[sp]["S"] != "fast":
            p_th = p_th + p.profiles["ni(10^19/m^3)"][:, sp] * p.profiles["ti(keV)"][:, sp]
    p_th = p_th * 1e19 * 1e3 * e_J
    Ip_A = abs(float(p.profiles["current(MA)"][-1])) * 1e6
    R = np.asarray(p.derived["R_surface"][0][-1])
    Z = np.asarray(p.derived["Z_surface"][0][-1])
    L_pol = float(np.sum(np.hypot(np.diff(np.append(R, R[0])), np.diff(np.append(Z, Z[0])))))
    Bpa = mu0 * Ip_A / L_pol
    beta_p = 2.0 * mu0 * p_th / Bpa**2
    psin = p.derived["psi_pol_n"]
    ibc = int(np.argmin(np.abs(p.profiles["rho(-)"] - rho_bc_rho)))
    j0 = ibc + 2   # first rewritten point
    return -np.diff(beta_p[j0:]) / np.diff(psin[j0:])


def make_mock():
    return SimpleNamespace(
        parameters_trans_beat={},
        counter_current=1,
        maestro_namelist={"plasma": {"parameters": {}}},
        prune_level=0,
    )


# ===========================================================================
# PART 1 - analytic self-consistency, both density treatments
# ===========================================================================

TARGET = 2.0
ok = True
print("===== PART 1: analytic self-consistency (target betap' = %.3f) =====" % TARGET)

for treatment in ("bc", "keep"):
    mock = make_mock()
    if treatment == "bc":
        mock.parameters_trans_beat["neped_20"] = 1.1   # exercise the prescribed-ne branch

    b = bc_beat(mock, method="betap", folder_name=folder_scratch / f"case_{treatment}")
    b.prepare(method="betap", density_treatment=treatment,
              betap_parameters={"betap_prime": TARGET})
    b.profiles_current = PROFILEStools.gacode_state(GACODE)
    b.profiles_current.derive_quantities(rederiveGeometry=True)
    results = b._run()

    # In-memory written state (machine precision expected)
    p_mem = b.profiles_output
    meas_mem = measure_betap_prime(p_mem, results["rho_bc_rho"])

    # State re-read from the written file (file precision)
    p_file = PROFILEStools.gacode_state(b.folder / "input.gacode.bc")
    p_file.derive_quantities(rederiveGeometry=True)
    meas_file = measure_betap_prime(p_file, results["rho_bc_rho"])

    dev_mem = abs(meas_mem - TARGET) / TARGET
    dev_file = abs(meas_file - TARGET) / TARGET
    pass_mem, pass_file = dev_mem < 1e-9, dev_file < 1e-4
    ok &= pass_mem and pass_file

    # Pressure-linear edge: every local slope on the rewritten edge interior equals betap'
    slopes_mem = measure_edge_slopes(p_mem, results["rho_bc_rho"])
    slopes_file = measure_edge_slopes(p_file, results["rho_bc_rho"])
    dev_sl_mem = float(np.max(np.abs(slopes_mem - TARGET)) / TARGET)
    dev_sl_file = float(np.max(np.abs(slopes_file - TARGET)) / TARGET)
    pass_sl = dev_sl_mem < 1e-6
    ok &= pass_sl
    print(f"      edge local d(beta_p)/dpsin ({slopes_mem.size} segments): "
          f"max rel dev vs target (memory) = {dev_sl_mem:.2e} [{'PASS' if pass_sl else 'FAIL'}], "
          f"(file) = {dev_sl_file:.2e}")
    print(f"  density_treatment='{treatment}': achieved (memory) = {meas_mem:.12f} "
          f"[rel dev {dev_mem:.2e}, {'PASS' if pass_mem else 'FAIL'}], "
          f"achieved (file) = {meas_file:.8f} [rel dev {dev_file:.2e}, {'PASS' if pass_file else 'FAIL'}]")
    print(f"      Te_bc = {results['Te_bc']:.4f} keV, ne_bc used = {results['ne_bc_used_1e19']:.4f} 1e19, "
          f"Bpa = {results['Bpa_T']:.4f} T, L_pol = {results['L_pol_m']:.3f} m, "
          f"delivered (incoming) betap' = {results['betap_prime_delivered']:.4f}")
    assert not results["Te_bc_at_floor"], "floor guard should not fire on this state"

# ===========================================================================
# PART 2 - validation raises
# ===========================================================================

print("===== PART 2: validation =====")
b = bc_beat(make_mock(), method="betap", folder_name=folder_scratch / "case_val")
try:
    b.prepare(method="betap", betap_prime=2.0)   # method-specific knob at top level
    raise AssertionError("top-level betap_prime did not raise")
except ValueError as e:
    assert "betap_parameters" in str(e)
    print("  top-level betap_prime raises with pointer: PASS")
try:
    b.prepare(method="betap", betap_parameters={"betap_prime_typo": 2.0})
    raise AssertionError("unknown key in betap_parameters did not raise")
except ValueError as e:
    assert "betap_parameters" in str(e)
    print("  unknown key inside betap_parameters raises: PASS")
try:
    b.prepare(method="betap", betap_parameters={"betap_prime": -1.0})
    raise AssertionError("negative betap_prime did not raise")
except ValueError:
    print("  negative betap_prime raises: PASS")

if not ok:
    print("\nPART 1 FAILED")
    sys.exit(1)

# ===========================================================================
# PART 3 - chain smoke: init (FreeGS + fixed BC) -> bc (betap)
# ===========================================================================

from mitim_modules.maestro.scripts import run_maestro

folder = __mitimroot__ / "tests" / "scratch" / "dev_bc_betap"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"
if folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

nml = IOtools.read_mitim_yaml(template)
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV

nml["maestro"]["beats"] = ["bc"]
# Method switch on the template's bc block: the confinement_parameters sub-dict it
# ships stays in place (ignored under method 'betap')
nml["maestro"]["bc"]["parameters_prepare"]["method"] = "betap"
nml["maestro"]["bc"]["parameters_prepare"]["betap_parameters"]["betap_prime"] = 2.0

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

m = run_maestro.run_maestro_local(
    namelist_file, folder=folder, terminal_outputs=True, force_cold_start=True, cpus=4,
)

sys.stdout = sys.__stdout__

r = np.load(folder / "Beats" / "Beat_1" / "beat_results" / "bc_results.npy",
            allow_pickle=True).item()
p_out = PROFILEStools.gacode_state(folder / "Beats" / "Beat_1" / "beat_results" / "input.gacode")
p_out.derive_quantities(rederiveGeometry=True)
meas_chain = measure_betap_prime(p_out, r["rho_bc_rho"])

print("\n===== PART 3: chain smoke (freegs init -> bc betap) =====")
print(f"  target betap' = {r['betap_prime']:.3f}, applied betap'_eff = {r['betap_prime_eff']:.4f}, "
      f"measured on beat_results state = {meas_chain:.4f}")
print(f"  Te_bc = {r['Te_bc']:.4f} keV (Tesep = {r['Te_sep']*1e3:.1f} eV), "
      f"Bpa = {r['Bpa_T']:.4f} T, Ip = {r['Ip_MA']:.3f} MA, L_pol = {r['L_pol_m']:.3f} m")
assert abs(meas_chain - r["betap_prime"]) / r["betap_prime"] < 1e-3, "chain-delivered betap' off target"
print(f"\nRun folder kept: {folder}")
print("ALL PASS")
