'''
Dev-test: MAESTRO chain with the MINUET beat substituting transp_soft.

Runs a two-beat MAESTRO chain ["minuet", "portals"] starting PURELY from engineering
parameters (no input.gacode): the template's SPARC-like plasma.parameters (R = 1.85 m,
a = 0.57 m, Bt = 12.2 T, Ip = 8.7 MA, neped_20 = 2.5) are turned into an initial state
by a FreeGS equilibrium + the 'fixed_bc' profile creator (Te pinned at x_bc, no
pedestal code involved -- same initialization as capability_tests/maestro_01_run.py).

    1. minuet  : current diffusion + sawteeth at fixed kinetics and fixed-boundary
                 equilibrium (coupled CD+GS), via the standalone MINUET package.
                 Gaussian auxiliary sources (heating.type = gaussian_sources) are
                 injected at beat output, exactly as a transp beat would do.
    2. portals : fast flux-matching smoke configuration (4 radii, te/ti only, SAT0
                 electrostatic, 2 BO iterations, in-process ctypes TGLF).

What this verifies:
    - The minuet beat runs end-to-end inside MAESTRO from an engineering-only start.
    - A 10-moment MXH refit of the FreeGS surfaces (n_mxh = 10; template default 5)
      gives a fold-free stored family, so the beat ingests the state WITHOUT the
      folded-surface boundary trim -- and exercises the no-ceiling MXH path
      (arbitrary shape_cos{n}/shape_sin{n} columns, incl. two-digit harmonics)
      through MITIM's state layer and minuet's reader end-to-end.
    - The q-profile actually evolved from the initializer state (current diffusion).
    - The kinetic profiles passed through VERBATIM (frozen-kinetics contract).
    - The gaussian sources were injected with the engineering Pe/Pi totals.
    - The cross-beat state (sawtooth_times) was persisted for downstream beats.
    - A PORTALS beat can consume the minuet beat's output state.

Run from the dev-pixi root:
    ./run_with_env.sh python MITIM-fusion/tests/dev_tests/test_maestro_minuet_beat.py

Re-run with cold_start = False to exercise the skip/restart path (both beats should
be detected as complete and only finalize/merge re-run).
'''

import numpy as np
import torch
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.maestro.scripts import run_maestro

cold_start = True

folder = __mitimroot__ / "tests" / "scratch" / "dev_maestro_minuet_beat"

# Gaussian-source engineering powers for this test [MW]
Pe_MW, Pi_MW = 10.0, 5.0

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(8)

# ------------------------------------------------------------------------------------------------
# Build the namelist from the template
# ------------------------------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.maestro.yaml")

# Initialize from ENGINEERING PARAMETERS only: FreeGS equilibrium (template default) +
# 'fixed_bc' creator (Te pinned at x_bc; ne from neped_20; BetaN/nu_ne matched by the
# core gradients). All engineering values (R, a, Bt, Ip, neped_20, separatrix shaping)
# come from the template's plasma.parameters block.
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV (Ti_bc: null -> same as Te_bc)

# Refit the traced FreeGS surfaces with 10 MXH moments (cos0..9). The template default
# of 5 under-resolves the shaped edge: the 2 outermost stored surfaces self-intersect
# when rebuilt from the file and the minuet beat has to trim its boundary to
# rho ~ 0.991; 7 moments already give a fold-free family. There is NO moment ceiling
# anymore (2026-07-31): MITIM's state layer and minuet both consume every
# shape_cos{n}/shape_sin{n} column the file carries, and 10 exercises the
# two-digit-harmonic path end-to-end. MEASURED dose-response on this case:
# 5 -> 2 folded surfaces (trim), 7 -> fold-free but edge-marginal, 10 -> fully
# clean (0 folds, 0 FSA-kernel failures), 20 -> OVERFIT: the export-side
# 19-harmonic refit of minuet's traced surfaces amplifies tracing noise and
# breaks MITIM's geometric-factors kernel at one surface (r/a ~ 0.972).
# More moments fit noise, not shape -- 10 is the sweet spot here.
# TGLF/CGYRO/GX exports still clamp at 6 (their input schemas); this only
# smooths what THEY see, never breaks them.
# NOTE: chains WITH a transp beat share this knob as the TRANSP boundary smoothing
# (mxh_coeffs_smooth_sep: null inherits it) -- there, pin that one back to 5.
nml["plasma"]["parameters"]["separatrix"]["n_mxh"] = 10

# Gaussian auxiliary sources: determined in the namelist, injected at minuet beat output
nml["plasma"]["heating"]["type"] = "gaussian_sources"
nml["plasma"]["heating"]["parameters"]["Pe"] = Pe_MW
nml["plasma"]["heating"]["parameters"]["Pi"] = Pi_MW
nml["plasma"]["heating"]["parameters"]["nu_source"] = 5.0
nml["plasma"]["heating"]["parameters"]["fmini"] = 0.0   # no minority physics without TRANSP

# The two-beat chain
nml["maestro"]["beats"] = ["minuet", "portals"]

# Short MINUET run for the test (production default is 20 s)
nml["maestro"]["minuet"]["parameters_prepare"]["t_end"] = 3.0
nml["maestro"]["minuet"]["parameters_prepare"]["n_save"] = 101

# Fast PORTALS configuration (far too few iterations for converged results; smoke only)
pp = nml["maestro"]["portals"]["parameters_prepare"]["portals_parameters"]
pp["solution"]["predicted_roa"] = [0.35, 0.55, 0.75, 0.9]
pp["solution"]["predicted_channels"] = ["te", "ti"]
pp["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT0"
pp["transport"]["options"]["tglf"]["run"]["extraOptions"] = {"USE_BPER": False}
pp["transport"]["in_process"] = True  # ctypes TGLF/NEO, no machine dispatch
pp.setdefault("optimization_options", {}).setdefault("convergence_options", {})["maximum_iterations"] = 2

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ------------------------------------------------------------------------------------------------
# Run MAESTRO
# ------------------------------------------------------------------------------------------------

run_maestro.run_maestro_local(
    namelist_file,
    folder = folder,
    terminal_outputs = True,
    force_cold_start = cold_start,
    cpus = 8,
)

# ------------------------------------------------------------------------------------------------
# Assertions
# ------------------------------------------------------------------------------------------------

print("\n" + "="*100)
print(" Checking results")
print("="*100)

b1 = folder / "Beats" / "Beat_1"
b2 = folder / "Beats" / "Beat_2"

# Both beats produced their canonical output state
assert (b1 / "beat_results" / "input.gacode").exists(), "minuet beat produced no beat_results/input.gacode"
assert (b2 / "beat_results" / "input.gacode").exists(), "portals beat produced no beat_results/input.gacode"

# Discovery-by-folder-name contract (mitim_plot_maestro / mitim_check_maestro key on run_minuet)
assert (b1 / "run_minuet").exists(), "minuet beat run folder is not named run_minuet"

# The 10-moment MXH refit (n_mxh = 10 above) must hand minuet a fold-free family:
# no folded-surface backoff, i.e. minuet's boundary is the last stored surface.
# (With the template's 5 moments this file exists and the boundary is trimmed.)
assert not (b1 / "run_minuet" / "input.gacode_trimmed").exists(), \
    "minuet beat trimmed folded surfaces despite the 10-moment MXH refit"

# ... and the state the beat received must actually CARRY the high harmonics
# (two-digit column names survive MITIM's writer/reader and minuet's ingest)
import re as _re
_txt = (b1 / "run_minuet" / "input.gacode").read_text()
assert _re.search(r"^# *shape_cos9", _txt, _re.M), \
    "input.gacode does not carry shape_cos9 -- n_mxh=10 did not reach the file"

# Sidecar + saved discharge object persisted for plotting/inheritance
assert (b1 / "beat_results" / "minuet_results.npy").exists(), "minuet_results.npy sidecar missing"
assert (b1 / "beat_results" / "run.minuet").exists(), "run.minuet discharge object missing"

# Pre-beat state = what the minuet beat received from the initializer (written by run())
p_in = PROFILEStools.gacode_state(b1 / "run_minuet" / "input.gacode")
p_out = PROFILEStools.gacode_state(b1 / "beat_results" / "input.gacode")
p_out.derive_quantities()

def _on_input_grid(key, column=None):
    y = p_out.profiles[key] if column is None else p_out.profiles[key][:, column]
    return np.interp(p_in.profiles["rho(-)"], p_out.profiles["rho(-)"], y)

# 1) Current diffusion happened: the q-profile evolved from the initializer state
dq = np.max(np.abs(_on_input_grid("q(-)") - p_in.profiles["q(-)"]))
print(f"\t- max |dq| between initializer and minuet output: {dq:.4f}")
assert dq > 0.01, "q-profile did not evolve (no current diffusion?)"

# 2) Frozen-kinetics contract: te passed through verbatim
dte = np.max(np.abs(_on_input_grid("te(keV)") - p_in.profiles["te(keV)"]))
print(f"\t- max |dTe| between initializer and minuet output: {dte:.2e} keV")
assert dte < 1e-3 * np.max(p_in.profiles["te(keV)"]), "kinetics leaked through the minuet beat"

# 3) Gaussian sources injected with the engineering totals
qRFe = p_out.derived["qRFe_MW"][-1]
qRFi = p_out.derived["qRFi_MW"][-1]
print(f"\t- injected auxiliary power: Pe = {qRFe:.3f} MW (target {Pe_MW}), Pi = {qRFi:.3f} MW (target {Pi_MW})")
assert abs(qRFe - Pe_MW) < 0.05 * Pe_MW, "electron gaussian source total does not match Pe"
assert abs(qRFi - Pi_MW) < 0.05 * Pi_MW, "ion gaussian source total does not match Pi"

# 4) Cross-beat state persisted (sawtooth_times for downstream beats)
d = np.load(b1 / "beat_results" / "minuet_results.npy", allow_pickle=True).item()
print(f"\t- sawtooth crashes in minuet beat: {len(d['sawtooth_times'])}")
assert (folder / "Outputs" / "trans_beat_parameters" / "beat_1.json").exists(), "cross-beat snapshot missing"

print("\nPASS: MAESTRO [minuet, portals] chain completed with evolved q, frozen kinetics and gaussian sources")
