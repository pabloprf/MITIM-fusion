"""
test_bc_relaxation.py
=====================
Runs a real (minimal) MAESTRO chain with one bc(confinement) beat followed by one
bc(sharpness) beat, both with relaxation=0.5, to exercise the Te_bc under-relaxation
(BCbeat.relax_bc + the shared 'Te_bc_applied' trans-beat memory):

    init (FreeGS + fixed BC) -> bc method=confinement (H98y2=1) -> bc method=sharpness (xi=1)

Beat 1 has no memory -> full step; beat 2 relaxes halfway between beat 1's
applied Te_bc and its own xi=1 target (reported as xi_eff). Everything runs
locally (no TRANSP/PORTALS). The run folder is KEPT for inspection/plotting:

    mitim_plot_maestro tests/scratch/dev_bc_relaxation

Usage
-----
    ./run_with_env.sh python tests/dev_tests/test_bc_relaxation.py
"""

import sys
import numpy as np
from mitim_modules.maestro.scripts import run_maestro
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

folder = __mitimroot__ / "tests" / "scratch" / "dev_bc_relaxation"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Namelist: template + minimal chain with relaxation on both BC beats
# ---------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(template)

# Initialization as in maestro_01_run.py: FreeGS equilibrium, fixed BC profiles
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV

nml["maestro"]["beats"] = ["bc_conf", "bc_sharp"]

# Two flavors of the single 'bc' beat type: block names are free, beat_type maps them.
# Common knobs at the parameters_prepare top level; method-specific ones nested in
# '<method>_parameters' (only the selected method's sub-dict is consumed)
nml["maestro"]["bc_conf"] = {
    "beat_type": "bc", "base_module": None,
    "parameters_prepare": {"method": "confinement", "relaxation": 0.5,
                           "confinement_parameters": {"confinement_scaling": "H98y2",
                                                      "confinement": 1.0}},
    "preprocess_prepare": None, "preprocess_prepare_parameters": {}, "preprocess_run": None,
}
nml["maestro"]["bc_sharp"] = {
    "beat_type": "bc", "base_module": None,
    "parameters_prepare": {"method": "sharpness", "relaxation": 0.5,
                           "sharpness_parameters": {"sharpness": 1.0}},
    "preprocess_prepare": None, "preprocess_prepare_parameters": {}, "preprocess_run": None,
}

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ---------------------------------------------------------------------------
# Run the chain
# ---------------------------------------------------------------------------

m = run_maestro.run_maestro_local(
    namelist_file,
    folder=folder,
    terminal_outputs=True,
    force_cold_start=True,
    cpus=4,
)

# ---------------------------------------------------------------------------
# Summary: the Te_bc trail across the two beats
# ---------------------------------------------------------------------------

# MAESTRO leaves stdout redirected to Outputs/maestro.log; bring the summary
# back to the terminal
sys.stdout = sys.__stdout__

r1 = np.load(folder / "Beats" / "Beat_1" / "beat_results" / "bc_results.npy",
             allow_pickle=True).item()
r2 = np.load(folder / "Beats" / "Beat_2" / "beat_results" / "bc_results.npy",
             allow_pickle=True).item()

print("\n----- BC relaxation trail -----")
print(f"Beat 1 (confinement, relaxation={r1['relaxation']}): "
      f"target {r1['Te_bc_target']:.4f} keV -> applied {r1['Te_bc']:.4f} keV "
      f"(no memory: full step), H98y2 achieved {r1['H_achieved']:.3f}")
print(f"Beat 2 (sharpness,   relaxation={r2['relaxation']}): "
      f"target {r2['Te_bc_target']:.4f} keV -> applied {r2['Te_bc']:.4f} keV "
      f"(memory {r1['Te_bc']:.4f} keV), xi prescribed {r2['sharpness']:.2f} -> "
      f"xi_eff {r2['xi_eff']:.3f}")
print(f"\nRun folder kept: {folder}")
