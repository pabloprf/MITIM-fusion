"""
test_lengyel_clean_beat.py
==========================
Runs a real (minimal) MAESTRO chain with the lengyel beat in mode='clean'
(non-detached forward conduction Tsep, impurities untouched) feeding a
sharpness beat -- the physics-based separatrix temperature then sets the
scale of the sharpness boundary condition (Te_bc = Tsep/(1 - xi*C)):

    init (FreeGS + fixed BC) -> lengyel (clean) -> sharpness (xi=1)

Checks printed at the end: the Tsep the lengyel beat applied, that the ion
densities are bit-identical through the beat (impurities untouched), and the
sharpness beat's resulting Te_bc. Run folder KEPT for inspection:

    mitim_plot_maestro tests/scratch/dev_lengyel_clean_beat

Usage
-----
    ./run_with_env.sh python tests/dev_tests/test_lengyel_clean_beat.py
"""

import sys
import numpy as np
from mitim_modules.maestro.scripts import run_maestro
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools

folder = __mitimroot__ / "tests" / "scratch" / "dev_lengyel_clean_beat"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Namelist: minimal chain, lengyel in clean (tsep-only) mode
# ---------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(template)

nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV

nml["maestro"]["beats"] = ["lengyel", "sharpness"]

nml["maestro"]["lengyel"]["parameters_prepare"]["mode"] = "clean"

nml["maestro"]["sharpness"]["parameters_prepare"]["sharpness"] = 1.0

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
# Summary (stdout is left redirected to Outputs/maestro.log by MAESTRO)
# ---------------------------------------------------------------------------

sys.stdout = sys.__stdout__

p_before = PROFILEStools.gacode_state(folder / "Beats" / "Beat_1" / "initializer_freegs" / "input.gacode")
p_after = PROFILEStools.gacode_state(folder / "Beats" / "Beat_1" / "beat_results" / "input.gacode")
r2 = np.load(folder / "Beats" / "Beat_2" / "beat_results" / "sharpness_results.npy",
             allow_pickle=True).item()

ni_identical = np.array_equal(p_before.profiles["ni(10^19/m^3)"], p_after.profiles["ni(10^19/m^3)"])
ne_identical = np.array_equal(p_before.profiles["ne(10^19/m^3)"], p_after.profiles["ne(10^19/m^3)"])

print("\n----- lengyel(clean) -> sharpness chain -----")
print(f"Tesep: initializer {p_before.profiles['te(keV)'][-1]*1e3:.1f} eV -> "
      f"lengyel clean {p_after.profiles['te(keV)'][-1]*1e3:.1f} eV")
print(f"densities untouched through lengyel beat: ne {ne_identical}, ni {ni_identical}")
print(f"sharpness beat: Te_sep {r2['Te_sep']*1e3:.1f} eV, C {r2['C']:.4f} "
      f"-> Te_bc {r2['Te_bc']*1e3:.1f} eV (xi={r2['sharpness']:.2f})")
print(f"\nRun folder kept: {folder}")
