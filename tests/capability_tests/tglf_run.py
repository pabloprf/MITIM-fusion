"""
CAPABILITY: Standalone TGLF runs
--------------------------------
This script teaches how to run TGLF directly from an existing input.tglf file
and compare physics settings.

Key teaching points:
    1. `code_settings` selects a preset (saturation rule + associated controls)
       from templates/input.tglf.models.yaml (e.g. "SAT0", "SAT1", "SAT2", "SAT3").
    2. `extraOptions` overrides individual input.tglf parameters on top of the
       preset — here used to toggle electromagnetic effects.
    3. read(label=..., save_and_cleanup=...) stores each run under a label in a
       single .npz file and removes the raw run folders; from_npz() restores the
       object later for plotting without re-running.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_run"
input_tglf = __mitimroot__ / "tests" / "data" / "input.tglf"
npz_file = folder / "tglf_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare the TGLF object from an existing input.tglf
# ---------------------------------------------------------------------------------------------------------------------

tglf = TGLFtools.TGLF()
tglf.prep_from_file(folder, input_tglf)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run with different settings and read each one under a label
# ---------------------------------------------------------------------------------------------------------------------

# Electrostatic run with the SAT1 saturation rule
tglf.run(
    "es_sat1/",
    code_settings="SAT1",
    extraOptions={"USE_BPER": False, "USE_BPAR": False},
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf.read(label="ES (SAT1)", save_and_cleanup=npz_file)

# Electromagnetic run with the SAT3 saturation rule
tglf.run(
    "em_sat3/",
    code_settings="SAT3",
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf.read(label="EM (SAT3)", save_and_cleanup=npz_file)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Restore from the .npz and plot both runs together
# ---------------------------------------------------------------------------------------------------------------------

tglf_loaded = TGLFtools.TGLF.from_npz(npz_file)
tglf_loaded.plot(labels=["ES (SAT1)", "EM (SAT3)"])
tglf_loaded.fn.show()
