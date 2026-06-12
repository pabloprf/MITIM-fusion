"""
CAPABILITY: Standalone TGLF runs from an existing input.tglf
------------------------------------------------------------
This script teaches how to run TGLF directly from a ready-made input.tglf file
and compare physics settings.

Key teaching points:
    1. prep_from_file() takes an existing input.tglf as-is (no plasma state
       needed). Results are in gyro-Bohm units only, since there is no
       experimental normalization attached (see tglf_1_run_from_inputgacode.py
       for the alternative).
    2. The input file each run receives is built in three levels, each
       overriding the previous one: the controls file
       (templates/input.tglf.controls, full defaults) -> the `code_settings`
       preset (templates/input.tglf.models.yaml, e.g. saturation rules
       "SAT0"..."SAT3") -> `extraOptions` (individual parameters, the final
       word — here used to toggle electromagnetic effects).
    3. read(label=..., save_and_cleanup=...) stores each run under a label in a
       single .npz file and removes the raw run folders. The results stay in
       memory for plotting; the .npz allows reloading later without re-running
       (see tglf_3_read_fromnpz.py).
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: each run() call below creates a subfolder in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_run_from_tglfinput"
input_tglf = __mitimroot__ / "tests" / "data" / "input.tglf"
npz_file = folder / "tglf_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare the TGLF object from an existing input.tglf
# ---------------------------------------------------------------------------------------------------------------------

# prep_from_file() stages the given input.tglf into the working folder and reads it;
# no plasma state is attached (results in gyro-Bohm units only)
tglf = TGLFtools.TGLF()
tglf.prep_from_file(folder, input_tglf)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run with different settings and read each one under a label
# ---------------------------------------------------------------------------------------------------------------------

# Electrostatic run with the SAT1 saturation rule
tglf.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "es_sat1/",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy):
    # saturation rule + the controls associated to it
    code_settings="SAT1",
    # Individual input.tglf parameters, applied on top of the preset (level 3):
    # here, turn off perpendicular and parallel magnetic fluctuations (electrostatic run)
    extraOptions={"USE_BPER": False, "USE_BPAR": False},
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
)
# read() parses the TGLF output files and stores the results in the object under the label
tglf.read(label="ES (SAT1)", save_and_cleanup=npz_file)

# Electromagnetic run with the SAT3 saturation rule (same pattern as above, with
# magnetic fluctuations turned on)
tglf.run(
    "em_sat3/",
    code_settings="SAT3",
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf.read(label="EM (SAT3)", save_and_cleanup=npz_file)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot both runs together (fluxes, spectra, eigenvalues per label)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI
tglf.plot(labels=["ES (SAT1)", "EM (SAT3)"])
tglf.fn.show()

