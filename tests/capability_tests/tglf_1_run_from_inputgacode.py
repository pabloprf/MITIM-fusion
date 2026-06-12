"""
CAPABILITY: Standalone TGLF runs from an input.gacode
-----------------------------------------------------
This script teaches how to run TGLF starting from a plasma state (input.gacode)
instead of a ready-made input.tglf.

Key teaching points:
    1. TGLF(rhos=[...]) + prep(input.gacode, ...) generates one input.tglf per
       requested radius from the profiles and geometry of the plasma state
       (look inside the run folder to see them).
    2. Because the plasma state is attached, results carry the experimental
       normalizations: fluxes can be plotted in both gyro-Bohm and physical
       units (contrast with tglf_2_run_from_tglfinput.py, gyro-Bohm only).
       Those extra plots in real units are added by default.
    3. The same code_settings/extraOptions hierarchy applies: controls file
       defaults -> code_settings preset -> extraOptions overrides.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared input files and run subfolders live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_run_from_inputgacode"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations (to recover fluxes in physical units)
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run and read (no save_and_cleanup here: the run folders are kept on disk)
# ---------------------------------------------------------------------------------------------------------------------

tglf.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "sat2_em/",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy):
    # SAT2 saturation rule + the controls associated to it
    code_settings="SAT2",
    # Individual input.tglf parameters, applied on top of the preset (level 3):
    # here, include perpendicular magnetic fluctuations (electromagnetic run)
    extraOptions={"USE_BPER": True},
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
)
# read() parses the TGLF output files and stores the results in the object under the label
tglf.read(label="EM (SAT2)")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot (fluxes appear both in gyro-Bohm and physical units thanks to the normalizations)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI
tglf.plot(labels=["EM (SAT2)"])
tglf.fn.show()

