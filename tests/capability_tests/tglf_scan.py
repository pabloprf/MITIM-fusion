"""
CAPABILITY: TGLF parameter scans
--------------------------------
This script teaches how to scan an input.tglf parameter at several radii,
starting from an input.gacode file.

Key teaching points:
    1. TGLF(rhos=[...]) + prep(input.gacode, ...) generates one input.tglf per
       requested radius from the plasma state.
    2. run_scan() varies one input.tglf parameter (`variable`) by the relative
       factors in `varUpDown`, around the base value at each radius.
    3. read_scan(label=..., save_and_cleanup=...) collects the scan into a
       single .npz; plot_scan() shows fluxes and spectra vs the scanned
       parameter.
"""

import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared input files and scan subfolders live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_scan"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
npz_file = folder / "scan_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from an input.gacode
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state and writes one input.tglf per requested rho into the folder
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Scan the electron temperature gradient
# ---------------------------------------------------------------------------------------------------------------------

# `variable` is the name of any parameter of the input.tglf file (open the generated
# input.tglf in the run folder, or see https://gafusion.github.io/doc/tglf/tglf_list.html
# for the full list). Here RLTS_1 = a/LTs of species 1, i.e. the normalized temperature
# gradient of electrons (in TGLF convention species 1 is electrons; ions follow as 2, 3, ...).
# `varUpDown` gives the multipliers applied to the base value at each radius.

tglf.run_scan(
    # Name of the subfolder (inside the working folder) where the scan points live
    subfolder="scan_aLTe",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy);
    # extraOptions could also be passed here, exactly as in a single run
    code_settings="SAT2",
    # Parameter of the input.tglf file to scan (see comment above)
    variable="RLTS_1",
    varUpDown=np.linspace(0.7, 1.3, 5),  # scan from 70% to 130% of the base a/LTe
    cold_start=cold_start,
)
# read_scan() parses all scan points and stores them in the object under the label
tglf.read_scan(label="scan_aLTe", variable="RLTS_1", save_and_cleanup=npz_file)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the scan (fluxes and spectra vs a/LTe at each radius)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI
tglf.plot_scan(labels=["scan_aLTe"])
tglf.fn.show()
