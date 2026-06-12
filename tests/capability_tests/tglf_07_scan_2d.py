"""
CAPABILITY: 2D TGLF parameter scans
-----------------------------------
This script teaches how to scan two input.tglf parameters simultaneously on a
2D grid — here the electron and ion temperature gradients, the classic
stiffness landscape — at several radii, starting from an input.gacode file.

Key teaching points:
    1. run_scan2d() runs one TGLF case per (value1, value2) combination:
       `variable1`/`variable2` are input.tglf parameter names (see tglf_05_scan.py
       for where they come from) and varUpDown1/2 are the multipliers of the
       base value at each radius — a 4x4 grid here, so 16 runs per radius.
    2. read_scan2d() aggregates everything into 2D maps per radius (fluxes of
       every channel and the linear growth rate, the latter evaluated at the
       spectrum ky closest to `ky_target`).
    3. The save_and_cleanup/.npz pattern works exactly as in 1D scans: the raw
       run folders are removed and the maps can be rebuilt from the .npz at
       any later time (as done below).
    4. plot_scan2d() shows the 2D flux/growth-rate maps vs the two scanned
       parameters at each radius.
"""

import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: the scan subfolders live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_scan2d"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
npz_file = folder / "scan2d_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state and writes one input.tglf per requested rho into the folder
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the 2D scan: a/LTe x a/LTi grid
# ---------------------------------------------------------------------------------------------------------------------

tglf.run_scan2d(
    # Name of the subfolder (inside the working folder) where the scan points live
    subfolder="scan2d",
    # The two input.tglf parameters of the grid: electron (RLTS_1) and ion (RLTS_2)
    # normalized temperature gradients, each from 50% to 150% of the base value
    variable1="RLTS_1",
    varUpDown1=np.linspace(0.5, 1.5, 4),
    variable2="RLTS_2",
    varUpDown2=np.linspace(0.5, 1.5, 4),
    # None runs with the bare controls-file defaults; any preset/extraOptions
    # combination can be passed here, exactly as in a single run
    code_settings=None,
    cold_start=cold_start,
    # Store everything in a single .npz and remove the raw run folders
    save_and_cleanup=npz_file,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Reload from the .npz, aggregate the 2D maps and plot
# ---------------------------------------------------------------------------------------------------------------------

# The raw folders are gone (save_and_cleanup): rebuild the object from the .npz
tglf_loaded = TGLFtools.TGLF.from_npz(npz_file)

# Aggregate into (n_rhos, n1, n2) maps; the growth-rate map is taken at the
# spectrum ky closest to ky_target
tglf_loaded.read_scan2d(label="scan2d", ky_target=0.3)

# All figures go into a multi-tab MITIM FigureNotebook (tglf_loaded.fn); show() opens the GUI
tglf_loaded.plot_scan2d(label="scan2d")
tglf_loaded.fn.show()

