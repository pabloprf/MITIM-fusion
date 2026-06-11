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

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_scan"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
npz_file = folder / "scan_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from an input.gacode
# ---------------------------------------------------------------------------------------------------------------------

tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Scan the electron temperature gradient (RLTS_1 = a/LTe; species 1 is electrons)
# ---------------------------------------------------------------------------------------------------------------------

tglf.run_scan(
    subfolder="scan_aLTe",
    code_settings="SAT2",
    variable="RLTS_1",
    varUpDown=np.linspace(0.7, 1.3, 5),  # scan from 70% to 130% of the base value
    cold_start=cold_start,
)
tglf.read_scan(label="scan_aLTe", variable="RLTS_1", save_and_cleanup=npz_file)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Restore from the .npz and plot the scan
# ---------------------------------------------------------------------------------------------------------------------

tglf_loaded = TGLFtools.TGLF.from_npz(npz_file)
tglf_loaded.plot_scan(labels=["scan_aLTe"])
tglf_loaded.fn.show()
