"""
CAPABILITY: Automatic scan of TGLF turbulence drives
----------------------------------------------------
This script teaches how to scan, in one call, the main drives of turbulence
at several radii, starting from an input.gacode file. This is useful to
understand what the dominant instability drives are at each location
(stiffness with respect to each gradient, collisionality, etc.).

Key teaching points:
    1. runScanTurbulenceDrives() launches one scan per drive. By default it
       scans RLTS_1 (a/LTe), RLTS_2 (a/LTi), RLNS_1 (a/Lne), XNUE
       (electron collisionality) and TAUS_2 (Ti/Te), each varied around its
       base value; the list can be changed with `variablesDrives`.
    2. The variation grid is controlled with `variation` (relative amplitude,
       default 0.5 = +-50%) and `resolutionPoints` (points per drive,
       default 5).
    3. plotScanTurbulenceDrives() collects all the individual scans in a
       single notebook, with fluxes vs each drive.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one scan subfolder per drive lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_turbulence_drives"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Scan all turbulence drives at once
# ---------------------------------------------------------------------------------------------------------------------

tglf.runScanTurbulenceDrives(
    # Base name of the scan subfolders (inside the working folder): one per drive,
    # e.g. turb_drives_RLTS_1, turb_drives_RLTS_2, ...
    subfolder="turb_drives",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy);
    # extraOptions could also be passed here, exactly as in a single run
    code_settings="SAT2",
    # 3 points per drive (70%, 100%, 130% of the base value) to keep this example cheap;
    # increase resolutionPoints (default 5) and variation (default 0.5) for real analyses
    resolutionPoints=3,
    variation=0.3,
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot fluxes vs each drive at each radius
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI.
# plotTGLFs=False skips the per-scan-point TGLF notebooks (set True to inspect each point)
tglf.plotScanTurbulenceDrives(label="turb_drives", plotTGLFs=False)
tglf.fn.show()

