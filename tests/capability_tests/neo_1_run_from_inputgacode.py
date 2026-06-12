"""
CAPABILITY: Standalone NEO runs from an input.gacode
----------------------------------------------------
This script teaches how to run the NEO neoclassical code starting from a
plasma state (input.gacode) at selected radii.

Key teaching points:
    1. NEO(rhos=[...]) + prep(input.gacode, ...) generates one input.neo per
       requested radius from the plasma state, same pattern as TGLF.
    2. The same three-level settings hierarchy applies: controls file
       (templates/input.neo.controls, full defaults) -> `code_settings` preset
       (templates/input.neo.models.yaml, e.g. "Sonic" -> ROTATION_MODEL=2) ->
       `extraOptions` (individual parameters, the final word).
    3. `multipliers` is an alternative to `extraOptions` for plasma parameters:
       instead of setting an absolute value, it multiplies the base value that
       prep() derived from the plasma state at each radius.
    4. run_scan()/read_scan()/plot_scan() work exactly as for TGLF (see
       tglf_5_scan.py), here used to scan the main-ion temperature gradient.
"""

import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: each run() call below creates a subfolder in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_neo_run_from_inputgacode"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare NEO at several radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.neo per requested rho into the folder
# and attaches the experimental normalizations
neo = NEOtools.NEO(rhos=np.linspace(0.5, 0.9, 5))
neo.prep(input_gacode, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run with different settings and read each one under a label
# ---------------------------------------------------------------------------------------------------------------------

# Preset only (level 2 of the hierarchy): "Sonic" is defined in
# templates/input.neo.models.yaml and solves the Hinton-Wong generalized
# equations (ROTATION_MODEL=2). Omitting code_settings entirely would run with
# the bare controls-file defaults (level 1). The first argument is the name of
# the subfolder (inside the working folder) where this run lives
neo.run("sonic/", code_settings="Sonic", cold_start=cold_start)
# read() parses the NEO output files and stores the results in the object under the label
neo.read(label="Sonic")

# Same preset, now lowering the velocity-space resolution on top of it via
# extraOptions (level 3): individual input.neo parameters, the final word
neo.run(
    "sonic_lowres/",
    code_settings="Sonic",
    extraOptions={"N_ENERGY": 5, "N_XI": 11, "N_THETA": 11},
    cold_start=cold_start,
)
neo.read(label="Sonic low res")

# Same as above, additionally increasing the temperature gradient of species 1
# (first ion) by 50%: multipliers scale the base value that prep() derived from
# the plasma state, instead of setting an absolute value like extraOptions
neo.run(
    "sonic_lowres_aLT/",
    code_settings="Sonic",
    extraOptions={"N_ENERGY": 5, "N_XI": 11, "N_THETA": 11},
    multipliers={"DLNTDR_1": 1.5},
    cold_start=cold_start,
)
neo.read(label="Sonic low res + 50% aLTi")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot all runs together
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (neo.fn); show() opens the GUI
neo.plot(labels=["Sonic", "Sonic low res", "Sonic low res + 50% aLTi"])

# ---------------------------------------------------------------------------------------------------------------------
# 4. Scan a parameter, exactly as in the TGLF scans
# ---------------------------------------------------------------------------------------------------------------------

# DLNTDR_1 is the input.neo name of the normalized temperature gradient (a/LT) of
# species 1 (first ion); varUpDown gives the multipliers applied to the base value
neo.run_scan(
    "scan_aLTi",
    variable="DLNTDR_1",
    varUpDown=np.linspace(0.5, 1.5, 4),
    cold_start=cold_start,
)
neo.read_scan(label="scan_aLTi", variable="DLNTDR_1")

# Adding the scan figures to the same notebook as the runs above (fn=neo.fn)
neo.plot_scan(labels=["scan_aLTi"], fn=neo.fn)
neo.fn.show()
