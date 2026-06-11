"""
CAPABILITY: Linear GX run from an input.gacode
----------------------------------------------
This script teaches how to run a (cheap) linear simulation with GX — the
GPU-native gyrokinetic code — starting from a plasma state (input.gacode).
GX runs on the machine configured for "gx" in config_user.json (it needs
GPUs; here each radius uses 4).

Key teaching points:
    1. GX(rhos=[...]) + prep(plasma_state, ...) follows the exact same pattern
       as TGLF/NEO/CGYRO: one input file per requested radius, experimental
       normalizations attached. The same settings hierarchy applies: controls
       file (templates/input.gx.controls) -> `code_settings` preset
       (templates/input.gx.models.yaml) -> `extraOptions`.
    2. Unlike CGYRO, GX computes the whole linear ky spectrum in a SINGLE run:
       the grid is set through y0 (kymin = 1/y0) and ny (nky = 1 + (ny-1)/3).
       See linear_gk_cgyro_vs_gx.py for a direct comparison of both codes.
    3. lumpIons() bundles all ions of the plasma state into a single effective
       species, making the runs cheaper.
    4. The save_figures switch shows the headless pattern for non-interactive
       environments (e.g. testing on an HPC node): save the notebook to a
       folder instead of opening the GUI.
"""

from mitim_tools.gacode_tools.PROFILEStools import gacode_state
from mitim_tools.simulation_tools.physics import GXtools
from mitim_tools.misc_tools import IOtools, GUItools
from mitim_tools import __mitimroot__

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

# If True, do not show the plots on screen, save them to a subfolder instead
save_figures = False

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared inputs, remote job files and outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_gx_linear"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare the plasma state (lumped ions) and GX at two radii
# ---------------------------------------------------------------------------------------------------------------------

# Lump all ions (main + impurities) into a single effective species to make the run cheaper
p = gacode_state(input_gacode)
p.lumpIons()

# prep() reads the plasma state, writes one GX input file per requested rho into the
# folder and attaches the experimental normalizations
gx = GXtools.GX(rhos=[0.5, 0.6])
gx.prep(p, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the linear spectrum (single run per radius)
# ---------------------------------------------------------------------------------------------------------------------

gx.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "gx1/",
    # Preset from templates/input.gx.models.yaml (level 2 of the hierarchy)
    code_settings="Linear Tokamak",
    # Individual input parameters, applied on top of the preset (level 3)
    extraOptions={
        "t_max": 5.0,  # simulated time (a/c_s units): very short, just for demonstration (~1 min on 8 A100s)
        "y0": 10.0,    # kymin = 1/y0 = 0.1
        "ny": 34,      # nky = 1 + (ny-1)/3 = 12 -> ky range 0.1 - 1.2
    },
    # Resources of each GX instance (one per radius): GPUs per call, and SLURM time limit
    allocation={"resources_per_call": 4, "minutes": 10},
    cold_start=cold_start,
)

# read() parses the GX output files and stores the results in the object under the label
gx.read("gx1")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot (growth rates and real frequencies vs ky at each radius)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook; show() opens the GUI
fn = GUItools.FigureNotebook("GX", geometry="1600x1000", show=not save_figures)
gx.plot(labels=["gx1"], fn=fn)

# Show on screen, or save to a subfolder when running non-interactively
if not save_figures:
    gx.fn.show()
    gx.fn.close()
else:
    gx.fn.save(f"{folder}/figs_gx/")
