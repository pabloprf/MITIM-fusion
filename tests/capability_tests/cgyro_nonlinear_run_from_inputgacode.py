"""
CAPABILITY: Nonlinear CGYRO run from an input.gacode
----------------------------------------------------
This script teaches how to run a (cheap) nonlinear CGYRO simulation starting
from a plasma state (input.gacode). CGYRO runs on the machine configured for
it in config_user.json (possibly remote, via SLURM).

Key teaching points:
    1. The same three-level settings hierarchy as TGLF/NEO applies: controls
       file (templates/input.cgyro.controls, full defaults) -> `code_settings`
       preset (templates/input.cgyro.models.yaml) -> `extraOptions` (individual
       parameters, the final word).
    2. The nonlinear presets form a fidelity ladder via `base:` inheritance in
       the models file: "Nonlinear_high" -> "Nonlinear_reduced1/2/3" ->
       "Nonlinear_silly", each lowering resolutions and physics fidelity
       (collisions, rotation) on top of its parent. Here we use
       "Nonlinear_silly", the cheapest one, meant only for testing workflows —
       NOT for physics production.
    3. Presets also carry `preprocess_options` (ky_min, L_x, N_radial, ...)
       from which MITIM automatically builds a consistent BOX_SIZE/N_RADIAL
       perpendicular grid at each radius.
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared inputs, remote job files and outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_nonlinear"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare CGYRO at one radius from the plasma state (one radius to keep it cheap)
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.cgyro per requested rho into the folder
# and attaches the experimental normalizations
cgyro = CGYROtools.CGYRO(rhos=[0.5])
cgyro.prep(input_gacode, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run a very coarse nonlinear simulation
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "nonlinear_silly",
    # Preset from templates/input.cgyro.models.yaml (level 2 of the hierarchy): the
    # lowest-fidelity rung of the nonlinear ladder (see docstring). It resolves, via
    # `base:` inheritance, to NONLINEAR_FLAG=1 with coarse grids (N_XI=8, N_THETA=8,
    # N_TOROIDAL=12), simplified collisions (diagonal Lorentz) and no rotation
    code_settings="Nonlinear_silly",
    # Individual input.cgyro parameters, applied on top of the preset (level 3): this
    # MAX_TIME overrides the preset's inherited MAX_TIME=1200 — extraOptions always wins
    extraOptions={
        "MAX_TIME": 10.0,  # very short, just for demonstration (real runs need saturated-flux statistics)
    },
    # Resources of each CGYRO instance (one per radius): cores or GPUs per call, and SLURM time limit
    allocation={"resources_per_call": 8, "minutes": 10},
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
    # 'normal' submits and waits for completion; 'submit' returns immediately
    # (come back later with cgyro.check() and cgyro.fetch()); 'prep' only writes the input files
    run_type="normal",
)
# read() parses the out.cgyro.* output files and stores the results in the object under the label
cgyro.read(label="nonlinear_silly")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot (flux time traces, spectra, 2D fluctuations)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (cgyro.fn); show() opens the GUI
cgyro.plot(labels=["nonlinear_silly"])
cgyro.fn.show()
