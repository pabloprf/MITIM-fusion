"""
CAPABILITY: TGYRO transport solver from an input.gacode
-------------------------------------------------------
This script teaches how to run TGYRO — the GACODE flux-matching transport
solver — through MITIM, starting from a plasma state. TGYRO iterates the
profiles with a classical (gradient-based) scheme until TGLF+NEO fluxes match
the targets: it is the traditional alternative to the surrogate-based PORTALS
(see portals_01_standard.py), and what PORTALS is benchmarked against. TGYRO runs
on the machine configured for "tgyro" in config_user.json.

Key teaching points:
    1. prep() builds the TGYRO run from the plasma state; run() launches it
       with the radial grid given by `special_radii` (or a uniform
       `vectorRange`), for a number of solver `iterations`.
    2. PredictionSet=[Te, Ti, ne] selects the evolved channels (here Te and
       Ti, density fixed), and TGYRO_physics_options/TGYRO_solver_options map
       to the TGYRO input controls (target model, iteration method, step
       sizes, relaxation).
    3. The TGLF model inside TGYRO is selected with `TGLFsettings` and
       modified with `extraOptionsTGLF` — the same preset/override notion as
       everywhere else in MITIM.
    4. read()/plot() parse and show the convergence history, profiles and
       flux matching of the run.
"""

from mitim_tools.gacode_tools import TGYROtools, PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run
folder = __mitimroot__ / "tests" / "scratch" / "capability_tgyro"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGYRO from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

profiles = PROFILEStools.gacode_state(input_gacode)

tgyro = TGYROtools.TGYRO()
# With cold_start=True, forceIfcold_start avoids the interactive confirmation prompt
tgyro.prep(folder, profilesclass_custom=profiles, cold_start=cold_start, forceIfcold_start=True)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the flux-matching iterations
# ---------------------------------------------------------------------------------------------------------------------

# Radii where the transport equations are solved
rhos = [0.3, 0.5, 0.6, 0.8]

# TGYRO solver controls: iteration method, jacobian/step sizes and relaxation
solver = {
    "step_jac": 1e-2,    # relative step for the jacobian evaluations
    "step_max": 1e-2,    # maximum relative step per iteration
    "res_method": 2,     # residual definition
    "tgyro_method": 6,   # iteration scheme
    "relax_param": 0.1,  # relaxation parameter
}

# TGYRO physics controls (here: target model selection)
physics_options = {"TypeTarget": 2}

tgyro.run(
    # Name of the subfolder (inside the working folder) where this run lives
    subFolderTGYRO="run1",
    # Number of TGYRO iterations (just a few here to keep the example cheap;
    # real flux-matching needs enough iterations to converge the residuals)
    iterations=3,
    cold_start=cold_start,
    forceIfcold_start=True,
    # Solve at these specific radii (alternatively, vectorRange=[from, to, n] for a uniform grid)
    special_radii=rhos,
    # Channels to evolve: [Te, Ti, ne] -> here temperatures only, density fixed
    PredictionSet=[1, 1, 0],
    # TGLF preset used inside TGYRO, and individual input.tglf overrides on top
    TGLFsettings=1,
    extraOptionsTGLF={"USE_BPER": True},
    TGYRO_solver_options=solver,
    TGYRO_physics_options=physics_options,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Read and plot (convergence history, profiles, flux matching)
# ---------------------------------------------------------------------------------------------------------------------

# read() parses the TGYRO output files and stores the results in the object under the label
tgyro.read(label="run1")

# All figures go into a multi-tab MITIM FigureNotebook (tgyro.fn); show() opens the GUI
tgyro.plot(labels=["run1"])
tgyro.fn.show()

