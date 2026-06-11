"""
CAPABILITY: Linear CGYRO run from an input.gacode
-------------------------------------------------
This script teaches how to run a (cheap) linear CGYRO simulation starting from
a plasma state (input.gacode). CGYRO runs on the machine configured for it in
config_user.json (possibly remote, via SLURM).

Key teaching points:
    1. CGYRO(rhos=[...]) + prep(input.gacode, ...) generates one input.cgyro
       per requested radius, same pattern as TGLF/NEO.
    2. The "Linear" preset (templates/input.cgyro.models.yaml) sets
       NONLINEAR_FLAG=0 with a single toroidal mode; `extraOptions` selects the
       binormal wavenumber KY of that mode and, here, a very short MAX_TIME to
       keep the run cheap.
    3. `allocation` controls the resources of each CGYRO instance (one per
       radius). run_type='normal' submits and waits; for long runs, use
       run_type='submit' and come back later with cgyro.check() + cgyro.fetch().
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared inputs, remote job files and outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_linear"
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
# 2. Run a single linear mode
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "linear_ky05",
    # Preset from templates/input.cgyro.models.yaml (level 2 of the hierarchy):
    # "Linear" sets NONLINEAR_FLAG=0 and a single toroidal mode (N_TOROIDAL=1)
    code_settings="Linear",
    # Individual input.cgyro parameters, applied on top of the preset (level 3)
    extraOptions={
        "KY": 0.5,        # binormal wavenumber (ky*rho_s) of the linear mode
        "MAX_TIME": 30.0, # very short, just for demonstration (real linear runs need convergence of the eigenvalue)
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
cgyro.read(label="linear_ky05")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot (eigenvalue convergence, fluctuation structure)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (cgyro.fn); show() opens the GUI
cgyro.plot(labels=["linear_ky05"])
cgyro.fn.show()
