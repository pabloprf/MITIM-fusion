"""
CAPABILITY: CGYRO warm start from a previous run's restart file
---------------------------------------------------------------
This script teaches how to chain CGYRO runs through restart files: a first
(seed) run writes its checkpoint, and a second run starts from that saturated
state instead of from noise — the standard way to extend statistics or to
continue with slightly different physics without paying the initial transient
again.

Key teaching points:
    1. CGYRO writes its checkpoint (bin.cgyro.restart) every RESTART_STEP data
       outputs; MITIM enforces a RESTART_STEP that guarantees at least one
       write at the end of the run, and retrieves the per-radius blobs as
       bin.cgyro.restart_<rho:.4f> in the run subfolder.
    2. To warm-start, the blob is staged into the new run renamed to the
       canonical bin.cgyro.restart, using `additional_files_to_send`
       ({rho: [(source_path, staged_name)]}). CGYRO auto-detects it at
       startup: with the binary present and no out.cgyro.tag, it uses the
       data as initial condition (warm start) — time restarts from 0 and
       MAX_TIME is the ADDITIONAL simulated time on top of the saved state.
    3. (The tag is deliberately NOT staged: tag+binary would request a true
       time-continuation, which is ill-defined if any input changed and
       requires the full previous output bundle on disk.)
    4. This is exactly what PORTALS automates with the
       `restart_from_folder` / `restart_from_cases` namelist options of the
       CGYRO transport model (see templates/namelist.portals.yaml).
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: both the seed and the warm-started run live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_restart"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

rho = 0.5

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Seed run: a very coarse nonlinear simulation that writes its checkpoint
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.cgyro per requested rho into the folder
# and attaches the experimental normalizations
cgyro = CGYROtools.CGYRO(rhos=[rho])
cgyro.prep(input_gacode, folder)

cgyro.run(
    "seed",
    # Lowest-fidelity nonlinear preset, only for testing workflows (see
    # cgyro_nonlinear_run_from_inputgacode.py for the fidelity ladder)
    code_settings="Nonlinear_silly",
    extraOptions={"MAX_TIME": 2.0},  # very short, just for demonstration
    allocation={"resources_per_call": 8, "minutes": 10},
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type="normal",
)
cgyro.read(label="seed")

# The retrieved checkpoint of the seed run (one per radius)
seed_restart = folder / "seed" / f"bin.cgyro.restart_{rho:.4f}"

# ---------------------------------------------------------------------------------------------------------------------
# 2. Warm-started run: continue from the seed's saturated state
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run(
    "warmstart",
    code_settings="Nonlinear_silly",
    # MAX_TIME is the ADDITIONAL simulated time on top of the saved state
    extraOptions={"MAX_TIME": 2.0},
    # Stage the seed's checkpoint into the new run, renamed to the canonical
    # name CGYRO looks for at startup
    additional_files_to_send={rho: [(seed_restart, "bin.cgyro.restart")]},
    allocation={"resources_per_call": 8, "minutes": 10},
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type="normal",
)
cgyro.read(label="warmstart")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot both runs together: the warm start has no initial transient
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (cgyro.fn); show() opens the GUI
cgyro.plot(labels=["seed", "warmstart"])
cgyro.fn.show()
