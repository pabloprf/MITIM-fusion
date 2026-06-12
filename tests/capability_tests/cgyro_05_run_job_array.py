"""
CAPABILITY: CGYRO at several radii as a SLURM job array
-------------------------------------------------------
This script teaches how MITIM dispatches a multi-radius CGYRO run and how to
control the submission mode. With several radii, each radius is an independent
CGYRO instance, and on clusters the natural layout is a SLURM job array: one
array element (with its own GPU/CPU allocation) per radius.

Key teaching points:
    1. MITIM resolves the submission mode automatically: 'bash' (concurrent
       local processes) on machines without SLURM; on SLURM machines,
       'slurm_standard' (a single allocation running all radii) when
       everything fits in one node, and 'slurm_array' (one sbatch array
       element per radius) otherwise. On GPU machines, CGYRO always uses an
       array so each radius gets its own GPU allocation.
    2. The heuristic can be overridden per run through the allocation dict:
       allocation={'submission_type': 'slurm_array' | 'slurm_standard' |
       'bash'}. Here we force an array explicitly.
    3. allocation={'exclusive': True} additionally forces --exclusive per
       array element — useful on clusters that do not enforce per-job GPU
       isolation (guarantees a whole node per element).
    4. Each array element runs, is monitored and is retrieved independently;
       read()/plot() then work exactly as in a single-radius run. NOTE: on a
       machine without SLURM (e.g. a laptop), the forced array silently falls
       back to concurrent bash processes — the knob matters on clusters.
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared inputs, job files and per-radius outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_array"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare CGYRO at three radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.cgyro per requested rho into the folder
# and attaches the experimental normalizations
cgyro = CGYROtools.CGYRO(rhos=[0.4, 0.55, 0.7])
cgyro.prep(input_gacode, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run cheap linear cases, one job-array element per radius
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run(
    "linear_array",
    code_settings="Linear",
    extraOptions={
        "KY": 0.5,        # binormal wavenumber (ky*rho_s) of the linear mode
        "MAX_TIME": 30.0, # very short, just for demonstration
    },
    allocation={
        # Resources of EACH array element (= each radius)
        "resources_per_call": 8,
        "minutes": 10,
        # Force one sbatch array element per radius (see docstring; on a machine
        # without SLURM this falls back to concurrent bash processes)
        "submission_type": "slurm_array",
        # Uncomment on clusters without strict per-job GPU isolation:
        # "exclusive": True,
    },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type="normal",
)

# read() parses the out.cgyro.* output files of every radius under the label
cgyro.read(label="linear_array")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot (one curve per radius in each panel)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (cgyro.fn); show() opens the GUI
cgyro.plot(labels=["linear_array"])
cgyro.fn.show()
