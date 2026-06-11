"""
CAPABILITY: CGYRO via submit / check / fetch (detached runs)
------------------------------------------------------------
This script teaches how to submit a CGYRO run without blocking on it: the
run() call returns right after the SLURM submission, and the results are
checked and retrieved later. This is the natural pattern for expensive runs
(hours/days of wall-clock), where you do not want a python process waiting
on the cluster queue.

Key teaching points:
    1. run_type='submit' stages the inputs, submits the SLURM job(s) and
       returns immediately (contrast with run_type='normal', which waits, and
       run_type='prep', which only writes the input files). Anything can be
       done between submission and retrieval — including other submissions.
    2. check(every_n_minutes=N) polls the queue every N minutes until the job
       leaves it (it also walks the CGYRO output files to report progress).
    3. fetch() retrieves the output files from the run machine and organizes
       them in the working folder; after that, read() and plot() work exactly
       as in a blocking run (see cgyro_linear_run_from_inputgacode.py).
    4. Note that CGYRO has a FOURTH settings level on top of the usual
       controls -> code_settings -> extraOptions hierarchy: `preprocess_options`
       (see cgyro_nonlinear_run_from_inputgacode.py).
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared inputs, remote job files and outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_submit"
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
# 2. Submit a cheap linear run and return immediately
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "linear_submitted",
    # Preset from templates/input.cgyro.models.yaml (level 2 of the hierarchy):
    # "Linear" sets NONLINEAR_FLAG=0 and a single toroidal mode (N_TOROIDAL=1)
    code_settings="Linear",
    # Individual input.cgyro parameters, applied on top of the preset (level 3)
    extraOptions={
        "KY": 0.5,        # binormal wavenumber (ky*rho_s) of the linear mode
        "MAX_TIME": 30.0, # very short, just for demonstration
    },
    # Resources of each CGYRO instance (one per radius): cores or GPUs per call, and SLURM time limit
    allocation={"resources_per_call": 8, "minutes": 10},
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
    # The point of this capability: submit and DO NOT wait
    run_type="submit",
)

# ... the job is now in the queue; this script could do other work here,
# e.g. submit more runs to other folders ...

# ---------------------------------------------------------------------------------------------------------------------
# 3. Wait for completion, retrieve the outputs and read them
# ---------------------------------------------------------------------------------------------------------------------

# Poll the queue every minute until the job leaves it
cgyro.check(every_n_minutes=1)

# Retrieve the output files from the run machine and organize them in the working folder
cgyro.fetch()

# From here on, everything is identical to a blocking run
# (read() parses the out.cgyro.* output files and stores the results under the label)
cgyro.read(label="linear_submitted")

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (cgyro.fn); show() opens the GUI
cgyro.plot(labels=["linear_submitted"])
cgyro.fn.show()
