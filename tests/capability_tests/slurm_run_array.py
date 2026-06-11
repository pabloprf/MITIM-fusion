"""
CAPABILITY: Farming many cases as a SLURM job array with run_slurm_array()
--------------------------------------------------------------------------
This script teaches how to launch the same script many times — once per input
value — as a single SLURM job array. This is the natural way to farm
parameter scans (e.g. one MAESTRO/PORTALS case per engineering point) without
submitting jobs one by one.

NOTE: as in slurm_run_portals.py, this example uses the "engaging_rpp"
machine block of config_user.json, which is an MIT-specific cluster (the PSFC
partition of the Engaging/ORCD system). If you are not at MIT, point
`machine_config` below to a machine block defined in your own
config_user.json. The script is meant to be executed on the cluster itself
(e.g. its login node).

Key teaching points:
    1. run_slurm_array(script, array_input, max_concurrent_jobs, ...) submits
       ONE sbatch job array: SLURM appends $SLURM_ARRAY_TASK_ID as the last
       argument of the script, so the script must accept the task id and
       decide from it what case to run.
    2. `array_input` is the list of task ids to run (they become the --array
       indices; they do not need to be consecutive), and `max_concurrent_jobs`
       throttles how many run simultaneously (--array=...%N).
    3. The same pattern launches heavy MITIM workflows: replace the toy script
       below with e.g. a script that reads its task id and calls
       mitim_run_portals/mitim_run_maestro on the corresponding case folder.
"""

from mitim_tools.opt_tools.scripts.slurm import run_slurm_array
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.CONFIGread import load_settings
from mitim_tools import __mitimroot__

# cold_start=True starts from scratch (here, removing the previous folder)
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: the task script, the sbatch file, the slurm logs and the
# per-task outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_slurm_array"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Write the toy script that each array task will execute
# ---------------------------------------------------------------------------------------------------------------------

# Each task runs `python task_script.py <folder> <task_id>`: the task id arrives as the
# LAST argument (appended by run_slurm_array as $SLURM_ARRAY_TASK_ID) and the script
# uses it to decide what to do — here, just write a file proving the task ran
task_script = folder / "task_script.py"
task_script.write_text(
    '''import sys

folder = sys.argv[1]
task_id = sys.argv[2]   # $SLURM_ARRAY_TASK_ID, appended by run_slurm_array

with open(f"{folder}/file_successfully_created_{task_id}.txt", "w") as f:
    f.write(f"Successfully created file from job #: {task_id}\\n")
'''
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Grab the SLURM settings of the cluster from config_user.json
# ---------------------------------------------------------------------------------------------------------------------

# MIT-specific example (see NOTE in the docstring); replace with a machine block
# from your own config_user.json
machine_config = "engaging_rpp"

settings = load_settings()

# Partition to submit to
partition = settings[machine_config]["slurm"]["partition"]

# Shell command(s) executed before the script inside each task, to set up the python
# environment; here we reuse the `modules` field of the machine configuration
environment = settings[machine_config].get("modules", "") or ""

# ---------------------------------------------------------------------------------------------------------------------
# 3. Submit the job array
# ---------------------------------------------------------------------------------------------------------------------

# Task ids to run: these become the --array indices of the sbatch submission. They can
# be any non-consecutive integers that your script knows how to interpret
array_input = [62, 63, 81]

run_slurm_array(
    # The command of each task; $SLURM_ARRAY_TASK_ID is appended as the last argument
    f"python {task_script} {folder}",
    array_input,
    # At most this many tasks run at the same time (--array=62,63,81%2)
    2,
    # Folder where the sbatch file and the slurm output/error logs are written
    folder,
    partition,
    venv=environment,
    # Job size PER TASK: wall-time (hours), cores (n) and memory
    hours=1,
    n=2,
    mem="4GB",
    # wait=False (default) returns right after submission; check with `squeue --me`
    wait=False,
)
