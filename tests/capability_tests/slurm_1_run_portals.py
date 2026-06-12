"""
CAPABILITY: Launching MITIM cases as SLURM jobs with run_slurm()
----------------------------------------------------------------
This script teaches how to submit any MITIM command as a SLURM job using
run_slurm(). As an example, it launches a PORTALS run through its command-line
interface (mitim_run_portals), which is how long cases are typically farmed
out to a cluster instead of being run in the local terminal.

NOTE: this example uses the "engaging_rpp" machine block of config_user.json,
which is an MIT-specific cluster (the PSFC partition of the Engaging/ORCD
system) — shown here as a concrete example of how a real configuration looks.
If you are not at MIT, simply point `machine_config` below to a machine block
defined in your own config_user.json. The script is meant to be executed on
the cluster itself (e.g. its login node), since run_slurm() submits to the
scheduler of the machine it runs on by default (machine="local").

Key teaching points:
    1. mitim_run_portals runs PORTALS from files on disk: a folder containing
       `input.gacode` and `namelist.portals.yaml` (the on-disk equivalent of
       the in-memory dictionary modification shown in portals_1_standard.py).
       The --batch flag makes it non-interactive (required inside a job).
    2. run_slurm(command, folder, partition, environment, ...) writes the
       sbatch file into the folder, submits it, and returns: monitor with
       squeue and read the slurm log files written in the same folder. Pass
       wait=True to block until completion instead.
    3. If hours > max_hours, run_slurm() automatically chains several
       dependent sbatch jobs (each up to max_hours) until the total
       wall-time is covered — useful on partitions with short time limits,
       since MITIM workflows restart from where they left off.
    4. run_slurm_array() is the sibling function to farm a list of inputs as
       a SLURM job array.
"""

import shutil
from mitim_tools.opt_tools.scripts.slurm import run_slurm
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.CONFIGread import load_settings
from mitim_tools import __mitimroot__

# cold_start=True starts from scratch (here, removing the previous folder); False would
# let mitim_run_portals continue from whatever the previous job already evaluated
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: PORTALS inputs, the sbatch file and the slurm logs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_slurm_portals"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare the run folder that mitim_run_portals expects (input.gacode + namelist.portals.yaml)
# ---------------------------------------------------------------------------------------------------------------------

# The plasma state, with the standard name the CLI looks for
shutil.copy2(input_gacode, folder / "input.gacode")

# Start from the default template namelist and modify it BEFORE writing it to the run
# folder — the on-disk file plays the role that the in-situ dictionary modification
# plays in portals_1_standard.py
nml = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.portals.yaml")
nml["solution"]["predicted_rho"] = [0.25, 0.45, 0.65, 0.85]
nml["optimization_options"]["convergence_options"]["maximum_iterations"] = 2
IOtools.write_mitim_yaml(nml, folder / "namelist.portals.yaml")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Grab the SLURM settings of the cluster from config_user.json
# ---------------------------------------------------------------------------------------------------------------------

# MIT-specific example (see NOTE in the docstring): the "engaging_rpp" block defines,
# among others, slurm.partition (e.g. "sched_mit_psfc_r8") and `modules` (the shell
# command that activates the MITIM environment on that cluster). Replace with the
# name of a machine block in your own config_user.json
machine_config = "engaging_rpp"

settings = load_settings()

# Partition to submit to
partition = settings[machine_config]["slurm"]["partition"]

# Shell command(s) executed before the script inside the job, to set up the python
# environment (e.g. "source ~/venvs/mitim/bin/activate" or a module-load string).
# Here we reuse the `modules` field of the machine configuration
environment = settings[machine_config].get("modules", "") or ""

# ---------------------------------------------------------------------------------------------------------------------
# 3. Submit mitim_run_portals as a SLURM job
# ---------------------------------------------------------------------------------------------------------------------

run_slurm(
    # Any shell command works here; --batch because there is no terminal inside the job
    f"mitim_run_portals {folder} --batch",
    # Folder where the sbatch file and the slurm output/error logs are written
    folder,
    partition,
    environment,
    # "local" (default) submits to the scheduler of the machine this script runs on —
    # i.e. execute this script on the cluster itself (here, engaging_rpp's login node)
    machine="local",
    # Job size: wall-time (hours) and number of cores (n)
    hours=1,
    n=8,
    # wait=False (default) returns right after submission; wait=True blocks until the job ends
    wait=False,
)

# After submission: check the job with `squeue --me`, follow the log files in the run
# folder, and once finished plot the results with `mitim_plot_portals <folder>`
