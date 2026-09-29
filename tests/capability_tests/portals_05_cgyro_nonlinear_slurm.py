"""
CAPABILITY: PORTALS driven entirely by nonlinear CGYRO on a SLURM/GPU machine
-----------------------------------------------------------------------------
This script teaches how to run a PORTALS flux-matching loop where the ONLY turbulence
model is nonlinear CGYRO, submitted through SLURM to the machine configured for
"cgyro" in config_user.json (one GPU allocation per radius, warm-start chain across
iterations, detached submit + polling). It uses the "Nonlinear_silly" preset so that
every evaluation is cheap: this is a WORKFLOW test at production SHAPE (5 radii, 3
channels, restarts, job arrays), not production PHYSICS.

Key teaching points:
    1. Single-fidelity CGYRO: `turbulence_model: "cgyro"` and everything in
       transport.options.cgyro. `run_base_tglf: True` (default) still runs TGLF at every
       iteration for comparison; it never enters the residual.
    2. On a GPU machine (gpus_per_node > 0 in the machine block), each radius becomes
       one element of a SLURM job array; `allocation.resources_per_call` is the number
       of GPUs per radius (MPI ranks = GPUs, one NUMA per GPU). Requires a machine block
       WITH a `slurm` section (account/qos/constraint); a block without one runs
       directly in the current shell/allocation.
    3. `run_type: "submit"` returns right after sbatch and PORTALS polls the queue every
       `every_n_minutes`; `check_existing_runs: True` re-attaches to a running job if
       the driver is restarted (kill + relaunch of this script resumes cleanly).
    4. Warm-start chain: `restart_from_cases: "best"` makes each radius restart from the
       previous iteration whose turbulent flux is closest to its current target. That
       needs RESTART_STEP set (restart written) and keep_files "all" (restart kept).
       MAX_TIME on iterations >= 1 is ADDITIONAL time on top of the saved state
       (CGYRO warm-start resets t to 0), hence the per-iteration overrides below.
    5. `extraOptions_special` / `allocation_special` take iteration selectors ("0",
       ">0", ">=5", ...) to give the seed iteration a longer window and more wall-clock.

Machine block example for NERSC Perlmutter, run from a Perlmutter login/compute shell
(machine "local", SLURM GPU queue; Perlmutter selects queues with --qos, not --partition):
    "perlmutter_gpu_slurm": {
        "machine":   "local",
        "username":  "<user>",
        "scratch":   "/pscratch/sd/<u>/<user>/scratch/",
        "modules":   "source <path-to-GPU-gacode-env>.src",
        "cores_per_node": 64,
        "gpus_per_node":  4,
        "slurm": {"account": "m3195_g", "constraint": "gpu", "qos": "debug"}
    }
with "preferences": {"cgyro": "perlmutter_gpu_slurm", ...}. The debug QOS caps jobs at
30 min and 8 nodes; use "regular" or "preempt" for real runs.

Cost of this script (silly resolution, 4 GPUs per radius): each evaluation is one
5-element array of ~5-10 min; with 5 SR seeds + up to 10 BO iterations expect ~2-3 h
wall-clock and ~60 GPU-node-hours-equivalent, most of it queue wait.
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# whatever is already in the folder (completed evaluations are detected and skipped, and
# with check_existing_runs a CGYRO job still in the queue is re-attached, not resubmitted)
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Working folder of the run: everything (inputs, per-iteration model runs, logs, results)
# is written under it
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_cgyro_nonlinear"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# --- Optimization controls -------------------------------------------------------------------------------------------
# `initial_training` simple-relaxation (SR) evaluations seed the surrogates, then up to
# `maximum_iterations` BO iterations follow (or fewer, if the residual-reduction criterion
# `maximum_value` = 5e-3 relative in the template fires first)
portals_fun.optimization_options["initialization_options"]["initial_training"] = 5
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 10
portals_fun.optimization_options["acquisition_options"]["optimizers"] = ["sr"]

# --- Solution: what to predict ---------------------------------------------------------------------------------------
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti", "ne"]
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.35, 0.55, 0.75, 0.875, 0.9]

# --- Transport models: nonlinear CGYRO only --------------------------------------------------------------------------
transport = portals_fun.portals_parameters["transport"]
transport["evaluator_instance_attributes"]["turbulence_model"] = "cgyro"
transport["evaluator_instance_attributes"]["neoclassical_model"] = "neo"

cgyro = transport["options"]["cgyro"]
run = cgyro["run"]

# Level 2 of the CGYRO hierarchy: cheapest rung of the nonlinear ladder (N_XI=8, N_THETA=8,
# N_TOROIDAL=12, diagonal Lorentz collisions, no rotation; perpendicular grid from the
# preset's own preprocess_options). Swap for "Nonlinear_reduced1"/"Nonlinear_high" for physics.
run["code_settings"] = "Nonlinear_silly"

# Level 3, common to all iterations. RESTART_STEP is in units of data outputs
# (DELTA_T*PRINT_STEP = 1 a/cs after MITIM's PRINT_STEP coercion), so 10 -> a restart
# blob every 10 a/cs; MITIM also guarantees one at the end of the run.
run["extraOptions"] = {"RESTART_STEP": 10}

# Per-iteration overrides: the seed iteration pays the transient and runs a long window;
# later iterations warm-start from a saturated state and only need statistics on top.
# (Silly resolution: these windows are still short for real saturation statistics.)
run["extraOptions_special"] = {
    "0":  {"MAX_TIME": 300.0},
    ">0": {"MAX_TIME": 150.0},
}

# Warm-start chain across PORTALS iterations (see docstring). "best" is non-fatal: a radius
# with no usable source is cold-started with a warning.
run["restart_from_cases"] = "best"

# Submission: detached submit + polling, re-attach on driver restart
run["run_type"] = "submit"
run["check_existing_runs"] = True
run["every_n_minutes"] = 2
# Retry SSH forever (only relevant when the CGYRO machine is remote from this driver)
run["ssh_retry_attempts"] = None

# Resources per radius: 4 GPUs (= one full Perlmutter GPU node, 4 MPI ranks x 1 NUMA each)
# and a 30-min limit (debug QOS ceiling). Iteration 0 keeps the same limit here; raise it
# via allocation_special when using a longer MAX_TIME or a heavier preset.
run["allocation"] = {"resources_per_call": 4, "minutes": 30}
run["allocation_special"] = {"0": {"minutes": 30}}

# Keep every CGYRO file: restarts must survive for the chain, and the traces are what
# you inspect afterwards (mitim_plot_portals <folder> --complete)
cgyro["keep_files"] = "all"

# Signal window for the flux average: last 30% of each run (template default)
cgyro["read"] = {"tmin": -0.3, "tmin_is_rel": True}

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state and the run
# ---------------------------------------------------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

# prep() defines DVs (a/LTe, a/LTi, a/Lne at the 5 radii = 15 DVs) and OFs (flux residuals)
# and snapshots the namelist into the folder — edits after this point are ignored
portals_fun.prep(plasma_state)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

# MITIM_BO is the generic optimization driver; askQuestions=False avoids interactive prompts.
# The driver blocks while polling the CGYRO job of each evaluation; if it is killed, rerun
# this script with cold_start=False to resume (finished evaluations are skipped, a queued or
# running CGYRO job is re-attached through cgyro_submission.json)
mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot results
# ---------------------------------------------------------------------------------------------------------------------

# Restart provenance of each iteration: Execution/Evaluation.<i>/transport_simulation_folder/
# base_cgyro/restart_sources.json (which source iteration each radius warm-started from)
portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()

# From the terminal at any time (also while running):
#   mitim_plot_portals <run-folder> --complete      (per-radius CGYRO time traces, restart-aware)
