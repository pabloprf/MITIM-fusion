"""
CAPABILITY: PORTALS with nonlinear CGYRO inside ONE pre-existing SLURM allocation
---------------------------------------------------------------------------------
This script teaches how to run a CGYRO-only PORTALS loop where no evaluation ever
submits a new SLURM job: you allocate the GPU nodes ONCE (salloc / sbatch), launch this
driver inside that allocation, and every CGYRO call becomes an `srun` step on one of
the allocated nodes. Contrast with portals_05_cgyro_nonlinear_slurm.py, where each
evaluation sbatches its own job array and waits in the queue.

Key teaching points:
    1. Machine block WITHOUT a `slurm` section (e.g. "local_gpu": machine "local",
       gpus_per_node 4, cores_per_node 64, modules -> GPU gacode env). MITIM then uses
       its "bash" submission mode: one script per evaluation that backgrounds the
       per-radius `cgyro -e <rho> -n 4 -nomp 16 -numa 4 -mpinuma 1` calls.
    2. Inside an allocation MITIM reads SLURM_JOB_NUM_NODES and lets as many radii run
       concurrently as the allocation holds (nodes x gpus_per_node / resources_per_call);
       each call is pinned to one node with its own GPUs (SLURM_NNODES=1,
       SLURM_GPUS_PER_NODE exported before the launch).
    3. `run_type: "normal"` (run and wait) is the right mode here: there is no queue to
       poll, the steps start immediately on the allocated nodes.
    4. Everything else (warm-start chain, RESTART_STEP, per-iteration MAX_TIME, keep_files)
       is identical to the sbatch-mode script.

How to run it on NERSC Perlmutter (2 radii -> 2 nodes, one radius per node):
    salloc -N 2 -C gpu -q interactive -t 02:00:00 -A m3195_g \
           --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=16
    # inside the allocation, with the pixi env active and preferences.cgyro = "local_gpu":
    python tests/capability_tests/portals_06_cgyro_nonlinear_in_allocation.py > portals_06.log 2>&1
Verify the placement afterwards: the two files
    Execution/Evaluation.3/transport_simulation_folder/base_cgyro/out.cgyro.hosts_<rho>
must name two DIFFERENT nodes, and slurm_output/mitim.out must show no GPU OOM.
Scale-up: allocation nodes = n_radii x (resources_per_call / gpus_per_node).
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_cgyro_in_allocation"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# 3 SR seeds + up to 2 BO iterations = 5 evaluations (2 radii each): short on purpose
portals_fun.optimization_options["initialization_options"]["initial_training"] = 3
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2
portals_fun.optimization_options["acquisition_options"]["optimizers"] = ["sr"]

portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti", "ne"]
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.55, 0.75]

# --- Transport models: nonlinear CGYRO only --------------------------------------------------------------------------
transport = portals_fun.portals_parameters["transport"]
transport["evaluator_instance_attributes"]["turbulence_model"] = "cgyro"
transport["evaluator_instance_attributes"]["neoclassical_model"] = "neo"

cgyro = transport["options"]["cgyro"]
run = cgyro["run"]

run["code_settings"] = "Nonlinear_silly"
run["extraOptions"] = {"RESTART_STEP": 10}
# Seed iteration: 100 a/cs; later iterations warm-start and add 50 a/cs on top
run["extraOptions_special"] = {"0": {"MAX_TIME": 100.0}, ">0": {"MAX_TIME": 50.0}}
run["restart_from_cases"] = "best"

# Run-and-wait inside the allocation (no queue, no polling)
run["run_type"] = "normal"

# 4 GPUs per radius = one full Perlmutter GPU node per radius. `minutes` is unused in
# bash mode (no sbatch) but harmless.
run["allocation"] = {"resources_per_call": 4, "minutes": 30}

cgyro["keep_files"] = "all"
cgyro["read"] = {"tmin": -0.3, "tmin_is_rel": True}

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare and run
# ---------------------------------------------------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

portals_fun.prep(plasma_state)

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------------------------------------------------

portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()

# From the terminal: mitim_plot_portals <run-folder> --complete
