"""
CAPABILITY: Nonlinear CGYRO on a CPU-only machine
-------------------------------------------------
This script teaches how to run nonlinear CGYRO on CPU nodes (a machine block with
gpus_per_node: 0 in config_user.json, e.g. engaging r8 CPU nodes with 64 cores or
Perlmutter CPU nodes with 128 cores) and how MITIM sizes the job there.

Key teaching points:
    1. On a CPU-only machine, allocation['resources_per_call'] is the number of CPU
       CORES given to each radius (on GPU machines it is the number of GPUs). MITIM
       turns it into the cgyro command `cgyro -e <folder> -n <ranks> -nomp <threads>`
       and into matching sbatch flags (--nodes, --ntasks-per-node, --cpus-per-task).
    2. allocation['omp_threads_cpu'] sets the OpenMP threads per MPI rank (default 1,
       pure MPI). The MPI rank count is resources_per_call / omp_threads_cpu, so the
       two must divide. Examples:
           engaging  (64 cores/node) : resources_per_call=128, omp_threads_cpu=1
                                       -> 2 nodes x 64 ranks x 1 thread  (cgyro -n 128 -nomp 1)
           Perlmutter (128 cores/node): resources_per_call=2048, omp_threads_cpu=4
                                       -> 16 nodes x 32 ranks x 4 threads (cgyro -n 512 -nomp 4)
    3. A radius larger than one node must take WHOLE nodes: resources_per_call must be
       a multiple of cores_per_node (MITIM raises otherwise), and for multi-node calls
       omp_threads_cpu must divide cores_per_node (same rank count on every node).
    4. As on GPU machines, each radius is its own SLURM array element by default;
       allocation['submission_type'] = 'slurm_standard' packs all radii into one job.
    5. TOROIDALS_PER_PROC is chosen from the MPI rank count (not the core count). If no
       value fits the velocity/configuration grid for that rank count, MITIM stops with
       an error listing the rank counts that do fit, instead of letting CGYRO abort.

Set preferences.cgyro in config_user.json to your CPU machine block before running.
Use run_type="prep" first to inspect the generated job without submitting anything.
"""

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools, CONFIGread

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

# 'normal' submits and waits; 'prep' only builds the inputs and the job (nothing is sent)
run_type = "normal"

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared inputs, remote job files and outputs live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_cgyro_cpu"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# The machine selected for CGYRO must be CPU-only for this example
machine = CONFIGread.machineSettings(code="cgyro")
print(f"CGYRO machine: {machine['machine']} | cores_per_node={machine.get('cores_per_node')} | gpus_per_node={machine.get('gpus_per_node')}")
if (machine.get("gpus_per_node") or 0) > 0:
    print("This machine has GPUs: resources_per_call below would be read as GPUs. Point preferences.cgyro to a CPU block.")

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare CGYRO at one radius from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

cgyro = CGYROtools.CGYRO(rhos=[0.5])
cgyro.prep(input_gacode, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run a very coarse nonlinear simulation on CPU cores
# ---------------------------------------------------------------------------------------------------------------------

cgyro.run(
    "nonlinear_cpu",
    # Lowest-fidelity nonlinear preset (N_TOROIDAL=12, N_XI=8, N_THETA=8); workflow testing only
    code_settings="Nonlinear_silly",
    extraOptions={
        "MAX_TIME": 5.0,  # very short, just for demonstration
    },
    allocation={
        # CPU cores per radius. 64 = one engaging r8 node; use 128 for two nodes, or
        # 2048 (with omp_threads_cpu=4) for 16 Perlmutter CPU nodes
        "resources_per_call": 64,
        # OpenMP threads per MPI rank: 64 cores / 1 thread = 64 MPI ranks (cgyro -n 64 -nomp 1)
        "omp_threads_cpu": 1,
        "minutes": 30,
    },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type=run_type,
)

# The job MITIM built: the cgyro line carries -n (ranks) and -nomp (threads); the sbatch
# flags (nodes, ntasks-per-node, cpus-per-task) were printed above when the job was defined
print("\nSLURM settings of the job:", cgyro.simulation_job.slurm_settings)
print("Launch line:", [line for line in "\n".join(cgyro.simulation_job.command).splitlines() if line.startswith("cgyro ")])

if run_type == "normal":
    # read() parses the out.cgyro.* output files and stores the results under the label
    cgyro.read(label="nonlinear_cpu")
    cgyro.plot(labels=["nonlinear_cpu"])
    cgyro.fn.show()
