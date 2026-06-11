import math
import os
import subprocess
from mitim_tools.misc_tools import FARMINGtools, IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

"""
This script is used to launch a slurm job with a scpecific script like... python3 run_case.py 0 --R 6.0
"""

def _fmt_minutes(minutes):
    """Format minutes as the sbatch 'time' string (MM:00 or HH:MM:00)."""
    minutes = int(minutes)
    if minutes >= 60:
        h, m = divmod(minutes, 60)
        return f"{h:02d}:{m:02d}:00"
    return f"{minutes:02d}:00"

def run_slurm(
        script,
        folder,
    # For where and how to launch the job:
        partition,
        venv,
        machine = "local",
        exclude = None,
        mem = None,
        exclusive = False, 
        qos = None,
    # Job size:
        n = 32,
        hours = 8,
        max_hours = 8,   # Max hours per sbatch allocation; hours>max_hours chains dependent jobs
        are_n_threads = True,
        ntasks_per_node = None,
    # For farming different seeds that the script understands:
        seeds = None,    # If not None, assume that the script is able to receive --seeds #
        seed_specific = 0,
    # Interaction settings:
        wait = False,
        nameJob = None,
    # For job arrays:
        job_array = None,
):

    folder = IOtools.expandPath(folder)

    seeds_explore = [None] if seeds is None else ([seed_specific] if seeds == 1 else list(range(seeds)))

    for seed in seeds_explore:

        extra_name = "" if (seed is None or seeds == 1) else f"_s{seed}"

        folder = IOtools.expandPath(folder)
        folder = folder.with_name(folder.name + extra_name)

        print(f"* Launching MITIM slurm job with random seed = {seed}")

        folder.mkdir(parents=True, exist_ok=True)

        command = [venv,script + (f" --seed {seed}" if seed is not None else "")]
        if nameJob is None:
            nameJob = f"mitim_{folder.name}{extra_name}"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocation information (e.g. partition, node exclusions and exclusivity)
        slurm_allocation = {
            "partition": partition,
            'qos': qos,
            'exclude': exclude,
            'exclusive': exclusive
            }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Split the requested wall-time into chunks of at most max_hours. Each
        # chunk is a separate sbatch submission; chunks after the first use
        # --dependency=afterany:<previous_job_id> so they launch as soon as
        # the preceding chunk ends (successfully or not), effectively running
        # the same script repeatedly until the total wall-time is covered.

        if max_hours <= 0:
            raise ValueError("max_hours must be positive")

        n_chunks = max(1, math.ceil(hours / max_hours))
        chunk_hours = [min(max_hours, hours - i * max_hours) for i in range(n_chunks)]

        if n_chunks > 1:
            print(f"* Requested {hours}h exceeds max_hours={max_hours}h; chaining {n_chunks} dependent jobs: {chunk_hours}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Slurm job information (settings for sbatch). One sbatch file per chunk.

        if are_n_threads: ntask, cpus_per_task = 1, n
        else:             ntask, cpus_per_task = n, 1

        sbatch_files = []
        for i, ch in enumerate(chunk_hours):
            label = "" if n_chunks == 1 else f"_chunk{i}"
            slurm_settings = {
                'job-name': nameJob + (f"_c{i}" if n_chunks > 1 else ""),
                'time': _fmt_minutes(int(60 * ch)),
                'ntasks': ntask,
                'ntasks-per-node': ntasks_per_node,
                'cpus-per-task': cpus_per_task,
                'mem': mem,
                'array': job_array,
            }

            _, fileSBATCH_i, _ = FARMINGtools.create_slurm_execution_files(
                command,
                folder,
                folder_local=folder,
                slurm_allocation=slurm_allocation,
                slurm_settings=slurm_settings,
                label_log_files=label,
                if_array_relabel=True,
                wait_until_sbatch=wait,
            )
            sbatch_files.append(fileSBATCH_i)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build the submission command. For a single chunk this is just a plain
        # sbatch call. For multiple chunks we capture each job id via
        # `sbatch --parsable` and pass it as the dependency of the next chunk.
        # When wait=True we only apply --wait to the last chunk so the caller
        # blocks until the full chain finishes.

        if n_chunks == 1:
            wait_flag = "--wait " if wait else ""
            if wait:
                print('* Waiting for job to complete...')
            command_execution = f"sbatch {wait_flag}{sbatch_files[0]}"
        else:
            parts = []
            for i, f in enumerate(sbatch_files):
                dep = f"--dependency=afterany:$JOBID{i-1} " if i > 0 else ""
                wait_flag = "--wait " if (wait and i == n_chunks - 1) else ""
                parts.append(f"JOBID{i}=$(sbatch --parsable {dep}{wait_flag}{f})")
                parts.append(f"echo \"Submitted chunk {i+1}/{n_chunks} as job $JOBID{i}\"")
            if wait:
                print('* Waiting for dependent job chain to complete...')
            command_execution = " && ".join(parts)

        if machine == "local":
            result = subprocess.run(command_execution + f' 2>&1 | tee {folder}/sbatch_submission.log', shell=True)
            if result.returncode != 0:
                print(f"\t- Local sbatch submission returned non-zero exit code ({result.returncode}), check {folder}/sbatch_submission.log", typeMsg='w')
        else:
            FARMINGtools.perform_quick_remote_execution(
                folder,
                machine,
                command_execution,
                input_files=sbatch_files,
                job_name = nameJob,
                )

def run_slurm_array(
        script,
    # Array information
        array_input,
        max_concurrent_jobs, 
    # Run paths
        folder,
        partition,
        venv = '',
    # Slurm specifications
        machine = "local",
        exclude = None, 
        mem = None, 
        exclusive = False, 
        qos = None,
    # Job size
        n = 32, 
        hours = 8, 
        are_n_threads = True,
        ntasks_per_node = None,
    # For farming different seeds that the script understands:
        seeds=None,    # If not None, assume that the script is able to receive --seeds 
        seed_specific = 0,
    # Interaction settings:
        wait = False,
        nameJob = None,
):

    folder = IOtools.expandPath(folder)

    # Set seeds_explore variable
    if seeds is not None:
        seeds_explore = [seed_specific] if seeds == 1 else list(range(seeds))
    else:
        seeds_explore = [None]

    for seed in seeds_explore:

        extra_name = "" if (seed is None or seeds == 1) else f"_s{seed}"

        folder = IOtools.expandPath(folder)
        folder = folder.with_name(folder.name + extra_name)

        print(f"* Launching slurm job of MITIM optimization with random seed = {seed}")

        folder.mkdir(parents=True, exist_ok=True)

        command = ['echo $SLURM_ARRAY_TASK_ID', venv, script + ' $SLURM_ARRAY_TASK_ID'+ (f" --seed {seed}" if seed is not None else "")]
        string_of_array_input = ','.join([str(i) for i in array_input])

        # Give job default name
        if nameJob is None:
            nameJob = f"mitim_{folder.name}{extra_name}"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocation information (e.g. partition, node exclusions and exclusivity)
        slurm_allocation = {
            "partition": partition,
            'qos': qos,
            'exclude': exclude,
            'exclusive': exclusive
            }
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Slurm job information  (settings for sbatch)
        
        if are_n_threads: ntask, cpus_per_task = 1, n
        else:             ntask, cpus_per_task = n, 1

        slurm_settings = {
            'job-name': nameJob,
            'time': _fmt_minutes(int(60 * hours)),
            'ntasks': ntask,
            'ntasks-per-node': ntasks_per_node,
            'cpus-per-task': cpus_per_task,
            'mem': mem,
            'array': f'{string_of_array_input}%{max_concurrent_jobs}',
        }


        _, fileSBATCH, _ = FARMINGtools.create_slurm_execution_files(
            command,
            folder,
            folder_local=folder,
            slurm_allocation=slurm_allocation,
            slurm_settings = slurm_settings,
            if_array_relabel=False,
            wait_until_sbatch=wait,
        )

        if wait:
            print('* Waiting for job to complete...')
            command_execution = f"sbatch --wait {fileSBATCH}"
        else:
            command_execution = f"sbatch {fileSBATCH}"

        if machine == "local":
            result = subprocess.run(command_execution, shell=True)
            if result.returncode != 0:
                print(f"\t- Local sbatch submission returned non-zero exit code ({result.returncode})", typeMsg='w')
        else:
            FARMINGtools.perform_quick_remote_execution(
                folder,
                machine,
                command_execution,
                input_files=[fileSBATCH],
                job_name = nameJob,
                )

def main():
    """
    CLI entry point (`mitim_slurm`): submit a command/script as a SLURM job.

    The positional `script` is the full command to execute (quote it if it has
    arguments) and `folder` is where the job runs and writes its slurm files.
    Restores the `main()` that the SLURM-launcher refactor dropped, leaving the
    advertised console script pointing at a nonexistent function.

    Examples:
        mitim_slurm 'python3 run_portals.py run1' run1 --partition sched_mit_psfc --env 'source ~/env/bin/activate'
        mitim_slurm 'python3 myopt.py' sweep --partition sched_mit_psfc --env 'module load mitim' --seeds 10 --hours 16 --max_hours 8
    """
    import argparse

    parser = argparse.ArgumentParser(description="Submit a MITIM script as a SLURM job")
    parser.add_argument("script", type=str, help="Command to execute (quote if it has arguments)")
    parser.add_argument("folder", type=str, help="Folder to run in")
    parser.add_argument("--partition", required=True, type=str, help="SLURM partition")
    parser.add_argument("--env", required=True, type=str, dest="venv",
                        help="Environment line executed before the script (e.g. 'source venv/bin/activate' or 'module load ...')")
    parser.add_argument("--machine", type=str, default="local", help="Machine to submit on (config name; default: local)")
    parser.add_argument("--n", type=int, default=32, help="Cores (threads by default; see --tasks)")
    parser.add_argument("--tasks", action="store_true", help="Interpret --n as MPI tasks instead of threads")
    parser.add_argument("--hours", type=float, default=8, help="Total wall-time hours")
    parser.add_argument("--max_hours", type=float, default=8, help="Max hours per sbatch; hours>max_hours chains dependent jobs")
    parser.add_argument("--mem", type=str, default=None, help="Memory request (e.g. 64GB)")
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None, help="Nodes to exclude")
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--ntasks_per_node", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Farm N seeded copies (script must accept --seed #)")
    parser.add_argument("--name", type=str, default=None, help="SLURM job name (default: mitim_<folder>)")
    parser.add_argument("--wait", action="store_true", help="Wait for completion instead of returning after submission")
    args = parser.parse_args()

    run_slurm(
        args.script,
        args.folder,
        args.partition,
        args.venv,
        machine=args.machine,
        exclude=args.exclude,
        mem=args.mem,
        exclusive=args.exclusive,
        qos=args.qos,
        n=args.n,
        hours=args.hours,
        max_hours=args.max_hours,
        are_n_threads=not args.tasks,
        ntasks_per_node=args.ntasks_per_node,
        seeds=args.seeds,
        wait=args.wait,
        nameJob=args.name,
    )
