import os
from mitim_tools.misc_tools import FARMINGtools, IOtools
from IPython import embed

"""
This script is used to launch a slurm job with a scpecific script like... python3 run_case.py 0 --R 6.0
"""

def run_slurm(
    script,
    folder,
    partition,
    venv,
    seeds=None,    # If not None, assume that the script is able to receive --seeds #
    hours=8,
    n=32,
    seed_specific=0,
    machine="local",
    exclude=None,
    mem=None,
    exclusive=False, 
    wait=False,
):

    folder = IOtools.expandPath(folder)

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

        command = [venv,script + (f" --seed {seed}" if seed is not None else "")]
        nameJob = f"mitim_{folder.name}{extra_name}"

        _, fileSBATCH, _ = FARMINGtools.create_slurm_execution_files(
            command,
            folder,
            folder_local=folder,
            slurm={"partition": partition, 'exclude': exclude,'exclusive': exclusive},
            slurm_settings = {
                'name': nameJob,
                'minutes': int(60 * hours),
                'ntasks': 1,
                'cpuspertask': n,
                'memory_req_by_job': mem
            }
        )

        if wait == True:
            print('* Waiting for job to complete...')
            command_execution = f"sbatch --wait {fileSBATCH}"
        else: 
            command_execution = f"sbatch {fileSBATCH}"

        if machine == "local":
            os.system(command_execution)
        else:
            FARMINGtools.perform_quick_remote_execution(
                folder,
                machine,
                command_execution,
                input_files=[fileSBATCH],
                job_name = nameJob,
                )


def run_slurm_array(
    script,
    array_input,
    folder,
    partition,
    max_concurrent_jobs, 
    venv = '',
    seeds=None,    # If not None, assume that the script is able to receive --seeds #
    hours=8,
    n=32,
    seed_specific=0,
    machine="local",
    exclude=None,
    mem=None, 
):

    folder = IOtools.expandPath(folder)

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

        nameJob = f"mitim_{folder.name}{extra_name}"

        _, fileSBATCH, _ = FARMINGtools.create_slurm_execution_files(
            command=command,
            folder_remote=folder,
            folder_local=folder,
            nameJob=nameJob,
            slurm={"partition": partition, 'exclude': exclude},
            minutes=int(60 * hours),
            ntasks=1,
            cpuspertask=n,
            memory_req_by_job=mem,
            job_array=f'{string_of_array_input}%{max_concurrent_jobs}',

        )

        command_execution = f"sbatch {fileSBATCH}"

        if machine == "local":
            os.system(command_execution)
        else:
            FARMINGtools.perform_quick_remote_execution(
                folder,
                machine,
                command_execution,
                input_files=[fileSBATCH],
                job_name = nameJob,
                )
