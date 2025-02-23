import os
import argparse
from mitim_tools.misc_tools import FARMINGtools, IOtools

"""
This script is used to launch a slurm job of a MITIM optimization.
The optimization script should receive both "folder" and "--seed", i.e.:

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("folder", type=str)
        parser.add_argument("--seed", type=int, required=False, default=0)
        args = parser.parse_args()
        folder = Path(args.folder)
        seed = args.seed

        # REST OF SCRIPT THAT PERFORMS THE JOB

To call:

    mitim_slurm runPORTALS.py --folder run1/

        [optional]
            --patition sched_mit_psfc_r8                (partition to run the job)
            --env /pool001/pablorf/env/mitim-env_311    (path to the virtual environment)
            --seeds 1                                   (number of seeds to run)
            --hours 8                                   (number of hours to run the job)
            --n 32                                      (number of CPUs per task)
            --seed_specific 0                           (specific seed to run, if seeds == 1)
            --extra 0.1                                 (extra arguments to pass to the script)

"""

def run_slurm(
    script,
    folder,
    partition,
    venv,
    seeds=1,
    hours=8,
    n=32,
    seed_specific=0,
    extra=None,
    machine="local"
):
    script = IOtools.expandPath(script)
    folder = IOtools.expandPath(folder)

    seeds_explore = [seed_specific] if seeds == 1 else list(range(seeds))

    for seed in seeds_explore:

        extra_name = "" if seeds == 1 else f"_s{seed}"

        folder = IOtools.expandPath(folder) / extra_name

        print(f"* Launching slurm job of MITIM optimization with random seed = {seed}")

        folder.mkdir(parents=True, exist_ok=True)

        extra_str = " ".join([str(e) for e in extra]) if extra is not None else ""

        command = [
            f"source {venv}/bin/activate",
            f"python3 {script} {folder} {extra_str} --seed {seed}",
        ]

        nameJob = f"mitim_opt_{folder.name}{extra_name}"

        _, fileSBATCH, _ = FARMINGtools.create_slurm_execution_files(
            command,
            folder_remote=folder,
            folder_local=folder,
            nameJob=nameJob,
            slurm={"partition": partition},
            minutes=int(60 * hours),
            ntasks=1,
            cpuspertask=n,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("--folder", type=str, required=False, default="run1/")
    parser.add_argument("--machine", type=str, required=False, default="local")
    parser.add_argument("--hours", type=int, required=False, default=8)
    parser.add_argument("--n", type=int, required=False, default=16)
    parser.add_argument("--seed_specific", type=int, required=False, default=0)
    parser.add_argument("--extra", type=float, required=False, default=None, nargs="*")
    parser.add_argument("--seeds", type=int, required=False, default=1)
    parser.add_argument(
        "--partition",
        type=str,
        required=False,
        default=IOtools.expandPath("$MFEIM_PARTITION"),
    )
    parser.add_argument(
        "--env", type=str, required=False, default=IOtools.expandPath("~/env/mitim-env")
    )

    args = parser.parse_args()

    run_slurm(
        args.script,
        args.folder,
        args.partition,
        args.env,
        seeds=args.seeds,
        hours=args.hours,
        n=args.n,
        extra=args.extra,
        seed_specific=args.seed_specific,
        machine=args.machine,
    )

if __name__ == "__main__":
    main()
