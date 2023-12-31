import os
import argparse
from mitim_tools.misc_tools import FARMINGtools, IOtools

"""
This script is used to launch a slurm job of a MITIM optimization.
The main script should receive: folderWork and --seed

To call:
	cd folder_where_all_runs_are/
	python3 ~/MITIM/mitim_opt/opt_tools/exe/slurm.py ~/STUDIES/analysis/_2022_SweepUpdate/runGSnew.py run1 

	or:
		python3 ~/MITIM/mitim_opt/opt_tools/exe/slurm.py runGSnew.py run1 --partition sched_mit_psfc
		python3 ~/MITIM/mitim_opt/opt_tools/exe/slurm.py runGSnew.py run1 --seeds 10

"""


def commander(
    script, folderWork0, num, partition, venv, n=32, hours=8, seed=0, extra_name=""
):
    folderWork = folderWork0 + extra_name + "/"

    print(f"* Launching slurm job of MITIM optimization with random seed = {seed}")

    if not os.path.exists(folderWork):
        os.system(f"mkdir {folderWork}")

    command = [
        f"source {venv}/bin/activate",
        f"python3 {script} {folderWork} --seed {seed}",
    ]

    _, fileSBTACH, _ = FARMINGtools.SLURM(
        command,
        folderWork,
        None,
        launchSlurm=True,
        nameJob=f"mitim_opt_{num}{extra_name}",
        partition=partition,
        minutes=int(60 * hours),
        ntasks=1,
        cpuspertask=n,
    )

    os.system(f"sbatch {fileSBTACH}")


def run_slurm(
    script,
    num,
    partition,
    venv,
    seeds=1,
    MainFolder="./",
    hours=8,
    n=32,
    seed_specific=0,
):
    script = IOtools.expandPath(script)
    folderWork = IOtools.expandPath(f"{MainFolder}/{num}")

    if seeds > 1:
        for i in range(seeds):
            j = i  # +10
            commander(
                script,
                folderWork,
                num,
                partition,
                venv,
                seed=j,
                extra_name=f"_s{j}",
                hours=hours,
                n=n,
            )
    else:
        commander(
            script,
            folderWork,
            num,
            partition,
            venv,
            seed=seed_specific,
            hours=hours,
            n=n,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument(
        "--partition", type=str, required=False, default="sched_mit_psfc"
    )
    parser.add_argument("--seeds", type=int, required=False, default=1)
    parser.add_argument("--hours", type=int, required=False, default=8)
    parser.add_argument("--n", type=int, required=False, default=64)
    parser.add_argument("--seed_specific", type=int, required=False, default=0)

    args = parser.parse_args()

    venv = IOtools.expandPath("~/.env/mitim-env")

    # Run
    run_slurm(
        args.script,
        args.name,
        args.partition,
        venv,
        seeds=args.seeds,
        hours=args.hours,
        n=args.n,
        seed_specific=args.seed_specific,
    )
