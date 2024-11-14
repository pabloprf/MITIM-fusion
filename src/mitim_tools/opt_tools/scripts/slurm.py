import os
import argparse
from mitim_tools.misc_tools import FARMINGtools, IOtools

"""
This script is used to launch a slurm job of a MITIM optimization.
The main script should receive: folderWork and --seed

To call:
	cd folder_where_all_runs_are/
	python3 ~/MITIM/mitim_opt/opt_tools/scripts/slurm.py ~/STUDIES/analysis/_2022_SweepUpdate/runGSnew.py run1 

	or:
		python3 ~/MITIM/mitim_opt/opt_tools/scripts/slurm.py runGSnew.py --folder run1 --partition sched_mit_psfc
		python3 ~/MITIM/mitim_opt/opt_tools/scripts/slurm.py runGSnew.py --folder run1 --seeds 10

"""


def commander(
    script,
    folderWork0,
    num,
    partition,
    venv,
    n=32,
    hours=8,
    seed=0,
    extra_name="",
    extra=None,
):
    folderWork = IOtools.expandPath(folderWork0) / f"{extra_name}"

    print(f"* Launching slurm job of MITIM optimization with random seed = {seed}")

    folderWork.mkdir(parents=True, exist_ok=True)

    if extra is not None:
        extra_str = " ".join([str(e) for e in extra])
    else:
        extra_str = ""

    command = [
        f"source {venv}/bin/activate",
        f"python3 {script} {folderWork} {extra_str} --seed {seed}",
    ]

    _, fileSBATCH, _ = FARMINGtools.create_slurm_execution_files(
        command,
        folderWork,
        None,
        folder_local=folderWork,
        launchSlurm=True,
        nameJob=f"mitim_opt_{num}{extra_name}",
        slurm={"partition": partition},
        minutes=int(60 * hours),
        ntasks=1,
        cpuspertask=n,
    )

    os.system(f"sbatch {fileSBATCH}")


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
):
    script = IOtools.expandPath(script)
    folderWork = IOtools.expandPath(folder)

    num = folderWork.name

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
                extra=extra,
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
            extra=extra,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("--folder", type=str, required=False, default="run1/")
    parser.add_argument(
        "--partition",
        type=str,
        required=False,
        default=IOtools.expandPath("$MITIM_PARTITION"),
    )
    parser.add_argument("--seeds", type=int, required=False, default=1)
    parser.add_argument(
        "--env", type=str, required=False, default=IOtools.expandPath("~/env/mitim-env")
    )
    parser.add_argument("--hours", type=int, required=False, default=8)
    parser.add_argument("--n", type=int, required=False, default=16)
    parser.add_argument("--seed_specific", type=int, required=False, default=0)
    parser.add_argument("--extra", type=float, required=False, default=None, nargs="*")

    args = parser.parse_args()

    # Run
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
    )

if __name__ == "__main__":
    main()
