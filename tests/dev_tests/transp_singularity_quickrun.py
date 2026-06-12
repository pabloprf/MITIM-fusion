"""
Quick standalone TRANSP run using the OLD singularity-apps container (23.x), for direct
comparison (correctness and timing) against transp_docker_quickrun.py on the same inputs.

Run it FROM the folder that contains the TRANSP inputs (<runid>TR.DAT + UFILEs +
predictive namelists):

    cd /path/to/run_folder
    python $MITIM_PATH/tests/dev_tests/transp_singularity_quickrun.py --tok CMOD --trmpi 8 --toricmpi 8

The image defaults to $TRANSP_SINGULARITY (the production sif, e.g. transp_23.1.sif);
pass --image to override. The runid is auto-detected from the *TR.DAT file in cwd.

The workflow mirrors what MITIM's runSINGULARITY does through the container apps:
transp-bashrc (sourced by every app) + pretr / trdat / link / transp / trlook.
Each phase is timed and everything is logged to <runid>tr.log.
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools import NMLtools


def run_phase(name, command, folder, log_file):
    print(f"\n>>> [{name}] {command}", flush=True)
    t0 = time.time()
    with open(log_file, "a") as f:
        f.write(f"\n########## [{name}] {command}\n")
        f.flush()
        process = subprocess.Popen(
            command, shell=True, cwd=folder,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        process.wait()
    dt = time.time() - t0
    print(f">>> [{name}] finished in {dt/60:.2f} min (exit code {process.returncode})", flush=True)
    return dt, process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tok", required=True, type=str, help="Tokamak name as TRANSP expects it (e.g. CMOD, D3D)")
    parser.add_argument("--image", default=os.environ.get("TRANSP_SINGULARITY"), type=str,
                        help="Path to the singularity-apps .sif (default: $TRANSP_SINGULARITY)")
    parser.add_argument("--runid", default=None, type=str, help="Run id (default: detected from *TR.DAT in cwd)")
    parser.add_argument("--trmpi", default=1, type=int, help="MPI tasks for NUBEAM")
    parser.add_argument("--toricmpi", default=1, type=int, help="MPI tasks for TORIC")
    parser.add_argument("--ptrmpi", default=1, type=int, help="MPI tasks for PT_SOLVER")
    args = parser.parse_args()

    if args.image is None:
        raise RuntimeError("No image: pass --image or set $TRANSP_SINGULARITY")

    folder = Path.cwd()

    runid = args.runid
    if runid is None:
        nml_files = sorted(folder.glob("*TR.DAT"))
        if len(nml_files) != 1:
            raise RuntimeError(
                f"Could not auto-detect runid: found {len(nml_files)} *TR.DAT files in {folder}, pass --runid"
            )
        runid = nml_files[0].name.removesuffix("TR.DAT")

    mpisettings = {"trmpi": args.trmpi, "toricmpi": args.toricmpi, "ptrmpi": args.ptrmpi}
    nparallel = max([1] + [int(v) for v in mpisettings.values()])
    # Same convention as TRANSPsingularity.defineRunParameters: 1 means "not used", exported as 0
    mpisettings = {k: (0 if int(v) == 1 else int(v)) for k, v in mpisettings.items()}

    # Point the namelist input paths to this folder (copied inputs carry stale absolute paths)
    nshot = IOtools.findValue(folder / f"{runid}TR.DAT", "nshot", "=")
    NMLtools.adaptNML(folder, runid, int(float(nshot)), str(folder))

    # pretr answers, same as runSINGULARITY
    with open(folder / "pre_mitim", "w") as f:
        f.write("00\nY\nLaunched by MITIM\nx\n")

    # transp-bashrc, sourced by every container app (same content runSINGULARITY generates)
    with open(folder / "transp-bashrc", "w") as f:
        f.write(f"""
export WORKDIR=./
export RESULTDIR={folder}/results/
mkdir -p {folder}/results/
export DATADIR={folder}/data/
mkdir -p {folder}/data/
export TMPDIR_TR={folder}/tmp/
mkdir -p {folder}/tmp/
export NPROCS={nparallel}
export NBI_NPROCS={mpisettings["trmpi"]}
export NTOR_NPROCS={mpisettings["toricmpi"]}
export NPTR_NPROCS={mpisettings["ptrmpi"]}
export NGEN_NPROCS=0
export NCQL3D_NPROCS=0
""")

    engine = "apptainer" if shutil.which("apptainer") else "singularity"
    if shutil.which(engine) is None:
        raise RuntimeError("No apptainer/singularity in PATH")

    # Bind the top-level folder, as in TRANSPsingularity (e.g. /pool001, /nobackup1, /orcd)
    start_folder = str(folder).split("/")[1]
    txt_bind = f"--bind /{start_folder} " if start_folder not in ["home", "Users"] else ""

    txt_mpi = " MPI" if nparallel > 1 else ""
    sif = args.image

    phases = [
        ("pretr",  f"{engine} run {txt_bind}--app pretr {sif} {args.tok}{txt_mpi} {runid} < pre_mitim"),
        ("trdat",  f"{engine} run {txt_bind}--app trdat {sif} {args.tok} {runid} w q"),
        ("link",   f"{engine} run {txt_bind}--app link {sif} {runid}"),
        ("transp", f"{engine} run {txt_bind}--cleanenv --app transp {sif} {runid}"),
        ("trlook", f"{engine} run {txt_bind}--app trlook {sif} {args.tok} {runid}"),
    ]

    log_file = folder / f"{runid}tr.log"
    print(f"Running TRANSP ({runid}, {args.tok}) via {engine} + singularity apps, NPROCS = {nparallel}")
    print(f"Image: {sif}")
    print(f"Log:   {log_file}")

    timings = {}
    for name, command in phases:
        dt, rc = run_phase(name, command, folder, log_file)
        timings[name] = dt
        if rc != 0:
            print(f"\nPhase [{name}] failed with exit code {rc}, stopping (check {log_file.name})")
            break

    print("\n================= Timing summary =================")
    for name, dt in timings.items():
        print(f"  {name:8s} : {dt/60:8.2f} min")
    print(f"  {'TOTAL':8s} : {sum(timings.values())/60:8.2f} min")

    cdf_file = folder / f"{runid}.CDF"
    if cdf_file.exists():
        print(f"\nDone. Final CDF: {cdf_file}")
    else:
        print(f"\nNo {cdf_file.name} produced, check {log_file.name}")


if __name__ == "__main__":
    main()
