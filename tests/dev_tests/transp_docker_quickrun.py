"""
Quick standalone TRANSP run using the docker-lineage container (v25.1.0+, no singularity apps).

Run it FROM the folder that contains the TRANSP inputs (<runid>TR.DAT + UFILEs +
predictive namelists), e.g. a MAESTRO Beat_1 run_transp folder:

    cd /path/to/run_folder
    python $MITIM_PATH/tests/dev_tests/transp_docker_quickrun.py --tok CMOD --image /orcd/path/transp_v25.1.0.sif --trmpi 32 --toricmpi 32

On the cluster the image is the .sif converted once from the OCI archive:
    apptainer build transp_v25.1.0.sif oci-archive://transp_v25.1.0_1.tar
On a workstation with docker, pass the loaded tag instead (e.g. --image transp_v25.1.0:latest).

The runid is auto-detected from the *TR.DAT file in the current folder (or pass --runid).
Container engine is auto-detected (docker > apptainer > singularity), or pass --engine.
"""

import argparse
from pathlib import Path
from mitim_tools.transp_tools.src import TRANSPdocker


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tok", required=True, type=str, help="Tokamak name as TRANSP expects it (e.g. CMOD, D3D)")
    parser.add_argument("--image", required=True, type=str, help="Container image: .sif path (apptainer) or docker tag")
    parser.add_argument("--runid", default=None, type=str, help="Run id (default: detected from *TR.DAT in cwd)")
    parser.add_argument("--engine", default=None, type=str, choices=["docker", "apptainer", "singularity"])
    parser.add_argument("--trmpi", default=1, type=int, help="MPI tasks for NUBEAM")
    parser.add_argument("--toricmpi", default=1, type=int, help="MPI tasks for TORIC")
    parser.add_argument("--ptrmpi", default=1, type=int, help="MPI tasks for PT_SOLVER")
    args = parser.parse_args()

    folder = Path.cwd()

    runid = args.runid
    if runid is None:
        nml_files = sorted(folder.glob("*TR.DAT"))
        if len(nml_files) != 1:
            raise RuntimeError(
                f"Could not auto-detect runid: found {len(nml_files)} *TR.DAT files in {folder}, pass --runid"
            )
        runid = nml_files[0].name.removesuffix("TR.DAT")

    cdf_file = TRANSPdocker.run_transp_docker(
        folder,
        runid,
        args.tok,
        image=args.image,
        mpisettings={"trmpi": args.trmpi, "toricmpi": args.toricmpi, "ptrmpi": args.ptrmpi},
        engine=args.engine,
    )

    print(f"\nDone. Final CDF: {cdf_file}")


if __name__ == "__main__":
    main()
