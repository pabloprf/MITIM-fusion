"""
fetch_cgyro_intermediate.py — pull whatever CGYRO output a PORTALS-submitted run
has produced so far on the cluster, organize it locally, read it, and plot the
per-rho results.

The script is intentionally self-contained:
  - It does NOT cancel the remote slurm job.
  - It does NOT delete the remote scratch folder.
  - It does NOT touch any PORTALS-side pickles (optimization_object.pkl,
    optimization_extra.pkl) or rewrite the cgyro_submission.json.
  - All local artifacts land under the user-supplied output folder.

Usage:
    python fetch_cgyro_intermediate.py \\
        /path/to/cgyro_submission.json \\
        /path/to/local/output_folder \\
        [--tmin -0.3] [--tmin-is-rel]

Remote side-effect: a temporary `mitim_receive.tar.gz` is created in the job's
remote folder and deleted again after download. The CGYRO simulation files
themselves are untouched.
"""

import argparse
import json
from pathlib import Path

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


def fetch_and_plot(submission_json, output_folder, tmin=0.0, tmin_is_rel=True):
    submission_json = Path(submission_json).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()

    if not submission_json.is_file():
        raise FileNotFoundError(f"cgyro_submission.json not found at {submission_json}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # Peek at the JSON to decide single vs batched, and pull the rho list
    # --------------------------------------------------------------------
    with open(submission_json) as f:
        meta = json.load(f)

    mode = meta.get("mode", "single")
    if mode != "single":
        raise NotImplementedError(
            f"This script currently supports single-plasma CGYRO submissions "
            f"(mode='single'), got mode={mode!r}. Batched submissions would "
            f"need per-plasma handling."
        )

    base_subfolder = meta.get("base_subfolder") or "base_cgyro"
    rhos_serialized = meta["kwargs_organize"]["code_executor"][base_subfolder]
    rhos = sorted(float(r) for r in rhos_serialized.keys())
    print(f"- Submission: jobid={meta['job'].get('jobid')} on "
          f"{meta['job'].get('machineSettings', {}).get('machine')}")
    print(f"- Remote folder: {meta['job'].get('folderExecution')}")
    print(f"- base_subfolder: {base_subfolder}")
    print(f"- Rhos ({len(rhos)}): {rhos}")
    print(f"- Local output: {output_folder}")

    # --------------------------------------------------------------------
    # Build a minimal CGYRO object + hydrate from the submission JSON
    # --------------------------------------------------------------------
    cgyro = CGYROtools.CGYRO(rhos=rhos)
    cgyro.load_submission_state(submission_json)

    # --------------------------------------------------------------------
    # Redirect EVERY local path into the user's output folder so the
    # script works regardless of where the original PORTALS run lived.
    # --------------------------------------------------------------------
    local_base = output_folder / base_subfolder
    local_base.mkdir(parents=True, exist_ok=True)

    cgyro.simulation_job.folder_local = output_folder
    cgyro.kwargs_organize["tmpFolder"] = output_folder / "tmp_retrieve"
    for sub, rhos_map in cgyro.kwargs_organize["code_executor"].items():
        for rho, v in rhos_map.items():
            v["folder"] = local_base
            v["folder"].mkdir(parents=True, exist_ok=True)
    cgyro.FolderSimLast = local_base

    # --------------------------------------------------------------------
    # Pull from remote. retrieve(check_if_files_received=False) skips the
    # 60-second retry + mandatory-file check, because partial output is
    # expected for an intermediate fetch. The underlying tar on remote
    # tolerates missing files (it logs an error but still emits the
    # tarball with whatever is available).
    # --------------------------------------------------------------------
    print("- Connecting to remote...")
    cgyro.simulation_job.connect()
    try:
        print("- Retrieving whatever output is available so far (intermediate)...")
        cgyro.simulation_job.retrieve(check_if_files_received=False)
    finally:
        cgyro.simulation_job.close()

    print("- Organizing pulled files under output folder...")
    cgyro._organize_results(**cgyro.kwargs_organize)

    # --------------------------------------------------------------------
    # Read + plot. CGYRO.plot iterates per-rho internally, producing one
    # multi-tab FigureNotebook with Fluxes / Intensities / Cross-phases /
    # Turbulence panels across every radius.
    # --------------------------------------------------------------------
    print(f"- Reading CGYRO outputs (tmin={tmin}, tmin_is_rel={tmin_is_rel})...")
    cgyro.read(label=base_subfolder, folder=local_base,
               tmin=tmin, tmin_is_rel=tmin_is_rel)

    print("- Plotting...")
    cgyro.plot(labels=[base_subfolder])

    return cgyro


def main():
    p = argparse.ArgumentParser(
        description="Pull intermediate CGYRO results from a PORTALS-submitted slurm job and plot them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("submission_json",
                   help="Path to the cgyro_submission.json written by PORTALS at submit time.")
    p.add_argument("output_folder",
                   help="Local folder where pulled results will be stored.")
    p.add_argument("--tmin", type=float, default=-0.3,
                   help="Left edge of the signal-analysis window for read(). "
                        "Negative + tmin-is-rel -> fraction of total sim time from the end.")
    p.add_argument("--tmin-is-rel", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether tmin<0 is interpreted as a relative fraction of total sim time.")
    args = p.parse_args()

    fetch_and_plot(
        args.submission_json,
        args.output_folder,
        tmin=args.tmin,
        tmin_is_rel=args.tmin_is_rel,
    )


if __name__ == "__main__":
    main()
