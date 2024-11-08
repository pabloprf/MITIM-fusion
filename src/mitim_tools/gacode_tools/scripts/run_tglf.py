"""
This example runs TGLF from an already existing file (no normalizations if no input_gacode file provided)

	run_tglf.py run0/ input.tglf [--gacode input.gacode] [--scan RLTS_2] [--drives] [--cold_start]

Sequence:
	- If drives: do drives analysis
	- If scans (and no drives): do scans
	- If no scans and no drives: do base run

"""

import argparse
from IPython import embed
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools

def main():

# ------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("tglf", type=str)
    parser.add_argument("--gacode", required=False, type=str, default=None)
    parser.add_argument("--scan", required=False, type=str, default=None)
    parser.add_argument("--drives", required=False, default=False, action="store_true")
    parser.add_argument(
        "--cold_start", "-r", required=False, default=False, action="store_true"
    )


    args = parser.parse_args()

    folder = IOtools.expandPath(args.folder)
    input_tglf = IOtools.expandPath(args.tglf)
    input_gacode = IOtools.expandPath(args.gacode) if args.gacode is not None else None
    scan = args.scan
    drives = args.drives
    cold_start = args.cold_start

# ------------------------------------------------------------------------------
#  Preparation
# ------------------------------------------------------------------------------

    tglf = TGLFtools.TGLF()
    tglf.prep_from_tglf(folder, input_tglf, input_gacode=input_gacode)

# ------------------------------------------------------------------------------
# Workflow
# ------------------------------------------------------------------------------

    if drives:
        tglf.runScanTurbulenceDrives(subFolderTGLF="scan_turb/", TGLFsettings=None)
        tglf.plotScanTurbulenceDrives(label="scan_turb")

    elif scan is not None:
        tglf.runScan(
            subFolderTGLF="scan1/",
            variable=scan,
            varUpDown=np.linspace(0.2, 2.0, 5),
            TGLFsettings=None,
            cold_start=cold_start,
        )
        tglf.readScan(label="scan1", variable=scan)
        tglf.plotScan(labels=["scan1"], variableLabel=scan)

    else:
        tglf.run(subFolderTGLF="run1/", TGLFsettings=None, cold_start=cold_start)
        tglf.read(label="run1")
        tglf.plot(labels=["run1"])

    tglf.fn.show()
    embed()

if __name__ == "__main__":
    main()
