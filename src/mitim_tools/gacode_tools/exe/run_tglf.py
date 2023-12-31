"""
This example runs TGLF from an already existing file (no normalizations if no input_gacode file provided)

	run_tglf.py --folder run0/ --tglf input.tglf [--gacode input.gacode] [--scan RLTS_2] [--drives True]

Sequence:
	- If drives: do drives analysis
	- If scans (and no drives): do scans
	- If no scans and no drives: do base run

"""

import argparse
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools

# ------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--tglf", required=True, type=str)
parser.add_argument("--gacode", required=False, type=str, default=None)
parser.add_argument("--scan", required=False, type=str, default=None)
parser.add_argument("--drives", required=False, type=bool, default=False)

args = parser.parse_args()

folder = IOtools.expandPath(args.folder)
input_tglf = IOtools.expandPath(args.tglf)
input_gacode = IOtools.expandPath(args.gacode) if args.gacode is not None else None
scan = args.scan
drives = args.drives

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
    )
    tglf.readScan(label="scan1", variable=scan)
    tglf.plotScan(labels=["scan1"], variableLabel=scan)

else:
    tglf.run(subFolderTGLF="run1/", TGLFsettings=None)
    tglf.read(label="run1")
    tglf.plotRun(labels=["run1"])
