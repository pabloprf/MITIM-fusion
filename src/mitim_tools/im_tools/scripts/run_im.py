"""
INPUTS:
	1) Inputs folder, where mitim.namelist is (mind folders written in namelist)
	2) params.in.1 containing values of DVs
	3,optional) Minimum debug option

EXAMPLE:
	python mitim.py ./ 7 (optional: --debug 0)

NOTES:
	- When running standalone, *mitim.namelist* and *params.in.X* must be in Evaluation folder
	- When running within a BO workflow, *mitim.namelist* must be where the BO is called
"""

import os, sys, argparse
from mitim_tools.misc_tools import IOtools
from mitim_tools.im_tools import IMtools

# User inputs
parser = argparse.ArgumentParser()
parser.add_argument("GeneralFolder", type=str)
parser.add_argument("numEv", type=int)
parser.add_argument("--debug", type=int, required=False, default=None)
args = parser.parse_args()

SpecificParams = {
    "IsItSingle": True,
    "forceMinimumDebug": args.debug,
    "automaticProcess": False,
}

IMnamelist = IOtools.expandPath(f"{args.GeneralFolder}/im.namelist")
FolderEvaluation = IOtools.expandPath(args.GeneralFolder, ensurePathValid=True) + "/"


im = IMtools.runIMworkflow(
    IMnamelist, FolderEvaluation, args.numEv, SpecificParams=SpecificParams
)
