"""
Quick way to look at (and plot) a TRANSP run in the current folder (either active or finished)
e.g.
		run_look.py 152895P01 CMOD
		run_look.py 152895P01 CMOD --nofull --plot --remove
"""

import sys, argparse
from mitim_tools.transp_tools import TRANSPtools
from mitim_tools.misc_tools import IOtools

# User inputs
parser = argparse.ArgumentParser()
parser.add_argument("runTot", type=str)
parser.add_argument("tokamak", type=str)
parser.add_argument("--nofull", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--remove", action="store_true")
args = parser.parse_args()

# Workflow
t = TRANSPtools.TRANSP(IOtools.expandPath("./"), args.tokamak)
t.defineRunParameters(args.runTot, args.runTot, ensureMPIcompatibility= False)

# Determine state
info, status, infoGrid = t.check()

retrieveAC = True

if status == 1:
    _ = t.fetch(label="run1", retrieveAC=retrieveAC)
else:
    t.get(label="run1", fullRequest=not args.nofull, retrieveAC=retrieveAC)

if bool(int(args.remove)):
    t.delete()

if bool(int(args.plot)):
    t.plot(label="run1")
