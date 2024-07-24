"""
Quick way to run TRANSP in the current folder (assumes NML and UFs exist & w/ right names)
e.g.
		run_transp.py 152895 P01 CMOD
		run_transp.py 152895 P01 CMOD --version tshare --trmpi 32 --toricmpi 32 --ptrmpi 32

"""

import sys, argparse
from mitim_tools.transp_tools import TRANSPtools
from mitim_tools.misc_tools import IOtools

# User inputs
parser = argparse.ArgumentParser()
parser.add_argument("shotnumber", type=str)
parser.add_argument("trid", type=str)
parser.add_argument("tokamak", type=str)
parser.add_argument("--version", type=str, required=False, default="pshare")
parser.add_argument("--trmpi", type=int, required=False, default=1)
parser.add_argument("--toricmpi", type=int, required=False, default=1)
parser.add_argument("--ptrmpi", type=int, required=False, default=1)
args = parser.parse_args()

runTot = args.shotnumber + args.trid

# --------------------------------------------------------------------------------------------------------------------
if args.toricmpi > 1:
    _ = IOtools.changeValue(
        IOtools.expandPath(f"./{runTot}TR.DAT"),
        "ntoric_pserve",
        "1",
        [""],
        "=",
        MaintainComments=True,
    )
else:
    _ = IOtools.changeValue(
        IOtools.expandPath(f"./{runTot}TR.DAT"),
        "ntoric_pserve",
        "0",
        [""],
        "=",
        MaintainComments=True,
    )
if args.trmpi > 1:
    _ = IOtools.changeValue(
        IOtools.expandPath(f"./{runTot}TR.DAT"),
        "nbi_pserve",
        "1",
        [""],
        "=",
        MaintainComments=True,
    )
else:
    _ = IOtools.changeValue(
        IOtools.expandPath(f"./{runTot}TR.DAT"),
        "nbi_pserve",
        "0",
        [""],
        "=",
        MaintainComments=True,
    )
if args.ptrmpi > 1:
    _ = IOtools.changeValue(
        IOtools.expandPath(f"./{runTot}TR.DAT"),
        "nptr_pserve",
        "1",
        [""],
        "=",
        MaintainComments=True,
    )
else:
    _ = IOtools.changeValue(
        IOtools.expandPath(f"./{runTot}TR.DAT"),
        "nptr_pserve",
        "0",
        [""],
        "=",
        MaintainComments=True,
    )
# --------------------------------------------------------------------------------------------------------------------

# Workflow
t = TRANSPtools.TRANSP(IOtools.expandPath("./"), args.tokamak)
t.defineRunParameters(
    runTot,
    args.shotnumber,
    mpisettings={"trmpi": args.trmpi, "toricmpi": args.toricmpi, "ptrmpi": args.ptrmpi},
)

t.run(version=args.version)
c = t.checkUntilFinished(label="run1", checkMin=30, retrieveAC=True)
