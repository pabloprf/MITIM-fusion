"""
To call:  
			run_clean.py 88664P CMOD --numbers 1,2,3  
 			run_clean.py 88664P CMOD --numbers 1
 			run_clean.py 88664P CMOD --numbers 1 --to 5
"""

import argparse
import numpy as np
from mitim_tools.misc_tools import CONFIGread, IOtools
from mitim_tools.transp_tools import TRANSPtools

# User inputs
parser = argparse.ArgumentParser()
parser.add_argument("runid_str", type=str)
parser.add_argument("tokamak", type=str)
parser.add_argument("--numbers", type=str, required=False, default="1")  # 1,2,3,5
parser.add_argument("--to", type=int, required=False, default=None)
args = parser.parse_args()

if args.to is None:
    arr = [int(i) for i in args.numbers.split(",")]
else:
    arr = np.arange(int(args.numbers), int(args.to) + 1, 1)

s = CONFIGread.load_settings()
user = s[s["preferences"]["ntcc"]]["username"]


def cancelRun(namerun):
    t = TRANSPtools.TRANSP(IOtools.expandPath(s['local']['scratch']), args.tokamak)
    t.defineRunParameters(namerun, namerun, ensureMPIcompatibility= False)

    _, _, infoGrid = t.check()

    if (infoGrid is not None) and "globus" in infoGrid:
        print(f"\t>>>> Cleaning TRANSP run {namerun} created by {user}")
        t.delete()
    else:
        print(
            f"\t>>>> Run {namerun} cannot be found on the grid, not performing any active action"
        )


def cancelAll(ParamsAll, cont):
    cancelRun(ParamsAll["name"] + str(arr[cont]).zfill(2))


for i in range(len(arr)):
    cancelRun(args.runid_str + str(arr[i]).zfill(2))
