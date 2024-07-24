"""
This script combines runs (using the same principles as in the read optimizaiton command in opt_tools), taking various together
and joining them in the same one, to be able to do sweep plotting stuff.

e.g.

    readGS_combine.py run10 run7 run12 --n 10 --store combine_10_7_12 --order 0 4 9

Then, run1_7_comb folder can be read with standard readGS_pickle.py, as if it was originally created by a optimization workflow!

"""

import sys, argparse
import numpy as np
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.freegsu import FREEGSUmain

# -------------------------------------------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="*")  # run10 run7 run12
parser.add_argument("--n", type=int, required=False, default=None)  # --n 5
parser.add_argument(
    "--store", type=str, required=False, default=None
)  # --store combine_10_7_12
parser.add_argument(
    "--times", type=float, nargs="*", required=False, default=None
)  # --times 0 0.1 0.2 0.3 0.4
parser.add_argument(
    "--order", type=float, nargs="*", required=False, default=None
)  # --order 0 2 4   (This means: run10 @ 0, run7 @ 0.2, run12 @ 0.4)
parser.add_argument("--resol", type=int, required=False, default=None)  # --resol 129

args = parser.parse_args()

foldersWork = [
    IOtools.expandPath(folder_reduced + "/", ensurePathValid=True)
    for folder_reduced in args.folders
]
n = args.n
folderToStore = (
    IOtools.expandPath(args.store, ensurePathValid=True)
    if args.store is not None
    else None
)
times = args.times if args.times is not None else np.linspace(0, 0.6, n)
orderInEquil = args.order
nResol = args.resol

# -------------------------------------------------------------------------------------------------------------------------------------
# Grab
# -------------------------------------------------------------------------------------------------------------------------------------

opt_funs = []
for folderWork in foldersWork:
    opt_fun = STRATEGYtools.opt_evaluator(folderWork)
    opt_fun.read_optimization_results(plotYN=False, analysis_level=1)
    opt_funs.append(opt_fun)

# -------------------------------------------------------------------------------------------------------------------------------------
# Combine
# -------------------------------------------------------------------------------------------------------------------------------------

fn = FREEGSUmain.combined_analysis(
    opt_funs,
    n=n,
    times=times,
    folderToStore=folderToStore,
    orderInEquil=orderInEquil,
    nResol=nResol,
)
