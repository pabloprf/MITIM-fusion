"""
This assumes that I have "analyzed_results" of the freegsu class and then I'm reading pickles
To avoid re-running them all of the time.
For comparison among optimizations.
Can only deal with freegs runs run through MITIM. I cannot combine stuff
"""

import sys
import pickle
import copy
import numpy as np
from mitim_modules.freegsu.utils import FREEGSUplotting
from mitim_modules.freegsu import FREEGSUtools
from mitim_tools.misc_tools.IOtools import printMsg as print

files = sys.argv[1:]

# ---------------------------
# Read pickles, extract prfs
# ---------------------------

onlyLast = len(files) > 1

prfs = []
for file in files:
    m = pickle.load(open(f"{file}/Outputs/final_analysis/results.pkl", "rb"))
    if onlyLast:
        prfs.extend([m["prfs"][-1]])
    else:
        prfs.extend(m["prfs"])

params = copy.deepcopy(m["params"])

# ---------------------------------------------------
# Re calculate metrics with these extended prfs
# ---------------------------------------------------

metrics, opts = FREEGSUtools.calculateMetrics(prfs, separatrixPoint=None)

params["times"] = np.arange(0, 0.3 * len(prfs), 0.3)
print("\n*** PRF WARNING: Remember to check the timing for voltages!!\n", typeMsg="w")

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

FREEGSUplotting.plotResult(
    prfs, metrics, m["function_parameters"]["Constraints"], ProblemExtras=params
)
