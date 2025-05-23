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
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

files = sys.argv[1:]
files = [IOtools.expandPath(file) for file in files]

# ---------------------------
# Read pickles, extract mitims
# ---------------------------

onlyLast = len(files) > 1

mitims = []
for file in files:
    m = pickle.load(open(file / "Outputs" / "final_analysis" / "results.pkl", "rb"))
    if onlyLast:
        mitims.extend([m["mitims"][-1]])
    else:
        mitims.extend(m["mitims"])

params = copy.deepcopy(m["params"])

# ---------------------------------------------------
# Re calculate metrics with these extended mitims
# ---------------------------------------------------

metrics, opts = FREEGSUtools.calculateMetrics(mitims, separatrixPoint=None)

params["times"] = np.arange(0, 0.3 * len(mitims), 0.3)
print("\n*** Remember to check the timing for voltages!!\n", typeMsg="w")

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

FREEGSUplotting.plotResult(
    mitims, metrics, m["function_parameters"]["Constraints"], ProblemExtras=params
)
