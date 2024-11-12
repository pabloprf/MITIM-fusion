import sys
import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools.utils import GACODErun

"""
run ~/MITIM/mitim/gacode_tools/scripts/run_profilesgen.py CMOD_2.7T_v3.cdf
"""

file_cdf = IOtools.expandPath(sys.argv[1], ensurePathValid=True)

folderWork, s2 = IOtools.reducePathLevel(file_cdf, level=1, isItFile=True)
nameWork = s2.split(".cdf")[0]


GACODErun.runPROFILES_GEN(
    folderWork,
    nameFiles=nameWork,
    UsePRFmodification=True,
    includeGEQ=True,
)

file_gacode = IOtools.expandPath("input.gacode", ensurePathValid=True)
file_gacode.rename(file_gacode.parent / f"{nameWork}_input.gacode")
