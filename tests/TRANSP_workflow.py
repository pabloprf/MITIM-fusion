"""
Regression test to run and plot TRANSP results from an example set of input files (CMOD 88664)

To run: python3  /tests/TRANSP_workflow.py 

Notes:
- This regression test will get TORIC and NUBEAM (DD products) files
- This regression launches both TRLOOK and TRFETCHS
- This regression is also predicting Ti with TGLF default namelist in TRANSP
- This regression test will also write the TRANSP outputs (requires gacode working)

In engaging, with 32 cores, should take ~1h20min
"""

import os
import shutil
from mitim_tools.transp_tools import TRANSPtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

scratch_folder = __mitimroot__ / 'tests/scratch'
scratch_folder.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------
# 	Input data
# ------------------------------------------------------------------------------------

folderInput = __mitimroot__ / "tests" / "data" / "FolderTRANSP"

# ------------------------------------------------------------------------------------
# 	Workflow
# ------------------------------------------------------------------------------------

folder = __mitimroot__ / "tests" / "scratch" / "transp_test"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

runid = 'Z99'

# ---- Prepare NML and UFILES
shutil.copytree(folderInput, folder, dirs_exist_ok=True)
(folder / '12345X01TR.DAT').replace(folder / f'12345{runid}TR.DAT')
# ---------------------------

# Define TRANSP class and where it is run
t = TRANSPtools.TRANSP(folder, "CMOD")

# Define user and run parameters
t.defineRunParameters(
    "12345" + runid, "12345",
    mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 32},
    minutesAllocation = 10
)

# Submit run
t.run()

# Check
c = t.checkUntilFinished(
    label="run1", checkMin=2, grabIntermediateEachMin=20, retrieveAC=True
)

# Plot
t.plot(label="run1")

# Required if running in non-interactive mode
t.fn.show()