"""
CAPABILITY: TRANSP run from a folder of input files
---------------------------------------------------
This script teaches how to launch, monitor and plot a TRANSP run from a
prepared set of input files (namelist + UFILES), here an example C-Mod
discharge. TRANSP runs on the machine configured for "transp" in
config_user.json — with 32 cores, this example takes over an hour, so this is
NOT a quick test.

Key teaching points:
    1. TRANSP inputs are a namelist (<shot><runid>TR.DAT) and UFILES living
       together in a folder; the tokamak name ("CMOD" here) selects the
       machine-specific settings.
    2. defineRunParameters() sets the runid and shot, the MPI resources of
       each parallelizable module (trmpi: main transport, toricmpi: TORIC ICRF
       solver, ptrmpi: NUBEAM fast ions) and the SLURM allocation time.
    3. run() submits and returns; checkUntilFinished() polls every `checkMin`
       minutes, optionally retrieving intermediate results
       (grabIntermediateEachMin) so progress can be inspected mid-run, and
       retrieveAC=True also fetches the TORIC/NUBEAM output files.
    4. plot() opens the standard MITIM TRANSP notebook with the time traces
       and profiles of the run.
"""

import shutil
from mitim_tools.transp_tools import TRANSPtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Example TRANSP inputs distributed with MITIM (C-Mod shot 12345: TR.DAT namelist + UFILES)
folderInput = __mitimroot__ / "tests" / "data" / "FolderTRANSP"

# Working folder of the run
folder = __mitimroot__ / "tests" / "scratch" / "capability_transp"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Stage the input files and rename the namelist to the chosen runid
# ---------------------------------------------------------------------------------------------------------------------

runid = "Z99"

shutil.copytree(folderInput, folder, dirs_exist_ok=True)
# The namelist must be named <shot><runid>TR.DAT
(folder / "12345X01TR.DAT").replace(folder / f"12345{runid}TR.DAT")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Define the run and submit it
# ---------------------------------------------------------------------------------------------------------------------

# The tokamak name selects machine-specific conventions
t = TRANSPtools.TRANSP(folder, "CMOD")

t.defineRunParameters(
    "12345" + runid,  # full run name: shot + runid
    "12345",          # shot number
    # MPI resources per parallelizable module: main transport (trmpi),
    # TORIC ICRF solver (toricmpi) and NUBEAM fast ions (ptrmpi)
    mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 32},
    minutesAllocation=10,
)

# Submit and return (TRANSP runs detached on the configured machine)
t.run()

# ---------------------------------------------------------------------------------------------------------------------
# 3. Monitor until finished, retrieving intermediate results along the way
# ---------------------------------------------------------------------------------------------------------------------

c = t.checkUntilFinished(
    label="run1",
    checkMin=2,                  # poll the run status every 2 minutes
    grabIntermediateEachMin=20,  # fetch intermediate outputs every 20 minutes
    retrieveAC=True,             # also fetch the TORIC and NUBEAM output files
)

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (t.fn); show() opens the GUI
t.plot(label="run1")
t.fn.show()
