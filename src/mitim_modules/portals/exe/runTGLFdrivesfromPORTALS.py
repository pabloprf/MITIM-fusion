import os, argparse, copy
import dill as pickle_dill
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.opt_tools import STRATEGYtools

"""

This script is useful to understand why MITIM may fail at reproducing TGLF fluxes. You can select the iteration
to use as base case to see how TGLF behaves (understand if it has discontinuities)
	e.g.
		runTGLFdrivesfrommitim.py --folder run11/ --ev -1 --pos 0 2 --var 0.05  --wf 0.2 1.0

Notes:
	- wf runs scan with waveform too (slightly more expensive, as it will require 1 extra sim per run, but cheaper)
"""

# --- Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--ev", type=int, required=False, default=-1)
parser.add_argument("--pos", type=int, required=False, default=[0.5], nargs="*")
parser.add_argument("--wf", type=float, required=False, default=None, nargs="*")
parser.add_argument(
    "--var", type=float, required=False, default=0.01
)  # Variation in inputs (1% default)

args = parser.parse_args()
folder = args.folder
ev = args.ev
pos = args.pos
wf = args.wf
var = args.var

# --- Workflow

workingFolder = f"{folder}/turb_drives/"

if not os.path.exists(workingFolder):
    os.system(f"mkdir {workingFolder}")

opt_fun = STRATEGYtools.FUNmain(folder)
opt_fun.read_optimization_results(analysis_level=4, plotYN=False)

rhos = []
for i in pos:
    rhos.append(opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"][i])

if ev < 0:
    ev = opt_fun.res.best_absolute_index

inputgacode = f"{folder}/Execution/Evaluation.{ev}/tgyro_complete/input.gacode"

if not os.path.exists(inputgacode):
    print(
        "input.gacode could not be read in Execution folder, likely because of a remote run. Resorting to MITIMextra"
    )
    self_complete = opt_fun.plot_optimization_results(analysis_level=4, plotYN=False)
    with open(self_complete.MITIMextra, "rb") as handle:
        MITIMextra_dict = pickle_dill.load(handle)

    inputgacode = f"{workingFolder}/input.gacode.start"
    MITIMextra_dict[ev]["tgyro"].profiles.writeCurrentStatus(file=inputgacode)


tglf = TGLFtools.TGLF(rhos=rhos)
cdf = tglf.prep(workingFolder, restart=False, inputgacode=inputgacode)

extraOptions = opt_fun.prfs_model.mainFunction.TGLFparameters["extraOptionsTGLF"]

extraOptions["NMODES"] = 2

name = f"turb"

tglf.runScanTurbulenceDrives(
    subFolderTGLF=name,
    resolutionPoints=5,
    variation=var,
    variablesDrives=["RLTS_1", "RLTS_2", "RLNS_1", "XNUE", "TAUS_2", "BETAE"],
    TGLFsettings=opt_fun.prfs_model.mainFunction.TGLFparameters["TGLFsettings"],
    extraOptions=extraOptions,
    restart=False,
    runWaveForms=wf,
)


# --- Extra TGLF plotting

tglf.plotScanTurbulenceDrives(label=name)
