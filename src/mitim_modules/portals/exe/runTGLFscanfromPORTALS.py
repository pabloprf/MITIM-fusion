import os, argparse, copy
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.opt_tools import STRATEGYtools

"""

This script is useful to understand why MITIM may fail at reproducing TGLF fluxes. You can select the iteration
to use as base case, and scan parameters to see how TGLF behaves (understand if it has discontinuities)
	e.g.
		runTGLFscanfrommitim.py --folder run11/ --ev -1 --pos 0 2 --params RLTS_2 RLTS_1 --wf 0.2 1.0

Notes:
	- wf runs scan with waveform too (slightly more expensive, as it will require 1 extra sim per run, but cheaper)
"""

# --- Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--ev", type=int, required=False, default=-1)
parser.add_argument("--pos", type=int, required=False, default=[0.5], nargs="*")
parser.add_argument("--params", type=str, required=False, default=["RLTS_2"], nargs="*")
parser.add_argument("--wf", type=float, required=False, default=None, nargs="*")

args = parser.parse_args()
folder = args.folder
ev = args.ev
params = args.params
pos = args.pos
wf = args.wf

# --- Workflow

opt_fun = STRATEGYtools.FUNmain(folder)
opt_fun.read_optimization_results(analysis_level=4, plotYN=False)

rhos = []
for i in pos:
    rhos.append(opt_fun.prfs_model.mainFunction.TGYROparameters["RhoLocations"][i])

if ev < 0:
    ev = opt_fun.res.best_absolute_index

inputgacode = f"{folder}/Execution/Evaluation.{ev}/tgyro_complete/input.gacode"
workingFolder = f"{folder}/scans{ev}/"
varUpDown = np.linspace(0.9, 1.1, 10)

tglf = TGLFtools.TGLF(rhos=rhos)
cdf = tglf.prep(workingFolder, restart=False, inputgacode=inputgacode)

labels = []
for param in params:
    extraOptions = copy.deepcopy(
        opt_fun.prfs_model.mainFunction.TGLFparameters["extraOptionsTGLF"]
    )

    tglf.runScan(
        subFolderTGLF=f"scan_{param}/",
        variable=param,
        varUpDown=varUpDown,
        TGLFsettings=opt_fun.prfs_model.mainFunction.TGLFparameters["TGLFsettings"],
        extraOptions=extraOptions,
        restart=False,
        runWaveForms=wf,
    )

    tglf.readScan(label=f"scan_{param}", variable=param)
    labels.append(f"scan_{param}")


# --- Extra TGLF plotting

tglf.plotScan(labels=labels, variableLabel="X", relativeX=True)
