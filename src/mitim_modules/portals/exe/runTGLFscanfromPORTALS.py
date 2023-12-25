import argparse
import copy
import numpy as np
from mitim_modules.portals.aux import PORTALSanalysis

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

portals = PORTALSanalysis.PORTALSanalyzer(folder)
tglf, TGLFsettings, extraOptions = portals.extractTGLF(positions=pos,step=ev)

varUpDown = np.linspace(0.9, 1.1, 10)

labels = []
for param in params:
    tglf.runScan(
        subFolderTGLF="scan",
        variable=param,
        varUpDown=varUpDown,
        TGLFsettings=TGLFsettings,
        extraOptions=extraOptions,
        restart=False,
        runWaveForms=wf,
    )

    tglf.readScan(label=f"scan_{param}", variable=param)
    labels.append(f"scan_{param}")

# --- Extra TGLF plotting

tglf.plotScan(labels=labels, variableLabel="X", relativeX=True)
