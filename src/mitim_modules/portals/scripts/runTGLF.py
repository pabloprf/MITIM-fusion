import argparse
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_modules.portals.utils import PORTALSanalysis

"""
This script is useful to understand why PORTALS may fail at reproducing TGLF fluxes. You can select the iteration
to use as base case, and scan parameters to see how TGLF behaves (understand if it has discontinuities)
	e.g.
		runTGLF.py --folder run11/ --ev 5 --pos 0 2 --params RLTS_2 RLTS_1 --wf 0.2 1.0 --var 0.05 --num 10

Notes:
	- wf runs scan with waveform too (slightly more expensive, as it will require 1 extra sim per run, but cheaper)
    - drives will simply run the drives analysis, ignoring the params option
"""

# --- Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--ev", type=int, required=False, default=None)
parser.add_argument("--pos", type=int, required=False, default=[0.5], nargs="*")
parser.add_argument("--params", type=str, required=False, default=["RLTS_2"], nargs="*")
parser.add_argument("--wf", type=float, required=False, default=[], nargs="*")
parser.add_argument("--var", type=float, required=False, default=0.05)  # Variation in inputs (5% default)
parser.add_argument("--num", type=int, required=False, default=10)
parser.add_argument("--cold_start", "-r", required=False, default=False, action="store_true")
parser.add_argument("--drives", required=False, default=False, action="store_true")
parser.add_argument("--ion", type=int, required=False, default=2)

args = parser.parse_args()
folder = IOtools.expandPath(args.folder)
ev = args.ev
params = args.params
pos = args.pos
wf = args.wf
var = args.var
cold_start = args.cold_start
drives = args.drives
num = args.num
ion = args.ion

# --- Workflow

portals = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
tglf, TGLFsettings, extraOptions = portals.extractTGLF(positions=pos, evaluation=ev, modified_profiles=True, cold_start=cold_start)

if not drives:
    varUpDown = np.linspace(1.0 - var, 1.0 + var, num)

    labels = []
    for param in params:
        tglf.runScan(
            subfolder="scan",
            variable=param,
            varUpDown=varUpDown,
            TGLFsettings=TGLFsettings,
            extraOptions=extraOptions,
            cold_start=cold_start,
            runWaveForms=wf,
        )

        tglf.readScan(label=f"scan_{param}", variable=param, positionIon = ion)
        labels.append(f"scan_{param}")

    # --- Extra TGLF plotting

    tglf.plotScan(labels=labels, variableLabel="X", relativeX=True)

else:
    tglf.runScanTurbulenceDrives(
        subfolder="turb",
        resolutionPoints=5,
        variation=var,
        variablesDrives=["RLTS_1", "RLTS_2", "RLNS_1", "XNUE", "TAUS_2", "BETAE"],
        TGLFsettings=TGLFsettings,
        extraOptions=extraOptions,
        cold_start=cold_start,
        runWaveForms=wf,
    )

    tglf.plotScanTurbulenceDrives(label="turb")
