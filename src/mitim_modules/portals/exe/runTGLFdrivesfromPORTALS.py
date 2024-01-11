import argparse
from mitim_modules.portals.aux import PORTALSanalysis

"""
This script is useful to understand why surrogates may fail at reproducing TGLF fluxes.
You can select the iteration to use as base case to see how TGLF behaves (if it has discontinuities)
	e.g.
		runTGLFdrivesfrommitim.py --folder run11/ --ev 5 --pos 0 2 --var 0.05  --wf 0.2 1.0

Notes:
	- wf runs scan with waveform too (slightly more expensive, as it will require 1 extra sim per run, but cheaper)
"""

# --- Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--ev", type=int, required=False, default=None)
parser.add_argument("--pos", type=int, required=False, default=[0], nargs="*")
parser.add_argument("--wf", type=float, required=False, default=None, nargs="*")
parser.add_argument(
    "--var", type=float, required=False, default=0.01
)  # Variation in inputs (1% default)
parser.add_argument("-r", required=False, default=False, action='store_true')

args = parser.parse_args()
folder = args.folder
ev = args.ev
pos = args.pos
wf = args.wf
var = args.var
restart = args.r

# --- Workflow

portals = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
tglf, TGLFsettings, extraOptions = portals.extractTGLF(positions=pos, evaluation=ev)

tglf.runScanTurbulenceDrives(
    subFolderTGLF="turb",
    resolutionPoints=5,
    variation=var,
    variablesDrives=["RLTS_1", "RLTS_2", "RLNS_1", "XNUE", "TAUS_2", "BETAE"],
    TGLFsettings=TGLFsettings,
    extraOptions=extraOptions,
    restart=restart,
    runWaveForms=wf,
)

tglf.plotScanTurbulenceDrives(label="turb")
