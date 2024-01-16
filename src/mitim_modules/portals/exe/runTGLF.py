import argparse
import numpy as np
from mitim_modules.portals.aux import PORTALSanalysis

"""
This script is useful to understand why PORTALS may fail at reproducing TGLF fluxes. You can select the iteration
to use as base case, and scan parameters to see how TGLF behaves (understand if it has discontinuities)
	e.g.
		runTGLF.py --folder run11/ --ev 5 --pos 0 2 --params RLTS_2 RLTS_1 --wf 0.2 1.0 --var 0.05

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
parser.add_argument("--wf", type=float, required=False, default=None, nargs="*")
parser.add_argument(
    "--var", type=float, required=False, default=0.05
)  # Variation in inputs (10% default)
parser.add_argument(
    "--restart", "-r", required=False, default=False, action="store_true"
)
parser.add_argument("--drives", required=False, default=False, action="store_true")

args = parser.parse_args()
folder = args.folder
ev = args.ev
params = args.params
pos = args.pos
wf = args.wf
var = args.var
restart = args.restart
drives = args.drives

# --- Workflow

portals = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
tglf, TGLFsettings, extraOptions = portals.extractTGLF(positions=pos, evaluation=ev)

if not drives:
    varUpDown = np.linspace(1.0 - var, 1.0 + var, 10)

    labels = []
    for param in params:
        tglf.runScan(
            subFolderTGLF="scan",
            variable=param,
            varUpDown=varUpDown,
            TGLFsettings=TGLFsettings,
            extraOptions=extraOptions,
            restart=restart,
            runWaveForms=wf,
        )

        tglf.readScan(label=f"scan_{param}", variable=param)
        labels.append(f"scan_{param}")

    # --- Extra TGLF plotting

    tglf.plotScan(labels=labels, variableLabel="X", relativeX=True)

else:
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
