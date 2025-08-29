import argparse
from mitim_tools.misc_tools import IOtools
from mitim_modules.portals.utils import PORTALSanalysis

"""
This script is useful to understand why surrogates may fail at reproducing TGLF fluxes.
You can select the iteration to use as base case to see how TGLF behaves (if it has discontinuities)
	e.g.
		runTGLFdrivesfrommitim.py --folder run11/ --ev 5 --pos 0 2 --var 0.05  --wf 0.2 1.0 --num 5

Notes:
	- wf runs scan with waveform too (slightly more expensive, as it will require 1 extra sim per run, but cheaper)
"""

# --- Inputs

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--ev", type=int, required=False, default=None)
parser.add_argument("--pos", type=int, required=False, default=[0], nargs="*")
parser.add_argument("--wf", type=float, required=False, default=None, nargs="*")
parser.add_argument("--var", type=float, required=False, default=0.01)  # Variation in inputs (1% default)
parser.add_argument("--num",type=int, required=False, default=5)
parser.add_argument("--cold_start", "-r", required=False, default=False, action="store_true")
parser.add_argument("--ion", type=int, required=False, default=2)


args = parser.parse_args()
folder = IOtools.expandPath(args.folder)
ev = args.ev
pos = args.pos
wf = args.wf
var = args.var
num = args.num
cold_start = args.cold_start
ion = args.ion

# --- Workflow

portals = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
tglf, code_settings, extraOptions = portals.extractTGLF(positions=pos, evaluation=ev, modified_profiles=True, cold_start=cold_start)

tglf.runScanTurbulenceDrives(
    subfolder="turb",
    resolutionPoints=num,
    variation=var,
    variablesDrives=["RLTS_1", "RLTS_2", "RLNS_1", "XNUE", "TAUS_2", "BETAE"],
    code_settings=code_settings,
    extraOptions=extraOptions,
    cold_start=cold_start,
    runWaveForms=wf,
    positionIon=ion,
)

tglf.plotScanTurbulenceDrives(label="turb")
