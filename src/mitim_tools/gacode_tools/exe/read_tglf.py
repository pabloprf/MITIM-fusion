"""
This example reads TGLF from an already existing folder (no normalizations if no input_gacode file provided)

	read_tglf.py --folder run0/ [--suffix _0.55] [--gacode input.gacode]

"""

import argparse
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)
parser.add_argument("--suffix", required=False, type=str, default="")
parser.add_argument("--gacode", required=False, type=str, default=None)

args = parser.parse_args()

folder = IOtools.expandPath(args.folder)
input_gacode = IOtools.expandPath(args.gacode) if args.gacode is not None else None
suffix = args.suffix

tglf = TGLFtools.TGLF()
tglf.prep_from_tglf(folder, f"{folder}/input.tglf{suffix}", input_gacode=input_gacode)
tglf.read(folder=f"{folder}/", suffix=suffix, label="run1")

tglf.plotRun(labels=["run1"])
