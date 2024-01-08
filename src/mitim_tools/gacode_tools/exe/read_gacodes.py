import argparse
from mitim_tools.gacode_tools import PROFILEStools

"""
Quick way to plot several input.gacode files together
e.g.
		read_gacodes.py --files input.gacode1 input.gacode2
"""

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="*")
args = parser.parse_args()

files = args.files

# Read
profs = []
for file in files:
    p = PROFILEStools.PROFILES_GACODE(file)
    profs.append(p)

    p.printInfo()

# Plot

PROFILEStools.plotAll(profs)
