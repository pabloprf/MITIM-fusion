import argparse
from mitim_tools.gacode_tools import TGYROtools, PROFILEStools

"""
Quick way to plot several tgyro results
e.g.
		read_tgyros.py --folders folderTGYRO1/ folderTGYRO2/
"""

parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="*")
args = parser.parse_args()

folders = args.folders

# ------ Read tgyros
tgyros = []
for folder in folders:
    prof_file = folder + "/input.gacode"
    prof = PROFILEStools.PROFILES_GACODE(prof_file)
    p = TGYROtools.TGYROoutput(folder, profiles=prof)
    tgyros.append(p)

# ------ Plot
TGYROtools.plotAll(tgyros, labels=None, fn=None)
