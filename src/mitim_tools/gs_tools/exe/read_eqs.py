import argparse
from mitim_tools.gs_tools import GEQtools

parser = argparse.ArgumentParser()
parser.add_argument("files", vtype=str, nargs="*")
args = parser.parse_args()

files = args.files

gs = []
for file in files:
    gs.append(GEQtools.MITIMgeqdsk(file))

if len(gs) == 1:
    gs[0].plot()
else:
    GEQtools.compareGeqdsk(gs)
