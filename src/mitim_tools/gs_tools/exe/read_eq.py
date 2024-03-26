import argparse
from mitim_tools.gs_tools import GEQtools

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="*")
args = parser.parse_args()

files = args.files

gs = []
for file in files:
    gs.extend(GEQtools.MITIMgeqdsk.timeslices(file))

if len(gs) == 1:
    gs[0].plot()
else:
    axs, fn = GEQtools.compareGeqdsk(gs)
