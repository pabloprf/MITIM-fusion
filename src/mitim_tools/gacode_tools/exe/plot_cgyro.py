import argparse
from mitim_tools.gacode_tools import CGYROtools

"""
e.g.	plot_cgyro.py folder
"""

parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="*")
args = parser.parse_args()

folders = args.folders

# Read
c = CGYROtools.CGYRO()

labels = []
for i, folder in enumerate(folders):
    labels.append(f"cgyro{i+1}")
    c.read(label=labels[-1], folder=folder)

c.plot(labels=labels)
