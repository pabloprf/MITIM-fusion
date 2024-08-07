import argparse
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import CGYROtools

"""
e.g.	plot_cgyro.py folder
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    args = parser.parse_args()

    folders = args.folders

    # Read
    c = CGYROtools.CGYRO()

    labels = []
    for i, folder in enumerate(folders):
        labels.append(f"{IOtools.reducePathLevel(folder)[-1]}")
        c.read(label=labels[-1], folder=folder)

    c.plot(labels=labels)

    c.fn.show()
    embed()

if __name__ == "__main__":
    main()
