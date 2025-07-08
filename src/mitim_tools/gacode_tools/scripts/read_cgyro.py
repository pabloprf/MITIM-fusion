import argparse
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import CGYROtools

"""
e.g.	read_cgyro.py folder
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--linear", action="store_true", help="linear run")
    parser.add_argument("--tmin", type=float, nargs="*", default=None, help="Minimum time to calculate mean and std")
    args = parser.parse_args()

    folders = args.folders
    linear = args.linear
    tmin = args.tmin
    
    if tmin is None:
        tmin = [0.0] * len(folders)
    
    # Read
    c = CGYROtools.CGYRO()

    labels = []
    for i, folder in enumerate(folders):
        labels.append(f"case {i + 1}")
        c.read(label=labels[-1], folder=folder, tmin=tmin[i])

    if linear:
        # Plot linear spectrum
        c.plotLS(labels=labels)
    else:
        c.plot(labels=labels)

    c.fn.show()
    embed()

if __name__ == "__main__":
    main()
