import argparse
from mitim_tools.simulation_tools.physics import GXtools
from IPython import embed
import os

"""
e.g.	read_gx.py folder
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--suffixes", required=False, type=str, nargs="*", default=None)
    parser.add_argument("--tmin", type=float, nargs="*", default=[0.0], help="Minimum time to calculate mean and std")
    parser.add_argument("--noplot", action="store_true", help="If set, it will not plot anything, just read the data.")
    parser.add_argument("--pickle", required=False, type=str, default=None, help="If set, it will save the read data in a pickle file for faster reading next time.")
    
    args = parser.parse_args()

    folders = args.folders
    tmin = args.tmin
    skip_plotting = args.noplot
    pkl = args.pickle

    suffixes = args.suffixes
    
    if suffixes is None:
        suffixes = ["" for _ in range(len(folders))]

    for i in range(len(suffixes)):
        if suffixes[i] == "_":
            suffixes[i] = ""
    
    # Read
    c = GXtools.GX()

    labels = []
    for i, folder in enumerate(folders):
        labels.append(f"case {i + 1}")
        
        c.read(
            label=labels[-1],
            folder=folder,
            tmin=tmin[i],
            suffix=suffixes[i],
        )

        if pkl is not None:
            c.save_pickle(pkl)

    if not skip_plotting:
        c.plot(labels=labels)
        c.fn.show()
    
    embed()




if __name__ == "__main__":
    main()
