import argparse
from pathlib import Path
from mitim_tools.misc_tools.GUItools import FigureNotebook
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
    parser.add_argument("--save", type=str, required=False, default=None,
                        help="Folder to save the figures.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")

    args = parser.parse_args()

    folders = args.folders
    tmin = args.tmin
    skip_plotting = args.noplot
    pkl = args.pickle

    folder_save = Path(args.save) if args.save is not None else None
    noshow = args.noshow
    dpi_fig = args.dpi

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
        fn = FigureNotebook("GX Notebook", geometry="1700x900", vertical=True, show=not noshow)
        c.plot(fn=fn, labels=labels)

        if not noshow:
            c.fn.show()

        if folder_save is not None:
            if not folder_save.exists():
                folder_save.mkdir(parents=True)
            c.fn.save(folder_save, dpi=dpi_fig)

    embed()




if __name__ == "__main__":
    main()
