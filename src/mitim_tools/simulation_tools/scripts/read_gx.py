import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
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
    parser.add_argument("--tmin", type=float, nargs="*", default=[0.0],
                        help="Left edge of the signal-analysis window per folder. tmin>=0 is absolute time (a/cs). "
                             "tmin<0 is either a fraction-of-run from the end (default) or an absolute a/cs offset "
                             "from the end if --tmin_absolute is set.")
    parser.add_argument("--tmin_absolute", action="store_true",
                        help="Interpret negative --tmin values as absolute a/cs offsets from the end of the run "
                             "(e.g. --tmin -200 means the last 200 a/cs). Without this flag, negative --tmin is "
                             "a fraction-of-run (e.g. --tmin -0.3 means the last 30%% of the run).")
    parser.add_argument("--noplot", action="store_true", help="If set, it will not plot anything, just read the data.")
    parser.add_argument("--pickle", required=False, type=str, default=None, help="If set, it will save the read data in a pickle file for faster reading next time.")
    parser.add_argument("--save", type=str, nargs="?", const=IOtools.SAVE_FOLDER_AUTO_SENTINEL, required=False, default=None,
                        help=f"Folder to save the figures. If flag given without a value, defaults to '<first folder>/{IOtools.SAVE_FOLDER_DEFAULT_SUBDIR}'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")

    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    if args.save == IOtools.SAVE_FOLDER_AUTO_SENTINEL and not args.folders:
        parser.error("--save without a value needs at least one positional folder argument")

    folders = args.folders
    tmin = args.tmin
    tmin_is_rel = not args.tmin_absolute
    skip_plotting = args.noplot
    pkl = args.pickle

    folder_save = IOtools.resolve_save_folder(args.save, folders[0] if folders else None)
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
            tmin_is_rel=tmin_is_rel,
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
