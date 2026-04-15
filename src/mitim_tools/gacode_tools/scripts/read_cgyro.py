import argparse
import pickle
from pathlib import Path
from mitim_tools.gacode_tools.utils.CGYROutils import CGYROoutput
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gacode_tools import CGYROtools
import os

"""
e.g.	read_cgyro.py folder
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--suffixes", required=False, type=str, nargs="*", default=None)
    parser.add_argument("--two", action="store_true", help="Include 2D plots")
    parser.add_argument("--linear", action="store_true", help="Just a plot of the linear spectra")
    parser.add_argument("--tmin", type=float, nargs="*", default=None, help="Minimum time to calculate mean and std")
    parser.add_argument("--scan_subfolder_id" , type=str, nargs="*", default="KY", help="If reading a linear scan, the subfolders contain this common identifier")
    parser.add_argument("--noplot", action="store_true", help="If set, it will not plot anything, just read the data.")
    parser.add_argument("--pickle", action="store_true", help="If set, it will save the read data in a pickle file for faster reading next time.")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--save", type=str, required=False, default=None,
                        help="Folder to save the figures.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")

    args = parser.parse_args()

    folders = args.folders
    linear = args.linear
    tmin = args.tmin
    include_2D = args.two
    skip_plotting = args.noplot
    pkl = args.pickle
    minimal = args.minimal

    folder_save = Path(args.save) if args.save is not None else None
    noshow = args.noshow
    dpi_fig = args.dpi

    suffixes = args.suffixes

    scan_subfolder_id = args.scan_subfolder_id

    if isinstance(scan_subfolder_id, str):
        scan_subfolder_id = [scan_subfolder_id for _ in range(len(folders))]

    if suffixes is None:
        suffixes = ["" for _ in range(len(folders))]

    for i in range(len(suffixes)):
        if suffixes[i] == "_":
            suffixes[i] = ""

    if tmin is None:
        tmin = [0.0] * len(folders)
        last_tmin_for_linear = True
    else:
        last_tmin_for_linear = False

    # Read
    c = CGYROtools.CGYRO()

    labels = []
    output_pickle = {}
    for i, folder in enumerate(folders):
        labels.append(f"case {i + 1}")

        if linear:
            c.read_linear_scan(
                label=labels[-1],
                folder=folder,
                suffix=suffixes[i],
                preffix=scan_subfolder_id[i],
                minimal=minimal
                )
        elif include_2D:
            c.read(
                label=labels[-1],
                folder=folder,
                tmin=tmin[i],
                last_tmin_for_linear=last_tmin_for_linear,
                suffix=suffixes[i],
                preffix=scan_subfolder_id[i],
                minimal=minimal
            )
        else:
            c.read(
                label=labels[-1],
                folder=folder,
                tmin=tmin[i],
                last_tmin_for_linear=last_tmin_for_linear,
                suffix=suffixes[i],
                preffix=scan_subfolder_id[i],
                minimal=minimal
            )

        if pkl:
            print("Pickling data...")
            print(c.results[labels[-1]]['output'])
            folder_abs = os.path.abspath(folder)
            simname = folder_abs.rstrip("/").split("/")[-1]
            print(f"Pickling to {simname}.pkl")

            with open(f"{folder}/{simname}_data.pkl", "wb") as f:
                pickle.dump(c.results[labels[-1]]['output'], f)
            print("Pickling done.")

    if not skip_plotting:
        if linear:
            # Plot linear spectrum
            fig = plt.figure(figsize=(15, 9))
            c.plot_quick_linear(labels=labels, fig=fig)
            if not noshow:
                plt.show()
            if folder_save is not None:
                if not folder_save.exists():
                    folder_save.mkdir(parents=True)
                GRAPHICStools.output_figure_papers(f"{folder_save}/figure", fig=fig, dpi=dpi_fig)
        else:
            fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000", show=not noshow)
            c.plot(labels=labels, fn=fn, include_2D=include_2D, common_colorbar=True)
            if not noshow:
                c.fn.show()
            if folder_save is not None:
                if not folder_save.exists():
                    folder_save.mkdir(parents=True)
                c.fn.save(folder_save, dpi=dpi_fig)

        embed()




if __name__ == "__main__":
    main()
