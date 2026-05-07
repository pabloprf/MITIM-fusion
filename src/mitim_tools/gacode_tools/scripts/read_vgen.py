"""
Read and plot results from an existing profiles_gen -vgen run.

    mitim_plot_vgen <folder>

<folder>  : directory containing the vgen run (i.e. the parent of the vgen/ sub-folder).
            When smooth_profiles=True was used, input.gacode.raw and input.gacode are both
            read automatically so the raw vs. smoothed comparison tab is populated.
"""

import argparse
from pathlib import Path
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gacode_tools import NEOtools

def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=str, help="Directory containing the VGEN run (parent of vgen/ sub-folder)")
    parser.add_argument("--save", type=str, nargs="?", const=IOtools.SAVE_FOLDER_AUTO_SENTINEL, required=False, default=None,
                        help=f"Folder to save the figures. If flag given without a value, defaults to '<folder>/{IOtools.SAVE_FOLDER_DEFAULT_SUBDIR}'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")

    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    folder = IOtools.expandPath(args.folder)
    folder_save = IOtools.resolve_save_folder(args.save, folder)
    noshow = args.noshow
    dpi_fig = args.dpi

    neo = NEOtools.NEO(rhos=[])
    neo.FolderGACODE = folder.parent
    neo.read_vgen(subfolder=folder.name)

    fn = FigureNotebook("NEO VGEN Notebook", geometry="1700x900", vertical=True, show=not noshow)
    neo.plot_vgen(fn=fn)

    if not noshow:
        neo.fn.show()

    if folder_save is not None:
        if not folder_save.exists():
            folder_save.mkdir(parents=True)
        neo.fn.save(folder_save, dpi=dpi_fig)

    embed()

if __name__ == "__main__":
    main()
