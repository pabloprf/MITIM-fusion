"""
This example reads TGLF from an already existing folder (no normalizations if no input_gacode file provided)

	read_tglf.py run0/ [--suffixes _0.55] [--gacode input.gacode]

"""

import argparse
from pathlib import Path
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gacode_tools import TGLFtools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--suffixes", required=False, type=str, nargs="*", default=None)
    parser.add_argument("--gacode", required=False, type=str, default=None)
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

    folders = [IOtools.expandPath(folder) for folder in args.folders]
    input_gacode = IOtools.expandPath(args.gacode) if args.gacode is not None else None
    suffixes = args.suffixes

    folder_save = IOtools.resolve_save_folder(args.save, folders[0] if folders else None)
    noshow = args.noshow
    dpi_fig = args.dpi

    if suffixes is None:
        suffixes = ["" for _ in range(len(folders))]

    for i in range(len(suffixes)):
        if suffixes[i] == "_":
            suffixes[i] = ""

    tglf = TGLFtools.TGLF()
    tglf.prep_from_file(
        folders[0], folders[0] / f"input.tglf{suffixes[0]}", input_gacode=input_gacode
    )
    for i, folder in enumerate(folders):
        tglf.read(folder=folder, suffix=suffixes[i], label=f"run{i}")

    fn = FigureNotebook("TGLF MITIM Notebook", geometry="1700x900", vertical=True, show=not noshow)
    tglf.plot(fn=fn, labels=[f"run{i}" for i in range(len(folders))])

    if not noshow:
        tglf.fn.show()

    if folder_save is not None:
        if not folder_save.exists():
            folder_save.mkdir(parents=True)
        tglf.fn.save(folder_save, dpi=dpi_fig)

    embed()

if __name__ == "__main__":
    main()
