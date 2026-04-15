import argparse
from pathlib import Path
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gacode_tools import TGYROtools, PROFILEStools

"""
Quick way to plot several tgyro results
e.g.
		read_tgyros.py --folders folderTGYRO1/ folderTGYRO2/
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--save", type=str, required=False, default=None,
                        help="Folder to save the figures.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")
    args = parser.parse_args()

    folders = [IOtools.expandPath(folder) for folder in args.folders]
    folder_save = Path(args.save) if args.save is not None else None
    noshow = args.noshow
    dpi_fig = args.dpi

    # ------ Read tgyros
    tgyros = []
    for folder in folders:
        prof_file = folder / "input.gacode"
        prof = PROFILEStools.gacode_state(prof_file)
        p = TGYROtools.TGYROoutput(folder, profiles=prof)
        tgyros.append(p)

    # ------ Plot
    fn = FigureNotebook("TGYRO Output Notebook", geometry="1800x900", show=not noshow)
    TGYROtools.plotAll(tgyros, labels=None, fn=fn)

    if not noshow:
        fn.show()

    if folder_save is not None:
        if not folder_save.exists():
            folder_save.mkdir(parents=True)
        fn.save(folder_save, dpi=dpi_fig)

    embed()

if __name__ == "__main__":
    main()
