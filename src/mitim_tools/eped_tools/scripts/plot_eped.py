import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.eped_tools import EPEDtools
from IPython import embed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--param", type=str, default="neped", help="Parameter that was scanned.")
    parser.add_argument("--save", type=str, nargs="?", const="figs", required=False, default=None,
                        help="Folder to save the figures. If flag given without a value, defaults to 'figs'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")

    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    folders = [IOtools.expandPath(folder) for folder in args.folders]
    param = args.param
    folder_save = Path(args.save) if args.save is not None else None
    noshow = args.noshow
    dpi_fig = args.dpi

    eped = EPEDtools.EPED(folder=None)

    for i, folder in enumerate(folders):
        eped.read(subfolder=folder, label=f"run{i}")

    fn = FigureNotebook("EPED", geometry="1600x900", show=not noshow)
    eped.plot(fn=fn, labels=[f"run{i}" for i in range(len(folders))], scan_params=[param])

    if not noshow:
        eped.fn.show()

    if folder_save is not None:
        if not folder_save.exists():
            folder_save.mkdir(parents=True)
        eped.fn.save(folder_save, dpi=dpi_fig)

    embed()

if __name__ == "__main__":
    main()
