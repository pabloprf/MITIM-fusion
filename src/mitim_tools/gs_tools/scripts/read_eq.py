import argparse
from pathlib import Path
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gs_tools import GEQtools
from mitim_tools.gs_tools.utils import GEQplotting


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    parser.add_argument("--save", type=str, nargs="?", const=IOtools.SAVE_FOLDER_AUTO_SENTINEL, required=False, default=None,
                        help=f"Folder to save the figures. If flag given without a value, defaults to '<dir of first file>/{IOtools.SAVE_FOLDER_DEFAULT_SUBDIR}'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")
    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    if args.save == IOtools.SAVE_FOLDER_AUTO_SENTINEL and not args.files:
        parser.error("--save without a value needs at least one positional GEQDSK file")

    files = [IOtools.expandPath(file) for file in args.files]
    folder_save = IOtools.resolve_save_folder(args.save, Path(files[0]).parent if files else None)
    noshow = args.noshow
    dpi_fig = args.dpi

    gs = []
    for file in files:
        gs.extend(GEQtools.MITIMgeqdsk.timeslices(file))

    fn = FigureNotebook("GEQDSK Notebook", geometry="1600x1000", show=not noshow)
    GEQplotting.compareGeqdsk(gs, fn=fn)

    if not noshow:
        fn.show()

    if folder_save is not None:
        if not folder_save.exists():
            folder_save.mkdir(parents=True)
        fn.save(folder_save, dpi=dpi_fig)

    embed()

if __name__ == "__main__":
    main()
