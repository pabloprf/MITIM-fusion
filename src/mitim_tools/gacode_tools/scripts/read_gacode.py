import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_tools.gacode_tools import PROFILEStools

"""
Quick way to plot several input.gacode files together
e.g.
		read_gacodes.py input.gacode1 input.gacode2 [--rho 0.9]
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    parser.add_argument("--rho", type=float, required=False, default=0.89)  # Last rho for gradients plot
    parser.add_argument("--print", required=False, default=False, action="store_true")  # Last rho for gradients plot
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
        parser.error("--save without a value needs at least one positional input.gacode file")

    files = args.files
    rho = args.rho
    print_only = args.print
    folder_save = IOtools.resolve_save_folder(args.save, Path(files[0]).parent if files else None)
    noshow = args.noshow
    dpi_fig = args.dpi

    # Read
    profs = []
    for file in files:
        p = PROFILEStools.gacode_state(file)
        profs.append(p)

        p.printInfo()

    # Plot

    if not print_only:

        fn = FigureNotebook("Profiles", geometry="1800x900", show=not noshow)
        figs = state_plotting.add_figures(fn)
        state_plotting.plotAll(profs, figs=figs, lastRhoGradients=rho)

        if not noshow:
            fn.show()

        if folder_save is not None:
            if not folder_save.exists():
                folder_save.mkdir(parents=True)
            fn.save(folder_save, dpi=dpi_fig)

    # Import IPython and embed an interactive session
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
