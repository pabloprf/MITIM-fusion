import argparse
from mitim_modules.maestro.utils import MAESTROplot
from mitim_tools.misc_tools import GUItools

"""
Quick way to plot several input.gacode files together
e.g.
		read_maestro.py folder [--beats 3]
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument(
        "--beats", type=int, required=False, default=2
    )  # Last beats to plot
    parser.add_argument(
        "--only", type=str, required=False, default=None
    )
    parser.add_argument(
        "--full", required=False, default=False, action="store_true"
    )  

    args = parser.parse_args()

    folders = args.folders
    beats = args.beats
    only = args.only
    full = args.full

    fn = GUItools.FigureNotebook("MAESTRO")

    for folder in folders:
        m = MAESTROplot.plotMAESTRO(folder, fn = fn, num_beats=beats, only_beats = only, full_plot = full)

    fn.show()

    # Import IPython and embed an interactive session
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
