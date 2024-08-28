import argparse
from mitim_modules.maestro.utils import MAESTROplot

"""
Quick way to plot several input.gacode files together
e.g.
		read_maestro.py folder [--beats 3]
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
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

    folder = args.folder
    beats = args.beats
    only = args.only
    full = args.full

    m = MAESTROplot.plotMAESTRO(folder, num_beats=beats, only_beats = only, full_plot = full)

    # Import IPython and embed an interactive session
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
