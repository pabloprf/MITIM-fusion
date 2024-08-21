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
    args = parser.parse_args()

    folder = args.folder
    beats = args.beats

    m = MAESTROplot.plotMAESTRO(folder, num_beats=beats)

    # Import IPython and embed an interactive session
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
