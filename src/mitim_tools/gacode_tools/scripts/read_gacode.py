import argparse
from mitim_tools.gacode_tools import PROFILEStools

"""
Quick way to plot several input.gacode files together
e.g.
		read_gacodes.py input.gacode1 input.gacode2 [--rho 0.9]
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    parser.add_argument(
        "--rho", type=float, required=False, default=0.89
    )  # Last rho for gradients plot
    args = parser.parse_args()

    files = args.files
    rho = args.rho

    # Read
    profs = []
    for file in files:
        p = PROFILEStools.PROFILES_GACODE(file)
        profs.append(p)

        p.printInfo()

    # Plot

    fn = PROFILEStools.plotAll(profs, lastRhoGradients=rho)

    fn.show()

    # Import IPython and embed an interactive session
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
