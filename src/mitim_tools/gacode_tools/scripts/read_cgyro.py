import argparse
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.gacode_tools import CGYROtools

"""
e.g.	read_cgyro.py folder
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--two", action="store_true", help="Include 2D plots")
    parser.add_argument("--linear", action="store_true", help="Just a plot of the linear spectra")
    parser.add_argument("--tmin", type=float, nargs="*", default=None, help="Minimum time to calculate mean and std")
    args = parser.parse_args()

    folders = args.folders
    linear = args.linear
    tmin = args.tmin
    include_2D = args.two
    
    if tmin is None:
        tmin = [0.0] * len(folders)
    
    # Read
    c = CGYROtools.CGYRO()

    labels = []
    for i, folder in enumerate(folders):
        labels.append(f"case {i + 1}")
        c.read(label=labels[-1], folder=folder, tmin=tmin[i])

    if linear:
        # Plot linear spectrum
        c.plot_quick_linear(labels=labels)
        plt.show()
    else:
        c.plot(labels=labels, include_2D=include_2D, common_colorbar=True)
        c.fn.show()

    
    embed()

if __name__ == "__main__":
    main()
