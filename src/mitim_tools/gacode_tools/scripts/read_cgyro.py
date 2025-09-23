import argparse
import pickle
from mitim_tools.gacode_tools.utils.CGYROutils import CGYROoutput
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
    parser.add_argument("--suffixes", required=False, type=str, nargs="*", default=None)
    parser.add_argument("--two", action="store_true", help="Include 2D plots")
    parser.add_argument("--linear", action="store_true", help="Just a plot of the linear spectra")
    parser.add_argument("--tmin", type=float, nargs="*", default=None, help="Minimum time to calculate mean and std")
    parser.add_argument("--scan_subfolder_id" , type=str, nargs="*", default="KY", help="If reading a linear scan, the subfolders contain this common identifier")
    parser.add_argument("--noplot", action="store_true", help="If set, it will not plot anything, just read the data.")
    parser.add_argument("--pickle", action="store_true", help="If set, it will save the read data in a pickle file for faster reading next time.")
    parser.add_argument("--minimal", action="store_true")
    
    args = parser.parse_args()

    folders = args.folders
    linear = args.linear
    tmin = args.tmin
    include_2D = args.two
    skip_plotting = args.noplot
    pkl = args.pickle
    minimal = args.minimal

    suffixes = args.suffixes
    
    scan_subfolder_id = args.scan_subfolder_id
    
    if isinstance(scan_subfolder_id, str):
        scan_subfolder_id = [scan_subfolder_id for _ in range(len(folders))]

    if suffixes is None:
        suffixes = ["" for _ in range(len(folders))]

    for i in range(len(suffixes)):
        if suffixes[i] == "_":
            suffixes[i] = ""
    
    if tmin is None:
        tmin = [0.0] * len(folders)
        last_tmin_for_linear = True
    else:
        last_tmin_for_linear = False
    
    # Read
    c = CGYROtools.CGYRO()

    labels = []
    output_pickle = {}
    for i, folder in enumerate(folders):
        labels.append(f"case {i + 1}")
        
        if linear:
            c.read_linear_scan(
                label=labels[-1],
                folder=folder,
                suffix=suffixes[i],
                preffix=scan_subfolder_id[i],
                minimal=minimal
                )   
        elif include_2D:
            c.read(
                label=labels[-1],
                folder=folder,
                tmin=tmin[i],
                last_tmin_for_linear=last_tmin_for_linear,
                suffix=suffixes[i],
                preffix=scan_subfolder_id[i],
                minimal=minimal
            )
        else:
            c.read(
                label=labels[-1],
                folder=folder,
                tmin=tmin[i],
                last_tmin_for_linear=last_tmin_for_linear,
                suffix=suffixes[i],
                preffix=scan_subfolder_id[i],
                minimal=minimal
            )

        if pkl:
            print("Pickling data...")
            print(c.results[labels[-1]]['output'])
            with open(folder + "/data.pkl", "wb") as f:
                pickle.dump(c.results[labels[-1]]['output'], f)
            print("Pickling done.")

    if not skip_plotting:
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
