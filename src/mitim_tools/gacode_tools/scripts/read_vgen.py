"""
Read and plot results from an existing profiles_gen -vgen run.

    mitim_plot_vgen <folder>

<folder>  : directory containing the vgen run (i.e. the parent of the vgen/ sub-folder).
            When smooth_profiles=True was used, input.gacode.raw and input.gacode are both
            read automatically so the raw vs. smoothed comparison tab is populated.
"""

import argparse
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import NEOtools

def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=str, help="Directory containing the VGEN run (parent of vgen/ sub-folder)")

    args = parser.parse_args()

    folder = IOtools.expandPath(args.folder)

    neo = NEOtools.NEO(rhos=[])
    neo.FolderGACODE = folder.parent
    neo.read_vgen(subfolder=folder.name)

    neo.plot_vgen()
    neo.fn.show()
    embed()

if __name__ == "__main__":
    main()
