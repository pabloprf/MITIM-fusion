"""
Read and plot results from an existing profiles_gen -vgen run.

    mitim_plot_vgen <folder> [--gacode input.gacode]

<folder>  : directory containing the vgen run (i.e. the parent of the vgen/ sub-folder)
--gacode  : optional path to the original input.gacode, used to show the w0 profile before VGEN
"""

import argparse
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import NEOtools, PROFILEStools

def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=str, help="Directory containing the VGEN run (parent of vgen/ sub-folder)")
    parser.add_argument("--gacode", required=False, type=str, default=None,
                        help="Original input.gacode (used to display w0 before VGEN)")

    args = parser.parse_args()

    folder     = IOtools.expandPath(args.folder)
    gacode     = IOtools.expandPath(args.gacode) if args.gacode is not None else None

    neo = NEOtools.NEO(rhos=[])

    # Reconstruct the profiles attribute so plot_vgen() can show the before/after comparison
    if gacode is not None:
        neo.profiles = PROFILEStools.gacode_state(gacode, derive_quantities=True)
    else:
        # Fall back to the input.gacode written inside the vgen folder before VGEN was run
        fallback = folder / "input.gacode"
        if fallback.exists():
            neo.profiles = PROFILEStools.gacode_state(fallback, derive_quantities=True)
        else:
            neo.profiles = None

    # folder_vgen must be set so read_vgen() knows where to look
    neo.FolderGACODE = folder.parent
    neo.read_vgen(subfolder=folder.name)

    neo.plot_vgen()
    neo.fn.show()
    embed()

if __name__ == "__main__":
    main()
