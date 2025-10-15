"""
This example reads TGLF from an already existing folder (no normalizations if no input_gacode file provided)

	read_tglf.py run0/ [--suffixes _0.55] [--gacode input.gacode]

"""

import argparse
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--suffixes", required=False, type=str, nargs="*", default=None)
    parser.add_argument("--gacode", required=False, type=str, default=None)

    args = parser.parse_args()

    folders = [IOtools.expandPath(folder) for folder in args.folders]
    input_gacode = IOtools.expandPath(args.gacode) if args.gacode is not None else None
    suffixes = args.suffixes

    if suffixes is None:
        suffixes = ["" for _ in range(len(folders))]

    for i in range(len(suffixes)):
        if suffixes[i] == "_":
            suffixes[i] = ""

    tglf = TGLFtools.TGLF()
    tglf.prep_from_file(
        folders[0], folders[0] / f"input.tglf{suffixes[0]}", input_gacode=input_gacode
    )
    for i, folder in enumerate(folders):
        tglf.read(folder=folder, suffix=suffixes[i], label=f"run{i}")

    tglf.plot(labels=[f"run{i}" for i in range(len(folders))])

    tglf.fn.show()
    embed()

if __name__ == "__main__":
    main()
