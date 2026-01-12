import argparse
from mitim_tools.misc_tools import IOtools
from mitim_tools.eped_tools import EPEDtools
from IPython import embed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")
    parser.add_argument("--param", type=str, default="neped", help="Parameter that was scanned.")

    args = parser.parse_args()

    folders = [IOtools.expandPath(folder) for folder in args.folders]
    param = args.param

    eped = EPEDtools.EPED(folder=None)

    for i, folder in enumerate(folders):
        eped.read(subfolder=folder, label=f"run{i}")

    eped.plot(labels=[f"run{i}" for i in range(len(folders))], scan_params=[param])

    eped.fn.show()

    embed()

if __name__ == "__main__":
    main()
