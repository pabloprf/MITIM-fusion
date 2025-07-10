import argparse
from mitim_tools.misc_tools import IOtools
from mitim_tools.eped_tools import EPEDtools
from IPython import embed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")

    args = parser.parse_args()

    folders = [IOtools.expandPath(folder) for folder in args.folders]

    eped = EPEDtools.EPED(folder=None)

    for i, folder in enumerate(folders):
        eped.read(subfolder=folder, label=f"run{i}")

    eped.plot(labels=[f"run{i}" for i in range(len(folders))])

    eped.fn.show()

    embed()

if __name__ == "__main__":
    main()
