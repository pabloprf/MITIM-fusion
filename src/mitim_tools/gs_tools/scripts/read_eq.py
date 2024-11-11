import argparse
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.gs_tools.utils import GEQplotting


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    args = parser.parse_args()

    files = [IOtools.expandPath(file) for file in args.files]

    gs = []
    for file in files:
        gs.extend(GEQtools.MITIMgeqdsk.timeslices(file))

    axs, fn = GEQplotting.compareGeqdsk(gs)
    fn.show()
    embed()

if __name__ == "__main__":
    main()
