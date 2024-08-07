import argparse
from IPython import embed
from mitim_tools.gs_tools import GEQtools

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    args = parser.parse_args()

    files = args.files

    gs = []
    for file in files:
        gs.extend(GEQtools.MITIMgeqdsk.timeslices(file))

    axs, fn = GEQtools.compareGeqdsk(gs)
    fn.show()
    embed()

if __name__ == "__main__":
    main()
