from pathlib import Path
import argparse
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.misc_tools import IOtools

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("folder", type=str, help="Simulation folder")
    parser.add_argument("--namelist", type=str, required=False, default=None) # namelist.portals.yaml file, otherwise what's in the current folder
    parser.add_argument("--input", type=str, required=False, default=None) # input.gacode file, otherwise what's in the current folder
    parser.add_argument('--cold', required=False, default=False, action='store_true')

    args = parser.parse_args()
    
    folderWork = Path(args.folder)
    portals_namelist = args.namelist
    inputgacode = args.input
    cold_start = args.cold
    
    if portals_namelist is None:
        portals_namelist = IOtools.expandPath('.') / "namelist.portals.yaml"
    else:
        portals_namelist = Path(portals_namelist)

    if inputgacode is None:
        inputgacode = IOtools.expandPath('.') / "input.gacode"
    else:
        inputgacode = Path(inputgacode)

    portals_fun = PORTALSmain.portals(folderWork, portals_namelist=portals_namelist)
    portals_fun.prep(inputgacode)

    mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
    mitim_bo.run()

if __name__ == "__main__":
    main()
