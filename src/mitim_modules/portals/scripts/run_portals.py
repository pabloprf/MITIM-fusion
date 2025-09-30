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
    
    # Actual PORTALS run 
    
    portals_namelist = Path(portals_namelist) if  portals_namelist is not None else IOtools.expandPath('.') / "namelist.portals.yaml"
    inputgacode = Path(inputgacode) if  inputgacode is not None else IOtools.expandPath('.') / "input.gacode"

    portals_fun = PORTALSmain.portals(folderWork, portals_namelist=portals_namelist)
    portals_fun.prep(inputgacode)

    mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start)
    mitim_bo.run()

if __name__ == "__main__":
    main()
