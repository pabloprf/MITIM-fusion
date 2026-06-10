from pathlib import Path
import argparse
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.misc_tools import IOtools

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("folder", type=str, help="Simulation folder")
    parser.add_argument("--namelist", type=str, required=False, default=None) # namelist.portals.yaml file, otherwise what's in the current folder
    parser.add_argument("--input", type=str, required=False, default=None) # input.gacode file, otherwise what's in the current folder
    parser.add_argument('--cold', required=False, default=False, action='store_true')
    parser.add_argument('--batch', required=False, default=False, action='store_true', help="If True, do not ask any questions and proceed with defaults.")
    parser.add_argument('--save', required=False, default=False, action='store_true')
    parser.add_argument('--no-log-file', required=False, default=False, action='store_true',
                        help="Skip the Outputs/optimization_log.txt stdout redirection; prints flow straight to the captured stdout (e.g. slurm_output.dat). Useful on clusters with slow IO where the in-folder log buffers and hides live progress.")

    args = parser.parse_args()

    folderWork = Path(args.folder)
    portals_namelist = args.namelist
    inputgacode = args.input
    cold_start = args.cold
    batch = args.batch
    save_figs = args.save
    write_log_file = not args.no_log_file

    portals_namelist = Path(portals_namelist) if  portals_namelist is not None else IOtools.expandPath('.') / "namelist.portals.yaml"
    inputgacode = Path(inputgacode) if  inputgacode is not None else IOtools.expandPath('.') / "input.gacode"

    portals_fun = PORTALSmain.portals(folderWork, portals_namelist=portals_namelist)
    portals_fun.prep(inputgacode, askQuestions=not batch)

    mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=not batch, write_log_file=write_log_file)
    mitim_bo.run()
    
    if save_figs:
        portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folderWork)
        portals_output.plotPORTALS(noshow=True)
        portals_output.fn.save(folderWork / "Analysis" / "portals_plots")

if __name__ == "__main__":
    main()
