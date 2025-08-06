import argparse
import matplotlib.pyplot as plt
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools.utils import remote_tools
from IPython import embed


def main():

    parser = argparse.ArgumentParser()
    
    # Standard options
    parser.add_argument("folders", type=str, nargs="*",
                        help="Paths to the folders to read.")

    # PORTALS specific options
    parser.add_argument("--max", type=int, required=False, default=None)  # Define max bounds of fluxes based on this one, like 0, -1 or None(best)
    parser.add_argument("--indeces_extra", type=int, required=False, default=[], nargs="*")
    parser.add_argument("--all", required=False, default=False, action="store_true")  # Plot all fluxes?
    parser.add_argument("--file", type=str, required=False, default=None)  # File to save .eps
    parser.add_argument("--complete", "-c", required=False, default=False, action="store_true")
   
    # Remote options
    parser.add_argument("--remote",type=str, required=False, default=None,
                        help="Remote machine to retrieve the folders from. If not provided, it will read the local folders.")
    parser.add_argument("--remote_folder_parent",type=str, required=False, default=None,
                        help="Parent folder in the remote machine where the folders are located. If not provided, it will use --remote_folders.")
    parser.add_argument("--remote_folders",type=str, nargs="*", required=False, default=None,
                        help="List of folders in the remote machine to retrieve. If not provided, it will use the local folder structures.")
    parser.add_argument("--remote_minimal", required=False, default=False, action="store_true",
                        help="If set, it will only retrieve the folder structure with a few key files.")
    parser.add_argument('--fix', required=False, default=False, action='store_true',
                        help="If set, it will fix the pkl optimization portals in the remote folders.")

    args = parser.parse_args()

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Retrieve from remote
    # --------------------------------------------------------------------------------------------------------------------------------------------

    only_folder_structure_with_files = None
    if args.remote_minimal:
        only_folder_structure_with_files = ["Outputs/optimization_data.csv","Outputs/optimization_extra.pkl","Outputs/optimization_object.pkl","Outputs/optimization_results.out"]
            
    folders = remote_tools.retrieve_remote_folders(args.folders, args.remote, args.remote_folder_parent, args.remote_folders, only_folder_structure_with_files)

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Fix pkl optimization portals in remote
    # --------------------------------------------------------------------------------------------------------------------------------------------

    if args.fix:
        for folder in folders:
            STRATEGYtools.clean_state(folder)

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # PORTALS reading
    # --------------------------------------------------------------------------------------------------------------------------------------------

    portals_total = [PORTALSanalysis.PORTALSanalyzer.from_folder(folderWork) for folderWork in folders]

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Actual PORTALS plotting
    # --------------------------------------------------------------------------------------------------------------------------------------------

    file = args.file
    indexToMaximize = args.max
    indeces_extra = args.indeces_extra
    plotAllFluxes = args.all
    complete = args.complete

    if not complete:
        size = 8
        plt.rc("font", family="serif", serif="Times", size=size)
        plt.rc("xtick.minor", size=size)
    plt.close("all")

    is_any_ini = False
    for i in range(len(folders)):
        is_any_ini = is_any_ini or isinstance(
            portals_total[i], PORTALSanalysis.PORTALSinitializer
        )

    requiresFN = (len(folders) > 1) or complete or is_any_ini

    if requiresFN:
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        fn = FigureNotebook("PORTALS", geometry="1600x1000")
    else:
        fn = None

    for i in range(len(folders)):
        lab = f"{IOtools.reducePathLevel(folders[i])[-1]}"

        portals_total[i].fn = fn

        # Plot metrics
        if (not complete) or isinstance(portals_total[i], PORTALSanalysis.PORTALSinitializer):
            if isinstance(portals_total[i], PORTALSanalysis.PORTALSinitializer):
                fig = None
            elif requiresFN:
                fig = fn.add_figure(label=lab)
            else:
                fig = plt.figure(figsize=(15, 8))

            portals_total[i].plotMetrics(
                fig=fig,
                indexToMaximize=indexToMaximize,
                plotAllFluxes=plotAllFluxes,
                indeces_extra=indeces_extra,
                file_save=file if len(folders) == 1 else None,
                extra_lab=lab,
            )

        # Plot PORTALS
        else:
            portals_total[i].plotPORTALS()

    if fn is not None:
        fn.show()
    else:
        plt.show()
    embed()


if __name__ == "__main__":
    main()
