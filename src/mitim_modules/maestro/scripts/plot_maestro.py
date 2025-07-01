import argparse
from mitim_modules.maestro.utils import MAESTROplot
from mitim_tools.misc_tools import IOtools, GUItools, FARMINGtools
from mitim_tools.opt_tools import STRATEGYtools
from pathlib import Path
from IPython import embed

"""
Quick way to plot several input.gacode files together (assumes unix in remote)
e.g.
		read_maestro.py folder [--beats 3] [--remote mfews15]

Notes:
    - remote option will copy it locally in the current directory
"""

def fix_maestro(folders):
    for folder in folders:
        folderB = folder / 'Beats'
        for beats_folder in folderB.iterdir():
            subdirs = [subdir for subdir in beats_folder.iterdir() if subdir.is_dir()]
            if 'run_portals' in [subdir.name for subdir in subdirs]:
                folder_portals = [subdir for subdir in subdirs if subdir.name == 'run_portals'][0]
                STRATEGYtools.clean_state(folder_portals) 

                results = [subdir for subdir in subdirs if subdir.name == 'beat_results']
                if len(results) > 0:
                    folder_output = results[0]
                    try:
                        STRATEGYtools.clean_state(folder_output) 
                    except:
                        pass

def main():

    parser = argparse.ArgumentParser()
    
    # Standard options
    parser.add_argument("folders", type=str, nargs="*",
                        help="Paths to the folders to read.")
    parser.add_argument("--beats", type=int, required=False, default=2,
                        help="Number of beats to plot. If 0, it will not plot beat information.")
    parser.add_argument("--only", type=str, required=False, default=None,
                        help="If provided, it will only plot the specified beats (e.g., transp)")
    parser.add_argument("--full", required=False, default=False, action="store_true",
                        help="If set, it will plot the full beat information.")
   
    # Remote options
    parser.add_argument("--remote",type=str, required=False, default=None,
                        help="Remote machine to retrieve the folders from. If not provided, it will read the local folders.")
    parser.add_argument("--remote_folder_parent",type=str, required=False, default=None,
                        help="Parent folder in the remote machine where the folders are located. If not provided, it will use --remote_folders.")
    parser.add_argument("--remote_folders",type=str, nargs="*", required=False, default=None,
                        help="List of folders in the remote machine to retrieve. If not provided, it will use the local folder structures.")
    parser.add_argument("--remote_minimal", required=False, default=False, action="store_true",
                        help="If set, it will only retrieve the folder structure with a few files (input.gacode, input.gacode_final, initializer_geqdsk/input.gacode).")
    parser.add_argument('--fix', required=False, default=False, action='store_true',
                        help="If set, it will fix the pkl optimization portals in the remote folders.")

    args = parser.parse_args()

    remote = args.remote
    folders = args.folders
    fix = args.fix
    beats = args.beats
    only = args.only
    full = args.full

    if args.remote_folder_parent is not None:
        folders_remote = [args.remote_folder_parent + '/' + folder.split('/')[-1] for folder in folders]
    elif args.remote_folders is not None:
        folders_remote = args.remote_folders
    else:
        folders_remote = folders
            

    # Retrieve remote
    if remote is not None:

        if args.remote_minimal:
            only_folder_structure_with_files = ["beat_results/input.gacode", "input.gacode_final","initializer_geqdsk/input.gacode"]
            
            beats = 0
            
        _, folders = FARMINGtools.retrieve_files_from_remote(
            IOtools.expandPath('./'),
            remote,
            folders_remote = folders_remote,
            purge_tmp_files = True,
            only_folder_structure_with_files=only_folder_structure_with_files)
    
    # Fix pkl optimization portals in remote
    if fix:
        fix_maestro([Path(folder) for folder in folders])

    # -----

    folders = [IOtools.expandPath(folder) for folder in folders]
    
    fn = GUItools.FigureNotebook("MAESTRO")

    ms = []
    for folder in folders:
        m = MAESTROplot.plotMAESTRO(folder, fn = fn, num_beats=beats, only_beats = only, full_plot = full)
        ms.append(m)

    fn.show()

    # Import IPython and embed an interactive session
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
