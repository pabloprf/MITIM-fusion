import os, shutil
from mitim_tools.misc_tools import IOtools, FARMINGtools, CONFIGread
from IPython import embed

def retrieve_remote_folders(folders_local, remote, remote_folder_parent, remote_folders, only_folder_structure_with_files):

    # Make sure folders_local is a list of complete Paths
    folders_local = [IOtools.expandPath(folder).resolve() for folder in folders_local]

    if remote_folder_parent is not None:
        folders_remote = [remote_folder_parent + '/' + folder.split('/')[-1] for folder in folders_local]
    elif remote_folders is not None:
        folders_remote = remote_folders
    else:
        folders_remote = folders_local

    # Retrieve remote
    s = CONFIGread.load_settings()
    scratch_local_folder = s['local']['scratch']
    
    if remote is not None:
            
        _, folders = FARMINGtools.retrieve_files_from_remote(
            scratch_local_folder,
            remote,
            folders_remote = folders_remote,
            purge_tmp_files = True,
            only_folder_structure_with_files=only_folder_structure_with_files)

        # Renaming
        for i in range(len(folders)):
            folder = IOtools.expandPath(folders[i])
            folder_orig = IOtools.expandPath(folders_local[i])
        
            if folder == folder_orig:
                continue
            
            if folder_orig.exists():
                IOtools.shutil_rmtree(folder_orig)
                
            shutil.copytree(folder, folder_orig)
            IOtools.shutil_rmtree(folder)
            

    return folders_local