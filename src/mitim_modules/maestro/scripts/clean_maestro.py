import os
import shutil
from pathlib import Path
from IPython import embed
import argparse

def cleanup_directory(root_path, aggressive_clean_flag = False):
    """
    Keep only directories and structure leading to beat_results folders.
    Delete all other files and subdirectories.
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"Path {root_path} does not exist")
        return
    
    # First pass: find all beat_results directories and their parents
    dirs_to_keep = set()  # Directories where all contents should be kept
    dir_structure_to_keep = set()  # Directories where only file structure should be kept
    files_to_keep = set()
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Keep all files directly in Zeff_* folders
        current_path = Path(dirpath)
        if current_path == root:
            for filename in filenames:
                file_path = current_path / filename
                files_to_keep.add(file_path)
                print(f"Keeping: {file_path.relative_to(root)}")
        
        # Keep all files named 'eped.input.1' or 'output_run1.nc'
        for filename in filenames:

            if not aggressive_clean_flag:
                keep_keywords = ['input.gacode', 'namelist', '.CDF', '.yaml', 'eped.input', 'output_run', '.nc', '.npy', 
                                'params.in.', 'results.out.', 'optimization_data.csv', '.json', 'figure', 'eped.config', 'input.separatrix.gacode']

            else:
                keep_keywords = ['input.gacode', 'namelist', '.yaml', 'eped.input', 'output_run', '.nc', '.npy', 
                                'params.in.', 'results.out.', 'optimization_data.csv', '.json', 'figure', 'eped.config', 'input.separatrix.gacode']

            if any(keyword in filename for keyword in keep_keywords):
                file_path = current_path / filename
                files_to_keep.add(file_path)
                print(f"Keeping: {file_path.relative_to(root)}")
                # Keep parent directories up to the root
                for ancestor in current_path.parents:
                    if ancestor == root:
                        break
                    dir_structure_to_keep.add(ancestor)
            
        
        # Keep all folders that include 'run_' in the name, kept even with aggressive clean
        if any('run_' in dirname for dirname in dirnames):
            for dirname in dirnames:
                if 'run_' in dirname:
                    run_folder_path = current_path / dirname
                    dir_structure_to_keep.add(run_folder_path)
                    print(f"Keeping: {run_folder_path.relative_to(root)}")

        # Keep all folders that include 'run_' in the name, kept even with aggressive clean
        if any('beat_results' in dirname for dirname in dirnames):
            for dirname in dirnames:
                if 'beat_results' in dirname:
                    beat_results_path = current_path / dirname
                    dir_structure_to_keep.add(beat_results_path)
                    print(f"Keeping: {beat_results_path.relative_to(root)}")
            
            # Mark parent directories for keeping
            current = Path(dirpath)
            while current != root.parent:
                dir_structure_to_keep.add(current)
                current = current.parent
        
        if 'Outputs' in dirnames:
            outputs_path = Path(dirpath) / 'Outputs'
            print(f"Keeping: {outputs_path.relative_to(root)}")
            # Mark Outputs and all its contents for keeping
            for root_check, dirs_check, files_check in os.walk(outputs_path):
                dirs_to_keep.add(Path(root_check))
            
            # Mark parent directories for keeping
            current = Path(dirpath)
            while current != root.parent:
                dir_structure_to_keep.add(current)
                current = current.parent

    # Second pass: delete everything not in keep list
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        current_path = Path(dirpath)
        
        # Delete directories not in keep list
        for dirname in dirnames:
            subdir_path = current_path / dirname
            # Keep directory if it's in dirs_to_keep or part of dir_structure_to_keep
            if subdir_path not in dirs_to_keep and subdir_path not in dir_structure_to_keep:
                # Check if any files in this directory are on the keep list
                has_kept_files = False
                for file_path in files_to_keep:
                    if file_path.parent == subdir_path or subdir_path in file_path.parents:
                        has_kept_files = True
                        break
                
                if not has_kept_files:
                    try:
                        shutil.rmtree(subdir_path)
                        print(f"Deleted: {subdir_path.relative_to(root)}/")
                    except Exception as e:
                        print(f"Error deleting {subdir_path}: {e}")
                else:
                    print(f"Skipping: {subdir_path.relative_to(root)}/ (contains files to keep)")
        
        # Delete all files in current directory that are not in keep list
        for filename in filenames:
            file_path = current_path / filename
            if file_path not in files_to_keep:
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path.relative_to(root)}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

                    



def main(): 

    parser = argparse.ArgumentParser()
    
    # Specify folder
    parser.add_argument("folders", type=str, nargs="*",
                        help="Paths to the folders to read.")
    parser.add_argument("--aggressive_clean", action="store_true",
                        help="If set, keep more selective set of files, focusing on final results and reproducibility.")
    args = parser.parse_args()
    for folder in args.folders:
        cleanup_directory(folder)

if __name__ == "__main__":
    main()
