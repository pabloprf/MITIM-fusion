import argparse
from mitim_tools.misc_tools import FARMINGtools, IOtools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("machine", type=str)
    parser.add_argument("--files", type=str, required=False, default=[], nargs="*")
    parser.add_argument("--folders", type=str, required=False, default=[], nargs="*")

    args = parser.parse_args()

    folder_local = IOtools.expandPath('./')
    machine = args.machine
    files_remote = args.files
    folders_remote = args.folders

    FARMINGtools.retrieve_files_from_remote(folder_local, machine, files_remote = files_remote, folders_remote = folders_remote)

    # Remote files created in this process
    for file in ['mitim_bash.src', 'mitim_shell_executor.sh', 'paramiko.log', 'mitim.out']:
        (folder_local / file).unlink(missing_ok=True)
    