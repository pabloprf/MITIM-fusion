import os
import json
import socket
import getpass
from pathlib import Path
from mitim_tools.misc_tools import IOtools, LOGtools
from mitim_tools.misc_tools.LOGtools import printMsg
from IPython import embed

# ---------------------------------------------------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------------------------------------------------
'''
Heavily based on results from chatGPT 4o (08/12/2024)
'''

class ConfigManager:
    _instance = None
    _config_file_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def set(self, path: str):
        if self._config_file_path is not None:
            printMsg(f"MITIM > Setting configuration file to: {path}")
        else:
            print(f"MITIM > Setting configuration file to: {path}")
        self._config_file_path = path

    def get(self):
        if self._config_file_path is None:
            if "MITIM_CONFIG" in os.environ:
                self._config_file_path = Path(os.environ["MITIM_CONFIG"]).expanduser()
            
            if self._config_file_path is not None and self._config_file_path.exists():
                printMsg(f"MITIM Configuration file path taken from $MITIM_CONFIG = {self._config_file_path}", typeMsg='i')
            else:
                from mitim_tools import __mitimroot__
                self._config_file_path = __mitimroot__ / "templates" / "config_user.json"
                printMsg(f"MITIM Configuration file path not set, assuming {self._config_file_path}", typeMsg='i')
        return self._config_file_path

config_manager = ConfigManager()

# ---------------------------------------------------------------------------------------------------------------------

def load_settings():

    _config_file_path = config_manager.get()

    # Load JSON
    with open(_config_file_path, "r") as f:
        settings = json.load(f)

    return settings

def read_verbose_level():
    s = load_settings()
    if "verbose_level" in s["preferences"]:
        verbose = int(s["preferences"]["verbose_level"])
    else:
        verbose = 5

    # Ignore warnings automatically if low level of verbose
    if verbose in [1, 2]:
        LOGtools.ignoreWarnings()

    return verbose

def read_dpi():
    s = load_settings()
    if "dpi_notebook" in s["preferences"]:
        dpi = int(s["preferences"]["dpi_notebook"])
    else:
        dpi = 100

    return dpi

def machineSettings(
    code="tgyro",
    nameScratch="mitim_tmp",
    forceUsername=None,
    forceMachine=None,
    append_folder_local=None,
):
    """
    This script uses the config json file and completes the information required to run each code

    forceUsername is used to override the json file (for TRANSP PRF), adding also an identity and scratch
    """

    # Determine where to run this code, depending on config file
    s = load_settings()
    machine = s["preferences"][code] if forceMachine is None else forceMachine

    # Paths in scratch should have a one-to-one (and only one) correspondence with local, to avoid overlapping
    nameScratch_full = IOtools.path_overlapping(nameScratch, append_folder_local) if append_folder_local is not None else nameScratch

    """
    Set-up per code and machine
    -------------------------------------------------
    """

    if forceUsername is not None:
        username = forceUsername
        scratch = f"/home/{username}/scratch/{nameScratch_full}"
    else:
        username = s[machine]["username"] if ("username" in s[machine]) else "dummy"
        scratch = f"{s[machine]['scratch']}/{nameScratch_full}"

    # General limit of 255 characters in path
    scratch = scratch[:255]
    # ----  

    machineSettings = {
        "machine": s[machine]["machine"],
        "user": username,
        "tunnel": None,
        "port": None,
        "identity": None,
        "modules": "", #"source ~/.bashrc",
        "folderWork": scratch,
        "slurm": {},
        "isTunnelSameMachine": (
            bool(s[machine]["isTunnelSameMachine"])
            if "isTunnelSameMachine" in s[machine]
            else False
        ),
    }

    # I can give extra things to load in the config file
    if (
        "modules" in s[machine]
        and s[machine]["modules"] is not None
        and s[machine]["modules"] != ""
    ):
        machineSettings["modules"] = (
            f'{machineSettings["modules"]}\n{s[machine]["modules"]}'
        )

    checkers = ["slurm", "identity", "tunnel", "port"]
    for i in checkers:
        if i in s[machine]:
            machineSettings[i] = s[machine][i]

    if "scratch_tunnel" in s[machine]:
        machineSettings["folderWorkTunnel"] = (
            f"{s[machine]['scratch_tunnel']}/{nameScratch_full}"
        )

    # ************************************************************************************************************************
    # Specific case of being already in the machine where I need to run
    # ************************************************************************************************************************

    # Am I already in this machine?
    if machineSettings["machine"] in socket.gethostname():
        # Avoid tunneling and porting if I'm already there
        machineSettings["tunnel"] = machineSettings["port"] = None

        # Avoid sshing if I'm already there except if I'm running with another specific user
        if (forceUsername is None) or (forceUsername == getpass.getuser()):
            machineSettings["machine"] = "local"

    # ************************************************************************************************************************

    if machineSettings["machine"] == "local":
        machineSettings["folderWork"] = IOtools.expandPath(
            machineSettings["folderWork"]
        )

    if forceUsername is not None:
        machineSettings["identity"] = f"~/.ssh/id_rsa_{forceUsername}"

    return machineSettings
