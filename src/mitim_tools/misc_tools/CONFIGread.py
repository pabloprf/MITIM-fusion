import os
import json
import socket
import warnings
import logging
import getpass
from mitim_tools.misc_tools import IOtools
from IPython import embed

# PRF Note: Do not load IOtools, otherwise circularity problem


def load_settings(filename=None):
    if filename is None:
        filename = os.path.expanduser(
            os.path.expandvars("$MITIM_PATH/config/config_user.json")
        )

    # Load JSON
    with open(filename, "r") as f:
        settings = json.load(f)

    return settings


def read_verbose_level():
    s = load_settings()
    if "verbose_level" in s["preferences"]:
        verbose = int(s["preferences"]["verbose_level"])
    else:
        verbose = 1

    # Ignore warnings automatically if low level of verbose
    if verbose in [1, 2]:
        ignoreWarnings()

    return verbose


def read_dpi():
    s = load_settings()
    if "dpi_notebook" in s["preferences"]:
        dpi = int(s["preferences"]["dpi_notebook"])
    else:
        dpi = 100

    return dpi


def ignoreWarnings(module=None):
    if module is None:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        warnings.filterwarnings("ignore", module=module)  # "matplotlib\..*" )


def isThisEngaging():
    try:
        hostname = os.environ["SLURM_SUBMIT_HOST"][:6]
    except:
        try:
            hostname = os.environ["HOSTNAME"][:6]
        except:
            return False

    bo = hostname in ["eofe7.", "eofe8.", "eofe10"]

    print(f"\t- Is this engaging? {hostname}: {bo}")

    return bo


def machineSettings(
    code="tgyro",
    nameScratch="mitim_tmp/",
    forceUsername=None,
):
    """
    This script uses the config json file and completes the information required to run each code

    forceUsername is used to override the json file (for TRANSP PRF), adding also an identity and scratch
    """

    # Determine where to run this code, depending on config file
    s = load_settings()
    machine = s["preferences"][code]

    """
    Set-up per code and machine
    -------------------------------------------------
    """

    if forceUsername is not None:
        username = forceUsername
        scratch = f"/home/{username}/scratch/{nameScratch}"
    else:
        username = s[machine]["username"] if ("username" in s[machine]) else "dummy"
        scratch = f"{s[machine]['scratch']}/{nameScratch}"

    machineSettings = {
        "machine": s[machine]["machine"],
        "user": username,
        "tunnel": None,
        "port": None,
        "identity": None,
        "modules": "source $MITIM_PATH/config/mitim.bashrc",
        "folderWork": scratch,
        "slurm": {},
        "isTunnelSameMachine": bool(s[machine]["isTunnelSameMachine"])
        if "isTunnelSameMachine" in s[machine]
        else False,
    }

    # I can give extra things to load in the config file
    if (
        "modules" in s[machine]
        and s[machine]["modules"] is not None
        and s[machine]["modules"] != ""
    ):
        machineSettings[
            "modules"
        ] = f'{machineSettings["modules"]}\n{s[machine]["modules"]}'

    checkers = ["slurm","identity", "tunnel", "port"]
    for i in checkers:
        if i in s[machine]:
            machineSettings[i] = s[machine][i]

    if "scratch_tunnel" in s[machine]:
        machineSettings[
            "folderWorkTunnel"
        ] = f"{s[machine]['scratch_tunnel']}/{nameScratch}"

    # ************************************************************************************************************************
    # Specific case of being already in the machine where I need to run
    # ************************************************************************************************************************

    # Am I already in this machine?
    if machine in socket.gethostname():
        # Avoid tunneling and porting if I'm already there
        machineSettings["tunnel"] = machineSettings["port"] = None

        # Avoid sshing if I'm already there except if I'm running with another specific user
        if (forceUsername is None) or (forceUsername == getpass.getuser()):
            machineSettings["machine"] = "local"

    # ************************************************************************************************************************

    if machineSettings['machine'] == 'local':
        machineSettings['folderWork'] = IOtools.expandPath(machineSettings['folderWork'])

    if forceUsername is not None:
        machineSettings["identity"] = f"~/.ssh/id_rsa_{forceUsername}"

    return machineSettings
