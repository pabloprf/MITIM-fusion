"""
CAPABILITY: Pedestal prediction with EPED (including a parameter scan)
----------------------------------------------------------------------
This script teaches how to predict the pedestal height and width with EPED
from a set of scalar engineering and pedestal parameters, scanning one of
them. EPED runs on the machine configured for "eped" in config_user.json.

Key teaching points:
    1. EPED does not start from an input.gacode: its inputs are the scalars in
       `input_params` (machine: ip, bt, r, a, kappa, delta, zeta; pedestal:
       neped, betan, zeffped; separatrix: nesep, tesep).
    2. `scan_param` repeats the run over a list of values of one input (here
       the pedestal density), submitted as a SLURM job array with at most
       `job_array_limit` concurrent cases. `keep_nsep_ratio` ties the
       separatrix density to the scanned pedestal density (nesep = ratio *
       neped) so the scan stays physically consistent.
    3. `eped_params_override` modifies parameters of the EPED configuration
       file itself (the EPED analog of extraOptions), e.g. the bounds and
       resolution of the pedestal-temperature search.
    4. read() parses the scan results and plot() shows pedestal height/width
       vs the scanned parameter(s).
"""

from mitim_tools.eped_tools import EPEDtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one subfolder per case lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_eped"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize EPED and run a pedestal-density scan
# ---------------------------------------------------------------------------------------------------------------------

eped = EPEDtools.EPED(folder=folder)

eped.run(
    # Name of the subfolder (inside the working folder) where this case lives
    subfolder="case1",
    # Base scalar inputs of EPED (SPARC-like values here)
    input_params={
        "ip": 8.7,        # plasma current (MA)
        "bt": 12.16,      # toroidal field (T)
        "r": 1.85,        # major radius (m)
        "a": 0.57,        # minor radius (m)
        "kappa": 1.9,     # elongation
        "delta": 0.5,     # triangularity
        "zeta": 0.01,     # squareness (if your EPED implementation supports it)
        "neped": 30.0,    # pedestal density (10^19 m^-3)
        "betan": 1.0,     # normalized beta
        "zeffped": 1.5,   # pedestal Zeff
        "nesep": 10.0,    # separatrix density (10^19 m^-3); superseded by keep_nsep_ratio below
        "tesep": 100.0,   # separatrix temperature (eV)
    },
    # Scan one of the inputs over these values (each value is one EPED case)
    scan_param={"variable": "neped", "values": [15.0, 30.0, 45.0, 60.0, 75.0]},
    # Tie the separatrix density to the scanned pedestal density: nesep = 0.4 * neped
    keep_nsep_ratio=0.4,
    # Cores for each EPED case
    nproc_per_run=64,
    # Override parameters of the EPED configuration file itself, e.g. the
    # [min, max, step] bounds of the pedestal-temperature search
    eped_params_override={"TEPED_BOUND": [0.3, 1.4, 0.01]},
    cold_start=cold_start,
    # At most this many cases of the scan run concurrently (SLURM job array)
    job_array_limit=5,
    # EPED scratch trees are enormous: only set to False for debugging
    removeScratchFolders=True,
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Read and plot pedestal height/width vs the scanned parameter
# ---------------------------------------------------------------------------------------------------------------------

# read() parses the EPED output files of every case of the scan
eped.read(subfolder="case1")

# All figures go into a multi-tab MITIM FigureNotebook (eped.fn); show() opens the GUI
eped.plot(
    labels=["case1"],
    scan_params=["neped"],
    scan_params_labels=["$n_{e,ped}\\ (10^{19}m^{-3})$"],
)
eped.fn.show()
