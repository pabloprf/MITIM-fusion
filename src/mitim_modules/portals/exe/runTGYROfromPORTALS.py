import sys, os, argparse, copy, time
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.gacode_tools import TGYROtools, PROFILEStools
from mitim_modules.portals import PORTALSmain

"""
This script will run TGYRO using the settings used for mitim. It will do it in a subfolder of the
mitim run.
e.g.:           runTGYROfrommitim.py --folder run1/ --seeds 5 --methods 1 6

It will run 16 in parallel

param 1: relax_param, 2: step_max, 3: step_jac

"""

# ------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--seeds", type=int, required=False, default=1)
parser.add_argument("--methods", type=int, required=False, default=[6], nargs="*")
parser.add_argument("--params", type=int, required=False, default=[1], nargs="*")
parser.add_argument("--parallel_calls", type=int, required=False, default=16)

args = parser.parse_args()
folderO = IOtools.expandPath(args.folder)
seeds = args.seeds
methods = args.methods
parallel_calls = args.parallel_calls
params = args.params

restart = False

# ------------------------------------------------------------------------------------------
# Preparation
# ------------------------------------------------------------------------------------------

# Folder to work on
folder = IOtools.expandPath(folderO + "/tgyro_std_analysis/")
if not os.path.exists(folder):
    IOtools.askNewFolder(folder)


# # Read mitim results
# opt_fun = STRATEGYtools.FUNmain(folderO)
# opt_fun.read_optimization_results(plotYN=False,analysis_level=1)
# portals = opt_fun.prfs_model.mainFunction

# # Prepare TGYRO
# readFile = f'{folder}/input.gacode_copy_initialization'
# with open(readFile,'w') as f:  f.writelines(portals.file_in_lines_initial_input_gacode)
# profiles = PROFILEStools.PROFILES_GACODE(readFile,calculateDerived=True)

opt_fun, portals, self_complete, MITIMextra_dict = PORTALSmain.readFullmitim(folderO)
profiles = MITIMextra_dict[0]["tgyro"].profiles

tgyro = TGYROtools.TGYRO()
tgyro.prep(folder, profilesclass_custom=profiles, restart=restart, forceIfRestart=True)

# ------------------------------------------------------------------------------------------
# Run TGYRO (function prep)
# ------------------------------------------------------------------------------------------


def run_tgyro_parallel(Params, cont):
    time.sleep(
        cont * 5
    )  # Wait a bit to avoid bandwidth problems of ssh or scp connections

    # Grab data
    restartTGYRO = Params["restart"] if "restart" in Params else restart
    tgyro_here = Params["tgyro"] if "tgyro" in Params else copy.deepcopy(tgyro)
    method = Params["method"]
    iterations = Params["iterations"]
    i = Params["scan"][cont]
    seconds_sleep = Params["seconds_sleep"]

    if param == 1:
        name = f"m{method}_r{i:.3f}"
        TGYRO_solver_options = {"tgyro_method": method, "relax_param": i}
    elif param == 2:
        name = f"m{method}_s{i:.3f}"
        TGYRO_solver_options = {"tgyro_method": method, "step_max": i}
    elif param == 3:
        name = f"m{method}_j{i:.3f}"
        TGYRO_solver_options = {"tgyro_method": method, "step_jac": i}

    tgyro_here.run(
        subFolderTGYRO=name + "/",
        TGYRO_solver_options=TGYRO_solver_options,
        iterations=iterations,
        restart=restartTGYRO,
        forceIfRestart=True,
        special_radii=portals.TGYROparameters["RhoLocations"],
        PredictionSet=[
            int("te" in portals.TGYROparameters["ProfilesPredicted"]),
            int("ti" in portals.TGYROparameters["ProfilesPredicted"]),
            int("ne" in portals.TGYROparameters["ProfilesPredicted"]),
        ],
        minutesJob=120,
        launchSlurm=True,
        TGLFsettings=portals.TGLFparameters["TGLFsettings"],
        extraOptionsTGLF=portals.TGLFparameters["extraOptionsTGLF"],
        TGYRO_physics_options={
            "TargetType": 3,
            "quasineutrality": [1, 2] if profiles.DTplasmaBool else [1],
        },
    )


# ------------------------------------------------------------------------------------------
# Run TGYRO
# ------------------------------------------------------------------------------------------

for method in methods:
    for param in params:
        if method == 6:
            scan = np.linspace(0.01, 0.2, seeds)
            iterations = 100
        elif method == 1:
            if param == 1:
                scan = np.linspace(1, 3, seeds)
            elif param == 2:
                scan = np.linspace(0.05, 0.5, seeds)
            elif param == 3:
                scan = np.linspace(0.02, 0.5, seeds)
            iterations = 50

        Params = {
            "method": method,
            "iterations": iterations,
            "scan": scan,
            "seconds_sleep": 5,
        }

        FARMINGtools.ParallelProcedure(
            run_tgyro_parallel,
            Params,
            parallel=parallel_calls,
            howmany=len(Params["scan"]),
        )

# ------------------------------------------------------------------------------------------
# Read (it will detect in run() that files are already in folders)
# ------------------------------------------------------------------------------------------

folders = []
names = []
nice_names = []
for method in methods:
    for param in params:
        if method == 6:
            scan = np.linspace(0.01, 0.2, seeds)
            iterations = 100
        elif method == 1:
            if param == 1:
                scan = np.linspace(1, 3, seeds)
            elif param == 2:
                scan = np.linspace(0.05, 0.5, seeds)
            elif param == 3:
                scan = np.linspace(0.02, 0.5, seeds)
            iterations = 50

        for cont, i in enumerate(scan):
            if param == 1:
                name = f"m{method}_r{i:.3f}"
                nice_name = f"TGYRO method {method}, $\\eta=${i:.3f}"
            elif param == 2:
                name = f"m{method}_s{i:.3f}"
                nice_name = f"TGYRO method {method}, $b_{{max}}=${i:.3f}"
            elif param == 3:
                name = f"m{method}_j{i:.3f}"
                nice_name = f"TGYRO method {method}, $\\Delta_{{x}}=${i:.3f}"

            run_tgyro_parallel(
                {
                    "method": method,
                    "iterations": iterations,
                    "tgyro": tgyro,
                    "seconds_sleep": 0,
                    "restart": False,
                    "scan": scan,
                },
                cont,
            )

            tgyro.read(label=name)

            names.append(name)
            nice_names.append(nice_name)
            folders.append(tgyro.FolderTGYRO)

# ------------------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------------------

tgyro.plotRun(labels=names, doNotShow=True)

from mitim_modules.portals.exe.comparemitim import compareSolvers

compareSolvers(
    folderO, folders, nice_names, fig=tgyro.fn.add_figure(label="mitim COMPARISON")
)

tgyro.fn.show()
