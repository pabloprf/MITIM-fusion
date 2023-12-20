import sys, os
from mitim_tools.gacode_tools import TGYROtools, PROFILEStools
from mitim_tools.misc_tools import IOtools

"""
Regression test to run and plot TGYRO results from am example input.gacode file
To run: python3  $MITIM_PATH/tests/TGYRO_workflow.py
"""

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

gacode_file = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")
folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/tgyro_test/")

if restart and os.path.exists(folder):
    os.system(f"rm -r {folder}")

profiles = PROFILEStools.PROFILES_GACODE(gacode_file)
tgyro = TGYROtools.TGYRO()
tgyro.prep(folder, profilesclass_custom=profiles, restart=True, forceIfRestart=True)

# ---
rhos = [0.3, 0.5, 0.6, 0.8]
solver = {
    "step_jac": 1e-2,
    "step_max": 1e-2,
    "res_method": 2,
    "tgyro_method": 6,
    "relax_param": 0.1,
}
physics_options = {"TargetType": 2}

tgyro.run(
    subFolderTGYRO="run1/",
    iterations=5,
    restart=True,
    forceIfRestart=True,
    special_radii=rhos,
    PredictionSet=[1, 1, 0],
    TGLFsettings=1,
    TGYRO_solver_options=solver,
    TGYRO_physics_options=physics_options,
)
tgyro.read(label="run1")
tgyro.plotRun(labels=["run1"])
