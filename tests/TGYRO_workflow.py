import os
from mitim_tools.gacode_tools import TGYROtools, PROFILEStools
from mitim_tools import __mitimroot__

"""
Regression test to run and plot TGYRO results from am example input.gacode file
To run: python3  tests/TGYRO_workflow.py
"""

cold_start = True

rundir = __mitimroot__ / "tests" / "scratch/"
rundir.mkdir(parents=True, exist_ok=True)

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "tgyro_test"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

profiles = PROFILEStools.PROFILES_GACODE(gacode_file)
tgyro = TGYROtools.TGYRO()
tgyro.prep(folder, profilesclass_custom=profiles, cold_start=True, forceIfcold_start=True)

# ---
rhos = [0.3, 0.5, 0.6, 0.8]
solver = {
    "step_jac": 1e-2,
    "step_max": 1e-2,
    "res_method": 2,
    "tgyro_method": 6,
    "relax_param": 0.1,
}
physics_options = {"TypeTarget": 2}

tgyro.run(
    subFolderTGYRO="run1",
    iterations=3,
    cold_start=True,
    forceIfcold_start=True,
    special_radii=rhos,
    PredictionSet=[1, 1, 0],
    TGLFsettings=1,
    extraOptionsTGLF={"USE_BPER": True},
    TGYRO_solver_options=solver,
    TGYRO_physics_options=physics_options,
)
tgyro.read(label="run1")
tgyro.plot(labels=["run1"])

# Required if running in non-interactive mode
tgyro.fn.show()
