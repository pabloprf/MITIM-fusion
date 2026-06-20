"""
CAPABILITY: PORTALS with per-iteration neoclassical E×B shear (NEO/VGEN)
-----------------------------------------------------------------------
This script teaches how to let PORTALS compute the neoclassical (+diamagnetic)
E×B rotation at EVERY transport evaluation and feed the resulting E×B shearing
rate to the turbulence model, and how to read the new rotation diagnostics.

The capability lives in the NEO options block (see the heavily-commented
templates/namelist.portals.yaml, transport.options.neo):

    transport.options.neo.vgen_exb_shear: {er: 2, vel: 1}

With this on, before each transport dispatch PORTALS runs NEO VGEN
(profiles_gen -vgen, weak-rotation limit) to solve the neoclassical radial
electric field Er from the ion pressure gradient + neoclassical poloidal flow
(Waltz-Miller, zero toroidal rotation). VGEN writes the implied w0(rad/s) back
into the state, and TGLF then sees a non-zero VEXB_SHEAR (built from -dw0/dr).
Because the kinetic profiles evolve along the BO loop, the diamagnetic drive
(-dp_i/dr) changes, so w0 and the E×B shear are recomputed every iteration.

Key teaching points:
    1. vgen_exb_shear is the ONLY switch you need; er=2/vel=1 are the NEO
       weak-rotation-limit methods (recommended when the toroidal rotation is
       zero or negligible). Set it to true for VGEN defaults, or null/false
       to disable (the standard PORTALS behavior).
    2. w0 is NOT a predicted channel here: vgen_exb_shear recomputes w0 every
       evaluation, so predicting "w0" at the same time is forbidden (PORTALS
       raises in prep()). Use one or the other.
    3. We ZERO the input rotation, so the rotation you see at the end comes
       ENTIRELY from the neoclassical VGEN Er (not from any rotation already in
       the input.gacode) -- a clean demonstration of the capability.
    4. The end-of-run notebook gains a "PORTALS Rotation" tab (added
       automatically whenever rotation is relevant): per-iteration evolution of
       the E×B rotation w0, the E×B shearing rate (VEXB_SHEAR), the ion pressure
       gradient -dp_i/dr (the diamagnetic drive), the ion pressure, and the
       per-radius VEXB_SHEAR vs evaluation. Profiles are shown over the predicted
       core (markers at the predicted radii).

*** WARNING ***: initial_training / maximum_iterations are cut to the bone so
this teaching script finishes quickly -- far too few for a converged flux match.
Do NOT trust the resulting profiles as physics.

*** REQUIREMENTS ***: subprocess TGLF / NEO / profiles_gen(-vgen) configured in
config_user.json (same dependencies as portals_01_standard.py, plus VGEN).

    python tests/capability_tests/portals_03_neoclassical_exb_shear.py
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch; False reuses completed evaluations in the folder
cold_start = False

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_exb_shear"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# --- Optimization controls (kept tiny for a quick teaching run; see the WARNING in the header) -----------------------
portals_fun.optimization_options["initialization_options"]["initial_training"] = 5
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2

# --- Solution: what to predict ---------------------------------------------------------------------------------------
# Predict Te, Ti only. Do NOT add "w0": it is mutually exclusive with vgen_exb_shear (PORTALS would
# overwrite the predicted rotation with the VGEN one every evaluation, and prep() raises if both are set).
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]
# Push the outer predicted radius to r/a=0.9, where the neoclassical Er (∝ dp_i/dr) is strongest
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.4, 0.65, 0.9]

# --- The capability under test: neoclassical E×B shear from NEO VGEN ------------------------------------------------
# er=2 -> NEO weak-rotation-limit Er (recommended for zero toroidal rotation); vel=1 -> weak-rotation velocities.
# (Set to True to use VGEN defaults; null/False disables it and PORTALS runs as standard.)
portals_fun.portals_parameters["transport"]["options"]["neo"]["vgen_exb_shear"] = {"er": 2, "vel": 1}

# Run TGLF/NEO/VGEN as subprocesses (run trees stay on disk under Execution/ for inspection)
portals_fun.portals_parameters["transport"]["in_process"] = False

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state (zero the rotation so the displayed w0 is entirely the VGEN neoclassical Er)
# ---------------------------------------------------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

# Zero the toroidal rotation -> clean baseline: any w0 in the results is the neoclassical VGEN rotation
plasma_state.profiles["w0(rad/s)"] = plasma_state.profiles["w0(rad/s)"] * 0.0
plasma_state.derive_quantities(rederiveGeometry=False)

# prep() snapshots the namelist into the folder -- edits after this point are ignored
portals_fun.prep(plasma_state)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot results -- the multi-tab notebook now includes the "PORTALS Rotation" tab (w0, E×B shear,
#    diamagnetic drive -dp_i/dr, and per-radius VEXB_SHEAR vs evaluation), added because rotation is relevant.
# ---------------------------------------------------------------------------------------------------------------------

portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()

# The same notebook (including the Rotation tab) can be reopened from the terminal at any time with:
#   mitim_plot_portals <run-folder> --complete
