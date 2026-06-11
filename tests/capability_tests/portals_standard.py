"""
CAPABILITY: Standard PORTALS run (TGLF turbulence + NEO neoclassical)
---------------------------------------------------------------------
This script teaches how to launch a PORTALS flux-matching optimization from a
plain input.gacode file, controlling the transport models from Python.

Key teaching points:
    1. PORTALSmain.portals() loads the *default* namelist from
       templates/namelist.portals.yaml into two plain dictionaries:
       `portals_fun.portals_parameters` and `portals_fun.optimization_options`.
       You do NOT need to write your own YAML — just modify those dictionaries
       in-situ before calling prep(). (prep() snapshots the namelist into the
       run folder; edits after prep() are ignored.)
    2. Code settings follow a three-level hierarchy (see the Transport models
       section below): controls file -> code_settings preset -> extraOptions.
    3. TGLF (turbulence) and NEO (neoclassical) are the default transport
       models; each has its own options block under transport.options.
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# whatever is already in the folder (completed evaluations are detected and skipped)
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Working folder of the run: everything (inputs, per-iteration model runs, logs, results)
# is written under it
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_standard"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# --- Optimization controls -------------------------------------------------------------------------------------------
# The run starts with `initial_training` simple-relaxation (SR) evaluations to seed the
# surrogates, then performs up to `maximum_iterations` Bayesian-optimization iterations.
portals_fun.optimization_options["initialization_options"]["initial_training"] = 5
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2

# The run can also stop earlier on residual reduction, e.g. requiring a 100x improvement:
#   portals_fun.optimization_options["convergence_options"]["stopping_criteria_parameters"]["maximum_value"] = 1e-2
#   portals_fun.optimization_options["convergence_options"]["stopping_criteria_parameters"]["maximum_value_is_rel"] = True

# --- Solution: what to predict ---------------------------------------------------------------------------------------
portals_fun.portals_parameters["solution"]["predicted_rho"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]

# --- Transport models ------------------------------------------------------------------------------------------------
# Turbulence and neoclassical backends are selected in
# transport.evaluator_instance_attributes (defaults: tglf + neo).
#
# The input file each code receives is built in three levels, each overriding the
# previous one:
#   1. Controls file (templates/input.<code>.controls): the full set of default
#      control parameters for the code.
#   2. Models file (templates/input.<code>.models.yaml): the preset named by
#      `code_settings` (e.g. TGLF saturation rules "SAT0"..."SAT3", NEO "Sonic")
#      overrides the specific controls that define that model.
#   3. `extraOptions`: a dictionary of individual input parameters, applied last —
#      the final word on any control, regardless of what the preset says.

# TGLF: choose the saturation rule (level 2) and force electrostatic fluctuations (level 3)
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT2"
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["extraOptions"] = {
    "USE_BPER": False,
    "USE_BPAR": False,
}

# NEO: choose the preset (level 2; "Sonic" -> ROTATION_MODEL=2 on top of templates/input.neo.controls)
# and increase the pitch-angle resolution on top of it (level 3)
portals_fun.portals_parameters["transport"]["options"]["neo"]["run"]["code_settings"] = "Sonic"
portals_fun.portals_parameters["transport"]["options"]["neo"]["run"]["extraOptions"] = {"N_XI": 25}

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state and the run
# ---------------------------------------------------------------------------------------------------------------------

# Load the input.gacode into a plasma-state object and apply standard corrections
# (recompute total pressure, make fast species thermal, enforce quasineutrality)
plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

# prep() defines the optimization problem (DVs = gradients at predicted_rho, OFs = flux
# residuals) and snapshots the namelist into the folder — edits after this point are ignored
portals_fun.prep(plasma_state)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

# MITIM_BO is the generic optimization driver; askQuestions=False avoids interactive prompts
mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot results (flux-matching evolution, surrogate behavior, profiles)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (portals_fun.fn); show() opens the GUI
portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()

# A run (finished or still going) can also be plotted from the terminal at any time with:
#   mitim_plot_portals <run-folder>
