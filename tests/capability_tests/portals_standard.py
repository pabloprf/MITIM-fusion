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
    2. The TGLF saturation rule and any individual TGLF input parameter are
       controlled via transport.options.tglf.run (code_settings / extraOptions).
    3. NEO is the default neoclassical model; its preset is controlled via
       transport.options.neo.run.code_settings.
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
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

# --- Solution: what to predict ---------------------------------------------------------------------------------------
portals_fun.portals_parameters["solution"]["predicted_rho"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]

# --- Transport models ------------------------------------------------------------------------------------------------
# Turbulence and neoclassical backends are selected in
# transport.evaluator_instance_attributes (defaults: tglf + neo), and each backend has
# its own options block. `code_settings` selects a preset from
# templates/input.<code>.models.yaml; `extraOptions` overrides individual input
# parameters of the code on top of that preset.

# TGLF: choose the saturation rule and force electrostatic fluctuations
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT2"
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["extraOptions"] = {
    "USE_BPER": False,
    "USE_BPAR": False,
}

# NEO: choose the preset ("Sonic" -> ROTATION_MODEL=2, see templates/input.neo.models.yaml)
portals_fun.portals_parameters["transport"]["options"]["neo"]["run"]["code_settings"] = "Sonic"

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state and the run
# ---------------------------------------------------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

portals_fun.prep(plasma_state)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot results (flux-matching evolution, surrogate behavior, profiles)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()
