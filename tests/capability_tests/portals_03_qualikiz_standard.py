"""
CAPABILITY: PORTALS run with QuaLiKiz turbulence (+ NEO neoclassical)
---------------------------------------------------------------------
This script teaches how to swap the turbulence model of a PORTALS run from the
default TGLF (see portals_01_tglf_standard.py first) to QuaLiKiz, keeping NEO
for the neoclassical side. Only ONE line actually performs the swap (see the
Transport models section); everything else is the standard PORTALS setup.

Requirements beyond portals_01_tglf_standard.py:
    - The external `qualikiz_tools` package (QuaLiKiz-pythontools, a submodule
      of the QuaLiKiz repo) must be importable:
          pip install -e <path>/QuaLiKiz/QuaLiKiz-pythontools
    - A "qualikiz" entry in config_user.json pointing to a machine whose
      `modules` string puts the `QuaLiKiz` executable on PATH.

Key teaching points:
    1. `evaluator_instance_attributes.turbulence_model` selects the turbulence
       backend by naming an `transport.options.<name>` block. The QuaLiKiz
       block is keyed "qlk" (it carries `code: qualikiz`, which is what the
       dispatcher actually switches on), so the swap is:
           turbulence_model = "qlk"
       The neoclassical side is independent and stays "neo".
    2. QuaLiKiz packs ALL radii into a SINGLE execution via its own internal
       "parallel" scan (dimx), unlike TGLF/NEO/CGYRO which produce one folder
       per rho. One PORTALS iteration therefore dispatches one QuaLiKiz job,
       and `allocation.resources_per_call` are the MPI ranks spread over
       dimx*dimn.
    3. PHYSICS CAVEAT — geometry: QuaLiKiz does not support Miller/MXH shaping;
       it uses a circular / s-alpha-like geometry, so the shaped-equilibrium
       information in the input.gacode is DROPPED on the mapping (see the
       module docstring of mitim_tools/qualikiz_tools/QLKtools.py). Fluxes are
       therefore NOT directly comparable to a TGLF run on a strongly shaped
       plasma. This is a property of QuaLiKiz, not of the MITIM interface.
    4. PHYSICS CAVEAT — turbulent exchange: QuaLiKiz does not output the
       turbulent electron-ion energy exchange Qie (it is filled with zeros in
       transport_qualikiz.py). So, unlike portals_02_tglf_multichannel_*.py,
       `turbulent_exchange_as_surrogate` must stay False (the template default)
       and the exchange is left to the analytical target model.
    5. Flux uncertainties: with `use_scan_trick_for_stds: null` (default) the
       std is the ad-hoc `percent_error` (10%). Setting it to a float (e.g.
       0.02) instead estimates the stds from a +/-delta gradient scan — and
       QuaLiKiz stacks EVERY perturbation case onto the same dimx scan, so the
       whole scan is still ONE execution (TGLF needs N_rho x N_var x N_delta).
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
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_qualikiz"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# --- Optimization controls (see portals_01_tglf_standard.py) ---------------------------------------------------------
# *** WARNING ***: as in portals_01_tglf_standard.py, maximum_iterations=2 only keeps this
# teaching script cheap; it is NOT enough to converge a real flux match.
portals_fun.optimization_options["initialization_options"]["initial_training"] = 5
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2

# --- Solution: what to predict ---------------------------------------------------------------------------------------
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.25, 0.45, 0.65, 0.85]

# --- Transport models ------------------------------------------------------------------------------------------------
# THE SWAP: point the turbulence side at the "qlk" options block (code: qualikiz).
# The neoclassical side is untouched and keeps using NEO.
portals_fun.portals_parameters["transport"]["evaluator_instance_attributes"]["turbulence_model"] = "qlk"
portals_fun.portals_parameters["transport"]["evaluator_instance_attributes"]["neoclassical_model"] = "neo"

# QuaLiKiz settings follow the same three-level hierarchy as TGLF (controls file ->
# code_settings preset -> extraOptions). "FAST" (templates/input.qualikiz.models.yaml)
# uses fewer eigenvalue solutions and looser tolerances — cheap, for teaching only.
# Use "STANDARD" (the template default) for physics runs, or "ROTATION" with w0.
portals_fun.portals_parameters["transport"]["options"]["qlk"]["run"]["code_settings"] = "FAST"

# MPI ranks for the single QuaLiKiz execution per iteration, and its SLURM time budget.
# Because all rhos live in one execution, this is the ONLY allocation knob that matters.
portals_fun.portals_parameters["transport"]["options"]["qlk"]["allocation"] = {
    "resources_per_call": 16,
    "minutes": 60,
}

# Flux stds: keep the ad-hoc 10% (default). To derive them from a +/-2% gradient scan
# instead — still a single QuaLiKiz execution — use:
#   portals_fun.portals_parameters["transport"]["options"]["qlk"]["use_scan_trick_for_stds"] = 0.02
portals_fun.portals_parameters["transport"]["options"]["qlk"]["percent_error"] = 10.0

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state and the run
# ---------------------------------------------------------------------------------------------------------------------

# Load the input.gacode into a plasma-state object and apply standard corrections
# (recompute total pressure, make fast species thermal, enforce quasineutrality)
plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

# prep() defines the optimization problem (DVs = gradients at predicted_roa, OFs = flux
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

# All figures go into a multi-tab MITIM FigureNotebook (portals_fun.fn); show() opens the GUI.
# The PORTALS plots are model-agnostic: they show the same flux-matching diagnostics as the
# TGLF runs, since the turbulence backend only changes how the fluxes were produced.
portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()

# A run (finished or still going) can also be plotted from the terminal at any time with:
#   mitim_plot_portals <run-folder>
