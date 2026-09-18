"""
CAPABILITY: Multi-fidelity PORTALS run (TGLF as low fidelity, nonlinear CGYRO as high fidelity)
-----------------------------------------------------------------------------------------------
This script teaches how to declare more than one turbulence model in a single PORTALS
run, so that the optimizer can evaluate cheap (TGLF) and expensive (nonlinear CGYRO)
transport models within the same flux-matching loop. CGYRO runs at the resolution of
the "Nonlinear_silly" preset with a tiny MAX_TIME: this is a WORKFLOW test, the fluxes
are physically meaningless.

Key teaching points:
    1. Multi-fidelity is declared by turning `turbulence_model` into an int-keyed dict
       (`{0: 'tglf', 1: 'cgyro'}`) instead of a string. Each value names a block in
       transport.options, whose `code:` field selects the backend. PORTALS then appends
       a RESERVED design variable, `fidelity_level`, bounded [0, N-1] and rounded to
       an integer at dispatch (0 -> TGLF, 1 -> CGYRO here).
    2. Not every optimizer stage handles that extra DV natively:
         - The simple-relaxation (SR) INITIALIZATION always evaluates at fidelity 0
           (cheapest): the first `initial_training` points are TGLF.
         - The `sr` and `root` ACQUISITION stages pin fidelity_level to N-1 (highest):
           every BO iteration here is therefore a CGYRO evaluation.
         - `botorch` and `ga` treat fidelity_level as a free DV.
       With `optimizers: ["sr"]` the run is deterministic: TGLF seeds, CGYRO refines.
    3. Each backend keeps its own options block: TGLF (three-level settings hierarchy)
       and CGYRO (four levels: controls -> code_settings -> extraOptions ->
       preprocess_options, see cgyro_02_nonlinear_run_from_inputgacode.py). CGYRO runs
       on the machine configured for "cgyro" in config_user.json; TGLF on its own.
    4. `run_base_tglf: True` (template default) runs TGLF alongside every CGYRO
       evaluation, so both models are available for comparison at every iteration.

CAVEATS (state of the code, not of this script):
    - The surrogate does NOT receive fidelity_level as an input: TGLF and CGYRO points
      are fitted by the same Gaussian process. Use this capability to exercise the
      workflow, and treat cross-fidelity surrogate behavior as work in progress.
    - The per-iteration flux cache (fluxes_turb.json / fluxes_neoc.json) is keyed by
      folder, not by fidelity: re-running with cold_start=False reuses whatever
      fidelity produced those files.
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
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_multifidelity"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# --- Optimization controls -------------------------------------------------------------------------------------------
# `initial_training` SR evaluations seed the surrogates (at fidelity 0 = TGLF), then up to
# `maximum_iterations` BO iterations follow (with the `sr` optimizer, at fidelity 1 = CGYRO).
#
# *** WARNING ***: these caps are set ONLY so that this teaching script finishes quickly.
# The resulting profiles must NOT be trusted.
portals_fun.optimization_options["initialization_options"]["initial_training"] = 3
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2
portals_fun.optimization_options["acquisition_options"]["optimizers"] = ["sr"]

# --- Solution: what to predict ---------------------------------------------------------------------------------------
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]
portals_fun.portals_parameters["solution"]["predicted_roa"] = [0.5, 0.7]

# --- Transport models: the multi-fidelity declaration ----------------------------------------------------------------
# A dict instead of a string. Keys are the fidelity levels (0 = lowest), values are the names
# of the blocks in transport.options ("tglf" and "cgyro" exist in the template; named
# instances such as "tglf_sat2" would need their own block with a `code:` field).
transport = portals_fun.portals_parameters["transport"]
transport["evaluator_instance_attributes"]["turbulence_model"] = {0: "tglf", 1: "cgyro"}
transport["evaluator_instance_attributes"]["neoclassical_model"] = "neo"

# Low fidelity: TGLF, electrostatic SAT2 (three-level hierarchy, see portals_01_tglf_standard.py)
transport["options"]["tglf"]["run"]["code_settings"] = "SAT2"
transport["options"]["tglf"]["run"]["extraOptions"] = {"USE_BPER": False, "USE_BPAR": False}

# High fidelity: nonlinear CGYRO at the cheapest rung of the preset ladder
cgyro_options = transport["options"]["cgyro"]
cgyro_options["run"]["code_settings"] = "Nonlinear_silly"
# extraOptions is level 3 of the CGYRO hierarchy: MAX_TIME here overrides the preset's
# inherited value. 5 a/cs is nowhere near saturated turbulence — demonstration only.
cgyro_options["run"]["extraOptions"] = {"MAX_TIME": 5.0}
# The template default is "prep" (write inputs, never submit); a real loop needs "normal"
# (send + submit + wait) or "submit" (detached, PORTALS polls every `every_n_minutes`)
cgyro_options["run"]["run_type"] = "normal"
# Resources of each CGYRO instance (one per radius): on a GPU machine (gpus_per_node > 0 in
# config_user.json) resources_per_call is GPUs per radius; on a CPU machine it is cores
cgyro_options["run"]["allocation"] = {"resources_per_call": 8, "minutes": 10}
# Keep every CGYRO output file: the template default "pickle" deletes files that restart
# chains (restart_from_cases) rely on, and here we want to inspect the run trees afterwards
cgyro_options["keep_files"] = "all"

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state and the run
# ---------------------------------------------------------------------------------------------------------------------

# Load the input.gacode into a plasma-state object and apply standard corrections
# (recompute total pressure, make fast species thermal, enforce quasineutrality)
plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

# prep() defines the optimization problem: DVs = gradients at predicted_roa PLUS the
# fidelity_level variable, OFs = flux residuals. The namelist is snapshotted here.
portals_fun.prep(plasma_state)

# The reserved DV is visible in the problem definition
dvs = list(portals_fun.optimization_options["problem_options"]["dvs"])
print(f"Design variables ({len(dvs)}): {dvs}")
assert dvs[-1] == "fidelity_level", "multi-fidelity declaration did not append fidelity_level"

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

# MITIM_BO is the generic optimization driver; askQuestions=False avoids interactive prompts
mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Inspect which fidelity each evaluation used
# ---------------------------------------------------------------------------------------------------------------------

# Outputs/optimization_data.csv carries one row per evaluation, including the raw (continuous)
# fidelity_level value; the dispatched integer is its rounding. Expected pattern here:
# initial_training rows at 0 (TGLF), BO-iteration rows at 1 (CGYRO).
import pandas as pd
data = pd.read_csv(folderWork / "Outputs" / "optimization_data.csv")
print(data[["fidelity_level"]].round().astype(int).T.to_string())

# The run trees tell the same story:
#   Initialization/initialization_simple_relax/portals_sr_ev_<i>/transport_simulation_folder/base_tglf/
#   Execution/Evaluation.<iter>/transport_simulation_folder/base_cgyro/   (+ base_tglf from run_base_tglf)

# ---------------------------------------------------------------------------------------------------------------------
# 5. Plot results (flux-matching evolution, surrogate behavior, profiles)
# ---------------------------------------------------------------------------------------------------------------------

# Analysis tabs show the highest-fidelity model; CGYRO time traces per radius are in the
# --complete notebook (mitim_plot_portals <run-folder> --complete)
portals_fun.plot_optimization_results(analysis_level=2)
portals_fun.fn.show()

# On a MacBook Pro (8 cores, CPU CGYRO) this script takes ~15 minutes, dominated by the two
# CGYRO evaluations; on a GPU machine set resources_per_call to the GPUs per radius (e.g. 1)
