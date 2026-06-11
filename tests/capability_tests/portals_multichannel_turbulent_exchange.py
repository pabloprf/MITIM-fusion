"""
CAPABILITY: PORTALS predicting te, ti, ne, nZ and w0 with turbulent exchange
----------------------------------------------------------------------------
This script teaches how to run PORTALS beyond the standard temperature
prediction (see portals_standard.py first): predicting also the electron
density, the density of a trace impurity and the rotation, and treating the
turbulent energy exchange as an additional surrogate.

Key teaching points:
    1. predicted_channels can include, on top of "te"/"ti": "ne" (electron
       density), "nZ" (density of the trace impurity selected with
       `trace_impurity`, driven by its particle flux), and "w0" (toroidal
       rotation, driven by the momentum flux).
    2. turbulent_exchange_as_surrogate=True fits the turbulent energy
       exchange from the transport code as an extra surrogate, so the
       electron-ion exchange in the targets accounts for it self-consistently.
    3. Once the run is over, the trained surrogates can be flux-matched
       directly (flux_match_surrogate), with no new transport-code calls —
       useful to explore e.g. different targets at negligible cost.
    4. Any finished (or running) PORTALS folder can be re-read with
       PORTALSanalyzer.from_folder() and its figures saved to disk.
"""

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSoptimization, PORTALSanalysis
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
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_portals_multichannel"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize the PORTALS object (reads templates/namelist.portals.yaml as defaults)
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# --- Optimization controls (see portals_standard.py) ------------------------------------------------------------------
portals_fun.optimization_options["initialization_options"]["initial_training"] = 5
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2

# --- Solution: what to predict ---------------------------------------------------------------------------------------
portals_fun.portals_parameters["solution"]["predicted_rho"] = [0.25, 0.45, 0.65, 0.85]

# Channels beyond te/ti: electron density, trace-impurity density and toroidal rotation
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti", "ne", "nZ", "w0"]

# The impurity whose density profile "nZ" refers to (matched by name in the input.gacode
# species list); its particle flux becomes the transport channel that drives nZ
portals_fun.portals_parameters["solution"]["trace_impurity"] = "N"

# Fit the turbulent energy exchange as an extra surrogate (see docstring)
portals_fun.portals_parameters["solution"]["turbulent_exchange_as_surrogate"] = True

# --- Transport models (see portals_standard.py for the settings hierarchy) --------------------------------------------
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT0"

# ---------------------------------------------------------------------------------------------------------------------
# 2. Prepare the plasma state and the run
# ---------------------------------------------------------------------------------------------------------------------

# Load the input.gacode and apply standard corrections; enforce_same_aLn forces all ion
# densities to share the electron density gradient, a clean starting point when
# predicting density channels
plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True, "enforce_same_aLn": True})

# prep() defines the optimization problem and snapshots the namelist into the folder —
# edits after this point are ignored
portals_fun.prep(plasma_state)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Run the optimization
# ---------------------------------------------------------------------------------------------------------------------

# MITIM_BO is the generic optimization driver; askQuestions=False avoids interactive prompts
mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot results
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (portals_fun.fn); show() opens the GUI
portals_fun.plot_optimization_results(analysis_level=2)

# ---------------------------------------------------------------------------------------------------------------------
# 5. Flux-match the trained surrogates (no new transport-code calls)
# ---------------------------------------------------------------------------------------------------------------------

# Solve the flux-matching problem on the last-step surrogates, starting from the original
# plasma state, and add the resulting figures to the same notebook
PORTALSoptimization.flux_match_surrogate(
    mitim_bo.steps[-1],
    PROFILEStools.gacode_state(inputgacode),
    fn=portals_fun.fn,
    plot_results=True,
    keep_within_bounds=False,
)

# ---------------------------------------------------------------------------------------------------------------------
# 6. Re-read the finished folder with the analyzer and save its figures
# ---------------------------------------------------------------------------------------------------------------------

# PORTALSanalyzer works on any PORTALS folder (also remote-synced or still running); this
# is what the CLI `mitim_plot_portals <folder>` uses under the hood
portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folderWork)
portals_output.plotPORTALS(noshow=True)
portals_output.fn.save(folderWork / "final_portals_plots")

# Required if running in non-interactive mode
portals_fun.fn.show()
