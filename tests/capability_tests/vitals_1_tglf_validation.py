"""
CAPABILITY: VITALS — validating TGLF against experimental data
--------------------------------------------------------------
This script teaches the VITALS workflow: given experimental fluxes AND
fluctuation measurements (with error bars), find the TGLF inputs — varied
within their experimental uncertainties — that best reproduce ALL the
measurements simultaneously. It uses the same Bayesian-optimization engine as
PORTALS, but the design variables are input.tglf parameters at one radius and
the objectives are measured quantities.

Key teaching points:
    1. The design variables (`dvs`) are input.tglf parameter names (gradients,
       Zeff, temperature ratio), varied as multipliers of the base value
       within [dvs_min, dvs_max] — i.e. their experimental uncertainty.
    2. The objectives (`ofs`) can mix transport fluxes (Qe, Qi) with
       fluctuation quantities from a synthetic diagnostic: TeFluct (Te
       fluctuation level, %) and neTe (ne-Te cross-phase, degrees). The
       synthetic diagnostic needs the perpendicular resolution of the
       measurement, passed as d_perp_cm at read() time.
    3. The experimental values and error bars are attached to the TGLF object
       through NormalizationSets["EXP"] (exp_<quantity>, exp_<quantity>_error,
       at exp_<quantity>_rho); fluxes default to the power-balance values from
       the plasma state, so here only their error bars are specified.
    4. VITALS is initialized from a saved TGLF object (save_pkl) and runs the
       standard MITIM_BO loop; the analysis level 4 plot shows the agreement
       of every objective within error bars as the optimization progresses.
"""

import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.vitals import VITALSmain
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# whatever is already in the folder (completed evaluations are detected and skipped)
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Working folder of the run
folderWork = __mitimroot__ / "tests" / "scratch" / "capability_vitals"

if cold_start and folderWork.exists():
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Definition of the validation problem
# ---------------------------------------------------------------------------------------------------------------------

# Radius of the measurements
rho = 0.5

# TGLF preset to validate
code_settings = "SAT2"

# input.tglf parameters to vary (as multipliers of their base value, within their
# experimental uncertainty): a/LTe, a/LTi, a/Lne, Zeff and Ti/Te
dvs = ["RLTS_1", "RLTS_2", "RLNS_1", "ZEFF", "TAUS_2"]
dvs_min = [0.7, 0.7, 0.7, 0.7, 0.7]
dvs_max = [1.3, 1.3, 1.3, 1.3, 1.3]

# Quantities to match: power-balance fluxes + fluctuation measurements
ofs = ["Qe", "Qi", "TeFluct", "neTe"]

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the base TGLF case and attach the experimental data
# ---------------------------------------------------------------------------------------------------------------------

tglf = TGLFtools.TGLF(rhos=[rho])
tglf.prep_using_tgyro(folderWork, cold_start=cold_start, inputgacode=inputgacode)
tglf.run(subfolder="run_base/", code_settings=code_settings, cold_start=cold_start)

# d_perp_cm is the perpendicular resolution of the fluctuation measurement, used by the
# synthetic diagnostic to filter the TGLF spectrum like the actual instrument does
tglf.read(label="run_base", d_perp_cm={rho: 0.501})

# Fluctuation measurements: value and absolute error at the measurement radius
tglf.NormalizationSets["EXP"]["exp_TeFluct_rho"] = [rho]
tglf.NormalizationSets["EXP"]["exp_TeFluct"] = [1.12]  # Te fluctuation level (%)
tglf.NormalizationSets["EXP"]["exp_TeFluct_error"] = [0.1]

tglf.NormalizationSets["EXP"]["exp_neTe_rho"] = [rho]
tglf.NormalizationSets["EXP"]["exp_neTe"] = [-130]  # ne-Te cross-phase (degrees)
tglf.NormalizationSets["EXP"]["exp_neTe_error"] = [17]

# Fluxes: the experimental values default to the power-balance ones carried by the plasma
# state, so only the error bars (20% here) need to be defined at the measurement radius
tglf.NormalizationSets["EXP"]["exp_Qe_rho"] = [rho]
Qe_base = tglf.NormalizationSets["EXP"]["exp_Qe"][np.argmin(np.abs(tglf.NormalizationSets["EXP"]["rho"] - rho))]
tglf.NormalizationSets["EXP"]["exp_Qe_error"] = [Qe_base * 0.2]

tglf.NormalizationSets["EXP"]["exp_Qi_rho"] = [rho]
Qi_base = tglf.NormalizationSets["EXP"]["exp_Qi"][np.argmin(np.abs(tglf.NormalizationSets["EXP"]["rho"] - rho))]
tglf.NormalizationSets["EXP"]["exp_Qi_error"] = [Qi_base * 0.2]

# ---------------------------------------------------------------------------------------------------------------------
# 3. Prepare and run VITALS
# ---------------------------------------------------------------------------------------------------------------------

# VITALS starts from the saved TGLF object (with its base run and experimental data)
file = folderWork / "tglf.pkl"
tglf.save_pkl(file)

vitals_fun = VITALSmain.vitals(folderWork)
# Keep the example cheap; remove to run until actual convergence
vitals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2
# TGLF is run with the same preset as the base case during the optimization
vitals_fun.TGLFparameters["code_settings"] = code_settings

vitals_fun.prep(file, rho, ofs, dvs, dvs_min, dvs_max)

# MITIM_BO is the same generic optimization driver used by PORTALS
MITIM_BO = STRATEGYtools.MITIM_BO(vitals_fun, cold_start=cold_start, askQuestions=False)
MITIM_BO.run()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot (analysis level 4 includes the per-objective agreement within error bars)
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (vitals_fun.fn); show() opens the GUI
vitals_fun.plot_optimization_results(analysis_level=4)
vitals_fun.fn.show()

