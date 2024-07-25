import os
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.vitals import VITALSmain

restart = True

# ********************************************************************************
# Inputs
# ********************************************************************************

inputgacode = IOtools.expandPath("$MITIM_PATH/tests/data/input.gacode")
folderWork = IOtools.expandPath("$MITIM_PATH/tests/scratch/vitals_test/")

if restart and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

rho = 0.5
TGLFsettings = 2

dvs = ["RLTS_1", "RLTS_2", "RLNS_1", "ZEFF", "TAUS_2"]
ofs = ["Qe", "Qi", "TeFluct", "neTe"]
dvs_min = [0.7, 0.7, 0.7, 0.7, 0.7]
dvs_max = [1.3, 1.3, 1.3, 1.3, 1.3]

# ********************************************************************************
# First, run TGLF as you would normally (follow the TGLF regression tests)
# ********************************************************************************

tglf = TGLFtools.TGLF(rhos=[rho])
cdf = tglf.prep(folderWork, restart=restart, inputgacode=inputgacode)
tglf.run(subFolderTGLF="run_base/", TGLFsettings=TGLFsettings, restart=restart)

# ********************************************************************************
# Then, add experimental data of fluctuation information and error bars
# ********************************************************************************

tglf.read(
    label="run_base", d_perp_cm={rho: 0.501}
)  # For synthetic fluctuation diagnostic

tglf.NormalizationSets["EXP"]["exp_TeFluct_rho"] = [rho]
tglf.NormalizationSets["EXP"]["exp_TeFluct"] = [1.12]  # Percent fluctuation
tglf.NormalizationSets["EXP"]["exp_TeFluct_error"] = [0.1]  # Abolute error on it

tglf.NormalizationSets["EXP"]["exp_neTe_rho"] = [rho]
tglf.NormalizationSets["EXP"]["exp_neTe"] = [-130]  # Degrees
tglf.NormalizationSets["EXP"]["exp_neTe_error"] = [17]  # Absolute error

tglf.NormalizationSets["EXP"]["exp_Qe_rho"] = [rho]
Qe_base = tglf.NormalizationSets["EXP"]["exp_Qe"][
    np.argmin(np.abs(tglf.NormalizationSets["EXP"]["rho"] - rho))
]
tglf.NormalizationSets["EXP"]["exp_Qe_error"] = [Qe_base * 0.2]

tglf.NormalizationSets["EXP"]["exp_Qi_rho"] = [rho]
Qi_base = tglf.NormalizationSets["EXP"]["exp_Qi"][
    np.argmin(np.abs(tglf.NormalizationSets["EXP"]["rho"] - rho))
]
tglf.NormalizationSets["EXP"]["exp_Qi_error"] = [Qi_base * 0.2]

# ********************************************************************************
# Prepare VITALS
# ********************************************************************************

file = folderWork + "tglf.pkl"
tglf.save_pkl(file)

vitals_fun = VITALSmain.vitals(folderWork)
vitals_fun.Optim["BO_iterations"] = 2
vitals_fun.TGLFparameters["TGLFsettings"] = TGLFsettings

vitals_fun.prep(file, rho, ofs, dvs, dvs_min, dvs_max)

# ********************************************************************************
# Run VITALS
# ********************************************************************************

PRF_BO = STRATEGYtools.PRF_BO(vitals_fun, restartYN=False, askQuestions=False)
PRF_BO.run()

vitals_fun.plot_optimization_results(analysis_level=4)

# Required if running in non-interactive mode
vitals_fun.fn.show()