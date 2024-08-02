'''
TO FIX: Right now this needs to run in the "freegs" parent folder
'''

import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.freegsu import FREEGSUmain

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

folderWork = IOtools.expandPath("$MITIM_PATH/tests/scratch/freegsu_test/")

if restart and os.path.exists(folderWork):
    os.system(f"rm -r {folderWork}")

optionsFREEGS = {
    "n": 65,
    "symmetricX": True,
    "plasmaParams": {"BR": 22.49, "Ip": 8.7, "R0": 1.85, "pAxis": 2.6e6},
    "coilsVersion": "V2new",
    "onlyUpperPlate": None,
    "fileWall": "reinke_20211213/v2b_t4slope_v3.csv",
}

Constraints = {
    "xpoints": [[1.54, -1.12]],
    "midplaneR": [1.28, 2.42],
    "xpoints2": [[1.54, 1.12]],
}

CoilCurrents_base = {
    "cs1": -13.498,
    "cs2": -1.190,
    "cs3": 5.574,
    "dv1": 0.0,
    "dv2": 0.0,
    "pf1": 4.266,
    "pf2": 5.012,
    "pf3": -2.928,
    "pf4": -4.517,
}

setCoils = ["pf4", "dv1", "dv2", "pf3", "pf2", "pf1", "cs3", "cs2", "cs1"]
rangeVar = 20

ofs_dict = {
    "xpRlow_1": 154.0,
    "xpZlow_1": -112.0,
    "mpo_1": 242.0,
    "mpi_1": 128.0,
    "xpR_1": 154.0,
    "xpZ_1": 112.0,
    "xpdRsep_1": 2.0,
    "vs_mag_1": 0.0,
}

RequirementsFile = IOtools.expandPath(
    "$SPARC_PATH/FREEGS_SPARC/coils/Limits_20211221.xlsx"
)

if not os.path.exists(RequirementsFile):
    raise Exception(
        "[mitim] The FREEGS_SPARC module is not available. Please ensure it is installed and accessible."
    )

freegsu_opt = FREEGSUmain.freegsu(
    folderWork,
    function_parameters={
        "Constraints": Constraints,
        "optionsFREEGS": optionsFREEGS,
        "CoilCurrents": CoilCurrents_base,
        "CoilCurrents_lower": CoilCurrents_base,
        "params": {"RequirementsFile": RequirementsFile},
    },
)

freegsu_opt.optimization_options["BO_iterations"] = 2

freegsu_opt.prep(ofs_dict, setCoils, rangeVar=rangeVar)

PRF_BO = STRATEGYtools.PRF_BO(
    freegsu_opt, restartYN=restart, askQuestions=False
)

PRF_BO.run()

freegsu_opt.plot_optimization_results(analysis_level=2)

# Required if running in non-interactive mode
freegsu_opt.fn.show()