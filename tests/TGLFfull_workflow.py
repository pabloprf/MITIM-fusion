"""
Regression test to run and plot TGLF results from a TRANSP output file
To run: python3  $MITIM_PATH/tests/TGLF_workflow.py
"""
import numpy as np
import os
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.misc_tools import IOtools

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

cdf_file = IOtools.expandPath("$MITIM_PATH/tests/data/12345.CDF")
folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/tglf_full_test/")

if restart and os.path.exists(folder):
    os.system(f"rm -r {folder}")

tglf = TGLFtools.TGLF(cdf=cdf_file, time=2.5, avTime=0.02, rhos=np.array([0.6, 0.8]))
_ = tglf.prep(folder, restart=restart)

tglf.run(
    subFolderTGLF="runBase/",
    TGLFsettings=5,
    runWaveForms=[0.3],
    restart=restart,
    forceIfRestart=True,
)
tglf.read(label="runBase", d_perp_cm={0.6: 0.5, 0.8: 0.5})

tglf.run(
    subFolderTGLF="runSAT0/",
    TGLFsettings=2,
    runWaveForms=[0.3],
    restart=restart,
    forceIfRestart=True,
)
tglf.read(label="runSAT0", d_perp_cm={0.6: 0.5, 0.8: 0.5})

tglf.NormalizationSets["EXP"]["exp_TeFluct_rho"] = [0.6, 0.8]
tglf.NormalizationSets["EXP"]["exp_TeFluct"] = [1.12, 1.49]
tglf.NormalizationSets["EXP"]["exp_TeFluct_error"] = 0.2
tglf.NormalizationSets["EXP"]["exp_Qe_error"] = 0.005
tglf.NormalizationSets["EXP"]["exp_Qi_error"] = 0.005

tglf.plotRun(labels=["runBase", "runSAT0"])
