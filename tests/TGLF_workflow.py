import os
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

restart = True

if not os.path.exists(__mitimroot__ + "/tests/scratch/"):
    os.system("mkdir " + __mitimroot__ + "/tests/scratch/")

folder = __mitimroot__ + "/tests/scratch/tglf_test/"
input_tglf = __mitimroot__ + "/tests/data/input.tglf"

if restart and os.path.exists(folder):
    os.system(f"rm -r {folder}")

tglf = TGLFtools.TGLF()
tglf.prep_from_tglf(folder, input_tglf)
tglf.run(
    subFolderTGLF="run1/",
    TGLFsettings=None,
    runWaveForms=[0.67, 1.2],
    restart=restart,
    forceIfRestart=True,
    extraOptions={"USE_BPER": True},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="run1")
tglf.plot(labels=["run1"])

# Required if running in non-interactive mode
tglf.fn.show()