import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import TGLFtools

restart = True

if not os.path.exists(IOtools.expandPath("$MITIM_PATH/tests/scratch/")):
    os.system("mkdir " + IOtools.expandPath("$MITIM_PATH/tests/scratch/"))

folder = IOtools.expandPath("$MITIM_PATH/tests/scratch/tglf_test/")
input_tglf = IOtools.expandPath("$MITIM_PATH/tests/data/input.tglf")

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
    slurm_setup={"cores": 4, "minutes": 5},
)

tglf.read(label="run1")
tglf.plot(labels=["run1"])

# Required if running in non-interactive mode
tglf.fn.show()