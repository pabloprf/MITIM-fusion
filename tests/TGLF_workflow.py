import os
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from pathlib import Path
from IPython import embed

restart = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglf_test"
input_tglf = __mitimroot__ / "tests" / "data" / "input.tglf"

if restart and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

tglf = TGLFtools.TGLF()
tglf.prep_from_tglf(folder, input_tglf)

tglf.run(
    subFolderTGLF="run1/",
    TGLFsettings=None,
    restart=restart,
    forceIfRestart=True,
    extraOptions={"USE_BPER": False, "USE_BPAR": False},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="ES")

tglf.run(
    subFolderTGLF="run1/",
    TGLFsettings=None,
    restart=restart,
    forceIfRestart=True,
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="EM")

tglf.plot(labels=["EM","ES"])
