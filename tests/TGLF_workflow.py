import os
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglf_test"
input_tglf = __mitimroot__ / "tests" / "data" / "input.tglf"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

tglf = TGLFtools.TGLF()
tglf.prep_from_file(folder, input_tglf)

tglf.run(
    "run1/",
    code_settings='SAT1',
    cold_start=cold_start,
    runWaveForms  = [0.67, 10.0],
    forceIfcold_start=True,
    extraOptions={"USE_BPER": False, "USE_BPAR": False},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="ES (SAT1)")

tglf.run(
    "run2/",
    code_settings='SAT1',
    cold_start=cold_start,
    forceIfcold_start=True,
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="EM (SAT1)")

tglf.run(
    "run3/",
    code_settings='SAT3',
    cold_start=cold_start,
    forceIfcold_start=True,
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    slurm_setup={"cores": 4, "minutes": 1},
)

tglf.read(label="EM (SAT3)")

tglf.plot(labels=["EM (SAT1)","ES (SAT1)", "EM (SAT3)"])

# Required if running in non-interactive mode
tglf.fn.show()