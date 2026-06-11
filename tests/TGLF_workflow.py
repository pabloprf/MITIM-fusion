import os
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglf_test"
input_tglf = __mitimroot__ / "tests" / "data" / "input.tglf"
npz_file   = folder / "tglf_results.npz"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

tglf = TGLFtools.TGLF()
tglf.prep_from_file(folder, input_tglf)

tglf.run(
    "run1/",
    code_settings='SAT1',
    cold_start=cold_start,
    runWaveForms  = [0.67, 10.0],
    forceIfcold_start=True,
    extraOptions={"USE_BPER": False, "USE_BPAR": False},
    allocation={"resources_per_call": 4, "minutes": 10},
)

tglf.read(label="ES (SAT1)", save_and_cleanup=npz_file)

tglf.run(
    "run2/",
    code_settings='SAT1',
    cold_start=cold_start,
    forceIfcold_start=True,
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    allocation={"resources_per_call": 4, "minutes": 10},
)

tglf.read(label="EM (SAT1)", save_and_cleanup=npz_file)

tglf.run(
    "run3/",
    code_settings='SAT3',
    cold_start=cold_start,
    forceIfcold_start=True,
    extraOptions={"USE_BPER": True, "USE_BPAR": True},
    allocation={"resources_per_call": 4, "minutes": 10},
)

tglf.read(label="EM (SAT3)", save_and_cleanup=npz_file)

tglf_loaded = TGLFtools.TGLF.from_npz(npz_file)
tglf_loaded.plot(labels=["ES (SAT1)", "EM (SAT1)", "EM (SAT3)"])
tglf_loaded.fn.show()
