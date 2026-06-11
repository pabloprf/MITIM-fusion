"""
CAPABILITY: TGLF in-process (ctypes) vs standard execution
----------------------------------------------------------
This script teaches the in-process execution mode of TGLF: the code runs via
ctypes against a shared library (libtglf_serial.so) INSIDE the python process
— no subprocess fork, no folders, no input/output files. The same case is run
both ways and plotted together to show the user there is no difference in the
physics results.

PREREQUISITE — build the shared library once per machine:
    cd src/mitim_tools/simulation_tools/interfaces
    bash build_tglf_lib.sh

Key teaching points:
    1. Only TWO changes with respect to a standard run: `in_process=True` in
       the constructor, and prep() without a folder (there is no file I/O at
       any step). Everything else (run/read/scans, code_settings,
       extraOptions) works identically.
    2. Why it exists: zero fork/file overhead makes massive evaluations cheap
       — this is what `transport.in_process: true` uses inside PORTALS, and
       scan methods parallelize across all CPU cores via threads.
    3. The physics is identical: the overlaid plot shows both runs on top of
       each other. Tiny differences (<0.1%) are file-precision artifacts —
       the standard route reads fluxes from out.tglf.gbflux (4-5 significant
       figures), while in-process keeps full double precision.
    4. Results from different TGLF objects can be combined into one notebook
       by copying the labeled entry across `results` dictionaries.
    5. Limitation: runWaveForms is not supported in-process (it is skipped
       with a warning).
"""

import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
rhos = [0.5, 0.7]

# Working folder — only needed by the STANDARD run; the in-process one writes nothing
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_inprocess"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Standard (subprocess) run
# ---------------------------------------------------------------------------------------------------------------------

tglf_sub = TGLFtools.TGLF(rhos=rhos, in_process=False)
tglf_sub.prep(input_gacode, folder, cold_start=cold_start)

tglf_sub.run(
    "run_subprocess/",
    code_settings="SAT2",
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf_sub.read(label="subprocess")

# ---------------------------------------------------------------------------------------------------------------------
# 2. The exact same case, in-process (zero file I/O — note: no folder in prep)
# ---------------------------------------------------------------------------------------------------------------------

tglf_ip = TGLFtools.TGLF(rhos=rhos, in_process=True)
tglf_ip.prep(input_gacode)

tglf_ip.run(
    "run_inprocess/",  # label only: nothing is written to disk
    code_settings="SAT2",
    cold_start=cold_start,
    forceIfcold_start=True,
)
tglf_ip.read(label="in-process")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Compare: numbers to the screen, profiles overlaid in one notebook
# ---------------------------------------------------------------------------------------------------------------------

for i, rho in enumerate(rhos):
    Qe_s = tglf_sub.results["subprocess"]["output"][i].Qe
    Qe_i = tglf_ip.results["in-process"]["output"][i].Qe
    Qi_s = tglf_sub.results["subprocess"]["output"][i].Qi
    Qi_i = tglf_ip.results["in-process"]["output"][i].Qi
    # 0.1% tolerance: the difference is just the file precision of the standard route
    agree = np.isclose(Qe_s, Qe_i, rtol=1e-3) and np.isclose(Qi_s, Qi_i, rtol=1e-3)
    print(f"rho={rho:.2f}  Qe: {Qe_s:.5f} (subprocess) vs {Qe_i:.5f} (in-process)"
          f"  |  Qi: {Qi_s:.5f} vs {Qi_i:.5f}  ->  {'AGREE' if agree else 'DISAGREE'}")

# Bring the in-process result into the standard object so both labels share one notebook
tglf_sub.results["in-process"] = tglf_ip.results["in-process"]

# All figures go into a multi-tab MITIM FigureNotebook (tglf_sub.fn); show() opens the GUI.
# The two labels should lie exactly on top of each other in every panel
tglf_sub.plot(labels=["subprocess", "in-process"])
tglf_sub.fn.show()
