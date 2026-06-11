"""
CAPABILITY: NEO in-process (ctypes) vs standard execution
---------------------------------------------------------
This script teaches the in-process execution mode of NEO: the code runs via
ctypes against a shared library (libneo_serial.so) INSIDE the python process
— no subprocess fork, no folders, no input/output files. The same case is run
both ways and plotted together to show the user there is no difference in the
physics results.

PREREQUISITE — build the shared library once per machine:
    cd src/mitim_tools/simulation_tools/interfaces
    bash build_neo_lib.sh

Key teaching points:
    1. Only TWO changes with respect to a standard run: `in_process=True` in
       the constructor, and prep() without a folder (there is no file I/O at
       any step). Everything else (run/read/run_scan, code_settings,
       extraOptions, multipliers) works identically.
    2. Why it exists: zero fork/file overhead makes the neoclassical side of
       PORTALS essentially free (`transport.in_process: true`), and scan
       methods parallelize across all CPU cores via threads.
    3. The physics is identical: the overlaid plot shows both runs on top of
       each other (sub-percent differences come from output-file precision in
       the standard route).
    4. Results from different NEO objects can be combined into one notebook
       by copying the labeled entry across `results` dictionaries.
"""

import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"
rhos = [0.8, 0.9]  # edge radii, where the neoclassical contribution matters most

# Working folder — only needed by the STANDARD run; the in-process one writes nothing
folder = __mitimroot__ / "tests" / "scratch" / "capability_neo_inprocess"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Standard (subprocess) run
# ---------------------------------------------------------------------------------------------------------------------

neo_sub = NEOtools.NEO(rhos=rhos, in_process=False)
neo_sub.prep(input_gacode, folder)

neo_sub.run("run_subprocess/", code_settings="Sonic", cold_start=cold_start, forceIfcold_start=True)
neo_sub.read(label="subprocess")

# ---------------------------------------------------------------------------------------------------------------------
# 2. The exact same case, in-process (zero file I/O — note: no folder in prep)
# ---------------------------------------------------------------------------------------------------------------------

neo_ip = NEOtools.NEO(rhos=rhos, in_process=True)
neo_ip.prep(input_gacode)

neo_ip.run("run_inprocess/", code_settings="Sonic", cold_start=cold_start, forceIfcold_start=True)
neo_ip.read(label="in-process")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Compare: numbers to the screen, profiles overlaid in one notebook
# ---------------------------------------------------------------------------------------------------------------------

for i, rho in enumerate(rhos):
    Qe_s = neo_sub.results["subprocess"]["output"][i].Qe
    Qe_i = neo_ip.results["in-process"]["output"][i].Qe
    Qi_s = neo_sub.results["subprocess"]["output"][i].Qi
    Qi_i = neo_ip.results["in-process"]["output"][i].Qi
    agree = np.isclose(Qe_s, Qe_i, rtol=5e-3) and np.isclose(Qi_s, Qi_i, rtol=5e-3)
    print(f"rho={rho:.2f}  Qe: {Qe_s:.4e} (subprocess) vs {Qe_i:.4e} (in-process)"
          f"  |  Qi: {Qi_s:.4e} vs {Qi_i:.4e}  ->  {'AGREE' if agree else 'DISAGREE'}")

# Bring the in-process result into the standard object so both labels share one notebook
neo_sub.results["in-process"] = neo_ip.results["in-process"]

# All figures go into a multi-tab MITIM FigureNotebook (neo_sub.fn); show() opens the GUI.
# The two labels should lie exactly on top of each other in every panel
neo_sub.plot(labels=["subprocess", "in-process"])
neo_sub.fn.show()
