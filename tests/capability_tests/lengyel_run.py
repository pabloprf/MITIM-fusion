"""
CAPABILITY: Lengyel divertor/SOL model from an input.gacode
-----------------------------------------------------------
This script teaches how to run the (extended) Lengyel model — a fast 0D/1D
scrape-off-layer model relating power exhaust, separatrix conditions and the
impurity concentration required for detachment — populating its inputs from a
plasma state. It runs locally and in seconds.

PREREQUISITES:
    - The `extended_lengyel` python package installed in the environment.
    - The RADAS_DIR environment variable pointing to a radas atomic-data
      directory (generate it once with: radas -c radas_config.yml -s tungsten
      ... with the proper config and impurities).

Key teaching points:
    1. The model is configured by a yaml namelist
       (templates/input.lengyel.controls.yaml by default); prep() reads it and
       overwrites the machine/plasma inputs (R, a, B, Ip, kappa95, delta95,
       P_sol, separatrix density, main-ion mass) with the values of the given
       plasma state.
    2. Any input can be overridden at run() time as a keyword argument — here
       the plasma current — without touching the yaml.
    3. run_scan() repeats the run over a list of values of one input (each
       case in its own subfolder) and plots the resulting trend — here the
       separatrix electron temperature vs the power crossing the separatrix.
    4. Inputs carry their units as strings (e.g. '2.0 MA', '10MW'): the model
       parses them, so there is no unit ambiguity.
"""

import os
from mitim_tools.simulation_tools.physics.LENGYELtools import Lengyel
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one subfolder per case lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_lengyel"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Atomic data location (see PREREQUISITES in the docstring)
radas_dir_env = os.getenv("RADAS_DIR")
if radas_dir_env is None:
    raise EnvironmentError("[MITIM] The RADAS_DIR environment variable is not set")

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Initialize and populate the inputs from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# Lengyel() reads the default controls yaml; a custom one can be passed with
# namelist_location=...
l = Lengyel()

# prep() sets the radas atomic-data directory and overwrites the machine/plasma inputs
# with the values derived from the input.gacode (printed to the terminal as it does so)
l.prep(
    radas_dir=radas_dir_env,
    input_gacode=input_gacode,
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Single run, overriding one input in-situ
# ---------------------------------------------------------------------------------------------------------------------

# Any input of the controls yaml can be overridden as a keyword argument; results are
# parsed into l.results (a dictionary of quantities with units)
l.run(
    folder / "tmp_run",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Scan the power crossing the separatrix (at a modified plasma current)
# ---------------------------------------------------------------------------------------------------------------------

# One run per value of `scan_name`; any other input override applies to all of them.
# plotYN=True plots the separatrix electron temperature vs the scanned input at the end
l.run_scan(
    folder=folder / "tmp_scan",
    scan_name="power_crossing_separatrix",
    scan_values=["10MW", "20MW", "30MW"],
    plasma_current="2.0 MA",
    plotYN=True,
)
