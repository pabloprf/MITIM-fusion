import os
from mitim_tools.simulation_tools.physics.LENGYELtools import Lengyel
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "lengyel_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Check if RADAS_DIR is set
radas_dir_env = os.getenv("RADAS_DIR")
if radas_dir_env is None:
    raise EnvironmentError("[MITIM] The RADAS_DIR environment variable is not set")

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

# Initialize Lengyel object
l = Lengyel()

# Prepare Lengyel with default inputs and changes from GACODE
l.prep(
    radas_dir = radas_dir_env,
    input_gacode = input_gacode
    )

# Run Lengyel standalone
l.run(
    folder / 'tmp_run',
    cold_start=cold_start
    )

# Run Lengyel scans
l.run_scan(
    folder = folder / 'tmp_scan',
    scan_name = 'power_crossing_separatrix',
    scan_values = ['10MW','20MW','30MW'],
    plasma_current = '2.0 MA', # Change any parameter from the default
    plotYN = True # Plot results after the scan
    )
