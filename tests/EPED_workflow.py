import os
import matplotlib.pyplot as plt
from mitim_tools.eped_tools import EPEDtools
from mitim_tools import __mitimroot__

cold_start = True

folder = __mitimroot__ / "tests" / "scratch" / "eped_test"

if cold_start and os.path.exists(folder):
    os.system(f"rm -r {folder}")

eped = EPEDtools.EPED(folder=folder)

eped.run(
    subfolder = 'case1',
    input_params = {
        'ip': 8.7,
        'bt': 12.16,
        'r': 1.85,
        'a': 0.57,
        'kappa': 1.9,
        'delta': 0.5,
        'neped': 30.0,
        'betan': 1.0,
        'zeffped': 1.5,
        'nesep': 10.0,
        'tesep': 100.0,
    },
    scan_param = {'variable': 'neped', 'values': [15.0, 30.0, 45.0, 60.0, 75.0]},
    keep_nsep_ratio = 0.4,
    nproc_per_run = 64,
    cold_start = True,
)

eped.read(subfolder='case1')

eped.plot(labels=['case1'])
plt.show()