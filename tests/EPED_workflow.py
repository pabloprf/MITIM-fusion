from mitim_tools.eped_tools import EPEDtools
from mitim_tools import __mitimroot__

folder = __mitimroot__ / "tests" / "scratch" / "eped_test"

eped = EPEDtools.EPED(folder=folder)

eped.run(
    subfolder = 'run1',
    input_params = {
        'ip': 12.0,
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
    nproc = 64,
    cold_start = True,
)

eped.read(subfolder='run1')


eped.plot(labels=['run1'])
