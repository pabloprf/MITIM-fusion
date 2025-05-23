from re import sub
from mitim_tools.eped_tools import EPEDtools
from mitim_tools import __mitimroot__

folder = __mitimroot__ / "tests" / "scratch" / "eped_test"

eped = EPEDtools.EPED(folder=folder)

eped.run(
    subfolder = 'run1',
    input_params = {
        'ip': 0.5,
        'bt': 1.0,
        'r': 1.0,
        'a': 0.5,
        'kappa': 1.5,
        'delta': 0.5,
        'neped': 1.0,
        'betan': 0.5,
        'zeffped': 1.0,
        'nesep': 0.25,
        'tesep': 1.0,
    },
    nproc = 64
)

eped.read(subfolder='run1')

