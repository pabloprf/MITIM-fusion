"""
The Lengyel beat moves the separatrix temperature of electrons and THERMAL ions only.

A fast species carries T = 2/3 W/n, whose separatrix value is not a boundary condition
(n -> 0 there, so it can be thousands of keV). Shifting a fast species so that its edge
value lands on Tesep used to move its whole profile by that amount (negative alpha
temperatures in MAESTRO chains without a pedestal top).

Usage:
    python tests/dev_tests/test_lengyel_fast_ions.py
"""

import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.maestro.utils.LENGYELbeat import _modify_temperatures
from mitim_tools import __mitimroot__

p0 = PROFILEStools.gacode_state(__mitimroot__ / "tests" / "data" / "input.gacode")

fast = [i for i, sp in enumerate(p0.Species) if sp["S"] == "fast"]
thermal = [i for i, sp in enumerate(p0.Species) if sp["S"] == "therm"]
assert len(fast) > 0, "tests/data/input.gacode no longer has a fast species"

# Fast-ion edge value as TRANSP conversions leave it where the fast density vanishes [keV]
p0.profiles["ti(keV)"][-1, fast] = 7809.25

Tesep = 0.150  # keV

for rhotop in [None, 0.9]:
    p = copy.deepcopy(p0)
    _modify_temperatures(p, Tesep, rhotop)

    assert np.array_equal(p.profiles["ti(keV)"][:, fast], p0.profiles["ti(keV)"][:, fast])
    assert np.isclose(p.profiles["te(keV)"][-1], Tesep)
    assert np.allclose(p.profiles["ti(keV)"][-1, thermal], Tesep)
    assert not np.array_equal(p.profiles["ti(keV)"][:, thermal], p0.profiles["ti(keV)"][:, thermal])

    print(f"rhotop={rhotop}: fast ions untouched, Te and thermal Ti at the separatrix = {Tesep*1e3:.0f} eV")

print("\nAll good")
