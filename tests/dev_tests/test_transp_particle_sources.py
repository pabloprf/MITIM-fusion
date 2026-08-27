"""
test_transp_particle_sources.py
===============================
Tests for two CDFtools fixes:

  1. transp_output.to_profiles now maps the TRANSP particle sources into the
     extracted state: qpar_beam from SBTH (fast-ion thermalization source, the
     rate at which beam ions join the thermal population — NOT the deposition
     BDEP/SDEP family) and qpar_wall from SWD (thermal ion source from
     wall/recycled neutrals; SVD would double-count the beam and SISRC adds
     volume-recombination re-ionization). Previously both were left at zero.

  2. transp_output._impurity_mass_from_namelist parses BOTH namelist forms,
     indexed (AIMPS(2) = 40.0) and multi-valued (AIMPS = 12.0, 40.0). The old
     lookup only understood the indexed form, so multi-impurity runs silently
     fell back to mass = 2*Zave (Ar came out as A~36 instead of the A=40
     TRANSP actually ran with).

Test 1 is CDF-free (namelist parsing only). Tests 2-3 need the local DIII-D
CDFs (207958/207965, ~laptop-only data) and are SKIPPED when absent; reference
values were validated against TRANSP's own BSTH volume integral (0.3-0.9%
agreement across 7 shots).

Run as:

    python tests/dev_tests/test_transp_particle_sources.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.transp_tools import CDFtools
from mitim_modules.powertorch.utils import CALCtools

DECKS = Path("/Users/pablorf/PROJECTS/project_2026_DIIIDexperiment/04_transp_interpretive/01_transp_run_decks")
CDF_207958 = DECKS / "207958Z01_ida_v4/deck/207958Z01.CDF"
CDF_207965 = DECKS / "207965Z01_ida_v4/deck/207965Z01.CDF"

# 207958 @ 3.6 s (window 0.2 s): TRANSP BSTH = 2.351e20 N/s; validated state integral 2.357e20 N/s
REF_BEAM_207958 = 2.357e20
# SWD volume integral, same window (from the CDF itself)
REF_WALL_207958 = 2.459e21


def volint(p, key):
    return float(CALCtools.volume_integration(p.profiles[key], p.derived["r"], p.derived["volp_geo"])[-1])


def test_aimps_parsing():
    dummy = types.SimpleNamespace()
    with tempfile.TemporaryDirectory() as tmp:
        nml = Path(tmp) / "TR.DAT"

        nml.write_text(" AIMPS = 12.0, 40.0\n XZIMPS = 6.0, 18.0\n")
        dummy.LocationNML = nml
        assert CDFtools.transp_output._impurity_mass_from_namelist(dummy, 0) == 12.0
        assert CDFtools.transp_output._impurity_mass_from_namelist(dummy, 1) == 40.0

        nml.write_text(" aimps(1)  = 12.0 ! Atomic mass of impurity species\n aimps(2)  = 20.0\n")
        assert CDFtools.transp_output._impurity_mass_from_namelist(dummy, 1) == 20.0

        nml.write_text(" XZIMPS = 6.0\n")  # no AIMPS at all -> must raise (caller falls back to 2*Zave)
        raised = False
        try:
            CDFtools.transp_output._impurity_mass_from_namelist(dummy, 0)
        except Exception:
            raised = True
        assert raised
    print("PASS: AIMPS parsing (multi-value, indexed, missing)")


def test_particle_sources_207958():
    if not CDF_207958.is_file():
        print("SKIP: 207958 CDF not available")
        return
    c = CDFtools.transp_output(CDF_207958)
    p = c.to_profiles(time_extraction=3.6, time_window=0.2)
    p.derive_quantities()

    beam = volint(p, "qpar_beam(1/m^3/s)")
    wall = volint(p, "qpar_wall(1/m^3/s)")
    assert abs(beam / REF_BEAM_207958 - 1) < 0.10, f"qpar_beam volint {beam:.3e} vs ref {REF_BEAM_207958:.3e}"
    assert abs(wall / REF_WALL_207958 - 1) < 0.10, f"qpar_wall volint {wall:.3e} vs ref {REF_WALL_207958:.3e}"
    assert np.all(p.profiles["qpar_beam(1/m^3/s)"] >= 0)
    print(f"PASS: 207958 sources (beam volint {beam:.3e} /s [ref {REF_BEAM_207958:.1e}], wall {wall:.3e} /s)")


def test_ar_mass_207965():
    if not CDF_207965.is_file():
        print("SKIP: 207965 CDF not available")
        return
    c = CDFtools.transp_output(CDF_207965)
    p = c.to_profiles(time_extraction=3.6, time_window=0.2)
    names = [str(n) for n in p.profiles["name"]]
    mass_ar = float(p.profiles["mass"][names.index("AR")])
    # A=40 amu in the D=2.0 gacode convention: 40 * u/mD * 2 = 39.73
    assert abs(mass_ar - 39.7) < 0.3, f"Ar mass {mass_ar:.2f}, expected ~39.7 (was ~36 with the 2*Zave fallback)"
    print(f"PASS: 207965 Ar mass native from namelist ({mass_ar:.2f} in D=2 units)")


if __name__ == "__main__":
    test_aimps_parsing()
    test_particle_sources_207958()
    test_ar_mass_207965()
    print("All tests passed.")
