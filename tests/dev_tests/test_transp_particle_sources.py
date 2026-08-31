"""
test_transp_particle_sources.py
===============================
Tests for four CDFtools fixes:

  1. transp_output.to_profiles now maps the TRANSP particle sources into the
     extracted state: qpar_beam from SBTH (fast-ion thermalization source, the
     rate at which beam ions join the thermal population — NOT the deposition
     BDEP/SDEP family) and qpar_wall from SWD (thermal ion source from
     wall/recycled neutrals; SVD would double-count the beam and SISRC adds
     volume-recombination re-ionization). Previously both were left at zero.

  2. transp_output.to_profiles pins the state's TOTAL radiation (qbrem+qsync+qline)
     to TRANSP's PRAD instead of mapping only the internally-computed
     PRAD_BR/PRAD_CY/PRAD_LI split. In decks that prescribe measured radiation
     (.QRA ufile) that split is a small subset of PRAD -- 5% (207958) and 20%
     (207965) here -- so the old mapping made every extracted state under-radiate
     and biased the electron target flux high.

  3. transp_output.to_profiles maps TRANSP's P0NET into qioni with the sign
     FLIPPED: P0NET is an ion LOSS (Pi_teo = Pi + Pei - Pcx) whereas gacode sums
     qioni into qi with +1. Previously the channel was absent altogether.

  4. transp_output._impurity_mass_from_namelist parses BOTH namelist forms,
     indexed (AIMPS(2) = 40.0) and multi-valued (AIMPS = 12.0, 40.0). The old
     lookup only understood the indexed form, so multi-impurity runs silently
     fell back to mass = 2*Zave (Ar came out as A~36 instead of the A=40
     TRANSP actually ran with).

The namelist test is CDF-free. The rest need the local DIII-D CDFs
(207958/207965, ~laptop-only data) and are SKIPPED when absent; the particle
reference values were validated against TRANSP's own BSTH volume integral
(0.3-0.9% agreement across 7 shots), and the radiation test compares against
TRANSP's own PRAD volume integral computed from the same CDF.

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

T_EXTRACT, T_WINDOW = 3.6, 0.2

# 207958 @ 3.6 s (window 0.2 s): TRANSP BSTH = 2.351e20 N/s; validated state integral 2.357e20 N/s
REF_BEAM_207958 = 2.357e20
# SWD volume integral, same window (from the CDF itself)
REF_WALL_207958 = 2.459e21

_LOADED = {}


def _loaded(cdf):
    """CDF + extracted state, cached so the tests below read each CDF once."""
    if cdf not in _LOADED:
        c = CDFtools.transp_output(cdf)
        p = c.to_profiles(time_extraction=T_EXTRACT, time_window=T_WINDOW)
        p.derive_quantities()
        _LOADED[cdf] = (c, p)
    return _LOADED[cdf]


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
    _, p = _loaded(CDF_207958)

    beam = volint(p, "qpar_beam(1/m^3/s)")
    wall = volint(p, "qpar_wall(1/m^3/s)")
    assert abs(beam / REF_BEAM_207958 - 1) < 0.10, f"qpar_beam volint {beam:.3e} vs ref {REF_BEAM_207958:.3e}"
    assert abs(wall / REF_WALL_207958 - 1) < 0.10, f"qpar_wall volint {wall:.3e} vs ref {REF_WALL_207958:.3e}"
    assert np.all(p.profiles["qpar_beam(1/m^3/s)"] >= 0)
    print(f"PASS: 207958 sources (beam volint {beam:.3e} /s [ref {REF_BEAM_207958:.1e}], wall {wall:.3e} /s)")


def _radiation_total(cdf, label):
    """The state's total radiation must equal TRANSP's PRAD, not the internal PRAD_BR/CY/LI split."""
    if not cdf.is_file():
        print(f"SKIP: {label} CDF not available")
        return
    c, p = _loaded(cdf)

    mask = np.abs(c.t - T_EXTRACT) <= T_WINDOW / 2
    prad = float(np.mean(c.PradT[mask]))                                 # TRANSP's own PRAD integral, MW
    subset = float(np.mean((c.PradT_b + c.PradT_c + c.PradT_l)[mask]))   # what to_profiles used to map

    qrad = sum(volint(p, k) for k in ["qbrem(MW/m^3)", "qsync(MW/m^3)", "qline(MW/m^3)"])

    # 2% covers the zone-centre->boundary spline and the fact that the state integrates on MITIM's
    # Miller volumes while TRANSP integrates on its own zone volumes; the mapping itself is exact
    assert abs(qrad / prad - 1) < 0.02, f"{label}: state qrad {qrad:.4f} MW vs TRANSP PRAD {prad:.4f} MW"
    # guard against silently regressing to the internal split on a deck where it is a small subset
    assert subset / prad < 0.5, f"{label}: PRAD_BR/CY/LI is {subset/prad:.2f} of PRAD, deck not representative"
    assert qrad > 2 * subset, f"{label}: state radiation {qrad:.4f} MW looks like the internal split {subset:.4f} MW"
    # the spline can leave a numerically negligible negative overshoot in qline; nothing structural
    qline = p.profiles["qline(MW/m^3)"]
    assert qline.min() > -1e-3 * qline.max(), f"{label}: qline dips to {qline.min():.3e} MW/m^3"

    print(f"PASS: {label} radiation total {qrad:.4f} MW vs TRANSP PRAD {prad:.4f} MW "
          f"({100*(qrad/prad-1):+.2f}%); old internal split was {subset:.4f} MW ({subset/prad:.3f} of PRAD)")


def test_radiation_207958():
    _radiation_total(CDF_207958, "207958 baseline")


def test_radiation_207965():
    _radiation_total(CDF_207965, "207965 Argon")


def test_cx_sink_207958():
    """qioni must be P0NET negated: CX is an ion loss, but gacode adds qioni into qi with +1."""
    if not CDF_207958.is_file():
        print("SKIP: 207958 CDF not available")
        return
    c, p = _loaded(CDF_207958)

    mask = np.abs(c.t - T_EXTRACT) <= T_WINDOW / 2
    pcx = float(np.mean(c.PcxT[mask]))            # TRANSP's own P0NET volume integral, MW, positive = loss
    qioni = volint(p, "qioni(MW/m^3)")

    assert pcx > 0.0, f"P0NET integral {pcx:.4f} MW is not a loss, TRANSP sign convention changed"
    assert qioni < 0.0, f"qioni volume integral {qioni:.4f} MW is not a sink, the sign flip was lost"
    # 5%: same grid/volume mismatch as the radiation check, but P0NET is small (~0.05 MW) and
    # edge-peaked, so the zone-centre->boundary spline costs relatively more here
    assert abs(qioni / -pcx - 1) < 0.05, f"qioni {qioni:.4f} MW vs -P0NET {-pcx:.4f} MW"

    print(f"PASS: 207958 CX sink qioni {qioni:.4f} MW vs -P0NET {-pcx:.4f} MW ({100*(qioni/-pcx-1):+.2f}%)")


def test_ar_mass_207965():
    if not CDF_207965.is_file():
        print("SKIP: 207965 CDF not available")
        return
    _, p = _loaded(CDF_207965)
    names = [str(n) for n in p.profiles["name"]]
    mass_ar = float(p.profiles["mass"][names.index("AR")])
    # A=40 amu in the D=2.0 gacode convention: 40 * u/mD * 2 = 39.73
    assert abs(mass_ar - 39.7) < 0.3, f"Ar mass {mass_ar:.2f}, expected ~39.7 (was ~36 with the 2*Zave fallback)"
    print(f"PASS: 207965 Ar mass native from namelist ({mass_ar:.2f} in D=2 units)")


if __name__ == "__main__":
    test_aimps_parsing()
    test_particle_sources_207958()
    test_radiation_207958()
    test_radiation_207965()
    test_cx_sink_207958()
    test_ar_mass_207965()
    print("All tests passed.")
