"""
test_transp_fast_ions_time_averaging.py
=======================================
Tests for the TRANSP -> input.gacode extraction (CDFtools.transp_output.to_profiles):

  1. getSpecies builds one [fast] species per injected beam isotope (BDENS_D/T/H) and a
     thermal H species (NH). Previously only fusion products and ICRF minorities were fast
     species, so NBI plasmas lost the beam density (dilution), the beam pressure and
     thermal H: quasineutrality of the extracted state was off by ~1.6% at mid-radius on
     the JET DT run used here, and ~12% of the pressure was missing.

  2. Each fast species carries the pressure-consistent temperature
     T = 2/3 (W_perp + W_par)/n from its OWN energy densities (UBPRP_D/UBPAR_D, ...,
     UFPRP_4/UFPAR_4), so sum_fast n*T reproduces TRANSP's stored fast pressure
     (2/3 (UFASTPP+UFASTPA), and PMHDF_IN for near-isotropic populations).

  3. Time windows (time_window > 0) are a trapezoidal integral over the output slices
     divided by the window duration (TRANSP's output grid is not uniform); the fast
     temperature is formed from the window-averaged W and n (not <T>), and the flux
     surfaces are averaged slice by slice before the MXH fit.

The weights test is CDF-free. The rest need the local JET CDF (42847V04, laptop-only)
and are SKIPPED when absent. Run as:

    python tests/dev_tests/test_transp_fast_ions_time_averaging.py
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from mitim_tools.transp_tools import CDFtools

CDF_JET = Path("/Users/pablorf/PROJECTS/project_2026_JETdt/tglf/42847V04.CDF")
T_EXTRACT = 14.087  # beams on; the last slice of this run is a post-beam collapsed state

_cache = {}


def _cdf():
    if "c" not in _cache:
        _cache["c"] = CDFtools.transp_output(CDF_JET)
    return _cache["c"]


def test_time_window_weights():
    fake = types.SimpleNamespace(t=np.array([0.0, 0.1, 0.15, 0.4]))
    w = CDFtools.transp_output._time_window_weights(fake, np.arange(4))
    assert abs(w.sum() - 1.0) < 1e-14
    # integral of f=t over [0, 0.4] divided by the duration: exact for trapezoid
    assert abs(np.dot(w, fake.t) - 0.2) < 1e-14, np.dot(w, fake.t)
    assert np.allclose(w, [0.125, 0.1875, 0.375, 0.3125])
    # single slice
    assert np.allclose(CDFtools.transp_output._time_window_weights(fake, np.array([2])), [1.0])
    # uniform grid: half-weight endpoints
    fake.t = np.linspace(0, 1, 5)
    assert np.allclose(CDFtools.transp_output._time_window_weights(fake, np.arange(5)), [0.125, 0.25, 0.25, 0.25, 0.125])
    print("PASS: trapezoidal time-window weights")


def _fast_pressure_on_x(c, p, it_range):
    fast = [i for i in range(len(p.profiles["name"])) if p.profiles["type"][i] == "[fast]"]
    pf = np.sum(p.profiles["ni(10^19/m^3)"][:, fast] * 1e19 * p.profiles["ti(keV)"][:, fast] * 1e3 * c.e_J, axis=1)
    return np.interp(c.x[it_range[0]], p.profiles["rho(-)"], pf)


def test_species_and_fast_pressure():
    if not CDF_JET.is_file():
        print("SKIP: JET 42847V04 CDF not available")
        return
    c = _cdf()
    p = c.to_profiles(time_extraction=T_EXTRACT)
    _cache["p_slice"] = p

    tags = [f"{n}{t}" for n, t in zip(p.profiles["name"], p.profiles["type"])]
    for must in ["H[therm]", "D[fast]", "T[fast]", "He[fast]"]:
        assert must in tags, f"{must} missing from {tags}"

    ne = p.profiles["ne(10^19/m^3)"]
    qn = (np.sum(p.profiles["ni(10^19/m^3)"] * p.profiles["z"], axis=1) - ne) / ne
    assert np.max(np.abs(qn)) < 1e-3, f"quasineutrality error {np.max(np.abs(qn)):.2e}"

    it = np.argmin(np.abs(c.t - T_EXTRACT))
    pf_transp = 2.0 / 3.0 * (c.f["UFASTPP"][it, :] + c.f["UFASTPA"][it, :]) * 1e6
    pf_state = _fast_pressure_on_x(c, p, np.array([it]))
    err = np.max(np.abs(pf_state - pf_transp)) / pf_transp.max()
    assert err < 1e-2, f"fast pressure mismatch {err:.2e}"

    # per-isotope beam temperature vs TRANSP's own mean beam energy (<E> = 3/2 T for the same n and W)
    for iso in ["D", "T"]:
        E = c.f[f"EBEAM_{iso}"][it, :25] * 1e-3
        T = getattr(c, f"Tb{iso}")[it, :25]
        assert np.allclose(1.5 * T, E, rtol=1e-2), f"Tb{iso} vs EBEAM_{iso}"

    print(f"PASS: species {tags}; |QN err| < {np.max(np.abs(qn)):.1e}; fast pressure vs TRANSP within {err:.1e}; "
          f"axis fast pressure {pf_state[0]*1e-3:.1f} kPa")


def test_time_window():
    if not CDF_JET.is_file():
        print("SKIP: JET 42847V04 CDF not available")
        return
    c = _cdf()
    tw = 0.1
    p = c.to_profiles(time_extraction=T_EXTRACT, time_window=tw)
    it_range = np.where(np.abs(c.t - T_EXTRACT) <= tw / 2)[0]
    assert len(it_range) > 1
    w = c._time_window_weights(it_range)

    W = c.f["UFASTPP"][:] + c.f["UFASTPA"][:]
    pf_transp = 2.0 / 3.0 * np.tensordot(w, W[it_range, :], axes=1) * 1e6
    pf_state = _fast_pressure_on_x(c, p, it_range)
    err = np.max(np.abs(pf_state - pf_transp)) / pf_transp.max()
    assert err < 1e-2, f"window fast pressure mismatch {err:.2e}"

    # the averaged state stays close to the central slice for this quiet window
    ps = _cache["p_slice"]
    for k in ["te(keV)", "kappa(-)", "rmaj(m)"]:
        d = np.max(np.abs(p.profiles[k] - ps.profiles[k])) / np.max(np.abs(ps.profiles[k]))
        assert d < 0.05, f"{k} window vs slice differ by {d:.2e}"

    # default call (time_extraction=None, time_window=0) must still work: single slice at ind_saw
    p0 = c.to_profiles()
    assert np.all(np.isfinite(p0.profiles["ti(keV)"]))

    print(f"PASS: {len(it_range)}-slice window, fast pressure vs trapezoid-averaged TRANSP within {err:.1e}")


if __name__ == "__main__":
    test_time_window_weights()
    test_species_and_fast_pressure()
    test_time_window()
    print("All tests passed.")
