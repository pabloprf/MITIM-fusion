"""
Decorrelation lag of the ACF standard-error estimator (GKaveraging._grab_ncorrelation): it is the
FIRST lag where the ACF drops to 1/e, not the lag closest to 1/e over all lags.

    python tests/dev_tests/test_gkaveraging_acf.py        (or pytest)
"""

import numpy as np
from mitim_tools.simulation_tools.utils import GKaveraging


def ar1(phi, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def test_ar1_matches_analytic_lag():
    # ACF(k) = phi^k -> 1/e at k = -1/ln(phi)
    for phi in (0.8, 0.9, 0.95):
        S = ar1(phi)
        n_corr, icor = GKaveraging._grab_ncorrelation(S)
        tau = -1.0 / np.log(phi)
        assert abs(icor / tau - 1) < 0.2, (phi, icor, tau)
        assert np.isclose(n_corr, len(S) / (3 * icor))


def test_sinusoid_first_crossing():
    # A sinusoid of period P has ACF ~ cos(2 pi k / P): first 1/e crossing at P acos(1/e) / (2 pi)
    P, n = 100.0, 2000
    rng = np.random.default_rng(1)
    S = np.sin(2 * np.pi * np.arange(n) / P) + 0.05 * rng.normal(size=n)
    _, icor = GKaveraging._grab_ncorrelation(S)
    first = P * np.arccos(1 / np.e) / (2 * np.pi)
    assert abs(icor / first - 1) < 0.1, (icor, first)


def test_late_return_to_one_over_e_is_ignored(monkeypatch):
    # ACF that crosses 1/e between lags 1 and 2 (samples 0.6, 0.2, neither close to 1/e) and later
    # oscillates back through exactly 1/e at lag 150: the old all-lags argmin latched onto lag 150
    n = 400
    acf = np.full(n, 0.1)
    acf[:3] = [1.0, 0.6, 0.2]
    acf[3:] = 0.1 + (1 / np.e - 0.1) * np.cos(2 * np.pi * (np.arange(3, n) - 150) / 200)
    assert np.abs(acf - 1 / np.e).argmin() == 150
    monkeypatch.setattr(GKaveraging.sm.tsa, "acf", lambda S, nlags: acf)
    n_corr, icor = GKaveraging._grab_ncorrelation(np.zeros(n))
    expected = 1 + (0.6 - 1 / np.e) / (0.6 - 0.2)
    assert np.isclose(icor, expected) and np.isclose(n_corr, n / (3 * expected))


def test_white_noise_floors_to_one_lag():
    S = np.random.default_rng(2).normal(size=5000)
    n_corr, icor = GKaveraging._grab_ncorrelation(S)
    assert icor == 1 and np.isclose(n_corr, len(S) / 3)


def test_never_reaching_one_over_e_uses_full_length(monkeypatch):
    # The biased ACF estimator decays to ~0 at the last lag, so force an ACF that stays above 1/e
    monkeypatch.setattr(GKaveraging.sm.tsa, "acf", lambda S, nlags: np.linspace(1.0, 0.5, len(S)))
    S = np.zeros(300)
    n_corr, icor = GKaveraging._grab_ncorrelation(S)
    assert icor == len(S) and np.isclose(n_corr, 1 / 3)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
