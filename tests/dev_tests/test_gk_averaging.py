"""
Unit tests for the GK flux time-averaging methods (GKaveraging.GKaverager) on a synthetic
trace: an exponential transient decaying onto a stationary AR(1) process with known mean.

    python tests/dev_tests/test_gk_averaging.py        (or pytest)
"""

import numpy as np
import pytest
from mitim_tools.simulation_tools.utils import GKaveraging


def synthetic_run(seed=0, n=900, dt=1.0, mean=(3.0, 1.5, -0.2), transient_amp=6.0, tau_transient=60.0, phi=0.85, sigma=0.6):
    """Qi/Qe/Ge traces: transient_amp*exp(-t/tau) overshoot + AR(1) noise (lag-1 correlation phi) around `mean`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    traces = {}
    for k, m in zip(GKaveraging.PRIMARY_CHANNELS, mean):
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + rng.normal(0.0, sigma * np.sqrt(1 - phi**2))
        traces[k] = m * (1.0 + transient_amp * np.exp(-t / tau_transient)) + noise * abs(m)
    return t, traces


def test_fixed_window_semantics():
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="fixed", tmin=-0.3, tmin_is_rel=True)
    assert a.flag == "fixed"
    assert np.isclose(a.t_start, t[-1] - 0.3 * (t[-1] - t[0]))
    b = GKaveraging.GKaverager(t, traces, method="fixed", tmin=-200, tmin_is_rel=False)
    assert np.isclose(b.t_start, t[-1] - 200)
    c = GKaveraging.GKaverager(t, traces, method="fixed", tmin=500.0)
    assert np.isclose(c.t_start, 500.0)


def test_fixed_mean_within_standard_error():
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="fixed", tmin=-0.5)
    for k, m in zip(GKaveraging.PRIMARY_CHANNELS, (3.0, 1.5, -0.2)):
        assert abs(a.stats[k]["mean"] - m) < 3 * a.stats[k]["std"], k
        assert a.stats[k]["std"] > 0


def test_howard_skips_transient_and_matches_mean():
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    assert a.flag in GKaveraging.GKwindow_howard.FLAGS_OK
    # The 6x overshoot (tau=60) has decayed to <10% of the level by t~110. Howard's importance
    # thresholds (DR_MAX, HUMP_IMP) are relative to the level, so a small residual transient in
    # the first block is tolerated: the start must be past the settle time, not past 5 tau.
    assert 100.0 < a.t_start < 450.0
    assert a.window_length >= GKaveraging.GKwindow_howard.min_window(t[-1] - t[0])
    for k, m in zip(GKaveraging.PRIMARY_CHANNELS, (3.0, 1.5, -0.2)):
        assert abs(a.stats[k]["mean"] - m) < 3 * a.stats[k]["std"], k


def test_howard_fallback_on_coarse_cadence():
    # Howard's trend/hump tests need >= 40 samples in a candidate window. With only 36 samples
    # (25 a/cs cadence, 875 a/cs run) no window qualifies, so the method must fall back to the
    # second half of the run and flag it (EARLY_SKIP trimming only applies above 50 samples).
    t, traces = synthetic_run(n=36, dt=25.0)
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    assert a.flag in GKaveraging.GKwindow_howard.FLAGS_FALLBACK
    assert np.isclose(a.t_start, t[0] + 0.5 * (t[-1] - t[0]))


def test_howard_accepts_mild_linear_drift():
    # Documented tolerance, not a bug: the trend test accepts |drift| < DR_MAX (45%) of the level
    # across the window, so a slow linear ramp still yields an "ok" window late in the run.
    t, traces = synthetic_run(transient_amp=0.0)
    traces = {k: v * (1.0 + 0.0005 * t) for k, v in traces.items()}
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    assert a.flag in GKaveraging.GKwindow_howard.FLAGS_OK


def test_mean_std_multidimensional_uses_window():
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    S = np.stack([traces["Qi"], traces["Qe"]])          # (2, nt)
    m, s = a.mean_std(S)
    assert m.shape == (2,) and s.shape == (2,)
    assert np.isclose(m[0], a.stats["Qi"]["mean"]) and np.isclose(s[0], a.stats["Qi"]["std"])


def test_to_dict_is_json_serializable():
    import json
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    d = json.loads(json.dumps(a.to_dict()))
    assert d["method"] == "howard_gkav" and d["flag"] == a.flag
    assert set(d["stats"]) == set(GKaveraging.PRIMARY_CHANNELS)


def test_plot_runs_headless():
    import matplotlib
    matplotlib.use("Agg")
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    axs = a.plot()
    assert axs.shape == (3, 2)


@pytest.mark.skipif(not GKaveraging.quends_available(), reason="quends not installed")
def test_quends_skips_transient_and_matches_mean():
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="quends")
    assert a.flag in ("ok", "questionable")           # Gaussian AR(1): the robust quantile trim should accept the tail
    assert 50.0 < a.t_start < 600.0
    for k, m in zip(GKaveraging.PRIMARY_CHANNELS, (3.0, 1.5, -0.2)):
        assert abs(a.stats[k]["mean"] - m) < 3 * a.stats[k]["std"], k
        assert a.diagnostics["stats"][k]["effective_sample_size"] > 1


@pytest.mark.skipif(not GKaveraging.quends_available(), reason="quends not installed")
def test_quends_uncertainty_with_howard_window():
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="howard_gkav", uncertainty="quends")
    b = GKaveraging.GKaverager(t, traces, method="howard_gkav")
    assert a.t_start == b.t_start and a.uncertainty == "quends" and b.uncertainty == "acf"
    for k in GKaveraging.PRIMARY_CHANNELS:
        assert np.isclose(a.stats[k]["mean"], b.stats[k]["mean"], rtol=0.05)      # same window, block vs plain mean
        assert 0.3 < a.stats[k]["std"] / b.stats[k]["std"] < 3.0                 # both are standard errors of the mean


def test_quends_missing_package_falls_back(monkeypatch):
    monkeypatch.setattr(GKaveraging, "quends_available", lambda: False)
    t, traces = synthetic_run()
    a = GKaveraging.GKaverager(t, traces, method="quends", tmin=-0.3)
    assert a.flag == "fallback" and np.isclose(a.t_start, t[-1] - 0.3 * (t[-1] - t[0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
