"""
Stationarity-based selection of the flux-averaging window for nonlinear gyrokinetic runs.

Vendored from N.T. Howard's `cgyro_window.py` (window-selection section, file dated
2026-09-17, retrieved from engaging /orcd/pool/003/pablorf_shared/nthoward). The code
between the VENDORED markers is kept verbatim so it can be diffed against the original
when that file evolves; only this header and the trailing helper are MITIM additions.

Conventions (from the original): time in a/cs, fluxes GB-normalized, output cadence
assumed ~1 a/cs (the smoothing width is converted to samples with the median dt).
The three traces are the ion heat flux (all ions), electron heat flux and electron
particle flux. `select_start` returns the absolute window start `t_start` and a flag:
  ok / ok2         : all three / two of three traces stationary from t_start
  questionable     : stationary window shorter than min_window(span)
  fallback         : no stationary window found; second half of the run used
  failure          : as fallback, but transients still active inside the window
  below_threshold  : |Qi| and |Qe| both below NOISE_LEVEL (GB); fluxes are noise
"""

import numpy as np

# ---------------------------------------------------------------- VENDORED (begin)
SMOOTH_W = 25          # samples (dt = 1 a/cs typically)
SETTLE_RUN = 30        # samples smooth trace must stay in band to settle
CAND_STEP = 10.0       # a/cs spacing of candidate start times
SUSTAIN = 3            # consecutive candidates that must pass
EARLY_SKIP = 15.0      # a/cs; ignore this initial stretch entirely (start-up
                       # spikes in the first few steps are not physical)

DN_MAX = 4.5           # trend significance: |drift| / block scatter
DR_MAX = 0.45          # trend importance:  |drift| / window level
HUMP_SIG = 7.0         # excursion significance: block-median range / in-block scatter
HUMP_IMP = 3.5         # excursion importance:  block-median range / window level
ACTIVITY_FRAC = 0.25   # trace is "active" once |smooth| exceeds this x its max

NOISE_LEVEL = 0.025    # |Qi| and |Qe| both below this -> fluxes are basically
                       # noise; stationarity failures are flagged
                       # 'below_threshold' rather than fallback/failure


def min_window(span):
    """Minimum acceptable averaging window (a/cs) for a run of this length."""
    if span < 500.0:
        return 150.0
    if span < 750.0:
        return 200.0
    return 250.0


def _smooth(y, w):
    w = max(3, int(w) | 1)
    # never wider than the trace: np.convolve(mode="same") returns max(len(y), w) points, so a run
    # stopped early (fewer samples than SMOOTH_W) would otherwise break every array operation after this
    w = min(w, max(1, (len(y) - 1) | 1))
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode="same")
    norm = np.convolve(np.ones_like(y), k, mode="same")
    return ys / norm


def _mad(x):
    x = np.asarray(x)
    if len(x) == 0:
        return 0.0
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _activity_time(t, ys):
    """First time the smoothed trace reaches a meaningful fraction of its max.

    Guards against 'settling' during a quiescent pre-turbulence phase in
    runs where the turbulence only develops late.
    """
    amp = np.max(np.abs(ys))
    if amp <= 0:
        return t[0]
    active = np.abs(ys) > ACTIVITY_FRAC * amp
    if not active.any():
        return t[0]
    return t[int(np.argmax(active))]


def _settle_time(t, y, ys, t_act):
    """First time (>= t_act) the smoothed trace enters and stays near the
    last-half level."""
    n = len(t)
    lh = slice(n // 2, n)
    ref = np.median(ys[lh])
    fluct = _mad(y[lh] - ys[lh])
    wander = _mad(ys[lh] - ref)
    band = max(3.0 * max(fluct, wander), 0.4 * abs(ref), 1e-12)
    inside = (np.abs(ys - ref) < band) & (t >= t_act)
    run = min(SETTLE_RUN, max(5, n // 4))
    ok = np.convolve(inside.astype(float), np.ones(run), mode="valid") >= run - 0.5
    if not ok.any():
        return None
    return t[int(np.argmax(ok))]


def _block_metrics(t, y, s):
    """Trend and excursion ratios of the suffix [s, end] for one trace.

    Returns (dn, dr, hn, hr):
      dn, dr : Theil-Sen trend across block medians, normalized to the
               detrended block scatter (significance) and to the window
               level (importance);
      hn, hr : range of the detrended block medians, normalized to the
               median in-block scatter (significance) and to the window
               level (importance) -- catches humps, intermittent bursts,
               and dead stretches that a monotonic trend fit cancels out.
    """
    m = t >= s
    tw, yw = t[m], y[m]
    n = len(yw)
    if n < 40:
        return np.inf, np.inf, np.inf, np.inf
    nb = int(np.clip(n // 60, 4, 10))
    edges = np.linspace(0, n, nb + 1).astype(int)
    bm = np.array([np.median(yw[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    bc = np.array([tw[(a + b) // 2] for a, b in zip(edges[:-1], edges[1:])])
    slopes = [(bm[j] - bm[i]) / (bc[j] - bc[i])
              for i in range(nb) for j in range(i + 1, nb)]
    slope = np.median(slopes)
    delta = slope * (tw[-1] - tw[0])
    resid = bm - slope * (bc - bc.mean())
    spread = _mad(resid)
    med = np.median(yw)
    level = max(abs(med), _mad(yw - med), 1e-12)
    dn = abs(delta) / max(spread, 1e-12)
    dr = abs(delta) / level
    rr = resid.max() - resid.min()
    wb = np.median([_mad(yw[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    hn = rr / max(wb, 1e-12)
    hr = rr / level
    return dn, dr, hn, hr


def _passes(dn, dr, hn, hr):
    trend_ok = dn < DN_MAX or dr < DR_MAX
    hump_ok = not (hn > HUMP_SIG and hr > HUMP_IMP)
    return trend_ok and hump_ok


def select_start(t, qi, qe, ge):
    """Pick the averaging start time for one simulation.

    Returns a dict with keys: t_start, flag, npass, settle_times,
    transient_end, window_length, means, metrics.
    """
    t = np.asarray(t, float)
    traces = {"qi": np.asarray(qi, float),
              "qe": np.asarray(qe, float),
              "ge": np.asarray(ge, float)}
    # discard the unphysical start-up spike region entirely
    keep = t >= t[0] + EARLY_SKIP
    if keep.sum() > 50:
        t = t[keep]
        traces = {k: v[keep] for k, v in traces.items()}
    T, t0 = t[-1], t[0]
    span = T - t0
    minwin = min_window(span)
    dt = np.median(np.diff(t))
    w = max(3, int(round(SMOOTH_W / dt)))
    smooth = {k: _smooth(v, w) for k, v in traces.items()}

    # --- transient / settle detection ---------------------------------
    actives = {k: _activity_time(t, smooth[k]) for k in traces}
    settles = {k: _settle_time(t, traces[k], smooth[k], actives[k])
               for k in traces}
    considered = {k: s for k, s in settles.items()
                  if s is not None and s < t0 + 0.75 * span}
    s_min = max(considered.values()) if considered else t0 + 0.5 * span
    # effective end of transient activity for failure flagging: traces that
    # never settle contribute their activity-onset time instead
    eff = [s if s is not None else max(actives[k], t0)
           for k, s in settles.items()]
    transient_end = max(max(eff), s_min)

    # --- candidate scan ------------------------------------------------
    cands = np.arange(s_min, T - minwin + 1e-9, CAND_STEP)
    npass_c = []
    for s in cands:
        mets = [_block_metrics(t, traces[k], s) for k in traces]
        npass_c.append(sum(_passes(*mm) for mm in mets))
    npass_c = np.array(npass_c, int)

    def earliest_sustained(target):
        for i in range(len(cands)):
            j = min(i + SUSTAIN, len(cands))
            if np.all(npass_c[i:j] >= target):
                return i
        return None

    t_start, flag, npass = None, None, 0
    i3 = earliest_sustained(3)
    i2 = earliest_sustained(2)
    if i3 is not None and (i2 is None or cands[i3] <= cands[i2] + 0.25 * span):
        # prefer the all-3 window unless the 2/3 window is much longer
        t_start, flag, npass = cands[i3], "ok", 3
    elif i2 is not None:
        t_start, flag, npass = cands[i2], "ok2", 2

    # --- fallback ------------------------------------------------------
    if t_start is None:
        t_start = t0 + 0.5 * span
        flag = "failure" if transient_end > t_start else "fallback"
        mets = [_block_metrics(t, traces[k], t_start) for k in traces]
        npass = sum(_passes(*mm) for mm in mets)

    window = T - t_start
    if window < minwin and flag in ("ok", "ok2"):
        flag = "questionable"

    # tiny-flux runs are essentially noise and do not really matter;
    # label them below_threshold regardless of stationarity outcome.
    # Judge the level from the chosen window (and the last quarter, so a
    # late turbulence onset with real flux is never mistaken for noise).
    wm = t >= t_start
    lq = t >= t0 + 0.75 * span
    qi_lvl = max(np.median(np.abs(traces["qi"][wm])),
                 np.median(np.abs(traces["qi"][lq])))
    qe_lvl = max(np.median(np.abs(traces["qe"][wm])),
                 np.median(np.abs(traces["qe"][lq])))
    if qi_lvl < NOISE_LEVEL and qe_lvl < NOISE_LEVEL:
        flag = "below_threshold"

    m = t >= t_start
    means = {k: float(np.mean(v[m])) for k, v in traces.items()}
    metrics = {k: _block_metrics(t, traces[k], t_start) for k in traces}
    return {"t_start": float(t_start), "flag": flag, "npass": int(npass),
            "settle_times": settles, "activity_times": actives,
            "transient_end": float(transient_end),
            "window_length": float(window), "means": means,
            "metrics": metrics}


# ======================================================================
# Extraction, statistics, MF inputs, batch collection (collect_results.py)

# ---------------------------------------------------------------- VENDORED (end)

SOURCE = "N.T. Howard, cgyro_window.py (2026-09-17)"
FLAGS_OK = ("ok", "ok2")
FLAGS_FALLBACK = ("fallback", "failure")


def thresholds():
    """Module-level tuning constants, for provenance/summary printing."""
    return {
        "SMOOTH_W": SMOOTH_W, "SETTLE_RUN": SETTLE_RUN, "CAND_STEP": CAND_STEP,
        "SUSTAIN": SUSTAIN, "EARLY_SKIP": EARLY_SKIP, "DN_MAX": DN_MAX, "DR_MAX": DR_MAX,
        "HUMP_SIG": HUMP_SIG, "HUMP_IMP": HUMP_IMP, "ACTIVITY_FRAC": ACTIVITY_FRAC,
        "NOISE_LEVEL": NOISE_LEVEL,
    }
