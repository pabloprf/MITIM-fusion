"""DIII-D overview plotting (layout, scaling, colors, overlay).

All the "how to draw it" lives here so the analysis scripts only declare
*what* to plot. Data model — two small dataclasses:

    Trace(spec, label, scale, avg, reduce)      one line in a panel
    Panel(ylabel, traces, units)                one subplot, 1+ traces

`spec` is usually one signal; pass a list of specs + `reduce` ('mean'|'sum'|
'diff') for a derived trace (e.g. delta = mean(tritop, tribot)).

`overview(shots, layout, ...)` fetches (via retrieval.py) and plots. `layout`
is either:
    * a flat list of Panels        -> auto 3-column grid (row-major), and
                                       empty panels are dropped
    * a list of columns            -> explicit column layout (each inner list
      ([col1_panels], [col2...])      is one column, top-to-bottom; positions
                                       are preserved, no-data panels show empty)

One shot -> color per trace (legend = trace labels). Many shots -> color per
shot, linestyle per trace (figure legend = shots). Large traces are resampled
server-side and cached on disk by the fetcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.interpolate import RegularGridInterpolator

from mitim_tools.experiment_tools.diiid.retrieval import DIIIDConnection, DIIIDFetcher

# Default display window [ms]; pass t_window=None to auto-detect from Ip.
DEFAULT_TWINDOW = (1300.0, 5000.0)

_NCOL = 3
_TRACE_LS = ["-", "--", ":", "-."]


# =============================================================================
# Declarative data model
# =============================================================================

@dataclass
class Trace:
    """One line in a panel. `scale`/`avg` are display-only (client-side).

    `spec` may be a list of specs combined by `reduce` ('mean'|'sum'|'diff'),
    e.g. Trace(["tritop", "tribot"], reduce="mean") for average triangularity.

    `avg` (>0) is a moving time-average window in **ms**: the raw trace is drawn
    faint in the background and the time-average is drawn bold on top (handy for
    beam-modulated NBI power / torque). `raw_alpha` sets the opacity of that faint
    raw trace (0 hides it entirely, leaving only the average).
    """
    spec:   str | list
    label:  str = ""
    scale:  float | list = 1.0          # scalar, or per-spec list (applied before reduce)
    avg:    float = 0.0
    raw_alpha: float = 0.1              # opacity of the faint raw trace behind a time-average
    ls:     str | None = None           # explicit linestyle ('-','--',':'); else auto
    marker: str | None = None           # marker (e.g. 'o') drawn on the points
    reduce: str = "mean"                # 'mean'|'sum'|'diff'|'ratio' (= spec0 / sum(rest))


@dataclass
class Panel:
    """One subplot. `units` overrides the MDS units in the y-label if given;
    `ylim=(lo, hi)` fixes the y-range (otherwise autoscaled);
    `show_source=True` annotates the panel with the signal spec(s) it plots."""
    ylabel: str
    traces: list = field(default_factory=list)
    units:  str = ""
    ylim:   tuple | None = None
    show_source: bool = False


@dataclass
class Equilibrium:
    """Layout marker for a whole column showing EFIT flux surfaces at `time` [ms]
    (R,Z), overlaying every shot in its color. `levels` are the ψ_N contours.
    `time=None` -> use the middle of the overview `shade` window (else 4000 ms)."""
    time:        float | None = None
    tree:        str = "EFIT01"
    levels:      tuple = tuple(round(v, 3) for v in np.arange(0.03, 1.0, 0.03))  # interior psi_N
    nscrape:     int = 4           # number of SOL flux surfaces to draw outside the LCFS
    deltascrape: float = 0.01      # SOL flux-surface spacing [m] at the outboard midplane
    label:       str = ""


@dataclass
class ProfilePanel:
    """One radial-profile sub-panel inside a Profiles column."""
    source:   str = "thomson"      # 'thomson' | 'cer'
    quantity: str = "te"           # te|ne (thomson) ; tit|rotct (cer)
    system:   str = "all"          # thomson view(s): core|tangential|divertor|list|'all'
    scale:    float = 1.0          # raw-unit -> display (e.g. 1e-3 eV->keV)
    units:    str = ""
    ylabel:   str = ""
    ylim:     tuple | None = None
    channels: object = None        # CER channel range (default range(1,49))


@dataclass
class Profiles:
    """Layout marker for a column of radial profiles (value vs ρ) at the analysis
    window, one ProfilePanel per row, overlaying every shot. Points are
    time-averaged over the overview `shade` window (or `time`±`window` if none),
    mapped to `coord` ('rho'|'rhopol'|'R'|'Z') through each shot's EFIT."""
    panels:    list                # list of ProfilePanel
    tree:      str = "EFIT01"
    coord:     str = "rho"
    time:      float = 4000.0      # used (with window) only if no shade window is given
    window:    float = 100.0
    errorbars: bool = True


# =============================================================================
# Fetch + plot driver
# =============================================================================

def _is_columnar(layout):
    return bool(layout) and isinstance(layout[0], (list, tuple, Equilibrium, Profiles))


def _flatten(layout):
    if not _is_columnar(layout):
        return list(layout)
    return [p for col in layout if not isinstance(col, (Equilibrium, Profiles)) for p in col]


def _all_specs(panels):
    specs = set()
    for p in panels:
        for tr in p.traces:
            for s in (tr.spec if isinstance(tr.spec, list) else [tr.spec]):
                specs.add(s)
    return sorted(specs)


def overview(shots, layout, name: str = "overview",
             t_window: tuple | None = DEFAULT_TWINDOW, max_points: int = 4000,
             use_cache: bool = True, cache_dir: str | Path | None = None,
             tunnel_host: str = "cybele", server: str | None = None,
             connection=None, colors: list | None = None,
             labels: list | None = None, shade: tuple | list | None = None,
             vlines: list | None = None, fig=None,
             save_dir: str | Path | None = None, show: bool = True):
    """Fetch the layout's signals for each shot (one connection) and plot them.

    `shade` shades a time window (or list of windows) on every time-trace panel,
    e.g. shade=(3800, 4100). `vlines` is a list parallel to `shots` of event
    times [ms] (or None) drawn as a dashed vertical line in that shot's color,
    e.g. vlines=[None, 2200, 1800]. Equilibrium columns (see Equilibrium) fetch
    their own EFIT slice per shot and overlay all shots in R,Z.

    Pass `fig` to draw into an existing figure (e.g. a `GUItools.FigureNotebook`
    tab) instead of creating one; pass `connection` (a DIIIDConnection) to reuse
    ONE tunnel across several overview()/profiles() calls (polite to the server).
    """
    if isinstance(shots, int):
        shots = [shots]
    specs = _all_specs(_flatten(layout))
    cols = list(layout) if _is_columnar(layout) else []
    eq_cols = [c for c in cols if isinstance(c, Equilibrium)]
    prof_cols = [(ci, c) for ci, c in enumerate(cols) if isinstance(c, Profiles)]
    # profile time/window: the shade window if given, else each Profiles' own time±window
    shade_win = (shade if (shade and isinstance(shade[0], (int, float)))
                 else (list(shade)[0] if shade else None))
    shade_center = 0.5 * (shade_win[0] + shade_win[1]) if shade_win else None

    def _fetch_eq(fetcher, shot, etime, tree):     # fetch once per (shot,time,tree)
        key = (shot, etime, tree)
        if key not in eq_data:
            try:
                eq_data[key] = fetcher.fetch_equilibrium(etime, tree)
            except Exception as excp:
                print(f"  ! equilibrium {tree}@{etime:.0f}ms #{shot}: {str(excp)[:45]}")
                eq_data[key] = None

    results, eq_data, prof_data = {}, {}, {}
    own_conn = connection is None
    conn = connection if connection is not None else \
        DIIIDConnection(server=server, tunnel_host=tunnel_host)
    try:
        for shot in shots:
            print(f"* fetching #{shot} ...")
            fetcher = DIIIDFetcher(shot, connection=conn, max_points=max_points,
                                   use_cache=use_cache, cache_dir=cache_dir)
            res = {}
            for sp in specs:
                try:
                    res[sp] = fetcher.fetch_signal(sp, name=sp)
                except Exception as excp:
                    print(f"  ! {sp} unavailable for #{shot}: {str(excp)[:60]}")
                    res[sp] = None
            results[shot] = res
            for eq in eq_cols:                        # None -> middle of the shade window
                etime = eq.time if eq.time is not None else (shade_center or 4000.0)
                _fetch_eq(fetcher, shot, etime, eq.tree)
            for ci, pc in prof_cols:                  # radial-profile columns
                twin = shade_win
                ptime = shade_center if shade_center is not None else pc.time
                _fetch_eq(fetcher, shot, ptime, pc.tree)
                for pi, pp in enumerate(pc.panels):
                    try:
                        if pp.source == "thomson":
                            prof_data[(ci, shot, pi)] = fetcher.fetch_thomson_profile(
                                ptime, pp.quantity, pp.system, window=pc.window, t_window=twin)
                        else:
                            prof_data[(ci, shot, pi)] = fetcher.fetch_cer_profile(
                                ptime, pp.quantity, channels=pp.channels or range(1, 49),
                                window=pc.window, t_window=twin)
                    except Exception as excp:
                        print(f"  ! profile {pp.source}.{pp.quantity} #{shot}: {str(excp)[:45]}")
                        prof_data[(ci, shot, pi)] = None
    finally:
        if own_conn:
            conn.close()

    if _is_columnar(layout):
        columns = [col if isinstance(col, (Equilibrium, Profiles)) else list(col) for col in layout]
    else:                               # flat: drop empty panels, fill 3-col grid
        def has_data(p):
            return any(_trace_xy(results[sh], tr) is not None
                       for sh in shots for tr in p.traces)
        kept = [p for p in layout if has_data(p)]
        ncol = _NCOL
        columns = [[kept[i] for i in range(c, len(kept), ncol)] for c in range(ncol)]

    _render(results, columns, shots, name=name, t_window=t_window, colors=colors,
            labels=labels, shade=shade, vlines=vlines, eq_data=eq_data,
            prof_data=prof_data, fig=fig, save_dir=save_dir, show=show)
    return results


def profiles(shots, time: float = 4000.0, source: str = "cer",
             quantity: str = "tit", system: str = "core", tree: str = "EFIT01",
             channels=range(1, 49), window: float = 100.0, scale: float = 1.0,
             units: str = "", ylabel: str = "", coord: str = "auto",
             ylim: tuple | None = None, label_channels: bool = True,
             colors: list | None = None, labels: list | None = None,
             name: str | None = None, use_cache: bool = True,
             cache_dir: str | Path | None = None,
             tunnel_host: str = "cybele", server: str | None = None,
             save_dir: str | Path | None = None, show: bool = True):
    """Diagnostic channel profile vs its coordinate, plus the channel (R,Z) on the EFIT.

    `source` selects the diagnostic:
      * "cer"     : CER chords; `quantity`='tit'(Ti)/'rotct'(rotation); profile vs R.
      * "thomson" : Thomson; `quantity`='te'(Te)/'ne'(density), `system`=
                    core|tangential|divertor; profile vs Z (core) or R (tangential).

    Left panel  : value vs coordinate per shot (points+line, channels labeled),
                  axis/separatrix marked. Right panel: the equilibrium with every
                  channel's (R,Z) marked and labeled. `coord` forces the x-axis
                  ('R'|'Z'); 'auto' picks whichever spans more.
    `coord` may also be 'rho'/'rhotor' (ρ_tor = sqrt(norm. toroidal flux), the
    transport ρ) or 'rhopol' (sqrt ψ_N): each shot's channels are mapped through
    ITS OWN equilibrium via `EquilibriumData.rho_of`, so a profile comparison is
    on a common flux coordinate. `scale` converts raw units (e.g. 1e-3 eV->keV,
    1e-20 m^-3 -> 1e20); `units`/`ylabel` label the y-axis (defaulted when blank).
    """
    if isinstance(shots, int):
        shots = [shots]

    def fetch_prof(f):
        if source == "thomson":
            return f.fetch_thomson_profile(time, quantity=quantity, system=system, window=window)
        return f.fetch_cer_profile(time, quantity=quantity, channels=channels, window=window)

    profs, eq_data = {}, {}
    with DIIIDConnection(server=server, tunnel_host=tunnel_host) as conn:
        for shot in shots:
            print(f"* fetching {source} {quantity} profile #{shot} ...")
            f = DIIIDFetcher(shot, connection=conn, use_cache=use_cache, cache_dir=cache_dir)
            profs[shot] = fetch_prof(f)
            try:
                eq_data[(shot, time, tree)] = f.fetch_equilibrium(time, tree)
            except Exception as e:
                print(f"  ! equilibrium unavailable #{shot}: {str(e)[:50]}")
                eq_data[(shot, time, tree)] = None

    shot_colors = list(colors) if colors else [plt.cm.tab10(i) for i in range(10)]
    shot_names = [f"{sh} ({labels[si]})" if labels and si < len(labels) and labels[si]
                  else str(sh) for si, sh in enumerate(shots)]
    ref = next((profs[sh] for sh in shots if profs[sh].r.size), None)
    ref_eq = next((eq_data[(sh, time, tree)] for sh in shots
                   if eq_data.get((sh, time, tree)) is not None), None)
    title = ref.label if ref is not None else f"{source} {quantity}"
    ylabel = ylabel or title
    if coord == "auto":                          # x-axis = coordinate that spans more
        coord = "Z" if (ref is not None and np.ptp(ref.z) > np.ptp(ref.r)) else "R"
    rho_kind = {"rho": "tor", "rhotor": "tor", "rhopol": "pol"}.get(coord)
    ref_shot = next((sh for sh in shots if profs[sh].r.size), None)

    def xcoord(sh, pr):                           # per-shot x: raw R/Z, or ρ from that shot's EFIT
        if rho_kind:
            ed = eq_data.get((sh, time, tree))
            return (ed.rho_of(pr.r, pr.z, kind=rho_kind) if ed is not None
                    else np.full(pr.r.size, np.nan))
        return pr.z if coord == "Z" else pr.r
    xlabel = {"tor": r"$\rho_{tor}$", "pol": r"$\rho_{pol}$"}.get(rho_kind, f"{coord} [m]")

    def tags_of(pr):                              # per-channel labels ('C5'/'T3'), else channel #
        return pr.tag if pr.tag is not None else np.array([str(c) for c in pr.channel])
    pref = lambda t: "".join(ch for ch in str(t) if ch.isalpha())   # TS view initial, '' for CER
    subsys = sorted(set(pref(t) for t in tags_of(ref))) if ref is not None else [""]
    MARK = {"": "o", "C": "o", "T": "s", "D": "^"}

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    axp, axe = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- left: value vs coordinate, channels labeled ----
    marks = ([(0.0, "axis"), (1.0, "sep")] if rho_kind else
             [(ref_eq.raxis, "axis"), (ref_eq.rbbbs.max(), "sep")]
             if (ref_eq is not None and coord == "R") else [])
    for xr, tag in marks:
        axp.axvline(xr, color="0.6", ls=":", lw=0.8)
        axp.text(xr, 0.98, tag, color="0.5", fontsize=6, rotation=90,
                 va="top", ha="right", transform=axp.get_xaxis_transform())
    for si, sh in enumerate(shots):
        pr = profs[sh]
        if not pr.r.size:
            continue
        x = xcoord(sh, pr); o = np.argsort(x); color = shot_colors[si % len(shot_colors)]
        axp.plot(x[o], pr.value[o] * scale, "-", color=color, lw=0.8, label=shot_names[si])
        pres = np.array([pref(t) for t in tags_of(pr)])
        for p in subsys:                          # markers per TS view (single set for CER)
            mm = pres == p
            if mm.any():
                axp.plot(x[mm], pr.value[mm] * scale, MARK.get(p, "o"), color=color, ms=4, lw=0)
    if ref is not None and label_channels:
        for t, x, v in zip(tags_of(ref), xcoord(ref_shot, ref), ref.value):
            axp.annotate(str(t), (x, v * scale), fontsize=5, color="0.4",
                         xytext=(0, 5), textcoords="offset points", ha="center")
    if ylim:
        axp.set_ylim(ylim)
    axp.set_xlabel(xlabel, fontsize=8)
    axp.set_ylabel(ylabel + (f"  [{units}]" if units else ""), fontsize=9)
    axp.set_title(f"{title} vs {xlabel}   t={time:.0f} ms", fontsize=9)
    axp.grid(alpha=0.3); axp.tick_params(labelsize=7)
    leg = axp.legend(fontsize=7, frameon=False, loc="upper right")
    if len([p for p in subsys if p]) > 1:         # 2nd legend: which marker = which TS view
        names = {"C": "core", "T": "tangential", "D": "divertor"}
        mh = [Line2D([], [], color="0.3", marker=MARK.get(p, "o"), ls="", ms=4,
                     label=names.get(p, p)) for p in subsys]
        axp.add_artist(leg)
        axp.legend(handles=mh, fontsize=6, frameon=False, loc="lower left")

    # ---- right: equilibrium + channel positions (labeled, staggered) ----
    _draw_equilibrium(axe, eq_data, shots, Equilibrium(time=time, tree=tree), shot_colors)
    if ref is not None:
        axe.plot(ref.r, ref.z, "o", ms=3, color="k", zorder=6)
        if label_channels:
            for k, (n, r, z) in enumerate(zip(tags_of(ref), ref.r, ref.z)):
                alt = k % 2 == 0
                if coord == "Z":                 # vertical chord -> stagger left/right
                    off, ha, va = (9 if alt else -9, 0), ("left" if alt else "right"), "center"
                else:                            # horizontal chord -> stagger up/down
                    off, ha, va = (0, 9 if alt else -9), "center", ("bottom" if alt else "top")
                axe.annotate(str(n), (r, z), fontsize=4.5, color="0.15", zorder=7,
                             xytext=off, textcoords="offset points", ha=ha, va=va,
                             arrowprops=dict(arrowstyle="-", color="0.7", lw=0.3))

    fig.suptitle(f"DIII-D — {title} profile  (#{', #'.join(map(str, shots))})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_dir is not None:                  # None -> just show, don't write a file
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{name or f'{source}_{quantity}_profile'}_{'_'.join(map(str, shots))}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"* Saved figure -> {out}")
    if show:
        plt.show()
    plt.close(fig)
    return profs


# =============================================================================
# Plotting
# =============================================================================

def _trace_xy(shot_sigs, trace: Trace):
    """(t, y) for a trace: per-spec scale, then combine (reduce). None if absent.

    `scale` may be a scalar (applied after the reduce) or a per-spec list
    (applied to each signal before the reduce) — the latter lets you sum
    quantities that need different unit conversions, e.g. NBI [kW] + OH [W].
    """
    specs = trace.spec if isinstance(trace.spec, list) else [trace.spec]
    perspec = isinstance(trace.scale, (list, tuple))
    pairs = []                               # (signal, per-spec scale)
    for i, sp in enumerate(specs):
        s = shot_sigs.get(sp)
        if s is not None and len(s.time) > 1:
            pairs.append((s, trace.scale[i] if perspec else 1.0))
    if not pairs:
        return None
    t = np.asarray(pairs[0][0].time, float)
    ys = []
    for s, sc in pairs:                      # put each on the first time base, then scale
        yi = np.asarray(s.data, float)
        if s is not pairs[0][0]:
            yi = np.interp(t, np.asarray(s.time, float), yi)
        ys.append(yi * sc)
    Y = np.vstack(ys)
    if trace.reduce == "sum":
        y = Y.sum(0)
    elif trace.reduce == "diff":
        y = Y[0] - Y[1]
    elif trace.reduce == "ratio":            # spec0 / sum(rest), e.g. Prad/Pin
        denom = Y[1:].sum(0)
        y = Y[0] / np.where(denom == 0, np.nan, denom)
    else:
        y = Y.mean(0)
    return t, y if perspec else y * trace.scale


def _moving_avg(t, y, window_ms):
    """Edge-corrected moving average of y over a `window_ms` time window."""
    if window_ms <= 0 or t.size < 3:
        return y
    dt = float(np.median(np.diff(t)))
    k = int(round(window_ms / dt)) if dt > 0 else 0
    if k < 2 or y.size <= k:
        return y
    kern = np.ones(k) / k
    norm = np.convolve(np.ones_like(y), kern, mode="same")   # fractional count at edges
    return np.convolve(y, kern, mode="same") / norm


def _trace_units(shot_sigs, trace: Trace):
    s0 = (trace.spec[0] if isinstance(trace.spec, list) else trace.spec)
    sig = shot_sigs.get(s0)
    return sig.units if sig is not None else ""


def _decimate(t, y, n_max=4000):
    step = max(1, len(t) // n_max)
    return t[::step], y[::step]


def _trace_ls(tr, ti, multishot):
    """Linestyle for a trace: explicit tr.ls wins; else auto (per-trace when
    overlaying shots, solid for a single shot)."""
    if tr.ls:
        return tr.ls
    return _TRACE_LS[ti % len(_TRACE_LS)] if multishot else "-"


def _spec_label(tr):
    """Readable provenance for a trace's spec(s), reflecting the `reduce` op:
    sum -> 'a + b', diff -> 'a - b', ratio -> 'a / (b + c)', mean -> 'mean(a, b)'."""
    if isinstance(tr.spec, str):
        return tr.spec
    s = tr.spec
    if tr.reduce == "sum":
        return " + ".join(s)
    if tr.reduce == "diff":
        return " - ".join(s)
    if tr.reduce == "ratio":                  # spec0 / sum(rest)
        denom = s[1] if len(s) == 2 else "(" + " + ".join(s[1:]) + ")"
        return f"{s[0]} / {denom}"
    return "mean(" + ", ".join(s) + ")"       # 'mean'


def _discharge_window(ip_sigs):
    """[t0,t1] ms from where |Ip| > 5% of peak (ignoring no-plasma shots)."""
    t0s, t1s = [], []
    for s in ip_sigs:
        if s is None or len(s.time) < 2:
            continue
        a = np.abs(s.data)
        m = a > 0.05 * np.nanmax(a)
        if m.any() and m.mean() < 0.8:
            t0s.append(max(0.0, float(s.time[m].min())))
            t1s.append(float(s.time[m].max()))
    return (min(t0s), max(t1s) * 1.02) if t1s else (0.0, 6000.0)


def _draw_panel(ax, results, panel: Panel, shots, t0, t1, multishot,
                shot_colors, shot_names):
    """Draw one panel; returns True if anything was plotted."""
    units = panel.units
    multitrace = len(panel.traces) > 1
    drew = False
    for si, shot in enumerate(shots):
        for ti, tr in enumerate(panel.traces):
            xy = _trace_xy(results[shot], tr)
            if xy is None:
                continue
            units = units or _trace_units(results[shot], tr)
            t, y = xy
            m = (t >= t0) & (t <= t1)
            t, y = t[m], y[m]
            if t.size == 0:
                continue
            ls = _trace_ls(tr, ti, multishot)
            if multishot:
                color = shot_colors[si % len(shot_colors)]
                label = shot_names[si] if ti == 0 else None
            else:
                color = plt.cm.tab10(ti % 10)
                label = tr.label or (tr.spec if isinstance(tr.spec, str) else "")
            if tr.avg and tr.avg > 0:                 # raw faint, time-average bold
                td, yd = _decimate(t, y)
                if tr.raw_alpha > 0:
                    ax.plot(td, yd, color=color, ls=ls, lw=0.7, alpha=tr.raw_alpha)
                ta, ya = _decimate(t, _moving_avg(t, y, tr.avg))
                ax.plot(ta, ya, color=color, ls=ls, lw=1.4, label=label,
                        marker=tr.marker, ms=3)
            else:
                td, yd = _decimate(t, y)
                ax.plot(td, yd, color=color, ls=ls, lw=0.9, label=label,
                        marker=tr.marker, ms=3)
            drew = True

    ax.set_ylabel(panel.ylabel + (f"\n[{units}]" if units else ""), fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.3)
    if panel.ylim is not None:
        ax.set_ylim(panel.ylim)
    if not drew:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="0.6")
    elif multitrace and not multishot:
        ax.legend(fontsize=5.5, loc="best", ncol=2, frameon=False)
    elif multitrace and multishot:
        handles = [Line2D([], [], color="0.3", ls=_trace_ls(tr, ti, True),
                          lw=1, label=(tr.label or ""))
                   for ti, tr in enumerate(panel.traces)]
        ax.legend(handles=handles, fontsize=5, loc="best", ncol=2, frameon=False)
    if panel.show_source:                      # expose the actual MDS spec(s) + operation
        specs = [_spec_label(tr) for tr in panel.traces]
        ax.text(0.99, 0.96, ", ".join(dict.fromkeys(specs)), transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5, color="0.45",
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
    return drew


def _separatrix_legs(ax, ed, color):
    """Draw the divertor separatrix legs: contour psi_N = 1 on a sub-grid spanning
    the A-file strike points to ~10 cm past the active X-point, and mark the active
    X-point. No-op for a limited plasma."""
    lower = (1 < ed.rxpt1 < 2) and ed.zvsin < 0 and ed.zvsout < 0
    upper = (1 < ed.rxpt2 < 2) and ed.zvsin > 0 and ed.zvsout > 0
    if lower:
        zxpt, (z0, z1) = ed.zxpt1, (float(ed.zgrid.min()), ed.zxpt1 + 0.1)
        rxpt = ed.rxpt1
    elif upper:
        zxpt, (z0, z1) = ed.zxpt2, (ed.zxpt2 - 0.1, float(ed.zgrid.max()))
        rxpt = ed.rxpt2
    else:
        return
    interp = RegularGridInterpolator((ed.zgrid, ed.rgrid), ed.psiN,
                                     bounds_error=False, fill_value=np.nan)
    rspln = np.linspace(ed.rvsin, ed.rvsout, 200)
    zspln = np.linspace(z0, z1, 200)
    RR, ZZ = np.meshgrid(rspln, zspln)
    sub = interp(np.stack([ZZ.ravel(), RR.ravel()], 1)).reshape(ZZ.shape)
    ax.contour(rspln, zspln, sub, levels=[1.0], colors=[color], linewidths=1.3)
    ax.plot(rxpt, zxpt, "x", color=color, ms=5, mew=1.3)


def _draw_equilibrium(ax, eq_data, shots, eq, shot_colors, time=None):
    """R,Z flux-surface panel from a GEQDSK: dashed interior flux surfaces (psi_N
    from axis to boundary), the bold LCFS, and the diverted separatrix legs to the
    strike points. Reference shot in grey; each shot's LCFS (+ legs) overlaid in
    its color. `time` overrides eq.time (e.g. the middle of the analysis window)."""
    t = time if time is not None else eq.time
    eds = [(si, (eq_data or {}).get((shots[si], t, eq.tree)))
           for si in range(len(shots))]
    eds = [(si, ed) for si, ed in eds if ed is not None]
    ax.set_title(f"{eq.label or f'EFIT {eq.tree}'}  t={t:.0f} ms", fontsize=8)
    if not eds:
        ax.text(0.5, 0.5, "(no equilibrium)", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="0.6")
        return

    ref = eds[0][1]
    interp = RegularGridInterpolator((ref.zgrid, ref.rgrid), ref.psiN,
                                     bounds_error=False, fill_value=np.nan)
    # clip flux surfaces to the vessel so the SOL contours don't sprawl outside it
    clip = PathPatch(MplPath(np.column_stack([ref.wall_r, ref.wall_z])),
                     transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    # interior flux surfaces (dashed), psi_N = levels
    ci = ax.contour(ref.rgrid, ref.zgrid, ref.psiN, levels=list(eq.levels),
                    colors="0.55", linewidths=0.35, linestyles="dashed")
    ci.set_clip_path(clip)
    # SOL surfaces: sample psi at points stepped outward in R from the outboard-
    # midplane boundary point by deltascrape (even real-space spacing -> no inboard
    # bunching), then contour those psi levels.
    imax = int(np.argmax(ref.rbbbs))
    r0, z0 = float(ref.rbbbs[imax]), float(ref.zbbbs[imax])
    rpts = r0 + np.arange(1, eq.nscrape + 1) * eq.deltascrape
    sol = interp(np.column_stack([np.full(rpts.size, z0), rpts]))
    sol = np.sort(sol[np.isfinite(sol)])
    if sol.size:
        cs = ax.contour(ref.rgrid, ref.zgrid, ref.psiN, levels=list(sol),
                        colors="0.6", linewidths=0.3)
        cs.set_clip_path(clip)
    ax.plot(ref.wall_r, ref.wall_z, "-", color="0.15", lw=0.9)     # vessel
    for si, ed in eds:                                            # per-shot LCFS + legs + axis
        color = shot_colors[si % len(shot_colors)]
        ax.plot(ed.rbbbs, ed.zbbbs, color=color, lw=1.3)
        _separatrix_legs(ax, ed, color)
        ax.plot(ed.raxis, ed.zaxis, "+", color=color, ms=7)

    ax.set_xlim(ref.wall_r.min() - 0.04, ref.wall_r.max() + 0.04)
    ax.set_ylim(ref.wall_z.min() - 0.04, ref.wall_z.max() + 0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]", fontsize=7)
    ax.set_ylabel("Z [m]", fontsize=8)
    ax.tick_params(labelsize=6.5)


def _draw_profile_panel(ax, prof_data, eq_data, shots, shot_colors, pc, c, pi, pp, time, window, last):
    """One radial-profile sub-panel (value vs ρ) in a Profiles column: every shot
    overlaid with error bars, points time-averaged over `window` [ms]."""
    coord = pc.coord
    rho_kind = {"rho": "tor", "rhotor": "tor", "rhopol": "pol"}.get(coord)
    drew = False
    for si, sh in enumerate(shots):
        prof = (prof_data or {}).get((c, sh, pi))
        if prof is None or not prof.r.size:
            continue
        ed = eq_data.get((sh, time, pc.tree))
        if rho_kind:
            x = ed.rho_of(prof.r, prof.z, rho_kind) if ed is not None else np.full(prof.r.size, np.nan)
        else:
            x = prof.z if coord == "Z" else prof.r
        o = np.argsort(x); color = shot_colors[si % len(shot_colors)]
        yerr = prof.error[o] * pp.scale if (pc.errorbars and prof.error is not None) else None
        ax.errorbar(x[o], prof.value[o] * pp.scale, yerr=yerr, fmt="-o", ms=2.5, lw=0.6,
                    color=color, elinewidth=0.5, capsize=0, alpha=0.85)
        drew = True
    if rho_kind:
        ax.axvline(0.0, color="0.8", ls=":", lw=0.6)         # magnetic axis
        ax.axvline(1.0, color="0.4", ls="--", lw=0.9)        # separatrix (ρ=1)
        ax.set_xlim(left=0.0)
    ax.set_ylabel((pp.ylabel or pp.quantity) + (f"\n[{pp.units}]" if pp.units else ""), fontsize=8)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=6.5)
    if pp.ylim is not None:
        ax.set_ylim(pp.ylim)
    if not drew:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="0.6")
    if pi == 0:                                   # top of the column: show the averaging window
        ax.set_title(f"profiles  (avg {window[0]:.0f}-{window[1]:.0f} ms)" if window
                     else f"profiles  t={time:.0f} ms", fontsize=8)
    if last:
        ax.set_xlabel({"tor": r"$\rho_{tor}$", "pol": r"$\rho_{pol}$"}.get(rho_kind, f"{coord} [m]"),
                      fontsize=8)
    else:
        ax.tick_params(labelbottom=False)


def _render(results, columns, shots, name="overview",
            t_window: tuple | None = DEFAULT_TWINDOW, colors: list | None = None,
            labels: list | None = None, shade: tuple | list | None = None,
            vlines: list | None = None, eq_data: dict | None = None,
            prof_data: dict | None = None, fig=None,
            save_dir: str | Path | None = None, show: bool = True):
    """Place `columns` (Panels, an Equilibrium, or a Profiles column) on a grid.
    `fig` (a Figure, e.g. a FigureNotebook tab) is drawn into if given; else a
    new figure is created and saved/shown per `save_dir`/`show`."""
    eq_data, prof_data = eq_data or {}, prof_data or {}
    shades = [] if not shade else \
        ([shade] if isinstance(shade[0], (int, float)) else list(shade))
    prof_window = shades[0] if shades else None
    multishot = len(shots) > 1
    ncol = len(columns)
    nrow = max((len(col.panels) if isinstance(col, Profiles) else len(col)
                for col in columns if not isinstance(col, Equilibrium)), default=1)
    t0, t1 = t_window if t_window is not None else \
        _discharge_window([results[sh].get("ip") for sh in shots])
    shot_colors = list(colors) if colors else [plt.cm.tab10(i) for i in range(10)]
    # legend name per shot: "<shot> (<label>)" when a label is given
    shot_names = [f"{sh} ({labels[si]})" if labels and si < len(labels) and labels[si]
                  else str(sh) for si, sh in enumerate(shots)]

    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=(4.6 * ncol, 1.4 * nrow))
    else:                                         # notebook tab: size to fill it, not a short
        fig.set_size_inches(max(12.0, 3.6 * ncol), max(8.5, 2.0 * nrow))  # strip at the top
    gs = fig.add_gridspec(nrow, ncol)
    xref = None
    shade_center = 0.5 * (prof_window[0] + prof_window[1]) if prof_window else None
    for c, col in enumerate(columns):
        if isinstance(col, Equilibrium):
            et = col.time if col.time is not None else (shade_center if shade_center is not None else 4000.0)
            _draw_equilibrium(fig.add_subplot(gs[:, c]), eq_data, shots, col, shot_colors, time=et)
            continue
        if isinstance(col, Profiles):
            ptime = shade_center if shade_center is not None else col.time
            sub = gs[:, c].subgridspec(len(col.panels), 1, hspace=0.12)
            paxref = None
            for pi, pp in enumerate(col.panels):
                pax = fig.add_subplot(sub[pi, 0], sharex=paxref); paxref = paxref or pax
                _draw_profile_panel(pax, prof_data, eq_data, shots, shot_colors, col, c, pi, pp,
                                    ptime, prof_window, last=(pi == len(col.panels) - 1))
            continue
        for r in range(len(col)):
            ax = fig.add_subplot(gs[r, c], sharex=xref)
            xref = xref or ax
            _draw_panel(ax, results, col[r], shots, t0, t1, multishot,
                        shot_colors, shot_names)
            for s0, s1 in shades:
                ax.axvspan(s0, s1, color="gold", alpha=0.18, lw=0, zorder=0)
            for si, vt in enumerate(vlines or []):       # per-shot event markers
                if vt is not None:
                    ax.axvline(vt, color=shot_colors[si % len(shot_colors)],
                               ls="--", lw=1.0, alpha=0.8, zorder=1)
            if r < len(col) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time  [ms]", fontsize=7)
    if xref is not None:
        xref.set_xlim(t0, t1)

    # title on top, legend stacked just below it (centered, single row) so they
    # never collide regardless of how many shots/labels there are.
    title = f"DIII-D — {name}"
    height = fig.get_figheight()
    if multishot:
        handles = [Line2D([], [], color=shot_colors[si % len(shot_colors)], lw=2,
                          label=shot_names[si]) for si in range(len(shots))]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1 - 0.55 / height),
                   ncol=min(len(shots), 4), fontsize=8, frameon=False)
        top_in = 0.85
    else:
        title += f"  #{shots[0]}"
        top_in = 0.45
    fig.suptitle(title, fontsize=12, y=1 - 0.26 / height)

    fig.tight_layout(rect=(0, 0, 1, 1 - top_in / height), h_pad=0.15, w_pad=0.4)
    fig.subplots_adjust(hspace=0.12)      # tight vertical packing (panels share x)
    if own_fig:                           # a provided fig (notebook tab) is shown/saved by the caller
        if save_dir is not None:
            save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{name}_{'_'.join(str(s) for s in shots)}.png"
            fig.savefig(save_path, dpi=150)
            print(f"* Saved figure -> {save_path}")
        if show:
            plt.show()
    return fig
