"""DIII-D overview plotting (layout, scaling, colors, overlay).

All the "how to draw it" lives here so the analysis scripts only declare
*what* to plot. Data model — two small dataclasses:

    Trace(spec, label, scale, avg, reduce)      one line in a panel
    Panel(ylabel, traces)                        one subplot, 1+ traces

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
from IPython import embed

# Default display window [ms]; pass t_window=None to auto-detect from Ip.
DEFAULT_TWINDOW = (1300.0, 5000.0)

_NCOL = 3
_TRACE_LS = ["-", "--", ":", "-."]
# Colours for the several traces WITHIN one single-shot panel (e.g. core/edge in a
# Thomson/CER panel). A colour-blind-safe (Okabe-Ito) set whose first pair is
# maximally distinct (bluish-green vs reddish-purple). Kept off the matplotlib default
# cycle so it doesn't collide with the shot/window colours (which use that cycle).
_TRACE_PALETTE = ["#009E73", "#CC79A7", "#D55E00", "#000000",
                  "#F0E442", "#999999"]
#                  green     magenta    vermilion black  yellow   grey

# CER analysis "flavors" = the `system` prefix on the flat pointnames (cerqtit/
# ceratit/cerftit/cerntit). On a manually-analysed shot CERAUTO is often the carbon
# fit and CERFIT the neon fit.
_CER_FLAVORS = ("cerq", "cera", "cerf", "cern")
_CER_FLAVOR_LABEL = {"cerq": "CERQUICK", "cera": "CERAUTO", "cerf": "CERFIT", "cern": "CERNEUR"}
_PROF_MARKERS = ["o", "s", "^", "D", "v", "P"]   # markers to distinguish overlaid scatter series

# CER quantities for the profiles_cer "check" plot, keyed by a friendly name ->
# (CER suffix, scale, units, ylabel). The suffix is the <quantity> in the pointname
# <flavor><quantity><view><ch>: tit=Ti, rotct=toroidal rotation, nzt=impurity density
# n_Z, fzt=impurity fraction n_Z/n_e, ampt=line intensity.
_CER_QTY_DISPLAY = {
    "ti":  ("tit",   1e-3,  "keV",                 r"$T_i$"),
    "rot": ("rotct", 1.0,   "km/s",                r"$v_\phi$"),
    "nz":  ("nzt",   1e-18, r"$10^{18}$ m$^{-3}$", r"$n_Z$"),
    "fz":  ("fzt",   1.0,   "%",                   r"$n_Z/n_e$"),
    "amp": ("ampt",  1.0,   "ph/s/m²/sr",          "intensity"),
}
_CER_QTY_ALL = ("ti", "rot", "nz", "fz")          # default rows for profiles_cer


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
    abs:    bool = False                # plot |value| (e.g. B_T is stored < 0 at DIII-D)


@dataclass
class Panel:
    """One subplot. Put any units directly in `ylabel` (e.g. r"$I_p$ [MA]").
    `ylim=(lo, hi)` fixes the y-range (otherwise autoscaled);
    `show_source=True` annotates the panel with the signal spec(s) it plots."""
    ylabel: str
    traces: list = field(default_factory=list)
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
    system:   str = "all"          # thomson view(s) core|tangential|divertor|list|'all';
    #                              # for CER, the flavor: cerq(QUICK)|cera(AUTO)|cerf(FIT)
    scale:    float = 1.0          # raw-unit -> display (e.g. 1e-3 eV->keV)
    ylabel:   str = ""             # y-axis label; put any units here (e.g. r"$T_e$ [keV]")
    ylim:     tuple | None = None
    channels: object = None        # CER channel range (default range(1,49))
    join:     bool = False         # connect the points with a line (default: scatter only)
    flavor_labels: dict | None = None  # CER legend overrides, e.g. {"cera":"CERAUTO (Ne)"}
    alpha:    float = 0.85         # point/line opacity for this panel (e.g. 0.5 to de-emphasize)

    def __post_init__(self):       # fail fast on a misplaced/typo'd argument
        if self.source not in ("thomson", "cer"):
            raise ValueError(
                f"ProfilePanel source must be 'thomson' or 'cer' (got {self.source!r}). "
                "A CER flavor goes in `system`, e.g. ProfilePanel('cer', 'tit', system='cera').")
        if self.source == "cer":           # system = flavor, or a list of flavors to overlay
            flavs = self.system if isinstance(self.system, (list, tuple)) else [self.system]
            bad = [s for s in flavs if s not in _CER_FLAVORS + ("all",)]
            if bad:
                raise ValueError(
                    f"CER ProfilePanel `system` is the flavor — one of {_CER_FLAVORS} "
                    f"('all' = CERQUICK), or a list of them to overlay; got {bad}.")
        else:                      # thomson: a view, a list of views, or 'all'
            views = self.system if isinstance(self.system, (list, tuple)) else [self.system]
            bad = [v for v in views if v not in ("core", "tangential", "divertor", "all")]
            if bad:
                raise ValueError(
                    "Thomson ProfilePanel `system` must be core|tangential|divertor|'all' "
                    f"(or a list of those); got {bad}.")


@dataclass
class Profiles:
    """Layout marker for a column of radial profiles (value vs ρ) at the analysis
    window, one ProfilePanel per row, overlaying every shot. Points are
    time-averaged over the overview `shade` window (or `time`±`window` if none),
    mapped to `coord` ('rho'|'rhopol'|'R'|'Z') through each shot's EFIT.
    `average=False` instead plots EVERY time sample in the window (a scatter cloud
    per channel, no error bars) — handy to see the raw spread."""
    panels:    list                # list of ProfilePanel
    tree:      str = "EFIT01"
    coord:     str = "rho"
    time:      float = 4000.0      # used (with window) only if no shade window is given
    window:    float = 100.0
    errorbars: bool = True
    rho_max:   float | None = None  # drop points past this ρ (e.g. 1.0 to hide the noisy SOL)
    average:   bool = True          # False -> plot all time samples in the window (no averaging)


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


def _parse_windows(shade):
    """`shade` -> a list of (t0, t1) analysis windows. A single (t0,t1) or a list
    of them are both accepted; [] when no shade. Several windows => the Equilibrium
    and Profiles columns draw one snapshot per window (great for one shot in time)."""
    if not shade:
        return []
    if isinstance(shade[0], (int, float)):
        return [tuple(shade)]
    return [tuple(w) for w in shade]


def overview(shots, layout, name: str = "overview",
             t_window: tuple | None = DEFAULT_TWINDOW, max_points: int = 4000,
             use_cache: bool = True, cache_dir: str | Path | None = None,
             tunnel_host: str | None = None, server: str | None = None,
             connection=None, colors: list | None = None,
             labels: list | None = None, shade: tuple | list | None = None,
             vlines: list | None = None, fig=None,
             label_scale: float = 1.0, line_scale: float = 1.0, marker_scale: float = 1.0,
             save_dir: str | Path | None = None, show: bool = True):
    """Fetch the layout's signals for each shot (one connection) and plot them.

    `shade` shades a time window (or list of windows) on every time-trace panel,
    e.g. shade=(3800, 4100). `vlines` is a list parallel to `shots` of event
    times [ms] (or None) drawn as a dashed vertical line in that shot's color,
    e.g. vlines=[None, 2200, 1800]. Equilibrium columns (see Equilibrium) fetch
    their own EFIT slice per shot and overlay all shots in R,Z.

    `label_scale` multiplies every label/title/tick/legend font size, `line_scale`
    every line width, and `marker_scale` every marker size — incl. the scatter
    points in Profiles panels (all default 1.0) — bump them for talk-sized figures.

    Pass `fig` to draw into an existing figure (e.g. a `GUItools.FigureNotebook`
    tab) instead of creating one; pass `connection` (a DIIIDConnection) to reuse
    ONE tunnel across several overview()/profiles() calls (polite to the server).

    Returns `(fig, axes)` — the Figure and the list of all its subplot axes (in
    creation order) — and does NOT close the figure, so you can keep plotting.
    With `show=False` the figure is neither displayed NOR auto-saved (even if
    `save_dir` is set), so you can add to it and `fig.savefig(...)` yourself.
    """
    if isinstance(shots, int):
        shots = [shots]
    specs = _all_specs(_flatten(layout))
    cols = list(layout) if _is_columnar(layout) else []
    eq_cols = [c for c in cols if isinstance(c, Equilibrium)]
    prof_cols = [(ci, c) for ci, c in enumerate(cols) if isinstance(c, Profiles)]
    # one equilibrium + profile per analysis (shade) window; falls back to a single
    # snapshot (each column's own time) when no shade window is given.
    windows = _parse_windows(shade)
    centers = [0.5 * (a + b) for a, b in windows]

    results, eq_data, prof_data = {}, {}, {}

    def _fetch_eq(fetcher, shot, etime, tree):     # fetch once per (shot,time,tree)
        key = (shot, etime, tree)
        if key not in eq_data:
            try:
                eq_data[key] = fetcher.fetch_equilibrium(etime, tree)
            except Exception as excp:
                print(f"  ! equilibrium {tree}@{etime:.0f}ms #{shot}: {str(excp)[:45]}")
                eq_data[key] = None

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
            for eq in eq_cols:                        # one slice per window (None->middle)
                for et in (centers or [eq.time if eq.time is not None else 4000.0]):
                    _fetch_eq(fetcher, shot, et, eq.tree)
            for ci, pc in prof_cols:                  # radial-profile columns, per window
                for wi, win in (list(enumerate(windows)) or [(0, None)]):
                    ptime = 0.5 * (win[0] + win[1]) if win else pc.time
                    _fetch_eq(fetcher, shot, ptime, pc.tree)
                    for pi, pp in enumerate(pc.panels):
                        # store a list of (variant_label, profile): 1 for thomson/single CER
                        # flavor, several when pp.system is a list of CER flavors to overlay
                        variants = []
                        if pp.source == "thomson":     # NB: distinct name from the outer trace `specs`
                            vspecs = [(None, pp.system)]
                        else:                        # CER: one entry per flavor (non-flavor -> CERQUICK)
                            flavs = pp.system if isinstance(pp.system, (list, tuple)) else [pp.system]
                            flab = pp.flavor_labels or {}     # user overrides for the legend
                            vspecs = [(flab.get(f, _CER_FLAVOR_LABEL.get(f, f)),
                                       f if f in _CER_FLAVORS else "cerq") for f in flavs]
                        for vlabel, sysn in vspecs:
                            try:
                                if pp.source == "thomson":
                                    pr = fetcher.fetch_thomson_profile(
                                        ptime, pp.quantity, sysn, window=pc.window, t_window=win,
                                        average=pc.average)
                                else:
                                    pr = fetcher.fetch_cer_profile(
                                        ptime, pp.quantity, channels=pp.channels or range(1, 49),
                                        window=pc.window, t_window=win, system=sysn, average=pc.average)
                                variants.append((vlabel, pr))
                            except Exception as excp:
                                print(f"  ! profile {pp.source}.{pp.quantity}"
                                      f"{('/' + sysn) if pp.source == 'cer' else ''} #{shot}: {str(excp)[:45]}")
                        prof_data[(ci, shot, wi, pi)] = variants
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

    fig = _render(results, columns, shots, name=name, t_window=t_window, colors=colors,
                  labels=labels, shade=shade, vlines=vlines, eq_data=eq_data,
                  prof_data=prof_data, fig=fig, label_scale=label_scale, line_scale=line_scale,
                  marker_scale=marker_scale, save_dir=save_dir, show=show)
    return fig, list(fig.axes)         # the figure + all its axes, so the caller can keep plotting


def profiles(shots, time: float = 4000.0, source: str = "cer",
             quantity: str = "tit", system: str = "core", tree: str = "EFIT01",
             channels=range(1, 49), window: float = 100.0, scale: float = 1.0,
             units: str = "", ylabel: str = "", coord: str = "auto",
             ylim: tuple | None = None, label_channels: bool = True,
             colors: list | None = None, labels: list | None = None,
             name: str | None = None, use_cache: bool = True,
             cache_dir: str | Path | None = None,
             tunnel_host: str | None = None, server: str | None = None,
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

    Returns `(fig, (axp, axe))` — the Figure, the profile axis and the equilibrium
    axis — and does NOT close the figure, so you can keep plotting on either.
    """
    if isinstance(shots, int):
        shots = [shots]
    if source not in ("thomson", "cer"):
        raise ValueError(f"profiles() source must be 'thomson' or 'cer' (got {source!r}); "
                         "to compare CER flavors use profiles_cer(...).")

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
    MARK = {"": "o", "C": "o", "T": "s", "D": "^", "V": "v"}   # CER/TS views: T=tang(sq), V=vert(tri)

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
    eqseries = [(shot_colors[si % len(shot_colors)], "-", shot_names[si], eq_data[(sh, time, tree)])
                for si, sh in enumerate(shots) if eq_data.get((sh, time, tree)) is not None]
    _draw_equilibrium(axe, eqseries, Equilibrium(time=time, tree=tree), title_time=time)
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
    if save_dir is not None and show:         # show=False -> composing further: save it yourself
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{name or f'{source}_{quantity}_profile'}_{'_'.join(map(str, shots))}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"* Saved figure -> {out}")
    if show:
        plt.show()
    return fig, (axp, axe)        # (profile axis, equilibrium axis); not closed -> keep plotting


def profiles_cer(shots, time: float = 4000.0, quantities=_CER_QTY_ALL,
                 flavors=("cera", "cerf"), flavor_labels: dict | None = None,
                 tree: str = "EFIT01", channels=range(1, 49), window: float = 100.0,
                 t_window=None, coord: str = "rho", rho_max: float | None = None,
                 label_channels: bool = True, colors: list | None = None,
                 name: str = "cer_check", use_cache: bool = True,
                 cache_dir: str | Path | None = None, tunnel_host: str | None = None,
                 server: str | None = None, connection=None,
                 save_dir: str | Path | None = None, show: bool = True):
    """CER "check" plot: each CER `quantity` as a ROW (value vs coordinate), overlaying
    the analysis FLAVORS, plus one equilibrium per flavor with its channels marked.

    This is the diagnostic look-at-everything plot; for publication panels use
    `overview(Profiles([...]))`. `quantities` picks which rows to show (friendly names,
    default all): 'ti' (Ti), 'rot' (toroidal rotation), 'nz' (impurity density n_Z),
    'fz' (impurity fraction n_Z/n_e), 'amp' (line intensity); the CER suffix and
    display scaling/units come from `_CER_QTY_DISPLAY`. `flavors` are the CER `system`
    prefixes ('cerq'|'cera'|
    'cerf'); each (shot, flavor) is a curve: colour=flavor for one shot, else
    colour=shot + linestyle=flavor. `coord='rho'` maps to ρ_tor via each shot's EFIT;
    `rho_max` drops the SOL. Points are time-averaged over `t_window=(t0,t1)` if given,
    else over `time`±`window`. Pass `flavor_labels` to relabel (e.g. the ion).

    Returns `(fig, (axes_prof, axes_eq))` — the per-quantity profile axes and the
    per-flavor equilibrium axes; the figure is left open so you can keep plotting."""
    if isinstance(shots, int):
        shots = [shots]
    if isinstance(quantities, str):
        quantities = [quantities]
    bad = [fl for fl in flavors if fl not in _CER_FLAVORS]
    if bad:
        raise ValueError(f"profiles_cer() flavors must be CER prefixes {_CER_FLAVORS}; got {bad}.")
    flav_lab = {**_CER_FLAVOR_LABEL, **(flavor_labels or {})}
    rho_kind = {"rho": "tor", "rhotor": "tor", "rhopol": "pol"}.get(coord)
    eqtime = 0.5 * (t_window[0] + t_window[1]) if t_window else time
    qdisp = lambda q: _CER_QTY_DISPLAY.get(q, (q, 1.0, "", q))   # -> (suffix, scale, units, ylabel)

    profs, eq_data = {}, {}                        # profs[(shot, flavor, quantity)]
    own_conn = connection is None
    conn = connection if connection is not None else DIIIDConnection(server=server, tunnel_host=tunnel_host)
    try:
        for sh in shots:
            print(f"* fetching CER {list(quantities)} flavors {list(flavors)} #{sh} ...")
            f = DIIIDFetcher(sh, connection=conn, use_cache=use_cache, cache_dir=cache_dir)
            for fl in flavors:
                for q in quantities:
                    try:
                        profs[(sh, fl, q)] = f.fetch_cer_profile(time, quantity=qdisp(q)[0], channels=channels,
                                                                 window=window, t_window=t_window, system=fl)
                    except Exception as e:
                        print(f"  ! {fl} {q} #{sh}: {str(e)[:40]}")
                        profs[(sh, fl, q)] = None
            try:
                eq_data[sh] = f.fetch_equilibrium(eqtime, tree)
            except Exception as e:
                print(f"  ! equilibrium #{sh}: {str(e)[:40]}")
                eq_data[sh] = None
    finally:
        if own_conn:
            conn.close()

    multishot = len(shots) > 1
    palette = list(colors) if colors else [plt.cm.tab10(i) for i in range(10)]

    def style(si, fi):                            # colour=flavor for one shot; else colour=shot, ls=flavor
        return ((palette[fi % len(palette)], "-") if not multishot
                else (palette[si % len(palette)], _TRACE_LS[fi % len(_TRACE_LS)]))

    def xof(sh, pr):
        if rho_kind:
            ed = eq_data.get(sh)
            return ed.rho_of(pr.r, pr.z, rho_kind) if ed is not None else np.full(pr.r.size, np.nan)
        return pr.z if coord == "Z" else pr.r
    xlabel = {"tor": r"$\rho_{tor}$", "pol": r"$\rho_{pol}$"}.get(rho_kind, f"{coord} [m]")

    avg0, avg1 = t_window if t_window else (time - window, time + window)
    when = (f"avg {avg0:.0f}-{avg1:.0f} ms" if t_window
            else f"t={time:.0f} ms  (avg {avg0:.0f}-{avg1:.0f} ms)")

    def masked(sh, pr):                           # (x, value, error, keep-mask), SOL (x>rho_max) dropped
        x = xof(sh, pr); m = np.isfinite(x)
        if rho_max is not None:
            m &= (x <= rho_max)
        return x[m], pr.value[m], (pr.error[m] if pr.error is not None else None), m

    nq, nf = len(quantities), len(flavors)
    fig = plt.figure(figsize=(5.8 + 2.6 * nf, 1.7 * nq + 0.6))
    gs = fig.add_gridspec(nq, 1 + nf, width_ratios=[2.4] + [1.0] * nf)
    axes_prof = []
    for qi in range(nq):
        axes_prof.append(fig.add_subplot(gs[qi, 0], sharex=axes_prof[0] if axes_prof else None))
    axes_eq = [fig.add_subplot(gs[:, 1 + fi]) for fi in range(nf)]

    # ---- left column: one profile ROW per quantity, flavors overlaid ----
    for qi, q in enumerate(quantities):
        ax = axes_prof[qi]; _suffix, scale, units, ylab = qdisp(q)
        if rho_kind:
            ax.axvline(0.0, color="0.8", ls=":", lw=0.6)
            ax.axvline(1.0, color="0.4", ls="--", lw=0.9)
        for si, sh in enumerate(shots):
            for fi, fl in enumerate(flavors):
                pr = profs.get((sh, fl, q))
                if pr is None or not pr.r.size:
                    continue
                x, val, err, m = masked(sh, pr); o = np.argsort(x); color, ls = style(si, fi)
                lab = flav_lab.get(fl, fl) if not multishot else f"{sh} {flav_lab.get(fl, fl)}"
                ax.errorbar(x[o], val[o] * scale, yerr=(err[o] * scale if err is not None else None),
                            fmt=ls + "o", ms=3.5, lw=0.7, color=color, elinewidth=0.4, capsize=0,
                            alpha=0.9, label=lab)
                if label_channels:
                    lbls = (pr.tag if pr.tag is not None else pr.channel)[m]   # 'T18'/'V14' if tagged
                    for ch, xi, vi in zip(lbls, x, val):
                        ax.annotate(str(ch), (xi, vi * scale), fontsize=3.5, color=color,
                                    xytext=(0, 3), textcoords="offset points", ha="center", alpha=0.8)
        if rho_kind:
            ax.set_xlim(0.0, rho_max)
        ax.set_ylabel(ylab + (f"  [{units}]" if units else ""), fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=6.5)
        if qi == 0:
            ax.set_title(f"CER check vs {xlabel}   {when}", fontsize=9)
            ax.legend(fontsize=6.5, frameon=False, loc="best")
        if qi == nq - 1:
            ax.set_xlabel(xlabel, fontsize=8)
        else:
            ax.tick_params(labelbottom=False)

    # ---- right: one equilibrium PER flavor, channels (of the first quantity) on the EFIT ----
    q0 = quantities[0]
    eqseries = [("0.4", "-", str(sh), eq_data[sh]) for sh in shots if eq_data.get(sh) is not None]
    for fi, fl in enumerate(flavors):
        ax = axes_eq[fi]
        _draw_equilibrium(ax, eqseries, Equilibrium(time=eqtime, tree=tree, label=flav_lab.get(fl, fl)),
                          title_time=eqtime)
        for si, sh in enumerate(shots):
            pr = profs.get((sh, fl, q0))
            if pr is None or not pr.r.size:
                continue
            color, _ls = style(si, fi); _x, _v, _e, m = masked(sh, pr)
            ax.plot(pr.r[m], pr.z[m], "o", ms=3, color=color, zorder=6)
            if label_channels:
                lbls = (pr.tag if pr.tag is not None else pr.channel)[m]
                for k, (ch, r, z) in enumerate(zip(lbls, pr.r[m], pr.z[m])):
                    ax.annotate(str(ch), (r, z), fontsize=4, color="0.15", zorder=7,
                                xytext=(0, 7 if k % 2 == 0 else -7), textcoords="offset points",
                                ha="center", va="bottom" if k % 2 == 0 else "top",
                                arrowprops=dict(arrowstyle="-", color="0.7", lw=0.3))
        if fi > 0:
            ax.set_ylabel("")

    fig.suptitle(f"CER check  (#{', #'.join(map(str, shots))})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_dir is not None and show:         # show=False -> composing further: save it yourself
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{name}_{'_'.join(map(str, shots))}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"* Saved figure -> {out}")
    if show:
        plt.show()
    return fig, (axes_prof, axes_eq)    # (per-quantity profile axes, per-flavor equilibrium axes)


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
    y = y if perspec else y * trace.scale
    return t, (np.abs(y) if trace.abs else y)


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


def _trace_legend_label(tr, show_source):
    """Legend text for a trace. With `show_source`, the MDS spec is folded in as a
    second line under the user label (one combined legend entry, not a separate note)."""
    spec = _spec_label(tr)
    if show_source and tr.label and tr.label != spec:
        return f"{tr.label}\n{spec}"
    return tr.label or spec


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
                shot_colors, shot_names, fs=1.0, lw=1.0, msc=1.0):
    """Draw one panel; returns True if anything was plotted. `fs`/`lw`/`msc` scale the
    label font sizes, the line widths and the marker sizes."""
    multitrace = len(panel.traces) > 1
    drew = False
    for si, shot in enumerate(shots):
        for ti, tr in enumerate(panel.traces):
            xy = _trace_xy(results[shot], tr)
            if xy is None:
                continue
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
                color = _TRACE_PALETTE[ti % len(_TRACE_PALETTE)]
                label = _trace_legend_label(tr, panel.show_source)   # source folds in as a 2nd line
            if tr.avg and tr.avg > 0:                 # raw faint, time-average bold
                td, yd = _decimate(t, y)
                if tr.raw_alpha > 0:
                    ax.plot(td, yd, color=color, ls=ls, lw=0.7 * lw, alpha=tr.raw_alpha)
                ta, ya = _decimate(t, _moving_avg(t, y, tr.avg))
                ax.plot(ta, ya, color=color, ls=ls, lw=1.4 * lw, label=label,
                        marker=tr.marker, ms=3 * msc)
            else:
                td, yd = _decimate(t, y)
                ax.plot(td, yd, color=color, ls=ls, lw=0.9 * lw, label=label,
                        marker=tr.marker, ms=3 * msc)
            drew = True

    ax.set_ylabel(panel.ylabel, fontsize=8 * fs)
    ax.tick_params(labelsize=6.5 * fs)
    ax.grid(alpha=0.3)
    if panel.ylim is not None:
        ax.set_ylim(panel.ylim)
    if not drew:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", fontsize=7 * fs, color="0.6")
    elif multitrace and not multishot:
        ax.legend(fontsize=5.5 * fs, loc="best", ncol=2, frameon=False)
    elif multitrace and multishot:
        handles = [Line2D([], [], color="0.3", ls=_trace_ls(tr, ti, True),
                          lw=1, label=_trace_legend_label(tr, panel.show_source))   # label (+ source)
                   for ti, tr in enumerate(panel.traces)]
        ax.legend(handles=handles, fontsize=5 * fs, loc="best", ncol=2, frameon=False)
    if panel.show_source and not multitrace:   # single trace -> source as a note (multitrace folds it
        specs = [_spec_label(tr) for tr in panel.traces]   # into the legend entries above)
        ax.text(0.99, 0.96, ", ".join(dict.fromkeys(specs)), transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5 * fs, color="0.45",
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
    return drew


def _separatrix_legs(ax, ed, color, lw=1.0):
    """Draw the divertor separatrix legs: contour psi_N = 1 on a sub-grid spanning
    the A-file strike points to ~10 cm past the active X-point, and mark the active
    X-point. No-op for a limited plasma. `lw` scales the line widths."""
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
    ax.contour(rspln, zspln, sub, levels=[1.0], colors=[color], linewidths=1.3 * lw)
    ax.plot(rxpt, zxpt, "x", color=color, ms=5, mew=1.3 * lw)


def _draw_equilibrium(ax, series, eq, title_time=None, fs=1.0, lw=1.0):
    """R,Z flux-surface panel from a GEQDSK: dashed interior flux surfaces (psi_N
    from axis to boundary), the bold LCFS, and the diverted separatrix legs to the
    strike points. `series` is a list of (color, linestyle, label, EquilibriumData),
    one per (shot, window). The interior flux surfaces + SOL are drawn for EVERY
    series (grey when there is a single case, else in each series' colour) so all
    selected times/shots are visible; the vessel is shared. `fs`/`lw` scale fonts
    and line widths."""
    ax.set_title(f"{eq.label or f'EFIT {eq.tree}'}"
                 + (f"  t={title_time:.0f} ms" if title_time is not None else ""), fontsize=8 * fs)
    if not series:
        ax.text(0.5, 0.5, "(no equilibrium)", transform=ax.transAxes,
                ha="center", va="center", fontsize=7 * fs, color="0.6")
        return

    ref = series[0][3]
    # clip flux surfaces to the vessel so the SOL contours don't sprawl outside it
    clip = PathPatch(MplPath(np.column_stack([ref.wall_r, ref.wall_z])),
                     transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    ax.plot(ref.wall_r, ref.wall_z, "-", color="0.15", lw=0.9 * lw)     # vessel (shared)
    multi = len(series) > 1
    for color, ls, _label, ed in series:                          # interior + SOL + LCFS PER case
        surf = color if multi else "0.55"
        interp = RegularGridInterpolator((ed.zgrid, ed.rgrid), ed.psiN,
                                         bounds_error=False, fill_value=np.nan)
        # interior flux surfaces (dashed), psi_N = levels
        ci = ax.contour(ed.rgrid, ed.zgrid, ed.psiN, levels=list(eq.levels),
                        colors=surf, linewidths=0.35 * lw, linestyles="dashed",
                        alpha=0.55 if multi else 1.0)
        ci.set_clip_path(clip)
        # SOL surfaces: sample psi at points stepped outward in R from the outboard-
        # midplane boundary point by deltascrape (even real-space spacing -> no inboard
        # bunching), then contour those psi levels.
        imax = int(np.argmax(ed.rbbbs))
        r0, z0 = float(ed.rbbbs[imax]), float(ed.zbbbs[imax])
        rpts = r0 + np.arange(1, eq.nscrape + 1) * eq.deltascrape
        sol = interp(np.column_stack([np.full(rpts.size, z0), rpts]))
        sol = np.sort(sol[np.isfinite(sol)])
        if sol.size:
            cs = ax.contour(ed.rgrid, ed.zgrid, ed.psiN, levels=list(sol),
                            colors=surf if multi else "0.6", linewidths=0.3 * lw,
                            alpha=0.55 if multi else 1.0)
            cs.set_clip_path(clip)
        ax.plot(ed.rbbbs, ed.zbbbs, color=color, lw=1.3 * lw, ls=ls)     # LCFS
        _separatrix_legs(ax, ed, color, lw=lw)
        ax.plot(ed.raxis, ed.zaxis, "+", color=color, ms=7 * fs)

    ax.set_xlim(ref.wall_r.min() - 0.04, ref.wall_r.max() + 0.04)
    ax.set_ylim(ref.wall_z.min() - 0.04, ref.wall_z.max() + 0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]", fontsize=7 * fs)
    ax.set_ylabel("Z [m]", fontsize=8 * fs)
    ax.tick_params(labelsize=6.5 * fs)


def _draw_profile_panel(ax, prof_data, eq_data, shots, pc, c, pi, pp, windows, centers,
                        series_style, last, fs=1.0, lw=1.0, msc=1.0):
    """One radial-profile sub-panel (value vs ρ) in a Profiles column: one curve per
    (shot, window), each time-averaged over its window and mapped through that
    window's equilibrium. `series_style(si, wi)` gives the (color, linestyle).
    `fs`/`lw`/`msc` scale the label font sizes, the line widths and the marker sizes."""
    coord = pc.coord
    rho_kind = {"rho": "tor", "rhotor": "tor", "rhopol": "pol"}.get(coord)
    wis = range(len(windows)) if windows else [0]
    drew = False
    flavor_ls = {}                                # CER flavor label -> linestyle (for the legend)
    for si, sh in enumerate(shots):
        for wi in wis:
            variants = (prof_data or {}).get((c, sh, wi, pi)) or []      # (label, profile) per flavor
            for vidx, (vlabel, prof) in enumerate(variants):
                if prof is None or not prof.r.size:
                    continue
                t = centers[wi] if centers else pc.time
                ed = eq_data.get((sh, t, pc.tree))
                if rho_kind:
                    x = ed.rho_of(prof.r, prof.z, rho_kind) if ed is not None else np.full(prof.r.size, np.nan)
                else:
                    x = prof.z if coord == "Z" else prof.r
                m = np.isfinite(x)
                if rho_kind and pc.rho_max is not None:           # drop SOL points beyond rho_max
                    m &= (x <= pc.rho_max)
                xm, vm = x[m], prof.value[m]
                em = prof.error[m] if (pc.errorbars and prof.error is not None) else None
                o = np.argsort(xm); color, ls = series_style(si, wi); mk = "o"
                if len(variants) > 1:                # overlaid flavors -> linestyle (joined) or marker
                    if pp.join:
                        ls = _TRACE_LS[vidx % len(_TRACE_LS)]
                    else:
                        mk = _PROF_MARKERS[vidx % len(_PROF_MARKERS)]
                if vlabel is not None:               # legend the flavor even when there's just one
                    flavor_ls[vlabel] = (ls if pp.join else "none", mk)
                fmt = (ls + mk) if pp.join else mk    # join with a line, else scatter only
                ax.errorbar(xm[o], vm[o] * pp.scale, yerr=(em[o] * pp.scale if em is not None else None),
                            fmt=fmt, ms=2.5 * msc, lw=0.6 * lw, color=color, elinewidth=0.5 * lw,
                            capsize=0, alpha=pp.alpha)
                drew = True
    if rho_kind:
        ax.axvline(0.0, color="0.8", ls=":", lw=0.6 * lw)        # magnetic axis
        ax.axvline(1.0, color="0.4", ls="--", lw=0.9 * lw)       # separatrix (ρ=1)
        ax.set_xlim(0.0, pc.rho_max)
    ax.set_ylabel(pp.ylabel or pp.quantity, fontsize=8 * fs)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=6.5 * fs)
    if pp.ylim is not None:
        ax.set_ylim(pp.ylim)
    if flavor_ls:                                 # legend: which line/marker is which CER flavor
        ax.legend(handles=[Line2D([], [], color="0.3", ls=l, marker=mk, lw=1.2 * lw, label=lab)
                           for lab, (l, mk) in flavor_ls.items()],
                  fontsize=6 * fs, frameon=False, loc="best")
    if not drew:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center",
                fontsize=7 * fs, color="0.6")
    if pi == 0:                                   # top of the column: how the points were reduced
        n_w = len(windows)
        how = "avg" if pc.average else "all pts"  # time-averaged vs every sample in the window
        ttl = (f"profiles  ({how} {windows[0][0]:.0f}-{windows[0][1]:.0f} ms)" if n_w == 1
               else f"profiles  ({how} per window)" if n_w > 1
               else f"profiles  t={centers[0] if centers else pc.time:.0f} ms")
        ax.set_title(ttl, fontsize=8 * fs)
    if last:
        ax.set_xlabel({"tor": r"$\rho_{tor}$", "pol": r"$\rho_{pol}$"}.get(rho_kind, f"{coord} [m]"),
                      fontsize=8 * fs)
    else:
        ax.tick_params(labelbottom=False)


def _render(results, columns, shots, name="overview",
            t_window: tuple | None = DEFAULT_TWINDOW, colors: list | None = None,
            labels: list | None = None, shade: tuple | list | None = None,
            vlines: list | None = None, eq_data: dict | None = None,
            prof_data: dict | None = None, fig=None, label_scale: float = 1.0,
            line_scale: float = 1.0, marker_scale: float = 1.0,
            save_dir: str | Path | None = None, show: bool = True):
    """Place `columns` (Panels, an Equilibrium, or a Profiles column) on a grid.
    `fig` (a Figure, e.g. a FigureNotebook tab) is drawn into if given; else a
    new figure is created and saved/shown per `save_dir`/`show`. `label_scale`/
    `line_scale`/`marker_scale` multiply every label font size / line width / marker."""
    fs, lw, msc = label_scale, line_scale, marker_scale
    eq_data, prof_data = eq_data or {}, prof_data or {}
    windows = _parse_windows(shade)               # analysis windows (one snapshot each)
    centers = [0.5 * (a + b) for a, b in windows]
    n_w = len(windows)
    multishot = len(shots) > 1
    ncol = len(columns)
    nrow = max((len(col.panels) if isinstance(col, Profiles) else len(col)
                for col in columns if not isinstance(col, Equilibrium)), default=1)
    t0, t1 = t_window if t_window is not None else \
        _discharge_window([results[sh].get("ip") for sh in shots])
    shot_colors = list(colors) if colors else [plt.cm.tab10(i) for i in range(10)]
    win_palette = list(colors) if colors else [plt.cm.tab10(i) for i in range(10)]
    shot_names = [f"{sh} ({labels[si]})" if labels and si < len(labels) and labels[si]
                  else str(sh) for si, sh in enumerate(shots)]
    by_window = (n_w > 1 and not multishot)       # one shot, many windows -> colour by window
    win_lab = lambda wi: f"{windows[wi][0]:.0f}-{windows[wi][1]:.0f} ms"

    def series_style(si, wi):                     # (color, linestyle) for one (shot, window) curve
        if by_window:
            return win_palette[wi % len(win_palette)], "-"
        return (shot_colors[si % len(shot_colors)],
                _TRACE_LS[wi % len(_TRACE_LS)] if n_w > 1 else "-")

    def series_label(si, wi):
        if by_window:
            return win_lab(wi)
        return shot_names[si] if n_w <= 1 else f"{shot_names[si]} @ {win_lab(wi)}"

    shade_color = lambda wi: win_palette[wi % len(win_palette)] if by_window else "gold"

    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=(np.min([17, 3.6 * ncol]), np.min([10, 3.0 * nrow])))
    elif np.allclose(fig.get_size_inches(), plt.rcParams["figure.figsize"]):
        # provided fig is UNSIZED (matplotlib default, e.g. a FigureNotebook tab) -> fill it so it
        # doesn't render as a short strip; an explicitly-sized fig is respected as given.
        fig.set_size_inches(max(12.0, 3.6 * ncol), max(8.5, 2.0 * nrow))
    gs = fig.add_gridspec(nrow, ncol)
    xref = None
    for c, col in enumerate(columns):
        if isinstance(col, Equilibrium):          # one boundary per (shot, window)
            ts = centers if centers else [col.time if col.time is not None else 4000.0]
            series = []
            for si, sh in enumerate(shots):
                for wi, t in enumerate(ts):
                    ed = eq_data.get((sh, t, col.tree))
                    if ed is not None:
                        color, ls = series_style(si, wi)
                        series.append((color, ls, series_label(si, wi), ed))
            _draw_equilibrium(fig.add_subplot(gs[:, c]), series, col,
                              title_time=(ts[0] if len(ts) == 1 else None), fs=fs, lw=lw)
            continue
        if isinstance(col, Profiles):
            sub = gs[:, c].subgridspec(len(col.panels), 1, hspace=0.12)
            paxref = None
            for pi, pp in enumerate(col.panels):
                pax = fig.add_subplot(sub[pi, 0], sharex=paxref); paxref = paxref or pax
                _draw_profile_panel(pax, prof_data, eq_data, shots, col, c, pi, pp,
                                    windows, centers, series_style,
                                    last=(pi == len(col.panels) - 1), fs=fs, lw=lw, msc=msc)
            continue
        for r in range(len(col)):
            ax = fig.add_subplot(gs[r, c], sharex=xref)
            xref = xref or ax
            _draw_panel(ax, results, col[r], shots, t0, t1, multishot,
                        shot_colors, shot_names, fs=fs, lw=lw, msc=msc)
            for wi, (s0, s1) in enumerate(windows):
                ax.axvspan(s0, s1, color=shade_color(wi), alpha=0.16, lw=0, zorder=0)
            for si, vt in enumerate(vlines or []):       # per-shot event markers
                if vt is not None:
                    ax.axvline(vt, color=shot_colors[si % len(shot_colors)],
                               ls="--", lw=1.0 * lw, alpha=0.8, zorder=1)
            if r < len(col) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time  [ms]", fontsize=7 * fs)
    if xref is not None:
        xref.set_xlim(t0, t1)

    # `name` is the title verbatim (no auto prefix); empty -> no title. Legend sits
    # just below it, wrapping to as many rows as the figure width needs, and the
    # subplot area top is set right under the legend (no big gap).
    title = (name or "") + ("" if multishot else f"  #{shots[0]}")
    height = fig.get_figheight()
    has_title = bool(title.strip())
    # Profiles / Equilibrium columns draw a title above their TOP panel; reserve a row
    # for it so it clears the figure legend instead of poking up into the shot labels.
    title_pad = 0.30 if any(isinstance(col, (Profiles, Equilibrium)) for col in columns) else 0.0
    leg = []                                       # colour = shot; for >1 window add a window key
    if multishot:
        leg += [Line2D([], [], color=shot_colors[si % len(shot_colors)], lw=2 * lw, label=shot_names[si])
                for si in range(len(shots))]
    if n_w > 1:
        leg += [Line2D([], [], lw=2 * lw, label=win_lab(wi),
                       color=(win_palette[wi % len(win_palette)] if by_window else "0.3"),
                       ls=("-" if by_window else _TRACE_LS[wi % len(_TRACE_LS)]))
                for wi in range(n_w)]
    if has_title:
        fig.suptitle(title, fontsize=12 * fs, y=1 - 0.22 / height)
    fig.tight_layout(h_pad=0.15, w_pad=0.4)        # sets left/right/bottom; we set the top below
    sub_top = 1 - ((0.34 if has_title else 0.07) + title_pad) / height
    if leg:
        ncol_leg = max(1, min(len(leg), int(fig.get_figwidth() / 1.6)))   # fit the figure width
        nrow_leg = -(-len(leg) // ncol_leg)                               # ceil division
        leg_y = 1 - (0.46 if has_title else 0.20) / height                # legend top, under the title
        fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, leg_y),
                   ncol=ncol_leg, fontsize=8 * fs, frameon=False)
        sub_top = leg_y - (0.25 * nrow_leg + 0.05 + title_pad) / height   # +room for the column title
    fig.subplots_adjust(top=sub_top, hspace=0.12)
    if own_fig:                           # a provided fig (notebook tab) is shown/saved by the caller
        if save_dir is not None and show:  # show=False -> composing further: save it yourself
            save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{name}_{'_'.join(str(s) for s in shots)}.png"
            fig.savefig(save_path, dpi=150)
            print(f"* Saved figure -> {save_path}")
        if show:
            plt.show()
    return fig
