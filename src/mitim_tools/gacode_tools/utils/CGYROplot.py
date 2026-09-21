"""
CGYRO-specific plotting helpers that are fully PORTALS-agnostic.

Owned by the CGYRO tools layer so any caller with a set of CGYRO iteration
folders (PORTALS is currently the only one, but the module has no PORTALS
imports) can produce the per-rho Qe/Qi/Ge time-trace figure. Callers handle
their own iteration discovery (folder layout) and namelist lookups (tmin,
restart mode); this module handles CGYRO-specific loading and drawing.
"""

import json
import os
import shlex
import shutil
import time
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mitim_tools.simulation_tools import SIMtools
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools import GRAPHICStools, FARMINGtools
from mitim_tools.misc_tools.LOGtools import HiddenPrints, printMsg as print


# Column chunk size: at most this many non-base iterations per column.
_CHUNK_SIZE = 5

# Channels drawn on the figure. Keeping this module-level so both grid
# layouts (rows=channels and rows=rhos) agree on order and labels.
_CHANNELS = [
    ('Qe', '$Q_e$ [GB]'),
    ('Qi', '$Q_i$ [GB]'),
    ('Ge', '$\\Gamma_e$ [GB]'),
]

# Shared cmap template; each column builds its own Normalize over just its
# chunk so column-local first->last reads as blue->red.
_CMAP_ITER = LinearSegmentedColormap.from_list("iter_bluered", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])

# Trace channel -> GB flux key used in the targets_per_iter dicts.
_CHANNEL_TO_GB = {'Qe': 'QeGB', 'Qi': 'QiGB', 'Ge': 'GeGB'}

# Fractional x-axis extension past the last trace so the in-axes legend
# (upper right) sits over empty space instead of covering data.
_XLIM_LEGEND_EXTEND = 0.20


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_tool_for_iteration(folder_execution, rhos, read_kwargs=None, base_subfolder="base_cgyro"):
    '''
    Best-effort load of a CGYRO tool object carrying per-rho CGYROoutput
    instances for one iteration folder. Tries the pickle fast path first
    (single-plasma keep_files='pickle'), then falls back to re-reading the
    raw CGYRO output files. Returns None if neither works — caller should
    skip the iteration.

    `read_kwargs` forwards tmin / tmin_is_rel (and anything else the read
    API accepts). The raw fallback uses those so the plot-time re-read
    reproduces the exact averaging window the owning driver used at
    simulation time. Without this the raw path defaults to tmin=0.0
    (full window) and every *_mean / _std displayed disagrees with the
    scalars the driver actually consumed.

    `base_subfolder` names the on-disk artifacts directory to look inside
    (defaults to "base_cgyro" for single-fidelity). For named multi-fidelity
    CGYRO instances callers should pass e.g. "base_cgyro1" to match what
    the dispatcher wrote at run time. The raw-fallback read also labels its
    results with `base_subfolder` so pick_output_for_rho's generic label
    iteration still finds the data.
    '''
    base = folder_execution / base_subfolder
    if not base.is_dir():
        return None

    pickle_file = base / "gk_object.pkl"
    if pickle_file.is_file():
        try:
            with HiddenPrints():
                return SIMtools.restore_class_pickle(pickle_file)
        except Exception as e:
            print(f"\t- CGYRO pickle unreadable at {pickle_file} ({e}); falling back to raw files", typeMsg='w')

    read_kwargs = dict(read_kwargs) if read_kwargs else {}
    try:
        with HiddenPrints():
            c = CGYROtools.CGYRO(rhos=list(rhos))
            c.read(folder=base, label=base_subfolder, minimal=True, **read_kwargs)
        return c
    except Exception as e:
        print(f"\t- CGYRO read failed at {base} ({e}); skipping iteration", typeMsg='w')
        return None


def pick_output_for_rho(tool, rho, fallback_idx):
    '''
    Locate the CGYROoutput instance inside a tool.results dict that
    corresponds to `rho`. Matches by nearest-rho against tool.rhos, falling
    back to positional index if tool.rhos is unavailable. Tries label
    "base_cgyro" first, then any other label the tool carries.
    '''
    if hasattr(tool, "rhos") and tool.rhos is not None and len(tool.rhos) > 0:
        idx = int(np.argmin(np.abs(np.asarray(tool.rhos, dtype=float) - float(rho))))
    else:
        idx = fallback_idx

    results = getattr(tool, "results", None) or {}
    for label in ("base_cgyro", *[lab for lab in results if lab != "base_cgyro"]):
        if label not in results:
            continue
        outputs = results[label].get("output")
        if outputs and 0 <= idx < len(outputs):
            return outputs[idx]
    return None


def load_tools_for_iterations(iteration_folders, rhos, read_kwargs=None, base_subfolder="base_cgyro"):
    '''
    Convenience: given an iterable of (iteration_index, folder) pairs,
    return {iteration_index: tool} skipping any iteration whose folder
    fails to load. Non-destructive if some iterations are missing on
    disk — the result is whatever subset was readable.

    `base_subfolder` is forwarded to load_tool_for_iteration so named
    multi-fidelity CGYRO instances pick up the right per-iteration directory.
    '''
    cache = {}
    for it, folder in iteration_folders:
        tool = load_tool_for_iteration(folder, rhos, read_kwargs=read_kwargs, base_subfolder=base_subfolder)
        if tool is not None:
            cache[it] = tool
    return cache


def load_restart_sources_for_iterations(iteration_folders, base_subfolder="base_cgyro"):
    '''
    Companion to `load_tools_for_iterations`: load each iteration's
    restart_sources.json (written by `_resolve_cgyro_restart_chain` in
    transport_cgyro for all three "first" / "all" / "best" modes via a
    single code path). The plotter consumes this to align CGYRO time
    traces across warm-started iterations.

    Returns {iteration_index: {"mode": "first|all|best",
                               "parents": {"0.2500": <source_iter>, ...}}}
    Iterations without a JSON contribute no entry — they are treated as
    cold starts by the plotter (offset = 0). This is also the behavior
    when no PORTALS run wrote any restart_sources.json files at all
    (e.g., older runs predating the persistence step, or runs with
    `restart_from_cases: null`).

    `base_subfolder` matches the on-disk artifacts directory name (e.g.
    "base_cgyro" for single-fidelity, "base_cgyro1" for named multi-
    fidelity instances) so the right JSON is picked up per iteration.
    '''
    sources_per_iter = {}
    for it, folder in iteration_folders:
        json_path = folder / base_subfolder / "restart_sources.json"
        if not json_path.is_file():
            continue
        try:
            with open(json_path, "r") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            print(f"\t- restart_sources.json unreadable at {json_path} ({e}); ignoring", typeMsg='w')
            continue
        parents_raw = payload.get("sources", {})
        if not isinstance(parents_raw, dict):
            continue
        parents = {}
        for k, v in parents_raw.items():
            try:
                parents[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        if not parents:
            continue
        sources_per_iter[it] = {
            "mode": str(payload.get("mode", "")),
            "parents": parents,
        }
    return sources_per_iter


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------


def _chunk_iterations(sorted_its, base_iter):
    '''Partition non-base iterations into column chunks of size _CHUNK_SIZE.'''
    non_base = [i for i in sorted_its if i != base_iter]
    chunks = (
        [non_base[i:i + _CHUNK_SIZE] for i in range(0, len(non_base), _CHUNK_SIZE)]
        if non_base else [[]]
    )
    return non_base, chunks


def _make_column_color_fn(chunk):
    '''Blue->red linear colormap scaled over exactly this chunk's ev range.'''
    if not chunk:
        return lambda it: (0.0, 0.0, 0.0)
    v0, v1 = chunk[0], chunk[-1]
    norm = (Normalize(vmin=v0, vmax=v1) if v0 != v1
            else Normalize(vmin=v0 - 0.5, vmax=v0 + 0.5))
    return lambda it: _CMAP_ITER(norm(it))


def _restart_label_from_sources(sources_per_iter):
    '''Derive a short human-readable label of the restart mode used across
    iterations. Returns "" when no iteration carries a sources record (so
    the plotter renders without a restart-aware title/xlabel suffix).'''
    if not sources_per_iter:
        return ""
    modes = {entry.get("mode", "") for entry in sources_per_iter.values()}
    modes.discard("")
    if not modes:
        return ""
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _xlabel_suffix_from_sources(sources_per_iter):
    '''xlabel suffix that matches the dominant restart mode in the JSONs.
    Empty string when no JSON was found, which preserves the legacy
    "no restart context" rendering.'''
    label = _restart_label_from_sources(sources_per_iter)
    return {
        "all": " (chained)",
        "first": " (branched from ev0)",
        "best": " (best per-rho)",
        "mixed": " (restart-aligned)",
    }.get(label, "")


def _compute_offsets_for_rho(rho, r_idx, sources_per_iter, cache, sorted_its):
    '''
    Per-(rho, iter) time offsets via parent-map walk. Each iter's offset
    is the sum of t[parent].t[-1] along its parent chain back to a root
    (an iter whose restart_sources.json has no entry for this rho — i.e.
    the rho was cold-started for that iter, or no JSON was written at all).
    Iters with no parent chain at this rho get offset = 0, which collapses
    to the legacy "no restart" rendering.

    The parent map is the per-rho mapping read from each child iter's
    restart_sources.json. "first" produces parent ≡ 0 for every (rho, iter);
    "all" produces parent ≡ k-1; "best" produces a per-(rho, iter) lookup.
    The walker is identical for all three modes — that is the whole point
    of routing every restart mode through this single path.

    Memoised + cycle-broken (cycles can't arise from the writer but the
    plotter is defensive against hand-edited JSON).
    '''
    rho_key = f"{rho:.4f}"

    def _tmax(it):
        if it not in cache:
            return 0.0
        out = pick_output_for_rho(cache[it], rho, r_idx)
        if out is None or not hasattr(out, "t") or len(out.t) == 0:
            return 0.0
        return float(out.t[-1])

    memo = {}
    visiting = set()

    def _offset(it):
        if it in memo:
            return memo[it]
        if it in visiting:
            return 0.0
        visiting.add(it)
        try:
            entry = sources_per_iter.get(it) or {}
            parent = entry.get("parents", {}).get(rho_key)
            if parent is None or parent not in cache:
                off = 0.0
            else:
                off = _offset(parent) + _tmax(parent)
        finally:
            visiting.discard(it)
        memo[it] = off
        return off

    return {it: _offset(it) for it in sorted_its}


def _draw_chunk_cell(ax, var, rho, r_idx, chunk, cache, base_out, offsets,
                     color_for, base_iter, targets_per_iter=None):
    '''
    Draw one subplot: non-base traces in `chunk` + base iter, for channel
    `var` at `rho`. Returns (trace_count, trace_means) so the outer grid
    owner can run per-row y-clamp based on the 2.5x-of-max-mean rule.

    `targets_per_iter` ({iter: {"QeGB": per-rho array, ...}}, values =
    target - neoclassical in GB) adds a color-matched star at the end of
    each iteration's trace marking the turbulence-only target it was
    trying to flux-match — the distance star <-> mean marker reads as the
    residual. Iterations / channels absent from the dict draw no star.

    Per trace we draw:
      1. The raw time trace.
      2. A dashed horizontal line at mean across [tmin, t[-1]] so the
         averaged scalar the driver consumed is visible alongside the
         signal.
      3. A shaded rectangle with x-range = [tmin, t[-1]] (the averaging
         window) and y-range = [mean - 2*sigma, mean + 2*sigma]. The box
         thus encodes both the window duration (horizontally) and the
         post-autocorr uncertainty of the scalar (vertically).
      4. A square errorbar marker at the trace end showing mean +/- 2*sigma
         at the precise (t_end, mean) point.
    The base iteration is drawn in black on top with thicker styling and
    a gray shading instead of a colour-matched one.
    '''
    trace_means = []
    trace_count = 0
    target_vals = []
    gb_key = _CHANNEL_TO_GB.get(var)

    def _draw_target(it, out, offset_it, color, is_base):
        '''Star at the end of the trace: the turbulence-only target
        (target - neoc) this iteration was trying to match.'''
        if not targets_per_iter or gb_key is None or out is None:
            return
        arr = (targets_per_iter.get(it) or {}).get(gb_key)
        if arr is None or r_idx >= len(arr):
            return
        tval = float(arr[r_idx])
        if not np.isfinite(tval) or not hasattr(out, "t") or len(out.t) == 0:
            return
        x_end = float(out.t[-1]) + offset_it
        ax.plot([x_end], [tval], marker='*', ls='',
                ms=12 if is_base else 10,
                color='black' if is_base else color,
                mec='black', mew=0.4, zorder=8)
        # Collected separately from trace_means: the row clamp keeps targets
        # in frame with a modest margin instead of the factor-scaled headroom.
        target_vals.append(tval)

    def _draw_single_trace(out, offset_it, color, is_base):
        nonlocal trace_count
        y = getattr(out, var, None)
        if y is None or not hasattr(out, "t"):
            return
        t_shifted = out.t + offset_it
        x_end = float(t_shifted[-1])
        tmin_it = getattr(out, 'tmin', None)
        mean_val = getattr(out, f"{var}_mean", None)
        std_val = getattr(out, f"{var}_std", None)

        # Styling split so ev0 reads as the reference.
        if is_base:
            lw_trace, lw_mean, ms_err, capsize_err, lw_err = 1.4, 1.0, 5, 3, 1.2
            z_trace, z_mean, z_err = 5, 6, 7
            alpha_trace, alpha_mean = 1.0, 0.85
            shade_color, shade_alpha = 'gray', 0.28
        else:
            lw_trace, lw_mean, ms_err, capsize_err, lw_err = 1.0, 0.7, 3, 2, 0.8
            z_trace, z_mean, z_err = 2, 3, 4
            alpha_trace, alpha_mean = 0.85, 0.8
            shade_color, shade_alpha = color, 0.22

        # 2*sigma x window box replaces the old full-height axvspan so the
        # rectangle's vertical extent is informative. Drawn first (low z)
        # so it sits behind the signal.
        if tmin_it is not None and mean_val is not None and std_val is not None:
            m, s2 = float(mean_val), 2.0 * float(std_val)
            ax.fill_between(
                [float(tmin_it) + offset_it, x_end],
                [m - s2, m - s2],
                [m + s2, m + s2],
                color=shade_color, alpha=shade_alpha, linewidth=0, zorder=0,
            )

        ax.plot(t_shifted, y, color=color if not is_base else 'black',
                lw=lw_trace, alpha=alpha_trace, zorder=z_trace)
        trace_count += 1

        # Dashed mean line across the averaging window, for every trace —
        # not just the base. Colour-matched for non-base so the reader can
        # associate mean line -> trace.
        if mean_val is not None and tmin_it is not None:
            ax.hlines(
                float(mean_val),
                float(tmin_it) + offset_it,
                x_end,
                colors=color if not is_base else 'black',
                linestyles='--', lw=lw_mean, alpha=alpha_mean, zorder=z_mean,
            )

        if mean_val is not None and std_val is not None:
            ax.errorbar(
                x_end, float(mean_val), yerr=2.0 * float(std_val),
                fmt='s',
                color=color if not is_base else 'black',
                ms=ms_err, capsize=capsize_err, lw=lw_err,
                mec='black', mew=0.3, zorder=z_err,
            )

        # Window provenance: a tick at t_start spanning the band, and the averaging
        # method/flag (GKaveraging) as a small label. The label goes on the base trace
        # always and on other traces only when the method did not converge ('fallback'/
        # 'failure'), so a bad window is visible without cluttering healthy cells.
        avg = getattr(out, 'averaging', None)
        if tmin_it is not None and mean_val is not None and std_val is not None:
            x0, m, s2 = float(tmin_it) + offset_it, float(mean_val), 2.0 * float(std_val)
            ax.vlines(x0, m - s2, m + s2, colors=color if not is_base else 'black', linestyles=':', lw=lw_mean, alpha=alpha_mean, zorder=z_mean)
            method, flag = getattr(avg, 'method', None), getattr(avg, 'flag', None)
            if method is not None and (is_base or flag in ('fallback', 'failure')):
                ax.text(x0, m + s2, f" {method}" + (f" ({flag})" if flag not in (None, 'ok', 'fixed') else ""),
                        color=color if not is_base else 'black', fontsize=6, ha='left', va='bottom', alpha=0.9, zorder=z_err)

        # Track (mean, std) per trace so the row-level clamp can apply the
        # factor-of-(mean+/-2sigma) rule. Missing std defaults to 0 (treat as
        # a point). Skip entries whose mean isn't finite so pre-window traces
        # don't push the clamp toward raw transient values.
        if mean_val is not None:
            try:
                m = float(mean_val)
                s = float(std_val) if std_val is not None else 0.0
                if np.isfinite(m) and np.isfinite(s):
                    trace_means.append((m, s))
            except (TypeError, ValueError):
                pass

    # Non-base traces first so the base overlays them.
    for it in chunk:
        out = pick_output_for_rho(cache[it], rho, r_idx)
        if out is None:
            continue
        # offsets is empty in local-time mode (time_mode="local"): every trace starts at 0
        _draw_single_trace(out, offsets.get(it, 0.0), color_for(it), is_base=False)
        _draw_target(it, out, offsets.get(it, 0.0), color_for(it), is_base=False)

    if base_out is not None:
        _draw_single_trace(base_out, offsets.get(base_iter, 0.0), color=None, is_base=True)
        _draw_target(base_iter, base_out, offsets.get(base_iter, 0.0), None, is_base=True)

    return trace_count, trace_means, target_vals


def _column_legend(ax, chunk, color_for, base_iter, base_has_window, has_targets=False):
    '''Compact legend shared by both grid layouts: base + chunk endpoints,
    the x=window x y=2*sigma shading patches, the dashed mean line, and
    the mean +/- 2*sigma errorbar marker. Keeps readers honest about what
    every visual channel on the plot is encoding. Anchored at the upper
    right, where the _XLIM_LEGEND_EXTEND x-axis extension guarantees
    data-free space.'''
    handles = [Line2D([0], [0], color='black', lw=1.4)]
    labels_ = [f'ev{base_iter} (base)']
    if chunk:
        first_it, last_it = chunk[0], chunk[-1]
        handles.append(Line2D([0], [0], color=color_for(first_it), lw=1.5))
        labels_.append(f'ev{first_it}')
        if last_it != first_it:
            handles.append(Line2D([0], [0], color=color_for(last_it), lw=1.5))
            labels_.append(f'ev{last_it}')
    if base_has_window:
        handles.append(Patch(facecolor='gray', alpha=0.35, edgecolor='none'))
        labels_.append(f'window $\\times 2\\sigma$ (ev{base_iter})')
    if chunk:
        handles.append(Patch(facecolor=color_for(chunk[-1]), alpha=0.3, edgecolor='none'))
        labels_.append(r'window $\times 2\sigma$ (per iter)')
    handles.append(Line2D([0], [0], color='gray', ls='--', lw=1.0))
    labels_.append(r'$\mu$ over window')
    handles.append(Line2D([0], [0], marker='s', color='gray', ls='',
                          markersize=4, mec='black', mew=0.3))
    labels_.append(r'$\mu \pm 2\sigma$')
    if has_targets:
        handles.append(Line2D([0], [0], marker='*', color='gray', ls='',
                              markersize=9, mec='black', mew=0.4))
        labels_.append(r'target$-$neo')
    ax.legend(handles, labels_, loc='upper right', prop={'size': 9}, framealpha=0.85)


def _column_title_for_chunk(chunk, base_iter):
    if chunk:
        return (f"ev{chunk[0]}\u2013ev{chunk[-1]}"
                if chunk[0] != chunk[-1] else f"ev{chunk[0]}")
    return f"ev{base_iter} only"


def _apply_row_clamp(axs_row, trace_means, trace_count, factor=2.5, target_vals=None):
    '''Clamp the row's y-axis using factor-scaled mean+/-2sigma bounds so
    transient peaks don't dominate while each trace's uncertainty bar stays
    inside the frame:

        ymax = factor * max(mean + 2*std)
        ymin = min(drawn-data bottom across the row, factor * min(mean - 2*std))

    The bottom is never pinned to zero: it follows the lowest drawn artist
    (raw traces included, via each axis' dataLim) so negative excursions
    (e.g. Gamma_e dipping below 0 in the saturated phase) stay visible.
    The data-driven bottom is capped at -|ymax| so a deep negative transient
    can't squash the row — the symmetric mirror of the factor rule cutting
    positive transients at the top. The stats-based bound still extends
    below the cap when the windowed mean-2*sigma is genuinely negative.

    `axs_row` is the full row of (sharey) axes; limits are set on the
    leftmost and propagate. `trace_means` is a list of (mean, std) tuples
    from _draw_chunk_cell. `factor` is exposed as a kwarg (default 2.5) so
    callers can widen / tighten the headroom without editing.

    `target_vals` (flat list of turbulence-only target values drawn as
    stars) only ever *extends* the limits, with a modest 15% margin —
    enough to keep every star in frame without granting targets the full
    factor-scaled headroom reserved for the trace statistics.

    Only skip when there are literally zero traces or no valid means.'''
    if trace_count < 1 or not trace_means:
        return
    uppers = [m + 2.0 * s for (m, s) in trace_means]
    lowers = [m - 2.0 * s for (m, s) in trace_means]
    ymax = factor * max(uppers)
    data_bottoms = [ax.dataLim.ymin for ax in axs_row if np.isfinite(ax.dataLim.ymin)]
    data_floor = max(min(data_bottoms), -abs(ymax)) if data_bottoms else 0.0
    ymin = min(factor * min(lowers), data_floor)
    finite_targets = [t for t in (target_vals or []) if np.isfinite(t)]
    if finite_targets:
        max_t, min_t = max(finite_targets), min(finite_targets)
        ymax = max(ymax, max_t * (1.15 if max_t > 0 else 0.85))
        ymin = min(ymin, min_t * (0.85 if min_t > 0 else 1.15))
    if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin:
        axs_row[0].set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# Top-level plot entry points
# ---------------------------------------------------------------------------


def plot_time_traces_per_radius(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    sources_per_iter=None,
    base_iter=0,
    title_prefix="CGYRO time traces",
    factor=2.5,
    targets_per_iter=None,
    time_mode="local",
):
    '''
    Build one FigureNotebook tab per rho. Rows = transport channels
    (Qe, Qi, Ge), columns = iteration chunks. `base_iter` is drawn in
    every column as the shared baseline; non-base iterations fill each
    column with a column-local blue->red gradient.

    When the driver used CGYRO's warm-start feature each iteration after
    base resets its clock at t=0; time axes are re-aligned via the
    per-(rho, iter) parent map persisted as restart_sources.json by
    `_resolve_cgyro_restart_chain`. The parent map is loaded by
    `load_restart_sources_for_iterations` and passed in as
    `sources_per_iter`. All three restart modes ("first", "all", "best")
    flow through the same offset walker — modes only differ in the shape
    of the per-rho parent map.

    If `sources_per_iter` is empty (no JSON found, including older runs),
    every iter's offset is 0 and the figure renders without restart
    alignment — equivalent to the legacy `restart_mode="none"`.

    Per-trace stats overlays: tinted axvspan window, raw trace, square
    errorbar marker at trace end for mean +/- 2*sigma. Base iter also
    gets a dashed mean line across its window. Per-row y-clamp tight to
    first-sample / mean+2*sigma bounds (skipped when <3 traces).
    '''
    if not tools_by_iteration:
        print("\t- No CGYRO time-trace data available across iterations; skipping CGYRO tabs", typeMsg='w')
        return

    sources_per_iter = sources_per_iter or {}
    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    non_base_its, chunks = _chunk_iterations(sorted_its, base_iter)
    n_cols = len(chunks)
    xlabel_suffix = _xlabel_suffix_from_sources(sources_per_iter) if time_mode == "chain" else " (per evaluation)"
    restart_label = _restart_label_from_sources(sources_per_iter) or "none"

    for r_idx, rho in enumerate(rhos):
        # Per-radius tabs all share one color so they read as a visual
        # group in the notebook (per-channel pivot uses a different single
        # color via the companion plot_time_traces_per_channel).
        fig = fn.add_figure(
            label=f"CGYRO traces (rho={float(rho):.3f})",
            tab_color=fn_color_start,
        )
        axs = fig.subplots(nrows=len(_CHANNELS), ncols=n_cols, squeeze=False, sharex=True, sharey='row')
        fig.set_size_inches(max(6.5, 3.8 * n_cols + 1.8), 7.8)

        fig.suptitle(
            f"{title_prefix} at $\\rho={float(rho):.3f}$  "
            f"(restart_mode={restart_label!r}; {len(non_base_its)} non-base iter"
            f"{'' if len(non_base_its) == 1 else 's'})",
            fontsize=11,
        )

        offsets = _compute_offsets_for_rho(rho, r_idx, sources_per_iter, cache, sorted_its) if time_mode == "chain" else {}
        base_out = pick_output_for_rho(cache[base_iter], rho, r_idx) if base_iter in cache else None
        base_has_window = base_out is not None and getattr(base_out, 'tmin', None) is not None

        # Per-row aggregators — rows are channels here.
        row_trace_means = {row_idx: [] for row_idx in range(len(_CHANNELS))}
        row_trace_counts = {row_idx: 0 for row_idx in range(len(_CHANNELS))}
        row_target_vals = {row_idx: [] for row_idx in range(len(_CHANNELS))}

        for c_idx, chunk in enumerate(chunks):
            col_axes = axs[:, c_idx]
            color_for = _make_column_color_fn(chunk)

            for row_idx, (var, ylabel) in enumerate(_CHANNELS):
                tc, tm, tv = _draw_chunk_cell(
                    col_axes[row_idx], var, rho, r_idx, chunk,
                    cache, base_out, offsets, color_for, base_iter,
                    targets_per_iter=targets_per_iter,
                )
                row_trace_counts[row_idx] += tc
                row_trace_means[row_idx].extend(tm)
                row_target_vals[row_idx].extend(tv)

            col_axes[0].set_title(_column_title_for_chunk(chunk, base_iter), fontsize=10)
            _column_legend(col_axes[0], chunk, color_for, base_iter, base_has_window,
                           has_targets=bool(targets_per_iter))

            for row_idx, (var, ylabel) in enumerate(_CHANNELS):
                ax = col_axes[row_idx]
                if c_idx == 0:
                    ax.set_ylabel(ylabel)
                if row_idx == len(_CHANNELS) - 1:
                    ax.set_xlabel("$t \\, c_s/a$" + xlabel_suffix)
                GRAPHICStools.addDenseAxis(ax)

        for row_idx in range(len(_CHANNELS)):
            _apply_row_clamp(axs[row_idx, :], row_trace_means[row_idx],
                             row_trace_counts[row_idx], factor=factor,
                             target_vals=row_target_vals[row_idx])

        # Extend the shared x-axis past the last trace so the upper-right
        # legend sits over empty space instead of covering data.
        t_max = 0.0
        for it in sorted_its:
            out = pick_output_for_rho(cache[it], rho, r_idx)
            if out is not None and hasattr(out, "t") and len(out.t) > 0:
                t_max = max(t_max, offsets.get(it, 0.0) + float(out.t[-1]))
        if t_max > 0.0:
            axs[0, 0].set_xlim(-0.02 * t_max, t_max * (1.0 + _XLIM_LEGEND_EXTEND))


def plot_time_traces_per_channel(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    sources_per_iter=None,
    base_iter=0,
    title_prefix="CGYRO time traces",
    factor=2.5,
    targets_per_iter=None,
    time_mode="local",
):
    '''
    Companion to `plot_time_traces_per_radius` with the axes pivoted:
    one FigureNotebook tab per channel (Qe, Qi, Ge), rows = rhos, columns
    = iteration chunks (same chunking rule). Handy when the interesting
    view is "how did Qi evolve at every radius over the PORTALS
    iterations" rather than "what's happening at rho=0.5 across channels".

    Restart-aware time alignment is identical to plot_time_traces_per_radius:
    `sources_per_iter` carries the per-(rho, iter) parent map loaded from
    each iteration's restart_sources.json. Empty / missing -> no offset.
    Per-cell semantics, color palette, and per-row y-clamp are identical
    to plot_time_traces_per_radius — rows just carry a different meaning
    (the rho value) so the clamp is now per-(channel, rho) rather than
    per-(rho, channel).
    '''
    if not tools_by_iteration:
        print("\t- No CGYRO time-trace data available across iterations; skipping CGYRO per-channel tabs", typeMsg='w')
        return

    sources_per_iter = sources_per_iter or {}
    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    non_base_its, chunks = _chunk_iterations(sorted_its, base_iter)
    n_cols = len(chunks)
    xlabel_suffix = _xlabel_suffix_from_sources(sources_per_iter) if time_mode == "chain" else " (per evaluation)"
    restart_label = _restart_label_from_sources(sources_per_iter) or "none"

    rho_list = list(rhos)
    n_rows = len(rho_list)

    for v_idx, (var, ylabel) in enumerate(_CHANNELS):
        # Per-channel tabs all share one color, mirroring the per-radius
        # grouping (see plot_time_traces_per_radius). Caller picks a
        # distinct fn_color_start so the two groups stay distinguishable.
        fig = fn.add_figure(
            label=f"CGYRO traces ({var})",
            tab_color=fn_color_start,
        )
        axs = fig.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, sharex=True, sharey='row')
        fig.set_size_inches(max(6.5, 3.8 * n_cols + 1.8), max(3.0, 2.4 * n_rows + 1.2))

        fig.suptitle(
            f"{title_prefix} - {ylabel.split(' [')[0]}  "
            f"(restart_mode={restart_label!r}; {len(non_base_its)} non-base iter"
            f"{'' if len(non_base_its) == 1 else 's'})",
            fontsize=11,
        )

        # Per-row aggregators — rows are rhos here. Each rho has its own
        # offsets / base_out because those depend on the rho's own CGYRO
        # output.
        row_trace_means = {row_idx: [] for row_idx in range(n_rows)}
        row_trace_counts = {row_idx: 0 for row_idx in range(n_rows)}
        row_target_vals = {row_idx: [] for row_idx in range(n_rows)}

        # Pre-resolve per-rho offsets and base_out once (shared across columns).
        offsets_per_rho = {
            r_idx: (_compute_offsets_for_rho(rho_list[r_idx], r_idx, sources_per_iter, cache, sorted_its)
                    if time_mode == "chain" else {})
            for r_idx in range(n_rows)
        }
        base_out_per_rho = {
            r_idx: (pick_output_for_rho(cache[base_iter], rho_list[r_idx], r_idx) if base_iter in cache else None)
            for r_idx in range(n_rows)
        }
        base_has_window_any = any(
            bo is not None and getattr(bo, 'tmin', None) is not None
            for bo in base_out_per_rho.values()
        )

        for c_idx, chunk in enumerate(chunks):
            col_axes = axs[:, c_idx]
            color_for = _make_column_color_fn(chunk)

            for row_idx in range(n_rows):
                tc, tm, tv = _draw_chunk_cell(
                    col_axes[row_idx], var,
                    rho_list[row_idx], row_idx, chunk,
                    cache, base_out_per_rho[row_idx], offsets_per_rho[row_idx],
                    color_for, base_iter,
                    targets_per_iter=targets_per_iter,
                )
                row_trace_counts[row_idx] += tc
                row_trace_means[row_idx].extend(tm)
                row_target_vals[row_idx].extend(tv)

            col_axes[0].set_title(_column_title_for_chunk(chunk, base_iter), fontsize=10)
            _column_legend(col_axes[0], chunk, color_for, base_iter, base_has_window_any,
                           has_targets=bool(targets_per_iter))

            for row_idx in range(n_rows):
                ax = col_axes[row_idx]
                if c_idx == 0:
                    ax.set_ylabel(f"$\\rho={float(rho_list[row_idx]):.3f}$  {ylabel}")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("$t \\, c_s/a$" + xlabel_suffix)
                GRAPHICStools.addDenseAxis(ax)

        for row_idx in range(n_rows):
            _apply_row_clamp(axs[row_idx, :], row_trace_means[row_idx],
                             row_trace_counts[row_idx], factor=factor,
                             target_vals=row_target_vals[row_idx])

        # Extend the shared x-axis past the last trace so the upper-right
        # legend sits over empty space instead of covering data. Max over
        # all rows since x is shared figure-wide.
        t_max = 0.0
        for row_idx in range(n_rows):
            offs = offsets_per_rho[row_idx]
            for it in sorted_its:
                out = pick_output_for_rho(cache[it], rho_list[row_idx], row_idx)
                if out is not None and hasattr(out, "t") and len(out.t) > 0:
                    t_max = max(t_max, offs.get(it, 0.0) + float(out.t[-1]))
        if t_max > 0.0:
            axs[0, 0].set_xlim(-0.02 * t_max, t_max * (1.0 + _XLIM_LEGEND_EXTEND))


# ---------------------------------------------------------------------------
# Overview figures: every evaluation at once
# ---------------------------------------------------------------------------

def _iteration_colors(sorted_its):
    '''Evaluation index -> color on a perceptually ordered map, plus the mappable for a colorbar.'''
    import matplotlib.cm as cm
    norm = Normalize(vmin=min(sorted_its), vmax=max(sorted_its) if max(sorted_its) > min(sorted_its) else min(sorted_its) + 1)
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    return (lambda it: sm.to_rgba(it)), sm


def _target_for(targets_per_iter, it, var, r_idx):
    '''Turbulence-only target (target - neoclassical, GB) of one iteration/channel/radius, or None.'''
    gb_key = _CHANNEL_TO_GB.get(var)
    if not targets_per_iter or gb_key is None:
        return None
    arr = (targets_per_iter.get(it) or {}).get(gb_key)
    if arr is None or r_idx >= len(arr):
        return None
    val = float(arr[r_idx])
    return val if np.isfinite(val) else None


def _robust_ylim(ax, means, stds, targets, pad=0.25):
    '''
    Frame the saturated windows, not the startup transients: the limits come from the window
    means +/- 2 sigma (and the targets), so one iteration's initial overshoot cannot flatten
    every other trace.
    '''
    vals = [m + 2.0 * s for m, s in zip(means, stds)] + [m - 2.0 * s for m, s in zip(means, stds)] + list(targets)
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 2:
        return
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or (abs(hi) or 1.0)
    ax.set_ylim(min(lo - pad * span, 0.0) if lo >= 0 else lo - pad * span, hi + pad * span)


def plot_time_traces_overview(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    sources_per_iter=None,
    base_iter=0,
    title_prefix="CGYRO time traces",
    targets_per_iter=None,
    chained_time=True,
):
    '''
    One figure with everything: rows = channels (Qe, Qi, Ge), columns = radii, every evaluation
    drawn in the same cell and colored by evaluation index (colorbar on the right).

    With `chained_time` (default) the x axis is the warm-start time: each evaluation is offset by
    the simulated time its restart parent had already accumulated (restart_sources.json), so a
    trace continues where its parent stopped and the axis reads as the total time invested at that
    radius. The window mean of every evaluation is drawn as a short horizontal bar, and the
    turbulence-only target of the last evaluation as a dashed line, so the approach to flux match
    is visible across the whole run.
    '''
    if not tools_by_iteration:
        return
    sources_per_iter = sources_per_iter or {}
    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    color_for, sm = _iteration_colors(sorted_its)

    fig = fn.add_figure(label="CGYRO traces (all)", tab_color=fn_color_start)
    # No shared y across columns: the flux scale changes by an order of magnitude between the
    # inner and outer radii, so a shared axis would flatten every inner-radius cell
    axs = fig.subplots(nrows=len(_CHANNELS), ncols=len(rhos), squeeze=False, sharex=True)
    fig.set_size_inches(max(9.0, 3.2 * len(rhos)), 8.0)
    fig.suptitle(
        f"{title_prefix} — all {len(sorted_its)} evaluations"
        f" ({'warm-start (chained) time' if chained_time else 'per-evaluation time'};"
        f" restart_mode={_restart_label_from_sources(sources_per_iter) or 'none'})",
        fontsize=11,
    )

    for r_idx, rho in enumerate(rhos):
        offsets = _compute_offsets_for_rho(rho, r_idx, sources_per_iter, cache, sorted_its) if chained_time else {}
        for row_idx, (var, ylabel) in enumerate(_CHANNELS):
            ax = axs[row_idx, r_idx]
            means, stds, targets = [], [], []
            for it in sorted_its:
                out = pick_output_for_rho(cache[it], rho, r_idx)
                if out is None or not hasattr(out, 't') or getattr(out, var, None) is None:
                    continue
                off = float(offsets.get(it, 0.0))
                c = color_for(it)
                ax.plot(out.t + off, getattr(out, var), color=c, lw=0.5, alpha=0.75,
                        zorder=2 + (it == sorted_its[-1]) * 3)
                m, s, tmin = getattr(out, f"{var}_mean", None), getattr(out, f"{var}_std", None), getattr(out, 'tmin', None)
                if m is not None and tmin is not None:
                    ax.hlines(float(m), float(tmin) + off, float(out.t[-1]) + off, colors=c, lw=1.6, zorder=6)
                    means.append(float(m)); stds.append(float(s) if s is not None else 0.0)
                tval = _target_for(targets_per_iter, it, var, r_idx)
                if tval is not None:
                    targets.append(tval)
            if targets:
                ax.axhline(targets[-1], color='k', ls='--', lw=1.0, alpha=0.8, zorder=7)
            _robust_ylim(ax, means, stds, targets)
            if r_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"$\\rho={float(rho):.3f}$", fontsize=10)
            if row_idx == len(_CHANNELS) - 1:
                ax.set_xlabel("$t \\, c_s/a$" + (" (chained)" if chained_time else ""))
            GRAPHICStools.addDenseAxis(ax)

    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("evaluation")
    axs[0, 0].plot([], [], color='k', ls='--', lw=1.0, label='target $-$ neoclassical (last)')
    axs[0, 0].plot([], [], color='gray', lw=1.6, label='mean over window')
    axs[0, 0].legend(loc='upper right', fontsize=7, framealpha=0.9)


def _tail_fraction(ky, spectrum, tail_start):
    '''
    Share of the flux carried by the high-ky end of the grid: sum|spectrum| over ky >= tail_start*ky_max
    divided by sum|spectrum| over all ky. Absolute values because a particle-flux spectrum changes sign,
    and a signed sum would hide a large tail behind cancellation. A resolved run keeps this small: the
    flux is carried by the modes the box resolves, not by the last bins.
    '''
    ky, spectrum = np.asarray(ky, dtype=float), np.abs(np.asarray(spectrum, dtype=float))
    if ky.size == 0 or spectrum.size != ky.size:
        return None
    total = spectrum.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    return float(spectrum[ky >= tail_start * ky.max()].sum() / total)


def plot_flux_spectra(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    tail_start=0.75,
    title_prefix="CGYRO flux spectra",
    live_iteration=None,
):
    '''
    Is the grid resolving the flux? Rows = channels (Qe, Qi, Ge) as spectra against k_theta*rho_s,
    columns = radii, one line per evaluation (color = evaluation, colorbar on the right). Each
    spectrum is the time average over that evaluation's own saturated window (<q>_ky_mean), with
    +/- sigma shaded for the last evaluation only. The dotted vertical line marks where the
    high-ky tail starts (tail_start*ky_max).

    The bottom row is the tail share per channel against evaluation: the fraction of |flux| carried
    by ky >= tail_start*ky_max. A few percent means the resolved modes carry the flux; a share that
    is large, or that grows as the profiles steepen, means the answer is set by the grid and the run
    needs more ky (or a smaller ky_min) before its fluxes mean anything.
    '''
    if not tools_by_iteration:
        return
    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    color_for, sm = _iteration_colors(sorted_its)

    fig = fn.add_figure(label="CGYRO spectra", tab_color=fn_color_start)
    axs = fig.subplots(nrows=len(_CHANNELS) + 1, ncols=len(rhos), squeeze=False, sharex="row",
                       gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    fig.set_size_inches(max(9.0, 3.2 * len(rhos)), 9.5)
    live_note = f" — evaluation {live_iteration} still running (its window is not final)" if live_iteration is not None else ""
    fig.suptitle(f"{title_prefix} — window-averaged, tail = ky >= {tail_start:.2f} $ky_{{max}}${live_note}", fontsize=11)

    for r_idx, rho in enumerate(rhos):
        tails = {var: ([], []) for var, _ in _CHANNELS}
        for row_idx, (var, ylabel) in enumerate(_CHANNELS):
            ax = axs[row_idx, r_idx]
            ky_max = None
            for it in sorted_its:
                out = pick_output_for_rho(cache[it], rho, r_idx)
                ky, mean = getattr(out, "ky", None), getattr(out, f"{var}_ky_mean", None)
                if out is None or ky is None or mean is None:
                    continue
                ky, mean = np.asarray(ky, dtype=float), np.asarray(mean, dtype=float)
                ky_max = ky.max()
                c = color_for(it)
                ax.plot(ky, mean, color=c, lw=1.0, marker='o', ms=2.0, alpha=0.9)
                if it == sorted_its[-1]:
                    std = getattr(out, f"{var}_ky_std", None)
                    if std is not None:
                        std = np.asarray(std, dtype=float)
                        ax.fill_between(ky, mean - std, mean + std, color=c, alpha=0.2, lw=0)
                frac = _tail_fraction(ky, mean, tail_start)
                if frac is not None:
                    tails[var][0].append(it)
                    tails[var][1].append(100.0 * frac)
            if ky_max is not None:
                ax.axvline(tail_start * ky_max, color='k', ls=':', lw=1.0, alpha=0.7)
            ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
            if r_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"$\\rho={float(rho):.3f}$", fontsize=10)
            GRAPHICStools.addDenseAxis(ax)

        ax = axs[-1, r_idx]
        for (var, _), c in zip(_CHANNELS, ("b", "r", "g")):
            its, vals = tails[var]
            if its:
                ax.plot(its, vals, color=c, marker='o', ms=3.0, lw=1.0, label=var)
        ax.set_ylim(bottom=0)
        ax.set_xticks(sorted_its)
        ax.set_xlim(min(sorted_its) - 0.5, max(sorted_its) + 0.5)
        ax.set_xlabel("evaluation")
        if r_idx == 0:
            ax.set_ylabel(f"tail share [%]")
            ax.legend(loc="best", fontsize=7, framealpha=0.9)
        GRAPHICStools.addDenseAxis(ax)

    for r_idx in range(len(rhos)):
        axs[len(_CHANNELS) - 1, r_idx].set_xlabel("$k_\\theta \\rho_s$")

    if len(sorted_its) > 1:
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("evaluation")


def plot_flux_convergence(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    base_iter=0,
    title_prefix="CGYRO fluxes",
    targets_per_iter=None,
):
    '''
    The question the traces are a diagnostic for: does each channel approach its target as the
    optimizer iterates? Rows = channels, columns = radii; per evaluation the window mean with
    +/- 2 sigma error bars against the evaluation index, over the turbulence-only target (line
    per evaluation, since the target moves with the profiles).
    '''
    if not tools_by_iteration:
        return
    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    color_for, sm = _iteration_colors(sorted_its)

    fig = fn.add_figure(label="CGYRO flux convergence", tab_color=fn_color_start)
    axs = fig.subplots(nrows=len(_CHANNELS), ncols=len(rhos), squeeze=False, sharex=True)
    fig.set_size_inches(max(9.0, 3.2 * len(rhos)), 8.0)
    fig.suptitle(f"{title_prefix} — window mean $\\pm 2\\sigma$ vs evaluation, and the target it chases", fontsize=11)

    for r_idx, rho in enumerate(rhos):
        for row_idx, (var, ylabel) in enumerate(_CHANNELS):
            ax = axs[row_idx, r_idx]
            its, means, stds, tgts, tgt_its = [], [], [], [], []
            for it in sorted_its:
                out = pick_output_for_rho(cache[it], rho, r_idx)
                m = getattr(out, f"{var}_mean", None) if out is not None else None
                if m is not None:
                    its.append(it); means.append(float(m))
                    s = getattr(out, f"{var}_std", None)
                    stds.append(float(s) if s is not None else 0.0)
                tval = _target_for(targets_per_iter, it, var, r_idx)
                if tval is not None:
                    tgts.append(tval); tgt_its.append(it)
            if its:
                ax.errorbar(its, means, yerr=[2.0 * s for s in stds], fmt='o-', ms=3.5, lw=1.0,
                            color='tab:blue', ecolor='tab:blue', elinewidth=0.8, capsize=2, label='CGYRO')
            if tgts:
                ax.plot(tgt_its, tgts, 's--', ms=3.5, lw=1.0, color='k', alpha=0.8, label='target $-$ neoc')
            _robust_ylim(ax, means, stds, tgts)
            if r_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"$\\rho={float(rho):.3f}$", fontsize=10)
            if row_idx == len(_CHANNELS) - 1:
                ax.set_xlabel("evaluation")
            GRAPHICStools.addDenseAxis(ax)
    axs[0, 0].legend(loc='best', fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Live status of a running CGYRO job
# ---------------------------------------------------------------------------

# The minimal read set, plus input.cgyro (PRINT_STEP, MAX_TIME) and .mitim_t0 (simulated time at
# launch). out.cgyro.time goes LAST: copied after bin.cgyro.ky_flux it can only be longer, which
# CGYROoutput._reconcile_time_vector trims.
_LIVE_FILES = [
    "input.cgyro", "input.cgyro.gen", ".mitim_t0",
    "bin.cgyro.geo", "out.cgyro.egrid", "out.cgyro.equilibrium", "out.cgyro.grids", "out.cgyro.hosts",
    "out.cgyro.memory", "out.cgyro.mpi", "out.cgyro.prec", "out.cgyro.rotation", "out.cgyro.startups",
    "out.cgyro.version", "out.cgyro.info", "out.cgyro.timing",
    "bin.cgyro.freq", "bin.cgyro.ky_cflux", "bin.cgyro.ky_flux", "out.cgyro.time",
]


def live_source_from_submission(submission_json):
    '''Where a submitted (run_type submit) CGYRO job runs, from its cgyro_submission.json:
    (machineSettings, scratch folder, [(subfolder, rho), ...]).'''
    meta = json.loads(Path(submission_json).read_text())
    pairs = [(sub, float(rho)) for sub, rhos in meta["kwargs_organize"]["code_executor"].items() for rho in rhos]
    return meta["job"]["machineSettings"], meta["job"]["folderExecution"], pairs


def live_source_from_bash(tmp_folder):
    '''
    Where a run_type normal (bash) CGYRO job runs, when no submission JSON exists: the scratch folder
    is the first `cd` of the execution script staged in tmp_<code>, and the radii are the staged
    <subfolder>/rho_* folders. Only LOCAL scratch is supported this way (the machine is not recorded),
    so this returns None when the folder is not on this filesystem.
    '''
    tmp_folder = Path(tmp_folder)
    for script in sorted(tmp_folder.glob("mitim_bash*.src")) + sorted(tmp_folder.glob("mitim_shell_executor*.sh")):
        for line in script.read_text().splitlines():
            if line.strip().startswith("cd "):
                folder_execution = shlex.split(line.strip())[1]
                if not Path(folder_execution).is_dir():
                    return None
                pairs = sorted((d.parent.name, float(d.name.split("rho_")[-1])) for d in tmp_folder.glob("*/rho_*") if d.is_dir())
                return {"machine": "local"}, folder_execution, pairs
    return None


def fetch_live_outputs(machine_settings, folder_execution, pairs, local_folder, files=None):
    '''
    Copy the in-progress outputs of every radius from the scratch folder (local copy, or SFTP get from a
    remote machine) into local_folder as <file>_<rho:.4f>, the layout of retrieved results, so the
    standard reader applies. Read-only on the scratch side: no tarball, no renames, the job is untouched.
    Returns {rho: {'mtime': last write of out.cgyro.time (epoch s) or None, 'machine': name}}.
    '''
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)
    job = FARMINGtools.mitim_job(local_folder)
    job.machineSettings = machine_settings
    job.connect()
    info = {}
    try:
        for sub, rho in pairs:
            info[rho] = {"machine": machine_settings["machine"], "mtime": None}
            remote = f"{folder_execution}/{sub}/rho_{rho:.4f}"
            for name in (files or _LIVE_FILES):
                src, dst = f"{remote}/{name}", local_folder / f"{name}_{rho:.4f}"
                try:
                    if job.sftp is None:
                        shutil.copyfile(src, dst)
                        mtime = os.stat(src).st_mtime
                    else:
                        job.sftp.get(src, str(dst))
                        mtime = job.sftp.stat(src).st_mtime
                except OSError:
                    continue
                if name == "out.cgyro.time":
                    info[rho]["mtime"] = mtime
    finally:
        job.close()
    return info


# Enough to tell, per radius, how far it got and whether it already ended (EXIT line, watchdog tags)
_STOP_STATUS_FILES = ["input.cgyro", ".mitim_t0", "out.cgyro.info", "mitim_budget.tag", "mitim_discard.tag", "mitim_stop", "out.cgyro.time"]


def live_radii_state(machine_settings, folder_execution, pairs, local_folder):
    '''
    Per-radius state of a running CGYRO job, read from its scratch without touching it:
    {(subfolder, rho): {'t', 't0', 'MAX_TIME', 'exit', 'budget', 'discard', 'stop_requested', 'mtime'}}.
    't' is the last simulated time in out.cgyro.time; 'mtime' its last write (epoch s).
    '''
    local_folder = Path(local_folder)
    state = {}
    for sub, rho in pairs:
        dst = local_folder / sub
        info = fetch_live_outputs(machine_settings, folder_execution, [(sub, rho)], dst, files=_STOP_STATUS_FILES)[rho]
        scal = _read_live_scalars(dst, rho)
        try:
            t = float((dst / f"out.cgyro.time_{rho:.4f}").read_text().split("\n")[-2].split()[0])
        except (OSError, IndexError, ValueError):
            t = None
        state[(sub, rho)] = {
            "t": t, "t0": scal.get("t0"), "MAX_TIME": scal.get("MAX_TIME"), "exit": scal.get("exit"),
            "budget": (dst / f"mitim_budget.tag_{rho:.4f}").exists(),
            "discard": (dst / f"mitim_discard.tag_{rho:.4f}").exists(),
            "stop_requested": (dst / f"mitim_stop_{rho:.4f}").exists(),
            "mtime": info["mtime"],
        }
    return state


def request_stop(machine_settings, folder_execution, pairs, note="mitim_kill_cgyro"):
    '''
    Drop a mitim_stop file in the scratch folder of each (subfolder, rho) in pairs. The watchdog
    around the running launch (CGYRO._wall_budget_wrap) picks it up within ~20 s, waits for the next
    restart write, leaves mitim_budget.tag and stops CGYRO; the radius is then read as finished,
    with its fluxes averaged over what it simulated. Launches started before the watchdog wrapped
    main radii (MITIM older than this function) ignore the file.
    '''
    job = FARMINGtools.mitim_job(Path.cwd())
    job.machineSettings = machine_settings
    job.connect()
    try:
        for sub, rho in pairs:
            path = f"{folder_execution}/{sub}/rho_{rho:.4f}/mitim_stop"
            text = f"{note} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            if job.sftp is None:
                Path(path).write_text(text)
            else:
                with job.sftp.open(path, "w") as f:
                    f.write(text)
            print(f"\t- Stop requested for {sub}/rho_{rho:.4f} ({machine_settings['machine']}:{path})")
    finally:
        job.close()


def _read_live_scalars(folder, rho):
    '''PRINT_STEP, DELTA_T and MAX_TIME from input.cgyro, simulated time at launch (.mitim_t0) and the EXIT line of out.cgyro.info.'''
    out = {}
    try:
        for line in (folder / f"input.cgyro_{rho:.4f}").read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() in ("PRINT_STEP", "DELTA_T", "MAX_TIME") and val.split():
                out[key.strip()] = float(val.split()[0])
    except (OSError, ValueError):
        pass
    try:
        out["t0"] = float((folder / f".mitim_t0_{rho:.4f}").read_text().strip() or 0.0)
    except (OSError, ValueError):
        pass
    try:
        out["exit"] = next((l.strip() for l in (folder / f"out.cgyro.info_{rho:.4f}").read_text().splitlines() if "EXIT" in l), None)
    except OSError:
        pass
    return out


def _fmt_duration(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "?"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def plot_live_status(fn, fn_color, rhos, tool, info, folder, label="CGYRO live", targets_per_iter=None, it=None):
    '''
    Status of a CGYRO job still running: rows = channels (Qe, Qi, Ge) + wall-clock timing, columns = radii.

    Flux rows: trace vs simulated time with the current window mean +/- std (the run's own averaging
    settings) and the turbulence-only target (dashed) when the evaluation already wrote it.
    Timing row: wall seconds per time step = TOTAL of each out.cgyro.timing row / PRINT_STEP (CGYRO
    writes one row per print interval and zeroes its timers after it), and the two sections that cost
    the most over the run. Its title gives the latest s/step and the wall time left to reach MAX_TIME,
    counted from .mitim_t0 (MAX_TIME is additional on warm starts). The column title gives the last
    simulated time and how long ago out.cgyro.time was written (a stall shows up there).
    '''
    fig = fn.add_figure(label=label, tab_color=fn_color)
    axs = fig.subplots(nrows=len(_CHANNELS) + 1, ncols=len(rhos), squeeze=False, sharex="col")
    fig.set_size_inches(max(9.0, 3.2 * len(rhos)), 9.5)
    now = time.time()
    etas = {}
    wall_peaks = []   # last row is wall s per a/cs: comparable across radii, so it shares one y axis

    for r_idx, rho in enumerate(rhos):
        out = pick_output_for_rho(tool, rho, r_idx) if tool is not None else None
        live = {**info.get(rho, {}), **_read_live_scalars(folder, rho)}
        has_t = out is not None and getattr(out, "t", None) is not None and len(out.t) > 0

        age = now - live["mtime"] if live.get("mtime") is not None else None
        status = "exited" if live.get("exit") else (f"{_fmt_duration(age)} ago" if age is not None else "no output yet")
        axs[0, r_idx].set_title(f"$\\rho={float(rho):.3f}$" + (f"  t={out.t[-1]:.1f} $a/c_s$" if has_t else "") + f"\n(last output {status})", fontsize=9)

        for row_idx, (var, ylabel) in enumerate(_CHANNELS):
            ax = axs[row_idx, r_idx]
            if has_t and getattr(out, var, None) is not None:
                ax.plot(out.t, getattr(out, var), color="b", lw=0.6)
                m, sd, tmin = getattr(out, f"{var}_mean", None), getattr(out, f"{var}_std", None), getattr(out, "tmin", None)
                if m is not None and tmin is not None:
                    ax.hlines(float(m), float(tmin), float(out.t[-1]), colors="r", lw=1.6, zorder=6)
                    if sd is not None:
                        ax.fill_between([float(tmin), float(out.t[-1])], float(m) - float(sd), float(m) + float(sd), color="r", alpha=0.15, lw=0)
            tval = _target_for(targets_per_iter, it, var, r_idx)
            if tval is not None:
                ax.axhline(tval, color="k", ls="--", lw=1.0, alpha=0.8, zorder=7)
            if r_idx == 0:
                ax.set_ylabel(ylabel)
            GRAPHICStools.addDenseAxis(ax)

        ax = axs[-1, r_idx]
        timing = getattr(out, "timing", None) if out is not None else None
        if has_t and timing is not None and len(timing):
            # A timing row covers one output interval: PRINT_STEP time steps = PRINT_STEP*DELTA_T of
            # simulated time (1 a/cs the way MITIM sets it), so the raw row divided by that span is the
            # wall cost of 1 a/cs and reads directly against MAX_TIME.
            dt_out = live["PRINT_STEP"] * live["DELTA_T"] if live.get("PRINT_STEP") and live.get("DELTA_T") else 1.0
            total = out.timing_total
            x = out.t[-len(total):] if len(out.t) >= len(total) else np.arange(len(total))
            ax.plot(x, total / dt_out, color="k", lw=1.0, label="TOTAL")
            share = timing.sum(axis=0)
            for i, c in zip(np.argsort(share)[::-1][:2], ("tab:red", "tab:orange")):
                ax.plot(x, timing[:, i] / dt_out, color=c, lw=0.8, label=f"{out.timing_names[i]} ({100 * share[i] / share.sum():.0f}%)")
            wall_peaks.append(float(np.max(total / dt_out)))
            ax.legend(loc="upper center", ncol=3, fontsize=7, framealpha=0.9, handlelength=1.2, columnspacing=0.8)
            eta = None
            if live.get("MAX_TIME") is not None:
                eta = max(live.get("t0", 0.0) + live["MAX_TIME"] - out.t[-1], 0.0) * total[-1] / dt_out
            etas[rho] = 0.0 if live.get("exit") else eta
            state = "done" if live.get("exit") else _fmt_duration(eta)
            ax.set_title(f"{total[-1] / dt_out:.4g} wall s / 1 $a/c_s$  \u00b7  to MAX_TIME: {state}", fontsize=8)
        if r_idx == 0:
            ax.set_ylabel("wall s / 1 $a/c_s$")
        ax.set_xlabel("$t \\, c_s/a$")
        GRAPHICStools.addDenseAxis(ax)

        # x out to where this radius stops (MAX_TIME is additional to the warm-start time in
        # .mitim_t0), so every column shows how much of its run is already done. Shared per column.
        if live.get("MAX_TIME") is not None:
            ax.set_xlim(0.0, live.get("t0", 0.0) + live["MAX_TIME"])

    # One shared y axis for the timing row: the cost per a/cs is the quantity being compared between
    # radii. Headroom above the curves holds the one-row legend.
    if wall_peaks:
        for c, ax in enumerate(axs[-1]):
            ax.set_ylim(0, 1.45 * max(wall_peaks))
            if c > 0:
                ax.tick_params(labelleft=False)

    # The radii of one evaluation run concurrently, so the evaluation ends with the slowest of them
    known = {r: e for r, e in etas.items() if e is not None}
    if known:
        slowest = max(known, key=known.get)
        left = known[slowest]
        text = "all radii done" if left == 0 else f"evaluation done in {_fmt_duration(left)} (slowest $\\rho$={slowest:.3f})"
        missing = len(rhos) - len(known)
        if missing:
            text += f" + {missing} radius/radii with no timing yet"
        fig.suptitle(f"{label}: {text}", fontsize=11)

    axs[0, 0].plot([], [], color="r", lw=1.6, label="window mean $\\pm\\sigma$")
    if targets_per_iter:
        axs[0, 0].plot([], [], color="k", ls="--", lw=1.0, label="target $-$ neoclassical")
    axs[0, 0].legend(loc="best", fontsize=7, framealpha=0.9)

