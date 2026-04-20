"""
CGYRO-specific plotting helpers that are fully PORTALS-agnostic.

Owned by the CGYRO tools layer so any caller with a set of CGYRO iteration
folders (PORTALS is currently the only one, but the module has no PORTALS
imports) can produce the per-rho Qe/Qi/Ge time-trace figure. Callers handle
their own iteration discovery (folder layout) and namelist lookups (tmin,
restart mode); this module handles CGYRO-specific loading and drawing.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mitim_tools.simulation_tools import SIMtools
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools import GRAPHICStools
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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_tool_for_iteration(folder_execution, rhos, read_kwargs=None):
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
    '''
    base = folder_execution / "base_cgyro"
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
            c.read(folder=base, label="base_cgyro", minimal=True, **read_kwargs)
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


def load_tools_for_iterations(iteration_folders, rhos, read_kwargs=None):
    '''
    Convenience: given an iterable of (iteration_index, folder) pairs,
    return {iteration_index: tool} skipping any iteration whose folder
    fails to load. Non-destructive if some iterations are missing on
    disk — the result is whatever subset was readable.
    '''
    cache = {}
    for it, folder in iteration_folders:
        tool = load_tool_for_iteration(folder, rhos, read_kwargs=read_kwargs)
        if tool is not None:
            cache[it] = tool
    return cache


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


def _xlabel_suffix_for(restart_mode, base_iter):
    return {
        "all": " (chained)",
        "first": f" (branched from ev{base_iter})",
    }.get(restart_mode, "")


def _compute_offsets_for_rho(rho, r_idx, restart_mode, cache, sorted_its, base_iter):
    '''Per-iteration time offsets at this rho (see restart_mode semantics).'''
    offsets = {it: 0.0 for it in sorted_its}
    if restart_mode == "all":
        cumulative = 0.0
        for it in sorted_its:
            offsets[it] = cumulative
            out = pick_output_for_rho(cache[it], rho, r_idx)
            if out is not None and hasattr(out, "t") and len(out.t) > 0:
                cumulative += float(out.t[-1])
    elif restart_mode == "first":
        base_tmax = 0.0
        if base_iter in cache:
            out0 = pick_output_for_rho(cache[base_iter], rho, r_idx)
            if out0 is not None and hasattr(out0, "t") and len(out0.t) > 0:
                base_tmax = float(out0.t[-1])
        for it in sorted_its:
            if it != base_iter:
                offsets[it] = base_tmax
    return offsets


def _draw_chunk_cell(ax, var, rho, r_idx, chunk, cache, base_out, offsets,
                     color_for, base_iter):
    '''
    Draw one subplot: non-base traces in `chunk` + base iter, for channel
    `var` at `rho`. Returns (trace_count, y_candidates) so the outer grid
    owner can run per-row y-clamp.

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
    y_candidates = []
    trace_count = 0

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
        try:
            y_candidates.append(float(y[0]))
        except (TypeError, IndexError):
            pass

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
            y_candidates.append(float(mean_val) + 2.0 * float(std_val))
            y_candidates.append(float(mean_val) - 2.0 * float(std_val))

    # Non-base traces first so the base overlays them.
    for it in chunk:
        out = pick_output_for_rho(cache[it], rho, r_idx)
        if out is None:
            continue
        _draw_single_trace(out, offsets[it], color_for(it), is_base=False)

    if base_out is not None:
        _draw_single_trace(base_out, offsets.get(base_iter, 0.0), color=None, is_base=True)

    return trace_count, y_candidates


def _column_legend(ax, chunk, color_for, base_iter, base_has_window):
    '''Compact legend shared by both grid layouts: base + chunk endpoints,
    the x=window x y=2*sigma shading patches, the dashed mean line, and
    the mean +/- 2*sigma errorbar marker. Keeps readers honest about what
    every visual channel on the plot is encoding.'''
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
    ax.legend(handles, labels_, loc='best', prop={'size': 6}, framealpha=0.85)


def _column_title_for_chunk(chunk, base_iter):
    if chunk:
        return (f"ev{chunk[0]}\u2013ev{chunk[-1]}"
                if chunk[0] != chunk[-1] else f"ev{chunk[0]}")
    return f"ev{base_iter} only"


def _apply_row_clamp(axs_row_leftmost, y_candidates, trace_count):
    '''Clamp the row's y-axis to the tightest interval containing every
    trace's first sample and mean+2*sigma. Skip when <3 traces are on the
    row — the transient-peak protection is worth less than just showing
    everything when there are so few lines.'''
    if trace_count < 3 or not y_candidates:
        return
    y_lo = min(y_candidates)
    y_hi = max(y_candidates)
    if y_hi > y_lo:
        pad = 0.05 * (y_hi - y_lo)
        axs_row_leftmost.set_ylim(y_lo - pad, y_hi + pad)


# ---------------------------------------------------------------------------
# Top-level plot entry points
# ---------------------------------------------------------------------------


def plot_time_traces_per_radius(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    restart_mode="none",
    base_iter=0,
    title_prefix="CGYRO time traces",
):
    '''
    Build one FigureNotebook tab per rho. Rows = transport channels
    (Qe, Qi, Ge), columns = iteration chunks. `base_iter` is drawn in
    every column as the shared baseline; non-base iterations fill each
    column with a column-local blue->red gradient.

    When the driver used CGYRO's warm-start feature each iteration after
    base resets its clock at t=0; time axes are re-aligned per
    `restart_mode`:
      - "none"  -> no warm-start; overlay every iter at t=0.
      - "first" -> every ev_N (N != base) restarted from base's final
                   state; all non-base iters share offset = tmax(base)
                   and fan out as parallel branches from that endpoint.
      - "all"   -> chained; iter N restarts from iter N-1, offsets are
                   cumulative over prior iterations' tmax.

    Per-trace stats overlays: tinted axvspan window, raw trace, square
    errorbar marker at trace end for mean +/- 2*sigma. Base iter also
    gets a dashed mean line across its window. Per-row y-clamp tight to
    first-sample / mean+2*sigma bounds (skipped when <3 traces).
    '''
    if not tools_by_iteration:
        print("\t- No CGYRO time-trace data available across iterations; skipping CGYRO tabs", typeMsg='w')
        return

    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    non_base_its, chunks = _chunk_iterations(sorted_its, base_iter)
    n_cols = len(chunks)
    xlabel_suffix = _xlabel_suffix_for(restart_mode, base_iter)

    for r_idx, rho in enumerate(rhos):
        fig = fn.add_figure(
            label=f"CGYRO traces (rho={float(rho):.3f})",
            tab_color=fn_color_start + r_idx,
        )
        axs = fig.subplots(nrows=len(_CHANNELS), ncols=n_cols, squeeze=False, sharex=True, sharey='row')
        fig.set_size_inches(max(6.5, 3.8 * n_cols + 1.8), 7.8)

        fig.suptitle(
            f"{title_prefix} at $\\rho={float(rho):.3f}$  "
            f"(restart_mode={restart_mode!r}; {len(non_base_its)} non-base iter"
            f"{'' if len(non_base_its) == 1 else 's'})",
            fontsize=11,
        )

        offsets = _compute_offsets_for_rho(rho, r_idx, restart_mode, cache, sorted_its, base_iter)
        base_out = pick_output_for_rho(cache[base_iter], rho, r_idx) if base_iter in cache else None
        base_has_window = base_out is not None and getattr(base_out, 'tmin', None) is not None

        # Per-row aggregators — rows are channels here.
        row_y_candidates = {row_idx: [] for row_idx in range(len(_CHANNELS))}
        row_trace_counts = {row_idx: 0 for row_idx in range(len(_CHANNELS))}

        for c_idx, chunk in enumerate(chunks):
            col_axes = axs[:, c_idx]
            color_for = _make_column_color_fn(chunk)

            for row_idx, (var, ylabel) in enumerate(_CHANNELS):
                tc, yc = _draw_chunk_cell(
                    col_axes[row_idx], var, rho, r_idx, chunk,
                    cache, base_out, offsets, color_for, base_iter,
                )
                row_trace_counts[row_idx] += tc
                row_y_candidates[row_idx].extend(yc)

            col_axes[0].set_title(_column_title_for_chunk(chunk, base_iter), fontsize=10)
            _column_legend(col_axes[0], chunk, color_for, base_iter, base_has_window)

            for row_idx, (var, ylabel) in enumerate(_CHANNELS):
                ax = col_axes[row_idx]
                if c_idx == 0:
                    ax.set_ylabel(ylabel)
                if row_idx == len(_CHANNELS) - 1:
                    ax.set_xlabel("$t \\, c_s/a$" + xlabel_suffix)
                GRAPHICStools.addDenseAxis(ax)

        for row_idx in range(len(_CHANNELS)):
            _apply_row_clamp(axs[row_idx, 0], row_y_candidates[row_idx], row_trace_counts[row_idx])


def plot_time_traces_per_channel(
    fn,
    fn_color_start,
    rhos,
    tools_by_iteration,
    restart_mode="none",
    base_iter=0,
    title_prefix="CGYRO time traces",
):
    '''
    Companion to `plot_time_traces_per_radius` with the axes pivoted:
    one FigureNotebook tab per channel (Qe, Qi, Ge), rows = rhos, columns
    = iteration chunks (same chunking rule). Handy when the interesting
    view is "how did Qi evolve at every radius over the PORTALS
    iterations" rather than "what's happening at rho=0.5 across channels".

    Per-cell semantics, color palette, and per-row y-clamp are identical
    to plot_time_traces_per_radius — rows just carry a different meaning
    (the rho value) so the clamp is now per-(channel, rho) rather than
    per-(rho, channel).
    '''
    if not tools_by_iteration:
        print("\t- No CGYRO time-trace data available across iterations; skipping CGYRO per-channel tabs", typeMsg='w')
        return

    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    non_base_its, chunks = _chunk_iterations(sorted_its, base_iter)
    n_cols = len(chunks)
    xlabel_suffix = _xlabel_suffix_for(restart_mode, base_iter)

    rho_list = list(rhos)
    n_rows = len(rho_list)

    for v_idx, (var, ylabel) in enumerate(_CHANNELS):
        fig = fn.add_figure(
            label=f"CGYRO traces ({var})",
            tab_color=fn_color_start + v_idx,
        )
        axs = fig.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, sharex=True, sharey='row')
        fig.set_size_inches(max(6.5, 3.8 * n_cols + 1.8), max(3.0, 2.4 * n_rows + 1.2))

        fig.suptitle(
            f"{title_prefix} - {ylabel.split(' [')[0]}  "
            f"(restart_mode={restart_mode!r}; {len(non_base_its)} non-base iter"
            f"{'' if len(non_base_its) == 1 else 's'})",
            fontsize=11,
        )

        # Per-row aggregators — rows are rhos here. Each rho has its own
        # offsets / base_out because those depend on the rho's own CGYRO
        # output.
        row_y_candidates = {row_idx: [] for row_idx in range(n_rows)}
        row_trace_counts = {row_idx: 0 for row_idx in range(n_rows)}

        # Pre-resolve per-rho offsets and base_out once (shared across columns).
        offsets_per_rho = {
            r_idx: _compute_offsets_for_rho(rho_list[r_idx], r_idx, restart_mode, cache, sorted_its, base_iter)
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
                tc, yc = _draw_chunk_cell(
                    col_axes[row_idx], var,
                    rho_list[row_idx], row_idx, chunk,
                    cache, base_out_per_rho[row_idx], offsets_per_rho[row_idx],
                    color_for, base_iter,
                )
                row_trace_counts[row_idx] += tc
                row_y_candidates[row_idx].extend(yc)

            col_axes[0].set_title(_column_title_for_chunk(chunk, base_iter), fontsize=10)
            _column_legend(col_axes[0], chunk, color_for, base_iter, base_has_window_any)

            for row_idx in range(n_rows):
                ax = col_axes[row_idx]
                if c_idx == 0:
                    ax.set_ylabel(f"$\\rho={float(rho_list[row_idx]):.3f}$  {ylabel}")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("$t \\, c_s/a$" + xlabel_suffix)
                GRAPHICStools.addDenseAxis(ax)

        for row_idx in range(n_rows):
            _apply_row_clamp(axs[row_idx, 0], row_y_candidates[row_idx], row_trace_counts[row_idx])
