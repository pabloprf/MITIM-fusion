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
    Build one FigureNotebook tab per rho with three rows of subplots
    (Qe, Qi, Ge time traces, GB units) overlaying every iteration in
    `tools_by_iteration`. `base_iter` (default 0) is drawn on top in a
    distinct style as the shared baseline reference and is replicated in
    every column so the eye has a consistent anchor.

    Columns chunk the non-base iterations CHUNK_SIZE at a time so panels
    stay readable as iteration count grows; within each column the first
    iter is blue and the last is red.

    When the driver used CGYRO's warm-start feature each iteration after
    base resets its clock at t=0; time axes are re-aligned per
    `restart_mode`:
      - "none"  -> no warm-start; overlay every iter at t=0.
      - "first" -> every ev_N (N != base) restarted from base's final
                   state; all non-base iters share offset = tmax(base)
                   and fan out as parallel branches from that endpoint.
      - "all"   -> chained; iter N restarts from iter N-1, offsets are
                   cumulative over prior iterations' tmax.

    Stat overlays per trace: a tinted axvspan over [out.tmin, out.t[-1]]
    (the window apply_ac used), plus a square errorbar at the trace end
    showing mean +/- 2*std. The base iteration additionally gets a
    dashed mean line so its stats value is obvious. Row y-limits are
    clamped to the tightest range containing every trace's first sample
    and every trace's mean+2*std (so mid-trace transient peaks don't
    dominate the view), *unless* the row has fewer than 3 traces — with
    so few lines the peaks are part of what the user wants to see.
    '''
    if not tools_by_iteration:
        print("\t- No CGYRO time-trace data available across iterations; skipping CGYRO tabs", typeMsg='w')
        return

    cache = tools_by_iteration
    sorted_its = sorted(cache.keys())
    varss = [('Qe', '$Q_e$ [GB]'), ('Qi', '$Q_i$ [GB]'), ('Ge', '$\\Gamma_e$ [GB]')]

    CHUNK_SIZE = 5
    non_base_its = [i for i in sorted_its if i != base_iter]
    chunks = (
        [non_base_its[i:i + CHUNK_SIZE] for i in range(0, len(non_base_its), CHUNK_SIZE)]
        if non_base_its else [[]]
    )
    n_cols = len(chunks)

    # Shared cmap template; each column builds its own Normalize over just
    # its chunk, so column-local first->last reads as blue->red.
    cmap_iter = LinearSegmentedColormap.from_list("iter_bluered", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
    def _make_column_color_fn(chunk):
        if not chunk:
            return lambda it: (0.0, 0.0, 0.0)
        v0, v1 = chunk[0], chunk[-1]
        norm = (Normalize(vmin=v0, vmax=v1) if v0 != v1
                else Normalize(vmin=v0 - 0.5, vmax=v0 + 0.5))
        return lambda it: cmap_iter(norm(it))

    _xlabel_suffix = {
        "all": " (chained)",
        "first": f" (branched from ev{base_iter})",
    }.get(restart_mode, "")

    for r_idx, rho in enumerate(rhos):
        fig = fn.add_figure(label=f"CGYRO traces (rho={float(rho):.3f})", tab_color=fn_color_start + r_idx)
        # squeeze=False keeps axs 2D even with n_cols=1; sharey='row' lets
        # the eye compare the same channel across iteration chunks at the
        # same vertical scale.
        axs = fig.subplots(nrows=len(varss), ncols=n_cols, squeeze=False, sharey='row')
        fig.set_size_inches(max(6.5, 3.8 * n_cols + 1.8), 7.8)

        fig.suptitle(
            f"{title_prefix} at $\\rho={float(rho):.3f}$  "
            f"(restart_mode={restart_mode!r}; {len(non_base_its)} non-base iter"
            f"{'' if len(non_base_its) == 1 else 's'})",
            fontsize=11,
        )

        # Per-iteration time offsets (see restart_mode semantics above).
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

        base_out = pick_output_for_rho(cache[base_iter], rho, r_idx) if base_iter in cache else None

        # Per-row y-limit candidates + trace counters. Clamp each row's
        # y-axis to the tightest interval containing every trace's first
        # sample and every trace's mean+2*std, UNLESS the row has fewer
        # than 3 traces (with <3 lines the transient-peak protection the
        # clamp provides is worth less than just showing everything).
        row_y_candidates = {row_idx: [] for row_idx in range(len(varss))}
        row_trace_counts = {row_idx: 0 for row_idx in range(len(varss))}

        for c_idx, chunk in enumerate(chunks):
            col_axes = axs[:, c_idx]
            _color_for = _make_column_color_fn(chunk)

            # Non-base traces for this column. For each iteration we draw:
            #   1. A tinted shaded band over [out.tmin, out.t[-1]] +offset
            #      (the window apply_ac collapsed into the scalar stat).
            #   2. The raw time trace.
            #   3. An errorbar marker at the trace end showing mean +/- 2*std.
            # out.<Var>_mean / _std come from CGYROutils.apply_ac.
            for it in chunk:
                out = pick_output_for_rho(cache[it], rho, r_idx)
                if out is None or not hasattr(out, "t"):
                    continue
                color = _color_for(it)
                t_shifted = out.t + offsets[it]
                x_end = float(t_shifted[-1])
                tmin_it = getattr(out, 'tmin', None)
                for row_idx, (var, _) in enumerate(varss):
                    y = getattr(out, var, None)
                    if y is None:
                        continue
                    ax = col_axes[row_idx]
                    if tmin_it is not None:
                        ax.axvspan(
                            float(tmin_it) + offsets[it],
                            x_end,
                            color=color, alpha=0.08, zorder=0,
                        )
                    ax.plot(t_shifted, y, color=color, lw=1.0, alpha=0.85, zorder=2)
                    row_trace_counts[row_idx] += 1
                    try:
                        row_y_candidates[row_idx].append(float(y[0]))
                    except (TypeError, IndexError):
                        pass
                    mean_val = getattr(out, f"{var}_mean", None)
                    std_val = getattr(out, f"{var}_std", None)
                    if mean_val is not None and std_val is not None:
                        ax.errorbar(
                            x_end, float(mean_val), yerr=2.0 * float(std_val),
                            fmt='s', color=color, ms=3, capsize=2, lw=0.8,
                            mec='black', mew=0.3, zorder=4,
                        )
                        row_y_candidates[row_idx].append(float(mean_val) + 2.0 * float(std_val))

            # Base iter on top of the gradient in every column — black,
            # slightly thicker. Its own averaging window is shaded gray
            # (neutral reference band) and its mean is drawn as a dashed
            # horizontal line across the window.
            if base_out is not None and hasattr(base_out, "t"):
                base_tmin = getattr(base_out, 'tmin', None)
                base_offset = offsets.get(base_iter, 0.0)
                for row_idx, (var, _) in enumerate(varss):
                    y = getattr(base_out, var, None)
                    if y is None:
                        continue
                    ax = col_axes[row_idx]
                    ax.plot(base_out.t + base_offset, y, color='black', lw=1.4, alpha=1.0, zorder=5)
                    row_trace_counts[row_idx] += 1
                    try:
                        row_y_candidates[row_idx].append(float(y[0]))
                    except (TypeError, IndexError):
                        pass

                    if base_tmin is not None:
                        ax.axvspan(
                            float(base_tmin) + base_offset,
                            float(base_out.t[-1]) + base_offset,
                            alpha=0.12, color='gray', zorder=0,
                        )

                    mean_val = getattr(base_out, f"{var}_mean", None)
                    std_val = getattr(base_out, f"{var}_std", None)
                    if mean_val is not None and std_val is not None:
                        if base_tmin is not None:
                            ax.hlines(
                                float(mean_val),
                                float(base_tmin) + base_offset,
                                float(base_out.t[-1]) + base_offset,
                                colors='black', linestyles='--', lw=0.8,
                                alpha=0.7, zorder=6,
                            )
                        ax.errorbar(
                            float(base_out.t[-1]) + base_offset,
                            float(mean_val), yerr=2.0 * float(std_val),
                            fmt='s', color='black', ms=5, capsize=3, lw=1.2,
                            zorder=7,
                        )
                        row_y_candidates[row_idx].append(float(mean_val) + 2.0 * float(std_val))

            # Column title shows the iteration range in this column.
            if chunk:
                col_title = (f"ev{chunk[0]}\u2013ev{chunk[-1]}"
                             if chunk[0] != chunk[-1] else f"ev{chunk[0]}")
            else:
                col_title = f"ev{base_iter} only"
            col_axes[0].set_title(col_title, fontsize=10)

            # Compact per-column legend on the top row: base iter + chunk
            # endpoints + markers that explain the averaging window and
            # the right-edge errorbar points.
            handles = [Line2D([0], [0], color='black', lw=1.4)]
            labels_ = [f'ev{base_iter} (base)']
            if chunk:
                first_it, last_it = chunk[0], chunk[-1]
                handles.append(Line2D([0], [0], color=_color_for(first_it), lw=1.5))
                labels_.append(f'ev{first_it}')
                if last_it != first_it:
                    handles.append(Line2D([0], [0], color=_color_for(last_it), lw=1.5))
                    labels_.append(f'ev{last_it}')
            if base_out is not None and getattr(base_out, 'tmin', None) is not None:
                handles.append(Patch(facecolor='gray', alpha=0.3, edgecolor='none'))
                labels_.append(f'avg window (ev{base_iter})')
            if chunk:
                handles.append(Patch(facecolor=_color_for(chunk[-1]), alpha=0.3, edgecolor='none'))
                labels_.append('avg window (per iter)')
            handles.append(Line2D([0], [0], marker='s', color='gray', ls='',
                                  markersize=4, mec='black', mew=0.3))
            labels_.append(r'$\mu \pm 2\sigma$')
            col_axes[0].legend(handles, labels_, loc='best', prop={'size': 6}, framealpha=0.85)

            # Axis decorations: y-label on the leftmost column only;
            # x-label on the bottom row only.
            for row_idx, (var, ylabel) in enumerate(varss):
                ax = col_axes[row_idx]
                if c_idx == 0:
                    ax.set_ylabel(ylabel)
                if row_idx == len(varss) - 1:
                    ax.set_xlabel("$t \\, c_s/a$" + _xlabel_suffix)
                GRAPHICStools.addDenseAxis(ax)

        # Per-row y-axis clamp (see docstring). Skipped when the row has
        # fewer than 3 total traces — there's not enough happening in the
        # panel to justify hiding transient peaks.
        for row_idx in range(len(varss)):
            if row_trace_counts[row_idx] < 3:
                continue
            candidates = row_y_candidates[row_idx]
            if not candidates:
                continue
            y_lo = min(candidates)
            y_hi = max(candidates)
            if y_hi > y_lo:
                pad = 0.05 * (y_hi - y_lo)
                axs[row_idx, 0].set_ylim(y_lo - pad, y_hi + pad)
