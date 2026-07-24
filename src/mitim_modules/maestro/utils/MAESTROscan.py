"""
Interpretation of MAESTRO parameter scans: a folder of ``case_*`` runs (as produced by
run_maestro_scan-style launchers) is turned into

  1. Seed-spread violin panels of performance scalars vs any scanned input, where the
     VIOLIN AT EACH POINT IS THE SEED SPREAD ONLY (genuine run-to-run scatter at one
     fixed operating point). Deterministic scan inputs (neped, fGped, nsep/nped, ...)
     are NEVER pooled into the spread: one goes on the x-axis and another becomes the
     x-dodged color series. A line joins the seed means of each series across x, and
     reference/benchmark runs are overlaid as marked points (placed by interpolation
     when their input value falls between the scanned grid values).
  2. Per-beat evolution traces of selected quantities, overplotted for all cases.
  3. Cumulative wall time along the beat chain (mean over seeds, max-min errorbars),
     from each case's ``Outputs/Performance/timing.jsonl``.
  4. A compiled PDF report: the summary figures first, then per-case MAESTRO "special"
     tab (from ``maestro_plots/``) and ``Outputs/maestro_summary.md``.

Case-name convention (matches the scan launchers): ``case_<tag>_<p1><v1>_<p2><v2>_..._seed<N>``,
e.g. ``case_geqdsk_nsep0.40_neped2.10_seed0`` or ``case_case7_nsep0.40_fG0.80_seed3``.
The first token after ``case_`` is the tag (machine / init method); the remaining
tokens are parsed as <name><value> scan parameters; ``seed`` is always split off.

Performance scalars are read from each case's ``Outputs/input.gacode_final`` (the final
converged MAESTRO plasma state) via ``gacode_state.derive_quantities()``; beat traces
from ``Beats/Beat_<n>/beat_results/input.gacode``.

Typical use (see also the mitim_plot_maestro_scan CLI):

    from mitim_modules.maestro.utils.MAESTROscan import maestro_scan
    scan = maestro_scan(folder)
    overlay = scan.benchmark_overlay(v3a_folder, x_value=0.766, series_value=0.4,
                                     label='V3A benchmark')
    scan.plot_performance(x='fG', series='nsep', overlays=[overlay])
    scan.plot_beat_evolution(color_by='fG')
    scan.plot_beat_timing(color_by='fG', panel_by=['nsep'])
    scan.compile_report()
"""

import re
import json
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# Default performance scalars to summarize: (derived key, axis label). All are scalars
# in gacode_state.derived (see MITIMstate.derive_quantities).
METRICS = [
    ("Pfus",  r"$P_{fus}$ [MW]"),
    ("fG",    r"$f_{G}$"),
    ("Q",     r"$Q$"),
    ("Wthr",  r"$W_{th}$ [MJ]"),
    ("BetaN", r"$\beta_N$"),
    ("H98",   r"$H_{98,y2}$"),
    ("Psol",  r"$P_{sol}$ [MW]"),
    ("Prad",  r"$P_{rad}$ [MW]"),
    ("q95",   r"$q_{95}$"),
]

# Fixed reference lines + guaranteed axis ranges for the headline panels:
# {metric: (dashed hline, minimum top-of-axis)}
REFERENCE_LINES = {
    "Pfus": (1100.0, 1200.0),
    "fG":   (1.0, 1.2),
}


def _pfus_masked(p):
    """Pfus [MW], blanked to NaN below ~1 kW (non-fusion beats are numerical noise),
    matching MAESTROplot.plot_special_quantities."""
    pfus = float(p.derived["Pfus"])
    return np.nan if pfus < 1e-3 else pfus


def _p_edge(p):
    """Thermal pressure at rho=0.9 [MPa] (the 'p(rho=0.9)' edge trace in the special tab)."""
    return float(np.interp(0.9, p.profiles["rho(-)"], p.derived["pthr_manual"]))


# Default per-beat quantities for the beat-evolution traces: (key, axis label, extractor).
# These mirror the corresponding curves in MAESTROplot.plot_special_quantities.
BEAT_QUANTITIES = [
    ("Pfus",       r"$P_{fus}$ [MW]",             _pfus_masked),
    ("BetaN",      r"$\beta_N$ (engineering)",    lambda p: float(p.derived["BetaN_engineering"])),
    ("ne_peaking", r"$\nu_{ne}$ (ne peaking)",    lambda p: float(p.derived["ne_peaking0.2"])),
    ("p_edge",     r"$p_{th}(\rho{=}0.9)$ [MPa]", _p_edge),
]

# Default profile rows for plot_profiles: (key, axis label, extractor -> (rho, values),
# core_ylim). core_ylim=True bounds the row's y-axis by the rho <= 0.9 values, so the
# pedestal/edge gradient spike does not squash the core structure the seeds differ in.
PROFILE_QUANTITIES = [
    ("Ti",   r"$T_i$ [keV]",                lambda p: (p.profiles["rho(-)"], p.profiles["ti(keV)"][:, 0]), False),
    ("ne",   r"$n_e$ [$10^{20}$ m$^{-3}$]", lambda p: (p.profiles["rho(-)"], p.profiles["ne(10^19/m^3)"] * 0.1), False),
    ("aLTi", r"$a/L_{T_i}$",                lambda p: (p.profiles["rho(-)"], p.derived["aLTi"][:, 0]), True),
    ("aLne", r"$a/L_{n_e}$",                lambda p: (p.profiles["rho(-)"], p.derived["aLne"]), True),
]

# timing.jsonl labels look like "Beat #3 (portals) - Run + Finalization"
_TIMING_RE = re.compile(r"Beat\s*#\s*(\d+)\s*\((\w+)\)")
_PARAM_RE = re.compile(r"^([A-Za-z]+)([0-9]+(?:\.[0-9]+)?)$")
_BEAT_DIR_RE = re.compile(r"Beat_(\d+)$")


class maestro_scan:
    """A folder of MAESTRO ``case_*`` runs, exposed for scan-level analysis/plotting."""

    def __init__(self, main_folder, metrics=None, beat_quantities=None, param_labels=None):
        self.main_folder = IOtools.expandPath(main_folder)
        self.metrics = metrics or METRICS
        self.beat_quantities = beat_quantities or BEAT_QUANTITIES
        # axis-label overrides per scan parameter, e.g. {'neped': r"$n_{e,ped}$ [$10^{20}$ m$^{-3}$]"}
        self.param_labels = param_labels or {}

        self.cases = self._discover_cases()
        if not self.cases:
            print(f"No case_* folders found under {self.main_folder}", typeMsg="w")
        self.tags = sorted({c["tag"] for c in self.cases})
        self.scan_params = sorted({k for c in self.cases for k in c["params"]})
        self.values = {p: sorted({c["params"][p] for c in self.cases if p in c["params"]})
                       for p in self.scan_params}

        self._performance = None       # cached load_performance() result
        self._out_folder = self.main_folder / "interpretation"
        self._summary_figs = []        # figures accumulated for compile_report

    # --------------------------------------------------------------------------------------------
    # Discovery / loading
    # --------------------------------------------------------------------------------------------

    def _discover_cases(self):
        """Parse every ``case_*`` subfolder name into {tag, params{...}, seed, folder}."""
        cases = []
        for folder in sorted(self.main_folder.glob("case_*")):
            if not folder.is_dir():
                continue
            tokens = folder.name.split("_")[1:]     # drop the 'case' prefix
            tag_tokens, params = [], {}
            for k, tok in enumerate(tokens):
                m = _PARAM_RE.match(tok) if k > 0 else None   # first token is ALWAYS the tag
                if m:
                    params[m.group(1)] = float(m.group(2))
                else:
                    tag_tokens.append(tok)
            seed = int(params.pop("seed", -1))
            cases.append(dict(tag="_".join(tag_tokens), params=params, seed=seed,
                              folder=folder, name=folder.name))
        return sorted(cases, key=lambda c: (c["tag"], tuple(sorted(c["params"].items())), c["seed"]))

    def _final_gacode(self, case):
        """Path to the case's final MAESTRO state, or None if it didn't finish."""
        f = case["folder"] / "Outputs" / "input.gacode_final"
        return f if f.exists() else None

    def load_performance(self):
        """Per-case performance records [{tag, params, seed, values{metric: float}}, ...]
        from each case's final state (cached). Unfinished cases are skipped and reported."""
        if self._performance is not None:
            return self._performance
        records, missing = [], []
        for case in self.cases:
            gfile = self._final_gacode(case)
            if gfile is None:
                missing.append(case["name"])
                continue
            p = PROFILEStools.gacode_state(gfile)
            p.derive_quantities()
            records.append(dict(tag=case["tag"], params=case["params"], seed=case["seed"],
                                values={key: float(p.derived[key]) for key, _ in self.metrics}))
        if missing:
            print(f"\t- {len(missing)} case(s) without Outputs/input.gacode_final (skipped): "
                  + ", ".join(missing), typeMsg="w")
        self._performance = records
        return records

    def benchmark_metrics(self, run_folder):
        """METRICS of a single reference MAESTRO run (its Outputs/input.gacode_final),
        or None if not reachable from this machine."""
        gfile = IOtools.expandPath(run_folder) / "Outputs" / "input.gacode_final"
        if not gfile.exists():
            print(f"\t- Benchmark run not found at {run_folder}; skipping its overlay", typeMsg="w")
            return None
        p = PROFILEStools.gacode_state(gfile)
        p.derive_quantities()
        return {key: float(p.derived[key]) for key, _ in self.metrics}

    def benchmark_overlay(self, run_folder, x_value, series_value, label, marker="s"):
        """Build a plot_performance overlay from a reference run folder: its metrics
        placed at ``x_value`` (interpolated between grid values if off-grid), colored
        like the ``series_value`` series. Returns None if the run isn't reachable."""
        bench = self.benchmark_metrics(run_folder)
        if bench is None:
            return None
        return ({key: [(x_value, val)] for key, val in bench.items()}, label, series_value, marker)

    # --------------------------------------------------------------------------------------------
    # Seed-spread violins
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def _series_colors(vals, cmap_name):
        """Map each (ordered) series value to a color from a sequential colormap, so the
        color ordering reflects the input-choice ordering (low -> high)."""
        vs = sorted(vals)
        cmap = plt.get_cmap(cmap_name)
        return {v: cmap(0.15 + 0.7 * k / max(len(vs) - 1, 1)) for k, v in enumerate(vs)}

    def _param_label(self, param):
        return self.param_labels.get(param, param)

    def plot_performance(self, x, series=None, tag=None, save_path=None, overlays=None,
                         cmap="viridis", title=None, stamp=None):
        """Violin panels of every metric vs scan parameter ``x``, one x-dodged color
        series per value of scan parameter ``series``; THE VIOLIN/SCATTER IS THE SEEDS
        ONLY. ``tag`` restricts to one case tag (required when several are present).
        ``overlays`` is a list of (points, label, series_value, marker) entries with
        points = {metric: [(xvalue, value), ...]} -- see benchmark_overlay(). Overlay x
        values off the scanned grid are placed by interpolation between grid positions.
        """
        records = self.load_performance()
        if tag is not None:
            records = [r for r in records if r["tag"] == tag]
        if not records:
            print(f"\t- No completed cases for tag={tag}; skipping violins", typeMsg="w")
            return None
        xvals = sorted({r["params"][x] for r in records if x in r["params"]})
        svals = sorted({r["params"][series] for r in records if series in r["params"]}) if series else [None]
        scolors = self._series_colors(svals, cmap) if series else {None: "tab:blue"}

        ncols = 3
        nrows = int(np.ceil(len(self.metrics) / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.6 * nrows))
        axs = np.atleast_1d(axs).ravel()

        positions = np.arange(len(xvals))
        xpos = {xv: i for i, xv in enumerate(xvals)}
        offsets = np.linspace(-0.26, 0.26, len(svals)) if len(svals) > 1 else [0.0]
        width = min(0.30, 0.46 / max(len(svals), 1))

        def _overlay_pos(xvalue):
            """Categorical position of an overlay x value; interpolated when off-grid."""
            if xvalue in xpos:
                return float(xpos[xvalue])
            if len(xvals) > 1 and xvals[0] <= xvalue <= xvals[-1]:
                return float(np.interp(xvalue, xvals, positions))
            return None

        for ax, (key, label) in zip(axs, self.metrics):
            for j, s in enumerate(svals):
                col, off = scolors[s], offsets[j]
                mean_pos, mean_val = [], []
                for xv in xvals:
                    g = np.array([r["values"][key] for r in records
                                  if r["params"].get(x) == xv
                                  and (series is None or r["params"].get(series) == s)], dtype=float)
                    g = g[~np.isnan(g)]
                    if g.size == 0:
                        continue
                    pos = xpos[xv] + off
                    if g.size >= 2 and np.ptp(g) > 0:   # violin needs spread
                        parts = ax.violinplot([g], positions=[pos], showmeans=False,
                                              showextrema=False, widths=width)
                        for body in parts["bodies"]:
                            body.set_facecolor(col)
                            body.set_alpha(0.30)
                    jitter = (np.random.RandomState(0).rand(g.size) - 0.5) * 0.06
                    ax.scatter(pos + jitter, g, s=6, color=col, zorder=3)
                    ax.plot(pos, g.mean(), marker="D", color=col, markeredgecolor="k",
                            markeredgewidth=0.5, markersize=6, zorder=4)
                    mean_pos.append(pos)
                    mean_val.append(g.mean())
                # line joining the seed means of this series across x (deterministic trend)
                if len(mean_pos) > 1:
                    ax.plot(mean_pos, mean_val, "-", color=col, lw=1.4, alpha=0.9, zorder=2)

            for points, _label, svalue, omarker in (overlays or []):
                if points is None or key not in points:
                    continue
                ocolor = scolors.get(svalue, "k")
                pp = [(pxv, v) for pxv, v in ((_overlay_pos(xv_), v) for xv_, v in sorted(points[key]))
                      if pxv is not None]
                if pp:
                    px, pv = zip(*pp)
                    ax.plot(px, pv, ":", color=ocolor, lw=1.2, zorder=5)
                    # face = series color, flashy magenta contour so it pops out of the violins
                    ax.plot(px, pv, omarker, color=ocolor, markersize=12,
                            markeredgecolor="magenta", markeredgewidth=2.0,
                            linestyle="", zorder=5)

            if key in REFERENCE_LINES:
                hline, floor = REFERENCE_LINES[key]
                vmax = max([r["values"][key] for r in records if np.isfinite(r["values"][key])],
                           default=0.0)
                for points, *_style in (overlays or []):
                    if points is not None:
                        vmax = max([vmax] + [v for _, v in points.get(key, [])])
                ax.set_ylim(0.0, max(floor, 1.03 * vmax))
                ax.axhline(hline, color="k", ls="--", lw=0.9, alpha=0.6, zorder=1)

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{xv:g}" for xv in xvals])
            ax.set_xlabel(self._param_label(x))
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        for ax in axs[len(self.metrics):]:
            ax.set_visible(False)

        handles = []
        if series:
            handles += [plt.Line2D([], [], marker="o", linestyle="", color=scolors[s],
                                   markeredgecolor="k", label=f"{self._param_label(series)}={s:g}")
                        for s in svals]
        for points, olabel, svalue, omarker in (overlays or []):
            if points is not None:
                handles.append(plt.Line2D([], [], marker=omarker, linestyle="",
                                          color=scolors.get(svalue, "k"),
                                          markeredgecolor="magenta", markeredgewidth=2.0,
                                          markersize=10, label=olabel))
        fig.legend(handles=handles, loc="upper right", title=self._param_label(series) if series else None,
                   fontsize=9)
        nseeds = len({r["seed"] for r in records})
        fig.suptitle(title or f"MAESTRO scan -- performance vs {x} "
                              f"(violin = {nseeds} seeds; color = {series})", fontsize=12)
        if stamp:
            fig.text(0.995, 0.005, stamp, ha="right", va="bottom", fontsize=7,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self._save(fig, save_path or self._out_folder / f"perf_vs_{x}.png")
        return fig

    # --------------------------------------------------------------------------------------------
    # Beat-evolution traces
    # --------------------------------------------------------------------------------------------

    def _beat_states(self, case):
        """[(beat_number, path to beat_results/input.gacode), ...], sorted by beat number."""
        hits = []
        for d in (case["folder"] / "Beats").glob("Beat_*"):
            m = _BEAT_DIR_RE.search(d.name)
            g = d / "beat_results" / "input.gacode"
            if m and g.exists():
                hits.append((int(m.group(1)), g))
        return sorted(hits)

    def load_beat_evolution(self, case):
        """Per-beat trace of each beat quantity for one case: (beat_numbers, {key: [values]}).
        Returns (None, None) if no beat states are readable."""
        states = self._beat_states(case)
        if not states:
            return None, None
        numbers, series = [], {key: [] for key, _, _ in self.beat_quantities}
        for n, gfile in states:
            try:
                p = PROFILEStools.gacode_state(gfile)
                p.derive_quantities()
            except Exception as exc:
                print(f"\t- Could not read {gfile}: {exc}", typeMsg="w")
                continue
            numbers.append(n)
            for key, _, extractor in self.beat_quantities:
                series[key].append(extractor(p))
        return numbers, series

    def plot_beat_evolution(self, color_by, style_by=None, tag=None, save_path=None,
                            cmap="plasma", title=None, stamp=None):
        """Overplot the per-beat evolution of the beat quantities for every case, one line
        per case, colored by scan parameter ``color_by`` (and optionally line-styled by
        ``style_by``, e.g. the tag when several init methods are present)."""
        cases = [c for c in self.cases if tag is None or c["tag"] == tag]
        cvals = sorted({c["params"].get(color_by) for c in cases if color_by in c["params"]})
        cmap_ = plt.get_cmap(cmap)
        ccol = {v: cmap_(k / max(len(cvals) - 1, 1)) for k, v in enumerate(cvals)}
        if style_by == "tag":
            styles = {t: ls for t, ls in zip(self.tags, ["-", "--", ":", "-."])}
        else:
            svals = sorted({c["params"].get(style_by) for c in cases}) if style_by else []
            styles = {v: ls for v, ls in zip(svals, ["-", "--", ":", "-."])}

        nq = len(self.beat_quantities)
        fig, axs = plt.subplots(nrows=2, ncols=int(np.ceil(nq / 2)), figsize=(13, 8))
        axs = np.atleast_1d(axs).ravel()

        all_numbers = []
        for case in cases:
            numbers, series = self.load_beat_evolution(case)
            if not numbers:
                continue
            if len(numbers) > len(all_numbers):
                all_numbers = numbers
            skey = case["tag"] if style_by == "tag" else case["params"].get(style_by)
            ls = styles.get(skey, "-")
            for ax, (key, _, _) in zip(axs, self.beat_quantities):
                ax.plot(numbers, series[key], ls, color=ccol.get(case["params"].get(color_by), "k"),
                        alpha=0.5, lw=1, marker="o", markersize=2.5)

        for ax, (key, ylabel, _) in zip(axs, self.beat_quantities):
            ax.set_xticks(all_numbers)
            ax.set_xticklabels([f"#{n}" for n in all_numbers], rotation=90, fontsize=6)
            ax.set_xlabel("beat")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        for ax in axs[nq:]:
            ax.set_visible(False)

        handles = [plt.Line2D([], [], color=ccol[v], lw=2,
                              label=f"{self._param_label(color_by)}={v:g}") for v in cvals]
        handles += [plt.Line2D([], [], color="k", ls=ls, lw=1, label=str(k))
                    for k, ls in styles.items()]
        fig.legend(handles=handles, loc="upper right", fontsize=9)
        fig.suptitle(title or f"MAESTRO scan -- per-beat evolution (color = {color_by})", fontsize=12)
        if stamp:
            fig.text(0.995, 0.005, stamp, ha="right", va="bottom", fontsize=7,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save(fig, save_path or self._out_folder / "beat_evolution.png")
        return fig

    # --------------------------------------------------------------------------------------------
    # Beat timing
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def load_beat_timings(folder):
        """Total wall time per beat for one run, {beat_counter: (beat_type, seconds)},
        by summing all mitim_timer phases (Initializer, Preparation, Run + Finalization, ...)
        in ``Outputs/Performance/timing.jsonl``. Returns {} if the file is missing."""
        totals = {}
        timing_file = IOtools.expandPath(folder) / "Outputs" / "Performance" / "timing.jsonl"
        if not timing_file.exists():
            return totals
        for raw in timing_file.read_text().splitlines():
            try:
                d = json.loads(raw)
                seconds = float(d["duration_s"])
            except (ValueError, KeyError, TypeError):
                continue
            m = _TIMING_RE.search(d.get("script", ""))
            if not m:
                continue
            counter, btype = int(m.group(1)), m.group(2)
            totals[counter] = (btype, totals.get(counter, (btype, 0.0))[1] + seconds)
        return totals

    @classmethod
    def cumulative_timing(cls, folder):
        """(labels, cumulative_hours) of a single run's beat chain from its timing.jsonl,
        or (None, None). Useful to overlay a benchmark run's own chain."""
        t = cls.load_beat_timings(folder)
        if not t:
            return None, None
        labels, cumul, running = [], [], 0.0
        for c in sorted(t):
            running += t[c][1] / 3600.0
            labels.append(f"#{c} {t[c][0]}")
            cumul.append(running)
        return labels, cumul

    def plot_beat_timing(self, color_by, panel_by=None, tag=None, save_path=None,
                         benchmark_timing=None, benchmark_panel=None, benchmark_color=None,
                         cores=None, benchmark_cores=None,
                         cmap="plasma", title=None, stamp=None):
        """CUMULATIVE wall time along the beat chain (like the mitim_plot_maestro timings
        tab), MEAN OVER SEEDS with errorbars spanning the max-min seed variation. One
        subplot per combination of ``panel_by`` parameters (plus the tag when several),
        color = ``color_by`` -- deterministic scan inputs are never pooled into the
        errorbar. ``benchmark_timing`` = (labels, cumulative_hours) of a reference run
        (see cumulative_timing()), overlaid dashed on panels matching ``benchmark_panel``
        (a dict {param: value}; None = all panels), in ``benchmark_color``'s series color.

        ``cores``: number of cores of each scan case's allocation -- when given, the
        y-axis becomes cumulative CPU-hours (wall x cores), with the benchmark chain
        scaled by ``benchmark_cores`` (defaults to ``cores``). NOTE: this scales by the
        HEAD-JOB allocation only; externally dispatched work (e.g. TRANSP on another
        machine/allocation) is not accounted separately.
        """
        panel_by = panel_by or []
        cores_factor = float(cores) if cores is not None else 1.0
        bench_factor = float(benchmark_cores if benchmark_cores is not None else (cores or 1.0))
        yunits = "CPU-hours" if cores is not None else "wall time [h]"
        cases = [c for c in self.cases if tag is None or c["tag"] == tag]
        timings = {c["name"]: self.load_beat_timings(c["folder"]) for c in cases}

        def _panel_key(case):
            return (case["tag"],) + tuple(case["params"].get(p) for p in panel_by)

        combos = sorted({_panel_key(c) for c in cases})
        ncols = min(3, len(combos))
        nrows = int(np.ceil(len(combos) / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows),
                                squeeze=False)
        axs = axs.ravel()
        cvals = sorted({c["params"].get(color_by) for c in cases if color_by in c["params"]})
        cmap_ = plt.get_cmap(cmap)
        ccolor = {v: cmap_(k / max(len(cvals) - 1, 1)) for k, v in enumerate(cvals)}

        for ax, combo in zip(axs, combos):
            cases_c = [c for c in cases if _panel_key(c) == combo]
            counters = sorted({ctr for case in cases_c for ctr in timings[case["name"]]})
            beat_names = {}
            for case in cases_c:
                for ctr, (btype, _s) in timings[case["name"]].items():
                    beat_names.setdefault(ctr, btype)
            x = np.arange(len(counters))
            for v in cvals:
                seeds = [timings[c["name"]] for c in cases_c
                         if c["params"].get(color_by) == v and timings[c["name"]]]
                if not seeds:
                    continue
                # per-seed running total up to and including each beat (a seed that died
                # early just stops contributing beyond its last beat)
                cumul = []
                for t in seeds:
                    running, per_beat = 0.0, {}
                    for ctr in sorted(t):
                        running += t[ctr][1] / 3600.0 * cores_factor
                        per_beat[ctr] = running
                    cumul.append(per_beat)
                mean, err_lo, err_hi = [], [], []
                for ctr in counters:
                    vals = np.array([pb[ctr] for pb in cumul if ctr in pb])
                    if vals.size == 0:
                        mean.append(np.nan); err_lo.append(0.0); err_hi.append(0.0)
                    else:
                        mean.append(vals.mean())
                        err_lo.append(vals.mean() - vals.min())
                        err_hi.append(vals.max() - vals.mean())
                ax.errorbar(x, mean, yerr=[err_lo, err_hi], color=ccolor[v],
                            lw=1.3, marker="o", markersize=3, capsize=2, alpha=0.85)

            panel_matches = benchmark_panel is None or all(
                combo[1 + panel_by.index(p)] == v for p, v in benchmark_panel.items()
                if p in panel_by)
            if benchmark_timing is not None and benchmark_timing[0] is not None and panel_matches:
                _blabels, bcumul = benchmark_timing
                ax.plot(np.arange(len(bcumul)), np.array(bcumul) * bench_factor, "--",
                        color=ccolor.get(benchmark_color, "k"),
                        lw=1.3, marker="s", markersize=5, markeredgecolor="magenta",
                        markeredgewidth=1.2, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"#{c} {beat_names.get(c, '?')}" for c in counters],
                               rotation=90, fontsize=7)
            ax.set_ylabel(f"cumulative {yunits}")
            ptitle = ", ".join([combo[0]] + [f"{p}={v:g}" for p, v in zip(panel_by, combo[1:])])
            ax.set_title(ptitle, fontsize=10)
            ax.grid(True, alpha=0.3)

        for ax in axs[len(combos):]:
            ax.set_visible(False)

        handles = [plt.Line2D([], [], color=ccolor[v], lw=2,
                              label=f"{self._param_label(color_by)}={v:g}") for v in cvals]
        if benchmark_timing is not None and benchmark_timing[0] is not None:
            handles.append(plt.Line2D([], [], color=ccolor.get(benchmark_color, "k"), ls="--",
                                      marker="s", markeredgecolor="magenta", markeredgewidth=1.2,
                                      markersize=6, label="benchmark run"))
        fig.legend(handles=handles, loc="upper right", fontsize=9)
        fig.suptitle(title or "MAESTRO scan -- cumulative wall time along the beat chain "
                              "(mean over seeds; errorbar = max-min)", fontsize=12)
        if stamp:
            fig.text(0.995, 0.005, stamp, ha="right", va="bottom", fontsize=7,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(fig, save_path or self._out_folder / "beat_timing.png")
        return fig

    # --------------------------------------------------------------------------------------------
    # Per-seed profile spread
    # --------------------------------------------------------------------------------------------

    def plot_profiles(self, per, columns, quantities=None, tag=None, save_prefix=None,
                      cmap="tab10", title=None, stamp=None,
                      annotate_metric="Pfus", annotate_units="MW"):
        """Per-seed profile variation at fixed deterministic inputs: ONE FIGURE PER VALUE
        of scan parameter ``per``, with one subplot column per value of scan parameter
        ``columns`` and one row per quantity in ``quantities`` (default: Ti, ne and
        their normalized inverse gradient scale lengths a/LTi, a/Lne, from each case's
        final state). Every line in a subplot is one seed -- everything deterministic is
        fixed there, so the line spread IS the seed-to-seed scatter of the profiles.
        ``annotate_metric`` (a derived scalar key; None disables) is printed per seed in
        the top subplot of each column, in the seed's color -- by default each seed's
        Pfus, so the profile spread reads directly against the performance it produced.
        Figures are saved as ``<save_prefix>_<per><value>.png`` (default prefix
        ``profiles``). Returns the list of figures."""
        quantities = quantities or PROFILE_QUANTITIES
        records = [c for c in self.cases if tag is None or c["tag"] == tag]
        pvals = sorted({c["params"][per] for c in records if per in c["params"]})
        cvals = sorted({c["params"][columns] for c in records if columns in c["params"]})
        seeds = sorted({c["seed"] for c in records})
        cmap_ = plt.get_cmap(cmap)
        scol = {s: cmap_(k % 10) for k, s in enumerate(seeds)}

        quantities = [(q + (False,))[:4] for q in quantities]   # pad missing core_ylim flag
        figs = []
        for pv in pvals:
            fig, axs = plt.subplots(nrows=len(quantities), ncols=len(cvals),
                                    figsize=(4.2 * len(cvals), 2.9 * len(quantities)),
                                    sharex=True, squeeze=False)
            core_max = np.zeros((len(quantities), len(cvals)))
            for jc, cv in enumerate(cvals):
                cases_cell = [c for c in records
                              if c["params"].get(per) == pv and c["params"].get(columns) == cv]
                annotations = []
                for case in cases_cell:
                    gfile = self._final_gacode(case)
                    if gfile is None:
                        continue
                    p = PROFILEStools.gacode_state(gfile)
                    p.derive_quantities()
                    for jr, (key, _ylabel, extractor, _core) in enumerate(quantities):
                        rho, vals = extractor(p)
                        axs[jr, jc].plot(rho, vals, color=scol[case["seed"]], lw=1.2,
                                         alpha=0.85, label=f"seed {case['seed']}")
                        core_max[jr, jc] = max(core_max[jr, jc],
                                               np.max(np.abs(np.array(vals)[np.array(rho) <= 0.9])))
                    if annotate_metric is not None:
                        annotations.append((case["seed"],
                                            float(p.derived[annotate_metric])))
                # per-seed metric readout (e.g. Pfus), stacked bottom-left of the top subplot
                for k, (seed, val) in enumerate(sorted(annotations)):
                    axs[0, jc].text(0.03, 0.04 + 0.08 * k,
                                    f"seed {seed}: {val:.0f} {annotate_units}",
                                    transform=axs[0, jc].transAxes, fontsize=7.5,
                                    color=scol[seed], va="bottom", weight="bold")
                axs[0, jc].set_title(f"{self._param_label(columns)} = {cv:g}", fontsize=10)
                axs[-1, jc].set_xlabel(r"$\rho_{tor}$")
            for jr, (key, ylabel, _extractor, core_ylim) in enumerate(quantities):
                axs[jr, 0].set_ylabel(ylabel)
                for jc in range(len(cvals)):
                    axs[jr, jc].grid(True, alpha=0.3)
                    if core_ylim and core_max[jr, jc] > 0:
                        # bound by the core (rho <= 0.9): the pedestal spike stays off-scale
                        axs[jr, jc].set_ylim(-0.15 * core_max[jr, jc], 1.25 * core_max[jr, jc])
            handles = [plt.Line2D([], [], color=scol[s], lw=2, label=f"seed {s}") for s in seeds]
            fig.legend(handles=handles, loc="upper right", fontsize=9)
            fig.suptitle(title or f"Profiles per seed -- {self._param_label(per)} = {pv:g} "
                                  f"(columns = {columns}; line spread = seeds)", fontsize=12)
            if stamp:
                fig.text(0.995, 0.005, stamp, ha="right", va="bottom", fontsize=7,
                         bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            prefix = save_prefix or (self._out_folder / "profiles")
            self._save(fig, prefix.parent / f"{prefix.name}_{per}{pv:g}.png")
            figs.append(fig)
        return figs

    # --------------------------------------------------------------------------------------------
    # PDF report
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def _find_special_tab(folder):
        """Locate the saved 'MAESTRO special' tab PNG for a case (run_maestro --save
        writes the FigureNotebook to <case>/maestro_plots/)."""
        for sub in ("maestro_plots", "Outputs/maestro_plots"):
            hits = sorted((folder / sub).glob("*special*")) if (folder / sub).is_dir() else []
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _image_page(pdf, image_path, title):
        img = plt.imread(image_path)
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
        ax.imshow(img)
        ax.axis("off")
        fig.suptitle(title, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    @staticmethod
    def _text_pages(pdf, text, title, lines_per_page=58, wrap=110):
        wrapped = []
        for line in text.splitlines():
            wrapped.extend(textwrap.wrap(line, width=wrap) or [""])
        for start in range(0, max(len(wrapped), 1), lines_per_page):
            chunk = "\n".join(wrapped[start:start + lines_per_page])
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(title, fontsize=12)
            fig.text(0.05, 0.93, chunk, family="monospace", fontsize=7.5, va="top", ha="left")
            pdf.savefig(fig)
            plt.close(fig)

    def compile_report(self, pdf_path=None, summary_figs=None):
        """Compile a single PDF: the summary figures first (defaults to every figure this
        object produced so far), then per-case special tab + maestro_summary.md."""
        pdf_path = pdf_path or self._out_folder / "scan_report.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for fig in (summary_figs if summary_figs is not None else self._summary_figs):
                if fig is not None:
                    pdf.savefig(fig)
            for case in self.cases:
                pdesc = ", ".join(f"{k}={v:g}" for k, v in case["params"].items())
                title = f"{case['name']}  (tag={case['tag']}, {pdesc}, seed={case['seed']})"
                special = self._find_special_tab(case["folder"])
                if special is not None:
                    self._image_page(pdf, special, f"{title} -- MAESTRO special")
                else:
                    self._text_pages(pdf, "[no MAESTRO special tab found in maestro_plots/]",
                                     f"{title} -- MAESTRO special")
                summary = case["folder"] / "Outputs" / "maestro_summary.md"
                if summary.exists():
                    self._text_pages(pdf, summary.read_text(), f"{title} -- maestro_summary.md")
                else:
                    self._text_pages(pdf, "[no Outputs/maestro_summary.md found]",
                                     f"{title} -- maestro_summary.md")
        print(f"\t- Saved {pdf_path}")

    # --------------------------------------------------------------------------------------------

    def set_output_folder(self, folder):
        """Where figures/report go by default (created on first save).
        Defaults to <main_folder>/interpretation."""
        self._out_folder = IOtools.expandPath(folder)

    def _save(self, fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        self._summary_figs.append(fig)
        print(f"\t- Saved {path}")
