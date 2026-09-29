"""
Time-averaging of nonlinear gyrokinetic flux traces (CGYRO, GX): pick the saturated
window and compute the mean and its uncertainty for every signal of the run.

The method is chosen with the `averaging` block of the read options (PORTALS namelist
`transport.options.<code>.read.averaging`, or `averaging=` kwarg of the output classes):

    method: "fixed"        window from tmin / tmin_is_rel (classic MITIM behaviour)
            "quends"       Sandia QUENDS (optional dependency `quends`): trims the transient
                           per channel and uses block-mean statistics for the uncertainty
            "howard_gkav"  N.T. Howard's stationarity scan (GKwindow_howard.py) picks the
                           window; the uncertainty is the ACF standard error

Uncertainty convention (all methods): `*_std` is the 1-sigma STANDARD ERROR OF THE MEAN,
not the fluctuation level. `uncertainty: "acf"` gives std/sqrt(n_corr), with n_corr = N/(3*tau)
independent samples and tau the 1/e decorrelation lag of the ACF; `uncertainty: "quends"` gives
the QUENDS block-mean SEM (autotuned block size, Ljung-Box independence test) for 1D signals.
Defaults: "acf" for fixed/howard_gkav, "quends" for the quends method.

Note on QUENDS trimming (2026-09-17, local CGYRO runs): its strategies rarely accept local
turbulent fluxes (15-20% RMS, bursty) as steady state and, if started at t=0, they declare the
quiescent linear phase steady. Hence start_time defaults to the turbulence onset and the
method falls back to the fixed window (flag "fallback") when nothing is found.

Times are in the code's own normalized units (a/cs for CGYRO); fluxes in GB units. The
window selection only looks at the primary channels (Qi, Qe, Ge); every other signal is
then averaged over that same window, so spectra and fluctuations stay consistent with
the fluxes.
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools import __version__ as mitim_version, __mitimroot__
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.simulation_tools.utils import GKwindow_howard
from mitim_tools.misc_tools.LOGtools import printMsg as print

METHODS = ("fixed", "quends", "howard_gkav")
PRIMARY_CHANNELS = ("Qi", "Qe", "Ge")

# QUENDS defaults: the package's own trim strategy (robust quantile test), trimming only
# after the turbulence onset ("onset" = first time the smoothed trace reaches 25% of its
# max, as in GKwindow_howard); stats_window_size=None lets QUENDS autotune the block size.
QUENDS_DEFAULTS = {
    "trim_method": "std",
    "window_size": 10,
    "threshold": None,
    "start_time": "onset",
    "robust": True,
    "stats_window_size": None,
}
UNCERTAINTIES = ("acf", "quends")


def resolve_fixed_tmin(t, tmin=0.0, tmin_is_rel=True, print_msg=True):
    """
    Absolute window start from the classic (tmin, tmin_is_rel) pair:
      tmin >= 0                    : absolute time
      tmin <  0, tmin_is_rel=True  : fraction of the run counted from the end (-0.3 -> last 30%)
      tmin <  0, tmin_is_rel=False : absolute offset from the end (-200 -> last 200 time units)
    """
    if tmin >= 0.0:
        return float(tmin)

    if tmin_is_rel:
        t_start = t[-1] + tmin * (t[-1] - t[0])
        if print_msg:
            print(f"\t- Negative relative tmin provided ({tmin}), setting tmin to {t_start:.3f} (last {-tmin*100:.1f}% of run)", typeMsg='i')
    else:
        t_start = t[-1] + tmin
        if print_msg:
            print(f"\t- Negative absolute tmin provided ({tmin} a/cs), setting tmin to {t_start:.3f} (= t[-1]={t[-1]:.3f} + {tmin})", typeMsg='i')
        if t_start < t[0] and print_msg:
            print(f"\t  Warning: computed tmin ({t_start:.3f}) is before the start of the run (t[0]={t[0]:.3f}); the full time series will be used", typeMsg='w')

    return float(t_start)


def quends_available():
    try:
        import quends  # noqa: F401
        return True
    except ImportError:
        return False


class GKaverager:
    """
    Window selection + statistics for one nonlinear gyrokinetic run.

    Parameters
    ----------
    t       : time base (1D)
    traces  : dict of the primary channels {'Qi': ..., 'Qe': ..., 'Ge': ...} (GB units), same length as t
    method  : one of METHODS
    tmin, tmin_is_rel : the classic fixed window (also the fallback for the other methods)
    quends  : options overriding QUENDS_DEFAULTS
    uncertainty : "acf" | "quends" | None (None -> "quends" for the quends method, "acf" otherwise)
    label   : text for prints/plots

    Attributes after construction
    -----------------------------
    t_start, t_end   : averaging window (absolute time units of t)
    flag             : "fixed" for the fixed method; Howard's flags (ok/ok2/questionable/fallback/failure/below_threshold);
                       "ok"/"fallback" for quends
    stats            : {channel: {'mean': ..., 'std': ...}} of the primary channels
    diagnostics      : per-method details (settle times, sss_start, ESS, ...)
    provenance       : method, options, code versions
    """

    def __init__(self, t, traces, method="fixed", tmin=0.0, tmin_is_rel=True, quends=None, uncertainty=None, label=""):

        if method not in METHODS:
            raise ValueError(f"Unknown GK averaging method '{method}', options are {METHODS}")
        self.uncertainty = uncertainty or ("quends" if method == "quends" else "acf")
        if self.uncertainty not in UNCERTAINTIES:
            raise ValueError(f"Unknown GK averaging uncertainty '{self.uncertainty}', options are {UNCERTAINTIES}")
        if self.uncertainty == "quends" and not quends_available():
            print("\t- quends is not installed (pip install mitim[quends]); using the ACF standard error", typeMsg='w')
            self.uncertainty = "acf"

        self.t = np.asarray(t, dtype=float)
        self.traces = {k: np.asarray(v, dtype=float) for k, v in traces.items()}
        self.method = method
        self.label = label
        self.tmin_requested = tmin
        self.tmin_is_rel = tmin_is_rel
        self.options_quends = {**QUENDS_DEFAULTS, **(quends or {})}

        self.t_end = float(self.t[-1])
        self.t_start_fixed = resolve_fixed_tmin(self.t, tmin, tmin_is_rel, print_msg=(method == "fixed"))
        self.diagnostics = {}

        self._select_window()

        self.stats = {k: dict(zip(("mean", "std"), self.mean_std(v, label_print=k, print_msg=True))) for k, v in self.traces.items()}

        self.provenance = self._provenance()

    # ------------------------------------------------------------------
    # Window selection
    # ------------------------------------------------------------------

    def _select_window(self):

        if self.method == "fixed":
            self.t_start, self.flag = self.t_start_fixed, "fixed"

        elif self.method == "howard_gkav":
            self._select_window_howard()

        elif self.method == "quends":
            self._select_window_quends()

        self.t_start = float(np.clip(self.t_start, self.t[0], self.t_end))
        self.window_length = self.t_end - self.t_start

        print(f"\t- GK averaging [{self.method}, {self.uncertainty} uncertainty]{(' ' + self.label) if self.label else ''}: window t = [{self.t_start:.1f}, {self.t_end:.1f}] ({self.window_length:.1f} long, {self.n_window} samples), flag = {self.flag}",
              typeMsg='i' if self.flag in ("fixed", "ok", "ok2") else 'w')

    def _primary(self):
        missing = [k for k in PRIMARY_CHANNELS if k not in self.traces]
        if missing:
            raise ValueError(f"GK averaging method '{self.method}' needs the primary channels {PRIMARY_CHANNELS}; missing {missing}")
        return [self.traces[k] for k in PRIMARY_CHANNELS]

    def _select_window_howard(self):

        qi, qe, ge = self._primary()
        res = GKwindow_howard.select_start(self.t, qi, qe, ge)

        self.t_start, self.flag = res["t_start"], res["flag"]
        self.diagnostics = {
            "settle_times": {k: (None if v is None else float(v)) for k, v in res["settle_times"].items()},
            "activity_times": {k: float(v) for k, v in res["activity_times"].items()},
            "transient_end": res["transient_end"],
            "npass": res["npass"],
            "metrics": {k: tuple(float(x) for x in v) for k, v in res["metrics"].items()},
            "min_window": GKwindow_howard.min_window(self.t[-1] - self.t[0]),
        }

        if self.flag in GKwindow_howard.FLAGS_FALLBACK:
            print(f"\t- howard_gkav could not find a stationary window (flag '{self.flag}'); proceeding with its fallback window (second half of the run)", typeMsg='w')

    def _select_window_quends(self):

        if not quends_available():
            print("\t- quends is not installed (pip install mitim[quends]); falling back to the fixed tmin window", typeMsg='w')
            self.t_start, self.flag = self.t_start_fixed, "fallback"
            self.diagnostics.update({"sss_start": {}, "quends_installed": False})
            return

        self._primary()
        sss = {k: self._quends_sss_start(k, v) for k, v in self.traces.items() if k in PRIMARY_CHANNELS}
        found = {k: v for k, v in sss.items() if v is not None}

        # Conservative: the window starts once ALL primary channels are in steady state
        if len(found) == len(sss):
            self.t_start, self.flag = max(found.values()), "ok"
        elif len(found) > 0:
            self.t_start, self.flag = max(found.values()), "questionable"
            print(f"\t- quends found no steady state for {[k for k in sss if k not in found]}; using the channels that did", typeMsg='w')
        else:
            self.t_start, self.flag = self.t_start_fixed, "fallback"
            print("\t- quends found no steady state in any primary channel; falling back to the fixed tmin window", typeMsg='w')

        self.diagnostics.update({"sss_start": sss, "quends_installed": True})

    def _onset_time(self, S):
        '''Turbulence onset as in GKwindow_howard: smoothed |S| first exceeds ACTIVITY_FRAC of its max.'''
        dt = np.median(np.diff(self.t))
        w = max(3, int(round(GKwindow_howard.SMOOTH_W / dt)))
        return float(GKwindow_howard._activity_time(self.t, GKwindow_howard._smooth(np.asarray(S, dtype=float), w)))

    def _quends_stream(self, S, t=None):
        import quends as qnds
        return qnds.from_numpy(np.asarray(S, dtype=float), "signal", time=np.asarray(self.t if t is None else t, dtype=float))

    def _quends_sss_start(self, name, S):
        o = self.options_quends
        start_time = self._onset_time(S) if o["start_time"] == "onset" else float(o["start_time"])
        self.diagnostics.setdefault("start_time", {})[name] = start_time
        trimmed = self._quends_stream(S).trim(
            column_name="signal", method=o["trim_method"], window_size=o["window_size"],
            start_time=start_time, threshold=o["threshold"], robust=o["robust"])
        sss = trimmed.trim_metadata.get("sss_start", None)
        if isinstance(sss, dict) or sss is None:
            print(f"\t- quends [{name}]: no steady state found ({sss.get('message', '') if isinstance(sss, dict) else 'empty'})", typeMsg='w')
            return None
        return float(sss)

    # ------------------------------------------------------------------
    # Statistics over the selected window
    # ------------------------------------------------------------------

    @property
    def n_window(self):
        return int(np.sum(self.t >= self.t_start))

    def _window_indices(self, tmax=None):
        it0 = int(np.argmin(np.abs(self.t - self.t_start)))
        it1 = int(np.argmin(np.abs(self.t - tmax))) if tmax is not None else len(self.t)
        return (it1, it1) if it1 <= it0 else (it0, it1)

    def mean_std(self, S, tmax=None, label_print="", print_msg=False):
        """
        Mean and standard error of S (time on the last axis) over [t_start, tmax].
        QUENDS statistics apply to 1D signals only; anything else uses the ACF estimator.
        """
        S = np.asarray(S)
        if self.uncertainty == "quends" and S.ndim == 1:
            return self._mean_std_quends(S, tmax=tmax, label_print=label_print, print_msg=print_msg)
        return apply_ac(self.t, S, tmin=self.t_start, tmax=tmax, label_print=label_print, print_msg=print_msg)

    def _mean_std_quends(self, S, tmax=None, label_print="", print_msg=False):
        it0, it1 = self._window_indices(tmax)
        stats = self._quends_stream(S[it0:it1 + 1], t=self.t[it0:it1 + 1]).compute_statistics(
            column_name="signal", method="non-overlapping", window_size=self.options_quends["stats_window_size"])["signal"]
        mean, std = float(stats["mean"]), float(stats["mean_uncertainty"])
        if label_print:
            self.diagnostics.setdefault("stats", {})[label_print] = {k: stats.get(k, None) for k in
                ("effective_sample_size", "window_size", "n_short_averages", "independence_status", "standard_deviation", "warning")}
        if print_msg:
            print(f"\t- {label_print}: quends block size {stats.get('window_size')} ({stats.get('independence_status')}), ESS {stats.get('effective_sample_size', float('nan')):.1f} -> {mean:.2e} +-{std:.2e}")
        return mean, std

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _provenance(self):
        prov = {"method": self.method, "uncertainty": self.uncertainty, "mitim_version": mitim_version}
        try:
            prov["mitim_branch"], prov["mitim_commit"] = IOtools.get_git_info(__mitimroot__)
        except Exception:
            pass
        if self.method == "fixed":
            prov["tmin"], prov["tmin_is_rel"] = self.tmin_requested, self.tmin_is_rel
        elif self.method == "howard_gkav":
            prov["source"], prov["thresholds"] = GKwindow_howard.SOURCE, GKwindow_howard.thresholds()
        if self.method == "quends" or self.uncertainty == "quends":
            prov["options_quends"] = dict(self.options_quends)
            try:
                from importlib.metadata import version
                prov["quends_version"] = version("quends")
            except Exception:
                prov["quends_version"] = None
        return prov

    def to_dict(self):
        """JSON-serializable record (written by PORTALS into fluxes_turb.json)."""
        return {
            "method": self.method,
            "uncertainty": self.uncertainty,
            "flag": self.flag,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "window_length": self.window_length,
            "n_samples": self.n_window,
            "stats": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in self.stats.items()},
            "diagnostics": _jsonable(self.diagnostics),
            "provenance": _jsonable(self.provenance),
        }

    def summary(self):
        lines = [f"GK averaging ({self.method}, {self.uncertainty} uncertainty) {self.label}".rstrip(),
                 f"  window: t = [{self.t_start:.1f}, {self.t_end:.1f}]  ({self.window_length:.1f} long, {self.n_window} samples)  flag = {self.flag}"]
        for k, v in self.stats.items():
            lines.append(f"  {k:>4}: {v['mean']:.3e} +- {v['std']:.3e}  ({100*abs(v['std']/v['mean']) if v['mean'] != 0 else float('inf'):.1f}%)")
        if self.method == "howard_gkav":
            d = self.diagnostics
            lines.append(f"  settle times: " + ", ".join(f"{k}={('%.0f' % v) if v is not None else 'none'}" for k, v in d["settle_times"].items())
                         + f";  transient end = {d['transient_end']:.0f};  channels passing = {d['npass']}/3;  min window = {d['min_window']:.0f}")
        elif self.method == "quends" and self.diagnostics.get("quends_installed", False):
            lines.append("  steady-state start: " + ", ".join(f"{k}={('%.0f' % v) if v is not None else 'none'}" for k, v in self.diagnostics["sss_start"].items()))
        return "\n".join(lines)

    def print_summary(self):
        for line in self.summary().split("\n"):
            print(f"\t{line}", typeMsg='i')

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, axs=None, fig=None, channels=None, color="b", label_plot=None, show_acf=True):
        """
        One row per primary channel: the trace with the window shaded, the mean +- 1-sigma
        standard error band and the per-method markers (settle times / steady-state start);
        optionally a second column with the ACF of the windowed trace and its 1/e lag.
        Returns the axes array (n_channels x 1 or x 2).
        """
        channels = list(self.traces.keys()) if channels is None else channels
        ncols = 2 if show_acf else 1
        if axs is None:
            if fig is None:
                fig = plt.figure(figsize=(14 if show_acf else 9, 3 * len(channels)))
            axs = fig.subplots(len(channels), ncols, squeeze=False, gridspec_kw={"width_ratios": [3, 1] if show_acf else [1]})
        axs = np.asarray(axs).reshape(len(channels), ncols)

        for i, ch in enumerate(channels):
            self._plot_trace(axs[i, 0], ch, color=color, label_plot=label_plot if i == 0 else None, xlabel=(i == len(channels) - 1))
            if show_acf:
                self._plot_acf(axs[i, 1], ch, color=color, xlabel=(i == len(channels) - 1))

        axs[0, 0].set_title(f"{self.method}: window [{self.t_start:.0f}, {self.t_end:.0f}], flag = {self.flag}", fontsize=10, loc="left")
        return axs

    def _plot_trace(self, ax, ch, color="b", label_plot=None, xlabel=True):
        S, (m, s) = self.traces[ch], (self.stats[ch]["mean"], self.stats[ch]["std"])
        ax.plot(self.t, S, c=color, lw=0.8, label=label_plot)
        ax.axvspan(self.t_start, self.t_end, color=color, alpha=0.08, lw=0)
        ax.fill_between([self.t_start, self.t_end], m - s, m + s, color=color, alpha=0.35, lw=0)
        ax.hlines(m, self.t_start, self.t_end, colors=color, linestyles="--", lw=1.2)
        ax.axvline(self.t_start, c=color, ls="-", lw=1.5)
        for tm, ls in self._marker_times(ch):
            ax.axvline(tm, c=color, ls=ls, lw=0.8, alpha=0.7)
        ax.text(0.99, 0.95, f"{m:.3f} $\\pm$ {s:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=color)
        ax.set_ylabel(f"{ch} (GB)")
        if xlabel:
            ax.set_xlabel("t (a/cs)")
        ax.set_xlim([self.t[0], self.t_end])
        GRAPHICStools.addDenseAxis(ax)
        if label_plot:
            ax.legend(loc="upper left", fontsize=8)

    def _marker_times(self, ch):
        """(time, linestyle) pairs of the method's per-channel diagnostics."""
        key = ch.lower()
        if self.method == "howard_gkav":
            return [(v, ":") for k, v in self.diagnostics["settle_times"].items() if k == key and v is not None] \
                 + [(self.diagnostics["transient_end"], "-.")]
        if self.method == "quends":
            return [(v, ":") for k, v in self.diagnostics.get("sss_start", {}).items() if k == ch and v is not None]
        return []

    def _plot_acf(self, ax, ch, color="b", xlabel=True):
        it0, it1 = self._window_indices()
        Sw = self.traces[ch][it0:it1 + 1]
        if len(Sw) < 3:
            return
        acf = sm.tsa.acf(Sw, nlags=len(Sw) - 1)
        n_corr, icor = _grab_ncorrelation(Sw)
        ax.plot(acf, "-", c=color, lw=1.0)
        ax.axhline(1 / np.e, c="k", ls="--", lw=0.8)
        ax.axvline(icor, c=color, ls=":", lw=0.8)
        ax.text(0.98, 0.95, f"$\\tau$ = {icor:.0f} samples\n{n_corr:.1f} indep. samples", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_xlim([0, min(len(acf), max(5 * icor, 20))])
        ax.set_ylabel(f"ACF({ch})")
        if xlabel:
            ax.set_xlabel("lag (samples)")
        GRAPHICStools.addDenseAxis(ax)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ----------------------------------------------------------------------
# ACF standard-error estimator (moved here from CGYROutils; same numerics)
# ----------------------------------------------------------------------

def _grab_ncorrelation(S, debug=False):
    # Calculate the autocorrelation function
    i_acf = sm.tsa.acf(S, nlags=len(S))

    if i_acf.min() > 1/np.e:
        print("Autocorrelation function does not reach 1/e, will use full length of time series for n_corr.", typeMsg='w')

    # Calculate how many time slices make the autocorrelation function is 1/e (conventional decorrelation level)
    icor = np.abs(i_acf-1/np.e).argmin()
    if icor < 1:
        # Window too short (or signal too noisy) for the ACF to be resolved: the
        # closest-to-1/e lag is 0 and n_corr would be infinite (std -> 0). Treat
        # every sample as correlated to the next one, i.e. one decorrelation lag.
        print("Autocorrelation lag resolved as 0 (signal window too short); using 1 lag for n_corr — flux uncertainty is unreliable.", typeMsg='w')
        icor = 1

    # Define number of samples
    n_corr = len(S) / ( 3.0 * icor ) #Define "sample" as 3 x autocor time

    if debug:
        fig, ax = plt.subplots()
        ax.plot(i_acf, '-o', label='ACF')
        ax.axhline(1/np.e, color='r', linestyle='--', label='1/e')
        ax.set_xlabel('Lags'); ax.set_xlim([0, icor+20])
        ax.set_ylabel('ACF')
        ax.legend()
        plt.show()
        embed()

    return n_corr, icor

def apply_ac(t, S, tmin = 0, tmax = None, label_print = '', print_msg = False, debug=False):

    it0 = np.argmin(np.abs(t - tmin))
    it1 = np.argmin(np.abs(t - tmax)) if tmax is not None else len(t)  # If tmax is None, use the full length of t

    if it1 <= it0:
        it0 = it1

    # Calculate the mean and std of the signal after tmin (last dimension is time)
    S_mean = np.mean(S[..., it0:it1+1], axis=-1)
    S_std = np.std(S[..., it0:it1+1], axis=-1)

    if S.ndim == 1:
        # 1D case: single time series
        n_corr, icor = _grab_ncorrelation(S[it0:it1+1], debug=debug)
        S_std = S_std / np.sqrt(n_corr)

        if print_msg:
            print(f"\t- {(label_print + ': a') if len(label_print)>0 else 'A'}utocorr time: {icor:.1f} -> {n_corr:.1f} samples -> {S_mean:.2e} +-{S_std:.2e}")

    else:
        # Multi-dimensional case: flatten all dimensions except the last one
        shape_orig = S.shape[:-1]  # Original shape without time dimension
        S_reshaped = S.reshape(-1, S.shape[-1])  # Flatten to (n_series, n_time)

        n_series = S_reshaped.shape[0]
        n_corr = np.zeros(n_series)
        icor = np.zeros(n_series)

        # Calculate correlation for each flattened time series
        for i in range(n_series):
            n_corr[i], icor[i] = _grab_ncorrelation(S_reshaped[i, it0:it1+1], debug=debug)

        # Reshape correlation arrays back to original shape (without time dimension)
        n_corr = n_corr.reshape(shape_orig)
        icor = icor.reshape(shape_orig)

        # Apply correlation correction to standard deviation
        S_std = S_std / np.sqrt(n_corr)

        # Print results - handle different dimensionalities
        if print_msg:
            if S.ndim == 2:
                # 2D case: print each series
                for i in range(S.shape[0]):
                    print(f"\t- {(label_print + f'_{i}: a') if len(label_print)>0 else 'A'}utocorr: {icor[i]:.1f} -> {n_corr[i]:.1f} samples -> {S_mean[i]:.2e} +-{S_std[i]:.2e}")
            else:
                # Higher dimensional case: print summary statistics
                print(f"\t- {(label_print + ': a') if len(label_print)>0 else 'A'}utocorr time: {icor.mean():.1f}±{icor.std():.1f} -> {n_corr.mean():.1f}±{n_corr.std():.1f} samples -> shape {S_mean.shape}")

    return S_mean, S_std
