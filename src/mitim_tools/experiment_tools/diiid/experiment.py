"""DIIIDExperiment: object-oriented DIII-D discharge analysis on top of the DIIIDFetcher retrieval.

    with DIIIDExperiment(207959, time=4000.0, tunnel_host="cybele") as exp:
        exp.overview()                      # fetch + plot engineering/kinetic traces
        exp.plot_cer_coverage()             # CER (rho, t) coverage
        fit = exp.fit_ti(robust=True)       # QUICKFIT (map2grid) profile fit at `time`
        cC, cimp, imp = exp.impurity_concentration()
        state = exp.to_gacode(plot_data=True)   # -> input.gacode (mitim gacode_state)

Retrieval / equilibrium / geqdsk / cer+thomson profiles / coverage are inherited from DIIIDFetcher.
The higher-level capabilities (QUICKFIT fits, impurity concentration from Zeff, input.gacode
translation, overview/CER plots) are added here.

QUICKFIT (Tomas Odstrcil's map2grid; https://github.com/odstrcilt/quickfit) is an OPTIONAL
capability: `pip install mitim-fusion[quickfit]` installs scikit-sparse<0.5, and the quickfit clone
is auto-located next to MITIM-fusion (../quickfit) or via $QUICKFIT_PATH. It is imported lazily, so
this module (and DIIIDExperiment retrieval/plotting) works without it; only the fit_*/to_gacode
methods require it.
"""
import os
import sys
from pathlib import Path
import numpy as np
from scipy.constants import m_e, e, c as c_light, epsilon_0
from scipy.interpolate import interp1d

from mitim_tools.experiment_tools.diiid.retrieval import DIIIDConnection, DIIIDFetcher
from mitim_tools.experiment_tools.diiid import plotting as _plotting
from mitim_tools.misc_tools.LOGtools import printMsg as print


# ------------------------------------------------------------------ QUICKFIT (optional)
_MAP2GRID = None


def _load_map2grid():
    """Lazy, path-configurable import of QUICKFIT's map2grid (optional capability)."""
    global _MAP2GRID
    if _MAP2GRID is not None:
        return _MAP2GRID
    try:
        from grid_map import map2grid
    except ImportError:
        roots = [os.environ.get("QUICKFIT_PATH"), str(Path(__file__).resolve().parents[5] / "quickfit")]
        for r in [x for x in roots if x]:
            if (Path(r) / "grid_map.py").exists():
                sys.path.insert(0, r)
                break
        try:
            from grid_map import map2grid
        except ImportError as e:
            raise ImportError(
                "QUICKFIT (map2grid) not found. Install the optional dependency `mitim-fusion[quickfit]` "
                "(scikit-sparse<0.5) and clone https://github.com/odstrcilt/quickfit next to MITIM-fusion "
                "(../quickfit) or set $QUICKFIT_PATH.") from e
    if not hasattr(np, "infty"):
        np.infty = np.inf
    np.seterr(divide="ignore", invalid="ignore")     # MITIM aLT does log(0) at zero-pressure edge
    _MAP2GRID = map2grid
    return map2grid


# ------------------------------------------------------------------ fit settings (validated defaults)
def _transforms():
    log = (lambda x: np.log(np.maximum(x, 0) / .1 + 1), lambda x: np.maximum(np.exp(x) - 1, 1e-6) * .1,
           lambda x: 1 / (.1 + np.maximum(0, x)))
    sqrt = (lambda x: np.sqrt(np.maximum(0, x)), np.square, lambda x: 0.5 / np.sqrt(np.maximum(1e-5, x)))
    asinh = (np.arcsinh, np.sinh, lambda x: 1.0 / np.hypot(x, 1))
    return {"log": log, "sqrt": sqrt, "asinh": asinh}


# per quantity: transform, zero_edge, null_outer_rho (mask data past this rho; None=off), lam, eta, and
# the npz value key / display scale. These are DEFAULTS; a project can override them via configure_fits().
QSET = {
    "te":    dict(trans="sqrt",  zero_edge=True,  null_outer_rho=None, lam=0.40, eta=0.5, vkey="Te", scale=1e-3),
    "ne":    dict(trans="sqrt",  zero_edge=False, null_outer_rho=None, lam=0.40, eta=0.5, vkey="ne", scale=1.0),
    "ti":    dict(trans="log",   zero_edge=True,  null_outer_rho=0.98, lam=0.52, eta=0.5, vkey="Ti", scale=1e-3),
    "omega": dict(trans="asinh", zero_edge=False, null_outer_rho=0.98, lam=0.52, eta=0.5, vkey="omega", scale=1.0),
    "nz":    dict(trans="sqrt",  zero_edge=True,  null_outer_rho=0.98, lam=0.52, eta=0.5, vkey="nz", scale=1.0),
}
_PED_RHO, _NR, _DT_S, _NNOISE, _EVEN = 0.97, 101, 0.20, 50, True


def configure_fits(fits):
    """Override the module-level map2grid fit defaults from a shots.json-style `fits` dict, shaped
    {'global': {pedestal_rho, lam, eta, nr_new, dt_s, n_noise_vec, even_fun, robust}, 'channels': {q: {
    transform, zero_edge, null_outer_rho, lam?, eta?, robust?}}}. Keeps a project's fit recipe in its
    config instead of hardcoded here (mitim_tools stays project-agnostic; the caller supplies the dict).
    Unspecified keys keep the QSET/global defaults. Returns {q: effective_robust} (per-channel `robust`
    overriding `global.robust`) so callers pass robust= to fit_*()/fit()."""
    global _PED_RHO, _NR, _DT_S, _NNOISE, _EVEN
    g = fits.get("global", {})
    _PED_RHO = g.get("pedestal_rho", _PED_RHO); _NR = g.get("nr_new", _NR)
    _DT_S = g.get("dt_s", _DT_S); _NNOISE = g.get("n_noise_vec", _NNOISE); _EVEN = g.get("even_fun", _EVEN)
    glam, geta, grobust = g.get("lam"), g.get("eta"), g.get("robust", False)
    robust_by_q = {}
    for q, d in fits.get("channels", {}).items():
        if q not in QSET:
            continue
        if "transform" in d:  QSET[q]["trans"] = d["transform"]
        if "zero_edge" in d:  QSET[q]["zero_edge"] = d["zero_edge"]
        if "null_outer_rho" in d:  QSET[q]["null_outer_rho"] = d["null_outer_rho"]
        QSET[q]["lam"] = d.get("lam", glam if glam is not None else QSET[q]["lam"])
        QSET[q]["eta"] = d.get("eta", geta if geta is not None else QSET[q]["eta"])
        robust_by_q[q] = bool(d.get("robust", grobust))
    return robust_by_q
_R0_BT, _NHARM, _ECE_DT_MS = 1.6955, 2, 5.0            # ECE: R0 for vacuum Bt, 2nd harmonic, thinning
_TS_SYS = ["core", "tangential"]
_CER_VIEWS = [("t", "TANGENTIAL"), ("v", "VERTICAL")]
Z_CHARGE = {"Carbon": 6, "Neon": 10, "Argon": 16}    # effective core charge (Argon Ne-like at 2 keV)

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "M": 1000}


def _roman_to_int(s):
    tot, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN.get(ch, 0)
        tot += -val if val < prev else val
        prev = val
    return tot


def _parse_lineid(lid):
    """CER LINEID -> (element, charge Z). The spectroscopic numeral IS the ion charge: 'C VI 8-7'->('C',6),
    'Ne X 11-10'->('Ne',10), 'B V 7-6'->('B',5), 'Ar XVI ...'->('Ar',16), 'Ar XVIII ...'->('Ar',18)."""
    parts = str(lid).strip().split()
    if len(parts) < 2:
        return (None, None)
    return (parts[0], _roman_to_int(parts[1]))


# impurity species label -> (element, charge Z) for CER IMPDENS selection
NZ_SPECIES = {"C6": ("C", 6), "Ne10": ("Ne", 10), "Ar16": ("Ar", 16), "Ar18": ("Ar", 18), "B5": ("B", 5)}
A_MASS = {"Carbon": 12.0, "Neon": 20.2, "Argon": 39.9}


def _with_conn(exp, kw):
    """Fill kw's connection/cache from `exp` UNLESS the caller already set them (so an explicit
    use_cache=/cache_dir=/connection= in the call overrides the instance's wired defaults instead
    of colliding). Mutates and returns kw."""
    for k, v in (("use_cache", exp.use_cache), ("cache_dir", exp.cache_dir), ("connection", exp._exp_conn)):
        kw.setdefault(k, v)
    return kw


class DIIIDExperiment(DIIIDFetcher):
    """Per-shot DIII-D discharge: retrieval (inherited) + fitting, concentration, input.gacode, plots."""

    def __init__(self, shot, time=4000.0, avg=200.0, t_range=(1400.0, 4150.0), tunnel_host="cybele",
                 cache_dir=None, connection=None, use_cache=True, tree="EFIT01"):
        self.time = float(time)
        self.avg = float(avg)
        self.t_range = (float(t_range[0]), float(t_range[1]))   # full fit window (all time nodes)
        self.tree = tree
        self._own_conn = connection is None
        self._exp_conn = connection or DIIIDConnection(tunnel_host=tunnel_host)
        super().__init__(shot, connection=self._exp_conn, use_cache=use_cache, cache_dir=cache_dir)
        self._cloud_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._own_conn:
            self._exp_conn.close()

    # -------------------------------------------------- rho mapping
    def _efit_slices(self, t0, t1, dt=150.0):
        ets = np.arange(t0, t1 + 1, dt)
        eqs = []
        for et in ets:
            try:
                eqs.append(self.fetch_equilibrium(et, self.tree))
            except Exception:
                eqs.append(None)
        return ets, eqs

    @staticmethod
    def _rho_of(ets, eqs, R, Z, t):
        rho_et = np.array([eq.rho_of(np.array([R]), np.array([Z]), "tor")[0] if eq is not None else np.nan for eq in eqs])
        ok = np.isfinite(rho_et)
        return np.interp(t, ets[ok], rho_et[ok]) if ok.sum() >= 2 else np.full(np.size(t), np.nan)

    def _window(self):
        return self.t_range                           # fit the full window -> all time nodes

    # -------------------------------------------------- clouds (fetch + quality cuts)
    def _ts_cloud(self, quantity, t0, t1):
        ets, eqs = self._efit_slices(t0, t1)
        r, t, v, e = [], [], [], []
        for sysn in [s.upper() for s in _TS_SYS]:
            base = rf"\ELECTRONS::TOP.TS.BLESSED.{sysn}"
            try:
                Te, _ = self._value_cached(f"{base}:TEMP", tree="ELECTRONS"); Tee, _ = self._value_cached(f"{base}:TEMP_E", tree="ELECTRONS")
                De, _ = self._value_cached(f"{base}:DENSITY", tree="ELECTRONS"); Dee, _ = self._value_cached(f"{base}:DENSITY_E", tree="ELECTRONS")
                tt, _ = self._value_cached(f"{base}:TIME", tree="ELECTRONS"); R, _ = self._value_cached(f"{base}:R", tree="ELECTRONS"); Zc, _ = self._value_cached(f"{base}:Z", tree="ELECTRONS")
            except Exception:
                continue
            Te, Tee, De, Dee = (np.atleast_2d(np.asarray(a, float)) for a in (Te, Tee, De, Dee))
            tt, R, Zc = (np.atleast_1d(np.asarray(a, float)) for a in (tt, R, Zc))
            if Te.shape[0] == tt.size and Te.shape[1] != tt.size:
                Te, Tee, De, Dee = Te.T, Tee.T, De.T, Dee.T
            try:
                Tm, _ = self._value_cached(f"{base}:TEMPMASK", tree="ELECTRONS"); Dm, _ = self._value_cached(f"{base}:DENSMASK", tree="ELECTRONS")
                Tm, Dm = np.atleast_2d(Tm), np.atleast_2d(Dm)
                if Tm.shape[0] == tt.size and Tm.shape[1] != tt.size:
                    Tm, Dm = Tm.T, Dm.T
            except Exception:
                Tm, Dm = np.ones_like(Te, bool), np.ones_like(De, bool)
            tsel = (tt >= t0) & (tt <= t1); ti = tt[tsel]
            for i in range(min(Te.shape[0], R.size)):
                rho_i = self._rho_of(ets, eqs, R[i], Zc[i], ti)
                Te_i, Tee_i, De_i, Dee_i = Te[i, tsel], Tee[i, tsel], De[i, tsel], Dee[i, tsel]
                Tm_i = np.bool_(Tm[i, tsel]) if Tm.shape[0] > i else np.ones_like(Te_i, bool)
                Dm_i = np.bool_(Dm[i, tsel]) if Dm.shape[0] > i else np.ones_like(De_i, bool)
                corrupt = ((Te_i < 20) & np.isfinite(Tee_i) & (rho_i < 0.95)) | (De_i == 1e19) | (Te_i == 100) | (R[i] > 2.2)
                if quantity == "te":
                    keep = (Tee_i > 0) & np.isfinite(Tee_i) & (Te_i > 5) & Tm_i & ~corrupt; vals, errs = Te_i, Tee_i   # eV
                else:
                    keep = (Dee_i > 0) & np.isfinite(Dee_i) & (De_i > 0) & Dm_i & ~corrupt; vals, errs = De_i / 1e19, Dee_i / 1e19
                keep &= np.isfinite(rho_i)
                r += list(rho_i[keep]); t += list(ti[keep]); v += list(vals[keep]); e += list(errs[keep])
        return tuple(np.asarray(a) for a in (r, t, v, e))

    def _cer_ti_cloud(self, sources, t0, t1):
        """sources: list of (flat_prefix, IONS_tree, impurity_name). Fetches TEMPC(->TEMP) tang+vert.
        Returns (rho, t, val, err, impurity_label, view_label) so callers can colour/store by species."""
        ets, eqs = self._efit_slices(t0, t1)
        r, t, v, e, imp, view = [], [], [], [], [], []
        for flavor, tree, imp_name in sources:
            for vlet, vname in _CER_VIEWS:
                vlabel = "tangential" if vlet == "t" else "vertical"
                for n in range(1, 81):
                    base = rf"\IONS::TOP.CER.{tree}.{vname}.CHANNEL{n:02d}:"
                    try:
                        sig = self.fetch_signal(base + "TEMPC")
                    except Exception:
                        try:
                            sig = self.fetch_signal(f"{flavor}ti{vlet}{n}")
                        except Exception:
                            continue
                    try:
                        err = np.asarray(self.fetch_signal(base + "TEMP_ERR").data, float)
                    except Exception:
                        err = np.full(np.asarray(sig.data).shape, np.nan)
                    try:
                        R = float(np.nanmedian(self.fetch_signal(f"{flavor}r{vlet}{n}").data)); Z = float(np.nanmedian(self.fetch_signal(f"{flavor}z{vlet}{n}").data))
                    except Exception:
                        continue
                    tt = np.asarray(sig.time, float); Ti = np.asarray(sig.data, float)
                    if err.shape != tt.shape:
                        err = np.full(tt.shape, np.nan)
                    m = (tt >= t0) & (tt <= t1) & np.isfinite(Ti) & (Ti > 0)
                    if not m.any():
                        continue
                    rho_i = self._rho_of(ets, eqs, R, Z, tt[m]); ok = np.isfinite(rho_i)
                    nk = int(ok.sum())
                    r += list(rho_i[ok]); t += list(tt[m][ok]); v += list(Ti[m][ok]); e += list(err[m][ok])   # eV
                    imp += [imp_name] * nk; view += [vlabel] * nk
        return (np.asarray(r), np.asarray(t), np.asarray(v), np.asarray(e),
                np.asarray(imp, dtype="U8"), np.asarray(view, dtype="U12"))

    def _cer_omega_cloud(self, t0, t1):
        """omega = ROTC[km/s]/R[m] -> krad/s, CERAUTO tangential only (ROTC only; raw ROT sign is bad)."""
        ets, eqs = self._efit_slices(t0, t1)
        r, t, v, e = [], [], [], []
        for n in range(1, 81):
            base = rf"\IONS::TOP.CER.CERAUTO.TANGENTIAL.CHANNEL{n:02d}:"
            try:
                sig = self.fetch_signal(base + "ROTC")
            except Exception:
                continue
            try:
                err = np.asarray(self.fetch_signal(base + "ROT_ERR").data, float)
            except Exception:
                err = np.full(np.asarray(sig.data).shape, np.nan)
            try:
                R = float(np.nanmedian(self.fetch_signal(f"cerart{n}").data)); Z = float(np.nanmedian(self.fetch_signal(f"cerazt{n}").data))
            except Exception:
                continue
            tt = np.asarray(sig.time, float); vt = np.asarray(sig.data, float)
            if err.shape != tt.shape:
                err = np.full(tt.shape, np.nan)
            m = (tt >= t0) & (tt <= t1) & np.isfinite(vt) & (R > 0)
            if not m.any():
                continue
            rho_i = self._rho_of(ets, eqs, R, Z, tt[m]); ok = np.isfinite(rho_i)
            r += list(rho_i[ok]); t += list(tt[m][ok]); v += list(vt[m][ok] / R); e += list(err[m][ok] / R)
        return tuple(np.asarray(a) for a in (r, t, v, e))

    def _nimp_geom(self, vlet, vname, n):
        """(R,Z) [m] of CER channel n: physical CERAUTO pointnames (cera{r,z}{t,v}{n}) as Ti/omega use;
        fall back to the analysis-independent CALIBRATION sightline median (some shots lack CERAUTO)."""
        try:
            return (float(np.nanmedian(self.fetch_signal(f"cerar{vlet}{n}").data)),
                    float(np.nanmedian(self.fetch_signal(f"ceraz{vlet}{n}").data)))
        except Exception:
            return (float(np.nanmedian(self._value(rf"\IONS::TOP.CER.CALIBRATION.{vname}.CHANNEL{n:02d}:PLASMA_R"))),
                    float(np.nanmedian(self._value(rf"\IONS::TOP.CER.CALIBRATION.{vname}.CHANNEL{n:02d}:PLASMA_Z"))))

    def _cer_nimp_cloud(self, element, Z, t0, t1, analysis_types=("CERAUTO", "CERQUICK", "CERFIT", "CERNEUR")):
        """Impurity density n_z [m^-3] for (element, charge Z) from the CER IMPDENS flat array — the same
        pre-computed density the QUICKFIT GUI shows in 'impdens' mode. Picks the first analysis type that
        has data; each channel's species is set by its LINEID; the channel's density segment is sliced out
        of the flat IMPDENS array via INDECIES (ordered by CALIBRATION:ARRAY_ORDER); geometry -> rho as for
        Ti/omega. Returns (rho, t_ms, nz, err); empty if the line was not measured this shot."""
        ets, eqs = self._efit_slices(t0, t1)
        self.conn.openTree("IONS", self.shot)          # _efit_slices left EFIT as the open tree; \IONS::_value needs IONS
        for atype in analysis_types:
            try:
                nz = np.atleast_1d(self._value(rf"\IONS::TOP.IMPDENS.{atype}:IMPDENS")).astype(float)
            except Exception:
                continue
            if nz.size <= 1:
                continue
            nzerr = np.atleast_1d(self._value(rf"\IONS::TOP.IMPDENS.{atype}:ERR_IMPDENS")).astype(float)
            ind = np.atleast_1d(self._value(rf"\IONS::TOP.IMPDENS.{atype}:INDECIES")).astype(int)
            timp = np.atleast_1d(self._value(rf"\IONS::TOP.IMPDENS.{atype}:TIME")).astype(float)   # ms
            ao = [str(a) for a in np.atleast_1d(self._value(r"\IONS::TOP.CER.CALIBRATION:ARRAY_ORDER"))]
            ao_fmt = [a[0] + ("0" + a[4:].strip())[-2:] for a in ao]                                # 'VERT1'->'V01'
            r, t, v, e = [], [], [], []
            for vlet, vname in _CER_VIEWS:
                for n in range(1, 49):
                    nm = vlet.upper() + f"{n:02d}"
                    if nm not in ao_fmt:
                        continue
                    try:
                        el, zc = _parse_lineid(self._value(rf"\IONS::TOP.CER.CALIBRATION.{vname}.CHANNEL{n:02d}:LINEID"))
                    except Exception:
                        continue
                    if el != element or zc != Z:
                        continue
                    j = ao_fmt.index(nm)
                    if j + 1 >= ind.size:
                        continue
                    sl = slice(int(ind[j]), int(ind[j + 1]))
                    nz_c, err_c, tc = nz[sl], nzerr[sl], timp[sl]
                    m = (tc >= t0) & (tc <= t1) & np.isfinite(nz_c) & (nz_c > 0) & (nz_c < 1e21)
                    if not m.any():
                        continue
                    try:
                        R, Zc = self._nimp_geom(vlet, vname, n)
                    except Exception:
                        continue
                    rho_i = self._rho_of(ets, eqs, R, Zc, tc[m]); ok = np.isfinite(rho_i)
                    r += list(rho_i[ok]); t += list(tc[m][ok]); v += list(nz_c[m][ok]); e += list(err_c[m][ok])
            if r:
                print(f"    n_{element}{Z}: {len(r)} pts from {atype}", typeMsg="i")
                return tuple(np.asarray(a) for a in (r, t, v, e))
        return tuple(np.array([]) for _ in range(4))

    def _zipfit(self, node, scale):
        """ZIPFIT profile x(rho_tor, t): returns (rho, t_ms, x[nrho,ntime]) scaled to SI."""
        c = self.conn
        c.openTree("ELECTRONS", self.shot)
        x = np.asarray(c.get(rf"_x=\ELECTRONS::TOP.PROFILE_FITS.ZIPFIT.{node}").data(), float)
        d0 = np.asarray(c.get("dim_of(_x,0)").data(), float); d1 = np.asarray(c.get("dim_of(_x,1)").data(), float)
        c.closeTree("ELECTRONS", self.shot)
        rho, t = (d0, d1) if (d0.max() <= 1.5) else (d1, d0)
        if x.shape[0] == t.size:
            x = x.T
        return rho, t, x * scale

    def _ece_cloud(self, t0, t1):
        """ECE-radiometer Te cloud (rho, t_ms, Te[eV], err), replicating QUICKFIT load_ece: 2nd-harmonic
        resonance freq->R (relativistic gamma from ZIPFIT Te), R->rho, with RHC density-cutoff /
        3rd-harmonic / invalid / goes-to-zero masks. ECE error is inflated in the core (soft constraint)."""
        c = self.conn
        freq = np.atleast_1d(np.asarray(self._value_cached(r"\ECE::TOP.SETUP.FREQ", tree="ECE")[0], float)) * 1e9
        z0 = float(np.atleast_1d(self._value_cached(r"\ECE::TOP.SETUP.ECEZH", tree="ECE")[0])[0])
        valid = np.bool_(np.atleast_1d(self._value_cached(r"\ECE::TOP.CALF:VALIDF", tree="ECE")[0]))
        nch = freq.size
        valid = np.r_[valid, np.zeros(max(0, nch - valid.size), bool)][:nch]
        c.openTree("ECE", self.shot)
        te0 = np.asarray(c.get(r"dim_of(\TECE01)").data(), float); pre = te0 < 0
        Te_raw = np.full((nch, te0.size), np.nan)
        for ch in range(1, nch + 1):
            try:
                y = np.asarray(c.get(rf"\TECE{ch:02d}").data(), float) * 1e3
            except Exception:
                continue
            if y.size == te0.size:
                Te_raw[ch - 1] = y - (y[pre].mean() if pre.any() else 0.0)
        c.closeTree("ECE", self.shot)
        win = (te0 >= t0) & (te0 <= t1); tw, Te_w = te0[win], Te_raw[:, win]
        step = max(1, int(round(_ECE_DT_MS / max(np.median(np.diff(tw)), 1e-6))))
        tw, Te_w = tw[::step], Te_w[:, ::step]
        bt = self.fetch_signal("bt")
        rz, tz, Tez = self._zipfit("ETEMPFIT", 1e3); rn, tn, nez = self._zipfit("EDENSFIT", 1e19)
        r_in = np.linspace(1.05, 2.35, 400); efit_t = np.arange(t0, t1 + 1, 150.0)
        rho_e = np.full((efit_t.size, nch), np.nan); fcut_e = np.full((efit_t.size, nch), np.nan)
        tez_e = np.full((efit_t.size, nch), np.nan); f3h_e = np.full(efit_t.size, np.inf)
        for i, et in enumerate(efit_t):
            try:
                eq = self.fetch_equilibrium(et, self.tree)
            except Exception:
                continue
            B0 = abs(float(np.interp(et, bt.time, bt.data))); wce_cold = e * (B0 * _R0_BT / r_in) / m_e
            hrho = eq.rho_of(r_in, np.full_like(r_in, z0), "tor")
            Te_zip = Tez[:, int(np.argmin(np.abs(tz - et)))]; ne_zip = nez[:, int(np.argmin(np.abs(tn - et)))]
            Te_R = np.interp(hrho, rz, Te_zip, left=Te_zip[0], right=0.0)
            ne_R = np.interp(hrho, rn, ne_zip, left=ne_zip[0], right=0.0)
            gamma = 1.0 / np.sqrt(1 - np.clip(2 * Te_R * e / m_e / c_light**2, 0, 0.99))
            R_res = np.interp(-2 * np.pi * freq, -(wce_cold / gamma) * _NHARM, r_in)
            rho_e[i] = eq.rho_of(R_res, np.full_like(R_res, z0), "tor")
            tez_e[i] = np.interp(rho_e[i], rz, Te_zip, left=Te_zip[0], right=0.0)
            f_ce = wce_cold / (2 * np.pi); f_pe = np.sqrt(np.maximum(ne_R, 0) * e**2 / (m_e * epsilon_0)) / (2 * np.pi)
            f_RHC = 0.5 * f_ce + np.sqrt((0.5 * f_ce)**2 + f_pe**2); f_cut = np.maximum.accumulate(f_RHC[::-1])[::-1]
            fcut_e[i] = np.interp(R_res, r_in, f_cut); f3h_e[i] = 3 * np.interp(float(eq.rbbbs.max()), r_in, f_ce)
        ok = np.isfinite(rho_e[:, 0])
        if ok.sum() < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])
        rho_c = interp1d(efit_t[ok], rho_e[ok], axis=0, fill_value="extrapolate")(tw)
        fcut_c = interp1d(efit_t[ok], fcut_e[ok], axis=0, fill_value="extrapolate")(tw)
        tez_c = interp1d(efit_t[ok], tez_e[ok], axis=0, fill_value="extrapolate")(tw)
        f3h_c = np.interp(tw, efit_t[ok], f3h_e[ok]); Te_c = Te_w.T
        keep = (np.isfinite(Te_c) & (Te_c > 20) & np.isfinite(rho_c) & valid[None, :]
                & (fcut_c <= freq[None, :]) & (freq[None, :] <= f3h_c[:, None]) & (Te_c >= 0.5 * tez_c))
        tt = np.broadcast_to(tw[:, None], Te_c.shape); r, t, v = rho_c[keep], tt[keep], Te_c[keep]
        err = (0.10 + 0.15 * np.clip(0.35 - r, 0, 0.35) / 0.35) * np.abs(v) + 100.0
        return r, t, v, err

    # -------------------------------------------------- fit
    def fit(self, quantity, cloud, robust=False, extract_time=None, labels=None):
        """map2grid fit of a (rho, t_ms, val, err) cloud; returns the profile at the node nearest
        `extract_time` (default self.time). `labels` (dict of per-point arrays) are filtered in sync
        with the data and returned as data_<key> (e.g. imp/view for CER-Ti species colouring)."""
        map2grid = _load_map2grid()
        trans = _transforms()[QSET[quantity]["trans"]]; s = QSET[quantity]
        rho, tms, val, err = cloud
        good = np.isfinite(rho) & np.isfinite(val)
        if quantity in ("te", "ne", "ti", "nz"):
            good &= (val > 0)
        good &= ~(np.isfinite(err) & (err < 0))   # drop CER "disabled-by-default" points (negative-err sentinel:
        #                                           edge-split/HFS chords the GUI greys out; map2grid masks err<=0)
        lab = {k: np.asarray(vv)[good] for k, vv in labels.items()} if labels else {}
        rho, tms, val, err = (a[good] for a in (rho, tms, val, err))
        ve = np.isfinite(err) & (err > 0)
        err_plot = np.where(ve, err, np.nan)              # measurement error for display (NaN where invalid)
        ef = np.where(ve, err, np.median(err[ve]) if ve.any() else 1.0).astype(float)
        if s["null_outer_rho"] is not None:
            ef[rho > s["null_outer_rho"]] = -1.0
        MG = map2grid(rho, tms / 1e3, val, ef, nr_new=_NR, dt=_DT_S)
        MG.PrepareCalculation(zero_edge=s["zero_edge"], core_discontinuties=[], edge_discontinuties=[],
                              transformation=trans, pedestal_rho=_PED_RHO, robust_fit=robust, elm_phase=None, even_fun=_EVEN)
        MG.PreCalculate(); MG.Calculate(s["lam"], s["eta"], n_noise_vec=_NNOISE)
        rg = np.asarray(MG.r_new); rg = rg[0] if rg.ndim == 2 else np.ravel(rg)
        tg = np.asarray(MG.g_t); tg_ms = (tg[:, 0] if tg.ndim == 2 else np.ravel(tg)) * 1e3
        it = int(np.argmin(np.abs(tg_ms - (extract_time or self.time))))
        res = dict(rgrid=rg, prof=np.asarray(MG.g)[it], gfit=np.asarray(MG.g), gu=np.asarray(MG.g_u),
                   gd=np.asarray(MG.g_d), tg_ms=tg_ms, t_node=float(tg_ms[it]), chi2=float(MG.chi2),
                   data_rho=rho, data_t=tms, data_val=val, data_err=err_plot)
        res.update({f"data_{k}": vv for k, vv in lab.items()})
        return res

    def _cache_cloud(self, key, fn):
        if key not in self._cloud_cache:
            self._cloud_cache[key] = fn()
        return self._cloud_cache[key]

    def fit_te(self, robust=False, use_ece=False):
        """Te fit. use_ece=True adds the ECE radiometer to the Thomson cloud (constrains the core,
        esp. for shots lacking tangential TS like 207959)."""
        t0, t1 = self._window()
        cloud = self._cache_cloud(("te", t0, t1), lambda: self._ts_cloud("te", t0, t1))
        if use_ece:
            ec = self._cache_cloud(("ece", t0, t1), lambda: self._ece_cloud(t0, t1))
            if ec[0].size:
                cloud = tuple(np.concatenate([a, b]) for a, b in zip(cloud, ec))
        return self.fit("te", cloud, robust)

    def fit_ne(self, robust=False):
        t0, t1 = self._window()
        return self.fit("ne", self._cache_cloud(("ne", t0, t1), lambda: self._ts_cloud("ne", t0, t1)), robust)

    def fit_ti(self, robust=False, sources=None):
        """sources: list of (flat_prefix, IONS_tree, impurity_name); default Carbon (CERAUTO) only.
        e.g. [("cera","CERAUTO","Carbon"),("cerf","CERFIT","Neon")] for Carbon + the CERFIT impurity."""
        t0, t1 = self._window()
        sources = tuple(tuple(s) for s in sources) if sources else (("cera", "CERAUTO", "Carbon"),)
        c6 = self._cache_cloud(("ti", t0, t1, sources), lambda: self._cer_ti_cloud(sources, t0, t1))
        return self.fit("ti", c6[:4], robust, labels={"imp": c6[4], "view": c6[5]})

    def fit_omega(self, robust=False):
        t0, t1 = self._window()
        return self.fit("omega", self._cache_cloud(("omega", t0, t1), lambda: self._cer_omega_cloud(t0, t1)), robust)

    def fit_nimp(self, species, robust=False):
        """Impurity density fit for `species` in NZ_SPECIES ('C6','Ne10','Ar16','Ar18','B5'), from the CER
        IMPDENS leaf (auto-picks CERAUTO/CERQUICK/...). Returns the fit() dict, or None if the shot has no
        IMPDENS data for that line (caller draws a 'no data' panel)."""
        element, Z = NZ_SPECIES[species]
        t0, t1 = self._window()
        cloud = self._cache_cloud(("nz", species, t0, t1), lambda: self._cer_nimp_cloud(element, Z, t0, t1))
        if cloud[0].size == 0:
            return None
        return self.fit("nz", cloud, robust)

    def load_fit(self, tag, cache_dir=None):
        """Load a previously-saved map2grid fit for THIS shot from disk: reads
        <cache_dir>/quickfit_<tag>_<shot>.npz and returns it as a plain dict (format-agnostic — no
        knowledge of the npz keys, so the saver owns the schema). No MDS/QUICKFIT needed. Lets
        multi-shot comparisons build one instance per shot and pull each shot's stored fit."""
        fp = Path(cache_dir or self.cache_dir) / f"quickfit_{tag}_{self.shot}.npz"
        d = np.load(fp)
        return {k: d[k] for k in d.files}

    # -------------------------------------------------- impurity concentration (Zeff, uniform model)
    def impurity_concentration(self, impurity=None, puff_time=None):
        """(c_C, c_imp, imp_name) at self.time from the measured Zeff (uniform-concentration model).
        Carbon pre-puff; if `impurity`+`puff_time` given and self.time>puff_time, split off c_imp."""
        z = self.fetch_signal("zeff"); tz = np.asarray(z.time, float); zeff = np.asarray(z.data, float)
        g = np.isfinite(zeff) & (zeff > 0.9) & (zeff < 5.0)
        zt, zv = tz[g], zeff[g]
        zeff_now = float(np.interp(self.time, zt, zv))
        cC = (zeff_now - 1.0) / 30.0
        cimp, imp = 0.0, None
        if impurity and puff_time and self.time > puff_time:
            pre = (zt >= puff_time - 500) & (zt <= puff_time - 50)
            cC = float(np.mean((zv[pre] - 1.0) / 30.0)) if pre.any() else cC
            Zi = Z_CHARGE[impurity]
            cimp = max(0.0, (zeff_now - 1.0 - 30.0 * cC) / (Zi * (Zi - 1)))
            imp = impurity
        return cC, cimp, imp

    # -------------------------------------------------- input.gacode
    def to_gacode(self, ti_sources=None, impurity=None, puff_time=None, out_dir=None, plot_data=False,
                  heating=False, radiation="experimental", fits=None, concentration=None,
                  heating_scalars=None, out_tag=None, ion_profiles=None):
        """Build a MITIM gacode_state / input.gacode from fits at self.time. Sources/heating=0 unless
        heating=True, which adds on-axis Gaussian (peaking-5) NBI (50/50 e/i), Ohmic and radiation from
        the experimental total powers, then recomputes qei/qfus (+qrad if radiation='analytic') from the
        fitted kinetics. w0(rad/s)=omega*1e3 (rigid-rotor v_tor/R; sign follows CER ROTC, unverified).
        Overrides for the merged multi-shot path (DIIIDMultiShot.to_gacode): `fits` (skip the per-shot
        map2grid fits and use these {q:{rgrid,prof,...}}), `concentration` ((cC,cimp,imp), skip the Zeff
        model), `heating_scalars` ((P_nbi,P_ohm,P_rad) MW, skip the per-shot power reads), `out_tag`
        (filename stem, default the shot number). All None -> unchanged single-shot behavior.
        `ion_profiles` {species: dict(rgrid, prof[m^-3], Z, mass, name)}: populate the thermal ions from
        MEASURED impurity DENSITY profiles (n_D by quasineutrality) instead of the uniform-concentration
        Zeff model -- one ion per stored charge state, fully-stripped assumption (n_C6~=n_C, n_Ne10~=n_Ne
        in the core; breaks for Ar). When given, the Zeff model + its Zeff-signal fetch are skipped."""
        from mitim_tools.gs_tools.GEQtools import MITIMgeqdsk
        _load_map2grid()
        out_dir = Path(out_dir) if out_dir else (Path(self.cache_dir).parent / "gacode_from_diiid" if self.cache_dir else Path("."))
        out_dir.mkdir(parents=True, exist_ok=True)

        geqf = self.fetch_geqdsk(self.time, self.tree, path=out_dir / f"g{self.shot}.{int(self.time):05d}")
        if fits is None:
            fits = {"te": self.fit_te(True), "ne": self.fit_ne(True),
                    "ti": self.fit_ti(True, sources=ti_sources or [("cera", "CERAUTO", "Carbon")]), "omega": self.fit_omega(True)}
        if concentration is not None:
            cC, cimp, imp = concentration
        elif ion_profiles is None:
            cC, cimp, imp = self.impurity_concentration(impurity, puff_time)
        else:
            cC, cimp, imp = 0.01, 0.0, None   # nominal: ion densities come from ion_profiles below; this only seeds geq.to_profiles
        for q, r in fits.items():
            print(f"    {q:<6s}: chi2={r['chi2']:.2f}, node@{r['t_node']:.0f} ms", typeMsg="i")
        if ion_profiles is not None:
            print(f"    ions from MEASURED n_z: {', '.join(ion_profiles)} (n_D quasineutral)", typeMsg="i")
        else:
            print(f"    c_C={cC*100:.2f}%" + (f", c_{imp}={cimp*100:.2f}%" if imp else " (Carbon only)"), typeMsg="i")

        geq = MITIMgeqdsk(geqf)
        p = geq.to_profiles(ne0_20=float(fits["ne"]["prof"][0]) / 10.0, Zeff=1.0 + 30.0 * cC, Z=6)
        rho = p.profiles["rho(-)"]
        onto = lambda q: np.interp(rho, fits[q]["rgrid"], fits[q]["prof"])
        te = np.maximum(onto("te") * 1e-3, 1e-3)          # eV -> keV
        ti = np.maximum(onto("ti") * 1e-3, 1e-3)          # eV -> keV
        ne = np.maximum(onto("ne"), 1e-3)                 # 10^19 m^-3 (gacode units)
        w0 = onto("omega") * 1e3                          # krad/s -> rad/s
        if ion_profiles is not None:                          # MEASURED impurity densities -> one ion per charge state, n_D quasineutral
            names, zs, masses, nis = ["D"], [1.0], [2.0], [None]
            charge_sum = np.zeros_like(rho)
            for sp, ip in ion_profiles.items():               # ip: dict(rgrid, prof[m^-3], Z, mass, name)
                nz = np.maximum(np.interp(rho, ip["rgrid"], ip["prof"]), 0.0) / 1e19   # m^-3 -> 10^19 (gacode units)
                names.append(ip["name"]); zs.append(float(ip["Z"])); masses.append(float(ip["mass"])); nis.append(nz)
                charge_sum = charge_sum + float(ip["Z"]) * nz
            nD = ne - charge_sum
            if np.any(nD <= 0):
                print(f"    WARNING: quasineutral n_D <= 0 at {int((nD <= 0).sum())}/{rho.size} radii "
                      f"(measured impurities over-count the charge here)", typeMsg="w")
            nis[0] = np.maximum(nD, 1e-3)
        else:                                                 # uniform-concentration Zeff model (default)
            nC = cC * ne; nimp = cimp * ne
            nD = ne - 6.0 * nC - (Z_CHARGE[imp] * nimp if imp else 0.0)
            if imp and cimp > 0:
                names, zs, masses, nis = ["D", "C", imp[:2]], [1.0, 6.0, float(Z_CHARGE[imp])], [2.0, 12.0, A_MASS[imp]], [nD, nC, nimp]
            else:
                names, zs, masses, nis = ["D", "C"], [1.0, 6.0], [2.0, 12.0], [nD, nC]
        nion = len(names)
        p.profiles["te(keV)"] = te; p.profiles["ne(10^19/m^3)"] = ne
        p.profiles["ni(10^19/m^3)"] = np.array(nis).T; p.profiles["ti(keV)"] = np.array([ti] * nion).T
        p.profiles["w0(rad/s)"] = w0
        p.profiles["nion"] = np.array([str(nion)]); p.profiles["name"] = np.array(names)
        p.profiles["type"] = np.array(["[therm]"] * nion); p.profiles["mass"] = np.array(masses); p.profiles["z"] = np.array(zs)
        for key in list(p.profiles.keys()):
            if any(key.startswith(s) for s in ("qohme", "qbeame", "qbeami", "qrfe", "qrfi", "qfuse", "qfusi",
                                               "qsync", "qbrem", "qline", "qei", "qione", "qpar", "qmom")):
                p.profiles[key] = np.zeros_like(rho)
        if heating:
            self._add_gaussian_heating(p, radiation=radiation, powers=heating_scalars)
        p.derive_quantities()
        assert len(p.profiles["z"]) == nion == len(p.profiles["name"]) == len(p.profiles["mass"]), "species mismatch"
        tag = out_tag if out_tag is not None else str(self.shot)
        out = out_dir / f"input.gacode_{tag}_{int(self.time)}"
        p.write_state(out)
        print(f"[{tag}] wrote {out} (nion={nion}, Zeff0={p.profiles['z_eff(-)'][0]:.2f}, q95={p.derived['q95']:.2f})", typeMsg="i")
        if plot_data:
            self._plot_to_gacode(geq, fits, p, cC, cimp, imp, out_dir, tag=tag, ion_profiles=ion_profiles)
        return p

    def _power_MW(self, signal, scale):
        """Window-averaged total power of a DIII-D signal (MW). `scale` converts the raw signal to MW
        (pinj: 1e-3 [kW->MW]; poh/prad_tot: 1e-6 [W->MW])."""
        s = self.fetch_signal(signal); t = np.asarray(s.time, float); y = np.asarray(s.data, float)
        m = (t >= self.time - self.avg) & (t <= self.time + self.avg)
        return float(np.nanmean(y[m])) * scale

    def _add_gaussian_heating(self, p, radiation="experimental", powers=None):
        """On-axis Gaussian (parabolicProfile nu=5 -> peaking 5) heating from the experimental total
        powers: NBI split 50/50 e/i, Ohmic (e), radiation (e). Normalization reuses the gacode volume
        machinery (a unit shape's integrated power). Then recompute the qei exchange + qfus alpha (and
        qrad radiation if radiation='analytic') self-consistently from the fitted Te/Ti/ne with the
        analytic target model. NOTE: pinj is INJECTED NBI power (not absorbed). `powers`: optional
        pre-pooled (P_nbi, P_ohm, P_rad) MW (merged path); else read this shot's signals."""
        from mitim_tools.misc_tools import PLASMAtools
        rho = p.profiles["rho(-)"]
        if powers is None:
            P_nbi = self._power_MW("pinj", 1e-3); P_ohm = self._power_MW("poh", 1e-6); P_rad = self._power_MW("prad_tot", 1e-6)
        else:
            P_nbi, P_ohm, P_rad = powers
        _, shape = PLASMAtools.parabolicProfile(Tbar=1.0, nu=5.0, rho=rho, Tedge=0.0)   # peaking-5, centered on axis
        p.profiles["qohme(MW/m^3)"] = shape; p.derive_quantities(rederiveGeometry=False)
        g = shape / float(p.derived["qOhm_MW"][-1])          # unit shape volume-normalized to 1 MW total
        p.profiles["qbeame(MW/m^3)"] = g * 0.5 * P_nbi       # NBI 50/50 electrons / ions
        p.profiles["qbeami(MW/m^3)"] = g * 0.5 * P_nbi
        p.profiles["qohme(MW/m^3)"]  = g * P_ohm             # Ohmic -> electrons
        targets = ["qie", "qfus"]
        if radiation == "analytic":
            targets.append("qrad")                            # analytic radiation (overwrites qbrem/qsync/qline)
        else:
            p.profiles["qbrem(MW/m^3)"] = g * P_rad          # experimental total radiation -> electron sink
        p.recompute_targets(targets=targets)
        print(f"    heating (Gaussian nu=5): P_NBI={P_nbi:.2f} (50/50), P_ohm={P_ohm:.2f}, "
              f"P_rad={P_rad:.2f} MW; recomputed {targets} (radiation={radiation})", typeMsg="i")

    def _plot_to_gacode(self, geq, fits, p, cC, cimp, imp, out_dir, tag=None, ion_profiles=None):
        import matplotlib.pyplot as plt
        lab = tag if tag is not None else str(self.shot)
        rho = p.profiles["rho(-)"]
        fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.28)
        axeq = fig.add_subplot(gs[:, 0])
        geq.plotFluxSurfaces(ax=axeq, fluxes=np.linspace(0.1, 1.0, 10), rhoPol=False, sqrt=True, color="0.6", plot1=True)
        p.plot_state_flux_surfaces(ax=axeq, surfaces_rho=np.linspace(0.1, 1.0, 10), color="tab:red")
        axeq.set_title(f"#{self.shot} EFIT @ {self.time:.0f} ms"); axeq.set_aspect("equal")
        for k, (q, ylab) in enumerate([("te", "$T_e$ [keV]"), ("ne", "$n_e$ [$10^{19}$]"), ("ti", "$T_i$ [keV]"), ("omega", r"$\omega_\phi$ [krad/s]")]):
            ax = fig.add_subplot(gs[k // 2, 1 + k % 2]); r = fits[q]; sc = QSET[q]["scale"]
            near = np.abs(r["data_t"] - r["t_node"]) <= (self.avg / 2 + 60)
            ax.plot(r["data_rho"][near], r["data_val"][near] * sc, ".", ms=3, color="0.6", alpha=0.4, label="data")
            ax.plot(r["rgrid"], r["prof"] * sc, "-", color="tab:red", lw=2, label="fit")
            ax.set_xlim(0, 1.05); ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.set_xlabel(r"$\rho_{tor}$"); ax.set_ylabel(ylab)
            ax.set_title(f"{q} (chi2={r['chi2']:.2f})"); ax.legend(fontsize=8)
        sp = "+".join(list(p.profiles["name"]))
        comp = (f"ions from measured $n_z$; $Z_{{eff}}$(0)={p.profiles['z_eff(-)'][0]:.2f}" if ion_profiles is not None
                else f"c_C={cC*100:.2f}%" + (f", c_{imp}={cimp*100:.2f}%" if imp else ""))
        fig.suptitle(f"{lab} @ {self.time:.0f} ms -> input.gacode   species: {sp}   {comp}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"diiid_to_gacode_{lab}_{int(self.time)}.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # -------------------------------------------------- convenience plots (wrap plotting.py, multi-shot capable)
    def _default_overview_layout(self):
        P, T = _plotting.Panel, _plotting.Trace
        return [
            P("$I_p$ [MA]", [T("ip", scale=1e-6)]),
            P("$B_t$ [T]", [T("bt")]),
            P(r"$\bar n_e$ [$10^{19}$]", [T("density", scale=1e-19)]),
            P("$Z_{eff}$, Z valve [V]", [T("zeff"), T(["gasb", "gasc"], reduce="sum")], ylim=(0.0, 3.5)),
            P("$T_e$ core TS [keV]", [T("tste_00", scale=1e-3)]),
            P("$P_{rad}$ [MW]", [T("prad_tot", scale=1e-6)]),
        ]

    def overview(self, layout=None, **kw):
        """Fetch + plot an engineering/kinetic overview for this shot (delegates to plotting.overview;
        a sensible default layout is used if none is given). kw overrides the wired connection/cache."""
        return _plotting.overview([self.shot], layout or self._default_overview_layout(), **_with_conn(self, kw))

    def plot_cer_coverage(self, quantity="tit", **kw):
        return _plotting.cer_coverage([self.shot], quantity=quantity, **_with_conn(self, kw))

    def plot_cer_profiles(self, **kw):
        kw.setdefault("time", self.time)
        return _plotting.profiles_cer([self.shot], **_with_conn(self, kw))

    # -------------------------------------------------- multi-shot comparison
    @classmethod
    def multishot(cls, *exps):
        """Group several DIIIDExperiment instances into a DIIIDMultiShot for comparison plots:
            ab = DIIIDExperiment.multishot(a, b); ab.overview(); fits = ab.load_fits("ti")
        Accepts multishot(a, b, ...) or multishot([a, b, ...])."""
        return DIIIDMultiShot(*exps)


class DIIIDMultiShot:
    """A group of DIIIDExperiment instances for multi-shot comparison — the object returned by
    DIIIDExperiment.multishot(). Build it once, then call comparison methods on it:

        ab = DIIIDExperiment.multishot(a, b)
        ab.overview()                 # overlay the engineering/kinetic traces (LIVE)
        ab.plot_cer_coverage()        # overlay the CER (rho, t) coverage (LIVE)
        fits = ab.load_fits("ti")   # {shot: stored fit} for project comparison figures

    The comparison overlays SEVERAL shots, so it can't be an instance method of a single-shot
    DIIIDExperiment (no shot is privileged). Every LIVE fetch is routed through the FIRST
    experiment's connection + cache, so a comparison opens exactly ONE SSH tunnel (the other
    instances' lazy connections are never touched). The overlay method names mirror the single-shot
    ones (exp.overview() <-> ab.overview()). This class is deliberately GENERAL: experiment-specific
    comparison figures live in the caller and consume load_fits(), not here."""

    def __init__(self, *exps):
        if len(exps) == 1 and not isinstance(exps[0], DIIIDExperiment):
            exps = tuple(exps[0])                      # multishot([a, b]) as well as multishot(a, b)
        self.exps = list(exps)

    @property
    def shots(self):
        return [e.shot for e in self.exps]

    def __len__(self):
        return len(self.exps)

    def __iter__(self):
        return iter(self.exps)

    def overview(self, layout=None, **kw):
        """Overlay the engineering/kinetic overview (default layout if none) of all shots. kw
        overrides the wired connection/cache (e.g. use_cache=False, shade=, vlines=, colors=)."""
        e0 = self.exps[0]
        return _plotting.overview(self.shots, layout or e0._default_overview_layout(), **_with_conn(e0, kw))

    def plot_cer_coverage(self, quantity="tit", **kw):
        """Overlay the CER (rho, t) coverage of all shots."""
        return _plotting.cer_coverage(self.shots, quantity=quantity, **_with_conn(self.exps[0], kw))

    def plot_cer_profiles(self, **kw):
        """Overlay the time-averaged CER profiles of all shots (at exps[0].time by default)."""
        kw.setdefault("time", self.exps[0].time)
        return _plotting.profiles_cer(self.shots, **_with_conn(self.exps[0], kw))

    def load_fits(self, tag, cache_dir=None):
        """{shot: load_fit(tag)} for every experiment that has that stored fit on disk (a shot whose
        npz is missing is skipped). No MDS/QUICKFIT; opens no tunnel."""
        out = {}
        for e in self.exps:
            try:
                out[e.shot] = e.load_fit(tag, cache_dir=cache_dir)
            except FileNotFoundError:
                pass
        return out

    # -------------------------------------------------- merged-repeat kinetic fit (pooled clouds)
    def _merged_cloud(self, q):
        """Concatenate the group's stored `<q>` data clouds (te/ne/ti/omega); fit settings are
        read from the first available shot's npz. Returns (rho, tms, val, err, meta) or None. No MDS:
        reads the per-shot robust clouds written by the profile-fit scripts (each mapped to rho by its
        own EFIT), so pooling repeats is a straight concatenation in (rho, t)."""
        vk = QSET[q]["vkey"]; tag = f"{q}"   # clean per-channel tag (single primary fit; robustness set by fits.global.robust)
        rho, tms, val, err, meta = [], [], [], [], None
        for e in self.exps:
            try:
                d = e.load_fit(tag)
            except FileNotFoundError:
                print(f"    ! missing quickfit_{tag}_{e.shot}.npz -> skipped", typeMsg="w"); continue
            rho.append(d["data_rho"]); tms.append(d["data_t_ms"])
            val.append(d[f"data_{vk}"]); err.append(d[f"data_{vk}_err"])
            if meta is None:
                meta = dict(transform=str(d["transform"]), lam=float(d["lam"]), eta=float(d["eta"]),
                            pedestal_rho=float(d["pedestal_rho"]), dt_s=float(d["dt_s"]),
                            zero_edge=bool(int(d["zero_edge"])) if "zero_edge" in d else QSET[q]["zero_edge"])
        if not rho:
            return None
        return (np.concatenate(rho), np.concatenate(tms), np.concatenate(val), np.concatenate(err), meta)

    def _refit_merged(self, q, cloud):
        """map2grid refit (robust) of the merged cloud, returning a fit dict in the single-shot fit()
        format (rgrid, gfit[time, rho], tg_ms, chi2, data_*) so it can feed to_gacode(fits=...). Same
        settings path as the standalone merged-repeat summary (transform/lam/eta/pedestal/dt from the
        npz; null_outer_rho per QSET; nr_new/n_noise module defaults)."""
        map2grid = _load_map2grid(); vk = QSET[q]["vkey"]
        rho, tms, val, err, meta = cloud
        v = val.astype(float); e = err.astype(float)
        good = np.isfinite(rho) & np.isfinite(v) & (v > 0)
        rho, tms, v, e = rho[good], tms[good], v[good], e[good]
        ve = np.isfinite(e) & (e > 0)
        e_fit = np.where(ve, e, np.median(e[ve]) if ve.any() else 1.0).astype(float)
        s = QSET[q]   # use the (configure_fits'd) module settings so per-shot and merged fits agree
        if s["null_outer_rho"] is not None:
            e_fit[rho > s["null_outer_rho"]] = -1.0
        MG = map2grid(rho, tms / 1e3, v, e_fit, nr_new=_NR, dt=_DT_S)
        MG.PrepareCalculation(zero_edge=s["zero_edge"], core_discontinuties=[], edge_discontinuties=[],
                              transformation=_transforms()[s["trans"]], pedestal_rho=_PED_RHO,
                              robust_fit=True, elm_phase=None, even_fun=_EVEN)
        MG.PreCalculate(); MG.Calculate(s["lam"], s["eta"], n_noise_vec=_NNOISE)
        rg = np.asarray(MG.r_new); rg = rg[0] if rg.ndim == 2 else np.ravel(rg)
        tg = np.asarray(MG.g_t); tg_ms = (tg[:, 0] if tg.ndim == 2 else np.ravel(tg)) * 1e3
        return dict(rgrid=rg, gfit=np.asarray(MG.g), gu=np.asarray(MG.g_u), gd=np.asarray(MG.g_d),
                    tg_ms=tg_ms, chi2=float(MG.chi2), data_rho=rho, data_t=tms, data_val=v,
                    data_err=np.where(ve, e, np.nan))

    def merged_fits(self, extract_time=None):
        """{q: fit dict} for te/ne/ti/omega from the pooled-repeat clouds, each at the fit time-node
        nearest `extract_time` (default the first shot's time). Dict shape matches the single-shot
        fit() so it can be passed straight to (DIIIDExperiment.)to_gacode(fits=...)."""
        t = extract_time if extract_time is not None else self.exps[0].time
        fits = {}
        for q in ("te", "ne", "ti", "omega"):
            cloud = self._merged_cloud(q)
            if cloud is None:
                continue
            r = self._refit_merged(q, cloud)
            it = int(np.argmin(np.abs(r["tg_ms"] - t)))
            r["prof"] = r["gfit"][it]; r["t_node"] = float(r["tg_ms"][it])
            fits[q] = r
            print(f"    {q:<6s} (merged {'+'.join(map(str, self.shots))}): {r['data_rho'].size} pts, "
                  f"chi2={r['chi2']:.2f}, node@{r['t_node']:.0f} ms", typeMsg="i")
        return fits

    def to_gacode(self, impurity=None, puff_times=None, out_dir=None, out_tag=None,
                  heating=False, radiation="experimental", plot_data=False):
        """Build ONE input.gacode for this CONDITION from the MERGED (pooled-repeat) kinetic fits, on
        the FIRST shot's equilibrium, with Zeff -> (cC, cimp) and the heating powers POOLED (mean) over
        the repeats. `puff_times`: per-shot impurity onsets [ms] (list matching self.exps) so each
        shot's pre-puff carbon is taken from its OWN window before averaging; `impurity`: the (shared)
        puffed species; `out_tag`: filename stem (e.g. the condition). Assembly is delegated to the
        first shot's to_gacode via its fits/concentration/heating_scalars overrides."""
        rep = self.exps[0]                                            # first shot -> equilibrium + assembly host
        puffs = puff_times if puff_times is not None else [None] * len(self.exps)
        ccs = [e.impurity_concentration(impurity, pt) for e, pt in zip(self.exps, puffs)]
        cC = float(np.mean([c[0] for c in ccs])); cimp = float(np.mean([c[1] for c in ccs]))
        imp = next((c[2] for c in ccs if c[2]), None)
        P = lambda sig, sc: float(np.mean([e._power_MW(sig, sc) for e in self.exps]))
        heating_scalars = (P("pinj", 1e-3), P("poh", 1e-6), P("prad_tot", 1e-6)) if heating else None
        print(f"  [{out_tag or 'condition'}] eq from first shot #{rep.shot}; pooled over {self.shots}: "
              f"c_C={cC*100:.2f}%" + (f", c_{imp}={cimp*100:.2f}%" if imp else " (Carbon only)"), typeMsg="i")
        fits = self.merged_fits(rep.time)
        return rep.to_gacode(out_dir=out_dir, plot_data=plot_data, heating=heating, radiation=radiation,
                             fits=fits, concentration=(cC, cimp, imp), heating_scalars=heating_scalars,
                             out_tag=out_tag)
