"""DIII-D experimental-data retrieval (MDSplus, via the pure-python `mdsthin`).

Reusable engine for pulling experimental traces, EFIT equilibria and diagnostic
profiles from the DIII-D tokamak. The *selection* of which signals to grab, and
the plotting, live elsewhere (see `plotting.py` and the capability test); this
module only knows how to connect and fetch.

Signal resolution uses the standard DIII-D MDSplus access cascade (the DIII-D
server-side TDI functions `findsig` -> `ptdata2` -> `pseudo`). A signal spec is
one of:
    * a bare pointname            ->  findsig() locates its tree+node; if that
                                      aborts it is a true PTDATA pointname and
                                      we fall back to `ptdata2()`, then `pseudo()`
    * `PTDATA::<name>`            ->  PTDATA explicitly
    * `<TREE>::<expr>`            ->  open <TREE>, evaluate <expr>
    * a full node `\\<TREE>::...`  ->  open <TREE>, evaluate the node path
Every fetch then reads value + `dim_of(_s,0)` (time base) + `units(_s)`.

Being polite to the server (atlas.gat.com is shared; admins notice heavy I/O):
    * ONE SSH tunnel + ONE mdsplus connection is reused across all shots
      (`DIIIDConnection`); do not open a tunnel per shot.
    * Large traces are **resampled on the server** (`resample(...)`) so only the
      reduced array crosses the wire — a raw PTDATA pointname is ~0.5-1 M points
      over the full [-4 s, +20 s] digitizer record; we transfer ~`max_points`.
    * Fetches are **cached to disk** (keyed by shot+spec+max_points), so
      re-running an analysis does not hit atlas again.
    * Fetching is serial (no parallel hammering).

Conventions / units (DIII-D):
    * Time base is **milliseconds** for PTDATA and tree nodes.
    * Values are returned as stored, with the units MDSplus reports.

Connecting to the server:
    The DIII-D MDSplus server `atlas.gat.com:8000` is only reachable from inside
    GA. With the default `tunnel_host=None` (and no "mdsplus"/"diiid" block in
    config_user.json, see `MDStools`) the client connects to it directly,
    which works on a GA host or over the GA VPN. Off-site, pass
    `tunnel_host=<your jump host>` (a passwordless `~/.ssh/config` entry that can
    reach atlas:8000, e.g. a GA gateway such as `cybele`) and an SSH tunnel is
    opened for you (reused across shots):

        ssh -N -L <localport>:atlas.gat.com:8000 <tunnel_host>

    so nothing has to be installed or running on the GA side. To target a
    different server explicitly, pass `server="host:port"`.

`mdsthin` is an optional dependency: ``pip install mitim-fusion[mds]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mitim_tools import __mitimroot__
# Machine-agnostic layer (re-exported here so existing `diiid.retrieval` imports keep working).
from mitim_tools.experiment_tools.MDStools import (       # noqa: F401
    _mds, _b2s, Signal, EquilibriumData, ChannelProfile, time_average, orient_psi,
    reduce_channels, _write_geqdsk, _pick_free_port, SSHTunnel, MDSConnection, MDSFetcher)

# CER stored measurement error: the per-chord `<quantity>_ERR` sibling leaf lives ONLY on
# the structured node (\IONS::TOP.CER.<FLAVOR>.<VIEW>.CHANNELnn:<leaf>), not as a flat
# pointname, so fetch_cer_profile builds that path to read the real error bar. Maps: flat
# flavor prefix -> IONS analysis name, view letter -> system name, qbase -> error leaf.
_CER_FLAVOR_TREE = {"cerq": "CERQUICK", "cera": "CERAUTO", "cerf": "CERFIT", "cern": "CERNEUR"}
_CER_VIEW_NAME = {"t": "TANGENTIAL", "v": "VERTICAL"}
_CER_ERR_LEAF = {"ti": "TEMP_ERR", "rotc": "ROT_ERR", "rot": "ROT_ERR",
                 "amp": "AMP_ERR", "vb": "VB_ERR"}
# Zeeman/rotation-CORRECTED value variants (the DIII-D standard product, QUICKFIT's default):
# Ti uses TEMPC (raw TEMP is Zeeman-broadened, biased high by ~5-7%), rotation uses ROTC. The
# error stays the raw `_ERR` (there is no TEMPC_ERR). Fetched from the structured node when it
# carries data; else the raw TEMP/ROT flat pointname is used.
_CER_CORRECTED_LEAF = {"ti": "TEMPC", "rotc": "ROTC"}



# =============================================================================
# Connection (shot-agnostic) — ONE tunnel + ONE mdsplus connection, reused
# =============================================================================

class DIIIDConnection(MDSConnection):
    """Holds a single SSH tunnel + mdsplus connection to atlas, reusable across shots.

    Open it once and share it among many DIIIDFetcher(shot, connection=...)
    instances so a multi-shot job uses one tunnel, not one per shot.
    """

    MACHINE = "diiid"
    MDS_SERVER = "atlas.gat.com:8000"


# =============================================================================
# DIII-D MDSplus fetcher (per shot; shares a DIIIDConnection)
# =============================================================================

class DIIIDFetcher(MDSFetcher):
    """Per-shot DIII-D MDSplus fetcher (shares a DIIIDConnection across shots).

        # single shot (owns its connection)
        with DIIIDFetcher(207959) as f:
            sig = f.fetch_signal("ip")

        # many shots, ONE tunnel/connection (polite):
        with DIIIDConnection() as conn:
            for shot in shots:
                sigs = DIIIDFetcher(shot, connection=conn).fetch_signals(specs)
    """

    CONNECTION = DIIIDConnection
    DEFAULT_CACHE = __mitimroot__ / "tests" / "scratch" / "diiid_fetcher"

    # ---- signal resolution (findsig -> ptdata2 -> pseudo cascade) -----------
    def _assign(self, spec: str) -> str:
        """Resolve `spec` and assign it to server-side `_s`; return provenance.

        `PTDATA::<name>` reads a PTDATA pointname; other `<TREE>::...` specs are
        generic (see MDSFetcher._assign); bare names go through `_assign_bare`.
        """
        if "::" in spec:
            head, rest = spec.split("::", 1)
            if head.lstrip("\\").strip().upper() == "PTDATA":
                self.conn.get(f'_s = ptdata2("{rest.strip()}",{self.shot})')
                return f"PTDATA:{rest.strip()}"
        return super()._assign(spec)

    def _assign_bare(self, spec: str) -> str:
        """`findsig` is the DIII-D signal finder: it returns the proper node and
        sets `_fstree` to the tree (e.g. wmhd -> EFIT01:\\WMHD). It aborts for
        true PTDATA pointnames (ip, bt, ece...), which then go through ptdata2.
        """
        try:
            node = _b2s(self._value(f'findsig("{spec}",_fstree)'))
            tree = _b2s(self._value("_fstree"))
            if tree and node:
                self._open_tree(tree)
                self.conn.get(f"_s = {node}")
                if self._ssize() > 1:
                    return f"{tree}:{node}"
        except Exception:
            pass

        self.conn.get(f'_s = ptdata2("{spec}",{self.shot})')
        if self._ssize() > 1:
            return f"PTDATA:{spec}"

        self.conn.get(f'_s = pseudo("{spec}",{self.shot})')
        return f"PSEUDO:{spec}"

    # ---- EFIT equilibrium (2D, one time slice) ------------------------------
    def fetch_equilibrium(self, time: float, tree: str = "EFIT01") -> EquilibriumData:
        """EFIT flux-surface snapshot nearest `time` [ms] from `tree`.

        Returns normalized ψ on the R,Z grid plus the LCFS, magnetic axis,
        vessel and A-file X-points/strike points. Cached on disk.

        CRITICAL: mdsthin returns the time axis REVERSED relative to the server's
        storage order, so the python time index must be applied to FULL python
        arrays (`arr[it]`), NEVER to a server-side `[it]` subscript — the latter
        silently returns a DIFFERENT time slice. Self-consistent but wrong-time.
        """
        cached = self._eq_cache_load(tree, time)
        if cached is not None:
            self.n_from_cache += 1
            print(f"Using cached equilibrium for tree {tree} at time {time}")
            return cached

        self.n_from_server += 1
        self.conn.openTree(tree, self.shot)
        G = rf"\{tree}::TOP.RESULTS.GEQDSK"
        gtime = np.atleast_1d(self._value(f"{G}:GTIME")).astype(float)
        it = int(np.argmin(np.abs(gtime - time)))
        t_act = float(gtime[it])

        # full arrays, indexed in python; PSIRZ python shape is (ntime, nz, nr)
        psi = np.asarray(self._value(f"{G}:PSIRZ"), float)[it]
        d0 = np.atleast_1d(self._value(f"dim_of({G}:PSIRZ,0)")).astype(float)
        d1 = np.atleast_1d(self._value(f"dim_of({G}:PSIRZ,1)")).astype(float)

        simag = float(np.atleast_1d(self._value(f"{G}:SSIMAG"))[it])
        sibry = float(np.atleast_1d(self._value(f"{G}:SSIBRY"))[it])
        rax = float(np.atleast_1d(self._value(f"{G}:RMAXIS"))[it])
        zax = float(np.atleast_1d(self._value(f"{G}:ZMAXIS"))[it])

        # R is the all-positive grid; Z spans negative.
        rgrid, zgrid = (d0, d1) if d1.min() < d0.min() else (d1, d0)
        # orient the slice to [nz, nr] by matching the psi extremum to the axis
        PSI = orient_psi(psi, rgrid, zgrid, simag, sibry, rax, zax)
        psiN = (PSI - simag) / (sibry - simag)

        nb = int(np.atleast_1d(self._value(f"{G}:NBBBS"))[it])
        rb = np.asarray(self._value(f"{G}:RBBBS"), float)[it][:nb]
        zb = np.asarray(self._value(f"{G}:ZBBBS"), float)[it][:nb]

        qfull = np.asarray(self._value(f"{G}:QPSI"), float)  # q on uniform ψ grid (for ρ_tor)
        qpsi = (qfull[it] if qfull.ndim == 2 and qfull.shape[0] == gtime.size
                else qfull[:, it] if qfull.ndim == 2 else qfull)

        lim = np.asarray(self._value(f"{G}:LIM"), float)     # vessel/limiter (R,Z) points
        wr, wz = (lim[0], lim[1]) if lim.shape[0] == 2 else (lim[:, 0], lim[:, 1])

        # A-file X-points (RXPT1/2) and divertor strike points (RVS*/ZVS*) [m],
        # indexed with the A-file's own time base (python full array).
        A = rf"\{tree}::TOP.RESULTS.AEQDSK"
        try:
            atime = np.atleast_1d(self._value(f"{A}:ATIME")).astype(float)
            ita = int(np.argmin(np.abs(atime - time)))
        except Exception:
            ita = it

        def asc(node):
            try:
                return float(np.atleast_1d(self._value(f"{A}:{node}"))[ita])
            except Exception:
                return float("nan")

        ed = EquilibriumData(
            self.shot, tree, t_act, rgrid, zgrid, psiN, rb, zb, rax, zax, wr, wz,
            asc("RXPT1"), asc("ZXPT1"), asc("RXPT2"), asc("ZXPT2"),
            asc("RVSIN"), asc("ZVSIN"), asc("RVSOUT"), asc("ZVSOUT"), qpsi)
        self._eq_cache_save(tree, time, ed)
        return ed

    def fetch_geqdsk(self, time: float, tree: str = "EFIT01", path=None) -> Path:
        """Write a standard GEQDSK (g-file) for the EFIT slice nearest `time` [ms].

        Reads the full `\\<tree>::TOP.RESULTS.GEQDSK` node group (ψ(R,Z), the 1D
        fpol/pres/ffprim/pprime/q profiles, the scalars, boundary and limiter) and
        writes a self-contained g-file in SI units, readable by `gs_tools.GEQtools`
        / megpy / OMFIT. The time slice is taken with PYTHON full-array indexing
        (mdsthin reverses the time axis). Returns the output path.
        """
        self.conn.openTree(tree, self.shot)
        G = rf"\{tree}::TOP.RESULTS.GEQDSK"
        gtime = np.atleast_1d(self._value(f"{G}:GTIME")).astype(float)
        it = int(np.argmin(np.abs(gtime - time))); t_act = float(gtime[it])

        sc = lambda n: float(np.atleast_1d(self._value(f"{G}:{n}"))[it])    # per-time scalar
        def prof(n):                                   # per-time 1D profile -> (nw,)
            a = np.asarray(self._value(f"{G}:{n}"), float)
            return a[it] if (a.ndim == 2 and a.shape[0] == gtime.size) else (a[:, it] if a.ndim == 2 else a)

        psi = np.asarray(self._value(f"{G}:PSIRZ"), float)[it]
        d0 = np.atleast_1d(self._value(f"dim_of({G}:PSIRZ,0)")).astype(float)
        d1 = np.atleast_1d(self._value(f"dim_of({G}:PSIRZ,1)")).astype(float)
        rgrid, zgrid = (d0, d1) if d1.min() < d0.min() else (d1, d0)
        simag, sibry, rax, zax = sc("SSIMAG"), sc("SSIBRY"), sc("RMAXIS"), sc("ZMAXIS")
        PSI = orient_psi(psi, rgrid, zgrid, simag, sibry, rax, zax)   # -> [nz, nr]

        nb = int(np.atleast_1d(self._value(f"{G}:NBBBS"))[it])
        rb = np.asarray(self._value(f"{G}:RBBBS"), float)[it][:nb]
        zb = np.asarray(self._value(f"{G}:ZBBBS"), float)[it][:nb]
        lim = np.asarray(self._value(f"{G}:LIM"), float)
        rl, zl = (lim[0], lim[1]) if lim.shape[0] == 2 else (lim[:, 0], lim[:, 1])

        data = dict(case=f"EFIT {tree} #{self.shot} {t_act:.0f}ms", nw=rgrid.size, nh=zgrid.size,
                    rdim=sc("XDIM"), zdim=sc("ZDIM"), rcentr=sc("RZERO"), rleft=float(rgrid.min()),
                    zmid=sc("ZMID"), rmaxis=rax, zmaxis=zax, simag=simag, sibry=sibry,
                    bcentr=sc("BCENTR"), current=sc("CPASMA"), psirz=PSI,
                    fpol=prof("FPOL"), pres=prof("PRES"), ffprime=prof("FFPRIM"),
                    pprime=prof("PPRIME"), qpsi=prof("QPSI"),
                    rbbbs=rb, zbbbs=zb, rlim=np.asarray(rl, float), zlim=np.asarray(zl, float))
        path = Path(path) if path is not None else (self.cache_dir / f"g{self.shot}.{int(round(t_act)):05d}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write_geqdsk(path, data)

    # ---- CER channel profile (value vs R,Z at one time) ---------------------
    def fetch_cer_profile(self, time: float, quantity: str = "tit",
                          channels=range(1, 49), window: float = 100.0,
                          t_window=None, system: str = "cerq",
                          views=("t", "v"), zeeman_corrected: bool = True,
                          average: bool = True) -> ChannelProfile:
        """CER profile: each channel's `quantity` plus its (R, Z), averaged in time,
        across the requested CER VIEWING SYSTEMS.

        DIII-D CER has two views: TANGENTIAL ('t', ~midplane chords) and VERTICAL
        ('v', looking down, reaching the core). `views` selects which to include
        (default BOTH — they are physically distinct chords, not duplicates); each
        channel is tagged 'T<n>'/'V<n>'. The flat pointname is
        <system><qbase><view><n>, geometry <system>r<view><n> / <system>z<view><n>
        — e.g. cerqtit3/cerqrt3 (tangential) and cerqtiv3/cerqrv3 (vertical).
        `quantity` is the suffix INCLUDING the trailing view letter ('tit'=Ti [eV],
        'rotct'=rotation); its base ('ti') is reused for every view. Averaged over
        [time-window, time+window], or the explicit `t_window=(t0,t1)`; the error bar
        is the STORED per-chord measurement error (the sibling `<quantity>_ERR` node —
        TEMP_ERR / ROT_ERR / AMP_ERR) averaged over the window, falling back to the
        temporal std of the samples only when no `*_ERR` node exists (e.g. the derived
        n_Z / n_Z-n_e). The stored error is essential for slow chords (a tangential Ti
        chord often has ONE sample per window -> a temporal std of 0 -> no error bar).

        `zeeman_corrected` (default True, matching QUICKFIT and the DIII-D standard):
        for Ti/rotation, use the CORRECTED node — Ti's TEMPC (raw TEMP is Zeeman-
        broadened, biased high by ~5-7%) and rotation's ROTC — whenever it carries data,
        else fall back to the raw TEMP/ROT flat pointname. Read from the structured node,
        so the corrected value caches under a DISTINCT key (…:TEMPC) from the raw one
        (the flat pointname) — the toggle can never return the wrong cached variant.
        Other quantities are unaffected. Set False for the raw (uncorrected) value.

        With `average=False` every time sample in the window is kept (a scatter cloud at each
        channel's R), each carrying its per-sample stored error (`<quantity>_ERR` at that time,
        NaN where none is stored). Missing channels are skipped (cached as misses). Sorted by R.
        """
        t0, t1 = t_window if t_window is not None else (time - window, time + window)
        qbase = quantity[:-1] if quantity and quantity[-1] in "tv" else quantity

        chs, tags, rs, zs, vals, errs, units = [], [], [], [], [], [], ""
        for view in views:                        # 't' tangential, 'v' vertical
            for n in channels:
                try:
                    v = self._cer_corrected_value(system, view, n, qbase) if zeeman_corrected else None
                    if v is None:                 # no corrected variant/data -> raw TEMP/ROT flat pointname
                        v = self.fetch_signal(f"{system}{qbase}{view}{n}")  # name=spec => cache shared w/ overview
                    r = self.fetch_signal(f"{system}r{view}{n}")
                    z = self.fetch_signal(f"{system}z{view}{n}")
                except Exception:
                    continue
                units = v.units
                R = float(np.nanmedian(r.data)); Z = float(np.nanmedian(z.data))   # geometry ~ steady
                if average:
                    vm, vs, _ = time_average(v.time, v.data, t0, t1)
                    if not np.isfinite(vm):       # no sample inside the window -> drop (no fallback)
                        continue
                    em = self._cer_stored_error(system, view, n, qbase, t0, t1)
                    chs.append(n); tags.append(f"{view.upper()}{n}")
                    vals.append(float(vm))
                    errs.append(em if em is not None else float(vs))   # stored err, else temporal std
                    rs.append(R); zs.append(Z)
                else:                             # keep every time sample in the window
                    tv, yv = np.asarray(v.time, float), np.asarray(v.data, float)
                    esig = self._cer_error_signal(system, view, n, qbase)   # per-sample error trace
                    ev = np.asarray(esig.data, float) if esig is not None else None
                    aligned = ev is not None and ev.shape == yv.shape       # shares the value time base
                    msk = (tv >= t0) & (tv <= t1) & np.isfinite(yv)
                    for k in np.flatnonzero(msk):
                        chs.append(n); tags.append(f"{view.upper()}{n}")
                        vals.append(float(yv[k])); rs.append(R); zs.append(Z)
                        errs.append(float(ev[k]) if aligned else np.nan)    # per-sample stored error
        order = np.argsort(rs) if rs else np.array([], int)
        arr = lambda a: np.asarray(a, float)[order]
        return ChannelProfile(self.shot, 0.5 * (t0 + t1), quantity,
                              np.asarray(chs, int)[order], arr(rs), arr(zs), arr(vals),
                              units, label=f"CER {quantity}",
                              tag=np.asarray(tags)[order],
                              error=arr(errs))     # per-point (average=False) or per-channel (average=True)

    def _cer_corrected_value(self, system, view, n, qbase):
        """Zeeman/rotation-CORRECTED CER value (TEMPC / ROTC) from the structured node, or
        None when this quantity has no corrected variant or the node carries no data (caller
        then uses the raw TEMP/ROT flat pointname). This is QUICKFIT's default product."""
        leaf = _CER_CORRECTED_LEAF.get(qbase)
        tree = _CER_FLAVOR_TREE.get(system)
        vname = _CER_VIEW_NAME.get(view)
        if leaf is None or tree is None or vname is None:
            return None
        try:
            return self.fetch_signal(rf"\IONS::TOP.CER.{tree}.{vname}.CHANNEL{n:02d}:{leaf}")
        except Exception:
            return None

    def _cer_error_signal(self, system, view, n, qbase):
        """The stored per-chord error SIGNAL (the `<qbase>_ERR` time trace), or None when this
        quantity/flavor has no such node. The error lives only on the structured node."""
        leaf = _CER_ERR_LEAF.get(qbase)
        tree = _CER_FLAVOR_TREE.get(system)
        vname = _CER_VIEW_NAME.get(view)
        if leaf is None or tree is None or vname is None:
            return None
        try:
            return self.fetch_signal(rf"\IONS::TOP.CER.{tree}.{vname}.CHANNEL{n:02d}:{leaf}")
        except Exception:
            return None

    def _cer_stored_error(self, system, view, n, qbase, t0, t1):
        """Window-AVERAGED stored per-chord measurement error (for average=True), or None
        when this quantity/flavor has no `<qbase>_ERR` node (caller uses the temporal std)."""
        e = self._cer_error_signal(system, view, n, qbase)
        if e is None:
            return None
        ea, _, _ = time_average(e.time, e.data, t0, t1)
        return float(ea) if np.isfinite(ea) else None

    def fetch_cer_coverage(self, quantity: str = "tit", channels=range(1, 49),
                           system: str = "cerq", views=("t", "v"), t_window=None,
                           zeeman_corrected: bool = True) -> list:
        """Per-channel valid-sample TIMES and chord (R,Z) for a CER `quantity` — the raw temporal
        coverage (no averaging), for diagnosing WHEN each chord actually measures (CER reports in
        bursts, not continuously). For each (view, channel) that has data it returns a dict
        {'view','channel','t' [ms, valid samples], 'R','Z'}; a sample is kept where the value is
        finite and non-zero and within `t_window`=(t0,t1). Value node is TEMPC when
        `zeeman_corrected` else the raw flat pointname; geometry from <system>r/z<view><n>."""
        qbase = quantity[:-1] if quantity and quantity[-1] in "tv" else quantity
        t0, t1 = t_window if t_window is not None else (-np.inf, np.inf)
        out = []
        for view in views:
            for n in channels:
                try:
                    v = self._cer_corrected_value(system, view, n, qbase) if zeeman_corrected else None
                    if v is None:
                        v = self.fetch_signal(f"{system}{qbase}{view}{n}")
                    r = self.fetch_signal(f"{system}r{view}{n}")
                    z = self.fetch_signal(f"{system}z{view}{n}")
                except Exception:
                    continue
                t = np.asarray(v.time, float); y = np.asarray(v.data, float)
                m = np.isfinite(y) & (y != 0) & (t >= t0) & (t <= t1)
                if not m.any():
                    continue
                out.append({"view": view, "channel": n, "t": t[m],
                            "R": float(np.nanmedian(r.data)), "Z": float(np.nanmedian(z.data))})
        return out

    # ---- Thomson-scattering profile (Te / ne vs R,Z at one time) ------------
    def fetch_thomson_profile(self, time: float, quantity: str = "te", system="core",
                              window: float = 100.0, t_window=None,
                              average: bool = True) -> ChannelProfile:
        """Thomson-scattering profile: Te or ne vs (R, Z) per channel, time-averaged.

        Reads the 2D BLESSED arrays `\\ELECTRONS::TOP.TS.BLESSED.<SYSTEM>:{TEMP|DENSITY}`
        plus its stored error `:{TEMP|DENSITY}_E` and `:R`/`:Z`/`:TIME` (all cached via
        `_value_cached`). Averages over [time-window, time+window] — or the explicit
        `t_window=(t0,t1)` if given — and drops channels with no valid (>0) sample.
        The per-channel error bar is the stored measurement error (`_E`) averaged
        over the window (the typical per-measurement uncertainty).

        `system` may be one of 'core'|'tangential'|'divertor', a LIST of them, or
        'all' (= core+tangential). Each channel is tagged by view ('C#','T#','D#').
        `quantity` 'te'->TEMP [eV], 'ne'->DENSITY [m^-3]. Points sorted by R. With
        `average=False` every time sample in the window is kept (scatter, error=None).
        """
        systems = (["core", "tangential"] if system == "all"
                   else [system] if isinstance(system, str) else list(system))
        node = "TEMP" if quantity.lower() in ("te", "temp") else "DENSITY"
        t0, t1 = t_window if t_window is not None else (time - window, time + window)
        Rs, Zs, Vs, Es, Tg, units = [], [], [], [], [], ""
        for sysname in systems:
            base = rf"\ELECTRONS::TOP.TS.BLESSED.{sysname.upper()}"
            try:                                  # a TS view can be absent on a given shot
                val2d, units = self._value_cached(f"{base}:{node}", tree="ELECTRONS")
                err2d, _ = self._value_cached(f"{base}:{node}_E", tree="ELECTRONS")
                R, _ = self._value_cached(f"{base}:R", tree="ELECTRONS")
                Z, _ = self._value_cached(f"{base}:Z", tree="ELECTRONS")
                tarr, _ = self._value_cached(f"{base}:TIME", tree="ELECTRONS")
            except Exception as e:
                print(f"  ! TS {sysname} unavailable for #{self.shot}: {str(e)[:45]}")
                continue
            pts = reduce_channels(val2d, err2d, R, Z, tarr, t0, t1, sysname[0].upper(), average)
            if average or pts[0].size:
                for acc, a in zip((Rs, Zs, Vs, Es, Tg), pts):
                    acc.append(a)
        empty = np.array([])
        R, Z, V, E, Tg = (np.concatenate(a) if a else empty for a in (Rs, Zs, Vs, Es, Tg))
        order = np.argsort(R)
        return ChannelProfile(self.shot, 0.5 * (t0 + t1), f"{'+'.join(systems)}.{node.lower()}",
                              np.arange(R.size)[order], R[order], Z[order], V[order], units,
                              label=f"TS {'+'.join(systems)} {quantity}", tag=Tg[order],
                              error=(E[order] if average else None))


# generic names by which the machine-agnostic plotting resolves this backend
Connection, Fetcher = DIIIDConnection, DIIIDFetcher
