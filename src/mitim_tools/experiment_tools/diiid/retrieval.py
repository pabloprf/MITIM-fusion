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
    GA. Off-site, pass `tunnel_host=<your jump host>` (a passwordless
    `~/.ssh/config` entry that can reach atlas:8000) and an SSH tunnel is opened
    for you (reused across shots):

        ssh -N -L <localport>:atlas.gat.com:8000 <tunnel_host>

    so nothing has to be installed or running on the GA side. (The MFE-IM group
    typically uses a GA gateway such as `cybele`.) On a GA host, instead pass
    `server="host:port"` (or `tunnel_host=None`) to connect directly.

`mdsthin` is an optional dependency: ``pip install mitim-fusion[mds]``.
"""

from __future__ import annotations

import atexit
import hashlib
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mitim_tools import __mitimroot__

# Pure-python MDSplus thin client (optional dependency; see module docstring).
# Note: the mdsthin backend returns multi-time arrays with the time axis reversed
# vs the server's storage order — see fetch_equilibrium() for the (mandatory)
# python-side indexing this requires. Do NOT swap in a different MDS backend
# without revisiting that.
try:
    import mdsthin as _mds
except Exception as _excp:
    _mds = None
    _MDS_IMPORT_ERROR = _excp


def _b2s(x) -> str:
    """Bytes / numpy scalar -> plain stripped str."""
    x = np.atleast_1d(np.asarray(x)).ravel()
    if x.size == 0:
        return ""
    v = x[0]
    return (v.decode(errors="replace") if isinstance(v, bytes) else str(v)).strip()


# =============================================================================
# Lightweight container for a single time trace
# =============================================================================

@dataclass
class Signal:
    """A 1D experimental trace in time."""
    name:  str
    time:  np.ndarray          # [ms]
    data:  np.ndarray
    units: str = ""
    label: str = ""
    source: str = ""           # provenance (tree:node / PTDATA:name / cache)

    def __repr__(self):
        n = 0 if self.time is None else len(self.time)
        return (f"Signal({self.name!r}, n={n}, units={self.units!r}, "
                f"source={self.source!r})")


@dataclass
class EquilibriumData:
    """An EFIT flux-surface snapshot at one time (everything in m / normalized ψ)."""
    shot:   int
    tree:   str
    time:   float            # actual EFIT time [ms]
    rgrid:  np.ndarray       # R axis [m]
    zgrid:  np.ndarray       # Z axis [m]
    psiN:   np.ndarray       # normalized poloidal flux, shape [nz, nr]
    rbbbs:  np.ndarray       # LCFS R [m]
    zbbbs:  np.ndarray       # LCFS Z [m]
    raxis:  float            # magnetic axis R [m]
    zaxis:  float            # magnetic axis Z [m]
    wall_r: np.ndarray       # limiter/vessel R [m]
    wall_z: np.ndarray       # limiter/vessel Z [m]
    # A-file X-points and divertor strike points [m], for the separatrix legs
    rxpt1:  float
    zxpt1:  float
    rxpt2:  float
    zxpt2:  float
    rvsin:  float
    zvsin:  float
    rvsout: float
    zvsout: float
    qpsi:   np.ndarray       # safety factor on the uniform ψ_N grid (axis->boundary)

    def rho_of(self, R, Z, kind: str = "tor"):
        """Map (R, Z) [m] to a normalized flux radius using this equilibrium.

        kind='tor': ρ_tor = sqrt(normalized toroidal flux) — the transport ρ,
                    Φ(ψ) = ∫ q dψ from the q-profile (QPSI).
        kind='pol': ρ_pol = sqrt(ψ_N).
        Off-grid -> NaN; in the SOL (ψ_N>1) both continue as sqrt(ψ_N) (ρ_tor is
        only defined inside, so it is matched to ρ_pol past the separatrix).
        """
        from scipy.interpolate import RegularGridInterpolator
        ip = RegularGridInterpolator((self.zgrid, self.rgrid), self.psiN,
                                     bounds_error=False, fill_value=np.nan)
        R = np.atleast_1d(np.asarray(R, float)); Z = np.atleast_1d(np.asarray(Z, float))
        psiN = np.clip(ip(np.column_stack([Z, R])), 0.0, None)
        if kind.startswith("pol"):
            return np.sqrt(psiN)
        q = np.abs(np.asarray(self.qpsi, float))
        pn = np.linspace(0.0, 1.0, q.size)              # uniform ψ_N grid of QPSI
        phi = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(pn))])
        rho = np.interp(psiN, pn, np.sqrt(phi / phi[-1]))   # ρ_tor for ψ_N<=1
        rho[psiN > 1.0] = np.sqrt(psiN[psiN > 1.0])         # continue into SOL
        return rho


@dataclass
class ChannelProfile:
    """A multi-channel diagnostic profile at one time: <quantity> vs (R, Z), one
    point per channel (CER chords, Thomson channels, ...). Sorted by R."""
    shot:     int
    time:     float          # actual window-center time [ms]
    quantity: str            # e.g. 'tit' (CER Ti), 'core.temp' (TS core Te)
    channel:  np.ndarray     # channel numbers/indices
    r:        np.ndarray     # major radius of each channel [m]
    z:        np.ndarray     # height of each channel [m]
    value:    np.ndarray     # quantity value near `time` (windowed mean)
    units:    str = ""
    label:    str = ""       # display label, e.g. "CER tit" / "TS core te"
    tag:      np.ndarray = None  # per-channel labels (e.g. 'C5'/'T3' for TS views); None -> use channel
    error:    np.ndarray = None  # per-channel 1σ error bar (stored meas. error or temporal std)


# =============================================================================
# General utilities
# =============================================================================

def time_average(t, y, t0, t1, axis=-1):
    """Average `y` over the time window [t0, t1] (ms) along `axis`.

    Returns (mean, std, n): the NaN-ignoring mean, standard deviation, and count
    of finite samples STRICTLY inside the window. If NO sample falls inside, the
    mean/std are NaN and n is 0 -- there is deliberately NO out-of-window fallback,
    so a too-narrow window honestly yields no data rather than a nearby slice.
    General-purpose (profiles, scalars, any windowed mean).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    idx = np.where((t >= t0) & (t <= t1))[0]
    if idx.size == 0:                              # nothing in the window -> NaN, no fallback
        out = y.shape[:axis % y.ndim] + y.shape[axis % y.ndim + 1:]
        nan = np.full(out, np.nan) if out else np.float64("nan")
        return nan, nan, (np.zeros(out, int) if out else 0)
    sl = np.take(y, idx, axis=axis)
    with np.errstate(invalid="ignore"):
        return (np.nanmean(sl, axis=axis), np.nanstd(sl, axis=axis),
                np.sum(np.isfinite(sl), axis=axis))


def _write_geqdsk(path, d):
    """Write a standard EFIT GEQDSK (g-file) from a dict of SI-unit fields.

    All quantities are SI as stored by EFIT: psi [Wb/rad], R/Z [m], fpol=R*Bt
    [m*T], pres [Pa], current [A], bcentr [T]. `d['psirz']` is the 2D slice
    oriented [nz, nr]; it is written row-major, i.e. ((psi(i=R, j=Z), i=1,nw), j=1,nh),
    the GEQDSK convention. Boundary/limiter are written as interleaved (R, Z)."""
    nw, nh = d["nw"], d["nh"]

    def block(arr):                                # 5 values per line, Fortran e16.9
        a = np.asarray(arr, float).ravel()
        return "\n".join("".join(f"{v: .9E}" for v in a[i:i + 5]) for i in range(0, a.size, 5))

    def row(*v):
        return "".join(f"{x: .9E}" for x in v)

    lines = [f"{d['case'][:48]:<48s}{3:4d}{nw:4d}{nh:4d}",
             row(d["rdim"], d["zdim"], d["rcentr"], d["rleft"], d["zmid"]),
             row(d["rmaxis"], d["zmaxis"], d["simag"], d["sibry"], d["bcentr"]),
             row(d["current"], d["simag"], 0.0, d["rmaxis"], 0.0),
             row(d["zmaxis"], 0.0, d["sibry"], 0.0, 0.0),
             block(d["fpol"]), block(d["pres"]), block(d["ffprime"]), block(d["pprime"]),
             block(d["psirz"]), block(d["qpsi"]),
             f"{len(d['rbbbs']):5d}{len(d['rlim']):5d}"]
    bdry = np.empty(2 * len(d["rbbbs"])); bdry[0::2] = d["rbbbs"]; bdry[1::2] = d["zbbbs"]
    lim = np.empty(2 * len(d["rlim"])); lim[0::2] = d["rlim"]; lim[1::2] = d["zlim"]
    lines += [block(bdry), block(lim)]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


# =============================================================================
# SSH -L tunnel to the DIII-D MDSplus server
# =============================================================================

def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SSHTunnel:
    """Background `ssh -N -L localport:mds_host:mds_port jump_host`.

    jump_host is a ~/.ssh/config alias, so all the ProxyJump / key / user
    details come from your config (no credentials handled here).
    """

    def __init__(self, jump_host: str, mds_host: str, mds_port: int,
                 local_port: int | None = None, timeout: float = 25.0):
        self.jump_host = jump_host
        self.mds_host = mds_host
        self.mds_port = int(mds_port)
        self.local_port = local_port or _pick_free_port()
        self.timeout = timeout
        self.proc = None

    def open(self) -> "SSHTunnel":
        cmd = ["ssh", "-N",
               "-o", "ExitOnForwardFailure=yes",
               "-o", "ServerAliveInterval=30",
               "-o", "BatchMode=yes",          # key/agent auth only; never hang on a prompt
               "-o", "ConnectTimeout=15",
               "-L", f"{self.local_port}:{self.mds_host}:{self.mds_port}",
               self.jump_host]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        atexit.register(self.close)

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode(errors="replace").strip()
                raise ConnectionError(
                    f"SSH tunnel via '{self.jump_host}' exited early: {err}")
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return self
            time.sleep(0.2)

        self.close()
        raise TimeoutError(
            f"SSH tunnel to {self.mds_host}:{self.mds_port} via "
            f"'{self.jump_host}' was not ready within {self.timeout}s")

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None

    @property
    def server(self) -> str:
        return f"127.0.0.1:{self.local_port}"


# =============================================================================
# Connection (shot-agnostic) — ONE tunnel + ONE mdsplus connection, reused
# =============================================================================

class DIIIDConnection:
    """Holds a single SSH tunnel + mdsplus connection, reusable across shots.

    Open it once and share it among many DIIIDFetcher(shot, connection=...)
    instances so a multi-shot job uses one tunnel, not one per shot.
    """

    def __init__(self, server: str | None = None, tunnel_host: str = "cybele",
                 mds_server: str = "atlas.gat.com:8000", tunnel_timeout: float = 25.0):
        self.server = server               # if set, connect directly (no tunnel)
        self.tunnel_host = tunnel_host
        self.mds_host, self.mds_port = mds_server.split(":")
        self.tunnel_timeout = tunnel_timeout
        self._tunnel = None
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def conn(self):
        """The thin-client connection, established (with tunnel) on first use."""
        if self._conn is None:
            if _mds is None:
                raise ImportError(
                    f"`mdsthin` is not installed ({_MDS_IMPORT_ERROR!r}). "
                    "Install the MDS extra: `pip install mitim-fusion[mds]` "
                    "(or `pip install mdsthin`).")
            if self.server is None:
                self._tunnel = SSHTunnel(self.tunnel_host, self.mds_host,
                                         self.mds_port,
                                         timeout=self.tunnel_timeout).open()
                self.server = self._tunnel.server
            self._conn = _mds.Connection(self.server)
        return self._conn

    def close(self):
        self._conn = None
        if self._tunnel is not None:
            self._tunnel.close()
            self._tunnel = None


# =============================================================================
# DIII-D MDSplus fetcher (per shot; shares a DIIIDConnection)
# =============================================================================

class DIIIDFetcher:
    """Per-shot DIII-D MDSplus fetcher (shares a DIIIDConnection across shots).

        # single shot (owns its connection)
        with DIIIDFetcher(207959) as f:
            sig = f.fetch_signal("ip")

        # many shots, ONE tunnel/connection (polite):
        with DIIIDConnection() as conn:
            for shot in shots:
                sigs = DIIIDFetcher(shot, connection=conn).fetch_signals(specs)
    """

    DEFAULT_CACHE = __mitimroot__ / "tests" / "scratch" / "diiid_fetcher"

    def __init__(self, shot: int, connection: DIIIDConnection | None = None,
                 max_points: int = 4000, use_cache: bool = True,
                 cache_dir: str | Path | None = None,
                 # forwarded to DIIIDConnection when we create our own:
                 server: str | None = None, tunnel_host: str = "cybele",
                 mds_server: str = "atlas.gat.com:8000", tunnel_timeout: float = 25.0):
        self.shot = int(shot)
        self.max_points = max_points
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.DEFAULT_CACHE

        self._own_conn = connection is None
        self.connection = connection or DIIIDConnection(
            server=server, tunnel_host=tunnel_host,
            mds_server=mds_server, tunnel_timeout=tunnel_timeout)

    # ---- lifecycle ----------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._own_conn:                 # never close a shared connection
            self.connection.close()

    @property
    def conn(self):
        return self.connection.conn

    def _value(self, expr: str):
        """Evaluate a TDI expression on the server, returned as numpy."""
        r = self.conn.get(expr)
        r = r.data() if hasattr(r, "data") else r   # mdsthin wraps results in a Data object
        return np.asarray(r)

    # ---- public fetch -------------------------------------------------------
    def fetch_signal(self, spec: str, label: str = "", name: str = "",
                     max_points: int | None = None) -> Signal:
        """Fetch a signal spec as a Signal (value + time + units).

        Large traces are resampled on the server to <= max_points before
        transfer. Results are cached to disk unless use_cache is False.
        """
        mp = self.max_points if max_points is None else max_points
        name = name or spec

        cached = self._cache_load(spec, mp, name)   # may raise (a cached miss)
        if cached is not None:
            return cached

        source = self._assign(spec.strip())
        self._server_reduce(mp)
        data = np.atleast_1d(self._value("_s"))
        if data.size <= 1:                     # scalar/empty => no usable trace
            self._cache_save_miss(spec, mp, name, source)   # remember the miss
            raise ValueError(f"no data ({source})")
        time_ = np.atleast_1d(self._value("dim_of(_s,0)"))
        sig = Signal(name, time_, data, units=self._safe_units(),
                     label=label or spec, source=source)
        self._cache_save(spec, mp, sig)
        return sig

    def fetch_signals(self, specs, max_points: int | None = None) -> dict:
        """Fetch many signals. `specs` is an iterable of (key, spec, label).

        Returns {key: Signal}; any signal that errors or is absent maps to None
        (real shots are routinely missing ECH/CER/etc., so we don't abort).
        """
        out = {}
        for key, spec, label in specs:
            try:
                out[key] = self.fetch_signal(spec, label=label, name=key,
                                             max_points=max_points)
            except Exception as excp:
                print(f"! {key} ({spec}) unavailable for #{self.shot}: {excp}")
                out[key] = None
        return out

    # ---- signal resolution (findsig -> ptdata2 -> pseudo cascade) -----------
    def _assign(self, spec: str) -> str:
        """Resolve `spec` and assign it to server-side `_s`; return provenance.

        `findsig` is the DIII-D signal finder: it returns the proper node and
        sets `_fstree` to the tree (e.g. wmhd -> EFIT01:\\WMHD). It aborts for
        true PTDATA pointnames (ip, bt, ece...), which then go through ptdata2.
        """
        if "::" in spec:
            head, rest = spec.split("::", 1)
            tree = head.lstrip("\\").strip()
            if tree.upper() == "PTDATA":
                self.conn.get(f'_s = ptdata2("{rest.strip()}",{self.shot})')
                return f"PTDATA:{rest.strip()}"
            tdi = spec if spec.startswith("\\") else rest.strip()
            self._open_tree(tree)
            self.conn.get(f"_s = {tdi}")
            return spec

        # bare name: findsig -> ptdata2 -> pseudo
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

    def _server_reduce(self, max_points: int):
        """Resample `_s` on the server to <= max_points (only if it is larger).

        Keeps the full time extent; just coarsens it so we transfer ~max_points
        instead of the full digitizer record. minval/maxval are scalars computed
        server-side, so nothing big crosses the wire before the resample.
        """
        if not max_points:
            return
        n = self._ssize()
        if n <= max_points:
            return
        tmin = float(np.atleast_1d(self._value("minval(dim_of(_s))")).ravel()[0])
        tmax = float(np.atleast_1d(self._value("maxval(dim_of(_s))")).ravel()[0])
        if tmax > tmin:
            dt = (tmax - tmin) / max_points
            self.conn.get(f"_s = resample(_s,{tmin},{tmax},{dt})")

    # ---- disk cache ---------------------------------------------------------
    def _cache_path(self, spec: str, max_points: int, name: str) -> Path:
        key = hashlib.md5(f"{self.shot}|{spec}|{max_points}".encode()).hexdigest()[:12]
        safe = "".join(c if c.isalnum() else "_" for c in name)[:24]
        return self.cache_dir / f"{self.shot}_{safe}_{key}.npz"

    def _cache_load(self, spec: str, max_points: int, name: str):
        if not self.use_cache:
            return None
        path = self._cache_path(spec, max_points, name)
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=False)
        if "nodata" in z.files:                # cached miss: re-raise without a server probe
            raise ValueError(f"no data ({str(z['source'])}) [cached]")
        return Signal(str(z["name"]), z["time"], z["data"], units=str(z["units"]),
                      label=str(z["label"]), source=f"{z['source']} (cache)")

    def _cache_save(self, spec: str, max_points: int, sig: Signal):
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(spec, max_points, sig.name)
        np.savez(path, time=sig.time, data=sig.data,
                 units=np.array(sig.units), label=np.array(sig.label),
                 source=np.array(sig.source), name=np.array(sig.name))

    def _cache_save_miss(self, spec: str, max_points: int, name: str, source: str):
        """Cache a 'no data' result so an absent signal isn't re-probed each run."""
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._cache_path(spec, max_points, name),
                 nodata=np.array(True), source=np.array(source))

    # ---- internals ----------------------------------------------------------
    def _open_tree(self, treename: str):
        # Always (re)open: openTree sets the *current* tree context, and node
        # lookups for `\TREE::NODE` resolve against it. Skipping a re-open when
        # another tree was opened in between makes the fetch silently return 0.
        self.conn.openTree(treename, self.shot)

    def _ssize(self) -> int:
        """Element count of server-side `_s` (cheap; no array transfer)."""
        try:
            return int(np.atleast_1d(self._value("size(_s)")).ravel()[0])
        except Exception:
            return 0

    def _safe_units(self) -> str:
        try:
            u = self._value("units(_s)")
            return str(np.atleast_1d(u)[0]).strip()
        except Exception:
            return ""

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
            return cached

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
        ii, jj = np.unravel_index(int(np.argmin(psi) if simag < sibry else np.argmax(psi)),
                                  psi.shape)
        err_zr = abs(rgrid[jj] - rax) + abs(zgrid[ii] - zax)   # psi is [Z, R]
        err_rz = abs(rgrid[ii] - rax) + abs(zgrid[jj] - zax)   # psi is [R, Z]
        PSI = psi if err_zr <= err_rz else psi.T               # -> [nz, nr]
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
        ii, jj = np.unravel_index(int(np.argmin(psi) if simag < sibry else np.argmax(psi)), psi.shape)
        err_zr = abs(rgrid[jj] - rax) + abs(zgrid[ii] - zax)
        err_rz = abs(rgrid[ii] - rax) + abs(zgrid[jj] - zax)
        PSI = psi if err_zr <= err_rz else psi.T       # -> [nz, nr]

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
                          views=("t", "v"), average: bool = True) -> ChannelProfile:
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
        [time-window, time+window], or the explicit `t_window=(t0,t1)`; the error
        bar is the temporal std over the window (CER has no readily-keyed stored
        error). With `average=False` every time sample in the window is kept (a
        scatter cloud at each channel's R, error=None). Missing channels are skipped
        (cached as misses). Sorted by R.
        """
        t0, t1 = t_window if t_window is not None else (time - window, time + window)
        qbase = quantity[:-1] if quantity and quantity[-1] in "tv" else quantity

        chs, tags, rs, zs, vals, errs, units = [], [], [], [], [], [], ""
        for view in views:                        # 't' tangential, 'v' vertical
            for n in channels:
                try:                              # name defaults to spec => cache shared with overview
                    v = self.fetch_signal(f"{system}{qbase}{view}{n}")
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
                    chs.append(n); tags.append(f"{view.upper()}{n}")
                    vals.append(float(vm)); errs.append(float(vs)); rs.append(R); zs.append(Z)
                else:                             # keep every time sample in the window
                    tv, yv = np.asarray(v.time, float), np.asarray(v.data, float)
                    msk = (tv >= t0) & (tv <= t1) & np.isfinite(yv)
                    for yi in yv[msk]:
                        chs.append(n); tags.append(f"{view.upper()}{n}")
                        vals.append(float(yi)); errs.append(np.nan); rs.append(R); zs.append(Z)
        order = np.argsort(rs) if rs else np.array([], int)
        arr = lambda a: np.asarray(a, float)[order]
        return ChannelProfile(self.shot, 0.5 * (t0 + t1), quantity,
                              np.asarray(chs, int)[order], arr(rs), arr(zs), arr(vals),
                              units, label=f"CER {quantity}",
                              tag=np.asarray(tags)[order],
                              error=(arr(errs) if average else None))

    # ---- Thomson-scattering profile (Te / ne vs R,Z at one time) ------------
    def _value_cached(self, node: str, tree: str | None = None):
        """Full-array fetch of `node` (any shape) with a disk cache; returns
        (data, units). On a cache hit nothing touches the connection (no tunnel)."""
        key = hashlib.md5(f"{self.shot}|{node}".encode()).hexdigest()[:12]
        path = self.cache_dir / f"{self.shot}_arr_{key}.npz"
        if self.use_cache and path.exists():
            z = np.load(path, allow_pickle=False)
            if "nodata" in z.files:                # cached miss -> re-raise, no server probe
                raise ValueError(f"no data ({node}) [cached]")
            return np.asarray(z["data"], float), str(z["units"])
        if tree:
            self.conn.openTree(tree, self.shot)
        try:
            data = np.asarray(self._value(node), float)
        except Exception as e:                     # absent / NODATA node -> cache the miss
            if self.use_cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(path, nodata=np.array(True))
            raise ValueError(f"no data ({node}): {str(e)[:40]}")
        try:
            units = str(self._value(f"units_of({node})"))
        except Exception:
            units = ""
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(path, data=data, units=np.array(units))
        return data, units

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
            R, Z, tarr = np.atleast_1d(R), np.atleast_1d(Z), np.atleast_1d(tarr)
            if val2d.shape[0] == tarr.size and val2d.shape[1] != tarr.size:
                val2d, err2d = val2d.T, err2d.T    # orient to (nchan, ntime)
            tag_of = lambda i: f"{sysname[0].upper()}{i}"
            if average:
                valid = np.where(val2d > 0, val2d, np.nan)            # 0 = no measurement
                v, _, _ = time_average(tarr, valid, t0, t1, axis=1)
                e, _, _ = time_average(tarr, np.where(val2d > 0, err2d, np.nan), t0, t1, axis=1)
                gd = np.isfinite(v) & (v > 0)
                Rs.append(R[gd]); Zs.append(Z[gd]); Vs.append(v[gd]); Es.append(e[gd])
                Tg.append(np.array([tag_of(i) for i in np.arange(R.size)[gd]]))
            else:                                 # every valid time sample in the window
                tm = (tarr >= t0) & (tarr <= t1)
                sub = val2d[:, tm]
                for i in range(R.size):
                    yi = sub[i]; good = yi > 0
                    if not good.any():
                        continue
                    nrep = int(good.sum())
                    Rs.append(np.full(nrep, R[i])); Zs.append(np.full(nrep, Z[i]))
                    Vs.append(yi[good]); Es.append(np.full(nrep, np.nan))
                    Tg.append(np.full(nrep, tag_of(i)))
        empty = np.array([])
        R, Z, V, E, Tg = (np.concatenate(a) if a else empty for a in (Rs, Zs, Vs, Es, Tg))
        order = np.argsort(R)
        return ChannelProfile(self.shot, 0.5 * (t0 + t1), f"{'+'.join(systems)}.{node.lower()}",
                              np.arange(R.size)[order], R[order], Z[order], V[order], units,
                              label=f"TS {'+'.join(systems)} {quantity}", tag=Tg[order],
                              error=(E[order] if average else None))

    def _eq_cache_path(self, tree: str, time: float) -> Path:
        return self.cache_dir / f"{self.shot}_eq_{tree}_{int(round(time))}.npz"

    def _eq_cache_load(self, tree: str, time: float):
        if not self.use_cache:
            return None
        p = self._eq_cache_path(tree, time)
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        if "qpsi" not in z.files:              # pre-qpsi cache -> re-fetch to populate it
            return None
        return EquilibriumData(int(z["shot"]), str(z["tree"]), float(z["time"]),
                               z["rgrid"], z["zgrid"], z["psiN"], z["rbbbs"],
                               z["zbbbs"], float(z["raxis"]), float(z["zaxis"]),
                               z["wall_r"], z["wall_z"],
                               *(float(z[k]) for k in ("rxpt1", "zxpt1", "rxpt2", "zxpt2",
                                                       "rvsin", "zvsin", "rvsout", "zvsout")),
                               z["qpsi"])

    def _eq_cache_save(self, tree: str, time: float, ed: EquilibriumData):
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._eq_cache_path(tree, time), shot=ed.shot,
                 tree=np.array(ed.tree), time=ed.time, rgrid=ed.rgrid,
                 zgrid=ed.zgrid, psiN=ed.psiN, rbbbs=ed.rbbbs, zbbbs=ed.zbbbs,
                 raxis=ed.raxis, zaxis=ed.zaxis, wall_r=ed.wall_r, wall_z=ed.wall_z,
                 rxpt1=ed.rxpt1, zxpt1=ed.zxpt1, rxpt2=ed.rxpt2, zxpt2=ed.zxpt2,
                 rvsin=ed.rvsin, zvsin=ed.zvsin, rvsout=ed.rvsout, zvsout=ed.zvsout,
                 qpsi=ed.qpsi)
