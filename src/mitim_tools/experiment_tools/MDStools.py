"""Machine-agnostic MDSplus retrieval layer (pure-python `mdsthin` thin client).

Shared by the per-machine packages (`experiment_tools.diiid`, `experiment_tools.cmod`):
the SSH tunnel, the connection (ONE tunnel + ONE mdsplus connection reused across
shots), the per-shot fetcher base (signal fetch with server-side resampling, disk
cache, full-array node reads) and the data containers (Signal, EquilibriumData,
ChannelProfile). What is machine-specific (signal-name resolution, EFIT node layout,
diagnostic profiles, time units) lives in the machine subclasses.

Access (where the MDSplus server is and how to reach it), resolved per machine:
    1. explicit kwargs:  server="host:port" (connect straight there), or
                         tunnel_host=<~/.ssh/config alias> (+ optional mds_server=)
    2. config_user.json top-level block, keyed by machine:
           "mdsplus": {"cmod":  {"tunnel_host": "mfews15", "mds_server": "alcdata.psfc.mit.edu:8000"},
                       "diiid": {"tunnel_host": null,      "mds_server": "atlas.gat.com:8000"}}
       (read only when neither `server` nor `tunnel_host` is passed)
    3. direct connection to the machine's default `MDS_SERVER` (on-site / VPN).
With a tunnel host, `ssh -N -L <localport>:<mds_server> <tunnel_host>` is opened for
you (key/agent auth only; ProxyJump etc. come from your ~/.ssh/config).

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
# Note: mdsthin returns arrays with the dimension order REVERSED vs the server's
# (Fortran) storage order, so e.g. a server [nr, nz, nt] array arrives as [nt, nz, nr].
# Time-slice with python full-array indexing (see the machine fetch_equilibrium).
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
        return self.rho_of_psin(ip(np.column_stack([Z, R])), kind)

    def rho_of_psin(self, psin, kind: str = "tor"):
        """Map normalized poloidal flux ψ_N to ρ_tor ('tor') or ρ_pol ('pol'), with the same
        conventions as `rho_of` (for profiles delivered on a ψ_N grid, e.g. C-Mod HIREXSR)."""
        psiN = np.clip(np.atleast_1d(np.asarray(psin, float)), 0.0, None)
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
    psin:     np.ndarray = None  # normalized ψ of each point when the profile comes on a flux grid
    #                              (inverted profiles, r/z are then NaN); plotting maps it to ρ directly


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


def orient_psi(psi, rgrid, zgrid, simag, sibry, raxis, zaxis):
    """Return an EFIT ψ(R,Z) slice oriented [nz, nr], deciding [Z,R] vs [R,Z] by which
    orientation puts the ψ extremum on the magnetic axis."""
    ii, jj = np.unravel_index(int(np.argmin(psi) if simag < sibry else np.argmax(psi)),
                              psi.shape)
    err_zr = abs(rgrid[jj] - raxis) + abs(zgrid[ii] - zaxis)   # psi is [Z, R]
    err_rz = abs(rgrid[ii] - raxis) + abs(zgrid[jj] - zaxis)   # psi is [R, Z]
    return psi if err_zr <= err_rz else psi.T                 # -> [nz, nr]


def reduce_channels(val2d, err2d, R, Z, tarr, t0, t1, tag_prefix, average=True):
    """Window-reduce one multi-channel diagnostic (value/error arrays over (channel, time)).

    Orients the arrays to (nchan, ntime), treats value <= 0 as "no measurement", and either
    averages over [t0, t1] (value and stored error; channels with no valid sample dropped) or,
    with `average=False`, keeps every valid sample in the window (error NaN). Returns the
    per-point arrays (R, Z, value, error, tag), with tags '<tag_prefix><channel index>'."""
    R, Z, tarr = np.atleast_1d(R), np.atleast_1d(Z), np.atleast_1d(tarr)
    if val2d.shape[0] == tarr.size and val2d.shape[1] != tarr.size:
        val2d, err2d = val2d.T, err2d.T    # orient to (nchan, ntime)
    tag_of = lambda i: f"{tag_prefix}{i}"
    if average:
        valid = np.where(val2d > 0, val2d, np.nan)            # 0 = no measurement
        v, _, _ = time_average(tarr, valid, t0, t1, axis=1)
        e, _, _ = time_average(tarr, np.where(val2d > 0, err2d, np.nan), t0, t1, axis=1)
        gd = np.isfinite(v) & (v > 0)
        return R[gd], Z[gd], v[gd], e[gd], np.array([tag_of(i) for i in np.arange(R.size)[gd]])
    Rs, Zs, Vs, Es, Tg = [], [], [], [], []
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
    return tuple(np.concatenate(a) if a else empty for a in (Rs, Zs, Vs, Es, Tg))


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
# SSH -L tunnel to an MDSplus server
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

class MDSConnection:
    """Holds a single SSH tunnel + mdsplus connection, reusable across shots.

    Open it once and share it among many <Machine>Fetcher(shot, connection=...)
    instances so a multi-shot job uses one tunnel, not one per shot. Subclasses set
    MACHINE (the config_user.json "mdsplus" key) and MDS_SERVER (default host:port).
    """

    MACHINE = None
    MDS_SERVER = None

    def __init__(self, server: str | None = None, tunnel_host: str | None = None,
                 mds_server: str | None = None, tunnel_timeout: float = 25.0):
        if server is None and tunnel_host is None:     # nothing explicit -> per-user config
            from mitim_tools.misc_tools.CONFIGread import read_mdsplus_access
            access = read_mdsplus_access(self.MACHINE)
            tunnel_host = access.get("tunnel_host")
            mds_server = mds_server or access.get("mds_server")
        self.server = server               # if set, connect directly to this host:port
        self.tunnel_host = tunnel_host     # None -> connect straight to mds_server (no tunnel)
        self.mds_host, self.mds_port = (mds_server or self.MDS_SERVER).split(":")
        self.tunnel_timeout = tunnel_timeout
        self._tunnel = None
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def conn(self):
        """The thin-client connection, established on first use (directly, or via an
        SSH tunnel when `tunnel_host` is set)."""
        if self._conn is None:
            if _mds is None:
                raise ImportError(
                    f"`mdsthin` is not installed ({_MDS_IMPORT_ERROR!r}). "
                    "Install the MDS extra: `pip install mitim-fusion[mds]` "
                    "(or `pip install mdsthin`).")
            if self.server is None:
                if self.tunnel_host is None:        # no jump host -> connect straight to the server
                    self.server = f"{self.mds_host}:{self.mds_port}"   # (on-site / over the VPN)
                else:                               # off-site -> open an SSH -L tunnel for you
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
# Per-shot fetcher base (shares an MDSConnection)
# =============================================================================

class MDSFetcher:
    """Per-shot MDSplus fetcher base (shares an MDSConnection across shots).

    Subclasses set CONNECTION (their MDSConnection subclass) and DEFAULT_CACHE, and
    implement `_assign_bare` (how a bare signal name resolves) and, if the machine's
    time base is not ms, `_time` (converts a fetched time axis to ms).
    """

    CONNECTION = MDSConnection
    DEFAULT_CACHE = __mitimroot__ / "tests" / "scratch" / "mds_fetcher"

    def __init__(self, shot: int, connection: MDSConnection | None = None,
                 max_points: int = 4000, use_cache: bool = True,
                 cache_dir: str | Path | None = None,
                 # forwarded to the connection when we create our own:
                 server: str | None = None, tunnel_host: str | None = None,
                 mds_server: str | None = None, tunnel_timeout: float = 25.0):
        self.shot = int(shot)
        self.max_points = max_points
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.DEFAULT_CACHE

        # provenance counters: how many fetches were served from disk cache vs the server
        # (a caller can snapshot the delta around a fetch to report "cached" vs "server").
        self.n_from_cache = 0
        self.n_from_server = 0

        self._own_conn = connection is None
        self.connection = connection or self.CONNECTION(
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

    def _time(self, t):
        """Machine time axis -> ms (identity: the stored time base is already ms)."""
        return t

    # ---- public fetch -------------------------------------------------------
    def fetch_signal(self, spec: str, label: str = "", name: str = "",
                     max_points: int | None = None) -> Signal:
        """Fetch a signal spec as a Signal (value + time [ms] + units).

        Large traces are resampled on the server to <= max_points before
        transfer. Results are cached to disk unless use_cache is False.
        """
        mp = self.max_points if max_points is None else max_points
        name = name or spec

        cached = self._cache_load(spec, mp, name)   # may raise (a cached miss); counts a cache hit
        if cached is not None:
            return cached

        print(f"  [MDS] #{self.shot}: '{name}' not in cache -> fetching from database")
        self.n_from_server += 1                      # cache miss -> hitting the server
        try:
            source = self._assign(spec.strip())
            self._server_reduce(mp)
            data = np.atleast_1d(self._value("_s"))
        except Exception as excp:              # a structured node path RAISES when the node is absent
            low = str(excp).lower()            # cache the miss ONLY for a definitive "node absent",
            if any(k in low for k in ("nnf", "nodata", "not found", "node not")):   # not a transient
                self._cache_save_miss(spec, mp, name, str(excp)[:40])               # error (VPN/conn)
            raise ValueError(f"no data ({spec}): {str(excp)[:45]}")
        if data.size <= 1:                     # scalar/empty => no usable trace
            self._cache_save_miss(spec, mp, name, source)   # remember the miss
            raise ValueError(f"no data ({source})")
        time_ = self._time(np.atleast_1d(self._value("dim_of(_s,0)")))
        sig = Signal(name, time_, data, units=self._safe_units(),
                     label=label or spec, source=source)
        self._cache_save(spec, mp, sig)
        return sig

    def fetch_signals(self, specs, max_points: int | None = None) -> dict:
        """Fetch many signals. `specs` is an iterable of (key, spec, label).

        Returns {key: Signal}; any signal that errors or is absent maps to None
        (real shots are routinely missing diagnostics, so we don't abort).
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

    # ---- signal resolution --------------------------------------------------
    def _assign(self, spec: str) -> str:
        """Resolve `spec` and assign it to server-side `_s`; return provenance.

        `<TREE>::<expr>` opens <TREE> and evaluates <expr>; a full node `\\<TREE>::...`
        opens <TREE> and evaluates the node path. Bare names go to `_assign_bare`.
        """
        if "::" in spec:
            head, rest = spec.split("::", 1)
            tree = head.lstrip("\\").strip()
            tdi = spec if spec.startswith("\\") else rest.strip()
            self._open_tree(tree)
            self.conn.get(f"_s = {tdi}")
            return spec
        return self._assign_bare(spec)

    def _assign_bare(self, spec: str) -> str:
        raise ValueError(f"'{spec}' is not a <TREE>::<expr> spec and {type(self).__name__} "
                         "has no bare-name resolution")

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
        self.n_from_cache += 1                 # file exists -> served from cache (data or cached-miss)
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
        print(f"  [MDS] #{self.shot}: '{node}' not in cache -> fetching from database")
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

    # ---- equilibrium disk cache ---------------------------------------------
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
