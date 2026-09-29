"""Alcator C-Mod experimental-data retrieval (MDSplus, via the pure-python `mdsthin`).

C-Mod counterpart of `experiment_tools.diiid.retrieval`, built on the shared
`experiment_tools.MDStools` layer (connection, cache, containers). It exposes the
same interface the plotting uses (`fetch_signal`, `fetch_equilibrium`,
`fetch_thomson_profile`, plus `fetch_hirex_profile`), so
`diiid.plotting.overview(..., machine="cmod")` draws C-Mod data.

Signal specs: a bare name from `CMODFetcher.SIGNALS` (ip, bt, kappa, q95, wmhd, nebar,
zeff, prad, ...) or any `\\<TREE>::<node>` / `<TREE>::<expr>` spec.

Conventions / units (C-Mod):
    * The tree time base is SECONDS; every time returned here (Signal.time, profile and
      equilibrium times) is converted to **ms** (x1e3) to match the plotting and the DIII-D
      module. Every time argument (`time`, `t_window`) is in ms too.
    * Ip and Bt are stored NEGATIVE in the standard C-Mod field/current direction.
    * Units in SIGNALS are checked against the stored magnitudes; they override the tree's
      unit tag where the tag is blank or wrong (see SIGNALS comments).

Connecting to the server:
    `alcdata.psfc.mit.edu:8000` is only reachable inside the PSFC network. Off-site, pass
    `tunnel_host=<ssh alias of a PSFC host>` or set it once in config_user.json
    ("mdsplus": {"cmod": {"tunnel_host": ...}}, see `MDStools`). No eqtools needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mitim_tools import __mitimroot__
from mitim_tools.experiment_tools.MDStools import (
    EquilibriumData, ChannelProfile, MDSConnection, MDSFetcher, time_average,
    orient_psi, reduce_channels, _write_geqdsk)

# EFIT (ANALYSIS tree) node groups. The C-Mod layout has a COLON after TOP.
_EFIT = r"\ANALYSIS::TOP:EFIT.RESULTS"
_A, _G = f"{_EFIT}.A_EQDSK", f"{_EFIT}.G_EQDSK"

# Thomson scattering: system -> (node group, {quantity: (value, error, factor->eV|m^-3)}, Z node)
_TS = {
    "core": (r"\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES",
             {"te": ("TE_RZ", "TE_ERR", 1e3), "ne": ("NE_RZ", "NE_ERR", 1.0)},   # core Te stored in keV
             None),                                                           # Z = dim_of(value, 1) [m]
    "edge": (r"\ELECTRONS::TOP.YAG_EDGETS.RESULTS",
             {"te": ("TE", "TE:ERROR", 1.0), "ne": ("NE", "NE:ERROR", 1.0)},      # edge Te stored in eV
             r"\ELECTRONS::TOP.YAG_EDGETS.DATA:FIBER_Z"),
}
_TS_LASER_R = r"\ELECTRONS::TOP.YAG.RESULTS.PARAM:R"     # vertical YAG laser major radius [m] (0.69)

# HIREXSR inverted profiles (THACO). Line -> node under ...HIREXSR.ANALYSIS[n].
_HIREX_LINES = {"w": "HELIKE.PROFILES.W", "x": "HELIKE.PROFILES.X", "z": "HELIKE.PROFILES.Z",
                "lya1": "HLIKE.PROFILES.LYA1", "j": "HLIKE.PROFILES.J", "mo4d": "HLIKE.PROFILES.MO4D"}
# PRO moment index -> (units). From THACO hirexsr_calc_profiles.pro (github mlreinke/THACO):
# ipro[*,0] emissivity (relative), ipro[*,1] toroidal rotation FREQUENCY f = omega/2pi [kHz] (l.199),
# ipro[*,3] Ti [keV] (l.257). (ipro[*,2] is the poloidal term in km/s/T, zero for solid-body fits.)
_HIREX_MOMENTS = {"emiss": (0, "arb"), "omega": (1, "kHz"), "ti": (3, "keV")}


def _efit_time_index(gtime_s, time_ms):
    """Index of the EFIT slice nearest `time_ms` on a time base stored in seconds."""
    return int(np.argmin(np.abs(np.asarray(gtime_s, float) * 1e3 - time_ms)))


class CMODConnection(MDSConnection):
    """Holds a single SSH tunnel + mdsplus connection to alcdata, reusable across shots."""

    MACHINE = "cmod"
    MDS_SERVER = "alcdata.psfc.mit.edu:8000"


class CMODFetcher(MDSFetcher):
    """Per-shot C-Mod MDSplus fetcher (shares a CMODConnection across shots).

        with CMODConnection(tunnel_host="mfews15") as conn:
            for shot in shots:
                f = CMODFetcher(shot, connection=conn)
                ip = f.fetch_signal("ip")                        # time [ms], Ip [A] (< 0)
                te = f.fetch_thomson_profile(1000.0, "te", "all")  # eV, (R, Z) per channel
    """

    CONNECTION = CMODConnection
    DEFAULT_CACHE = __mitimroot__ / "tests" / "scratch" / "cmod_fetcher"

    # bare name -> (spec, units). Units verified against the stored values; the tree tag is
    # blank for nebar/nl04/zeff/kappa/q95 and WRONG for prad (tag 'MW', values in W, consistent
    # with \twopi_foil which is tagged W).
    SIGNALS = {
        "ip":       (r"\MAGNETICS::IP", "A"),
        "bt":       (r"\MAGNETICS::BTOR", "T"),
        "kappa":    (rf"{_A}:KAPPA", ""),
        "tritop":   (rf"{_A}:TRITOP", ""),                # upper triangularity
        "tribot":   (rf"{_A}:TRIBOT", ""),                # lower triangularity
        "q95":      (rf"{_A}:Q95", ""),
        "wmhd":     (rf"{_A}:WMHD", "J"),                 # EFIT stored energy (== WPLASM)
        "wdia":     (rf"{_A}:WDIA", "J"),                 # diamagnetic stored energy
        "betap":    (rf"{_A}:BETAP", ""),
        "li":       (rf"{_A}:LI", ""),
        "vloop":    (rf"{_A}:VLOOPT", "V"),
        "nebar":    (r"\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT", "m^-3"),   # line-avg ne (TCI+EFIT)
        "nl04":     (r"\ELECTRONS::TOP.TCI.RESULTS:NL_04", "m^-2"),                 # central TCI chord ∫ne dl
        "zeff":     (r"\SPECTROSCOPY::Z_AVE", ""),        # visible-bremsstrahlung line-averaged Zeff
        "prad":     (r"\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.FOIL:MAIN_POWER", "W"),  # foil-bolometer P_rad
        "prad_2pi": (r"SPECTROSCOPY::\twopi_foil", "W"),  # 2pi foil-bolometer estimate
    }

    # ---- time + signal resolution -------------------------------------------
    def _time(self, t):
        """C-Mod stores time in seconds -> ms."""
        return np.asarray(t, float) * 1e3

    def _assign_bare(self, spec: str) -> str:
        if spec.lower() not in self.SIGNALS:
            raise ValueError(f"unknown C-Mod signal '{spec}': use one of {sorted(self.SIGNALS)} "
                             r"or a '\TREE::NODE' spec")
        return self._assign(self.SIGNALS[spec.lower()][0])

    def fetch_signal(self, spec: str, label: str = "", name: str = "",
                     max_points: int | None = None):
        """As MDSFetcher.fetch_signal; a SIGNALS alias also gets its verified units."""
        sig = super().fetch_signal(spec, label=label, name=name, max_points=max_points)
        if spec.lower() in self.SIGNALS:
            sig.units = self.SIGNALS[spec.lower()][1]
        return sig

    # ---- EFIT equilibrium (ANALYSIS tree) -------------------------------------
    def _efit_slice(self, time: float, tree: str):
        """Raw EFIT G-file quantities at the slice nearest `time` [ms].

        mdsthin layouts (python order): PSIRZ (nt, nz, nr) -> [it]; the 1D profiles
        (QPSI/FPOL/PRES/FFPRIM/PPRIME) and RBBBS/ZBBBS are (n, nt) -> [:, it]; per-time
        scalars (nt,) -> [it]; XDIM/ZDIM/RZERO/ZMID are shot scalars. All SI (m, Wb/rad).
        """
        self.conn.openTree(tree, self.shot)
        G = _G.replace("ANALYSIS", tree.upper(), 1)
        gtime = np.atleast_1d(self._value(f"{G}:GTIME")).astype(float)       # [s]
        it = _efit_time_index(gtime, time)
        v = lambda n: np.asarray(self._value(f"{G}:{n}"), float)
        rgrid, zgrid = np.atleast_1d(v("RGRID")), np.atleast_1d(v("ZGRID"))
        simag, sibry = float(v("SSIMAG")[it]), float(v("SSIBRY")[it])
        rax, zax = float(v("RMAXIS")[it]), float(v("ZMAXIS")[it])
        nb = int(np.atleast_1d(self._value(f"{G}:NBBBS"))[it])
        lim = v("LIM")                                                         # (nlim, 2) [m]
        return dict(it=it, t_act=float(gtime[it]) * 1e3, rgrid=rgrid, zgrid=zgrid,
                    psi=orient_psi(v("PSIRZ")[it], rgrid, zgrid, simag, sibry, rax, zax),
                    simag=simag, sibry=sibry, rax=rax, zax=zax,
                    rb=v("RBBBS")[:nb, it], zb=v("ZBBBS")[:nb, it],
                    rlim=lim[:, 0], zlim=lim[:, 1], qpsi=v("QPSI")[:, it], G=G, v=v)

    def fetch_equilibrium(self, time: float, tree: str = "ANALYSIS") -> EquilibriumData:
        """EFIT flux-surface snapshot nearest `time` [ms] from the ANALYSIS tree (cached).

        X-points (A_EQDSK RXPT/ZXPT) are in m with -9.99 = none (-> NaN); the tag of ZXPT2
        says 'cm' but its values are m. Strike points RVSIN/ZVSIN/RVSOUT/ZVSOUT are stored
        in cm (-> m); they are 0 for a limited plasma (then no separatrix legs are drawn).
        """
        cached = self._eq_cache_load(tree, time)
        if cached is not None:
            self.n_from_cache += 1
            print(f"Using cached equilibrium for tree {tree} at time {time}")
            return cached

        self.n_from_server += 1
        s = self._efit_slice(time, tree)
        A = _A.replace("ANALYSIS", tree.upper(), 1)
        ita = _efit_time_index(np.atleast_1d(self._value(f"{A}:ATIME")), time)

        def asc(node, factor=1.0):
            x = float(np.ravel(self._value(f"{A}:{node}"))[ita])
            return np.nan if x <= -9.0 else x * factor                        # -9.99 = no X-point

        ed = EquilibriumData(
            self.shot, tree, s["t_act"], s["rgrid"], s["zgrid"],
            (s["psi"] - s["simag"]) / (s["sibry"] - s["simag"]), s["rb"], s["zb"],
            s["rax"], s["zax"], s["rlim"], s["zlim"],
            asc("RXPT1"), asc("ZXPT1"), asc("RXPT2"), asc("ZXPT2"),
            asc("RVSIN", 1e-2), asc("ZVSIN", 1e-2), asc("RVSOUT", 1e-2), asc("ZVSOUT", 1e-2),
            s["qpsi"])
        self._eq_cache_save(tree, time, ed)
        return ed

    def fetch_geqdsk(self, time: float, tree: str = "ANALYSIS", path=None) -> Path:
        """Write a standard GEQDSK (g-file) for the EFIT slice nearest `time` [ms] (SI units,
        C-Mod signs as stored: Ip and B_T negative). Returns the output path."""
        s = self._efit_slice(time, tree)
        v, it = s["v"], s["it"]
        data = dict(case=f"EFIT {tree} #{self.shot} {s['t_act']:.0f}ms",
                    nw=s["rgrid"].size, nh=s["zgrid"].size,
                    rdim=float(v("XDIM")), zdim=float(v("ZDIM")), rcentr=float(v("RZERO")),
                    rleft=float(s["rgrid"].min()), zmid=float(v("ZMID")),
                    rmaxis=s["rax"], zmaxis=s["zax"], simag=s["simag"], sibry=s["sibry"],
                    bcentr=float(v("BCENTR")[it]), current=float(v("CPASMA")[it]), psirz=s["psi"],
                    fpol=v("FPOL")[:, it], pres=v("PRES")[:, it], ffprime=v("FFPRIM")[:, it],
                    pprime=v("PPRIME")[:, it], qpsi=s["qpsi"],
                    rbbbs=s["rb"], zbbbs=s["zb"], rlim=s["rlim"], zlim=s["zlim"])
        path = Path(path) if path is not None else (self.cache_dir / f"g{self.shot}.{int(round(s['t_act'])):05d}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write_geqdsk(path, data)

    # ---- Thomson scattering -------------------------------------------------
    def fetch_thomson_profile(self, time: float, quantity: str = "te", system="core",
                              window: float = 100.0, t_window=None,
                              average: bool = True) -> ChannelProfile:
        """Thomson-scattering Te [eV] or ne [m^-3] vs (R, Z) per channel, time-averaged.

        `system`: 'core' (YAG_NEW.RESULTS.PROFILES: TE_RZ [keV -> eV], NE_RZ [m^-3], errors
        TE_ERR/NE_ERR, Z = dim_of(value, 1) [m]), 'edge' (YAG_EDGETS.RESULTS: TE [eV], NE
        [m^-3], errors TE:ERROR/NE:ERROR, Z = DATA:FIBER_Z [m]), 'all' (= both) or a list.
        Every channel sits on the vertical laser at R = YAG.RESULTS.PARAM:R (0.69 m); ρ is
        obtained by mapping (R, Z) through EFIT, like DIII-D (the stored R_MID_T/RHO_T/PSINORM
        are not used). Averaging/window/tags ('C#', 'E#') as in the DIII-D fetcher.
        """
        systems = (["core", "edge"] if system == "all"
                   else [system] if isinstance(system, str) else list(system))
        q = "te" if quantity.lower() in ("te", "temp") else "ne"
        t0, t1 = t_window if t_window is not None else (time - window, time + window)
        Rs, Zs, Vs, Es, Tg = [], [], [], [], []
        for sysname in systems:
            base, leaves, znode = _TS[sysname]
            vleaf, eleaf, fac = leaves[q]
            try:
                val2d, _ = self._value_cached(f"{base}:{vleaf}", tree="ELECTRONS")
                err2d, _ = self._value_cached(f"{base}:{eleaf}", tree="ELECTRONS")
                tarr, _ = self._value_cached(f"dim_of({base}:{vleaf},0)", tree="ELECTRONS")
                Z, _ = self._value_cached(znode or f"dim_of({base}:{vleaf},1)", tree="ELECTRONS")
                R, _ = self._value_cached(_TS_LASER_R, tree="ELECTRONS")
            except Exception as e:
                print(f"  ! TS {sysname} unavailable for #{self.shot}: {str(e)[:45]}")
                continue
            Z = np.atleast_1d(Z)
            pts = reduce_channels(val2d * fac, err2d * fac, np.full(Z.size, float(R)), Z,
                                  self._time(tarr), t0, t1, sysname[0].upper(), average)
            if average or pts[0].size:
                for acc, a in zip((Rs, Zs, Vs, Es, Tg), pts):
                    acc.append(a)
        empty = np.array([])
        R, Z, V, E, Tg = (np.concatenate(a) if a else empty for a in (Rs, Zs, Vs, Es, Tg))
        order = np.argsort(R, kind="stable")
        return ChannelProfile(self.shot, 0.5 * (t0 + t1), f"{'+'.join(systems)}.{q}",
                              np.arange(R.size)[order], R[order], Z[order], V[order],
                              "eV" if q == "te" else "m^-3",
                              label=f"TS {'+'.join(systems)} {quantity}", tag=Tg[order],
                              error=(E[order] if average else None))

    # ---- HIREXSR (x-ray imaging crystal spectrometer, THACO inversions) -----
    def fetch_hirex_profile(self, time: float, quantity: str = "ti", system: str = "z",
                            window: float = 100.0, t_window=None, average: bool = True,
                            tht: int = 0) -> ChannelProfile:
        """HIREXSR inverted profile on its ψ_N grid (`ChannelProfile.psin`; r/z are NaN).

        `quantity`: 'ti' [keV], 'omega' = toroidal rotation FREQUENCY f = v_tor/(2 pi R) [kHz]
        (not a velocity, not rad/s), 'emiss' (relative). Stored sign is kept: the THACO code
        does not document the sign relative to Ip/B_T. `system` is the line: He-like
        'w'|'x'|'z' (Ar16+, whole plasma) or H-like 'lya1'|'j'|'mo4d' (Ar17+, core). `tht` picks
        the THACO analysis branch (0 -> HIREXSR.ANALYSIS, n -> HIREXSR.ANALYSIS<n>).
        Node PRO is (moment, time, ρ) in python order with dim_of(PRO,1) = time [s]; RHO is
        ψ_N (tree units 'psin') per (time, ρ). Fill values (time or ψ_N == -1) are dropped.
        When THACO also fitted a poloidal m=1 term, PRO holds 2*nρ radial points (THACO
        hirexsr_calc_profiles.pro: ipro[0:nrho-1,*] flux-surface profile, ipro[nrho:*,*] the m=1
        term); only the first nρ (the flux-surface profile, matching RHO) are returned.
        No quality filter is applied: THACO stores unconverged inversions too (e.g. Ti of
        100s of keV), so check `error` and the line choice ('lya1' is the core Ti workhorse).
        The error bar is the stored PROERR (window-averaged with `average=True`).
        """
        m, units = _HIREX_MOMENTS[quantity.lower()]
        base = (r"\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS" + (str(tht) if tht else "")
                + "." + _HIREX_LINES[system.lower()])
        pro, _ = self._value_cached(f"{base}:PRO", tree="SPECTROSCOPY")
        err, _ = self._value_cached(f"{base}:PROERR", tree="SPECTROSCOPY")
        psin, _ = self._value_cached(f"{base}:RHO", tree="SPECTROSCOPY")
        tsec, _ = self._value_cached(f"dim_of({base}:PRO,1)", tree="SPECTROSCOPY")
        t = np.where(tsec == -1, np.nan, self._time(tsec))
        pro, err = pro[..., :psin.shape[1]], err[..., :psin.shape[1]]   # drop the m=1 half, if any
        good = (psin >= 0) & np.isfinite(t)[:, None]
        val = np.where(good, pro[m], np.nan)
        er = np.where(good, err[m], np.nan)
        ps = np.where(good, psin, np.nan)
        t0, t1 = t_window if t_window is not None else (time - window, time + window)
        if average:
            vals, _, _ = time_average(t, val, t0, t1, axis=0)
            errs, _, _ = time_average(t, er, t0, t1, axis=0)
            psis, _, _ = time_average(t, ps, t0, t1, axis=0)
        else:
            tm = (t >= t0) & (t <= t1)
            vals, errs, psis = val[tm].ravel(), er[tm].ravel(), ps[tm].ravel()
        keep = np.isfinite(vals) & np.isfinite(psis)
        vals, errs, psis = vals[keep], errs[keep], psis[keep]
        order = np.argsort(psis, kind="stable")
        ch = np.flatnonzero(keep)[order] % pro.shape[-1]
        nan = np.full(vals.size, np.nan)
        return ChannelProfile(self.shot, 0.5 * (t0 + t1), f"hirexsr.{system}.{quantity}",
                              ch, nan, nan, vals[order], units,
                              label=f"HIREXSR {system} {quantity}",
                              tag=np.array([f"{system}{i}" for i in ch]),
                              error=errs[order], psin=psis[order])


# generic names by which the machine-agnostic plotting resolves this backend
Connection, Fetcher = CMODConnection, CMODFetcher
