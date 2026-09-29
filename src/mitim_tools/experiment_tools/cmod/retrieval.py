"""Alcator C-Mod experimental-data retrieval (MDSplus, via the pure-python `mdsthin`).

C-Mod counterpart of `experiment_tools.diiid.retrieval`. Signal fetch, the EFIT reader,
the Thomson reduction, caching and the connection all come from the shared
`experiment_tools.MDStools` base; this module declares C-Mod's node map (SIGNALS, EFIT,
Thomson) and adds HIREXSR. `diiid.plotting.overview(..., connection=CMODConnection(...))`
draws C-Mod data with the machine defaults below (EFIT tree, time window).

Signal specs: a bare name from `CMODFetcher.SIGNALS` (ip, bt, kappa, q95, wmhd, nebar,
zeff, prad, ...) or any `\\<TREE>::<node>` / `<TREE>::<expr>` spec.

Conventions / units (C-Mod):
    * The tree time base is SECONDS; every time returned here (Signal.time, profile and
      equilibrium times) is converted to **ms** (x1e3) to match the plotting and the DIII-D
      module. Every time argument (`time`, `t_window`) is in ms too.
    * Ip and B_T are stored NEGATIVE in the standard C-Mod field/current direction; the
      'ip'/'bt' aliases flip the sign (positive = standard direction; raw nodes stay as stored).
    * Units in SIGNALS are checked against the stored magnitudes; they override the tree's
      unit tag where the tag is blank or wrong (see SIGNALS comments).

Connecting to the server:
    `alcdata.psfc.mit.edu:8000` is only reachable inside the PSFC network. Off-site, pass
    `tunnel_host=<ssh alias of a PSFC host>` or set it once in config_user.json
    ("mdsplus": {"cmod": {"tunnel_host": ...}}, see `MDStools`). No eqtools needed.
"""

from __future__ import annotations

import numpy as np

from mitim_tools import __mitimroot__
from mitim_tools.experiment_tools.MDStools import (
    ChannelProfile, MDSConnection, MDSFetcher, time_average)

_A = r"\{tree}::TOP:EFIT.RESULTS.A_EQDSK"       # the C-Mod EFIT layout has a COLON after TOP
_A_ANALYSIS = _A.format(tree="ANALYSIS")

# Thomson scattering: system -> (node group, {q: (value, error, factor -> eV | m^-3)}, Z node)
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


class CMODConnection(MDSConnection):
    """Holds a single SSH tunnel + mdsplus connection to alcdata, reusable across shots."""

    MACHINE = "cmod"
    MDS_SERVER = "alcdata.psfc.mit.edu:8000"


class CMODFetcher(MDSFetcher):
    """Per-shot C-Mod MDSplus fetcher (shares a CMODConnection across shots).

        with CMODConnection(tunnel_host="<your ssh alias>") as conn:
            for shot in shots:
                f = CMODFetcher(shot, connection=conn)
                ip = f.fetch_signal("ip")                        # time [ms], Ip [A] (> 0)
                te = f.fetch_thomson_profile(1000.0, "te", "all")  # eV, (R, Z) per channel
    """

    CONNECTION = CMODConnection
    DEFAULT_CACHE = __mitimroot__ / "tests" / "scratch" / "cmod_fetcher"
    TIME_TO_MS = 1e3                  # tree time base is seconds
    T_WINDOW = (0.0, 2000.0)          # [ms]
    T_REF = 1000.0                    # [ms]
    EFIT = dict(tree="ANALYSIS", g=r"\{tree}::TOP:EFIT.RESULTS.G_EQDSK", a=_A,
                time_axis=-1,         # QPSI/FPOL/PRES/FFPRIM/PPRIME/RBBBS/ZBBBS are (n, nt)
                # strike points stored in cm; X-points are in m (ZXPT2's tag says cm, values are m)
                a_scale={"RVSIN": 1e-2, "ZVSIN": 1e-2, "RVSOUT": 1e-2, "ZVSOUT": 1e-2},
                no_xpoint=-9.0)       # -9.99 = no X-point
    TS_ALL = ("core", "edge")

    # bare name -> (spec, units, factor). Units verified against the stored values; the tree tag
    # is blank for nebar/nl04/zeff/kappa/q95 and WRONG for prad (tag 'MW', values in W, consistent
    # with \twopi_foil which is tagged W). Ip/B_T are stored < 0 -> factor -1.
    SIGNALS = {
        "ip":       (r"\MAGNETICS::IP", "A", -1),
        "bt":       (r"\MAGNETICS::BTOR", "T", -1),
        "kappa":    (rf"{_A_ANALYSIS}:KAPPA", "", 1),
        "tritop":   (rf"{_A_ANALYSIS}:TRITOP", "", 1),       # upper triangularity
        "tribot":   (rf"{_A_ANALYSIS}:TRIBOT", "", 1),       # lower triangularity
        "q95":      (rf"{_A_ANALYSIS}:Q95", "", 1),
        "wmhd":     (rf"{_A_ANALYSIS}:WMHD", "J", 1),        # EFIT stored energy (== WPLASM)
        "wdia":     (rf"{_A_ANALYSIS}:WDIA", "J", 1),        # diamagnetic stored energy
        "betap":    (rf"{_A_ANALYSIS}:BETAP", "", 1),
        "li":       (rf"{_A_ANALYSIS}:LI", "", 1),
        "vloop":    (rf"{_A_ANALYSIS}:VLOOPT", "V", 1),
        "nebar":    (r"\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT", "m^-3", 1),   # line-avg ne
        "nl04":     (r"\ELECTRONS::TOP.TCI.RESULTS:NL_04", "m^-2", 1),                 # central chord ∫ne dl
        "zeff":     (r"\SPECTROSCOPY::Z_AVE", "", 1),    # visible-bremsstrahlung line-averaged Zeff
        "prad":     (r"\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.FOIL:MAIN_POWER", "W", 1),  # foil-bolometer P_rad
        "prad_2pi": (r"SPECTROSCOPY::\twopi_foil", "W", 1),                             # 2pi foil estimate
    }

    # ---- Thomson scattering -------------------------------------------------
    def _thomson_arrays(self, system: str, q: str):
        """'core': YAG_NEW.RESULTS.PROFILES TE_RZ [keV -> eV] / NE_RZ [m^-3], errors TE_ERR/NE_ERR,
        Z = dim_of(value, 1) [m]. 'edge': YAG_EDGETS.RESULTS TE [eV] / NE [m^-3], errors
        TE:ERROR/NE:ERROR, Z = DATA:FIBER_Z [m]. Every channel sits on the vertical laser at
        R = YAG.RESULTS.PARAM:R (0.69 m); ρ comes from mapping (R, Z) through EFIT (the stored
        R_MID_T/RHO_T/PSINORM are not used). Time dim_of(value, 0) [s -> ms]."""
        base, leaves, znode = _TS[system]
        vleaf, eleaf, fac = leaves[q]
        val2d, _ = self._value_cached(f"{base}:{vleaf}", tree="ELECTRONS")
        err2d, _ = self._value_cached(f"{base}:{eleaf}", tree="ELECTRONS")
        tarr, _ = self._value_cached(f"dim_of({base}:{vleaf},0)", tree="ELECTRONS")
        Z, _ = self._value_cached(znode or f"dim_of({base}:{vleaf},1)", tree="ELECTRONS")
        R, _ = self._value_cached(_TS_LASER_R, tree="ELECTRONS")
        Z = np.atleast_1d(Z)
        return (val2d * fac, err2d * fac, np.full(Z.size, float(R)), Z, self._time(tarr),
                "eV" if q == "te" else "m^-3")


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
