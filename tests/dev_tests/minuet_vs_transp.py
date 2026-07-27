"""
Shared machinery for the MINUET-vs-TRANSP current-diffusion benchmarks.

Run the thin drivers, not this module:
    python tests/dev_tests/minuet_vs_transp_01_no_sawteeth.py
    python tests/dev_tests/minuet_vs_transp_02_sawteeth.py

Both run the SAME problem — the SPARC PRD input.gacode, poloidal-field
diffusion only, matched physics — through a genuine MAESTRO `transp_soft`
beat and through MINUET, and open a multi-tab FigureNotebook comparison.

Physics matching:
    | aspect        | TRANSP (transp_soft beat)          | MINUET                     |
    |---------------|------------------------------------|----------------------------|
    | CD equation   | nlmdif=T on xb = sqrt(Phi/Phi_b)   | same equation, same label  |
    | equilibrium   | TEQ fixed-boundary (levgeo=11)     | fixed-boundary GS          |
    | GS pressure   | its own evolving p (PTOWB)         | TRANSP's p(x, t), commanded|
    | resistivity   | Sauter (nlres_sau=T, default)      | Sauter, coulomb_log=transp |
    | bootstrap     | Sauter (bootstrap_model='sauter')  | SauterBootstrap — MATCHED  |
    | sawteeth      | per variant (beat `sawteeth` knob) | per variant                |
    | Ip            | file HEADER value via CUR ufile    | same, via DiffusionSettings|
    | kinetics      | interpretive UFILES, constant      | frozen from file           |
    | q(t=0)        | QPR ufile from the same file       | file q                     |

Boundary provenance (answer to "same boundary?"): BOTH boundaries descend
from the same input.gacode LCFS, through different parameterizations —
TRANSP receives it as RFS/ZFS moment UFILES written by to_transp (MXH
smoothing), MINUET reconstructs it from the file's own MXH coefficients.
The measured mismatch is quantified on the Boundary tab (mm-level).

Known remaining model differences (the "why are they different" list, also
shown on the notebook's last tab):
  - TRANSP mixes Te/Ti at each sawtooth crash (nlsawe/nlsawi, even in
    interpretive mode) -> its resistivity/Porcelli inputs evolve; MINUET's
    kinetics are frozen (its GS pressure follows TRANSP's, but eta and the
    trigger see the unmixed file profiles).
  - Sawtooth period floors: TRANSP c_sawtooth(2)*tau_PM (~10 ms here) vs
    MINUET min_interval = 20 ms.
  - Porcelli input approximations differ (kappa1, Bp1 evaluation).
  - Porcelli TRIGGER: fully matched -- route (eq13; with no fast ions
    TRANSP's Eq 13 threshold is identically zero, so it crashes the moment
    -dW crosses 0; Eq 14/15b are never satisfied in either code here),
    inputs (perimeter-averaged Bp1, shelf-side s1, outermost q=1 crossing;
    see the TRANSP-PARITY SETTINGS block) and dW assembly (verified term by
    term; li1 is the paper's area-weighted definition). The last culprit
    was the betap1 PRESSURE SOURCE: TRANSP feeds its own evolving TOTAL
    pressure state -- matched by MINUET now using the equilibrium pressure
    (= the commanded p_of_x_t) in the margin. Phase-aligned scalars agree
    to ~1% (Bp1 3.5%); the residual cycle difference is 22 vs 25 crashes,
    464 vs 402 ms, core level -0.8%.
  - Porcelli REDISTRIBUTION is matched in TRANSP-parity mode (identified
    empirically on the hires CDF, dtmaxb = 2 ms): island width follows
    fporcelli * x_q1 (not a fraction of x_mix), and the axial region
    RECONNECTS outward -- the new axis inherits the helical flux of
    (nearly) the mixing-radius surface, giving a flat core ~1.5% (in q)
    ABOVE the paper's helicity-conserving Taylor value (two-anchor rule
    reproduces the measured post-crash core to 4 sig figs). MINUET's
    DEFAULT stays paper-faithful (eq 24 Taylor core); the benchmark passes
    PorcelliReconnection(0.63, width_convention='x_q1', core='reconnect',
    core_anchor_inset=1.5/nzones). Without this, the marginally-balanced
    limit cycle integrates the per-crash core-lift wedge into a persistent
    core-q level offset (~0.875 vs ~0.93 -- measured with the Taylor core).
    TRANSP's cycle statistics are stepping-robust (hires vs coarse:
    identical 25 crashes / ~400 ms / level).
  - TEQ vs MINUET's stretched-polar GS; flux-average conventions of the
    reported current density (CUR vs <J.B>/B0); nzones=200 = n_cells=200.
  - TRANSP's NEAR-AXIS bootstrap fix-up: in the first ~4 zones its CURBS is
    a smooth fill, not the Sauter evaluation (provable from the CDF itself:
    there CURBS != CURBSNE+CURBSTE+CURBSNI+CURBSTI, identical everywhere
    else). MINUET keeps the pointwise formula (jbs -> 0 at axis with ft).
    Deliberately NOT copied: it is a regularization artifact (~3 kA,
    ~+0.1% on the core q band).
  - Spitzer N(Z) charge factor: TRANSP's chain uses a 0.76 numerator
    variant; MINUET always uses Sauter's published 0.74 (deliberate):
    ~+1% flat eta difference.
  - The SPARC PRD file header Ip (8.7 MA) is inconsistent with its own
    equilibrium (8.32 MA): both codes are forced to the header value.
"""

import json
import time

import numpy as np
import matplotlib.pyplot as plt

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.transp_tools import CDFtools
from mitim_tools.transp_tools.CDFtools import getFluxSurface
from mitim_modules.maestro.scripts import run_maestro

from minuet import (minuet, Settings, DiffusionSettings, SauterBootstrap,
                    SauterResistivity, PorcelliSawtooth, PorcelliReconnection,
                    InputGacode)

GACODE = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"
GEQ = __mitimroot__ / "tests" / "data" / "SPARC_DN_PRD_freegs_20221013.geq"
TEMPLATE = __mitimroot__ / "templates" / "namelist.maestro.yaml"

# The BENCHMARK SOURCE is built, not taken raw: the PRD input.gacode's
# internal equilibrium (MXH moments from an old TRANSP statefile) is NOT the
# GS solution of its own boundary+p+q, and its header Ip (8.7 MA) disagrees
# with its own flux content (8.32 MA) -- every code absorbs those
# inconsistencies differently, which polluted the comparison at the 2-3%
# level. Instead: initialize MINUET from the FreeGS geqdsk (a true GS
# solution, Ip-consistent to 8e-5) with the PRD kinetics, and export that
# solved state as a fresh input.gacode (kinetics verbatim). BOTH codes then
# read the SAME exactly-GS-consistent file.


def _build_bench_source():
    bench = __mitimroot__ / "tests" / "scratch" / "bench_SPARC_PRD_geq.input.gacode"
    if bench.exists():
        return bench
    from minuet import GEQDSK, TabulatedKineticProfiles, InputGacode as _IG
    print("\t- Building the GS-consistent benchmark source (geqdsk + PRD kinetics)")
    gq = GEQDSK.from_file(str(GEQ))
    kin = TabulatedKineticProfiles.from_input_gacode(_IG.from_file(str(GACODE)))
    m0 = minuet(gq, profiles=kin,
                settings=Settings(t_end=0.02, evolve_equilibrium=True,
                                  geqdsk_boundary_psin=BOUNDARY_PSIN,
                                  # minuet defaults bootstrap+sawteeth ON; this is a
                                  # 0.02 s GS-CONSISTENCY solve whose only job is to
                                  # emit a clean source file, so no redistribution
                                  # and no bootstrap should touch it
                                  bootstrap=None, sawtooth=None,
                                  diffusion=DiffusionSettings(n_save=3)))
    m0.run(verbose=False)
    m0.export_input_gacode(str(bench), keep_kinetics=str(GACODE))
    return bench

# transp_soft beat timings (template defaults): current diffusion switches on
# at transition_window + currentheating_window — MINUET's t=0
TIME_DIFFUSION = 0.1 + 0.001

C_T, C_M = "#3f7cac", "#d1495b"         # TRANSP blue, MINUET red



# TRANSP radial zones: the template's transp_soft default (60) is too coarse
# here -- TRANSP's built-in ~1-2-zone bootstrap smoothing eats the pedestal
# jbs peak at that resolution (verified by a resolution scan). Pinned at 200.
NZONES = 200

# =========================== TRANSP-PARITY SETTINGS ==========================
# COMPLETE list of the MINUET NON-DEFAULTS this benchmark needs to match
# TRANSP. MINUET's defaults stay physical / paper-faithful; every value here
# was identified EMPIRICALLY on TRANSP's own output (CDF variables, namelist
# it actually ran with) and verified quantitatively -- see the per-knob
# docstrings in minuet for the full evidence. Everything not listed here runs
# at the MINUET default.
#
# --- resistivity (SauterResistivity) ---
# ln-Lambda: TRANSP's RESISTIVITY chain uses its Zeff-dependent CLOGE
# convention (matched to machine precision); its BOOTSTRAP uses Sauter's own
# 31.3 formula = minuet's default -> match module by module, not code-wide.
RESIST_COULOMB_LOG = "transp"            # default 'sauter'
# trapped fraction: TRANSP's eta chain uses the shaping-blind analytic
# epsilon formula (never Lin-Liu-upgraded, unlike its bootstrap):
# +3.6% ft -> +7% eta at the pedestal, exactly the measured band
RESIST_TRAPPED_FRACTION = "circular"     # default 'exact'
# (its N(Z) numerator is a 0.76 variant of Sauter's published 0.74 -- MINUET
#  always uses the published 0.74: a deliberate, documented ~1%-flat eta
#  difference, see the known-differences list)
#
# --- bootstrap (SauterBootstrap) ---
# TRANSP's discrete chain (one-zone stencils + current filters) acts like a
# ~2-zone sliding triangular average (kernel fit at nzones 60 and 200) with
# ZERO-PADDED edge behavior (its pedestal-edge CURBS follows a zero-padded
# kernel of the raw evaluation, not an edge-renormalized one)
BS_SMOOTH_X = 2.0 / NZONES               # default None (no smoothing)
BS_SMOOTH_BND = "zero"                   # default 'renorm'
#
# --- sawtooth trigger (PorcelliSawtooth) ---
# q=1 shear read on the INNER (shelf) side of the crossing over a finite
# stencil: TRANSP's exported trigger shear dips to ~0.09 right after each
# crash (the shelf-side slope); via the 1/s_norm factor on every dW term
# this drives its deep, fast -dW reset cycle
SAW_S1_HALFWIDTH = 0.02                  # default None (pointwise slope)
SAW_S1_SIDE = "inner"                    # default 'centered'
# the Bp fed to the trigger: TRANSP's "outboard midplane" bpol array is
# EMPIRICALLY the perimeter-averaged (Ampere) field -- its exported BPOL
# matches mu0 I_enc / L_pol to 1% at the q=1 surface
SAW_BP1_CONVENTION = "perimeter"         # default 'fsa'
# (eq13=True and q1_crossing='outer' are ALSO required for parity but are
#  now MINUET defaults -- passed explicitly below only for the record)
#
# --- sawtooth redistribution (PorcelliReconnection) ---
# island width = fporcelli * x_q1 (its measured convention; the namelist
# fporcelli = 0.63), NOT a fraction of the Kadomtsev mixing radius
MIX_WIDTH_CONVENTION = "x_q1"            # default 'x_mix' (paper)
# the axial region RECONNECTS outward (new axis inherits the helical flux
# of ~the mixing-radius surface): reproduces TRANSP's measured post-crash
# flat core to 4 sig figs; the paper's helicity-conserving Taylor core is
# ~1.5% (in q) below it
MIX_CORE = "reconnect"                   # default 'taylor' (paper eq 24)
# axis anchor sits ~1.5 of ITS radial zones inside the psi*-return radius
# (its discrete mixing-boundary determination; fitted, 0.2% rms in core q)
MIX_CORE_ANCHOR_INSET = 1.5 / NZONES     # default 0.0
#
# ALSO required for parity but now MINUET DEFAULTS (benchmark-identified
# values promoted to defaults): island_fraction = 0.63, current_sheet_width
# = 0.025, DiffusionSettings.ip_match_buffer = 0.05, PorcelliSawtooth
# eq13 = True and q1_crossing = 'outer'.
# =============================================================================

# ONE boundary for the whole benchmark: the traced psi_N=0.95 surface of the
# geqdsk (minuet Settings.geqdsk_boundary_psin). Deep enough off the
# separatrix that the MXH fit is smooth and BOTH consumers ingest the same
# file cleanly (0.995/0.99 still have near-X flux-bunched tips: TRDAT kink
# aborts, MINUET re-read tracer/Picard failures -- measured). The benchmark
# plasma is "the PRD core inside psi_N=0.95": ~identical CD physics, exactly
# shared boundary.
BOUNDARY_PSIN = 0.95


def run_benchmark(sawteeth, cold_start=False, flattop_window=10.0, show=True,
                  transp_overrides=None, tag_extra="", n_save=401):
    """transp_overrides: dict merged into the transp_soft parameters_prepare
    (flat TRANSP-namelist keys, e.g. dtOut_ms / dtCurrentDiffusion_ms).
    tag_extra: appended to the scratch-folder tag so variants (e.g. '_hires')
    live next to the base runs. n_save: MINUET saved frames over the window."""
    tag = ("sawteeth" if sawteeth else "no_sawteeth") + f"_geq_nz{NZONES}" + tag_extra
    folder = __mitimroot__ / "tests" / "scratch" / f"dev_minuet_transp_{tag}"
    if cold_start and folder.exists():
        IOtools.shutil_rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------------
    # 1. TRANSP: a genuine MAESTRO chain with a single transp_soft beat
    # -------------------------------------------------------------------------------
    nml = IOtools.read_mitim_yaml(TEMPLATE)

    # Initialize both equilibrium and profiles from the input.gacode, verbatim
    nml["plasma"]["profiles_initialization"]["initialization_type"] = "profiles"
    nml["plasma"]["profiles_initialization"]["creator_type"] = None
    bench_source = _build_bench_source()
    nml["plasma"]["profiles_initialization"]["parameters"]["profiles_file"] = str(bench_source)
    nml["plasma"]["parameters"]["Bt"] = None
    nml["plasma"]["parameters"]["Ip"] = None
    # The bench LCFS is already the smooth psi_N=0.995 surface (BOUNDARY_PSIN,
    # baked in by the prep) -> TRANSP takes it as-is: SAME boundary as MINUET
    nml["plasma"]["parameters"]["separatrix"]["boundary_surface_psin"] = 1.0

    nml["maestro"]["beats"] = ["transp_soft"]

    pp = nml["maestro"]["transp_soft"]["parameters_prepare"]
    pp["flattop_window"] = flattop_window
    # MAESTRO launches SPARC AS the machine_initialization tokamak (benign
    # machine equilibrium at t=0, UFILES morph it into SPARC by 0.1 s)
    pp["machine_initialization"] = "D3D"
    pp["sawteeth"] = sawteeth
    if not sawteeth:
        pp["extract_at"] = "last"           # no sawtooth to anchor extraction on
    # MATCH the bootstrap model to MINUET's (NMLtools default is Hager)
    pp["bootstrap_model"] = "sauter"
    pp["nzones"] = NZONES
    if transp_overrides:
        pp.update(transp_overrides)

    namelist_file = folder / "namelist.maestro.yaml"
    IOtools.write_mitim_yaml(nml, namelist_file)

    run_maestro.run_maestro_local(
        namelist_file, folder=folder,
        terminal_outputs=True, force_cold_start=cold_start, cpus=4,
    )

    cdfs = [f for f in (folder / "Beats").glob("Beat_*/run_transp/*.CDF")
            if not f.name.endswith("PH.CDF")]
    assert len(cdfs) > 0, "No CDF found in the transp beat folder — did the beat finish?"
    cdf_file = max(cdfs, key=lambda f: f.stat().st_size)
    print(f"\t- Reading TRANSP output: {cdf_file}")
    c = CDFtools.transp_output(cdf_file)

    # -------------------------------------------------------------------------------
    # 2. TRANSP evolution on the aligned time base + the MINUET pressure drive
    # -------------------------------------------------------------------------------
    tT_all = c.t - TIME_DIFFUSION
    sel = tT_all >= 0.0
    tT = tT_all[sel]

    # Plain arrays for the pressure closure (NEVER close over `c`: it holds an
    # open netCDF handle and MINUET's save() dill-serializes closures)
    _tp = np.ascontiguousarray(tT)
    _xp = np.ascontiguousarray(c.x[sel])
    _pp = np.ascontiguousarray(c.p_kin[sel] * 1e6)      # total pressure [Pa]

    def p_of_x_t(t, _tp=_tp, _xp=_xp, _pp=_pp):
        """MINUET GS-pressure drive = TRANSP's own evolving total pressure."""
        k = int(np.argmin(np.abs(_tp - float(t))))
        xk, pk = _xp[k], _pp[k]
        return lambda xq: np.interp(np.asarray(xq, dtype=float), xk, pk)

    # -------------------------------------------------------------------------------
    # 3. MINUET: same file, matched physics, TRANSP's pressure evolution
    # -------------------------------------------------------------------------------
    minuet_file = folder / f"minuet_{tag}.minuet"
    timing_file = folder / "minuet_timing.json"
    fresh = cold_start or not minuet_file.exists()
    t_wall0 = time.time()
    # MINUET runs on TRANSP'S REALIZED BOUNDARY -- the exact contour of the
    # CDF's own moment surfaces (getFluxSurface at x=1): no MXH bottleneck in
    # MINUET's path at all (the new boundary-polygon source). Kinetics, q_init
    # and F_boundary come from the same bench file TRANSP consumed; Ip is
    # commanded to the file header (what TRANSP's CUR ufile enforces).
    from minuet import TabulatedKineticProfiles
    ig_b = InputGacode.from_file(str(bench_source))
    _kin = TabulatedKineticProfiles.from_input_gacode(ig_b)
    _rho_b = np.ascontiguousarray(ig_b.profiles["rho"])
    _q_b = np.ascontiguousarray(ig_b.profiles["q"])

    def _q_init(x, _r=_rho_b, _q=_q_b):
        return np.interp(np.asarray(x, dtype=float), _r, _q)

    F_bnd = float(ig_b.scalars["rcentr"] * ig_b.scalars["bcentr"])
    Ip_file_A = float(ig_b.scalars["current"]) * 1e6
    Rb_leg, Zb_leg = getFluxSurface(c.f, TIME_DIFFUSION, 1.0)
    m = minuet.cached(
        minuet_file, (np.asarray(Rb_leg[::4]), np.asarray(Zb_leg[::4])),
        profiles=_kin, q_init=_q_init, F_boundary=F_bnd,
        cold_start=cold_start,
        settings=Settings(
            t_end=flattop_window,
            evolve_equilibrium=True,
            # ALL TRANSP-parity non-defaults come from the module-level
            # TRANSP-PARITY SETTINGS block (top of this file), which lists
            # each one with its evidence. eq13 / q1_crossing='outer' are
            # minuet defaults, passed explicitly only for the record.
            resistivity=SauterResistivity(
                coulomb_log=RESIST_COULOMB_LOG,
                trapped_fraction=RESIST_TRAPPED_FRACTION),
            bootstrap=SauterBootstrap(smooth_x=BS_SMOOTH_X,
                                      smooth_boundary=BS_SMOOTH_BND),
            sawtooth=(PorcelliSawtooth(eq13=True, q1_crossing="outer",
                                       s1_halfwidth=SAW_S1_HALFWIDTH,
                                       s1_side=SAW_S1_SIDE,
                                       bp1_convention=SAW_BP1_CONVENTION,
                                       redistribution=PorcelliReconnection(
                                           width_convention=MIX_WIDTH_CONVENTION,
                                           core=MIX_CORE,
                                           core_anchor_inset=MIX_CORE_ANCHOR_INSET))
                      if sawteeth else None),
            p_of_x_t=p_of_x_t,
            diffusion=DiffusionSettings(n_save=n_save, Ip=Ip_file_A),
        ),
    )
    if fresh:
        timing_file.write_text(json.dumps({"minuet_wall_s": time.time() - t_wall0}))
    minuet_wall_s = json.loads(timing_file.read_text())["minuet_wall_s"] \
        if timing_file.exists() else np.nan
    res = m.result

    # -------------------------------------------------------------------------------
    # 4. Aligned traces, crash records, metrics
    # -------------------------------------------------------------------------------
    # Core q at a SMALL FINITE radius (both codes' q(0) is an extrapolation that
    # is noisy right after crashes; q(x=0.10) is a solved value in both)
    X_CORE = 0.10
    q0_T = np.array([np.interp(X_CORE, c.xb[sel][k, :], c.q[sel][k, :])
                     for k in range(len(tT))])
    li3_T, vs_T, Ip_T = c.Li3[sel], c.Vsurf[sel], c.Ip[sel]

    tM = res.t
    q0_M = np.array([np.interp(X_CORE, res.x_q, res.q[k, :])
                     for k in range(len(tM))])
    li3_M = res.li3
    vs_M = res.v_loop[:, -1]
    Ip_M = res.ip_enc[:, -1] / 1e6

    # Crash times from each code's OWN record
    crashes_T = np.asarray(getattr(c, "tlastsawU", []), dtype=float) - TIME_DIFFUSION
    crashes_T = crashes_T[(crashes_T >= 0.0) & (crashes_T <= tT[-1])]
    crashes_M = np.asarray(m.history.get("crashes", []), dtype=float)

    # Final q(x) on a common grid (TRANSP xb == MINUET x, sqrt norm. tor. flux)
    x_common = np.linspace(0.1, 0.9, 81)
    q_T_end = np.interp(x_common, c.xb[-1, :], c.q[-1, :])
    q_M_end = np.interp(x_common, res.x_q, res.q[-1, :])
    q_rms = float(np.sqrt(np.mean(((q_T_end - q_M_end) / q_T_end) ** 2)))

    # Boundary agreement [mm]: polar radius about a common center vs angle
    RbT, ZbT = getFluxSurface(c.f, c.t[-1], 1.0)
    RbM, ZbM = m.geom_last.boundary
    Rc, Zc = float(np.mean(RbT)), float(np.mean(ZbT))
    th = np.linspace(-np.pi, np.pi, 512, endpoint=False)

    def _rho_of(Rb, Zb):
        a = np.arctan2(np.asarray(Zb) - Zc, np.asarray(Rb) - Rc)
        r = np.hypot(np.asarray(Rb) - Rc, np.asarray(Zb) - Zc)
        o = np.argsort(a)
        return np.interp(th, a[o], r[o], period=2 * np.pi)

    drho_mm = 1e3 * np.abs(_rho_of(RbT, ZbT) - _rho_of(RbM, ZbM))
    bnd_mean_mm, bnd_max_mm = float(drho_mm.mean()), float(drho_mm.max())

    # Timings: TRANSP physics-loop CPU = SUM of the per-step CPTIM trace
    # (the beat's ~10 min wall-clock is staging/container/SLURM overhead, not
    # physics); MINUET wall time is recorded on the fresh cached() run
    transp_cpu_s = float(np.sum(c.cptim)) if hasattr(c, "cptim") else np.nan

    dli_T, dli_M = li3_T[-1] - li3_T[0], li3_M[-1] - li3_M[0]

    def _period(tc):
        return np.mean(np.diff(tc)) if len(tc) > 1 else np.nan

    def _fmt_t(s):
        if not np.isfinite(s):
            return "n/a"
        return f"{s:.1f} s" if s < 120 else f"{s/60:.1f} min"

    # total bootstrap current fraction at t=end, SAME convention on both
    # sides: both jbs profiles integrated with TRANSP's own zone areas
    # (DAREA), so the row compares the profiles and nothing else. (An earlier
    # version mapped MINUET's <J.B>/B0 to a toroidal current with
    # V' dx/(2 pi Rgeo) -- that approximation understates I_bs by ~3.6% here
    # and faked a bootstrap deficit that band-by-band profile comparison
    # refutes; TRANSP's own CURBS*DAREA sum reproduces its IpB exactly.)
    geomL = m.geom_last
    fbs_T = fbs_M = np.nan
    try:
        from minuet.diffusion import CurrentDiffusion as _CD4, \
            DiffusionSettings as _DS4
        k1_ = len(res.t) - 1
        _cdb = _CD4(geomL, res.profiles, psi_init=res.psi[k1_],
                    bootstrap=SauterBootstrap(smooth_x=BS_SMOOTH_X, smooth_boundary=BS_SMOOTH_BND),
                    settings=_DS4(Ip=float(res.ip_enc[k1_, -1])),
                    t0=float(res.t[k1_]))
        j_bs = _cdb.jb_bootstrap(res.psi[k1_], 0.0) / geomL.B0     # A/m^2
        xT_ = c.x[sel][-1, :]
        darea_ = np.asarray(c.f["DAREA"][:], float)[-1] * 1e-4  # m^2
        jT_ = np.asarray(c.f["CURBS"][:], float)[-1] * 1e4      # A/m^2
        j_bs_onT = np.interp(xT_, res.x_c, j_bs)
        fbs_M = float(np.sum(j_bs_onT * darea_) / (Ip_M[-1] * 1e6))
        fbs_T = float(np.sum(jT_ * darea_) / (Ip_T[-1] * 1e6))
    except Exception as e:
        print(f"\t- bootstrap-fraction row skipped: {e}")

    def _row(label, vT, vM, fmt="{:.3f}", pct=True):
        sT, sM = fmt.format(vT), fmt.format(vM)
        rel = (f"{100 * (vM - vT) / vT:+10.2f}" if pct and np.isfinite(vT)
               and np.isfinite(vM) and abs(vT) > 1e-12 else f"{'--':>10}")
        print(f"  {label:<38} {sT:>12} {sM:>12} {rel}")

    print("\n" + "=" * 78)
    print(f" MINUET vs TRANSP current diffusion (SPARC PRD, {tag.replace('_', ' ')})")
    print("=" * 78)
    print(f"  {'quantity':<38} {'TRANSP':>12} {'MINUET':>12} {'M/T-1 [%]':>10}")
    _row(f"q(x={X_CORE}) start", q0_T[0], q0_M[0])
    _row(f"q(x={X_CORE}) end", q0_T[-1], q0_M[-1])
    if sawteeth:
        _row("sawtooth crashes", len(crashes_T), len(crashes_M), fmt="{:d}")
        _row("mean sawtooth period [ms]", 1e3 * _period(crashes_T),
             1e3 * _period(crashes_M), fmt="{:.0f}")
    _row("li3 start", li3_T[0], li3_M[0])
    _row("delta li3 over flattop", dli_T, dli_M, pct=False)
    _row("V_surf at end [V]", vs_T[-1], vs_M[-1])
    _row("Ip at end [MA]", Ip_T[-1], Ip_M[-1])
    _row("bootstrap fraction I_bs/Ip at end", fbs_T, fbs_M)
    print(f"  {'physics compute time (1 core)':<38} {_fmt_t(transp_cpu_s):>12} {_fmt_t(minuet_wall_s):>12} {'--':>10}")
    print(f"  {'  (TRANSP: sum of CPTIM; beat wall-clock is staging/container)':<70}")
    print(f"  q(x) rel. rms difference at end (0.1 < x < 0.9): {q_rms * 100:.2f}%")
    print(f"  boundary agreement: mean {bnd_mean_mm:.1f} mm, max {bnd_max_mm:.1f} mm")

    consistent = (
        q_rms < 0.10
        and abs(dli_T - dli_M) < 0.10
        and (len(crashes_T) > 0) == (len(crashes_M) > 0)
    )
    print(f"\n  --> {'BENCHMARK CONSISTENT' if consistent else 'BENCHMARK DISCREPANT (inspect the notebook)'}")
    print("=" * 76 + "\n")

    # -------------------------------------------------------------------------------
    # 5. FigureNotebook
    # -------------------------------------------------------------------------------
    times_cmp = np.linspace(0.0, min(tT[-1], tM[-1]), 5)
    cmap = plt.get_cmap("viridis")

    def _iT(t):
        return int(np.argmin(np.abs(tT - t)))

    def _iM(t):
        return int(np.argmin(np.abs(tM - t)))

    fn = FigureNotebook(f"MINUET vs TRANSP — SPARC PRD current diffusion ({tag.replace('_', ' ')})")

    # ------------------------------------------------------------- Time traces
    fig = fn.add_figure(label="Time traces")
    fig.set_layout_engine("constrained")
    axs = fig.subplots(2, 3)
    ax = axs[0, 0]
    ax.plot(tT, q0_T, color=C_T, lw=1.2, label="TRANSP")
    ax.plot(tM, q0_M, color=C_M, lw=1.0, ls="--", label="MINUET")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel(f"q(x={X_CORE})"); ax.set_ylim(bottom=0)
    ax.legend(); ax.set_title("core safety factor")

    ax = axs[0, 1]
    ax.plot(tT, li3_T, color=C_T, lw=1.6, label="TRANSP LI_3")
    ax.plot(tM, li3_M, color=C_M, lw=1.3, ls="--", label="MINUET li3")
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("li(3)")
    ax.legend(); ax.set_title("internal inductance")

    ax = axs[0, 2]
    ax.plot(tT, vs_T, color=C_T, lw=1.6, label="TRANSP VSURC")
    # mask MINUET frames adjacent to GS refreshes (and sawtooth cuts): the
    # stitched dpsi/dt across a metric swap is an operator-splitting artifact,
    # not physics (MINUET's own notebook masks these frames too)
    t_ref = np.asarray(m.history.get("t", []), dtype=float)
    dtf = float(np.median(np.diff(tM)))
    okM = np.ones(len(tM), dtype=bool)
    for tr in t_ref:
        okM &= np.abs(tM - tr) > 1.6 * dtf
    ax.plot(tM, np.where(okM, vs_M, np.nan), color=C_M, lw=1.3, ls="--",
            label="MINUET V_loop(edge)")
    lo, hi = min(0.0, vs_T.min()), vs_T.max()
    pad = 0.5 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("V [V]")
    ax.legend(); ax.set_title("surface loop voltage (refresh frames masked)")

    ax = axs[1, 0]
    ax.plot(tT, Ip_T, color=C_T, lw=1.6, label="TRANSP PCUR")
    ax.plot(tM, Ip_M, color=C_M, lw=1.3, ls="--", label="MINUET Ip_enc(edge)")
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("Ip [MA]")
    ax.legend(); ax.set_title("plasma current (input, must match)")

    ax = axs[1, 1]
    pT_core = np.array([np.interp(X_CORE, c.x[sel][k, :], c.p_kin[sel][k, :])
                        for k in range(len(tT))])
    pM_core = np.array([float(p_of_x_t(t)(X_CORE)) / 1e6 for t in tM])
    ax.plot(tT, pT_core, color=C_T, lw=1.6, label="TRANSP PTOWB")
    ax.plot(tM, pM_core, color=C_M, lw=1.3, ls="--", label="MINUET GS drive")
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel(f"p(x={X_CORE}) [MPa]")
    ax.legend(); ax.set_title("pressure (MINUET commanded = TRANSP)")

    ax = axs[1, 2]
    if sawteeth:
        for tc in crashes_T:
            ax.axvline(tc, color=C_T, lw=1.4, alpha=0.9, ymin=0.55, ymax=1.0)
        for tc in crashes_M:
            ax.axvline(tc, color=C_M, lw=1.4, alpha=0.9, ymin=0.0, ymax=0.45)
        ax.text(0.02, 0.97, f"TRANSP: {len(crashes_T)} crashes", color=C_T,
                transform=ax.transAxes, va="top")
        ax.text(0.02, 0.08, f"MINUET: {len(crashes_M)} crashes", color=C_M,
                transform=ax.transAxes, va="bottom")
        ax.set_xlim(0, max(tT[-1], tM[-1]))
        ax.set_title("sawtooth crash times")
    else:
        ax.text(0.5, 0.5, "sawteeth disabled in BOTH codes\n"
                          "(beat knob sawteeth=false; no MINUET trigger armed)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("sawteeth: off")
    ax.set_xlabel("t - t_CD [s]"); ax.set_yticks([]); ax.grid(False)

    # ------------------------------------------------------------- q profiles
    fig = fn.add_figure(label="q profiles")
    fig.set_layout_engine("constrained")
    axs = fig.subplots(1, 3)
    ax = axs[0]
    for i, tc in enumerate(times_cmp):
        col = cmap(i / max(len(times_cmp) - 1, 1))
        ax.plot(c.xb[sel][_iT(tc), :], c.q[sel][_iT(tc), :], color=col, lw=1.8,
                marker="o", ms=1.5, mfc="none", markevery=1, label=f"t={tc:.1f} s")
        ax.plot(res.x_q, res.q[_iM(tc), :], color=col, lw=1.4, ls="--", marker=".", ms=2, markevery=1)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("x = sqrt(Phi/Phi_b)"); ax.set_ylabel("q"); ax.set_ylim(bottom=0)
    ax.legend(title="solid TRANSP / dashed MINUET")
    ax.set_title("q(x, t) family")

    ax = axs[1]
    ax.plot(c.xb[sel][-1, :], c.q[sel][-1, :], color=C_T, lw=2.0,
            marker="o", ms=1.5, mfc="none", markevery=1, label="TRANSP (end)")
    ax.plot(res.x_q, res.q[-1, :], color=C_M, lw=1.6, ls="--",
            marker=".", ms=2, markevery=1, label="MINUET (end)")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("x"); ax.set_ylabel("q"); ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(x_common, 100 * (q_M_end - q_T_end) / q_T_end, color="gray", lw=1.0)
    ax2.set_ylabel("MINUET-TRANSP [%]", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.legend(loc="upper left")
    ax.set_title(f"final q  (rel. rms {q_rms * 100:.2f}%)")

    ax = axs[2]
    tmid = 0.5 * min(tT[-1], tM[-1])
    win = (max(0.0, tmid - 1.5), tmid + 1.5)
    mskT = (tT >= win[0]) & (tT <= win[1])
    mskM = (tM >= win[0]) & (tM <= win[1])
    ax.plot(tT[mskT], q0_T[mskT], color=C_T, lw=1.4, marker=".", ms=3, label="TRANSP")
    ax.plot(tM[mskM], q0_M[mskM], color=C_M, lw=1.1, ls="--", marker=".", ms=3,
            label="MINUET")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel(f"q(x={X_CORE})")
    ax.legend(); ax.set_title("mid-flattop zoom")

    # --------------------------------------------------------------- Pressure
    fig = fn.add_figure(label="Pressure")
    fig.set_layout_engine("constrained")
    axs = fig.subplots(1, 3)
    ax = axs[0]
    for i, tc in enumerate(times_cmp):
        col = cmap(i / max(len(times_cmp) - 1, 1))
        ax.plot(c.x[sel][_iT(tc), :], c.p_kin[sel][_iT(tc), :], color=col, lw=1.8,
                marker="o", ms=1.5, mfc="none", markevery=1, label=f"t={tc:.1f} s")
        xg = np.linspace(0, 1, 101)
        ax.plot(xg, p_of_x_t(tc)(xg) / 1e6, color=col, lw=1.4, ls="--")
    ax.set_xlabel("x"); ax.set_ylabel("p [MPa]")
    ax.legend(title="solid TRANSP / dashed MINUET drive")
    ax.set_title("total pressure (PTOWB) family")

    ax = axs[1]
    ax.plot(c.x[sel][0, :], c.p_kin[sel][0, :], color=C_T, lw=2.0, marker="o", ms=1.5, mfc="none", markevery=1, label="t = 0")
    ax.plot(c.x[sel][-1, :], c.p_kin[sel][-1, :], color=C_M, lw=2.0, marker="o", ms=1.5, mfc="none", markevery=1, label="t = end")
    ax.set_xlabel("x"); ax.set_ylabel("p [MPa]")
    ax.legend()
    ax.set_title("TRANSP pressure: start vs end"
                 + (" (sawtooth mixing)" if sawteeth else ""))

    ax = axs[2]
    for xprobe, ls in [(0.1, "-"), (0.4, "--"), (0.8, ":")]:
        pTx = np.array([np.interp(xprobe, c.x[sel][k, :], c.p_kin[sel][k, :])
                        for k in range(len(tT))])
        ax.plot(tT, pTx, color=C_T, lw=1.3, ls=ls, label=f"x={xprobe}")
    ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("p [MPa]")
    ax.legend(); ax.set_title("TRANSP pressure traces (the MINUET GS drive)")

    # ------------------------------------------------- Boundary and surfaces
    fig = fn.add_figure(label="Boundary and surfaces")
    fig.set_layout_engine("constrained")
    # R-Z views: one full-height column EACH; metrics stacked in a wide third
    gs_ = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.9])
    ax_b = fig.add_subplot(gs_[:, 0])
    ax_s = fig.add_subplot(gs_[:, 1])
    ax_m = fig.add_subplot(gs_[0, 2])
    ax_p = fig.add_subplot(gs_[1, 2])

    # x of each stored surface, from the geometry's authoritative per-surface
    # psi_N (surfaces_psin) mapped onto (psin, x). Do NOT re-derive the storage
    # subsample rule: the geometry now also stores the outermost traced level,
    # so the stored set is no longer a plain round-linspace and reconstructing
    # it mislabeled every surface's x (TRANSP fetched at the wrong x -> the
    # "same x" surfaces looked spuriously shifted inward).
    geomL = m.geom_last
    x_surfL = np.interp(geomL.surfaces_psin, geomL.psin, geomL.x)

    ax = ax_b
    Rb0, Zb0 = getFluxSurface(c.f, TIME_DIFFUSION, 1.0)
    ax.plot(Rb0, Zb0, color=C_T, lw=2.0, label="TRANSP t=0")
    ax.plot(RbT, ZbT, color=C_T, lw=1.2, ls=":", label="TRANSP t=end")
    for geom, ls, lab in [(m.geom_first, "--", "MINUET t=0"),
                          (geomL, "-.", "MINUET t=end")]:
        Rb, Zb = geom.boundary
        ax.plot(Rb, Zb, color=C_M, lw=1.3, ls=ls, label=lab)
        ax.plot(geom.R_axis, geom.Z_axis, "+", color=C_M, ms=10, mew=1.5)
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.legend()
    ax.set_title(f"boundary (same LCFS)\nmismatch mean {bnd_mean_mm:.1f} / "
                 f"max {bnd_max_mm:.1f} mm")

    # Surfaces compared at IDENTICAL x: TRANSP contours are evaluated at the
    # exact x of each stored MINUET surface (plotting nearest-stored against
    # round numbers faked cm-level offsets: near the edge the stored spacing
    # is ~0.04 in x)
    in_range = (x_surfL >= 0.15) & (x_surfL <= 0.95)
    idx_cmp = np.where(in_range)[0][::2]          # every other stored surface
    ax = ax_s
    for k in idx_cmp:
        RT, ZT = getFluxSurface(c.f, c.t[-1], float(x_surfL[k]))
        ax.plot(RT, ZT, color=C_T, lw=1.4)
        RM, ZM = geomL.surfaces[k]
        ax.plot(np.append(RM, RM[0]), np.append(ZM, ZM[0]), color=C_M,
                lw=1.1, ls="--")
    ax.plot(RbT, ZbT, color="gray", lw=1.0)
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title("surfaces at t=end, same x\n(solid TRANSP / dashed MINUET)")

    # Per-surface shape metrics on the full stored set (identical x by
    # construction): elongation and outboard-midplane radius offset
    kap_T, kap_M, dRout_mm, x_met = [], [], [], []
    for k in np.where(in_range)[0]:
        RT, ZT = getFluxSurface(c.f, c.t[-1], float(x_surfL[k]))
        RM, ZM = geomL.surfaces[k]
        kap_T.append((ZT.max() - ZT.min()) / (RT.max() - RT.min()))
        kap_M.append((ZM.max() - ZM.min()) / (RM.max() - RM.min()))
        dRout_mm.append(1e3 * (RT.max() - RM.max()))
        x_met.append(float(x_surfL[k]))
    x_met = np.asarray(x_met)

    ax = ax_m
    ax.plot(x_met, kap_T, "o-", color=C_T, ms=4, label="TRANSP")
    ax.plot(x_met, kap_M, "s--", color=C_M, ms=4, label="MINUET")
    ax.set_xlabel("x"); ax.set_ylabel("elongation (Zmax-Zmin)/(Rmax-Rmin)")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x_met, dRout_mm, color="gray", lw=1.2, marker=".", ms=4)
    ax2.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax2.set_ylabel("Rout(TRANSP) - Rout(MINUET) [mm]", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_title("surface-shape metrics at t=end (same-x pairs)")

    # psi(x): the integrated CD state -- differences here accumulate the q
    # mismatch. TRANSP PLFLX and MINUET psi are both Wb/rad, gauge psi(axis)=0
    ax = ax_p
    psi_T = c.psi[sel][-1, :] - c.psi[sel][-1, 0]
    ax.plot(c.x[sel][-1, :], psi_T, color=C_T, lw=1.8,
            marker="o", ms=1.5, mfc="none", label="TRANSP PLFLX (t=end)")
    ax.plot(res.x_c, res.psi[-1, :] - res.psi[-1, 0], color=C_M, lw=1.4,
            ls="--", marker=".", ms=2, label="MINUET psi (t=end)")
    ax.set_xlabel("x"); ax.set_ylabel("psi - psi_axis [Wb/rad]")
    ax.legend()
    dpsi_b = float(np.interp(1.0, c.x[sel][-1, :], psi_T)
                   - (res.psi[-1, -1] - res.psi[-1, 0]))
    ax.set_title(f"poloidal flux vs x (edge difference {dpsi_b:+.3f} Wb/rad)")

    # ------------------------------------------------ Resistivity and currents
    fig = fn.add_figure(label="Resistivity and currents")
    fig.set_layout_engine("constrained")
    axs = fig.subplots(1, 3)
    ax = axs[0]
    for i, tc in enumerate([times_cmp[0], times_cmp[-1]]):
        col = cmap(0.15 + 0.7 * i)
        ax.semilogy(c.x[sel][_iT(tc), :], c.eta[sel][_iT(tc), :], color=col, lw=1.8,
                    marker="o", ms=1.5, mfc="none", markevery=1, label=f"t={tc:.1f} s")
        ax.semilogy(res.x_c, res.eta[_iM(tc), :], color=col, lw=1.4, ls="--", marker=".", ms=2, markevery=1)
    ax.set_xlabel("x"); ax.set_ylabel("eta_par [Ohm m]")
    ax.legend(title="solid TRANSP / dashed MINUET")
    ax.set_title("parallel (Sauter) resistivity")

    ax = axs[1]
    for i, tc in enumerate([times_cmp[0], times_cmp[-1]]):
        col = cmap(0.15 + 0.7 * i)
        ax.plot(c.x[sel][_iT(tc), :], c.j[sel][_iT(tc), :], color=col, lw=1.8,
                marker="o", ms=1.5, mfc="none", markevery=1, label=f"t={tc:.1f} s")
        ax.plot(res.x_c, res.j_par[_iM(tc), :] / 1e6, color=col, lw=1.4, ls="--", marker=".", ms=2, markevery=1)
    ax.set_xlabel("x"); ax.set_ylabel("j [MA/m^2]")
    ax.legend(title="solid TRANSP CUR / dashed MINUET <J.B>/B0\n"
                                "(flux-average conventions differ)")
    ax.set_title("parallel current density")

    ax = axs[2]
    ax.plot(c.x[sel][-1, :], c.jB[sel][-1, :], color=C_T, lw=1.8,
            marker="o", ms=1.5, mfc="none", markevery=1, label="TRANSP jB (Sauter)")
    ax.plot(c.x[sel][-1, :], c.jOh[sel][-1, :], color=C_T, lw=1.2, ls=":",
            marker="o", ms=1.5, mfc="none", markevery=1, label="TRANSP jOh")
    # MINUET's Sauter bootstrap at t=end, recomputed with the same recipe the
    # Poynting tab uses (CurrentDiffusion rebuild at the final state)
    jbs_M = None
    try:
        from minuet.diffusion import CurrentDiffusion as _CD, DiffusionSettings as _DS
        k1 = len(res.t) - 1

        def _jbs_with(bs_model):
            cdx = _CD(geomL, res.profiles, psi_init=res.psi[k1],
                      bootstrap=bs_model,
                      settings=_DS(Ip=float(res.ip_enc[k1, -1])),
                      t0=float(res.t[k1]))
            return cdx.jb_bootstrap(res.psi[k1], 0.0) / geomL.B0 / 1e6

        jbs_M = _jbs_with(SauterBootstrap(smooth_x=BS_SMOOTH_X, smooth_boundary=BS_SMOOTH_BND))   # as run
        jbs_raw = _jbs_with(SauterBootstrap())                     # physical
        ax.plot(res.x_c, jbs_M, color=C_M, lw=1.6, ls="--",
                marker=".", ms=2, markevery=1,
                label=f"MINUET jbs as run (smooth_x={BS_SMOOTH_X:.3f})")
        ax.plot(res.x_c, jbs_raw, color=C_M, lw=0.9, ls=":",
                label="MINUET jbs unsmoothed (physical)")
    except Exception as e:
        print(f"\t- MINUET bootstrap overlay skipped: {e}")
    ax.set_xlabel("x"); ax.set_ylabel("j [MA/m^2]")
    ax.legend()
    ax.set_title("bootstrap + ohmic at t=end (SAME Sauter model;\nMINUET run smoothing matched to TRANSP zone width)")

    # ---------------------------------------------------------------- Sawteeth
    if sawteeth:
        fig = fn.add_figure(label="Sawteeth")
        fig.set_layout_engine("constrained")
        axs = fig.subplots(1, 3)
        ax = axs[0]
        if len(crashes_T) > 1:
            ax.plot(crashes_T[1:], 1e3 * np.diff(crashes_T), "o-", color=C_T, ms=4,
                    label="TRANSP")
        if len(crashes_M) > 1:
            ax.plot(crashes_M[1:], 1e3 * np.diff(crashes_M), "s--", color=C_M, ms=4,
                    label="MINUET")
        ax.set_xlabel("crash time [s]"); ax.set_ylabel("inter-crash period [ms]")
        ax.set_ylim(bottom=0); ax.legend(); ax.set_title("sawtooth period evolution")

        ax = axs[1]
        ax.plot(tT, q0_T, color=C_T, lw=1.2, label="TRANSP")
        ax.plot(tM, q0_M, color=C_M, lw=1.0, ls="--", label="MINUET")
        for tc in crashes_T:
            ax.axvline(tc, color=C_T, lw=0.5, alpha=0.3)
        for tc in crashes_M:
            ax.axvline(tc, color=C_M, lw=0.5, alpha=0.3)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel(f"q(x={X_CORE})")
        ax.legend(); ax.set_title("core q with recorded crash times")

        # pre/post-crash q of the LAST crash in each code. TRANSP writes an
        # EVENT FRAME PAIR at each crash: the frame AT the crash time holds
        # the pre-crash state, the one 0.1 ms later the instantaneous
        # post-crash state -- and its crash then COMPLETES over its next few
        # internal steps (hires run, dtmaxb = 2 ms: complete within ~3 ms,
        # core +0.008 instantaneous / +0.019 total; at the transp_soft
        # default dtmaxb = 0.1 s the same completion smears over ~40 ms), so
        # a third, slightly later curve shows the full sawtooth signature.
        # MINUET: the redistribution's own fine-grid record (exact crash
        # instant, already complete). NOTE the completed TRANSP core lands
        # ~0.5% (in q) ABOVE the helicity-conserving Taylor value q0f --
        # the one remaining mixing difference (see module docstring).
        ax = axs[2]
        tc_T = crashes_T[crashes_T < tT[-1] - 0.05][-1] if len(crashes_T) else None
        if tc_T is not None:
            kb_ = int(np.searchsorted(tT, tc_T - 1e-6))       # frame AT tc: pre
            ka_ = min(kb_ + 1, len(tT) - 1)                   # +0.1 ms: post
            # sample the completed crash as EARLY as the frame cadence
            # allows: the shelf-bounding current sheets are the fastest-
            # diffusing structures, so a late frame shows an already-eroded
            # (narrower) q = 1 island and fakes a mixing-extent mismatch
            # (measured: at true completion, +3 ms, the two codes' flat
            # q = 1 extents agree to 0.5% of x; at +38 ms TRANSP's has
            # visibly shrunk from both sides)
            kr_ = min(int(np.searchsorted(tT, tc_T + 0.012)), len(tT) - 1)
            ax.plot(c.xb[sel][kb_, :], c.q[sel][kb_, :], color=C_T, lw=1.6,
                    label=f"TRANSP before (t={tT[kb_]:.3f})")
            ax.plot(c.xb[sel][ka_, :], c.q[sel][ka_, :], color=C_T, lw=1.2,
                    ls=":", label="TRANSP instant after (+0.1 ms)")
            ax.plot(c.xb[sel][kr_, :], c.q[sel][kr_, :], color=C_T, lw=1.0,
                    ls="-.", label=f"TRANSP +{1e3 * (tT[kr_] - tc_T):.0f} ms "
                                   "(crash completes)")
        lc = m.history.get("last_crash") or {}
        if lc.get("crashed") and "q_before" in lc:
            ax.plot(lc["xx"], lc["q_before"], color=C_M, lw=1.4, ls="--",
                    label=f"MINUET before (t={crashes_M[-1]:.2f})")
            ax.plot(lc["xx"], lc["q_after"], color=C_M, lw=1.0, ls="-.",
                    label="MINUET after")
            for xv, lb in [(lc.get("x1"), "x1"), (lc.get("x_mix"), "x_mix")]:
                if xv:
                    ax.axvline(xv, color=C_M, lw=0.6, alpha=0.4)
                    ax.text(xv, ax.get_ylim()[0], f" {lb}", color=C_M,
                            fontsize="x-small", va="bottom")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(0, 0.9); ax.set_ylim(0.7, 1.6)
        ax.set_xlabel("x"); ax.set_ylabel("q")
        ax.legend(); ax.set_title("last crash: q before/after")

    # --------------------------------------------------------- Porcelli trigger
    # Term-by-term comparison of the SAME Porcelli conditions in both codes:
    # TRANSP exports its trigger terms (PORC13/14/15* on the CDF time base),
    # MINUET records the same quantities per save in history['porcelli'].
    # Eq 13/14 share the LHS (-dW); Eq 13's RHS is 0 with no fast ions (the
    # route that actually fires TRANSP here), Eq 14's is 0.5 w*i tau_A.
    # Eq 15b (w*i < c_star gamma_rho) gates the resistive route: it is never
    # satisfied in EITHER code for this plasma (w*i ~ 2x c_star gamma_rho).
    # Post-crash -dW shapes: MINUET shows a one-save NEEDLE (margin evaluated
    # on the sheet-fresh profile: the local s1 at the new q=1 crossing dips,
    # and every dW term carries 1/s_norm), healing within ~0.05 s to the
    # controlling reset level; TRANSP's dip is smooth/shallower because its
    # crash completes over ~40 ms and its trigger uses a smoothed,
    # cycle-constant s1 (PORCDIAG1 ~ 0.32).
    if sawteeth and m.history.get("porcelli"):
        hp = m.history["porcelli"]
        tP = np.asarray(hp["t"])
        fig = fn.add_figure(label="Porcelli trigger")
        fig.set_layout_engine("constrained")
        axs = fig.subplots(1, 3)

        ax = axs[0]
        ax.plot(tT, np.asarray(c.f["PORC14L"][:], float)[sel], color=C_T,
                lw=1.4, label="-dW (TRANSP)")
        ax.plot(tP, np.asarray(hp["neg_dW"]), color=C_M, lw=1.2, ls="--",
                label="-dW (MINUET)")
        ax.plot(tT, np.asarray(c.f["PORC14R"][:], float)[sel], color=C_T,
                lw=1.0, ls=":", label="Eq14 RHS: 0.5 w*i tauA (TRANSP)")
        ax.plot(tP, np.asarray(hp["half_wsi_tauA"]), color=C_M, lw=1.0,
                ls=":", label="Eq14 RHS (MINUET)")
        ax.axhline(0.0, color="gray", lw=0.8,
                   label="Eq13 RHS (no fast ions) = 0")
        for tc in crashes_T:
            ax.axvline(tc, color=C_T, lw=0.5, alpha=0.25)
        for tc in crashes_M:
            ax.axvline(tc, color=C_M, lw=0.5, alpha=0.25)
        ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("normalized dW terms")
        ax.legend()
        ax.set_title("ideal m=1 drive vs thresholds\n(crash: -dW above a threshold)")

        ax = axs[1]
        ax.plot(tT, np.asarray(c.f["PORC15BL"][:], float)[sel], color=C_T,
                lw=1.4, label="w*i (TRANSP)")
        ax.plot(tP, np.asarray(hp["wsi"]), color=C_M, lw=1.2, ls="--",
                label="w*i (MINUET)")
        ax.plot(tT, np.asarray(c.f["PORC15BR"][:], float)[sel], color=C_T,
                lw=1.0, ls=":", label="c* gamma_rho (TRANSP)")
        ax.plot(tP, np.asarray(hp["c_star_grho"]), color=C_M, lw=1.0, ls=":",
                label="c* gamma_rho (MINUET)")
        ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("[1/s]")
        ax.set_ylim(bottom=0); ax.legend()
        ax.set_title("Eq 15b resistive gate: w*i < c* gamma_rho\n"
                     "(never satisfied here, in either code)")

        ax = axs[2]
        xq1_T = np.full(len(tT), np.nan)
        s1_T = np.full(len(tT), np.nan)
        for k in range(len(tT)):
            qk, xk = c.q[sel][k, :], c.xb[sel][k, :]
            cr = np.where(np.diff(np.sign(qk - 1.0)) != 0)[0]
            if cr.size:
                i = cr[-1]
                xq1_T[k] = xk[i] + (1.0 - qk[i]) / (qk[i + 1] - qk[i]) \
                    * (xk[i + 1] - xk[i])
                s1_T[k] = xq1_T[k] * np.gradient(qk, xk)[i]
        ax.plot(tT, xq1_T, color=C_T, lw=1.4, label="x(q=1) TRANSP")
        ax.plot(tP, np.asarray(hp["x_q1"]), color=C_M, lw=1.2, ls="--",
                label="x(q=1) MINUET")
        # NOTE: TRANSP's trigger normalizes dW with a SMOOTHED, nearly
        # cycle-constant shear (its PORCDIAG1 ~ 0.32 throughout), while its
        # actual q=1 profile shear sawtooths; MINUET's s1 is the local
        # instantaneous value, which dips inside the crash current sheet and
        # deepens the post-crash -dW reset (longer period; see benchmark
        # discussion).
        ax.plot(tT, s1_T, color=C_T, lw=0.9, ls="-.", alpha=0.7,
                label="s1 from Q profile, TRANSP")
        ax.plot(tP, np.asarray(hp["s1"]), color=C_M, lw=0.9, ls="-.",
                alpha=0.7, label="s1 (local), MINUET")
        ax.set_xlabel("t - t_CD [s]"); ax.set_ylabel("x(q=1), s1")
        ax.set_ylim(bottom=0); ax.legend()
        ax.set_title("q=1 surface location and shear")

    # ------------------------------------------------- Neoclassical diagnostics
    fig = fn.add_figure(label="Neoclassical diagnostics")
    fig.set_layout_engine("constrained")
    axs = fig.subplots(1, 3)
    import minuet.neoclassical as _neo
    xTe = c.x[sel][-1, :]
    Te_c = np.maximum(res.profiles.Te_eV(res.x_c), 1.0)
    ne_c = res.profiles.ne_m3(res.x_c)
    Zf_c = res.profiles.Zeff_x(res.x_c)
    eta_sp_Mc = 1.0 / _neo.spitzer_conductivity(ne_c, Te_c, Zf_c)

    ax = axs[0]
    ax.plot(xTe, c.eta[sel][-1, :] / c.etas_sp[sel][-1, :], color=C_T, lw=1.8,
            marker="o", ms=1.5, mfc="none", markevery=1, label="TRANSP ETA_USE/ETA_SP")
    ax.plot(res.x_c, res.eta[-1, :] / eta_sp_Mc, color=C_M, lw=1.4, ls="--",
            marker=".", ms=2, markevery=1, label="MINUET eta/eta_Spitzer")
    ax.set_xlabel("x"); ax.set_ylabel("neoclassical factor")
    ax.legend()
    ax.set_title("neoclassical resistivity enhancement\n"
                 "(isolates trapped fraction / collisionality; lnLambda cancels)")

    ax = axs[1]
    ax.plot(xTe, c.f["CLOGE"][:][sel][-1, :], color=C_T, lw=1.8, marker="o", ms=1.5, mfc="none", markevery=1, label="TRANSP CLOGE")
    ax.plot(res.x_c, _neo.coulomb_log_e(ne_c, Te_c), color=C_M, lw=1.4, ls="--",
            marker=".", ms=2, markevery=1, label="MINUET (Sauter-1999)")
    ax.set_xlabel("x"); ax.set_ylabel("ln Lambda_e")
    ax.legend()
    ax.set_title("Coulomb logarithm conventions (curves = the two formulas;\nthis MINUET run USES coulomb_log=transp)")

    ax = axs[2]
    try:
        if jbs_M is None:
            raise NameError("jbs_M")
        jbs_M_onT = np.interp(xTe, res.x_c, jbs_M)
        rj = c.jB[sel][-1, :] / np.where(np.abs(jbs_M_onT) > 0.05, jbs_M_onT, np.nan)
        ax.plot(xTe, rj, color="k", lw=1.6)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel("x"); ax.set_ylabel("jB_TRANSP / jbs_MINUET")
        ax.set_title("bootstrap ratio (SAME Sauter model)\n"
                     "departures only at x>0.85: edge-equilibrium representation")
    except NameError:
        ax.text(0.5, 0.5, "jbs_M unavailable", ha="center", va="center",
                transform=ax.transAxes)

    if show:
        fn.show()

    return {"c": c, "m": m, "fn": fn, "q_rms": q_rms,
            "crashes_T": crashes_T, "crashes_M": crashes_M,
            "bnd_mean_mm": bnd_mean_mm, "bnd_max_mm": bnd_max_mm,
            "transp_cpu_s": transp_cpu_s, "minuet_wall_s": minuet_wall_s}


if __name__ == "__main__":
    raise SystemExit("Run minuet_vs_transp_01_no_sawteeth.py or "
                     "minuet_vs_transp_02_sawteeth.py instead.")
