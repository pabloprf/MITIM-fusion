"""
DEV TEST: MINUET on the RAW SPARC double-null separatrix -- sharp-corner capture.

The minuet_vs_transp benchmarks deliberately sidestep the boundary-corner
question: they run on the traced psi_N=0.95 surface (BOUNDARY_PSIN), which is
round near the X-points, precisely because the raw SPARC separatrix polygon
has near-kinks at the two X-points (the geometry docstring records tracer /
Picard failures on MXH re-reads of 0.99-0.995 surfaces). This test faces the
corners directly:

  START from the SPARC PRD geqdsk (FreeGS double-null, tests/data), take its
  STORED separatrix polygon (rbbbs/zbbbs, X-point kinks included) as the
  fixed boundary, and ask -- as a function of GS resolution -- how well MINUET
  captures the sharp corners.

Three layers, each with its own figure tab and summary rows:

  1. BOUNDARY INGESTION (resolution-independent). MINUET represents the
     boundary as a periodic cubic spline rho(theta) about the polygon
     centroid (gs._boundary_spline). A cubic spline through a kink
     necessarily rounds it and can overshoot nearby. Measured: distance
     of the spline curve to the raw polygon vs theta (corner windows vs the
     rest), and the effective radius of curvature the spline assigns to each
     X-point corner (Menger curvature on a finely sampled curve) vs the
     polygon's own vertex scale.

  2. SOLVED FIELD vs THE FILE, resolution scan. For each (gs_ns, gs_ntheta)
     in the scan, reproduce the equilibrium from the file's own boundary +
     p' + FF' (minuet.verification.verify_gs_geqdsk -- the same check the
     verification suite runs once at 160x320) and compare against the file's
     own PSIRZ: flux-depth / Ip / interior-psi errors, q(psi_N) vs the file
     QPSI, and -- the corner-specific part -- near-edge flux contours
     (psi_N = 0.95 / 0.99 / 0.995) traced from BOTH fields about the SAME
     center, with the mismatch split into X-point corner windows vs the
     rest of the poloidal angle. Self-convergence of the psi_N = 0.995
     contour (each resolution vs the finest) isolates MINUET's own
     convergence from the file's 129x129 FreeGS accuracy floor.

  3. FULL COUPLED RUN at the highest scan resolution, raw separatrix, short
     flattop (the configuration the vs-TRANSP benchmark avoided): PRD
     kinetics, Sauter resistivity+bootstrap, evolving equilibrium -- does the
     whole trace/Picard/CD loop survive the corners at high resolution, and
     what does its own consistency report say?

Honest limitations:
  - The file PSIRZ is a 129x129 free-boundary FreeGS solution: agreement
    beyond ~1% of the flux depth (the verification-suite thresholds) is not
    expected of ANY resolution -- that is the file's floor, hence the
    self-convergence layer.
  - The stored rbbbs polygon is EFIT-style, numerically just inside the true
    separatrix; its "corners" are the polygon's sharpest vertices, not exact
    X-points (where the true boundary tangents would be discontinuous and
    q -> infinity). The test quantifies capture of the FILE's boundary as
    given, which is the contract of a fixed-boundary solver.
  - gs_ntheta sets the finest poloidal feature the mapped solution can carry
    (arc scale ~ 2 pi rho / ntheta ~ 20 mm at 256, ~7 mm at 768): corner
    capture is expected to be ingestion-limited (spline) at fine grids and
    grid-limited at coarse ones. The scan separates the two.
  - History (recorded so it is not rediscovered): the FIRST refined coupled
    run used ntheta=768 with UNIFORM 5 mm knots and showed Picard-floor
    warnings at anchoring (initial q-consistency 5.3e-2, core q(0) shifted
    0.934 -> 0.897), mm-scale dips on traced surfaces along the corner
    mapping rays, and an oscillatory magnetic shear (found by Pablo). All
    of it was ALIASING: the theta grid must sample the refined knots
    locally (ntheta >= 2 pi rho_knot / spacing ~ 1460 for 5 mm here);
    resampling was then made corner-localized, the [gs] Nyquist warning
    added, and at 256x2048 everything heals (initial q-consistency 9e-4,
    one outer iter, no warnings, clean surfaces/shear -- cap17 measures
    this). Raw-ingestion runs at moderate resolution never alias: a smooth
    fit through a coarse polygon has no sub-cell content.

Measured picture (2026-07-15 first run, log in tests/scratch): corner capture
is INGESTION-limited, not grid-limited -- the psi_N=0.995 corner-window
mismatch vs the file moves only 5.9 -> 4.9 mm max across the plain scan
(off-corner 1.0 mm, at the file's own 129x129 floor), while MINUET's
self-convergence at the corners collapses 1.4 -> 0.3 mm; the raw spline
rounds the X-point kinks to r_curv ~ 24 / 37 mm regardless of resolution.
That conclusion drove the boundary-ingestion knobs added the same day
(Settings.gs_boundary_refine_mm + gs_boundary_interp='pchip': resample the
polygon along its own edges to a max knot spacing, fit with a
shape-preserving C1 interpolant that cannot ring). The scan's last case and
the coupled run exercise them: ingestion error alone drops 12.0 -> 0.36 mm
max (measured standalone; spline ringing gone, corner radius down to the
knot scale).

Run:  ./run_with_env.sh python MITIM-fusion/tests/dev_tests/minuet_geqdsk_sharp_boundary.py
      [--no-notebook] (save PNGs to scratch instead of opening the GUI)
      [--cold]        (rerun the coupled case even if cached)
"""

import sys
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.GUItools import FigureNotebook

from minuet import (GEQDSK, InputGacode, TabulatedKineticProfiles,
                    minuet, Settings, DiffusionSettings, SauterBootstrap)
from minuet.gs import _boundary_spline
from minuet.geometry import RectPsiField, _ray_boundary_distance
from minuet.verification import verify_gs_geqdsk

GEQ = __mitimroot__ / "tests" / "data" / "SPARC_DN_PRD_freegs_20221013.geq"
GACODE = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"
OUT = __mitimroot__ / "tests" / "scratch" / "dev_minuet_sharp_boundary"
OUT.mkdir(parents=True, exist_ok=True)

NO_NOTEBOOK = "--no-notebook" in sys.argv
COLD = "--cold" in sys.argv

# The scan: (gs_ns, gs_ntheta, FixedBoundaryGS boundary kwargs, tag).
# First = coupled-loop default, then the verification-suite hires,
# "pedestal-resolving production", the corner-stress grid -- and finally the
# boundary-ingestion knobs added 2026-07-15 (Settings.gs_boundary_refine_mm /
# gs_boundary_interp): 5 mm knots near the X-point corners (corner-localized
# resampling) + shape-preserving PCHIP rho(theta). Ingestion alone improves
# 12.0 -> 0.65 mm max (measured standalone). NOTE the refined case runs at
# ntheta=2048, NOT 768: the theta grid must sample the refined knots LOCALLY
# (ntheta >= 2 pi rho_knot / spacing ~ 1460 at 5 mm) -- at 512-1024 the
# metric ALIASES (measured: 8-17x interior-field noise, mm glitches on
# traced surfaces along the corner rays, oscillatory shear; found by Pablo
# on the first refined coupled run). A [gs] WARNING now fires on violation.
SCAN = [(128, 256, {}, ""),
        (192, 384, {}, ""),
        (256, 512, {}, ""),
        (384, 768, {}, ""),
        (256, 2048, dict(boundary_refine_mm=5.0, boundary_interp="pchip"),
         "+5mm/pchip")]
LEVELS = [0.95, 0.99, 0.995]      # traceable in BOTH fields (rbbbs is inside
                                  # the true separatrix: 0.999 has no root in
                                  # the file field along every ray)
LEVEL_SELF = 0.998                # GS-only self-convergence level. NOT closer
                                  # to 1: the psi_N in [0.999, 1] sliver is
                                  # radially thinner (sub-mm) than the spline-
                                  # vs-polygon boundary offset, so a root
                                  # bracket capped at the polygon misses it
CORNER_HALF_WIDTH = np.deg2rad(12.0)   # corner window about each X-point


# =============================================================================
# Helpers
# =============================================================================
def menger_curvature(R, Z):
    """Unsigned curvature at each vertex of a closed polyline (Menger:
    kappa = 4 A / (|ab| |bc| |ca|) on consecutive triplets)."""
    Rm, Zm = np.roll(R, 1), np.roll(Z, 1)
    Rp, Zp = np.roll(R, -1), np.roll(Z, -1)
    twoA = np.abs((R - Rm) * (Zp - Zm) - (Rp - Rm) * (Z - Zm))
    a = np.hypot(R - Rm, Z - Zm)
    b = np.hypot(Rp - R, Zp - Z)
    c = np.hypot(Rp - Rm, Zp - Zm)
    return 2.0 * twoA / np.maximum(a * b * c, 1e-300)


def dist_to_polygon(Rq, Zq, Rp, Zp):
    """Distance of query points to a closed polygon (min over segments)."""
    R1, Z1 = Rp, Zp
    R2, Z2 = np.roll(Rp, -1), np.roll(Zp, -1)
    dR, dZ = R2 - R1, Z2 - Z1
    L2 = np.maximum(dR**2 + dZ**2, 1e-300)
    t = ((Rq[:, None] - R1) * dR + (Zq[:, None] - Z1) * dZ) / L2
    t = np.clip(t, 0.0, 1.0)
    dx = Rq[:, None] - (R1 + t * dR)
    dz = Zq[:, None] - (Z1 + t * dZ)
    return np.sqrt((dx**2 + dz**2).min(axis=1))


def trace_level(psin_at, rc, zc, r_hi_of_theta, level, theta):
    """Radii r(theta) of the psi_N = level contour about (rc, zc), by root
    finding along rays. NaN where the level has no root inside r_hi."""
    r = np.full(theta.size, np.nan)
    for k, th in enumerate(theta):
        ct, st = np.cos(th), np.sin(th)
        f = lambda rr: psin_at(rc + rr * ct, zc + rr * st) - level
        r_hi = r_hi_of_theta[k]
        if f(1e-3) < 0.0 <= f(r_hi):
            r[k] = brentq(f, 1e-3, r_hi, xtol=1e-10)
    return r


# =============================================================================
# 0. Read the geqdsk; locate the X-point corners of the stored separatrix
# =============================================================================
gq = GEQDSK.from_file(str(GEQ))
Rb, Zb = gq.rbbbs.copy(), gq.zbbbs.copy()
if np.hypot(Rb[-1] - Rb[0], Zb[-1] - Zb[0]) < 1e-12:
    Rb, Zb = Rb[:-1], Zb[:-1]

kap_poly = menger_curvature(Rb, Zb)
i_top = np.argmax(np.where(Zb > 0, kap_poly, -1.0))
i_bot = np.argmax(np.where(Zb < 0, kap_poly, -1.0))
corners = [(Rb[i_top], Zb[i_top]), (Rb[i_bot], Zb[i_bot])]
seg = np.median(np.hypot(np.diff(Rb), np.diff(Zb)))
print(f"[corners] sharpest vertices of the stored separatrix "
      f"(median segment {1e3*seg:.1f} mm):")
for (Rx, Zx), lab in zip(corners, ("upper", "lower")):
    print(f"  {lab} X-point corner at (R, Z) = ({Rx:.4f}, {Zx:.4f}) m")

# Common ray center for every contour comparison: the FILE axis
rc, zc = float(gq.rmaxis), float(gq.zmaxis)
th_corners = [np.arctan2(Zx - zc, Rx - rc) for (Rx, Zx) in corners]


def in_corner_window(theta):
    d = [np.abs((theta - tc + np.pi) % (2 * np.pi) - np.pi) for tc in th_corners]
    return np.minimum(*d) < CORNER_HALF_WIDTH


THETA = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
IN_CORNER = in_corner_window(THETA)
R_BND = _ray_boundary_distance(rc, zc, THETA, Rb, Zb)

# =============================================================================
# 1. Boundary ingestion: the spline rho(theta) vs the raw polygon
# =============================================================================
Rc_s, Zc_s, rho_s, _ = _boundary_spline(Rb, Zb)
th_fine = np.linspace(0.0, 2 * np.pi, 8192, endpoint=False)
Rspl = Rc_s + rho_s(th_fine) * np.cos(th_fine)
Zspl = Zc_s + rho_s(th_fine) * np.sin(th_fine)

# the refined-ingestion boundary (the new knobs): 5 mm knots + pchip
from minuet.gs import _resample_polygon
Rc_r, Zc_r, rho_r, _ = _boundary_spline(*_resample_polygon(Rb, Zb, 5e-3),
                                        interp="pchip")
Rrefd = Rc_r + rho_r(th_fine) * np.cos(th_fine)
Zrefd = Zc_r + rho_r(th_fine) * np.sin(th_fine)
d_refd = dist_to_polygon(Rrefd, Zrefd, Rb, Zb)                    # [m]
d_spl = dist_to_polygon(Rspl, Zspl, Rb, Zb)                       # [m]
in_c_fine = in_corner_window(np.arctan2(Zspl - zc, Rspl - rc))

# GS-side root brackets must follow each case's OWN boundary curve (the
# rho(theta) interpolant, spline or refined pchip), not the raw polygon --
# off the corners the interpolant and the polygon differ by more than the
# whole psi_N > 0.999 sliver (sub-mm), and the refined boundary reaches
# ~12 mm closer to the corner tips than the raw spline
def ray_cap_of(gs_solver):
    tf = np.linspace(0.0, 2 * np.pi, 4096, endpoint=False)
    Rbc = gs_solver.Rc + gs_solver.rho(tf) * np.cos(tf)
    Zbc = gs_solver.Zc + gs_solver.rho(tf) * np.sin(tf)
    return _ray_boundary_distance(rc, zc, THETA, Rbc[::4], Zbc[::4])

kap_spl = menger_curvature(Rspl, Zspl)
rcurv_spl = []            # spline's effective corner radius of curvature
for (Rx, Zx) in corners:
    near = np.hypot(Rspl - Rx, Zspl - Zx) < 0.03
    rcurv_spl.append(1.0 / kap_spl[near].max())

ing = {
    "d_corner_max_mm": 1e3 * d_spl[in_c_fine].max(),
    "d_corner_mean_mm": 1e3 * d_spl[in_c_fine].mean(),
    "d_rest_max_mm": 1e3 * d_spl[~in_c_fine].max(),
    "rcurv_top_mm": 1e3 * rcurv_spl[0],
    "rcurv_bot_mm": 1e3 * rcurv_spl[1],
}
print(f"\n[ingestion] periodic-spline boundary representation "
      f"(resolution-independent):")
print(f"  distance to raw polygon: corners max {ing['d_corner_max_mm']:.2f} mm "
      f"(mean {ing['d_corner_mean_mm']:.2f}), elsewhere max "
      f"{ing['d_rest_max_mm']:.2f} mm")
print(f"  X-point corner rounded to radius of curvature: upper "
      f"{ing['rcurv_top_mm']:.1f} mm, lower {ing['rcurv_bot_mm']:.1f} mm "
      f"(polygon segment scale {1e3*seg:.1f} mm)")
print(f"  with refine 5 mm + pchip: distance to polygon max "
      f"{1e3*d_refd.max():.2f} mm everywhere (ringing eliminated)")

# =============================================================================
# 2. Resolution scan: solve from the file's own boundary + sources
# =============================================================================
file_field = RectPsiField(gq.R, gq.Z, gq.psirz)
dpsi_file = gq.sibry - gq.simag


def psin_file(R, Z):
    return float((file_field.ev(R, Z) - gq.simag) / dpsi_file)


print("\n[file] tracing reference contours from the geqdsk PSIRZ ...")
r_file = {lev: trace_level(psin_file, rc, zc, 1.05 * R_BND, lev, THETA)
          for lev in LEVELS}

cases = []
for ns, nt, bkw, btag in SCAN:
    print(f"\n[scan] gs {ns} x {nt}{' ' + btag if btag else ''}: "
          f"solving from the file boundary + p'/FF' ...")
    chk = verify_gs_geqdsk(gq, ns=ns, ntheta=nt, **bkw)
    gs, sol, u_file = chk["_fields"]

    def psin_gs(R, Z, _sol=sol):
        return float(1.0 - _sol.field.ev(R, Z) / _sol.u_ax)

    # near-edge contours from the SOLVED field, same rays/center as the file
    r_cap = 0.9999 * ray_cap_of(gs)
    r_gs = {lev: trace_level(psin_gs, rc, zc, r_cap, lev, THETA)
            for lev in LEVELS + [LEVEL_SELF]}

    # contour mismatch vs the file, split corner windows / rest [mm]
    cmp_ = {}
    for lev in LEVELS:
        d = 1e3 * np.abs(r_gs[lev] - r_file[lev])
        ok = np.isfinite(d)
        cmp_[lev] = {
            "corner_mean": float(np.nanmean(d[ok & IN_CORNER])),
            "corner_max": float(np.nanmax(d[ok & IN_CORNER])),
            "rest_mean": float(np.nanmean(d[ok & ~IN_CORNER])),
            "rest_max": float(np.nanmax(d[ok & ~IN_CORNER])),
        }

    # q(psi_N) from the traced geometry vs the file QPSI
    geom = sol.geometry(n_surfaces=120, n_theta=1024)
    q_geo = np.abs(geom.q)
    pn_geo = geom.psin
    msk = (gq.psin_grid > 0.05) & (gq.psin_grid < min(0.95, pn_geo.max()))
    q_file_on = np.abs(gq.qpsi[msk])
    q_gs_on = PchipInterpolator(pn_geo, q_geo)(gq.psin_grid[msk])
    q_rms = float(np.sqrt(np.mean(((q_gs_on - q_file_on) / q_file_on) ** 2)))

    cases.append(dict(ns=ns, nt=nt, btag=btag, refined=bool(bkw),
                      chk=chk, sol=sol, geom=geom,
                      r_gs=r_gs, cmp=cmp_, q_rms=q_rms,
                      pn_geo=pn_geo, q_geo=q_geo))
    print(f"  Picard iters {chk['gs_iterations']}, depth err "
          f"{100*chk['depth_rel_err']:.2f}%, Ip err {100*chk['Ip_rel_err']:.2f}%, "
          f"psi rms {100*chk['psi_rel_err_rms']:.2f}% | "
          f"psiN=0.995 contour, corners: mean {cmp_[0.995]['corner_mean']:.1f} / "
          f"max {cmp_[0.995]['corner_max']:.1f} mm")

# self-convergence of the near-edge contour against the finest UNREFINED
# case (the refined case has a genuinely different -- sharper -- boundary,
# so comparing it to the unrefined reference would measure the boundary
# change, not grid convergence)
plain = [c for c in cases if not c["refined"]]
r_ref = plain[-1]["r_gs"][LEVEL_SELF]
for c in cases:
    if c["refined"] or c is plain[-1]:
        c["self_corner_max"] = c["self_rest_max"] = np.nan
        continue
    d = 1e3 * np.abs(c["r_gs"][LEVEL_SELF] - r_ref)
    ok = np.isfinite(d)
    c["self_corner_max"] = float(d[ok & IN_CORNER].max())
    c["self_rest_max"] = float(d[ok & ~IN_CORNER].max())

# =============================================================================
# 3. Full coupled run at the top scan resolution, raw separatrix
# =============================================================================
NS_HI, NT_HI, BKW_HI, _ = SCAN[-1]
print(f"\n[coupled] full minuet on the raw separatrix, gs {NS_HI} x {NT_HI} "
      f"({BKW_HI or 'raw ingestion'}), PRD kinetics, t_end = 0.5 s ...")
kin = TabulatedKineticProfiles.from_input_gacode(InputGacode.from_file(str(GACODE)))
m = minuet.cached(
    OUT / "coupled_refined_nt2048.minuet", gq, profiles=kin, cold_start=COLD,
    settings=Settings(
        t_end=0.5,
        evolve_equilibrium=True,
        gs_ns=NS_HI, gs_ntheta=NT_HI,
        gs_boundary_refine_mm=BKW_HI.get("boundary_refine_mm"),
        gs_boundary_interp=BKW_HI.get("boundary_interp", "spline"),
        n_surfaces=100, n_theta_trace=1024,
        bootstrap=SauterBootstrap(),
        diffusion=DiffusionSettings(n_save=51),
    ),
)
res = m.result

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 96)
print(" MINUET sharp-corner capture -- SPARC DN separatrix, resolution scan")
print("   (contour rows: |r_GS - r_file| at psi_N = 0.995; self rows: "
      "psi_N = 0.999 vs finest)")
print("=" * 96)
hdr = (f"  {'gs grid':<20} {'iters':>5} {'depth%':>7} {'Ip%':>6} {'psi_rms%':>9} "
       f"{'q_rms%':>7} {'corner mm':>12} {'rest mm':>10} {'self-c mm':>10}")
print(hdr)
for c in cases:
    cm = c["cmp"][0.995]
    sc = f"{c['self_corner_max']:.2f}" if np.isfinite(c["self_corner_max"]) else "--"
    grid = f"{c['ns']}x{c['nt']}{c['btag']}"
    print(f"  {grid:<20} {c['chk']['gs_iterations']:>5} "
          f"{100*c['chk']['depth_rel_err']:>7.2f} {100*c['chk']['Ip_rel_err']:>6.2f} "
          f"{100*c['chk']['psi_rel_err_rms']:>9.3f} {100*c['q_rms']:>7.2f} "
          f"{cm['corner_mean']:>5.1f}/{cm['corner_max']:<5.1f} "
          f"{cm['rest_mean']:>4.1f}/{cm['rest_max']:<4.1f} "
          f"{sc:>10}")
print(f"\n  boundary ingestion (all resolutions): spline-vs-polygon corners max "
      f"{ing['d_corner_max_mm']:.2f} mm; corner radius of curvature "
      f"{ing['rcurv_top_mm']:.0f} / {ing['rcurv_bot_mm']:.0f} mm (up/down)")
print(f"  coupled hires run: {len(res.t)} frames to t = {res.t[-1]:.2f} s, "
      f"q0 {np.abs(res.q[0, 0]):.3f} -> {np.abs(res.q[-1, 0]):.3f}, "
      f"li3 {res.li3[0]:.3f} -> {res.li3[-1]:.3f}")
print("=" * 96 + "\n")

# =============================================================================
# FigureNotebook
# =============================================================================
fn = FigureNotebook("MINUET sharp-corner capture -- SPARC DN separatrix")
figs = []
cols = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(SCAN)))

# ------------------------------------------------------- Boundary ingestion
fig = fn.add_figure(label="Boundary ingestion"); figs.append(("ingestion", fig))
fig.set_layout_engine("constrained")
gs_ = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.6])
ax = fig.add_subplot(gs_[:, 0])
ax.plot(np.append(Rb, Rb[0]), np.append(Zb, Zb[0]), "k.-", ms=2, lw=0.6,
        label="raw rbbbs polygon")
ax.plot(Rspl, Zspl, color="#d1495b", lw=1.0, label="spline rho(theta)")
for (Rx, Zx) in corners:
    ax.plot(Rx, Zx, "x", color="#3f7cac", ms=10, mew=2)
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.legend(fontsize="small"); ax.set_title("stored separatrix + X-point corners")

for j, ((Rx, Zx), lab) in enumerate(zip(corners, ("upper", "lower"))):
    ax = fig.add_subplot(gs_[j, 1])
    ax.plot(np.append(Rb, Rb[0]), np.append(Zb, Zb[0]), "k.-", ms=4, lw=0.8,
            label="polygon")
    ax.plot(Rspl, Zspl, color="#d1495b", lw=1.2, label="spline (raw)")
    ax.plot(Rrefd, Zrefd, color="#3f7cac", lw=1.2, ls="--",
            label="refine 5mm + pchip")
    ax.set_xlim(Rx - 0.05, Rx + 0.05); ax.set_ylim(Zx - 0.05, Zx + 0.05)
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"{lab} corner zoom (r_curv "
                 f"{ing['rcurv_top_mm' if j == 0 else 'rcurv_bot_mm']:.0f} mm)")
    if j == 0:
        ax.legend(fontsize="x-small")

ax = fig.add_subplot(gs_[0, 2])
th_q = np.arctan2(Zspl - zc, Rspl - rc) % (2 * np.pi)
o = np.argsort(th_q)
ax.plot(np.rad2deg(th_q[o]), 1e3 * d_spl[o], color="#d1495b", lw=0.8)
for tc in th_corners:
    ax.axvspan(np.rad2deg((tc - CORNER_HALF_WIDTH) % (2 * np.pi)),
               np.rad2deg((tc + CORNER_HALF_WIDTH) % (2 * np.pi)),
               color="#3f7cac", alpha=0.15)
ax.set_xlabel("theta about file axis [deg]"); ax.set_ylabel("distance [mm]")
ax.set_title("spline-to-polygon distance (shaded: corner windows)")

ax = fig.add_subplot(gs_[1, 2])
ax.semilogy(np.rad2deg(th_fine), np.maximum(kap_spl, 1e-3), color="#d1495b",
            lw=0.8, label="spline curve")
ax.axhline(1.0 / max(rcurv_spl), color="#3f7cac", lw=0.8, ls=":",
           label="corner peak")
ax.set_xlabel("theta (spline parameter) [deg]"); ax.set_ylabel("curvature [1/m]")
ax.legend(fontsize="small"); ax.set_title("boundary curvature: corners = spikes")

# ------------------------------------------------------- Corner contours
fig = fn.add_figure(label="Corner contours"); figs.append(("contours", fig))
fig.set_layout_engine("constrained")
axs = fig.subplots(2, len(SCAN), sharex="row", sharey="row")
for i, c in enumerate(cases):
    for j, ((Rx, Zx), lab) in enumerate(zip(corners, ("upper", "lower"))):
        ax = axs[j, i]
        for lev, lw in zip(LEVELS, (0.7, 0.9, 1.1)):
            rf, rg = r_file[lev], c["r_gs"][lev]
            ax.plot(rc + rf * np.cos(THETA), zc + rf * np.sin(THETA),
                    color="k", lw=lw)
            ax.plot(rc + rg * np.cos(THETA), zc + rg * np.sin(THETA),
                    color="#d1495b", lw=lw, ls="--")
        ax.plot(np.append(Rb, Rb[0]), np.append(Zb, Zb[0]), color="gray",
                lw=0.6)
        ax.set_xlim(Rx - 0.12, Rx + 0.12); ax.set_ylim(Zx - 0.12, Zx + 0.12)
        ax.set_aspect("equal")
        if j == 0:
            ax.set_title(f"gs {c['ns']}x{c['nt']}{c['btag']}")
        if i == 0:
            ax.set_ylabel(f"{lab} X-point\nZ [m]")
        if j == 1:
            ax.set_xlabel("R [m]")
axs[0, 0].text(0.03, 0.03, "solid file / dashed MINUET\npsi_N = 0.95, 0.99, 0.995",
               transform=axs[0, 0].transAxes, fontsize="x-small", va="bottom")

# ------------------------------------------------------- Convergence
fig = fn.add_figure(label="Convergence"); figs.append(("convergence", fig))
fig.set_layout_engine("constrained")
axs = fig.subplots(1, 3)
nts = [c["nt"] for c in plain]
ref_case = next((c for c in cases if c["refined"]), None)

ax = axs[0]
for lev, mk in zip(LEVELS, ("o", "s", "^")):
    ax.plot(nts, [c["cmp"][lev]["corner_max"] for c in plain], mk + "-",
            color="#3f7cac", ms=5, label=f"corners, psi_N={lev}")
    ax.plot(nts, [c["cmp"][lev]["rest_max"] for c in plain], mk + "--",
            color="#d1495b", ms=5, label=f"rest, psi_N={lev}")
    if ref_case is not None:
        ax.plot(ref_case["nt"], ref_case["cmp"][lev]["corner_max"], mk,
                color="#3f7cac", ms=9, mfc="none", mew=2)
ax.axhline(ing["d_corner_max_mm"], color="gray", lw=0.8, ls=":",
           label="ingestion floor (raw spline)")
ax.set_xlabel("gs_ntheta"); ax.set_ylabel("max |r_GS - r_file| [mm]")
ax.set_yscale("log"); ax.legend(fontsize="x-small")
ax.set_title("contour mismatch vs file\n(open markers: +refine 5mm / pchip)")

ax = axs[1]
sc = [c["self_corner_max"] for c in plain[:-1]]
sr = [c["self_rest_max"] for c in plain[:-1]]
ax.loglog(nts[:-1], sc, "o-", color="#3f7cac", ms=5, label="corner windows")
ax.loglog(nts[:-1], sr, "s--", color="#d1495b", ms=5, label="rest")
if len(sc) > 1 and min(sc) > 0:
    p = np.polyfit(np.log(nts[:-1]), np.log(sc), 1)[0]
    ax.set_title(f"self-convergence, psi_N={LEVEL_SELF} contour vs finest\n"
                 f"corner-window order ~ ntheta^{p:.1f} (unrefined cases)")
else:
    ax.set_title(f"self-convergence, psi_N={LEVEL_SELF} contour vs finest")
ax.set_xlabel("gs_ntheta"); ax.set_ylabel("max diff to finest [mm]")
ax.legend(fontsize="small")

ax = axs[2]
for key, lab, col in [("depth_rel_err", "flux depth", "#3f7cac"),
                      ("Ip_rel_err", "Ip", "#d1495b"),
                      ("psi_rel_err_rms", "psi rms", "gray")]:
    ax.plot(nts, [100 * c["chk"][key] for c in plain], "o-", color=col,
            ms=5, label=lab)
    if ref_case is not None:
        ax.plot(ref_case["nt"], 100 * ref_case["chk"][key], "o",
                color=col, ms=9, mfc="none", mew=2)
ax.plot(nts, [100 * c["q_rms"] for c in plain], "^-", color="k", ms=5,
        label="q rms (0.05<psi_N<0.95)")
if ref_case is not None:
    ax.plot(ref_case["nt"], 100 * ref_case["q_rms"], "^", color="k",
            ms=9, mfc="none", mew=2)
ax.set_xlabel("gs_ntheta"); ax.set_ylabel("error vs file [%]")
ax.legend(fontsize="small")
ax.set_title("global metrics vs the file\n(open markers: +refine 5mm / pchip)")

# ------------------------------------------------------- q profiles
fig = fn.add_figure(label="q vs file"); figs.append(("q_profiles", fig))
fig.set_layout_engine("constrained")
axs = fig.subplots(1, 2)
ax = axs[0]
ax.plot(gq.psin_grid, np.abs(gq.qpsi), "k-", lw=2.0, label="file QPSI")
for c, col in zip(cases, cols):
    ax.plot(c["pn_geo"], c["q_geo"], ls="--", lw=1.2, color=col,
            label=f"gs {c['ns']}x{c['nt']}{c['btag']}")
ax.set_xlabel("psi_N"); ax.set_ylabel("|q|"); ax.legend(fontsize="small")
ax.set_title("safety factor: file vs solved+traced")
ax = axs[1]
for c, col in zip(cases, cols):
    msk = (gq.psin_grid > 0.05) & (gq.psin_grid < min(0.95, c["pn_geo"].max()))
    qf = np.abs(gq.qpsi[msk])
    qg = PchipInterpolator(c["pn_geo"], c["q_geo"])(gq.psin_grid[msk])
    ax.plot(gq.psin_grid[msk], 100 * (qg - qf) / qf, lw=1.2, color=col,
            label=f"gs {c['ns']}x{c['nt']}{c['btag']} (rms {100*c['q_rms']:.2f}%)")
ax.axhline(0.0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("psi_N"); ax.set_ylabel("(q_GS - q_file)/q_file [%]")
ax.legend(fontsize="small"); ax.set_title("q relative difference")

# ------------------------------------------------------- Coupled hires run
fig = fn.add_figure(label="Coupled hires run"); figs.append(("coupled", fig))
fig.set_layout_engine("constrained")
axs = fig.subplots(1, 3)
ax = axs[0]
for k, lab, ls in [(0, "t = 0", "-"), (len(res.t) - 1, "t = end", "--")]:
    ax.plot(res.x_q, np.abs(res.q[k, :]), ls=ls, color="#d1495b", lw=1.5,
            label=lab)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("x = sqrt(Phi/Phi_b)"); ax.set_ylabel("|q|")
ax.legend(); ax.set_title(f"coupled run, gs {NS_HI}x{NT_HI}, raw separatrix")
ax = axs[1]
ax.plot(res.t, res.li3, color="#d1495b", lw=1.3)
ax.set_xlabel("t [s]"); ax.set_ylabel("li(3)"); ax.set_title("internal inductance")
ax = axs[2]
geomL = m.geom_last
Rbl, Zbl = geomL.boundary
ax.plot(np.append(Rb, Rb[0]), np.append(Zb, Zb[0]), "k-", lw=1.5,
        label="file separatrix")
ax.plot(Rbl, Zbl, color="#d1495b", lw=1.0, ls="--", label="MINUET boundary")
for k in range(0, len(geomL.surfaces), 4):
    RM, ZM = geomL.surfaces[k]
    ax.plot(np.append(RM, RM[0]), np.append(ZM, ZM[0]), color="#d1495b",
            lw=0.4, alpha=0.5)
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.legend(fontsize="small"); ax.set_title("traced surfaces at t = end")

if NO_NOTEBOOK:
    for name, fig in figs:
        f = OUT / f"sharp_boundary_{name}.png"
        fig.savefig(f, dpi=200)
        print(f"  saved {f}")
else:
    fn.show()
