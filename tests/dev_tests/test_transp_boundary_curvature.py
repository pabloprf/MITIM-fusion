"""
test_transp_boundary_curvature.py
=================================
Characterize and test the TRANSP fixed-boundary curvature ratio along the two knobs that
control it, exercising the ACTUAL code paths:

  1. n_mxh              -- the MXH moment smoothing applied when the boundary is written for
                           TRANSP (mitim_tools...TRANSPhelpers.prepare_RZsep_for_TRANSP).
  2. boundary_surface_psin -- the flux surface (psi_N) used as the fixed boundary, i.e. backing
                           the boundary off the separatrix (TRANSPhelpers.transp_input_time.
                           _produce_geometry_profiles, the new knob threaded from the maestro
                           separatrix namelist block).

Motivation: a sharp / near-X-point separatrix makes TRANSP abort on its internal boundary
"curvature ratio too small" check. Two ways to raise the curvature ratio: round the boundary by
lowering n_mxh (distorts kappa/delta), or back the boundary off to a rounder interior surface
(shape-preserving). This test measures the curvature ratio -- TRANSP's own definition (minimum
radius of curvature from a circle through successive triples of boundary points, normalized to the
midplane half-width) -- as a function of each, produces a diagnostic figure, and asserts the
invariants that matter for the boundary_surface_psin feature.

Usage
-----
    run_with_env.sh python tests/dev_tests/test_transp_boundary_curvature.py

What it does
------------
1. Loads the SPARC double-null geqdsk (tests/data/SPARC_DN_PRD_freegs_20221013.geq) and
   converts it to a state via MITIMgeqdsk.to_profiles() -- a sharp separatrix, near the crash regime.
2. n_mxh sweep: writes the separatrix through prepare_RZsep_for_TRANSP at several n_mxh and
   measures the curvature ratio + retained elongation/triangularity.
3. psi_N sweep: builds the fixed boundary through _produce_geometry_profiles at several
   boundary_surface_psin and measures the curvature ratio.
4. Saves a 4-panel figure to tests/scratch/.
5. Asserts:
   - boundary_surface_psin=1.0 reproduces the separatrix (last flux surface) exactly;
   - backing the boundary off (decreasing psi_N) is monotone-rounder (curvature ratio rises);
   - every curvature ratio is finite and positive.
   Exits non-zero if any assertion fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mitim_tools import __mitimroot__
from mitim_tools.gs_tools import GEQtools
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.transp_tools.utils.TRANSPhelpers import prepare_RZsep_for_TRANSP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# SPARC double-null geqdsk: a genuinely sharp (near-X-point) separatrix, so the curvature
# ratio at the boundary is small (~crash regime) and the backoff / n_mxh effects are large --
# a far more representative test than a rounded input.gacode boundary.
GEQDSK = __mitimroot__ / "tests" / "data" / "SPARC_DN_PRD_freegs_20221013.geq"
OUT_DIR = __mitimroot__ / "tests" / "scratch"
N_MXH_VALS = [7, 5, 4, 3, 2, 1]
PSIN_VALS = [1.0, 0.995, 0.99, 0.98, 0.95, 0.90]


# ---------------------------------------------------------------------------
# Curvature ratio -- TRANSP's definition
# ---------------------------------------------------------------------------
def _circumradius(p1, p2, p3):
    a = np.hypot(*(p2 - p3)); b = np.hypot(*(p1 - p3)); c = np.hypot(*(p1 - p2))
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return np.inf if area < 1e-12 else a * b * c / (4 * area)


def curvature_ratio(R, Z):
    """Minimum radius of curvature (3-point circle fit through successive triples) over the
    midplane half-width -- TRANSP's boundary 'curvature ratio'."""
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    if np.hypot(R[-1] - R[0], Z[-1] - Z[0]) < 1e-9:   # drop a duplicated closing point
        R, Z = R[:-1], Z[:-1]
    n = len(R); P = np.column_stack([R, Z])
    Rc = np.array([_circumradius(P[(i - 1) % n], P[i], P[(i + 1) % n]) for i in range(n)])
    return float(np.min(Rc) / (0.5 * (R.max() - R.min())))


def shape_params(R, Z):
    """Geometric elongation and triangularity."""
    R = np.asarray(R); Z = np.asarray(Z)
    a = (R.max() - R.min()) / 2.0
    R0 = (R.max() + R.min()) / 2.0
    kappa = (Z.max() - Z.min()) / (2 * a)
    delta = (R0 - 0.5 * (R[np.argmax(Z)] + R[np.argmin(Z)])) / a
    return kappa, delta


# ---------------------------------------------------------------------------
# Sweeps (exercise the real code)
# ---------------------------------------------------------------------------
def sweep_n_mxh(p):
    """Curvature ratio + retained shape of the separatrix, smoothed at each n_mxh."""
    Rsep, Zsep = p.derived["R_surface"][0, -1, :], p.derived["Z_surface"][0, -1, :]
    out = {}
    for n in N_MXH_VALS:
        _, Rs, Zs = prepare_RZsep_for_TRANSP(np.array(Rsep), np.array(Zsep), n_coeff=n)
        kap, dlt = shape_params(Rs, Zs)
        out[n] = dict(R=np.array(Rs), Z=np.array(Zs), cr=curvature_ratio(Rs, Zs), kappa=kap, delta=dlt)
    return out


def sweep_psin(p):
    """Curvature ratio of the fixed boundary built at each boundary_surface_psin (the real path)."""
    ti = TRANSPhelpers.transp_input_time(None)   # only _produce_geometry_profiles is exercised
    ti.p = p
    out = {}
    for psin in PSIN_VALS:
        ti._produce_geometry_profiles(boundary_surface_psin=psin)
        R, Z = np.array(ti.geometry["R_sep"]), np.array(ti.geometry["Z_sep"])
        out[psin] = dict(R=R, Z=Z, cr=curvature_ratio(R, Z))
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(nmxh, psins, p, out_path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    cmap = plt.cm.viridis

    # (0,0) boundary shape vs n_mxh
    ax = axs[0, 0]
    Rsep, Zsep = p.derived["R_surface"][0, -1, :], p.derived["Z_surface"][0, -1, :]
    ax.plot(Rsep, Zsep, "k-", lw=2.4, label="separatrix", zorder=5)
    for n, col in zip(N_MXH_VALS, cmap(np.linspace(0, 0.85, len(N_MXH_VALS)))):
        d = nmxh[n]
        ax.plot(d["R"], d["Z"], "-", color=col, lw=1.3, label=f"n_mxh={n} (cr={d['cr']:.3f})")
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title("Boundary vs n_mxh (MXH smoothing)"); ax.legend(fontsize=8, loc="upper right")

    # (0,1) curvature ratio + retained shape vs n_mxh
    ax = axs[0, 1]
    ns = N_MXH_VALS
    ax.plot(ns, [nmxh[n]["cr"] for n in ns], "o-", color="tab:blue", label="curvature ratio")
    ax.set_xlabel("n_mxh"); ax.set_ylabel("curvature ratio", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue"); ax.grid(alpha=0.3)
    ax.set_title("Curvature & retained shape vs n_mxh")
    ax2 = ax.twinx()
    ax2.plot(ns, [nmxh[n]["kappa"] for n in ns], "s--", color="tab:red", label="kappa")
    ax2.plot(ns, [nmxh[n]["delta"] for n in ns], "^--", color="tab:orange", label="delta")
    ax2.set_ylabel("kappa / delta")
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")

    # (1,0) boundary shape vs psi_N
    ax = axs[1, 0]
    for psin, col in zip(PSIN_VALS, cmap(np.linspace(0, 0.85, len(PSIN_VALS)))):
        d = psins[psin]
        ax.plot(d["R"], d["Z"], "-", color=col, lw=1.4, label=f"psi_N={psin:.3f} (cr={d['cr']:.3f})")
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title("Boundary vs boundary_surface_psin (backoff)"); ax.legend(fontsize=8, loc="upper right")

    # (1,1) curvature ratio vs psi_N
    ax = axs[1, 1]
    ax.plot(PSIN_VALS, [psins[ps]["cr"] for ps in PSIN_VALS], "o-", color="tab:green")
    ax.set_xlabel("boundary_surface_psin (1.0 = separatrix)"); ax.set_ylabel("curvature ratio")
    ax.set_title("Curvature ratio rises as the boundary backs off"); ax.invert_xaxis(); ax.grid(alpha=0.3)

    fig.suptitle("TRANSP fixed-boundary curvature ratio vs n_mxh and vs boundary_surface_psin",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TRANSP boundary curvature ratio: n_mxh & boundary_surface_psin sweep")
    print("=" * 70)
    p = GEQtools.MITIMgeqdsk(GEQDSK).to_profiles()

    nmxh = sweep_n_mxh(p)
    psins = sweep_psin(p)

    print("\n  n_mxh sweep (separatrix smoothing):")
    for n in N_MXH_VALS:
        d = nmxh[n]
        print(f"    n_mxh={n:>2}:  curvature ratio={d['cr']:.4f}   kappa={d['kappa']:.3f}  delta={d['delta']:.3f}")

    print("\n  boundary_surface_psin sweep (backoff):")
    for ps in PSIN_VALS:
        print(f"    psi_N={ps:.3f}:  curvature ratio={psins[ps]['cr']:.4f}")

    out_png = OUT_DIR / "test_transp_boundary_curvature.png"
    make_figure(nmxh, psins, p, out_png)
    print(f"\n  Figure saved: {out_png}")

    # -------------------- assertions --------------------
    failures = []

    # (1) default psi_N=1.0 reproduces the separatrix (last flux surface)
    Rsep = np.array(p.derived["R_surface"][0, -1, :]); Zsep = np.array(p.derived["Z_surface"][0, -1, :])
    if not (np.allclose(psins[1.0]["R"], Rsep) and np.allclose(psins[1.0]["Z"], Zsep)):
        failures.append("boundary_surface_psin=1.0 does NOT reproduce the separatrix")

    # (2) backing off (lower psi_N) is monotonically rounder (curvature ratio increases)
    cr_psin = np.array([psins[ps]["cr"] for ps in PSIN_VALS])   # PSIN_VALS descending
    if not np.all(np.diff(cr_psin) > -1e-4):
        failures.append(f"curvature ratio not monotone-increasing as psi_N backs off: {cr_psin}")
    if not (cr_psin[-1] > cr_psin[0]):
        failures.append(f"most-backed-off boundary is not rounder than the separatrix: {cr_psin[0]} -> {cr_psin[-1]}")

    # (3) all curvature ratios finite & positive
    all_cr = [nmxh[n]["cr"] for n in N_MXH_VALS] + list(cr_psin)
    if not all(np.isfinite(c) and c > 0 for c in all_cr):
        failures.append(f"non-finite or non-positive curvature ratio encountered: {all_cr}")

    print()
    print("=" * 70)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print("TEST FAILED")
        print("=" * 70)
        sys.exit(1)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
