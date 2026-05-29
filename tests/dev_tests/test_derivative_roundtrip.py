"""
Comprehensive test and visualization of the T -> a/LT -> T roundtrip problem.

Compares the old Lagrange-polynomial derivative (derivation_into_Lx_lagrange)
with the new central-difference derivative (derivation_into_Lx) that is matched
to integration_Lx. Performs roundtrips on both fine and coarse grids, with
varying resolution.

Usage:
    python tests/test_derivative_roundtrip.py
"""

import numpy as np
import torch
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GUItools
from mitim_modules.powertorch.utils import CALCtools
from mitim_tools import __mitimroot__

# ============================================================================
# Load data
# ============================================================================

print("\nLoading input.gacode...")
profiles = PROFILEStools.gacode_state(__mitimroot__ / "tests" / "data" / "input.gacode")

rho = profiles.profiles["rho(-)"]
roa = profiles.derived["roa"]  # r/a coordinate
te = profiles.profiles["te(keV)"]
ti = profiles.profiles["ti(keV)"][:, 0]  # First thermal ion
ne = profiles.profiles["ne(10^19/m^3)"]

all_profiles = {
    r"$T_e$ (keV)": te,
    r"$T_i$ (keV)": ti,
    r"$n_e$ ($10^{19}/m^3$)": ne,
}
profile_short = ["Te", "Ti", "ne"]
profile_labels = list(all_profiles.keys())
profile_arrays = list(all_profiles.values())

# Coarse grid: 5 radial points (typical PORTALS setup)
rho_coarse = np.array([0.2, 0.4, 0.55, 0.7, 0.85])
roa_coarse = np.interp(rho_coarse, rho, roa)
coarse_indices = [np.argmin(np.abs(roa - rc)) for rc in roa_coarse]

# High-resolution grid (like PORTALS improve_resolution_profiles)
N_hires = 200
rho_hires = np.linspace(rho[0], rho[-1], N_hires)
roa_hires = np.interp(rho_hires, rho, roa)
hires_profiles = [np.interp(rho_hires, rho, p) for p in profile_arrays]
coarse_indices_hires = [np.argmin(np.abs(roa_hires - rc)) for rc in roa_coarse]


# ============================================================================
# Helper functions
# ============================================================================

def fine_grid_roundtrip(roa_arr, profile_arr, deriv_func):
    """T -> a/LT -> T on the fine grid."""
    r_torch = torch.from_numpy(roa_arr).unsqueeze(0).double()
    p_torch = torch.from_numpy(profile_arr).unsqueeze(0).double()
    aLT = deriv_func(r_torch[0], p_torch[0], array=False)
    aLT_batch = aLT.unsqueeze(0)
    T_reconstructed = CALCtools.integration_Lx(r_torch, aLT_batch, p_torch[0, -1])
    return aLT.numpy(), T_reconstructed[0].numpy()


def coarse_grid_roundtrip(roa_arr, profile_arr, coarse_idx, deriv_func):
    """
    PORTALS-like roundtrip:
    T -> a/LT (fine) -> sample at coarse -> piecewise-linear interp -> integrate -> T

    The BC is at the last coarse point (e.g. rho=0.85), and the integration
    only covers [0, roa_last_coarse]. This mirrors PORTALS, which fixes T at
    the last prediction radius and stitches the original trailing edge beyond.
    """
    r_torch = torch.from_numpy(roa_arr).double()
    p_torch = torch.from_numpy(profile_arr).double()
    aLT_fine = deriv_func(r_torch, p_torch, array=False).numpy()

    # Control points: zero at axis + values at coarse locations
    roa_cp = np.concatenate([[0.0], roa_arr[coarse_idx]])
    aLT_cp = np.concatenate([[0.0], aLT_fine[coarse_idx]])

    # Only work on the domain [0, last coarse point]
    ir_last = coarse_idx[-1]
    roa_domain = roa_arr[: ir_last + 1]
    prof_domain = profile_arr[: ir_last + 1]

    # Piecewise-linear interpolation of gradients within the domain
    aLT_interp = np.interp(roa_domain, roa_cp, aLT_cp)

    # BC at the last coarse point
    f_bound = p_torch[ir_last]

    r_batch = torch.from_numpy(roa_domain).unsqueeze(0).double()
    aLT_batch = torch.from_numpy(aLT_interp).unsqueeze(0).double()
    T_reconstructed = CALCtools.integration_Lx(r_batch, aLT_batch, f_bound)

    return (aLT_fine, aLT_cp, roa_cp, aLT_interp,
            T_reconstructed[0].numpy(), roa_domain, prof_domain)


def relative_error(original, reconstructed):
    """Relative error (%), avoiding division by zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs((reconstructed - original) / original) * 100.0
        err[~np.isfinite(err)] = 0.0
    return err


# ============================================================================
# Create FigureNotebook
# ============================================================================

fn = GUItools.FigureNotebook("Derivative Roundtrip Analysis", geometry="1800x950")

# ============================================================================
# Tab 1: Original profiles overview
# ============================================================================

fig = fn.add_figure(label="Original Profiles")
grid = fig.add_gridspec(1, 3)
for col, (label, prof) in enumerate(all_profiles.items()):
    ax = fig.add_subplot(grid[0, col])
    ax.plot(rho, prof, "k-o", lw=2, ms=3)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(label)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{len(rho)} grid points")


# ============================================================================
# Tab 2: Derivative comparison (Lagrange vs Central Difference)
# ============================================================================

fig = fn.add_figure(label="Derivative Comparison")
grid = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

for row in range(3):
    for icol, (r_arr, p_arr, rh_arr, nlabel) in enumerate([
        (roa, profile_arrays[row], rho, f"Original ({len(rho)} pts)"),
        (roa_hires, hires_profiles[row], rho_hires, f"Hi-res ({N_hires} pts)"),
    ]):
        r_t = torch.from_numpy(r_arr).double()
        p_t = torch.from_numpy(p_arr).double()
        aLT_lagrange = CALCtools.derivation_into_Lx(r_t, p_t, array=False).numpy()
        aLT_central = CALCtools.derivation_into_Lx_central(r_t, p_t, array=False).numpy()

        ax = fig.add_subplot(grid[row, icol])
        ax.plot(rh_arr, aLT_lagrange, "b-", lw=1.5, label="Lagrange (old)")
        ax.plot(rh_arr, aLT_central, "r--", lw=1.5, label="Central diff (new)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"a/L for {profile_labels[row]}")
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=8)
        ax.set_title(nlabel)
        ax.grid(True, alpha=0.3)

    # Difference column
    ax = fig.add_subplot(grid[row, 2])
    r_t = torch.from_numpy(roa).double()
    p_t = torch.from_numpy(profile_arrays[row]).double()
    diff_orig = (CALCtools.derivation_into_Lx_central(r_t, p_t, array=False).numpy()
                 - CALCtools.derivation_into_Lx(r_t, p_t, array=False).numpy())
    r_t2 = torch.from_numpy(roa_hires).double()
    p_t2 = torch.from_numpy(hires_profiles[row]).double()
    diff_hires = (CALCtools.derivation_into_Lx_central(r_t2, p_t2, array=False).numpy()
                  - CALCtools.derivation_into_Lx(r_t2, p_t2, array=False).numpy())
    ax.plot(rho, diff_orig, "g-o", lw=1.5, ms=3, label=f"Orig (max={np.max(np.abs(diff_orig)):.3f})")
    ax.plot(rho_hires, diff_hires, "m-", lw=1, label=f"Hi-res (max={np.max(np.abs(diff_hires)):.4f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Central - Lagrange")
    ax.set_xlabel(r"$\rho$")
    ax.legend(fontsize=8)
    ax.set_title("Difference (converges with resolution)")
    ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 3: Fine-grid roundtrip on both original and hi-res grids
# ============================================================================

fig = fn.add_figure(label="Fine-Grid Roundtrip")
grid = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

for row in range(3):
    for icol, (r_arr, p_arr, rh_arr, nlabel) in enumerate([
        (roa, profile_arrays[row], rho, f"Original ({len(rho)} pts)"),
        (roa_hires, hires_profiles[row], rho_hires, f"Hi-res ({N_hires} pts)"),
    ]):
        _, T_lag = fine_grid_roundtrip(r_arr, p_arr, CALCtools.derivation_into_Lx)
        _, T_cen = fine_grid_roundtrip(r_arr, p_arr, CALCtools.derivation_into_Lx_central)
        err_lag = relative_error(p_arr, T_lag)
        err_cen = relative_error(p_arr, T_cen)

        ax = fig.add_subplot(grid[row, icol * 2])
        ax.plot(rh_arr, p_arr, "k-", lw=2, label="Original")
        ax.plot(rh_arr, T_lag, "b--", lw=1.5, label="Lagrange")
        ax.plot(rh_arr, T_cen, "r:", lw=2, label="Central diff")
        ax.set_ylabel(profile_labels[row])
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=7)
        ax.set_title(nlabel)
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(grid[row, icol * 2 + 1])
        ax.semilogy(rh_arr[1:], err_lag[1:], "b-", lw=1.5,
                     label=f"Lagrange (max={np.max(err_lag[1:]):.3f}%)")
        ax.semilogy(rh_arr[1:], err_cen[1:], "r-", lw=1.5,
                     label=f"Central (max={np.max(err_cen[1:]):.3f}%)")
        ax.set_ylabel("Relative error (%)")
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=6)
        ax.set_title(f"Error - {nlabel}")
        ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 4: Coarse-grid roundtrip (5 control points) on both grids
# ============================================================================

fig = fn.add_figure(label="Coarse-Grid Roundtrip (5 pts)")
fig.suptitle(r"Coarse-Grid Roundtrip, 5 control points at $\rho$ = " + str(list(rho_coarse))
             + r", BC at last point", fontsize=11)
grid = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

for row in range(3):
    for icol, (r_arr, p_arr, rh_arr, c_idx, nlabel) in enumerate([
        (roa, profile_arrays[row], rho, coarse_indices, f"Orig ({len(rho)} pts)"),
        (roa_hires, hires_profiles[row], rho_hires, coarse_indices_hires, f"Hi-res ({N_hires} pts)"),
    ]):
        aLT_f_lag, aLT_cp_lag, roa_cp, aLT_i_lag, T_c_lag, roa_dom_lag, p_dom_lag = coarse_grid_roundtrip(
            r_arr, p_arr, c_idx, CALCtools.derivation_into_Lx)
        aLT_f_cen, aLT_cp_cen, _, aLT_i_cen, T_c_cen, roa_dom_cen, p_dom_cen = coarse_grid_roundtrip(
            r_arr, p_arr, c_idx, CALCtools.derivation_into_Lx_central)
        rho_cp = np.interp(roa_cp, r_arr, rh_arr)
        rho_dom = np.interp(roa_dom_lag, r_arr, rh_arr)

        ax = fig.add_subplot(grid[row, icol * 2])
        ax.plot(rh_arr, aLT_f_lag, "b-", lw=1, alpha=0.4, label="Fine (Lagrange)")
        ax.plot(rh_arr, aLT_f_cen, "r-", lw=1, alpha=0.4, label="Fine (Central)")
        ax.plot(rho_dom, aLT_i_lag, "b--", lw=1.5, alpha=0.8, label="Interp (Lagrange)")
        ax.plot(rho_dom, aLT_i_cen, "r:", lw=2, alpha=0.8, label="Interp (Central)")
        ax.plot(rho_cp, aLT_cp_lag, "bs", ms=6, zorder=5)
        ax.plot(rho_cp, aLT_cp_cen, "r^", ms=6, zorder=5)
        ax.axvline(rho_dom[-1], color="gray", ls="--", lw=0.8, alpha=0.6, label="BC")
        ax.set_ylabel(f"a/L for {profile_labels[row]}")
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=5)
        ax.set_title(f"Gradients - {nlabel}")
        ax.grid(True, alpha=0.3)

        err_lag = relative_error(p_dom_lag, T_c_lag)
        err_cen = relative_error(p_dom_cen, T_c_cen)
        ax = fig.add_subplot(grid[row, icol * 2 + 1])
        ax.plot(rho_dom, p_dom_lag, "k-", lw=2, label="Original")
        ax.plot(rho_dom, T_c_lag, "b--", lw=1.5, label=f"Lagrange (err max={np.max(err_lag[1:]):.2f}%)")
        ax.plot(rho_dom, T_c_cen, "r:", lw=2, label=f"Central (err max={np.max(err_cen[1:]):.2f}%)")
        ax.axvline(rho_dom[-1], color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylabel(profile_labels[row])
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=6)
        ax.set_title(f"Profiles - {nlabel}")
        ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 5: Error at control points (bar charts)
# ============================================================================

fig = fn.add_figure(label="Error at Control Points")
grid = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

for col in range(3):
    _, _, _, _, T_c_lag, _, p_dom = coarse_grid_roundtrip(
        roa_hires, hires_profiles[col], coarse_indices_hires, CALCtools.derivation_into_Lx)
    _, _, _, _, T_c_cen, _, _ = coarse_grid_roundtrip(
        roa_hires, hires_profiles[col], coarse_indices_hires, CALCtools.derivation_into_Lx_central)

    # coarse_indices_hires are all <= ir_last, so valid in the truncated arrays
    orig_at_cp = p_dom[coarse_indices_hires]
    lag_at_cp = T_c_lag[coarse_indices_hires]
    cen_at_cp = T_c_cen[coarse_indices_hires]
    err_lag_cp = np.abs((lag_at_cp - orig_at_cp) / orig_at_cp) * 100.0
    err_cen_cp = np.abs((cen_at_cp - orig_at_cp) / orig_at_cp) * 100.0

    x_pos = np.arange(len(rho_coarse))
    width = 0.35

    ax = fig.add_subplot(grid[0, col])
    ax.bar(x_pos - width / 2, orig_at_cp, width * 0.9, label="Original", color="gray", alpha=0.8)
    ax.bar(x_pos - width / 2, lag_at_cp, width * 0.9, label="Lagrange", color="blue", alpha=0.3)
    ax.bar(x_pos + width / 2, cen_at_cp, width * 0.9, label="Central", color="red", alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r:.2f}" for r in rho_coarse])
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(profile_labels[col])
    ax.legend(fontsize=7)
    ax.set_title(f"{profile_labels[col]} at control points")

    ax = fig.add_subplot(grid[1, col])
    ax.bar(x_pos - width / 2, err_lag_cp, width,
           label=f"Lagrange (mean={np.mean(err_lag_cp):.2f}%)", color="blue", alpha=0.7)
    ax.bar(x_pos + width / 2, err_cen_cp, width,
           label=f"Central (mean={np.mean(err_cen_cp):.2f}%)", color="red", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r:.2f}" for r in rho_coarse])
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Relative error (%)")
    ax.legend(fontsize=7)


# ============================================================================
# Tab 6: Analytic test with exact solution
# ============================================================================

fig = fn.add_figure(label="Analytic Test")
fig.suptitle(r"Analytic: $T(r) = T_0 (1 - (r/a)^2)^2$, exact $a/L_T = 4x/(1-x^2)$", fontsize=11)
grid = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)

N_an = 201
roa_an = np.linspace(1e-4, 0.95, N_an)
T0 = 5.0
T_an = T0 * (1.0 - roa_an**2) ** 2
aLT_exact = 4.0 * roa_an / (1.0 - roa_an**2 + 1e-30)

r_t = torch.from_numpy(roa_an).double()
p_t = torch.from_numpy(T_an).double()
aLT_lag_an = CALCtools.derivation_into_Lx(r_t, p_t, array=False).numpy()
aLT_cen_an = CALCtools.derivation_into_Lx_central(r_t, p_t, array=False).numpy()

_, T_rt_lag = fine_grid_roundtrip(roa_an, T_an, CALCtools.derivation_into_Lx)
_, T_rt_cen = fine_grid_roundtrip(roa_an, T_an, CALCtools.derivation_into_Lx_central)

roa_coarse_an = np.array([0.15, 0.35, 0.5, 0.65, 0.8])
ci_an = [np.argmin(np.abs(roa_an - rc)) for rc in roa_coarse_an]
_, aLT_cp_lag, rcp_lag, aLT_i_lag, T_c_lag, roa_dom_an, p_dom_an = coarse_grid_roundtrip(
    roa_an, T_an, ci_an, CALCtools.derivation_into_Lx)
_, aLT_cp_cen, rcp_cen, aLT_i_cen, T_c_cen, _, _ = coarse_grid_roundtrip(
    roa_an, T_an, ci_an, CALCtools.derivation_into_Lx_central)
aLT_exact_dom = 4.0 * roa_dom_an / (1.0 - roa_dom_an**2 + 1e-30)

# Row 0: Fine grid
ax = fig.add_subplot(grid[0, 0])
ax.plot(roa_an, aLT_exact, "k-", lw=2, label="Exact")
ax.plot(roa_an, aLT_lag_an, "b--", lw=1.5, label="Lagrange")
ax.plot(roa_an, aLT_cen_an, "r:", lw=2, label="Central diff")
ax.set_ylabel("a/LT"); ax.set_xlabel("r/a"); ax.set_ylim(0, 20)
ax.legend(fontsize=8); ax.set_title("Fine: Derivatives vs Exact"); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(grid[0, 1])
ax.semilogy(roa_an[1:-1], np.abs(aLT_lag_an[1:-1] - aLT_exact[1:-1]), "b-", lw=1.5, label="Lagrange")
ax.semilogy(roa_an[1:-1], np.abs(aLT_cen_an[1:-1] - aLT_exact[1:-1]), "r-", lw=1.5, label="Central diff")
ax.set_ylabel("|Error in a/LT|"); ax.set_xlabel("r/a")
ax.legend(fontsize=8); ax.set_title("Fine: Derivative Error"); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(grid[0, 2])
ax.plot(roa_an, T_an, "k-", lw=2, label="Original")
ax.plot(roa_an, T_rt_lag, "b--", lw=1.5, label="Lagrange")
ax.plot(roa_an, T_rt_cen, "r:", lw=2, label="Central diff")
ax.set_ylabel("T (keV)"); ax.set_xlabel("r/a")
ax.legend(fontsize=8); ax.set_title("Fine: Roundtrip"); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(grid[0, 3])
err_lag_an = relative_error(T_an, T_rt_lag)
err_cen_an = relative_error(T_an, T_rt_cen)
ax.semilogy(roa_an[1:], err_lag_an[1:], "b-", lw=1.5,
            label=f"Lagrange (max={np.max(err_lag_an[1:]):.4f}%)")
ax.semilogy(roa_an[1:], err_cen_an[1:], "r-", lw=1.5,
            label=f"Central (max={np.max(err_cen_an[1:]):.4f}%)")
ax.set_ylabel("Relative error (%)"); ax.set_xlabel("r/a")
ax.legend(fontsize=7); ax.set_title("Fine: Roundtrip Error"); ax.grid(True, alpha=0.3)

# Row 1: Coarse grid (domain up to last coarse point only)
ax = fig.add_subplot(grid[1, 0])
ax.plot(roa_dom_an, aLT_exact_dom, "k-", lw=1, alpha=0.4, label="Exact")
ax.plot(roa_dom_an, aLT_i_lag, "b--", lw=1.5, label="Interp (Lagrange)")
ax.plot(roa_dom_an, aLT_i_cen, "r:", lw=2, label="Interp (Central)")
ax.plot(rcp_lag, aLT_cp_lag, "bs", ms=7, zorder=5)
ax.plot(rcp_cen, aLT_cp_cen, "r^", ms=7, zorder=5)
ax.axvline(roa_dom_an[-1], color="gray", ls="--", lw=0.8, alpha=0.6, label="BC")
ax.set_ylabel("a/LT"); ax.set_xlabel("r/a"); ax.set_ylim(0, 20)
ax.legend(fontsize=7); ax.set_title("Coarse: Piecewise-linear"); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(grid[1, 1])
ax.plot(roa_dom_an, np.abs(aLT_i_lag - aLT_exact_dom), "b-", lw=1.5, label="Lagrange")
ax.plot(roa_dom_an, np.abs(aLT_i_cen - aLT_exact_dom), "r-", lw=1.5, label="Central diff")
ax.set_ylabel("|Gradient interp error|"); ax.set_xlabel("r/a")
ax.legend(fontsize=8); ax.set_title("Coarse: Interp Error"); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(grid[1, 2])
ax.plot(roa_dom_an, p_dom_an, "k-", lw=2, label="Original")
ax.plot(roa_dom_an, T_c_lag, "b--", lw=1.5, label="Lagrange")
ax.plot(roa_dom_an, T_c_cen, "r:", lw=2, label="Central diff")
ax.axvline(roa_dom_an[-1], color="gray", ls="--", lw=0.8, alpha=0.6)
ax.set_ylabel("T (keV)"); ax.set_xlabel("r/a")
ax.legend(fontsize=8); ax.set_title("Coarse: Roundtrip"); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(grid[1, 3])
err_c_lag = relative_error(p_dom_an, T_c_lag)
err_c_cen = relative_error(p_dom_an, T_c_cen)
ax.semilogy(roa_dom_an[1:], err_c_lag[1:], "b-", lw=1.5,
            label=f"Lagrange (max={np.max(err_c_lag[1:]):.2f}%)")
ax.semilogy(roa_dom_an[1:], err_c_cen[1:], "r-", lw=1.5,
            label=f"Central (max={np.max(err_c_cen[1:]):.2f}%)")
ax.set_ylabel("Relative error (%)"); ax.set_xlabel("r/a")
ax.legend(fontsize=7); ax.set_title("Coarse: Roundtrip Error"); ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 7: Grid resolution convergence study
# ============================================================================

fig = fn.add_figure(label="Resolution Convergence")
grid = fig.add_gridspec(1, 3, wspace=0.3)

resolutions = [21, 41, 81, 101, 151, 201, 301, 501]

for col in range(3):
    max_err_lag_res, max_err_cen_res = [], []
    for N in resolutions:
        roa_test = np.linspace(roa[0], roa[-1], N)
        p_test = np.interp(roa_test, roa, profile_arrays[col])
        _, T_lag = fine_grid_roundtrip(roa_test, p_test, CALCtools.derivation_into_Lx)
        _, T_cen = fine_grid_roundtrip(roa_test, p_test, CALCtools.derivation_into_Lx_central)
        max_err_lag_res.append(np.max(relative_error(p_test, T_lag)[1:]))
        max_err_cen_res.append(np.max(relative_error(p_test, T_cen)[1:]))

    ax = fig.add_subplot(grid[0, col])
    ax.loglog(resolutions, max_err_lag_res, "b-o", lw=2, ms=6, label="Lagrange (old)")
    ax.loglog(resolutions, max_err_cen_res, "r-s", lw=2, ms=6, label="Central diff (new)")
    h = np.array(resolutions, dtype=float)
    ref2 = max_err_lag_res[0] * (h / h[0]) ** (-2)
    ax.loglog(resolutions, ref2, "k--", lw=0.8, alpha=0.5, label=r"$\propto N^{-2}$")
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Max relative error (%)")
    ax.set_title(profile_short[col])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


# ============================================================================
# Tab 8: Summary bar chart
# ============================================================================

fig = fn.add_figure(label="Summary")
grid = fig.add_gridspec(1, 3, wspace=0.3)

categories = [
    f"Fine\n({len(rho)} pts)",
    f"Fine\n({N_hires} pts)",
    f"Coarse 5pts\n({len(rho)} pts)",
    f"Coarse 5pts\n({N_hires} pts)",
]

all_errors_lag = [[] for _ in range(4)]
all_errors_cen = [[] for _ in range(4)]

for i in range(3):
    _, T = fine_grid_roundtrip(roa, profile_arrays[i], CALCtools.derivation_into_Lx)
    all_errors_lag[0].append(np.max(relative_error(profile_arrays[i], T)[1:]))
    _, T = fine_grid_roundtrip(roa, profile_arrays[i], CALCtools.derivation_into_Lx_central)
    all_errors_cen[0].append(np.max(relative_error(profile_arrays[i], T)[1:]))

    hp = hires_profiles[i]
    _, T = fine_grid_roundtrip(roa_hires, hp, CALCtools.derivation_into_Lx)
    all_errors_lag[1].append(np.max(relative_error(hp, T)[1:]))
    _, T = fine_grid_roundtrip(roa_hires, hp, CALCtools.derivation_into_Lx_central)
    all_errors_cen[1].append(np.max(relative_error(hp, T)[1:]))

    _, _, _, _, T, _, p_dom = coarse_grid_roundtrip(roa, profile_arrays[i], coarse_indices, CALCtools.derivation_into_Lx)
    all_errors_lag[2].append(np.max(relative_error(p_dom, T)[1:]))
    _, _, _, _, T, _, p_dom = coarse_grid_roundtrip(roa, profile_arrays[i], coarse_indices, CALCtools.derivation_into_Lx_central)
    all_errors_cen[2].append(np.max(relative_error(p_dom, T)[1:]))

    _, _, _, _, T, _, p_dom = coarse_grid_roundtrip(roa_hires, hp, coarse_indices_hires, CALCtools.derivation_into_Lx)
    all_errors_lag[3].append(np.max(relative_error(p_dom, T)[1:]))
    _, _, _, _, T, _, p_dom = coarse_grid_roundtrip(roa_hires, hp, coarse_indices_hires, CALCtools.derivation_into_Lx_central)
    all_errors_cen[3].append(np.max(relative_error(p_dom, T)[1:]))

for col in range(3):
    ax = fig.add_subplot(grid[0, col])
    x = np.arange(len(categories))
    width = 0.35
    lag_vals = [all_errors_lag[j][col] for j in range(4)]
    cen_vals = [all_errors_cen[j][col] for j in range(4)]
    bars1 = ax.bar(x - width / 2, lag_vals, width, label="Lagrange (old)", color="blue", alpha=0.7)
    bars2 = ax.bar(x + width / 2, cen_vals, width, label="Central diff (new)", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Max relative error (%)")
    ax.set_title(profile_short[col])
    ax.legend(fontsize=8)
    ax.bar_label(bars1, fmt="%.2f%%", fontsize=7, padding=2)
    ax.bar_label(bars2, fmt="%.2f%%", fontsize=7, padding=2)
    ax.grid(True, alpha=0.3, axis="y")


# ============================================================================
# Tab 9: Multi-pass degradation (original -> 1 pass -> 2 passes -> 3 passes)
# ============================================================================

def multi_pass_coarse_roundtrip(roa_arr, profile_arr, coarse_idx, deriv_func, n_passes):
    """
    Perform n successive T -> a/LT -> T roundtrips through the coarse
    parameterization. BC is always fixed to the original profile value
    at the last coarse point (as PORTALS does).
    """
    ir_last = coarse_idx[-1]
    roa_dom = roa_arr[: ir_last + 1]
    roa_cp = np.concatenate([[0.0], roa_arr[coarse_idx]])
    bc_value = profile_arr[ir_last]  # fixed BC from original

    current_T = profile_arr[: ir_last + 1].copy()
    results = []  # (aLT_on_domain, T_result) per pass

    for _ in range(n_passes):
        r_t = torch.from_numpy(roa_dom).double()
        p_t = torch.from_numpy(current_T).double()
        aLT = deriv_func(r_t, p_t, array=False).numpy()

        aLT_cp = np.concatenate([[0.0], aLT[coarse_idx]])
        aLT_interp = np.interp(roa_dom, roa_cp, aLT_cp)

        r_b = torch.from_numpy(roa_dom).unsqueeze(0).double()
        z_b = torch.from_numpy(aLT_interp).unsqueeze(0).double()
        T_new = CALCtools.integration_Lx(r_b, z_b, torch.tensor(bc_value).double())[0].numpy()

        results.append((aLT_interp.copy(), T_new.copy()))
        current_T = T_new

    return roa_dom, profile_arr[: ir_last + 1], results


pass_colors = ["k", "#1f77b4", "#ff7f0e", "#d62728"]
pass_styles = ["-", "--", "-.", ":"]
pass_labels = ["Original", "1 pass", "2 passes", "3 passes"]

fig = fn.add_figure(label="Multi-Pass Degradation")
grid = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.3)
fig.suptitle("Profile degradation through repeated coarse-grid roundtrips (BC fixed at last control point)", fontsize=11)

for method_row, (deriv_func, method_name) in enumerate([
    (CALCtools.derivation_into_Lx_central, "Central diff"),
    (CALCtools.derivation_into_Lx, "Lagrange (GACODE-matched)"),
]):
    for prof_idx in range(3):
        row = method_row * 3 + prof_idx

        roa_dom, p_orig, passes = multi_pass_coarse_roundtrip(
            roa_hires, hires_profiles[prof_idx], coarse_indices_hires, deriv_func, n_passes=3)
        rho_dom = np.interp(roa_dom, roa_hires, rho_hires)

        # Also compute the original a/LT on the domain for comparison
        r_t = torch.from_numpy(roa_dom).double()
        p_t = torch.from_numpy(p_orig).double()
        aLT_orig = deriv_func(r_t, p_t, array=False).numpy()

        # -- Left: a/LT --
        ax = fig.add_subplot(grid[row, 0])
        ax.plot(rho_dom, aLT_orig, pass_colors[0], ls=pass_styles[0], lw=2, label=pass_labels[0])
        for ip, (aLT_p, _) in enumerate(passes):
            ax.plot(rho_dom, aLT_p, pass_colors[ip + 1], ls=pass_styles[ip + 1], lw=1.5,
                    label=pass_labels[ip + 1])
        # Mark coarse control points on original
        rho_cp = rho_hires[coarse_indices_hires]
        ax.plot(rho_cp, aLT_orig[coarse_indices_hires], "ks", ms=5, zorder=5)
        ax.set_ylabel(f"a/L {profile_short[prof_idx]}")
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"{method_name} - a/L{profile_short[prof_idx]}")
        ax.grid(True, alpha=0.3)

        # -- Right: T --
        ax = fig.add_subplot(grid[row, 1])
        ax.plot(rho_dom, p_orig, pass_colors[0], ls=pass_styles[0], lw=2, label=pass_labels[0])
        for ip, (_, T_p) in enumerate(passes):
            err_max = np.max(relative_error(p_orig, T_p)[1:])
            ax.plot(rho_dom, T_p, pass_colors[ip + 1], ls=pass_styles[ip + 1], lw=1.5,
                    label=f"{pass_labels[ip + 1]} (err={err_max:.2f}%)")
        ax.plot(rho_cp, p_orig[coarse_indices_hires], "ks", ms=5, zorder=5)
        ax.axvline(rho_dom[-1], color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(profile_labels[prof_idx])
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"{method_name} - {profile_short[prof_idx]}")
        ax.grid(True, alpha=0.3)


# ============================================================================
# Print summary and show
# ============================================================================

print("\n" + "=" * 90)
print("ROUNDTRIP ERROR SUMMARY (Max Relative Error %)")
print("=" * 90)
print(f"{'Profile':<8} {'Fine Lag':>10} {'Fine Cen':>10} {'Fine200 Lag':>12} {'Fine200 Cen':>12} {'Coarse Lag':>12} {'Coarse Cen':>12}")
print("-" * 90)
for i, name in enumerate(profile_short):
    print(f"{name:<8} {all_errors_lag[0][i]:>9.4f}% {all_errors_cen[0][i]:>9.4f}% "
          f"{all_errors_lag[1][i]:>11.4f}% {all_errors_cen[1][i]:>11.4f}% "
          f"{all_errors_lag[2][i]:>11.4f}% {all_errors_cen[2][i]:>11.4f}%")
print("=" * 90)

fn.show()
