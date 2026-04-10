"""
Visualization of the trailing edge blending in profile_constructors_fine.

Compares the old hard-concatenation approach (if still present) with the
new Hermite blending, focusing on the junction between the reconstructed
profile and the original trailing edge.

Usage:
    python tests/test_trailing_edge.py
"""

import numpy as np
import torch
import copy
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GUItools
from mitim_modules.powertorch.utils import CALCtools, TRANSFORMtools
from mitim_modules.powertorch.physics_models import parameterizers
from mitim_modules.powertorch import STATEtools
from mitim_tools import __mitimroot__


# ============================================================================
# Load data and create powerstate
# ============================================================================

print("\nLoading input.gacode...")
profiles = PROFILEStools.gacode_state(__mitimroot__ / "tests" / "data" / "input.gacode")

rho_coarse = torch.from_numpy(np.array([0.25, 0.45, 0.65, 0.85])).to(dtype=torch.double)

print("Creating powerstate...")
s = STATEtools.powerstate(
    copy.deepcopy(profiles),
    evolution_options={
        "ProfilePredicted": ["te", "ti", "ne"],
        "rhoPredicted": rho_coarse,
    },
)

# ============================================================================
# Extract the data we need for visualization
# ============================================================================

roa_fine = s.profiles.derived["roa"]
rho_fine = s.profiles.profiles["rho(-)"]

# The profile_constructors_fine were created during powerstate init
# Let's use them to reconstruct profiles and visualize

fn = GUItools.FigureNotebook("Trailing Edge Blending", geometry="1800x950")


# ============================================================================
# Tab 1: Full profile reconstruction for each channel
# ============================================================================

channels = ["te", "ti", "ne"]
channel_labels = [r"$T_e$ (keV)", r"$T_i$ (keV)", r"$n_e$ ($10^{19}/m^3$)"]
channel_gacode = ["te(keV)", "ti(keV)", "ne(10^19/m^3)"]
channel_ions = [None, 0, None]

fig = fn.add_figure(label="Profile Reconstruction")
grid = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
fig.suptitle("Profile reconstruction via profile_constructors_fine (Hermite trailing edge)", fontsize=11)

for row, (ch, ch_label, ch_gacode, ion_idx) in enumerate(zip(channels, channel_labels, channel_gacode, channel_ions)):

    # Original profile on the fine grid
    orig = s.profiles.profiles[ch_gacode] if ion_idx is None else s.profiles.profiles[ch_gacode][:, ion_idx]

    # Reconstruct via profile_constructors_fine with current gradients
    roa_ps = s.plasma["roa"][0, :]
    aLT = s.plasma[f"aL{ch}"][0, :]

    x_rec, y_rec = s.profile_constructors_fine[ch](roa_ps, aLT)
    x_rec = x_rec.numpy()
    y_rec = y_rec[0].numpy()

    # Map to rho for plotting
    rho_rec = np.interp(x_rec, roa_fine, rho_fine)

    # Coarse profile from profile_constructors_coarse
    _, y_coarse = s.profile_constructors_coarse[ch](roa_ps.unsqueeze(0), aLT.unsqueeze(0))
    rho_coarse_pts = np.interp(roa_ps.numpy(), roa_fine, rho_fine)

    # Also compute gradients of the reconstructed profile to check for kinks
    r_rec_t = torch.from_numpy(x_rec).double()
    y_rec_t = torch.from_numpy(y_rec).double()
    aLT_rec = CALCtools.derivation_into_Lx(r_rec_t, y_rec_t, array=False).numpy()

    aLT_orig = CALCtools.derivation_into_Lx(
        torch.from_numpy(roa_fine).double(),
        torch.from_numpy(orig).double(),
        array=False
    ).numpy()

    # Left: Full profiles
    ax = fig.add_subplot(grid[row, 0])
    ax.plot(rho_fine, orig, "k-", lw=2, label="Original")
    ax.plot(rho_rec, y_rec, "r--", lw=1.5, label="Reconstructed")
    ax.plot(rho_coarse_pts, y_coarse[0].numpy(), "go", ms=6, zorder=5, label="Coarse pts")
    last_cp_rho = rho_coarse_pts[-1]
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6, label="Last CP")
    ax.set_ylabel(ch_label)
    ax.set_xlabel(r"$\rho$")
    ax.legend(fontsize=7)
    ax.set_title(f"{ch_label} - Full view")
    ax.grid(True, alpha=0.3)

    # Middle: Zoom on trailing edge
    ax = fig.add_subplot(grid[row, 1])
    ax.plot(rho_fine, orig, "k-", lw=2, label="Original")
    ax.plot(rho_rec, y_rec, "r--", lw=1.5, label="Reconstructed")
    ax.plot(rho_coarse_pts, y_coarse[0].numpy(), "go", ms=6, zorder=5)
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(last_cp_rho - 0.05, min(rho_fine[-1], last_cp_rho + 0.15))
    ax.set_ylabel(ch_label)
    ax.set_xlabel(r"$\rho$")
    ax.legend(fontsize=7)
    ax.set_title(f"Zoom: trailing edge")
    ax.grid(True, alpha=0.3)

    # Right: Gradients (check for kinks)
    ax = fig.add_subplot(grid[row, 2])
    ax.plot(rho_fine, aLT_orig, "k-", lw=2, label="Original a/LT")
    ax.plot(rho_rec, aLT_rec, "r--", lw=1.5, label="Reconstructed a/LT")
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6, label="Last CP")
    ax.set_xlim(last_cp_rho - 0.1, min(rho_fine[-1], last_cp_rho + 0.15))
    ax.set_ylabel(f"a/L for {ch_label}")
    ax.set_xlabel(r"$\rho$")
    ax.legend(fontsize=7)
    ax.set_title(f"Gradient near trailing edge")
    ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 2: Grid spacing and blend region detail
# ============================================================================

fig = fn.add_figure(label="Grid & Blend Detail")
grid_spec = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
fig.suptitle("Grid spacing and Hermite blend weight near trailing edge", fontsize=11)

for col, (ch, ch_label, ch_gacode, ion_idx) in enumerate(zip(channels, channel_labels, channel_gacode, channel_ions)):

    orig = s.profiles.profiles[ch_gacode] if ion_idx is None else s.profiles.profiles[ch_gacode][:, ion_idx]

    roa_ps = s.plasma["roa"][0, :]
    aLT = s.plasma[f"aL{ch}"][0, :]
    x_rec, y_rec = s.profile_constructors_fine[ch](roa_ps, aLT)
    x_rec = x_rec.numpy()
    y_rec = y_rec[0].numpy()
    rho_rec = np.interp(x_rec, roa_fine, rho_fine)
    last_cp_rho = np.interp(roa_ps[-1].item(), roa_fine, rho_fine)

    # Grid spacing
    ax = fig.add_subplot(grid_spec[0, col])
    dr = np.diff(x_rec)
    rho_mid = 0.5 * (rho_rec[:-1] + rho_rec[1:])
    ax.semilogy(rho_mid, dr, "b.-", ms=3, lw=1)
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6, label="Last CP")
    ax.set_ylabel("dr (r/a spacing)")
    ax.set_xlabel(r"$\rho$")
    ax.set_title(f"{ch} - Grid spacing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Relative error between original and reconstructed
    ax = fig.add_subplot(grid_spec[1, col])
    orig_interp = np.interp(x_rec, roa_fine, orig)
    rel_err = np.abs((y_rec - orig_interp) / (orig_interp + 1e-30)) * 100
    ax.semilogy(rho_rec[1:], rel_err[1:], "r-", lw=1.5)
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6, label="Last CP")
    ax.set_ylabel("Relative error (%)")
    ax.set_xlabel(r"$\rho$")
    ax.set_title(f"{ch} - Reconstruction error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 3: Multi-pass stability with the new trailing edge
# ============================================================================

fig = fn.add_figure(label="Multi-Pass Stability")
grid_spec = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)
fig.suptitle("Multi-pass stability: does the profile change after re-parameterization?", fontsize=11)

pass_colors = ["k", "#1f77b4", "#ff7f0e", "#d62728"]
pass_styles = ["-", "--", "-.", ":"]
pass_labels = ["Original", "1 pass", "2 passes", "3 passes"]

for row, (ch, ch_label, ch_gacode, ion_idx) in enumerate(zip(channels, channel_labels, channel_gacode, channel_ions)):

    orig_full = s.profiles.profiles[ch_gacode] if ion_idx is None else s.profiles.profiles[ch_gacode][:, ion_idx]

    roa_ps = s.plasma["roa"][0, :]
    aLT_current = s.plasma[f"aL{ch}"][0, :].clone()

    # Pass 0: reconstruct from current gradients
    x_rec, y_rec = s.profile_constructors_fine[ch](roa_ps, aLT_current)
    x_rec_np = x_rec.numpy()
    rho_rec = np.interp(x_rec_np, roa_fine, rho_fine)
    last_cp_rho = np.interp(roa_ps[-1].item(), roa_fine, rho_fine)

    profiles_passes = [y_rec[0].numpy()]
    gradients_passes = [CALCtools.derivation_into_Lx(
        torch.from_numpy(x_rec_np).double(),
        y_rec[0].double(), array=False).numpy()]

    # Subsequent passes: re-derive gradients from reconstructed profile, sample at coarse, reconstruct again
    current_profile = y_rec[0].numpy()
    for p in range(3):
        r_t = torch.from_numpy(x_rec_np).double()
        p_t = torch.from_numpy(current_profile).double()
        aLT_full = CALCtools.derivation_into_Lx(r_t, p_t, array=False).numpy()

        # Find coarse point indices on the reconstructed grid
        coarse_roa = roa_ps[1:].numpy()  # skip the axis zero
        coarse_idx = [np.argmin(np.abs(x_rec_np - cr)) for cr in coarse_roa]

        # Sample and re-inject
        aLT_new = torch.zeros_like(aLT_current)
        aLT_new[0] = 0.0  # axis
        for ci, idx in enumerate(coarse_idx):
            aLT_new[ci + 1] = aLT_full[idx]

        x_new, y_new = s.profile_constructors_fine[ch](roa_ps, aLT_new)
        current_profile = y_new[0].numpy()
        profiles_passes.append(current_profile)
        gradients_passes.append(CALCtools.derivation_into_Lx(
            torch.from_numpy(x_rec_np).double(),
            torch.from_numpy(current_profile).double(), array=False).numpy())

    # Left: Profiles
    ax = fig.add_subplot(grid_spec[row, 0])
    ax.plot(rho_fine, orig_full, "k-", lw=2, alpha=0.3, label="Original gacode")
    for ip, (yp, lab, col_p, sty) in enumerate(zip(profiles_passes, pass_labels, pass_colors, pass_styles)):
        if ip == 0:
            lab_full = f"Pass 0 (first reconstruction)"
        else:
            err = np.max(np.abs((yp - profiles_passes[0]) / (profiles_passes[0] + 1e-30))) * 100
            lab_full = f"Pass {ip} (max diff={err:.3f}%)"
        ax.plot(rho_rec, yp, color=col_p, ls=sty, lw=1.5, label=lab_full)
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylabel(ch_label)
    ax.set_xlabel(r"$\rho$")
    ax.legend(fontsize=6, loc="best")
    ax.set_title(f"{ch} profiles")
    ax.grid(True, alpha=0.3)

    # Right: Gradients near trailing edge
    ax = fig.add_subplot(grid_spec[row, 1])
    ax.plot(rho_fine, CALCtools.derivation_into_Lx(
        torch.from_numpy(roa_fine).double(),
        torch.from_numpy(orig_full).double(), array=False).numpy(),
        "k-", lw=2, alpha=0.3, label="Original gacode")
    for ip, (gp, lab, col_p, sty) in enumerate(zip(gradients_passes, pass_labels, pass_colors, pass_styles)):
        ax.plot(rho_rec, gp, color=col_p, ls=sty, lw=1.5, label=pass_labels[ip])
    ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlim(last_cp_rho - 0.15, min(rho_fine[-1], last_cp_rho + 0.15))
    ax.set_ylabel(f"a/L {ch}")
    ax.set_xlabel(r"$\rho$")
    ax.legend(fontsize=6, loc="best")
    ax.set_title(f"{ch} gradients near trailing edge")
    ax.grid(True, alpha=0.3)


# ============================================================================
# Tab 4: Comparison with the SPARC case (if available)
# ============================================================================

from pathlib import Path
sparc_path = Path("/Users/pablorf/PROJECTS/project_2026_Development/development_portals/roundtrip/sparc_3.7")
if (sparc_path / "Execution/Evaluation.0/transport_simulation_folder/input.gacode").exists():

    fig = fn.add_figure(label="SPARC Case")
    grid_spec = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle("SPARC case: profile reconstruction with Hermite trailing edge", fontsize=11)

    p_sparc = PROFILEStools.gacode_state(sparc_path / "Execution/Evaluation.0/transport_simulation_folder/input.gacode")

    rho_sparc = torch.from_numpy(np.array([0.302734, 0.478464, 0.668839, 0.808734, 0.840279])).to(dtype=torch.double)

    s_sparc = STATEtools.powerstate(
        copy.deepcopy(p_sparc),
        evolution_options={
            "ProfilePredicted": ["te", "ti", "ne"],
            "rhoPredicted": rho_sparc,
        },
        increase_profile_resol=False,  # already at enhanced resolution
    )

    roa_sparc = s_sparc.profiles.derived["roa"]
    rho_sparc_fine = s_sparc.profiles.profiles["rho(-)"]

    for col, (ch, ch_label, ch_gacode, ion_idx) in enumerate(zip(channels, channel_labels, channel_gacode, channel_ions)):

        orig = p_sparc.profiles[ch_gacode] if ion_idx is None else p_sparc.profiles[ch_gacode][:, ion_idx]
        roa_ps = s_sparc.plasma["roa"][0, :]
        aLT = s_sparc.plasma[f"aL{ch}"][0, :]
        x_rec, y_rec = s_sparc.profile_constructors_fine[ch](roa_ps, aLT)
        x_rec_np = x_rec.numpy()
        y_rec_np = y_rec[0].numpy()
        rho_rec = np.interp(x_rec_np, roa_sparc, rho_sparc_fine)
        last_cp_rho = np.interp(roa_ps[-1].item(), roa_sparc, rho_sparc_fine)

        # Profile
        ax = fig.add_subplot(grid_spec[0, col])
        ax.plot(rho_sparc_fine, orig, "k-", lw=2, label="Original")
        ax.plot(rho_rec, y_rec_np, "r--", lw=1.5, label="Reconstructed")
        ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(last_cp_rho - 0.05, min(rho_sparc_fine[-1], last_cp_rho + 0.15))
        ax.set_ylabel(ch_label)
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=8)
        ax.set_title(f"SPARC {ch} trailing edge")
        ax.grid(True, alpha=0.3)

        # Gradient
        ax = fig.add_subplot(grid_spec[1, col])
        aLT_orig = CALCtools.derivation_into_Lx(
            torch.from_numpy(roa_sparc).double(),
            torch.from_numpy(orig).double(), array=False).numpy()
        aLT_rec = CALCtools.derivation_into_Lx(
            torch.from_numpy(x_rec_np).double(),
            torch.from_numpy(y_rec_np).double(), array=False).numpy()
        ax.plot(rho_sparc_fine, aLT_orig, "k-", lw=2, label="Original a/LT")
        ax.plot(rho_rec, aLT_rec, "r--", lw=1.5, label="Reconstructed a/LT")
        ax.axvline(last_cp_rho, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(last_cp_rho - 0.1, min(rho_sparc_fine[-1], last_cp_rho + 0.15))
        ax.set_ylabel(f"a/L {ch}")
        ax.set_xlabel(r"$\rho$")
        ax.legend(fontsize=8)
        ax.set_title(f"SPARC {ch} gradient")
        ax.grid(True, alpha=0.3)


# ============================================================================
# Show
# ============================================================================

fn.show()
