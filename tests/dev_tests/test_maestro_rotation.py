"""
DEV TEST: MAESTRO rotation flow (TRANSP -> PORTALS -> TRANSP -> PORTALS)
-----------------------------------------------------------------------
Exercises and lets you MONITOR the toroidal rotation (w0) as it moves through a
MAESTRO chain, using the rotation plumbing added alongside this test:
  - PORTALS predicts w0 (rotation added to predicted_channels), so each PORTALS
    beat evolves the rotation profile.
  - each TRANSP beat passes the incoming w0 INTO TRANSP as the 'omg' U-File
    (gacode_state.to_transp auto-writes it when w0 != 0) and writes the TRANSP
    rotation back out to w0 (OMEGA, with the NCLASS neoclassical Er/omega now in
    the CDF by default).
So rotation should flow: (seed w0=0) -> PORTALS predicts a w0 -> next TRANSP
ingests it -> next PORTALS evolves it. The point is to WATCH that propagation,
not to converge it.

*** WARNING ***: both the TRANSP flattop and the PORTALS iteration cap are cut
to the bone here ONLY so the chain finishes fast enough to inspect. These are
FAR too short for converged physics — do not read the numbers as results.

*** REQUIREMENTS ***: the "transp" machine in config_user.json (TRANSP runs) and
TGLF/NEO for the PORTALS beats (same dependencies as maestro_01_run.py).

Instead of the full per-beat MAESTRO plots, this script ends with a focused
ROTATION-FLOW analysis (a FigureNotebook) that follows the rotation end to end:
  - TRANSP: the angular-rotation "versions" (input omg U-File, the OMEGA TRANSP
    used / wrote to input.gacode, and the NCLASS neoclassical OMEGA_NC) and the
    neoclassical E_r decomposition into its sources (pressure / toroidal / poloidal);
  - the w0 profile propagated across every beat output;
  - PORTALS: the predicted w0 turned into the E×B shear (VEXB_SHEAR) that TGLF
    actually receives, scattered at the TGLF prediction radii.
(You can still get the complete per-beat detail with `mitim_plot_maestro <folder> --beats 4`.)
"""

import numpy as np
import torch

from mitim_modules.maestro.scripts import run_maestro
from mitim_modules.maestro.utils import MAESTROplot
from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.misc_tools.GUItools import FigureNotebook

cold_start = False
folder = __mitimroot__ / "tests" / "scratch" / "dev_maestro_rotation"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(8)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Namelist: template + in-situ edits
# ---------------------------------------------------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(template)

# Constant-BC initialization (FreeGS + fixed_bc), as in maestro_01 — avoids the EPED
# dependency so the chain is just TRANSP + PORTALS.
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0

# The chain to monitor: TRANSP -> PORTALS -> TRANSP -> PORTALS. Each beat type appears
# twice and shares its single config block below.
nml["maestro"]["beats"] = ["transp", "portals", "transp", "portals"]

# --- PORTALS beats: PREDICT ROTATION (add w0), kept very short --------------------------
pp = nml["maestro"]["portals"]["parameters_prepare"]["portals_parameters"]
pp["solution"]["predicted_roa"] = [0.4, 0.6, 0.8]
pp["solution"]["predicted_channels"] = ["te", "ti", "ne", "w0"]   # <-- rotation is now predicted
pp.setdefault("optimization_options", {}).setdefault("convergence_options", {})["maximum_iterations"] = 2

# --- TRANSP beats: short flattop (see the WARNING) --------------------------------------
# NUBEAM/TORIC still dominate the wall time; the flattop is the main length knob.
nml["maestro"]["transp"]["parameters_prepare"]["flattop_window"] = 0.5

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run the chain
# ---------------------------------------------------------------------------------------------------------------------

m = run_maestro.run_maestro_local(
    namelist_file,
    folder=folder,
    terminal_outputs=True,
    force_cold_start=cold_start,
    cpus=8,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Monitor the rotation across beats
# ---------------------------------------------------------------------------------------------------------------------

objs, _, _ = MAESTROplot.collect_beat_states(m)

print("\n" + "=" * 64)
print(" Toroidal rotation w0(rad/s) across the MAESTRO chain")
print("=" * 64)
print(f" {'state':<22}{'w0(0)':>13}{'w0(rho=0.5)':>15}")
print("-" * 64)
for label, st in objs.items():
    if st is None:
        continue
    rho, w0 = st.profiles["rho(-)"], st.profiles["w0(rad/s)"]
    print(f" {label:<22}{w0[0]:>13.3e}{np.interp(0.5, rho, w0):>15.3e}")
print("=" * 64)

# ---------------------------------------------------------------------------------------------------------------------
# 4. Rotation-flow analysis figure: TRANSP sources -> input.gacode -> PORTALS w0 -> TGLF E×B shear
# ---------------------------------------------------------------------------------------------------------------------

w0_factor = 2 * np.pi * 1e3   # CDF rotation is stored as kHz; * w0_factor -> rad/s (the input.gacode w0 convention)


def transp_beats(maestro):
    '''(label, transp_output, it) per TRANSP beat. `it` is the CDF time slice that
    transp_beat.interpret() extracts and writes to the beat's input.gacode (ind_saw-1
    by default), so everything we read at `it` is exactly what the next beat ingests.'''
    out = []
    for counter, beat in maestro.beats.items():
        if not isinstance(beat, transp_beat):
            continue
        cdf, _ = beat.grab_output()
        if cdf is None:
            print(f"\t- [skip] TRANSP beat #{counter}: CDF not on disk", typeMsg="w")
            continue
        it = -1 if beat.extract_last_instead_of_sawtooth else cdf.ind_saw - 1
        out.append((f"TRANSP b#{counter}", cdf, it))
    return out


def portals_beats(maestro):
    '''(label, PORTALSanalyzer) per PORTALS beat.'''
    out = []
    for counter, beat in maestro.beats.items():
        if not isinstance(beat, portals_beat):
            continue
        out.append((f"PORTALS b#{counter}", PORTALSanalysis.PORTALSanalyzer.from_folder(beat.folder_output)))
    return out


def vexb_shear_from_state(state):
    '''Reproduce the E×B-shear input TGLF receives, straight from a gacode_state and
    with the SAME formula as MITIMstate.to_tglf (so this IS what TGLF got):
        gamma_E    = -dw0/dr * r/|q|                       [1/s]   (physical shearing rate)
        VEXB_SHEAR = -sign(I_t) * gamma_E * a/c_s          [c_s/a-normalized, dimensionless]
    Returns (rho, gamma_E[1/s], VEXB_SHEAR[norm]).'''
    sign_it = -np.sign(state.profiles["current(MA)"][-1])
    dw0dr   = state._deriv_gacode(state.profiles["w0(rad/s)"])                 # rad/s/m
    gamma_E = -dw0dr * state.derived["r"] / np.abs(state.profiles["q(-)"])     # 1/s
    vexb    = -sign_it * gamma_E * state.derived["a"] / state.derived["c_s"]   # normalized
    return state.profiles["rho(-)"], gamma_E, vexb


fn = FigureNotebook("MAESTRO rotation flow", geometry="1900x1000")

# --- Tab 1: TRANSP rotation sources (and the omega that goes to input.gacode) ---------------------
tb = transp_beats(m)
if tb:
    fig = fn.add_figure(label="TRANSP: rotation sources", tab_color=2)
    axs = fig.subplots(nrows=len(tb), ncols=2, squeeze=False)
    for row, (label, cdf, it) in enumerate(tb):
        x = cdf.x[it]

        ax = axs[row][0]   # the three angular-rotation "versions"
        ax.plot(x, cdf.VtorkHz_data[it] * w0_factor, c="g", lw=2, label=r"$\omega_{input}$ (omg U-File)")
        ax.plot(x, cdf.VtorkHz[it]      * w0_factor, c="b", lw=2.5, label=r"$\omega_{TRANSP}$ (OMEGA) $\rightarrow$ input.gacode")
        ax.plot(x, cdf.VtorkHz_nc[it]   * w0_factor, c="r", lw=2, ls="--", label=r"$\omega_{NCLASS}$ (OMEGA_NC)")
        ax.axhline(0, c="k", lw=0.5, ls=":")
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title(f"{label}: toroidal rotation"); ax.legend(fontsize=7, loc="best")
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[row][1]   # neoclassical Er and its additive sources
        ax.plot(x, cdf.Er_LF[it]     * 1e-3, c="k",  lw=2.5,           label=r"$E_r$ total")
        ax.plot(x, cdf.Er_p_LF[it]   * 1e-3, c="C0", lw=1.5, ls="--", label=r"$E_r$ pressure ($\nabla p$)")
        ax.plot(x, cdf.Er_tor_LF[it] * 1e-3, c="C1", lw=1.5, ls="--", label=r"$E_r$ toroidal ($v_\phi B_\theta$)")
        ax.plot(x, cdf.Er_pol_LF[it] * 1e-3, c="C3", lw=1.5, ls="--", label=r"$E_r$ poloidal ($v_\theta B_\phi$)")
        ax.axhline(0, c="k", lw=0.5, ls=":")
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$E_r$ (kV/m)")
        ax.set_title(f"{label}: neoclassical $E_r$ sources"); ax.legend(fontsize=7, loc="best")
        GRAPHICStools.addDenseAxis(ax)
    fig.suptitle(r"TRANSP — rotation 'versions' (input / TRANSP / NCLASS) and the neoclassical $E_r$ sources")
    fig.tight_layout()

# --- Tab 2: how w0 propagates across the chain ----------------------------------------------------
states = [(lab, st) for lab, st in objs.items() if st is not None]
if states:
    fig = fn.add_figure(label="Rotation propagation", tab_color=3)
    ax0, ax1 = fig.subplots(ncols=2)
    colors = GRAPHICStools.listColors()
    for k, (lab, st) in enumerate(states):
        ax0.plot(st.profiles["rho(-)"], st.profiles["w0(rad/s)"], c=colors[k], lw=2, label=lab)
    ax0.axhline(0, c="k", lw=0.5, ls=":")
    ax0.set_xlabel(r"$\rho$"); ax0.set_ylabel(r"$w_0$ (rad/s)")
    ax0.set_title("w0 profile written by each beat"); ax0.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax0)

    labs = [lab for lab, _ in states]
    w0_axis = [st.profiles["w0(rad/s)"][0] for _, st in states]
    w0_half = [np.interp(0.5, st.profiles["rho(-)"], st.profiles["w0(rad/s)"]) for _, st in states]
    ax1.plot(range(len(labs)), w0_axis, "-o", c="C0", label=r"$w_0(\rho=0)$")
    ax1.plot(range(len(labs)), w0_half, "-s", c="C1", label=r"$w_0(\rho=0.5)$")
    ax1.axhline(0, c="k", lw=0.5, ls=":")
    ax1.set_xticks(range(len(labs))); ax1.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel(r"$w_0$ (rad/s)"); ax1.set_title("rotation passed along the chain")
    ax1.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax1)
    fig.tight_layout()

# --- Tab 3: PORTALS predicted w0 -> the E×B shear TGLF actually used -------------------------------
pb = portals_beats(m)
obj_by_label = {lab: st for lab, st in objs.items() if st is not None}
if pb:
    fig = fn.add_figure(label="PORTALS: rotation -> TGLF E×B shear", tab_color=1)
    axs = fig.subplots(nrows=len(pb), ncols=2, squeeze=False)
    for row, (label, pa) in enumerate(pb):
        st = obj_by_label.get(label)             # the beat's output input.gacode (predicted w0, fully derived)
        if st is None:
            print(f"\t- [skip] {label}: output state not available", typeMsg="w")
            continue
        rho_p = pa.rhos                           # TGLF prediction radii (rho)

        ax = axs[row][0]   # predicted rotation, with the TGLF radii marked
        rho, w0 = st.profiles["rho(-)"], st.profiles["w0(rad/s)"]
        ax.plot(rho, w0, c="b", lw=2, label=r"$w_0$ (best iter)")
        ax.plot(rho_p, np.interp(rho_p, rho, w0), "o", c="r", ms=8, label="TGLF predicted radii")
        ax.axhline(0, c="k", lw=0.5, ls=":")
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$w_0$ (rad/s)")
        ax.set_title(f"{label}: predicted rotation"); ax.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax)

        ax = axs[row][1]   # the E×B shear TGLF received, built from that w0
        rho_s, gamma_E, vexb = vexb_shear_from_state(st)
        ax.plot(rho_s, vexb, c="b", lw=2, label=r"$VEXB\_SHEAR$ (TGLF input)")
        ax.plot(rho_p, np.interp(rho_p, rho_s, vexb), "o", c="r", ms=8, label="at TGLF radii")
        ax.axhline(0, c="k", lw=0.5, ls=":")
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$VEXB\_SHEAR$  ($c_s/a$ norm.)")
        ax.set_title(f"{label}: E×B shear TGLF got from $w_0$"); ax.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax)
        axt = ax.twinx()   # physical shearing rate for context
        axt.plot(rho_s, gamma_E, c="gray", lw=1, ls=":")
        axt.set_ylabel(r"$\gamma_E=-\partial_r w_0\, r/|q|$ (1/s)", color="gray", fontsize=8)
    fig.suptitle(r"PORTALS — the predicted $w_0$ becomes the E×B shear TGLF uses at the prediction radii")
    fig.tight_layout()

fn.show()
