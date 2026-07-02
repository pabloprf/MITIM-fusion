"""
DEV TEST: MAESTRO rotation flow — TWO chains compared (zero seed vs strong w0 seed)
-----------------------------------------------------------------------------------
Both chains are TRANSP -> PORTALS -> TRANSP -> PORTALS with w0 added to PORTALS'
predicted_channels, under the default rotation_source='echo' (PASS-THROUGH): each
TRANSP beat feeds the incoming w0 to TRANSP as the 'omg' U-File (rotation modeling +
NCLASS Er diagnostics in the CDF) and carries the SAME w0 out unchanged. The chain's
rotation therefore changes ONLY across PORTALS beats, which predict it.

  RUN A ("w0=0")  : constant-BC init (FreeGS + fixed_bc) -> rotation SEEDED AT ZERO.
                    The first TRANSP beat must hand w0=0 to PORTALS (pass-through of a
                    zero seed — NCLASS's OMEGA_NC stays in the CDF as a diagnostic, it
                    is NOT written out under 'echo'). PORTALS then evolves w0 from
                    zero, and the second TRANSP beat must carry that predicted
                    rotation through untouched.

  RUN B ("w0!=0") : same chain but the initial input.gacode is SEEDED with a strong,
                    artificial rotation (~1e5 rad/s on axis, ~20x the neoclassical
                    scale and opposite sign), so pass-through vs re-derivation is
                    unambiguous: the first TRANSP beat must hand EXACTLY this seed
                    to PORTALS.

Point of the comparison: w0 must change ONLY across PORTALS beats (which predict it)
and NEVER across a TRANSP beat (pass-through contract of 'echo'). The TRANSP tabs also
show what TRANSP would have written instead — OMEGA, OMEGA_NC, and the E×B rotation
that rotation_source='neoclassical_transp' writes — for reference.

*** WARNING ***: TRANSP flattop and PORTALS iteration cap are cut to the bone here ONLY
so the chains finish fast enough to inspect, and Run B's w0 seed is artificial — do NOT
read any number as physics.

*** REQUIREMENTS ***: the "transp" machine in config_user.json (TRANSP) and TGLF/NEO for
the PORTALS beats (same dependencies as maestro_01_run.py).

The script ends with a rotation-flow FigureNotebook: a seed-comparison tab (w0 across
every beat output, side by side) plus, per run, the TRANSP rotation 'versions' (input
omg = what 'echo' carries out / OMEGA / NCLASS OMEGA_NC / the E×B rotation) with the
neoclassical Er sources, and the PORTALS predicted w0 -> VEXB_SHEAR that TGLF receives
at the prediction radii.
(Full per-beat detail is still available via `mitim_plot_maestro <folder> --beats 4`.)
"""

import numpy as np
import torch

from mitim_modules.maestro.scripts import run_maestro
from mitim_modules.maestro.utils import MAESTROplot
from mitim_modules.maestro.utils.TRANSPbeat import transp_beat, _extraction_index
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.misc_tools.GUItools import FigureNotebook

cold_start = False
torch.set_num_threads(8)

root = __mitimroot__ / "tests" / "scratch"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"
w0_factor = 2 * np.pi * 1e3   # CDF rotation is stored as kHz; * w0_factor -> rad/s (input.gacode w0 convention)


# =====================================================================================
# Namelist / seed builders
# =====================================================================================

def base_namelist():
    '''Common edits for both chains: the beat chain, predicted channels (incl. w0),
    and the deliberately short iteration/flattop caps.'''
    nml = IOtools.read_mitim_yaml(template)
    nml["maestro"]["beats"] = ["transp", "portals", "transp", "portals"]
    pp = nml["maestro"]["portals"]["parameters_prepare"]["portals_parameters"]
    pp["solution"]["predicted_roa"] = [0.4, 0.6, 0.8]
    pp["solution"]["predicted_channels"] = ["te", "ti", "ne", "w0"]   # rotation is predicted
    pp.setdefault("optimization_options", {}).setdefault("convergence_options", {})["maximum_iterations"] = 2
    nml["maestro"]["transp"]["parameters_prepare"]["flattop_window"] = 0.5
    return nml


def make_rotating_seed(base_gacode, out_path, w0_axis=1.0e5):
    '''Write an input.gacode identical to `base_gacode` but with a strong, peaked-on-axis
    toroidal rotation injected: w0(rho) = w0_axis * (1 - rho^2). Artificial — chosen large
    (~20x neoclassical) and opposite-sign so propagation vs overwrite is unambiguous.'''
    st = PROFILEStools.gacode_state(base_gacode)
    rho = st.profiles["rho(-)"]
    st.profiles["w0(rad/s)"] = w0_axis * (1.0 - rho ** 2)
    st.derive_quantities(rederiveGeometry=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_state(file=out_path)
    return out_path


def launch(nml, folder):
    if cold_start and folder.exists():
        IOtools.shutil_rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nml_file = folder / "namelist.maestro.yaml"
    IOtools.write_mitim_yaml(nml, nml_file)
    return run_maestro.run_maestro_local(
        nml_file, folder=folder, terminal_outputs=True, force_cold_start=cold_start, cpus=8)


# =====================================================================================
# Analysis helpers
# =====================================================================================

def transp_beats(maestro):
    '''(label, transp_output, it) per TRANSP beat. `it` is the CDF slice that
    transp_beat.finalize() extracts (extract_at grammar; the flattop floor is ignored
    here — close enough for these display tabs).'''
    out = []
    for counter, beat in maestro.beats.items():
        if not isinstance(beat, transp_beat):
            continue
        cdf, _ = beat.grab_output()
        if cdf is None:
            print(f"\t- [skip] TRANSP beat #{counter}: CDF not on disk", typeMsg="w")
            continue
        it = _extraction_index(cdf, getattr(beat, 'extract_at', 'saw-1'))
        out.append((f"TRANSP b#{counter}", cdf, it))
    return out


def portals_beats(maestro):
    '''(label, PORTALSanalyzer) per PORTALS beat.'''
    out = []
    for counter, beat in maestro.beats.items():
        if isinstance(beat, portals_beat):
            out.append((f"PORTALS b#{counter}", PORTALSanalysis.PORTALSanalyzer.from_folder(beat.folder_output)))
    return out


def vexb_shear_from_state(state):
    '''The E×B-shear input TGLF receives, from a gacode_state with the SAME formula as
    MITIMstate.to_tglf: gamma_E = -dw0/dr * r/|q| [1/s]; VEXB_SHEAR = -sign(I_t)*gamma_E*a/c_s
    (c_s/a-normalized). Returns (rho, gamma_E[1/s], VEXB_SHEAR[norm]).'''
    sign_it = -np.sign(state.profiles["current(MA)"][-1])
    dw0dr   = state._deriv_gacode(state.profiles["w0(rad/s)"])
    gamma_E = -dw0dr * state.derived["r"] / np.abs(state.profiles["q(-)"])
    vexb    = -sign_it * gamma_E * state.derived["a"] / state.derived["c_s"]
    return state.profiles["rho(-)"], gamma_E, vexb


def print_w0_table(objs, title):
    print("\n" + "=" * 64)
    print(f" {title}")
    print("=" * 64)
    print(f" {'state':<22}{'w0(0)':>13}{'w0(rho=0.5)':>15}")
    print("-" * 64)
    for label, st in objs.items():
        if st is None:
            continue
        rho, w0 = st.profiles["rho(-)"], st.profiles["w0(rad/s)"]
        print(f" {label:<22}{w0[0]:>13.3e}{np.interp(0.5, rho, w0):>15.3e}")
    print("=" * 64)


def add_comparison_tab(fn, runs):
    '''runs = [(tag, objs), ...]. One panel per run: w0(rho) for every beat output.'''
    fig = fn.add_figure(label="Rotation propagation: seed comparison", tab_color=3)
    axs = np.atleast_1d(fig.subplots(ncols=len(runs), squeeze=False)[0])
    for ax, (tag, objs) in zip(axs, runs):
        colors = GRAPHICStools.listColors()
        for k, (lab, st) in enumerate((l, s) for l, s in objs.items() if s is not None):
            ax.plot(st.profiles["rho(-)"], st.profiles["w0(rad/s)"], c=colors[k], lw=2, label=lab)
        ax.axhline(0, c="k", lw=0.5, ls=":")
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$w_0$ (rad/s)")
        ax.set_title(f"seed: {tag}"); ax.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax)
    fig.suptitle(r"w0 propagated across the chain — zero seed vs strong non-zero seed")
    fig.tight_layout()


def add_run_tabs(fn, maestro, objs, tag):
    '''Per-run detail tabs: TRANSP rotation 'versions' + Er sources, and PORTALS -> TGLF E×B shear.'''
    tb = transp_beats(maestro)
    if tb:
        fig = fn.add_figure(label=f"TRANSP sources — {tag}", tab_color=2)
        axs = fig.subplots(nrows=len(tb), ncols=2, squeeze=False)
        for row, (label, cdf, it) in enumerate(tb):
            x = cdf.x[it]
            ax = axs[row][0]
            ax.plot(x, cdf.VtorkHz_data[it] * w0_factor, c="g", lw=2, label=r"$\omega_{input}$ (omg U-File) $\rightarrow$ carried out ('echo')")
            ax.plot(x, cdf.VtorkHz[it]      * w0_factor, c="b", lw=2.5, label=r"$\omega_{TRANSP}$ (OMEGA, adopted)")
            ax.plot(x, cdf.VtorkHz_nc[it]   * w0_factor, c="r", lw=2, ls="--", label=r"$\omega_{NCLASS}$ (OMEGA_NC)")
            ax.plot(x, cdf.TGLF_w0_exb[it], c="m", lw=1.5, ls=":", label=r"$\omega_{E\times B}$ ($E_r/(d\psi/dR)$) $\rightarrow$ 'neoclassical_transp'")
            ax.axhline(0, c="k", lw=0.5, ls=":")
            ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\omega$ (rad/s)")
            ax.set_title(f"{label}: toroidal rotation"); ax.legend(fontsize=7, loc="best")
            GRAPHICStools.addDenseAxis(ax)
            ax = axs[row][1]
            ax.plot(x, cdf.Er_LF[it]     * 1e-3, c="k",  lw=2.5,           label=r"$E_r$ total")
            ax.plot(x, cdf.Er_p_LF[it]   * 1e-3, c="C0", lw=1.5, ls="--", label=r"$E_r$ pressure ($\nabla p$)")
            ax.plot(x, cdf.Er_tor_LF[it] * 1e-3, c="C1", lw=1.5, ls="--", label=r"$E_r$ toroidal ($v_\phi B_\theta$)")
            ax.plot(x, cdf.Er_pol_LF[it] * 1e-3, c="C3", lw=1.5, ls="--", label=r"$E_r$ poloidal ($v_\theta B_\phi$)")
            ax.axhline(0, c="k", lw=0.5, ls=":")
            ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$E_r$ (kV/m)")
            ax.set_title(f"{label}: neoclassical $E_r$ sources"); ax.legend(fontsize=7, loc="best")
            GRAPHICStools.addDenseAxis(ax)
        fig.suptitle(f"TRANSP rotation 'versions' & neoclassical $E_r$ sources — {tag} seed")
        fig.tight_layout()

    pb = portals_beats(maestro)
    obj_by_label = {lab: st for lab, st in objs.items() if st is not None}
    if pb:
        fig = fn.add_figure(label=f"PORTALS E×B — {tag}", tab_color=1)
        axs = fig.subplots(nrows=len(pb), ncols=2, squeeze=False)
        for row, (label, pa) in enumerate(pb):
            st = obj_by_label.get(label)
            if st is None:
                print(f"\t- [skip] {label}: output state not available", typeMsg="w")
                continue
            rho_p = pa.rhos
            ax = axs[row][0]
            rho, w0 = st.profiles["rho(-)"], st.profiles["w0(rad/s)"]
            ax.plot(rho, w0, c="b", lw=2, label=r"$w_0$ (best iter)")
            ax.plot(rho_p, np.interp(rho_p, rho, w0), "o", c="r", ms=8, label="TGLF predicted radii")
            ax.axhline(0, c="k", lw=0.5, ls=":")
            ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$w_0$ (rad/s)")
            ax.set_title(f"{label}: predicted rotation"); ax.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax)
            ax = axs[row][1]
            rho_s, gamma_E, vexb = vexb_shear_from_state(st)
            ax.plot(rho_s, vexb, c="b", lw=2, label=r"$VEXB\_SHEAR$ (TGLF input)")
            ax.plot(rho_p, np.interp(rho_p, rho_s, vexb), "o", c="r", ms=8, label="at TGLF radii")
            ax.axhline(0, c="k", lw=0.5, ls=":")
            ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$VEXB\_SHEAR$  ($c_s/a$ norm.)")
            ax.set_title(f"{label}: E×B shear TGLF got from $w_0$"); ax.legend(fontsize=8); GRAPHICStools.addDenseAxis(ax)
            axt = ax.twinx()
            axt.plot(rho_s, gamma_E, c="gray", lw=1, ls=":")
            axt.set_ylabel(r"$\gamma_E=-\partial_r w_0\, r/|q|$ (1/s)", color="gray", fontsize=8)
        fig.suptitle(f"PORTALS predicted $w_0$ -> TGLF E×B shear at the prediction radii — {tag} seed")
        fig.tight_layout()


# =====================================================================================
# RUN A: original chain, constant-BC init -> seed w0 = 0
# =====================================================================================

folderA = root / "dev_maestro_rotation"
nmlA = base_namelist()
nmlA["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nmlA["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nmlA["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0
mA = launch(nmlA, folderA)

# =====================================================================================
# RUN B: same chain, profiles init from an input.gacode seeded with a strong w0
# =====================================================================================

folderB = root / "dev_maestro_rotation_seeded"
if cold_start and folderB.exists():
    IOtools.shutil_rmtree(folderB)
folderB.mkdir(parents=True, exist_ok=True)
seed_file = make_rotating_seed(
    __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD", folderB / "seed_input.gacode")
nmlB = base_namelist()
nmlB["plasma"]["profiles_initialization"]["initialization_type"] = "profiles"
nmlB["plasma"]["profiles_initialization"]["creator_type"] = None   # use the seed file's profiles as-is (no creator beat); must be None, not "null"
nmlB["plasma"]["profiles_initialization"]["parameters"]["profiles_file"] = str(seed_file)
mB = launch(nmlB, folderB)

# =====================================================================================
# Monitor + plot both chains
# =====================================================================================

objsA, _, _ = MAESTROplot.collect_beat_states(mA)
objsB, _, _ = MAESTROplot.collect_beat_states(mB)
print_w0_table(objsA, "Toroidal rotation w0(rad/s) — RUN A (zero seed)")
print_w0_table(objsB, "Toroidal rotation w0(rad/s) — RUN B (strong w0 seed)")

fn = FigureNotebook("MAESTRO rotation flow (seed comparison)", geometry="1900x1000")
add_comparison_tab(fn, [("w0=0", objsA), ("w0!=0", objsB)])
add_run_tabs(fn, mA, objsA, "w0=0")
add_run_tabs(fn, mB, objsB, "w0!=0")
fn.show()
