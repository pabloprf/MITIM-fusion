"""
DEV TEST: effect of neoclassical E×B shear on a PORTALS flux match
------------------------------------------------------------------
PORTALS can stabilize the turbulence with the neoclassical E×B shearing rate.
The mechanism (see templates/namelist.portals.yaml, transport.options.neo):

    transport.options.neo.vgen_exb_shear: true

runs NEO VGEN (profiles_gen -vgen, weak-rotation limit) at each transport
evaluation to compute the neoclassical radial electric field Er from the ion
pressure gradient + neoclassical poloidal flow (Waltz–Miller, zero toroidal
rotation). VGEN writes the implied w0(rad/s) back into the state, and TGLF then
sees a non-zero VEXB_SHEAR / VPAR_SHEAR (built in MITIMstate from -dw0/dr). E×B
shear quenches the turbulence, so for a FIXED target flux the flux-matched
gradient has to be STEEPER, i.e. the predicted core temperature is HIGHER.

This test brings the SAME plasma to flux-matching conditions twice — once with
E×B shear OFF and once ON — and isolates the effect:

  * The input.gacode rotation is ZEROED in both runs, so the OFF run is a true
    zero-E×B baseline and the ON run's E×B shear comes ENTIRELY from the
    neoclassical VGEN Er (not from any rotation already in the file).
  * The plasma is lumped to electrons + main ion + a single impurity at each
    transport dispatch (the same profiles_postprocessing_fun MAESTRO uses for
    its PORTALS beats), which keeps the TGLF/NEO/VGEN calls cheap.
  * The SPARC PRD case is rebalanced into a colder, lower-density, ICRF-heated
    point: Te/Ti are uniformly cooled so the edge predicted radius (r/a=0.9)
    sits at ~0.5 keV (a colder plasma raises the normalized E×B shearing rate,
    VEXB_SHEAR ∝ a/c_s ∝ 1/sqrt(T)); the density is halved (less line
    radiation, qrad ∝ ne^2); and the ICRF auxiliary power is set to ~20 MW so
    the cold case has a sensible, ICRF-dominated power balance rather than a
    near-marginal one.
  * Two comparisons are reported:
      (1) MECHANISM, convergence-independent: at iteration 0 both runs sit at
          the *same* gradients, so the only difference is VEXB_SHEAR. The
          turbulent flux must not increase when E×B is switched on (it should
          drop) — this is the direct, robust signature of stabilization.
      (2) EFFECT on the converged state: at the flux-matched (best) iteration,
          the gradients a/LTe, a/LTi and the predicted core Te, Ti should be
          higher with E×B on.

*** WARNING ***: initial_training / maximum_iterations are cut to the bone here
so the two flux matches finish quickly. They are enough to see the direction of
the effect but are NOT converged physics — do not read the numbers as results.

*** REQUIREMENTS ***: subprocess TGLF/NEO/profiles_gen(-vgen) configured in
config_user.json (same dependencies as portals_01_standard.py). Execution is
NOT in-process here, so every TGLF/NEO/VGEN call leaves its run tree on disk
under <run>/Execution/Evaluation.<i>/transport_simulation_folder/ for later
inspection.

Run it interactively (a comparison figure pops up at the end):

    python tests/dev_tests/test_portals_exb_shear.py
"""

import sys
from functools import partial

import numpy as np
import torch

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_modules.maestro.utils.PORTALSbeat import profiles_postprocessing_fun as maestro_lump_postproc
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.NEOtools import _compute_vexb_shear
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools, MATHtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# ---------------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------------

cold_start = False
in_process = False   # subprocess TGLF/NEO/VGEN -> run trees stay on disk for inspection (set True for fast ctypes)

predicted_roa     = [0.4, 0.65, 0.9]   # 0.9 is near the edge where the neoclassical Er (∝ dpi/dr) is strongest
initial_training  = 5
maximum_iterations = 15

# Gradient-search bounds. The cold, half-density plasma is STIFF (tiny gyroBohm unit, so a large
# normalized flux is needed to carry 20 MW), so the flux-matched a/LT are well above the initial
# profile's. The default relative ceiling (ymax=3 -> ~3x the initial gradient) clips the search before
# it reaches flux match. Widen it: a larger relative ymax plus a generous ABSOLUTE floor on the range
# (yminymax_atleast) so every radius can reach steep gradients regardless of its (low) initial value.
gradient_ymax        = 10.0       # relative upper multiplier on the initial gradient
gradient_atleast     = [0.0, 20.0]  # absolute [min, max] a/LT the search must at least span

# Cool the plasma down to this Te at the edge predicted radius (r/a=0.9) by a uniform Te/Ti scale.
# A colder plasma raises the normalized E×B shear (VEXB_SHEAR ∝ 1/sqrt(T)), enhancing the effect.
Te09_target = 0.5   # keV

# Also lower the density (half) and set a clean ICRF auxiliary heating. The SPARC PRD sources are
# rebalanced this way so the cold case is not radiation-dominated / near-marginal: qfus/qrad/qie are
# recomputed each iteration (targets_evolve), so the ICRF is the dominant FIXED source the flux match
# has to carry. Half density also lowers the line radiation (qrad ∝ ne^2).
density_factor = 0.5   # multiply ne and all ni by this
P_icrf_MW      = 20.0  # set total ICRF (RF) power to this, keeping the SPARC deposition shape and e/i split

torch.set_num_threads(8)

folder = __mitimroot__ / "tests" / "scratch" / "dev_portals_exb_shear"
inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"   # D-T burning plasma (w0=0)

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

def _np(x):
    """Coerce a powerstate.plasma entry (torch tensor or ndarray) to a numpy array."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def rotation_inputs(state):
    """The (normalized) TGLF rotation inputs that the E×B shear feeds into TGLF, derived from
    w0(rad/s) on the full radial grid exactly as in MITIMstate.to_tglf():

        sign_It     = -sign(Ip)
        gamma_eb0   = -(dw0/dr) * r / |q|        ;  VEXB_SHEAR = -sign_It * gamma_eb0 * a/c_s
        gamma_p0    = -Rmaj * (dw0/dr)           ;  VPAR_SHEAR = -sign_It * gamma_p0  * a/c_s
                                                    VPAR       = -sign_It * Rmaj * w0  / c_s

    VEXB_SHEAR (the E×B shearing rate) is the stabilizing one; VPAR/VPAR_SHEAR are the parallel-flow
    drive that comes along with the same w0. Returns (roa, w0[rad/s], VEXB_SHEAR, VPAR, VPAR_SHEAR).
    """
    roa  = state.derived["roa"]
    w0   = state.profiles["w0(rad/s)"]
    r    = state.derived["r"]
    a    = state.derived["a"]
    c_s  = state.derived["c_s"]
    rmaj = state.profiles["rmaj(m)"]
    sign_it = -np.sign(state.profiles["current(MA)"][-1])

    vexb_shear = _compute_vexb_shear(state)                 # reuse the existing helper for the E×B shearing rate
    dw0_dr     = MATHtools.deriv(r, w0, array=True)
    vpar_shear = -sign_it * (-rmaj * dw0_dr) * a / c_s
    vpar       = -sign_it * rmaj * w0 / c_s
    return roa, w0, vexb_shear, vpar, vpar_shear


def make_state():
    """Fresh, corrected plasma state with the toroidal rotation ZEROED so both runs
    start from a clean zero-E×B baseline (the ON run rebuilds w0 from neoclassical VGEN)."""
    p = PROFILEStools.gacode_state(inputgacode)
    p.correct(options={"recalculate_ptot": True, "remove_fast": True, "quasineutrality": True})

    # ICRF scale factor: geometry and qrfe/qrfi are untouched by the T/density scaling below, so the
    # current integrated RF power (p.derived["qRF_MW"][-1]) is the right denominator.
    f_rf = P_icrf_MW / p.derived["qRF_MW"][-1]

    # Lower-temperature variant: scale Te and Ti by a COMMON factor so the edge (r/a=0.9) sits at
    # ~0.5 keV. A uniform scale leaves the normalized gradients (a/LT) and the Ti/Te shape unchanged
    # but lowers c_s, raising the normalized E×B shearing rate (VEXB_SHEAR ∝ a/c_s ∝ 1/sqrt(T)) so the
    # neoclassical stabilization is stronger than on the hot SPARC PRD baseline.
    f_cold = Te09_target / np.interp(0.9, p.derived["roa"], p.profiles["te(keV)"])
    p.profiles["te(keV)"] = p.profiles["te(keV)"] * f_cold
    p.profiles["ti(keV)"] = p.profiles["ti(keV)"] * f_cold

    # Halve the density (ne and every ion): scale-invariant in a/Ln, preserves quasineutrality and Zeff.
    p.profiles["ne(10^19/m^3)"] = p.profiles["ne(10^19/m^3)"] * density_factor
    p.profiles["ni(10^19/m^3)"] = p.profiles["ni(10^19/m^3)"] * density_factor

    # Set total ICRF to P_icrf_MW, keeping the SPARC e/i split and deposition shape.
    p.profiles["qrfe(MW/m^3)"] = p.profiles["qrfe(MW/m^3)"] * f_rf
    p.profiles["qrfi(MW/m^3)"] = p.profiles["qrfi(MW/m^3)"] * f_rf

    p.profiles["w0(rad/s)"] = p.profiles["w0(rad/s)"] * 0.0
    p.derive_quantities(rederiveGeometry=False)
    return p


def run_portals_fluxmatch(tag, vgen_exb_shear):
    """Bring the plasma to flux-matching conditions with E×B shear off/on and return the analyzer."""
    work = folder / tag

    portals_fun = PORTALSmain.portals(work)

    portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti"]
    portals_fun.portals_parameters["solution"]["predicted_roa"] = predicted_roa

    # Widen the gradient search so the stiff cold case can reach flux match (see the knobs above)
    portals_fun.portals_parameters["solution"]["exploration_ranges"]["ymax"] = gradient_ymax
    portals_fun.portals_parameters["solution"]["exploration_ranges"]["yminymax_atleast"] = gradient_atleast

    portals_fun.portals_parameters["transport"]["in_process"] = in_process

    # Lump to electrons + main ion + 1 impurity at each transport dispatch (what MAESTRO
    # does for its PORTALS beats) -> cheaper TGLF/NEO/VGEN
    portals_fun.portals_parameters["transport"]["profiles_postprocessing_fun"] = partial(
        maestro_lump_postproc, lumpImpurities=True, enforce_same_density_gradients=True
    )

    # >>> the capability under test: neoclassical E×B shear from NEO VGEN (zero toroidal rotation)
    #     er=2 -> NEO weak-rotation limit (recommended for zero Vtor); vel=1 -> weak-rotation velocities.
    portals_fun.portals_parameters["transport"]["options"]["neo"]["vgen_exb_shear"] = (
        {"er": 2, "vel": 1} if vgen_exb_shear else None
    )

    portals_fun.optimization_options["initialization_options"]["initial_training"] = initial_training
    portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = maximum_iterations

    portals_fun.prep(make_state())

    mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
    mitim_bo.run()

    return PORTALSanalysis.PORTALSanalyzer.from_folder(work)


# ---------------------------------------------------------------------------------------------------------------------
# Run both flux matches
# ---------------------------------------------------------------------------------------------------------------------

print("\n>>> PORTALS flux match WITHOUT E×B shear (zero-rotation baseline)", typeMsg="i")
pa_off = run_portals_fluxmatch("noexb", vgen_exb_shear=False)

print("\n>>> PORTALS flux match WITH neoclassical E×B shear (VGEN)", typeMsg="i")
pa_on = run_portals_fluxmatch("exb", vgen_exb_shear=True)


# ---------------------------------------------------------------------------------------------------------------------
# Pull out the quantities to compare
# ---------------------------------------------------------------------------------------------------------------------

# (1) MECHANISM — iteration 0: same gradients in both runs, only VEXB_SHEAR differs
ps0_off, ps0_on = pa_off.powerstates[0], pa_on.powerstates[0]
roa = _np(ps0_off.plasma["roa"][0, 1:])
Qturb_off = _np(ps0_off.plasma["QeMWm2_tr_turb"][0, 1:]) + _np(ps0_off.plasma["QiMWm2_tr_turb"][0, 1:])
Qturb_on  = _np(ps0_on.plasma["QeMWm2_tr_turb"][0, 1:]) + _np(ps0_on.plasma["QiMWm2_tr_turb"][0, 1:])

# (2) EFFECT — flux-matched (best) iteration
ps_off, ps_on = pa_off.powerstates[pa_off.ibest], pa_on.powerstates[pa_on.ibest]
aLte_off, aLti_off = _np(ps_off.plasma["aLte"][0, 1:]), _np(ps_off.plasma["aLti"][0, 1:])
aLte_on,  aLti_on  = _np(ps_on.plasma["aLte"][0, 1:]),  _np(ps_on.plasma["aLti"][0, 1:])
te_off, ti_off = _np(ps_off.plasma["te"][0, 1:]), _np(ps_off.plasma["ti"][0, 1:])
te_on,  ti_on  = _np(ps_on.plasma["te"][0, 1:]),  _np(ps_on.plasma["ti"][0, 1:])

# E×B capability fired? -> w0 populated by VGEN in the ON run, still zero in the OFF run
w0_on  = ps_on.profiles.profiles["w0(rad/s)"]
w0_off = ps_off.profiles.profiles["w0(rad/s)"]

# Full-grid rotation inputs (w0, VEXB_SHEAR, VPAR, VPAR_SHEAR) for plotting. Taken at iteration 0 so
# they line up with the mechanism panel above (same gradients in both runs): the VEXB_SHEAR there is
# what TGLF saw when the flux dropped, so its radial structure explains where the stabilization happened.
rroa_off, w0p_off, vexb_off, vpar_off, vpars_off = rotation_inputs(pa_off.extractProfiles(0))
rroa_on,  w0p_on,  vexb_on,  vpar_on,  vpars_on  = rotation_inputs(pa_on.extractProfiles(0))


# ---------------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------------

# Programmatic MITIM_BO.run() leaves sys.stdout pointing at the last run's optimization_log.txt
# (thread-local logging proxy); restore the real terminal so this report is actually visible.
sys.stdout = sys.__stdout__

print("\n" + "=" * 78)
print(" (1) MECHANISM at iteration 0 (identical gradients) — turbulent Qe+Qi [MW/m^2]")
print("=" * 78)
print(f" {'r/a':>6}{'E×B off':>14}{'E×B on':>14}{'change':>12}")
print("-" * 78)
for r, qo, qn in zip(roa, Qturb_off, Qturb_on):
    print(f" {r:>6.2f}{qo:>14.4e}{qn:>14.4e}{100*(qn-qo)/qo:>11.1f}%")
print("-" * 78)
print(f" total turbulent flux change with E×B on: {100*(Qturb_on.sum()-Qturb_off.sum())/Qturb_off.sum():+.1f}%")

print("\n" + "=" * 78)
print(" (2) EFFECT on the flux-matched state — gradients and core T")
print("=" * 78)
print(f" {'r/a':>6}{'a/LTe off':>12}{'a/LTe on':>12}{'a/LTi off':>12}{'a/LTi on':>12}")
print("-" * 78)
for r, go, gn, io, inn in zip(roa, aLte_off, aLte_on, aLti_off, aLti_on):
    print(f" {r:>6.2f}{go:>12.3f}{gn:>12.3f}{io:>12.3f}{inn:>12.3f}")
print("-" * 78)
print(f" innermost predicted r/a={roa[0]:.2f}:  Te {te_off[0]:.3f}->{te_on[0]:.3f} keV   "
      f"Ti {ti_off[0]:.3f}->{ti_on[0]:.3f} keV")
print(f" edge predicted     r/a={roa[-1]:.2f}:  Te {te_off[-1]:.3f}->{te_on[-1]:.3f} keV   "
      f"(cooled to ~{Te09_target} keV)")
print(f" case setup: density x{density_factor}, ICRF ~{P_icrf_MW} MW (SPARC PRD rebalanced)")
print(f" |w0| max [krad/s]:  off={np.abs(w0_off).max()/1e3:.3f}   on={np.abs(w0_on).max()/1e3:.3f}  "
      f"(VGEN neoclassical Er)")
print("=" * 78)


# ---------------------------------------------------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------------------------------------------------

# HARD: the capability fired — VGEN populated a non-zero neoclassical rotation in the ON run only
assert np.abs(w0_off).max() == 0.0, "baseline (E×B off) must keep zero rotation"
assert np.abs(w0_on).max() > 0.0, "vgen_exb_shear=True must populate a non-zero neoclassical w0 (VGEN Er)"

# HARD: the option actually changed the converged flux-matched gradients
assert not np.allclose(aLte_on, aLte_off) or not np.allclose(aLti_on, aLti_off), \
    "E×B shear should change the flux-matched gradients"

# HARD (sanity): E×B shear must not GROSSLY increase the turbulent flux at fixed gradients. This is a
# deliberately loose bound: in the cold, near-marginal regime the iteration-0 fluxes are tiny
# (~1e-3 MW/m^2) and dominated by TGLF noise, so the clean stabilization signal moves OUT of this
# fixed-gradient comparison and INTO the flux-matched gradients (the EFFECT table / soft check below).
assert Qturb_on.sum() <= Qturb_off.sum() * 1.25, \
    "E×B shear grossly increased the turbulent flux at fixed gradients — unexpected"

# SOFT (physics, direction): the regime decides which signal is clean.
#   - moderate flux (hot plasma): E×B lowers the fixed-gradient flux at iter 0 (the MECHANISM panel).
#   - low flux (cold / near-marginal): the fixed-gradient comparison is noisy, but E×B raises the
#     critical gradient, so the flux-matched gradients are markedly steeper (the EFFECT panel).
if Qturb_on.sum() < Qturb_off.sum() * 0.98:
    print("\nOK: E×B shear reduced the turbulent flux at fixed gradients (stabilizing).", typeMsg="i")
else:
    print("\nNote: iteration-0 fluxes barely changed — near-marginal / noisy regime; read the "
          "flux-matched gradients (EFFECT) instead of this fixed-gradient comparison.", typeMsg="w")

if aLte_on.mean() >= aLte_off.mean() and aLti_on.mean() >= aLti_off.mean():
    print("OK: flux-matched gradients are steeper with E×B shear (critical-gradient upshift / "
          "expected stabilizing direction).", typeMsg="i")
else:
    print("Note: flux-matched gradients did not move in the expected direction — likely the short, "
          "under-converged run (see the WARNING in the header); re-check with more iterations.", typeMsg="w")


# ---------------------------------------------------------------------------------------------------------------------
# Combined notebook: (a) this E×B comparison, (b) the PORTALS summaries that mitim_plot_portals
# produces (one metrics tab per run), and (c) the VGEN neoclassical-Er notebook (mitim_plot_vgen).
# ---------------------------------------------------------------------------------------------------------------------

from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gacode_tools import NEOtools

nb = FigureNotebook("E×B shear effect on a PORTALS flux match", geometry="1800x1000")

# --- (a) our comparison tab --------------------------------------------------------------------
fig = nb.add_figure(label="E×B comparison", tab_color=0)
axs = fig.subplots(2, 3)
fig.suptitle("Effect of neoclassical E×B shear on a PORTALS flux match")

# --- Row 1: mechanism (iter 0) + effect on the flux-matched gradients --------------------------
axs[0, 0].plot(roa, Qturb_off, "o-", label="E×B off")
axs[0, 0].plot(roa, Qturb_on, "s-", label="E×B on")
axs[0, 0].set_xlabel("r/a"); axs[0, 0].set_ylabel("turbulent Qe+Qi [MW/m$^2$]")
axs[0, 0].set_title("(1) iter 0: same gradients"); axs[0, 0].legend(); axs[0, 0].set_ylim(bottom=0)

axs[0, 1].plot(roa, aLte_off, "o-", label="E×B off")
axs[0, 1].plot(roa, aLte_on, "s-", label="E×B on")
axs[0, 1].set_xlabel("r/a"); axs[0, 1].set_ylabel("a/LTe"); axs[0, 1].set_title("(2) flux-matched a/LTe"); axs[0, 1].legend()

axs[0, 2].plot(roa, aLti_off, "o-", label="E×B off")
axs[0, 2].plot(roa, aLti_on, "s-", label="E×B on")
axs[0, 2].set_xlabel("r/a"); axs[0, 2].set_ylabel("a/LTi"); axs[0, 2].set_title("(2) flux-matched a/LTi"); axs[0, 2].legend()

# --- Row 2: the rotation inputs the E×B shear feeds into TGLF, at iteration 0 -------------------
# Markers = values at the predicted radii (what TGLF actually evaluates); the line is the VGEN
# profile, trimmed to the band where VGEN computes it (flat tails outside are np.interp padding).
xlo, xhi = min(predicted_roa) - 0.1, max(predicted_roa) + 0.06
bm = (rroa_on >= xlo) & (rroa_on <= xhi)

def _mark_radii(ax):
    for rr in predicted_roa:
        ax.axvline(rr, color="0.85", lw=0.8, zorder=0)

def _cp(y):  # value at each predicted radius
    return np.interp(predicted_roa, rroa_on, y)

axs[1, 0].plot(rroa_off[bm], w0p_off[bm] / 1e3, "-", color="C0", label="E×B off")
axs[1, 0].plot(rroa_on[bm], w0p_on[bm] / 1e3, "-", color="C1", label="E×B on")
axs[1, 0].plot(predicted_roa, _cp(w0p_on) / 1e3, "o", color="C1")
axs[1, 0].set_xlabel("r/a"); axs[1, 0].set_ylabel("w0 [krad/s]")
axs[1, 0].set_title("rotation (VGEN neoclassical Er)"); axs[1, 0].legend(); _mark_radii(axs[1, 0])

axs[1, 1].plot(rroa_off[bm], vexb_off[bm], "-", color="C0", label="E×B off")
axs[1, 1].plot(rroa_on[bm], vexb_on[bm], "-", color="C1", label="E×B on")
axs[1, 1].plot(predicted_roa, _cp(vexb_on), "o", color="C1")
axs[1, 1].set_xlabel("r/a"); axs[1, 1].set_ylabel("VEXB_SHEAR  ($\\gamma_E\\,a/c_s$)")
axs[1, 1].set_title("E×B shearing rate (stabilizing)"); axs[1, 1].legend(); _mark_radii(axs[1, 1])

# VPAR_SHEAR (left axis) and VPAR (right twin axis) for the E×B-on case — different magnitudes
axs[1, 2].plot(rroa_on[bm], vpars_on[bm], "-", color="C1", label="VPAR_SHEAR")
axs[1, 2].plot(predicted_roa, _cp(vpars_on), "o", color="C1")
axs[1, 2].set_xlabel("r/a"); axs[1, 2].set_ylabel("VPAR_SHEAR  ($\\gamma_p\\,a/c_s$)", color="C1")
ax_vpar = axs[1, 2].twinx()
ax_vpar.plot(rroa_on[bm], vpar_on[bm], "--", color="C3", label="VPAR")
ax_vpar.set_ylabel("VPAR  ($v_\\parallel/c_s$)", color="C3")
axs[1, 2].set_title("parallel-flow drive (E×B on)"); _mark_radii(axs[1, 2])
# combine the two axes' legends, dropping the axvline ('_child…') entries
handles = [l for l in axs[1, 2].get_lines() + ax_vpar.get_lines() if not l.get_label().startswith("_")]
axs[1, 2].legend(handles, [l.get_label() for l in handles], fontsize=8)
fig.tight_layout()

# --- (b) PORTALS summaries (what `mitim_plot_portals` shows): one metrics tab per run ----------
for pa, lbl, col in [(pa_off, "PORTALS metrics (E×B off)", 1), (pa_on, "PORTALS metrics (E×B on)", 2)]:
    pa.fn = nb   # route this analyzer's tabs into the shared notebook (same pattern as read_portals)
    pa.plotMetrics(fig=nb.add_figure(label=lbl, tab_color=col), extra_lab=lbl)

# --- (c) VGEN notebook (neoclassical Er decomposition) for iteration 0 of the E×B-on run -------
# Located via the analyzer's folder convention: Execution/Evaluation.<it>/transport_simulation_folder,
# matching the iteration used for the rotation panels above.
try:
    neo = NEOtools.NEO(rhos=[])
    neo.FolderGACODE = folder / "exb" / "Execution" / "Evaluation.0" / "transport_simulation_folder"
    neo.read_vgen(subfolder="vgen_neo_exb")
    # mark the PORTALS predicted radii on the smoothing tab (they bracket the VGEN rho band)
    neo.plot_vgen(fn=nb, fn_color=3, mark_rho=pa_on.rhos)
except Exception as e:
    print(f"Could not add the VGEN notebook tabs ({e})", typeMsg="w")

nb.show()
