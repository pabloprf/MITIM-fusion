"""
DEV TEST: TRANSP/NCLASS vs VGEN/NEO neoclassical rotation (w0 / omega) and Er
----------------------------------------------------------------------------
Pure NEOCLASSICAL comparison on the SAME plasma state (a bundled input.gacode
with w0=0, i.e. no rotation prescribed). NO anomalous transport, NO PT_SOLVER,
NO predictive momentum equation is involved on either side — both codes solve
the neoclassical force balance and report the implied toroidal angular rotation
and radial electric field:

    PATH A  --  TRANSP / NCLASS (Houlberg 2004): NCLASS is on by default in the
                MITIM namelist (NMLtools.addNCLASS). With the NCLASS neoclassical
                potential ON (nlvwnc=T, now the MITIM default) the .CDF carries
                the neoclassical omega (OMEGA_NC / EPOTNC) and the neoclassical Er
                decomposition (ERPRESS/ERVTOR/ERVPOL), which CDFtools exposes.
                The run is deliberately stripped to the bone for speed -- no
                ICRF/TORIC, no NBI/NUBEAM, coarse timesteps (same spirit as
                MAESTRO's transp_soft beat) -- since NCLASS only needs the kinetic
                profiles + the (zero) rotation U-File to close the Er balance.

    PATH B  --  VGEN (profiles_gen -vgen) with er=2 = the NEO WEAK-ROTATION
                neoclassical limit: NEO is solved over the flux surfaces and
                returns the neoclassical Er and the consistent w0(rad/s).

The two neoclassical rotation profiles are printed as a table and overlaid in a
matplotlib figure with THREE panels: (1) w0/omega vs rho, (2) Er vs rho with TRANSP's
ERPRESS/ERVTOR/ERVPOL decomposition, and (3) a TERM-BY-TERM Er check vs an independent
diamagnetic Er = (1/(Z_i e n_i)) dp_i/dr.

*** WHAT THE COMPARISON SHOWS (and a w0-vs-Er caveat) ***
    Compare Er, NOT w0. The w0 panel mixes two DIFFERENT decompositions: TRANSP's
    OMEGA_NC = omega_ExB + omega_diamag (the diamagnetic part largely CANCELS -> small
    intrinsic value), while VGEN's w0 RETAINS the diamagnetic term (diamagnetic-dominated).
    The frame-independent Er is the clean check, and it shows:
      - the DIAMAGNETIC term agrees across NCLASS, NEO, and the independent dp/dr/Zen
        (it is model-independent), and
      - the neoclassical POLOIDAL-FLOW term (ERVPOL vs NEO's) is the entire residual --
        opposite sign in the core here -- the expected NCLASS(Houlberg)-vs-NEO model
        spread (NEO is the higher-fidelity drift-kinetic solver).
    This does NOT affect MAESTRO's rotation chain: the TRANSP beat writes w0 = OMEGA
    (cdf.VtorkHz / TGLF_w0, CDFtools.to_profiles), the rotation TRANSP actually used --
    NOT OMEGA_NC -- so the w0 round-trip stays in the GACODE convention end to end.

*** REQUIREMENTS ***
    - PATH A requires a configured TRANSP machine ("transp" in config_user.json).
      Even a short flattop run is minutes-scale, so this is NOT a CI test.
      (Same dependency as tests/capability_tests/maestro_01_run.py.)
    - PATH B requires "profiles_gen" configured (the GACODE install providing
      profiles_gen -vgen, which wraps NEO). Much cheaper than PATH A.
      (Same dependency as tests/capability_tests/neo_02_vgen_from_inputgacode.py.)
    - matplotlib for the comparison figure.

*** NEOCLASSICAL POTENTIAL IS ON BY DEFAULT ***
    This test relies on the NCLASS neoclassical potential being written to the
    CDF (EPOTNC / OMEGA_NC). The MITIM namelist now does this by default through
    the `computeNCLASSpotential` flag (NMLtools.py), which sets nlvwnc=T. No
    namelist patching is needed here, and NO PT_SOLVER / lpredict_* / anomalous
    momentum transport is enabled — this is a default MITIM TRANSP run with
    NCLASS as the (only) neoclassical model.

*** UNITS / SIGN CONVENTIONS (verify before trusting the comparison) ***
    Toroidal angular rotation:
      - GACODE/VGEN convention: w0(rad/s), the field 'w0(rad/s)' in input.gacode.
        VGEN populates it from the NEO neoclassical Er. Starts at 0 in this file.
      - TRANSP/CDFtools NEOCLASSICAL angular frequency, two equivalent reads:
          * transp_output.VtorkHz_nc   (kHz; CDFtools.py:3309, from CDF 'OMEGA_NC')
          * transp_output.VtorkHz_nc_check (kHz; CDFtools.py:3381, = -dPhi_nc/dpsi
            / 2pi, from the neoclassical potential EPOTNC -> Epot_nc)
        Both -> rad/s by multiplying by 2*pi*1e3. We compare against VtorkHz_nc.
      - SIGN: GACODE w0 follows the input.gacode COCOS; TRANSP follows nlbccw/
        nljccw (NMLtools.py:589-590, both default False). These need NOT agree a
        priori. Compare magnitude and shape; reconcile the overall sign against
        the field/current directions of YOUR case before drawing conclusions.
    Radial electric field Er (V/m):
      - VGEN: er_exp in out.vgen.vel -> NEO.vgen_vel["er_exp"] (NEOtools.py:822).
      - TRANSP/CDFtools NEOCLASSICAL Er: transp_output.Er (CDFtools.py:3345, from
        CDF 'ERTOT', *1e2 cm->m) with the additive neoclassical decomposition
        Er = Er_p + Er_tor + Er_pol  (ERPRESS/ERVTOR/ERVPOL, CDFtools.py:3348-3355),
        the quantity CDFtools itself titles "Neoclassical Er". Same sign caution.
    Radial coordinate:
      - input.gacode / VGEN: 'rho(-)' = sqrt(normalized toroidal flux).
      - CDFtools: x (zone center) and xb (zone boundary) are ALSO sqrt normalized
        toroidal flux (CDFtools.py:684-685), directly comparable to gacode rho.
        VtorkHz_nc lives on x; Er on the xb-derived grid. We interpolate TRANSP
        onto the VGEN rho grid for the table.
"""

import numpy as np
import matplotlib.pyplot as plt

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.gacode_tools import PROFILEStools, NEOtools
from mitim_tools.transp_tools import CDFtools

# cold_start=True starts from scratch (removing the previous folder); False reuses
# results already present (so a finished TRANSP CDF / vgen folder is not recomputed)
cold_start = False  # reuse the good (w0=0) CDF/vgen; set True only if inputs change

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "test_transp_vs_vgen_rotation"

# Bundled DT SPARC PRD plasma state (T,D fuel + F,W,He). NOTE: this file carries a
# finite w0(rad/s) (~-11 krad/s mid-radius) -- it is NOT a zero-rotation state. We zero
# w0 below (see A.1) so both paths genuinely PREDICT the neoclassical rotation instead
# of TRANSP echoing the prescribed rotation.
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# Tokamak name selects machine-specific TRANSP conventions
tokamak = "AUGD"

# =====================================================================================
# PATH A: TRANSP run with NCLASS neoclassical potential output (no anomalous transport)
# =====================================================================================

folderTRANSP = folder / "transp_neoclassical"
folderTRANSP.mkdir(parents=True, exist_ok=True)

shot = "12345"
runid = "R01"

# Short flattop: NCLASS evaluates the neoclassical Er diagnostically from the
# (fixed, experimental-UFILE) kinetic profiles, so a long run is not needed for
# the neoclassical quantities — the flattop just lets the equilibrium settle.
time_init = 0.0
time_current_diffusion = 0.0
time_end = 0.5          # s of flattop
time_extraction = None  # no AC snapshot: OMEGA_NC/EPOTNC live in the regular CDF, and
                        # there are no TORIC/NUBEAM AC files to extract in this stripped run

# ---------------------------------------------------------------------------------------------------------------------
# A.1 Build the TRANSP run from input.gacode (the canonical input.gacode -> TRANSP path)
# ---------------------------------------------------------------------------------------------------------------------

# gacode_state.to_transp() (MITIMstate.py:3209) returns a TRANSPhelpers.transp_run
# already populated with the UFILE-able quantities at the requested times. It also
# ships the toroidal rotation as the 'omg' U-File by default — here a ZERO omg U-File,
# since w0=0 in this state. That zero rotation is the input NCLASS needs to close the
# Er force balance (the toroidal-rotation term is not predicted by neoclassical theory),
# and it puts NCLASS in exactly the weak-rotation limit compared against NEO er=2 below.
profiles = PROFILEStools.gacode_state(input_gacode)

# IMPORTANT: the bundled input.gacode actually carries a finite toroidal rotation
# (w0(rad/s) ~ -11 krad/s at mid-radius, NOT zero). The neoclassical-PREDICTION
# comparison only works from a NO-rotation state: with a finite w0, to_transp ships it
# as the omg U-File and NCLASS just ECHOES it (OMEGA_NC ~ input, Er dominated by the
# v_phi*B_theta term), while VGEN er=2 (weak-rotation) DISCARDS it -> the two are then
# computing different things (a measured ~3-5x, sign-flipped gap that is NOT a model
# disagreement; the diamagnetic Er terms actually match). Zero w0 so BOTH genuinely
# predict the neoclassical rotation from the kinetic profiles, and feed the SAME zeroed
# state to both paths.
profiles.profiles["w0(rad/s)"][:] = 0.0
input_gacode = folder / "input.gacode_w0zero"
profiles.write_state(file=input_gacode)

times = [time_init, time_end + 1.0]  # bracket the flattop (matches TRANSPbeat usage)

# Separatrix smoothing for the RFS/ZFS boundary U-Files. The bundled state is a
# single-null DIVERTED plasma (lower X-point: Z in [-0.88, +0.77]); the boundary is
# MXH-fit and reconstructed for TRANSP. With many harmonics the fit REPRODUCES the
# X-point cusp (n_coeff=5 keeps the raw ~6 cm turn radius), and TRANSP's fixed-boundary
# equilibrium then fails its flux-surface Jacobian check at rho=1 (det(J) changes sign).
# A low n_coeff rounds the X-point (n_coeff=2 -> ~20 cm turn radius) while keeping
# triangularity + squareness. Drop to 1 if the Jacobian check still trips.
mxh_coeffs_smooth_sep = 2

transp = profiles.to_transp(
    folder=folderTRANSP,
    shot=shot,
    runid=runid,
    times=times,
    Vsurf=0.0,
    mxh_coeffs_smooth=mxh_coeffs_smooth_sep,
)

# ---------------------------------------------------------------------------------------------------------------------
# A.2 Write a MINIMAL namelist: neoclassical rotation only, as cheap as possible
# ---------------------------------------------------------------------------------------------------------------------

# We want the lightest TRANSP that still closes the NCLASS Er/rotation force balance,
# so every expensive auxiliary-heating / fast-ion module is switched OFF (same spirit
# as MAESTRO's transp_soft beat, just inlined here):
#   - Pich=False               -> no ICRF, so TORIC never runs
#   - Pnbi=False               -> no NBI, so NUBEAM beams never run (also the default)
#   - useNUBEAMforAlphas=False -> D-T alphas use the analytic fast model (nalpha=1),
#                                 NOT the NUBEAM Monte Carlo
# and all timesteps are coarsened to ~100 ms. NO PTsolver either, so NCLASS (Houlberg)
# is the ONLY model active. NCLASS still writes EPOTNC/OMEGA_NC because it only needs
# the rotation U-File shipped above (the zero omg U-File -> nlvphi=T, nlvwnc=T, both on
# by default).
#
# Ufiles: feed the separatrix as RFS/ZFS boundary U-Files (the moments path that
# write_ufiles(use_mry_file=False, the default) actually produces below), NOT the
# scrunched MRY file. The default NMLtools list still lists "mry" (+ df4/vc4/gfd
# He4/gas U-Files that the to_transp/write_ufiles path never writes), so leaving it
# unset makes TRDAT request a MIT<shot>.MRY that is never written -> "MRY FILE OPEN
# ERROR". This mirrors the MAESTRO TRANSP beat (TRANSPbeat.py:131).
transp.write_namelist(
    timings={
        "time_start": time_init,
        "time_current_diffusion": time_current_diffusion,
        "time_end": time_end,
        "time_extraction": time_extraction,
    },
    Ufiles=["qpr", "cur", "vsf", "ter", "ti2", "ner", "rbz", "lim", "zf2", "rfs", "zfs"],
    # --- Strip auxiliary heating + fast ions: no TORIC, no NUBEAM ---
    Pich=False,
    Pnbi=False,
    useNUBEAMforAlphas=False,
    DTplasma=True,                 # keep the D-T species mix (cheap; rotation is unaffected)
    # --- Coarse time resolution for speed (mirrors transp_soft) ---
    dtEquilMax_ms=100.0,           # MHD equilibrium step (dtmaxg)
    dtHeating_ms=100.0,            # ICRF/NBI step (unused here, harmless)
    dtCurrentDiffusion_ms=100.0,   # poloidal-field diffusion step (dtmaxb)
    dtOut_ms=100.0,                # output cadence (sedit/stedit)
    dtIn_ms=100.0,                 # input-data cadence (tgrid1/tgrid2)
)

# ---------------------------------------------------------------------------------------------------------------------
# A.3 Write UFILEs and submit; wait for completion; fetch the AC/CDF outputs
# ---------------------------------------------------------------------------------------------------------------------

transp.write_ufiles(mxh_coeffs_smooth=mxh_coeffs_smooth_sep)

# transp_run.run() wraps TRANSPtools.TRANSP + defineRunParameters + run +
# checkUntilFinished (TRANSPhelpers.run:382). No TORIC/NUBEAM here, so the toric/ptr
# MPI pools are set to 1 and retrieveAC=False (there are no AC files to pull; the
# neoclassical rotation lives in the regular CDF).
#
# cold_start=False makes run() reuse an existing {shot}{runid}.CDF instead of re-staging
# and re-submitting to SLURM -- so a cold_start=False rerun just re-reads + re-plots (the
# cheap local input regen above still runs). cold_start=True forces a fresh TRANSP run.
transp.run(
    tokamak,
    mpisettings={"trmpi": 32, "toricmpi": 1, "ptrmpi": 1},
    minutesAllocation=30,
    case="neoclassical",
    checkMin=2,
    grabIntermediateEachMin=1e6,
    retrieveAC=False,
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# A.4 Read the NEOCLASSICAL rotation / Er from the TRANSP CDF (last sawtooth/AC slice)
# ---------------------------------------------------------------------------------------------------------------------

# transp_output() auto-finds the .CDF in the directory (CDFtools.py:135) and reads
# OMEGA_NC / EPOTNC / ERTOT (+decomposition) in __init__.
cdf = CDFtools.transp_output(folderTRANSP)

it = cdf.ind_saw  # last-sawtooth (steady) time index used throughout CDFtools

# Neoclassical toroidal angular rotation. VtorkHz_nc is kHz (from OMEGA_NC) -> rad/s.
transp_rho      = cdf.x[it, :]                          # sqrt(norm tor flux) == gacode rho
transp_w0_nc    = cdf.VtorkHz_nc[it, :] * (2 * np.pi * 1e3)        # rad/s (NCLASS)
# Cross-check from the neoclassical potential (-dPhi_nc/dpsi/2pi), on the xb grid.
transp_rho_xb   = cdf.xb[it, :]
transp_w0_nc_chk = cdf.VtorkHz_nc_check[it, :] * (2 * np.pi * 1e3)  # rad/s

# Neoclassical Er and its decomposition (V/m); the sum is the "Neoclassical Er".
transp_Er       = cdf.Er_LF[it, :]                      # V/m (ERTOT), LF-mapped onto the x (zone-center) grid
transp_Er_p     = cdf.Er_p_LF[it, :]                    # V/m (diamagnetic / grad-p)
transp_Er_tor   = cdf.Er_tor_LF[it, :]                  # V/m (toroidal-flow term)
transp_Er_pol   = cdf.Er_pol_LF[it, :]                  # V/m (poloidal-flow term)
# NOTE: raw cdf.Er/Er_p/Er_tor/Er_pol live on the Rmaj midplane grid (LF->HF, ~2x longer),
# NOT on x/xb — use the _LF variants so they align with transp_rho (the x grid) below.

# =====================================================================================
# PATH B: VGEN / NEO neoclassical rotation on the SAME input.gacode
# =====================================================================================

folder_vgen_parent = folder / "vgen_neoclassical"

# rhos=[] because VGEN sweeps the flux surfaces of the state (see
# neo_02_vgen_from_inputgacode.py); it is not a per-rho run.
neo = NEOtools.NEO(rhos=[])
neo.prep(input_gacode, folder_vgen_parent)

neo.run_vgen(
    subfolder="vgen1",
    # Restrict to the core/gradient region (edge neoclassical well is the noisy part).
    rho_range=[0.1, 0.90],
    vgenOptions={
        # er=2: NEO WEAK-rotation neoclassical limit (recommended when toroidal
        # rotation is ~0, which is exactly the bundled state). vel=1: weak-rot vel.
        "er": 2,
        "vel": 1,
        "nth": "17,39",
        "matched_ion": 1,
    },
    # Smooth kinetic profiles first so piecewise-linear gradient kinks don't pollute
    # the NEO Er (original state untouched).
    smooth_profiles=True,
    cold_start=cold_start,
)

neo.read_vgen()

# w0 populated by NEO into the updated input.gacode (rad/s, GACODE convention)
vgen_rho   = neo.profiles_vgen.profiles["rho(-)"]
vgen_w0    = neo.profiles_vgen.profiles["w0(rad/s)"]

# Er used/derived by VGEN (out.vgen.vel -> vgen_vel["er_exp"], NEOtools.py:822).
# *** UNITS ***: er_exp is in kV/m, NOT V/m (vgen.f90 builds it in CGS-Gaussian and ends
# with a /1000; the output values, ~-0.4..-4.5, are kV/m). The NEOtools comment calling it
# V/m is wrong. Multiply by 1e3 to put it on the V/m axis next to the TRANSP Er.
# This lives on the (possibly truncated) vgen rho grid, separate from vgen_rho.
if neo.vgen_vel and "er_exp" in neo.vgen_vel:
    vgen_Er_rho = neo.vgen_vel["rho"]
    vgen_Er     = neo.vgen_vel["er_exp"] * 1e3   # kV/m -> V/m
else:
    vgen_Er_rho = None
    vgen_Er     = None

# =====================================================================================
# COMPARISON: table + figure
# =====================================================================================

# Common rho grid for the table: the VGEN window, interpolating TRANSP onto it.
rho_common = vgen_rho[(vgen_rho >= 0.1) & (vgen_rho <= 0.90)]

w0_vgen_c      = np.interp(rho_common, vgen_rho, vgen_w0)
w0_transp_c    = np.interp(rho_common, transp_rho, transp_w0_nc)

print("\n" + "=" * 70)
print(" Neoclassical toroidal angular rotation w0 [rad/s]")
print("   TRANSP/NCLASS  vs  VGEN/NEO  (weak-rotation limit)")
print("=" * 70)
print(f" {'rho':>6} | {'w0 VGEN/NEO':>16} | {'w0 TRANSP/NCLASS':>18}")
print("-" * 70)
for r, wv, wt in zip(rho_common, w0_vgen_c, w0_transp_c):
    print(f" {r:6.3f} | {wv:16.4e} | {wt:18.4e}")
print("=" * 70 + "\n")

# Term-by-term Er verification (V/m). The main-ion diamagnetic Er = (1/(Z_i e n_i)) dp_i/dr
# is MODEL-INDEPENDENT, so it anchors the pressure term of BOTH codes; the poloidal-flow
# term is then whatever each code's total Er has beyond it. (profiles still holds n_i/T_i;
# zeroing w0 does not change them.) This isolates the claim: diamagnetic agrees, the
# neoclassical poloidal-flow term is where NCLASS (Houlberg) and NEO actually differ.
_e = 1.602176634e-19
_rho_p = profiles.profiles["rho(-)"]
_ni    = profiles.profiles["ni(10^19/m^3)"][:, 0] * 1e19   # main ion (D), matched_ion=1
_Ti    = profiles.profiles["ti(keV)"][:, 0] * 1e3 * _e
_Zi    = profiles.profiles["z"][0]
Er_dia_indep = np.gradient(_ni * _Ti, profiles.derived["r"]) / (_Zi * _e * _ni)  # V/m
# VGEN poloidal Er: with er=2 (vtor=0) Er = diamagnetic + poloidal, so the poloidal piece
# is total - diamagnetic. (TRANSP gives ERVPOL directly.)
if vgen_Er is not None:
    vgen_Er_pol = vgen_Er - np.interp(vgen_Er_rho, _rho_p, Er_dia_indep)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Panel 1: neoclassical toroidal angular rotation w0 / omega ---
ax = axs[0]
ax.plot(vgen_rho, vgen_w0, "-o", color="C0", lw=1.8, ms=3, label=r"$\omega_0$ VGEN/NEO")
ax.plot(transp_rho, transp_w0_nc, "-s", color="C1", lw=1.8, ms=3, label=r"$\omega_{nc}$ TRANSP (OMEGA_NC)")
ax.plot(transp_rho_xb, transp_w0_nc_chk, "--^", color="C3", lw=1.2, ms=3, label=r"$\omega_{nc}$ TRANSP ($-d\Phi_{nc}/d\psi$)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$  (sqrt norm. tor. flux)")
ax.set_ylabel(r"$\omega_0$  (rad/s)")
ax.set_xlim([0.0, 1.0])
ax.set_title("Neoclassical toroidal angular rotation")
ax.legend(loc="best", fontsize=8)

# --- Panel 2: neoclassical radial electric field Er ---
ax = axs[1]
if vgen_Er is not None:
    ax.plot(vgen_Er_rho, vgen_Er, "-o", color="C0", lw=1.8, ms=3, label=r"$E_r$ VGEN/NEO")
ax.plot(transp_rho, transp_Er, "-s", color="C1", lw=1.8, ms=3, label=r"$E_r$ TRANSP (total)")
ax.plot(transp_rho, transp_Er_p, ":", color="C2", lw=1.2, label=r"$E_r$ TRANSP ($\nabla p$)")
ax.plot(transp_rho, transp_Er_tor, ":", color="C4", lw=1.2, label=r"$E_r$ TRANSP (tor)")
ax.plot(transp_rho, transp_Er_pol, ":", color="C5", lw=1.2, label=r"$E_r$ TRANSP (pol)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$  (sqrt norm. tor. flux)")
ax.set_ylabel(r"$E_r$  (V/m)")
ax.set_xlim([0.0, 1.0])
ax.set_title("Neoclassical radial electric field")
ax.legend(loc="best", fontsize=8)

# --- Panel 3: TERM-BY-TERM Er verification (diamagnetic agrees; poloidal differs) ---
# Diamagnetic curves should OVERLAP (model-independent physics); the poloidal-flow curves
# are where NCLASS and NEO diverge (opposite sign in the core here), which is the entire
# residual once the spurious input rotation is removed.
ax = axs[2]
ax.plot(_rho_p, Er_dia_indep, "--", color="k", lw=1.8,
        label=r"$E_r^{\nabla p}$ independent ($dp_i/dr/Z_ien_i$)")
ax.plot(transp_rho, transp_Er_p, "-s", color="C2", lw=1.6, ms=3,
        label=r"$E_r^{\nabla p}$ TRANSP (ERPRESS)")
ax.plot(transp_rho, transp_Er_pol, "-^", color="C5", lw=1.6, ms=3,
        label=r"$E_r^{v_\theta}$ TRANSP (ERVPOL)")
if vgen_Er is not None:
    ax.plot(vgen_Er_rho, vgen_Er_pol, "-o", color="C0", lw=1.6, ms=3,
            label=r"$E_r^{v_\theta}$ VGEN ($E_r-E_r^{\nabla p}$)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$  (sqrt norm. tor. flux)")
ax.set_ylabel(r"$E_r$ component  (V/m)")
ax.set_xlim([0.0, 1.0])
ax.set_title(r"Term-by-term: $\nabla p$ agrees, $v_\theta$ differs")
ax.legend(loc="best", fontsize=7)

fig.suptitle("TRANSP/NCLASS vs VGEN/NEO neoclassical rotation (SAME plasma state, w0 zeroed)")
fig.tight_layout()

figure_file = folder / "transp_vs_vgen_rotation.png"
fig.savefig(figure_file, dpi=150)
print(f"\t- Comparison figure saved to {IOtools.clipstr(figure_file)}", typeMsg="i")

plt.show()
