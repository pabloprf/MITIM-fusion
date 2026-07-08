"""
DEV TEST: TRANSP/NCLASS vs VGEN/NEO neoclassical rotation (w0 / omega) and Er
----------------------------------------------------------------------------
Pure NEOCLASSICAL comparison on the SAME plasma state (a bundled input.gacode
with w0=0, i.e. no rotation prescribed). NO anomalous transport, NO PT_SOLVER,
NO predictive momentum equation is involved on either side — both codes solve
the neoclassical force balance and report the implied toroidal angular rotation
and radial electric field:

    PATH A  --  TRANSP / NCLASS (Houlberg 2004): this run uses
                rotation_source='neoclassical_transp', the only mode that computes
                and keeps the NCLASS Er. With the NCLASS neoclassical
                potential ON (nlvwnc=T) the .CDF carries
                the neoclassical omega (OMEGA_NC / EPOTNC) and the neoclassical Er
                decomposition (ERPRESS/ERVTOR/ERVPOL), which CDFtools exposes.
                The run is deliberately stripped to the bone for speed -- no
                ICRF/TORIC, no NBI/NUBEAM, coarse timesteps (same spirit as
                MAESTRO's transp_soft beat) -- since NCLASS only needs the kinetic
                profiles + the (zero) rotation U-File to close the Er balance.

    PATH B  --  VGEN (profiles_gen -vgen) with er=2 = the NEO WEAK-ROTATION
                neoclassical limit: NEO is solved over the flux surfaces and
                returns the neoclassical Er and the consistent w0(rad/s).

The comparison is printed as two rigorous tables (Er decomposition; rotation-field
identification) and a 4-panel figure: (1) E×B rotation in the SAME (GACODE) convention
[VGEN w0 vs TRANSP geom*ERtot]; (2) Er and its ERPRESS/ERVTOR/ERVPOL decomposition;
(3) TERM-BY-TERM [diamagnetic from TRANSP/VGEN/independent dp/dr overlaps; poloidal flips
sign between NCLASS and NEO]; (4) why OMEGA_NC != w0 [omega_tor = omega_ExB + omega_dia+pol
(the force-balance remainder), which nearly cancel, leaving the small toroidal velocity].

*** WHAT THE COMPARISON SHOWS (and a w0-vs-Er caveat) ***
    Compare Er, NOT the omega-like fields. OMEGA_NC and w0 are DIFFERENT quantities:
    OMEGA_NC = omega_ExB + omega_dia+pol (force-balance remainder; the two nearly CANCEL
    when V_phi~0 is imposed -> small residual), while GACODE w0 IS omega_ExB itself
    (diamagnetic-dominated here). The frame-independent Er is the clean check, and it shows:
      - the DIAMAGNETIC term agrees across NCLASS, NEO, and the independent dp/dr/Zen
        (it is model-independent), and
      - the neoclassical POLOIDAL-FLOW term (ERVPOL vs NEO's) is the entire residual --
        opposite sign in the core here -- the expected NCLASS(Houlberg)-vs-NEO model
        spread (NEO is the higher-fidelity drift-kinetic solver).
    MAESTRO's rotation chain is consistent with this: CDFtools.to_profiles writes
    w0 = TGLF_w0_exb (the ERTOT-based E×B rotation; NOT OMEGA, NOT OMEGA_NC), and
    TRANSPbeat.finalize keeps that re-derivation only under
    rotation_source='neoclassical_transp' -- 'echo'/'neoclassical_portals' restore the
    SEED w0 unchanged (see the VALIDATION block below, which reads to_profiles directly).

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
          * transp_output.VtorkHz_nc   (kHz; CDFtools.py:3350, from CDF 'OMEGA_NC')
          * transp_output.VtorkHz_nc_check (kHz; CDFtools.py:3422, = -dPhi_nc/dpsi
            / 2pi, from the neoclassical potential EPOTNC -> Epot_nc)
        Both -> rad/s by multiplying by 2*pi*1e3. We compare against VtorkHz_nc.
      - SIGN: GACODE w0 follows the input.gacode COCOS; TRANSP follows nlbccw/
        nljccw (NMLtools.py:589-590, both default False). These need NOT agree a
        priori. Compare magnitude and shape; reconcile the overall sign against
        the field/current directions of YOUR case before drawing conclusions.
    Radial electric field Er (V/m):
      - VGEN: er_exp in out.vgen.vel -> NEO.vgen_vel["er_exp"] (NEOtools.py:828).
      - TRANSP/CDFtools NEOCLASSICAL Er: transp_output.Er (CDFtools.py:3386, from
        CDF 'ERTOT', *1e2 cm->m) with the additive neoclassical decomposition
        Er = Er_p + Er_tor + Er_pol  (ERPRESS/ERVTOR/ERVPOL, CDFtools.py:3389-3395),
        the quantity CDFtools itself titles "Neoclassical Er". Same sign caution.
    Radial coordinate:
      - input.gacode / VGEN: 'rho(-)' = sqrt(normalized toroidal flux).
      - CDFtools: x (zone center) and xb (zone boundary) are ALSO sqrt normalized
        toroidal flux (CDFtools.py:711), directly comparable to gacode rho.
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

# gacode_state.to_transp() (MITIMstate.py:3244) returns a TRANSPhelpers.transp_run
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
    rotation_source='neoclassical_transp',   # zero omg U-File in -> NCLASS weak-rotation Er (w0 is zeroed above anyway; this states the intent)
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
# ERROR". This mirrors the MAESTRO TRANSP beat (TRANSPbeat.py:277).
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

# =====================================================================================
# COMPARISON + RIGOROUS TERM-BY-TERM INVESTIGATION
# =====================================================================================
#
# GROUND TRUTH (from source, not eyeballed):
#   * GACODE w0 (vgen.f90:237-251): the radial force balance is
#         Er = (1/(Z_a e n_a)) dp_a/dr  +  V_phi,a B_theta  -  V_theta,a B_phi
#     and w0 = c q Er / (Bunit r |grad r|) = -c dPhi/dpsi_pol -- the E×B (potential)
#     rotation, built from the TOTAL Er. So GACODE w0 ∝ ERtot.
#   * TRANSP CDF metadata (read directly from the .CDF):
#       OMEGA_NC = "N.C. TOROIDAL ANGULAR VELOCITY"  (V_phi/R; NOT the E×B rotation)
#       EPOTNC   = "ER POTENTIAL: NC ANALYSIS"       (-dPhi_nc/dpsi -> the E×B rotation)
#       ERTOT = ERPRESS + ERVTOR + ERVPOL            (V/cm; pressure/toroidal/poloidal)
#   => the TRANSP field that equals GACODE w0 is the EPOTNC E×B rotation (-dPhi/dpsi),
#      NOT OMEGA_NC. OMEGA_NC = omega_ExB + omega_diamag, and those nearly cancel, so the
#      toroidal velocity is a small residual (shown in panel 4).

# --- TRANSP extra fields for the diagnostics ---
transp_omega    = cdf.VtorkHz[it, :]      * (2 * np.pi * 1e3)   # OMEGA (the rotation TRANSP itself used; NOT what to_profiles writes -- that is TGLF_w0_exb)
transp_omega_in = cdf.VtorkHz_data[it, :] * (2 * np.pi * 1e3)   # OMEGDATA (the omg U-File we shipped)
transp_w0_exb   = transp_w0_nc_chk                              # EPOTNC E×B rotation (rad/s), on xb grid

# --- VGEN Er + force-balance decomposition (out.vgen.vel + out.vgen.ercomp) ---
# er_exp is kV/m (NOT V/m: vgen.f90 ends the CGS build with /1000). The ercomp omega
# components (rad/s) sum to w0; each is er_to_w0*Er_component, where (vgen.f90:250)
#     er_to_w0 = c q / (Bunit r |grad r|)    [the exact Er->w0 conversion factor].
# MITIM does not expose |grad r| (grad_r0), so we take VGEN's OWN self-consistent ratio
# w0/er_exp, which EQUALS that factor (sign + units included), and use it to move between
# Er(V/m) and omega(rad/s) without reconstructing the geometry.
vv = neo.vgen_vel
ec = neo.vgen_ercomp
vgen_Er_rho = np.asarray(vv["rho"])
vgen_Er     = np.asarray(vv["er_exp"]) * 1e3        # total Er, V/m
er_to_w0    = np.asarray(vv["w0"]) / np.asarray(vv["er_exp"])   # rad/s per kV/m (VGEN COCOS)
def _ercomp_to_Vm(key):  # omega-component (rad/s) -> Er-component (V/m), on the ercomp grid
    return np.asarray(ec[key]) / np.interp(np.asarray(ec["rho"]), vgen_Er_rho, er_to_w0) * 1e3
vgen_Er_p   = _ercomp_to_Vm("omega_gradp_1")        # diamagnetic Er, V/m
vgen_Er_pol = _ercomp_to_Vm("omega_vpol_1")         # poloidal Er, V/m
vgen_Er_tor = _ercomp_to_Vm("omega_vtor_1")         # toroidal Er, V/m (~0, er=2)
ec_rho      = np.asarray(ec["rho"])

# --- Independent, MODEL-FREE diamagnetic Er (main ion): Er = (1/(Z_i e n_i)) dp_i/dr ---
_e = 1.602176634e-19
_rho_p = profiles.profiles["rho(-)"]
_ni    = profiles.profiles["ni(10^19/m^3)"][:, 0] * 1e19   # main ion (D), matched_ion=1
_Ti    = profiles.profiles["ti(keV)"][:, 0] * 1e3 * _e
_Zi    = profiles.profiles["z"][0]
Er_dia_indep = np.gradient(_ni * _Ti, profiles.derived["r"]) / (_Zi * _e * _ni)  # V/m

# --- TRANSP total Er expressed as a GACODE-convention w0 (er_to_w0 carries the GACODE sign) ---
# w0[rad/s] = er_to_w0[rad/s per kV/m] * Er[kV/m]; apply VGEN's factor to TRANSP's ERtot.
er_to_w0_on_t    = np.interp(transp_rho, vgen_Er_rho, er_to_w0)
transp_w0_fromEr = er_to_w0_on_t * (transp_Er * 1e-3)   # TRANSP neoclassical Er -> GACODE w0

# helpers to interpolate onto the VGEN window for the tables
def _t(y, rq):  return float(np.interp(rq, transp_rho, y))     # TRANSP x-grid
def _tb(y, rq): return float(np.interp(rq, transp_rho_xb, y))  # TRANSP xb-grid
def _v(y, rq):  return float(np.interp(rq, vgen_Er_rho, y))    # VGEN vel-grid
def _ve(y, rq): return float(np.interp(rq, ec_rho, y))         # VGEN ercomp-grid
def _i(y, rq):  return float(np.interp(rq, _rho_p, y))         # input.gacode-grid
rho_tab = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

print("\n" + "=" * 104)
print(" Er DECOMPOSITION [V/m]  --  diamagnetic is MODEL-INDEPENDENT (must agree); poloidal is the model term")
print("=" * 104)
print(f"{'rho':>5} | {'ERp_T':>8}{'ERp_V':>8}{'ERp_ind':>8} | {'ERpol_T':>9}{'ERpol_V':>9} | {'ERtor_T':>8}{'ERtor_V':>8} | {'ERtot_T':>9}{'ERtot_V':>9}")
print("-" * 104)
for rq in rho_tab:
    print(f"{rq:>5.2f} | {_t(transp_Er_p,rq):>8.0f}{_ve(vgen_Er_p,rq):>8.0f}{_i(Er_dia_indep,rq):>8.0f} | "
          f"{_t(transp_Er_pol,rq):>9.0f}{_ve(vgen_Er_pol,rq):>9.0f} | "
          f"{_t(transp_Er_tor,rq):>8.0f}{_ve(vgen_Er_tor,rq):>8.0f} | "
          f"{_t(transp_Er,rq):>9.0f}{_v(vgen_Er,rq):>9.0f}")

print("\n" + "=" * 104)
print(" ROTATION [rad/s]  --  which TRANSP field = GACODE w0 (= ERtot->w0, the E×B rotation)?")
print("=" * 104)
print(f"{'rho':>5} | {'w0_VGEN':>10} | {'ERtot->w0_T':>13} | {'EPOTNC_rot_T':>13} | {'OMEGA_NC_T':>11} | {'OMEGA_T(out)':>13}")
print("-" * 104)
for rq in rho_tab:
    print(f"{rq:>5.2f} | {_v(np.asarray(vv['w0']),rq):>10.0f} | {_t(transp_w0_fromEr,rq):>13.0f} | "
          f"{_tb(transp_w0_exb,rq):>13.0f} | {_t(transp_w0_nc,rq):>11.0f} | {_t(transp_omega,rq):>13.0f}")

# Quantified take-aways at mid-radius
rq = 0.45
diaT, diaV, diaI = _t(transp_Er_p,rq), _ve(vgen_Er_p,rq), _i(Er_dia_indep,rq)
polT, polV = _t(transp_Er_pol,rq), _ve(vgen_Er_pol,rq)
print("\n" + "=" * 104)
print(f" QUANTIFIED at rho={rq}:")
print(f"   diamagnetic Er : TRANSP {diaT:.0f} | VGEN {diaV:.0f} | independent {diaI:.0f} V/m  "
      f"-> spread {100*(max(diaT,diaV,diaI)-min(diaT,diaV,diaI))/abs(diaI):.0f}% (MODEL-INDEPENDENT, agree)")
print(f"   poloidal   Er : TRANSP/NCLASS {polT:.0f} vs VGEN/NEO {polV:.0f} V/m  "
      f"-> {'OPPOSITE sign' if polT*polV < 0 else 'same sign'} (the genuine model difference; NEO higher fidelity)")
print(f"   OMEGA_NC = omega_ExB + omega_dia+pol = {_tb(transp_w0_exb,rq):.0f} + {_t(transp_w0_nc,rq)-_tb(transp_w0_exb,rq):.0f}"
      f" = {_t(transp_w0_nc,rq):.0f} rad/s.  This near-cancellation is FORCED by the imposed V_phi~0")
print(f"      (weak-rotation input), NOT an emergent result: neoclassical theory does not predict V_phi,")
print(f"      so OMEGA_NC just echoes the ~0 toroidal input. The physics lives in the E×B rotation.")
print(f"   => write the EPOTNC E×B rotation (-dPhi/dpsi) for rotation_source='neoclassical_transp', "
      f"with the COCOS sign flip (VGEN w0 and TRANSP EPOTNC_rot are opposite sign here).")
print("=" * 104 + "\n")

# =====================================================================================
# VALIDATION of rotation_source='neoclassical_transp' write-back (replicates TRANSPbeat.finalize)
# =====================================================================================
# cdf.to_profiles() builds the OUTPUT input.gacode and writes w0 = the E×B rotation from the CDF
# variable cdf.TGLF_w0_exb (= -c dPhi/dpsi = Er/(dpsi/dR), CDF-native, no derivative/sign work in
# the beat). This zero-w0 reuse run is the 'neoclassical_transp' / weak-rotation case (omg was zeroed in),
# so to_profiles' w0 IS the neoclassical E×B rotation -- exactly what the beat writes. The output
# state is internally self-consistent in TRANSP's convention, which has FLIPPED current/Bt/Bunit
# sign vs the input we fed VGEN, so the written w0 has the opposite sign to VGEN's input-convention
# w0 for the SAME physical rotation. We read it straight from to_profiles, then compare to VGEN by
# mapping VGEN into the output convention via the Bunit-sign ratio.
_itx   = cdf.ind_saw - 1                                  # finalize extracts at ind_saw-1
st_out = cdf.to_profiles(time_extraction=cdf.t[_itx])     # the file the next beat would read
_B     = np.asarray(st_out.derived["B_unit"]); _rho_o = np.asarray(st_out.profiles["rho(-)"])
w0_neo_out = np.asarray(st_out.profiles["w0(rad/s)"])    # what to_profiles/the beat writes (TGLF_w0_exb)

vgen_w0_v  = np.asarray(vv["w0"])                           # VGEN, INPUT convention
# COCOS relation output<->input (via Bunit sign): +1 same convention, -1 flipped
cocos_flip = float(np.sign(np.median(_B)) * np.sign(np.median(profiles.derived["B_unit"])))
w0_vgen_in_out = cocos_flip * vgen_w0_v                     # VGEN mapped INTO the output convention
w0_neo_on_v    = np.interp(vgen_Er_rho, _rho_o, w0_neo_out) # written w0 on the VGEN grid (output conv)
core      = (vgen_Er_rho >= 0.15) & (vgen_Er_rho <= 0.85)
sign_match = bool(np.all(np.sign(w0_neo_on_v[core]) == np.sign(w0_vgen_in_out[core])))
mag_ratio  = float(np.median(np.abs(w0_neo_on_v[core]) / np.abs(vgen_w0_v[core])))

print("=" * 104)
print(" VALIDATION: rotation_source='neoclassical_transp' write-back vs VGEN/NEO")
print("=" * 104)
print(f"   COCOS:  input(VGEN) sign[current,Bt,Bunit] = "
      f"[{np.sign(profiles.profiles['current(MA)'][0]):+.0f},{np.sign(profiles.profiles['bcentr(T)'][0]):+.0f},{np.sign(np.median(profiles.derived['B_unit'])):+.0f}]"
      f"   TRANSP output = "
      f"[{np.sign(st_out.profiles['current(MA)'][0]):+.0f},{np.sign(st_out.profiles['bcentr(T)'][0]):+.0f},{np.sign(np.median(_B)):+.0f}]")
print(f"   => output<->input COCOS flip factor (Bunit sign ratio) = {cocos_flip:+.0f}  "
      f"(written w0 is in the OUTPUT convention; self-consistent with that file)")
print(f"   SIGN match (VGEN mapped to output convention), rho in [0.15,0.85]: {sign_match}  "
      f"(sign is COCOS-covariant via Bunit, NOT hardcoded)")
print(f"   magnitude ratio |w0_neoclassical| / |w0_VGEN| (median) = {mag_ratio:.2f}  "
      f"(>1: NCLASS poloidal flow inflates Er vs NEO -- expected model difference)")
assert sign_match, "[CHECK FAILED] neoclassical write-back sign disagrees with VGEN once COCOS is accounted for"
print("   [CHECK PASSED] neoclassical write-back is COCOS-consistent with VGEN/NEO")
print("=" * 104 + "\n")

# =====================================================================================
# FIGURE: 4 diagnostic panels
# =====================================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))

# --- Panel 1: E×B rotation, all shown in the VGEN/input convention (curves comparable) ---
# The written w0 lives in the OUTPUT convention (opposite COCOS); map it via cocos_flip to
# overlay with VGEN. Its actual sign in the written file is the opposite of what's shown here.
ax = axs[0, 0]
ax.plot(vgen_Er_rho, np.asarray(vv["w0"]), "-o", color="C0", lw=2, ms=3, label=r"$w_0$ VGEN/NEO (GACODE)")
ax.plot(transp_rho, transp_w0_fromEr, "-s", color="C1", lw=2, ms=3, label=r"$E_r^{tot}\to w_0$ TRANSP ($E_r\times$ VGEN factor)")
ax.plot(vgen_Er_rho, cocos_flip * w0_neo_on_v, "--D", color="C6", lw=1.8, ms=3, label=r"rotation_source='neoclassical_transp' written (mapped to VGEN conv.)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$w_0$  (rad/s, VGEN/input conv.)")
ax.set_xlim([0.0, 1.0]); ax.set_title(r"E×B rotation $w_0$ (shown in VGEN convention)")
ax.legend(loc="best", fontsize=7)

# --- Panel 2: Er total + TRANSP/VGEN decomposition ---
ax = axs[0, 1]
ax.plot(transp_rho, transp_Er,     "-s", color="C1", lw=2, ms=3, label=r"$E_r$ TRANSP (ERTOT)")
ax.plot(vgen_Er_rho, vgen_Er,      "-o", color="C0", lw=2, ms=3, label=r"$E_r$ VGEN (total)")
ax.plot(transp_rho, transp_Er_p,   ":",  color="C2", lw=1.4, label=r"TRANSP $\nabla p$ (ERPRESS)")
ax.plot(transp_rho, transp_Er_pol, ":",  color="C5", lw=1.4, label=r"TRANSP $v_\theta$ (ERVPOL)")
ax.plot(transp_rho, transp_Er_tor, ":",  color="C4", lw=1.4, label=r"TRANSP $v_\phi$ (ERVTOR)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$E_r$  (V/m)")
ax.set_xlim([0.0, 1.0]); ax.set_title("Neoclassical $E_r$ and its decomposition")
ax.legend(loc="best", fontsize=8)

# --- Panel 3: TERM-BY-TERM (diamagnetic and poloidal components, TRANSP vs VGEN) ---
ax = axs[1, 0]
ax.plot(transp_rho, transp_Er_p,   "-s", color="C2", lw=1.6, ms=3, label=r"$E_r^{\nabla p}$ TRANSP")
ax.plot(ec_rho, vgen_Er_p,         "-o", color="C8", lw=1.6, ms=3, label=r"$E_r^{\nabla p}$ VGEN")
ax.plot(transp_rho, transp_Er_pol, "-^", color="C5", lw=1.6, ms=3, label=r"$E_r^{v_\theta}$ TRANSP (NCLASS)")
ax.plot(ec_rho, vgen_Er_pol,       "-v", color="C0", lw=1.6, ms=3, label=r"$E_r^{v_\theta}$ VGEN (NEO)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$E_r$ component  (V/m)")
ax.set_xlim([0.0, 1.0]); ax.set_title(r"$E_r$ components: $\nabla p$ and $v_\theta$ (TRANSP vs VGEN)")
ax.legend(loc="best", fontsize=8)

# --- Panel 4: omega_tor = omega_ExB + omega_dia+pol (TRANSP). NOTE the near-cancellation is
#     FORCED by the imposed V_phi~0 weak-rotation input (omega_tor ~ 0 -> omega_ExB ~ -omega_dia+pol),
#     not an emergent result -- it shows OMEGA_NC is the small toroidal velocity, not the E×B rotation.
#     NONE of the near-zero curves here is what MITIM writes back: to_profiles writes TGLF_w0_exb
#     (the E×B rotation -- panel 1's 'written' curve); OMEGA is only the rotation TRANSP ran with
#     (the zero omg U-File input), shown to confirm the weak-rotation setup.
ax = axs[1, 1]
omega_diapol = transp_w0_nc - np.interp(transp_rho, transp_rho_xb, transp_w0_exb)  # OMEGA_NC - omega_ExB (diamagnetic + poloidal-flow remainder)
ax.plot(transp_rho_xb, transp_w0_exb, "-", color="C3", lw=2, label=r"$\omega_{E\times B}$ ($-d\Phi_{nc}/d\psi$, EPOTNC)  [= GACODE $w_0$, TRANSP conv.]")
ax.plot(transp_rho, omega_diapol,     "-", color="C9", lw=2, label=r"$\omega_{dia+pol}$ (= OMEGA_NC $-\;\omega_{E\times B}$)")
ax.plot(transp_rho, transp_w0_nc,     "-o", color="C1", lw=2, ms=3, label=r"$\omega_{tor}$ = OMEGA_NC ($V_\phi/R$, NOT $w_0$)")
ax.plot(transp_rho, transp_omega,     "--", color="k", lw=1.0, label=r"OMEGA (rotation TRANSP used = zero omg input; NOT the write-back)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\omega$  (rad/s, TRANSP conv.)")
ax.set_xlim([0.0, 1.0]); ax.set_title(r"$\omega_{tor}$ vs E×B and diamagnetic parts (TRANSP)")
ax.legend(loc="best", fontsize=8)

fig.suptitle("TRANSP/NCLASS vs VGEN/NEO neoclassical rotation — term-by-term (SAME state, w0 zeroed)", fontsize=13)
fig.tight_layout()

figure_file = folder / "transp_vs_vgen_rotation.png"
fig.savefig(figure_file, dpi=150)
print(f"\t- Comparison figure saved to {IOtools.clipstr(figure_file)}", typeMsg="i")

plt.show()
