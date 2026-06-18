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

    PATH B  --  VGEN (profiles_gen -vgen) with er=2 = the NEO WEAK-ROTATION
                neoclassical limit: NEO is solved over the flux surfaces and
                returns the neoclassical Er and the consistent w0(rad/s).

The two neoclassical rotation profiles are printed as a table and overlaid in a
matplotlib figure (w0 / omega vs rho, and Er vs rho).

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
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "test_transp_vs_vgen_rotation"

# Bundled DT SPARC PRD plasma state (T,D fuel + F,W,He; w0 = 0 everywhere, so both
# paths genuinely compute the neoclassical rotation rather than echoing an input).
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# Tokamak name selects machine-specific TRANSP conventions (SPARC adds df4/vc4
# UFILEs and nteq_mode=2; see TRANSPhelpers.default_nml:1433).
tokamak = "SPARC"

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
time_extraction = 0.5   # s at which to write the averaged AC/output snapshot

# ---------------------------------------------------------------------------------------------------------------------
# A.1 Build the TRANSP run from input.gacode (the canonical input.gacode -> TRANSP path)
# ---------------------------------------------------------------------------------------------------------------------

# gacode_state.to_transp() (MITIMstate.py:3209) returns a TRANSPhelpers.transp_run
# already populated with the UFILE-able quantities at the requested times.
profiles = PROFILEStools.gacode_state(input_gacode)

times = [time_init, time_end + 1.0]  # bracket the flattop (matches TRANSPbeat usage)
transp = profiles.to_transp(
    folder=folderTRANSP,
    shot=shot,
    runid=runid,
    times=times,
    Vsurf=0.0,
)

# ---------------------------------------------------------------------------------------------------------------------
# A.2 Write a DEFAULT namelist (NCLASS + neoclassical potential both on by default)
# ---------------------------------------------------------------------------------------------------------------------

# Pich=True keeps the ICRF heating that the SPARC PRD case uses; DTplasma=True for
# the D-T fuel mix. NO PTsolver, so NO predictive/anomalous momentum machinery is
# emitted — NCLASS (Houlberg) runs as the neoclassical model only. The NCLASS
# neoclassical potential (nlvwnc) is on by default (computeNCLASSpotential=True),
# so EPOTNC/OMEGA_NC are written to the CDF.
transp.write_namelist(
    timings={
        "time_start": time_init,
        "time_current_diffusion": time_current_diffusion,
        "time_end": time_end,
        "time_extraction": time_extraction,
    },
    Pich=True,
    DTplasma=True,
)

# ---------------------------------------------------------------------------------------------------------------------
# A.3 Write UFILEs and submit; wait for completion; fetch the AC/CDF outputs
# ---------------------------------------------------------------------------------------------------------------------

transp.write_ufiles(mxh_coeffs_smooth=5)

# transp_run.run() wraps TRANSPtools.TRANSP + defineRunParameters + run +
# checkUntilFinished (TRANSPhelpers.run:382). retrieveAC=True pulls the averaged
# output snapshot written at time_extraction.
transp.run(
    tokamak,
    mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 32},
    minutesAllocation=30,
    case="neoclassical",
    checkMin=2,
    grabIntermediateEachMin=1e6,
    retrieveAC=True,
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
transp_Er       = cdf.Er[it, :]                         # V/m (ERTOT)
transp_Er_p     = cdf.Er_p[it, :]                       # V/m (diamagnetic / grad-p)
transp_Er_tor   = cdf.Er_tor[it, :]                     # V/m (toroidal-flow term)
transp_Er_pol   = cdf.Er_pol[it, :]                     # V/m (poloidal-flow term)

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

# Er used/derived by VGEN, V/m (out.vgen.vel -> vgen_vel["er_exp"], NEOtools.py:822).
# This lives on the (possibly truncated) vgen rho grid, separate from vgen_rho.
if neo.vgen_vel and "er_exp" in neo.vgen_vel:
    vgen_Er_rho = neo.vgen_vel["rho"]
    vgen_Er     = neo.vgen_vel["er_exp"]
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

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

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
ax.plot(transp_rho_xb, transp_Er, "-s", color="C1", lw=1.8, ms=3, label=r"$E_r$ TRANSP (total)")
ax.plot(transp_rho_xb, transp_Er_p, ":", color="C2", lw=1.2, label=r"$E_r$ TRANSP ($\nabla p$)")
ax.plot(transp_rho_xb, transp_Er_tor, ":", color="C4", lw=1.2, label=r"$E_r$ TRANSP (tor)")
ax.plot(transp_rho_xb, transp_Er_pol, ":", color="C5", lw=1.2, label=r"$E_r$ TRANSP (pol)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"$\rho$  (sqrt norm. tor. flux)")
ax.set_ylabel(r"$E_r$  (V/m)")
ax.set_xlim([0.0, 1.0])
ax.set_title("Neoclassical radial electric field")
ax.legend(loc="best", fontsize=8)

fig.suptitle("TRANSP/NCLASS vs VGEN/NEO neoclassical rotation (SAME plasma state)")
fig.tight_layout()

figure_file = folder / "transp_vs_vgen_rotation.png"
fig.savefig(figure_file, dpi=150)
print(f"\t- Comparison figure saved to {IOtools.clipstr(figure_file)}", typeMsg="i")

plt.show()
