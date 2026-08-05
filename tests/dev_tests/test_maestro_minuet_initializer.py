'''
Dev-test: MINUET as the fixed-boundary Grad-Shafranov solver behind MAESTRO's initialization.

MITIM initializes MAESTRO equilibria from engineering parameters at three places that used to be
FREEGS-only. All three now go through the MINUET fixed-boundary solver (GEQtools.minuet_millerized)
when the optional minuet package is installed, and this test exercises each one:

    PART A: initialization_type = "minuet" -- a full (short) MAESTRO chain that starts PURELY from
            engineering parameters (SPARC-like R = 1.85 m, a = 0.57 m, Bt = 12.2 T, Ip = 8.7 MA),
            builds the equilibrium with MINUET, writes minuet.geqdsk, and hands it to the geqdsk
            initializer + the 'fixed_bc' profile creator. This is an EXPLICIT request, so there is
            no FREEGS fallback: no freegs artifact may appear anywhere.

    PART B: initialization_type = "separatrix" -- here the equilibrium solver only CORRECTS the
            guessed profiles (the shaping is kept). MINUET is PREFERRED and FREEGS is the fallback,
            so the same initializer object is driven twice: once normally (must produce
            minuet.geqdsk.helper) and once with GEQtools.minuet_available monkeypatched to False
            (must produce freegs.geqdsk.helper). Driven directly, without a MAESTRO chain.

    PART C: the TRANSP t = 0 "machine initialization" morph -- transp_input_time.from_minuet vs
            .from_freegs on the hardcoded C-Mod machine parameters that TRANSPbeat uses. Checks
            that the MINUET path lays down the SAME set of time-slice quantities as the FREEGS
            path (contract equivalence) and that the equilibrium it encodes is the requested one.

After every assertion has passed, a VISUALIZATION stage builds a FigureNotebook so the equilibria
can be looked at, not just trusted from numbers (all of it via the standard MITIM plotting
machinery -- MITIMgeqdsk.plot, state_plotting.plotAll, mitim_flux_surfaces.plot):

    A) the minuet.geqdsk the initializer wrote, through the full MITIMgeqdsk notebook (shape,
       plasma profiles, flux surfaces, currents, fields, GS quality, geometry);
    A) requested Miller boundary vs the LCFS traced from the solved MINUET field (fixed boundary:
       these must lie on top of each other);
    A) initializer state vs post-beat state overlaid (the q profile relaxes by current diffusion);
    B) MINUET-corrected vs FREEGS-corrected separatrix states overlaid (shaping is identical by
       construction -- what differs is q / pressure / poloidal flux);
    C) the TRANSP t = 0 slice from both solvers: boundary polygons, q(rhotor), p(rhotor).

The notebook is shown on screen by default and the PNGs are ALWAYS written to
tests/scratch/dev_maestro_minuet_initializer/figures/.

Run from the dev-pixi root:
    ./run_with_env.sh python MITIM-fusion/tests/dev_tests/test_maestro_minuet_initializer.py
    [--no-notebook]  (headless: only write the PNGs, do not open the GUI)
'''

import sys
import numpy as np
import torch
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_modules.maestro.scripts import run_maestro

cold_start = True
no_notebook = "--no-notebook" in sys.argv

folder = __mitimroot__ / "tests" / "scratch" / "dev_maestro_minuet_initializer"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(8)

assert GEQtools.minuet_available(), "this test requires the optional minuet package to be installed"

# ================================================================================================
# PART A: full MAESTRO chain initialized with initialization_type = "minuet"
# ================================================================================================

print("\n" + "="*100)
print(" PART A: MAESTRO chain with initialization_type = 'minuet'")
print("="*100)

folder_A = folder / "partA_initialization_type_minuet"
folder_A.mkdir(parents=True, exist_ok=True)

nml = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.maestro.yaml")

# Equilibrium from MINUET, profiles from the fixed_bc creator (no pedestal code involved)
nml["plasma"]["profiles_initialization"]["initialization_type"] = "minuet"
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV (Ti_bc: null -> same as Te_bc)

# 10 MXH moments: fold-free stored family for the minuet beat (see test_maestro_minuet_beat.py)
nml["plasma"]["parameters"]["separatrix"]["n_mxh"] = 10

# Gaussian auxiliary sources (no TRANSP in this chain)
nml["plasma"]["heating"]["type"] = "gaussian_sources"
nml["plasma"]["heating"]["parameters"]["Pe"] = 10.0
nml["plasma"]["heating"]["parameters"]["Pi"] = 5.0
nml["plasma"]["heating"]["parameters"]["nu_source"] = 5.0
nml["plasma"]["heating"]["parameters"]["fmini"] = 0.0

# Single cheap beat: the point of this test is the INITIALIZER, not the beat
nml["maestro"]["beats"] = ["minuet"]
nml["maestro"]["minuet"]["parameters_prepare"]["t_end"] = 2.0
nml["maestro"]["minuet"]["parameters_prepare"]["n_save"] = 51

Ip_nml = nml["plasma"]["parameters"]["Ip"]
R_nml = nml["plasma"]["parameters"]["separatrix"]["R"]
a_nml = nml["plasma"]["parameters"]["separatrix"]["a"]

namelist_file = folder_A / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

run_maestro.run_maestro_local(
    namelist_file,
    folder = folder_A,
    terminal_outputs = True,
    force_cold_start = cold_start,
    cpus = 8,
)

b1 = folder_A / "Beats" / "Beat_1"

# The MINUET initializer ran and wrote its geqdsk in its own initializer folder
assert (b1 / "initializer_minuet" / "minuet.geqdsk").exists(), "MINUET initializer produced no minuet.geqdsk"
assert (b1 / "initializer_minuet" / "input.geqdsk.gacode").exists(), "MINUET initializer did not chain to the geqdsk initializer"

# initialization_type = minuet is EXPLICIT: no freegs anywhere
assert not (b1 / "initializer_freegs").exists(), "a freegs initializer folder appeared despite initialization_type=minuet"
for stray in ["freegs.geqdsk", "freegs.geqdsk.helper"]:
    assert not (b1 / "initializer_minuet" / stray).exists(), f"freegs artifact {stray} appeared in the minuet initializer folder"
log_file = folder_A / "Outputs" / "maestro.log"
if log_file.exists():
    log_txt = log_file.read_text()
    assert "with FREEGS" not in log_txt, "the FREEGS solver banner appears in the maestro log"
    assert "with MINUET" in log_txt, "the MINUET solver banner is missing from the maestro log"

# The state the beat received (pre-beat) must carry the requested engineering parameters
p_in = PROFILEStools.gacode_state(b1 / "run_minuet" / "input.gacode")
p_in.derive_quantities()

Ip_state = float(p_in.profiles["current(MA)"][0])
R_state = float(p_in.profiles["rmaj(m)"][-1])
a_state = float(p_in.profiles["rmin(m)"][-1])
q_state = p_in.profiles["q(-)"]

print(f"\t- Ip: {Ip_state:.5f} MA (namelist {Ip_nml} MA, {100*(Ip_state/Ip_nml-1):+.3f}%)")
print(f"\t- R0: {R_state:.5f} m  (namelist {R_nml} m, {100*(R_state/R_nml-1):+.3f}%)")
print(f"\t- a : {a_state:.5f} m  (namelist {a_nml} m, {100*(a_state/a_nml-1):+.3f}%)")
print(f"\t- q : q(0) = {q_state[0]:.3f}, q(edge) = {q_state[-1]:.3f}")

assert abs(Ip_state/Ip_nml - 1) < 0.01, "Ip of the MINUET-initialized state is off by more than 1%"
assert abs(R_state/R_nml - 1) < 0.005, "R0 of the MINUET-initialized state is off by more than 0.5%"
assert abs(a_state/a_nml - 1) < 0.005, "a of the MINUET-initialized state is off by more than 0.5%"
assert np.all(q_state > 0), "q profile is not positive everywhere"
assert q_state[-1] > q_state[0], "q profile is not rising from axis to edge"

# The chain completed
assert (b1 / "beat_results" / "input.gacode").exists(), "minuet beat produced no beat_results/input.gacode"

print("\nPART A OK: MINUET built the initial equilibrium and the chain completed, no FREEGS involved")

# ================================================================================================
# PART B: initialization_type = "separatrix" -- MINUET preferred, FREEGS fallback
# ================================================================================================

print("\n" + "="*100)
print(" PART B: separatrix initializer, MINUET-preferred correction + forced FREEGS fallback")
print("="*100)

folder_B = folder / "partB_separatrix_correction"
folder_B.mkdir(parents=True, exist_ok=True)

from mitim_modules.maestro.utils import MAESTRObeat

class _StubMaestro:
    '''Minimal stand-in for the maestro instance the initializer talks back to'''
    def __init__(self):
        self.maestro_namelist = {}
        self.parameters_trans_beat = {}

class _StubBeat:
    '''Minimal stand-in for the beat that owns the initializer (only folder_beat is used)'''
    def __init__(self, folder_beat):
        self.folder_beat = folder_beat
        self.maestro_instance = _StubMaestro()

# Same engineering inputs as PART A
kwargs_sep = dict(
    Paux_MW = 15.0, Zeff = 1.5, netop_20 = 2.5, coeffs_MXH = 5,
    extract_995_from = None,
    R = R_nml, a = a_nml, kappa_sep = nml["plasma"]["parameters"]["separatrix"]["kappa_sep"],
    delta_sep = nml["plasma"]["parameters"]["separatrix"]["delta_sep"],
    zeta_sep = nml["plasma"]["parameters"]["separatrix"]["zeta_sep"],
    z0 = 0.0, Ip_MA = Ip_nml, B_T = nml["plasma"]["parameters"]["Bt"],
    BetaN = 1.0, internal_flux_file = None, rz_boundary_file = None,
    )

# ---- B.1: minuet available -> MINUET must do the correction -----------------------------------

folder_B_minuet = folder_B / "with_minuet"
folder_B_minuet.mkdir(parents=True, exist_ok=True)

ini_minuet = MAESTRObeat.initializer_from_separatrix(_StubBeat(folder_B_minuet))
ini_minuet(**kwargs_sep)

assert (ini_minuet.folder / "minuet.geqdsk.helper").exists(), "separatrix correction did not go through MINUET"
assert not (ini_minuet.folder / "freegs.geqdsk.helper").exists(), "FREEGS ran even though MINUET succeeded"
assert (ini_minuet.folder / "input.separatrix.gacode").exists(), "separatrix initializer produced no state"

Ip_B_minuet = float(ini_minuet.p.profiles["current(MA)"][0])
print(f"\t- [MINUET] corrected state Ip = {Ip_B_minuet:.5f} MA (requested {Ip_nml} MA)")
assert abs(Ip_B_minuet/Ip_nml - 1) < 0.01, "MINUET-corrected separatrix state Ip off by more than 1%"

# ---- B.2: minuet forced unavailable -> FREEGS fallback ----------------------------------------

print("\n" + "-"*100)
print(" B.2: minuet_available() monkeypatched to False -- the FREEGS banner below is EXPECTED:")
print("      this leg verifies the fallback, it is NOT a leak of FREEGS into the minuet paths")
print("-"*100)

folder_B_freegs = folder_B / "forced_freegs"
folder_B_freegs.mkdir(parents=True, exist_ok=True)

minuet_available_original = GEQtools.minuet_available
GEQtools.minuet_available = lambda: False
try:
    ini_freegs = MAESTRObeat.initializer_from_separatrix(_StubBeat(folder_B_freegs))
    ini_freegs(**kwargs_sep)
finally:
    GEQtools.minuet_available = minuet_available_original

assert (ini_freegs.folder / "freegs.geqdsk.helper").exists(), "FREEGS fallback did not run when minuet was unavailable"
assert not (ini_freegs.folder / "minuet.geqdsk.helper").exists(), "MINUET ran despite being reported unavailable"
assert (ini_freegs.folder / "input.separatrix.gacode").exists(), "FREEGS fallback produced no state"

Ip_B_freegs = float(ini_freegs.p.profiles["current(MA)"][0])
print(f"\t- [FREEGS] corrected state Ip = {Ip_B_freegs:.5f} MA (requested {Ip_nml} MA)")
assert abs(Ip_B_freegs/Ip_nml - 1) < 0.05, "FREEGS-corrected separatrix state Ip off by more than 5%"

print("\nPART B OK: MINUET corrects the separatrix guess when available, FREEGS takes over when not")

# ================================================================================================
# PART C: TRANSP machine-initialization morph, from_minuet vs from_freegs
# ================================================================================================

print("\n" + "="*100)
print(" PART C: transp_input_time.from_minuet vs .from_freegs (C-Mod machine initialization)")
print("="*100)

from mitim_tools.transp_tools.utils import TRANSPhelpers

# Hardcoded C-Mod initialization machine of TRANSPbeat._additional_operations_add_initialization
R_c, a_c, kappa_c, delta_c, zeta_c, z0_c = 0.68, 0.22, 1.5, 0.46, 0.0, 0.0
p0_c, Ip_c, B_c, ne0_c = 0.3, 1.0, 5.4, 1.0
time_c = 0.0

class _StubTransp:
    '''Minimal stand-in for the transp instance transp_input_time populates'''
    def __init__(self):
        self.variables = {}
        self.geometry = {}

pt_minuet = TRANSPhelpers.transp_input_time(_StubTransp())
pt_minuet.from_minuet(time_c, R_c, a_c, kappa_c, delta_c, zeta_c, z0_c, p0_c, Ip_c, B_c, ne0_20 = ne0_c)

pt_freegs = TRANSPhelpers.transp_input_time(_StubTransp())
pt_freegs.from_freegs(time_c, R_c, a_c, kappa_c, delta_c, zeta_c, z0_c, p0_c, Ip_c, B_c, ne0_20 = ne0_c)

v_m = pt_minuet.transp_instance.variables[time_c]
v_f = pt_freegs.transp_instance.variables[time_c]
g_m = pt_minuet.transp_instance.geometry[time_c]
g_f = pt_freegs.transp_instance.geometry[time_c]

# Contract equivalence: same laid-down quantities
assert set(v_m.keys()) == set(v_f.keys()), f"from_minuet lays down {set(v_m.keys())} vs from_freegs {set(v_f.keys())}"
assert set(g_m.keys()) == set(g_f.keys()), f"from_minuet geometry {set(g_m.keys())} vs from_freegs {set(g_f.keys())}"
print(f"\t- time-slice quantities laid down (both solvers): {sorted(v_m.keys())}")

# Physics of the MINUET slice
Ip_A = float(v_m['CUR']['z'])
RB_cmT = float(v_m['RBZ']['z'])
q_m = np.array(v_m['QPR']['z'])
p_m = np.array(v_m['TEL']['z'])  # T_eV, from pressure; positive iff pressure is
Rsep, Zsep = np.array(g_m['R_sep']), np.array(g_m['Z_sep'])

print(f"\t- Ip   = {Ip_A*1E-6:.5f} MA (requested {Ip_c} MA, {100*(Ip_A*1E-6/Ip_c-1):+.3f}%)")
print(f"\t- R*Bt = {RB_cmT:.3f} cm*T (requested {R_c*B_c*100:.3f} cm*T, {100*(RB_cmT/(R_c*B_c*100)-1):+.3f}%)")
print(f"\t- q    = [{q_m.min():.3f}, {q_m.max():.3f}], T_eV = [{p_m.min():.1f}, {p_m.max():.1f}]")

assert abs(Ip_A*1E-6/Ip_c - 1) < 0.01, "from_minuet Ip off by more than 1%"
assert abs(RB_cmT/(R_c*B_c*100) - 1) < 0.01, "from_minuet R*Bt off by more than 1%"
assert np.all(np.isfinite(q_m)) and np.all(q_m > 0), "from_minuet q profile not finite and positive"
assert np.all(np.isfinite(p_m)) and np.all(p_m >= 0) and p_m[0] > 0, "from_minuet pressure/temperature not finite and non-negative"

# Boundary is a closed curve (the theta grid excludes the 2*pi endpoint, so check the wrap gap)
gaps = np.hypot(np.diff(np.append(Rsep, Rsep[0])), np.diff(np.append(Zsep, Zsep[0])))
print(f"\t- boundary: {Rsep.size} points, max point-to-point gap {gaps.max()*1E3:.2f} mm (wrap gap {gaps[-1]*1E3:.2f} mm)")
assert gaps.max() < 0.05 * a_c, "from_minuet boundary polygon is not closed / too coarse"

# The boundary is the requested Miller curve (fixed-boundary: exact by construction)
R0_b = 0.5*(Rsep.max() + Rsep.min())
a_b = 0.5*(Rsep.max() - Rsep.min())
kappa_b = 0.5*(Zsep.max() - Zsep.min())/a_b
print(f"\t- boundary R0 = {R0_b:.5f} m, a = {a_b:.5f} m, kappa = {kappa_b:.5f}")
assert abs(R0_b/R_c - 1) < 1E-3 and abs(a_b/a_c - 1) < 1E-3 and abs(kappa_b/kappa_c - 1) < 1E-3, \
    "from_minuet boundary does not reproduce the requested Miller curve"

print("\nPART C OK: from_minuet lays down the same TRANSP time-slice contract as from_freegs")

# ================================================================================================

print("\n" + "="*100)
print("PASS: MINUET drives all three MAESTRO initialization sites")
print("  A) initialization_type='minuet' -> minuet.geqdsk, correct Ip/R/a/q, no FREEGS fallback used")
print("  B) separatrix correction -> MINUET when available, FREEGS when not (both produce a state)")
print("  C) TRANSP machine morph -> from_minuet matches from_freegs's contract with the right Ip/R*Bt/boundary")
print("="*100)

# ================================================================================================
# VISUALIZATION: the numbers are checked above, here the equilibria are made visible
# ================================================================================================

print("\n" + "="*100)
print(" VISUALIZATION: assembling the figure notebook")
print("="*100)

folder_figs = folder / "figures"

fn = FigureNotebook("MAESTRO initialization with MINUET", geometry="1800x900", show=not no_notebook)

# ---- A: the geqdsk MINUET wrote, through the standard MITIMgeqdsk notebook ---------------------

g_A = GEQtools.MITIMgeqdsk(b1 / "initializer_minuet" / "minuet.geqdsk")
g_A.plot(fn=fn, extraLabel="A) geq - ", tab_color=0)

# ---- A: requested Miller boundary vs the LCFS written into minuet.geqdsk ------------------------
# MINUET is fixed-boundary, so the boundary it exports (rbbbs/zbbbs) should BE the requested curve

sep_nml = nml["plasma"]["parameters"]["separatrix"]

sep_requested = GEQtools.mitim_flux_surfaces()
sep_requested.reconstruct_from_miller(
    R_nml, a_nml, sep_nml["kappa_sep"],
    0.0,  # z0: MAESTRO hardcodes the magnetic-axis height of the requested boundary to 0
    sep_nml["delta_sep"], sep_nml["zeta_sep"],
    thetas = np.linspace(0, 2*np.pi, 1000, endpoint=False))

def _max_separation(R, Z):
    '''
    Max over the curve of the distance to the requested Miller curve [m]. Distance is measured to
    the closed POLYLINE (nearest point on each segment), not to its vertices: a vertex-only metric
    would report half the requested curve's own point spacing instead of a geometry error
    '''
    R1, Z1 = sep_requested.R[0], sep_requested.Z[0]
    dR, dZ = np.roll(R1, -1) - R1, np.roll(Z1, -1) - Z1
    t = np.clip(((np.asarray(R)[:, None] - R1)*dR + (np.asarray(Z)[:, None] - Z1)*dZ) / (dR**2 + dZ**2), 0.0, 1.0)
    return np.hypot(
        np.asarray(R)[:, None] - (R1 + t*dR),
        np.asarray(Z)[:, None] - (Z1 + t*dZ)).min(axis=1).max()

fig = fn.add_figure(label="A) Boundary req. vs achieved", tab_color=0)
ax = fig.add_subplot(111)
sep_requested.plot(ax=ax, color="b", label="requested Miller boundary (namelist)")
ax.plot(np.asarray(g_A.Rb_gfile), np.asarray(g_A.Yb_gfile), "--", color="r", lw=2, label="LCFS written to minuet.geqdsk")
ax.set_aspect("equal")
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.legend(loc="best", prop={"size": 8})
ax.set_title("PART A: fixed boundary -- the requested curve IS the boundary")

print(f"\t- [A] geqdsk LCFS vs requested Miller curve: max separation "
      f"{_max_separation(g_A.Rb_gfile, g_A.Yb_gfile)*1E3:.3f} mm")

# ---- A: what the beat did to the initialized state (q relaxes by current diffusion) ------------

p_beat = PROFILEStools.gacode_state(b1 / "beat_results" / "input.gacode")

figs_A = state_plotting.add_figures(fn, fnlab_pre="A) ", tab_color=0)
state_plotting.plotAll([p_in, p_beat], figs=figs_A, extralabs=["initializer (pre-beat)", "post minuet beat"])

# ---- B: MINUET-corrected vs FREEGS-corrected separatrix states ---------------------------------
# The separatrix initializer keeps the SHAPING (it only corrects the internal equilibrium), so the
# geometry tabs are identical by construction: what the two solvers disagree on is q / p / polflux

print("\t- [B] shaping is identical by construction (the separatrix initializer only corrects the")
print("\t      internal equilibrium): look at q, pressure and poloidal flux for the solver difference")

figs_B = state_plotting.add_figures(fn, fnlab_pre="B) ", tab_color=1)
state_plotting.plotAll([ini_minuet.p, ini_freegs.p], figs=figs_B, extralabs=["MINUET-corrected", "FREEGS-corrected"])

# ---- C: the TRANSP t = 0 slice from both solvers ------------------------------------------------
# No MITIM plotter reads a raw transp_input_time slice, so this one is plain matplotlib

fig = fn.add_figure(label="C) TRANSP t=0 slice", tab_color=2)
axs = fig.subplots(1, 3)

e_J = 1.60217662e-19

for v, g, color, label in [(v_m, g_m, "b", "MINUET"), (v_f, g_f, "r", "FREEGS")]:

    Rs, Zs = np.asarray(g['R_sep']), np.asarray(g['Z_sep'])
    axs[0].plot(np.append(Rs, Rs[0]), np.append(Zs, Zs[0]), "-", color=color, lw=2, label=label)

    rhotor = np.asarray(v['QPR']['x'])
    axs[1].plot(rhotor, np.asarray(v['QPR']['z']), "-", color=color, lw=2, label=label)

    # TEL is T_eV = p/(2*e*ne) and NEL is ne in cm^-3, so p [Pa] = TEL * 2*e*NEL*1E6
    p_Pa = np.asarray(v['TEL']['z']) * 2.0 * e_J * np.asarray(v['NEL']['z']) * 1E6
    axs[2].plot(rhotor, p_Pa * 1E-3, "-", color=color, lw=2, label=label)

axs[0].set_aspect("equal")
axs[0].set_xlabel("R [m]")
axs[0].set_ylabel("Z [m]")
axs[0].set_title("C-Mod boundary polygon")

axs[1].set_xlabel("$\\rho_{tor}$")
axs[1].set_ylabel("q")
axs[1].set_xlim([0, 1])
axs[1].set_title("safety factor")

axs[2].set_xlabel("$\\rho_{tor}$")
axs[2].set_ylabel("$p$ [kPa]")
axs[2].set_xlim([0, 1])
axs[2].set_title("total pressure ($p = 2 e n_e T$)")

for ax in axs:
    ax.legend(loc="best", prop={"size": 8})
    GRAPHICStools.addDenseAxis(ax)

# ---- Always write the PNGs, show the notebook unless asked not to -------------------------------

saved = fn.save(folder_figs, dpi=120, prefix="minuet_init")
print(f"\n{len(saved)} figures written to {folder_figs}")

if not no_notebook:
    fn.show()
