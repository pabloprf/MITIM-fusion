"""
test_bc_servo_response_fit.py
=============================
Exercises the 'response_fit' BC servo (SHARPNESSbeat.record_bc_response + servo_step),
which replaces the fixed under-relaxation by a step derived from the MEASURED delivered
response of previous cycles.

Two parts:

1) Unit ladder test (no MAESTRO). A synthetic plant H(Te_bc) = (Te_bc/T1)**alpha_true
   stands in for "apply the BC, run the downstream beats, measure H". The frozen-shape
   solve is emulated exactly as it behaves in the beat: alpha = 1 (H proportional to
   Te_bc at frozen a/L). With the measured alpha_true = 0.4 that solve is 2.5x too
   stiff, which is precisely what the servo learns and the fixed relaxation cannot.
   Checks: convergence speed vs a fixed relaxation of 0.75, boundedness of the ladder
   for a stiff plant (alpha_true = 1.2), and exclusion of railed pairs from the fits.

2) Real-chain smoke test: a minimal MAESTRO run with three confinement beats in a row,
   all with servo_mode='response_fit'. Nothing runs between them, so the delivered
   response is ~the identity (alpha ~ 1 and the state is already at H = 1): this checks
   the plumbing (history accumulation in parameters_trans_beat AND in the JSON snapshot
   on disk, rung ladder, landing at the target), not the physics. Assert-free, printed.
   The run folder is KEPT for inspection:

       mitim_plot_maestro tests/scratch/dev_bc_servo_response_fit

Usage
-----
    ./run_with_env.sh python tests/dev_tests/test_bc_servo_response_fit.py
"""

import sys
import json
import numpy as np
from mitim_modules.maestro.scripts import run_maestro
from mitim_modules.maestro.utils.SHARPNESSbeat import record_bc_response, servo_step
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# ===========================================================================
# PART 1 - unit ladder test
# ===========================================================================

KIND   = "H98y2"
TARGET = 1.0
T1     = 1.0     # Te_bc (keV) at which the synthetic plant delivers H = 1
TE0    = 0.5     # first applied Te_bc (as if a previous incarnation had taken a full step)
BOUNDS = (0.05, 10.0)


class FakeMaestro:
    """Stand-in for maestro_instance: the servo only reads parameters_trans_beat/counter_current."""

    def __init__(self):
        self.parameters_trans_beat = {"Te_bc_applied": TE0, "Te_bc_applied_railed": False}
        self.counter_current = 0


def plant(Te_bc, alpha_true):
    """Synthetic delivered response: H = (Te_bc/T1)**alpha_true."""
    return (Te_bc / T1) ** alpha_true


def frozen_target(Te_last, H_last):
    """The frozen-shape solve, which believes alpha = 1 exactly: H proportional to Te_bc."""
    return Te_last * TARGET / H_last


def run_ladder(alpha_true, n_cycles, mode="response_fit", relaxation=0.75):
    """One servo cycle = record the delivered response, step, apply. Returns the trail."""

    m = FakeMaestro()
    trail = []
    for cycle in range(1, n_cycles + 1):
        m.counter_current = cycle
        Te_prev = m.parameters_trans_beat["Te_bc_applied"]
        H_delivered = plant(Te_prev, alpha_true)
        record_bc_response(m, KIND, H_delivered)

        Te_frozen = frozen_target(Te_prev, H_delivered)
        if mode == "response_fit":
            Te_new, diag = servo_step(m, KIND, TARGET, Te_frozen, BOUNDS)
            rung = diag["rung"]
        else:
            Te_new = Te_prev + relaxation * (Te_frozen - Te_prev)
            rung = f"relaxation={relaxation}"

        m.parameters_trans_beat["Te_bc_applied"] = Te_new
        trail.append({"cycle": cycle, "Te_prev": Te_prev, "H_delivered": H_delivered,
                      "rung": rung, "Te_new": Te_new, "H_next": plant(Te_new, alpha_true)})
    return m, trail


def print_trail(title, trail):
    print(f"\n  {title}")
    print("    cycle  Te_applied_prev   H_delivered   rung            Te_applied_new   H_next")
    for t in trail:
        print(f"    {t['cycle']:>5}  {t['Te_prev']:>15.4f}   {t['H_delivered']:>11.4f}   "
              f"{t['rung']:<14}  {t['Te_new']:>14.4f}   {t['H_next']:>6.4f}")


def first_cycle_converged(trail, tol=0.02):
    for t in trail:
        if abs(t["H_next"] - TARGET) < tol:
            return t["cycle"]
    return None


print("\n===========================================================================")
print(" PART 1 - unit ladder test (synthetic plant, no MAESTRO)")
print("===========================================================================")

# --- soft plant (the measured median): alpha_true = 0.4 -> frozen solve is 2.5x too stiff
_, trail_servo = run_ladder(0.4, 3)
_, trail_relax = run_ladder(0.4, 10, mode="relaxation", relaxation=0.75)

print_trail("alpha_true = 0.40, response_fit servo", trail_servo)
print_trail("alpha_true = 0.40, fixed relaxation = 0.75", trail_relax)

n_servo = first_cycle_converged(trail_servo)
n_relax = first_cycle_converged(trail_relax)
print(f"\n  Cycles to |H-1| < 0.02:  response_fit = {n_servo},  relaxation(0.75) = {n_relax}")

assert abs(trail_servo[-1]["H_next"] - TARGET) < 0.02, \
    f"response_fit did not land within 0.02 of the target in 3 cycles ({trail_servo[-1]['H_next']})"
assert n_relax > 6, f"fixed relaxation expected to need > 6 cycles, took {n_relax}"

# --- stiff plant: alpha_true = 1.2 -> the seeded rung would overshoot; the trust clamp holds it
_, trail_stiff = run_ladder(1.2, 4)
print_trail("alpha_true = 1.20, response_fit servo (trust factor 1.5)", trail_stiff)

ratios = [t["Te_new"] / t["Te_prev"] for t in trail_stiff]
print(f"    per-cycle Te_bc ratios: {['%.3f' % r for r in ratios]}")
assert all(1.0 / 1.5 - 1e-9 <= r <= 1.5 + 1e-9 for r in ratios), \
    f"trust clamp violated: {ratios}"
assert abs(trail_stiff[-1]["H_next"] - TARGET) < 0.02, \
    f"stiff plant did not converge in 4 cycles ({trail_stiff[-1]['H_next']})"
print(f"  Stiff plant: ladder bounded by the trust factor and converged in "
      f"{first_cycle_converged(trail_stiff)} cycles")

# --- railed pairs must not enter the fits
_railed_pair = {"kind": KIND, "beat": 2, "Te_bc": 2.0, "value": 2.5, "railed": True}
_good_pair   = {"kind": KIND, "beat": 1, "Te_bc": 0.5, "value": 0.7579, "railed": False}

m = FakeMaestro()
m.parameters_trans_beat["bc_response_history"] = [_good_pair, dict(_railed_pair)]
_, diag_excl = servo_step(m, KIND, TARGET, frozen_target(0.5, 0.7579), BOUNDS)

m = FakeMaestro()
m.parameters_trans_beat["bc_response_history"] = [_good_pair, dict(_railed_pair, railed=False)]
_, diag_incl = servo_step(m, KIND, TARGET, frozen_target(0.5, 0.7579), BOUNDS)

print(f"\n  Railed-pair exclusion: railed=True -> rung={diag_excl['rung']} (n_pairs={diag_excl['n_pairs']}), "
      f"same pair with railed=False -> rung={diag_incl['rung']} (n_pairs={diag_incl['n_pairs']})")
assert (diag_excl["n_pairs"], diag_excl["rung"]) == (1, "seed")
assert (diag_incl["n_pairs"], diag_incl["rung"]) == (2, "fit")

print("\n  PART 1 PASSED")

# ===========================================================================
# PART 2 - real-chain smoke test
# ===========================================================================

folder   = __mitimroot__ / "tests" / "scratch" / "dev_bc_servo_response_fit"
template = __mitimroot__ / "templates" / "namelist.maestro.yaml"

if folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

nml = IOtools.read_mitim_yaml(template)

# Initialization as in maestro_01_run.py: FreeGS equilibrium, fixed BC profiles
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV

nml["maestro"]["beats"] = ["confinement", "confinement", "confinement"]

nml["maestro"]["confinement"]["parameters_prepare"]["confinement_scaling"] = "H98y2"
nml["maestro"]["confinement"]["parameters_prepare"]["confinement"] = 1.0
nml["maestro"]["confinement"]["parameters_prepare"]["servo_mode"] = "response_fit"

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

m = run_maestro.run_maestro_local(
    namelist_file,
    folder=folder,
    terminal_outputs=True,
    force_cold_start=True,
    cpus=4,
)

# MAESTRO leaves stdout redirected to Outputs/maestro.log; bring the summary back
sys.stdout = sys.__stdout__

print("\n===========================================================================")
print(" PART 2 - real chain: 3 x confinement beat with servo_mode=response_fit")
print("===========================================================================")

print("\n  beat  frozen target   applied      rung    n_pairs   alpha      H_initial -> H_achieved")
for i in (1, 2, 3):
    r = np.load(folder / "Beats" / f"Beat_{i}" / "beat_results" / "confinement_results.npy",
                allow_pickle=True).item()
    alpha = r.get("servo_alpha")
    print(f"  {i:>4}  {r['Te_bc_target']:>13.4f}  {r['Te_bc']:>9.4f}  {r.get('servo_rung', '-'):>8}  "
          f"{str(r.get('servo_n_pairs', '-')):>7}   {('%.3f' % alpha) if alpha is not None else '   -  '}   "
          f"{r['H_initial']:.4f} -> {r['H_achieved']:.4f}")

history_live = m.parameters_trans_beat["bc_response_history"]
snapshot = json.loads((folder / "Outputs" / "trans_beat_parameters" / "beat_3.json").read_text())
history_disk = snapshot["bc_response_history"]

print(f"\n  Response history in live trans-beat parameters ({len(history_live)} pairs) "
      f"and in the beat_3 JSON snapshot ({len(history_disk)} pairs):")
for h in history_live:
    print(f"    beat {h['beat']}: {h['kind']} = {h['value']:.4f} delivered at "
          f"Te_bc = {h['Te_bc']:.4f} keV (railed={h['railed']})")

print(f"\n  Note: nothing runs between the beats, so the state comes back unchanged --")
print(f"  the actuator barely moves, the abscissa spread stays below the 2% fit gate and")
print(f"  the ladder legitimately stays on the seeded rung. The physics of the fit rung is")
print(f"  exercised by PART 1; PART 2 checks the plumbing and that H stays at the target.")
print(f"\n  Run folder kept: {folder}")
