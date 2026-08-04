"""
CAPABILITY: SOL / separatrix temperature estimates from a plasma state
----------------------------------------------------------------------
This script teaches the `calculate_sol()` method of the MITIM state, which
collapses all separatrix-temperature estimates in one place:

    - derived['Te_lcfs_estimate']  : legacy 2-point model (Brunner lambda_q,
      Bp = eps*Bt/q95 -- a very rough AVERAGED poloidal field). DEPRECATED,
      kept for backwards compatibility; it will be removed in the future.
    - derived['Bpol_omp']          : outboard-midplane poloidal field from the
      poloidal-flux gradient (exact -- typically ~2x the rough average).
    - derived['Te_lcfs_2pt']       : the same 2-point model with the real OMP
      Bp (the recommended analytic estimate; ~x0.75-0.85 of the legacy one
      through the (q_par)^(2/7) stiffness).
    - derived['Te_lcfs_lengyel']   : the extended-Lengyel model (OPTIONAL
      dependency: pip install -e .[lengyel] + a radas atomic-data directory
      via the RADAS_DIR env variable). If the dependency or data is missing,
      the method prints a warning and stores NaN -- it never raises.

Here we scan the injected RF power of a plasma state (which moves Psol) and
plot the three estimates side by side. Key teaching points:
    1. The analytic estimates are computed automatically at every
       derive_quantities() call -- no user action needed.
    2. The Lengyel estimate must be requested explicitly
       (calculate_sol(lengyel=True, ...)) because it runs an external model
       with atomic data; its connection length defaults to the SAME
       L = pi*R*q95 the 2-point model uses (0.44*L divertor leg), so the
       comparison is apples to apples; any input of the Lengyel controls yaml
       can be overridden as a keyword argument.
    3. All three estimates scale close to Psol^(2/7) -- the Spitzer-conduction
       stiffness -- so even large model differences compress into modest
       temperature offsets.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GRAPHICStools, IOtools
from mitim_tools import __mitimroot__

# cold_start=True starts the Lengyel runs from scratch; False reuses results already
# in the folder (the analytic estimates are recomputed either way, they are instant)
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)
folder = __mitimroot__ / "tests" / "scratch" / "capability_sol_estimates"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Load the state; the analytic estimates are already there
# ---------------------------------------------------------------------------------------------------------------------

p = PROFILEStools.gacode_state(input_gacode)

print(f"As loaded: Psol = {p.derived['Psol']:.2f} MW")
print(f"   legacy Te_lcfs_estimate = {p.derived['Te_lcfs_estimate']*1e3:.1f} eV (DEPRECATED)")
print(f"   Bpol_omp                = {p.derived['Bpol_omp']:.2f} T")
print(f"   Te_lcfs_2pt             = {p.derived['Te_lcfs_2pt']*1e3:.1f} eV")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Scan the RF power (moves Psol) and collect all estimates per point
# ---------------------------------------------------------------------------------------------------------------------

# Without RADAS_DIR the Lengyel column will be NaN (with a printed warning per point)
# -- the script still runs to completion, demonstrating the graceful degradation.
if os.getenv("RADAS_DIR") is None:
    print(">> RADAS_DIR not set: the Lengyel estimate will show as NaN (analytic ones unaffected)")

Prf_scan = [1.0, 2.0, 5.0, 10.0, 20.0]  # MW

results = {"Psol": [], "legacy": [], "twopt": [], "lengyel": []}
for i, Prf in enumerate(Prf_scan):
    p.changeRFpower(PrfMW=Prf)                       # rederives -> analytic estimates updated
    p.calculate_sol(lengyel=True, lengyel_folder=folder / f"point_{i}", cold_start=cold_start)

    results["Psol"].append(p.derived["Psol"])
    results["legacy"].append(p.derived["Te_lcfs_estimate"] * 1e3)
    results["twopt"].append(p.derived["Te_lcfs_2pt"] * 1e3)
    results["lengyel"].append(p.derived["Te_lcfs_lengyel"] * 1e3)

    print(f"Prf = {Prf:5.1f} MW -> Psol = {results['Psol'][-1]:5.2f} MW: "
          f"legacy {results['legacy'][-1]:6.1f} eV, 2pt {results['twopt'][-1]:6.1f} eV, "
          f"Lengyel {results['lengyel'][-1]:6.1f} eV")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the comparison
# ---------------------------------------------------------------------------------------------------------------------

Psol = np.array(results["Psol"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Psol, results["legacy"], "o-", label="Te_lcfs_estimate (legacy 2-pt, rough Bp) [DEPRECATED]")
ax.plot(Psol, results["twopt"], "s-", label="Te_lcfs_2pt (2-pt, exact OMP Bp)")
if np.isfinite(results["lengyel"]).any():
    ax.plot(Psol, results["lengyel"], "^-", label="Te_lcfs_lengyel (extended-Lengyel)")

# Spitzer-conduction stiffness reference: Tsep ~ Psol^(2/7)
ax.plot(Psol, results["twopt"][0] * (Psol / Psol[0]) ** (2.0 / 7.0), "k--", lw=0.8,
        label=r"$\propto P_{sol}^{2/7}$ reference")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$P_{sol}$ [MW]")
ax.set_ylabel("$T_{e,sep}$ [eV]")
ax.set_title("Separatrix temperature estimates (calculate_sol)")
ax.legend(loc="best", fontsize=8)
GRAPHICStools.addDenseAxis(ax)

plt.show()
