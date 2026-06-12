"""
CAPABILITY: TGLF over multiple plasmas in one parallel submission
-----------------------------------------------------------------
This script teaches the "plasmas" parallelism axis: a single TGLF instance
runs several independent plasma states (e.g. variations of a baseline, or
different discharges) at the same radii, dispatched together in one parallel
submission. This is how the batched evaluators of PORTALS fan multiple
powerstate plasmas through one code call.

Key teaching points:
    1. run_over_plasmas(list_of_states, base_subfolder=...) runs every
       (plasma, rho) work unit concurrently through the same parallel pipeline
       that scans use, creating one subfolder per plasma
       (base_plasma0, base_plasma1, ...). It returns the
       {plasma index -> results label} mapping.
    2. Each plasma keeps its own prep-time state: read_plasma(p) temporarily
       restores that plasma's profiles, inputs and normalizations so the
       standard read() path applies per plasma.
    3. Results land in tglf.results under the per-plasma labels, so anything
       downstream (plotting, flux extraction) works as usual.
"""

import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools import TGLFtools, PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools, GRAPHICStools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one subfolder per plasma lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_multiplasma"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Build two distinct plasma states from the same input.gacode
# ---------------------------------------------------------------------------------------------------------------------

# Plasma 0: the baseline
state_a = PROFILEStools.gacode_state(input_gacode)

# Plasma 1: a deliberate perturbation (30% hotter electrons, 10% hotter ions),
# guaranteeing different TGLF fluxes at the same radii
state_b = PROFILEStools.gacode_state(input_gacode)
state_b.profiles["te(keV)"] = state_b.profiles["te(keV)"] * 1.3
state_b.profiles["ti(keV)"] = state_b.profiles["ti(keV)"] * 1.1

list_of_states = [state_a, state_b]

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run TGLF on both plasmas in one parallel submission
# ---------------------------------------------------------------------------------------------------------------------

# prep() with the first state defines the radii and the folder
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(state_a, folder, cold_start=cold_start)

# One subfolder per plasma (base_plasma0, base_plasma1); every (plasma, rho) work
# unit is dispatched concurrently. code_settings/extraOptions can be passed here
# exactly as in a single run
plasma_labels = tglf.run_over_plasmas(
    list_of_states,
    base_subfolder="base",
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
)
print("\nrun_over_plasmas returned plasma -> label mapping:", plasma_labels)

# read_plasma() restores each plasma's own prep-time state (profiles, inputs,
# normalizations) and reads its results under the corresponding label
for p in plasma_labels:
    tglf.read_plasma(p, cold_startWF=False)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Compare the two plasmas: fluxes vs radius, side by side
# ---------------------------------------------------------------------------------------------------------------------

rhos = np.array(tglf.rhos)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
colors = ["tab:blue", "tab:red"]
names = {0: "baseline", 1: "hotter (Te x1.3, Ti x1.1)"}
for p, label in plasma_labels.items():
    outputs = tglf.results[label]["output"]
    axs[0].plot(rhos, [o.Qe for o in outputs], "o-", color=colors[p], label=names[p])
    axs[1].plot(rhos, [o.Qi for o in outputs], "o-", color=colors[p], label=names[p])

axs[0].set_xlabel("$\\rho$"); axs[0].set_ylabel("$Q_e$ (GB)"); axs[0].set_title("Electron energy flux")
axs[1].set_xlabel("$\\rho$"); axs[1].set_ylabel("$Q_i$ (GB)"); axs[1].set_title("Ion energy flux")
axs[0].legend()
for ax in axs:
    GRAPHICStools.addDenseAxis(ax)
fig.suptitle("TGLF over two plasmas, shared radii, one parallel submission")
fig.tight_layout()
plt.show()
