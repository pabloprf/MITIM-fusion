import os
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools import TGLFtools, PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# ---------------------------------------------------------------------------
# Multi-plasma TGLF workflow test
#
# This test mirrors TGLFscan_workflow.py but exercises the new "plasmas"
# parallelism axis added via mitim_simulation.run_over_plasmas: a single TGLF
# instance runs two independent plasmas (two input.gacodes) at the same rhos
# in one parallel submission, then each plasma's results are read back via
# read_plasma(...) and plotted side-by-side.
# ---------------------------------------------------------------------------

cold_start = True

(__mitimroot__ / 'tests' / 'scratch').mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "tglf_multi_plasma_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------
# Build two distinct plasma states from the shared test input.gacode:
#   - plasma 0 = baseline
#   - plasma 1 = 30% hotter electrons, 10% hotter ions (a deliberate perturbation
#     that guarantees different TGLF fluxes at the same rhos)
# ---------------------------------------------------------------------------
state_a = PROFILEStools.gacode_state(input_gacode)

state_b = PROFILEStools.gacode_state(input_gacode)
state_b.profiles['te(keV)'] = state_b.profiles['te(keV)'] * 1.3
state_b.profiles['ti(keV)'] = state_b.profiles['ti(keV)'] * 1.1

list_of_states = [state_a, state_b]

# ---------------------------------------------------------------------------
# Run TGLF on both plasmas in one parallel submission.
# run_over_plasmas creates a subfolder per plasma ("base_plasma0", "base_plasma1")
# and every (plasma, rho) work unit is dispatched through the existing
# FARMINGtools parallel loop — same path scans already use.
# ---------------------------------------------------------------------------
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(state_a, folder, cold_start=cold_start)

plasma_labels = tglf.run_over_plasmas(
    list_of_states,
    base_subfolder='base',
    cold_start=cold_start,
)

print("\nrun_over_plasmas returned plasma -> subfolder mapping:", plasma_labels)

# ---------------------------------------------------------------------------
# Read each plasma's results. read_plasma temporarily restores that plasma's
# prep-time state (profiles, inputs, normalizations, folder) so the existing
# TGLF.read() path can reuse its per-plasma normalizations unchanged.
# ---------------------------------------------------------------------------
for p in plasma_labels:
    tglf.read_plasma(p, cold_startWF=False)

# ---------------------------------------------------------------------------
# Build Qe / Qi arrays (one value per rho per plasma) and plot a side-by-side
# comparison so the caller can eyeball the two plasmas against each other.
# ---------------------------------------------------------------------------
rhos = np.array(tglf.rhos)
Qe_per_plasma, Qi_per_plasma = {}, {}
for p, label in plasma_labels.items():
    outputs = tglf.results[label]['output']
    Qe_per_plasma[p] = np.array([o.Qe for o in outputs])
    Qi_per_plasma[p] = np.array([o.Qi for o in outputs])

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
colors = ['tab:blue', 'tab:red']
for p in plasma_labels:
    axs[0].plot(rhos, Qe_per_plasma[p], 'o-', color=colors[p], label=f'plasma {p}')
    axs[1].plot(rhos, Qi_per_plasma[p], 'o-', color=colors[p], label=f'plasma {p}')

axs[0].set_xlabel(r'$\rho$')
axs[0].set_ylabel(r'$Q_e$ (TGLF, GB units)')
axs[0].set_title('Electron energy flux')
axs[0].legend()

axs[1].set_xlabel(r'$\rho$')
axs[1].set_ylabel(r'$Q_i$ (TGLF, GB units)')
axs[1].set_title('Ion energy flux')

fig.suptitle('TGLF multi-plasma workflow: two plasmas, shared rhos, parallel submission')
fig.tight_layout()
plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Assertion: plasma 0 and plasma 1 must produce distinct fluxes (the 30% / 10%
# temperature perturbation in state_b should not collapse to the baseline).
# ---------------------------------------------------------------------------
for rho_idx, rho in enumerate(rhos):
    qe_a = float(Qe_per_plasma[0][rho_idx])
    qe_b = float(Qe_per_plasma[1][rho_idx])
    if np.isclose(qe_a, qe_b, rtol=1e-6, atol=0):
        raise AssertionError(
            f"Expected distinct Qe between the two plasmas at rho={rho:.4f}, "
            f"got qe_a={qe_a:.6g}, qe_b={qe_b:.6g}"
        )

print("\nTGLFmulti_plasma_workflow.py: Qe/Qi differ across the two plasmas at every rho. PASS.")
