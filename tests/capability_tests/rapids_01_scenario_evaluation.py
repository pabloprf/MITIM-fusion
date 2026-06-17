"""
CAPABILITY: RAPIDS — fast whole-discharge scenario evaluation (and scan)
------------------------------------------------------------------------
RAPIDS builds a self-consistent input.gacode scenario in *seconds* from a
handful of engineering scalars plus PRESCRIBED core gradients — the fast,
approximate counterpart to MAESTRO for scoping a design space. For each point
it: sets the geometry/heating/species from the scalars, predicts the pedestal
with a trained EPED-NN surrogate, imposes the prescribed core gradients
(aLTe, aLn, aLTi), and recomputes the analytic targets (radiation, fusion,
exchange) on the resulting profiles — no PORTALS flux-matching and no TRANSP.

Key teaching points:
    1. The pedestal comes from a device-specific EPED-NN surrogate. Here we use
       the ARC-trained model from the (private) MFE-IM repo, loaded via the
       $MFEIM_PATH environment variable — so the engineering point evaluated
       below is ARC-class. The bundled SPARC PRD input.gacode is only a DT
       *seed template* (species mix + radial grid); RAPIDS overrides its
       geometry, pedestal and core.
    2. The core gradients are INPUTS you prescribe, NOT predicted. This is the
       central approximation vs MAESTRO/PORTALS: a RAPIDS-vs-MAESTRO gap mixes
       pedestal-NN differences with this fixed core-transport assumption.
    3. `rapids_evaluator(nn, core, p_base, <engineering scalars>)` returns
       (ptop_kPa, wtop_psipol, p_new, eped_eval, neped_transition_estimate);
       `p_new` is a full gacode_state with derived Pfus, BetaN, Q, ...
    4. Because each evaluation is cheap, RAPIDS ships its own scan-and-plot:
       `scan_parameter` sweeps one engineering knob (here the pedestal density),
       warm-starting each BetaN solve from the previous point, and builds the
       summary figure (Ptop and Pfus vs the knob and vs the Greenwald fraction,
       plus Pfus vs q*, volume, H98 and BetaN). It returns the results dict.

REQUIREMENT: the ARC EPED-NN under $MFEIM_PATH/private_code_mitim/NN_DATA/ (the
analog of eped_01 needing the EPED machine, or maestro_01 needing TRANSP).
"""

import numpy as np
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.popcon_tools import RAPIDStools
from mitim_tools import __mitimroot__

# ---------------------------------------------------------------------------------------------------------------------
# 1. Seed template + EPED-NN surrogate
# ---------------------------------------------------------------------------------------------------------------------

# Seed: provides the DT species mix and the radial grid; geometry/pedestal/core
# are all overridden by RAPIDS below.
p_base = PROFILEStools.gacode_state(__mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD")


def load_eped_nn():
    """ARC-trained EPED-NN from the MFE-IM repo (path from $MFEIM_PATH, never hardcoded)."""
    nn = NNtools.eped_nn(type="tf")
    nn_folder = IOtools.expandPath("$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC_expanded")
    nn.load(
        f"{nn_folder}/eped_nn_expanded.keras",
        norm=f"{nn_folder}/eped_nn_expanded.txt",
        transform_inputs_fun=NNtools.engineering_arguments_eped,
    )
    return nn


nn = load_eped_nn()

# ---------------------------------------------------------------------------------------------------------------------
# 2. Engineering point (ARC-class, matching the EPED-NN domain) + prescribed core
# ---------------------------------------------------------------------------------------------------------------------

n_G = PLASMAtools.Greenwald_density(11.9, 1.16)   # 1e20 m^-3, n_G = Ip / (pi a^2)
print(f"\nGreenwald density n_G = {n_G:.3f}e20")

# The full engineering point: every key is a rapids_evaluator argument. scan_parameter
# sweeps one of these (neped below) and holds the rest fixed.
nominal_parameters = {
    "R": 4.4, "a": 1.16, "Bt": 9.4, "Ip": 11.9,        # geometry (m) / field (T) / current (MA)
    "kappa995": 1.9, "delta995": 0.5,                  # 99.5% flux-surface shaping
    "kappa_sep": 1.9 * 1.07, "delta_sep": 0.5 * 1.22,  # separatrix shaping
    "neped": 0.85 * n_G, "Zeff": 1.47,                 # pedestal density (1e20) + Zeff
    "tesep_eV": 200.0, "nesep_ratio": 0.3,             # separatrix Te + nesep/neped
}

# Prescribed core gradients (RAPIDS does not predict these)
core = {"aLTe": 1.9, "aLn": 0.46, "aLTi": 1.61}

# ---------------------------------------------------------------------------------------------------------------------
# 3. A single scenario
# ---------------------------------------------------------------------------------------------------------------------

ptop, wtop, p_new, eped_eval, _ = RAPIDStools.rapids_evaluator(nn, core, p_base, **nominal_parameters)
print(f"\nNominal scenario (neped={nominal_parameters['neped']:.2f}e20):")
print(f"  Pfus = {p_new.derived['Pfus']:.1f} MW   BetaN = {p_new.derived['BetaN']:.2f}   "
      f"Q = {p_new.derived['Q']:.1f}")
print(f"  pedestal: ptop = {ptop:.1f} kPa,  width = {wtop:.4f} psi_pol")

# Save the scenario for inspection (e.g. mitim_plot_gacode on it)
out = __mitimroot__ / "tests" / "scratch" / "capability_rapids"
out.mkdir(parents=True, exist_ok=True)
p_new.write_state(out / "input.gacode")

# ---------------------------------------------------------------------------------------------------------------------
# 4. A rapid pedestal-density scan, using RAPIDS' own scan-and-plot
# ---------------------------------------------------------------------------------------------------------------------

results = RAPIDStools.scan_parameter(
    nn, p_base,
    xparam="neped",                            # which engineering knob to sweep
    x=np.array([0.65, 0.78, 0.90]) * n_G,      # pedestal densities (1e20 m^-3)
    nominal_parameters=nominal_parameters,
    core=core,
    xparamlab=r"$n_{e,ped}$ ($10^{20}\,m^{-3}$)",
    goal_pfusion=500,                          # green target band on the Pfus panels (MW)
    type_plot="full",                          # 8-panel summary; "simple" gives 3 panels
)

print(f"\nScan Pfus (MW): {[round(v, 1) for v in results['Pfus']]}")
print(f"Scan BetaN:     {[round(v, 2) for v in results['betaN']]}")

import matplotlib.pyplot as plt
plt.ioff()   # scan_parameter enables interactive mode (plt.ion); turn it off so that
plt.show()   # show() blocks and the figure stays open under a plain `python script.py`
