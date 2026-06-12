"""
CAPABILITY: The MITIM plasma state — read, derive, manipulate and compare
-------------------------------------------------------------------------
This script teaches the object at the center of everything in MITIM: the
plasma state (gacode_state), loaded from an input.gacode file. Every workflow
(TGLF, NEO, CGYRO, PORTALS, MAESTRO, ...) starts from one of these, so this is
the natural first capability to learn. Everything here runs locally and in
seconds.

Three modified variants of the same plasma are built and compared against the
original, both with the standard state plots (overlaid) and with custom
figures of the quantities of interest:

    - "hotter":   20% hotter electrons (direct profile edit + corrections)
    - "impure":   a tungsten trace added and Zeff raised to 2.5
    - "denser":   30% denser thermal species and more RF power

Key teaching points:
    1. gacode_state(file) reads profiles/equilibrium/species/sources and
       derives the physics quantities (state.derived: stored energies, BetaN,
       tauE, Greenwald fraction, volume averages, ...); printInfo() summarizes.
       The raw columns live in state.profiles (a plain dictionary) and can be
       edited directly — call derive_quantities() afterwards to refresh.
    2. Species machinery: addSpecie(Z, mass, fi_vol) appends an impurity at a
       given fraction of ne; changeZeff(Zeff) rebalances the impurity and the
       quasineutral main ions to reach a target Zeff. Related methods worth
       knowing: remove(), moveSpecie(), lumpIons()/lumpImpurities()/lumpDT().
    3. Engineering-style knobs: scaleAllThermalDensities() (e.g. Greenwald
       fraction changes), changeRFpower(), imposeBCtemps()/imposeBCdens()
       (boundary conditions), parabolizePlasma(), introduceRotationProfile(),
       changeResolution(), smooth_profiles().
    4. correct(options) applies the standard clean-ups (quasineutrality,
       recompute Ptot, make fast species thermal) — same options used by the
       workflows when prepping runs.
    5. Comparisons: state_plotting.plotAll overlays full standard plots of
       several states, and custom figures can be added to the same notebook
       (here: kinetic profiles, impurity content/Zeff, and a relative-change
       summary of the scalars of interest).
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools import __mitimroot__

input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Working folder: only used to save the modified states at the end
folder = __mitimroot__ / "tests" / "scratch" / "capability_profiles"
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Read the plasma state and inspect it
# ---------------------------------------------------------------------------------------------------------------------

state = PROFILEStools.gacode_state(input_gacode)

# Scalar summary of the plasma (geometry, content, performance)
state.printInfo()

# The species list as read from the file (D + C + N + fast D in this example)
print(f"\nSpecies of the original state: {[str(n) for n in state.profiles['name']]}")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Variant "hotter": direct profile edit + standard corrections
# ---------------------------------------------------------------------------------------------------------------------

state_hot = copy.deepcopy(state)

# Direct modification of a raw column: 20% hotter electrons everywhere
state_hot.profiles["te(keV)"] = state_hot.profiles["te(keV)"] * 1.2

# Standard clean-ups: quasineutrality (adjusting thermal ion densities), recompute the
# total pressure consistently with the new Te, make fast species thermal
state_hot.correct(options={"recalculate_ptot": True, "quasineutrality": True, "remove_fast": True})

# Refresh the derived quantities so they reflect the modifications
state_hot.derive_quantities()

# ---------------------------------------------------------------------------------------------------------------------
# 3. Variant "impure": add a tungsten trace and raise Zeff
# ---------------------------------------------------------------------------------------------------------------------

state_imp = copy.deepcopy(state)
state_imp.correct(options={"recalculate_ptot": True, "quasineutrality": True, "remove_fast": True})

# Append a new thermal species: tungsten-like trace at a fraction 1e-5 of ne
state_imp.addSpecie(Z=74.0, mass=184.0, fi_vol=1e-5, forcename="W")

# Raise Zeff to 2.5 by rebalancing the impurity content and the quasineutral main ions
# (by default it acts on the ion in position 2 of the ion list; see also keep_fmain)
state_imp.changeZeff(2.5)

state_imp.derive_quantities()
print(f"\nSpecies after adding the tungsten trace: {[str(n) for n in state_imp.profiles['name']]}")

# ---------------------------------------------------------------------------------------------------------------------
# 4. Variant "denser": engineering-style knobs
# ---------------------------------------------------------------------------------------------------------------------

state_den = copy.deepcopy(state)
state_den.correct(options={"recalculate_ptot": True, "quasineutrality": True, "remove_fast": True})

# 30% denser plasma (e.g. exploring a higher Greenwald fraction): scale the electron
# density AND all thermal ion densities together, so quasineutrality is preserved
# (scaleAllThermalDensities acts on the ions; ne is a raw column edit)
state_den.profiles["ne(10^19/m^3)"] = state_den.profiles["ne(10^19/m^3)"] * 1.3
state_den.scaleAllThermalDensities(scaleFactor=1.3)

# More auxiliary power: rescale the RF deposition profiles to this total
state_den.changeRFpower(PrfMW=2.0)

state_den.derive_quantities()

# ---------------------------------------------------------------------------------------------------------------------
# 5. Quantify what happened: derived quantities of interest, per variant
# ---------------------------------------------------------------------------------------------------------------------

states = [state, state_hot, state_imp, state_den]
names = ["original", "hotter", "impure", "denser"]
colors = ["b", "r", "g", "m"]

quantities = {  # derived key -> pretty label
    "Wthr": "$W_{th}$ (MJ)",
    "BetaN_engineering": "$\\beta_N$",
    "tauE": "$\\tau_E$ (s)",
    "fG": "$f_G$",
    "Zeff_vol": "$\\langle Z_{eff}\\rangle$",
    "Psol": "$P_{sol}$ (MW)",
}

print("\nDerived quantities of interest:")
print(f"{'':18s}" + "".join(f"{n:>12s}" for n in names))
for key in quantities:
    vals = [s.derived[key] for s in states]
    print(f"{key:18s}" + "".join(f"{v:12.3f}" for v in vals))

# ---------------------------------------------------------------------------------------------------------------------
# 6. Plot: standard overlaid state plots + custom comparison figures
# ---------------------------------------------------------------------------------------------------------------------

fn = FigureNotebook("MITIM plasma state", geometry="1800x900")

# --- Standard full state plots, all variants overlaid ---------------------------------
figs = state_plotting.add_figures(fn)
state_plotting.plotAll(states, figs=figs, extralabs=names)

# --- Custom tab: kinetic profiles pre/post --------------------------------------------
fig = fn.add_figure(label="Compare: kinetic", tab_color=1)
axs = fig.subplots(1, 3)
for s, n, c in zip(states, names, colors):
    rho = s.profiles["rho(-)"]
    axs[0].plot(rho, s.profiles["te(keV)"], c=c, lw=1.5, label=n)
    axs[1].plot(rho, s.profiles["ti(keV)"][:, 0], c=c, lw=1.5)
    axs[2].plot(rho, s.profiles["ne(10^19/m^3)"] * 0.1, c=c, lw=1.5)
for ax, ylab, title in zip(axs, ["$T_e$ (keV)", "$T_i$ (keV)", "$n_e$ ($10^{20}m^{-3}$)"],
                           ["Electron temperature", "Ion temperature", "Electron density"]):
    ax.set_xlabel("$\\rho_N$"); ax.set_ylabel(ylab); ax.set_title(title)
    GRAPHICStools.addDenseAxis(ax)
axs[0].legend()

# --- Custom tab: impurity content and Zeff pre/post ------------------------------------
fig = fn.add_figure(label="Compare: impurities", tab_color=2)
axs = fig.subplots(1, 2)
ax = axs[0]
for s, n, c in zip([state, state_imp], ["original", "impure"], ["b", "g"]):
    ax.plot(s.profiles["rho(-)"], s.profiles["z_eff(-)"], c=c, lw=1.5, label=n)
ax.set_xlabel("$\\rho_N$"); ax.set_ylabel("$Z_{eff}$"); ax.set_title("Effective charge")
ax.legend(); GRAPHICStools.addDenseAxis(ax)

# Per-species densities of the "impure" variant (log scale: the W trace sits ~5 orders
# of magnitude below the main ions, exactly the fi_vol=1e-5 that was requested)
ax = axs[1]
for i, ion_name in enumerate(state_imp.profiles["name"]):
    ax.plot(state_imp.profiles["rho(-)"], state_imp.profiles["ni(10^19/m^3)"][:, i] * 0.1, lw=1.5, label=str(ion_name))
ax.set_yscale("log")
ax.set_xlabel("$\\rho_N$"); ax.set_ylabel("$n_i$ ($10^{20}m^{-3}$)"); ax.set_title('Ion densities ("impure")')
ax.legend(); GRAPHICStools.addDenseAxis(ax)

# --- Custom tab: relative change of the scalars of interest ----------------------------
fig = fn.add_figure(label="Compare: scalars", tab_color=3)
ax = fig.subplots()
x = np.arange(len(quantities))
width = 0.25
for j, (s, n, c) in enumerate(zip(states[1:], names[1:], colors[1:])):
    rel = [100.0 * (s.derived[k] - state.derived[k]) / state.derived[k] for k in quantities]
    ax.bar(x + (j - 1) * width, rel, width, color=c, label=n)
ax.set_xticks(x); ax.set_xticklabels(list(quantities.values()))
ax.axhline(0, c="k", lw=1)
ax.set_ylabel("Change from original (%)")
ax.set_title("What each manipulation did to the plasma")
ax.legend(); GRAPHICStools.addDenseAxis(ax)

fn.show()

# ---------------------------------------------------------------------------------------------------------------------
# 7. Save any variant as a new input.gacode (standard file, usable by any MITIM workflow)
# ---------------------------------------------------------------------------------------------------------------------

state_imp.write_state(file=folder / "input.gacode.modified")
