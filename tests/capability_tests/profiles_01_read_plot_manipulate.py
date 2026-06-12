"""
CAPABILITY: The MITIM plasma state — read, derive, manipulate and plot
----------------------------------------------------------------------
This script teaches the object at the center of everything in MITIM: the
plasma state (gacode_state), loaded from an input.gacode file. Every workflow
(TGLF, NEO, CGYRO, PORTALS, MAESTRO, ...) starts from one of these, so this is
the natural first capability to learn. Everything here runs locally and in
seconds.

Key teaching points:
    1. gacode_state(file) reads the profiles, equilibrium, species and sources,
       and derives a large set of physics quantities (state.derived: stored
       energies, BetaN, confinement times, volume averages, gradients, ...);
       printInfo() summarizes them.
    2. The raw profiles live in state.profiles (a dictionary mirroring the
       input.gacode columns) and can be modified directly; after any
       modification, derive_quantities() refreshes the derived set so the
       consequences (here, of hotter electrons) can be quantified.
    3. Common manipulations have dedicated methods: correct() applies standard
       clean-ups (quasineutrality, recompute Ptot, make fast species thermal),
       lumpIons() bundles the ion species into one effective ion,
       changeResolution() regrids, write_state() saves a new input.gacode.
    4. Several states can be overlaid in one notebook (state_plotting.plotAll)
       — here the original vs the modified one, so every consequence of the
       manipulation is visible panel by panel.
"""

import copy
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.plasmastate_tools.utils import state_plotting
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools import __mitimroot__

input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

# Working folder: only used to save the modified state at the end
folder = __mitimroot__ / "tests" / "scratch" / "capability_profiles"
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Read the plasma state and inspect it
# ---------------------------------------------------------------------------------------------------------------------

state = PROFILEStools.gacode_state(input_gacode)

# Scalar summary of the plasma (geometry, content, performance)
state.printInfo()

# The derived dictionary carries the physics quantities computed from the profiles
print(f"\nSome derived quantities of the original state:")
print(f"   Wthr  = {state.derived['Wthr']:.3f} MJ (thermal stored energy)")
print(f"   BetaN = {state.derived['BetaN_engineering']:.3f}")
print(f"   tauE  = {state.derived['tauE']:.3f} s")
print(f"   Psol  = {state.derived['Psol']:.3f} MW")

# The species list as read from the file
print(f"   Species = {[str(n) for n in state.profiles['name']]}")

# ---------------------------------------------------------------------------------------------------------------------
# 2. Manipulate a copy: hotter electrons, standard corrections, lumped ions
# ---------------------------------------------------------------------------------------------------------------------

state_mod = copy.deepcopy(state)

# Direct profile modification: 20% hotter electrons everywhere. state_mod.profiles is
# just a dictionary of the input.gacode columns
state_mod.profiles["te(keV)"] = state_mod.profiles["te(keV)"] * 1.2

# Standard clean-ups: enforce quasineutrality (adjusting thermal ion densities),
# recompute the total pressure consistently with the new Te, make fast species thermal
state_mod.correct(options={"recalculate_ptot": True, "quasineutrality": True, "remove_fast": True})

# Bundle all ions (main + impurities) into a single effective species
state_mod.lumpIons()

# Refresh the derived quantities so they reflect all the modifications above
state_mod.derive_quantities()

print(f"\nSame quantities after the modifications (Te x1.2, corrections, lumped ions):")
print(f"   Wthr  = {state_mod.derived['Wthr']:.3f} MJ")
print(f"   BetaN = {state_mod.derived['BetaN_engineering']:.3f}")
print(f"   tauE  = {state_mod.derived['tauE']:.3f} s")
print(f"   Psol  = {state_mod.derived['Psol']:.3f} MW")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot both states overlaid, panel by panel
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook; show() opens the GUI.
# plotAll overlays the states (one color each), so the effect of every
# manipulation — Te, pressures, powers, gradients, ion content — is visible
fn = FigureNotebook("MITIM plasma state", geometry="1800x900")
figs = state_plotting.add_figures(fn)
state_plotting.plotAll([state, state_mod], figs=figs, extralabs=["original", "modified"])

fn.show()

# ---------------------------------------------------------------------------------------------------------------------
# 4. Save the modified state as a new input.gacode
# ---------------------------------------------------------------------------------------------------------------------

# The written file is a standard input.gacode, usable by any code/workflow in MITIM
state_mod.write_state(file=folder / "input.gacode.modified")
