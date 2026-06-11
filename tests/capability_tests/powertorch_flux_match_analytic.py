"""
CAPABILITY: POWERTORCH — flux matching with an analytic transport model
-----------------------------------------------------------------------
This script teaches the powerstate object, the PyTorch-based plasma state at
the heart of PORTALS: profiles parametrized by their gradients on a coarse
radial grid, with transport fluxes and targets evaluated as (batched,
differentiable) tensors. Everything here runs in memory, locally and in
seconds — no files are written and no external code is called.

Key teaching points:
    1. powerstate is built from a plasma state (input.gacode) plus
       `evolution_options`: which channels to evolve ("te", "ti", ...) and at
       which coarse radii.
    2. The transport model is pluggable through `transport_options`: here a
       simple analytic diffusion model (conducted flux from the given chi_e,
       chi_i diffusivities at each radius) takes the place that TGLF+NEO
       occupy in a real PORTALS run. This is ideal to understand and test the
       machinery without the cost of a turbulence code.
    3. calculate() evaluates transport and target fluxes for the current
       profiles; flux_match() solves directly for the profiles that make
       transport = targets (algorithm='root' uses a scipy root finder;
       'simple_relax' is the relaxation used to seed PORTALS surrogates).
    4. plot(compare_to_state=...) compares two states — here the original
       profiles vs the flux-matched ones.
"""

import copy
import torch
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.physics_models import transport_analytic
from mitim_tools import __mitimroot__

# ---------------------------------------------------------------------------------------------------------------------
# 1. Build the powerstate from a plasma state
# ---------------------------------------------------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(__mitimroot__ / "tests" / "data" / "input.gacode")

# Coarse radial grid where the profiles are parametrized and the fluxes matched
# (powertorch works with torch tensors in double precision)
rho = torch.from_numpy(np.linspace(0.1, 0.9, 9)).to(dtype=torch.double)

s = STATEtools.powerstate(
    plasma_state,
    # Channels to evolve and the coarse radii where they are parametrized
    evolution_options={
        "ProfilePredicted": ["te", "ti"],
        "rhoPredicted": rho,
    },
    # Pluggable transport model: an analytic diffusion model with prescribed
    # diffusivities (m^2/s) at each radius, instead of TGLF+NEO
    transport_options={
        "evaluator": transport_analytic.diffusion_model,
        "options": {
            "chi_e": torch.ones(rho.shape[0]).to(rho) * 0.8,
            "chi_i": torch.ones(rho.shape[0]).to(rho) * 1.2,
        },
    },
)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Evaluate the original state, then solve the flux-matching problem
# ---------------------------------------------------------------------------------------------------------------------

# Keep a copy of the original profiles, with its transport and target fluxes evaluated
s_orig = copy.deepcopy(s)
s_orig.calculate()

# Solve for the profiles that make transport flux = target flux at every radius
# (with this analytic model, the whole solve takes seconds)
s.flux_match(algorithm="root")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot the flux-matched state against the original one
# ---------------------------------------------------------------------------------------------------------------------

# Profiles, gradients, transport vs target fluxes of both states in one notebook
fn = s.plot(compare_to_state=s_orig)
