# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **New MAESTRO `confinement` beat**, which sets the temperature boundary condition at a chosen radial location such that the plasma matches a target confinement level (H-factor, `H98y2` or `H89p`). Since the H-factor cannot be inverted analytically for T_bc, it is found by Nelder-Mead minimization (same spirit as the eped_initializer BetaN matching), applying the BC at each trial Te_bc with the sharpness-beat machinery (core preserves a/LT and a/Ln; edge anchored at the BC and separatrix values, with `edge_shape` selecting a straight line in psi_n or the initializer's pedestal tanh in r/a). An optional `alpha_power_feedback` recomputes qfuse/qfusi analytically at every trial Te_bc so the H-factor accounts for the alpha-heating response (relevant for burning plasmas; no-op for non-DT), and `density_treatment` chooses whether the density profiles are rescaled to ne_bc or left completely untouched by the BC change (the latter option is also available in the sharpness beat). Includes all standard beat methods: restart robustness, plots (optimization diagnostics + H-factor-inputs panel), and trans-beat parameter hand-off so subsequent PORTALS beats reuse surrogate data. See the `confinement` section of `templates/namelist.maestro.yaml`.

### Bug Fixes

*   🐛 **NEW BUG FIX**, description

### Changes for developers (internal execution)

*   🔎 **NEW CHANGE**, description

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
