# v5.1.0 — MAESTRO confinement beat, GKNN surrogate, full CGYRO transport channels, and boundary-gradient fixes

This release introduces the MAESTRO `confinement` beat (matching a target H-factor through the temperature boundary condition), the `gknn` neural-network turbulence surrogate for PORTALS, complete CGYRO transport channels (momentum, exchange, ion particle fluxes) with PORTALS-CGYRO support for them, and a set of important fixes to the gradients that PORTALS receives at its boundary-condition location, the TGLF flux-uncertainty model, and the trace-impurity D/V analysis.

### New Features

*   💥 **New MAESTRO `confinement` beat**: sets the temperature boundary condition at a chosen radial location such that the plasma matches a target H-factor (`H98y2` or `H89p`), found by minimization in the spirit of the eped_initializer BetaN matching. Options include the edge profile shape (`edge_shape`: linear or pedestal tanh), alpha-heating feedback during the scan (`alpha_power_feedback`), and what to do with density (`density_treatment`, also available in the sharpness beat). Full beat support: restarts, plots, and surrogate-data reuse by subsequent PORTALS beats. See the `confinement` section of `templates/namelist.maestro.yaml`.

*   💥 **New `gknn` turbulence model for PORTALS** — a TGLF-GKNN-JAX neural-network surrogate (TGLF `SAT3` + GKNN corrections), selectable via the `gknn` block under `transport.options`, with an `apply_gknn` toggle and a dedicated two-ion lumping postprocessing. The `tglf_gknn_jax` package is an optional dependency — runs degrade gracefully when absent. See `templates/namelist.portals.yaml`.

*   💥 **MAESTRO backs up a previous run's finalization when restarted on the same folder**: the final `input.gacode`, summary report and saved figures move to a timestamped `Outputs/finalization_backup_*/` folder and are regenerated when the restarted run finalizes. Plot-only consumers (`mitim_plot_maestro`) never touch the artifacts; `maestro.unfinalize()` can also be called manually.

*   💥 **CGYRO nonlinear outputs now carry all transport channels, and PORTALS-CGYRO can consume them.** The reader extracts the momentum flux and the turbulent energy exchange (per species, ES/EM split, saturated means/stds; exchange gracefully absent on older outputs), and the "Fluxes" tabs gain ion-particle, momentum and exchange panels. PORTALS-CGYRO passes `GZ`/`Mt`/`Qie` to the optimizer instead of zeros, so those channels can be included in predictions as with PORTALS-TGLF. Also fixed: the EM contribution to the ion particle fluxes was silently dropped.

*   💥 **MAESTRO plotting additions** — the `Special` tab gains a `Confinement Evolution` column (energy confinement time and `H98y2`/`H89p` across beats), the sharpness beat gains a profiles-vs-coordinates tab, plus Special-tab cosmetics (spacing, no meaningless non-fusion `Q`/`Pfus` traces, density-peaking autoscale, multi-case `Profiles ALL` tab only when comparing runs).

### Bug Fixes

*   🐛 **Fixed the trace-impurity D/V analysis (`TGLF.runAnalysis(analysisType="Z")`) using a trace species a factor ~2 too heavy**: the physical mass in amu was written into `input.tglf`, whose `MASS_X` entries are deuterium-normalized. Note: this changes the `DZ`/`VZ`/`VoD` coefficients of re-run trace analyses.

*   🐛 **Fixed artificial a/L spikes at the MAESTRO boundary-condition location (the last PORTALS control point).** Three sources addressed: profile creation placed the core/pedestal kink exactly at the pedestal top (~2x too-high temperature gradients, ~5x for nearly-flat ne); the PORTALS re-gridding rang across that kink (now shape-preserving PCHIP); and the sharpness/confinement beats recreated the kink at every BC change. Gradients at the BC are now clean and preserved across beats. Note: the creation/re-gridding fixes take effect on fresh runs (cold start from the first beat).

*   🐛 **Fixed the TGLF scan-based flux-uncertainty model (PORTALS stds) producing artificially zero or inconsistent stds around zero values**: the minimum-delta floor was a no-op exactly at zero, rotation scans had no floor at all (zero-rotation plasmas reported exactly-zero `Mt` stds), and co-varied species keys were not floored with their representative. The `tglf_ball.npz` reuse file now records its scan-variable keys and is discarded if a restart changes the predicted channels.

*   🐛 **Fixed MAESTRO's PORTALS-to-PORTALS warm start being silently disabled for `predicted_rho`-style namelists**: a key-presence check stored `predicted_roa=None` in the trans-beat hand-off, so the next beat declared a bogus boundary "move" — forcing a full simple-relax initialization instead of one surrogate-flux-matched point and discarding the last-location surrogate data at every beat.

*   🐛 **Fixed an `IndexError` crash at the end of a converged PORTALS run resumed from a previous attempt** (`finalize_evaluation`): the best-evaluation index can exceed the stored powerstates on resumed runs; it is now clamped to the last available evaluation.

*   🐛 **Fixed a crash at PORTALS initialization when the optimization data contains coincident evaluations** (e.g. a flux-match trajectory revisiting a point): the tabular lookup now uses the first matching row.

*   🐛 **Fixed a startup crash in the LHS initialization of Bayesian optimization** when the design-variable bounds arrive as a numpy array. Also applied to `main` as a hotfix.

*   🐛 **MAESTRO/PORTALS runs no longer crash when a SLURM status poll cannot retrieve `squeue`** (e.g. a transient cluster or VPN hiccup) — the poll degrades to a pending state instead of raising.

*   🐛 **Remote `exec_command` calls are now covered by the SSH transient-retry policy** (used for `squeue`/`tar` status polls), so long PORTALS/CGYRO runs survive brief connection drops during status checks.

### Changes for developers (internal execution)

*   🔎 **The MAESTRO namelist now warns when a `fixed_bc` creator uses a `bc_coordinate` other than `'rho'`**, since the boundary condition is expected in `rho`.

### Back-compatibility considerations and defaults

*   🔮 **The new `gknn` turbulence model needs the optional `tglf_gknn_jax` package** (`pip install tglf_gknn_jax[onnx]`). It is not a hard dependency: code paths degrade gracefully when it is not installed, so existing environments are unaffected.

---

*Thanks to everyone who contributed to this release: Garud Snoep, Audrey Saltzman. Portions of this release were developed with AI-assisted coding (Claude Code).*
