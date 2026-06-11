# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **New MAESTRO `confinement` beat**, which sets the temperature boundary condition at a chosen radial location such that the plasma matches a target confinement level (H-factor, `H98y2` or `H89p`). Since the H-factor cannot be inverted analytically for T_bc, it is found by Nelder-Mead minimization (same spirit as the eped_initializer BetaN matching), applying the BC at each trial Te_bc with the sharpness-beat machinery (core preserves a/LT and a/Ln; edge anchored at the BC and separatrix values, with `edge_shape` selecting a straight line in psi_n or the initializer's pedestal tanh in r/a). An optional `alpha_power_feedback` recomputes qfuse/qfusi analytically at every trial Te_bc so the H-factor accounts for the alpha-heating response (relevant for burning plasmas; no-op for non-DT), and `density_treatment` chooses whether the density profiles are rescaled to ne_bc or left completely untouched by the BC change (the latter option is also available in the sharpness beat). Includes all standard beat methods: restart robustness, plots (profiles and a/L gradient diagnostics around the BC, optimization convergence, H-factor-inputs panel), and trans-beat parameter hand-off so subsequent PORTALS beats reuse surrogate data. See the `confinement` section of `templates/namelist.maestro.yaml`.

*   💥 **New `gknn` turbulence model for PORTALS** — a TGLF-GKNN-JAX neural-network surrogate (TGLF `SAT3` + GKNN corrections), selectable via the `gknn` block under `transport.options`. It ships a dedicated `profiles_postprocessing_fun` that lumps the plasma to two ions (main + a single lumped impurity) and enforces quasineutrality for network compatibility, plus an `apply_gknn` toggle to run with or without the GKNN correction factors on top of the base TGLF-NN. The `tglf_gknn_jax` package is an optional dependency — runs degrade gracefully when it is absent. See the `gknn` block in `templates/namelist.portals.yaml`.

*   💥 **New `mitim_clean_maestro` tool** — a CLI (`mitim_clean_maestro FOLDER ...`) that prunes a finished MAESTRO run folder down to the directory structure and key files needed to reload results (`input.gacode`, namelists, `.nc`/`.npy` outputs, optimization data, figures, …), with an aggressive mode for deeper cleaning. Useful for shrinking runs before archiving or transfer.

*   💥 **MAESTRO plotting additions** — the `Special` tab gains a `Confinement Evolution` column (energy confinement time and `H98y2`/`H89p` across beats), and the sharpness beat gains a profiles-vs-coordinates tab (Te/Ti/ne against `rho_tor`, `r/a`, `psi_n`). Special-tab cosmetics too: wider inter-column spacing, non-fusion `Q`/`Pfus` are no longer drawn as meaningless ~1e-14 noise, the density-peaking axis autoscales instead of clamping, and the multi-case `Profiles ALL` tab is only built when more than one run is plotted.

### Bug Fixes

*   🐛 **Fixed artificial a/L spikes at the MAESTRO boundary-condition location (the last PORTALS control point).** Three contributing causes, addressed at the source: (1) profile creation (`MITIMfunctional_aLyTanh`, behind the eped/parameterization/fixed_bc initializers) placed the core/pedestal slope discontinuity exactly at the pedestal top, so the derivative there mixed core and pedestal slopes (~2x too high for the temperatures, ~5x for nearly-flat ne) — the pedestal is now re-anchored one grid point past the top, and the derivative at the top reads exactly the prescribed core a/L; (2) the PORTALS fine-grid re-gridding interpolated with a cubic spline across that kink, ringing into non-monotonic wiggles next to the control point (artificial a/Lne dips) — it now uses shape-preserving PCHIP interpolation; (3) the sharpness/confinement beats placed their analytical edge exactly at the BC point, recreating the kink at every beat — the edge now starts one grid point past the BC and the core scaling is an exact multiplication, so a/L at the BC is preserved to machine precision across these beats. Unpredicted channels (e.g. ne when PORTALS evolves only te/ti) no longer inherit corrupted gradients between beats. Note: the creation/re-gridding fixes take effect on fresh runs (cold start from the first beat); states produced by earlier runs still carry the old kink.

*   🐛 **Fixed the TGLF scan-based flux-uncertainty model (PORTALS stds) producing artificially zero or inconsistent stds around zero values.** The `minimum_delta_abs` floor — meant to avoid no-op scans at zero gradients — was a no-op exactly at zero (`np.sign(0)=0`); rotation scans (`VEXB_SHEAR`, and the `VPAR_SHEAR` of all species co-varied with it) had no floor at all, so zero-rotation plasmas reported `Mt` stds of exactly zero (a zero-noise GP on the momentum channel); and the species keys co-varied by `completeVariation_TGLF` (`RLTS_3+`, `RLNS_2+`) were not floored alongside their representative, making the group perturbation inconsistent at near-zero gradients. The `tglf_ball.npz` reuse file now also records its scan-variable keys and is discarded (with a warning) if a restart changes the predicted channels, instead of silently misinterpreting the stored inputs. Impurity (`GZ`), rotation (`Mt`) and exchange (`S`) outputs were audited and read consistently across base/scans/ball; the scan-variable construction is now a single documented helper shared by the single-plasma and batched paths.

*   🐛 **Fixed MAESTRO's PORTALS-to-PORTALS warm start being silently disabled for `predicted_rho`-style namelists.** The PORTALS template carries both `predicted_roa` (null) and `predicted_rho`; the trans-beat hand-off checked key presence instead of value, stored `predicted_roa=None`, and the next PORTALS beat then could not find its `predicted_rho` to verify the boundary location had not moved — declaring a bogus "move" (e.g. "from 0.9 to 0.9") that disabled `try_flux_match_only_for_first_point` (forcing a full multi-point simple-relax initialization with real transport evaluations instead of one surrogate-flux-matched point) and discarded the last-location surrogate data at every beat. Runs whose portals overlay used `predicted_roa` were unaffected. Checks are now value-aware.

*   🐛 **Fixed a crash at PORTALS initialization when the optimization data contains coincident evaluations** (`ValueError: can only convert an array of size 1`): a simple-relax flux-match trajectory that oscillates back onto a point it already evaluated leaves duplicate rows in `optimization_data.csv`, which the tabular lookup did not tolerate. The first matching row is now used (with a warning).

*   🐛 **Fixed a startup crash in the LHS initialization of Bayesian optimization** when the design-variable bounds arrive as a numpy array (the GPU device/dtype-adoption path required a tensor). PORTALS/BO runs no longer fail at initialization on this path. Also applied to `main` as a hotfix.

*   🐛 **MAESTRO/PORTALS runs no longer crash when a SLURM status poll cannot retrieve `squeue`** (e.g. a transient cluster or VPN hiccup) — the poll degrades to a pending state instead of raising.

*   🐛 **Remote `exec_command` calls are now covered by the SSH transient-retry policy** (used for `squeue`/`tar` status polls), so long PORTALS/CGYRO runs survive brief connection drops during status checks.

### Changes for developers (internal execution)

*   🔎 **The MAESTRO namelist now warns when a `fixed_bc` creator uses a `bc_coordinate` other than `'rho'`**, since the boundary condition is expected in `rho`.

### Back-compatibility considerations and defaults

*   🔮 **The new `gknn` turbulence model needs the optional `tglf_gknn_jax` package** (`pip install tglf_gknn_jax[onnx]`). It is not a hard dependency: importing it is wrapped in a try/except, and code paths that reference `tglf_gknn` degrade gracefully (no warning spam) when it is not installed, so existing environments are unaffected.

---

*Thanks to everyone who contributed to this release: Garud Snoep, Audrey Saltzman. Portions of this release were developed with AI-assisted coding (Claude Code).*
