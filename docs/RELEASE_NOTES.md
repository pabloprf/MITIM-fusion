# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   ⚛️ **EPED plasma composition (full EPED)**: the MAESTRO EPED beat and `EPEDtools.EPED.run` now feed EPED the actual plasma's main-ion mass and an effective impurity derived from the state (reproducing both Zeff and fuel dilution, `zi_eff = (Zeff − d)/(1 − d)`), instead of a hardcoded 50/50 D-T + neon. `m`/`z`/`mi`/`zi` default to the old values and are overridable via the beat's `corrections_set`; a new `zeff_location` knob (`vol_avg` default, `pedestal`) sets where Zeff and the dilution are taken; the EPED-NN path is unaffected. **NOTE:** in EPED1 the only composition quantity that enters the solve is `Zeff` (via TOQ / bootstrap-collisionality); `m`/`z`/`mi`/`zi` are recorded but inert, so scanning them at fixed `Zeff` leaves the predicted pedestal unchanged.

*   🔌 **`gacode_state.recompute_targets()`**: re-derives the radiation (qbrem/qsync/qline), fusion alpha-heating (qfuse/qfusi) and electron-ion exchange (qei) power profiles from the kinetic profiles with the analytic target model, evaluated on the full radial grid (no edge points left stale). It is now the single entry point used by the MAESTRO confinement beat and RAPIDS instead of their inline powerstate round-trips; `debug=True` plots each recomputed channel against the profiles that drive it.

*   📊 **MAESTRO summary report** (`Outputs/maestro_summary.md`) now embeds the per-beat "special quantities" evolution and the timing breakdown (`maestro_special.png`, `maestro_timing.png`) next to the existing beat-flow diagram — the same plots produced when plotting a case, now in the standalone report.

*   🎼 **`mitim_plot_maestro --summary`** (alias `--special`) plots only the cross-beat "MAESTRO special" and "MAESTRO timings" summary tabs, skipping the per-beat / profile / transition tabs — a fast at-a-glance view of a run.

*   🧭 **`mitim_plot_neo`** reads and plots NEO results from an existing folder, mirroring `mitim_plot_tglf` (positional folders, `--suffixes`, `--gacode` for normalizations) via a new `NEO.prep_from_file`.

*   ⚙️ **SR acquisition optimizer (`halt_on`)**: new `optimizer_options.sr.halt_on` (`best` | `all`). The batched restarts halt together when the *best* restart meets the tolerance (`best`, default, unchanged) or only once *every* restart does (`all`) — use `all` when more than one `x_best` is consumed, so all returned candidates are comparably converged instead of the slower ones being truncated. The batched-restart behavior of both ROOT and SR (and what `relative_improvement_for_stopping` controls) is now documented in `namelist.optimization.yaml` and the solver docstrings.

*   💾 **MAESTRO disk footprint — lean PORTALS pickles**: under `keep_all_files: false`, PORTALS beats persist a *lean* `optimization_object.pkl` that drops the fitted GP surrogates (`steps`, ≈⅔ of each file); the per-iteration checkpoints written *during* the run stay full, so a preempted run still resumes. Only the **last** PORTALS beat keeps its heavy outputs; intermediate beats are stripped of pickles plus per-iteration artifacts, since chaining only needs `surrogate_data.csv` and each beat's `input.gacode` (both kept). Together these roughly halve a finished case (e.g. ~210 MB → ~100 MB). Replotting metrics still works (output identical; only the GP-posterior "Expected" plots are unavailable), and re-running/extending a finished run stays idempotent via per-beat parameter snapshots. New low-level switch: `MITIM_BO.save(lean=True)`. Standalone PORTALS runs and `keep_all_files: true` are unchanged.

*   💾 **PORTALS `optimization_extra.pkl` — drop recomputable `derived`**: the per-iteration powerstates stored in `extra.pkl` no longer carry each profile's full `derived` dict (flux-surface geometry, gradients, …). Since `derived` is fully recomputable from the base profiles via `derive_quantities()` and is only read at plot time, it is now stripped before pickling and rebuilt lazily on read in `PORTALSanalysis`. Cuts `extra.pkl` by ~60% (e.g. ~104 → ~40 MB). Base profiles and transport fluxes (not recomputable) are kept at full float64; in-run consumers use those, never `derived`. Gated by `store_lean_powerstates` (default `True`; set `False` to keep `derived`).


*   🪚 **MAESTRO TRANSP adaptive sawtooth-period floor** (`min_sawtooth_period_ms`): on compact, cold-core plasmas the Park-Monticello period `tau_PM ~ R²·Te0^1.5/Zeff` — the floor on the Porcelli trigger's crash interval (`c_sawtooth(2)·tau_PM`) — is sub-ms, so sawteeth crash every ~1 ms, forcing sub-ms timesteps and a multi-GB output CDF. The TRANSP beat now sets `c_sawtooth(2) = min_sawtooth_period_ms / tau_PM` per case (`tau_PM` from new `PLASMAtools.park_monticello_sawtooth_period`), imposing an absolute floor on the crash interval while leaving large machines (whose `tau_PM` already exceeds the floor) untouched. Default **10 ms** in the maestro template; `null` bypasses.

*   🧊 **MAESTRO 99.5% shaping re-freeze timing** (`maestro.refreeze_995_after_beat`): controls WHEN the pedestal shaping (kappa995/delta995/zeta995) fed to EPED is fixed, orthogonal to `separatrix.freeze_995_from` (which sets HOW it is extracted). `0` (default, old behavior) freezes at initialization; `N>0` re-extracts once from beat N's evolved equilibrium (e.g. after the first TRANSP beat) and reuses it thereafter; `null` never freezes, so each EPED beat recomputes from its own current equilibrium. Motivated by separatrix-initialized runs, whose internal flux surfaces (and hence the init 99.5% values) are a parametric guess rather than a solved equilibrium. Also adds a runtime warning that `freeze_995_from`'s parameterization is only applied for geqdsk init (ignored for separatrix, which always interpolates the built profiles to psiN=0.995).

*   🎚️ **MAESTRO TRANSP extraction-slice selector** (`extract_at`): which CDF time slice is handed to the next beat is now configurable in the transp beat's `parameters_prepare` — `saw` / `saw-N` (the last sawtooth, or N coarse slices before it) or `last` / `last-N` (N before the last simulated slice). Default `saw-1` reproduces the historical behavior (`ind_saw-1`, a small step-back that avoids sampling the sawtoothing crash profiles on the coarse MAESTRO grid).


### Bug Fixes

*   🐛 **TRANSP CDF file-descriptor leak (major MAESTRO scratch regression)**: the multi-GB TRANSP output CDF was held open for the **entire** MAESTRO run and fork-inherited by every later PORTALS/TGLF worker, so once the run folder was wiped (`keep_all_files: false`) the deleted CDF stayed pinned on disk as an NFS silly-rename (`.nfsXXXX`) until the case ended — inflating each case's scratch use by 2–5 GB and overrunning per-user quotas on large scans. The TRANSP beat now releases the cached `transp_output` (and any `self.transp.t.cdfs`) at the end of `run()`, before the PORTALS beats fork; `transp_output` gained an idempotent `close()`, and the various metadata/finalize/completeness readers now close their handles.

*   🐛 **MAESTRO TRANSP Porcelli sawtooth triggering**: earlier MAESTRO runs could finish without ever triggering sawteeth through the Porcelli model if the Park-Monticello predicted period was too long (10% of it was taking as the minimum). The TRANSP beat now has the flexibility of using a user-specified sawtooth-period floor (see new features above).

*   🐛 **MAESTRO TRANSP early-extraction floor** (`min_extraction_flattop_fraction`, default 0.5): a plasma whose only sawtooth fired early (then never again) had its profiles extracted too soon — before heating / current diffusion settled. The extraction is now floored at this fraction of the flattop window: if the `extract_at` slice lands earlier, it moves to the first slice at/after the floor. Healthy runs (last sawtooth already past mid-flattop) are unchanged; `null` disables.

*   🐛 **MAESTRO engineering scans** (`launch_scan`): the `exclude` and `qos` SLURM allocation settings were silently dropped and are now forwarded to the array submission, so node exclusions actually take effect.

*   🐛 **`mitim_check_maestro`** now recognizes the sharpness, confinement and lengyel beats (previously shown as `UNKNOWN`) by their `run_<type>` folder.

*   🐛 **TRANSP (singularity) finish** no longer leaves a duplicate copy of the retrieved `results/` tree (notably the heavy `.CDF`): its contents are surfaced into the run folder and the redundant `results/` is removed.

*   🐛 **`mitim_plot_cgyro` timing panels** no longer clip later cases: the per-output and cumulative-cost y-axes now expand to fit every overlaid case instead of freezing to the first case's range (`set_ylim(bottom=0)` was disabling y-autoscale).

*   🐛 **MAESTRO per-beat logs** (`Outputs/Logs/beat_<n>_*.log`) are now line-buffered, so a long-running beat's log (e.g. a multi-hour TRANSP run) streams progress live instead of staying empty until the beat finishes — the block buffer previously only flushed on close.

*   🐛 **PORTALS restart robustness**: a resumed optimization whose pkl checkpoint lagged behind `optimization_data.csv` could leave `x_next` empty and crash the results writer with an `IndexError` (indexing one past the end of `train_X`). `MITIM_BO.updateSet` now treats an empty `x_next` as a no-op and skips the step instead of crashing.

*   🐛 **`mitim_job.run()` retry robustness**: `run()` is now idempotent w.r.t. its input file/folder lists. The "repeat once after a transient error" retry (e.g. a code returning incomplete output) re-ran the in-place relativization on already-relative paths and crashed in `relative_to()` with a misleading `'mitim_bash.src' is not in the subpath…` error; it now re-runs cleanly and surfaces the real failure.

*   🐛 **`mitim_plot_transp` boundary-less g-file**: a TRANSP run whose reference g-eqdsk has no plasma-boundary contour (`nbbbs=0`) no longer crashes the whole read. `getGFILE` now skips a g-file that megpy cannot derive (it needs the boundary to build `R_psi_lfs`) with a warning and falls through to the next candidate extension, instead of raising.

*   🐛 **TRANSP run-abort detection & CDF-build retry**: a TRANSP run that aborts during initialization (e.g. a t=0 TEQ equilibrium failure) is now flagged `stopped` instead of `finished` — a fatal `ABORTR`/`bad_exit`/segfault in the log outranks the singularity wrapper's unconditional `Finished TRANSP run app.` line, so the run fails fast with the real error. When the finish step fails to build `{runid}.CDF` for a *completed* run, `TRANSPsingularity.fetch` falls back to a `look` rebuild that re-stages the `.PLN` files from the remote run folder and re-runs `plotcon`, instead of hard-failing on the missing CDF. Mid-run intermediate grabs in `checkUntilFinished` are now best-effort: a failed intermediate `look`/CDF is logged and skipped instead of killing a healthy run or masking the real `stopped` error.

*   🩺 **Informative TRANSP failure message** (`TRANSPdebug`): a stopped TRANSP run now raises with a *best-effort estimate* of the cause parsed from the run log — the signal and a plain-language gloss, the simulation time/step, the physics event at the trap (MHD equilibrium/TEQ vs Porcelli sawtooth), and geometry/underflow breadcrumbs — instead of a bare "TRANSP stopped". It also flags InfiniBand/RDMA container-launch failures (mlx5 UD-QP denied) as infrastructure rather than physics. The message is a heuristic reading of the log, **not** a guaranteed root cause. New `TRANSPdebug` module, usable interactively via `diagnose_transp_logfile`.

*   🐛 **geqdsk → `input.gacode` `B_unit`** (`MITIMgeqdsk.to_profiles`): `torfluxa` was divided by an extra `2π` — megpy's `derived['phi']` is already per-radian (`∫q dψ == phi[-1]`) — so `B_unit`, and with it the gyroBohm normalization and `P_PRIME_LOC`/`BETAE` (`∝ 1/B_unit²`), came out `2π` too small in any `input.gacode` built through `to_profiles` (fixed: drop the `/(2π)`). **In practice this affects almost nobody**: standard `input.gacode` come from `profiles_gen`/TRANSP (unaffected), and MAESTRO's main equilibrium path calls `equilibrium_to_profiles` directly (unaffected) while a geqdsk-initialized beat is overwritten by the first TRANSP beat before any transport/PORTALS run. It only bites when a `to_profiles` output is fed *directly* into a transport code (e.g. a geqdsk-built `input.gacode` sent straight to TGLF with no TRANSP step).

### Changes for developers (internal execution)

*   🤖 **MAESTRO investigation subagent** (`.claude/agents/maestro.md`): a Claude Code agent that forensically compares and debugs MAESTRO runs — it knows the `Beats/` layout, each beat's inputs/outputs, where the logs/timing/namelist artifacts live, and how to load and overlay states headlessly. Shipped in-repo by un-ignoring `.claude/agents/` (the rest of `.claude/` stays local).

*   🔎 **MAESTRO scan per-case logs** now symlink each case's `slurm.out`/`slurm.err` to the live SLURM array logs (`slurm_output/slurm_error_<jobid>_<task>.dat`) instead of redirecting — logs stream live and are reachable from both the case and main folders (links dangle only if a case folder is copied away on its own).

*   🔎 **MAESTRO EPED beat** failure diagnostics: reports the EPED inputs (R, a, BetaN, …) alongside the "no stable solution" warning and the final failure, and now distinguishes a compute-node execution failure (TOQ/ELITE produced no output files) from a genuine pedestal no-solution — the former is surfaced immediately as an execution error instead of being masked by futile teped-lowering retries and a misleading "no stable solution".

*   🔧 **SLURM `exclusive` accepts a string**: setting `exclusive: "user"` (or `"mcs"`) now emits `#SBATCH --exclusive=user` instead of plain `--exclusive`, so a scan can keep nodes free of *other* users while still packing the user's own array tasks onto each node — node isolation for large scans without the one-task-per-node core waste. A bare `True` is unchanged (plain `--exclusive`). MAESTRO scan arrays (`_submit_array`) now forward `slurm['exclusive']` (it was hardcoded off), so `exclusive="user"` set in a scan launcher actually takes effect.

*   🌀 **PORTALS neoclassical E×B shear** now has a standalone dev test (`tests/dev_tests/test_portals_exb_shear.py`): it flux-matches the same plasma with `transport.options.neo.vgen_exb_shear` off vs on (the NEO-VGEN neoclassical Er at zero toroidal rotation) to isolate the stabilization, and bundles the comparison, the per-run PORTALS metrics and the VGEN notebook into one figure. Alongside it, `NEO.plot_vgen` (and `mitim_plot_vgen`) gained an optional `mark_rho` to scatter chosen radii — e.g. the PORTALS predicted radii — on the smoothed profiles in the VGEN smoothing tab.

*   🧮 **E×B / parallel-velocity shear as state quantities**: `mitim_state.derive_quantities` now exposes `derived['gamma_exb']` and `derived['gamma_p']` — the E×B and parallel-velocity shearing rates in TGLF's `VEXB_SHEAR`/`VPAR_SHEAR` normalization (`c_s/a`), computed once from `w0`, `q`, `r`, `a`, `c_s`. `to_tglf` now consumes them instead of recomputing the rotation-shear formula inline (behavior-preserving), so the E×B shearing-rate profile is available directly on any state (`p.derive_quantities(); p.derived['gamma_exb']`) without a TGLF prep.

### Back-compatibility considerations and defaults

*   💾 **Lean PORTALS pickles under `keep_all_files: false`**: intermediate PORTALS beats' `optimization_object.pkl`/`optimization_extra.pkl` are pruned, and the retained (last-beat) `optimization_object.pkl` is lean (no GP surrogates). Replotting metrics still works; the GP-posterior ("Expected") plots and a pickle-based surrogate resume of those finished beats are not available. Set `keep_all_files: true` to retain full pickles.

*   🪚 **Sawtooth floor defaults**: `min_sawtooth_period_ms` defaults to **1 ms** in `TRANSPbeat.prepare` (and `NMLtools` keeps `c_sawtooth(2)=0.1`), so namelists without the key reproduce the historical ~1 ms behavior; the maestro **template** sets **10 ms**. `null` bypasses the adaptive floor.

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
