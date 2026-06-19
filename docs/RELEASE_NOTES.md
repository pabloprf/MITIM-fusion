# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   ⚛️ **EPED plasma composition (full EPED)**: the MAESTRO EPED beat and `EPEDtools.EPED.run` now feed EPED the actual plasma's main-ion mass and an effective impurity derived from the state, instead of a hardcoded 50/50 D-T + neon. The effective impurity charge reproduces both Zeff and the fuel dilution (`zi_eff = (Zeff − d)/(1 − d)`). `m`/`z`/`mi`/`zi` default to the old values (preserving EPED-NN consistency) and are overridable via the beat's `corrections_set`; a new `zeff_location` knob (`vol_avg` default, `pedestal`) sets where Zeff and the dilution are taken. The EPED-NN path is unaffected. **NOTE: in the current EPED1 build the *only* composition quantity that enters the pedestal solve is `Zeff` (via the TOQ equilibrium / bootstrap-collisionality); the `m`/`z`/`mi`/`zi` fields are passed in and recorded in the output state but are inert in the model — the KBM-width and peeling-ballooning stages carry no ion-mass or impurity-charge dependence. Scanning them at fixed `Zeff` therefore leaves the predicted pedestal unchanged (verified in `tests/dev_tests/test_eped_fuel_impurity.py`: every physics output is bit-identical across the scan, only the echoed input differs). So this change makes EPED record the true composition and honor it through `Zeff`, but it does not add an isotope/charge sensitivity that EPED1 itself does not model.**

*   🔌 **`gacode_state.recompute_targets()`**: re-derives the radiation (qbrem/qsync/qline), fusion alpha-heating (qfuse/qfusi) and electron-ion exchange (qei) power profiles from the kinetic profiles with the analytic target model, evaluated on the full radial grid (no edge points left stale). It is now the single entry point used by the MAESTRO confinement beat and RAPIDS instead of their inline powerstate round-trips; `debug=True` plots each recomputed channel against the profiles that drive it.

*   📊 **MAESTRO summary report** (`Outputs/maestro_summary.md`) now embeds the per-beat "special quantities" evolution and the timing breakdown (`maestro_special.png`, `maestro_timing.png`) next to the existing beat-flow diagram — the same plots produced when plotting a case, now in the standalone report.

*   🎼 **`mitim_plot_maestro --summary`** (alias `--special`) plots only the cross-beat "MAESTRO special" and "MAESTRO timings" summary tabs, skipping the per-beat / profile / transition tabs — a fast at-a-glance view of a run.

*   🧭 **`mitim_plot_neo`** reads and plots NEO results from an existing folder, mirroring `mitim_plot_tglf` (positional folders, `--suffixes`, `--gacode` for normalizations) via a new `NEO.prep_from_file`.

*   ⚙️ **SR acquisition optimizer (`halt_on`)**: new `optimizer_options.sr.halt_on` (`best` | `all`). The batched restarts halt together when the *best* restart meets the tolerance (`best`, default, unchanged) or only once *every* restart does (`all`) — use `all` when more than one `x_best` is consumed, so all returned candidates are comparably converged instead of the slower ones being truncated. The batched-restart behavior of both ROOT and SR (and what `relative_improvement_for_stopping` controls) is now documented in `namelist.optimization.yaml` and the solver docstrings.


### Bug Fixes

*   🐛 **MAESTRO engineering scans** (`launch_scan`): the `exclude` and `qos` SLURM allocation settings were silently dropped and are now forwarded to the array submission, so node exclusions actually take effect.

*   🐛 **`mitim_check_maestro`** now recognizes the sharpness, confinement and lengyel beats (previously shown as `UNKNOWN`) by their `run_<type>` folder.

*   🐛 **TRANSP (singularity) finish** no longer leaves a duplicate copy of the retrieved `results/` tree (notably the heavy `.CDF`): its contents are surfaced into the run folder and the redundant `results/` is removed.

*   🐛 **`mitim_plot_cgyro` timing panels** no longer clip later cases: the per-output and cumulative-cost y-axes now expand to fit every overlaid case instead of freezing to the first case's range (`set_ylim(bottom=0)` was disabling y-autoscale).

*   🐛 **MAESTRO per-beat logs** (`Outputs/Logs/beat_<n>_*.log`) are now line-buffered, so a long-running beat's log (e.g. a multi-hour TRANSP run) streams progress live instead of staying empty until the beat finishes — the block buffer previously only flushed on close.

*   🐛 **PORTALS restart robustness**: a resumed optimization whose pkl checkpoint lagged behind `optimization_data.csv` could leave `x_next` empty and crash the results writer with an `IndexError` (indexing one past the end of `train_X`). `MITIM_BO.updateSet` now treats an empty `x_next` as a no-op and skips the step instead of crashing.

*   🐛 **`mitim_job.run()` retry robustness**: `run()` is now idempotent w.r.t. its input file/folder lists. The "repeat once after a transient error" retry (e.g. a code returning incomplete output) re-ran the in-place relativization on already-relative paths and crashed in `relative_to()` with a misleading `'mitim_bash.src' is not in the subpath…` error; it now re-runs cleanly and surfaces the real failure.

*   🐛 **TRANSP run-abort detection & CDF-build retry**: a TRANSP run that aborts during initialization (e.g. a t=0 TEQ equilibrium failure) is now flagged as `stopped` instead of `finished` — the singularity wrapper's unconditional `Finished TRANSP run app.` line no longer outranks a fatal `ABORTR`/`bad_exit`/segfault in the log, so the run fails fast with the real error in the log tail instead of proceeding to a confusing missing-CDF / failed-`look` prompt. When the finish step's `trlook`/`plotcon` does fail to build `{runid}.CDF` for a *completed* run (e.g. a transient "TF.PLN file not found" abort), `TRANSPsingularity.fetch` falls back to a `look` rebuild that re-stages the `.PLN` files from the remote run folder (`job.folderExecution`, not the local run directory where they never live) and re-runs `plotcon`, instead of hard-failing downstream on the missing CDF.

### Changes for developers (internal execution)

*   🤖 **MAESTRO investigation subagent** (`.claude/agents/maestro.md`): a Claude Code agent that forensically compares and debugs MAESTRO runs — it knows the `Beats/` layout, each beat's inputs/outputs, where the logs/timing/namelist artifacts live, and how to load and overlay states headlessly. Shipped in-repo by un-ignoring `.claude/agents/` (the rest of `.claude/` stays local).

*   🔎 **MAESTRO scan per-case logs** now symlink each case's `slurm.out`/`slurm.err` to the live SLURM array logs (`slurm_output/slurm_error_<jobid>_<task>.dat`) instead of redirecting — logs stream live and are reachable from both the case and main folders (links dangle only if a case folder is copied away on its own).

*   🔎 **MAESTRO EPED beat** failure diagnostics: reports the EPED inputs (R, a, BetaN, …) alongside the "no stable solution" warning and the final failure, and now distinguishes a compute-node execution failure (TOQ/ELITE produced no output files) from a genuine pedestal no-solution — the former is surfaced immediately as an execution error instead of being masked by futile teped-lowering retries and a misleading "no stable solution".

*   🔧 **SLURM `exclusive` accepts a string**: setting `exclusive: "user"` (or `"mcs"`) now emits `#SBATCH --exclusive=user` instead of plain `--exclusive`, so a scan can keep nodes free of *other* users while still packing the user's own array tasks onto each node — node isolation for large scans without the one-task-per-node core waste. A bare `True` is unchanged (plain `--exclusive`). MAESTRO scan arrays (`_submit_array`) now forward `slurm['exclusive']` (it was hardcoded off), so `exclusive="user"` set in a scan launcher actually takes effect.

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
