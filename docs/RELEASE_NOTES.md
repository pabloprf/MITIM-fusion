# v5.3.0 — MAESTRO control and robustness, QuaLiKiz interface

This release is centered on MAESTRO: new control over its TRANSP and pedestal-shaping steps (sawtooth-period floor, boundary-surface backoff, extraction-slice selection, q-seed sanitization, a configurable and re-freezable 99.5% shaping surface, Lengyel Zeff relaxation), plus a broad robustness pass on the failure modes seen on compact, strongly shaped devices — TRANSP aborts now surface their real cause instead of a bare "stopped", EPED survives preemption and requeue, and sawteeth actually trigger. It also adds a **QuaLiKiz** interface usable as a PORTALS turbulence backend, a real plasma composition for EPED, local-optima mining in the Bayesian-optimization loop, a pulse-duration (volt-second) calculator, and a set of changes that cut a finished MAESTRO case's disk footprint several-fold.

### New Features

*   💾 **Lean PORTALS pickles under `keep_all_files: false`**: PORTALS beats drop the fitted GP surrogates from `optimization_object.pkl` (≈⅔ of the file), and only the **last** beat keeps its heavy outputs — chaining only needs `surrogate_data.csv` and each beat's `input.gacode`. Per-iteration checkpoints stay full, so a preempted run still resumes. Roughly halves a finished case; replotting metrics is unchanged (only the GP-posterior "Expected" plots are lost). New switch `MITIM_BO.save(lean=True)`.

*   🧹 **`mitim_prune_maestro`**: applies the same savings to a run that already **finished** with `keep_all_files: true` — wipes each beat's `run_<name>/` folder (multi-GB TRANSP CDF included) and slims the PORTALS outputs, leaving `beat_results/` and `initializer_*` untouched. Dry-run by default, `--apply` to delete; reclaims e.g. ~950 → ~200 MB.

*   💾 **PORTALS `optimization_extra.pkl` drops the recomputable `derived` dict**: stripped before pickling and rebuilt lazily on read in `PORTALSanalysis` (~60% smaller, e.g. ~104 → ~40 MB). Base profiles and fluxes are kept at full float64. Gated by `store_lean_powerstates` (default `True`).

*   🌀 **QuaLiKiz interface**: new `mitim_tools.qualikiz_tools` runs and reads QuaLiKiz standalone from an `input.gacode`, and is selectable as a PORTALS turbulence backend with `turbulence_model: "qlk"` (the neoclassical side keeps NEO). All radii, and every gradient-perturbation case used for flux uncertainties, are packed into a *single* execution via QuaLiKiz's own `dimx` scan — one job per PORTALS iteration. Requires the external `qualikiz_tools` package and a `qualikiz` entry in `config_user.json`; the import is caught, so workflows are unaffected when it is absent. **NOTE**: QuaLiKiz uses circular / s-alpha geometry (shaping is dropped in `to_qualikiz`, so fluxes are not comparable to TGLF on a shaped plasma) and provides no turbulent Qie (zero-filled, so `turbulent_exchange_as_surrogate` must stay `False`) — both are properties of QuaLiKiz itself, not of this interface. Teaching scripts: `qualikiz_01_run_from_inputgacode.py`, `portals_03_qualikiz_standard.py`.

*   ⚛️ **EPED plasma composition**: the MAESTRO EPED beat and `EPEDtools.EPED.run` now feed EPED the actual main-ion mass and an effective impurity derived from the state (matching both Zeff and fuel dilution, `zi_eff = (Zeff − d)/(1 − d)`) instead of a hardcoded 50/50 D-T + neon. The old values remain the defaults, overridable via `corrections_set`; new `zeff_location` knob (`vol_avg` default, `pedestal`); EPED-NN unaffected. **NOTE**: in EPED1 only `Zeff` enters the solve (via TOQ / bootstrap collisionality) — the mass/charge entries are recorded but inert.

*   ⛏️ **Local optima mining in Bayesian optimization** (`local_optima_options`): every `n_acq_batches_per_cycle` acquisition batches, the BO loop can mine the GP surrogate for diverse local optima with independent L-BFGS-B restarts, evaluate them with the physics model and add them to the training data. `apply: false` by default. Points are labeled by source in `optimization_data.csv` (`training`, `acquisition`, `local_optima`).

*   🪚 **MAESTRO TRANSP adaptive sawtooth-period floor** (`min_sawtooth_period_ms`): TRANSP's Porcelli trigger only enforces a *relative* floor on the crash interval, `c_sawtooth(2) * tau_PM`, with the Park-Monticello period `tau_PM ~ R^2 * Te0^1.5 / Zeff`. At the previous fixed `c_sawtooth(2) = 0.1` that floor is sub-ms on compact, cold-core plasmas — crashes every few timesteps and a multi-GB output CDF — and on large machines long enough that sawteeth could go untriggered for the whole flattop. The beat now sets `c_sawtooth(2) = min_sawtooth_period_ms / tau_PM` per case (new `PLASMAtools.park_monticello_sawtooth_period`), making the floor an absolute time. Template default **10 ms**; `null` bypasses.

*   🧩 **MAESTRO TRANSP boundary-surface backoff** (`separatrix.boundary_surface_psin`): the fixed TRANSP boundary can be taken at a flux surface just inside the separatrix (e.g. `0.995`), which is rounder and clears TRANSP's curvature-ratio abort while preserving the true plasma shape (unlike lowering `n_mxh`). Default `1.0` = separatrix. The boundary curve (backoff + MXH smoothing) is built by the first TRANSP beat and reused verbatim by every later one: TRANSP is fixed-boundary, so its output psiN=1 surface *is* the boundary it was given, and backing off from that again each beat would compound into a steady shrink of the plasma. Best used with `separatrix.internal_flux_file`, whose radial shaping decay makes interior surfaces physically rounder; that path now also carries its poloidal-flux mapping through the initializer's FREEGS correction step.

*   🧊 **MAESTRO 99.5% shaping re-freeze timing** (`maestro.refreeze_995_after_beat`): controls WHEN the pedestal shaping fed to EPED is fixed, orthogonal to `freeze_995_from` (which sets HOW). `0` (default) freezes at initialization; `N>0` re-extracts once from beat N's evolved equilibrium; `null` lets each EPED beat recompute from its own equilibrium. Motivated by separatrix-initialized runs, whose init 99.5% values are a parametric guess rather than a solved equilibrium.

*   🎯 **MAESTRO configurable 99.5% shaping surface** (`separatrix.shaping_extraction_psin`): the psiN at which the pedestal shaping is sampled is now a knob, orthogonal to HOW and WHEN above. Default `0.995` is bit-for-bit unchanged and the dict keys stay named "995"; raising it toward the separatrix yields higher kappa/delta. Honored by both extraction paths (geqdsk flux-surface tracer and profile interpolation).

*   🎚️ **MAESTRO TRANSP extraction-slice selector** (`extract_at`): which CDF time slice is handed to the next beat — `saw` / `saw-N` (the last sawtooth, or N coarse slices before it) or `last` / `last-N`. Default `saw-1` reproduces the historical step-back, which avoids sampling crash profiles on the coarse MAESTRO grid.

*   🩹 **MAESTRO TRANSP q-seed sanitization** (`sanitize_q_input`): an over-peaked seed (very low q0 → q=1 surface far toward the boundary) can make TRANSP's Kadomtsev model hard-exit on its first crash, before current diffusion relaxes q. If set (e.g. `0.95`), the seed q-profile is rescaled to that q0 anchored on q95 (edge/shape preserved). Seed only; `null` (default) is a no-op.

*   ⚖️ **MAESTRO Lengyel beat Zeff relaxation** (`zeff_relaxation_factor`): the core-Zeff update can now be damped toward the state the beat received instead of jumping to the divertor's demand in one shot (`[0,1]`, default `1.0` = old behavior); the blended Zeff is re-solved for the seed-impurity density. Operates on aggregate Zeff, not species identity.

*   ⏱️ **Pulse-duration calculator** (`calc_pulse_duration.calc_flattop_time`): estimates the maximum flattop duration of a `gacode_state` from a central-solenoid volt-second balance, chaining `cfspopcon`'s inductance, resistivity and flux-consumption algorithms off the state's geometry, peaking, Zeff and dilution, and printing the full breakdown. The available CS flux is given directly (`overwrite_flux`) or derived from `cs_change_in_field` + `inboard_to_CS_distance`. New teaching script `pulse_duration_01.py`.

*   🔌 **`gacode_state.recompute_targets()`**: re-derives radiation (qbrem/qsync/qline), alpha heating (qfuse/qfusi) and electron-ion exchange (qei) from the kinetic profiles with the analytic target model, on the full radial grid (no stale edge points). Now the single entry point used by the MAESTRO confinement beat and RAPIDS; `debug=True` plots each recomputed channel.

*   📊 **MAESTRO summary plots**: `Outputs/maestro_summary.md` now embeds the special-quantities and timing plots next to the beat-flow diagram, and `mitim_plot_maestro --summary` plots only those two cross-beat tabs — a fast at-a-glance view of a run.

*   🧭 **`mitim_plot_neo`**: plots NEO results from an existing folder, mirroring `mitim_plot_tglf` (`--suffixes`, `--gacode`) via a new `NEO.prep_from_file`.

*   ⚙️ **SR acquisition optimizer `halt_on`** (`best` | `all`): batched restarts halt when the *best* restart meets the tolerance (`best`, default) or only once *every* restart does (`all`) — use `all` when more than one `x_best` is consumed, so all candidates are comparably converged.

### Bug Fixes

*   🐛 **TRANSP CDF file-descriptor leak (major MAESTRO scratch consumer)**: the multi-GB output CDF was held open for the **entire** MAESTRO run and fork-inherited by every later PORTALS/TGLF worker, so once the run folder was wiped (`keep_all_files: false`) the deleted file stayed pinned as an NFS silly-rename (`.nfsXXXX`) until the case ended — 2–5 GB of extra scratch per case, overrunning quotas on large scans. The TRANSP beat now releases the cached `transp_output` at the end of `run()`, before the PORTALS beats fork.

*   🐛 **TRANSP run-abort detection & CDF-build retry**: a run that aborts during initialization (e.g. a t=0 TEQ equilibrium failure) is now flagged `stopped` rather than `finished` — a fatal `ABORTR`/`bad_exit`/segfault outranks the singularity wrapper's unconditional "Finished TRANSP run app." — so it fails fast with the real error. A failed `{runid}.CDF` build for a *completed* run falls back to a `look` rebuild, and mid-run intermediate grabs are now best-effort.

*   🩺 **Informative TRANSP failure message** (`TRANSPdebug`): a stopped run now raises with a best-effort cause parsed from the log — the signal and a plain-language gloss, the simulation time/step, and the physics event at the trap (TEQ equilibrium vs Porcelli sawtooth). Controlled Fortran aborts (`ERRSET`/`%bad_exit`, which print no signal line) are parsed too, naming the reason on the `??<reason>` line (e.g. "curvature ratio too small" in the PRGCHK boundary check); InfiniBand/RDMA launch failures are flagged as infrastructure rather than physics. **NOTE**: heuristic, not a guaranteed root cause. Usable interactively via `diagnose_transp_logfile`.

*   🐛 **MAESTRO TRANSP early-extraction floor** (`min_extraction_flattop_fraction`, default 0.5): a plasma whose only sawtooth fired early had its profiles extracted before heating / current diffusion settled; extraction is now floored at this fraction of the flattop window. Healthy runs unchanged; `null` disables.

*   🐛 **geqdsk → `input.gacode` `B_unit`** (`MITIMgeqdsk.to_profiles`): `torfluxa` was divided by an extra `2π` (megpy's `derived['phi']` is already per-radian), so `B_unit` — and with it the gyroBohm normalization and `P_PRIME_LOC`/`BETAE` (`∝ 1/B_unit²`) — came out `2π` too small. **NOTE**: this affects almost nobody, since standard `input.gacode` come from `profiles_gen`/TRANSP and MAESTRO uses `equilibrium_to_profiles`; it only bites when a `to_profiles` output is fed *directly* into a transport code.

*   🐛 **MAESTRO Lengyel beat edge-temperature bump**: when the Lengyel separatrix temperature far exceeded the incoming one (factor ~2 or more), the `rhotop`→separatrix rescaling lifted the mid-pedestal above the pedestal top (e.g. a 7.7 keV bump on a 5 keV top). The blend is now an additive quadratic offset bounded by the separatrix change; the `rhotop is None` fallback shifts by a constant instead of rescaling the whole profile.

*   🐛 **MAESTRO EPED cold-start on preemption+requeue** (`forceifcold_start`): a requeued job re-cold-started EPED, found the killed attempt's `output_run1.nc`, and hit an interactive "rerun from scratch?" prompt that raised `InteractiveTerminalError` in batch — reported upstream as "EPED failed to run". A cold-start that finds an existing output now warns and reruns from scratch. Defaulted `True` in the beat; standalone `EPED.run` keeps the prompt.

*   🐛 **PORTALS restart robustness**: a resumed run whose pkl checkpoint lagged behind `optimization_data.csv` left `x_next` empty and crashed the results writer with an `IndexError`; `MITIM_BO.updateSet` now skips the step instead.

*   🐛 **PORTALS on GPU**: numpy operations were being applied to PyTorch CUDA tensors (silently fine on CPU) — `yminymax_atleast` now uses `torch.minimum`/`torch.maximum` and `improve_resolution_profiles` coerces `rhoMODEL` to numpy first; `print_machine_info` no longer crashes on newer PyTorch. **NOTE**: not an exhaustive sweep, other instances may remain.

*   🐛 **`mitim_job.run()` retry robustness**: the retry-once-after-transient-error path re-ran the in-place relativization on already-relative paths and crashed in `relative_to()` with a misleading error; `run()` is now idempotent w.r.t. its input file/folder lists.

*   🐛 **MAESTRO per-beat logs** (`Outputs/Logs/beat_<n>_*.log`) are now line-buffered, so a long-running beat streams progress live instead of staying empty until it finishes.

*   🐛 **MAESTRO engineering scans** (`launch_scan`): the `exclude` and `qos` SLURM settings were silently dropped and are now forwarded to the array submission.

*   🐛 **`mitim_check_maestro`** now recognizes the sharpness, confinement and lengyel beats (previously shown as `UNKNOWN`).

*   🐛 **TRANSP (singularity) finish** no longer leaves a duplicate copy of the retrieved `results/` tree (notably the heavy `.CDF`).

*   🐛 **`mitim_plot_transp` boundary-less g-file**: a reference g-eqdsk with no plasma-boundary contour (`nbbbs=0`) no longer crashes the read — `getGFILE` warns, skips it and tries the next candidate extension.

*   🐛 **`mitim_plot_cgyro` timing panels** no longer clip later cases: the y-axes now expand to fit every overlaid case (`set_ylim(bottom=0)` was disabling y-autoscale).

*   🐛 **`initialization_simple_relax` folder copy** now preserves symlinks, so a transport folder containing one (e.g. a QuaLiKiz run folder) no longer breaks the copy.

### Changes for developers (internal execution)

*   🤖 **MAESTRO investigation subagent** (`.claude/agents/maestro.md`): a Claude Code agent that compares and debugs MAESTRO runs — it knows the `Beats/` layout, each beat's inputs/outputs, where the logs and namelist artifacts live, and how to overlay states headlessly. Shipped in-repo by un-ignoring `.claude/agents/`.

*   🔎 **MAESTRO EPED beat diagnostics**: reports the EPED inputs (R, a, BetaN, …) with the "no stable solution" warning, and now distinguishes a compute-node execution failure (TOQ/ELITE produced no output files) from a genuine no-solution instead of masking it behind futile teped-lowering retries.

*   🔎 **MAESTRO scan per-case logs** now symlink `slurm.out`/`slurm.err` to the live SLURM array logs instead of redirecting, so they stream live and are reachable from both the case and main folders.

*   🔧 **SLURM `exclusive` accepts a string**: `exclusive: "user"` (or `"mcs"`) emits `--exclusive=user`, keeping other users off a node while still packing the scan's own array tasks onto it. A bare `True` is unchanged; MAESTRO scan arrays now forward `slurm['exclusive']` (it was hardcoded off).

*   🧮 **E×B / parallel-velocity shear as state quantities**: `derive_quantities` now exposes `derived['gamma_exb']` and `derived['gamma_p']` in TGLF's `VEXB_SHEAR`/`VPAR_SHEAR` normalization (`c_s/a`); `to_tglf` consumes them instead of recomputing the formula inline (behavior-preserving), so the shearing rate is available on any state without a TGLF prep. `NEO.plot_vgen` / `mitim_plot_vgen` gained `mark_rho` to scatter chosen radii on the smoothed profiles.

### Back-compatibility considerations and defaults

*   💾 **Lean PORTALS pickles under `keep_all_files: false`**: intermediate beats' pickles are pruned and the retained last-beat `optimization_object.pkl` is lean. Replotting metrics still works; the GP-posterior ("Expected") plots and a pickle-based surrogate resume of those beats are not available. Set `keep_all_files: true` to retain full pickles.

*   🪚 **Sawtooth floor defaults**: `min_sawtooth_period_ms` defaults to **1 ms** in `TRANSPbeat.prepare`, so namelists without the key reproduce the historical behavior; the maestro **template** sets **10 ms**. `null` bypasses.

*   🧊 **MAESTRO `freeze_995_from` default → `analytic`**: the template now extracts the 99.5% shaping by tracing the actual psiN=0.995 flux surface and fitting the Miller-analytic shape to it, instead of `analytic_interpolation` (interpolating the per-surface Miller coefficients to psiN=0.995). Both target the same surface; the traced fit is the more direct measure. Set `analytic_interpolation` to restore the old default.

*   🔮 **PORTALS capability tests renamed** to name their turbulence model: `portals_01_standard.py` → `portals_01_tglf_standard.py`, `portals_02_multichannel_turbulent_exchange.py` → `portals_02_tglf_multichannel_turbulent_exchange.py`, joined by `portals_03_qualikiz_standard.py`. Teaching scripts only (no API change), but links to the old paths need updating.

---

*Thanks to everyone who contributed to this release: Aaron Ho, Audrey Saltzman. Portions of this release were developed with AI-assisted coding (Claude Code).*
