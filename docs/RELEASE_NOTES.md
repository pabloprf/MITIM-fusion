# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   🌀 **QuaLiKiz interface**: new `mitim_tools.qualikiz_tools` (`QLKtools.QuaLiKiz`) runs and reads QuaLiKiz standalone from an `input.gacode`, and a matching PORTALS turbulence backend is selected with `transport.evaluator_instance_attributes.turbulence_model: "qlk"` (the neoclassical side is independent and keeps NEO). Settings follow the usual controls → `code_settings` preset (`templates/input.qualikiz.models.yaml`: `STANDARD`/`FAST`/`MINIMAL`/`ROTATION`) → `extraOptions`/`multipliers` hierarchy. All radii are packed into a *single* execution via QuaLiKiz's own `dimx` scan (one job per PORTALS iteration, not one folder per rho), and `use_scan_trick_for_stds` stacks every gradient-perturbation case onto that same execution, so flux-uncertainty estimation stays one job (TGLF needs N_rho × N_var × N_delta). Requires the external `qualikiz_tools` (QuaLiKiz-pythontools) package and a `qualikiz` entry in `config_user.json`; the import is caught, so workflows are unaffected when it is absent. **NOTE: QuaLiKiz uses a circular / s-alpha-like geometry and does not support Miller/MXH shaping, so shaped-equilibrium information is dropped in the `gacode_state.to_qualikiz` mapping — fluxes are not directly comparable to TGLF on a strongly shaped plasma. QuaLiKiz also provides no turbulent electron-ion energy exchange Qie (it is zero-filled), so `turbulent_exchange_as_surrogate` must stay `False` and the exchange is left to the analytical target model. Both are properties of QuaLiKiz itself, not of this interface.** Teaching scripts: `tests/capability_tests/qualikiz_01_run_from_inputgacode.py` (standalone) and `portals_03_qualikiz_standard.py` (PORTALS).

*   ⚛️ **EPED plasma composition (full EPED)**: the MAESTRO EPED beat and `EPEDtools.EPED.run` now feed EPED the actual plasma's main-ion mass and an effective impurity derived from the state, instead of a hardcoded 50/50 D-T + neon. The effective impurity charge reproduces both Zeff and the fuel dilution (`zi_eff = (Zeff − d)/(1 − d)`). `m`/`z`/`mi`/`zi` default to the old values (preserving EPED-NN consistency) and are overridable via the beat's `corrections_set`; a new `zeff_location` knob (`vol_avg` default, `pedestal`) sets where Zeff and the dilution are taken. The EPED-NN path is unaffected. **NOTE: in the current EPED1 build the *only* composition quantity that enters the pedestal solve is `Zeff` (via the TOQ equilibrium / bootstrap-collisionality); the `m`/`z`/`mi`/`zi` fields are passed in and recorded in the output state but are inert in the model — the KBM-width and peeling-ballooning stages carry no ion-mass or impurity-charge dependence. Scanning them at fixed `Zeff` therefore leaves the predicted pedestal unchanged (verified in `tests/dev_tests/test_eped_fuel_impurity.py`: every physics output is bit-identical across the scan, only the echoed input differs). So this change makes EPED record the true composition and honor it through `Zeff`, but it does not add an isotope/charge sensitivity that EPED1 itself does not model.**

*   🔌 **`gacode_state.recompute_targets()`**: re-derives the radiation (qbrem/qsync/qline), fusion alpha-heating (qfuse/qfusi) and electron-ion exchange (qei) power profiles from the kinetic profiles with the analytic target model, evaluated on the full radial grid (no edge points left stale). It is now the single entry point used by the MAESTRO confinement beat and RAPIDS instead of their inline powerstate round-trips; `debug=True` plots each recomputed channel against the profiles that drive it.

*   📊 **MAESTRO summary report** (`Outputs/maestro_summary.md`) now embeds the per-beat "special quantities" evolution and the timing breakdown (`maestro_special.png`, `maestro_timing.png`) next to the existing beat-flow diagram — the same plots produced when plotting a case, now in the standalone report.

*   🎼 **`mitim_plot_maestro --summary`** (alias `--special`) plots only the cross-beat "MAESTRO special" and "MAESTRO timings" summary tabs, skipping the per-beat / profile / transition tabs — a fast at-a-glance view of a run.

*   🧭 **`mitim_plot_neo`** reads and plots NEO results from an existing folder, mirroring `mitim_plot_tglf` (positional folders, `--suffixes`, `--gacode` for normalizations) via a new `NEO.prep_from_file`.

*   ⚙️ **SR acquisition optimizer (`halt_on`)**: new `optimizer_options.sr.halt_on` (`best` | `all`). The batched restarts halt together when the *best* restart meets the tolerance (`best`, default, unchanged) or only once *every* restart does (`all`) — use `all` when more than one `x_best` is consumed, so all returned candidates are comparably converged instead of the slower ones being truncated. The batched-restart behavior of both ROOT and SR (and what `relative_improvement_for_stopping` controls) is now documented in `namelist.optimization.yaml` and the solver docstrings.


### Bug Fixes

*   🐛 **PORTALS on GPU**: a few numpy operations were being applied to PyTorch CUDA tensors — these silently work on CPU tensors but raise on GPU. The `yminymax_atleast` bounds in `PORTALSinit` now use `torch.minimum`/`torch.maximum` instead of `np.min`/`np.max`, and `improve_resolution_profiles` coerces a CUDA `rhoMODEL` to numpy before its numpy-based work. Separately, `print_machine_info` no longer crashes on newer PyTorch (`props.total_mem` → `total_memory`). **NOTE: this is not an exhaustive sweep — other numpy-on-CUDA-tensor instances may well remain.**

*   🐛 **`initialization_simple_relax` folder copy** now preserves symlinks (`shutil.copytree(..., symlinks=True)`), so a transport folder containing one (e.g. a QuaLiKiz run folder) no longer breaks the copy. The link is copied as a link (dangling in the copy), which is harmless since it is never re-run from there.

*   🐛 **MAESTRO engineering scans** (`launch_scan`): the `exclude` and `qos` SLURM allocation settings were silently dropped and are now forwarded to the array submission, so node exclusions actually take effect.

*   🐛 **`mitim_check_maestro`** now recognizes the sharpness, confinement and lengyel beats (previously shown as `UNKNOWN`) by their `run_<type>` folder.

*   🐛 **TRANSP (singularity) finish** no longer leaves a duplicate copy of the retrieved `results/` tree (notably the heavy `.CDF`): its contents are surfaced into the run folder and the redundant `results/` is removed.

*   🐛 **`mitim_plot_cgyro` timing panels** no longer clip later cases: the per-output and cumulative-cost y-axes now expand to fit every overlaid case instead of freezing to the first case's range (`set_ylim(bottom=0)` was disabling y-autoscale).

*   🐛 **MAESTRO per-beat logs** (`Outputs/Logs/beat_<n>_*.log`) are now line-buffered, so a long-running beat's log (e.g. a multi-hour TRANSP run) streams progress live instead of staying empty until the beat finishes — the block buffer previously only flushed on close.

*   🐛 **PORTALS restart robustness**: a resumed optimization whose pkl checkpoint lagged behind `optimization_data.csv` could leave `x_next` empty and crash the results writer with an `IndexError` (indexing one past the end of `train_X`). `MITIM_BO.updateSet` now treats an empty `x_next` as a no-op and skips the step instead of crashing.

*   🐛 **`mitim_job.run()` retry robustness**: `run()` is now idempotent w.r.t. its input file/folder lists. The "repeat once after a transient error" retry (e.g. a code returning incomplete output) re-ran the in-place relativization on already-relative paths and crashed in `relative_to()` with a misleading `'mitim_bash.src' is not in the subpath…` error; it now re-runs cleanly and surfaces the real failure.

*   🐛 **TRANSP run-abort detection & CDF-build retry**: a TRANSP run that aborts during initialization (e.g. a t=0 TEQ equilibrium failure) is now flagged as `stopped` instead of `finished` — the singularity wrapper's unconditional `Finished TRANSP run app.` line no longer outranks a fatal `ABORTR`/`bad_exit`/segfault in the log, so the run fails fast with the real error in the log tail instead of proceeding to a confusing missing-CDF / failed-`look` prompt. When the finish step's `trlook`/`plotcon` does fail to build `{runid}.CDF` for a *completed* run (e.g. a transient "TF.PLN file not found" abort), `TRANSPsingularity.fetch` falls back to a `look` rebuild that re-stages the `.PLN` files from the remote run folder (`job.folderExecution`, not the local run directory where they never live) and re-runs `plotcon`, instead of hard-failing downstream on the missing CDF. Separately, the mid-run intermediate grabs in `checkUntilFinished` (the periodic monitor and the stopped-run cleanup) are now best-effort: a failed intermediate `look`/CDF is logged and skipped instead of killing a healthy run on a transient miss, or masking the real `stopped` error behind an `InteractiveTerminalError` in batch.

### Changes for developers (internal execution)

*   🤖 **MAESTRO investigation subagent** (`.claude/agents/maestro.md`): a Claude Code agent that forensically compares and debugs MAESTRO runs — it knows the `Beats/` layout, each beat's inputs/outputs, where the logs/timing/namelist artifacts live, and how to load and overlay states headlessly. Shipped in-repo by un-ignoring `.claude/agents/` (the rest of `.claude/` stays local).

*   🔎 **MAESTRO scan per-case logs** now symlink each case's `slurm.out`/`slurm.err` to the live SLURM array logs (`slurm_output/slurm_error_<jobid>_<task>.dat`) instead of redirecting — logs stream live and are reachable from both the case and main folders (links dangle only if a case folder is copied away on its own).

*   🔎 **MAESTRO EPED beat** failure diagnostics: reports the EPED inputs (R, a, BetaN, …) alongside the "no stable solution" warning and the final failure, and now distinguishes a compute-node execution failure (TOQ/ELITE produced no output files) from a genuine pedestal no-solution — the former is surfaced immediately as an execution error instead of being masked by futile teped-lowering retries and a misleading "no stable solution".

*   🔧 **SLURM `exclusive` accepts a string**: setting `exclusive: "user"` (or `"mcs"`) now emits `#SBATCH --exclusive=user` instead of plain `--exclusive`, so a scan can keep nodes free of *other* users while still packing the user's own array tasks onto each node — node isolation for large scans without the one-task-per-node core waste. A bare `True` is unchanged (plain `--exclusive`). MAESTRO scan arrays (`_submit_array`) now forward `slurm['exclusive']` (it was hardcoded off), so `exclusive="user"` set in a scan launcher actually takes effect.

*   🌀 **PORTALS neoclassical E×B shear** now has a standalone dev test (`tests/dev_tests/test_portals_exb_shear.py`): it flux-matches the same plasma with `transport.options.neo.vgen_exb_shear` off vs on (the NEO-VGEN neoclassical Er at zero toroidal rotation) to isolate the stabilization, and bundles the comparison, the per-run PORTALS metrics and the VGEN notebook into one figure. Alongside it, `NEO.plot_vgen` (and `mitim_plot_vgen`) gained an optional `mark_rho` to scatter chosen radii — e.g. the PORTALS predicted radii — on the smoothed profiles in the VGEN smoothing tab.

### Back-compatibility considerations and defaults

*   🔮 **PORTALS capability tests renamed** to name their turbulence model, now that it is a real choice: `portals_01_standard.py` → `portals_01_tglf_standard.py` and `portals_02_multichannel_turbulent_exchange.py` → `portals_02_tglf_multichannel_turbulent_exchange.py`, joined by the new `portals_03_qualikiz_standard.py`. Only the teaching scripts moved (no API change), but any bookmark or doc link pointing at the old paths needs updating.

---

*Thanks to everyone who contributed to this release: Aaron Ho. Portions of this release were developed with AI-assisted coding (Claude Code).*
