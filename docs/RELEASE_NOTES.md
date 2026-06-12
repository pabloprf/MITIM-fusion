# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **NEW FEATURE**, descriptions

*   📚 **New `tests/capability_tests/` folder with standalone teaching scripts**: verbose, tutorial-like, runnable examples of every major MITIM capability — all wrapped codes (TGLF including scans, turbulence drives, waveforms, incremental-diffusivity analyses (electron/ion/cross), trace-impurity D/V, from-TRANSP-CDF, in-process and multi-plasma parallel submissions; NEO including VGEN E×B and in-process; CGYRO including detached submit/check/fetch, grid preprocessing, warm-start restart chaining and SLURM job arrays over radii; GX; TGYRO; TRANSP; EPED; FreeGS; Lengyel), the PORTALS (standard and multi-channel) and MAESTRO workflows, VITALS, powertorch flux matching, the generic Bayesian-optimization engine, a CGYRO-vs-GX linear comparison, and SLURM job/array submission. They replace the `tutorials/` folder and ALL the legacy `tests/*_workflow.py` smoke tests (both removed), and are kept up to date as capabilities are added or APIs change.

*   🩹 **Full-EPED beats no longer die on the 30-minute SLURM default**: the MAESTRO EPED beat now passes a `minutes_slurm` knob (default 240, documented in the template namelist) to each EPED case — the `EPEDtools` default of 30 min was far too short for the full width x teped stability sweep and jobs were cancelled on the time limit. Also, single-case EPED submissions no longer carry the confusing `--array=1%N` concurrency suffix.

*   🩹 **MAESTRO EPED beat can now retry instead of dying on "EPED failed to find any stable solution"**: new namelist knobs `teped_retries` (default 2; set 0 for the old fail-immediately behavior) and `teped_retry_lower_factor` (default 0.7) re-run EPED with the floor of the explored pedestal-temperature window (`TEPED_BOUND`) lowered relative to the original per attempt.

*   ⏱️ **New "Timing" tab in the CGYRO plot notebook**, characterizing the computational cost of the run from `out.cgyro.timing`: wall time per data output, cumulative wall time (setup included), and the share of run time spent in each code section (nl, str, field, shear, coll, io, ...).


### Bug Fixes

*   🐛 **NEW BUG FIX**, description

*   🐛 **`TGLF.plotAnalysis` crashed for `analysisType='chi_i'`** (unset scan-variable label); it now plots against RLTS_2 like the cross-term analysis.

*   🐛 **The TRANSP trdat/pretr prep stage now runs as its own minimal SLURM job** (1 task, 1 cpu, 10 min, non-exclusive) instead of executing the singularity container front-end: login nodes may not allow the user namespaces singularity needs (e.g. engaging), which forced node-targeted machine configs. On machines without SLURM the behavior is unchanged.

*   🐛 **TRANSP-singularity runs no longer hang until the job time limit when the container fails to launch** (e.g. nodes whose OS image disables unprivileged user namespaces and lacks `apptainer-suid`, seen on some `mit_preemptable` nodes): the namespace-creation error is now caught at the trdat prep step (raising immediately) and in the run-status checker (flagging the run as stopped), so the failure surfaces within minutes instead of burning the full allocation.

*   🐛 **Hardened the initializer's BetaN and ne-peaking matching against numerical path-sensitivity.** The parameterization/eped/fixed_bc profile creators matched BetaN (via aLTi) and ne peaking (via aLn) with Nelder-Mead at `tol=1e-3`, which accepts up to ~3% target error and stalls unpredictably near the aLTi bound — bit-level FP differences (e.g. EPED-NN inference on different cluster nodes) could shift the starting plasma's BetaN by ~5%, seeding run-to-run divergence of full MAESTRO chains from identical inputs. Both matches are monotonic in their gradient knob, so they now use bracketed root finding (`brentq`) on the signed mismatch — deterministic and exact — and an unreachable target saturates at the closest bound with an explicit warning instead of silently stalling.

*   🐛 **`mitim_check_maestro` was blind to SLURM cancellations** (e.g. preemption on `mit_preemptable`): requeued jobs showed an innocent PENDING/RUNNING. Cancellation notices in `slurm_error.dat` are now surfaced — live requeued jobs are annotated with the cancellation time/reason, and cancelled jobs no longer in the queue are reported as definite FAILED with the reason instead of a generic timestamp. Cases whose recorded job id went stale (e.g. manual resubmissions) are now also recognized via their deterministic job name (`mitim_<folder>`) instead of being reported as failed.

*   🐛 **Crashes with modern SciPy/NumPy in integration helpers**: `scipy.integrate.cumtrapz` (removed in SciPy 1.14) and `np.trapz` (removed in NumPy 2.0) were still used in `PLASMAtools.chi_inc`, `MATHtools` and `PROFILEStools` — breaking e.g. the TGLF incremental-diffusivity analysis. All call sites migrated to `cumulative_trapezoid`/`np.trapezoid`.

*   🐛 **Standalone CGYRO runs never returned their `bin.cgyro.restart` files**: the stale-warm-start cleanup in the execution script compared the restart's mtime against `out.cgyro.info`, which CGYRO finalizes AFTER writing the restart — so every fresh restart was deleted before retrieval. The baseline is now a marker file touched at run start. (PORTALS warm-start chaining was mostly unaffected because wall-clock-killed runs never append the EXIT line.)

*   🐛 **Plotting in-process TGLF results crashed** (`AttributeError` on `scalar_sat_params`, then `IndexError` in the fluctuation spectra): `TGLFoutput.from_inprocess` was missing attributes added later to the file-reading path, and its placeholder spectral arrays were sized 1 along the species/ion axes while plot loops iterate the actual species counts. All placeholders now carry consistent dimensions; verified by running the full standard-vs-in-process comparison and notebook build end-to-end.

*   🐛 **The analytic diffusion transport model of powertorch was broken** (`transport_analytic.diffusion_model` still wrote fluxes as object attributes instead of into `powerstate.plasma`, crashing `calculate()`/`flux_match()` with `KeyError: 'QeMWm2_tr'`). Now follows the current evaluator contract.

*   🐛 **CGYRO crashed at startup when the core count did not divide `N_TOROIDAL`** (e.g. nonlinear presets with `N_TOROIDAL=12` on 8 local cores: "MPI processes not a multiple of N_TOROIDAL/N_TOROIDAL_PER_PROCESS"). The automatic `TOROIDALS_PER_PROC` selection now implements CGYRO's actual constraint (ranks must be a multiple of toroidal groups), picking the smallest valid value for any core count.

### Changes for developers (internal execution)

*   🖥️ **Notebook/plotting UX**: FigureNotebooks always open on the FIRST tab (script-built notebooks could open on an arbitrary mid-list tab after internal tab reordering, e.g. PORTALS `plot_optimization_results`); a long tab bar now has jump-to-first/last corner buttons next to the scroll arrows; and MAESTRO plots place the "MAESTRO special" and "MAESTRO timings" summary tabs first, as the default view (both from `m.plot()` and `mitim_plot_maestro`).

*   📈 **TGLF plotting improvements**: waveform tabs no longer clip the evaluated-ky markers out of the eigenvalue axes (relevant when `forceClosestUnstableWF` lands far from the requested ky); 1D and 2D scans now also show the momentum flux and turbulent exchange (and the selected ion particle flux in 2D); `plotAnalysis` can compose several analyses into one notebook (`fn`/`fn_color`/`extratitle`) and skip per-point notebooks (`plotTGLFs`).

*   🔒 **`run_slurm`/`run_slurm_array` gained lock-file protection and log-append mode**: `lock_file=True` makes the sbatch script create a `job.lock` in the run folder and abort if a recent one exists (timeout via `lock_file_timeout_hours`, default 12h), preventing redundant submissions of the same case from racing on the same files; `append_mode=True` adds `--open-mode=append` so re-launched jobs append to `slurm_output/error.dat` instead of overwriting them (preserving e.g. preemption notices across requeues).

*   🔎 **New `lengyel` optional-dependencies group** (`pip install mitim-fusion[lengyel]`): installs `extended-lengyel` and `radas`, required by the Lengyel divertor/SOL model wrapper (same pattern as the `vmec` extra).

*   🔎 **All MITIM-generated sbatch files now request `--requeue`**: on preemption (e.g. preemptable partitions) or node failure, SLURM puts the job back in the queue under the same id instead of killing it, and MITIM workflows resume from their on-disk checkpoints when re-executed. Opt out per job with `slurm_settings={'requeue': False}` (emits `--no-requeue`) or `None` (cluster default).

*   🔎 **All `os.system()` calls replaced by stdlib equivalents** (`shutil`/`tarfile`/`pathlib` for file operations, `subprocess.run` for command executions): paths with special characters are now handled safely, failed local `sbatch` submissions warn instead of passing silently, and two lingering `os.chdir` side effects were removed. Also fixes `IOtools.renameCommand`, which crashed on non-mfe hosts. Pure-stdlib change, no new dependencies.

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

*   🔮 **The MAESTRO template namelist now defaults to full EPED** (`use_full_EPED: true`): the EPED-NN entries (`nn_location`, `norm_location`, `transform_inputs_fun`) default to null and the NN-specific `corrections_set` was removed from the template (both documented as comments for users who switch back to the NN surrogate).

---

*Thanks to everyone who contributed to this release: @AudreySaltzman. Portions of this release were developed with AI-assisted coding (Claude Code).*
