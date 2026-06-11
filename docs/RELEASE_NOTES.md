# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **NEW FEATURE**, descriptions

*   🩹 **MAESTRO EPED beat can now retry instead of dying on "EPED failed to find any stable solution"**: new namelist knobs `teped_retries` (default 0, old behavior) and `teped_retry_lower_factor` (default 0.7) re-run EPED with the floor of the explored pedestal-temperature window (`TEPED_BOUND`) lowered relative to the original per attempt.

*   ⏱️ **New "Timing" tab in the CGYRO plot notebook**, characterizing the computational cost of the run from `out.cgyro.timing`: wall time per data output, cumulative wall time (setup included), and the share of run time spent in each code section (nl, str, field, shear, coll, io, ...).


### Bug Fixes

*   🐛 **NEW BUG FIX**, description

*   🐛 **`mitim_check_maestro` was blind to SLURM cancellations** (e.g. preemption on `mit_preemptable`): requeued jobs showed an innocent PENDING/RUNNING. Cancellation notices in `slurm_error.dat` are now surfaced — live requeued jobs are annotated with the cancellation time/reason, and cancelled jobs no longer in the queue are reported as definite FAILED with the reason instead of a generic timestamp.

*   🐛 **Plotting in-process TGLF results crashed** (`AttributeError` on `scalar_sat_params`, then `IndexError` in the fluctuation spectra): `TGLFoutput.from_inprocess` was missing attributes added later to the file-reading path, and its placeholder spectral arrays were sized 1 along the species/ion axes while plot loops iterate the actual species counts. All placeholders now carry consistent dimensions; verified by running the full standard-vs-in-process comparison and notebook build end-to-end.

*   🐛 **The analytic diffusion transport model of powertorch was broken** (`transport_analytic.diffusion_model` still wrote fluxes as object attributes instead of into `powerstate.plasma`, crashing `calculate()`/`flux_match()` with `KeyError: 'QeMWm2_tr'`). Now follows the current evaluator contract.

*   🐛 **CGYRO crashed at startup when the core count did not divide `N_TOROIDAL`** (e.g. nonlinear presets with `N_TOROIDAL=12` on 8 local cores: "MPI processes not a multiple of N_TOROIDAL/N_TOROIDAL_PER_PROCESS"). The automatic `TOROIDALS_PER_PROC` selection now implements CGYRO's actual constraint (ranks must be a multiple of toroidal groups), picking the smallest valid value for any core count.

### Changes for developers (internal execution)

*   🔎 **All MITIM-generated sbatch files now request `--requeue`**: on preemption (e.g. preemptable partitions) or node failure, SLURM puts the job back in the queue under the same id instead of killing it, and MITIM workflows resume from their on-disk checkpoints when re-executed. Opt out per job with `slurm_settings={'requeue': False}` (emits `--no-requeue`) or `None` (cluster default).

*   🔎 **All `os.system()` calls replaced by stdlib equivalents** (`shutil`/`tarfile`/`pathlib` for file operations, `subprocess.run` for command executions): paths with special characters are now handled safely, failed local `sbatch` submissions warn instead of passing silently, and two lingering `os.chdir` side effects were removed. Also fixes `IOtools.renameCommand`, which crashed on non-mfe hosts. Pure-stdlib change, no new dependencies.

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
