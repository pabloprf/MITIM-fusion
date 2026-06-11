# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **NEW FEATURE**, descriptions

*   📚 **New `tests/capability_tests/` folder with standalone teaching scripts**: verbose, tutorial-like, runnable examples of MITIM capabilities, kept up to date as capabilities are added or APIs change. First entries cover PORTALS (TGLF+NEO with in-situ namelist modification), standalone TGLF runs (from input.tglf or input.gacode), TGLF scans, turbulence-drives scans, incremental-diffusivity analysis, eigenfunction waveforms, reloading results from .npz, NEO runs, the neoclassical E×B from VGEN, cheap linear/nonlinear CGYRO runs (including detached submit/check/fetch and grid preprocessing), a linear-spectrum CGYRO-vs-GX comparison, multi-channel PORTALS with trace impurity and turbulent exchange, SLURM submission of single jobs and job arrays (`run_slurm` / `run_slurm_array`), EPED pedestal prediction with scans, FreeGS equilibria from shape parameters, the generic Bayesian-optimization engine on a custom function, powertorch flux matching with an analytic transport model, TRANSP runs, and VITALS validation of TGLF against fluxes and fluctuation measurements. The `tutorials/` folder and the workflow tests fully reproduced by these scripts (OPT, PORTALS, TGLF, TGLF scans, NEO, NEO-VGEN, CGYRO, EPED, FREEGS, POWERTORCH, TRANSP, VITALS) have been removed, replaced by them.

*   ⏱️ **New "Timing" tab in the CGYRO plot notebook**, characterizing the computational cost of the run from `out.cgyro.timing`: wall time per data output, cumulative wall time (setup included), and the share of run time spent in each code section (nl, str, field, shear, coll, io, ...).


### Bug Fixes

*   🐛 **NEW BUG FIX**, description

*   🐛 **`TGLF.plotAnalysis` crashed for `analysisType='chi_i'`** (unset scan-variable label); it now plots against RLTS_2 like the cross-term analysis.

*   🐛 **`mitim_check_maestro` was blind to SLURM cancellations** (e.g. preemption on `mit_preemptable`): requeued jobs showed an innocent PENDING/RUNNING. Cancellation notices in `slurm_error.dat` are now surfaced — live requeued jobs are annotated with the cancellation time/reason, and cancelled jobs no longer in the queue are reported as definite FAILED with the reason instead of a generic timestamp.

*   🐛 **The analytic diffusion transport model of powertorch was broken** (`transport_analytic.diffusion_model` still wrote fluxes as object attributes instead of into `powerstate.plasma`, crashing `calculate()`/`flux_match()` with `KeyError: 'QeMWm2_tr'`). Now follows the current evaluator contract.

*   🐛 **CGYRO crashed at startup when the core count did not divide `N_TOROIDAL`** (e.g. nonlinear presets with `N_TOROIDAL=12` on 8 local cores: "MPI processes not a multiple of N_TOROIDAL/N_TOROIDAL_PER_PROCESS"). The automatic `TOROIDALS_PER_PROC` selection now implements CGYRO's actual constraint (ranks must be a multiple of toroidal groups), picking the smallest valid value for any core count.

### Changes for developers (internal execution)

*   🔎 **All `os.system()` calls replaced by stdlib equivalents** (`shutil`/`tarfile`/`pathlib` for file operations, `subprocess.run` for command executions): paths with special characters are now handled safely, failed local `sbatch` submissions warn instead of passing silently, and two lingering `os.chdir` side effects were removed. Also fixes `IOtools.renameCommand`, which crashed on non-mfe hosts. Pure-stdlib change, no new dependencies.

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
