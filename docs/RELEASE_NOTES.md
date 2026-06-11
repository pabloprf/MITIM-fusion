# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **NEW FEATURE**, descriptions

*   📚 **New `tests/capability_tests/` folder with standalone teaching scripts**: verbose, tutorial-like, runnable examples of MITIM capabilities, kept up to date as capabilities are added or APIs change. First entries cover PORTALS (TGLF+NEO with in-situ namelist modification), standalone TGLF runs (from input.tglf or input.gacode), TGLF scans, turbulence-drives scans, incremental-diffusivity analysis, eigenfunction waveforms, reloading results from .npz, NEO runs, the neoclassical E×B from VGEN, and cheap linear/nonlinear CGYRO runs. The outdated `tutorials/TGLF_tutorial.py` and `tutorials/PORTALS_tutorial.py` have been removed, replaced by these scripts.


### Bug Fixes

*   🐛 **NEW BUG FIX**, description

*   🐛 **`TGLF.plotAnalysis` crashed for `analysisType='chi_i'`** (unset scan-variable label); it now plots against RLTS_2 like the cross-term analysis.

### Changes for developers (internal execution)

*   🔎 **All `os.system()` calls replaced by stdlib equivalents** (`shutil`/`tarfile`/`pathlib` for file operations, `subprocess.run` for command executions): paths with special characters are now handled safely, failed local `sbatch` submissions warn instead of passing silently, and two lingering `os.chdir` side effects were removed. Also fixes `IOtools.renameCommand`, which crashed on non-mfe hosts. Pure-stdlib change, no new dependencies.

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
