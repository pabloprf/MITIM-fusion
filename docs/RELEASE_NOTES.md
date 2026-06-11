# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   💥 **NEW FEATURE**, descriptions

*   📚 **New `tests/capability_tests/` folder with standalone teaching scripts**: tutorial-like, runnable examples of MITIM capabilities, kept up to date as capabilities are added or APIs change. First entries: `portals_standard.py` (PORTALS with TGLF+NEO, in-situ namelist modification), `tglf_run.py` (standalone TGLF, `code_settings` presets vs `extraOptions`), and `tglf_scan.py` (TGLF parameter scans). Expected to eventually replace the `tutorials/` folder.


### Bug Fixes

*   🐛 **NEW BUG FIX**, description

### Changes for developers (internal execution)

*   🔎 **All `os.system()` calls replaced by stdlib equivalents** (`shutil`/`tarfile`/`pathlib` for file operations, `subprocess.run` for command executions): paths with special characters are now handled safely, failed local `sbatch` submissions warn instead of passing silently, and two lingering `os.chdir` side effects were removed. Also fixes `IOtools.renameCommand`, which crashed on non-mfe hosts. Pure-stdlib change, no new dependencies.

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
