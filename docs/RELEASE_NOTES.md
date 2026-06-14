# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   📊 **MAESTRO summary report** (`Outputs/maestro_summary.md`) now embeds the per-beat "special quantities" evolution and the timing breakdown (`maestro_special.png`, `maestro_timing.png`) next to the existing beat-flow diagram — the same plots produced when plotting a case, now in the standalone report.

*   🎼 **`mitim_plot_maestro --summary`** (alias `--special`) plots only the cross-beat "MAESTRO special" and "MAESTRO timings" summary tabs, skipping the per-beat / profile / transition tabs — a fast at-a-glance view of a run.

*   🧭 **`mitim_plot_neo`** reads and plots NEO results from an existing folder, mirroring `mitim_plot_tglf` (positional folders, `--suffixes`, `--gacode` for normalizations) via a new `NEO.prep_from_file`.


### Bug Fixes

*   🐛 **MAESTRO engineering scans** (`launch_scan`): the `exclude` and `qos` SLURM allocation settings were silently dropped and are now forwarded to the array submission, so node exclusions actually take effect.

*   🐛 **`mitim_check_maestro`** now recognizes the sharpness, confinement and lengyel beats (previously shown as `UNKNOWN`) by their `run_<type>` folder.

### Changes for developers (internal execution)

*   🔎 **MAESTRO scan per-case logs** now symlink each case's `slurm.out`/`slurm.err` to the live SLURM array logs (`slurm_output/slurm_error_<jobid>_<task>.dat`) instead of redirecting — logs stream live and are reachable from both the case and main folders (links dangle only if a case folder is copied away on its own).

*   🔎 **MAESTRO EPED beat** failure diagnostics: reports the EPED inputs (R, a, BetaN, …) alongside the "no stable solution" warning and the final failure, and now distinguishes a compute-node execution failure (TOQ/ELITE produced no output files) from a genuine pedestal no-solution — the former is surfaced immediately as an execution error instead of being masked by futile teped-lowering retries and a misleading "no stable solution".

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
