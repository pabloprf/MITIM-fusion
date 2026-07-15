# MITIM-fusion — Orientation for Claude Code

> This is the repo-level briefing. Pablo's global preferences (`~/.claude/CLAUDE.md`) and the dev-pixi workspace file (`dev-pixi/CLAUDE.md`, if present) still apply — this file adds repo-specific context on top.

MITIM (MIT Integrated Modeling) is a Python toolbox for plasma physics / fusion-energy
modeling and optimization, developed at the MIT Plasma Science and Fusion Center
by Pablo Rodriguez-Fernandez (pablorf@mit.edu) and the MFE-IM group. It wraps
external transport codes (TGLF, NEO, CGYRO, GX, TGYRO, TRANSP, EPED, FREEGS,
VMEC, ASTRA, …) behind a single object-oriented API and adds two flagship
optimization workflows on top: **PORTALS** (surrogate-based steady-state
profile prediction) and **MAESTRO** (multi-step integrated modeling pipeline
that chains those workflows). This file is the briefing for future Claude
sessions working in this repo.

Reference papers:
- PORTALS: Rodriguez-Fernandez et al., Nucl. Fusion **64** 076034 (2024).
- PORTALS-CGYRO: Rodriguez-Fernandez, Howard, Candy, Nucl. Fusion **62** 076036 (2022).

Public docs: https://mitim-fusion.readthedocs.io

---

## 0. Repository layout

```
src/
  mitim_tools/             # standalone interfaces to external codes + utilities
    gacode_tools/          # TGLF, NEO, CGYRO, TGYRO, profiles_gen wrappers
    transp_tools/          # TRANSP wrapper
    eped_tools/            # EPED + EPED-NN
    gs_tools/              # FREEGS / VMEC equilibrium tools
    opt_tools/             # generic Bayesian-optimization framework (MITIM_BO)
    simulation_tools/      # SLURM / SSH / mitim_job submission, in-process libs
    plasmastate_tools/     # mitim_state object
    surrogate_tools/       # NN / GP utilities (e.g. EPED-NN)
    misc_tools/            # FARMINGtools, IOtools, LOGtools, GUItools, GRAPHICStools
  mitim_modules/           # high-level workflows built on mitim_tools
    portals/               # PORTALS surrogate-based BO loop
    maestro/               # MAESTRO multi-beat orchestrator
    powertorch/            # transport-evaluator dispatch + STATEtools
    vitals/                # VITALS validation workflow
    freegsu/               # FreeGS-based equilibrium optimization
templates/                 # YAML namelists + config_user_example.json
tests/                     # unit tests + capability_tests/ teaching scripts
tests/capability_tests/    # standalone teaching scripts (see §9; replaced tutorials/)
docs/                      # Sphinx sources for readthedocs
```

CLI entry points are declared in `pyproject.toml` under `[project.scripts]`. The
ones you will use most:

| Command | Purpose |
|---|---|
| `mitim_run_portals <folder>` | Launch a PORTALS run from `namelist.portals.yaml` + `input.gacode` |
| `mitim_plot_portals <folder> [--complete] [--remote <m>]` | Read+plot PORTALS results |
| `mitim_run_maestro <folder> --namelist namelist.maestro.yaml` | Launch a MAESTRO multi-beat run |
| `mitim_plot_maestro <folder> [--beats N] [--only transp]` | Plot MAESTRO results |
| `mitim_check_maestro` | Inspect MAESTRO state / progress |
| `mitim_plot_gacode / _tglf / _neo / _cgyro / _gx / _eq / _eped / _transp / _vgen` | Read+plot per-code outputs |
| `mitim_run_tglf` / `mitim_run_transp` | Run a single code instance |
| `mitim_slurm` | Submit a wrapper job to SLURM |
| `mitim_scp` | Pull files/folders from a configured remote |
| `mitim_compare_nml` | Diff two namelists |

---

## 1. Capability tests

You can learn about how to run the different MITIM capabilities by exploring and
reproducing what's in `tests/capability_tests/`, a subfolder that contains tons of
well-explained examples of the main capabilities.

## 2. User config (`config_user.json`) — required for any non-local run

MITIM dispatches per-code work to whichever machine you select. Configuration
lives in `templates/config_user.json` (or the path in env `MITIM_CONFIG`, or
the path passed to `from mitim_tools import config_manager; config_manager.set(path)`).

`templates/config_user_example.json` shows the structure. The two pieces:

- `"preferences"`: maps each code (`tglf`, `cgyro`, `gx`, `neo`, `transp`, `eped`,
  `tgyro`, `eq`, `astra`, `profiles_gen`, …) to a machine name. `verbose_level` 0–5
  controls log noise; `dpi_notebook` scales matplotlib figures.
- One block per machine name: `machine`, `username`, `scratch`, `modules` (string
  sourced before each run, e.g. `export GACODE_ROOT=…; . ${GACODE_ROOT}/shared/bin/gacode_setup`),
  `cores_per_node`, `gpus_per_node`, optional `slurm: {partition, account, exclusive,
  exclude, mem, constraint, email}`, optional `identity` (SSH key path).

Everything ssh/sftp goes through `mitim_tools.misc_tools.FARMINGtools` and the
`mitim_job` abstraction in `mitim_tools.simulation_tools.SIMtools`. Long
PORTALS-CGYRO runs survive overnight VPN drops via per-job `ssh_retry_attempts`
(set `null` to retry forever).

---

## 3. PORTALS — the BO transport-prediction workflow

### 3.1 What it does

PORTALS solves the steady-state core transport problem (find `(a/LTe, a/LTi, a/Lne, …)`
profiles such that turbulent + neoclassical fluxes match the local target sources,
channel by channel and radius by radius) by treating each transport-code call as
an expensive black-box evaluation and fitting Gaussian-process surrogates over
"physics-informed" inputs. A botorch acquisition picks the next candidate; an
inner classical solver (`sr` = simple-relax conjugate-thermo, `root` = scipy
Levenberg-Marquardt) refines that candidate; the loop runs until residuals
converge or `maximum_iterations` is hit.

### 3.2 Inputs you provide

For a typical run:
- An `input.gacode` file (initial profiles + equilibrium, GACODE convention).
- A `namelist.portals.yaml` (copy `templates/namelist.portals.yaml` and edit).

The defaults in `templates/namelist.portals.yaml` are heavily commented and are
the **single source of truth** for what each knob does. Keep that file authoritative;
do not duplicate per-knob docs in CLAUDE.md.

Top-level sections of the PORTALS namelist:
- `solution`: radial grid (`predicted_rho` or `predicted_roa`), channels
  (`["te","ti","ne","nZ","w0"]` subset), exploration ranges (`ymin`, `ymax`,
  `yminymax_atleast`, `enforce_finite_aLT`), trace-impurity name, scalar
  multipliers, `portals_transformation_variables` (the physics-informed inputs
  fitted by the surrogate, switched at iteration thresholds).
- `transport`: which evaluator class, `evaluator_instance_attributes.{turbulence_model, neoclassical_model}`,
  per-backend `options.{tglf,neo,cgyro,gx}` blocks, `applyCorrections`
  (Ti-thermals, ni-thermals, recompute Ptot, …), `flatten_gradients_at_control_points`,
  `profiles_postprocessing_fun` (global) plus per-backend `profiles_postprocessing_fun` overrides.
- `target`: target evaluator (`analytical_model` by default), evolved targets
  (`["qie","qrad","qfus"]`), `targets_resolution`, `force_zero_particle_flux`.
- `optimization_options`: overrides on top of `templates/namelist.optimization.yaml` —
  `initialization_options.initial_training` (default 5 SR points),
  `convergence_options.{maximum_iterations, stopping_criteria_parameters.maximum_value, …}`,
  `acquisition_options.{type, optimizers}` (`optimizers: ["sr","root","botorch"]`
  applied sequentially), `surrogate_options`, `strategy_options.AllowedExcursions`.

#### Multi-fidelity (`turbulence_model` as int-keyed dict)

```yaml
turbulence_model:
  0: 'tglf'
  1: 'cgyro'
```

appends a reserved DV `fidelity_level` ∈ `[0, N-1]`. `botorch` and `ga` handle
it natively. `sr` and `root` cannot (they iterate `Δx_i ∝ residual_i` and need
`len(DVs)==len(residuals)`); the framework auto-strips it for those stages and
pins `fidelity_level=N-1` (highest fidelity) — see
`OPTtools._METHODS_HANDLE_FIDELITY_LEVEL` and the docstring at
`PORTALSmain.py:89-99`.

#### CGYRO restart machinery

`transport.options.cgyro.run.restart_from_cases` controls warm-start chaining
across BO iterations: `null` (cold), `"first"` (always reuse iter 0's restart),
`"all"` (iter N restarts from N-1), `"best"` (per-rho pick the prior iteration
with the closest L2 turbulent flux to the current target). `bin.cgyro.restart`
is staged into each rho subfolder; `out.cgyro.tag` is deliberately NOT staged
(would trigger restart_flag=1 / io_control=3 = true rewind, ill-defined when
input.cgyro changes per iteration). `MAX_TIME` on warm-starts is *additional*
time on top of the saved state, not cumulative. `restart_sources.json` records
which source iter each rho used; it is persisted in `cgyro_submission.json`
metadata so a kill+reattach restores it.

`run_type` options for CGYRO: `prep` (build only), `send` (stage to remote),
`submit` (sbatch), `normal` (send+submit+wait). Auto-resubmit on stalled rhos
is on by default (`auto_resubmit_enabled: True`, `stall_*_kill_seconds: 1800`,
`max_resubmits_per_rho: 1`).

### 3.3 Launching a run

CLI (recommended):

```bash
cd <workdir-with-input.gacode-and-namelist.portals.yaml>
mitim_run_portals .                      # uses ./namelist.portals.yaml, ./input.gacode
mitim_run_portals myrun --namelist mynl.yaml --input my.gacode --cold   # explicit
mitim_run_portals myrun --batch          # non-interactive (CI / SLURM)
mitim_run_portals myrun --no-log-file    # don't redirect stdout to Outputs/optimization_log.txt
```

Programmatic (`tests/capability_tests/portals_01_tglf_standard.py` is the canonical example):

```python
from mitim_modules.portals import PORTALSmain
from mitim_tools.opt_tools import STRATEGYtools

portals_fun = PORTALSmain.portals(folder, portals_namelist=path)   # path optional
portals_fun.portals_parameters["solution"]["predicted_rho"] = [0.35, 0.55, 0.75]
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 10
portals_fun.prep(input_gacode_path)
mitim_bo = STRATEGYtools.MITIM_BO(portals_fun)
mitim_bo.run()
```

Note: edit `portals_parameters` and `optimization_options` BEFORE calling
`prep()`. `prep()` snapshots the namelist into the run folder; later edits are
ignored.

### 3.4 Folder layout produced

```
<run-folder>/
  Outputs/
    optimization_log.txt        # main log (unless --no-log-file)
    optimization_data.csv       # all evaluations, one row per iter
    optimization_extra.pkl      # large extras (powerstates, GP blobs, …)
    optimization_object.pkl     # the MITIM_BO object (read by analyzer)
    optimization_results.out    # human-readable summary
    timing.jsonl                # per-iter wallclock
    portals_profiles/           # input.gacode.<iter> snapshots
  Initialization/
    initialization_simple_relax/portals_sr_ev_<i>/  # SR seed evaluations
  Execution/
    Evaluation.<iter>/transport_simulation_folder/{base_tglf,base_cgyro,base_neo}/
                                # per-iteration model run trees (kept per keep_files)
```

### 3.5 Reading + plotting results

```bash
mitim_plot_portals <folder>                         # metrics-only (single fig)
mitim_plot_portals <folder> --complete              # full notebook (multi-tab GUI)
mitim_plot_portals <folder> --max -1                # bound y-axes by last iter
mitim_plot_portals <folder> --indeces_extra 5 10    # also annotate iters 5 and 10
mitim_plot_portals <folder> --remote engaging       # rsync the folder back first
mitim_plot_portals <folder> --remote engaging --remote_minimal   # bring only key files
mitim_plot_portals <folder> --save                  # write figs to <folder>/figures_plotting_save
```

Programmatic:

```python
from mitim_modules.portals.utils import PORTALSanalysis
pa = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
pa.plotMetrics(...)        # single-figure metrics
pa.plotPORTALS(...)        # full FigureNotebook with all tabs
```

### 3.6 Interpreting common results

- **Residual** (Outputs/optimization_results.out, "Sum of residuals" column):
  channel-summed |Q_target − Q_predicted|/Q_target across rhos. Convergence is
  declared when this drops by `maximum_value` (default `5e-3` ⇒ 200× reduction
  from iter 0) OR Ricci metric < `ricci_value` (0.05 default) OR DVs stop
  varying for `minimum_inputs_variation` consecutive iters.
- **PORTALS Metrics tab**: residual vs iter (log), DV trajectories, profile
  evolution. `i0` = first iter, `ibest` = best-residual iter, `iextra` =
  user-marked.
- **PORTALS Expected tab**: GP posterior mean and std on the next candidate.
- **Per-channel/per-radius CGYRO trace tabs** (`--complete`): time series of
  Q_GB, Γ_GB, Π_GB at each rho with restart-aware time appending (so a
  warm-started run shows continuous time from the parent iter, even though
  CGYRO internally reset t to 0).
- **fluxes_turb.json / fluxes_neoc.json**: per-iteration JSON blobs with the
  GB-normalized turbulent and neoclassical fluxes per channel/rho. These are
  what `restart_from_cases: "best"` uses to pick parents.

---

## 4. MAESTRO — chained whole-discharge integrated modeling

### 4.1 What it does

MAESTRO orchestrates a sequence of "beats", each running one external code or
PORTALS instance, and feeds the output of one beat into the next. Typical
chain (matches `templates/namelist.maestro.yaml`):

```
beats: ["transp", "eped", "portals", "eped", "portals"]
```

= *spin up TRANSP equilibrium → solve pedestal with EPED → run PORTALS core
prediction → re-solve EPED with the new pressure → re-run PORTALS to
self-consistency*. Other beat types: `transp_soft`, `transp_final`,
`portals_soft`, `eped_initializer`, `lengyel`, `sharpness`, **`confinement`
(sets the temperature BC by minimizing over T_bc until a target H-factor —
`H98y2` or `H89p` — is matched; sources stay frozen during the scan)**. Each beat defines
`prepare()`, `run()`, `interpret()`, and merges its output into a frozen
`profiles_with_engineering_parameters` that the next beat starts from.

Engineering inputs that stay fixed across beats (R, a, Bt, Ip, separatrix
shape, total heating power, edge density, Zeff, fuel mix) are "frozen" at
initialization. Per-beat physics state (ψ, profiles, fast-ion populations) is
updated and passed forward via `parameters_trans_beat`.

### 4.2 Inputs

- `namelist.maestro.yaml` (copy `templates/namelist.maestro.yaml`).

Top-level sections:
- `seed`: master RNG seed.
- `plasma.profiles_initialization`: how to bootstrap the first beat
  (`initialization_type` ∈ `freegs|fibe|geqdsk|separatrix|profiles`,
  `creator_type` ∈ `eped_initializer|fixed_profiles|fixed_bc|null`).
- `plasma.parameters`: `Bt`, `Ip`, `neped_20` (or `fGped`), `ne_ratio_sep_ped`,
  `Tesep_eV`, `separatrix.{R,a,delta_sep,kappa_sep,zeta_sep,n_mxh, geqdsk_file,
  rz_boundary_file, internal_flux_file, freeze_995_from}`.
- `plasma.species`: `fuel: ['D','T']` or `['D']`, `Zeff`, `mix.{fmain, highZ,
  fhighZ, CShighZ_estimate}` (used to lump low-Z impurities to match Zeff).
- `plasma.heating.{type, parameters}` — `ICRH | NBI | gaussian_sources` and
  the corresponding params (`P_icrh`, `minority`, `fmini`, `freq_ICH`, `P_nbi`,
  `Pe`, `Pi`, `nu_source`).
- `maestro.beats: [...]` — the ordered chain.
- `maestro.<beat_name>`: per-beat config. Every beat has:
  - `beat_type`: maps to a beat class (`transp`, `eped`, `portals`, `lengyel`,
    `sharpness`, `eped_initializer`).
  - `base_module`: optional pointer to another beat's config to inherit from
    (`portals_soft` inherits from `portals` and overrides only the relaxed
    knobs).
  - `parameters_prepare`: kwargs forwarded to `beat.prepare()`.
  - `preprocess_prepare` (optional callable, `import::module.fn`): mutates the
    prepare-namelist using the rest of the maestro namelist.
  - `preprocess_prepare_parameters`: extra args to that callable.
  - `preprocess_run`: same idea, for the `run()` kwargs.

For PORTALS beats, `parameters_prepare.portals_parameters` is a partial
overlay on top of the PORTALS template namelist — only specify the keys you
want to override (the maestro template shows several typical overrides:
`predicted_roa`, TGLF `code_settings: SAT2astra`, `force_zero_particle_flux:
true`, `optimizers: ["sr"]`, `AllowedExcursions: [0.25, 0.0]`, etc.). PORTALS
beats also have their own knobs:
`thermalize_fast`, `quasineutrality`, `change_last_radial_call`,
`use_previous_residual`, `use_previous_surrogate_data`, `use_previous_ranges`,
`try_flux_match_only_for_first_point`, `enforce_impurity_radiation_existence`,
plus `lumpImpurities` and `enforce_same_density_gradients` in
`preprocess_prepare_parameters`.

### 4.3 Launching

```bash
mitim_run_maestro <folder> --namelist namelist.maestro.yaml --cpus 8
mitim_run_maestro <folder> --namelist nl.yaml --terminal       # log to terminal too
mitim_run_maestro <folder> --namelist nl.yaml --coldstart      # ignore checkpoints
mitim_run_maestro <folder> --namelist nl.yaml --save           # auto-save plots after
mitim_run_maestro <folder> --namelist nl.yaml \
    --slurm sched_mit_psfc mitim-env 24 64GB                    # submit as SLURM job
```

Re-running with the same folder is idempotent: each beat checks for its output
file and skips when present. The *first* beat that needs to run forces every
later beat to cold-start (state has changed). To force a particular beat to
re-run, delete its `Beats/<n>_*/beat_results/` output file.

### 4.4 Folder layout produced

```
<run-folder>/
  Outputs/
    maestro.log                 # combined log (only if --terminal)
    warnings.log
    Logs/                       # per-beat stdout
    Performance/                # per-beat timing
  Beats/
    1_transp/                   # ufiles, namelist.dat, run_transp/, beat_results/
    2_eped/                     # eped run + beat_results/input.gacode
    3_portals/                  # full PORTALS run subtree (see §3.4) + beat_results/
    4_eped/
    5_portals/
  maestro.namelist.actual.yaml  # exact namelist used (post-preprocess)
```

### 4.5 Reading + plotting results

```bash
mitim_plot_maestro <folder>                     # last 2 beats, summary
mitim_plot_maestro <folder> --beats 5 --full    # all 5 beats, full detail
mitim_plot_maestro <folder> --only transp       # restrict to TRANSP beats
mitim_plot_maestro <folder> --remote engaging   # rsync first
mitim_plot_maestro <folder> --save              # auto-save figs
mitim_check_maestro <folder>                    # quick textual progress check
```

The interesting per-beat output for downstream beats and external use is
`Beats/<n>_<type>/beat_results/input.gacode` (PORTALS, EPED) or
`run_transp/*.cdf` (TRANSP).

---

## 5. Shared concepts and pitfalls

### 5.1 `mitim_state` / PROFILEStools

The `mitim_state` (loaded from `input.gacode` via
`mitim_tools.gacode_tools.PROFILEStools`) is the canonical plasma state
passed between modules. It carries profiles, equilibrium, species, sources.
Most workflows accept either a path to `input.gacode` or a live state.

### 5.2 In-process transport (TGLF / NEO)

`transport.in_process: true` in the PORTALS namelist runs TGLF (and NEO,
when supported) via ctypes against `libtglf_serial.so` / `libneo_serial.so`,
no folder I/O, no subprocess fork. Build the libs once per machine via
`src/mitim_tools/simulation_tools/interfaces/build_{tglf,neo}_lib.sh`.
CAVEAT: in-process execution currently returns MINIMAL data (fluxes only);
spectral quantities are zero-filled placeholders.
Tests covering this path: `tests/dev_tests/test_tglf_inprocess.py` and
`tests/dev_tests/test_neo_inprocess.py` (unit), plus the teaching comparisons
`tests/capability_tests/tglf_11_run_inprocess.py` and
`tests/capability_tests/neo_03_run_inprocess.py` (standard vs in-process,
overlaid).

### 5.3 `profiles_postprocessing_fun`

A user-supplied callable (typically `functools.partial(...)`) with signature
`fn(file_profs: Path) -> gacode_state` that mutates `input.gacode` *before*
each transport-code dispatch — used to lump impurities, force quasineutrality,
thermalize fast ions, etc. Resolves per side (turbulence vs neoclassical) at
evaluation time: per-backend `transport.options.<name>.profiles_postprocessing_fun`
overrides the global `transport.profiles_postprocessing_fun`. When the two
sides resolve to *different* callables, the layer materialises two parallel
files (`input.gacode.turb` / `input.gacode.neo`) and tracks per-side
`impurityPosition_transport_{turb,neo}`.

### 5.4 Optimization options

Lower-level BO knobs (acquisition kernel, surrogate hyperprior, GP fit
options, transformation_variables thresholds, …) live in
`templates/namelist.optimization.yaml`. The PORTALS namelist's
`optimization_options:` block deep-merges into it; you only override what
differs.

### 5.5 SLURM / remote dispatch

Per-code remote submission goes through `FARMINGtools` and `mitim_job`. CGYRO
and GX have their own multi-task allocators (`slurm_array` for one element
per rho on GPU partitions; `slurm_standard` for a single allocation with `&`
parallelism). Per-iteration overrides (`extraOptions_special`,
`allocation_special`) accept selectors like `"0"`, `">5"`, `"<=10"`.

### 5.6 Logging conventions

Use `from mitim_tools.misc_tools.LOGtools import printMsg as print` and pass
`typeMsg=...` for styled output: `'i'` info, `'w'` warning, `'q'` interactive
yes/no question (returns bool — used to gate retries on transient errors,
e.g. `FARMINGtools.retrieve()`).

---

## 6. Tests as documented entry points

The legacy `tests/*_workflow.py` smoke tests have ALL been replaced by the
standalone teaching scripts in `tests/capability_tests/` (see §9): verbose,
runnable, end-to-end examples covering every wrapped code and workflow
(PORTALS, MAESTRO, TGLF, NEO, CGYRO, GX, TGYRO, TRANSP, EPED, FreeGS, VGEN,
Lengyel, VITALS, powertorch, the generic BO engine, in-process execution and
SLURM submission). They double as smoke tests.

Unit tests (in `tests/dev_tests/`):
- `test_cgyro_auto_resubmit.py` — CGYRO stall/resubmit logic.
- `test_*_inprocess.py` — ctypes-backed in-process TGLF/NEO.

If you change something that touches PORTALS, MAESTRO, or a transport
interface, run the relevant `tests/capability_tests/` script (or at minimum
`tests/dev_tests/test_*_inprocess.py` and
`tests/dev_tests/test_cgyro_auto_resubmit.py`) before committing.

---

## 7. Conventions to honor when editing

- **Namelists are the docs**: `templates/namelist.portals.yaml` and
  `templates/namelist.maestro.yaml` carry the per-knob comments. When adding
  or changing a knob, update the comment in the same edit. Don't move that
  documentation into Python or here.
- **Reserved DV names**: `fidelity_level` is reserved by the multi-fidelity
  machinery — do not use it as a user DV name; see
  `OPTtools._METHODS_HANDLE_FIDELITY_LEVEL` and `PORTALSmain.py:89-99`.
- **Scope localization for transient-failure fixes**: keep retry / disk-out /
  VPN-flap fixes inside the failing function (e.g. `FARMINGtools.retrieve()`'s
  `typeMsg='q'` retry prompt) rather than spreading layered defenses across
  the call chain.
- **Backwards-compat sentinels**: deprecated namelist keys still work as
  aliases and print a notice (`restart_from_first` →
  `restart_from_cases: "first"`; `extraOptions_first` →
  `extraOptions_special: {"0": ...}`; `allocation_first` →
  `allocation_special: {"0": ...}`). Keep that pattern when retiring keys.
- **`prep()` snapshots**: PORTALS/MAESTRO snapshot their namelist at `prep()` /
  `define_beat()` time. Edits after that point are silently ignored. If a knob
  needs to change mid-run, it has to be re-read from the on-disk snapshot.


---

## 8. Logging of changes

- The github logic in MITIM-fusion is that the "main" branch remains untouched until
  a new release is ready. Changes occur in "development", with the exception of
  important bug fixes.
- Whenever you commit anything to main directly or development, I want you to log
  the changes in the RELEASE_NOTES.md file that exists in "docs/". That file should
  follow the same template as common release documents in github: 
  https://github.com/pabloprf/MITIM-fusion/releases (see also as an example:
  "docs/RELEASE_NOTES_template.md"). Once a release has happened, you will clear up
  that RELEASE document and start again.
- Do not add trivial stuff, only add items to the document that are worth pointing
  to users and developers. Keep all the changes for a specific capability inside
  that bullet point as much as possible. For example, if improvements to a MAESTRO
  beat, do not populate the document with more than one bullet, unless needed.
- Contributors should not be added per item, but at the end of the document. Don't
  add the github repo main author.
- Do not add to RELEASE_NOTES at the moment of implementation, add stuff at the moment
  of making the commit.
- The information in the RELEASE document shouldn't be comprehensive, do not necessarily
  explain all the logic that led to that capability or bug fix. Except when it is very
  important, each bullet should not expand more than 5 lines.


## 9. Teaching

- In "MITIM-fusion/tests/capability_tests/", individual standalone scripts provide
  "tutorial-like" capabilities to teach users how to use the code base.
- When a new capability is added and you deem it worth it of adding it to the tests,
  go ahead and create the script (and commit it together with the capability).
- Make sure that you fix bugs, or change argument definitions, etc, that you modify
  the test accordingly.
- These files are meant to be verbose, lots of info for users to understand what is 
  going on.


## 10. Outside codes

- If you have questions about how TRANSP work, looking into the following websites may help:
  https://transp.pppl.gov/
  https://transp.jetdata.eu/docs/Help/HelpFile/body_transp_hlp.html
  But don't consume them all, search in them what you need