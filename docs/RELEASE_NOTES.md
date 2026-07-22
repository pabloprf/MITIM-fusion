# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   🌀 **TRANSP toroidal rotation I/O**: `w0` is no longer silently dropped on the way into TRANSP — `to_transp` ships it as the `omg` U-File (rad/s, main-ion), enabling rotation modeling (`nlvphi=T`) when the seed carries rotation, and — only under `neoclassical_transp` — the NCLASS neoclassical Er (`nlvwnc=T`; `OMEGA_NC`/`EPOTNC` in the CDF, analysis window auto-sized to the PORTALS prediction grid, overridable via `NCrotation_window`). The MAESTRO transp beat's `rotation_source` knob sets the `w0` carried to the next beat: `echo` (default, pass-through), `neoclassical_transp` (NCLASS weak-rotation E×B rotation), `neoclassical_portals` (downstream PORTALS beats recompute it with NEO-VGEN every evaluation), or `off`; legacy `write_rotation` still accepted. Full per-mode docs in `templates/namelist.maestro.yaml`. New dev tests: `test_transp_vs_vgen_rotation.py`, `test_maestro_rotation.py`.

*   📈 **PORTALS rotation diagnostics tab**: `mitim_plot_portals --complete` (any `plotPORTALS` call) now adds a "PORTALS Rotation" tab whenever rotation is relevant (non-trivial `w0`, or per-evaluation `transport.options.neo.vgen_exb_shear`): per-iteration evolution of the ion pressure, the ion pressure gradient `-dp_i/dr` (the diamagnetic drive — raw piecewise-linear `a/L` vs the VGEN-smoothed gradient actually integrated for Er), the E×B rotation `w0`, the VEXB_SHEAR profile into TGLF, and the per-radius `|VEXB_SHEAR|` vs evaluation (log scale). Profiles are drawn over the predicted core with markers at the predicted radii. New capability test `tests/capability_tests/portals_03_neoclassical_exb_shear.py` runs PORTALS with `vgen_exb_shear` on and opens this tab.


### Bug Fixes

*   🐛 **VGEN reuse on `cold_start=False`**: `NEO.run_vgen`'s "results already present" check listed the `out.vgen.*`/`vgen.dat` outputs without the `vgen/` subfolder prefix where VGEN actually writes them, so the check never found them and VGEN re-ran on every `cold_start=False` call. The paths are corrected, so a finished VGEN folder is now reused.

*   🐛 **TRANSP `w0` is the E×B rotation, not the toroidal velocity**: `CDFtools.to_profiles` wrote `w0(rad/s) = OMEGA` (the toroidal angular velocity), but the GACODE `w0` is the E×B/potential rotation `-c dPhi/dpsi` (the same quantity VGEN populates). For a plasma with a neoclassical `Er` but small toroidal rotation this wrote a `w0` ~40x too small (the toroidal velocity nearly vanishes while the E×B rotation does not). A new CDF variable `TGLF_w0_exb = Er/(dpsi/dR)` (CDF-native, no `EPOTNC` derivative) is now written as `w0`, making the TRANSP→input.gacode path consistent with the VGEN/NEO path. The CDF-direct TGLF parameters (`getTGLFparameters_all`) now also derive `VEXB_SHEAR`/`VPAR`/`VPAR_SHEAR` from this E×B `w0` instead of `OMEGA`, matching what `to_tglf` produces from the written `input.gacode`.

*   🐛 **TRANSP no-ICRF runs no longer crash in TORIC**: a `Pich=False` run with auto-generated (non-machine-fixed) structures still wrote the antenna geometry (`rmjicha`/`rmnicha`/`thicha`) to the namelist, and TRANSP turns the TORIC solver ON from the mere presence of those keys — then segfaults (`t4_tofpp_init.jpsedg`, "not enough zones") because no TORIC grid was emitted. The antenna block is now written only when ICRF is actually modeled (`Pich=True`), so lightweight no-heating TRANSP runs (e.g. neoclassical-only) run cleanly.

### Changes for developers (internal execution)

*   🔧 **`transp_run.run(cold_start=...)`** gained a skip-if-done guard: with `cold_start=False` it reuses an existing `{shot}{runid}.CDF` in the run folder (printing a notice) instead of re-staging and re-submitting to SLURM — the standalone TRANSP path previously had no idempotency of its own (only MAESTRO's beat wrapper did). Default `cold_start=True` preserves the always-(re)run behaviour, so existing callers (MAESTRO included) are unaffected.

### Back-compatibility considerations and defaults

*   🔒 **PORTALS `w0` prediction vs `vgen_exb_shear`**: predicting `w0` (`solution.predicted_channels`) together with `transport.options.neo.vgen_exb_shear` now raises in `prep()`. `vgen_exb_shear` recomputes `w0` from the neoclassical Er at every evaluation, which would overwrite the predicted rotation and make its momentum-flux match a no-op — the two are mutually exclusive, so pick one.

*   🌀 **TRANSP rotation defaults**: the default `rotation_source: echo` is a true pass-through — with zero rotation it now generates the pre-rotation TRANSP namelist bit-for-bit (no `omg` U-File, `nlvphi`/`nlvwnc` off), so existing chains are unchanged. The NCLASS neoclassical Er (`nlvwnc=T`) is computed **only** under `neoclassical_transp` (the one mode that keeps it); `echo`/`neoclassical_portals` model rotation only when the seed actually carries it and never compute the discarded Er. `off` opts out entirely; rotation physics in the carried `w0` is strictly opt-in.

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
