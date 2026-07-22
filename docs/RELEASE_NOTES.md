# vX.Y.Z — TITLE

DESCRIPTION

### New Features

*   🔬 **DIII-D experimental data (MDSplus)**: new `mitim_tools/experiment_tools/diiid/` (`retrieval.py` + `plotting.py`) pulls DIII-D time traces, EFIT equilibria and Thomson/CER profiles via the pure-python `mdsthin` thin client (optional extra: `pip install mitim-fusion[mds]`), over a direct connection or ONE reused SSH tunnel, with on-disk caching (server-side resampling, polite to the shared server). Signals are resolved with the standard DIII-D MDSplus cascade (findsig → ptdata2 → pseudo). `plotting.overview(shots, layout, …)` draws a declarative multi-column layout — scalar time traces, flux-surface equilibria (R,Z), and Te/ne/Ti profiles vs ρ (time-averaged over a chosen analysis window and mapped through each shot's EFIT, with error bars). CER profiles select the analysis flavor (CERQUICK/CERAUTO/CERFIT, including a per-panel overlay of several) and pull BOTH viewing systems — tangential and vertical; the `Profiles` column supports multiple analysis windows (one equilibrium+profile snapshot each), an all-time-samples scatter (`average=False`) and a ρ cutoff (`rho_max`). Window-averaging is STRICT — no out-of-window/nearest-slice fallback — so a too-narrow window honestly yields no point rather than a nearby one. `plotting.profiles_cer(…)` is a per-shot CER "check" plot that stacks the standard quantities (Ti, toroidal rotation, n_Z, n_Z/n_e) as rows (pick via `quantities`), each analysis flavor overlaid with one equilibrium per flavor; `ProfilePanel` likewise plots any CER quantity and takes a per-panel `alpha`. `retrieval.fetch_geqdsk(time, tree)` writes a GEQDSK file from the EFIT tree for a shot+time (read-back verified against `fetch_equilibrium`). `overview(…)` returns `(fig, axes)`, folds a trace's label and its source signal into one legend entry, and exposes `label_scale`/`line_scale`/`marker_scale` for talk-sized figures. See `tests/capability_tests/diiid_01_fetch_and_plot.py`.

*   🔬 **DIII-D `DIIIDExperiment` analysis class**: new `experiment_tools/diiid/experiment.py` adds an object-oriented layer on top of the retrieval — `DIIIDExperiment(shot, time, …)` with `overview`/`plot_cer_coverage`/`plot_cer_profiles`, QUICKFIT (Tomas Odstrcil's map2grid) profile fits `fit_te/ne/ti/omega/nimp`, `impurity_concentration()` from Zeff, and `to_gacode()` translating the fits into an `input.gacode` (`gacode_state`). QUICKFIT is an OPTIONAL, lazily-imported capability (`pip install mitim-fusion[quickfit]` → scikit-sparse; the quickfit clone is auto-located at `../quickfit` or `$QUICKFIT_PATH`), so retrieval/plotting still work without it. `DIIIDExperiment.multishot(…)` returns a `DIIIDMultiShot` group (one shared tunnel) with `overview`/`load_fits`/`merged_fits`/`to_gacode` across shots. See `tests/capability_tests/diiid_02_experiment_class.py`.


### Bug Fixes

*   🐛 **NEW BUG FIX**, description

### Changes for developers (internal execution)

*   🔎 **NEW CHANGE**, description

### Back-compatibility considerations and defaults

*   🔮 **NEW CONSIDERATION**, description

---

*Thanks to everyone who contributed to this release: USER LIST. Portions of this release were developed with AI-assisted coding (Claude Code).*
