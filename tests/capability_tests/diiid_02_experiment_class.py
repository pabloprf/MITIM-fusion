"""
CAPABILITY: Analyze a DIII-D discharge end-to-end with the DIIIDExperiment class
--------------------------------------------------------------------------------
`diiid_01_fetch_and_plot.py` teaches the low-level, multi-shot fetch+plot API
(declarative `overview(shots, layout, ...)`, custom columns, `Equilibrium`,
`ProfilePanel`). THIS script teaches the object-oriented layer on top of it:
`DIIIDExperiment` bundles a single shot + its analysis time window and exposes
the whole per-shot workflow as methods — fetch, overview, impurity
concentration, profile fitting, and export to a MITIM `input.gacode`.

    exp = DIIIDExperiment(shot, time=..., tunnel_host=..., cache_dir=...)
    exp.overview(...)                 # fetch + plot engineering/kinetic traces
    exp.impurity_concentration(...)   # c_C, c_imp from the measured Zeff
    exp.fit_te(); exp.fit_ti(...)     # map2grid profile fits (QUICKFIT)
    exp.to_gacode(...)                # -> input.gacode (fits + EFIT geometry)

Optional dependencies (guarded, so partial installs still teach something):
    pip install mitim-fusion[mds]       # `mdsthin` — needed for ALL of it (fetch)
    pip install mitim-fusion[quickfit]  # `scikit-sparse` — needed for the FITS/to_gacode
The fitting also needs Tomas Odstrcil's `map2grid` (`quickfit`), which is NOT on
PyPI: clone https://github.com/odstrcilt/quickfit next to MITIM-fusion (../quickfit)
or point $QUICKFIT_PATH at it. Without it, overview + impurity_concentration still
run; the fit/to_gacode block is skipped with a message.

ACCESS / being polite to the server (same as diiid_01):
    The DIII-D MDSplus server (atlas.gat.com:8000) is only reachable from inside
    GA. On a GA host / VPN set tunnel_host=None to connect directly; off-site,
    set tunnel_host to a passwordless SSH jump host in ~/.ssh/config that reaches
    atlas:8000 (e.g. 'cybele'). The object owns ONE connection for its lifetime
    (opened lazily, closed by the `with` block); every fetch is cached to disk
    (cache_dir) keyed by shot+spec, so a re-run never hits the server.

Key teaching points:
    1. Construct once with the analysis context (shot, extraction `time`, window
       averaging `avg`, fit window `t_range`); every method reuses it. Use it as
       a context manager so the owned SSH connection is closed on exit.
    2. `overview(layout=None)` fetches AND plots; with no layout it draws a
       sensible engineering/kinetic default. Pass `fig=` to render into a
       FigureNotebook tab, exactly like the low-level `overview()`.
    3. `fit_te/fit_ne/fit_ti/fit_omega` return a dict with the fitted profile on
       a rho grid (`rgrid`,`prof`), uncertainty band, the measured cloud
       (`data_rho/data_val/...`) and the fit `chi2`. `robust=True` down-weights
       outliers; `fit_ti(sources=[...])` selects which CER species to fit.
    4. `to_gacode(...)` ties it together: robust fits of Te/ne/Ti/omega + EFIT
       geometry (geqdsk -> MXH) + a uniform-concentration impurity model from
       Zeff -> a quasineutral, zero-source MITIM `input.gacode` you can hand to
       PORTALS. Units are flagged at each conversion inside the method.
    5. To COMPARE shots, group instances: `ab = DIIIDExperiment.multishot(a, b)`
       returns a `DIIIDMultiShot` whose overlay methods (`ab.overview()`,
       `ab.plot_cer_coverage()`) mirror the single-shot ones but draw every shot
       through ONE connection; `ab.load_fits(tag)` returns {shot: stored fit} for
       your own comparison figures.
"""

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.experiment_tools.diiid.experiment import DIIIDExperiment

# ----------------------------------------------------------------------------
# USER SETTINGS — edit these
# ----------------------------------------------------------------------------
shot        = 207959                      # DIII-D shot to analyze
time        = 4000.0                      # ms — extraction time (fit node / profile export)
avg         = 200.0                       # ms — half-window averaging for overview traces
t_range     = (1400.0, 4150.0)            # ms — fit window fed to map2grid
tunnel_host = "cybele"                    # None on a GA host/VPN; else YOUR ssh jump host
cache_dir   = __mitimroot__ / "tests" / "scratch" / "diiid_experiment"
# For to_gacode on a puffed shot, name the puffed impurity + its onset time so the
# extra Zeff above frozen Carbon is attributed to it (None,None = intrinsic Carbon only):
puff_impurity, puff_time = "Neon", 1800.0
compare_shot = 207958                     # a second shot to overlay against `shot` (multishot demo)
# ----------------------------------------------------------------------------

fn = FigureNotebook("DIII-D experiment — %d" % shot)

with DIIIDExperiment(shot, time=time, avg=avg, t_range=t_range,
                     tunnel_host=tunnel_host, cache_dir=cache_dir) as exp:

    # --- Tab 1: engineering/kinetic overview (default layout; MDS only) ---
    exp.overview(fig=fn.add_figure(label="Overview"),
                 shade=(time - avg, time + avg), show=False)

    # --- impurity concentrations from the measured Zeff (uniform model; MDS only) ---
    cC, cimp, imp = exp.impurity_concentration(impurity=puff_impurity, puff_time=puff_time)
    print(f"[{shot}] @ {time:.0f} ms  c_C = {cC*100:.2f}%"
          + (f",  c_{imp} = {cimp*100:.2f}%" if imp else "  (intrinsic Carbon only)"))

    # --- profile fits + input.gacode (needs the QUICKFIT extra + map2grid clone) ---
    try:
        # Ti from Carbon (CERAUTO) + the puffed impurity via CERFIT, if any:
        ti_sources = [("cera", "CERAUTO", "Carbon")]
        if imp:
            ti_sources.append(("cerf", "CERFIT", imp))

        te = exp.fit_te(robust=True, use_ece=True)     # ECE pins the core where tangential TS is absent
        ti = exp.fit_ti(robust=True, sources=ti_sources)
        print(f"[{shot}] robust fits:  Te core = {te['prof'][0]*1e-3:.2f} keV (chi2={te['chi2']:.2f}),  "
              f"Ti core = {ti['prof'][0]*1e-3:.2f} keV (chi2={ti['chi2']:.2f}),  "
              f"CER species fitted = {sorted(set(ti['data_imp']))}")

        # full export: robust Te/ne/Ti/omega + EFIT geometry -> quasineutral input.gacode
        p = exp.to_gacode(ti_sources=ti_sources, impurity=puff_impurity, puff_time=puff_time,
                          plot_data=True)   # plot_data -> diiid_to_gacode_<shot>_<time>.png next to the file
        print(f"[{shot}] input.gacode: nion={len(p.profiles['name'])} ({'+'.join(p.profiles['name'])}), "
              f"Zeff(0)={p.profiles['z_eff(-)'][0]:.2f}, q95={p.derived['q95']:.2f}")
    except ImportError as e:
        print("\n[QUICKFIT missing] skipping the fit / to_gacode block — overview + impurity "
              "concentration above only need the [mds] extra.\n  " + str(e).splitlines()[0])

# --- multi-shot comparison: group instances into a DIIIDMultiShot and overlay in one figure ---
# DIIIDExperiment.multishot(a, b, ...) returns a group object you build once; its overlay methods
# (overview / plot_cer_coverage / plot_cer_profiles) mirror the single-shot ones but draw every shot.
# All shots are fetched through the FIRST instance's connection, so the whole comparison opens ONE
# SSH tunnel (the other instances' lazy connections are never touched). For experiment-specific
# comparison figures, pull the stored fits with ab.load_fits("ti_robust") -> {shot: fit} and plot
# them yourself — the group stays general and doesn't bake in any study's styling.
with DIIIDExperiment(shot, time=time, tunnel_host=tunnel_host, cache_dir=cache_dir) as a, \
     DIIIDExperiment(compare_shot, time=time, tunnel_host=tunnel_host, cache_dir=cache_dir) as b:
    ab = DIIIDExperiment.multishot(a, b)
    ab.overview(fig=fn.add_figure(label=f"Compare {shot}/{compare_shot}"),
                colors=["green", "blue"], labels=[str(shot), str(compare_shot)], show=False)

fn.show()
