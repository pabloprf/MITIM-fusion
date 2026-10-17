"""
CAPABILITY: Pull Alcator C-Mod experimental data (MDSplus) into a multi-tab notebook
------------------------------------------------------------------------------------
This teaches how to retrieve experimental time traces, EFIT equilibria and
diagnostic profiles (Thomson Te/ne, HIREXSR Ti/rotation) from Alcator C-Mod,
overlay several shots, and collect the views into a `GUItools.FigureNotebook`.
It is the C-Mod twin of `diiid_01_fetch_and_plot.py`: the SAME plotting layer
(`experiment_tools.diiid.plotting`) draws both machines; only the connection
(`CMODConnection`) differs.

Requires the optional MDS extra (pure-python thin client):
    pip install mitim-fusion[mds]      # installs `mdsthin`

ACCESS / being polite to the server:
    The C-Mod MDSplus server (alcdata.psfc.mit.edu:8000) is only reachable inside
    the PSFC network. Inside, leave tunnel_host=None; off-site, set tunnel_host to a
    passwordless SSH alias in your ~/.ssh/config that reaches alcdata:8000 (a PSFC
    workstation), or set it once for all scripts in config_user.json:
        "mdsplus": {"cmod": {"tunnel_host": "<alias>", "mds_server": "alcdata.psfc.mit.edu:8000"}}
    Here we open ONE `CMODConnection` and pass it to every overview() call, so a
    single SSH tunnel is reused across all tabs; every fetch is cached to disk
    (cache_dir) keyed by shot+node, so a re-run never hits the server.

Key teaching points:
    1. A "spec" is a C-Mod signal: a short name known to CMODFetcher.SIGNALS
       (ip, bt, kappa, tritop, tribot, q95, wmhd, nebar, zeff, prad, ...) or any
       tree node (`\\ANALYSIS::TOP:EFIT.RESULTS.A_EQDSK:KAPPA`, `SPECTROSCOPY::\\twopi_foil`).
    2. C-Mod stores time in SECONDS; the fetcher converts everything to ms, so
       t_window / shade / Equilibrium(time=...) are all in ms, exactly as for DIII-D.
    3. Machine defaults travel with the connection: the EFIT tree (ANALYSIS), the
       default time window (0-2000 ms) and the sign of Ip/B_T (stored negative at C-Mod;
       the 'ip'/'bt' aliases return them positive for the standard direction).
    4. `fetcher.inventory(t_window=...)` gives a found / not-found row per signal.
    5. Profiles: Thomson views are 'core', 'edge' or 'all' (Te in eV, ne in m^-3,
       mapped to rho through EFIT from their (R, Z)); HIREXSR ('hirex') profiles
       come on a psi_N grid, per spectral line ('z'/'w'/'x' He-like Ar, whole
       plasma; 'lya1' H-like Ar, core), quantity 'ti' [keV] or 'omega' = toroidal
       rotation frequency f = v_tor/(2 pi R) [kHz], sign as stored by THACO.
"""

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.experiment_tools.cmod.retrieval import CMODConnection, CMODFetcher
from mitim_tools.experiment_tools.diiid.plotting import (
    Trace, Panel, Equilibrium, Profiles, ProfilePanel, overview)

# ----------------------------------------------------------------------------
# USER SETTINGS — edit these
# ----------------------------------------------------------------------------
shots = [1120210021, 1120210007]        # N-seeded / unseeded ohmic, 5.4 T (Ennever thesis)
tunnel_host = None                       # None = config_user.json "mdsplus" block, else direct (inside PSFC);
                                         # off-site put YOUR ssh alias here (a passwordless ~/.ssh/config entry)
cache_dir = __mitimroot__ / "tests" / "scratch" / "cmod_fetcher"   # where to cache fetches
eq_shot, eq_time = 1120615015, 960.0     # equilibrium example (shot, time [ms])
# ----------------------------------------------------------------------------

fn = FigureNotebook("C-Mod experimental data")
common = dict(shots=shots, shade=(900, 1100),
              colors=["red", "blue"], labels=["N-seeded", "unseeded"],
              cache_dir=cache_dir, show=False)

# ONE connection reused across all tabs; each overview() draws into its tab figure.
with CMODConnection(tunnel_host=tunnel_host) as conn:

    # --- Tab 1: a broad "overview" set, one signal per panel (auto 3-column grid)
    overview(layout=[
        Panel(r"$I_p$ [MA]",          [Trace("ip", scale=1e-6)]),
        Panel(r"$B_T$ [T]",           [Trace("bt")]),
        Panel(r"$\bar{n}_e$ [$10^{20}$m$^{-3}$]", [Trace("nebar", scale=1e-20)]),
        Panel(r"$W_{MHD}$ [kJ]",        [Trace("wmhd", scale=1e-3)]),
        Panel(r"$P_{rad}$ [MW]",        [Trace("prad", scale=1e-6), Trace("prad_2pi", scale=1e-6, avg=20)]),
        Panel(r"$Z_{eff}$ (VB)",        [Trace("zeff", avg=20)], ylim=(0, 8)),
        Panel(r"$q_{95}$",              [Trace("q95")], ylim=(2, 8)),
        Panel(r"$\kappa$",              [Trace("kappa")]),
        Panel(r"$\delta$ (mean up/low)", [Trace(["tritop", "tribot"], reduce="mean")]),
    ], name="overview", connection=conn, fig=fn.add_figure(label="Overview"), **common)

    # --- Tab 2: Te/ne (Thomson core+edge) and Ti/rotation (HIREXSR) vs rho_tor + equilibrium
    overview(layout=[
        Profiles([
            ProfilePanel("thomson", "te", "all", scale=1e-3,  ylabel=r"$T_e$ [keV]"),
            ProfilePanel("thomson", "ne", "all", scale=1e-20, ylabel=r"$n_e$ [$10^{20}$m$^{-3}$]"),
        ], coord="rho", rho_max=1.1),
        Profiles([
            ProfilePanel("hirex", "ti", ["z", "lya1"], ylabel=r"$T_i$ [keV]", ylim=(0, 2.5)),
            ProfilePanel("hirex", "omega", ["z", "lya1"], ylabel=r"$f_\phi = v_\phi/2\pi R$ [kHz]",
                         ylim=(-20, 20)),
        ], coord="rho", rho_max=1.1),
        Equilibrium(),                  # time=None -> middle of the shade window; tree -> ANALYSIS
    ], name="profiles", connection=conn, fig=fn.add_figure(label="Profiles"), **common)

    # --- Tab 3: equilibrium example, single shot at a given time, and its g-file
    overview(shots=[eq_shot], layout=[
        [Panel(r"$I_p$ [MA]", [Trace("ip", scale=1e-6)]),
         Panel(r"$\kappa$", [Trace("kappa")]),
         Panel(r"$q_{95}$", [Trace("q95")], ylim=(2, 8))],
        Equilibrium(time=eq_time),
    ], name="equilibrium", connection=conn, cache_dir=cache_dir,
       show=False, fig=fn.add_figure(label="Equilibrium"))
    f = CMODFetcher(eq_shot, connection=conn, cache_dir=cache_dir)
    gfile = f.fetch_geqdsk(eq_time)
    print(f"* g-file for #{eq_shot} @ {eq_time:.0f} ms -> {gfile}")
    for name, row in f.inventory(t_window=(900, 1100)).items():   # found / not-found per signal
        print(f"  {name:>9s}: {row}")

fn.show()
