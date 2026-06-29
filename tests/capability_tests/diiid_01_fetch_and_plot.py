"""
CAPABILITY: Pull DIII-D experimental data (MDSplus) into a multi-tab notebook
-----------------------------------------------------------------------------
This teaches how to retrieve experimental time traces, EFIT equilibria and
diagnostic profiles from the DIII-D tokamak, overlay several shots, and collect
the views into a `GUItools.FigureNotebook` with one plot per tab.

Requires the optional MDS extra (pure-python thin client):
    pip install mitim-fusion[mds]      # installs `mdsthin`

ACCESS / being polite to the server:
    The DIII-D MDSplus server (atlas.gat.com:8000) is only reachable from inside
    GA. On a GA host / VPN leave tunnel_host=None to connect directly; off-site,
    set tunnel_host below to a passwordless SSH jump host in your ~/.ssh/config
    that can reach atlas:8000 (e.g. a GA gateway such as 'cybele').
    Here we open ONE `DIIIDConnection` and pass it to every overview() call, so
    a single SSH tunnel is reused across all three tabs; every fetch is cached
    to disk (CACHE_DIR) keyed by shot+spec, so a re-run never hits the server.

Key teaching points:
    1. A "spec" is a DIII-D signal: a bare pointname (ip, wmhd, density,
       cerqtit1, ...), a `PTDATA::<name>`, or a tree node (`\\EFIT01::...`).
       It is resolved with the standard DIII-D cascade: findsig -> ptdata2 -> pseudo.
    2. `overview(shots, layout, fig=...)` draws a declarative `layout` of columns
       into a given figure (a FigureNotebook tab). A column is a list of Panels
       (time traces), an `Equilibrium` (R,Z flux surfaces), or a `Profiles`
       column (Te/ne/Ti vs rho, time-averaged over the shaded analysis window
       and mapped through each shot's EFIT). `Equilibrium(time=None)` plots at
       the middle of the shade window, so equilibrium and profiles share a time.
    3. A `Trace` can derive a quantity from several specs via `reduce`
       ('sum'|'diff'|'ratio'|'mean'); a Thomson `ProfilePanel` can combine the
       core and tangential views with `system='all'`.
"""

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.experiment_tools.diiid.retrieval import DIIIDConnection
from mitim_tools.experiment_tools.diiid.plotting import (
    Trace, Panel, Equilibrium, Profiles, ProfilePanel, overview)

# ----------------------------------------------------------------------------
# USER SETTINGS — edit these
# ----------------------------------------------------------------------------
shots = [207958, 207959]                 # DIII-D shot(s) to overlay
tunnel_host = None                        # None = connect directly (GA host/VPN); else YOUR ssh jump host
cache_dir = __mitimroot__ / "tests" / "scratch" / "diiid_fetcher"   # where to cache fetches
# ----------------------------------------------------------------------------

fn = FigureNotebook("DIII-D experimental data")
common = dict(shots=shots, t_window=(1300, 4600), shade=(3800, 4100),
              colors=["blue", "green"], labels=["shot A", "shot B"],
              cache_dir=cache_dir, show=False)

# ONE connection reused across all tabs; each overview() draws into its tab figure.
with DIIIDConnection(tunnel_host=tunnel_host) as conn:

    # --- Tab 1: a broad "overview" set, one signal per panel (auto 3-column grid)
    #     (units go straight into the ylabel string)
    overview(layout=[
        Panel(r"$I_p$ [MA]",            [Trace("ip", scale=1e-6)]),
        Panel(r"$B_T$ [T]",             [Trace("bt")]),
        Panel(r"$\bar{n}_e$ [$10^{20}$m$^{-3}$]", [Trace("density", scale=1e-14)]),
        Panel(r"$W_{MHD}$ [MJ]",        [Trace("wmhd", scale=1e-6)]),
        Panel(r"$P_{NBI}$ [MW]",        [Trace("pinj", scale=1e-3, avg=200)]),
        Panel(r"$P_{rad}$ [MW]",        [Trace("prad_tot", scale=1e-6)]),
        Panel(r"$\beta_N$",             [Trace("betan")]),
        Panel(r"$q_{95}$",              [Trace("q95")]),
        Panel(r"$\tau_E$ [s]",          [Trace("taue")]),
    ], name="overview", connection=conn, fig=fn.add_figure(label="Overview"), **common)

    # --- Tab 2: a curated "custom" engineering set in explicit columns + equilibrium
    overview(layout=[
        [   # column 1
            Panel(r"$I_p$ [MA]", [Trace("ip", scale=1e-6)]),
            Panel(r"$\kappa$",   [Trace("kappa")]),
            Panel(r"$\delta$",   [Trace(["tritop", "tribot"], reduce="mean")]),
            Panel(r"$q_{95}$",   [Trace("q95")]),
        ],
        [   # column 2
            Panel(r"$\bar{n}_e$ [$10^{20}$m$^{-3}$]", [Trace("density", scale=1e-14)]),
            Panel(r"$W_{MHD}$ [MJ]", [Trace("wmhd", scale=1e-6)]),
            Panel(r"$Z_{eff}$",      [Trace("zeff")], ylim=(1.0, 3.0)),
            Panel(r"$\tau_E$ [s]",   [Trace("taue")]),
        ],
        Equilibrium(),               # time=None -> middle of the shade window
    ], name="custom", connection=conn, fig=fn.add_figure(label="Custom"), **common)

    # --- Tab 3: Te and ne radial profiles (Thomson core+tangential) vs rho_tor
    overview(layout=[
        Profiles([
            ProfilePanel("thomson", "te", "all", scale=1e-3,  ylabel=r"$T_e$ [keV]"),
            ProfilePanel("thomson", "ne", "all", scale=1e-20, ylabel=r"$n_e$ [$10^{20}$m$^{-3}$]"),
        ], coord="rho"),
        Equilibrium(),               # context: where the diagnostics sit
    ], name="profiles", connection=conn, fig=fn.add_figure(label="Te / ne profiles"), **common)

fn.show()
