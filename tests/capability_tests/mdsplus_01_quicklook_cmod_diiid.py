"""
CAPABILITY: quick look at one Alcator C-Mod and one DIII-D shot (MDSplus)
-------------------------------------------------------------------------
Pulls Ip, B_T, stored energy, q95, Thomson Te/ne profiles and the EFIT equilibrium of one shot per
machine and plots them with the same layout. The per-machine details (trees, time
window, EFIT tree) come with each machine's fetcher.

SETUP (once):
    1. pip install mitim-fusion[mds]            # the pure-python MDSplus thin client (mdsthin)
    2. Both MDSplus servers are only reachable from inside their site (C-Mod:
       alcdata.psfc.mit.edu:8000 inside PSFC; DIII-D: atlas.gat.com:8000 inside GA).
       Off-site, MITIM opens an SSH tunnel through a jump host. Tell it which one per
       machine in your config_user.json (templates/config_user.json, or $MITIM_CONFIG):

           "mdsplus": {
               "cmod":  {"tunnel_host": "<ssh alias that reaches alcdata:8000>"},
               "diiid": {"tunnel_host": "<ssh alias that reaches atlas:8000>"}
           }

       Each value is a Host alias from YOUR ~/.ssh/config that logs in without a
       password prompt (keys/agent; ProxyJump chains are fine), e.g. a PSFC
       workstation for C-Mod and a GA gateway for DIII-D. Leave a machine out (or
       set null) when you run from inside its site: MITIM then connects directly.
       Check with `ssh <alias> hostname` before running this script.

HOW THE DATA FLOWS:
    * `ov_cmod = overview(...)` and `ov_diiid = overview(...)` each fetch their shot from that
      machine's server (one lazily-opened tunnel per machine, only on a cache miss), cache every
      fetch on disk (tests/scratch/), and draw one figure. Each returns an `Overview` object that
      keeps everything it fetched in memory (it still unpacks as `fig, axes`).
    * `overview_together([ov_cmod, ov_diiid], ...)` overlays both machines in ONE figure from
      those in-memory objects only: it opens no connection, no tunnel and reads no cache. Each
      shot keeps its own EFIT tree, analysis window (shaded in its colour) and label, and the
      equilibrium panel overlays both boundaries and both walls.
"""

import matplotlib.pyplot as plt
from mitim_tools.experiment_tools.diiid.plotting import (
    Trace, Panel, Equilibrium, Profiles, ProfilePanel, overview, overview_together)

layout = [
    [Panel(r"$I_p$ [MA]", [Trace("ip", scale=1e-6, abs=True)]),
     Panel(r"$B_T$ [T]", [Trace("bt", abs=True)])],
    [Panel(r"$W_{MHD}$ [MJ]", [Trace("wmhd", scale=1e-6)]),
     Panel(r"$q_{95}$", [Trace("q95")], ylim=(2, 8))],
    Profiles([
        ProfilePanel("thomson", "te", "all", scale=1e-3, ylabel=r"$T_e$ [keV]"),
        ProfilePanel("thomson", "ne", "all", scale=1e-20, ylabel=r"$n_e$ [$10^{20}$m$^{-3}$]"),
    ], coord="rho"),
    Equilibrium(),
]

# t_window [ms] spans the whole discharge (without it, each machine uses its default display window)

# 1) C-Mod: LOADS from alcdata (via the config_user.json tunnel; disk cache on repeat runs) and plots
ov_cmod = overview(shots=[1120615015], layout=layout, machine="cmod", t_window=(0, 2000), shade=(910, 1010), name="C-Mod 1120615015", show=False)

# 2) DIII-D: LOADS from atlas the same way and plots
ov_diiid = overview(shots=[207959], layout=layout, machine="diiid", t_window=(0, 5200), shade=(3900, 4100), name="DIII-D 207959", show=False)

# 3) Both machines in one figure: REUSES the data already held by ov_cmod / ov_diiid, touches no server
overview_together([ov_cmod, ov_diiid], name="C-Mod vs DIII-D", show=False)

plt.show()
