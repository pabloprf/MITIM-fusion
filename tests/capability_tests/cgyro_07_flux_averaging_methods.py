"""
CAPABILITY: Choosing how nonlinear CGYRO fluxes are time-averaged
------------------------------------------------------------------
A nonlinear gyrokinetic run produces flux time traces Q(t), Gamma(t), Pi(t) that must be
averaged over their saturated phase. Where that phase starts, and how uncertain the
average is, has always been the user's guess (the `tmin` window). This script teaches
the `averaging` option of the CGYRO reader, shared with GX and used by PORTALS-CGYRO
(namelist `transport.options.cgyro.read.averaging`).

Key teaching points:
    1. Three methods, selected with averaging={'method': ...}:
         "fixed"       : the classic tmin / tmin_is_rel window (e.g. tmin=-0.3 -> last 30%).
         "quends"      : Sandia QUENDS (pip install mitim[quends]) trims the transient of each
                         primary channel (Qi, Qe, Ge) and computes block-mean statistics.
         "howard_gkav" : N.T. Howard's stationarity scan finds the earliest start from which
                         the three traces show no trend and no hump; the uncertainty is the
                         autocorrelation-corrected standard error over that window.
    2. The window is selected from the primary fluxes only, but it is then applied to EVERY
       signal of the run (ky spectra, fluctuation intensities, cross-phases), so the whole
       object is consistent with one window. `output.tmin` is the selected start.
    3. In all methods `*_std` is the 1-sigma STANDARD ERROR OF THE MEAN (what PORTALS uses
       as the surrogate noise), not the fluctuation amplitude.
    4. Each output carries `output.averaging` (a GKaverager) with the flag of the selection
       (ok / ok2 / questionable / fallback / failure / below_threshold for howard_gkav),
       diagnostics, provenance, `summary()` and `plot()`. PORTALS writes `to_dict()` per
       rho into fluxes_turb.json under additional_info['averaging'].
    5. A method that cannot find a steady state does not stop: howard_gkav proceeds with
       the second half of the run and flags it; quends falls back to the fixed window.

Usage:
    python cgyro_07_flux_averaging_methods.py [folder] [--suffix _0.55]
    Without arguments it reads the output of cgyro_02 (tests/scratch/capability_cgyro_nonlinear).
"""

import argparse
import matplotlib.pyplot as plt
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.simulation_tools.utils import GKaveraging

parser = argparse.ArgumentParser()
parser.add_argument("folder", nargs="?", default=str(__mitimroot__ / "tests" / "scratch" / "capability_cgyro_nonlinear" / "cgyro_run"))
parser.add_argument("--suffix", default="", help="per-rho suffix of the output files when several radii share a folder (e.g. _0.55)")
parser.add_argument("--noshow", action="store_true")
args = parser.parse_args()

methods = ["fixed", "howard_gkav"] + (["quends"] if GKaveraging.quends_available() else [])
if "quends" not in methods:
    print("quends is not installed; skipping that method (pip install mitim[quends])")

# ------------------------------------------------------------------------------------------
# Read the same run once per method. The averaging block is passed to read() exactly as
# PORTALS forwards `transport.options.cgyro.read` to the CGYRO output class.
# ------------------------------------------------------------------------------------------
c = CGYROtools.CGYRO()
for method in methods:
    c.read(
        label=method,
        folder=args.folder,
        suffix=args.suffix,
        tmin=-0.3, tmin_is_rel=True,             # only used by "fixed" (and as fallback by the others)
        averaging={"method": method},
        minimal=True,                             # fluxes only, no fluctuation fields
    )

# ------------------------------------------------------------------------------------------
# Compare: window start, flag, and the primary-flux means with their standard errors
# ------------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"{'method':>12} {'t_start':>9} {'t_end':>8} {'flag':>16} {'Qi (GB)':>22} {'Qe (GB)':>22} {'Ge (GB)':>24}")
for method in methods:
    out = c.results[method]["output"][0]
    a = out.averaging
    row = f"{method:>12} {a.t_start:9.1f} {a.t_end:8.1f} {a.flag:>16}"
    for ch in ("Qi", "Qe", "Ge"):
        row += f"   {a.stats[ch]['mean']:9.4f} +- {a.stats[ch]['std']:8.4f}"
    print(row)
print("=" * 100)

for method in methods:
    print()
    c.results[method]["output"][0].averaging.print_summary()

# ------------------------------------------------------------------------------------------
# Plot: one figure per method with the traces, the shaded window, the mean +- std band and
# the ACF of the windowed signal. The "Averaging" tabs also appear in the standard CGYRO
# notebook (mitim_plot_cgyro <folder> --averaging howard_gkav).
# ------------------------------------------------------------------------------------------
fn = FigureNotebook("CGYRO flux averaging methods", geometry="1500x900", show=not args.noshow)
for i, method in enumerate(methods):
    fig = fn.add_figure(label=method)
    c.results[method]["output"][0].averaging.plot(fig=fig, color=["b", "r", "g"][i], label_plot=method)

if not args.noshow:
    fn.show()
    plt.show()
