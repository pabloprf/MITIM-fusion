"""
CAPABILITY: Linear gyrokinetic spectra — CGYRO vs GX comparison
---------------------------------------------------------------
This script teaches how to compute the linear spectrum (real frequency and
growth rate vs binormal wavenumber ky) of a plasma state with two different
gyrokinetic codes, CGYRO and GX, and compare them in the same figure. Each
code runs on the machine configured for it in config_user.json (possibly
remote, via SLURM).

Key teaching points:
    1. CGYRO computes one linear mode per execution: the spectrum is obtained
       with run_scan() over the variable KY. Note relativeChanges=False, so
       varUpDown contains the *absolute* KY values to run, not multipliers of
       the base value (contrast with the TGLF scans). read_linear_scan()
       collects the converged eigenvalue of every KY run into a spectrum.
    2. GX computes the whole spectrum in a *single* run: the ky grid is set
       in extraOptions through y0 (kymin = 1/y0) and ny (nky = 1 + (ny-1)/3).
    3. lumpIons() bundles all ions (main + impurities) of the plasma state
       into a single effective species: fewer kinetic species makes the runs
       cheaper and the cross-code comparison cleaner.
    4. Results of each code can be combined in custom figures: a FigureNotebook
       accepts user-made tabs (fn.add_figure) next to the standard per-code
       plots, and the spectra are accessible as attributes (ky, f_mean, g_mean).
"""

import numpy as np
from mitim_tools.gacode_tools.PROFILEStools import gacode_state
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools.physics import GXtools
from mitim_tools.misc_tools import IOtools, GUItools, GRAPHICStools
from mitim_tools import __mitimroot__

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

# If True, do not show the plots on screen, save them to a subfolder instead
# (useful when running non-interactively, e.g. on an HPC node)
save_figures = False

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one subfolder per code lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_linear_gk"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare the plasma state (one radius, lumped ions)
# ---------------------------------------------------------------------------------------------------------------------

p = gacode_state(input_gacode)
# Lump all ions (main + impurities) into a single effective species (see docstring)
p.lumpIons()

rho = 0.7

# ---------------------------------------------------------------------------------------------------------------------
# 2. Linear CGYRO: one run per KY, collected into a spectrum
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.cgyro at the requested rho into the folder
# and attaches the experimental normalizations
cgyro = CGYROtools.CGYRO(rhos=[rho])
cgyro.prep(p, folder / "cgyro")

cgyro.run_scan(
    # Name of the subfolder (inside the working folder) where the scan points live
    "scan1",
    # Preset from templates/input.cgyro.models.yaml (level 2 of the hierarchy):
    # "Linear" sets NONLINEAR_FLAG=0 and a single toroidal mode (N_TOROIDAL=1)
    code_settings="Linear",
    # Individual input.cgyro parameters, applied on top of the preset (level 3)
    extraOptions={
        "MAX_TIME": 50.0,  # simulated time (a/c_s units) for the eigenvalue to converge
    },
    # Scan KY with absolute values (relativeChanges=False), not multipliers of the base
    variable="KY",
    varUpDown=np.linspace(0.1, 2.3, 24),
    relativeChanges=False,
    # Resources of each CGYRO instance (one per KY value): cores or GPUs per call, and SLURM time limit
    allocation={"resources_per_call": 16, "minutes": 30},
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
    # 'normal' submits and waits for completion ('submit' returns immediately:
    # come back later with cgyro.check() and cgyro.fetch())
    run_type="normal",
)

# read_linear_scan() collects the converged eigenvalue (real frequency f and growth rate g)
# of every KY run into a single spectrum stored under the label
cgyro.read_linear_scan(label="scan1", variable="KY", store_as_label="scan1", irho=0)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Linear GX: the whole spectrum in a single run
# ---------------------------------------------------------------------------------------------------------------------

# prep() works exactly as for CGYRO: one input file at the requested rho + normalizations
gx = GXtools.GX(rhos=[rho])
gx.prep(p, folder)

gx.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "gx1/",
    # Preset from templates/input.gx.models.yaml (level 2 of the hierarchy)
    code_settings="Linear Tokamak",
    # Individual input parameters, applied on top of the preset (level 3):
    # the ky grid below matches the CGYRO scan above (kymin = 0.1, 24 modes up to 2.3)
    extraOptions={
        "t_max": 50.0,  # simulated time (a/c_s units), as in the CGYRO runs
        "y0": 10.0,     # kymin = 1/y0 = 0.1
        "ny": 70,       # nky = 1 + (ny-1)/3 = 24 -> ky range 0.1 - 2.3
    },
    # GX runs all modes together, so a single allocation (GPUs where configured)
    allocation={"resources_per_call": 4, "minutes": 30},
    cold_start=cold_start,
)

# read() parses the GX output files and stores the results in the object under the label
gx.read("gx1")

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot: custom CGYRO-vs-GX comparison + standard per-code figures
# ---------------------------------------------------------------------------------------------------------------------

fn = GUItools.FigureNotebook("Linear GK", geometry="1600x1000", show=not save_figures)

# Custom tab: real frequency and growth rate vs ky, both codes together
fig = fn.add_figure(label="Comparison")
axs = fig.subplot_mosaic(
    """
    fg
    """
)

ax = axs["f"]
ax.plot(np.abs(cgyro.results["scan1"].ky), cgyro.results["scan1"].f_mean, "o-", label="CGYRO")
ax.plot(np.abs(gx.results["gx1"]["output"][0].ky), gx.results["gx1"]["output"][0].f_mean, "o-", label="GX")
ax.set_xlabel("$k_y \\rho_s$")
ax.set_ylabel("Real frequency ($a/c_s$)")
ax.axhline(0, color="k", ls="--")
ax.legend()
GRAPHICStools.addDenseAxis(ax)

ax = axs["g"]
ax.plot(np.abs(cgyro.results["scan1"].ky), cgyro.results["scan1"].g_mean, "o-", label="CGYRO")
ax.plot(np.abs(gx.results["gx1"]["output"][0].ky), gx.results["gx1"]["output"][0].g_mean, "o-", label="GX")
ax.set_xlabel("$k_y \\rho_s$")
ax.set_ylabel("Growth rate ($a/c_s$)")
ax.legend()
GRAPHICStools.addDenseAxis(ax)

# Standard per-code figures, added to the same notebook
fig = fn.add_figure(label="CGYRO")
cgyro.plot_quick_linear(labels=["scan1"], fig=fig)

gx.plot(labels=["gx1"], fn=fn)

# Show on screen, or save to a subfolder when running non-interactively
if not save_figures:
    fn.show()
    fn.close()
else:
    fn.save(f"{folder}/figs_gk/")
