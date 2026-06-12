"""
CAPABILITY: Incremental diffusivity (stiffness) analyses with TGLF
------------------------------------------------------------------
This script teaches how to extract thermal transport coefficients from the
local flux-gradient response of TGLF, starting from an input.gacode file. All
three thermal flavors of runAnalysis() are run:

    - 'chi_e'  : electron stiffness — scan the electron temperature gradient
                 (RLTS_1) and analyze the Qe response.
    - 'chi_i'  : ion stiffness — scan the ion temperature gradient (RLTS_2)
                 and analyze the Qi response.
    - 'chi_ei' : cross-coupling — scan the SAME ion temperature gradient but
                 analyze the response of the ELECTRON heat flux, i.e. how
                 strongly a/LTi drives electron transport (the diagnostic of
                 choice when electron transport seems enslaved to the ion
                 channel, e.g. ITG-dominated regimes).

Key teaching points:
    1. Each analysis scans its gradient by +-5% around the base value (11
       points) and, from the flux response, computes at each radius:
         - chi_inc : incremental (stiffness) diffusivity, dQ/d(grad T)
         - chi_eff : effective diffusivity, Q/(n grad T)
         - chi_pb  : power-balance diffusivity at the base gradient
         - Vpinch  : the equivalent heat pinch
       This requires the experimental normalizations, so the object must be
       prepared from a plasma state (input.gacode), not from a bare input.tglf.
    2. Each runAnalysis() call performs its own scan in its own subfolder
       (note 'chi_i' and 'chi_ei' scan the same parameter, but the runs are
       not shared between them).
    3. The three analyses share ONE FigureNotebook: plotAnalysis() accepts an
       external notebook (fn=...) with a per-analysis tab color (fn_color) and
       tab-label prefix (extratitle), so everything lives in a single window.
    4. The remaining flavor, 'Z' (trace-impurity D and V), has its own
       capability test: tglf_9_analysis_DV_trace_impurity.py.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools, GUItools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one scan subfolder per analysis lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_incremental_chi"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at one radius from the plasma state (one radius to keep the 33 runs cheap)
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations (required by runAnalysis to produce
# diffusivities in physical units)
tglf = TGLFtools.TGLF(rhos=[0.5])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Electron incremental diffusivity: a/LTe scan -> Qe response
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    # Name of the subfolder (inside the working folder) where the scan lives
    subfolder="chi_e",
    # Results are stored in the object under this label
    label="chi_e_sat2",
    # Which analysis to perform (see docstring for the available types)
    analysisType="chi_e",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy);
    # extraOptions could also be passed here, exactly as in a single run
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Ion incremental diffusivity: a/LTi scan -> Qi response
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    subfolder="chi_i",
    label="chi_i_sat2",
    analysisType="chi_i",
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 4. Cross diffusivity: the SAME a/LTi scan -> Qe response
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    subfolder="chi_ei",
    label="chi_ei_sat2",
    analysisType="chi_ei",
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 5. Plot the three analyses in ONE notebook (one tab color per analysis)
# ---------------------------------------------------------------------------------------------------------------------

# A single notebook shared by the three analyses: each plotAnalysis() call adds its tabs
# with its own color (fn_color) and label prefix (extratitle).
# plotTGLFs=False skips the per-scan-point TGLF notebooks (11 of them per analysis);
# set it to True to inspect the spectra of each individual scan point
fn = GUItools.FigureNotebook("TGLF incremental-diffusivity analyses", geometry="1500x900")

tglf.plotAnalysis(labels=["chi_e_sat2"], analysisType="chi_e", plotTGLFs=False, fn=fn, fn_color=0, extratitle="chi_e - ")
tglf.plotAnalysis(labels=["chi_i_sat2"], analysisType="chi_i", plotTGLFs=False, fn=fn, fn_color=1, extratitle="chi_i - ")
tglf.plotAnalysis(labels=["chi_ei_sat2"], analysisType="chi_ei", plotTGLFs=False, fn=fn, fn_color=2, extratitle="chi_ei - ")

fn.show()

