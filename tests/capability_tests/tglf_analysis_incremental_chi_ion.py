"""
CAPABILITY: Ion and cross incremental diffusivities with TGLF
-------------------------------------------------------------
This script teaches the two remaining thermal flavors of runAnalysis() (see
tglf_analysis_incremental_chi.py first for the electron one and the meaning
of the derived coefficients):

    - 'chi_i'  : ion stiffness — scan the ion temperature gradient (RLTS_2)
                 and analyze the Qi response.
    - 'chi_ei' : cross-coupling — scan the SAME ion temperature gradient but
                 analyze the response of the ELECTRON heat flux, i.e. how
                 strongly a/LTi drives electron transport.

Key teaching points:
    1. Both analyses scan RLTS_2 by +-5% around the base value (11 points);
       what changes is the flux channel from which chi_inc/chi_eff/chi_pb and
       the pinch are derived.
    2. Because each runAnalysis() call performs its own scan in its own
       subfolder, the two analyses are independent runs (the TGLF cases of
       'chi_i' are not reused by 'chi_ei').
    3. Each plotAnalysis() call opens its own notebook; the cross analysis is
       the diagnostic of choice when electron transport seems enslaved to the
       ion channel (e.g. ITG-dominated regimes).
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: one scan subfolder per analysis lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_chi_ion"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at one radius from the plasma state (one radius to keep it cheap)
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations (required by runAnalysis)
tglf = TGLFtools.TGLF(rhos=[0.5])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Ion incremental diffusivity: a/LTi scan -> Qi response
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    subfolder="chi_i",
    label="chi_i_sat2",
    analysisType="chi_i",
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Cross diffusivity: the SAME a/LTi scan -> Qe response
# ---------------------------------------------------------------------------------------------------------------------

tglf.runAnalysis(
    subfolder="chi_ei",
    label="chi_ei_sat2",
    analysisType="chi_ei",
    code_settings="SAT2",
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 4. Plot both analyses (each call opens its own notebook)
# ---------------------------------------------------------------------------------------------------------------------

# plotTGLFs=False skips the per-scan-point TGLF notebooks (11 of them per analysis)
tglf.plotAnalysis(labels=["chi_i_sat2"], analysisType="chi_i", plotTGLFs=False)
fn_ion = tglf.fn

tglf.plotAnalysis(labels=["chi_ei_sat2"], analysisType="chi_ei", plotTGLFs=False)

fn_ion.show()
tglf.fn.show()
