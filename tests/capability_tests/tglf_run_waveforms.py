"""
CAPABILITY: TGLF runs with eigenfunction waveforms
--------------------------------------------------
This script teaches how to obtain, together with a standard TGLF run, the
parallel structure (waveform along the field line) of the fluctuation
eigenfunctions at selected wavenumbers.

Key teaching points:
    1. Passing runWaveForms=[ky1, ky2, ...] to run() triggers, after the
       standard run, one extra TGLF execution per requested ky in waveform
       mode. By default the closest *unstable* mode of the spectrum to each
       requested ky is the one actually computed.
    2. The requested values are in ky*rho_s units, so e.g. 0.67 probes the
       ion-scale (ITG/TEM) part of the spectrum and 10.0 the electron-scale
       (ETG) part.
    3. plot() of a run with waveforms adds the eigenfunction-structure tabs
       to the usual fluxes/spectra ones.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: prepared input files and run subfolders live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_tglf_waveforms"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare TGLF at two radii from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# prep() reads the plasma state, writes one input.tglf per requested rho into the folder
# and attaches the experimental normalizations
tglf = TGLFtools.TGLF(rhos=[0.5, 0.7])
tglf.prep(input_gacode, folder, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run TGLF, requesting waveforms at an ion-scale and an electron-scale ky
# ---------------------------------------------------------------------------------------------------------------------

tglf.run(
    # Name of the subfolder (inside the working folder) where this run lives
    "waveforms/",
    # Preset from templates/input.tglf.models.yaml (level 2 of the hierarchy);
    # extraOptions could also be passed here, exactly as in a single run
    code_settings="SAT2",
    # ky*rho_s values at which to compute the eigenfunction waveforms (see docstring)
    runWaveForms=[0.67, 10.0],
    cold_start=cold_start,
    # With cold_start=True, remove previous results without asking for confirmation interactively
    forceIfcold_start=True,
)
# read() parses the TGLF output files (including the waveforms) and stores the
# results in the object under the label
tglf.read(label="SAT2 with WF")

# ---------------------------------------------------------------------------------------------------------------------
# 3. Plot fluxes, spectra and the eigenfunction structures
# ---------------------------------------------------------------------------------------------------------------------

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI
tglf.plot(labels=["SAT2 with WF"])
tglf.fn.show()
