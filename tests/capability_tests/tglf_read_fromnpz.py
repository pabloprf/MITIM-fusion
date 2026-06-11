"""
CAPABILITY: Reload TGLF results from a .npz file
------------------------------------------------
NOTE: Run tglf_run_from_tglfinput.py first — this script reads the .npz file
that one produces.

This script teaches how to restore TGLF results without re-running the code.

Key teaching points:
    1. read(..., save_and_cleanup=file.npz) during a run session stores all
       labeled results in a single portable .npz file (and removes the raw run
       folders).
    2. TGLF.from_npz(file.npz) reconstructs the TGLF object with all its labels,
       anywhere and at any later time — useful to share results or revisit
       plots without access to the original run folders.
"""

from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

npz_file = __mitimroot__ / "tests" / "scratch" / "capability_tglf_run_from_tglfinput" / "tglf_results.npz"

if not npz_file.exists():
    raise FileNotFoundError(f"[MITIM] {npz_file} not found: run tglf_run_from_tglfinput.py first")

# ---------------------------------------------------------------------------------------------------------------------
# 1. Restore the TGLF object from the .npz and plot the stored labels
# ---------------------------------------------------------------------------------------------------------------------

# from_npz() rebuilds the TGLF object with every label that was saved into the file;
# no run folders or remote access needed
tglf = TGLFtools.TGLF.from_npz(npz_file)

# All figures go into a multi-tab MITIM FigureNotebook (tglf.fn); show() opens the GUI
tglf.plot(labels=["ES (SAT1)", "EM (SAT3)"])
tglf.fn.show()
