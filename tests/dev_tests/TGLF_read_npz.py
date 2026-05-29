"""
Read and plot TGLF results previously saved as npz files.

Requires that TGLF_workflow.py and TGLFscan_workflow.py have already been run
so that the npz files exist under tests/scratch/.
"""
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

scratch = __mitimroot__ / "tests" / "scratch"

# ---------------------------------------------------------------------------
# 1. Standard TGLF runs  (produced by TGLF_workflow.py)
# ---------------------------------------------------------------------------

tglf = TGLFtools.TGLF.from_npz(scratch / "tglf_test" / "tglf_results.npz")
tglf.plot(labels=["ES (SAT1)", "EM (SAT1)", "EM (SAT3)"])
tglf.fn.show()

# ---------------------------------------------------------------------------
# 2. Scan results  (produced by TGLFscan_workflow.py)
# ---------------------------------------------------------------------------

tglf_scan = TGLFtools.TGLF.from_npz(scratch / "tglfscan_test" / "scan_results.npz")

tglf_scan.plot_scan(labels=["scan1"], plotTGLFs=False)
tglf_scan.fn.show()

tglf_scan.plotScanTurbulenceDrives(label="turb_drives", plotTGLFs=False)
tglf_scan.fn.show()
