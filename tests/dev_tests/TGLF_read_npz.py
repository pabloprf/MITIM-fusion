"""
Read and plot TGLF results previously saved as npz files.

Requires that capability_tests/tglf_run_from_tglfinput.py and
capability_tests/tglf_scan.py have already been run so that the npz files
exist under tests/scratch/.
"""
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools import __mitimroot__

scratch = __mitimroot__ / "tests" / "scratch"

# ---------------------------------------------------------------------------
# 1. Standard TGLF runs  (produced by capability_tests/tglf_run_from_tglfinput.py)
# ---------------------------------------------------------------------------

tglf = TGLFtools.TGLF.from_npz(scratch / "capability_tglf_run_from_tglfinput" / "tglf_results.npz")
tglf.plot(labels=["ES (SAT1)", "EM (SAT3)"])
tglf.fn.show()

# ---------------------------------------------------------------------------
# 2. Scan results  (produced by capability_tests/tglf_scan.py)
# ---------------------------------------------------------------------------

tglf_scan = TGLFtools.TGLF.from_npz(scratch / "capability_tglf_scan" / "scan_results.npz")

tglf_scan.plot_scan(labels=["scan_aLTe"], plotTGLFs=False)
tglf_scan.fn.show()
