"""
test_organize_results_missing.py
===============================
What SIMtools.mitim_simulation._organize_results does when one radius did not come
back from the remote. The previous result of that radius must survive (it used to be
unlinked before the retrieved file was even known to exist), and the run must be
reported as incomplete so the scratch folder is kept for forensics.

Run as:

    python tests/dev_tests/test_organize_results_missing.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools import SIMtools

RHO_OK, RHO_MISSING = 0.3486, 0.6712
FILE = "out.cgyro.gbflux"


def _build():
    """Results folder with a previous result for both radii; scratch has only RHO_OK."""
    d = Path(tempfile.mkdtemp())
    results = d / "base_cgyro"
    results.mkdir()
    for rho in (RHO_OK, RHO_MISSING):
        (results / f"{FILE}_{rho:.4f}").write_text(f"previous result {rho}\n")

    tmpFolder = d / "tmp_cgyro"
    (tmpFolder / "base_cgyro" / f"rho_{RHO_OK:.4f}").mkdir(parents=True)
    (tmpFolder / "base_cgyro" / f"rho_{RHO_OK:.4f}" / FILE).write_text("fresh result\n")
    (tmpFolder / "base_cgyro" / f"rho_{RHO_MISSING:.4f}").mkdir(parents=True)

    code_executor = {"base_cgyro": {rho: {"folder": results} for rho in (RHO_OK, RHO_MISSING)}}
    sim = types.SimpleNamespace(simulation_job=None, FolderGACODE=d)
    return d, results, tmpFolder, code_executor, sim


def test_missing_file_keeps_previous_result_and_scratch():
    d, results, tmpFolder, code_executor, sim = _build()
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            SIMtools.mitim_simulation._organize_results(sim, code_executor, tmpFolder, [FILE])
        out = log.getvalue()

        assert (results / f"{FILE}_{RHO_OK:.4f}").read_text() == "fresh result\n"
        assert (results / f"{FILE}_{RHO_MISSING:.4f}").read_text() == f"previous result {RHO_MISSING}\n"
        assert tmpFolder.exists(), "scratch folder must be kept when something is missing"
        assert "Some files were not retrieved" in out, out
        assert "could not be retrieved" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: missing file -> previous result kept, scratch kept, run reported incomplete")


def test_all_present_moves_everything_and_wipes_scratch():
    d, results, tmpFolder, code_executor, sim = _build()
    (tmpFolder / "base_cgyro" / f"rho_{RHO_MISSING:.4f}" / FILE).write_text("fresh result 2\n")
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            SIMtools.mitim_simulation._organize_results(sim, code_executor, tmpFolder, [FILE])
        out = log.getvalue()

        assert (results / f"{FILE}_{RHO_OK:.4f}").read_text() == "fresh result\n"
        assert (results / f"{FILE}_{RHO_MISSING:.4f}").read_text() == "fresh result 2\n"
        assert not tmpFolder.exists(), "scratch folder must be removed on a complete retrieval"
        assert "All files were successfully retrieved" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: all files present -> all moved, scratch removed")


if __name__ == "__main__":
    test_missing_file_keeps_previous_result_and_scratch()
    test_all_present_moves_everything_and_wipes_scratch()
    print("\nALL TESTS PASSED")
