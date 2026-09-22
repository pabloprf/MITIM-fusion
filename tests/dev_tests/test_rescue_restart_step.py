"""
test_rescue_restart_step.py
===========================
When SIMtools._rescue_interrupted_runs continues an interrupted CGYRO radius in place it
trims MAX_TIME to the time left. RESTART_STEP was sized by _enforce_restart_step from the
FULL MAX_TIME, so without the 'after_trim' hook the trigger
mod(i_time, RESTART_STEP*PRINT_STEP) == 0, i_time = 1..nint(MAX_TIME/DELTA_T)
no longer fires in the shortened window: the continuation writes no restart at all and
mitim_kill_cgyro's watchdog (which waits for out.cgyro.tag to move) blocks on that radius.

Run as:

    python tests/dev_tests/test_rescue_restart_step.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools

# DELTA_T=0.01, PRINT_STEP=100, MAX_TIME=450 -> n_outputs = 45000//100 = 450, so
# _enforce_restart_step coerced RESTART_STEP to 450 ("one restart at end of run")
INPUT = """#-------------------------------------------------------------------------
# cgyro input file modified by MITIM
#-------------------------------------------------------------------------


# Control parameters
# ------------------

DELTA_T                 = 1.00000E-02
DELTA_T_METHOD          = 0
PRINT_STEP              = 100
RESTART_STEP            = 450
MAX_TIME                = 4.50000E+02
N_RADIAL                = 24
"""

_after_trim = CGYROtools.CGYRO._restart_step_after_trim


def _key(text, key):
    return re.search(rf"^{key}\s*=\s*(\S+)", text, flags=re.M).group(1)


def test_trim_shrinks_restart_step():
    text, note = _after_trim(INPUT, 25.0)
    assert _key(text, "RESTART_STEP") == "25", _key(text, "RESTART_STEP")
    assert "RESTART_STEP 450 -> 25" in note, note
    # everything else untouched
    assert _key(text, "PRINT_STEP") == "100" and _key(text, "DELTA_T") == "1.00000E-02"
    print("PASS: RESTART_STEP 450 -> 25 when only 25 a/cs remain (fires at the last output step)")


def test_valid_divisor_is_kept():
    text, note = _after_trim(INPUT.replace("= 450", "= 5"), 425.0)
    # n_outputs_remaining = 42500//100 = 425, and 425 % 5 == 0
    assert _key(text, "RESTART_STEP") == "5", _key(text, "RESTART_STEP")
    assert note == "", note
    print("PASS: a RESTART_STEP that still divides the remaining n_outputs is kept")


def test_remaining_below_one_print_step():
    text, _ = _after_trim(INPUT, 0.5)
    # nint(0.5/0.01) = 50 < PRINT_STEP -> n_outputs_remaining floors to 0, clamped to 1
    assert _key(text, "RESTART_STEP") == "1", _key(text, "RESTART_STEP")
    print("PASS: remaining window shorter than one PRINT_STEP -> RESTART_STEP = 1")


def test_missing_keys_left_alone():
    stripped = "".join(l for l in INPUT.splitlines(keepends=True) if not l.startswith("PRINT_STEP"))
    text, note = _after_trim(stripped, 25.0)
    assert text == stripped and note == ""
    print("PASS: missing PRINT_STEP -> input returned unchanged")


# ---------------------------------------------------------------------------
# End-to-end through SIMtools._rescue_interrupted_runs with a fake job
# ---------------------------------------------------------------------------


class FakeJob:
    """probe_interrupted_runs stand-in: reports the radius as rescuable."""

    def __init__(self, md5, progress):
        self.run_in_place = False
        self.preserve_subfolders = None
        self._md5, self._progress = md5, progress

    def probe_interrupted_runs(self, folders_red, required, checksum_file, **kwargs):
        return {rel: (self._md5, self._progress, "out.cgyro.tag=26") for rel in folders_red}


def test_end_to_end_rescue_rewrites_both_keys():
    d = Path(tempfile.mkdtemp())
    try:
        rel = "base_cgyro/rho_0.6712"
        folder = d / rel
        folder.mkdir(parents=True)
        (folder / "input.cgyro").write_text(INPUT)
        (folder / "bin.cgyro.restart").write_text("staged blob")  # must be wiped by the rescue

        cgyro = CGYROtools.CGYRO()
        spec = cgyro.run_specifications["rescue_spec"]
        ignored = tuple(spec["checksum_ignore"])
        kept = "".join(l for l in INPUT.splitlines(keepends=True) if not l.startswith(ignored))
        sim = types.SimpleNamespace(
            run_specifications=cgyro.run_specifications,
            simulation_job=FakeJob(hashlib.md5(kept.encode()).hexdigest(), "425.0"),
        )

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            SIMtools.mitim_simulation._rescue_interrupted_runs(
                sim, {"rescue_interrupted": True}, d, [folder], [rel], "input.cgyro")
        out = log.getvalue()

        text = (folder / "input.cgyro").read_text()
        assert sim.simulation_job.preserve_subfolders == [rel], sim.simulation_job.preserve_subfolders
        assert not (folder / "bin.cgyro.restart").exists(), "staged restart must not survive the rescue"
        assert float(_key(text, "MAX_TIME")) == 25.0, _key(text, "MAX_TIME")
        assert _key(text, "RESTART_STEP") == "25", _key(text, "RESTART_STEP")
        assert "RESTART_STEP 450 -> 25" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: rescue trims MAX_TIME 450 -> 25 and RESTART_STEP 450 -> 25, logging both")


if __name__ == "__main__":
    test_trim_shrinks_restart_step()
    test_valid_divisor_is_kept()
    test_remaining_below_one_print_step()
    test_missing_keys_left_alone()
    test_end_to_end_rescue_rewrites_both_keys()
    print("\nALL TESTS PASSED")
