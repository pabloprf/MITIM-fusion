"""
test_submit_needs_minutes.py
============================
A detached submission (run_type 'submit') must carry an explicit wall clock: without allocation['minutes']
MITIM used to submit every array element with a 5-minute limit (engaging 23542221: four reduced3 radii,
all TIMEOUT after 5 min). Also pins the default for the other run types at 10 minutes.

Run as:  python tests/dev_tests/test_submit_needs_minutes.py
"""
import sys
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools import SIMtools


def _sim():
    return types.SimpleNamespace(
        run_specifications={"code": "cgyro", "input_file": "input.cgyro"},
        nameRunid="x",
        FolderGACODE=Path("/tmp/mitim_test"),
        output_files_simulation={"complete": [], "minimal": [], "optional": []},
        _default_allocation=lambda code, minutes=10: {"resources_per_call": 1, "minutes": minutes},
    )


def _settings(run_type, allocation):
    return SIMtools.mitim_simulation._run_settings(_sim(), run_type, {"allocation": allocation, "extra_name": ""})


def test_submit_without_minutes_raises():
    try:
        _settings("submit", {"resources_per_call": 4})
    except ValueError as e:
        assert "minutes" in str(e), e
    else:
        raise AssertionError("submit without minutes did not raise")
    assert _settings("submit", {"resources_per_call": 4, "minutes": 480}).minutes == 480
    print("PASS test_submit_without_minutes_raises")


def test_other_run_types_default_to_ten_minutes():
    assert _settings("normal", {"resources_per_call": 4}).minutes == 10
    assert _settings("normal", {"resources_per_call": 4, "minutes": 30}).minutes == 30
    print("PASS test_other_run_types_default_to_ten_minutes")


if __name__ == "__main__":
    test_submit_without_minutes_raises()
    test_other_run_types_default_to_ten_minutes()
    print("\nALL PASS")
