"""
test_scheduler_extras.py
========================
Which extra cases of load_balance 'extra_points' the in-allocation scheduler keeps.

An extra is usable when it ran to its end (CGYRO's EXIT line in out.cgyro.info) or when the
watchdog stopped it past min_time (mitim_budget.tag). Until 2026-09-21 only the tag counted, so
extras that finished at MAX_TIME - the best-converged ones - were discarded (engaging
v3a_reduced2_engaging_papercompare, Evaluation 5: three extras at t=450 thrown away). The
"ended on its own" line was also printed again on every poll.

Run as:

    python tests/dev_tests/test_scheduler_extras.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools.utils import SCHEDULERtools
from mitim_modules.powertorch.physics_models import transport_cgyro

MARKER = ("out.cgyro.info", "EXIT")

# extra -> body: ran to MAX_TIME / crashed / stopped by the watchdog past min_time
EXTRAS = {
    "extra/rho_0.3486": 'mkdir -p extra/rho_0.3486; echo "EXIT: (CGYRO) Reached MAX_TIME" > extra/rho_0.3486/out.cgyro.info',
    "extra/rho_0.4808": 'mkdir -p extra/rho_0.4808; echo "ERROR: (CGYRO) something broke" > extra/rho_0.4808/out.cgyro.info; exit 1',
    "extra/rho_0.6673": 'mkdir -p extra/rho_0.6673; echo "STOP t=320" > extra/rho_0.6673/mitim_budget.tag; touch extra/rho_0.6673/out.cgyro.info',
}


def _run(completion_marker):
    d = Path(tempfile.mkdtemp())
    queue = list(EXTRAS.items())
    # each main call ends at once except the last, which runs long enough for every extra to end on its own
    bodies = {f"main/rho_{i}": ("sleep 3" if i == 3 else ":") for i in range(4)}
    sched = SCHEDULERtools.InAllocationScheduler(
        bodies, hosts=[], concurrency=4, poll_seconds=0.2, completion_marker=completion_marker,
        on_call_finished=lambda rel: queue.pop(0) if queue else None)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            result = sched.run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return result, log.getvalue()


def test_finished_extras_are_accepted():
    result, log = _run(MARKER)
    assert sorted(result["accepted"]) == ["extra/rho_0.3486", "extra/rho_0.6673"], (result, log)
    assert result["discarded"] == ["extra/rho_0.4808"], (result, log)
    print("PASS test_finished_extras_are_accepted")


def test_without_completion_marker_only_the_tag_counts():
    result, _ = _run(None)
    assert result["accepted"] == ["extra/rho_0.6673"], result
    print("PASS test_without_completion_marker_only_the_tag_counts")


def test_ended_on_its_own_logged_once():
    _, log = _run(MARKER)
    for rel in EXTRAS:
        n = log.count(f"extra {rel} ended on its own")
        assert n == 1, f"{rel}: logged {n} times\n{log}"
    print("PASS test_ended_on_its_own_logged_once")


def test_reader_accepts_finished_extras():
    '''transport_cgyro reads extras that finished (EXIT) as well as stopped ones (tag).'''
    d = Path(tempfile.mkdtemp())
    try:
        (d / "out.cgyro.info").write_text("INFO: ...\nEXIT: (CGYRO) Reached MAX_TIME\n")
        assert transport_cgyro._extra_point_usable(d)
        (d / "out.cgyro.info").write_text("ERROR: (CGYRO) something broke\n")
        assert not transport_cgyro._extra_point_usable(d)
        (d / "mitim_budget.tag").write_text("STOP t=320\n")
        assert transport_cgyro._extra_point_usable(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_reader_accepts_finished_extras")


if __name__ == "__main__":
    test_finished_extras_are_accepted()
    test_without_completion_marker_only_the_tag_counts()
    test_ended_on_its_own_logged_once()
    test_reader_accepts_finished_extras()
    print("\nALL PASS")
