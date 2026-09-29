"""
test_scheduler_extras.py
========================
Which extra cases of load_balance 'extra_points' the in-allocation scheduler keeps.

An extra is usable when it ran to its end (CGYRO's EXIT line in out.cgyro.info) or when the
watchdog stopped it past min_time (mitim_budget.tag). Until 2026-09-21 only the tag counted, so
extras that finished at MAX_TIME - the best-converged ones - were discarded (engaging
v3a_reduced2_engaging_papercompare, Evaluation 5: three extras at t=450 thrown away). The
"ended on its own" line was also printed again on every poll.

Slots that no main call takes (a rescue relaunching only the radius an earlier driver job left
unfinished) must be offered extras too, built from the radii that did finish, under the same
idle-window test (engaging v3a_reduced2_engaging_papercompare, driver 23495963, 2026-09-23: four
of five nodes idle for ~1.5 h while one radius re-ran, no extra launched).

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
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
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


def _run_idle(remaining, needed, sources=("main/rho_1", "main/rho_2")):
    '''One main call on a 4-slot allocation; the extras record the MITIM_CALL (slot) they got.'''
    d = Path(tempfile.mkdtemp())
    calls = d / "calls"

    def extra(rel):
        rel_extra = rel.replace("main/", "extra/")
        return rel_extra, f'mkdir -p {rel_extra}; echo "$MITIM_CALL" >> {calls}; echo "EXIT: (CGYRO) Normal" > {rel_extra}/out.cgyro.info'

    sched = SCHEDULERtools.InAllocationScheduler(
        {"main/rho_0": "sleep 2"}, hosts=[], concurrency=4, poll_seconds=0.2, completion_marker=MARKER,
        on_call_finished=extra, estimate_remaining=lambda rel: remaining, estimate_to_accept=lambda rel: needed,
        idle_slot_sources=lambda mains: list(sources))
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            result = sched.run(d)
        slots = sorted(int(x) for x in calls.read_text().split()) if calls.exists() else []
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return result, slots, log.getvalue()


def test_idle_slots_get_extras_from_finished_calls():
    result, slots, log = _run_idle(remaining=3600.0, needed=600.0)
    assert sorted(result["accepted"]) == ["extra/rho_1", "extra/rho_2"], (result, log)
    assert slots == [2, 3], f"extras must take the slots the main call (call 1) does not hold: {slots}\n{log}"
    assert "on the idle slot 2" in log and "on the idle slot 3" in log, log
    print("PASS test_idle_slots_get_extras_from_finished_calls")


def test_idle_slots_respect_the_window():
    result, slots, log = _run_idle(remaining=60.0, needed=600.0)
    assert result == {"accepted": [], "discarded": []} and slots == [], (result, log)
    for rel in ("main/rho_1", "main/rho_2"):
        assert log.count(f"{rel} idle window") == 1, f"{rel}: declined more than once or never\n{log}"
    print("PASS test_idle_slots_respect_the_window")


def test_idle_slots_wait_for_an_estimate():
    '''Before the running main call reports its remaining time, the window test cannot run: no extra.'''
    result, slots, log = _run_idle(remaining=None, needed=600.0)
    assert slots == [] and "idle window" not in log, (result, log)
    print("PASS test_idle_slots_wait_for_an_estimate")


def test_cgyro_stages_finished_radii():
    '''CGYRO offers the radii stored as finished (not the relaunched one, not one whose extra is done),
    symlinked into scratch under their plain names.'''
    d = Path(tempfile.mkdtemp())
    try:
        local, scratch = d / "local" / "base_cgyro", d / "scratch"
        local.mkdir(parents=True)
        for rho, info in ((0.3486, "[t: 2.050E+02]"), (0.4808, "EXIT: (CGYRO) Normal"),
                          (0.6673, "EXIT: (CGYRO) Normal"), (0.8028, "[t: 1.000E+02]")):
            (local / f"out.cgyro.info_{rho:.4f}").write_text(info + "\n")
            (local / f"out.cgyro.timing_{rho:.4f}").write_text("1 2 3 12.5\n")
            (local / f"bin.cgyro.restart_{rho:.4f}").write_bytes(b"blob")
        done = d / "local" / "extra_cgyro" / "rho_0.6673"
        done.mkdir(parents=True)
        (done / "out.cgyro.info").write_text("EXIT: (CGYRO) Normal\n")

        fake = types.SimpleNamespace(
            FolderGACODE=d / "local",
            run_specifications={"completion_marker": ("out.cgyro.info", "EXIT"), "completion_alt_file": "mitim_budget.tag"},
            _scratch=lambda rel: scratch / rel,
            _stage_finished_radius=CGYROtools.CGYRO._stage_finished_radius)
        sources = CGYROtools.CGYRO._idle_slot_sources(fake, ["base_cgyro/rho_0.3486"])
        assert sources == ["base_cgyro/rho_0.4808"], sources
        staged = scratch / "base_cgyro" / "rho_0.4808"
        assert (staged / "out.cgyro.timing").is_symlink() and (staged / "bin.cgyro.restart").read_bytes() == b"blob"
        assert "EXIT" in (staged / "out.cgyro.info").read_text()
        # a second call (scratch folder already there) stages nothing new and still offers it
        assert CGYROtools.CGYRO._idle_slot_sources(fake, ["base_cgyro/rho_0.3486"]) == sources
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_cgyro_stages_finished_radii")


if __name__ == "__main__":
    test_finished_extras_are_accepted()
    test_without_completion_marker_only_the_tag_counts()
    test_ended_on_its_own_logged_once()
    test_reader_accepts_finished_extras()
    test_idle_slots_get_extras_from_finished_calls()
    test_idle_slots_respect_the_window()
    test_idle_slots_wait_for_an_estimate()
    test_cgyro_stages_finished_radii()
    print("\nALL PASS")
