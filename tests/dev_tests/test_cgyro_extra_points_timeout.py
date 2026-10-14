"""
test_cgyro_extra_points_timeout.py
==================================
Array-mode extra cases when a MAIN radius hits its element wall clock (TIMED_OUT, no mitim_budget.tag).

When the next submission of the evaluation continues that radius in place (rescue_interrupted, with a
driver that re-attaches), the evaluation is not over: its extra cases are left running and handed over
(ArrayExtraPoints._hand_over), the next submission takes them back into its ledger (take_over, through
GKRunControl._extra_points_handed_over), stops them when its own main radius really ends and collects
them. When the radius is NOT continued in place, the extras are stopped as before.

Reuses the fakes of test_cgyro_extra_points_array.py (real shell on a temporary scratch, canned slurm).

Run as:

    python tests/dev_tests/test_cgyro_extra_points_timeout.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

here = Path(__file__).resolve().parent
if str(here.parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(here.parents[1] / "src"))

_spec = importlib.util.spec_from_file_location("extra_points_array_fakes", here / "test_cgyro_extra_points_array.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools

FAST, SLOW, EXTRA = fx.FAST, fx.SLOW, fx.EXTRA
STOP = Path(EXTRA) / "mitim_stop"


def hand_over_sim(d, hand_over=True):
    '''A submission whose fast radius finished and got its extra (job 200), with the hand over allowed or not.'''
    sim = fx.FakeSim(d)
    sim.kwargs_organize["extra_points"]["hand_over"] = hand_over
    fx.quiet_poll(sim, fx.RUNNING_ROWS)
    assert sim._extra_ledger[FAST]["status"] == "RUNNING"
    fx.write_running_extra(d, t=120)
    sim.simulation_job.infoSLURM = {"STATE": "TIMEOUT"}     # the array left the queue
    return sim


def timeout_rows(tag="500"):
    return [fx.row(FAST, "FINISHED", cost=10.0), dict(fx.row(SLOW, "TIMED_OUT", cost=40.0), tag_token=tag)]


def next_submission(d, ledger, continued):
    '''The next driver's fresh submission of the same evaluation: only the slow radius, its own plan.'''
    sim = fx.FakeSim(d)
    sim.kwargs_organize["extra_points"] = {"commands": {SLOW: "cgyro -e extra_cgyro/rho_0.8105"},
                                           "min_time": 300.0, "stop_grace_seconds": 1800, "hand_over": True}
    sim.simulation_job.in_queue.add("200")                # the handed-over extra is still running
    sim.simulation_job.preserve_subfolders = list(continued)
    sim._finished_in_scratch = []
    sim._extra_ledger = ledger                            # what load_submission_state read back
    with contextlib.redirect_stdout(io.StringIO()):
        sim._extra_ledger = CGYROtools.GKRunControl._extra_points_handed_over(sim)
    return sim


def roundtrip(ledger):
    record = SIMtools.SubmissionRecord(code="cgyro", slurm_output="slurm_output.dat",
                                       kwargs_organize={"tmpFolder": "/tmp/x", "filesToRetrieve": [], "code_executor": {}},
                                       extra_ledger=ledger, job={})
    d = Path(tempfile.mkdtemp())
    try:
        return SIMtools.SubmissionRecord.read(record.write(d / "cgyro_submission.json")).extra_ledger
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_timed_out_main_hands_the_extras_over():
    with fx.scratch_with_finished_radius() as d:
        sim = hand_over_sim(d)
        job = sim.simulation_job
        writes = sim.metadata_writes
        fx.quiet_poll(sim, timeout_rows())

        entry = sim._extra_ledger[FAST]
        assert not (d / STOP).exists() and job.cancelled == [], "the extra must keep running"
        assert entry["status"] == "HANDED_OVER" and entry["handed_over_from"] == "RUNNING" and entry["awaits"] == [SLOW], entry
        assert sim.metadata_writes > writes, "the hand over must reach cgyro_submission.json"
        assert SIMtools.mitim_simulation._child_jobids(sim) == [], "check() must not wait for a handed-over extra"

        with contextlib.redirect_stdout(io.StringIO()):
            CGYROtools.ArrayExtraPoints(sim).collect()      # the failing driver's fetch
        assert job.cancelled == [] and not (d / STOP).exists()
        assert sim._extra_ledger[FAST]["status"] == "HANDED_OVER"
        assert job.scheduler_result == {"accepted": [], "discarded": []} and EXTRA not in job.output_folders

        assert roundtrip(sim._extra_ledger)[FAST]["status"] == "HANDED_OVER"
    print("PASS test_timed_out_main_hands_the_extras_over")


def test_next_submission_takes_them_over_and_stops_them_at_its_end():
    with fx.scratch_with_finished_radius() as d:
        sim = hand_over_sim(d)
        fx.quiet_poll(sim, timeout_rows())
        ledger = roundtrip(sim._extra_ledger)

        nxt = next_submission(d, ledger, continued=[SLOW])
        job = nxt.simulation_job
        entry = nxt._extra_ledger[FAST]
        assert entry["status"] == "RUNNING" and "released" not in entry and "awaits" not in entry, entry
        assert EXTRA in job.preserve_subfolders, "a scratch wipe going in must keep the extra's folder"
        assert SIMtools.mitim_simulation._child_jobids(nxt) == ["200"], "its poll waits for the extra again"

        running = [fx.row(SLOW, "RUNNING", cost=40.0, remaining=9000.0)]
        fx.quiet_poll(nxt, running)                         # the rescued main integrates: extra untouched
        assert not (d / STOP).exists() and job.cancelled == [] and job.submitted == []

        fx.quiet_poll(nxt, [fx.row(SLOW, "FINISHED", cost=40.0)])
        assert (d / STOP).exists() and nxt._extra_ledger[FAST]["status"] == "STOPPING"

        fx.write_running_extra(d, t=330, tag="mitim_budget.tag")
        job.in_queue.clear()
        fx.quiet_poll(nxt, [fx.row(SLOW, "FINISHED", cost=40.0)])
        assert nxt._extra_ledger[FAST]["status"] == "ACCEPTED"
        with contextlib.redirect_stdout(io.StringIO()):
            CGYROtools.ArrayExtraPoints(nxt).collect()
        assert EXTRA in job.output_folders and job.scheduler_result == {"accepted": [EXTRA], "discarded": []}
    print("PASS test_next_submission_takes_them_over_and_stops_them_at_its_end")


def test_timeout_not_continued_stops_the_extras_as_before():
    with fx.scratch_with_finished_radius() as d:
        # no hand over allowed (rescue_interrupted off, or a driver that does not re-attach)
        sim = hand_over_sim(d, hand_over=False)
        fx.quiet_poll(sim, timeout_rows())
        assert (d / STOP).exists() and sim._extra_ledger[FAST]["status"] == "STOPPING"
        assert SIMtools.mitim_simulation._child_jobids(sim) == ["200"]

    with fx.scratch_with_finished_radius() as d:
        # timed out before its first restart write: nothing to continue in place
        sim = hand_over_sim(d)
        fx.quiet_poll(sim, timeout_rows(tag="-"))
        assert (d / STOP).exists() and sim._extra_ledger[FAST]["status"] == "STOPPING"

    with fx.scratch_with_finished_radius() as d:
        # handed over, but the next submission does not continue the radius in place (rescue refused)
        sim = hand_over_sim(d)
        fx.quiet_poll(sim, timeout_rows())
        nxt = next_submission(d, roundtrip(sim._extra_ledger), continued=[])
        assert nxt._extra_ledger[FAST].get("released") is True
        fx.quiet_poll(nxt, [fx.row(SLOW, "RUNNING", cost=40.0, remaining=9000.0)])
        assert (d / STOP).exists() and nxt._extra_ledger[FAST]["status"] == "STOPPING"
    print("PASS test_timeout_not_continued_stops_the_extras_as_before")


def test_plan_allows_the_hand_over_only_with_rescue_and_reattach():
    script = types.SimpleNamespace(folders=[FAST, SLOW], _redirect=lambda f: "", _call=lambda f, **k: f"body {f}")
    settings = types.SimpleNamespace(run_type=SIMtools.RunType.SUBMIT)
    for rescue, reattach, expected in ((True, True, True), (True, False, False), (False, True, False)):
        sim = types.SimpleNamespace(_load_balance={"strategy": "extra_points", "min_time": 300},
                                    extra_point_builder=lambda *a: None, _rescue_interrupted=rescue, _reattach_enabled=reattach)
        with contextlib.redirect_stdout(io.StringIO()):
            plan = CGYROtools.CGYRO._extra_point_array_plan(sim, script, settings)
        assert plan["hand_over"] is expected, (rescue, reattach, plan)
    assert SIMtools.mitim_simulation._extra_points_handed_over(None) == {}
    print("PASS test_plan_allows_the_hand_over_only_with_rescue_and_reattach")


if __name__ == "__main__":
    test_timed_out_main_hands_the_extras_over()
    test_next_submission_takes_them_over_and_stops_them_at_its_end()
    test_timeout_not_continued_stops_the_extras_as_before()
    test_plan_allows_the_hand_over_only_with_rescue_and_reattach()
