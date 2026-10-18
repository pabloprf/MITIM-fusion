"""
test_cgyro_extra_points_array.py
================================
load_balance 'extra_points' when every radius is its own slurm array element (run_type 'submit').

In bash mode the in-allocation scheduler gives the node of a radius that finished early to one extra
case. In array mode the finished element releases its node, so the driver's poll submits the extra as
a standalone job (CGYROtools.ArrayExtraPoints). These tests cover that bookkeeping without a cluster:
when an extra is submitted (idle window, pending mains, max_concurrent_calls), how it is stopped when
the last main radius ends, how it is classified, what fetch() retrieves, and what a re-attaching
driver reads back from cgyro_submission.json.

The fake job runs the real shell snippets (the probe template, the staging mkdir/cp, the stop file)
against a temporary "scratch" folder, and cans only the slurm commands.

Run as:

    python tests/dev_tests/test_cgyro_extra_points_array.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools import FARMINGtools

FAST, SLOW = "base_cgyro/rho_0.3360", "base_cgyro/rho_0.8105"
EXTRA = "extra_cgyro/rho_0.3360"
TRAILER = FARMINGtools.SqueueAnswer.TRAILER


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeJob:
    """mitim_job stand-in on a shared filesystem (ssh None): real shell, canned slurm."""

    def __init__(self, scratch):
        self.jobid = "100"
        self.folderExecution = str(scratch)
        self.folder_local = Path(scratch) / "_local"
        self.launchSlurm = True
        self.ssh = None
        self.infoSLURM = {"STATE": "RUNNING"}
        self.slurm_settings = {"job-name": "cgyro_E_pt03_ev0"}
        self.output_folders = [FAST, SLOW]
        self.output_folders_selective = {FAST: ["out.cgyro.info"], SLOW: ["out.cgyro.info"]}
        self.scheduler_result = None
        self.in_queue = set()          # jobids squeue still lists
        self.squeue_answers = True     # False: slurmctld gave no verdict
        self.next_jobid = 200
        self.submitted, self.cancelled, self.put = [], [], []

    @contextlib.contextmanager
    def session(self, **kwargs):
        yield self

    def execute(self, cmd, printYN=False):
        if cmd.startswith("squeue"):
            if not self.squeue_answers:
                return b"", b"slurm_load_jobs error: Socket timed out"
            ids = cmd.split("-j")[1].split()[0].split(",")
            table = "".join(f"{j:>15} {'RUNNING':>10}\n" for j in ids if j in self.in_queue)
            return f"{table}{TRAILER}0".encode(), b""
        if cmd.startswith("scancel"):
            jobid = cmd.split()[1]
            self.cancelled.append(jobid)
            self.in_queue.discard(jobid)
            return b"", b""
        done = subprocess.run(["bash", "-c", cmd], capture_output=True)
        return done.stdout, done.stderr

    def put_file(self, local_file, remote_path):
        self.put.append((str(local_file), str(remote_path)))
        shutil.copy2(local_file, remote_path)

    def resubmit_single_task(self, code_call_str, label, exclude_node=None):
        jobid = str(self.next_jobid)
        self.next_jobid += 1
        self.submitted.append((code_call_str, label, jobid))
        self.in_queue.add(jobid)
        return jobid


class FakeSim:
    def __init__(self, scratch, min_time=300.0):
        self.simulation_job = FakeJob(scratch)
        self.FolderGACODE = Path(scratch) / "_results"
        self.run_specifications = {"completion_marker": ("out.cgyro.info", "EXIT"), "completion_alt_file": "mitim_budget.tag"}
        self.kwargs_organize = {
            "filesToRetrieve": ["out.cgyro.info"],
            "extra_points": {"commands": {FAST: f"cgyro -e {EXTRA}", SLOW: "cgyro -e extra_cgyro/rho_0.8105"},
                             "min_time": min_time, "stop_grace_seconds": 1800},
        }
        self._resubmit_ledger, self._extra_ledger = {}, {}
        self._base_subfolder = "base_cgyro"
        self.metadata_writes = 0
        self.built = []

    def extra_point_builder(self, rho, finished_dir, local_dir):
        self.built.append((rho, Path(finished_dir)))
        input_cgyro = Path(local_dir) / "input.cgyro_built"
        input_cgyro.write_text("MAX_TIME = 1000\n")
        return input_cgyro

    def _write_submission_metadata(self, base_subfolder):
        self.metadata_writes += 1


def row(folder, effective, cost=None, remaining=None):
    return {"folder": folder, "effective": effective, "cost_last": cost, "remaining_s": remaining}


@contextlib.contextmanager
def scratch_with_finished_radius():
    d = Path(tempfile.mkdtemp())
    (d / FAST).mkdir(parents=True)
    (d / FAST / "bin.cgyro.restart").write_text("blob")
    (d / FAST / "out.cgyro.tag").write_text("1000\n")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def quiet_poll(sim, rows):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        CGYROtools.ArrayExtraPoints(sim, session=sim.simulation_job).poll(rows)
    return log.getvalue()


def write_running_extra(scratch, t=120, tag=None, started=True):
    """Extra folder as CGYRO leaves it: info + timing + time (t a/cs done), optionally a watchdog tag."""
    d = Path(scratch) / EXTRA
    d.mkdir(parents=True, exist_ok=True)
    if started:
        (d / "out.cgyro.info").write_text("init\n")
        (d / "out.cgyro.timing").write_text("Run time\n" + "".join("1 " * 13 + "20.0\n" for _ in range(t)))
        (d / "out.cgyro.time").write_text("".join(f"{i + 1} 0\n" for i in range(t)))
    if tag is not None:
        (d / tag).write_text("STOP t=320\n")


RUNNING_ROWS = [row(FAST, "FINISHED", cost=10.0), row(SLOW, "RUNNING", cost=40.0, remaining=10000.0)]
OVER_ROWS = [row(FAST, "FINISHED", cost=10.0), row(SLOW, "FINISHED", cost=40.0)]


# ---------------------------------------------------------------------------
# The probe: what is left of a running launch
# ---------------------------------------------------------------------------


def test_probe_reports_remaining_time():
    with scratch_with_finished_radius() as d:
        r = d / SLOW
        r.mkdir(parents=True)
        (r / "out.cgyro.info").write_text("init\n")
        (r / "out.cgyro.timing").write_text("Run time\n" + "".join("1 " * 13 + f"{c}\n" for c in (30.0, 50.0)))
        (r / "out.cgyro.time").write_text("601 0\n650 0\n")
        (r / ".mitim_t0").write_text("600\n")            # in-place continuation from t=600
        (r / "input.cgyro").write_text("N_RADIAL=4\nMAX_TIME = 400\n")
        lines = CGYROtools.CgyroProbe(FakeJob(d), session=FakeJob(d)).run([SLOW, "base_cgyro/rho_0.9999"])
        status = CGYROtools.RadiusStatus.from_probe_line(lines[0])
        pending = CGYROtools.RadiusStatus.from_probe_line(lines[1])
    # (MAX_TIME - (t - t0)) a/cs at the LAST step cost: (400 - 50) * 50 s
    assert status.remaining_seconds == 350 * 50.0, lines
    assert status.to_row()["cost_last"] == 50.0 and status.to_row()["remaining_s"] == 17500.0
    assert pending.raw == CGYROtools.TaskState.NOT_STARTED and pending.remaining_seconds is None
    # a probe line of the older 8-field shape still parses
    old = CGYROtools.RadiusStatus.from_probe_line(f"{SLOW}|RUNNING|40.0|10|100|5|-|0")
    assert old.remaining_seconds is None and old.state == CGYROtools.TaskState.RUNNING
    print("PASS test_probe_reports_remaining_time")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def test_finished_radius_gets_one_extra():
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)
        log = quiet_poll(sim, RUNNING_ROWS)
        job = sim.simulation_job
        # needed = 300 a/cs * 10 s + 300 s = 3300 s < 10000 s left on the slow radius
        assert job.submitted == [(f"cgyro -e {EXTRA}", "_extra_rho_0.3360", "200")], (job.submitted, log)
        assert sim.built == [(0.336, d / FAST)], sim.built      # shared filesystem: read in scratch
        assert (d / EXTRA / "input.cgyro").read_text() == "MAX_TIME = 1000\n"
        assert (d / EXTRA / "bin.cgyro.restart").read_text() == "blob"
        assert not (d / EXTRA / "out.cgyro.tag").exists()        # warm start, t from 0
        assert sim._extra_ledger[FAST]["status"] == "RUNNING" and sim._extra_ledger[FAST]["jobid"] == "200"
        assert sim.metadata_writes == 1
        # the next poll does not submit a second one
        quiet_poll(sim, RUNNING_ROWS)
        assert len(job.submitted) == 1
    print("PASS test_finished_radius_gets_one_extra")


def test_short_idle_window_is_skipped_for_good():
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)
        rows = [row(FAST, "FINISHED", cost=10.0), row(SLOW, "RUNNING", cost=40.0, remaining=3000.0)]
        log = quiet_poll(sim, rows)
        assert sim.simulation_job.submitted == [] and sim.built == []
        assert sim._extra_ledger[FAST]["status"] == "SKIPPED" and "idle window" in sim._extra_ledger[FAST]["reason"], log
        quiet_poll(sim, RUNNING_ROWS)   # a longer estimate later does not reopen it
        assert sim.simulation_job.submitted == []
    print("PASS test_short_idle_window_is_skipped_for_good")


def test_no_extra_while_a_main_radius_waits_for_a_node():
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)
        rows = [row(FAST, "FINISHED", cost=10.0), row(SLOW, "NOT_STARTED")]
        quiet_poll(sim, rows)
        assert sim.simulation_job.submitted == [] and sim._extra_ledger == {}
        quiet_poll(sim, RUNNING_ROWS)   # once it runs and has an estimate, the extra goes
        assert len(sim.simulation_job.submitted) == 1
    print("PASS test_no_extra_while_a_main_radius_waits_for_a_node")


def test_max_concurrent_calls_bounds_mains_plus_extras():
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)
        sim.simulation_job.slurm_settings["array_limit"] = 1
        quiet_poll(sim, RUNNING_ROWS)
        assert sim.simulation_job.submitted == [] and sim._extra_ledger == {}
        sim.simulation_job.slurm_settings["array_limit"] = 2
        quiet_poll(sim, RUNNING_ROWS)
        assert len(sim.simulation_job.submitted) == 1
    print("PASS test_max_concurrent_calls_bounds_mains_plus_extras")


def test_builder_failure_leaves_the_node_alone():
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)
        sim.extra_point_builder = lambda rho, finished_dir, local_dir: (_ for _ in ()).throw(ValueError("no fluxes"))
        quiet_poll(sim, RUNNING_ROWS)
        assert sim.simulation_job.submitted == []
        assert sim._extra_ledger[FAST]["status"] == "SKIPPED" and "no fluxes" in sim._extra_ledger[FAST]["reason"]
    print("PASS test_builder_failure_leaves_the_node_alone")


def test_without_the_strategy_nothing_happens():
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)
        sim.kwargs_organize["extra_points"] = None
        quiet_poll(sim, RUNNING_ROWS)
        assert sim.simulation_job.submitted == [] and sim._extra_ledger == {} and sim.metadata_writes == 0
    print("PASS test_without_the_strategy_nothing_happens")


# ---------------------------------------------------------------------------
# Stop and classification
# ---------------------------------------------------------------------------


def _launched(d):
    sim = FakeSim(d)
    quiet_poll(sim, RUNNING_ROWS)
    assert sim._extra_ledger[FAST]["status"] == "RUNNING"
    return sim


def test_stop_when_the_last_main_radius_ends():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        job = sim.simulation_job
        write_running_extra(d, t=320)

        quiet_poll(sim, RUNNING_ROWS)                     # mains still running: no stop
        assert not (d / EXTRA / "mitim_stop").exists()

        quiet_poll(sim, OVER_ROWS)
        assert (d / EXTRA / "mitim_stop").exists() and job.cancelled == []
        assert sim._extra_ledger[FAST]["status"] == "STOPPING"
        assert SIMtools.mitim_simulation._child_jobids(sim) == ["200"]   # check() keeps polling

        # the watchdog stopped it past min_time and the job left the queue
        write_running_extra(d, t=330, tag="mitim_budget.tag")
        job.in_queue.clear()
        quiet_poll(sim, OVER_ROWS)
        assert sim._extra_ledger[FAST]["status"] == "ACCEPTED"
        assert SIMtools.mitim_simulation._child_jobids(sim) == []        # nothing holds the poll loop
    print("PASS test_stop_when_the_last_main_radius_ends")


def test_extra_below_min_time_is_discarded():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        write_running_extra(d, t=100)
        quiet_poll(sim, OVER_ROWS)
        write_running_extra(d, t=100, tag="mitim_discard.tag")
        sim.simulation_job.in_queue.clear()
        quiet_poll(sim, OVER_ROWS)
        assert sim._extra_ledger[FAST]["status"] == "DISCARDED"
    print("PASS test_extra_below_min_time_is_discarded")


def test_extra_that_ran_to_max_time_is_accepted_while_mains_run():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        write_running_extra(d, t=1000)
        (d / EXTRA / "out.cgyro.info").write_text("init\nEXIT: (CGYRO) Reached MAX_TIME\n")
        sim.simulation_job.in_queue.clear()
        quiet_poll(sim, RUNNING_ROWS)
        assert sim._extra_ledger[FAST]["status"] == "ACCEPTED"
        assert len(sim.simulation_job.submitted) == 1    # one extra per radius, not a second one
    print("PASS test_extra_that_ran_to_max_time_is_accepted_while_mains_run")


def test_pending_extra_is_cancelled_when_mains_end():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        write_running_extra(d, started=False)             # still waiting for a node
        quiet_poll(sim, OVER_ROWS)
        assert sim.simulation_job.cancelled == ["200"]
        assert sim._extra_ledger[FAST]["status"] == "DISCARDED"
        assert not (d / EXTRA / "mitim_stop").exists()
    print("PASS test_pending_extra_is_cancelled_when_mains_end")


def test_extra_that_ignores_the_stop_is_cancelled_after_the_grace():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        write_running_extra(d, t=320)
        quiet_poll(sim, OVER_ROWS)
        (d / EXTRA / "mitim_stop").unlink()               # a requeued extra clears it at restart
        quiet_poll(sim, OVER_ROWS)
        assert (d / EXTRA / "mitim_stop").exists() and sim.simulation_job.cancelled == []   # dropped again, within grace
        sim._extra_ledger[FAST]["stop_requested_at"] = "2020-01-01T00:00:00Z"
        quiet_poll(sim, OVER_ROWS)
        assert sim.simulation_job.cancelled == ["200"]
        quiet_poll(sim, OVER_ROWS)                        # gone from the queue, no tag
        assert sim._extra_ledger[FAST]["status"] == "DISCARDED"
    print("PASS test_extra_that_ignores_the_stop_is_cancelled_after_the_grace")


def test_no_answer_from_slurm_keeps_the_extra_in_flight():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        write_running_extra(d, t=320)
        sim.simulation_job.in_queue.clear()
        sim.simulation_job.squeue_answers = False
        quiet_poll(sim, RUNNING_ROWS)
        assert sim._extra_ledger[FAST]["status"] == "RUNNING"
    print("PASS test_no_answer_from_slurm_keeps_the_extra_in_flight")


def test_radius_on_a_rescue_job_keeps_the_extras_running():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        write_running_extra(d, t=320)
        sim.simulation_job.infoSLURM = {"STATE": "NOT FOUND"}       # the array left the queue...
        sim._resubmit_ledger[SLOW] = {"n_attempts": 1, "child_jobids": ["150"], "status": "active"}
        rows = [row(FAST, "FINISHED", cost=10.0), row(SLOW, "RUNNING", cost=40.0, remaining=5000.0)]
        quiet_poll(sim, rows)                                       # ...but the slow radius runs as a rescue job
        assert not (d / EXTRA / "mitim_stop").exists()
        del sim._resubmit_ledger[SLOW]
        quiet_poll(sim, rows)                                       # array gone and no rescue: mains are over
        assert (d / EXTRA / "mitim_stop").exists()
    print("PASS test_radius_on_a_rescue_job_keeps_the_extras_running")


# ---------------------------------------------------------------------------
# Fetch and re-attach
# ---------------------------------------------------------------------------


def test_collect_adds_accepted_extras_to_the_retrieval():
    with scratch_with_finished_radius() as d:
        sim = _launched(d)
        job = sim.simulation_job
        write_running_extra(d, t=330, tag="mitim_budget.tag")
        with contextlib.redirect_stdout(io.StringIO()):
            CGYROtools.ArrayExtraPoints(sim).collect()              # still in the queue at fetch time
        assert job.cancelled == ["200"]
        assert sim._extra_ledger[FAST]["status"] == "ACCEPTED"
        assert job.output_folders == [FAST, SLOW, EXTRA]
        assert job.output_folders_selective[EXTRA] == ["out.cgyro.info"]
        assert job.scheduler_result == {"accepted": [EXTRA], "discarded": []}   # what _organize_results stores
        with contextlib.redirect_stdout(io.StringIO()):
            CGYROtools.ArrayExtraPoints(sim).collect()              # a second fetch does not duplicate
        assert job.output_folders == [FAST, SLOW, EXTRA]
    print("PASS test_collect_adds_accepted_extras_to_the_retrieval")


def test_submission_record_carries_the_extras():
    record = SIMtools.SubmissionRecord(
        code="cgyro", slurm_output="slurm_output.dat",
        kwargs_organize={"tmpFolder": "/tmp/x", "filesToRetrieve": [], "code_executor": {},
                         "extra_points": {"commands": {FAST: "cgyro"}, "min_time": 300.0, "stop_grace_seconds": 1800}},
        extra_ledger={FAST: {"folder": EXTRA, "jobid": "200", "status": "RUNNING"}},
        job={})
    d = Path(tempfile.mkdtemp())
    try:
        back = SIMtools.SubmissionRecord.read(record.write(d / "cgyro_submission.json"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert back.extra_ledger == {FAST: {"folder": EXTRA, "jobid": "200", "status": "RUNNING"}}
    assert back.kwargs_organize["extra_points"]["commands"] == {FAST: "cgyro"}
    print("PASS test_submission_record_carries_the_extras")


# ---------------------------------------------------------------------------
# The plan built at submit time
# ---------------------------------------------------------------------------


def _plan(load_balance, run_type="submit", builder=lambda *a: None):
    calls = []
    script = types.SimpleNamespace(
        folders=[FAST, SLOW],
        _redirect=lambda folder: f"1> {folder}/slurm_output.dat",
        _call=lambda folder, **kwargs: calls.append((folder, kwargs)) or f"body {folder}")
    sim = types.SimpleNamespace(_load_balance=load_balance, extra_point_builder=builder)
    settings = types.SimpleNamespace(run_type=SIMtools.RunType.parse(run_type))
    with contextlib.redirect_stdout(io.StringIO()):
        return CGYROtools.CGYRO._extra_point_array_plan(sim, script, settings), calls


def test_plan_holds_one_stop_mode_body_per_radius():
    plan, calls = _plan({"strategy": "extra_points", "min_time": 250})
    assert plan["commands"] == {FAST: f"body {EXTRA}", SLOW: "body extra_cgyro/rho_0.8105"}
    assert plan["min_time"] == 250.0
    # STOP watchdog (discard below min_time, no cost guard), output redirected into the extra's own folder
    assert calls[0] == (EXTRA, {"watchdog": "stop", "additional_command": f"1> {EXTRA}/slurm_output.dat"})
    print("PASS test_plan_holds_one_stop_mode_body_per_radius")


def test_no_plan_without_strategy_polling_or_builder():
    assert _plan({"strategy": "wall_budget"})[0] is None
    assert _plan(None)[0] is None
    assert _plan({"strategy": "extra_points"}, run_type="normal")[0] is None   # the driver does not poll
    assert _plan({"strategy": "extra_points"}, builder=None)[0] is None        # standalone CGYRO run
    print("PASS test_no_plan_without_strategy_polling_or_builder")


def test_builder_reduces_per_radius_lists_to_its_radius():
    from mitim_modules.powertorch.physics_models.utils.cgyro_extra_points import ExtraPointHarvester
    options = {"RESTART_STEP": [25, 25, 10], "MAX_TIME": 1000.0, "COLLISION_MODEL": 4}
    assert ExtraPointHarvester._at_radius(options, 2) == {"RESTART_STEP": 10, "MAX_TIME": 1000.0, "COLLISION_MODEL": 4}
    assert ExtraPointHarvester._at_radius(None, 0) == {}
    print("PASS test_builder_reduces_per_radius_lists_to_its_radius")


if __name__ == "__main__":
    test_builder_reduces_per_radius_lists_to_its_radius()
    test_probe_reports_remaining_time()
    test_finished_radius_gets_one_extra()
    test_short_idle_window_is_skipped_for_good()
    test_no_extra_while_a_main_radius_waits_for_a_node()
    test_max_concurrent_calls_bounds_mains_plus_extras()
    test_builder_failure_leaves_the_node_alone()
    test_without_the_strategy_nothing_happens()
    test_stop_when_the_last_main_radius_ends()
    test_extra_below_min_time_is_discarded()
    test_extra_that_ran_to_max_time_is_accepted_while_mains_run()
    test_pending_extra_is_cancelled_when_mains_end()
    test_extra_that_ignores_the_stop_is_cancelled_after_the_grace()
    test_no_answer_from_slurm_keeps_the_extra_in_flight()
    test_radius_on_a_rescue_job_keeps_the_extras_running()
    test_collect_adds_accepted_extras_to_the_retrieval()
    test_submission_record_carries_the_extras()
    test_plan_holds_one_stop_mode_body_per_radius()
    test_no_plan_without_strategy_polling_or_builder()
    print("\nALL TESTS PASSED")
