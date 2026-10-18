"""
test_cgyro_extras_yield.py
==========================
Extra cases (load_balance 'extra_points', slurm_array) yield to the main radii, and every sbatch of an
evaluation respects a per-user qos cap.

    nice                    extras are submitted with sbatch --nice=<nice>; array elements and rescues are not.
    yield_to_pending_mains  no extra is submitted while squeue shows a main or rescue GK job of ANY run of the
                            user waiting for nodes (a pending extra, or an array throttled by its own %N, does not count).
    rescues_first           no extra while a rescue job of THIS evaluation (resubmit ledger) is pending.
    squeue unreadable       no extra, said once in the log.
    qos_cap / fallback_qos  before an sbatch, the user's jobs in the configured qos + the new ones > cap -> --qos=<fallback>;
                            unset -> no squeue and the same sbatch text and command as before.

No cluster: the poll tests reuse the fakes of test_cgyro_extra_points_array.py; the sbatch tests run the real
mitim_job.resubmit_single_task / full_process on the local machine with a canned execute().

Run as:

    python tests/dev_tests/test_cgyro_extras_yield.py

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mitim_tools.misc_tools import FARMINGtools
from mitim_tools.simulation_tools import SIMtools
from test_cgyro_extra_points_array import (FakeJob, FakeSim, FAST, SLOW, RUNNING_ROWS, TRAILER, quiet_poll,
                                           scratch_with_finished_radius, _plan)

YIELD_PLAN = {"nice": 100, "yield_to_pending_mains": True, "rescues_first": True}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class QueueJob(FakeJob):
    """FakeJob that also answers the user-wide squeue of pending jobs."""

    def __init__(self, scratch):
        super().__init__(scratch)
        self.pending = {}               # jobid -> (name, reason)
        self.user_squeue_answers = True
        self.user_squeues = 0

    def execute(self, cmd, printYN=False):
        if cmd.startswith("squeue -h -u"):
            self.user_squeues += 1
            if not self.user_squeue_answers:
                return b"", b"slurm_load_jobs error: Socket timed out"
            table = "".join(f"{j}|{name}|{reason}\n" for j, (name, reason) in self.pending.items())
            return f"{table}{TRAILER}0".encode(), b""
        return super().execute(cmd, printYN=printYN)


def yield_sim(d, **plan):
    sim = FakeSim(d)
    sim.simulation_job = QueueJob(d)
    sim.kwargs_organize["extra_points"].update(YIELD_PLAN, **plan)
    return sim


def real_job(d, qos_cap=None, in_qos=0):
    """A local mitim_job whose remote commands are canned: squeue counts `in_qos` jobs, sbatch prints a jobid."""
    job = FARMINGtools.mitim_job(d / "local")
    job.folder_local.mkdir(parents=True)
    job.folderExecution = str(d / "exec")
    Path(job.folderExecution).mkdir()
    job.launchSlurm = True
    job.machineSettings = {"machine": "local", "modules": None, "slurm": {"partition": "gpu", "qos": "premium"}}
    job.slurm_settings = {"job-name": "cgyro_E_pt03_ev0", "time": "01:00:00", "array": "0,1,2", "nodes": 1}
    job.qos_cap = qos_cap
    job.run_in_place = False
    job.commands = []

    def execute(cmd, printYN=False, **kwargs):
        job.commands.append(cmd)
        if cmd.startswith("squeue"):
            return ("".join(f"{1000 + i}\n" for i in range(in_qos)) + f"{TRAILER}0").encode(), b""
        return b"12345\n", b""

    job.execute = execute
    return job


def sbatch_text(job, label):
    return (job.folder_local / f"mitim_bash{label}.src").read_text()


@contextlib.contextmanager
def tmp():
    d = Path(tempfile.mkdtemp())
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# nice
# ---------------------------------------------------------------------------


def test_extras_carry_nice_rescues_do_not():
    with scratch_with_finished_radius() as d:
        sim = yield_sim(d)
        quiet_poll(sim, RUNNING_ROWS)
        assert sim.simulation_job.directives == [["--nice=100"]], sim.simulation_job.directives
    with scratch_with_finished_radius() as d:
        sim = FakeSim(d)          # a plan written before the knobs: no nice, no user squeue
        quiet_poll(sim, RUNNING_ROWS)
        assert sim.simulation_job.directives == [None]
    with tmp() as d:
        job = real_job(d)
        assert job.resubmit_single_task("cgyro -e rho", "_extra_rho_0.3360", extra_directives=["--nice=100"]) == "12345"
        assert job.resubmit_single_task("cgyro -e rho", "_resubmit_rho_0.3360_a1") == "12345"
        extra, rescue = sbatch_text(job, "_extra_rho_0.3360"), sbatch_text(job, "_resubmit_rho_0.3360_a1")
        directives = [l for l in extra.splitlines() if l.startswith("#SBATCH")]
        assert directives[-1] == "#SBATCH --nice=100" and directives[-2] == "#SBATCH --profile=all", directives
        assert "--nice" not in rescue
        # the same script but for the label and the --nice line
        assert extra.replace("#SBATCH --nice=100\n", "").replace("_extra_rho_0.3360", "_X") == rescue.replace("_resubmit_rho_0.3360_a1", "_X")
    print("PASS test_extras_carry_nice_rescues_do_not")


def test_plan_defaults_yield():
    plan = _plan({"strategy": "extra_points", "min_time": 250})[0]
    assert (plan["nice"], plan["yield_to_pending_mains"], plan["rescues_first"]) == (100, True, True)
    plan = _plan({"strategy": "extra_points", "extra_points": {"nice": None, "yield_to_pending_mains": False, "rescues_first": False}})[0]
    assert (plan["nice"], plan["yield_to_pending_mains"], plan["rescues_first"]) == (None, False, False)
    print("PASS test_plan_defaults_yield")


# ---------------------------------------------------------------------------
# yield_to_pending_mains / rescues_first
# ---------------------------------------------------------------------------


def test_main_job_name_pattern():
    from mitim_tools.simulation_tools.gk_control.extra_points import ArrayExtraPoints
    match = ArrayExtraPoints.MAIN_JOB_NAME.match
    assert match("cgyro_E_pt03_ev3") and match("cgyro_E_pt03_ev3_resubmit_rho_0.5500_a1")
    assert match("cgyro_E_pt03_ev3_continue_rho_0.5500_c2"), "an in-link continuation is a main radius"
    assert not match("cgyro_E_pt03_ev3_extra_rho_0.5500")
    print("PASS test_main_job_name_pattern")


def test_pending_main_of_another_run_holds_the_extra():
    with scratch_with_finished_radius() as d:
        sim = yield_sim(d)
        job = sim.simulation_job
        job.pending = {"900": ("cgyro_OtherRun_ev3", "Priority"), "901": ("cgyro_OtherRun_ev3_resubmit_rho_0.5500_a1", "Resources")}
        log = quiet_poll(sim, RUNNING_ROWS) + quiet_poll(sim, RUNNING_ROWS)
        assert job.submitted == [] and sim._extra_ledger == {}
        assert log.count("main GK job(s) 900, 901 of the user pending") == 1, log      # said once, not every poll
        assert job.user_squeues == 2                                                  # one squeue per poll
        job.pending = {"900": ("cgyro_OtherRun_ev3", "Priority")}                     # the rescue started
        quiet_poll(sim, RUNNING_ROWS)
        assert job.submitted == []
        # what does not wait for a node, or is not a main: another extra, an array held by its own %N, other codes
        job.pending = {"902": ("cgyro_OtherRun_ev3_extra_rho_0.5500", "Priority"), "903_[2-4]": ("cgyro_Other_ev1", "JobArrayTaskLimit"),
                       "904": ("transp_run", "Priority"), "905": ("cgyro_sim", "Priority")}
        quiet_poll(sim, RUNNING_ROWS)
        assert len(job.submitted) == 1 and sim._extra_ledger[FAST]["status"] == "RUNNING"
    print("PASS test_pending_main_of_another_run_holds_the_extra")


def test_own_pending_rescue_holds_the_extra():
    with scratch_with_finished_radius() as d:
        sim = yield_sim(d, yield_to_pending_mains=False)
        job = sim.simulation_job
        sim._resubmit_ledger = {SLOW: {"n_attempts": 1, "child_jobids": ["555"], "status": "active"}}
        job.pending = {"555": ("anything", "Priority")}
        log = quiet_poll(sim, RUNNING_ROWS)
        assert job.submitted == [] and "rescue job(s) 555 of this evaluation pending" in log, log
        job.pending = {"900": ("cgyro_OtherRun_ev3", "Priority")}     # foreign mains ignored with yield off
        quiet_poll(sim, RUNNING_ROWS)
        assert len(job.submitted) == 1
    print("PASS test_own_pending_rescue_holds_the_extra")


def test_unreadable_squeue_holds_the_extra():
    with scratch_with_finished_radius() as d:
        sim = yield_sim(d)
        job = sim.simulation_job
        job.user_squeue_answers = False
        log = quiet_poll(sim, RUNNING_ROWS) + quiet_poll(sim, RUNNING_ROWS)
        assert job.submitted == [] and log.count("gave no answer") == 1, log
        job.user_squeue_answers = True
        quiet_poll(sim, RUNNING_ROWS)
        assert len(job.submitted) == 1
    print("PASS test_unreadable_squeue_holds_the_extra")


def test_no_squeue_without_a_candidate():
    with scratch_with_finished_radius() as d:
        sim = yield_sim(d)
        quiet_poll(sim, [{"folder": FAST, "effective": "RUNNING", "cost_last": 10.0, "remaining_s": 900.0},
                         {"folder": SLOW, "effective": "RUNNING", "cost_last": 40.0, "remaining_s": 10000.0}])
        assert sim.simulation_job.user_squeues == 0
    print("PASS test_no_squeue_without_a_candidate")


def test_pending_jobs_of_user_parses_the_table():
    job = QueueJob(tempfile.gettempdir())
    job.pending = {"7": ("cgyro_A_ev1", "Priority"), "8": ("cgyro_B_ev2", "Dependency"), "9_[0-3]": ("cgyro_C_ev0", "Resources")}
    assert SIMtools.pending_jobs_of_user(job) == {"7": "cgyro_A_ev1", "9_[0-3]": "cgyro_C_ev0"}
    job.user_squeue_answers = False
    assert SIMtools.pending_jobs_of_user(job) is None
    print("PASS test_pending_jobs_of_user_parses_the_table")


# ---------------------------------------------------------------------------
# qos cap
# ---------------------------------------------------------------------------


def test_qos_cap_on_single_task_submissions():
    cap = {"cap": 5, "fallback": "regular"}
    with tmp() as d:
        job = real_job(d, qos_cap=cap, in_qos=4)          # 4 + 1 = 5: at the cap, configured qos
        job.resubmit_single_task("cgyro -e rho", "_extra_rho_0.3360")
        assert job.commands[0].startswith('squeue -h -r -u "$USER" -q premium'), job.commands
        assert job.commands[-1].endswith("sbatch --parsable mitim_bash_extra_rho_0.3360.src"), job.commands
    with tmp() as d:
        job = real_job(d, qos_cap=cap, in_qos=5)          # 5 + 1 > 5: fallback
        job.resubmit_single_task("cgyro -e rho", "_resubmit_rho_0.3360_a1")
        assert job.commands[-1].endswith("sbatch --parsable --qos=regular mitim_bash_resubmit_rho_0.3360_a1.src"), job.commands
        capped = sbatch_text(job, "_resubmit_rho_0.3360_a1").replace(str(d), "D")
    with tmp() as d:
        job = real_job(d)                                 # unset: no squeue, same command and script
        job.resubmit_single_task("cgyro -e rho", "_resubmit_rho_0.3360_a1")
        assert len(job.commands) == 1 and job.commands[0].endswith("sbatch --parsable mitim_bash_resubmit_rho_0.3360_a1.src")
        assert sbatch_text(job, "_resubmit_rho_0.3360_a1").replace(str(d), "D") == capped   # the cap never touches the script
    print("PASS test_qos_cap_on_single_task_submissions")


def array_dispatch(job):
    job.session = lambda **kwargs: contextlib.nullcontext(job)
    job.create_scratch_folder = job.send = lambda: None
    job.retrieve = lambda **kwargs: True
    job.full_process("cd /scratch && ./mitim_shell_executor.sh > mitim.out",
                     removeScratchFolders_goingIn=False, removeScratchFolders_goingOut=False)
    return job.commands[-1]


def test_qos_cap_on_the_array_submission():
    with tmp() as d:
        job = real_job(d, qos_cap={"cap": 5, "fallback": "regular"}, in_qos=2)    # 2 + 3 elements = 5
        assert array_dispatch(job) == "cd /scratch && ./mitim_shell_executor.sh > mitim.out"
    with tmp() as d:
        job = real_job(d, qos_cap={"cap": 5, "fallback": "regular"}, in_qos=3)    # 3 + 3 > 5
        assert array_dispatch(job) == "export SBATCH_QOS=regular && cd /scratch && ./mitim_shell_executor.sh > mitim.out"
    with tmp() as d:
        job = real_job(d)
        assert array_dispatch(job) == "cd /scratch && ./mitim_shell_executor.sh > mitim.out" and len(job.commands) == 1
    print("PASS test_qos_cap_on_the_array_submission")


def test_qos_cap_is_persisted_for_a_reattach():
    assert FARMINGtools.mitim_job(tempfile.gettempdir()).qos_cap is None
    d = Path(tempfile.mkdtemp())
    job = {"folder_local": str(d), "folderExecution": "/x", "jobid": "1", "launchSlurm": True, "slurm_settings": {},
           "machineSettings": {}, "output_files": [], "output_folders": [], "check_files_in_folder": {},
           "output_folders_selective": {}, "log_simulation_file": None, "qos_cap": {"cap": 5, "fallback": "regular"}}
    record = SIMtools.SubmissionRecord(code="cgyro", slurm_output="slurm_output.dat", job=job,
                                       kwargs_organize={"tmpFolder": "/tmp/x", "filesToRetrieve": [], "code_executor": {}})
    try:
        sim = types.SimpleNamespace()
        SIMtools.SubmissionRecord.read(record.write(d / "cgyro_submission.json")).apply_to(sim)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert sim.simulation_job.qos_cap == {"cap": 5, "fallback": "regular"}
    print("PASS test_qos_cap_is_persisted_for_a_reattach")


if __name__ == "__main__":
    test_extras_carry_nice_rescues_do_not()
    test_plan_defaults_yield()
    test_main_job_name_pattern()
    test_pending_main_of_another_run_holds_the_extra()
    test_own_pending_rescue_holds_the_extra()
    test_unreadable_squeue_holds_the_extra()
    test_no_squeue_without_a_candidate()
    test_pending_jobs_of_user_parses_the_table()
    test_qos_cap_on_single_task_submissions()
    test_qos_cap_on_the_array_submission()
    test_qos_cap_is_persisted_for_a_reattach()
    print("\nALL TESTS PASSED")
