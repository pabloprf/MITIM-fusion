"""
CGYRO job names and jobid handling of a detached (run_type='submit') submission.

Checks:
  1. gk_job_name_suffix tags the job with its PORTALS run (MAESTRO: case + beat), keeps "_ev<N>",
     sanitizes to slurm-safe characters and caps long run names with a stable hash; transport_cgyro
     uses it for every GK submission.
  2. An sbatch that refuses the submission raises at once with sbatch's own message, before any
     submission record is written.
  3. An sbatch that went through but whose jobid was not read back is not an error: the record is
     written and check() finds the job by name AND submit folder (test_job_identity_by_folder).
  4. delete() cancels by jobid, the parent and the rescue / extra-case children (handed-over extras
     included), never `scancel -n`.

Naming and the delete children are mitim_7's (development_cgyro_job_names).

Run: python tests/dev_tests/test_cgyro_job_names.py
"""

import contextlib
import io
import shutil
import tempfile
import types
from pathlib import Path

from mitim_modules.powertorch.physics_models import transport_cgyro
from mitim_tools.misc_tools import FARMINGtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.gk_control.submission import gk_job_name_suffix, _JOB_TAG_MAX_LENGTH


def _quiet(fn, *args, **kwargs):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = fn(*args, **kwargs)
    return result, log.getvalue()


def _eval_folder(run, n=3):
    return Path(run) / "Execution" / f"Evaluation.{n}" / "transport_simulation_folder"


def test_job_name_carries_the_run():
    assert gk_job_name_suffix(_eval_folder("/runs/V4C/E_pt03"), 0) == "_E_pt03_ev0"
    assert gk_job_name_suffix(_eval_folder("/runs/V4C/A_pt00"), 0) == "_A_pt00_ev0"
    assert gk_job_name_suffix(_eval_folder("/runs/V4C/E_pt03"), 12).endswith("_ev12")
    # MAESTRO: every PORTALS run folder is run_portals
    assert gk_job_name_suffix(_eval_folder("/scan/case_007/Beats/Beat_3/run_portals"), 1) == "_case_007_Beat_3_ev1"
    # simple-relax seeds: <run>/Initialization/initialization_simple_relax/portals_sr_ev_N/transport_simulation_folder
    sr = Path("/runs/V4C/E_pt03/Initialization/initialization_simple_relax/portals_sr_ev_2/transport_simulation_folder")
    assert gk_job_name_suffix(sr, 2) == "_E_pt03_ev2"
    sr = Path("/scan/case_007/Beats/Beat_3/run_portals/Initialization/initialization_simple_relax/portals_sr_ev_0/transport_simulation_folder")
    assert gk_job_name_suffix(sr, 0) == "_case_007_Beat_3_ev0"
    # Not a PORTALS evaluation folder: the folder itself names the job
    assert gk_job_name_suffix(Path("/somewhere/my_cgyro_test"), 0) == "_my_cgyro_test_ev0"
    assert transport_cgyro.gk_job_name_suffix is gk_job_name_suffix
    print("PASS test_job_name_carries_the_run")


def test_job_name_sanitized_and_capped():
    assert gk_job_name_suffix(_eval_folder("/runs/ARC V4 (mod#1)"), 0) == "_ARC_V4__mod_1__ev0"

    long_a = _eval_folder("/runs/" + "x" * 60 + "_caseA")
    long_b = _eval_folder("/runs/" + "x" * 60 + "_caseB")
    a, b = gk_job_name_suffix(long_a, 2), gk_job_name_suffix(long_b, 2)
    assert a != b, "two long run names sharing a prefix must still differ"
    assert len(a) == len("_") + _JOB_TAG_MAX_LENGTH + len("_ev2"), a
    assert a.endswith("_ev2") and a == gk_job_name_suffix(long_a, 2), "the hash must be stable"
    print("PASS test_job_name_sanitized_and_capped")


class FakeSubmitJob:
    '''mitim_job.run() of a detached submission, with what the dispatch printed.'''

    SBATCH_REFUSAL_PATTERNS = FARMINGtools.mitim_job.SBATCH_REFUSAL_PATTERNS

    def __init__(self, stdout, stderr):
        self.stdout, self.stderr = stdout, stderr
        self.folder_local = Path(tempfile.mkdtemp())
        self.launchSlurm = True

    def _squeue_job_name(self):
        return "cgyro_E_pt03_ev0"

    def run(self, **kwargs):
        # the real jobid parse and refusal detection of mitim_job.run
        (self.folder_local / "mitim.out").write_text(self.stdout)
        self.last_dispatch_output, self.last_dispatch_error = self.stdout.encode(), self.stderr.encode()
        self.jobid = next((l.split()[-1] for l in self.stdout.splitlines() if "Submitted batch job " in l), None)
        self.sbatch_refusal = (FARMINGtools.mitim_job._sbatch_refusal(self)
                               if (self.launchSlurm and self.jobid is None) else None)


def _dispatch(job):
    '''_dispatch_detached on `job`; returns the submission record it wrote (None when it raised first).'''
    job.folderExecution, job.slurm_settings = "/scratch/x", {}
    sim = types.SimpleNamespace(simulation_job=job, _extra_points_handed_over=lambda: {},
                                _submission_metadata_filename="cgyro_submission.json", written=[])
    sim._write_submission_metadata = lambda base: sim.written.append(base)
    settings = types.SimpleNamespace(launch_slurm=True, code="cgyro", cold_start=False, tmpFolder="tmp",
                                     files_to_retrieve=[], optional_files_to_retrieve=[], base_subfolder="base_cgyro")
    script = types.SimpleNamespace(array_index_by_folder={}, per_folder_commands={})
    SIMtools.mitim_simulation._dispatch_detached(sim, {}, script, settings)
    return sim.written


def test_refused_sbatch_raises():
    job = FakeSubmitJob("", "bashrc noise\nsbatch: error: Job cost estimated at 1234 hours exceeds the remaining balance\n"
                            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified\n")
    try:
        _quiet(_dispatch, job)
    except RuntimeError as e:
        msg = str(e)
        assert "sbatch refused" in msg and "cgyro_E_pt03_ev0" in msg and "exceeds the remaining balance" in msg, msg
        assert "bashrc noise" not in msg, msg
    else:
        raise AssertionError("a refused sbatch must raise")
    shutil.rmtree(job.folder_local, ignore_errors=True)
    print("PASS test_refused_sbatch_raises")


def test_unread_jobid_is_not_an_error():
    # sbatch went through (no error line) but its banner did not come back: no raise, the record is written
    job = FakeSubmitJob("", "some unrelated warning from the login shell\n")
    try:
        written, _ = _quiet(_dispatch, job)
    except RuntimeError as e:
        raise AssertionError(f"an sbatch without error must not raise: {e}")
    assert job.jobid is None and job.sbatch_refusal is None and written == ["base_cgyro"], written
    # and a normal submission parses its jobid and is not a refusal
    job = FakeSubmitJob("Submitted batch job 4242\n", "")
    written, _ = _quiet(_dispatch, job)
    assert job.jobid == "4242" and job.sbatch_refusal is None and written == ["base_cgyro"]
    print("PASS test_unread_jobid_is_not_an_error")


class FakeDeleteJob:
    def __init__(self, jobid):
        self.jobid = jobid
        self.launchSlurm = True
        self.commands = []

    def _squeue_job_name(self):
        return "cgyro_E_pt03_ev0"

    def prep(self, command, **kwargs):
        self.commands.append(command)

    def run(self, **kwargs):
        pass


def _delete_sim(jobid):
    sim = types.SimpleNamespace(
        simulation_job=FakeDeleteJob(jobid),
        _resubmit_ledger={"base_cgyro/rho_0.5500": {"child_jobids": ["201"]}},
        _extra_ledger={"x": {"status": "RUNNING", "jobid": "301"}, "y": {"status": "DONE", "jobid": "302"},
                       "z": {"status": "HANDED_OVER", "jobid": "303"}},
    )
    sim._child_jobids = lambda **kw: SIMtools.mitim_simulation._child_jobids(sim, **kw)
    return sim


def test_delete_by_jobid():
    sim = _delete_sim("100")
    _quiet(SIMtools.mitim_simulation.delete, sim)
    # a handed-over extra (left running for the next submission) goes too; the poll never waits on it
    assert sim.simulation_job.commands == ["scancel 100 201 301 303"], sim.simulation_job.commands
    assert SIMtools.mitim_simulation._child_jobids(sim) == ["201", "301"]
    assert sim.simulation_job.launchSlurm is True

    sim = _delete_sim(None)
    sim.simulation_job._jobs_by_folder_bash = lambda: "LOOKUP"
    _quiet(SIMtools.mitim_simulation.delete, sim)
    (command,) = sim.simulation_job.commands
    assert command.startswith("LOOKUP;") and "scancel -n" not in command and " 201 301 303" in command, command
    print("PASS test_delete_by_jobid")


if __name__ == "__main__":
    test_job_name_carries_the_run()
    test_job_name_sanitized_and_capped()
    test_refused_sbatch_raises()
    test_unread_jobid_is_not_an_error()
    test_delete_by_jobid()
    print("\nALL TESTS PASSED")
