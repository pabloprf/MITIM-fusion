"""
test_driver_wall_marker.py
==========================
A PORTALS driver launched with sbatch polls its CGYRO job with no idea of its own SLURM wall: when the
last link of a chain is killed mid-poll, finished radii stay uncollected, silently. GKSubmission's poll
now asks squeue (%L) for the driver's time left and, when the next poll would not come before the wall
while the job is live, leaves <run>/Outputs/MITIM_DRIVER_WALL_UNFINISHED.json for the launcher.

Covered here (no cluster: squeue and SLURM_JOB_ID are faked):
- %L parsing (D-HH:MM:SS, HH:MM:SS, MM:SS, UNLIMITED, NOT_SET, junk);
- plenty of time -> no marker; little time with a live job -> marker with the unfinished radii;
- little time, job and children gone -> no marker; outside SLURM -> no marker, squeue never called;
- a failing squeue never breaks the poll;
- the marker sits in <run>/Outputs for both PORTALS layouts, next to the metadata otherwise;
- cleanup() after a successful read removes it.

Run as:

    python tests/dev_tests/test_driver_wall_marker.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools.gk_control import driver_wall, markers
from mitim_tools.simulation_tools.gk_control.submission import GKSubmission

ROWS = [
    {"folder": "base_cgyro/rho_0.35", "effective": "FINISHED"},
    {"folder": "base_cgyro/rho_0.55", "effective": "RUNNING"},
    {"folder": "base_cgyro/rho_0.75", "effective": "STALLED"},
]


class FakeSim:
    _submission_metadata_filename = markers.CGYRO_SUBMISSION

    def __init__(self, status=1, children=()):
        self.simulation_job = types.SimpleNamespace(jobid="4242", status=status, infoSLURM={"STATE": "RUNNING"})
        self.children = list(children)

    def _live_child_jobids(self, alive_if_unreachable=False):
        return self.children

    @staticmethod
    def _custom_check_callback(sim):
        return ROWS


class FakeSqueue:
    def __init__(self, stdout="", returncode=0, raises=False):
        self.stdout, self.returncode, self.raises, self.calls = stdout, returncode, raises, 0

    def __call__(self, cmd, **kwargs):
        self.calls += 1
        assert cmd[:3] == ["squeue", "-h", "-j"] and cmd[-2:] == ["-o", "%L"], cmd
        if self.raises:
            raise OSError("squeue: command not found")
        return types.SimpleNamespace(stdout=self.stdout, returncode=self.returncode)


def poll(folder, sim, squeue, job_id="777", every_n_minutes=10):
    real_run, real_env = subprocess.run, os.environ.get("SLURM_JOB_ID")
    driver_wall.subprocess.run = squeue
    if job_id is None:
        os.environ.pop("SLURM_JOB_ID", None)
    else:
        os.environ["SLURM_JOB_ID"] = job_id
    try:
        submission = GKSubmission(sim, folder, "base_cgyro", every_n_minutes=every_n_minutes)
        submission._poll_checker(every_n_minutes)(sim)
        return submission
    finally:
        driver_wall.subprocess.run = real_run
        if real_env is None:
            os.environ.pop("SLURM_JOB_ID", None)
        else:
            os.environ["SLURM_JOB_ID"] = real_env


def evaluation_folder(root):
    folder = Path(root) / "run_a" / "Execution" / "Evaluation.3" / "transport_simulation_folder"
    folder.mkdir(parents=True)
    return folder


def marker_of(root):
    return Path(root) / "run_a" / "Outputs" / markers.DRIVER_WALL_UNFINISHED


def test_parse_time_left():
    cases = {"1-02:03:04": 93784, "02:03:04": 7384, "03:04": 184, "0:00": 0, "UNLIMITED": None,
             "NOT_SET": None, "INVALID": None, "": None, "1:2:3:4": None}
    for text, expected in cases.items():
        assert driver_wall.parse_slurm_time_left(text) == expected, (text, driver_wall.parse_slurm_time_left(text))
    print("PASS test_parse_time_left")


def test_plenty_of_time_no_marker():
    with tempfile.TemporaryDirectory() as root:
        poll(evaluation_folder(root), FakeSim(), FakeSqueue("2:00:00\n"))
        assert not marker_of(root).exists()
        # boundary: exactly one poll + margin left is still enough
        poll(Path(root) / "run_a/Execution/Evaluation.3/transport_simulation_folder", FakeSim(), FakeSqueue("15:00\n"))
        assert not marker_of(root).exists()
    print("PASS test_plenty_of_time_no_marker")


def test_little_time_live_job_writes_marker():
    with tempfile.TemporaryDirectory() as root:
        folder = evaluation_folder(root)
        poll(folder, FakeSim(children=["5555"]), FakeSqueue("14:59\n"))
        data = json.loads(marker_of(root).read_text())
        assert data["slurm_job_id"] == "777" and data["driver_seconds_left"] == 899
        assert data["evaluation folder"] == str(folder)
        assert data["array jobid"] == "4242" and data["child jobids"] == ["5555"]
        assert data["unfinished radii"] == [{"folder": "base_cgyro/rho_0.55", "state": "RUNNING"},
                                            {"folder": "base_cgyro/rho_0.75", "state": "STALLED"}]
        assert "check_existing_runs" in data["note"] and data["time"]
        # a later poll overwrites it with the latest state
        poll(folder, FakeSim(), FakeSqueue("4:00\n"))
        data = json.loads(marker_of(root).read_text())
        assert data["driver_seconds_left"] == 240 and data["child jobids"] == []
    print("PASS test_little_time_live_job_writes_marker")


def test_little_time_job_gone_no_marker():
    with tempfile.TemporaryDirectory() as root:
        poll(evaluation_folder(root), FakeSim(status=2), FakeSqueue("1:00\n"))
        assert not marker_of(root).exists()
        poll(Path(root) / "run_a/Execution/Evaluation.3/transport_simulation_folder", FakeSim(status=2, children=["9"]), FakeSqueue("1:00\n"))
        assert marker_of(root).exists()
    print("PASS test_little_time_job_gone_no_marker")


def test_outside_slurm_or_failing_squeue_no_marker():
    with tempfile.TemporaryDirectory() as root:
        folder = evaluation_folder(root)
        squeue = FakeSqueue("1:00\n")
        poll(folder, FakeSim(), squeue, job_id=None)
        assert squeue.calls == 0 and not marker_of(root).exists()
        for broken in (FakeSqueue(raises=True), FakeSqueue("", returncode=1), FakeSqueue("UNLIMITED\n")):
            poll(folder, FakeSim(), broken)
            assert broken.calls == 1 and not marker_of(root).exists()
    print("PASS test_outside_slurm_or_failing_squeue_no_marker")


def test_marker_location():
    with tempfile.TemporaryDirectory() as root:
        seed = Path(root) / "run_b" / "Initialization" / "initialization_simple_relax" / "portals_sr_ev_1" / "transport_simulation_folder"
        seed.mkdir(parents=True)
        poll(seed, FakeSim(), FakeSqueue("1:00\n"))
        assert (Path(root) / "run_b" / "Outputs" / markers.DRIVER_WALL_UNFINISHED).exists()

        plain = Path(root) / "standalone"
        plain.mkdir()
        poll(plain, FakeSim(), FakeSqueue("1:00\n"))
        assert (plain / "base_cgyro" / markers.DRIVER_WALL_UNFINISHED).exists()
    print("PASS test_marker_location")


def test_cleanup_removes_marker():
    with tempfile.TemporaryDirectory() as root:
        folder = evaluation_folder(root)
        poll(folder, FakeSim(), FakeSqueue("1:00\n"))
        assert marker_of(root).exists()
        # the next link: a fresh submission object of a later evaluation of the same run reads its results
        later = Path(root) / "run_a" / "Execution" / "Evaluation.4" / "transport_simulation_folder"
        later.mkdir(parents=True)
        GKSubmission(FakeSim(), later, "base_cgyro").cleanup()
        assert not marker_of(root).exists()
    print("PASS test_cleanup_removes_marker")


if __name__ == "__main__":
    test_parse_time_left()
    test_plenty_of_time_no_marker()
    test_little_time_live_job_writes_marker()
    test_little_time_job_gone_no_marker()
    test_outside_slurm_or_failing_squeue_no_marker()
    test_marker_location()
    test_cleanup_removes_marker()
