"""
test_cgyro_auto_resubmit.py
===========================
Sanity tests for the per-rho stall-detect-and-resubmit orchestrator added in
CGYROtools._cgyro_handle_stalled_tasks. Mocks the mitim_job interface
(connect / close / execute / resubmit_single_task) so the assertions exercise
the orchestrator's decision tree, ledger semantics, and metadata-write
hookup without contacting any cluster.

Run as:

    python tests/test_cgyro_auto_resubmit.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[1] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeJob:
    """Minimal mitim_job stand-in capturing calls and returning canned outputs."""

    def __init__(self):
        self.jobid = "12345"
        self.folderExecution = "/scratch/test"
        self.launchSlurm = True
        self.infoSLURM = {"NODELIST": "node05", "STATE": "RUNNING"}
        self.machineSettings = {"slurm": {}}
        self.slurm_settings = {"job-name": "cgyro_test"}
        self.executed = []
        self.scancel_should_raise = False
        self.resubmit_should_return = "67890"
        self.resubmit_should_raise = False
        self.last_resubmit_args = None
        self.connect_should_raise = False

    def connect(self):
        self.executed.append(("connect",))
        if self.connect_should_raise:
            raise RuntimeError("connect boom")

    def close(self):
        self.executed.append(("close",))

    def execute(self, cmd, printYN=False):
        self.executed.append(("execute", cmd))
        if cmd.startswith("scancel"):
            if self.scancel_should_raise:
                raise RuntimeError("scancel boom")
            return b"", b""
        return b"", b""

    def resubmit_single_task(self, code_call_str, label, exclude_node=None):
        self.last_resubmit_args = (code_call_str, label, exclude_node)
        if self.resubmit_should_raise:
            raise RuntimeError("resubmit boom")
        return self.resubmit_should_return


class FakeSim:
    def __init__(self, n_attempts_init=0, cap=1):
        self.simulation_job = FakeJob()
        self.kwargs_organize = {
            "code_executor": {"base_cgyro": {0.6712: {"folder": Path("/local/0.6712")}}},
            "array_index_by_folder": {"base_cgyro/rho_0.6712": 2},
            "per_folder_commands": {
                "base_cgyro/rho_0.6712": "cd /scratch/test/base_cgyro/rho_0.6712 && cgyro -e .",
            },
        }
        self.auto_resubmit_settings = {
            "enabled": True,
            "stall_init_kill_seconds": 1800,
            "stall_running_kill_seconds": 1800,
            "max_resubmits_per_rho": cap,
        }
        self._resubmit_ledger = {}
        if n_attempts_init > 0:
            self._resubmit_ledger["base_cgyro/rho_0.6712"] = {
                "n_attempts": n_attempts_init,
                "child_jobids": [f"6789{n_attempts_init}"],
                "last_action_at": "2026-05-07T00:00:00Z",
                "status": "active",
            }
        self._base_subfolder = "base_cgyro"
        self.metadata_write_calls = []

    def _write_submission_metadata(self, base_subfolder):
        self.metadata_write_calls.append(base_subfolder)


def make_row(folder, effective, since):
    return {
        "folder": folder,
        "raw_state": "RUNNING",
        "effective": effective,
        "since_update_i": since,
        "wall_i": since,
        "avg_f": 5.0,
        "stale_threshold_warn": 300,
        "tag_token": "-",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_running_below_threshold_no_action():
    sim = FakeSim()
    rows = [make_row("base_cgyro/rho_0.6712", "RUNNING", 100)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    assert sim.simulation_job.executed == [], sim.simulation_job.executed
    assert sim._resubmit_ledger == {}, sim._resubmit_ledger
    assert sim.metadata_write_calls == [], sim.metadata_write_calls
    print("PASS: running below threshold -> no action")


def test_stalled_runs_full_rescue_path():
    sim = FakeSim(cap=1)
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    cmds = [e for e in sim.simulation_job.executed if e[0] == "execute"]
    assert any("scancel 12345_2" in e[1] for e in cmds), cmds
    assert any("rm -f out.cgyro.timing" in e[1] for e in cmds), cmds
    assert sim.simulation_job.last_resubmit_args is not None
    _code, label, exclude = sim.simulation_job.last_resubmit_args
    assert exclude == "node05", exclude
    assert "rho_0.6712" in label, label
    ledger = sim._resubmit_ledger["base_cgyro/rho_0.6712"]
    assert ledger["n_attempts"] == 1, ledger
    assert ledger["child_jobids"] == ["67890"], ledger
    assert sim.metadata_write_calls == ["base_cgyro"], sim.metadata_write_calls
    print("PASS: stalled -> scancel + cleanup + resubmit + ledger++")


def test_cap_exhausted_no_action():
    sim = FakeSim(n_attempts_init=1, cap=1)
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    cmds = [e for e in sim.simulation_job.executed if e[0] == "execute"]
    assert not any("scancel" in e[1] for e in cmds), cmds
    assert sim.simulation_job.last_resubmit_args is None
    assert sim._resubmit_ledger["base_cgyro/rho_0.6712"]["status"] == "EXHAUSTED"
    print("PASS: cap exhausted -> mark EXHAUSTED, no scancel/resubmit")


def test_resubmit_no_jobid_does_not_increment_ledger():
    sim = FakeSim(cap=2)
    sim.simulation_job.resubmit_should_return = None
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    ledger = sim._resubmit_ledger.get("base_cgyro/rho_0.6712", {})
    assert ledger.get("n_attempts", 0) == 0, ledger
    # No metadata write since nothing was committed.
    assert sim.metadata_write_calls == [], sim.metadata_write_calls
    print("PASS: resubmit returning no jobid -> ledger NOT incremented")


def test_scancel_failure_skips_rescue():
    sim = FakeSim(cap=1)
    sim.simulation_job.scancel_should_raise = True
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    assert sim.simulation_job.last_resubmit_args is None
    ledger = sim._resubmit_ledger.get("base_cgyro/rho_0.6712", {})
    assert ledger.get("n_attempts", 0) == 0, ledger
    assert sim.metadata_write_calls == [], sim.metadata_write_calls
    print("PASS: scancel raises -> no resubmit, ledger unchanged")


def test_resubmit_raises_no_increment():
    sim = FakeSim(cap=2)
    sim.simulation_job.resubmit_should_raise = True
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    ledger = sim._resubmit_ledger.get("base_cgyro/rho_0.6712", {})
    assert ledger.get("n_attempts", 0) == 0, ledger
    print("PASS: resubmit_single_task raising -> ledger unchanged")


def test_disabled_does_nothing():
    sim = FakeSim()
    sim.auto_resubmit_settings["enabled"] = False
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    assert sim.simulation_job.executed == [], sim.simulation_job.executed
    assert sim._resubmit_ledger == {}, sim._resubmit_ledger
    print("PASS: enabled=False -> no-op")


def test_separate_init_running_thresholds():
    sim = FakeSim(cap=1)
    sim.auto_resubmit_settings["stall_init_kill_seconds"] = 600
    sim.auto_resubmit_settings["stall_running_kill_seconds"] = 9999
    rows = [
        make_row("base_cgyro/rho_0.6712", "STALLED", 2000),
        make_row("base_cgyro/rho_0.4736", "STALLED_INIT", 700),
    ]
    sim.kwargs_organize["array_index_by_folder"]["base_cgyro/rho_0.4736"] = 1
    sim.kwargs_organize["per_folder_commands"]["base_cgyro/rho_0.4736"] = "cd ... && cgyro -e ."
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    assert sim.simulation_job.last_resubmit_args is not None
    _code, label, _exclude = sim.simulation_job.last_resubmit_args
    assert "rho_0.4736" in label, label
    print("PASS: separate init/running thresholds honoured")


def test_no_array_metadata_means_noop():
    sim = FakeSim()
    sim.kwargs_organize["array_index_by_folder"] = {}
    sim.kwargs_organize["per_folder_commands"] = {}
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    assert sim.simulation_job.executed == [], sim.simulation_job.executed
    print("PASS: non-array submission (no per_folder_commands) -> no-op")


def test_no_node_in_squeue_falls_back_to_no_exclude():
    sim = FakeSim(cap=1)
    sim.simulation_job.infoSLURM["NODELIST"] = "(null)"
    rows = [make_row("base_cgyro/rho_0.6712", "STALLED", 2000)]
    CGYROtools._cgyro_handle_stalled_tasks(sim, rows)
    _code, _label, exclude = sim.simulation_job.last_resubmit_args
    assert exclude is None, exclude
    print("PASS: NODELIST '(null)' -> no --exclude on resubmit")


def test_metadata_payload_keys_round_trip():
    """The new fields persist+restore through JSON without surprises."""
    payload = {
        "kwargs_organize": {
            "tmpFolder": "/tmp",
            "filesToRetrieve": [],
            "optional_files_to_retrieve": [],
            "code_executor": {},
            "array_index_by_folder": {"base_cgyro/rho_0.6712": 2},
            "per_folder_commands": {"base_cgyro/rho_0.6712": "cgyro -e ."},
        },
        "resubmit_ledger": {
            "base_cgyro/rho_0.6712": {
                "n_attempts": 1,
                "child_jobids": ["67890"],
                "status": "active",
                "last_action_at": "2026-05-07T00:00:00Z",
            }
        },
        "base_subfolder": "base_cgyro",
    }
    parsed = json.loads(json.dumps(payload))
    assert parsed["resubmit_ledger"]["base_cgyro/rho_0.6712"]["n_attempts"] == 1
    assert parsed["kwargs_organize"]["array_index_by_folder"]["base_cgyro/rho_0.6712"] == 2
    assert parsed["base_subfolder"] == "base_cgyro"
    print("PASS: metadata payload structure round-trips through JSON")


if __name__ == "__main__":
    test_running_below_threshold_no_action()
    test_stalled_runs_full_rescue_path()
    test_cap_exhausted_no_action()
    test_resubmit_no_jobid_does_not_increment_ledger()
    test_scancel_failure_skips_rescue()
    test_resubmit_raises_no_increment()
    test_disabled_does_nothing()
    test_separate_init_running_thresholds()
    test_no_array_metadata_means_noop()
    test_no_node_in_squeue_falls_back_to_no_exclude()
    test_metadata_payload_keys_round_trip()
    print("\nALL TESTS PASSED")
