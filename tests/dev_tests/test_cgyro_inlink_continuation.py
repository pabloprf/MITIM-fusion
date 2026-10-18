"""
test_cgyro_inlink_continuation.py
=================================
In-link continuation of a wall-hit CGYRO array element (gk_control.rescue.StallRescuer,
namelist max_continuations_per_rho, on only with rescue_interrupted): an element sacct reports
TIMEOUT / PREEMPTED / NODE_FAIL whose radius wrote a restart (out.cgyro.tag first token a
positive output index) is continued in place by a single-task job of its own, with its remote
input.cgyro trimmed so the run ends at the original t_end (.mitim_t_end) and out.cgyro.tag kept.

Mocks the mitim_job (execute / resubmit_single_task / squeue), no cluster.

Run as:

    python tests/dev_tests/test_cgyro_inlink_continuation.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.gk_control import markers, rescue

FOLDER = "base_cgyro/rho_0.6712"
INPUT = "DELTA_T=0.01\nMAX_TIME=600.0\nPRINT_STEP=100\nRESTART_STEP=10\nN_RADIAL=128\n"
RESCUE_SPEC = {"required": ["bin.cgyro.restart", "out.cgyro.tag"], "progress_file": "out.cgyro.tag", "progress_line": 2,
               "time_key": "MAX_TIME", "after_trim": CGYROtools.CGYRO._restart_step_after_trim,
               "checksum_ignore": ["MAX_TIME", "RESTART_STEP"]}


class FakeJob:
    def __init__(self):
        self.jobid = "12345"
        self.folderExecution = "/scratch/test"
        self.launchSlurm = True
        self.infoSLURM = {"STATE": "RUNNING"}
        self.machineSettings = {"slurm": {}}
        self.executed = []
        self.sacct = {}                 # target -> sacct State token
        self.t_tag = "240.0"            # line 2 of out.cgyro.tag
        self.t_end_file = "12345_2 600"  # .mitim_t_end
        self.t0_file = "0"
        self.remote_input = INPUT
        self.resubmits = []
        self.next_jobid = "67890"
        self.in_queue = []              # jobids squeue still lists

    def node_of(self, array_index):
        return "node05"

    def connect(self):
        self.executed.append(("connect",))

    def close(self):
        self.executed.append(("close",))

    def execute(self, cmd, printYN=False):
        self.executed.append(("execute", cmd))
        if cmd.startswith("sacct"):
            target = cmd.split()[2]
            return (f"{self.sacct[target]}|01:00:00" if target in self.sacct else ""), ""
        if cmd.startswith("squeue -h -j") and "%.15i" in cmd:
            rows = "".join(f"{j} RUNNING\n" for j in cmd.split()[3].split(",") if j in self.in_queue)
            return f"{rows}{SIMtools.FARMINGtools.SqueueAnswer.TRAILER}0", ""
        if cmd.startswith("squeue"):
            return "", ""
        if cmd.startswith("cd ") and markers.CONTINUE_SPLIT in cmd:
            sep = f"{markers.CONTINUE_SPLIT}\n"
            return f"{self.t_tag}\n{sep}{self.t_end_file}\n{sep}{self.t0_file}\n{sep}{self.remote_input}", ""
        if cmd.startswith("cat > "):
            self.remote_input = cmd.split("<<'MITIM_INPUT_EOF'\n", 1)[1].rsplit("\nMITIM_INPUT_EOF", 1)[0] + "\n"
            return "", ""
        return "", ""

    def resubmit_single_task(self, code_call_str, label, exclude_node=None):
        self.resubmits.append((code_call_str, label, exclude_node))
        return self.next_jobid

    def commands(self, prefix):
        return [c[1] for c in self.executed if c[0] == "execute" and c[1].startswith(prefix)]


class FakeSim:
    def __init__(self, rescue_interrupted=True, ledger=None, stall_enabled=True):
        self.simulation_job = FakeJob()
        self.kwargs_organize = {
            "array_index_by_folder": {FOLDER: 2},
            "per_folder_commands": {FOLDER: f"cd /scratch/test/{FOLDER} && cgyro -e ."},
        }
        # what transport_cgyro builds: the continuation cap is 0 unless rescue_interrupted
        self.auto_resubmit_settings = {
            "enabled": stall_enabled,
            "stall_init_kill_seconds": 1800,
            "stall_running_kill_seconds": 1800,
            "max_resubmits_per_rho": 1,
            "max_continuations_per_rho": 2 if rescue_interrupted else 0,
        }
        self.run_specifications = {"input_file": "input.cgyro", "rescue_spec": RESCUE_SPEC}
        self._resubmit_ledger = dict(ledger or {})
        self._base_subfolder = "base_cgyro"
        self.metadata_writes = 0

    def _write_submission_metadata(self, base_subfolder):
        self.metadata_writes += 1


def row(effective, since, tag="120"):
    return {"folder": FOLDER, "raw_state": "RUNNING", "effective": effective, "since_update_i": since,
            "wall_i": since, "avg_f": 5.0, "stale_threshold_warn": 300, "tag_token": tag}


def poll(sim, *rows):
    rescue._cgyro_handle_stalled_tasks(sim, list(rows))


def key(text, k):
    return float(re.search(rf"^{k}\s*=\s*(\S+)", text, flags=re.M).group(1))


def test_timeout_with_tag_continues_to_original_t_end():
    sim = FakeSim()
    job = sim.simulation_job
    job.sacct["12345_2"] = "TIMEOUT"
    # stale for 400 s only: well before the 1800 s stall kill threshold
    poll(sim, row("STALLED", 400))
    assert len(job.resubmits) == 1, job.resubmits
    body, label, exclude = job.resubmits[0]
    assert body == sim.kwargs_organize["per_folder_commands"][FOLDER] and label == "_continue_rho_0.6712_c1" and exclude is None
    # t_end 600 reached from t_tag 240: 360 a/cs more, nothing added on top
    assert abs(key(job.remote_input, "MAX_TIME") - 360.0) < 1e-6, job.remote_input
    assert key(job.remote_input, "RESTART_STEP") == 10 and "N_RADIAL=128" in job.remote_input
    # nothing cancelled, nothing deleted: out.cgyro.tag is what makes CGYRO continue
    assert not job.commands("scancel") and not [c for c in job.commands("cd ") if "rm -f" in c]
    entry = sim._resubmit_ledger[FOLDER]
    assert entry["child_jobids"] == ["67890"] and entry["continuation_jobids"] == ["67890"] and entry["n_attempts"] == 0
    assert entry["status"] == "active" and sim.metadata_writes == 1
    print("PASS test_timeout_with_tag_continues_to_original_t_end")


def test_restart_step_follows_the_trim_and_t_end_fallback():
    sim = FakeSim()
    job = sim.simulation_job
    job.sacct["12345_2"] = "PREEMPTED"
    job.t_tag, job.t_end_file, job.t0_file = "594.0", "", "0"   # no .mitim_t_end: t0 + MAX_TIME = 600
    poll(sim, row("TIMED_OUT", 0))
    assert len(job.resubmits) == 1
    assert abs(key(job.remote_input, "MAX_TIME") - 6.0) < 1e-6
    # 6 a/cs = 6 outputs: RESTART_STEP 10 would never fire
    assert key(job.remote_input, "RESTART_STEP") == 6, job.remote_input
    print("PASS test_restart_step_follows_the_trim_and_t_end_fallback")


def test_second_continuation_targets_the_child_and_cap_closes_it():
    first = {"n_attempts": 0, "child_jobids": ["67890"], "last_action_at": "2026-10-18T00:00:00Z",
             "status": "active", "continuation_jobids": ["67890"]}
    sim = FakeSim(ledger={FOLDER: first})
    job = sim.simulation_job
    job.sacct["67890"] = "TIMEOUT"
    job.t_tag, job.t_end_file = "480.0", "67890_0 600"
    job.next_jobid = "67891"
    poll(sim, row("STALLED", 400))
    assert job.resubmits[0][1] == "_continue_rho_0.6712_c2"
    assert abs(key(job.remote_input, "MAX_TIME") - 120.0) < 1e-6
    entry = sim._resubmit_ledger[FOLDER]
    assert entry["continuation_jobids"] == ["67890", "67891"] and entry["child_jobids"][-1] == "67891"

    # cap (2) spent: the third wall hit is left to the stall path, which closes it as today
    job.sacct["67891"] = "TIMEOUT"
    job.resubmits.clear()
    poll(sim, row("STALLED", 400))
    assert not job.resubmits and not job.commands("sacct -j 67891"), "no sacct before the kill threshold once the cap is spent"
    poll(sim, row("STALLED", 2000))
    assert not job.resubmits
    assert sim._resubmit_ledger[FOLDER]["status"] == "TERMINAL_NO_RESCUE:TIMEOUT"
    print("PASS test_second_continuation_targets_the_child_and_cap_closes_it")


def test_no_restart_tag_stays_terminal_no_rescue():
    sim = FakeSim()
    job = sim.simulation_job
    job.sacct["12345_2"] = "TIMEOUT"
    poll(sim, row("STALLED", 400, tag="-"))
    assert not job.executed, "no restart written: nothing to ask before the kill threshold"
    poll(sim, row("STALLED", 2000, tag="0"))
    assert not job.resubmits
    assert sim._resubmit_ledger[FOLDER]["status"] == "TERMINAL_NO_RESCUE:TIMEOUT"
    assert "continuation_jobids" not in sim._resubmit_ledger[FOLDER]
    print("PASS test_no_restart_tag_stays_terminal_no_rescue")


def test_running_element_keeps_the_stall_path():
    sim = FakeSim()
    job = sim.simulation_job
    job.sacct["12345_2"] = "RUNNING"
    poll(sim, row("STALLED", 400))
    assert not job.resubmits and not job.commands("scancel") and FOLDER not in sim._resubmit_ledger
    job.executed.clear()
    poll(sim, row("STALLED", 2000))
    # a genuine hang: scancel, clean (out.cgyro.tag deleted), resubmit, counted as a stall retry
    assert len(job.commands("sacct")) == 1, "one sacct per target and poll"
    assert job.commands("scancel 12345_2") and [c for c in job.commands("cd ") if "rm -f" in c and "out.cgyro.tag" in c]
    assert job.resubmits[-1][1] == "_resubmit_rho_0.6712_a1"
    entry = sim._resubmit_ledger[FOLDER]
    assert entry["n_attempts"] == 1 and "continuation_jobids" not in entry
    print("PASS test_running_element_keeps_the_stall_path")


def test_rescue_interrupted_false_is_unchanged():
    sim = FakeSim(rescue_interrupted=False)
    job = sim.simulation_job
    job.sacct["12345_2"] = "TIMEOUT"
    poll(sim, row("STALLED", 400), row("TIMED_OUT", 0))
    assert not job.executed, "below the kill threshold nothing is asked, as before"
    poll(sim, row("STALLED", 2000))
    assert not job.resubmits
    assert sim._resubmit_ledger[FOLDER] == {"n_attempts": 0, "child_jobids": [], "last_action_at": None,
                                            "status": "TERMINAL_NO_RESCUE:TIMEOUT"}
    # and with the stall rescue off too, the rescuer is not even built
    sim = FakeSim(rescue_interrupted=False, stall_enabled=False)
    sim.simulation_job.sacct["12345_2"] = "TIMEOUT"
    poll(sim, row("STALLED", 2000))
    assert not sim.simulation_job.executed and not sim._resubmit_ledger
    print("PASS test_rescue_interrupted_false_is_unchanged")


def test_continuation_without_stall_rescue():
    sim = FakeSim(stall_enabled=False)
    job = sim.simulation_job
    job.sacct["12345_2"] = "NODE_FAIL"
    poll(sim, row("STALLED", 2000))
    assert len(job.resubmits) == 1 and job.resubmits[0][2] == "node05", "node failure: that node is excluded"
    assert not job.commands("scancel")
    print("PASS test_continuation_without_stall_rescue")


def _interrupted_sim(ledger, in_queue):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "input.cgyro").write_text(INPUT)
    kept = "".join(l for l in INPUT.splitlines(keepends=True) if not l.startswith(("MAX_TIME", "RESTART_STEP")))
    sim = FakeSim(ledger=ledger)
    job = sim.simulation_job
    job.in_queue = in_queue
    job.preserve_subfolders = []
    job.probe_interrupted_runs = lambda *a, **k: {FOLDER: (hashlib.md5(kept.encode()).hexdigest(), "240.0", "", False)}
    return sim, tmp


def test_next_link_does_not_double_continue():
    entry = {"n_attempts": 0, "child_jobids": ["67890"], "last_action_at": None, "status": "active", "continuation_jobids": ["67890"]}
    sim, tmp = _interrupted_sim({FOLDER: entry}, in_queue=["67890"])
    try:
        rescue.rescue_interrupted_runs(sim, {"rescue_interrupted": True}, [tmp], [FOLDER], "input.cgyro")
        raise AssertionError("a radius whose continuation is still running must not be continued again")
    except RuntimeError as e:
        assert "67890" in str(e) and FOLDER in str(e)

    # the continuation ended: the next submission continues the radius in place as usual
    sim, tmp = _interrupted_sim({FOLDER: entry}, in_queue=[])
    rescue.rescue_interrupted_runs(sim, {"rescue_interrupted": True}, [tmp], [FOLDER], "input.cgyro")
    assert sim.simulation_job.preserve_subfolders == [FOLDER]
    text = (tmp / "input.cgyro").read_text()
    assert abs(key(text, "MAX_TIME") - 360.0) < 1e-6 and key(text, "RESTART_STEP") == 10
    print("PASS test_next_link_does_not_double_continue")


def test_ledger_round_trip():
    old = {"n_attempts": 1, "child_jobids": ["1"], "last_action_at": None, "status": "active"}
    assert rescue.LedgerEntry.from_json(old).to_json() == old
    new = dict(old, continuation_jobids=["2"], child_jobids=["1", "2"])
    assert rescue.LedgerEntry.from_json(new).to_json() == new and rescue.LedgerEntry.from_json(new).n_continuations == 1
    print("PASS test_ledger_round_trip")


if __name__ == "__main__":
    test_timeout_with_tag_continues_to_original_t_end()
    test_restart_step_follows_the_trim_and_t_end_fallback()
    test_second_continuation_targets_the_child_and_cap_closes_it()
    test_no_restart_tag_stays_terminal_no_rescue()
    test_running_element_keeps_the_stall_path()
    test_rescue_interrupted_false_is_unchanged()
    test_continuation_without_stall_rescue()
    test_next_link_does_not_double_continue()
    test_ledger_round_trip()
