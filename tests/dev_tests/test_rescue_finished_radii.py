"""
test_rescue_finished_radii.py
=============================
In bash/in-allocation mode the per-radius CGYRO outputs are retrieved only when the WHOLE
evaluation ends. A chain link that times out after some radii finished leaves them finished in
the scratch folder only, so the next link re-attaches with every radius in its work plan, and the
interrupted-run rescue used to "continue" the finished ones too: MAX_TIME 750 -> 1 remaining,
~1 min more to t~751, and only then extras on their nodes (Perlmutter slot2 pt00 link 2, job
58888645, 2026-09-26: rho 0.3316/0.5435/0.6342).

Now the rescue probe also reports whether a scratch radius already ENDED (EXIT line in
out.cgyro.info, or mitim_budget.tag). Such a radius passes the same input identity check, is kept
through the scratch wipe, is not launched, and is retrieved with the others; its node is offered
to extras from the start. When every radius ended, nothing is launched at all.

Run as:

    python tests/dev_tests/test_rescue_finished_radii.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import hashlib
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
from mitim_tools.misc_tools import FARMINGtools
from mitim_tools.simulation_tools import SIMtools

INPUT = """DELTA_T                 = 1.00000E-02
PRINT_STEP              = 100
RESTART_STEP            = 750
MAX_TIME                = 7.50000E+02
N_RADIAL                = 24
"""
IGNORED = ("MAX_TIME", "RESTART_STEP")
EXIT_LINE = "EXIT: (CGYRO) Simulation time limit reached\n"


def _md5(text):
    kept = "".join(l for l in text.splitlines(keepends=True) if not l.startswith(IGNORED))
    return hashlib.md5(kept.encode()).hexdigest()


def _write(folder, **files):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name.replace("__", ".")).write_text(text)


# ---------------------------------------------------------------------------
# The probe's shell, run for real
# ---------------------------------------------------------------------------


def _local_job(scratch):
    '''probe_interrupted_runs needs only these members of a mitim_job.'''
    def execute(cmd):
        out = subprocess.run(["bash", "-c", cmd], capture_output=True)
        return out.stdout, out.stderr
    return types.SimpleNamespace(run_in_place=False, folderExecution=scratch, folder_local=scratch.parent,
                                 session=lambda log_file=None: contextlib.nullcontext(), execute=execute)


def test_probe_flags_finished_radii():
    d = Path(tempfile.mkdtemp())
    try:
        scratch = d / "scratch"
        tag = "  75000\n  6.00000E+02\n"
        # interrupted: restart + tag, no EXIT yet
        _write(scratch / "base_cgyro/rho_0.3000", input__cgyro=INPUT, bin__cgyro__restart="x", out__cgyro__tag=tag, out__cgyro__info="running\n")
        # ran to MAX_TIME
        _write(scratch / "base_cgyro/rho_0.5000", input__cgyro=INPUT, bin__cgyro__restart="x", out__cgyro__tag="  75000\n  7.50000E+02\n",
               out__cgyro__info="running\n" + EXIT_LINE)
        # ended cleanly without any restart write: no tag, no restart, still finished
        _write(scratch / "base_cgyro/rho_0.6000", input__cgyro=INPUT, out__cgyro__info=EXIT_LINE)
        # stopped by the watchdog (mitim_kill_cgyro): no EXIT line, but the budget tag
        _write(scratch / "base_cgyro/rho_0.7000", input__cgyro=INPUT, bin__cgyro__restart="x", out__cgyro__tag=tag, mitim_budget__tag="stop\n")
        # nothing usable: neither the restart files nor an end marker
        _write(scratch / "base_cgyro/rho_0.8000", input__cgyro=INPUT, out__cgyro__info="running\n")

        rels = [f"base_cgyro/rho_0.{i}000" for i in (3, 5, 6, 7, 8)]
        spec = CGYROtools.CGYRO().run_specifications
        rescue = spec["rescue_spec"]
        found = FARMINGtools.mitim_job.probe_interrupted_runs(
            _local_job(scratch), rels, rescue["required"], "input.cgyro",
            progress_file=rescue["progress_file"], progress_line=rescue["progress_line"],
            checksum_ignore_prefix=rescue["checksum_ignore"], report_files=["out.cgyro.tag"],
            completion=(*spec["completion_marker"], spec["completion_alt_file"]),
        )
        md5 = _md5(INPUT)
        assert set(found) == set(rels[:4]), found
        assert found[rels[0]][0] == md5 and found[rels[0]][1] == "6.00000E+02" and found[rels[0]][3] is False, found[rels[0]]
        assert found[rels[1]][0] == md5 and found[rels[1]][3] is True, found[rels[1]]
        # empty progress token (no tag file) must not shift the md5 or the flag
        assert found[rels[2]][0] == md5 and found[rels[2]][1] is None and found[rels[2]][3] is True, found[rels[2]]
        assert found[rels[3]][3] is True, found[rels[3]]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: probe flags EXIT / budget-tag radii as finished, also without restart files")


# ---------------------------------------------------------------------------
# SIMtools._rescue_interrupted_runs
# ---------------------------------------------------------------------------


class FakeJob:
    def __init__(self, found):
        self.run_in_place = False
        self.preserve_subfolders = None
        self._found = found

    def probe_interrupted_runs(self, folders_red, required, checksum_file, **kwargs):
        assert kwargs["completion"] == ("out.cgyro.info", "EXIT", "mitim_budget.tag"), kwargs["completion"]
        return {rel: v for rel, v in self._found.items() if rel in folders_red}


def test_rescue_keeps_finished_radii_without_relaunching():
    d = Path(tempfile.mkdtemp())
    try:
        rels = ["base_cgyro/rho_0.3316", "base_cgyro/rho_0.5435", "base_cgyro/rho_0.6342"]
        folders = [d / r for r in rels]
        for f in folders:
            _write(f, input__cgyro=INPUT, bin__cgyro__restart="staged blob")
        md5 = _md5(INPUT)
        sim = types.SimpleNamespace(
            run_specifications=CGYROtools.CGYRO().run_specifications,
            simulation_job=FakeJob({
                rels[0]: (md5, "7.50000E+02", "out.cgyro.tag=26", True),       # finished
                rels[1]: (md5, "4.25000E+02", "out.cgyro.tag=26", False),      # interrupted
                rels[2]: ("stale", "7.50000E+02", "out.cgyro.tag=26", True),   # finished, other input
            }),
        )
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            finished = SIMtools.mitim_simulation._rescue_interrupted_runs(
                sim, {"rescue_interrupted": True}, folders, rels, "input.cgyro")
        out = log.getvalue()

        assert finished == [rels[0]] and sim._finished_in_scratch == [rels[0]], finished
        assert sim.simulation_job.preserve_subfolders == [rels[1], rels[0]], sim.simulation_job.preserve_subfolders
        assert (folders[0] / "input.cgyro").read_text() == INPUT, "a finished radius must not be trimmed"
        assert not (folders[0] / "bin.cgyro.restart").exists(), "staged restart must not overwrite the finished run"
        assert "MAX_TIME                = 3.25000E+02" in (folders[1] / "input.cgyro").read_text()
        assert (folders[2] / "bin.cgyro.restart").exists(), "a discarded radius keeps its normal staging"
        assert "not relaunched" in out and "finished run found but its input.cgyro differs" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: finished radius kept and not trimmed, interrupted one trimmed, stale finished one discarded")


# ---------------------------------------------------------------------------
# SIMtools._run: what is launched vs what is retrieved
# ---------------------------------------------------------------------------


def _fake_sim(run_type, finished, rels):
    calls = {}
    settings = SIMtools._RunSettings(
        run_type=SIMtools.RunType.parse(run_type), code="cgyro", input_file="input.cgyro", code_call=None,
        name="cgyro_test", job_name_suffix="_sim", launch_slurm=True, allocation={"minutes": 10},
        resources_per_call=4, minutes=10, submission_type_override=None, exclusive=None,
        attempts_execution=1, cold_start=False, helper_lostconnection=False, base_subfolder=None,
        tmpFolder=Path("/nonexistent"), files_to_retrieve=[], optional_files_to_retrieve=[])

    def record(name, ret=None):
        def f(*args):
            calls[name] = args
            return ret
        return f

    sim = types.SimpleNamespace(simulation_job="job")
    sim._run_settings = lambda run_type, kwargs_run: settings
    sim._stage_inputs = record("stage", ([Path(r) for r in rels], list(rels)))
    sim._rescue_interrupted_runs = record("rescue", list(finished))
    sim._resolve_allocation = record("resolve", "resolved")
    sim._build_script = record("build", "script")
    sim._prepare_job = record("prepare")
    sim._dispatch_blocking = record("blocking")
    sim._dispatch_detached = record("detached")
    return sim, calls


def _run(sim, run_type):
    original = SIMtools.WorkPlan.from_code_executor
    SIMtools.WorkPlan.from_code_executor = classmethod(lambda cls, ce: ["call"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            SIMtools.mitim_simulation._run(sim, {"base_cgyro": {}}, run_type=run_type)
    finally:
        SIMtools.WorkPlan.from_code_executor = original


def test_run_launches_only_unfinished():
    rels = ["base_cgyro/rho_0.3316", "base_cgyro/rho_0.5435"]

    sim, calls = _fake_sim("normal", [rels[0]], rels)
    _run(sim, "normal")
    assert calls["resolve"][0] == [rels[1]] and calls["build"][0] == [rels[1]], calls
    assert calls["prepare"][1] == rels, "every radius, finished or not, is retrieved"
    assert calls["resolve"][1].launch_slurm is True and "blocking" in calls and "detached" not in calls

    # submit with every radius finished: no job, retrieved now, then check()/fetch() are no-ops
    sim, calls = _fake_sim("submit", rels, rels)
    _run(sim, "submit")
    s = calls["resolve"][1]
    assert calls["resolve"][0] == [] and s.launch_slurm is False and s.run_type is SIMtools.RunType.NORMAL, s
    assert "blocking" in calls and "detached" not in calls and sim.simulation_job is None, calls

    # submit with one left: the array holds only that one
    sim, calls = _fake_sim("submit", [rels[0]], rels)
    _run(sim, "submit")
    assert calls["build"][0] == [rels[1]] and "detached" in calls and "blocking" not in calls, calls

    # prep never dispatches, even with everything finished
    sim, calls = _fake_sim("prep", rels, rels)
    _run(sim, "prep")
    assert "blocking" not in calls and "detached" not in calls, calls
    print("PASS: finished radii are retrieved but not launched; all finished -> nothing submitted")


# ---------------------------------------------------------------------------
# Extras on the nodes of the finished radii
# ---------------------------------------------------------------------------


def test_idle_slots_offered_for_scratch_finished_radii():
    d = Path(tempfile.mkdtemp())
    try:
        (d / "base_cgyro").mkdir()
        fake = types.SimpleNamespace(
            run_specifications=CGYROtools.CGYRO().run_specifications, FolderGACODE=d,
            _finished_in_scratch=["base_cgyro/rho_0.3316", "base_cgyro/rho_0.5435"])
        main = ["base_cgyro/rho_0.6342"]
        assert CGYROtools.CGYRO._idle_slot_sources(fake, main) == fake._finished_in_scratch
        # an extra already done locally for rho 0.3316 is not repeated
        _write(d / "extra_cgyro/rho_0.3316", out__cgyro__info=EXIT_LINE)
        assert CGYROtools.CGYRO._idle_slot_sources(fake, main) == ["base_cgyro/rho_0.5435"]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: scratch-finished radii are idle-slot sources for extras")


def test_stall_rescue_ignores_radii_outside_the_array():
    r = object.__new__(CGYROtools.StallRescuer)
    r.array_index_by_folder = {"base_cgyro/rho_0.5435": 0}
    r.job = types.SimpleNamespace(jobid="123")
    with contextlib.redirect_stdout(io.StringIO()):
        target, _ = r._resolve_target("base_cgyro/rho_0.3316", CGYROtools.LedgerEntry())
    assert target is None, target
    print("PASS: a finished radius left out of the array is never scancelled")


if __name__ == "__main__":
    test_probe_flags_finished_radii()
    test_rescue_keeps_finished_radii_without_relaunching()
    test_run_launches_only_unfinished()
    test_idle_slots_offered_for_scratch_finished_radii()
    test_stall_rescue_ignores_radii_outside_the_array()
    print("\nALL TESTS PASSED")
