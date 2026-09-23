"""
test_cgyro_completion_gate.py
=============================
A CGYRO step killed mid-run (preemption, crash, GPU OOM, node failure) must never be
handed back as a finished evaluation.

CGYRO writes every output file from its first step, so a truncated run passes the
"files exist" retrieval check. Two fixes are tested here:

- SIMtools.mitim_simulation._verify_completion: after a normal run, every radius must
  carry CGYRO's own completion marker (the EXIT line in out.cgyro.info, or the
  wall-budget watchdog's mitim_budget.tag), otherwise the run raises.
- CGYROtools.body_keeping_exit_status: the call body's exit status is CGYRO's, not the
  trailing cleanup's, so the in-allocation scheduler no longer logs rc=0 for a killed step.
- CGYROtools.CgyroLaunchBody.exit_verdict: gacode's `cgyro` script exits 0 even when the
  executable crashed, so a body whose CGYRO left no EXIT line reports rc=1 (engaging array
  23547066, 2026-09-23: "Disk quota exceeded" at t=205 of 750, recorded as COMPLETED 0:0).

The end-to-end test reproduces the Perlmutter incident of 2026-09-21 (job 58658847): a
step SIGTERM'd at t=526 of 750, logged by the scheduler as "ended (rc=0)".

Run as:

    python tests/dev_tests/test_cgyro_completion_gate.py

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
from mitim_tools.simulation_tools.utils import SCHEDULERtools

MARKER = ("out.cgyro.info", "EXIT")
ALT = "mitim_budget.tag"
RHO = 0.8332


def _fake_sim(completion_marker=MARKER, completion_alt_file=ALT):
    '''Just enough of a mitim_simulation to call _verify_completion.'''
    sim = types.SimpleNamespace(run_specifications={
        "completion_marker": completion_marker, "completion_alt_file": completion_alt_file})
    sim._verify_completion = types.MethodType(SIMtools.mitim_simulation._verify_completion, sim)
    return sim


def _stored(folder, info_text=None, budget_tag=False, rho=RHO):
    '''A radius as _organize_results leaves it: files renamed <file>_<rho>.'''
    folder.mkdir(parents=True, exist_ok=True)
    if info_text is not None:
        (folder / f"out.cgyro.info_{rho:.4f}").write_text(info_text)
    if budget_tag:
        (folder / f"{ALT}_{rho:.4f}").write_text("t=300\n")
    return {"base_cgyro": {rho: {"folder": folder}}}


TRUNCATED = "INFO: (CGYRO) GPU-aware code triggered.\n[t: 5.260E+02]\n"
FINISHED = TRUNCATED + "EXIT: (CGYRO) Normal\n"


# ---------------------------------------------------------------------------
def test_radius_finished():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cases = {
            "normal EXIT": (FINISHED, False, True),
            "linear EXIT": ("EXIT: (CGYRO) Linear converged\n", False, True),
            "truncated": (TRUNCATED, False, False),
            "CGYRO error": ("ERROR: (CGYRO) Invalid label found\n", False, False),
            "watchdog stop": (TRUNCATED, True, True),
            "info missing": (None, False, False),
        }
        for name, (text, tag, want) in cases.items():
            folder = d / name.replace(" ", "_")
            _stored(folder, text, tag)
            got, mfile = SIMtools.radius_finished(folder, RHO, MARKER, ALT)
            assert got == want, f"{name}: finished={got}, expected {want}"
            assert mfile.name == f"out.cgyro.info_{RHO:.4f}"
        # no alt file configured: a watchdog tag must not count
        folder = d / "watchdog_no_alt"
        _stored(folder, TRUNCATED, budget_tag=True)
        assert SIMtools.radius_finished(folder, RHO, MARKER, None)[0] is False
    print("PASS test_radius_finished")


def test_cold_start_checker_unchanged():
    '''The pre-launch check now uses the same helper; its verdicts must not move.'''
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rhos = [0.5, 0.9]
        for rho, text in ((0.5, FINISHED), (0.9, TRUNCATED)):
            (d / f"out.cgyro.info_{rho:.4f}").write_text(text)
        need = SIMtools.cold_start_checker(rhos, ["out.cgyro.info"], d,
                                           completion_marker=MARKER, completion_alt_file=ALT)
        assert need == [0.9], need
        (d / f"{ALT}_0.9000").write_text("t=300\n")
        need = SIMtools.cold_start_checker(rhos, ["out.cgyro.info"], d,
                                           completion_marker=MARKER, completion_alt_file=ALT)
        assert need == [], need
    print("PASS test_cold_start_checker_unchanged")


def test_verify_completion():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # finished and watchdog-stopped radii pass
        _fake_sim()._verify_completion(_stored(d / "ok", FINISHED), "cgyro")
        _fake_sim()._verify_completion(_stored(d / "budget", TRUNCATED, budget_tag=True), "cgyro")
        # codes without a marker (TGLF, NEO) are never gated
        _fake_sim(completion_marker=None)._verify_completion(_stored(d / "tglf", TRUNCATED), "tglf")
        # a truncated radius raises, and the message says which one and where it stopped
        try:
            _fake_sim()._verify_completion(_stored(d / "cut", TRUNCATED), "cgyro")
        except RuntimeError as e:
            msg = str(e)
            assert "CGYRO" in msg and f"rho={RHO:.4f}" in msg and "5.260E+02" in msg, msg
        else:
            raise AssertionError("truncated radius was accepted")
        # one bad radius among good ones is enough to raise
        code_executor = _stored(d / "mix_ok", FINISHED)
        code_executor["base_cgyro"][0.9] = {"folder": d / "mix_bad"}
        _stored(d / "mix_bad", "ERROR: (CGYRO) boom\n", rho=0.9)
        try:
            _fake_sim()._verify_completion(code_executor, "cgyro")
        except RuntimeError as e:
            assert "rho=0.9000" in str(e) and f"rho={RHO:.4f}" not in str(e), str(e)
        else:
            raise AssertionError("mixed batch with one failed radius was accepted")
    print("PASS test_verify_completion")


def test_body_keeping_exit_status():
    with tempfile.TemporaryDirectory() as d:
        cleaned = Path(d) / "cleaned"
        cleanup = f'rm -f "{Path(d) / "nothing"}"; touch "{cleaned}"'
        for main_cmd, want in (("true", 0), ("false", 1), ("bash -c 'kill -TERM $$'", 143)):
            cleaned.unlink(missing_ok=True)
            body = CGYROtools.body_keeping_exit_status(":", main_cmd, cleanup)
            rc = subprocess.run(["bash", "-c", body]).returncode
            assert rc == want, f"{main_cmd!r}: rc={rc}, expected {want}"
            assert cleaned.exists(), f"{main_cmd!r}: cleanup did not run"
        # appended commands still run (slurm_array / bash group case)
        body = CGYROtools.body_keeping_exit_status(":", "false", ":") + f'touch "{cleaned}.after"\n'
        subprocess.run(["bash", "-c", body])
        assert Path(f"{cleaned}.after").exists(), "a command appended after the body did not run"
    print("PASS test_body_keeping_exit_status")


def test_killed_step_end_to_end():
    '''A step SIGTERM'd mid-run: the scheduler reports its real rc and the run refuses it.'''
    d = Path(tempfile.mkdtemp())
    try:
        rel = f"base_cgyro/rho_{RHO:.4f}"
        (d / rel).mkdir(parents=True)
        fake_cgyro = (f'printf "INFO: (CGYRO) GPU-aware code triggered.\\n[t: 5.260E+02]\\n" > "{d / rel}/out.cgyro.info"; '
                      "bash -c 'kill -TERM $$'")
        body = CGYROtools.body_keeping_exit_status(":", fake_cgyro, ":")
        sched = SCHEDULERtools.InAllocationScheduler({rel: body}, hosts=[], concurrency=1, poll_seconds=0.2)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            sched.run(d)
        assert "ended (rc=143)" in log.getvalue(), f"scheduler did not report the kill:\n{log.getvalue()}"

        # what _organize_results would store, then the post-run gate
        stored = d / "stored"
        stored.mkdir()
        shutil.copy(d / rel / "out.cgyro.info", stored / f"out.cgyro.info_{RHO:.4f}")
        try:
            _fake_sim()._verify_completion({"base_cgyro": {RHO: {"folder": stored}}}, "cgyro")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the killed step was accepted as a finished evaluation")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_killed_step_end_to_end")


def test_exit_verdict():
    '''A launcher that returns 0 after CGYRO crashed is reported as rc=1; clean ends and watchdog stops are not.'''
    with tempfile.TemporaryDirectory() as d:
        rel = f"base_cgyro/rho_{RHO:.4f}"
        run_dir = Path(d) / rel
        run_dir.mkdir(parents=True)
        launch = CGYROtools.CgyroLaunchBody(rel, d)
        info = run_dir / "out.cgyro.info"
        crashed = 'printf "[t: 2.050E+02]\\n" > "{info}"; true'
        cases = (
            ("crash, launcher rc 0", crashed, None, 1),
            ("clean end", 'printf "[t: 7.500E+02]\\nEXIT: (CGYRO) Normal\\n" > "{info}"; true', None, 0),
            ("error line, launcher rc 0", 'printf "ERROR: (CGYRO) bad input\\n" > "{info}"; true', None, 1),
            ("watchdog stop", crashed, "mitim_budget.tag", 0),
            ("watchdog discard", crashed, "mitim_discard.tag", 0),
            ("killed, rc kept", 'printf "[t: 2.050E+02]\\n" > "{info}"; bash -c \'kill -TERM $$\'', None, 143),
        )
        for label, main_cmd, tag, want in cases:
            for f in run_dir.iterdir():
                f.unlink()
            if tag:
                (run_dir / tag).touch()
            body = CGYROtools.body_keeping_exit_status(":", main_cmd.format(info=info), ":", launch.exit_verdict())
            rc = subprocess.run(["bash", "-c", body], capture_output=True).returncode
            assert rc == want, f"{label}: rc={rc}, expected {want}"
    print("PASS test_exit_verdict")


if __name__ == "__main__":
    test_radius_finished()
    test_cold_start_checker_unchanged()
    test_verify_completion()
    test_body_keeping_exit_status()
    test_killed_step_end_to_end()
    test_exit_verdict()
    print("\nALL PASS")
