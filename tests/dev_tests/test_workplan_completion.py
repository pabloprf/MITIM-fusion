"""
test_workplan_completion.py
===========================
The two small types that own the per-radius conventions in SIMtools:

  - RadialCall.rel / .result_name  -> the one definition of "sub/rho_0.3486" and
    "<file>_0.3486", so the staging loop and the retrieval loop cannot drift apart.
  - WorkPlan.from_code_executor / .to_code_executor -> the boundary with the
    external `code_executor` contract must be a round trip that changes nothing.
  - CompletionSpec.finished -> the single answer to "did this radius finish?",
    equal to what radius_finished() returns, and reusable with plain file names
    (scratch layout) by SCHEDULERtools and transport_cgyro.

Run as:

    python tests/dev_tests/test_workplan_completion.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.utils import SCHEDULERtools

RHO_A, RHO_B = 0.3486, 0.6712
MARKER = ("out.cgyro.info", "EXIT")
ALT = "mitim_budget.tag"


def test_radial_call_names():
    call = SIMtools.RadialCall("base_cgyro", RHO_A, folder=Path("/results"))
    assert call.rel == "base_cgyro/rho_0.3486", call.rel
    assert call.result_name("out.cgyro.gbflux") == "out.cgyro.gbflux_0.3486", call.result_name("out.cgyro.gbflux")
    assert SIMtools.rho_folder(RHO_A) == "rho_0.3486"
    assert SIMtools.rho_suffix(RHO_A) == "_0.3486"
    print("PASS test_radial_call_names")


def test_workplan_round_trip_is_identity():
    # Full entries as _run_prepare builds them, a folder-only entry as
    # load_submission_state rehydrates them, and a subfolder with nothing pending.
    full = {
        "folder": Path("/results/base_cgyro"),
        "dictionary": object(),
        "inputs": "TEXT\n",
        "extraOptions": {"MAX_TIME": 200},
        "multipliers": None,
        "additional_files_to_send": [("/a/bin.cgyro.restart_0.3486", "bin.cgyro.restart")],
    }
    code_executor = {
        "base_cgyro": {RHO_A: full, RHO_B: dict(full, inputs="OTHER\n")},
        "base_cgyro_plasma1": {RHO_A: {"folder": Path("/results/p1")}},
        "base_cgyro_cached": {},
    }

    plan = SIMtools.WorkPlan.from_code_executor(code_executor)
    assert len(plan) == 3, len(plan)
    assert plan.rel_paths == ["base_cgyro/rho_0.3486", "base_cgyro/rho_0.6712", "base_cgyro_plasma1/rho_0.3486"], plan.rel_paths

    back = plan.to_code_executor()
    assert back == code_executor, back
    assert list(back) == list(code_executor), list(back)
    assert back["base_cgyro"][RHO_A] is full, "entries must be handed back untouched"

    # A plan built by hand still serialises the standard field set
    hand = SIMtools.WorkPlan([SIMtools.RadialCall("base_tglf", RHO_A, folder=Path("/r"), inputs="X")])
    assert set(hand.to_code_executor()["base_tglf"][RHO_A]) == set(SIMtools._CODE_EXECUTOR_FIELDS)
    print("PASS test_workplan_round_trip_is_identity")


def test_completion_spec_matches_radius_finished():
    d = Path(tempfile.mkdtemp())
    try:
        spec = SIMtools.CompletionSpec(MARKER[0], MARKER[1], alt_file=ALT)

        cases = {
            "marker present": lambda: (d / f"{MARKER[0]}_{RHO_A:.4f}").write_text("stuff\n EXIT (CGYRO)\n"),
            "marker absent": lambda: (d / f"{MARKER[0]}_{RHO_A:.4f}").write_text("stuff\n running\n"),
            "alt file": lambda: (d / f"{ALT}_{RHO_A:.4f}").write_text("budget\n"),
        }
        expected = {"marker present": True, "marker absent": False, "alt file": True, "nothing": False}

        for label in ("nothing", "marker absent", "marker present", "alt file"):
            for f in d.iterdir():
                f.unlink()
            if label in cases:
                cases[label]()
            got, mfile = spec.finished(d, RHO_A)
            ref, ref_mfile = SIMtools.radius_finished(d, RHO_A, MARKER, ALT)
            assert got == ref == expected[label], (label, got, ref, expected[label])
            assert mfile == ref_mfile == d / f"{MARKER[0]}_{RHO_A:.4f}", (label, mfile, ref_mfile)

        # 'alt file' on its own, with the marker file saying the run was cut short
        for f in d.iterdir():
            f.unlink()
        (d / f"{MARKER[0]}_{RHO_A:.4f}").write_text("no marker here\n")
        (d / f"{ALT}_{RHO_A:.4f}").write_text("budget\n")
        assert spec.finished(d, RHO_A)[0] is True
        assert SIMtools.CompletionSpec(MARKER[0], MARKER[1]).finished(d, RHO_A)[0] is False

        # Plain names (scratch layout): same test without the rho suffix
        for f in d.iterdir():
            f.unlink()
        assert spec.finished(d)[0] is False
        (d / ALT).write_text("budget\n")
        assert spec.finished(d)[0] is True

        # unfinished() over a plan
        results = d / "results"
        results.mkdir()
        (results / f"{MARKER[0]}_{RHO_A:.4f}").write_text("EXIT (CGYRO)\n")
        (results / f"{MARKER[0]}_{RHO_B:.4f}").write_text("still going\n")
        plan = SIMtools.WorkPlan.from_code_executor(
            {"base_cgyro": {RHO_A: {"folder": results}, RHO_B: {"folder": results}}}
        )
        missing = spec.unfinished(plan)
        assert [c.rho for c, _ in missing] == [RHO_B], missing
        assert spec.unfinished(plan.to_code_executor()) and len(spec.unfinished(plan.to_code_executor())) == 1

        # No completion_marker declared -> no spec at all
        assert SIMtools.CompletionSpec.from_run_specifications({"code": "tglf"}) is None
        assert SIMtools.CompletionSpec.from_run_specifications(
            {"completion_marker": MARKER, "completion_alt_file": ALT}
        ) == spec
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_completion_spec_matches_radius_finished")


def test_scheduler_accepts_tuple_and_spec():
    d = Path(tempfile.mkdtemp())
    try:
        folder = d / "extra_cgyro" / "rho_0.3486"
        folder.mkdir(parents=True)

        for completion_marker in (MARKER, SIMtools.CompletionSpec(MARKER[0], MARKER[1])):
            sched = SCHEDULERtools.InAllocationScheduler({}, hosts=[], concurrency=1, completion_marker=completion_marker)
            assert sched.completion_marker is completion_marker, "constructor must keep what it was given"
            assert sched._accepted(folder) is False

            (folder / MARKER[0]).write_text("EXIT (CGYRO)\n")
            assert sched._accepted(folder) is True
            (folder / MARKER[0]).unlink()

            # accepted_marker (the scheduler's own default) still wins on its own
            (folder / ALT).write_text("budget\n")
            assert sched._accepted(folder) is True
            (folder / ALT).unlink()

        # Without a completion marker only the accepted_marker counts
        sched = SCHEDULERtools.InAllocationScheduler({}, hosts=[], concurrency=1)
        (folder / MARKER[0]).write_text("EXIT (CGYRO)\n")
        assert sched._accepted(folder) is False
        (folder / ALT).write_text("budget\n")
        assert sched._accepted(folder) is True
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_scheduler_accepts_tuple_and_spec")


if __name__ == "__main__":
    test_radial_call_names()
    test_workplan_round_trip_is_identity()
    test_completion_spec_matches_radius_finished()
    test_scheduler_accepts_tuple_and_spec()
    print("\nAll tests passed")
