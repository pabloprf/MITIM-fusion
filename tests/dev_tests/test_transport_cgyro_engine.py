"""
test_transport_cgyro_engine.py
==============================
The pieces the PORTALS-CGYRO driver was split into: the per-iteration override merge, the
re-attach decision table, the flux extraction, the warm-start chain and the extra-point harvest.

Every test runs against fakes, so nothing here touches a cluster or a real CGYRO output.

Run as:

    python tests/dev_tests/test_transport_cgyro_engine.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_modules.powertorch.physics_models import transport_cgyro
from mitim_modules.powertorch.physics_models.utils.cgyro_extra_points import ExtraPointHarvester
from mitim_modules.powertorch.physics_models.utils.cgyro_restart import RestartChain
from mitim_modules.powertorch.physics_models.utils.gk_submission import GKSubmission
from mitim_modules.powertorch.physics_models.utils.per_iter_overrides import PerIterOverrides


def _quiet(fn, *args, **kwargs):
    '''Run fn capturing its log, returning (result, log).'''
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = fn(*args, **kwargs)
    return result, log.getvalue()


# ======================================================================================
# PerIterOverrides
# ======================================================================================

def test_merge_returns_a_new_dict():
    baseline = {"MAX_TIME": 800}
    ov = PerIterOverrides({"3": {"MAX_TIME": 100}}, label="extraOptions_special")

    merged, _ = _quiet(ov.merge, baseline, 3)
    assert merged == {"MAX_TIME": 100}, merged
    assert baseline == {"MAX_TIME": 800}, "the namelist dict was mutated"

    # no match: still a copy, never the shared namelist dict
    unmatched, _ = _quiet(ov.merge, baseline, 4)
    assert unmatched == baseline and unmatched is not baseline

    # a None baseline with no match stays None, so run() still sizes the allocation itself
    assert _quiet(ov.merge, None, 4)[0] is None
    print("PASS test_merge_returns_a_new_dict")


def test_selectors():
    ov = PerIterOverrides({"0": {"a": 1}, ">5": {"b": 2}, "<=10": {"c": 3}}, label="extraOptions_special")

    assert _quiet(ov.merge, {}, 0)[0] == {"a": 1, "c": 3}
    assert _quiet(ov.merge, {}, 5)[0] == {"c": 3}
    assert _quiet(ov.merge, {}, 7)[0] == {"b": 2, "c": 3}
    assert _quiet(ov.merge, {}, 20)[0] == {"b": 2}
    # PORTALS hands the evaluation number as a string during the Execution phase
    assert _quiet(ov.merge, {}, "7")[0] == {"b": 2, "c": 3}
    print("PASS test_selectors")


def test_exact_key_wins_over_range():
    ov = PerIterOverrides({">4": {"MAX_TIME": 100}, "5": {"MAX_TIME": 200}}, label="extraOptions_special")
    assert _quiet(ov.merge, {}, 5)[0] == {"MAX_TIME": 200}
    assert _quiet(ov.merge, {}, 6)[0] == {"MAX_TIME": 100}
    print("PASS test_exact_key_wins_over_range")


def test_legacy_aliases():
    for name, key in (("extraOptions", "extraOptions_first"), ("allocation", "allocation_first")):
        run_options = {key: {"minutes": 480}}
        ov, log = _quiet(PerIterOverrides.from_run_options, run_options, name)
        assert "deprecated" in log, log
        assert _quiet(ov.merge, {}, 0)[0] == {"minutes": 480}
        assert _quiet(ov.merge, {}, 1)[0] == {}
    print("PASS test_legacy_aliases")


def test_log_lists_changed_keys_too():
    ov = PerIterOverrides({"3": {"MAX_TIME": 100, "RESTART_STEP": 50}}, label="extraOptions_special")
    _, log = _quiet(ov.merge, {"MAX_TIME": 800}, 3)
    assert "added {'RESTART_STEP': 50}" in log, log
    assert "changed {'MAX_TIME': 100}" in log, log
    print("PASS test_log_lists_changed_keys_too")


# ======================================================================================
# GKSubmission decision table
# ======================================================================================

class FakeJob:
    def __init__(self, status):
        self.status = status
        self.jobid = "12345"
        self.infoSLURM = {"STATE": "RUNNING" if status != 2 else "NOT FOUND"}
        self.connection_retry_settings = None

    def check(self, file_output=None):
        pass


class FakeSim:
    _submission_metadata_filename = "cgyro_submission.json"

    def __init__(self, job_status=2, local_complete=False, complete_after_fetch=False, child_alive=False):
        self.simulation_job = FakeJob(job_status)
        self.slurm_output = "slurm_output.dat"
        self._local_complete = local_complete
        self._complete_after_fetch = complete_after_fetch
        self._child_alive = child_alive
        self.fetched = 0

    def load_submission_state(self, path):
        return {"job": {"jobid": "12345", "machineSettings": {"machine": "engaging"},
                        "folderExecution": "/scratch/x"}, "created_utc": "now", "schema_version": 1}

    def _any_child_job_alive(self):
        return self._child_alive

    def _child_jobids(self):
        return ["999"]

    def _local_results_complete(self):
        return self._local_complete

    def fetch(self):
        self.fetched += 1
        self._local_complete = self._complete_after_fetch


def _submission(tmp, sim, write_metadata=True):
    (tmp / "base_cgyro").mkdir(parents=True, exist_ok=True)
    if write_metadata and sim._submission_metadata_filename:
        (tmp / "base_cgyro" / sim._submission_metadata_filename).write_text("{}")
    return GKSubmission(sim, tmp, "base_cgyro", every_n_minutes=1, enabled=True)


def test_reattach_decision_table():
    tmp = Path(tempfile.mkdtemp())
    try:
        # 1. no metadata on disk -> fresh submission, nothing fetched
        sim = FakeSim()
        outcome, log = _quiet(_submission(tmp, sim, write_metadata=False).try_reattach)
        assert (outcome.reattached, outcome.skip_check_fetch) == (False, False), outcome
        assert "No prior CGYRO submission" in log and sim.fetched == 0

        # 2. metadata + job alive -> re-attach and poll it
        sim = FakeSim(job_status=1)
        outcome, log = _quiet(_submission(tmp, sim).try_reattach)
        assert (outcome.reattached, outcome.skip_check_fetch) == (True, False), outcome
        assert "still live" in log and sim.fetched == 0

        # 2b. parent gone but a rescue child still queued counts as alive
        sim = FakeSim(job_status=2, child_alive=True)
        outcome, log = _quiet(_submission(tmp, sim).try_reattach)
        assert (outcome.reattached, outcome.skip_check_fetch) == (True, False), outcome
        assert "rescue child jobid(s) still alive" in log

        # 3. metadata + job gone + local results complete -> read directly, never fetch
        sim = FakeSim(job_status=2, local_complete=True)
        outcome, log = _quiet(_submission(tmp, sim).try_reattach)
        assert (outcome.reattached, outcome.skip_check_fetch) == (True, True), outcome
        assert sim.fetched == 0, "fetch() must not run when the results are already local"

        # 3b. one fetch fills the local set -> still a re-attach, no polling
        sim = FakeSim(job_status=2, local_complete=False, complete_after_fetch=True)
        outcome, _ = _quiet(_submission(tmp, sim).try_reattach)
        assert (outcome.reattached, outcome.skip_check_fetch) == (True, True), outcome
        assert sim.fetched == 1

        # 4. metadata + job gone + fetch cannot fill it -> fresh submission
        sim = FakeSim(job_status=2, local_complete=False, complete_after_fetch=False)
        sub = _submission(tmp, sim)
        called = []
        outcome, log = _quiet(sub.try_reattach, on_fresh_fallback=lambda: called.append(True))
        assert (outcome.reattached, outcome.skip_check_fetch) == (False, False), outcome
        assert sim.fetched == 1 and called == [True]
        assert not sub.path.exists(), "the stale metadata must be removed before resubmitting"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_reattach_decision_table")


def test_backend_without_metadata_warns_and_skips():
    tmp = Path(tempfile.mkdtemp())
    try:
        class FakeGX(FakeSim):
            _submission_metadata_filename = None

        GKSubmission._warned_backends.discard("FakeGX")
        sim = FakeGX()
        sub = GKSubmission(sim, tmp, "base_gx", enabled=True)
        assert sub.path is None
        outcome, log = _quiet(sub.try_reattach)
        assert (outcome.reattached, outcome.skip_check_fetch) == (False, False)
        assert "writes no submission metadata" in log, log
        # warned once per process, not once per PORTALS iteration
        _, log2 = _quiet(GKSubmission(FakeGX(), tmp, "base_gx", enabled=True).try_reattach)
        assert "writes no submission metadata" not in log2, log2
        # cleanup is a no-op rather than a crash
        _quiet(sub.cleanup)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_backend_without_metadata_warns_and_skips")


def test_cleanup_unlinks_metadata():
    tmp = Path(tempfile.mkdtemp())
    try:
        sim = FakeSim()
        sub = _submission(tmp, sim)
        assert sub.path.exists()
        _quiet(sub.cleanup, False)
        assert not sub.path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_cleanup_unlinks_metadata")


# ======================================================================================
# Flux extraction
# ======================================================================================

class FakeOutput:
    def __init__(self, seed, with_exchange=True, n_species=4):
        self.Qe_mean, self.Qe_std = 1.0 + seed, 0.1 + seed
        self.Qi_mean, self.Qi_std = 2.0 + seed, 0.2 + seed
        self.Ge_mean, self.Ge_std = 3.0 + seed, 0.3 + seed
        self.Mt_mean, self.Mt_std = 5.0 + seed, 0.5 + seed
        self.Gi_all_mean = np.array([10.0 + seed + i for i in range(n_species)])
        self.Gi_all_std = np.array([1.0 + seed + i for i in range(n_species)])
        if with_exchange:
            self.Se_mean, self.Se_std = 6.0 + seed, 0.6 + seed
        self.averaging = _FakeAveraging(seed)


class _FakeAveraging:
    def __init__(self, seed):
        self.seed = seed

    def to_dict(self):
        return {"method": "fixed", "seed": self.seed}


class FakePower(transport_cgyro.gyrokinetic_model):
    IMPURITY_POSITION = 2

    def _impurity_position_transport_for(self, side):
        return self.IMPURITY_POSITION


def _ctx(batched, nrho):
    ctx = transport_cgyro._GKRun()
    ctx.batched = batched
    ctx.rho_locations = [0.25 + 0.2 * i for i in range(nrho)]
    return ctx


def test_collect_fluxes_single():
    power = FakePower()
    outputs = [FakeOutput(i) for i in range(3)]
    _quiet(power._gk_collect_fluxes, _ctx(False, 3), [outputs])

    assert power.QeGB_turb.shape == (3,), power.QeGB_turb.shape
    assert np.allclose(power.QeGB_turb, [1.0, 2.0, 3.0])
    assert np.allclose(power.QeGB_turb_stds, [0.1, 1.1, 2.1])
    assert np.allclose(power.QiGB_turb, [2.0, 3.0, 4.0])
    assert np.allclose(power.GeGB_turb, [3.0, 4.0, 5.0])
    assert np.allclose(power.MtGB_turb, [5.0, 6.0, 7.0])
    # GZ is Gi_all at the turbulence-side impurity position
    assert np.allclose(power.GZGB_turb, [12.0, 13.0, 14.0])
    assert np.allclose(power.GZGB_turb_stds, [3.0, 4.0, 5.0])
    assert np.allclose(power.QieGB_turb, [6.0, 7.0, 8.0])
    assert np.allclose(power.QieGB_turb_stds, [0.6, 1.6, 2.6])
    assert power.averaging_info_turb == [{"method": "fixed", "seed": i} for i in range(3)]
    print("PASS test_collect_fluxes_single")


def test_collect_fluxes_batched():
    power = FakePower()
    per_plasma = [[FakeOutput(10 * p + i) for i in range(3)] for p in range(2)]
    _quiet(power._gk_collect_fluxes, _ctx(True, 3), per_plasma)

    assert power.QeGB_turb.shape == (2, 3), power.QeGB_turb.shape
    assert np.allclose(power.QeGB_turb, [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
    assert np.allclose(power.QiGB_turb, [[2.0, 3.0, 4.0], [12.0, 13.0, 14.0]])
    assert np.allclose(power.GZGB_turb, [[12.0, 13.0, 14.0], [22.0, 23.0, 24.0]])
    assert np.allclose(power.QieGB_turb, [[6.0, 7.0, 8.0], [16.0, 17.0, 18.0]])
    assert len(power.averaging_info_turb) == 2
    assert power.averaging_info_turb[1][0] == {"method": "fixed", "seed": 10}
    print("PASS test_collect_fluxes_batched")


def test_missing_exchange_moment():
    # none of the radii has it -> zeros, same shape and dtype as the heat flux arrays
    power = FakePower()
    outputs = [FakeOutput(i, with_exchange=False) for i in range(3)]
    _, log = _quiet(power._gk_collect_fluxes, _ctx(False, 3), [outputs])
    assert "no turbulent-exchange moment" in log
    assert np.allclose(power.QieGB_turb, 0.0) and power.QieGB_turb.shape == (3,)
    assert np.allclose(power.QieGB_turb_stds, 0.0) and power.QieGB_turb_stds.shape == (3,)

    # a MIX across radii is an inconsistent output set, not a zero
    mixed = [FakeOutput(0), FakeOutput(1, with_exchange=False), FakeOutput(2)]
    try:
        _quiet(power._gk_collect_fluxes, _ctx(False, 3), [mixed])
    except RuntimeError as e:
        assert "0.4500" in str(e), e
    else:
        raise AssertionError("a mixed exchange moment must raise")

    # and a mix ACROSS PLASMAS of one batch is caught too (checked once per batch)
    per_plasma = [[FakeOutput(i) for i in range(2)], [FakeOutput(i, with_exchange=False) for i in range(2)]]
    try:
        _quiet(power._gk_collect_fluxes, _ctx(True, 2), per_plasma)
    except RuntimeError as e:
        assert "plasma 1" in str(e), e
    else:
        raise AssertionError("a mix across plasmas must raise")
    print("PASS test_missing_exchange_moment")


# ======================================================================================
# RestartChain
# ======================================================================================

def _build_tree(root, iterations, rhos, fluxes=None, base="base_cgyro"):
    '''<root>/Execution/Evaluation.{i}/transport_simulation_folder/{base}/bin.cgyro.restart_<rho>'''
    for i in iterations:
        eval_root = root / "Execution" / f"Evaluation.{i}" / "transport_simulation_folder"
        (eval_root / base).mkdir(parents=True, exist_ok=True)
        for rho in rhos:
            (eval_root / base / f"bin.cgyro.restart_{rho:.4f}").write_text("blob")
        if fluxes is not None and i in fluxes:
            (eval_root / "fluxes_turb.json").write_text(json.dumps({"fluxes_mean": fluxes[i]}))


def test_restart_chain_first():
    root = Path(tempfile.mkdtemp())
    try:
        rhos = [0.25, 0.45]
        _build_tree(root, [0, 1], rhos)
        folder = root / "Execution" / "Evaluation.2" / "transport_simulation_folder"
        folder.mkdir(parents=True, exist_ok=True)

        chain = RestartChain({"restart_from_cases": "first"}, 2, folder, rhos)
        plan, _ = _quiet(chain.resolve, None, None)

        assert plan.mode == "first" and plan.sources == {"0.2500": 0, "0.4500": 0}, plan.sources
        assert sorted(plan.files_per_rho) == rhos
        src, dst = plan.files_per_rho[0.25][0]
        assert dst == "bin.cgyro.restart" and "Evaluation.0" in str(src), src
        # the parent map is on disk for the trace plotter, and the payload is what gets embedded
        on_disk = json.loads((folder / "base_cgyro" / "restart_sources.json").read_text())
        assert on_disk == plan.payload and on_disk["mode"] == "first"

        # "all" chains from N-1 instead
        plan_all, _ = _quiet(RestartChain({"restart_from_cases": "all"}, 2, folder, rhos).resolve, None, None)
        assert plan_all.sources == {"0.2500": 1, "0.4500": 1}, plan_all.sources

        # a missing binary is fatal for the uniform modes
        (root / "Execution" / "Evaluation.0" / "transport_simulation_folder" / "base_cgyro" / "bin.cgyro.restart_0.4500").unlink()
        try:
            _quiet(chain.resolve, None, None)
        except FileNotFoundError as e:
            assert "bin.cgyro.restart_0.4500" in str(e), e
        else:
            raise AssertionError("a missing per-rho restart must stop the run")

        # iteration 0 has nothing to restart from, and that is not an error
        chain0 = RestartChain({"restart_from_cases": "first"}, 0, folder, rhos)
        plan0, log0 = _quiet(chain0.resolve, None, None)
        assert plan0.sources == {} and plan0.files_per_rho is None and "no prior iteration" in log0.lower()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_restart_chain_first")


def test_restart_chain_best():
    root = Path(tempfile.mkdtemp())
    try:
        rhos = [0.25, 0.45]
        # iter 0 is closest at the inner radius, iter 1 at the outer one
        fluxes = {
            0: {"QeGB": [1.0, 9.0], "QiGB": [2.0, 9.0]},
            1: {"QeGB": [9.0, 3.0], "QiGB": [9.0, 4.0]},
        }
        _build_tree(root, [0, 1], rhos, fluxes=fluxes)
        folder = root / "Execution" / "Evaluation.2" / "transport_simulation_folder"
        folder.mkdir(parents=True, exist_ok=True)

        target = {"QeGB": np.array([1.0, 3.0]), "QiGB": np.array([2.0, 4.0])}
        chain = RestartChain({"restart_from_cases": "best"}, 2, folder, rhos)
        plan, _ = _quiet(chain.resolve, None, target)

        assert plan.mode == "best"
        assert plan.sources == {"0.2500": 0, "0.4500": 1}, plan.sources
        assert "Evaluation.0" in str(plan.files_per_rho[0.25][0][0])
        assert "Evaluation.1" in str(plan.files_per_rho[0.45][0][0])

        # a rho with no binary anywhere cold-starts instead of raising
        for i in (0, 1):
            (root / "Execution" / f"Evaluation.{i}" / "transport_simulation_folder"
             / "base_cgyro" / "bin.cgyro.restart_0.4500").unlink()
        plan2, log2 = _quiet(chain.resolve, None, target)
        assert plan2.sources == {"0.2500": 0}, plan2.sources
        assert 0.45 not in (plan2.files_per_rho or {})
        assert "cold start" in log2
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_restart_chain_best")


def test_restart_chain_no_op_modes():
    root = Path(tempfile.mkdtemp())
    try:
        folder = root / "Execution" / "Evaluation.2" / "transport_simulation_folder"
        folder.mkdir(parents=True, exist_ok=True)
        existing = {0.25: [("x", "y")]}

        # null mode, unknown mode and restart_from_folder precedence all hand the input back
        for run_options in ({}, {"restart_from_cases": "bogus"},
                            {"restart_from_cases": "first", "restart_from_folder": "/somewhere"}):
            plan, _ = _quiet(RestartChain(run_options, 2, folder, [0.25]).resolve, existing, None)
            assert plan.files_per_rho is existing and plan.mode is None, run_options

        # the retired flag still works
        plan, log = _quiet(RestartChain({"restart_from_first": True}, 0, folder, [0.25]).resolve, None, None)
        assert "deprecated" in log, log
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_restart_chain_no_op_modes")


# ======================================================================================
# ExtraPointHarvester
# ======================================================================================

class _FakeExtraOutput:
    t = [0.0, 320.0]
    Qe_mean, Qe_std = 1.5, 0.15
    Qi_mean, Qi_std = 2.5, 0.25
    Ge_mean, Ge_std = 3.5, 0.35


class _FakePowerstate:
    def __init__(self, outputs_folder):
        self.transport_options = {"folder": outputs_folder}
        self.predicted_channels = ["te", "ti"]


class _FakePowerTransport:
    def __init__(self, folder, outputs_folder):
        self.folder = folder
        self.powerstate = _FakePowerstate(outputs_folder)


def test_extra_points_marker_only_after_csv():
    from mitim_tools.gacode_tools.utils import CGYROutils

    root = Path(tempfile.mkdtemp())
    original = CGYROutils.CGYROoutput
    CGYROutils.CGYROoutput = lambda *a, **k: _FakeExtraOutput()
    try:
        folder = root / "transport_simulation_folder"
        d = folder / "extra_cgyro" / "rho_0.3486"
        d.mkdir(parents=True)
        (d / "out.cgyro.info").write_text("EXIT: (CGYRO) Reached MAX_TIME\n")
        (d / "out.cgyro.time").write_text("320.0\n")
        (d / "mitim_extra_point.json").write_text(json.dumps(
            {"x": {"aLte": 2.0, "aLti": 3.0}, "evaluation_number": 4, "rho": 0.3486,
             "channel": "te", "factor": 0.85, "radius_index": 1}))

        # an Outputs/ that is a FILE makes the CSV write fail
        blocked = root / "blocked"
        blocked.mkdir()
        (blocked / "Outputs").write_text("not a directory")
        harvester = ExtraPointHarvester(_FakePowerTransport(folder, blocked), "cgyro", [0.3486], {}, {})
        try:
            _quiet(harvester.harvest)
        except OSError:
            pass
        else:
            raise AssertionError("the blocked CSV write should have raised")
        assert not (d / "mitim_harvested").exists(), "folders were marked before the CSV was written"

        # with a writable Outputs/ the rows land and only then is the folder marked
        good = root / "good"
        good.mkdir()
        harvester = ExtraPointHarvester(_FakePowerTransport(folder, good), "cgyro", [0.3486], {}, {})
        _quiet(harvester.harvest)
        csv = good / "Outputs" / "extra_points.csv"
        assert csv.exists(), "the CSV was not written"
        assert (d / "mitim_harvested").exists(), "the folder was not marked after a successful write"
        body = csv.read_text()
        for model in ("Qe_tr_turb_1", "Qi_tr_turb_1", "Ge_tr_turb_1"):
            assert model in body, body

        # harvested once: a second pass adds nothing
        _quiet(harvester.harvest)
        assert csv.read_text() == body
    finally:
        CGYROutils.CGYROoutput = original
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_extra_points_marker_only_after_csv")


def test_extra_point_usable_still_importable():
    # tests/dev_tests/test_scheduler_extras.py reads it off transport_cgyro
    d = Path(tempfile.mkdtemp())
    try:
        (d / "out.cgyro.info").write_text("ERROR: (CGYRO) something broke\n")
        assert not transport_cgyro._extra_point_usable(d)
        (d / "mitim_budget.tag").write_text("STOP t=320\n")
        assert transport_cgyro._extra_point_usable(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_extra_point_usable_still_importable")


if __name__ == "__main__":
    test_merge_returns_a_new_dict()
    test_selectors()
    test_exact_key_wins_over_range()
    test_legacy_aliases()
    test_log_lists_changed_keys_too()
    test_reattach_decision_table()
    test_backend_without_metadata_warns_and_skips()
    test_cleanup_unlinks_metadata()
    test_collect_fluxes_single()
    test_collect_fluxes_batched()
    test_missing_exchange_moment()
    test_restart_chain_first()
    test_restart_chain_best()
    test_restart_chain_no_op_modes()
    test_extra_points_marker_only_after_csv()
    test_extra_point_usable_still_importable()
    print("\nALL TESTS PASSED")
