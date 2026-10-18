"""
test_cgyro_cost_guard.py
========================
The cost guard of a PORTALS-CGYRO radius (namelist transport.options.cgyro.run.cost_guard):
a radius that costs too many wall seconds per a/cs while its heat flux is far above target is
stopped by the watchdog, at once, and counts as finished.

Three layers are covered, none needs CGYRO:
  - templates/gk_control/cgyro_guard.py, the judge that runs on the compute node: the binary flux reader
    and every clause of the stop rule, on synthetic radius folders;
  - templates/gk_control/cgyro_watchdog.sh: the real bash around a fake CGYRO (`sleep`), with the guard
    files staged as CGYROtools.CostGuard stages them;
  - the Python wiring: CostGuard staging and the targets the transport layer hands it.
A CGYRO that ends on the Cash-Karp abort while over-driven is accepted as a guard stop
(cgyro_guard.abort_verdict, the watchdog's abort branch), and every guard-stopped radius gets
its flux std floored (cost_guard.std_floor).

Run as:

    python tests/dev_tests/test_cgyro_cost_guard.py

Exits non-zero on any assertion failure. Each test prints PASS on success (~1.5 min in total).
"""

from __future__ import annotations

import array
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2]
if str(mitim_root / "src") not in sys.path:
    sys.path.insert(0, str(mitim_root / "src"))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_modules.powertorch.physics_models import transport_cgyro

_spec = importlib.util.spec_from_file_location("cgyro_guard", mitim_root / "templates" / "gk_control" / "cgyro_guard.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

RHO = 0.8040
N_N, N_SPECIES, N_FIELD = 2, 3, 2     # electrons are the last species
TARGETS = {"QeGB": 10.0, "QiGB": 10.0}


def _fake_radius(d, nt=300, cost=600.0, qe=80.0, qi=30.0, t_end=450.0, cost_recent=None,
                 options=None, precision="f", n_moment=4):
    '''
    A radius folder as a running CGYRO leaves it, one output per a/cs up to t = nt: every output
    took `cost` wall seconds (the last 5 `cost_recent`), and the heat fluxes are constant at qe, qi
    (GB), spread over fields and ky. The other moments hold 999 so a wrong index shows.
    The guard files are staged by CostGuard itself, at 100 s per a/cs and floor_time 50 unless
    `options` says otherwise (the tests do not depend on the machine of config_user.json).
    '''
    d.mkdir(parents=True, exist_ok=True)
    (d / "out.cgyro.time").write_text("".join(f" {t:.4E}  1.0E-03  1.0E-08  1.0E-02\n" for t in range(1, nt + 1)))
    totals = [cost] * nt
    if cost_recent is not None:
        totals[-5:] = [cost_recent] * 5
    (d / "out.cgyro.timing").write_text(
        "Setup time\n   input  str_init\n   0.001   10.0\nRun time\n     str      nl     TOTAL\n"
        + "".join(f"   1.000   2.000   {x:.3f}\n" for x in totals))
    (d / "out.cgyro.grids").write_text(f"{N_N}\n{N_SPECIES}\n{N_FIELD}\n184\n16\n")
    (d / "input.cgyro.gen").write_text("0.00000E+00  GAMMA_E\n")
    (d / ".mitim_t_end").write_text(f"none_0 {t_end:g}\n")

    record = np.full((N_SPECIES, n_moment, N_FIELD, N_N), 999.0)
    record[-1, 1] = qe / (N_FIELD * N_N)
    record[:-1, 1] = qi / ((N_SPECIES - 1) * N_FIELD * N_N)
    data = array.array(precision, list(record.flatten(order="F")) * nt)
    (d / "bin.cgyro.ky_flux").write_bytes(data.tobytes())

    options = {"seconds_per_acs": 100.0, "floor_time": 50.0, **(options or {})}
    cost_guard = CGYROtools.CostGuard({"enabled": True, "targets_GB": {RHO: TARGETS}, **options})
    for src, dst in cost_guard.stage(d, None)[RHO]:
        shutil.copy(src, d / dst)
    return d


def _burst(d, nt, n_last, qi):
    '''Overwrite Qi of the last n_last outputs of a _fake_radius flux file (float32, 4 moments).'''
    data = array.array("f")
    data.frombytes((d / "bin.cgyro.ky_flux").read_bytes())
    per_record = N_SPECIES * 4 * N_FIELD * N_N
    for k in range(nt - n_last, nt):
        for i_n in range(N_N):
            for i_field in range(N_FIELD):
                base = k * per_record + N_SPECIES * (1 + 4 * (i_field + N_FIELD * i_n))
                for i_species in range(N_SPECIES - 1):
                    data[base + i_species] = qi / ((N_SPECIES - 1) * N_FIELD * N_N)
    (d / "bin.cgyro.ky_flux").write_bytes(data.tobytes())


def _ramp(d, nt, n_last, qi_start, qi_end):
    '''Qi of the last n_last outputs of a _fake_radius rising linearly from qi_start to qi_end.'''
    for n in range(n_last, 0, -1):
        _burst(d, nt, n, qi_start + (qi_end - qi_start) * (n_last - n) / (n_last - 1))


def test_flux_reader():
    '''Qe is the last species, Qi the sum of the others, both summed over fields and ky; the
    precision and the number of moments are inferred from the file size.'''
    root = Path(tempfile.mkdtemp())
    try:
        for precision, n_moment in (("f", 4), ("f", 3), ("d", 4)):
            d = _fake_radius(root / f"{precision}{n_moment}", nt=120, qe=80.0, qi=30.0, precision=precision, n_moment=n_moment)
            q = guard.heat_fluxes(str(d), 120, 20)
            assert abs(q["QeGB"] - 80.0) < 1e-4 and abs(q["QiGB"] - 30.0) < 1e-4, (precision, n_moment, q)
        # a flux file out of step with out.cgyro.time is not trusted
        try:
            guard.heat_fluxes(str(d), 60, 20)
            raise AssertionError("a record count far from the time rows must be refused")
        except ValueError:
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_flux_reader")


def test_stop_rule():
    '''Every clause of the rule, with the defaults: 100 s per a/cs, 5x target, t >= 250 a/cs
    (waived when >= 12 h remain), never before 50 a/cs.'''
    root = Path(tempfile.mkdtemp())
    cases = [
        # name,                              kwargs,                                           stops
        ("slow, far, past min_time",         dict(nt=300),                                     True),
        ("pt24: t=90, 600 s, 60 h left",     dict(nt=90),                                      True),
        ("before min_time, 5 h left",        dict(nt=200, cost=120.0, t_end=350.0),            False),
        ("waiver never below floor_time",    dict(nt=40),                                      False),
        ("slow but close to target",         dict(nt=300, qe=30.0, qi=30.0),                   False),
        ("far but cheap",                    dict(nt=300, cost=40.0),                          False),
        ("cost recovering",                  dict(nt=300, cost_recent=30.0),                   False),
        ("flux far BELOW target",            dict(nt=300, qe=0.5, qi=0.5),                     False),
        ("only Qi far",                      dict(nt=300, qe=10.0, qi=60.0),                   True),
    ]
    try:
        for i, (name, kwargs, stops) in enumerate(cases):
            v = guard.verdict(str(_fake_radius(root / f"case{i}", **kwargs)))
            assert v.startswith("STOP" if stops else "WAIT"), (name, v)
            print(f"\t({name}: {v})")

        # pt07 H ev1 (2026-10-13): an ion burst in the last 20 a/cs (8.8x) over a trace at ~3x is
        # not a stop, because the ratio is averaged over [max(floor_time, t/2), t]
        d = _fake_radius(root / "burst", nt=348, qe=10.0, qi=30.0)
        _burst(d, nt=348, n_last=20, qi=88.0)
        v = guard.verdict(str(d))
        assert v.startswith("WAIT") and "x_QiGB" in v and "ratio=3." in v, v
        print(f"\t(burst in the window, trace at 3x: {v})")

        # saturation test: over [max(floor_time, t/2), t] = [150, 300] a Qi still rising 30 -> 90
        # (late/early ~1.7) is left running; a slow drift 50 -> 56 (~1.06) is within the 10% tolerance
        d = _fake_radius(root / "rising", nt=300, qe=10.0, qi=30.0)
        _ramp(d, nt=300, n_last=150, qi_start=30.0, qi_end=90.0)
        v = guard.verdict(str(d))
        assert v.startswith("WAIT") and "x_QiGB" in v and "late/early=1.6" in v, v
        print(f"\t(still rising: {v})")
        d = _fake_radius(root / "drifting", nt=300, qe=10.0, qi=30.0)
        _ramp(d, nt=300, n_last=150, qi_start=50.0, qi_end=56.0)
        v = guard.verdict(str(d))
        assert v.startswith("STOP") and "late/early=1.06" in v, v
        d = _fake_radius(root / "rising_tolerant", nt=300, qe=10.0, qi=30.0, options={"rising_tolerance": 1.0})
        _ramp(d, nt=300, n_last=150, qi_start=30.0, qi_end=90.0)
        assert guard.verdict(str(d)).startswith("STOP")
        d = _fake_radius(root / "rising_off", nt=300, qe=10.0, qi=30.0, options={"rising_tolerance": None})
        _ramp(d, nt=300, n_last=150, qi_start=30.0, qi_end=90.0)
        v = guard.verdict(str(d))
        assert v.startswith("STOP") and "late/early" not in v, v

        # the cost is per a/cs, not per output: the same rows with outputs 0.5 a/cs apart cost double
        d = _fake_radius(root / "spacing", nt=300, cost=60.0)
        (d / "out.cgyro.time").write_text("".join(f" {0.5 * t:.4E}  1.0E-03  1.0E-08  1.0E-02\n" for t in range(1, 301)))
        v = guard.verdict(str(d))
        assert "cost=120s_per_acs" in v, v

        # nothing readable is never a stop, and never an exception out of the script
        d = _fake_radius(root / "broken", nt=300)
        (d / "bin.cgyro.ky_flux").write_bytes(b"123")
        out = subprocess.run([sys.executable, str(d / "mitim_guard.py"), str(d)], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.startswith("WAIT guard error"), out
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_stop_rule")


def _watchdog(rho_dir, cmd, mode=None, guard_on=True):
    return CGYROtools.Watchdog.from_load_balance(str(rho_dir), {"min_time": 300}, mode=mode, guard=guard_on).wrap(cmd)


def _finished(rho_dir):
    '''radius_finished on the files as retrieval stores them (<file>_<rho>).'''
    shutil.copy(rho_dir / "mitim_budget.tag", rho_dir / f"mitim_budget.tag_{RHO:.4f}")
    return SIMtools.radius_finished(rho_dir, RHO, ("out.cgyro.info", "EXIT"), "mitim_budget.tag")[0]


def test_watchdog_stops_at_once():
    '''The real watchdog around a CGYRO that would run for 10 more minutes and never writes a
    restart: the guard ends it on its first check (~20 s), far below load_balance.min_time (300),
    and the radius is accepted as finished. A harvest record of it carries guard_stop = 1.'''
    d = Path(tempfile.mkdtemp())
    try:
        _fake_radius(d, nt=90)
        t0 = time.time()
        subprocess.run(["bash", "-c", _watchdog(d, "sleep 600")], timeout=90)
        tag = (d / "mitim_budget.tag").read_text()
        assert tag.startswith("GUARD t=90 ") and "ratio=8.0x_QeGB" in tag, tag
        assert CGYROtools.CostGuard.stopped(tag) and not CGYROtools.CostGuard.stopped("STOP t=90")
        assert not (d / "mitim_discard.tag").exists()
        assert _finished(d), "radius_finished must accept a guard-stopped radius"
        from mitim_tools.gacode_tools.utils import CGYROutils
        (d / "out.cgyro.info").write_text("")
        flags = CGYROutils.harvest_run_fields(types.SimpleNamespace(folder=d, suffix_read=''))
        assert flags['budget_stop'] == 1 and flags['guard_stop'] == 1, flags
        print(f"\t(stopped after {time.time() - t0:.0f} s: {tag.strip()})")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_watchdog_stops_at_once")


def test_watchdog_leaves_the_rest_alone():
    '''No stop when the guard is off for this launch, when the launch is a scheduler extra, when the
    radius is close to target, or when python3 gives no verdict.'''
    root = Path(tempfile.mkdtemp())
    try:
        # python3 that prints nothing, first on PATH
        (root / "bin").mkdir()
        (root / "bin" / "python3").write_text("#!/bin/bash\nexit 127\n")
        (root / "bin" / "python3").chmod(0o755)

        runs = {
            "guard off":       dict(radius=dict(nt=90), kwargs=dict(guard_on=False), env=None),
            "scheduler extra": dict(radius=dict(nt=90), kwargs=dict(mode="stop"), env=None),
            "close to target": dict(radius=dict(nt=300, qe=30.0, qi=30.0), kwargs={}, env=None),
            "no python3":      dict(radius=dict(nt=90), kwargs={}, env={**os.environ, "PATH": f"{root / 'bin'}:{os.environ['PATH']}"}),
        }
        procs = {}
        for i, (name, r) in enumerate(runs.items()):
            d = _fake_radius(root / f"run{i}", **r["radius"])
            procs[name] = (d, subprocess.Popen(["bash", "-c", _watchdog(d, "sleep 35; exit 3", **r["kwargs"])], env=r["env"]))
        for name, (d, proc) in procs.items():
            assert proc.wait(timeout=90) == 3, f"{name}: CGYRO's own exit status must pass through"
            assert not (d / "mitim_budget.tag").exists(), f"{name}: must not be stopped"
        assert (procs["close to target"][0] / "mitim_guard.status").read_text().split()[2] == "WAIT"
        assert "no verdict" in (procs["no python3"][0] / "mitim_guard.status").read_text()
        assert not (procs["guard off"][0] / "mitim_guard.status").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_watchdog_leaves_the_rest_alone")


def test_staging():
    '''CostGuard adds its two files to each radius with targets, on top of what was already being sent
    (restart blobs), and stays off when disabled or without targets.'''
    d = Path(tempfile.mkdtemp())
    try:
        cg = CGYROtools.CostGuard({"enabled": True, "flux_ratio": 8, "seconds_per_acs": 100, "targets_GB": {RHO: TARGETS, 0.5: {"QeGB": 1.0}}})
        files = cg.stage(d, {RHO: [("blob", "bin.cgyro.restart")]})
        assert files[RHO][0] == ("blob", "bin.cgyro.restart") and [dst for _, dst in files[RHO][1:]] == ["mitim_guard.json", "mitim_guard.py"]
        assert [dst for _, dst in files[0.5]] == ["mitim_guard.json", "mitim_guard.py"]
        staged = json.loads(Path(files[RHO][1][0]).read_text())
        assert staged["targets_GB"] == TARGETS and staged["flux_ratio"] == 8.0 and staged["seconds_per_acs"] == 100.0, staged
        assert Path(files[RHO][2][0]).is_file()

        assert not CGYROtools.CostGuard(None).enabled
        assert not CGYROtools.CostGuard({"enabled": False, "targets_GB": {RHO: TARGETS}}).enabled
        assert not CGYROtools.CostGuard({"enabled": True}).enabled, "no targets (standalone run): off"

        assert "_lb_guard=1" in _watchdog(d, "true") and "_lb_guard=0" in _watchdog(d, "true", guard_on=False)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_staging")


def test_seconds_per_acs_per_machine():
    '''seconds_per_acs {gpu, cpu} takes the number of the class of the CGYRO machine in
    config_user.json (gpus_per_node 0 -> cpu); a plain number is used as is.'''
    from mitim_tools.misc_tools import CONFIGread
    original = CONFIGread.machineSettings
    try:
        for gpus, expected in ((4, 100.0), (0, 1000.0)):
            CONFIGread.machineSettings = lambda code=None, gpus=gpus, **kw: {"gpus_per_node": gpus}
            cg = CGYROtools.CostGuard({"enabled": True, "seconds_per_acs": {"gpu": 100, "cpu": 1000}, "targets_GB": {RHO: TARGETS}})
            assert cg.thresholds["seconds_per_acs"] == expected, (gpus, cg.thresholds)
            assert CGYROtools.CostGuard({"enabled": True, "targets_GB": {RHO: TARGETS}}).thresholds["seconds_per_acs"] == expected
        assert CGYROtools.CostGuard({"enabled": True, "seconds_per_acs": 250, "targets_GB": {RHO: TARGETS}}).thresholds["seconds_per_acs"] == 250.0
    finally:
        CONFIGread.machineSettings = original
    print("PASS test_seconds_per_acs_per_machine")


def test_targets_from_transport_layer():
    '''The transport layer hands the guard the turbulent Qe and Qi targets per radius (GB), leaves
    the particle channel out and does not touch the namelist dict.'''
    rhos = [0.5, RHO]
    turb_target = {"QeGB": np.array([3.0, 27.3]), "QiGB": np.array([4.0, 14.4]), "GeGB": np.array([0.0, 0.01])}
    namelist = {"cost_guard": {"enabled": True, "flux_ratio": 5}}

    ctx = types.SimpleNamespace(batched=False, code="cgyro", rho_locations=rhos, turb_target_GB=turb_target,
                                run_kwargs={"cost_guard": namelist["cost_guard"]})
    transport_cgyro.gyrokinetic_model._gk_cost_guard(None, ctx, namelist)
    assert ctx.run_kwargs["cost_guard"]["targets_GB"] == {0.5: {"QeGB": 3.0, "QiGB": 4.0}, RHO: {"QeGB": 27.3, "QiGB": 14.4}}
    assert "targets_GB" not in namelist["cost_guard"]

    # batched dispatch: run_over_plasmas does not take cost_guard, so the key never reaches run_kwargs
    ctx = types.SimpleNamespace(batched=True, code="cgyro", rho_locations=rhos, turb_target_GB=turb_target, run_kwargs={})
    transport_cgyro.gyrokinetic_model._gk_cost_guard(None, ctx, namelist)
    assert "cost_guard" not in ctx.run_kwargs
    assert "cost_guard" not in transport_cgyro._RUN_OVER_PLASMAS_KEYS
    print("PASS test_targets_from_transport_layer")


CASH_KARP = "ERROR: (CGYRO) Cash-Carp step exceeded max iteration count"


def test_abort_rule():
    '''An abort is a guard stop only past floor_time (50) and with Qe or Qi >= abort_flux_ratio x target
    (null = flux_ratio, 5) over the second half of the trace; the cost does not enter.'''
    root = Path(tempfile.mkdtemp())
    cases = [
        # name,                              kwargs,                                                   stops
        ("pt07: t=77, cheap at the end",     dict(nt=77, cost=10.0),                                   True),
        ("abort before floor_time",          dict(nt=40),                                              False),
        ("abort close to target",            dict(nt=77, qe=20.0, qi=20.0),                            False),
        ("accept_abort off",                 dict(nt=77, options={"accept_abort": False}),             False),
        ("abort_flux_ratio above the flux",  dict(nt=77, options={"abort_flux_ratio": 10}),            False),
        ("abort_flux_ratio below flux_ratio", dict(nt=77, qe=40.0, options={"flux_ratio": 5, "abort_flux_ratio": 3}), True),
    ]
    try:
        for i, (name, kwargs, stops) in enumerate(cases):
            v = guard.abort_verdict(str(_fake_radius(root / f"case{i}", **kwargs)))
            assert v.startswith("STOP" if stops else "WAIT"), (name, v)
            print(f"\t({name}: {v})")
        d = root / "case0"
        out = subprocess.run([sys.executable, str(d / "mitim_guard.py"), str(d), "--abort"], capture_output=True, text=True)
        assert out.stdout.startswith("STOP t=77 ratio=8.0x_QeGB"), out
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_abort_rule")


def test_watchdog_accepts_abort():
    '''The real watchdog around a CGYRO that ends on the Cash-Karp abort: over-driven past floor_time
    it leaves a GUARD tag with reason=cash_karp_abort (finished, guard_stop = 1); close to target, or
    with the guard off, no tag. The launch keeps its own exit status either way.'''
    root = Path(tempfile.mkdtemp())
    abort = f"echo '{CASH_KARP}' >> out.cgyro.info; exit 3"
    try:
        d = _fake_radius(root / "far", nt=77)
        rc = subprocess.run(["bash", "-c", f"cd {d}; " + _watchdog(d, abort)], timeout=60).returncode
        tag = (d / "mitim_budget.tag").read_text()
        assert rc == 3, rc
        assert tag.startswith("GUARD t=77 ratio=8.0x_QeGB reason=cash_karp_abort"), tag
        assert _finished(d), "radius_finished must accept an accepted abort"
        from mitim_tools.gacode_tools.utils import CGYROutils
        flags = CGYROutils.harvest_run_fields(types.SimpleNamespace(folder=d, suffix_read=''))
        assert flags['budget_stop'] == 1 and flags['guard_stop'] == 1, flags
        print(f"\t(far: {tag.strip()})")

        for name, kwargs, guard_on in (("near", dict(qe=20.0, qi=20.0), True), ("off", {}, False)):
            d = _fake_radius(root / name, nt=77, **kwargs)
            subprocess.run(["bash", "-c", f"cd {d}; " + _watchdog(d, abort, guard_on=guard_on)], timeout=60)
            assert not (d / "mitim_budget.tag").exists(), name

        # an abort that is not Cash-Karp is never judged
        d = _fake_radius(root / "other", nt=77)
        subprocess.run(["bash", "-c", f"cd {d}; " + _watchdog(d, "echo 'ERROR: (CGYRO) something else' >> out.cgyro.info")], timeout=60)
        assert not (d / "mitim_budget.tag").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_watchdog_accepts_abort")


def test_std_floor():
    '''The transport layer raises every turbulent std of a guard-stopped radius to std_floor x |flux|
    and leaves the other radii, and a radius stopped by hand (STOP tag), untouched.'''
    root = Path(tempfile.mkdtemp())
    try:
        rhos = [0.5, 0.65, RHO]
        sub = root / "Evaluation"
        sub.mkdir()
        (sub / f"mitim_budget.tag_{RHO:.4f}").write_text("GUARD t=77 ratio=8.0x_QeGB reason=cash_karp_abort elapsed=10s\n")
        (sub / "mitim_budget.tag_0.6500").write_text("STOP t=300 elapsed=10s budget=0s min_time=0\n")

        model = transport_cgyro.gyrokinetic_model.__new__(transport_cgyro.gyrokinetic_model)
        model.folder = root
        for key in [spec[0] for spec in transport_cgyro._FLUX_SPEC + (transport_cgyro._EXCHANGE_SPEC,)]:
            setattr(model, f"{key}_turb", np.array([2.0, 3.0, -80.0]))
            setattr(model, f"{key}_turb_stds", np.array([0.1, 0.1, 1.0]))
        ctx = types.SimpleNamespace(rho_locations=rhos, subfolder_name="Evaluation",
                                    run_kwargs={"cost_guard": {"enabled": True, "std_floor": 0.5}})
        model._gk_report_guard_stops(ctx)
        assert np.allclose(model.QeGB_turb_stds, [0.1, 0.1, 40.0]), model.QeGB_turb_stds
        assert np.allclose(model.MtGB_turb_stds, [0.1, 0.1, 40.0]), model.MtGB_turb_stds

        # no floor configured: nothing changes
        model.QeGB_turb_stds = np.array([0.1, 0.1, 1.0])
        ctx.run_kwargs = {"cost_guard": {"enabled": True, "std_floor": None}}
        model._gk_report_guard_stops(ctx)
        assert np.allclose(model.QeGB_turb_stds, [0.1, 0.1, 1.0])
        assert CGYROtools.CostGuard({"enabled": True, "targets_GB": {RHO: TARGETS}}).thresholds["abort_flux_ratio"] == 5.0
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_std_floor")


if __name__ == "__main__":
    test_flux_reader()
    test_stop_rule()
    test_staging()
    test_seconds_per_acs_per_machine()
    test_targets_from_transport_layer()
    test_abort_rule()
    test_std_floor()
    test_watchdog_accepts_abort()
    test_watchdog_stops_at_once()
    test_watchdog_leaves_the_rest_alone()
    print("\nALL PASS")
