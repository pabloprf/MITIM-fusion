"""
test_cgyro_kill.py
==================
Stopping a CGYRO radius on request (`mitim_kill_cgyro`), keeping what it simulated.

Every CGYRO launch runs inside the watchdog of CGYROtools.CGYRO._wall_budget_wrap. A
`mitim_stop` file in the radius folder makes it wait for the next restart write, leave
`mitim_budget.tag` (accepted as finished by SIMtools.radius_finished) and stop CGYRO.
Main radii are never discarded; scheduler extras below min_time still are.

A fake CGYRO (a bash loop that appends to out.cgyro.time and rewrites out.cgyro.tag
every second) stands in for the real code, so this runs anywhere in ~1.5 min.

Run as:

    python tests/dev_tests/test_cgyro_kill.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest import mock

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.gacode_tools.scripts import kill_cgyro

RHO = 0.8334

# Endless fake CGYRO: one a/cs per 0.2 s, a restart write (out.cgyro.tag: step, time) every 5 a/cs
FAKE_CGYRO = r'''t=0
while true; do
    t=$((t+1)); echo "$t.0 0.0" >> "DIR/out.cgyro.time"
    (( t % 5 == 0 )) && printf "%d\n%d.0\n" $t $t > "DIR/out.cgyro.tag"
    sleep 0.2
done'''


def _watchdog(rho_dir, mode=None, load_balance=None, cmd=None):
    cg = types.SimpleNamespace(_load_balance=load_balance, _WALL_BUDGET_WATCHDOG=CGYROtools.CGYRO._WALL_BUDGET_WATCHDOG)
    wrap = types.MethodType(CGYROtools.CGYRO._wall_budget_wrap, cg)
    return wrap(cmd or FAKE_CGYRO.replace("DIR", str(rho_dir)), str(rho_dir), mode=mode)


def _launch(body):
    return subprocess.Popen(["bash", "-c", body], start_new_session=True)


def _finished(rho_dir):
    '''radius_finished on the files as retrieval stores them (<file>_<rho>).'''
    for name in ("out.cgyro.info", "mitim_budget.tag"):
        if (rho_dir / name).exists():
            shutil.copy(rho_dir / name, rho_dir / f"{name}_{RHO:.4f}")
    return SIMtools.radius_finished(rho_dir, RHO, ("out.cgyro.info", "EXIT"), "mitim_budget.tag")[0]


def test_main_radius_stopped_on_request():
    '''Default (manual) mode, stop requested far below min_time: accepted, never discarded.'''
    d = Path(tempfile.mkdtemp())
    try:
        proc = _launch(_watchdog(d, load_balance={"strategy": "extra_points", "min_time": 300}))
        time.sleep(3)
        (d / "mitim_stop").write_text("test\n")
        t_req = time.time()
        proc.wait(timeout=90)
        tag = (d / "mitim_budget.tag").read_text()
        assert tag.startswith("STOP"), tag
        assert not (d / "mitim_discard.tag").exists(), "a main radius must never be discarded"
        assert _finished(d), "radius_finished must accept a stopped main radius"
        t_stop = float(tag.split("t=")[1].split()[0])
        print(f"\t(stopped {time.time() - t_req:.0f} s after the request, at t={t_stop:g})")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_main_radius_stopped_on_request")


# Fake mpirun/prterun: the "rank" runs in its own process group (as prterun does), the launcher
# forwards TERM to it and exits at once, and the rank needs 3 s to die
FAKE_MPIRUN = r"""python3 -c '
import os, signal, time
os.setpgid(0, 0)
signal.signal(signal.SIGTERM, lambda *a: (time.sleep(3), os._exit(0)))
t = 0
while True:
    t += 1
    open("DIR/out.cgyro.time", "a").write(f"{t}.0 0.0\n")
    if t % 5 == 0: open("DIR/out.cgyro.tag", "w").write(f"{t}\n{t}.0\n")
    time.sleep(0.2)
' & _rank=$!
echo $_rank > "DIR/rank.pid"
trap 'kill -TERM $_rank; exit 143' TERM
wait $_rank"""


def test_stop_waits_for_ranks_in_own_process_group():
    '''The launch only returns once every rank is gone, even ranks outside its process group
    (a rank still alive would hold the node/GPUs the next call is placed on).'''
    d = Path(tempfile.mkdtemp())
    try:
        proc = _launch(_watchdog(d, cmd=FAKE_MPIRUN.replace("DIR", str(d))))
        time.sleep(3)
        (d / "mitim_stop").write_text("test\n")
        proc.wait(timeout=120)
        rank = int((d / "rank.pid").read_text())
        alive = subprocess.run(["kill", "-0", str(rank)], capture_output=True).returncode == 0
        assert not alive, "the watchdog returned while a rank was still running"
        assert (d / "mitim_budget.tag").exists()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_stop_waits_for_ranks_in_own_process_group")


def test_howard_on_short_stopped_trace():
    '''A radius stopped with fewer samples than howard_gkav's smoothing window (25) still averages
    (fallback window) instead of crashing the whole evaluation.'''
    import numpy as np
    from mitim_tools.simulation_tools.utils import GKwindow_howard
    for n in (2, 8, 24):
        t = np.arange(n, dtype=float)
        y = 1.0 + 0.1 * np.sin(t)
        res = GKwindow_howard.select_start(t, y, y, y)
        assert res["flag"] in GKwindow_howard.FLAGS_FALLBACK and t[0] <= res["t_start"] <= t[-1], (n, res["flag"], res["t_start"])
    print("PASS test_howard_on_short_stopped_trace")


def test_extra_below_min_time_discarded():
    '''Scheduler extras keep the old behavior: a stop below min_time discards the case.'''
    d = Path(tempfile.mkdtemp())
    try:
        proc = _launch(_watchdog(d, mode="stop", load_balance={"strategy": "extra_points", "min_time": 300}))
        time.sleep(3)
        (d / "mitim_stop").write_text("test\n")
        proc.wait(timeout=90)
        assert (d / "mitim_discard.tag").exists()
        assert not (d / "mitim_budget.tag").exists()
        assert not _finished(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_extra_below_min_time_discarded")


def test_normal_end_is_fast_and_keeps_rc():
    '''Without a request the watchdog adds no delay (1 s polling) and passes CGYRO's exit status through.
    A stop file left over from an earlier launch is cleared at launch.'''
    d = Path(tempfile.mkdtemp())
    try:
        (d / "mitim_stop").write_text("stale\n")
        t0 = time.time()
        rc = subprocess.run(["bash", "-c", _watchdog(d, cmd="sleep 1; exit 3")]).returncode
        assert rc == 3, rc
        assert time.time() - t0 < 5, f"watchdog delayed a finished call by {time.time() - t0:.0f} s"
        assert not (d / "mitim_stop").exists()
        assert not (d / "mitim_budget.tag").exists()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_normal_end_is_fast_and_keeps_rc")


def test_cli_on_bash_mode_run():
    '''mitim_kill_cgyro on a fake bash-mode PORTALS run: finds the scratch from the staged script,
    reports each radius, and drops mitim_stop only where asked and only in running radii.'''
    root = Path(tempfile.mkdtemp())
    try:
        scratch = root / "scratch"
        (root / "namelist.portals.yaml").write_text(
            "transport:\n  evaluator_instance_attributes:\n    turbulence_model: cgyro\n  options:\n    cgyro:\n      read: {}\n")
        tmp = root / "Execution" / "Evaluation.3" / "transport_simulation_folder" / "tmp_cgyro"
        for rho, info in ((0.6673, "EXIT: (CGYRO) Reached MAX_TIME\n"), (RHO, "")):
            (tmp / "base_cgyro" / f"rho_{rho:.4f}").mkdir(parents=True)
            r = scratch / "base_cgyro" / f"rho_{rho:.4f}"
            r.mkdir(parents=True)
            (r / "input.cgyro").write_text("MAX_TIME=4.25000E+02\nPRINT_STEP=100\n")
            (r / ".mitim_t0").write_text("25\n")
            (r / "out.cgyro.info").write_text(info)
            (r / "out.cgyro.time").write_text("".join(f"{t}.0 0.0\n" for t in range(26, 187)))
        (tmp / "mitim_bash.src").write_text(f"cd {scratch}\n")

        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["mitim_kill_cgyro", str(root), "--all", "--yes"]), contextlib.redirect_stdout(out):
            kill_cgyro.main()
        text = out.getvalue()
        assert (scratch / "base_cgyro" / f"rho_{RHO:.4f}" / "mitim_stop").exists(), text
        assert not (scratch / "base_cgyro" / "rho_0.6673" / "mitim_stop").exists(), "a finished radius must not be targeted"
        assert "450.0" in text, "the end time is MAX_TIME past the launch start t0 (425 + 25)"
        assert "Evaluation" in text or "evaluation 3" in text, text

        # a second look reports the pending request
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["mitim_kill_cgyro", str(root)]), contextlib.redirect_stdout(out):
            kill_cgyro.main()
        assert "stop requested" in out.getvalue(), out.getvalue()

        # --rho matching nothing running is refused
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["mitim_kill_cgyro", str(root), "--rho", "0.5", "--yes"]), contextlib.redirect_stdout(out):
            kill_cgyro.main()
        assert "matches no running radius" in out.getvalue(), out.getvalue()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS test_cli_on_bash_mode_run")


if __name__ == "__main__":
    test_normal_end_is_fast_and_keeps_rc()
    test_cli_on_bash_mode_run()
    test_howard_on_short_stopped_trace()
    test_main_radius_stopped_on_request()
    test_extra_below_min_time_discarded()
    test_stop_waits_for_ranks_in_own_process_group()
    print("\nALL PASS")
