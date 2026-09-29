"""
test_run_over_plasmas_kwargs.py
===============================
The batched CGYRO path (transport_cgyro) forwards a fixed allow-list of
`transport.options.cgyro.run` keys into SIMtools.mitim_simulation.run_over_plasmas.
That method takes no **kwargs, so any key in the allow-list that is not one of its
parameters raises TypeError at submission time — on the shipped template namelist,
which carries rescue_interrupted and load_balance.

Run as:

    python tests/dev_tests/test_run_over_plasmas_kwargs.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.simulation_tools import SIMtools
from mitim_modules.powertorch.physics_models import transport_cgyro


def test_allow_list_is_accepted_by_run_over_plasmas():
    accepted = set(inspect.signature(SIMtools.mitim_simulation.run_over_plasmas).parameters)
    missing = transport_cgyro._RUN_OVER_PLASMAS_KEYS - accepted
    assert not missing, f"run_over_plasmas does not accept: {sorted(missing)}"
    print("PASS: every forwarded run key is a run_over_plasmas parameter")


def test_template_run_keys_are_forwarded():
    # The knobs the shipped namelist sets and that were being dropped/crashing
    for key in ("rescue_interrupted", "load_balance"):
        assert key in transport_cgyro._RUN_OVER_PLASMAS_KEYS, key
    print("PASS: rescue_interrupted and load_balance are in the allow-list")


def test_rescue_interrupted_reaches_the_run_layer():
    # run_over_plasmas must hand rescue_interrupted to _run, which reads it from kwargs_run
    captured = {}

    class Fake(SIMtools.mitim_simulation):
        def __init__(self):
            self.FolderGACODE = Path(".")
            self.run_specifications = {"code": "cgyro"}

        def _prepare_plasmas_state(self, *args, **kwargs):
            return {}, {}, {}

        def _run(self, code_executor, **kwargs_run):
            captured.update(kwargs_run)
            captured["_load_balance_during_run"] = getattr(self, "_load_balance", None)

    fake = Fake()
    fake.run_over_plasmas([], "base", rescue_interrupted=True, load_balance={"strategy": "extra_points"})
    assert captured["rescue_interrupted"] is True, captured
    assert captured["_load_balance_during_run"] == {"strategy": "extra_points"}, captured
    assert fake._load_balance is None, "load_balance must be reset after the run"
    print("PASS: rescue_interrupted forwarded to _run, load_balance set then reset")


if __name__ == "__main__":
    test_allow_list_is_accepted_by_run_over_plasmas()
    test_template_run_keys_are_forwarded()
    test_rescue_interrupted_reaches_the_run_layer()
    print("\nALL TESTS PASSED")
