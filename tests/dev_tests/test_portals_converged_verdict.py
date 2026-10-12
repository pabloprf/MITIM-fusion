"""
test_portals_converged_verdict.py
=================================
Stopping is not converging. MITIM_BO.check_stopping_criteria returns the STOP flag of the
stopping criteria, while MITIM_BO.converged (what MAESTRO writes to portals_converged.txt and
counts toward maestro.max_unconverged_portals_beats) is, for PORTALS, the Ricci metric alone:
a PORTALS run stopped by stagnating inputs or by the residual target with Ricci above threshold
is NOT converged. Generic BO (no verdict set by its criteria) keeps converged == stop.

The default criteria and the Ricci metric are monkeypatched -- no PORTALS run.

Run as:

    python tests/dev_tests/test_portals_converged_verdict.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALStools


def _stub(criteria):
    return SimpleNamespace(optimization_options={"convergence_options": {
        "stopping_criteria": criteria,
        "stopping_criteria_parameters": {"ricci_value": 0.1},
    }})


def _check(default_stop, chiR_min):
    STRATEGYtools.stopping_criteria_default = lambda mitim_bo, parameters={}: (default_stop, np.zeros(3))
    PORTALStools.calculate_Ricci_portals_step = lambda mitim_bo, d0=2.0, la=0.5: np.array([chiR_min, 1.0])
    mitim_bo = _stub(PORTALStools.stopping_criteria_portals)
    stop = STRATEGYtools.MITIM_BO.check_stopping_criteria(mitim_bo)
    return stop, mitim_bo.converged


def test_stagnated_or_residual_stop_is_not_converged():
    assert _check(default_stop=True, chiR_min=0.906) == (True, False)
    print("PASS test_stagnated_or_residual_stop_is_not_converged")


def test_ricci_below_threshold_is_converged():
    assert _check(default_stop=False, chiR_min=0.05) == (True, True)
    assert _check(default_stop=True, chiR_min=0.05) == (True, True)
    print("PASS test_ricci_below_threshold_is_converged")


def test_running_is_not_converged():
    assert _check(default_stop=False, chiR_min=0.5) == (False, False)
    print("PASS test_running_is_not_converged")


def test_generic_bo_converged_is_stop():
    for flag in (True, False):
        mitim_bo = _stub(lambda mitim_bo, parameters={}: (flag, None))
        assert STRATEGYtools.MITIM_BO.check_stopping_criteria(mitim_bo) == flag
        assert mitim_bo.converged == flag
    print("PASS test_generic_bo_converged_is_stop")


if __name__ == "__main__":
    test_stagnated_or_residual_stop_is_not_converged()
    test_ricci_below_threshold_is_converged()
    test_running_is_not_converged()
    test_generic_bo_converged_is_stop()
