"""Toroidal modes that connect to themselves after one poloidal turn (CGYROutils.self_connected_modes) and their warning."""
import contextlib
import io
from types import SimpleNamespace

import pytest

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.gacode_tools.utils.CGYROutils import self_connected_modes

N_TOROIDAL = 16


def _turns(box_size, n_radial, n):
    # radial indices mode n visits before coming back: walk CGYRO's end-of-field-line connection ir -> ir + n*BOX_SIZE
    ir, visited = 0, set()
    while ir not in visited:
        visited.add(ir)
        ir = (ir + n * box_size) % n_radial
    return len(visited)


@pytest.mark.parametrize("ratio, expected", [
    (8, [8]),
    (7, [7, 14]),
    (9, [9]),
    (10, [10]),
    (5, [5, 10, 15]),
    (16, []),
    (38, []),
])
@pytest.mark.parametrize("box_size", [1, 7, 30])
def test_integer_ratio(ratio, expected, box_size):
    assert self_connected_modes(box_size, ratio * box_size, N_TOROIDAL) == expected


@pytest.mark.parametrize("box_size, n_radial", [(6, 16), (30, 100), (12, 250), (31, 240)])
def test_n_radial_not_a_multiple_of_box_size(box_size, n_radial):
    assert n_radial % box_size
    brute = [n for n in range(1, N_TOROIDAL) if _turns(box_size, n_radial, n) == 1]
    assert self_connected_modes(box_size, n_radial, N_TOROIDAL) == brute
    if (box_size, n_radial) == (6, 16):
        assert brute == [8]       # 16 divides 6*n only for n = 8
    if (box_size, n_radial) == (30, 100):
        assert brute == [10]      # 100 divides 30*n only for n = 10


def test_formula_matches_the_walk_everywhere():
    for box_size in range(1, 41):
        for n_radial in range(2, 301, 2):
            assert self_connected_modes(box_size, n_radial, N_TOROIDAL) == \
                [n for n in range(1, N_TOROIDAL) if _turns(box_size, n_radial, n) == 1], (box_size, n_radial)


def test_single_mode_has_nothing_to_report():
    assert self_connected_modes(1, 12, 1) == []


def _prepare_log(extraOptions, code_settings="Nonlinear_reduced2", rhos=(0.35, 0.7743)):
    # what CGYRO._enforce_toroidals_per_proc prints for the grid that will be written (controls -> model -> extraOptions)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        cg = CGYROtools.CGYRO(rhos=list(rhos))
        cg.inputs_files = {rho: SimpleNamespace(controls={}) for rho in cg.rhos}
        out = cg._enforce_toroidals_per_proc(dict(extraOptions), {"resources_per_call": 4}, code_settings=code_settings)
    return log.getvalue(), out


def test_warning_per_affected_radius_and_nothing_else_changes():
    # Nonlinear_reduced2 has N_TOROIDAL 16: ratio 9 at the first radius closes n = 9, ratio 8 at the second closes n = 8
    grid = {"BOX_SIZE": [30, 30], "N_RADIAL": [270, 240]}
    log, out = _prepare_log(grid)
    lines = [l for l in log.splitlines() if "connect to themselves" in l]
    assert len(lines) == 2
    assert "rho=0.3500" in lines[0] and "BOX_SIZE=30 and N_RADIAL=270 (N_RADIAL/BOX_SIZE=9)" in lines[0] and "n=[9] of N_TOROIDAL=16" in lines[0]
    assert "rho=0.7743" in lines[1] and "N_RADIAL/BOX_SIZE=8" in lines[1] and "n=[8] of N_TOROIDAL=16" in lines[1]
    assert "N_RADIAL/BOX_SIZE >= N_TOROIDAL" in lines[1] and "set by the grid and not by the plasma" in lines[1]
    assert {k: out[k] for k in grid} == grid


def test_warning_also_with_a_user_toroidals_per_proc():
    # the hand-set TOROIDALS_PER_PROC branch returns early; the warning must not depend on it
    log, _ = _prepare_log({"BOX_SIZE": 30, "N_RADIAL": 240, "TOROIDALS_PER_PROC": 4})
    assert log.count("connect to themselves") == 2   # same grid at both radii


def test_no_warning_when_no_mode_closes_or_the_run_is_not_nonlinear_multi_mode():
    assert "connect to themselves" not in _prepare_log({"BOX_SIZE": [6, 7], "N_RADIAL": [96, 266]})[0]      # ratios 16 and 38
    assert "connect to themselves" not in _prepare_log({"BOX_SIZE": 30, "N_RADIAL": 240, "N_TOROIDAL": 1})[0]
    assert "connect to themselves" not in _prepare_log({"BOX_SIZE": 30, "N_RADIAL": 240, "N_TOROIDAL": 16}, code_settings="Linear")[0]
