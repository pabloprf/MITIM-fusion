"""FFT-friendly BOX_SIZE/N_RADIAL selection (CGYROutils.fft_friendly_grid)."""
import pytest

from mitim_tools.gacode_tools.utils.CGYROutils import _is_smooth, compute_box_and_nradial, fft_friendly_grid


@pytest.mark.parametrize("box, n, expected", [
    ((31, 248), None, (30, 240)),    # 372 = 2^2*3*31 -> 360
    ((38, 266), None, (36, 252)),    # 399 = 3*7*19   -> 378 (tie with 40/280 goes to the smaller N)
    ((33, 264), None, (32, 256)),    # 396 = ...*11   -> 384
    ((7, 266), None, (7, 252)),      # only N moves: kx_max -5.3%
])
def test_bad_grids_move_to_nearest_fast_one(box, n, expected):
    assert fft_friendly_grid(*box) == expected


@pytest.mark.parametrize("pair", [(30, 270), (32, 256), (36, 252), (40, 240), (9, 252), (25, 250)])
def test_fast_grids_are_untouched(pair):
    assert fft_friendly_grid(*pair) == pair


@pytest.mark.parametrize("pair", [(11, 264), (24, 264)])
def test_no_candidate_within_tolerance_keeps_grid(pair):
    assert fft_friendly_grid(*pair) == pair


def test_every_result_is_fast_even_and_multiple_of_box():
    for b in range(5, 46):
        r = round(256 / b); r += (r * b) % 2
        box, n = fft_friendly_grid(b, r * b)
        assert n % 2 == 0 and n % box == 0
        if (box, n) != (b, r * b):
            assert _is_smooth(3 * n // 2)
            assert abs(box / b - 1) <= 0.06 + 1e-9
            assert abs(((n / 2 - 1) / box) / ((r * b / 2 - 1) / b) - 1) <= 0.06 + 1e-9


def test_compute_box_and_nradial_flag():
    # rho=0.7743 of the ARC rapids case: q 2.133, s 4.101, r/a 0.875, KY 0.08 -> BOX 30 / N 270 either way
    kw = dict(q=2.13268, shear=4.10091, rmin=0.875012, ky_min=0.08, L_x=90, N_radial=256)
    assert compute_box_and_nradial(**kw) == compute_box_and_nradial(**kw, fft_friendly=False) == (30, 270)
    # a shear that lands on BOX 31 is moved only when the flag is on
    kw["shear"] = 4.30
    old = compute_box_and_nradial(**kw, fft_friendly=False)
    new = compute_box_and_nradial(**kw)
    assert old == (31, 248) and new == (30, 240)
