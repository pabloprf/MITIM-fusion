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
    assert compute_box_and_nradial(**kw) == compute_box_and_nradial(**kw, fft_friendly=True) == (30, 270)
    # a shear that lands on BOX 31 is moved only when the flag is on; the code default is off (plain recipe)
    kw["shear"] = 4.30
    assert compute_box_and_nradial(**kw) == compute_box_and_nradial(**kw, fft_friendly=False) == (31, 248)
    assert compute_box_and_nradial(**kw, fft_friendly=True) == (30, 240)


def _prepared_grid(run_preprocess_options, code_settings="Nonlinear_reduced2"):
    # KY / BOX_SIZE / N_RADIAL that CGYRO._run_prepare hands on for one radius, with the model's preprocess_options
    # merged per key under the run-level ones (the SIMtools part and the other input enforcement stubbed out)
    import contextlib, io
    from types import SimpleNamespace
    from mitim_tools.gacode_tools import CGYROtools
    from mitim_tools.simulation_tools import SIMtools

    with contextlib.redirect_stdout(io.StringIO()):
        cg = CGYROtools.CGYRO(rhos=[0.7743])
        cg.inputs_files = {cg.rhos[0]: SimpleNamespace(plasma={"Q": 2.13268, "S": 4.30, "RMIN": 0.875012})}
        for name in ("_enforce_toroidals_per_proc", "_enforce_print_step", "_enforce_restart_step"):
            setattr(cg, name, lambda extraOptions, *a, **k: extraOptions)
        seen = {}
        keep = SIMtools.mitim_simulation._run_prepare
        SIMtools.mitim_simulation._run_prepare = lambda self, *a, **k: seen.update(k["extraOptions"]) or (None, None)
        try:
            cg._preprocess_options = run_preprocess_options
            cg._run_prepare("base_cgyro", extraOptions={}, code_settings=code_settings)
        finally:
            SIMtools.mitim_simulation._run_prepare = keep
    return seen


GRID_KEYS = ("KY", "BOX_SIZE", "N_RADIAL")


def test_run_level_fft_friendly_merges_per_key_over_the_model():
    # Nonlinear_reduced2 (ky_min 0.08, L_x 90, N_radial 256, min_box_size 85) at a shear that lands on BOX_SIZE 31
    def grid(run_level):
        seen = _prepared_grid(run_level)
        return seen["KY"], seen["BOX_SIZE"], seen["N_RADIAL"]

    # key absent (older namelist, standalone call) or false: the plain recipe, with the model's other keys
    for run_level in (None, {}, {"fft_tol": 0.06}, {"fft_friendly": False}):
        assert grid(run_level) == ([0.08], [31], [248]), run_level
    # the namelist switch (transport.options.cgyro.run.preprocess_options.fft_friendly) on: still the model's other keys
    for run_level in ({"fft_friendly": True}, {"fft_friendly": True, "fft_tol": 0.06}):
        assert grid(run_level) == ([0.08], [30], [240]), run_level
    # a tolerance too tight for 31 -> 30 (box length -3.2%) keeps the plain grid
    assert grid({"fft_friendly": True, "fft_tol": 0.01}) == ([0.08], [31], [248])


def test_fft_keys_alone_do_not_switch_preprocessing_on():
    # "Linear" has no preprocess_options: with no block, or only the fft keys, KY / BOX_SIZE / N_RADIAL are left alone
    for run_level in (None, {}, {"fft_friendly": True}, {"fft_friendly": False}, {"fft_friendly": True, "fft_tol": 0.06}):
        assert not set(GRID_KEYS) & set(_prepared_grid(run_level, code_settings="Linear")), run_level
    # a linear run that passes a grid key on purpose still gets the preprocessing (plain recipe unless asked otherwise)
    grid_keys = {"ky_min": 0.08, "L_x": 90, "N_radial": 256, "min_box_size": 85}
    seen = _prepared_grid(grid_keys, code_settings="Linear")
    assert (seen["KY"], seen["BOX_SIZE"], seen["N_RADIAL"]) == ([0.08], [31], [248])
    assert _prepared_grid({**grid_keys, "fft_friendly": True}, code_settings="Linear")["BOX_SIZE"] == [30]
    assert _prepared_grid({"ky_min": 0.3}, code_settings="Linear")["KY"] == [0.3]


def test_portals_template_switch_is_live_and_reaches_the_grid():
    # templates/namelist.portals.yaml carries the switch on; what PORTALS forwards from it to CGYRO.run gives the fast grid
    from mitim_tools import __mitimroot__
    from mitim_tools.misc_tools import IOtools
    from mitim_tools.gacode_tools import CGYROtools
    from mitim_modules.powertorch.physics_models import transport_cgyro

    run = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.portals.yaml")["transport"]["options"]["cgyro"]["run"]
    assert run["preprocess_options"] == {"fft_friendly": True, "fft_tol": 0.06}
    forwardable = transport_cgyro._forwardable_kwargs(CGYROtools.CGYRO, "run", transport_cgyro._SINGLE_EXPLICIT_KWARGS)
    run_kwargs = {k: v for k, v in run.items() if k in forwardable}
    assert _prepared_grid(run_kwargs["preprocess_options"])["BOX_SIZE"] == [30]
    # a user namelist without the block is taken as is (PORTALSmain reads the given file, no merge under it): selector off
    del run["preprocess_options"]
    assert _prepared_grid(run.get("preprocess_options"))["BOX_SIZE"] == [31]
