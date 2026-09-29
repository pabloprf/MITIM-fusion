"""
test_maestro_prune_levels.py
============================
Sanity tests for MAESTRO's graded pruning (maestro.prune_level, 0-3) added in
MAESTRObeat.beat.prune_run_folder / prune_initializer, plus the per-beat
scratch tables and the keep_all_files back-compat sentinel.

Pruning is irreversible, so what these tests really lock down is the SURVIVAL
set: at every level, beat_results/ and the load-bearing initializer files must
still be there (they are the sole idempotence key, the engineering-parameter
freeze source, and what mitim_plot_maestro reads).

Builds synthetic beat trees on disk -- no MAESTRO run, no cluster.

Run as:

    python tests/dev_tests/test_maestro_prune_levels.py

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

from mitim_modules.maestro.MAESTROmain import _resolve_prune_level
from mitim_modules.maestro.utils.MAESTRObeat import (
    PRUNE_NOTHING, PRUNE_SCRATCH, PRUNE_RUN, PRUNE_OUTPUTS)
from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.EPEDbeat import eped_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat


class _FakeMaestro:
    '''Minimal stand-in: the prune paths only ever read folder_beats and prune_level'''

    def __init__(self, folder, prune_level):
        self.folder = Path(folder)
        self.folder_beats = Path(folder)
        self.prune_level = prune_level
        self.counter_current = 1
        self.master_seed = 0


def _touch(path, size_kb=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * (size_kb * 1024))


def _build_tree(root, beat_cls, run_name):
    '''One beat folder with a run tree, beat_results and two initializers'''

    beat_folder = root / 'Beat_1'
    run = beat_folder / f'run_{run_name}'

    # Artifacts common to every beat
    _touch(beat_folder / 'beat_results' / 'input.gacode')
    _touch(run / 'input.gacode')

    if run_name == 'transp':
        _touch(run / '83126P01.CDF', 40)                       # kept until level 2
        _touch(run / '83126P01PH.CDF', 20)                     # scratch
        _touch(run / 'results' / 'CMOD.00' / '83126P01.CDF', 40)  # duplicate, scratch
        _touch(run / 'paramiko.log')
        _touch(run / 'MIT83126.CUR')                           # ufile, kept until level 2
    elif run_name == 'eped':
        _touch(run / 'case1' / 'output_run1.nc', 10)           # kept until level 2
        _touch(run / 'case1' / 'run1' / 'eped.input.1')        # file, kept until level 2
        _touch(run / 'case1' / 'run1' / 'height_1' / 'toq.log', 30)   # scratch
        _touch(run / 'case1' / 'run1' / 'height_2' / 'toq.log', 30)   # scratch
    elif run_name == 'portals':
        _touch(run / 'Outputs' / 'optimization_object.pkl', 10)       # kept until level 2
        _touch(run / 'Execution' / 'Evaluation.0' / 'model' / 'out', 50)  # scratch
        _touch(run / 'Initialization' / 'sr_0' / 'powerstate.pkl', 20)    # scratch
        _touch(run / 'flux_match' / 'optimization_data.csv')              # scratch

    # Initializers: load-bearing files + throwaway intermediates + a nested EPED tree
    ini = beat_folder / 'initializer_freegs'
    _touch(ini / 'input.gacode')
    _touch(ini / 'input.geqdsk', 10)
    _touch(ini / 'freegs.geqdsk', 10)
    _touch(ini / 'input.geqdsk.gacode', 5)

    ini_eped = beat_folder / 'initializer_eped'
    _touch(ini_eped / 'input.gacode')
    _touch(ini_eped / 'beat_results' / 'eped_results.npy')
    _touch(ini_eped / 'run_eped' / 'case1' / 'run1' / 'height_1' / 'toq.log', 60)

    return beat_folder


def _run_prune(beat_cls, run_name, level):
    tmp = Path(tempfile.mkdtemp())
    try:
        beat_folder = _build_tree(tmp, beat_cls, run_name)
        m = _FakeMaestro(tmp, level)
        b = beat_cls(m, folder_name=beat_folder) if run_name == 'eped' else beat_cls(m)
        b.prune_run_folder()
        b.prune_initializer()
        yield_paths = {str(p.relative_to(beat_folder)) for p in beat_folder.rglob('*') if p.is_file()}
        return yield_paths
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _assert_survivors(surviving, must_have, must_not_have, label):
    for f in must_have:
        assert f in surviving, f'{label}: expected {f} to SURVIVE, got {sorted(surviving)}'
    for f in must_not_have:
        assert f not in surviving, f'{label}: expected {f} to be PRUNED, got {sorted(surviving)}'


def test_level_0_keeps_everything():
    for cls, name in [(transp_beat, 'transp'), (portals_beat, 'portals')]:
        surviving = _run_prune(cls, name, PRUNE_NOTHING)
        assert 'beat_results/input.gacode' in surviving
        assert f'run_{name}/input.gacode' in surviving
        assert 'initializer_freegs/freegs.geqdsk' in surviving, 'level 0 must not touch initializers'
        assert 'initializer_eped/run_eped/case1/run1/height_1/toq.log' in surviving
    print('PASS test_level_0_keeps_everything')


def test_level_1_drops_scratch_only():
    surviving = _run_prune(transp_beat, 'transp', PRUNE_SCRATCH)
    _assert_survivors(
        surviving,
        must_have=['run_transp/83126P01.CDF', 'run_transp/MIT83126.CUR', 'beat_results/input.gacode'],
        must_not_have=['run_transp/83126P01PH.CDF', 'run_transp/results/CMOD.00/83126P01.CDF',
                       'run_transp/paramiko.log'],
        label='transp level 1')

    surviving = _run_prune(portals_beat, 'portals', PRUNE_SCRATCH)
    _assert_survivors(
        surviving,
        must_have=['run_portals/Outputs/optimization_object.pkl', 'beat_results/input.gacode'],
        must_not_have=['run_portals/Execution/Evaluation.0/model/out',
                       'run_portals/Initialization/sr_0/powerstate.pkl',
                       'run_portals/flux_match/optimization_data.csv'],
        label='portals level 1')

    # eped drops the per-height dirs but keeps the sibling files and output_run1.nc
    surviving = _run_prune(eped_beat, 'eped', PRUNE_SCRATCH)
    _assert_survivors(
        surviving,
        must_have=['run_eped/case1/output_run1.nc', 'run_eped/case1/run1/eped.input.1'],
        must_not_have=['run_eped/case1/run1/height_1/toq.log', 'run_eped/case1/run1/height_2/toq.log'],
        label='eped level 1')

    # Initializers are untouched below level 3
    assert 'initializer_freegs/freegs.geqdsk' in surviving
    print('PASS test_level_1_drops_scratch_only')


def test_level_2_wipes_run_folder():
    for cls, name in [(transp_beat, 'transp'), (portals_beat, 'portals')]:
        surviving = _run_prune(cls, name, PRUNE_RUN)
        assert not [f for f in surviving if f.startswith(f'run_{name}/')], \
            f'level 2 must empty run_{name}/, got {sorted(surviving)}'
        assert 'beat_results/input.gacode' in surviving
        assert 'initializer_freegs/freegs.geqdsk' in surviving, 'level 2 must not touch initializers'
    print('PASS test_level_2_wipes_run_folder')


def test_level_3_prunes_initializers_but_keeps_the_load_bearing_ones():
    surviving = _run_prune(portals_beat, 'portals', PRUNE_OUTPUTS)
    _assert_survivors(
        surviving,
        must_have=[
            'beat_results/input.gacode',                      # the sole idempotence key
            'initializer_freegs/input.gacode',                # engineering-parameter freeze
            'initializer_freegs/input.geqdsk',                # mitim_plot_maestro
            'initializer_eped/input.gacode',
            'initializer_eped/beat_results/eped_results.npy',  # EPED creator _inform_save on restart
        ],
        must_not_have=[
            'initializer_freegs/freegs.geqdsk',
            'initializer_freegs/input.geqdsk.gacode',
            'initializer_eped/run_eped/case1/run1/height_1/toq.log',
        ],
        label='level 3')
    print('PASS test_level_3_prunes_initializers_but_keeps_the_load_bearing_ones')


def test_keep_all_files_backcompat():
    assert _resolve_prune_level(None) == PRUNE_NOTHING
    assert _resolve_prune_level(None, keep_all_files=True) == PRUNE_NOTHING
    assert _resolve_prune_level(None, keep_all_files=False) == PRUNE_OUTPUTS
    assert _resolve_prune_level(2, keep_all_files=True) == 2, 'explicit prune_level must win'
    for bad in (7, -1, 'high'):
        try:
            _resolve_prune_level(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for prune_level={bad}')
    print('PASS test_keep_all_files_backcompat')


if __name__ == '__main__':
    test_level_0_keeps_everything()
    test_level_1_drops_scratch_only()
    test_level_2_wipes_run_folder()
    test_level_3_prunes_initializers_but_keeps_the_load_bearing_ones()
    test_keep_all_files_backcompat()
    print('\nAll MAESTRO prune-level tests passed.')
