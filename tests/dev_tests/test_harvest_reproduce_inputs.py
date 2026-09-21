"""
test_harvest_reproduce_inputs.py
================================
Round trip of real input files through the harvest: input.tglf / input.neo / input.cgyro as MITIM
wrote them for real runs (tests/data) -> the record path of the simulation objects
(SIMtools.mitim_simulation.harvest_records + the output classes' harvest_inputs) -> staging ->
push into a temporary netCDF -> harvest_database.input_file() -> the reconstructed file must parse
(SIMtools.buildDictFromInput, the parser MITIM itself uses) to exactly the same keys, in the same
order, with identical values and identical Python types (bool / int / float / str).

Cases: TGLF (bools, a string key), NEO (5 species, general Miller SHAPE_COS/SIN), CGYRO with 4 and
7 species (general Miller) staged in the SAME run (different key sets: the NaN fills of the 4-species
record must not leak into its file), and the same run pushed twice.

What is stubbed: the transport codes are not run. TGLF/NEO: the simulation object carries the
parsed input file exactly as SIMtools.read() builds it (buildDictFromInput of the file text) and a
plain GACODEoutput with fluxes. CGYRO: a real CGYROoutput (no pygacode read) pointed at a folder
holding the input file, so its own harvest_inputs() reads input.cgyro<suffix>.

Run as:

    python tests/dev_tests/test_harvest_reproduce_inputs.py
"""

import sys
import json
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))

from mitim_tools.harvest_tools import HARVESTtools as H
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.SIMtools import buildDictFromInput
from mitim_tools.gacode_tools.utils.CGYROutils import CGYROoutput

DATA = root / "tests" / "data"


def _sim(code, rhos, parsed, outputs):
    '''Stand-in simulation object after read(): what the base harvest_records() touches'''
    ns = SimpleNamespace(run_specifications={'code': code}, rhos=list(rhos),
                         results={'base': {'parsed': parsed, 'output': outputs, 'x': np.array(rhos)}},
                         simulation_job=SimpleNamespace(machineSettings={'machine': 'local', 'modules': ''}),
                         in_process=False, FolderSimLast=Path('base'), inputs_files={})
    ns.harvest_records = lambda label, folder=None: SIMtools.mitim_simulation.harvest_records(ns, label, folder=folder)
    return ns


def _gacode_output(**fluxes):
    o = SIMtools.GACODEoutput()
    for k, v in fluxes.items():
        setattr(o, k, v)
    return o


def _cgyro_output(folder, suffix):
    c = CGYROoutput.__new__(CGYROoutput)
    c.folder, c.suffix_read, c.params1D = folder, suffix, {'q_gb_norm': 0.0}
    c.Qe_mean, c.Qe_std, c.Qi_mean, c.Qi_std, c.Ge_mean, c.Ge_std = 1.0, 0.1, 2.0, 0.2, 0.0, 0.01
    c.t, c.linear, c.cgyro_version = np.linspace(0, 100, 101), False, 'test'
    return c


def _assert_same(original_text, rebuilt_text, label):
    a, b = buildDictFromInput(original_text), buildDictFromInput(rebuilt_text)
    assert list(a) == list(b), f"{label}: key set/order differs\n  only original: {set(a) - set(b)}\n  only rebuilt: {set(b) - set(a)}"
    for k in a:
        assert type(a[k]) is type(b[k]), f"{label}: {k} type {type(a[k]).__name__} -> {type(b[k]).__name__}"
        assert a[k] == b[k], f"{label}: {k} value {a[k]!r} -> {b[k]!r}"
    return len(a)


def test_roundtrip(tmp):
    staging = tmp / 'run' / 'Outputs' / 'harvest'
    rec = H.harvest_recorder(H.options_from_namelist({'enabled': True}, staging, run_meta_extra={'run_folder': str(tmp / 'run')}))

    originals = {}

    # TGLF: parsed dict as SIMtools.read() builds it
    txt = (DATA / 'input.tglf').read_text()
    assert rec.record(_sim('tglf', [0.5], [buildDictFromInput(txt)], [_gacode_output(Qe=1.0, Qi=2.0, Ge=0.1)]), 'base') == 1
    originals[('tglf', 0)] = txt

    # NEO: 5 species, general Miller
    txt = (DATA / 'input.neo').read_text()
    assert rec.record(_sim('neo', [0.5], [buildDictFromInput(txt)], [_gacode_output(Qe=0.01, Qi=0.02, Ge=0.0)]), 'base') == 1
    originals[('neo', 0)] = txt

    # CGYRO: 4 and 7 species radii of the same run, the output class reads its own input.cgyro<suffix>
    folder = tmp / 'base_cgyro'
    folder.mkdir()
    for rho, name in ((0.3, 'input.cgyro_4species'), (0.8, 'input.cgyro_7species')):
        shutil.copy(DATA / name, folder / f"input.cgyro_{rho:.4f}")
    outs = [_cgyro_output(folder, f"_{rho:.4f}") for rho in (0.3, 0.8)]
    assert rec.record(_sim('cgyro', [0.3, 0.8], [None, None], outs), 'base') == 2
    originals[('cgyro', 0)] = (DATA / 'input.cgyro_4species').read_text()
    originals[('cgyro', 1)] = (DATA / 'input.cgyro_7species').read_text()
    assert len(buildDictFromInput(originals[('cgyro', 1)])) > len(buildDictFromInput(originals[('cgyro', 0)])), "different key sets"

    db = H.harvest_database(tmp / 'db' / 'central.nc')
    assert db.push([staging]) == {'tglf': 1, 'neo': 1, 'cgyro': 2}

    for code in ('tglf', 'neo', 'cgyro'):
        df = db.load(code)
        for i in range(len(df)):
            original = originals[(code, i)]
            # by hash and by row, and written to disk
            n = _assert_same(original, db.input_file(code, df['hash'].iloc[i]), f"{code}[{i}] by hash")
            _assert_same(original, db.input_file(code, df.iloc[i]), f"{code}[{i}] by row")
            path = db.write_input_file(code, df['hash'].iloc[i], tmp / f"rebuilt_{code}_{i}")
            _assert_same(original, path.read_text(), f"{code}[{i}] written")
            print(f"PASS {code}[{i}]: {n} keys reproduced (names, order, values, types)")

    # the 4-species CGYRO record carries NaN fills for the 7-species keys; none may appear in its file
    rebuilt4 = buildDictFromInput(db.input_file('cgyro', db.load('cgyro')['hash'].iloc[0]))
    assert 'Z_7' not in rebuilt4 and rebuilt4['N_SPECIES'] == 4 and 'Z_7' in buildDictFromInput(originals[('cgyro', 1)])

    # the type map lives once per (run, code) in the runs table; a record whose types deviate from the
    # run's first record (KY = 3 in the 7-species file, 8.00000E-02 in the 4-species one) carries only those
    runs = db.runs()
    assert set(runs['code']) == {'tglf', 'neo', 'cgyro'} and all(len(s) > 2 for s in runs['input_types'])
    assert list(db.load('cgyro', with_run_info=False)['input_types_record']) == ['', '{"KY": "int"}']

    # the same run pushed again with a new record (e.g. a manual push of a live run, then the final
    # one): its type map gains the new key in place, the old records still reproduce
    extra = buildDictFromInput(originals[('tglf', 0)])
    extra['NEW_KEY'] = 3
    extra['RLTS_1'] = 9   # an int where the first record had a float
    assert rec.record(_sim('tglf', [0.5], [extra], [_gacode_output(Qe=1.0, Qi=2.0, Ge=0.1)]), 'base') == 1
    assert db.push([staging]) == {'tglf': 1}
    df = db.load('tglf')
    assert len(db.runs()) == 3, "no duplicate runs row"
    _assert_same(originals[('tglf', 0)], db.input_file('tglf', df['hash'].iloc[0]), 'tglf[0] after second push')
    rebuilt = buildDictFromInput(db.input_file('tglf', df['hash'].iloc[1]))
    assert rebuilt['NEW_KEY'] == 3 and type(rebuilt['NEW_KEY']) is int and list(rebuilt)[-1] == 'NEW_KEY'
    assert rebuilt['RLTS_1'] == 9 and type(rebuilt['RLTS_1']) is int
    print("PASS second push of the same run: type map extended in place, per-record deviation kept, key order kept")

    # a pre-schema-5 CGYRO record (pygacode params1D, lowercase) in the same file must be refused, not rebuilt
    legacy = staging / 'cgyro.jsonl'
    legacy.write_text(json.dumps({'run': 'legacy', 'hash': 'legacyhash000000', 'in_q': 2.0, 'in_kappa': 1.5, 'out_Qe_mean': 1.0}) + '\n')
    assert db.push([staging]) == {'cgyro': 1}
    try:
        db.input_file('cgyro', 'legacyhash000000')
        raise AssertionError("legacy CGYRO record was rebuilt")
    except ValueError:
        pass
    print("PASS pre-schema-5 CGYRO record refused")


def main():
    tmp = Path(tempfile.mkdtemp(prefix='mitim_harvest_repro_'))
    try:
        test_roundtrip(tmp)
        print("\nALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
