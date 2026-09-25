"""
test_harvest.py
===============
Sanity tests for the harvest capability (mitim_tools.harvest_tools.HARVESTtools): the per-run
recorder (staging JSONL with rolling gzip compression, dedup by input hash, numpy/NaN
serialization, provenance written once per run and code), the polymorphic
harvest_records/harvest_outputs interface on stand-in simulation objects (TGLF/NEO layout,
CGYRO, GX, EPED), and the central netCDF database (schema-union append, runs table join,
selective loading, two-process concurrent push under the mkdir lock, stale-lock recovery,
rebuild, summary/interpret/plot), column type flips across pushes (all-or-nothing), per-process
staging files (two writers in one folder, legacy <code>.jsonl names), and the CGYRO run-history
fields (restart chain, completion, cost).

Everything runs in a temporary folder -- no transport code, no cluster.

Run as:

    python tests/dev_tests/test_harvest.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import os
import sys
import json
import gzip
import time
import shutil
import tempfile
import multiprocessing
from types import SimpleNamespace
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

import matplotlib
matplotlib.use("Agg")

from mitim_tools.harvest_tools import HARVESTtools as H
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.gacode_tools.utils.CGYROutils import CGYROoutput
from mitim_tools.simulation_tools.physics.GXtools import GXoutput


def _opts(folder, **block):
    return H.options_from_namelist({'enabled': True, **block}, folder, run_meta_extra={'run_folder': str(folder)})


def _fake_sim(code, rhos, inputs, outputs_attrs, sim_folder='base_x'):
    '''Stand-in for a mitim_simulation after read(): the base harvest_records() only touches these'''
    outs = []
    for attrs in outputs_attrs:
        o = SIMtools.GACODEoutput()
        for k, v in attrs.items():
            setattr(o, k, v)
        outs.append(o)
    ns = SimpleNamespace(run_specifications={'code': code}, rhos=list(rhos),
                         results={'base': {'parsed': list(inputs), 'output': outs, 'x': np.array(rhos)}},
                         simulation_job=SimpleNamespace(machineSettings={'machine': 'local', 'modules': 'gacode_setup'}),
                         in_process=False, FolderSimLast=Path(sim_folder), inputs_files={})
    ns.harvest_records = lambda label, folder=None: SIMtools.mitim_simulation.harvest_records(ns, label, folder=folder)
    return ns


def _lines(folder, code):
    '''Plain staged lines of `code` (this process writes <code>.<host>-<pid>.jsonl)'''
    return [json.loads(l) for f in sorted(Path(folder).glob(f"{code}.*.jsonl")) for l in f.read_text().splitlines() if l.strip()]


def _meta(folder):
    return json.loads((Path(folder) / 'run_meta.json').read_text())


# ------------------------------------------------------------------------------------------------

def test_options_and_run_meta(tmp):
    folder = tmp / 'run1' / 'Outputs' / 'harvest'
    o = _opts(folder)
    assert o['enabled'] and (folder / 'run_meta.json').exists()
    rid = o['run_meta']['run']
    assert len(rid) == 12 and o['run_meta']['maestro_beat'] == -1 and o['run_meta']['mitim_version'] and o['run_meta']['codes'] == {}
    o2 = _opts(folder, maestro_beat=3)
    assert o2['run_meta']['run'] == rid, "resumed run must keep its id"
    assert o2['run_meta']['maestro_beat'] == 3
    o3 = H.options_from_namelist({'enabled': False}, tmp / 'never')
    assert not o3['enabled'] and not (tmp / 'never').exists()
    o4 = _opts(tmp / 'run2' / 'harvest', run_id='abc', push=False, scan_trick_members=False)
    assert o4['run_meta']['run'] == 'abc' and o4['push'] is False and o4['scan_trick_members'] is False
    print("PASS options_from_namelist / run_meta.json")

def test_drop_and_type_fallback(tmp):
    import netCDF4
    from mitim_tools.simulation_tools import SIMtools
    file = tmp / 'dbD' / 'central.nc'
    hashes = {}
    for name, n_val in (('A', 3), ('B', 3), ('C', 3.5)):   # run C disagrees on the type of N
        folder = tmp / 'dbD' / name / 'Outputs' / 'harvest'
        rec = H.harvest_recorder(_opts(folder))
        for i in range(3):
            rec._write({'code': 'tglf', 'inputs': {'USE_X': True, 'N': n_val, 'RLTS_1': 1.0 + i}, 'outputs': {'Qe': float(i)}, 'meta': {}})
        rec._write({'code': 'neo', 'inputs': {'RMIN_OVER_A': 0.5}, 'outputs': {'Qe': 1.0}, 'meta': {}})
        H.harvest_database(file).push([folder])
    db = H.harvest_database(file)
    df = db.load('tglf')
    run_a = _meta(tmp / 'dbD' / 'A' / 'Outputs' / 'harvest')['run']

    # a run pushed before the per-run type maps existed: its keys take the types all other runs agree on
    with netCDF4.Dataset(file, 'a') as ds:
        grp = ds.groups[H.RUNS_GROUP]
        for i, (r, c) in enumerate(zip(grp.variables['run'][:], grp.variables['code'][:])):
            if r == run_a and c == 'tglf':
                grp.variables['input_types'][i] = ''
    db = H.harvest_database(file)
    row = df[df['run'] == run_a].iloc[0]
    text = db.input_file('tglf', row)
    assert 'USE_X' in text and 'True' in text, "bool type borrowed from the other runs"
    assert 'N                       = 3.0' in text, "N is int in A/B but float in C: not borrowed, stays a float"
    assert db._code_types('tglf').get('USE_X') == 'bool' and 'N' not in db._code_types('tglf')
    assert H._format_input_value(9.000000026, 'int') == '9.000000026', "a non-integer never becomes an int"

    # drop: only the listed records go, every other group and run is kept, the previous file is kept aside
    gone = df[df['run'] == run_a]['hash'].iloc[:2].tolist()
    assert db.drop('tglf', gone, run=run_a) == 2
    df2 = db.load('tglf')
    assert len(df2) == len(df) - 2 and not set(gone) & set(df2[df2['run'] == run_a]['hash'])
    assert len(db.load('neo')) == 3 and len(db.runs()) == len(H.harvest_database(next(file.parent.glob('central.nc.before-drop-*'))).runs())
    kept = df2.iloc[0]
    assert H.input_hash('tglf', H._scalar_dict(SIMtools.buildDictFromInput(db.input_file('tglf', kept))), None) == kept['hash'] or kept['run'] == run_a
    print("PASS drop() rewrites the file without the listed records; runs without a type map borrow the unambiguous types")

def test_eped_input_files(tmp):
    import f90nml
    src = tmp / 'eped_src'
    src.mkdir()
    f90nml.write(f90nml.Namelist({'eped_input': {'ip': 14.00002, 'bt': 8.5085839, 'r': 4.595, 'a': 1.28, 'kappa': 2.159294983157618,
                                                 'delta': 0.627, 'neped': 21.7, 'betan': 2.0, 'zeffped': 1.75, 'nesep': 8.68,
                                                 'tesep': 200.0, 'shot': 0, 'num_scan': 1, 'teped': -1, 'zi': 9.000000026, 'mi': 18}}),
                 src / 'eped.input.1', force=True)
    (src / 'eped.config').write_text("NMODES = 5 6 8 10 15 20 30\nWIDTHS = 3 4 5 7 9\nTEPED_BOUND = 0.4 1.4 0.01\nNOT_ASKED = 1\n")
    r = H.collect_eped(None, toq_eq_choice='standard', dataset={'stability_rule': 'G', 'stability_threshold': 0.03},
                       eped_input_file=src / 'eped.input.1', eped_config_file=src / 'eped.config')
    folder = tmp / 'eped_run' / 'Outputs' / 'harvest'
    H.harvest_recorder(_opts(folder))._write(r)
    file = tmp / 'eped_db.nc'
    H.harvest_database(file).push([folder])
    db = H.harvest_database(file)
    row = db.load('eped').iloc[0]
    out = tmp / 'eped_rebuilt'
    out.mkdir()
    db.write_input_file('eped', row, out)
    again = H.collect_eped(None, toq_eq_choice=row['in_toq_eq_choice'], dataset={'stability_rule': 'G', 'stability_threshold': 0.03},
                           eped_input_file=out / 'eped.input', eped_config_file=out / 'eped.config')
    assert H.input_hash('eped', H._scalar_dict(again['inputs']), None) == row['hash'], "eped.input + eped.config rebuilt exactly"
    assert 'NOT_ASKED' not in (out / 'eped.config').read_text() and 'toq_eq_choice' not in (out / 'eped.input').read_text()
    print("PASS eped.input / eped.config rebuilt from an EPED record (same record hash)")

def test_qualikiz_records(tmp):
    '''QuaLiKiz: inputs are the dimx coordinates (plan), GB fluxes normalized like transport_qualikiz, read_cases members recorded'''
    import xarray as xr
    from mitim_tools.qualikiz_tools import QLKtools
    from mitim_tools.gacode_tools import PROFILEStools
    from mitim_tools.misc_tools import PLASMAtools
    rhos = [0.4, 0.6]
    def dataset(n):
        d = {'efe_SI': ('dimx', np.linspace(1e4, 2e4, n)), 'pfe_SI': ('dimx', np.linspace(1e18, 2e18, n)),
             'efi_SI': (('dimx', 'nions'), np.ones((n, 2)) * 5e3), 'pfi_SI': (('dimx', 'nions'), np.ones((n, 2)) * 1e17),
             'vfi_SI': (('dimx', 'nions'), np.ones((n, 2))), 'cke': ('dimx', np.zeros(n))}
        c = {'x': ('dimx', np.tile([0.45, 0.65], n // 2)), 'Ate': ('dimx', np.linspace(5, 6, n)), 'Ane': ('dimx', np.ones(n)),
             'Ati': (('dimx', 'nions'), np.ones((n, 2)) * 6), 'Te': ('dimx', np.ones(n) * 5), 'phi': (('ntheta', 'dimx'), np.zeros((4, n))),
             'coll_flag': 1.0}
        return xr.Dataset(d, coords=c)
    q = QLKtools.QuaLiKiz(rhos=rhos)
    q.profiles = PROFILEStools.gacode_state(Path(__file__).resolve().parents[1] / 'data' / 'input.gacode')
    ds = dataset(2).assign_coords(rho=('dimx', rhos))
    q.results['base'] = {'dataset': ds, 'x': np.array(rhos), 'output': [ds.isel(dimx=i) for i in range(2)]}
    recs = q.harvest_records('base')
    r0 = recs[0]
    assert {'x', 'Ate', 'Ati_0', 'Ati_1', 'Te', 'coll_flag', 'rho'} <= set(r0['inputs']) and 'cke' not in r0['inputs'] and 'phi' not in r0['inputs']
    assert r0['meta']['roa'] == 0.45 and {'efe_SI', 'efi_SI_0', 'Qe', 'Qi', 'Ge', 'Gi_1', 'Mt'} <= set(r0['outputs'])
    p = q.profiles
    Qgb, Ggb, Pgb, _, _ = PLASMAtools.gyrobohmUnits(np.interp(0.4, p.profiles['rho(-)'], p.profiles['te(keV)']),
                                                    np.interp(0.4, p.profiles['rho(-)'], p.profiles['ne(10^19/m^3)']) * 0.1,
                                                    PLASMAtools.md_u, np.interp(0.4, p.profiles['rho(-)'], p.derived['B_unit']), p.derived['a'])
    assert np.isclose(r0['outputs']['Qe'], 1e4 / (Qgb * 1e6)) and np.isclose(r0['outputs']['Qi'], 1e4 / (Qgb * 1e6))
    assert np.isclose(r0['outputs']['Ge'], 1e18 / (Ggb * 1e20)) and np.isclose(r0['outputs']['Mt'], 2 / Pgb)

    # the stacked scan trick (read_cases layout [case][rho]) is recorded too, one record per dimx point
    ds4 = dataset(4).assign_coords(rho=('dimx', rhos * 2), case=('dimx', [0, 0, 1, 1]))
    q.results['scan'] = {'dataset': ds4, 'x': np.array(rhos), 'output': [[ds4.isel(dimx=2 * c + i) for i in range(2)] for c in range(2)]}
    scan = q.harvest_records('scan')
    assert len(scan) == 4 and len({json.dumps(r['inputs'], sort_keys=True) for r in scan}) == 4 and 'case' not in scan[0]['inputs']
    db = H.harvest_database(tmp / 'qlk.nc')
    assert db.drive_label('qualikiz', 'Te') == 'R0/LTe' and db.drive_label('tglf', 'Te') == 'a/LTe'
    print("PASS QuaLiKiz records: plan inputs (coords), GB fluxes as PORTALS normalizes them, stacked scan members")

def test_mixed_cgyro_schemas(tmp):
    import pandas as pd
    df = pd.DataFrame({'in_N_SPECIES': [3.0, 3.0, np.nan], 'in_RMIN': [0.4, 0.55, np.nan], 'in_rmin': [np.nan, np.nan, 0.4],
                       'out_Qi_mean': [1.0, 2.0, 3.0]})
    kept = H.harvest_database._single_schema('cgyro', df)
    assert list(kept['in_RMIN']) == [0.4, 0.55] and 'in_rmin' not in kept, "schema-5 rows only, legacy-only columns dropped"
    assert len(H.harvest_database._single_schema('cgyro', df.iloc[2:])) == 1, "legacy-only frame kept"
    assert len(H.harvest_database._single_schema('tglf', df)) == 3
    print("PASS mixed CGYRO schemas: analyses keep the schema-5 records")

def test_no_default_file(tmp):
    from mitim_tools.misc_tools import CONFIGread
    configured = CONFIGread.read_harvest_file
    CONFIGread.read_harvest_file = lambda: None   # as a config_user.json without preferences.harvest_file
    try:
        assert H.checked_block({'enabled': True})['enabled'] is False, "no namelist file, no config file: harvest off"
        assert H.checked_block({'enabled': True, 'file': str(tmp / 'x.nc')})['enabled'] is True
        assert H.checked_block({'enabled': True, 'push': False})['enabled'] is True, "MAESTRO beats: the driver pushes"
        try:
            H.harvest_database(None)
            raise AssertionError("harvest_database without any file must raise")
        except ValueError:
            pass
        CONFIGread.read_harvest_file = lambda: str(tmp / 'from_config.nc')
        assert H.checked_block({'enabled': True})['enabled'] is True
        assert H.harvest_database(None).file.name == 'from_config.nc'
    finally:
        CONFIGread.read_harvest_file = configured
    print("PASS no default harvest file: namelist -> config -> harvest off with a warning")


def test_recorder_tglf_layout_and_dedup(tmp):
    folder = tmp / 'p1' / 'Outputs' / 'harvest'
    rec = H.harvest_recorder(_opts(folder)).with_context(evaluation=7)
    sim = _fake_sim('tglf', [0.3, 0.6],
                    [{'RLTS_1': 2.0, 'RLTS_2': np.float64(3.0), 'USE_BPER': np.bool_(True), 'NS': 3},
                     {'RLTS_1': 2.5, 'RLTS_2': 3.5, 'USE_BPER': False, 'NS': 3}],
                    [{'Qe': 1.0, 'Qi': np.float64(2.0), 'Ge': 0.1, 'Mt': 0.0, 'Se': np.nan, 'GiAll': np.array([0.2, 0.3]), 'roa': 0.31, 'tglf_version': 'abc123 [2025-10-02]\nPIXI_OPENMP\nTue'},
                     {'Qe': 1.5, 'Qi': 2.5, 'Ge': 0.2, 'Mt': 0.1, 'Se': 0.01, 'GiAll': np.array([0.4, 0.5]), 'roa': 0.61, 'tglf_version': ''}],
                    sim_folder='base_tglf')
    n = rec.record(sim, 'base', folder=Path('/some/where/base_tglf'))
    assert n == 2, n
    rows = _lines(folder, 'tglf')
    r = rows[0]
    # a record is inputs -> outputs plus two short keys, nothing else
    assert set(k for k in r if not k.startswith(('in_', 'out_'))) == {'run', 'hash'}, r.keys()
    assert r['run'] == rec.options['run_meta']['run'] and len(r['hash']) == 16
    assert r['in_RLTS_2'] == 3.0 and r['in_USE_BPER'] is True and r['in_NS'] == 3
    assert r['out_Qi'] == 2.0 and r['out_Gi_1'] == 0.2 and r['out_Gi_2'] == 0.3 and (r['out_Se'] is None or np.isnan(r['out_Se']))
    # provenance is written once per (run, code) into run_meta.json, from the first record
    prov = _meta(folder)['codes']['tglf']
    assert prov['code_version'].startswith('abc123') and prov['machine'] == 'local' and prov['modules'] == 'gacode_setup' and prov['in_process'] == 0

    # same inputs again (e.g. batched base re-read) -> nothing new
    assert rec.record(sim, 'base') == 0
    # fresh process view (registry cleared) must rescan the file and still dedup
    H._SEEN.clear()
    assert H.harvest_recorder(_opts(folder)).record(sim, 'base') == 0
    # a scan-trick member with a perturbed input is a NEW record
    sim.results['turb_drives_RLTS_1_1.02'] = {**sim.results['base'], 'parsed': [{'RLTS_1': 2.0 * 1.02, 'RLTS_2': 3.0, 'USE_BPER': True, 'NS': 3}, sim.results['base']['parsed'][1]]}
    assert rec.record(sim, 'turb_drives_RLTS_1_1.02') == 1, "the perturbed radius is new; the untouched radius is a duplicate"
    assert len(_lines(folder, 'tglf')) == 3
    # recorder pickles/deep-copies to plain dicts
    import pickle, copy
    rec2 = pickle.loads(pickle.dumps(rec))
    assert rec2.options == rec.options and rec2.context == rec.context and copy.deepcopy(rec).enabled

    # scan-trick members: tagged by the backend via context (not stored), skippable by the namelist knob
    sim.results['turb_drives_RLTS_2_1.02'] = {**sim.results['base'], 'parsed': [{'RLTS_1': 2.0, 'RLTS_2': 3.06, 'USE_BPER': True, 'NS': 3}, sim.results['base']['parsed'][1]]}
    assert rec.with_context(scan_member=1).record(sim, 'turb_drives_RLTS_2_1.02') == 1
    folder2 = tmp / 'p2' / 'Outputs' / 'harvest'
    rec_noscan = H.harvest_recorder(_opts(folder2, scan_trick_members=False))
    assert rec_noscan.with_context(scan_member=1).record(sim, 'turb_drives_RLTS_2_1.02') == 0, "knob off: scan members skipped"
    assert rec_noscan.record(sim, 'base') == 2, "knob off: base points still recorded"
    print("PASS recorder: lean records, provenance once per run/code, numpy scalars, dedup in-memory + on disk, pickle, scan knob")


def test_shared_staging_folder(tmp):
    '''A driver (MAESTRO) hands its own staging folder to the runs it chains: one folder, one run_meta.json, maestro_beat per record'''
    driver = tmp / 'chain' / 'Outputs' / 'harvest'
    o_driver = H.options_from_namelist({'enabled': True}, driver, run_meta_extra={'run_folder': str(tmp / 'chain')})
    own = tmp / 'chain' / 'Beats' / 'Beat_4' / 'run_portals' / 'Outputs' / 'harvest'
    o_beat = H.options_from_namelist({'enabled': True, 'push': False, 'run_id': o_driver['run_meta']['run'], 'maestro_beat': 4,
                                      'staging_folder': str(driver)}, own, run_meta_extra={'run_folder': str(own.parents[1])})
    assert o_beat['folder'] == str(driver) and not own.exists(), "the beat stages in the driver folder, its own is never created"
    assert _meta(driver)['maestro_beat'] == -1 and _meta(driver)['run_folder'] == str(tmp / 'chain'), "the shared run_meta.json stays the driver's"
    sim = _fake_sim('tglf', [0.5], [{'RLTS_1': 1.0}], [{'Qe': 1.0, 'Qi': 2.0, 'Ge': 0.0, 'Mt': 0.0, 'Se': 0.0, 'roa': 0.5, 'tglf_version': 'v'}])
    assert H.harvest_recorder(o_beat).record(sim, 'base') == 1
    o_beat7 = H.options_from_namelist({'enabled': True, 'run_id': o_driver['run_meta']['run'], 'maestro_beat': 7, 'staging_folder': str(driver)}, own)
    assert H.harvest_recorder(o_beat7).record(sim, 'base') == 0, "same inputs in a later beat are deduplicated across the chain"
    sim2 = _fake_sim('tglf', [0.5], [{'RLTS_1': 2.0}], [{'Qe': 3.0, 'Qi': 4.0, 'Ge': 0.0, 'Mt': 0.0, 'Se': 0.0, 'roa': 0.5, 'tglf_version': 'v'}])
    assert H.harvest_recorder(o_beat7).record(sim2, 'base') == 1
    rows = _lines(driver, 'tglf')
    assert [r['maestro_beat'] for r in rows] == [4, 7] and len({r['run'] for r in rows}) == 1
    db = H.harvest_database(tmp / 'chain.nc')
    assert db.push([driver]) == {'tglf': 2}
    df = db.load('tglf')
    assert sorted(df['maestro_beat'].astype(int)) == [4, 7] and 'maestro_beat_x' not in df.columns, "per-record beat wins over the runs table"
    # a standalone run (no staging_folder) keeps no per-record beat column
    solo = tmp / 'solo' / 'Outputs' / 'harvest'
    assert H.harvest_recorder(_opts(solo)).record(sim, 'base') == 1 and 'maestro_beat' not in _lines(solo, 'tglf')[0]
    print("PASS shared staging folder (driver-owned run_meta, per-record maestro_beat, cross-beat dedup)")


def test_cgyro_gx_eped_interfaces(tmp):
    c = CGYROoutput.__new__(CGYROoutput)
    c.params1D = {'n_species': 3, 'dlntdr_0': 2.0, 'nonlinear_flag': True, 'q_gb_norm': 0.5}
    c.Qe_mean, c.Qe_std, c.Qi_mean, c.Qi_std, c.Ge_mean, c.Ge_std, c.Mt_mean, c.Mt_std = 1., .1, 2., .2, .3, .03, 0., 0.
    c.Gi_all_mean, c.Gi_all_std = np.array([0.1, 0.2]), np.array([0.01, 0.02])
    c.t, c.tmin, c.linear = np.linspace(0, 500, 11), 250.0, False
    c.cgyro_version = '26-Jun-11 [0e6c00ed3 [2025-10-02]][PIXI_OPENMP][0.0]'
    # the averager as GKaveraging builds it (fixed window, ACF standard error), on a real trace for Qe
    from mitim_tools.simulation_tools.utils.GKaveraging import GKaverager
    rng = np.random.default_rng(1)
    c.t = np.linspace(0, 500, 501)
    c.Qe = 1.0 + np.convolve(rng.normal(size=len(c.t)), np.ones(10) / 10, mode='same')
    c.averaging = GKaverager(c.t, {'Qi': c.Qe * 2, 'Qe': c.Qe, 'Ge': c.Qe * 0.1}, method='fixed', tmin=250.0, tmin_is_rel=False)
    o = c.harvest_outputs()
    assert o['Qi_mean'] == 2. and o['Gi_2_std'] == 0.02 and 'Se_mean' not in o and o['t_last'] == 500. and np.isnan(o['tmax_fluct']) and o['linear'] == 0
    # averaging: window from the averager, per-flux ACF diagnostics recomputed on that window, method once per run/code
    assert o['avg_tmin'] == 250. and o['avg_tmax'] == 500. and o['avg_npoints'] == 251 and o['avg_dt'] == 1. and o['avg_flag'] == 'fixed'
    assert 1 < o['Qe_icor'] < 30 and o['Qe_ncorr'] < 251 / 3 and 'Qi_ncorr' not in o, "diagnostics only for fluxes whose trace is on the object"
    assert abs(c.averaging.stats['Qe']['std'] - c.Qe[250:].std() / np.sqrt(o['Qe_ncorr'])) < 1e-9, "std is the sample std / sqrt(ncorr) of the same window"
    prov = json.loads(c.harvest_provenance()['averaging'])
    assert prov['method'] == 'fixed' and prov['uncertainty'] == 'acf' and prov['tmin'] == 250.0 and 'sqrt(<flux>_ncorr)' in prov['std']
    assert o['derived_q_gb_norm'] == 0.5 and 'derived_n_species' not in o, "CGYRO-derived quantities (not inputs) travel as out_derived_*"
    assert c.harvest_inputs() is None, "no folder -> the simulation object falls back to its own parsed inputs"
    (tmp / 'cg').mkdir()
    (tmp / 'cg' / 'input.cgyro_0.5000').write_text("NONLINEAR_FLAG=1\nN_SPECIES=3\nDELTA_T=0.04\n")
    c.folder, c.suffix_read = tmp / 'cg', '_0.5000'
    assert c.harvest_inputs() == {'NONLINEAR_FLAG': 1, 'N_SPECIES': 3, 'DELTA_T': 0.04} and c.harvest_hash_extra() == {'t_last': 500.0}
    assert c.harvest_version().startswith('26-Jun-11')
    # no averager (e.g. a partially read object) -> window fields NaN, provenance empty, no crash
    c2 = CGYROoutput.__new__(CGYROoutput)
    c2.t, c2.params1D = c.t, {}
    assert np.isnan(c2.harvest_outputs()['avg_tmin']) and c2.harvest_provenance()['averaging'] == ''

    g = GXoutput.__new__(GXoutput)
    g.inputclass = SimpleNamespace(controls={'nstep': 1000}, plasma={'tprim_1': 3.0})
    g.Qe_mean, g.Qe_std, g.Qi_mean, g.Qi_std, g.Ge_mean, g.Ge_std = 1., .1, 2., .2, .3, .03
    g.Qi_all_mean, g.Qi_all_std = np.array([1.5, 0.5]), np.array([.1, .1])
    g.t, g.tmin, g.gx_version = np.linspace(0, 100, 5), 50., 'git_hash=deadbeef'
    g.averaging = SimpleNamespace(t_start=50., t_end=100., n_window=3, flag='fixed', method='fixed', uncertainty='acf', provenance={'tmin': 50.}, diagnostics={})
    o = g.harvest_outputs()
    assert o['Qi_2_mean'] == 0.5 and o['t_last'] == 100. and g.harvest_inputs() == {'nstep': 1000, 'tprim_1': 3.0} and g.harvest_version() == 'git_hash=deadbeef'
    assert o['avg_tmin'] == 50. and o['avg_tmax'] == 100. and o['avg_npoints'] == 3 and o['avg_flag'] == 'fixed' and '"method": "fixed"' in g.harvest_provenance()['averaging']

    import xarray as xr
    import f90nml
    ds = xr.Dataset({'ptop': 12.3, 'wptop': 0.05, 'tped': 1.1, 'ttop': 1.3, 'pped': 10.0, 'wpped': 0.04, 'wrped': 0.03,
                     'stability_index': 4, 'n_limiting': 12, 'dome_frac': 0.2, 'stability_rule': (('dim_one',), ['W']),
                     'stability_threshold': (('dim_one',), [1.0]), 'gamma': (('dim_height', 'dim_nmodes'), np.zeros((3, 2)))},
                    attrs={'eped_version': '1.0'})
    # the files as EPEDtools writes them per case: the f90 namelist and the effective config (retry lowered TEPED_BOUND)
    case = tmp / 'eped_case'
    case.mkdir()
    f90nml.write(f90nml.Namelist({'eped_input': {'ip': 8.7, 'bt': 12.2, 'neped': 30.0, 'm': 2.5, 'z': 1, 'mi': 20, 'zi': 10,
                                                  'teped': 1.2, 'tewid': 0.03, 'num_scan': 1}}), case / 'eped.input.1')
    (case / 'eped.config1').write_text("# cfg\nTAG = template\n[EPED_DRIVER]\n    NMODES = 5 6 8 10 15 20 30\n    WIDTHS = 3 4 5 7 9\n"
                                       "    TEPED_BOUND = 0.28 1.4 0.01\n    CLEAN_AFTER = 1\n[OTHER]\n    CLEAN_AFTER = 0\n")
    rec = H.collect_eped(input_params={'ip': -1}, composition={'m': -1},
                         eped_params_override={'TEPED_BOUND': [0.28, 1.4, 0.01], 'CLEAN_AFTER': 1}, toq_eq_choice='mxh', dataset=ds,
                         ptop_kPa=12.3, wtop_psipol=0.05, limiting_mode_info={'limiting_mode': 'peeling', 'n_limiting': 12, 'dome_frac': 0.2},
                         eped_folder=Path('/x/run_eped'), job=SimpleNamespace(machineSettings={'machine': 'engaging', 'modules': 'ips'}),
                         eped_input_file=case / 'eped.input.1', eped_config_file=case / 'eped.config1')
    folder = tmp / 'm1' / 'Outputs' / 'harvest'
    r = H.harvest_recorder(_opts(folder, maestro_beat=2))
    assert r._write(rec) == 1
    row = _lines(folder, 'eped')[0]
    assert row['in_ip'] == 8.7 and row['in_mi'] == 20 and row['in_teped'] == 1.2 and row['in_tewid'] == 0.03, "the namelist as run wins over the dict"
    assert row['maestro_beat'] == 2, "the beat travels with the record"
    assert H.harvest_recorder(_opts(tmp / 'eped_ctx')).with_context(maestro_beat=5)._write(rec) == 1 and _lines(tmp / 'eped_ctx', 'eped')[0]['maestro_beat'] == 5, "EPED beats pass it as context"
    assert row['in_toq_eq_choice'] == 'mxh' and row['in_cfg_NMODES_0'] == 5 and row['in_cfg_NMODES_6'] == 30 and row['in_cfg_WIDTHS_4'] == 9
    assert row['in_cfg_TEPED_BOUND_0'] == 0.28 and row['in_cfg_TEPED_BOUND_1'] == 1.4 and row['in_cfg_CLEAN_AFTER'] == 1, "effective config, first occurrence"
    assert row['in_stability_rule'] == 'W' and row['in_stability_threshold'] == 1.0
    assert row['out_ptop'] == 12.3 and row['out_n_limiting'] == 12 and row['out_limiting_mode'] == 'peeling' and 'out_stability_threshold' not in row
    # without the files, the dict fallback still gives a record
    rec2 = H.collect_eped(input_params={'ip': 8.7}, composition={'m': 2.5}, eped_params_override={'TEPED_BOUND': [0.4, 1.4, 0.01]}, dataset=ds)
    assert rec2['inputs']['ip'] == 8.7 and rec2['inputs']['cfg_TEPED_BOUND'] == [0.4, 1.4, 0.01] and 'cfg_NMODES' not in rec2['inputs']
    prov = _meta(folder)['codes']['eped']
    assert prov['code_version'] == 'eped_version=1.0' and prov['machine'] == 'engaging' and prov['averaging'] == '' and _meta(folder)['maestro_beat'] == 2
    # the averaging description travels with the provenance of averaged codes
    r._write({'code': 'cgyro', 'inputs': {'ky': 0.3}, 'outputs': c.harvest_outputs(), 'meta': {'machine': 'engaging', **c.harvest_provenance()}})
    assert '"method": "fixed"' in _meta(folder)['codes']['cgyro']['averaging']
    print("PASS CGYRO / GX output interfaces and EPED collector")


def test_push_schema_union_and_load(tmp):
    file = tmp / 'db' / 'central.nc'
    fA = tmp / 'runA' / 'Outputs' / 'harvest'
    fB = tmp / 'runB' / 'Outputs' / 'harvest'
    rA = H.harvest_recorder(_opts(fA))
    rB = H.harvest_recorder(_opts(fB))
    assert rA._write({'code': 'tglf', 'inputs': {'A': 1.0, 'B': 2.0}, 'outputs': {'Qe': 1.0}, 'meta': {'machine': 'local', 'modules': 'gacode_setup'}}) == 1
    assert rA._write({'code': 'tglf', 'inputs': {'A': 1.5, 'B': 2.0}, 'outputs': {'Qe': np.nan}, 'meta': {'machine': 'ignored (first record wins)'}}) == 1
    assert rB._write({'code': 'tglf', 'inputs': {'B': 3.0, 'C': 4.0}, 'outputs': {'Qe': 3.0}, 'meta': {'code_version': 'v2', 'machine': 'engaging'}}) == 1
    assert rB._write({'code': 'neo', 'inputs': {'N_XI': 25}, 'outputs': {'Qi': 0.5}, 'meta': {}}) == 1

    db = H.harvest_database(file)
    n = db.push([fA])
    assert n == {'tglf': 2}, n
    pushed = list(fA.glob('tglf.*jsonl.pushed-*'))
    assert not list(fA.glob('tglf.*.jsonl')) and len(pushed) == 1 and pushed[0].name.endswith('.gz'), "pushed file is gzipped in place"
    assert db.push([fA]) == {}, "nothing left to push"
    H._SEEN.clear()
    assert H.harvest_recorder(_opts(fA))._write({'code': 'tglf', 'inputs': {'A': 1.0, 'B': 2.0}, 'outputs': {'Qe': 1.0}, 'meta': {}}) == 0, "dedup against the gzipped archive"
    n = db.push([fB])
    assert n == {'tglf': 1, 'neo': 1}, n
    assert not (file.parent / 'central.nc.lock').exists()

    assert sorted(db.codes()) == ['neo', 'tglf']
    df = db.load('tglf')
    assert len(df) == 3
    assert list(df['in_A'])[:2] == [1.0, 1.5] and np.isnan(df['in_A'][2]), "column missing for the later record reads NaN"
    assert np.isnan(df['in_C'][0]) and df['in_C'][2] == 4.0, "column introduced by a later push is NaN for earlier records"
    assert np.isnan(df['out_Qe'][1]) and df['out_Qe'][2] == 3.0
    # provenance lives once per (run, code) in the runs group and is joined back on load
    assert list(df['code_version']) == ['', '', 'v2'] and list(df['machine']) == ['local', 'local', 'engaging'] and list(df['modules']) == ['gacode_setup', 'gacode_setup', '']
    assert df['run'].nunique() == 2 and df['mitim_version'].iloc[0] != '' and df['run_folder'].iloc[0] == str(fA)
    runs = db.runs()
    assert set(zip(runs['run'], runs['code'])) == {(rA.options['run_meta']['run'], 'tglf'), (rB.options['run_meta']['run'], 'tglf'), (rB.options['run_meta']['run'], 'neo')}
    raw = db.load('tglf', with_run_info=False)
    assert set(c for c in raw.columns if not c.startswith(('in_', 'out_'))) == {'run', 'hash'}, "records carry no provenance strings"
    sub = db.load('tglf', columns=['in_A'], run=rA.options['run_meta']['run'])
    assert len(sub) == 2 and 'in_A' in sub and 'in_B' not in sub and 'run' in sub
    dn = db.load('neo')
    assert len(dn) == 1 and dn['in_N_XI'][0] == 25.0 and dn['out_Qi'][0] == 0.5
    assert db.load('cgyro').empty

    import xarray as xr
    with xr.open_dataset(file, group='tglf') as ds:
        assert ds.sizes['record'] == 3 and float(ds['in_B'][2]) == 3.0
    print("PASS push: schema union, gzip archive + dedup against it, runs group join, selective load, xarray")


def test_rolling_compression(tmp):
    '''Plain JSON never accumulates: the tail is folded into gzip members of <code>.jsonl.gz as the run goes'''
    folder = tmp / 'roll' / 'Outputs' / 'harvest'
    rec = H.harvest_recorder(_opts(folder))
    saved = H.ROLL_BYTES
    H.ROLL_BYTES = 3000
    try:
        for i in range(55):
            rec._write({'code': 'tglf', 'inputs': {f'K{j}': float(j) for j in range(20)} | {'RLTS_1': float(i)}, 'outputs': {'Qe': float(i)}, 'meta': {}})
    finally:
        H.ROLL_BYTES = saved
    stem = H._writer_stem('tglf')
    gz, plain = folder / f'{stem}.jsonl.gz', folder / f'{stem}.jsonl'
    assert gz.exists() and plain.stat().st_size < 3000, "tail stays below the threshold"
    with gzip.open(gz, 'rb') as fi:
        uncompressed = len(fi.read())
    assert gz.stat().st_size < 0.5 * uncompressed, f"rolled members are compressed ({gz.stat().st_size} vs {uncompressed} bytes)"
    n_gz, n_plain = len(H._read_jsonl(gz)), len(H._read_jsonl(plain))
    assert n_gz + n_plain == 55 and n_gz >= 40, (n_gz, n_plain)
    with open(gz, 'rb') as fi:
        assert fi.read().count(b'\x1f\x8b') >= 4, "several gzip members appended over time"
    H._SEEN.clear()
    assert H.harvest_recorder(_opts(folder))._write({'code': 'tglf', 'inputs': {f'K{j}': float(j) for j in range(20)} | {'RLTS_1': 3.0}, 'outputs': {'Qe': 3.0}, 'meta': {}}) == 0
    data = gz.read_bytes()
    trunc = folder / 'trunc.jsonl.gz'
    trunc.write_bytes(data[:-200])
    n_trunc = len(H._read_jsonl(trunc))
    assert 0 < n_trunc < n_gz, (n_trunc, n_gz)
    trunc.unlink()
    file = tmp / 'db5' / 'central.nc'
    db = H.harvest_database(file)
    assert db.push([folder]) == {'tglf': 55}
    assert not gz.exists() and not plain.exists() and len(list(folder.glob('tglf.*jsonl.pushed-*.gz'))) == 1
    assert len(db.load('tglf')) == 55 and db.push([folder]) == {}
    print("PASS rolling gzip staging: bounded plain tail, multi-member archive, dedup, truncated member, push")


def _worker(args):
    folder, file, n, tag = args
    sys.path.insert(0, str(mitim_root))
    from mitim_tools.harvest_tools import HARVESTtools as H2
    rec = H2.harvest_recorder(H2.options_from_namelist({'enabled': True}, folder))
    for i in range(n):
        rec._write({'code': 'tglf', 'inputs': {'RLTS_1': float(i), 'tag': tag}, 'outputs': {'Qe': float(i)}, 'meta': {}})
    return H2.harvest_database(file).push([folder], timeout_s=120)


def test_concurrent_push(tmp):
    file = tmp / 'db2' / 'central.nc'
    args = [(tmp / f'conc{k}' / 'harvest', file, 200, f'p{k}') for k in range(2)]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(2) as pool:
        res = pool.map(_worker, args)
    assert all(r == {'tglf': 200} for r in res), res
    df = H.harvest_database(file).load('tglf')
    assert len(df) == 400 and df['run'].nunique() == 2
    assert not (file.parent / 'central.nc.lock').exists()
    print("PASS concurrent push from two processes under the mkdir lock (400 records, no lock left)")


def test_stale_and_busy_lock(tmp):
    file = tmp / 'db3' / 'central.nc'
    file.parent.mkdir(parents=True)
    folder = tmp / 'runL' / 'harvest'
    H.harvest_recorder(_opts(folder))._write({'code': 'tglf', 'inputs': {'A': 1.0}, 'outputs': {'Qe': 1.0}, 'meta': {}})
    lock = file.parent / 'central.nc.lock'
    lock.mkdir()
    old = time.time() - 2 * 3600
    os.utime(lock, (old, old))
    assert H.harvest_database(file).push([folder], stale_s=3600) == {'tglf': 1}, "a 2 h old lock must be broken"
    assert not lock.exists()

    H.harvest_recorder(_opts(folder))._write({'code': 'tglf', 'inputs': {'A': 2.0}, 'outputs': {'Qe': 2.0}, 'meta': {}})
    lock.mkdir()
    t0 = time.time()
    try:
        H.harvest_database(file).push([folder], timeout_s=2, stale_s=3600)
        raise AssertionError("push must time out on a fresh lock")
    except TimeoutError:
        pass
    assert time.time() - t0 >= 2 and (folder / f"{H._writer_stem('tglf')}.jsonl.gz").exists() and len(list(folder.glob('*.pushed-*'))) == 1 and not list(folder.glob('*.claim-*')), "staging must be left intact (rolled, unpushed) after a failed push"
    lock.rmdir()
    assert H.harvest_database(file).push([folder]) == {'tglf': 1}
    print("PASS lock: stale lock broken, fresh lock times out and leaves staging intact")


def test_database_inspection_and_rebuild(tmp):
    from mitim_tools.misc_tools.GUItools import FigureNotebook
    file = tmp / 'db4' / 'central.nc'
    folder = tmp / 'runI' / 'Outputs' / 'harvest'
    rec = H.harvest_recorder(_opts(folder))
    rng = np.random.default_rng(0)
    for i in range(30):
        rec._write({'code': 'tglf', 'inputs': {'RLTS_1': 1 + i * 0.1, 'RLTS_2': 2 + rng.random(), 'RLNS_1': 0.5, 'NS': 3},
                    'outputs': {'Qe': i * 0.2, 'Qi': i * 0.4, 'Ge': 0.0, 'Gi_1': 0.1}, 'meta': {'code_version': 'h1 [d]', 'machine': 'local'}})
    for i in range(5):
        rec._write({'code': 'eped', 'inputs': {'neped': 20 + i, 'betan': 1.0 + 0.1 * i, 'ip': 8.7}, 'outputs': {'ptop_kPa': 50 + 5 * i, 'wtop_psipol': 0.05, 'limiting_mode': 'peeling' if i % 2 else 'ballooning'}, 'meta': {}})
    db = H.harvest_database(file)
    db.push([folder])

    summ = db.summary()
    assert set(summ['code']) == {'tglf', 'eped'} and int(summ.set_index('code').loc['tglf', 'records']) == 30
    report = db.interpret()
    assert 'Distinct inputs: 30' in report and 'RLTS_1' in report and 'h1 [d]' in report and 'local' in report

    fn = FigureNotebook("harvest test", show=False)
    db.plotDatabase(fn=fn)
    assert fn.tab_titles == ['Overview', 'TGLF', 'TGLF by radius', 'TGLF ranges', 'TGLF pairs', 'EPED', 'EPED ranges', 'EPED pairs'], fn.tab_titles
    # input coverage: categories, discrete flag relative to the record count, constants
    cov = db.coverage('tglf').set_index('input')
    assert cov.loc['RLTS_1', 'category'] == 'drives' and cov.loc['NS', 'n_distinct'] == 1 and not cov.loc['NS', 'discrete']
    assert cov.loc['RLTS_1', 'n_distinct'] == 30 and not cov.loc['RLTS_1', 'discrete'] and cov.loc['RLNS_1', 'width'] == 0
    ce = db.coverage('eped').set_index('input')
    assert ce.loc['neped', 'category'] == 'pedestal' and ce.loc['ip', 'category'] == 'engineering' and ce.loc['neped', 'discrete']
    ax = db.plot('tglf', 'RLTS_2', 'Qi')
    assert ax.get_xlabel() == 'in_RLTS_2' and ax.get_ylabel() == 'out_Qi'

    file.write_bytes(b'garbage')
    n = db.rebuild([tmp / 'runI'])
    assert n == {'tglf': 30, 'eped': 5}, n
    assert len(db.load('tglf')) == 30 and len(list(file.parent.glob('central.nc.corrupt-*'))) == 1
    assert len(db.runs()) == 2 and db.load('tglf')['code_version'].iloc[0] == 'h1 [d]'
    assert H.staging_folders_of(tmp / 'runI') == [folder]
    # peek at staging without pushing: works on pushed archives, rolled archives and plain tails alike, touches nothing
    rec._write({'code': 'tglf', 'inputs': {'RLTS_1': 99.0, 'NS': 3}, 'outputs': {'Qe': 1.0}, 'meta': {}})
    before = sorted(p.name for p in folder.iterdir())
    peek = H.harvest_database.from_staging([tmp / 'runI'])
    assert peek.file != db.file and len(peek.load('tglf')) == 31 and len(peek.load('eped')) == 5 and peek.load('tglf')['machine'].iloc[0] == 'local'
    assert sorted(p.name for p in folder.iterdir()) == before, "the peek must not archive or rename anything"
    peek2 = H.harvest_database.from_staging([folder])
    assert len(peek2.load('tglf')) == 31

    # parity between codes run at the same plasma points (same run, r/a, q, electron gradients);
    # scan-trick members (perturbed gradients) never match, CGYRO stds become error bars
    fp = tmp / 'runP' / 'Outputs' / 'harvest'
    rp = H.harvest_recorder(_opts(fp))
    for i, roa in enumerate([0.3, 0.5, 0.7]):
        base = {'RMIN_LOC': roa, 'Q_LOC': 1.5 + i, 'RLTS_1': 2.0 + i, 'RLNS_1': 0.8, 'RLTS_2': 2.5, 'SAT_RULE': 2 if i < 2 else 3}
        rp._write({'code': 'tglf', 'inputs': base, 'outputs': {'Qe': 1.0 + i, 'Qi': 2.0 + i, 'Ge': 0.1}, 'meta': {}})
        rp._write({'code': 'tglf', 'inputs': {**base, 'RLTS_1': (2.0 + i) * 1.02}, 'outputs': {'Qe': 9.0, 'Qi': 9.0, 'Ge': 9.0}, 'meta': {}})
        rp._write({'code': 'cgyro', 'inputs': {'rmin': roa, 'q': 1.5 + i, 'z_0': 1.0, 'z_1': -1.0, 'dlntdr_0': 3.0, 'dlntdr_1': 2.0 + i, 'dlnndr_1': 0.8},
                   'outputs': {'Qe_mean': 1.2 + i, 'Qe_std': 0.1, 'Qi_mean': 2.5 + i, 'Qi_std': 0.2, 'Ge_mean': 0.0, 'Ge_std': 0.05}, 'meta': {}})
    rp._write({'code': 'cgyro', 'inputs': {'rmin': 0.9, 'q': 4.0, 'z_0': 1.0, 'z_1': -1.0, 'dlntdr_0': 3.0, 'dlntdr_1': 5.0, 'dlnndr_1': 0.8},
               'outputs': {'Qe_mean': 7.0, 'Qe_std': 0.1, 'Qi_mean': 8.0, 'Qi_std': 0.2, 'Ge_mean': 0.0, 'Ge_std': 0.05}, 'meta': {}})
    dbp = H.harvest_database(tmp / 'db6' / 'central.nc')
    dbp.push([fp])
    pairs = dbp.match_records('tglf', 'cgyro')
    assert len(pairs) == 3 and list(pairs['Qe_a']) == [1.0, 2.0, 3.0] and list(pairs['Qe_b']) == [1.2, 2.2, 3.2] and list(pairs['Qi_std_b']) == [0.2] * 3
    assert 'Qe_std_a' not in pairs.columns, "TGLF has no stored std"
    assert list(pairs['SAT_RULE']) == ['SAT2', 'SAT2', 'SAT3'], "the TGLF saturation rule labels each matched pair"
    fnp = FigureNotebook("parity test", show=False)
    dbp.plotDatabase(fn=fnp)
    assert fnp.tab_titles == ['Overview', 'TGLF', 'TGLF by radius', 'TGLF settings', 'TGLF ranges', 'TGLF pairs',
                              'CGYRO', 'CGYRO by radius', 'CGYRO ranges', 'CGYRO pairs', 'Parity TGLF-CGYRO'], fnp.tab_titles
    assert dbp.match_records('tglf', 'eped').empty

    # species resolved by charge (NEO as MITIM writes it has electrons LAST), collisionality per code, 3x4 grid tab
    import pandas as pd
    neo = pd.DataFrame({'in_Z_1': [1.0], 'in_Z_2': [6.0], 'in_Z_3': [-1.0], 'in_DLNTDR_1': [2.0], 'in_DLNTDR_3': [3.0],
                        'in_DLNNDR_3': [0.5], 'in_NU_1': [1e-4]})
    assert dbp._drives('neo', neo) == {'Te': 'in_DLNTDR_3', 'ne': 'in_DLNNDR_3', 'Ti': 'in_DLNTDR_1'}
    assert dbp._collisionality('neo', neo) == ('in_NU_1', 'NU_1 (collision frequency of NEO species 1, ion Z=1)')
    tg = pd.DataFrame({'in_ZS_1': [-1.0], 'in_ZS_2': [1.0], 'in_RLTS_1': [2.0], 'in_RLTS_2': [3.0], 'in_RLNS_1': [1.0], 'in_XNUE': [0.1]})
    assert dbp._drives('tglf', tg) == {'Te': 'in_RLTS_1', 'ne': 'in_RLNS_1', 'Ti': 'in_RLTS_2'} and dbp._collisionality('tglf', tg)[0] == 'in_XNUE'
    fig = dbp.plotFluxesVsDrives('tglf')
    assert fig is not None and len(fig.axes) == 12, "3 fluxes x (3 drives + distribution); colored by radius (legend, no colorbar)"
    print("PASS database summary / interpret / plotDatabase / plot / rebuild / staging_folders_of")


def test_statistics(tmp):
    '''Known answers: Qi = (a/LTi)^3, Qe = (a/LTe)^2, Ge = a/Lne, with +-2% one-at-a-time scans around each base point'''
    from mitim_tools.misc_tools.GUItools import FigureNotebook
    rng = np.random.default_rng(3)
    folder = tmp / 'runS' / 'Outputs' / 'harvest'
    rec = H.harvest_recorder(_opts(folder))

    def write(aLTe, aLTi, aLne, xnue, q):
        inputs = {'ZS_1': -1.0, 'ZS_2': 1.0, 'RLTS_1': aLTe, 'RLTS_2': aLTi, 'RLTS_3': aLTi, 'RLNS_1': aLne, 'XNUE': xnue,
                  'Q_LOC': q, 'NS': 3}
        rec._write({'code': 'tglf', 'inputs': inputs, 'outputs': {'Qe': aLTe ** 2, 'Qi': aLTi ** 3, 'Ge': aLne}, 'meta': {}})

    for _ in range(40):
        aLTe, aLTi, aLne, xnue = 1 + 2 * rng.random(), 1 + 2 * rng.random(), 0.2 + rng.random(), 0.01 + 0.1 * rng.random()
        q = 1.0 + 20 * xnue + 1e-3 * rng.random()   # q follows collisionality (confounded), fixed per base point
        write(aLTe, aLTi, aLne, xnue, q)
        for m in (0.98, 1.02):
            write(aLTe * m, aLTi, aLne, xnue, q)
            write(aLTe, aLTi * m, aLne, xnue, q)
            write(aLTe, aLTi, aLne * m, xnue, q)
    db = H.harvest_database(tmp / 'db7' / 'central.nc')
    db.push([folder])
    st = db.statistics('tglf')
    assert st.enough and list(st.inputs) == ['a/LTe (RLTS_1)', 'a/LTi (RLTS_2)', 'a/Lne (RLNS_1)', 'XNUE', 'Q_LOC'], list(st.inputs)
    pr = st.prcc()
    assert pr.loc['a/LTi (RLTS_2)', 'Qi'] > 0.95 and pr.loc['a/LTe (RLTS_1)', 'Qe'] > 0.95 and pr.loc['a/Lne (RLNS_1)', 'Ge'] > 0.95
    assert abs(pr.loc['a/LTi (RLTS_2)', 'Qe']) < 0.2, "Qe does not depend on a/LTi"
    loc = st.local_sensitivities()
    assert set(loc['group']) == {'RLTS_1', 'RLTS_2,RLTS_3', 'RLNS_1'}, "tied columns (RLTS_2 = RLTS_3) form one scanned group"
    assert len(loc) == 40 * 3 * 3, len(loc)
    el = loc.groupby(['column', 'flux'])['elasticity'].median()
    assert abs(el[('in_RLTS_2', 'Qi')] - 3) < 0.01 and abs(el[('in_RLTS_1', 'Qe')] - 2) < 0.01, el
    assert abs(el[('in_RLTS_1', 'Qi')]) < 1e-9 and np.isnan(el[('in_RLNS_1', 'Ge')]), "no Ge elasticity (only dGe/dlnx)"
    ge = loc[(loc.column == 'in_RLNS_1') & (loc.flux == 'Ge')]
    assert np.allclose(ge['dQdlnx'], ge['x0'], rtol=1e-6), "Ge = a/Lne -> dGe/dln(a/Lne) = a/Lne"
    report = st.interpret()
    assert 'strongest partial rank correlations' in report and 'a/LTi (RLTS_2)' in report and '120 one-at-a-time clusters' in report
    fn = FigureNotebook("stats test", show=False)
    assert st.plotImportance(fn=fn) is not None and st.plotSensitivities(fn=fn) is not None
    assert fn.tab_titles == ['TGLF stats', 'TGLF sensitivities']
    assert db.statistics('tglf') is st, "cached per code"
    print("PASS statistics: Spearman/PRCC pick the true drives, scan-trick elasticities 3 and 2 recovered, plots")


def test_type_flips(tmp):
    '''A column keeps the type it was created with; later pushes of the other type are converted or go to <col>__str, never fail halfway'''
    file = tmp / 'dbT' / 'central.nc'
    db = H.harvest_database(file)
    fA, fB, fC = (tmp / f'runT{k}' / 'harvest' for k in 'ABC')
    rA = H.harvest_recorder(_opts(fA))
    rA._write({'code': 'tglf', 'inputs': {'X': 1.0, 'MODE': 'GYRO'}, 'outputs': {'Qe': 1.0, 'kind': 'ITG'}, 'meta': {}})
    assert db.push([fA]) == {'tglf': 1}
    rB = H.harvest_recorder(_opts(fB))
    rB._write({'code': 'tglf', 'inputs': {'X': 'abc', 'MODE': 3}, 'outputs': {'Qe': 2.0, 'kind': 7.5}, 'meta': {}})
    rB._write({'code': 'tglf', 'inputs': {'X': 2.5, 'MODE': 'x'}, 'outputs': {'Qe': 3.0, 'kind': 'TEM'}, 'meta': {}})
    rB._write({'code': 'neo', 'inputs': {'N': 1}, 'outputs': {'Qi': 0.1}, 'meta': {}})
    assert db.push([fB]) == {'tglf': 2, 'neo': 1}
    df = db.load('tglf', with_run_info=False)
    # str into f8: parseable -> number, else NaN + the string in the sibling
    assert df['in_X'].iloc[0] == 1.0 and np.isnan(df['in_X'].iloc[1]) and df['in_X'].iloc[2] == 2.5
    assert list(df['in_X__str']) == ['', 'abc', ''], list(df['in_X__str'])
    # numbers into a str column: written as strings
    assert list(df['in_MODE']) == ['GYRO', '3', 'x'] and list(df['out_kind']) == ['ITG', '7.5', 'TEM']
    # a push whose reconciliation fails writes NOTHING (no group appended), and its staging goes back intact
    import netCDF4
    rC = H.harvest_recorder(_opts(fC))
    rC._write({'code': 'tglf', 'inputs': {'X': 9.0}, 'outputs': {'Qe': 9.0}, 'meta': {}})
    rC._write({'code': 'neo', 'inputs': {'N': 2}, 'outputs': {'Qi': 0.2}, 'meta': {}})
    orig = H._reconcile_frame
    def failing(df, existing):
        if 'in_N' in df.columns:
            raise TypeError("injected failure on the second group")
        return orig(df, existing)
    H._reconcile_frame = failing
    try:
        db.push([fC])
        raise AssertionError("push must raise")
    except TypeError:
        pass
    finally:
        H._reconcile_frame = orig
    with netCDF4.Dataset(file) as ds:
        assert len(ds.groups['tglf'].dimensions['record']) == 3 and len(ds.groups['neo'].dimensions['record']) == 1, "nothing appended"
    assert not list(fC.glob('*.claim-*')) and not list(fC.glob('*.pushed-*')) and len(list(fC.glob('*.jsonl.gz'))) == 2, "staging restored as rolled archives"
    assert db.push([fC]) == {'tglf': 1, 'neo': 1} and len(db.load('tglf')) == 4
    print("PASS type flips: f8<-str parse or __str sibling, str<-number, failed reconciliation writes nothing")


def _stage_worker(args):
    folder, n, tag, roll = args
    sys.path.insert(0, str(mitim_root))
    from mitim_tools.harvest_tools import HARVESTtools as H2
    H2.ROLL_BYTES = roll
    rec = H2.harvest_recorder(H2.options_from_namelist({'enabled': True}, folder))
    for i in range(n):
        rec._write({'code': 'tglf', 'inputs': {'RLTS_1': float(i), 'tag': tag, 'pad': 'x' * 200}, 'outputs': {'Qe': float(i)}, 'meta': {}})
    return os.getpid()


def test_concurrent_staging_same_folder(tmp):
    '''Two processes staging (and rolling) into ONE folder at the same time: each owns its files, no line lost'''
    folder = tmp / 'shared' / 'harvest'
    H.options_from_namelist({'enabled': True}, folder)
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(2) as pool:
        pids = pool.map(_stage_worker, [(folder, 300, 'a', 2000), (folder, 300, 'b', 2000)])
    stems = {H._stem_of(f) for f in folder.glob('tglf.*.jsonl*')}
    assert len(stems) == 2 and all(any(str(p) in s for s in stems) for p in pids), stems
    assert len(list(folder.glob('*.jsonl.gz'))) == 2, "both writers rolled"
    db = H.harvest_database(tmp / 'dbS' / 'central.nc')
    assert db.push([folder]) == {'tglf': 600}
    df = db.load('tglf')
    assert sorted(df.groupby('in_tag').size()) == [300, 300]
    print("PASS two processes staging into the same folder with rolls: 600/600 records, one file set per writer")


def test_legacy_staging_files(tmp):
    '''Runs staged before per-process files (<code>.jsonl, <code>.jsonl.gz, <code>.jsonl.pushed-<ts>.gz) still dedup, push and rebuild'''
    folder = tmp / 'legacy' / 'Outputs' / 'harvest'
    o = _opts(folder)
    rid = o['run_meta']['run']
    rows = []
    for i in range(4):
        inputs = {'RLTS_1': float(i)}
        rows.append(json.dumps({'run': rid, 'hash': H.input_hash('tglf', inputs), 'in_RLTS_1': float(i), 'out_Qe': float(i)}))
    with gzip.open(folder / 'tglf.jsonl.gz', 'wt') as fo:
        fo.write('\n'.join(rows[:2]) + '\n')
    (folder / 'tglf.jsonl').write_text('\n'.join(rows[2:]) + '\n')
    assert H._code_of(folder / 'tglf.jsonl') == 'tglf' and H._code_of(folder / 'tglf.host-12.jsonl.gz') == 'tglf'
    H._SEEN.clear()
    rec = H.harvest_recorder(o)
    assert rec._write({'code': 'tglf', 'inputs': {'RLTS_1': 2.0}, 'outputs': {'Qe': 2.0}, 'meta': {}}) == 0, "dedup reads the legacy tail"
    assert rec._write({'code': 'tglf', 'inputs': {'RLTS_1': 0.0}, 'outputs': {'Qe': 0.0}, 'meta': {}}) == 0, "... and the legacy archive"
    assert rec._write({'code': 'tglf', 'inputs': {'RLTS_1': 7.0}, 'outputs': {'Qe': 7.0}, 'meta': {}}) == 1
    db = H.harvest_database(tmp / 'dbL' / 'central.nc')
    assert db.push([folder]) == {'tglf': 5}
    pushed = sorted(f.name for f in folder.glob('*.pushed-*'))
    assert len(pushed) == 2 and any(n.startswith('tglf.jsonl.pushed-') for n in pushed), pushed
    assert not list(folder.glob('*.jsonl')) and not list(folder.glob('*.jsonl.gz'))
    # an old pushed archive from before the change is re-read by rebuild
    assert db.rebuild([tmp / 'legacy']) == {'tglf': 5}
    print("PASS legacy <code>.jsonl staging: dedup, push, archive and rebuild")


def _cgyro_folder(root, ctx, it, rho, t_last, sources=None, sub='base_cgyro', json_sub=None, info=None, extra=None):
    d = root / f"{ctx}{it}" / 'transport_simulation_folder' / sub
    d.mkdir(parents=True, exist_ok=True)
    s = f"_{rho:.4f}"
    t = np.arange(1.0, t_last + 0.5, 1.0)
    (d / f"out.cgyro.time{s}").write_text("\n".join(f" {x:.4E}  1.0E-03  1.0E-06  1.0E-02" for x in t) + "\n")
    if sources is not None:
        (root / f"{ctx}{it}" / 'transport_simulation_folder' / (json_sub or sub)).mkdir(parents=True, exist_ok=True)
        (root / f"{ctx}{it}" / 'transport_simulation_folder' / (json_sub or sub) / 'restart_sources.json').write_text(
            json.dumps({'mode': 'all', 'evaluation_number': it, 'context_label': 'Evaluation', 'sources': sources}))
    header = " nc_loc | nv_loc | nsplit | n_jtheta | n_MPI | n_OMP\n  2944      512     2048          4        8   16\n"
    (d / f"out.cgyro.info{s}").write_text(info if info is not None else f"INFO: (CGYRO) Initializing with restart data.\n{header}INFO: (CGYRO) GPU-aware code triggered.\nEXIT: (CGYRO) Normal\n")
    (d / f"out.cgyro.hosts{s}").write_text("".join(f"RANK={r} C1=0 C2={r} host=node{r // 4}\n" for r in range(8)))
    (d / f"input.cgyro{s}").write_text("NONLINEAR_FLAG=1\nMAX_TIME=100\nDELTA_T=0.04\n")
    for name, text in (extra or {}).items():
        (d / f"{name}{s}").write_text(text)
    obj = SimpleNamespace(folder=d, suffix_read=s, t=t, timing_total=np.full(len(t), 30.0), timing_setup={'input': 1.0, 'coll_init': 9.0},
                          cgyro_version='26-May-07 18:39:35 [208e0edea [2025-04-29]][PSFCLUSTER_GPU][0.0]')
    return obj


def test_cgyro_run_fields(tmp):
    from mitim_tools.gacode_tools.utils.CGYROutils import harvest_run_fields
    root, rho, k = tmp / 'pc' / 'Execution', 0.5, '0.5000'
    # chain ("all"): ev0 cold (t=100) <- ev1 (t=150) <- ev2 (t=120) <- ev3 (t=80)
    cold = harvest_run_fields(_cgyro_folder(root, 'Evaluation.', 0, rho, 100,
                                            info=" nc_loc | nv_loc | nsplit | n_jtheta | n_MPI | n_OMP\n 1 1 1 1 4 8\nEXIT: (CGYRO) Normal\n"))
    assert cold['restart_warm'] == 0 and cold['restart_t_inherited'] == 0 and np.isnan(cold['restart_source_iter'])
    _cgyro_folder(root, 'Evaluation.', 1, rho, 150, sources={k: 0})
    _cgyro_folder(root, 'Evaluation.', 2, rho, 120, sources={k: 1})
    o = harvest_run_fields(_cgyro_folder(root, 'Evaluation.', 3, rho, 80, sources={k: 2}))
    assert o['restart_warm'] == 1 and o['restart_source_iter'] == 2 and o['restart_t_inherited'] == 100 + 150 + 120, o
    # completion and cost
    assert o['max_time'] == 100 and o['t_start'] == 0 and o['reached_max_time'] == 1 and o['budget_stop'] == 0
    assert o['cost_s_per_acs'] == 30.0 and o['wall_s'] == 30.0 * 80 + 10.0
    assert o['n_mpi'] == 8 and o['n_omp'] == 16 and o['n_nodes'] == 2 and o['gpu'] == 1
    # the plotter's alignment gives the same offset (its walker is what harvest calls)
    from mitim_tools.gacode_tools.utils import CGYROplot
    src = CGYROplot.load_restart_sources_for_iterations([(i, root / f"Evaluation.{i}" / 'transport_simulation_folder') for i in range(4)])
    assert src[3]['parents'][k] == 2
    # a parent missing on disk -> NaN, never an undercount
    import shutil as sh
    sh.rmtree(root / 'Evaluation.1')
    assert np.isnan(harvest_run_fields(_cgyro_folder(root, 'Evaluation.', 3, rho, 80, sources={k: 2}))['restart_t_inherited'])
    # truncated (no EXIT after the last launch), stopped by the budget watchdog, continued in place
    header = " nc_loc | nv_loc | nsplit | n_jtheta | n_MPI | n_OMP\n 1 1 1 1 8 16\n"
    t = harvest_run_fields(_cgyro_folder(root, 'Evaluation.', 4, rho, 60, sources={k: 3}, info=f"EXIT: (CGYRO) Normal\nINFO: (CGYRO) Restart data found.\n{header}",
                                         extra={'mitim_budget.tag': 'BUDGET t=60'}))
    assert t['reached_max_time'] == 0 and t['budget_stop'] == 1 and np.isnan(t['t_start']), "EXIT of an earlier launch does not count"
    t = harvest_run_fields(_cgyro_folder(root, 'Evaluation.', 5, rho, 60, sources={k: 3}, info=f"INFO: (CGYRO) Restart data found.\n{header}",
                                         extra={'.mitim_t0': '40.0\n'}))
    assert t['t_start'] == 40.0 and t['reached_max_time'] == 0
    # warm start without a restart_sources.json (restart_from_folder): warm, parent unknown
    w = harvest_run_fields(_cgyro_folder(tmp / 'rf', 'Evaluation.', 0, rho, 50))
    assert w['restart_warm'] == 1 and np.isnan(w['restart_source_iter']) and np.isnan(w['restart_t_inherited'])
    # batched: JSON in <base>, traces in <base>_plasma<p>, parents are plasma 0
    rb = tmp / 'pb' / 'Execution'
    _cgyro_folder(rb, 'Evaluation.', 0, rho, 90, sub='base_cgyro_plasma0')
    b = harvest_run_fields(_cgyro_folder(rb, 'Evaluation.', 1, rho, 30, sub='base_cgyro_plasma1', sources={k: 0}, json_sub='base_cgyro'))
    assert b['restart_source_iter'] == 0 and b['restart_t_inherited'] == 90, b
    # simple-relax initialization folders chain the same way
    ri = tmp / 'sr' / 'Initialization' / 'initialization_simple_relax'
    _cgyro_folder(ri, 'portals_sr_ev_', 0, rho, 70)
    assert harvest_run_fields(_cgyro_folder(ri, 'portals_sr_ev_', 1, rho, 10, sources={k: 0}))['restart_t_inherited'] == 70
    # nothing on disk -> NaN everywhere, no exception
    e = harvest_run_fields(SimpleNamespace())
    assert all(np.isnan(v) for v in e.values()), e
    print("PASS CGYRO run fields: restart chain (all/batched/SR), missing parent NaN, completion, budget stop, in-place t_start, cost, ranks")


def test_radius_settings_windows(tmp):
    """r/a per code, radial groups, code settings (matched physics points), averaging-window tab"""
    import matplotlib
    matplotlib.use('Agg')
    folder = tmp / 'runS' / 'Outputs' / 'harvest'
    rec = H.harvest_recorder(_opts(folder))
    for i, roa in enumerate([0.35, 0.55, 0.75]):
        for sat in (2, 3):   # the same plasma point with two saturation rules
            base = {'RMIN_LOC': roa, 'ZS_1': -1.0, 'ZS_2': 1.0, 'RLTS_1': 2.0 + i, 'RLTS_2': 2.5, 'RLNS_1': 0.8, 'XNUE': 0.1, 'SAT_RULE': sat, 'NS': 2}
            rec._write({'code': 'tglf', 'inputs': base, 'outputs': {'Qe': sat * (1.0 + i), 'Qi': 2.0 + i, 'Ge': 0.1}, 'meta': {}})
        rec._write({'code': 'tglf', 'inputs': {**base, 'RLTS_2': 4.0, 'SAT_RULE': 3}, 'outputs': {'Qe': 1.0, 'Qi': 9.0, 'Ge': 0.1}, 'meta': {}})
        rec._write({'code': 'cgyro', 'inputs': {'RMIN': roa, 'Z_1': 1.0, 'Z_2': -1.0, 'DLNTDR_1': 2.5, 'DLNTDR_2': 2.0 + i, 'DLNNDR_2': 0.8,
                                                'NU_EE': 0.05, 'N_TOROIDAL': 16, 'MAX_TIME': 500.0 + 100 * i},
                    'outputs': {'Qe_mean': 1.0, 'Qe_std': 0.1, 'Qi_mean': 2.0, 'Qi_std': 0.2, 'Ge_mean': 0.0, 'Ge_std': 0.05,
                                'avg_tmin': 100.0, 'avg_tmax': 400.0, 't_last': 400.0, 'max_time': 500.0, 'reached_max_time': 0,
                                'restart_warm': 1, 'restart_t_inherited': 300.0, 'Qi_ncorr': 5.0, 'cost_s_per_acs': 20.0, 'wall_s': 8000.0, 'n_nodes': 1},
                    'meta': {}})
    db = H.harvest_database(tmp / 'db8' / 'central.nc')
    db.push([folder])
    tg, cg = db.load('tglf'), db.load('cgyro')
    assert list(db.radius('tglf', tg).unique()) == [0.35, 0.55, 0.75] and list(db.radius('cgyro', cg)) == [0.35, 0.55, 0.75]
    labels, colors = db._radial_bins(db.radius('tglf', tg))
    assert list(colors) == ['r/a=0.35', 'r/a=0.55', 'r/a=0.75'] and labels.iloc[0] == 'r/a=0.35'
    import pandas as pd
    labels, colors = db._radial_bins(pd.Series(np.linspace(0.3, 0.95, 20)))
    assert list(colors)[0] == 'r/a 0.30-0.40' and list(colors)[-1] == 'r/a 0.90-1.00' and len(colors) == 7, "more than 8 radii -> 0.1-wide bins"

    # settings: SAT_RULE varies (NS does not), MAX_TIME is run control, not a CGYRO setting
    _, varying, lab = db.settings('tglf')
    assert varying == ['in_SAT_RULE'] and set(lab) == {'SAT_RULE=2', 'SAT_RULE=3'}
    assert db.settings('cgyro')[1] == [], "N_TOROIDAL constant, MAX_TIME is run control"
    pairs = db.match_settings('tglf')
    assert len(pairs) == 3 and list(pairs['Qe']) == [2.0, 4.0, 6.0] and list(pairs['Qe_ref']) == [3.0, 6.0, 9.0], "only the same physics point pairs up"
    assert pairs.attrs['reference'] == 'SAT_RULE=3', "reference = the most common settings (3+3 records vs 3)"
    assert db.coverage('tglf').set_index('input').loc['SAT_RULE', 'category'] == 'settings'

    fig = db.plotSettings('tglf')
    assert fig is not None and db.plotSettings('neo') is None
    fig = db.plotWindows('cgyro')
    assert fig is not None and len(fig.axes) == 5 and db.plotWindows('tglf') is None
    fig = db.plotFluxesByRadius('cgyro')
    assert len(fig.axes) == 3 * 3 + 1, "3 fluxes x 3 radii + NU_EE colorbar"
    print("PASS radius per code, radial groups, settings + matched physics points, windows tab")


def main():
    tmp = Path(tempfile.mkdtemp(prefix='mitim_harvest_test_'))
    try:
        test_options_and_run_meta(tmp)
        test_no_default_file(tmp)
        test_mixed_cgyro_schemas(tmp)
        test_qualikiz_records(tmp)
        test_drop_and_type_fallback(tmp)
        test_eped_input_files(tmp)
        test_recorder_tglf_layout_and_dedup(tmp)
        test_shared_staging_folder(tmp)
        test_cgyro_gx_eped_interfaces(tmp)
        test_push_schema_union_and_load(tmp)
        test_rolling_compression(tmp)
        test_concurrent_push(tmp)
        test_stale_and_busy_lock(tmp)
        test_database_inspection_and_rebuild(tmp)
        test_radius_settings_windows(tmp)
        test_statistics(tmp)
        test_type_flips(tmp)
        test_concurrent_staging_same_folder(tmp)
        test_legacy_staging_files(tmp)
        test_cgyro_run_fields(tmp)
        print("\nALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
