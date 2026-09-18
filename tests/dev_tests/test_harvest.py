"""
test_harvest.py
===============
Sanity tests for the harvest capability (mitim_tools.harvest_tools.HARVESTtools): the per-run
recorder (staging JSONL with rolling gzip compression, dedup by input hash, numpy/NaN
serialization, provenance written once per run and code), the polymorphic
harvest_records/harvest_outputs interface on stand-in simulation objects (TGLF/NEO layout,
CGYRO, GX, EPED), and the central netCDF database (schema-union append, runs table join,
selective loading, two-process concurrent push under the mkdir lock, stale-lock recovery,
rebuild, summary/interpret/plot).

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
    return [json.loads(l) for l in (Path(folder) / f"{code}.jsonl").read_text().splitlines() if l.strip()]


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
    c.params1D = {'n_species': 3, 'dlntdr_0': 2.0, 'nonlinear_flag': True}
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
    assert c.harvest_inputs()['nonlinear_flag'] == 1 and c.harvest_hash_extra() == {'t_last': 500.0}
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
    pushed = list(fA.glob('tglf.jsonl.pushed-*'))
    assert not (fA / 'tglf.jsonl').exists() and len(pushed) == 1 and pushed[0].name.endswith('.gz'), "pushed file is gzipped in place"
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
    gz, plain = folder / 'tglf.jsonl.gz', folder / 'tglf.jsonl'
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
    assert not gz.exists() and not plain.exists() and len(list(folder.glob('tglf.jsonl.pushed-*.gz'))) == 1
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
    assert time.time() - t0 >= 2 and (folder / 'tglf.jsonl.gz').exists() and len(list(folder.glob('*.pushed-*'))) == 1, "staging must be left intact (rolled, unpushed) after a failed push"
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
    assert fn.tab_titles == ['Overview', 'TGLF', 'EPED'], f"Overview + TGLF + EPED tabs expected, got {fn.tab_titles}"
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
    assert fnp.tab_titles == ['Overview', 'TGLF', 'CGYRO', 'Parity TGLF-CGYRO'], fnp.tab_titles
    assert dbp.match_records('tglf', 'eped').empty
    print("PASS database summary / interpret / plotDatabase / plot / rebuild / staging_folders_of")


def main():
    tmp = Path(tempfile.mkdtemp(prefix='mitim_harvest_test_'))
    try:
        test_options_and_run_meta(tmp)
        test_recorder_tglf_layout_and_dedup(tmp)
        test_shared_staging_folder(tmp)
        test_cgyro_gx_eped_interfaces(tmp)
        test_push_schema_union_and_load(tmp)
        test_rolling_compression(tmp)
        test_concurrent_push(tmp)
        test_stale_and_busy_lock(tmp)
        test_database_inspection_and_rebuild(tmp)
        print("\nALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
