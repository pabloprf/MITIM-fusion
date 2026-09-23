"""
test_harvest_recover.py
=======================
Sanity tests for mitim_harvester (mitim_tools.harvest_tools.HARVESTrecover) on a synthetic MAESTRO
folder built from tests/data/input.tglf: TGLF records rebuilt from a PORTALS evaluation left on disk
(base point + one scan-trick member, tagged with their maestro_beat), provenance marked
`recovered_by`, --dry-run pushes nothing, a second run appends nothing (dedup against the file), a
parent folder is searched for runs, and eped.input / eped.config rebuilt from an EPED output .nc
give the same EPED record inputs as the originals.

Everything runs in a temporary folder -- no transport code, no cluster.

    python tests/dev_tests/test_harvest_recover.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.harvest_tools import HARVESTtools as H
from mitim_tools.harvest_tools import HARVESTrecover as R

DATA = Path(__file__).resolve().parents[1] / "data"


def _fake_maestro(root, name='case_A'):
    '''Beats/Beat_2/run_portals/.../portals_sr_ev_0/transport_simulation_folder/{base_tglf, turb_drives_RLTS_1_RLTS_1_1.02}'''
    run = root / 'scan' / name
    tsf = run / 'Beats' / 'Beat_2' / 'run_portals' / 'Initialization' / 'initialization_simple_relax' / 'portals_sr_ev_0' / 'transport_simulation_folder'
    text = (DATA / 'input.tglf').read_text()
    for sub, scale in (('base_tglf', 1.0), ('turb_drives_RLTS_1_RLTS_1_1.02', 1.02)):
        d = tsf / sub
        d.mkdir(parents=True)
        lines = [f"RLTS_1       = {2.5144 * scale}" if l.startswith('RLTS_1 ') else l for l in text.splitlines()]
        (d / 'input.tglf_0.5500').write_text("\n".join(lines) + "\n")
        vals = np.concatenate([[0.1, 0.2, 0.3], [2.0 * scale, 3.0, 0.5], [0.0, 0.01, 0.02], [0.0, 0.0, 0.0]])
        (d / 'out.tglf.gbflux_0.5500').write_text(" ".join(f"{v: .4E}" for v in vals) + "\n")
    (run / 'maestro.namelist.actual.yaml').write_text("maestro:\n  beats: [transp, portals]\n")
    return run


def test_tglf_records_and_dedup(tmp):
    run = _fake_maestro(tmp).resolve()   # macOS: /var -> /private/var
    file = tmp / 'recovered.nc'

    assert R.find_runs([tmp]) == [run], "parent folder searched for runs"
    assert R.harvest_runs([run], file, dry_run=True) == {'tglf': 2, 'neo': 0, 'eped': 0}
    assert not file.exists(), "--dry-run pushes nothing"

    assert R.harvest_runs([tmp / 'scan'], file, stage=tmp / 'stage') == {'tglf': 2, 'neo': 0, 'eped': 0}
    db = H.harvest_database(file)
    df = db.load('tglf')
    assert len(df) == 2 and set(df['maestro_beat']) == {2.0}
    assert sorted(df['out_Qe']) == [2.0, 2.04], "Qe read back from out.tglf.gbflux (base and scan member)"
    assert sorted(df['in_RLTS_1'].round(6)) == [2.5144, round(2.5144 * 1.02, 6)]
    runs = db.runs()
    assert runs['recovered_by'].iloc[0].startswith('mitim_harvester') and runs['run_folder'].iloc[0] == str(run)
    assert runs['run'].iloc[0] == R.run_id_of(run), "stable run id"

    assert R.harvest_runs([run], file, stage=tmp / 'stage') == {'tglf': 0, 'neo': 0, 'eped': 0}, "same staging: nothing new"
    assert R.harvest_runs([run], file) == {'tglf': 0, 'neo': 0, 'eped': 0}, "fresh staging: known (run, hash) skipped"
    assert len(H.harvest_database(file).load('tglf')) == 2
    assert R.harvest_runs([run], file, scan_trick_members=False, dry_run=True)['tglf'] == 0
    print("PASS TGLF records from disk, maestro_beat, provenance, dry run, dedup on re-run")


def test_eped_files_from_nc(tmp):
    import f90nml
    import xarray as xr
    inputs = {'a': 0.43, 'betan': 1.4, 'bt': 11.4, 'delta': 0.53, 'ip': 3.09, 'kappa': 1.44, 'm': 2.5, 'mi': 7.98, 'neped': 24.1,
              'nesep': 9.64, 'num_scan': 1, 'ptotped': -1, 'ptotwid': 0.03, 'r': 1.68, 'runid': 0, 'shot': 0, 'teped': -1,
              'tesep': 200.0, 'tewid': 0.03, 'timeid': 0, 'z': 1.0, 'zeffped': 1.5, 'zi': 4.19}
    config = {'NMODES': [5, 6, 8, 10, 15, 20, 30], 'WIDTHS': [3, 4, 5, 7, 9], 'TEPED_BOUND': [0.1, 1.4, 0.01]}
    d = tmp / 'eped'
    d.mkdir()
    f90nml.write(f90nml.Namelist({'eped_input': inputs}), d / 'eped.input.1', force=True)
    (d / 'eped.config1').write_text("\n".join(f"    {k} = {' '.join(str(v) for v in vals)}" for k, vals in config.items()) + "\n")
    ds = xr.Dataset({**{k: ('dim_one', [float(v)]) for k, v in inputs.items() if k != 'num_scan'}, 'zeta': ('dim_one', [0.0]),
                     'nmodes': ('dim_nmodes', config['NMODES']), 'widths': ('dim_widths', config['WIDTHS']),
                     'teped_bound': ('dim_three', config['TEPED_BOUND'])})
    ds.to_netcdf(d / 'output_run1.nc')

    a = H.collect_eped(None, eped_input_file=d / 'eped.input.1', eped_config_file=d / 'eped.config1')['inputs']
    b = H.collect_eped(None, eped_input_file=R.write_eped_input(d / 'output_run1.nc', d / 'rebuilt.input', 'standard'),
                       eped_config_file=R.write_eped_config(d / 'output_run1.nc', d / 'rebuilt.config'))['inputs']
    assert H._scalar_dict(a) == H._scalar_dict(b), "rebuilt eped.input/config give the same record inputs (zeta dropped for 'standard')"
    print("PASS eped.input / eped.config rebuilt from output_run1.nc")


def main():
    tmp = Path(tempfile.mkdtemp(prefix='mitim_harvester_test_'))
    try:
        test_tglf_records_and_dedup(tmp)
        test_eped_files_from_nc(tmp)
        print("\nALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
