'''
Harvester: rebuild the harvest records of MAESTRO (or PORTALS) runs that ran WITHOUT harvesting, from
what they left on disk, and append them to a harvest file (record layout: see HARVESTtools).

    TGLF / NEO : every TGLF/NEO run folder of every PORTALS evaluation still on disk
                 (Initialization/initialization_simple_relax/portals_sr_ev_*/ and Execution/Evaluation.*/,
                 transport_simulation_folder/<base_tglf, base_neo, turb_drives_*>), read with MITIM's own
                 readers and recorded by the same harvest_recorder a live run uses: same fields, same hash.
    QuaLiKiz   : base_qlk and the stacked scan-trick run (turb_drives/) of every evaluation, GB-normalized with
                 the evaluation's input.gacode_torun as the live run does.
    EPED       : every full-EPED evaluation whose output_run1.nc survived (Beat_N/run_eped/case1/, else the
                 pruned Beat_N/beat_results/ copy; the eped_initializer creator as maestro_beat 0), recorded
                 with collect_eped. When pruning removed eped.input.1 / eped.config1, they are rebuilt from
                 the .nc, which carries every eped.input value and the NMODES/WIDTHS/TEPED_BOUND scan.

Not recoverable from disk (a live run would have them): TGLF/NEO evaluations whose run folders were
pruned (MAESTRO keep_all_files: false keeps none), EPED retries overwritten by the final attempt, the
machine/modules the codes ran with, and the MITIM version of the run. The runs table marks recovered
runs with `recovered_by` (mitim_harvester version@commit).

Each run stages into its own folder (like a live run's Outputs/harvest) and is pushed with
harvest_database.push. Dedup: a run keeps a stable id (the live one when the run staged anything,
else a hash of its folder path) and every (run, hash) already in the target file is skipped, so
running the harvester twice on the same run appends nothing.

    mitim_harvester --from-disk <run or parent folder> [...] [--file F] [--dry-run] [--stage DIR]   (HARVESTtools.main_harvester)
'''

import re
import copy
import json
import yaml
import hashlib
import datetime
import tempfile
import numpy as np
from pathlib import Path
from mitim_tools import __version__ as mitim_version, __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print, HiddenPrints
from mitim_tools.harvest_tools import HARVESTtools as H

RECOVERED_CODES = ('tglf', 'neo', 'eped', 'qualikiz')

# Output file whose presence at a radius means the code finished there (+ its input file, always retrieved)
_GACODE_FILES = {'tglf': ('out.tglf.gbflux', 'input.tglf'), 'neo': ('out.neo.transport_flux', 'input.neo')}

# eped.input as EPEDtools.EPED._prep writes it: MAESTRO's input_params (+ zeta / s_three, s_four for the
# non-standard TOQ equilibria), the fixed keys and the composition. num_scan is the only one not in the .nc.
_EPED_INPUT_KEYS = ['ip', 'bt', 'r', 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep']
_EPED_SHAPE_KEYS = {'standard': [], 'full_turnbull_miller': ['zeta'], 'mxh': ['zeta', 's_three', 's_four']}
_EPED_FIXED_KEYS = ['shot', 'timeid', 'runid', 'tewid', 'ptotwid', 'teped', 'ptotped', 'm', 'z', 'mi', 'zi']
_EPED_INT_KEYS = {'num_scan', 'shot', 'timeid', 'runid'}
_EPED_CONFIG_VARS = {'NMODES': 'nmodes', 'WIDTHS': 'widths', 'TEPED_BOUND': 'teped_bound'}

def _trailing_int(path):
    last = re.split(r'[._]', path.name)[-1]
    return int(last) if last.isdigit() else None

_SKIPPED = set()

def _numbered(paths):
    '''Beat_<n> / Evaluation.<n> / portals_sr_ev_<n> sorted by n; renamed copies (e.g. Beat_14old) are skipped, since
    their records would be tagged with the number of a different attempt'''
    paths = [p for p in paths if p.is_dir()]
    for p in paths:
        if _trailing_int(p) is None and p not in _SKIPPED:
            _SKIPPED.add(p)
            print(f"\t- mitim_harvester: skipping {p} (name does not end in a number)")
    return sorted((p for p in paths if _trailing_int(p) is not None), key=_trailing_int)

def _first_existing(paths):
    return next((p for p in paths if p is not None and p.exists()), None)

def _owner(path):
    try:
        import pwd
        return pwd.getpwuid(path.stat().st_uid).pw_name
    except (ImportError, KeyError):
        return ''

def run_id_of(folder):
    '''The live id if the run staged anything (so its live records dedup), else a hash of the resolved folder path'''
    live = Path(folder) / 'Outputs' / 'harvest' / 'run_meta.json'
    if live.exists():
        return json.loads(live.read_text())['run']
    return hashlib.sha1(str(Path(folder).resolve()).encode()).hexdigest()[:12]

class harvester:
    '''One MAESTRO or PORTALS run folder -> its recoverable TGLF/NEO/QuaLiKiz/EPED records, staged in `staging_folder`'''

    def __init__(self, folder, staging_folder=None, scan_trick_members=True):
        self.folder = Path(folder).resolve()
        self.is_maestro = (self.folder / 'Beats').is_dir()
        self.staging_folder = Path(staging_folder) if staging_folder is not None else None   # needed by stage() only
        self.scan_trick_members = scan_trick_members
        self.run_id = run_id_of(self.folder)
        nml = self.folder / ('maestro.namelist.actual.yaml' if self.is_maestro else 'namelist.portals.yaml')
        self.namelist_file = nml if nml.exists() else None
        self.namelist = (yaml.safe_load(nml.read_text()) or {}) if nml.exists() else {}
        self.recorder = None
        self.failed = []

    @staticmethod
    def is_run(folder):
        folder = Path(folder)
        return (folder / 'Beats').is_dir() or (folder / 'Execution').is_dir() or (folder / 'Initialization' / 'initialization_simple_relax').is_dir()

    # -------------------------------------------------------------------------- what is on disk
    def _beats(self):
        return _numbered((self.folder / 'Beats').glob('Beat_*')) if self.is_maestro else []

    def portals_folders(self):
        '''[(maestro_beat or None, PORTALS folder)]: run_portals/ and beat_results/ twins alike (the hash dedups them)'''
        if not self.is_maestro:
            return [(None, self.folder)]
        return [(_trailing_int(b), b / sub) for b in self._beats() for sub in ('run_portals', 'beat_results')
                if (b / sub / 'Initialization').is_dir() or (b / sub / 'Execution').is_dir()]

    @staticmethod
    def transport_folders(portals):
        sr = _numbered((portals / 'Initialization' / 'initialization_simple_relax').glob('portals_sr_ev_*'))
        ev = _numbered((portals / 'Execution').glob('Evaluation.*'))
        return [f / 'transport_simulation_folder' for f in sr + ev if (f / 'transport_simulation_folder').is_dir()]

    @staticmethod
    def code_folders(transport_folder):
        '''[(run folder, code)] of one evaluation: base_tglf, base_neo, the TGLF scan-trick members, and the QuaLiKiz
        runs (base_qlk and the stacked scan-trick run under turb_drives/: any folder with parameters.json and output/)'''
        out = []
        for d in sorted(p for p in transport_folder.iterdir() if p.is_dir()):
            for code, (out_file, _) in _GACODE_FILES.items():
                if next(d.glob(f"{out_file}_*"), None) is not None:
                    out.append((d, code))
        out += [(f.parent, 'qualikiz') for f in sorted(transport_folder.rglob('parameters.json')) if (f.parent / 'output').is_dir()]
        return out

    def eped_folders(self):
        '''[(maestro_beat, folder)] of the full-EPED evaluations on disk; the eped_initializer creator is beat 0'''
        out = [(0, c) for b in self._beats() for c in sorted(b.glob('initializer_*/creator_eped'))
               if (c / 'run_eped' / 'case1' / 'output_run1.nc').exists()] if self.is_maestro else []
        out += [(_trailing_int(b), b) for b in self._beats()
                if (b / 'run_eped' / 'case1' / 'output_run1.nc').exists() or (b / 'beat_results' / 'output_run1.nc').exists()]
        # Full EPED only, as in live harvest: an EPED-NN beat (use_full_EPED false, the default) never records,
        # even if a stale output_run1.nc from an earlier full-EPED attempt sits in its folder
        return [(beat, f) for beat, f in out if not self.is_maestro or self._eped_parameters(beat).get('use_full_EPED', False)]

    # -------------------------------------------------------------------------- staging
    def _host(self):
        '''Host the codes ran on, from the first simulation log kept on disk (live runs store the driver's host)'''
        for _, portals in self.portals_folders():
            for tsf in self.transport_folders(portals)[:1]:
                for log in sorted(tsf.glob('mitim_simulation_*.log')):
                    m = re.search(r'^Host:\s*(\S+)', log.read_text(errors='ignore'), re.M)
                    if m:
                        return m.group(1)
        return ''

    def _run_meta_extra(self):
        _, commit = IOtools.get_git_info(__mitimroot__)
        ref = self.namelist_file or self.folder
        return {'run_folder': str(self.folder), 'user': _owner(self.folder), 'host': self._host(),
                'created': datetime.datetime.fromtimestamp(ref.stat().st_mtime).isoformat(timespec='seconds'),
                'mitim_version': '', 'git_branch': '', 'git_commit': '',
                'recovered_by': f"mitim_harvester {mitim_version}@{(commit or '')[:10]}"}

    def stage(self, known_hashes=()):
        '''
        Stage every recoverable record; returns {code: new records}. known_hashes (this run's hashes
        already in the target file) seed the recorder's dedup set, so they are not staged again.
        '''
        opts = H.options_from_namelist({'enabled': True, 'push': False, 'run_id': self.run_id,
                                        'scan_trick_members': self.scan_trick_members},
                                       self.staging_folder, run_meta_extra=self._run_meta_extra())
        self.recorder = H.harvest_recorder(opts)
        H._seen(self.staging_folder).update(known_hashes)

        counts = {c: 0 for c in RECOVERED_CODES}
        for beat, portals in self.portals_folders():
            for tsf in self.transport_folders(portals):
                for sim_folder, code in self.code_folders(tsf):
                    record = self._record_qualikiz if code == 'qualikiz' else self._record_gacode
                    counts[code] += self._counted(record, code, sim_folder, beat)
        for beat, folder in self.eped_folders():
            counts['eped'] += self._counted(self._record_eped, beat, folder)
        return counts

    def _counted(self, fn, *args):
        '''New hashes this call staged (reads never raise: a broken folder is reported and skipped)'''
        seen = H._seen(self.staging_folder)
        n0 = len(seen)
        try:
            with HiddenPrints(show_if_contains='harvest: could not'):
                fn(*args)
        except Exception as e:
            self.failed.append(f"{' '.join(str(a) for a in args)}: {type(e).__name__}: {e}")
        return len(seen) - n0

    def _context(self, beat, **extra):
        return {**({'maestro_beat': beat} if beat is not None else {}), **extra}

    def _record_gacode(self, code, sim_folder, beat):
        from mitim_tools.gacode_tools import TGLFtools, NEOtools
        out_file, in_file = _GACODE_FILES[code]
        rhos = sorted(float(f.name.rsplit('_', 1)[1]) for f in sim_folder.glob(f"{out_file}_*")
                      if (sim_folder / f"{in_file}_{f.name.rsplit('_', 1)[1]}").exists())
        if not rhos:
            return
        # TGLF folders other than base_* are the std scan trick (turb_drives_*): skippable like live
        scan_member = int(code == 'tglf' and not sim_folder.name.startswith('base'))
        sim = (TGLFtools.TGLF if code == 'tglf' else NEOtools.NEO)(rhos=rhos)
        sim.harvest = self.recorder.with_context(**self._context(beat, scan_member=scan_member))
        if code == 'tglf':
            sim.read(label='recovered', folder=sim_folder, require_all_files=False)
        else:
            sim.read(label='recovered', folder=sim_folder)

    def _record_qualikiz(self, code, sim_folder, beat):
        '''One QuaLiKiz run: base_qlk (read, one dimx point per radius) or the stacked scan trick (read_cases, radii repeated
        per case). The radii are the plan's own rho; the GB normalization uses the evaluation's input.gacode_torun'''
        from mitim_tools.qualikiz_tools import QLKtools
        from mitim_tools.gacode_tools import PROFILEStools
        tsf = next(p for p in sim_folder.parents if p.name == 'transport_simulation_folder')
        rho_all = QLKtools.qualikiz_folder_to_xarray(sim_folder)['rho'].values.astype(float).tolist()
        rhos = list(dict.fromkeys(rho_all))   # base: every dimx point; scan: the radii of the first case
        sim = QLKtools.QuaLiKiz(rhos=rhos)
        sim.profiles = PROFILEStools.gacode_state(_first_existing([tsf / 'input.gacode_torun', tsf / 'input.gacode']))
        base = sim_folder.name.startswith('base')
        sim.harvest = self.recorder.with_context(**self._context(beat, scan_member=int(not base)))
        if base:
            sim.read(label='recovered', folder=sim_folder)
        else:
            sim.read_cases('recovered', n_cases=len(rho_all) // len(rhos), folder=sim_folder)

    def _eped_parameters(self, beat):
        '''parameters_prepare of the EPED beat (base_module merged; the creator also takes profiles_initialization.parameters)'''
        m = self.namelist.get('maestro', {}) or {}
        init = (self.namelist.get('plasma', {}) or {}).get('profiles_initialization', {}) or {}
        name = init.get('creator_type') if beat == 0 else (m.get('beats') or [None] * beat)[beat - 1]
        block = m.get(name) or {}
        prep = copy.deepcopy(block.get('parameters_prepare') or {})
        if block.get('base_module'):
            prep = IOtools.deep_dict_update(copy.deepcopy(m[block['base_module']].get('parameters_prepare') or {}), prep)
        if beat == 0:
            prep = IOtools.deep_dict_update(prep, copy.deepcopy(init.get('parameters') or {}))
        return prep

    def _record_eped(self, beat, folder):
        from mitim_tools.eped_tools import EPEDtools
        run = folder / 'run_eped' / 'case1'
        nc = _first_existing([run / 'output_run1.nc', folder / 'beat_results' / 'output_run1.nc'])
        input_file = _first_existing([run / 'run1' / 'eped.input.1', folder / 'beat_results' / 'eped.input'])
        config_file = _first_existing([run / 'run1' / 'eped.config1'])
        results_file = _first_existing([folder / 'beat_results' / 'eped_results.npy', folder / 'run_eped' / 'eped_results.npy', folder / 'eped_results.npy'])
        gacode = _first_existing([folder / 'run_eped' / 'input.gacode', folder / 'beat_results' / 'input.gacode',
                                  folder / 'initializer_previous_beat' / 'input.gacode', folder.parent / 'input.gacode'])

        prep = self._eped_parameters(beat)
        toq = prep.get('toq_eq_choice', 'standard')
        results = np.load(results_file, allow_pickle=True).item() if results_file is not None else {}
        rule, threshold = (list(results.get('stability_rule') or prep.get('stability_rule') or ['G', 0.03]) + [None])[:2]

        eped = EPEDtools.EPED(folder=None)
        eped.read(subfolder=nc, label='recovered', print_results=False, diamagnetic_stab_rule=rule, stability_threshold=threshold,
                  gacode_state=gacode if rule == 'W' else None)
        ds = eped.results['recovered']['run1']

        with tempfile.TemporaryDirectory() as tmp:
            input_file = input_file or write_eped_input(nc, Path(tmp) / 'eped.input.1', toq)
            config_file = config_file or write_eped_config(nc, Path(tmp) / 'eped.config1')
            self.recorder.with_context(**self._context(beat)).record_eped(
                input_params=None, composition=None, eped_params_override=prep.get('eped_params_override') or None,
                toq_eq_choice=toq, dataset=ds, ptop_kPa=ds['ptop'].item(), wtop_psipol=ds['wptop'].item(),
                limiting_mode_info=EPEDtools.limiting_mode_from_dataset(ds), eped_folder=None, job=None,
                eped_input_file=input_file, eped_config_file=config_file)

# ------------------------------------------------------------------------------------------------
# eped.input / eped.config rebuilt from output_run1.nc (validated equal to the originals where both survive)
# ------------------------------------------------------------------------------------------------

def write_eped_input(nc, file, toq_eq_choice='standard'):
    import f90nml
    import xarray as xr
    keys = _EPED_INPUT_KEYS + _EPED_SHAPE_KEYS.get(toq_eq_choice, []) + _EPED_FIXED_KEYS
    with xr.open_dataset(nc) as ds:
        vals = {k: np.asarray(ds[k].values).reshape(-1)[0].item() for k in keys if k in ds}
    vals['num_scan'] = 1
    # the .nc stores every number as a float: ints where _prep always writes ints (teped/ptotped are -1 from MAESTRO);
    # m, z, mi, zi stay floats (ints only when MAESTRO fell back to the default composition, not decidable from the .nc)
    vals = {k: (int(v) if k in _EPED_INT_KEYS or (k in ('teped', 'ptotped') and float(v).is_integer()) else float(v))
            for k, v in sorted(vals.items())}
    f90nml.write(f90nml.Namelist({'eped_input': vals}), file, force=True)
    return file

def write_eped_config(nc, file):
    import xarray as xr
    with xr.open_dataset(nc) as ds:
        lines = [f"{key} = {' '.join(repr(v) for v in np.asarray(ds[var].values).reshape(-1).tolist())}"
                 for key, var in _EPED_CONFIG_VARS.items() if var in ds]
    Path(file).write_text("\n".join(lines) + "\n")
    return file

# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------

def find_runs(paths, max_depth=3):
    '''Run folders among `paths`: each path is a MAESTRO/PORTALS run, or a folder with runs up to max_depth levels below (never inside a run)'''
    def walk(p, depth):
        if harvester.is_run(p):
            return [p]
        if depth == 0 or not p.is_dir():
            return []
        return [r for c in sorted(c for c in p.iterdir() if c.is_dir()) for r in walk(c, depth - 1)]
    return [r for p in paths for r in walk(IOtools.expandPath(p), max_depth)]

def known_hashes(db, run_ids):
    '''{run: {hash}} of the TGLF/NEO/QuaLiKiz/EPED records the file already holds for these runs'''
    known = {r: set() for r in run_ids}
    ids = np.array(sorted(run_ids), dtype=object)
    for code in [c for c in db.codes() if c in RECOVERED_CODES]:
        df = db._read_group(code, columns=['run', 'hash'], mask_fn=lambda read: np.isin(read('run'), ids))
        for r, h in zip(df.get('run', []), df.get('hash', [])):
            known[r].add(h)
    return known

def harvest_runs(paths, file, stage=None, dry_run=False, scan_trick_members=True, batch=25):
    '''
    Stage and push (in batches of `batch` runs) every recoverable record of the runs under `paths` into
    `file`. stage: keep the staging under <stage>/<file stem>/<run>_<id>/ (pushed archives kept, as in a
    live run); None stages in a temporary folder removed at the end. dry_run: stage in a temporary folder,
    report what would be appended, push nothing. Returns {code: records} (appended, or new for dry_run).
    '''
    db = H.harvest_database(file)
    runs = find_runs(paths)
    if stage is None or dry_run:
        root = Path(tempfile.mkdtemp(prefix='mitim_harvester_'))
    else:
        root = IOtools.expandPath(stage) / Path(db.file).stem
    hs = [harvester(r, root / f"{r.name}_{run_id_of(r)}", scan_trick_members=scan_trick_members) for r in runs]
    print(f"- mitim_harvester: {len(hs)} run(s) -> {IOtools.clipstr(db.file)}{' (dry run)' if dry_run else ''}", typeMsg='i')
    known = known_hashes(db, {h.run_id for h in hs})

    totals, pending = {c: 0 for c in RECOVERED_CODES}, []
    for i, h in enumerate(hs):
        counts = h.stage(known.get(h.run_id, ()))
        print(f"\t{h.folder}: run {h.run_id}, new records " + ", ".join(f"{c} {n}" for c, n in counts.items())
              + (f", {len(h.failed)} unreadable folder(s)" if h.failed else ''))
        for msg in h.failed:
            print(f"\t\t- could not read {msg}", typeMsg='w')
        if dry_run:
            totals = {c: totals[c] + counts[c] for c in totals}
            continue
        pending.append(h.staging_folder)
        if len(pending) >= batch or i == len(hs) - 1:
            appended = db.push(pending)
            totals = {c: totals[c] + appended.get(c, 0) for c in totals}
            pending = []

    if stage is None or dry_run:
        IOtools.shutil_rmtree(root)
    print(f"- mitim_harvester: {'would append' if dry_run else 'appended'} " + ", ".join(f"{c} {n}" for c, n in totals.items()), typeMsg='i')
    return totals
