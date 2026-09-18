'''
Harvest: archive every individual transport-code evaluation (TGLF, NEO, CGYRO, GX, QuaLiKiz) and
every full-EPED evaluation of a PORTALS or MAESTRO run, so the data can be reused off-line
(surrogate training, physics studies) instead of dying with the run folder.

A record is exactly an input -> output map of one code evaluation at one radius: the FULL input
file (`in_<KEY>`), the scalar fluxes the code returned (`out_<name>`, in the code's own units: GB for
TGLF/NEO/CGYRO/GX, SI for QuaLiKiz), plus two short keys: `run` (which run produced it) and `hash`
(of the inputs, for deduplication). Who produced it (code version, machine, modules, MITIM commit,
user, host, run folder) is stored ONCE per run and code in the `runs` table and joined on load.
TGLF records are individual runs: the base point AND each perturbed member of the std scan trick
(`scan_trick_members: false` drops the latter, which are most of the volume).

Two tiers:
    1. Staging, per run (opt-in via the `harvest:` namelist block): JSON-lines, one file per code,
       `<run>/Outputs/harvest/<code>.jsonl` (+ run_meta.json with the provenance). Append-only and
       crash-safe, so a dead or preempted run keeps its records and can be pushed by hand
       (`mitim_harvest <folder>`). Plain JSON never accumulates: once the tail exceeds ROLL_BYTES it
       is compressed as one more gzip member of `<code>.jsonl.gz` (~20-30x smaller, since only a
       handful of inputs change between records), so a run holds a few MB at most. Pushed archives
       are renamed `<code>.jsonl.pushed-<ts>.gz` and kept, so the central file can be rebuilt.
    2. Central store, per user: one netCDF-4 file, one group per code plus `runs`, unlimited `record`
       dimension, appended in place under an NFS-safe mkdir lock (IOtools.mkdir_lock). A variable
       that later records introduce reads back as NaN for the earlier ones.

Objects:
    harvest_recorder : attached to a simulation object as `sim.harvest`; `record(sim, label)`
                       asks `sim.harvest_records(label)` (polymorphic, base default in SIMtools) and
                       stages the records. Picklable/deep-copy safe (two plain dicts; the dedup set
                       lives in a module registry keyed by staging folder).
    harvest_database : the central file: push staging folders into it, load (selectively), runs
                       table, summary, interpret, plot (FigureNotebook) and rebuild.
'''

import os
import json
import gzip
import uuid
import socket
import getpass
import hashlib
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from mitim_tools import __version__ as mitim_version, __mitimroot__
from mitim_tools.misc_tools import IOtools, CONFIGread, GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print

DEFAULT_FILE = "~/mitim_harvest/mitim_harvest.nc"
SCHEMA_VERSION = 3
ROLL_BYTES = 256 * 1024   # plain-JSON tail size that triggers compression into <code>.jsonl.gz (~60 TGLF records)
CODES = ('tglf', 'neo', 'cgyro', 'gx', 'qualikiz', 'eped')
RUNS_GROUP = 'runs'

RECORD_KEYS = ['run', 'hash']
# Provenance, once per run (run_meta.json) ...
RUN_KEYS = ['run', 'run_folder', 'user', 'host', 'mitim_version', 'git_branch', 'git_commit', 'created', 'maestro_beat']
# ... and once per (run, code), captured from the first record of that code (`averaging`: how the
# time-averaged fluxes and their std were computed, for CGYRO/GX; empty for single-value codes)
RUN_CODE_KEYS = ['machine', 'modules', 'code_version', 'in_process', 'averaging']

# ------------------------------------------------------------------------------------------------
# Options / central file resolution
# ------------------------------------------------------------------------------------------------

def resolve_central_file(file=None):
    '''namelist `file` -> config_user.json preferences.harvest_file -> DEFAULT_FILE'''
    file = file or CONFIGread.read_harvest_file() or DEFAULT_FILE
    file = IOtools.expandPath(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    return file

def options_from_namelist(block, staging_folder, run_meta_extra=None):
    '''
    Turn the `harvest:` namelist block into the plain, JSON-serializable options dict that travels
    inside transport_options / the maestro instance (no Path objects, no custom classes: it gets
    deep-copied and dill-pickled with the powerstate). When enabled, creates the staging folder and
    its run_meta.json (re-read if present, so a resumed run keeps its id).
    Extra keys a driver may inject into the block: `push` (False for MAESTRO beats), `run_id`,
    `maestro_beat`.
    '''
    block = dict(block or {})
    opts = {
        'enabled': bool(block.get('enabled', False)),
        'file': block.get('file', None),
        'push': bool(block.get('push', True)),
        'scan_trick_members': bool(block.get('scan_trick_members', True)),
        'folder': str(staging_folder),
        'run_meta': {},
    }
    if not opts['enabled']:
        return opts

    folder = Path(staging_folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta_file = folder / 'run_meta.json'

    if meta_file.exists():
        run_meta = json.loads(meta_file.read_text())
    else:
        branch, commit = IOtools.get_git_info(__mitimroot__)
        run_meta = {
            'run': block.get('run_id') or uuid.uuid4().hex[:12],
            'run_folder': '',
            'maestro_beat': -1,
            'mitim_version': str(mitim_version),
            'git_branch': str(branch or ''),
            'git_commit': str(commit or ''),
            'user': getpass.getuser(),
            'host': socket.gethostname(),
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'codes': {},
        }
    if block.get('run_id'):
        run_meta['run'] = block['run_id']
    if 'maestro_beat' in block:
        run_meta['maestro_beat'] = int(block['maestro_beat'])
    run_meta.update(run_meta_extra or {})
    run_meta['run_folder'] = str(run_meta.get('run_folder', ''))
    run_meta.setdefault('codes', {})

    meta_file.write_text(json.dumps(run_meta, indent=2))
    opts['run_meta'] = run_meta
    return opts

# ------------------------------------------------------------------------------------------------
# Scalar helpers
# ------------------------------------------------------------------------------------------------

def _json_default(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return str(v)

def _to_scalar(v, default=np.nan):
    '''float/int/bool/str/None from numpy scalars, 0-d arrays, xarray DataArrays; default otherwise'''
    if v is None:
        return default
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    if hasattr(v, 'values'):
        v = v.values
    if isinstance(v, np.ndarray):
        if v.size != 1:
            return default
        v = v.reshape(-1)[0]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (int, float)):
        return v
    return default

def _scalar_dict(d):
    '''Keep only JSON-friendly scalars (str/int/float/bool); short 1-D arrays (<=16, e.g. EPED NMODES) are expanded as KEY_i'''
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, str):
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.size <= 16:
                for i, vi in enumerate(arr):
                    s = _to_scalar(vi, default=None)
                    if s is not None:
                        out[f"{k}_{i}"] = s
            continue
        s = _to_scalar(v, default=None)
        if s is not None:
            out[str(k)] = s
    return out

def input_hash(code, inputs, extra=None):
    payload = json.dumps({'code': code, 'inputs': inputs, 'extra': extra}, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]

def machine_info(job):
    '''machine name + modules string of a mitim_job (None-safe)'''
    ms = getattr(job, 'machineSettings', None) or {}
    return {'machine': str(ms.get('machine', '') or ''), 'modules': str(ms.get('modules', '') or '')}

# ------------------------------------------------------------------------------------------------
# Staged files (plain while the run is alive, gzipped once pushed)
# ------------------------------------------------------------------------------------------------

def _open_text(file):
    return gzip.open(file, 'rt') if str(file).endswith('.gz') else open(file, 'r')

def _read_jsonl(file):
    '''
    Records of a plain or (multi-member) gzip staging file. A member truncated by a kill mid-roll
    ends the read at the last complete line instead of raising.
    '''
    rows = []
    try:
        with _open_text(file) as fi:
            for line in fi:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
    except (EOFError, gzip.BadGzipFile, OSError):
        pass
    return rows

def _roll(plain):
    '''
    Compress the plain <code>.jsonl tail as one more gzip member appended to <code>.jsonl.gz, then
    truncate the tail. Member first, truncate second: a kill in between duplicates lines, which the
    hash dedup absorbs; the reverse order could lose them.
    '''
    plain = Path(plain)
    if not plain.exists() or plain.stat().st_size == 0:
        return
    with open(plain, 'rb') as fi:
        data = fi.read()
    with gzip.open(plain.with_name(plain.name + '.gz'), 'ab') as fo:
        fo.write(data)
    with open(plain, 'wb'):
        pass

def _is_staged(file):
    return '.jsonl' in file.name and not file.name.startswith('run_meta')

def _code_of(file):
    return file.name.split('.jsonl')[0]

def _is_pushed(file):
    return '.pushed-' in file.name

# ------------------------------------------------------------------------------------------------
# Recorder (attached to simulation objects)
# ------------------------------------------------------------------------------------------------

_SEEN = {}   # staging folder -> set of input hashes already staged (this process)

def _seen(folder):
    key = str(folder)
    if key not in _SEEN:
        seen = set()
        for f in Path(folder).glob('*.jsonl*'):
            try:
                for row in _read_jsonl(f):
                    if row.get('hash'):
                        seen.add(row['hash'])
            except Exception:
                pass
        _SEEN[key] = seen
    return _SEEN[key]

class harvest_recorder:

    def __init__(self, options, context=None):
        self.options = dict(options or {})
        self.context = dict(context or {})

    def __getstate__(self):
        return {'options': self.options, 'context': self.context}

    def __setstate__(self, state):
        self.options, self.context = state['options'], state['context']

    @property
    def enabled(self):
        return bool(self.options.get('enabled', False))

    @property
    def folder(self):
        return Path(self.options['folder'])

    def with_context(self, **context):
        '''Context is NOT stored in records; it only steers the recorder (e.g. scan_member=1 -> skippable)'''
        return harvest_recorder(self.options, {**self.context, **context})

    def record(self, obj, label=None, folder=None):
        '''
        Stage the records `obj.harvest_records(label, folder)` describes. Never raises: a harvesting
        failure must not kill the simulation that produced the data.
        '''
        if not self.enabled:
            return 0
        if self.context.get('scan_member', 0) and not self.options.get('scan_trick_members', True):
            return 0
        try:
            records = obj.harvest_records(label, folder=folder)
        except Exception as e:
            print(f"\t- harvest: could not extract records for label '{label}' ({type(e).__name__}: {e})", typeMsg='w')
            return 0
        n = sum(self._write(rec) for rec in records)
        if n > 0:
            print(f"\t- harvest: staged {n} new record(s) of '{records[0]['code']}' (label {label}) in {IOtools.clipstr(self.folder)}", typeMsg='i')
        return n

    def record_eped(self, **kwargs):
        if not self.enabled:
            return 0
        try:
            rec = collect_eped(**kwargs)
        except Exception as e:
            print(f"\t- harvest: could not extract the EPED record ({type(e).__name__}: {e})", typeMsg='w')
            return 0
        return self._write(rec)

    def _write(self, rec):
        '''Flatten one {code, inputs, outputs, meta, hash_extra} record and append it to <code>.jsonl'''
        try:
            code = rec['code']
            inputs = _scalar_dict(rec.get('inputs'))
            outputs = _scalar_dict(rec.get('outputs'))
            h = input_hash(code, inputs, rec.get('hash_extra'))

            seen = _seen(self.folder)
            if h in seen:
                return 0

            self.folder.mkdir(parents=True, exist_ok=True)
            self._note_provenance(code, rec.get('meta', {}))

            flat = {'run': self.options.get('run_meta', {}).get('run', ''), 'hash': h}
            flat.update({f"in_{k}": v for k, v in inputs.items()})
            flat.update({f"out_{k}": v for k, v in outputs.items()})

            plain = self.folder / f"{code}.jsonl"
            with open(plain, 'a') as fo:
                fo.write(json.dumps(flat, default=_json_default) + '\n')
                size = fo.tell()
            seen.add(h)
            if size >= ROLL_BYTES:
                _roll(plain)
            return 1
        except Exception as e:
            print(f"\t- harvest: could not stage a record ({type(e).__name__}: {e})", typeMsg='w')
            return 0

    def _note_provenance(self, code, meta):
        '''Machine / modules / code version of this code, written once into run_meta.json (first record wins)'''
        meta_file = self.folder / 'run_meta.json'
        run_meta = json.loads(meta_file.read_text()) if meta_file.exists() else dict(self.options.get('run_meta', {}))
        codes = run_meta.setdefault('codes', {})
        if code not in codes:
            codes[code] = {k: (int(meta[k]) if k == 'in_process' else str(meta.get(k, ''))) if k in meta else ('' if k != 'in_process' else 0) for k in RUN_CODE_KEYS}
            meta_file.write_text(json.dumps(run_meta, indent=2))

# ------------------------------------------------------------------------------------------------
# EPED collector (EPED is not a SIMtools simulation object)
# ------------------------------------------------------------------------------------------------

_EPED_DS_OUTPUTS = ['ptop', 'wptop', 'pped', 'tped', 'ttop', 'wpped', 'wrped',
                    'stability_index', 'n_limiting', 'dome_frac']
# Driver settings of eped.config that define the scan (plus any key the run overrode)
_EPED_CFG_KEYS = ['NMODES', 'WIDTHS', 'TEPED_BOUND']

def _eped_config_values(config_file, keys):
    '''`KEY = v1 v2 ...` lines of an (effective, per-case) eped.config for the requested keys; first occurrence wins'''
    found = {}
    if config_file is None or not Path(config_file).exists():
        return found
    for line in Path(config_file).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#') or '=' not in s:
            continue
        key, val = (x.strip() for x in s.split('=', 1))
        if key in keys and key not in found:
            vals = []
            for tok in val.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    vals.append(tok)
            found[key] = vals[0] if len(vals) == 1 else vals
    return found

def collect_eped(input_params, composition=None, eped_params_override=None, toq_eq_choice='',
                 dataset=None, ptop_kPa=np.nan, wtop_psipol=np.nan, limiting_mode_info=None,
                 eped_folder=None, job=None, eped_input_file=None, eped_config_file=None):
    '''
    One full-EPED evaluation as inputs -> outputs.
    Inputs: the complete eped.input namelist as run (ip, bt, ..., zeffped, nesep, tesep, m, z, mi, zi,
    the teped guess, tewid/ptotwid scan widths, ...; falls back to `input_params` + `composition` when
    the file is not available), the effective driver settings of eped.config (NMODES, WIDTHS,
    TEPED_BOUND, plus every key the run overrode) as `cfg_<KEY>`, the TOQ equilibrium choice, and the
    MITIM stability rule that picked the pedestal from the ELITE spectrum (`stability_rule` 'G' = flat
    gamma/omega_A cut or 'W' = EPED1 diamagnetic criterion, `stability_threshold`).
    Outputs: ptop, wptop, pped, tped, ttop, wpped, wrped, stability_index, n_limiting, dome_frac,
    limiting_mode.
    '''
    inputs = {}
    if eped_input_file is not None and Path(eped_input_file).exists():
        try:
            import f90nml
            nml = f90nml.read(str(eped_input_file))
            inputs.update({k: v for k, v in dict(nml.get('eped_input', {})).items()})
        except Exception as e:
            print(f"\t- harvest: could not parse {IOtools.clipstr(eped_input_file)} ({e}); using the input_params dict", typeMsg='w')
    if not inputs:
        inputs.update(input_params or {})
        inputs.update(composition or {})

    inputs['toq_eq_choice'] = str(toq_eq_choice)
    cfg_keys = set(_EPED_CFG_KEYS) | set((eped_params_override or {}).keys())
    cfg = _eped_config_values(eped_config_file, cfg_keys)
    for k in cfg_keys:
        v = cfg.get(k, (eped_params_override or {}).get(k))
        if v is not None:
            inputs[f"cfg_{k}"] = v
    if dataset is not None:
        if 'stability_rule' in dataset:
            inputs['stability_rule'] = str(np.asarray(dataset['stability_rule']).reshape(-1)[0])
        if 'stability_threshold' in dataset:
            inputs['stability_threshold'] = _to_scalar(dataset['stability_threshold'])

    outputs = {'ptop_kPa': ptop_kPa, 'wtop_psipol': wtop_psipol}
    if dataset is not None:
        for name in _EPED_DS_OUTPUTS:
            if name in dataset:
                outputs[name] = _to_scalar(dataset[name])
    lm = limiting_mode_info or {}
    outputs['limiting_mode'] = str(lm.get('limiting_mode', '') or '')

    attrs = dict(getattr(dataset, 'attrs', {}) or {})
    code_version = "; ".join(f"{k}={v}" for k, v in attrs.items() if any(s in str(k).lower() for s in ('version', 'git', 'hash', 'commit')))

    meta = {'code_version': code_version, 'in_process': False, **machine_info(job)}
    return {'code': 'eped', 'inputs': inputs, 'outputs': outputs, 'meta': meta}

# ------------------------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------------------------

_STRING_COLS = {'run', 'hash', 'code', 'run_folder', 'user', 'host', 'mitim_version', 'git_branch', 'git_commit',
                'created', 'machine', 'modules', 'code_version', 'averaging'}

def _frame_from_rows(rows):
    '''DataFrame with the union of keys; a column is string if any value is a string, numeric (f8) otherwise'''
    df = pd.DataFrame(rows)
    for col in df.columns:
        s = df[col]
        is_str = (col in _STRING_COLS) or s.map(lambda v: isinstance(v, str)).any()
        if is_str:
            df[col] = s.map(lambda v: '' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)).astype(object)
        else:
            df[col] = pd.to_numeric(s.map(lambda v: (float(v) if isinstance(v, (bool, np.bool_)) else v)), errors='coerce').astype('float64')
    return df

class harvest_database:

    def __init__(self, file=None):
        self.file = resolve_central_file(file)

    def __repr__(self):
        return f"harvest_database({self.file})"

    # -------------------------------------------------------------------------- reading
    def codes(self):
        import netCDF4
        if not self.file.exists():
            return []
        with netCDF4.Dataset(self.file, 'r') as ds:
            present = [g for g in ds.groups if g != RUNS_GROUP and len(ds.groups[g].dimensions.get('record', [])) > 0]
        return [c for c in CODES if c in present] + [c for c in present if c not in CODES]

    def _read_group(self, code, columns=None, mask_fn=None):
        import netCDF4
        if not self.file.exists():
            return pd.DataFrame()
        with netCDF4.Dataset(self.file, 'r') as ds:
            if code not in ds.groups:
                return pd.DataFrame()
            grp = ds.groups[code]
            grp.set_auto_mask(False)

            def read(name):
                var = grp.variables[name]
                vals = var[:]
                if var.dtype == str:
                    return np.array(['' if v is None else str(v) for v in vals], dtype=object)
                return np.asarray(vals, dtype='float64')

            mask = mask_fn(read) if mask_fn is not None else None
            names = [n for n in grp.variables if columns is None or n in columns]
            data = {n: (read(n) if mask is None else read(n)[mask]) for n in names}
        return pd.DataFrame(data)

    def runs(self):
        '''One row per (run, code): folder, user, host, MITIM version/commit, machine, modules, code version'''
        return self._read_group(RUNS_GROUP)

    def load(self, code, columns=None, run=None, with_run_info=True):
        '''
        Records of one code as a pandas DataFrame (values missing for a record are NaN).
        columns: subset of variable names to read (`run` and `hash` are always included);
        run: one run id or a list of ids to keep (filtered while reading, cheap on a big file);
        with_run_info: join the provenance from the `runs` group (machine, code_version, ...).
        '''
        if columns is not None:
            columns = set(columns) | set(RECORD_KEYS)
        mask_fn = None
        if run is not None:
            ids = {run} if isinstance(run, str) else set(run)
            mask_fn = lambda read: np.array([r in ids for r in read('run')], dtype=bool)
        df = self._read_group(code, columns=columns, mask_fn=mask_fn)
        if with_run_info and len(df) and 'run' in df:
            runs = self.runs()
            if len(runs):
                runs = runs[runs['code'] == code].drop(columns=['code'])
                df = df.merge(runs, on='run', how='left')
                for c in RUN_KEYS + RUN_CODE_KEYS:
                    if c in df and not pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].fillna('')
        return df

    # -------------------------------------------------------------------------- writing
    def push(self, staging_folders, timeout_s=600, stale_s=3600):
        '''
        Append every unpushed staging file (<code>.jsonl tail + rolled <code>.jsonl.gz) of the given
        folders to the central file, under the lock, then archive them as <code>.jsonl.pushed-<ts>.gz.
        Records are deduplicated by (code, run, hash) within the call (MAESTRO may hand the
        run_portals/ and beat_results/ twins). Returns {code: records_appended}.
        '''
        files = [f for folder in [Path(f) for f in staging_folders] if folder.is_dir()
                 for f in sorted(folder.glob('*.jsonl*')) if _is_staged(f) and not _is_pushed(f)]
        return self._push_files(files, timeout_s=timeout_s, stale_s=stale_s, archive=True)

    def _push_files(self, files, timeout_s=600, stale_s=3600, archive=True):
        import netCDF4

        # Fold every plain tail into its rolled archive first, so each (folder, code) is one .gz.
        # A peek (archive=False) reads the files as they are and touches nothing.
        if archive:
            rolled = []
            for f in files:
                if f.suffix == '.jsonl':
                    _roll(f)
                    f.unlink(missing_ok=True)
                    f = f.with_name(f.name + '.gz')
                if f.exists() and f not in rolled:
                    rolled.append(f)
            files = rolled

        frames, runs_rows, seen = {}, {}, set()
        for f in files:
            code = _code_of(f)
            meta_file = f.parent / 'run_meta.json'
            run_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            rows = []
            for row in _read_jsonl(f):
                key = (code, row.get('run', ''), row.get('hash', ''))
                if key in seen:
                    continue
                seen.add(key)
                rkey = (row.get('run', ''), code)
                if rkey not in runs_rows:
                    per_code = run_meta.get('codes', {}).get(code, {})
                    runs_rows[rkey] = {'run': rkey[0], 'code': code,
                                       **{k: run_meta.get(k, '') for k in RUN_KEYS if k != 'run'},
                                       **{k: per_code.get(k, 0 if k == 'in_process' else '') for k in RUN_CODE_KEYS}}
                rows.append(row)
            if rows:
                frames.setdefault(code, []).extend(rows)

        appended = {}
        if frames:
            with IOtools.mkdir_lock(self.file, timeout_s=timeout_s, stale_s=stale_s):
                mode = 'a' if self.file.exists() else 'w'
                with netCDF4.Dataset(self.file, mode, format='NETCDF4') as ds:
                    for code, rows in frames.items():
                        appended[code] = self._append_group(ds, code, _frame_from_rows(rows))
                    self._upsert_runs(ds, list(runs_rows.values()))
            print(f"\t- harvest: appended {sum(appended.values())} record(s) to {IOtools.clipstr(self.file)} ({', '.join(f'{k}: {v}' for k, v in appended.items())})", typeMsg='i')

        if archive:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            for f in files:
                if not _is_pushed(f):
                    os.replace(f, f.with_name(f"{_code_of(f)}.jsonl.pushed-{ts}.gz"))

        return appended

    @staticmethod
    def _append_group(ds, code, df):
        grp = ds.groups[code] if code in ds.groups else ds.createGroup(code)
        if 'record' not in grp.dimensions:
            grp.createDimension('record', None)
            grp.setncattr('schema_version', SCHEMA_VERSION)
        n, k = len(grp.dimensions['record']), len(df)
        for col in df.columns:
            is_str = not pd.api.types.is_numeric_dtype(df[col])
            if col not in grp.variables:
                if is_str:
                    grp.createVariable(col, str, ('record',))
                else:
                    grp.createVariable(col, 'f8', ('record',), fill_value=np.nan, zlib=True, chunksizes=(4096,))
            var = grp.variables[col]
            if is_str:
                var[n:n + k] = np.array([str(v) for v in df[col].to_numpy()], dtype=object)
            else:
                var[n:n + k] = df[col].to_numpy(dtype='float64')
        grp.setncattr('last_push', datetime.datetime.now().isoformat(timespec='seconds'))
        return k

    def _upsert_runs(self, ds, rows):
        '''Append the (run, code) provenance rows not already present in the `runs` group'''
        if not rows:
            return 0
        grp = ds.groups[RUNS_GROUP] if RUNS_GROUP in ds.groups else ds.createGroup(RUNS_GROUP)
        existing = set()
        if 'record' in grp.dimensions and len(grp.dimensions['record']):
            grp.set_auto_mask(False)
            existing = set(zip([str(v) for v in grp.variables['run'][:]], [str(v) for v in grp.variables['code'][:]]))
        new = [r for r in rows if (r['run'], r['code']) not in existing]
        if new:
            self._append_group(ds, RUNS_GROUP, _frame_from_rows(new))
        return len(new)

    @classmethod
    def from_staging(cls, folders, file=None):
        '''
        Peek at staged records WITHOUT pushing them: build a (temporary, or `file`) database from the
        staging files of `folders` (run folders or harvest/ folders, plain tails, rolled and already
        pushed archives alike), leaving the staging untouched. For a copy of a live run's
        Outputs/harvest/ pulled with mitim_scp, or for a look at a run before it finishes.
        '''
        import tempfile
        staging = []
        for f in folders:
            staging += staging_folders_of(f)
        files = sorted({p for s in staging for p in s.glob('*.jsonl*') if _is_staged(p)})
        if file is None:
            file = Path(tempfile.mkdtemp(prefix='mitim_harvest_peek_')) / 'harvest_peek.nc'
        db = cls(file)
        if db.file.exists():
            db.file.unlink()
        db._push_files(files, archive=False)
        print(f"\t- harvest: peek database {IOtools.clipstr(db.file)} built from {len(files)} staged file(s); staging left untouched", typeMsg='i')
        return db

    def rebuild(self, roots):
        '''Move the current file aside and re-push every staged file (pushed or not) found under roots'''
        if self.file.exists():
            aside = self.file.with_name(f"{self.file.name}.corrupt-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.replace(self.file, aside)
            print(f"\t- harvest: previous file moved to {IOtools.clipstr(aside)}", typeMsg='w')
        files = sorted({f for root in [Path(r) for r in roots] for f in root.rglob('*.jsonl*')
                        if f.parent.name == 'harvest' and _is_staged(f)})
        return self._push_files(files, archive=True)

    # -------------------------------------------------------------------------- interpreting
    def summary(self):
        runs = self.runs()
        rows = []
        for code in self.codes():
            df = self.load(code, with_run_info=False)
            r = runs[runs['code'] == code] if len(runs) else pd.DataFrame()
            ts = pd.to_datetime(r['created'], errors='coerce') if 'created' in r else pd.Series(dtype='datetime64[ns]')
            rows.append({
                'code': code, 'records': len(df),
                'runs': df['run'].nunique() if 'run' in df else 0,
                'machines': ', '.join(sorted({m for m in r.get('machine', pd.Series(dtype=object)) if m})),
                'code_versions': r['code_version'].map(lambda s: s.split('\n')[0]).nunique() if 'code_version' in r else 0,
                'first_run': ts.min(), 'last_run': ts.max(),
            })
        return pd.DataFrame(rows)

    def interpret(self, code=None, max_rows=40):
        '''Printed text report of what the database holds (all codes, or one)'''
        lines = [f"Harvest database: {self.file}"]
        summ = self.summary()
        lines.append(summ.to_string(index=False) if len(summ) else "  (empty)")
        for c in ([code] if code else self.codes()):
            df = self.load(c)
            if len(df) == 0:
                continue
            lines.append(f"\n===== {c}: {len(df)} records =====")
            if 'run' in df:
                top = df.groupby('run').size().sort_values(ascending=False).head(5)
                lines.append("Top runs by records:\n" + "\n".join(f"  {rid}  {n}   {df.loc[df.run == rid, 'run_folder'].iloc[0] if 'run_folder' in df else ''}" for rid, n in top.items()))
            if 'hash' in df:
                lines.append(f"Distinct inputs: {df['hash'].nunique()} (duplicates across runs: {len(df) - df['hash'].nunique()})")
            if 'code_version' in df:
                vers = df['code_version'].map(lambda s: s.split('\n')[0] if s else '(unknown)').value_counts()
                lines.append("Code versions:\n" + "\n".join(f"  {n:6d}  {v}" for v, n in vers.items()))
            in_cols = [k for k in df.columns if k.startswith('in_') and pd.api.types.is_numeric_dtype(df[k])]
            out_cols = [k for k in df.columns if k.startswith('out_') and pd.api.types.is_numeric_dtype(df[k])]
            if in_cols:
                rng = df[in_cols].agg(['min', 'max', 'mean']).T
                rng = rng[(rng['max'] - rng['min']).abs() > 0].sort_values('max')
                lines.append(f"Inputs that vary ({len(rng)} of {len(in_cols)}):\n" + rng.head(max_rows).to_string(float_format=lambda x: f"{x:.4g}"))
            if out_cols:
                lines.append("Fluxes:\n" + df[out_cols].describe().T[['mean', 'std', 'min', 'max']].to_string(float_format=lambda x: f"{x:.4g}"))
                zero = (df[out_cols].fillna(0).abs().sum(axis=1) == 0).mean()
                lines.append(f"Records with all-zero fluxes: {100*zero:.1f}%")
        report = "\n".join(lines)
        print(report)
        return report

    # -------------------------------------------------------------------------- plotting
    _DRIVES = {   # candidates for the "main drive" of each channel, first present wins
        'tglf':     {'Te': ['RLTS_1'], 'Ti': ['RLTS_2'], 'ne': ['RLNS_1']},
        'neo':      {'Te': ['DLNTDR_1'], 'Ti': ['DLNTDR_2'], 'ne': ['DLNNDR_1']},
        'cgyro':    {'Te': ['DLNTDR_1', 'dlntdr_0'], 'Ti': ['DLNTDR_2', 'dlntdr_1'], 'ne': ['DLNNDR_1', 'dlnndr_0']},
        'gx':       {'Te': ['tprim_1', 'tprim_0'], 'Ti': ['tprim_2', 'tprim_1'], 'ne': ['fprim_1', 'fprim_0']},
        'qualikiz': {'Te': ['Ate'], 'Ti': ['Ati_0', 'Ati'], 'ne': ['Ane']},
    }
    _FLUXES = {'Qe': ['Qe', 'Qe_mean', 'efe_SI'], 'Qi': ['Qi', 'Qi_mean', 'efi_SI_0'], 'Ge': ['Ge', 'Ge_mean', 'pfe_SI']}

    @staticmethod
    def _first(df, prefix, candidates):
        for c in candidates:
            if f"{prefix}{c}" in df.columns:
                return f"{prefix}{c}"
        return None

    @staticmethod
    def _cgyro_drives(df):
        '''CGYRO species are 0-indexed in params1D with no fixed order: find the electron (z=-1) and first main ion (z=1) by charge'''
        z = {i: df[f'in_z_{i}'].dropna().iloc[0] for i in range(12) if f'in_z_{i}' in df.columns and df[f'in_z_{i}'].notna().any()}
        ie = next((i for i, v in z.items() if v == -1), None)
        ii = next((i for i, v in z.items() if v == 1), None)
        drives = {}
        if ie is not None:
            drives['Te'], drives['ne'] = [f'dlntdr_{ie}'], [f'dlnndr_{ie}']
        if ii is not None:
            drives['Ti'] = [f'dlntdr_{ii}']
        return drives

    def _color_by(self, df, key='run'):
        cols = GRAPHICStools.listColors()
        keys = list(dict.fromkeys(df[key])) if key in df else ['']
        return {k: cols[i % len(cols)] for i, k in enumerate(keys)}

    def plot(self, code, x, y, color_by='run', ax=None, **kw):
        '''
        Scatter of column y vs column x of one code (names with or without the in_/out_ prefix).
        Pass an `ax` from a FigureNotebook tab to keep it inside the notebook (a bare plt.show() next
        to a notebook re-opens every tab as a window).
        '''
        import matplotlib.pyplot as plt
        df = self.load(code)
        x = x if x in df.columns else (f"in_{x}" if f"in_{x}" in df.columns else f"out_{x}")
        y = y if y in df.columns else (f"out_{y}" if f"out_{y}" in df.columns else f"in_{y}")
        if ax is None:
            _, ax = plt.subplots()
        colors = self._color_by(df, color_by)
        for k, c in colors.items():
            sub = df[df[color_by] == k] if color_by in df else df
            ax.scatter(sub[x], sub[y], s=8, color=c, alpha=0.6, label=str(k)[:12], **kw)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(code)
        GRAPHICStools.addDenseAxis(ax)
        if len(colors) <= 12:
            ax.legend(fontsize=6, loc='best')
        return ax

    def plotDatabase(self, fn=None, codes=None):
        '''FigureNotebook: an Overview tab plus one tab per code present in the file'''
        from mitim_tools.misc_tools.GUItools import FigureNotebook
        if fn is None:
            fn = FigureNotebook("MITIM harvest", geometry="1700x900")
        self.fn = fn
        codes = codes or self.codes()
        self._plot_overview(fn, codes)
        for code in codes:
            if code == 'eped':
                self._plot_eped(fn)
            else:
                self._plot_transport(fn, code)
        return fn

    def _plot_overview(self, fn, codes):
        fig = fn.add_figure(label='Overview')
        axs = fig.subplots(2, 2)
        frames = {c: self.load(c) for c in codes}
        ax = axs[0, 0]
        ax.bar(list(frames), [len(d) for d in frames.values()])
        ax.set_ylabel('records'); ax.set_title('Records per code')
        ax = axs[0, 1]
        for c, df in frames.items():
            if 'created' in df and len(df):
                per_run = df.groupby('run').agg(n=('hash', 'size'), t=('created', 'first'))
                per_run['t'] = pd.to_datetime(per_run['t'], errors='coerce')
                per_run = per_run.sort_values('t')
                ax.step(per_run['t'], per_run['n'].cumsum(), where='post', label=c)
        ax.set_ylabel('cumulative records'); ax.set_title('Records vs run start time'); ax.legend(fontsize=7)
        ax.tick_params(axis='x', labelrotation=30)
        ax = axs[1, 0]
        runs = pd.concat([d[['run']].assign(code=c) for c, d in frames.items() if 'run' in d], ignore_index=True) if frames else pd.DataFrame()
        if len(runs):
            top = runs.groupby('run').size().sort_values(ascending=False).head(15)
            ax.barh(list(top.index), top.values)
            ax.invert_yaxis()
        ax.set_xlabel('records'); ax.set_title('Records per run (top 15)')
        ax = axs[1, 1]
        prov = pd.concat([d[['machine', 'code_version']].assign(code=c) for c, d in frames.items() if 'machine' in d], ignore_index=True) if frames else pd.DataFrame()
        if len(prov):
            prov['key'] = prov['code'] + ' @ ' + prov['machine'].replace('', '?') + ' / ' + prov['code_version'].map(lambda s: s.split('\n')[0][:20] if s else '?')
            cnt = prov.groupby('key').size().sort_values(ascending=False).head(12)
            ax.barh(cnt.index, cnt.values, color='gray')
            ax.invert_yaxis()
            ax.tick_params(axis='y', labelsize=6)
        ax.set_xlabel('records'); ax.set_title('Machine / code version')
        for a in axs.flatten():
            GRAPHICStools.addDenseAxis(a)
        GRAPHICStools.adjust_figure_layout(fig)

    def _plot_transport(self, fn, code):
        df = self.load(code)
        if len(df) == 0:
            return
        fig = fn.add_figure(label=code.upper())
        axs = fig.subplots(2, 3)
        colors = self._color_by(df)
        fluxes = {k: self._first(df, 'out_', v) for k, v in self._FLUXES.items()}
        candidates = {**self._DRIVES.get(code, {}), **(self._cgyro_drives(df) if code == 'cgyro' else {})}
        drives = {k: self._first(df, 'in_', v) for k, v in candidates.items()}
        for ax, (name, col) in zip(axs[0, :], fluxes.items()):
            if col is not None:
                vals = df[col].dropna()
                ax.hist(vals, bins=40, color='gray')
                ax.set_xlabel(col)
            ax.set_ylabel('records'); ax.set_title(f'{name} distribution')
        pairs = [(drives.get('Ti'), fluxes['Qi']), (drives.get('Te'), fluxes['Qe']), (drives.get('ne'), fluxes['Ge'])]
        for ax, (xcol, ycol) in zip(axs[1, :], pairs):
            if xcol is None or ycol is None:
                ax.text(0.5, 0.5, 'no drive/flux column found', ha='center', va='center', transform=ax.transAxes)
                continue
            for k, c in colors.items():
                sub = df[df['run'] == k]
                ax.scatter(sub[xcol], sub[ycol], s=6, color=c, alpha=0.5)
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        axs[1, 0].set_title(f'{len(colors)} runs (colors)')
        for a in axs.flatten():
            GRAPHICStools.addDenseAxis(a)
        GRAPHICStools.adjust_figure_layout(fig)

    def _plot_eped(self, fn):
        df = self.load('eped')
        if len(df) == 0:
            return
        fig = fn.add_figure(label='EPED')
        axs = fig.subplots(2, 2)
        colors = self._color_by(df)
        pairs = [('in_neped', 'out_ptop_kPa'), ('in_betan', 'out_ptop_kPa'), ('out_ptop_kPa', 'out_wtop_psipol')]
        for ax, (xcol, ycol) in zip(axs.flatten()[:3], pairs):
            if xcol in df and ycol in df:
                for k, c in colors.items():
                    sub = df[df['run'] == k]
                    ax.scatter(sub[xcol], sub[ycol], s=12, color=c, alpha=0.7)
                ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        ax = axs[1, 1]
        if 'out_limiting_mode' in df:
            cnt = df['out_limiting_mode'].replace('', '(none)').value_counts()
            ax.bar(cnt.index, cnt.values, color='gray')
        ax.set_ylabel('records'); ax.set_title('Limiting mode')
        for a in axs.flatten():
            GRAPHICStools.addDenseAxis(a)
        GRAPHICStools.adjust_figure_layout(fig)

# ------------------------------------------------------------------------------------------------
# CLIs
# ------------------------------------------------------------------------------------------------

def staging_folders_of(run_folder):
    '''Staging dirs of a PORTALS run, a MAESTRO run (all its beats), or a staging dir itself'''
    run_folder = Path(run_folder)
    if run_folder.name == 'harvest' and run_folder.is_dir():
        return [run_folder]
    folders = [run_folder / 'Outputs' / 'harvest']
    if (run_folder / 'Beats').is_dir():
        for beat in sorted((run_folder / 'Beats').glob('Beat_*')):
            folders += [beat / 'beat_results' / 'Outputs' / 'harvest', beat / 'run_portals' / 'Outputs' / 'harvest']
    return [f for f in folders if f.is_dir()]

def main_push():
    parser = argparse.ArgumentParser(description="Push the staged evaluations of PORTALS/MAESTRO runs into the central harvest file")
    parser.add_argument("folders", type=str, nargs="+", help="run folders (PORTALS or MAESTRO) or harvest staging folders")
    parser.add_argument("--file", type=str, default=None, help="central netCDF file (default: config preferences.harvest_file or ~/mitim_harvest/mitim_harvest.nc)")
    parser.add_argument("--rebuild", action="store_true", help="move the central file aside and re-push everything (pushed or not) found under the folders")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be pushed")
    args = parser.parse_args()

    db = harvest_database(args.file)
    if args.rebuild:
        print(db.rebuild(args.folders))
        return
    folders = [f for run in args.folders for f in staging_folders_of(IOtools.expandPath(run))]
    if args.dry_run:
        for f in folders:
            print(f"{f}: {[p.name for p in sorted(f.glob('*.jsonl*')) if not _is_pushed(p)]}")
        return
    print(db.push(folders))

def main_plot():
    parser = argparse.ArgumentParser(description="Inspect a harvest database, or peek at the staging of a (running) run without pushing it")
    parser.add_argument("path", type=str, nargs="?", default=None,
                        help="central netCDF file (default: the configured one), or a run folder / harvest staging folder to peek at")
    parser.add_argument("--code", type=str, default=None, help="restrict to one code")
    parser.add_argument("--x", type=str, default=None, help="with --y: single scatter instead of the notebook")
    parser.add_argument("--y", type=str, default=None)
    parser.add_argument("--noplot", action="store_true", help="only print the interpretation report")
    args = parser.parse_args()

    if args.path is not None and IOtools.expandPath(args.path).is_dir():
        db = harvest_database.from_staging([IOtools.expandPath(args.path)])
    else:
        db = harvest_database(args.path)
    db.interpret(code=args.code)
    if args.noplot:
        return
    if args.x and args.y:
        import matplotlib.pyplot as plt
        db.plot(args.code or db.codes()[0], args.x, args.y)
        plt.show()
    else:
        fn = db.plotDatabase(codes=[args.code] if args.code else None)
        fn.show()


if __name__ == "__main__":
    # `python -m mitim_tools.harvest_tools.HARVESTtools push|plot ...` when the console scripts are not installed
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "plot"):
        print("usage: python -m mitim_tools.harvest_tools.HARVESTtools push <folders...> [--file F] | plot [file|folder] [--code C] [--x X --y Y] [--noplot]")
        sys.exit(2)
    cmd = sys.argv.pop(1)
    main_push() if cmd == "push" else main_plot()
