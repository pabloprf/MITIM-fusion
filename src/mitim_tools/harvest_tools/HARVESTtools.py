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
    1. Staging, per run (opt-in via the `harvest:` namelist block): JSON-lines, one file per code and
       writer process, `<run>/Outputs/harvest/<code>.<host>-<pid>.jsonl` (+ run_meta.json with the
       provenance), so a file never has two writers (legacy runs: one shared `<code>.jsonl`, still read). A driver that
       chains several runs (MAESTRO) hands its own folder to each of them (`staging_folder`), so the
       whole chain stages in ONE place and each record carries its `maestro_beat`. Append-only and
       crash-safe, so a dead or preempted run keeps its records and can be pushed by hand
       (`mitim_harvest <folder>`). Plain JSON never accumulates: once the tail exceeds ROLL_BYTES it
       is compressed as one more gzip member of `<stem>.jsonl.gz` (~20-30x smaller, since only a
       handful of inputs change between records), so a run holds a few MB at most. Pushed archives
       are renamed `<stem>.jsonl.pushed-<ts>.gz` and kept, so the central file can be rebuilt.
    2. Central store, per user: one netCDF-4 file, one group per code plus `runs`, unlimited `record`
       dimension, appended in place under an NFS-safe mkdir lock (IOtools.mkdir_lock). A variable
       that later records introduce reads back as NaN for the earlier ones. A variable keeps the
       type of its first appearance: later numbers into a string variable are stored as strings,
       later strings into a numeric one as numbers when they parse, else as NaN with the string in
       the sibling `<col>__str`. Every group is typed before anything is written, so a push
       appends all its records or none.

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
SCHEMA_VERSION = 5   # 4: optional per-record `maestro_beat` (chained runs share one staging folder); 5: CGYRO in_ = input.cgyro keys (+ out_derived_*, run history), input type maps, `<col>__str` siblings
ROLL_BYTES = 256 * 1024   # plain-JSON tail size that triggers compression into <code>.jsonl.gz (~60 TGLF records)
CODES = ('tglf', 'neo', 'cgyro', 'gx', 'qualikiz', 'eped')
RUNS_GROUP = 'runs'

RECORD_KEYS = ['run', 'hash']
# Provenance, once per run (run_meta.json) ...
RUN_KEYS = ['run', 'run_folder', 'user', 'host', 'mitim_version', 'git_branch', 'git_commit', 'created', 'maestro_beat',
            'recovered_by']   # '' for live runs; 'mitim_harvester <version>@<commit>' for records rebuilt from disk (HARVESTrecover)
# ... and once per (run, code), captured from the first record of that code (`averaging`: how the
# time-averaged fluxes and their std were computed, for CGYRO/GX; empty for single-value codes)
RUN_CODE_KEYS = ['machine', 'modules', 'code_version', 'in_process', 'averaging']
# ... plus `input_types` (see RECORD_TYPES), built at push time from the staged records

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
    `maestro_beat`, and `staging_folder` (overrides the argument: the driver's own staging folder,
    shared by every run of the chain). With a shared folder the run-level run_meta.json belongs to
    the driver and is left untouched; `maestro_beat` then travels per record instead.
    '''
    block = dict(block or {})
    shared = bool(block.get('staging_folder'))
    staging_folder = block.get('staging_folder') or staging_folder
    opts = {
        'enabled': bool(block.get('enabled', False)),
        'file': block.get('file', None),
        'push': bool(block.get('push', True)),
        'scan_trick_members': bool(block.get('scan_trick_members', True)),
        'folder': str(staging_folder),
        'maestro_beat': int(block['maestro_beat']) if 'maestro_beat' in block else None,
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
    if shared and meta_file.exists():
        opts['run_meta'] = run_meta
        return opts

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
    '''gzip by content, not by name (a file claimed by a push is <name>.claim-<tag>)'''
    with open(file, 'rb') as fi:
        is_gz = fi.read(2) == b'\x1f\x8b'
    return gzip.open(file, 'rt') if is_gz else open(file, 'r')

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
    Compress the plain <stem>.jsonl tail as one more gzip member appended to <stem>.jsonl.gz, then
    truncate the tail. Member first, truncate second: a kill in between duplicates lines, which the
    hash dedup absorbs; the reverse order could lose them. Only the process that owns the file (its
    single writer) rolls it, so no line can be appended between the read and the truncation.
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

# Staging file names. Each process writes its own <code>.<host>-<pid>.jsonl (+ its rolled .jsonl.gz),
# so no file ever has two writers and no lock is needed (NFS-safe). Legacy runs staged one shared
# <code>.jsonl per folder; those names are still read and pushed (the stem is then just <code>).
# Pushed archives: <stem>.jsonl.pushed-<ts>.gz. Files claimed by a push in progress: <name>.claim-<tag>.
_HOST = ''.join(c if c.isalnum() else '_' for c in socket.gethostname().split('.')[0]) or 'host'

def _writer_stem(code):
    return f"{code}.{_HOST}-{os.getpid()}"

def _is_staged(file):
    return '.jsonl' in file.name and not file.name.startswith('run_meta') and '.claim-' not in file.name

def _stem_of(file):
    return file.name.split('.jsonl')[0]

def _code_of(file):
    return _stem_of(file).split('.')[0]

def _is_pushed(file):
    return '.pushed-' in file.name

def _gz_bytes(file):
    '''Content of a staging file as gzip member(s): rolled archives as they are, plain tails compressed'''
    data = Path(file).read_bytes()
    return data if data[:2] == b'\x1f\x8b' or not data else gzip.compress(data)

def _is_orphan_claim(file, stale_s):
    '''A file claimed by a push that died before archiving or restoring it (claim older than stale_s)'''
    if '.claim-' not in file.name:
        return False
    try:
        t = datetime.datetime.strptime(file.name.rsplit('.claim-', 1)[1][:15], '%Y%m%d_%H%M%S')
    except ValueError:
        return False
    return (datetime.datetime.now() - t).total_seconds() > stale_s

def _claim(files):
    '''Rename each unpushed file to <name>.claim-<tag> (pushed archives, re-pushed by rebuild, stay put); [(original, claimed)]'''
    tag = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{_HOST}-{os.getpid()}"
    claimed = []
    for f in files:
        f = Path(f)
        if _is_pushed(f):
            claimed.append((f, f))
            continue
        c = f.with_name(f"{f.name}.claim-{tag}")
        try:
            os.replace(f, c)
        except FileNotFoundError:
            continue
        claimed.append((f, c))
    return claimed

def _fold(claimed, target_of):
    '''Append every claimed (not pushed) file, as gzip members, to target_of(folder, stem), then drop it'''
    for orig, c in claimed:
        if c == orig and _is_pushed(c):
            continue
        data = _gz_bytes(c)
        if data:
            with open(target_of(c.parent, _stem_of(c)), 'ab') as fo:
                fo.write(data)
        c.unlink(missing_ok=True)

def _archive(claimed):
    '''Pushed: one <stem>.jsonl.pushed-<ts>.gz per (folder, writer), kept so the central file can be rebuilt'''
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _fold(claimed, lambda folder, stem: folder / f"{stem}.jsonl.pushed-{ts}.gz")

def _unclaim(claimed):
    '''Failed push: back to staging as rolled, unpushed archives <stem>.jsonl.gz'''
    _fold(claimed, lambda folder, stem: folder / f"{stem}.jsonl.gz")

# ------------------------------------------------------------------------------------------------
# Recorder (attached to simulation objects)
# ------------------------------------------------------------------------------------------------

_SEEN = {}   # staging folder -> set of input hashes already staged (this process)

def _seen(folder):
    '''Hashes staged in the folder, seeded once per process from EVERY staging file there (all writers, legacy names, pushed archives)'''
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
        '''Context steers the recorder (e.g. scan_member=1 -> skippable) and is NOT stored in records, except `maestro_beat`'''
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
            beat = self.context.get('maestro_beat', self.options.get('maestro_beat'))
            if beat is not None and int(beat) >= 0:
                flat['maestro_beat'] = int(beat)
            flat.update({f"in_{k}": v for k, v in inputs.items()})
            flat.update({f"out_{k}": v for k, v in outputs.items()})

            plain = self.folder / f"{_writer_stem(code)}.jsonl"
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
                'created', 'machine', 'modules', 'code_version', 'averaging', 'input_types', 'input_types_record', 'recovered_by'}

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

STR_SUFFIX = '__str'

def _parse_float(s):
    if s == '':
        return np.nan
    try:
        return float(s)
    except ValueError:
        return None

def _reconcile_frame(df, existing):
    '''
    The columns of `df` as the arrays to append: (n_records, {column: (is_str, array)}). A variable
    keeps the type it was created with (`existing`: {variable: is_str} already in the file; new
    columns take their type from df). Numbers arriving into a string variable are written as
    strings; strings arriving into a numeric variable are written as numbers when they parse, and
    otherwise as NaN there with the string in the string sibling `<col>__str`. Nothing is dropped.
    '''
    cols = {}
    for col in df.columns:
        is_str = not pd.api.types.is_numeric_dtype(df[col])
        vals = df[col].to_numpy()
        as_str = existing.get(col, is_str)
        if as_str and is_str:
            cols[col] = (True, np.array([str(v) for v in vals], dtype=object))
        elif as_str:
            cols[col] = (True, np.array(['' if np.isnan(v) else str(v) for v in vals.astype('float64')], dtype=object))
        elif not is_str:
            cols[col] = (False, vals.astype('float64'))
        else:
            strs = [str(v) for v in vals]
            parsed = [_parse_float(s) for s in strs]
            cols[col] = (False, np.array([np.nan if p is None else p for p in parsed], dtype='float64'))
            rest = [s if p is None else '' for s, p in zip(strs, parsed)]
            if any(rest):
                side = f"{col}{STR_SUFFIX}"
                if not existing.get(side, True):
                    raise TypeError(f"harvest: '{side}' exists as a numeric variable, cannot hold the strings of '{col}'")
                cols[side] = (True, np.array(rest, dtype=object))
    return len(df), cols

# Input types, so an input file can be written back exactly (the netCDF stores every number as f8): per
# (run, code) in the runs table, `input_types` = JSON {KEY: 'bool'|'int'|'float'|'str'} in the order of
# the input file, each key typed as its FIRST record had it (never rewritten, later keys appended); a
# record whose own types differ (e.g. KY = 3 in one input.cgyro, 8.0E-02 in another) carries just those
# keys in its `input_types_record` (JSON, '' otherwise).
RECORD_TYPES = 'input_types_record'

def _input_types(row):
    out = {}
    for k, v in row.items():
        if not k.startswith('in_'):
            continue
        if isinstance(v, bool):
            out[k[3:]] = 'bool'
        elif isinstance(v, int):
            out[k[3:]] = 'int'
        elif isinstance(v, float) and not np.isnan(v):
            out[k[3:]] = 'float'
        elif isinstance(v, str) and v != '':
            out[k[3:]] = 'str'
    return out

def _type_overrides(run_types, row_types):
    '''Add the keys run_types has not seen (first record wins) and return this record's deviations from it'''
    for k, t in row_types.items():
        run_types.setdefault(k, t)
    return {k: t for k, t in row_types.items() if run_types[k] != t}

def _format_input_value(v, typ):
    '''One value as MITIM's GACODE writers emit it (bools as True/False, ints as ints); floats as the
    shortest string that reads back to the same double, always with a '.' so buildDictFromInput keeps it a float'''
    if typ == 'bool':
        return "True" if bool(v) else "False"
    if typ == 'int':
        return str(int(round(float(v))))
    if typ == 'float':
        s = repr(float(v))
        if '.' not in s:
            s = s.replace('e', '.0e') if 'e' in s else s + '.0'
        return s
    return str(v)

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
        A `<col>__str` column holds the values of numeric `<col>` that arrived as non-numeric
        strings (NaN in `<col>` for those records); it is returned as is, next to `<col>`.
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
                runs = runs[runs['code'] == code].drop(columns=['code', 'input_types'], errors='ignore')   # input_file() reads the type map itself
                runs = runs.drop(columns=[c for c in runs.columns if c in df.columns and c != 'run'])   # per-record maestro_beat wins
                df = df.merge(runs, on='run', how='left')
                for c in RUN_KEYS + RUN_CODE_KEYS:
                    if c in df and not pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].fillna('')
        return df

    # -------------------------------------------------------------------------- input files
    _INPUT_FILES = {'tglf': 'input.tglf', 'neo': 'input.neo', 'cgyro': 'input.cgyro'}

    def record(self, code, record):
        '''One record as a pandas Series with its run's provenance (incl. the `input_types` map), from its input hash or a row of load()'''
        if isinstance(record, str):
            df = self._read_group(code, mask_fn=lambda read: read('hash') == record)
            if len(df) == 0:
                raise KeyError(f"harvest: no {code} record with hash {record} in {self.file}")
            record = df.iloc[0]
        row = pd.Series(record)
        runs = self.runs()
        if len(runs) and 'input_types' in runs:
            prov = runs[(runs['run'] == row['run']) & (runs['code'] == code)]
            if len(prov):
                row = pd.concat([row, prov.iloc[0].drop(labels=[c for c in prov.columns if c in row.index or c == 'code'])])
        return row

    def input_file(self, code, record):
        '''
        Text of the input file (input.tglf / input.neo / input.cgyro) of one record, `record` = its input
        hash or a row of load(). Keys in the order of the original file, types restored from the run's
        `input_types` map (bools True/False as MITIM writes them, ints as ints, floats exact), missing
        values (NaN / '' fills of keys this record did not have) dropped, `<KEY>__str` siblings used.
        Records pushed before the type map existed come back with every number as a float.
        '''
        if code not in self._INPUT_FILES:
            raise ValueError(f"harvest: input files can be written for {list(self._INPUT_FILES)}, not '{code}'")
        row = self.record(code, record)
        n_species = row.get('in_N_SPECIES', np.nan)
        if code == 'cgyro' and not (isinstance(n_species, (int, float, np.number)) and np.isfinite(n_species)):
            # schema < 5 CGYRO records hold pygacode params1D (lowercase), not input.cgyro: refuse instead of writing a bogus file
            raise ValueError("harvest: this CGYRO record predates schema 5 and does not contain its input.cgyro")
        types = {}
        for col in ('input_types', RECORD_TYPES):   # run map, then this record's deviations
            if isinstance(row.get(col, ''), str) and row.get(col, ''):
                types.update(json.loads(row[col]))
        keys = [k[3:] for k in row.index if k.startswith('in_') and not k.endswith(STR_SUFFIX) and k not in RUN_CODE_KEYS]   # in_process is provenance
        lines = []
        for key in [k for k in types if k in keys] + sorted(k for k in keys if k not in types):
            v = row[f'in_{key}']
            if isinstance(v, float) and np.isnan(v) or v is None or v == '':
                v = row.get(f'in_{key}{STR_SUFFIX}', '')
                if v is None or v == '' or (isinstance(v, float) and np.isnan(v)):
                    continue
            typ = types.get(key, 'str' if isinstance(v, str) else 'float')
            lines.append(f"{key.ljust(23)} = {_format_input_value(v, typ)}")
        return "\n".join(lines) + "\n"

    def write_input_file(self, code, record, path):
        '''Write input_file(code, record) to `path` (a folder gets <folder>/input.<code>); returns the path'''
        path = Path(path)
        if path.is_dir():
            path = path / self._INPUT_FILES[code]
        path.write_text(self.input_file(code, record))
        return path

    # -------------------------------------------------------------------------- writing
    def push(self, staging_folders, timeout_s=600, stale_s=3600):
        '''
        Append every unpushed staging file of the given folders (plain tails and rolled .jsonl.gz of
        every writer process, legacy <code>.jsonl names included) to the central file, under the
        lock, then archive them as <stem>.jsonl.pushed-<ts>.gz. Records are deduplicated by
        (code, run, hash) within the call (MAESTRO may hand the run_portals/ and beat_results/
        twins). Returns {code: records_appended}. A push that fails (lock timeout, unreadable file,
        ...) appends nothing and leaves its files in staging as rolled, unpushed archives.
        '''
        files = [f for folder in [Path(f) for f in staging_folders] if folder.is_dir()
                 for f in sorted(folder.glob('*.jsonl*')) if (_is_staged(f) and not _is_pushed(f)) or _is_orphan_claim(f, stale_s)]
        return self._push_files(files, timeout_s=timeout_s, stale_s=stale_s, archive=True)

    def _push_files(self, files, timeout_s=600, stale_s=3600, archive=True):
        '''
        archive=True claims each unpushed file by renaming it first (atomic, also on NFS): a writer
        still alive simply starts a new tail, so nothing it appends or rolls later can be archived
        without having been pushed. archive=False (peek) reads the files as they are and touches nothing.
        '''
        claimed = _claim(files) if archive else [(f, f) for f in files]
        try:
            appended = self._append_files([c for _, c in claimed], timeout_s=timeout_s, stale_s=stale_s)
        except BaseException:
            if archive:
                _unclaim(claimed)
            raise
        if archive:
            _archive(claimed)
        return appended

    def _append_files(self, files, timeout_s=600, stale_s=3600):
        import netCDF4

        frames, runs_rows, typed, seen = {}, {}, [], set()
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
                typed.append((rkey, row, _input_types(row)))
                rows.append(row)
            if rows:
                frames.setdefault(code, []).extend(rows)

        appended = {}
        if not frames:
            return appended
        with IOtools.mkdir_lock(self.file, timeout_s=timeout_s, stale_s=stale_s):
            # 1. Type every column of every group against what the file already holds, writing
            #    nothing yet: a type conflict can no longer abort a push halfway through the groups
            schema, run_index = self._schema()
            # input types: extend each run's map (a run pushed again, e.g. by hand while alive, keeps
            # its map and only appends new keys), deviating records carry their own
            run_types = {rkey: json.loads(run_index[rkey][1] or '{}') if rkey in run_index else {} for rkey in runs_rows}
            for rkey, row, row_types in typed:
                over = _type_overrides(run_types[rkey], row_types)
                if over:
                    row[RECORD_TYPES] = json.dumps(over)
            for rkey, row in runs_rows.items():
                row['input_types'] = json.dumps(run_types[rkey])
            type_updates = {run_index[rkey][0]: row['input_types'] for rkey, row in runs_rows.items()
                            if rkey in run_index and row['input_types'] != (run_index[rkey][1] or '{}')}
            plan = {code: _reconcile_frame(_frame_from_rows(rows), schema.get(code, {})) for code, rows in frames.items()}
            new_runs = [r for r in runs_rows.values() if (r['run'], r['code']) not in run_index]
            if new_runs:
                plan[RUNS_GROUP] = _reconcile_frame(_frame_from_rows(new_runs), schema.get(RUNS_GROUP, {}))
            # 2. Write
            with netCDF4.Dataset(self.file, 'a' if self.file.exists() else 'w', format='NETCDF4') as ds:
                for name, (k, cols) in plan.items():
                    self._write_group(ds, name, cols, k)
                    if name != RUNS_GROUP:
                        appended[name] = k
                if type_updates:
                    grp = ds.groups[RUNS_GROUP]
                    if 'input_types' not in grp.variables:
                        grp.createVariable('input_types', str, ('record',))
                    for idx, js in type_updates.items():
                        grp.variables['input_types'][idx] = js
        print(f"\t- harvest: appended {sum(appended.values())} record(s) to {IOtools.clipstr(self.file)} ({', '.join(f'{k}: {v}' for k, v in appended.items())})", typeMsg='i')
        return appended

    def _schema(self):
        '''({group: {variable: is_string}}, {(run, code): (index, input_types)} of the runs group) of the central file'''
        import netCDF4
        schema, run_index = {}, {}
        if not self.file.exists():
            return schema, run_index
        with netCDF4.Dataset(self.file, 'r') as ds:
            for name, grp in ds.groups.items():
                schema[name] = {v: grp.variables[v].dtype == str for v in grp.variables}
            grp = ds.groups.get(RUNS_GROUP)
            if grp is not None and 'record' in grp.dimensions and len(grp.dimensions['record']):
                grp.set_auto_mask(False)
                n = len(grp.dimensions['record'])
                it = grp.variables['input_types'][:] if 'input_types' in grp.variables else [''] * n
                for i, (r, c, js) in enumerate(zip(grp.variables['run'][:], grp.variables['code'][:], it)):
                    run_index.setdefault((str(r), str(c)), (i, '' if js is None else str(js)))
        return schema, run_index

    @staticmethod
    def _write_group(ds, name, cols, k):
        grp = ds.groups[name] if name in ds.groups else ds.createGroup(name)
        if 'record' not in grp.dimensions:
            grp.createDimension('record', None)
            grp.setncattr('schema_version', SCHEMA_VERSION)
        n = len(grp.dimensions['record'])
        for col, (is_str, arr) in cols.items():
            if col not in grp.variables:
                if is_str:
                    grp.createVariable(col, str, ('record',))
                else:
                    grp.createVariable(col, 'f8', ('record',), fill_value=np.nan, zlib=True, chunksizes=(4096,))
            grp.variables[col][n:n + k] = arr
        grp.setncattr('last_push', datetime.datetime.now().isoformat(timespec='seconds'))
        return k

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
                        if f.parent.name == 'harvest' and (_is_staged(f) or _is_orphan_claim(f, 3600))})
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

    def statistics(self, code, run=None, **kw):
        '''harvest_statistics for one code (Spearman, PRCC, local sensitivities); cached per (code, run)'''
        from mitim_tools.harvest_tools.HARVESTstatistics import harvest_statistics
        cache = self.__dict__.setdefault('_stats_cache', {})
        key = (code, str(run), tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = harvest_statistics(self, code, run=run, **kw)
        return cache[key]

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
            if c != 'eped':
                stats = self.statistics(c)
                if stats.enough:
                    lines.append(stats.interpret())
        report = "\n".join(lines)
        print(report)
        return report

    # -------------------------------------------------------------------------- plotting
    # Fallback drive names for codes whose species are not resolved by charge (see _SPECIES)
    _DRIVES = {
        'gx':       {'Te': ['tprim_1', 'tprim_0'], 'Ti': ['tprim_2', 'tprim_1'], 'ne': ['fprim_1', 'fprim_0']},
        'qualikiz': {'Te': ['Ate'], 'Ti': ['Ati_0', 'Ati'], 'ne': ['Ane']},
    }
    _FLUXES = {'Qe': ['Qe', 'Qe_mean', 'efe_SI'], 'Qi': ['Qi', 'Qi_mean', 'efi_SI_0'], 'Ge': ['Ge', 'Ge_mean', 'pfe_SI']}

    # Species order differs between codes (TGLF: electrons first; NEO as MITIM writes it: electrons LAST;
    # CGYRO: any order), so electrons (charge -1) and the main ion (first charge +1) are found by charge:
    # (charge key, a/LT key, a/Ln key, first index); CGYRO records before schema 5 held pygacode's 0-indexed names
    _SPECIES = {
        'tglf':  ('ZS_{}', 'RLTS_{}', 'RLNS_{}', 1),
        'neo':   ('Z_{}', 'DLNTDR_{}', 'DLNNDR_{}', 1),
        'cgyro': ('Z_{}', 'DLNTDR_{}', 'DLNNDR_{}', 1),
    }
    _SPECIES_LEGACY = {'cgyro': ('z_{}', 'dlntdr_{}', 'dlnndr_{}', 0)}

    @classmethod
    def _species_keys(cls, code, df):
        keys = cls._SPECIES[code]
        legacy = cls._SPECIES_LEGACY.get(code)
        if legacy is not None and f"in_{keys[0].format(keys[3])}" not in df.columns and f"in_{legacy[0].format(legacy[3])}" in df.columns:
            return legacy
        return keys

    # Collisionality used to color flux-vs-drive plots, as each code's own input (no conversion between
    # normalizations): TGLF XNUE is the electron-ion collision frequency; NEO only takes NU_1, the collision
    # frequency of ITS species 1 (the others are scaled from it internally)
    _COLLISIONALITY = {
        'tglf': ('XNUE', 'XNUE (e-i collision frequency, TGLF input)'),
        'neo':  ('NU_1', 'NU_1 (collision frequency of NEO species 1, {species})'),
        'cgyro': ('NU_EE', 'NU_EE (e-e collision frequency, CGYRO input)'),
    }

    # Radial location of a record, r/a from its own input file (every code writes r/a, under its own name)
    _RADIUS = {'tglf': ['RMIN_LOC'], 'neo': ['RMIN_OVER_A'], 'cgyro': ['RMIN', 'rmin'], 'gx': ['rhoc'], 'qualikiz': ['x']}

    def radius(self, code, df):
        '''r/a of each record of `df` (a frame of load(code)); NaN when the radial input is not stored'''
        col = self._first(df, 'in_', self._RADIUS.get(code, []))
        return df[col].astype(float) if col is not None else pd.Series(np.nan, index=df.index)

    @staticmethod
    def _radial_bins(roa, max_exact=8, dr=0.1):
        '''
        Radial groups for plotting: each distinct r/a when there are at most `max_exact` of them (one radial
        grid), else bins [k*dr, (k+1)*dr) in r/a, so runs on slightly different grids (0.944 vs 0.95) fall
        together. Returns (group label per record, {label: color} ordered inner (dark) to outer (bright)).
        '''
        import matplotlib.pyplot as plt
        v = roa.to_numpy(dtype=float)
        ok = np.isfinite(v)
        distinct = np.unique(np.round(v[ok], 3))
        if len(distinct) <= max_exact:
            names = np.array([f'r/a={x:.3g}' for x in np.round(np.where(ok, v, 0), 3)], dtype=object)
            order = [f'r/a={x:.3g}' for x in distinct]
        else:
            lo = np.floor(np.where(ok, v, 0) / dr + 1e-6) * dr
            names = np.array([f'r/a {a:.2f}-{a + dr:.2f}' for a in lo], dtype=object)
            order = [f'r/a {a:.2f}-{a + dr:.2f}' for a in np.unique(lo[ok])]
        names[~ok] = 'r/a unknown'
        cmap = plt.get_cmap('plasma')
        colors = {k: cmap(0.85 * i / max(len(order) - 1, 1)) for i, k in enumerate(order)}
        if not ok.all():
            colors['r/a unknown'] = 'gray'
        return pd.Series(names, index=roa.index), colors

    @staticmethod
    def _first(df, prefix, candidates):
        for c in candidates:
            if f"{prefix}{c}" in df.columns:
                return f"{prefix}{c}"
        return None

    @classmethod
    def _species_index(cls, code, df):
        '''(electron index, main-ion index) from the charges in the input file; None when not found'''
        if code not in cls._SPECIES:
            return None, None
        zkey, _, _, i0 = cls._species_keys(code, df)
        z = {i: df[f'in_{zkey.format(i)}'].dropna().iloc[0] for i in range(i0, i0 + 12)
             if f'in_{zkey.format(i)}' in df.columns and df[f'in_{zkey.format(i)}'].notna().any()}
        ie = next((i for i, v in z.items() if v == -1), None)
        ii = next((i for i, v in z.items() if v == 1), None)
        return ie, ii

    @classmethod
    def _species_drives(cls, code, df):
        '''{'Te': [col], 'Ti': [col], 'ne': [col]} with the electron / main-ion gradient names of this code'''
        if code not in cls._SPECIES:
            return {}
        _, tkey, nkey, _ = cls._species_keys(code, df)
        ie, ii = cls._species_index(code, df)
        drives = {}
        if ie is not None:
            drives['Te'], drives['ne'] = [tkey.format(ie)], [nkey.format(ie)]
        if ii is not None:
            drives['Ti'] = [tkey.format(ii)]
        return drives

    @classmethod
    def _cgyro_drives(cls, df):
        return cls._species_drives('cgyro', df)

    def _drives(self, code, df):
        '''{'Te': col, 'Ti': col, 'ne': col} for this code (charge-resolved when possible, else the fallback names)'''
        candidates = {**self._DRIVES.get(code, {}), **self._species_drives(code, df)}
        return {k: self._first(df, 'in_', v) for k, v in candidates.items()}

    def _collisionality(self, code, df):
        '''(column, label) of the collisionality input of this code, or (None, None)'''
        if code not in self._COLLISIONALITY or f'in_{self._COLLISIONALITY[code][0]}' not in df.columns:
            return None, None
        key, label = self._COLLISIONALITY[code]
        if code == 'neo':
            z1 = df['in_Z_1'].dropna().iloc[0] if 'in_Z_1' in df.columns and df['in_Z_1'].notna().any() else None
            species = 'electrons' if z1 == -1 else (f'ion Z={z1:g}' if z1 is not None else 'unknown')
            label = label.format(species=species)
        return f'in_{key}', label

    def _color_by(self, df, key='run'):
        cols = GRAPHICStools.listColors()
        keys = list(dict.fromkeys(df[key])) if key in df else ['']
        return {k: cols[i % len(cols)] for i, k in enumerate(keys)}

    @staticmethod
    def _std_of(df, ycol):
        '''Column holding the std of an averaged flux column (out_Qi_mean -> out_Qi_std), if stored'''
        if ycol is not None and ycol.endswith('_mean') and ycol[:-5] + '_std' in df.columns:
            return ycol[:-5] + '_std'
        return None

    @staticmethod
    def _scatter(ax, sub, xcol, ycol, stdcol, color, **kw):
        '''Scatter with 1-sigma error bars when the flux has a stored std (CGYRO/GX time averages)'''
        if stdcol is not None:
            ax.errorbar(sub[xcol], sub[ycol], yerr=sub[stdcol], fmt='o', ms=3, color=color, alpha=0.7, elinewidth=0.8, capsize=2, **kw)
        else:
            ax.scatter(sub[xcol], sub[ycol], s=8, color=color, alpha=0.6, **kw)

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
        stdcol = self._std_of(df, y)
        for k, c in colors.items():
            sub = df[df[color_by] == k] if color_by in df else df
            self._scatter(ax, sub, x, y, stdcol, c, label=str(k)[:12], **kw)
        ax.set_xlabel(x); ax.set_ylabel(y + (' (1-sigma bars)' if stdcol else '')); ax.set_title(code)
        GRAPHICStools.addDenseAxis(ax)
        if len(colors) <= 12:
            ax.legend(fontsize=6, loc='best')
        return ax

    def plotDatabase(self, fn=None, codes=None):
        '''
        FigureNotebook: an Overview tab; per transport code its fluxes vs drives (all radii, then one column
        per radius), the averaging windows (CGYRO/GX), the effect of its settings, coverage and statistics;
        EPED; and a parity tab per pair of codes run on the same points
        '''
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
                self.plotFluxesVsDrives(code, fn=fn)
                self.plotFluxesByRadius(code, fn=fn)
                self.plotWindows(code, fn=fn)
                self.plotSettings(code, fn=fn)
            self.plotCoverage(code, fn=fn)
            self.plotPairs(code, fn=fn)
            if code != 'eped':
                stats = self.statistics(code)
                if stats.enough:
                    stats.plotImportance(fn=fn)
                    stats.plotSensitivities(fn=fn)
        for a, b in (('tglf', 'cgyro'), ('tglf', 'gx'), ('cgyro', 'gx')):
            if a in codes and b in codes:
                self.plotParity(a, b, fn=fn)
        return fn

    # -------------------------------------------------------------------------- parity between codes
    # Physics coordinates that identify "the same plasma point" in each code's input file (electron
    # species resolved by charge for CGYRO; GX names are template dependent and left out)
    _COORDS = {
        'tglf':  {'roa': ['RMIN_LOC'], 'aLTe': ['RLTS_1'], 'aLne': ['RLNS_1'], 'q': ['Q_LOC']},
        'cgyro': {'roa': ['RMIN', 'rmin'], 'q': ['Q', 'q']},   # + electron gradients from _cgyro_drives
    }
    _FLUX_PAIRS = [('Qe', ['Qe', 'Qe_mean']), ('Qi', ['Qi', 'Qi_mean']), ('Ge', ['Ge', 'Ge_mean'])]

    def _coords(self, code, df):
        cols = {k: self._first(df, 'in_', v) for k, v in self._COORDS.get(code, {}).items()}
        if code == 'cgyro':
            d = self._cgyro_drives(df)
            cols['aLTe'] = self._first(df, 'in_', d.get('Te', []))
            cols['aLne'] = self._first(df, 'in_', d.get('ne', []))
        return {k: v for k, v in cols.items() if v is not None}

    def match_records(self, code_a='tglf', code_b='cgyro', rtol=2e-3):
        '''
        Records of code_a and code_b from the SAME run at the SAME plasma point: r/a within 1e-3 and q,
        electron a/LTe and a/Lne within `rtol` relative (CGYRO's parsed inputs carry 4-5 significant
        digits, hence a tolerance rather than equality; the 2% scan-trick members never match). Returns
        one row per pair with the coordinates (code_a's values), the fluxes of both codes (`<flux>_a`,
        `<flux>_b`) and the stds when stored (`<flux>_std_a/b`).
        '''
        A, B = self.load(code_a, with_run_info=False), self.load(code_b, with_run_info=False)
        if len(A) == 0 or len(B) == 0:
            return pd.DataFrame()
        ca, cb = self._coords(code_a, A), self._coords(code_b, B)
        keys = [k for k in ('roa', 'q', 'aLTe', 'aLne') if k in ca and k in cb]
        if 'roa' not in keys:
            return pd.DataFrame()

        def table(df, coords, suffix):
            t = pd.DataFrame({'run': df['run'], 'roa_key': (df[coords['roa']] * 1000).round().astype(int)})
            for k in keys:
                t[f'{k}_{suffix}'] = df[coords[k]].astype(float)
            # model settings that label the comparison (TGLF saturation rule)
            if 'in_SAT_RULE' in df.columns:
                t['SAT_RULE'] = df['in_SAT_RULE'].map(lambda v: f'SAT{int(v)}' if pd.notna(v) else '')
            for name, cands in self._FLUX_PAIRS:
                col = self._first(df, 'out_', cands)
                if col is not None:
                    t[f'{name}_{suffix}'] = df[col]
                    std = self._std_of(df, col)
                    if std is not None:
                        t[f'{name}_std_{suffix}'] = df[std]
            return t

        m = table(A, ca, 'a').merge(table(B, cb, 'b'), on=['run', 'roa_key'], how='inner')
        close = np.ones(len(m), dtype=bool)
        for k in keys:
            if k == 'roa':
                continue
            a, b = m[f'{k}_a'].to_numpy(), m[f'{k}_b'].to_numpy()
            close &= np.abs(a - b) <= rtol * np.maximum(np.abs(a), 1e-12) + 1e-9
        m = m[close].copy()
        for k in keys:
            m[k] = m[f'{k}_a']
            m = m.drop(columns=[f'{k}_a', f'{k}_b'])
        subset = ['run'] + keys + [c for c in ('SAT_RULE',) if c in m.columns]
        cols = ['run'] + keys + [c for c in m.columns if c not in ['run', 'roa_key'] + keys]
        return m[cols].drop_duplicates(subset=subset).reset_index(drop=True)

    def plotParity(self, code_a='tglf', code_b='cgyro', fn=None, axs=None, symlog_linthresh=None, color_by=None):
        '''
        Parity plots (code_b vs code_a) of Qe, Qi, Ge for the records matched by match_records, error bars
        from the stored stds. Colors by TGLF saturation rule when TGLF is one of the codes (else by run),
        markers by run. Heat fluxes on log-log axes (non-positive values sit on the lower limit); particle
        flux on symlog axes, linear within +-symlog_linthresh (default: the median |Ge|).
        '''
        import matplotlib.pyplot as plt
        pairs = self.match_records(code_a, code_b)
        if len(pairs) == 0:
            print(f"\t- harvest: no {code_a}/{code_b} records at the same plasma points; no parity plot", typeMsg='i')
            return pairs
        if axs is None:
            if fn is not None:
                fig = fn.add_figure(label=f'Parity {code_a.upper()}-{code_b.upper()}')
                axs = fig.subplots(1, 3)
            else:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        color_by = color_by or ('SAT_RULE' if 'SAT_RULE' in pairs.columns else 'run')
        colors = self._color_by(pairs, color_by)
        markers = {r: 'osD^v<>ph*'[i % 10] for i, r in enumerate(dict.fromkeys(pairs['run']))}
        for ax, (name, _) in zip(np.atleast_1d(axs), self._FLUX_PAIRS):
            if f'{name}_a' not in pairs or f'{name}_b' not in pairs:
                ax.text(0.5, 0.5, f'{name} not in both', ha='center', va='center', transform=ax.transAxes)
                continue
            for k, c in colors.items():
                for r, mk in markers.items():
                    sub = pairs[(pairs[color_by] == k) & (pairs['run'] == r)]
                    if len(sub) == 0:
                        continue
                    ax.errorbar(sub[f'{name}_a'], sub[f'{name}_b'],
                                xerr=sub[f'{name}_std_a'] if f'{name}_std_a' in sub else None,
                                yerr=sub[f'{name}_std_b'] if f'{name}_std_b' in sub else None,
                                fmt=mk, ms=4, color=c, alpha=0.8, elinewidth=0.8, capsize=2,
                                label=(str(k) if color_by != 'run' else str(k)[:12]) if r == next(iter(markers)) else None)
            vals = np.concatenate([pairs[f'{name}_a'].to_numpy(dtype=float), pairs[f'{name}_b'].to_numpy(dtype=float)])
            vals = vals[np.isfinite(vals)]
            if name == 'Ge':
                lin = symlog_linthresh or max(float(np.median(np.abs(vals[vals != 0]))) if np.any(vals != 0) else 0.1, 1e-3)
                ax.set_xscale('symlog', linthresh=lin); ax.set_yscale('symlog', linthresh=lin)
                lo, hi = -np.max(np.abs(vals)) * 1.5, np.max(np.abs(vals)) * 1.5
                ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
                scale_note = f'symlog, linear within +-{lin:.2g}'
            else:
                pos = vals[vals > 0]
                lo, hi = (np.min(pos) * 0.5, np.max(pos) * 2.0) if len(pos) else (1e-3, 1.0)
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                n_np = int(np.sum(vals <= 0))
                scale_note = 'log' + (f', {n_np} non-positive value(s) clipped to the lower limit' if n_np else '')
            ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1)
            ax.set_xlabel(f'{name} {code_a} (GB)'); ax.set_ylabel(f'{name} {code_b} (GB)')
            ax.set_title(f'{name}: {len(pairs)} matched points ({scale_note})', fontsize=9)
            if len(colors) <= 12:
                ax.legend(fontsize=7, loc='best', title=(f'color: {color_by}' + (f', marker: run ({len(markers)})' if len(markers) > 1 else '')), title_fontsize=7)
        GRAPHICStools.adjust_figure_layout(np.atleast_1d(axs)[0].figure)
        return pairs

    def _plot_overview(self, fn, codes):
        fig = fn.add_figure(label='Overview')
        axs = fig.subplots(2, 2)
        frames = {c: self.load(c) for c in codes}
        # Record counts span orders of magnitude between codes (thousands of TGLF runs vs a handful of EPED):
        # every count axis is logarithmic and each bar carries its number
        ax = axs[0, 0]
        bars = ax.bar(list(frames), [len(d) for d in frames.values()])
        ax.bar_label(bars, fontsize=8)
        ax.set_yscale('log'); ax.set_ylim(bottom=0.8)
        ax.set_ylabel('records (log)'); ax.set_title('Records per code')
        ax = axs[0, 1]
        for c, df in frames.items():
            if 'created' in df and len(df):
                per_run = df.groupby('run').agg(n=('hash', 'size'), t=('created', 'first'))
                per_run['t'] = pd.to_datetime(per_run['t'], errors='coerce')
                per_run = per_run.sort_values('t')
                ax.step(per_run['t'], per_run['n'].cumsum(), '-o', ms=4, where='post', label=c)
        ax.set_yscale('log'); ax.set_ylim(bottom=0.8)
        ax.set_ylabel('cumulative records (log)'); ax.set_title('Records vs run start time'); ax.legend(fontsize=7)
        ax.tick_params(axis='x', labelrotation=30)
        ax = axs[1, 0]
        runs = pd.concat([d[['run']].assign(code=c) for c, d in frames.items() if 'run' in d], ignore_index=True) if frames else pd.DataFrame()
        if len(runs):
            top = runs.groupby('run').size().sort_values(ascending=False).head(15)
            bars = ax.barh(list(top.index), top.values)
            ax.bar_label(bars, fontsize=8, padding=2)
            ax.invert_yaxis()
        ax.set_xscale('log'); ax.set_xlim(left=0.8)
        ax.set_xlabel('records (log)'); ax.set_title('Records per run (top 15)')
        ax = axs[1, 1]
        prov = pd.concat([d[['machine', 'code_version']].assign(code=c) for c, d in frames.items() if 'machine' in d], ignore_index=True) if frames else pd.DataFrame()
        if len(prov):
            prov['key'] = prov['code'] + ' @ ' + prov['machine'].replace('', '?') + ' / ' + prov['code_version'].map(lambda s: s.split('\n')[0][:20] if s else '?')
            cnt = prov.groupby('key').size().sort_values(ascending=False).head(12)
            bars = ax.barh(cnt.index, cnt.values, color='gray')
            ax.bar_label(bars, fontsize=8, padding=2)
            ax.invert_yaxis()
            ax.tick_params(axis='y', labelsize=6)
        ax.set_xscale('log'); ax.set_xlim(left=0.8)
        ax.set_xlabel('records (log)'); ax.set_title('Machine / code version')
        for a in axs.flatten():
            a.grid(True, which='major', alpha=0.4)
        GRAPHICStools.adjust_figure_layout(fig)

    _DRIVE_LABELS = {'Te': 'a/LTe', 'Ti': 'a/LTi', 'ne': 'a/Lne'}

    @staticmethod
    def _flux_axis(values, name, nbins=40):
        '''Axis spec for a flux spanning orders of magnitude: log for heat fluxes, symlog for the particle flux'''
        v = values[np.isfinite(values)]
        if name == 'Ge' or len(v[v > 0]) == 0:
            nz = np.abs(v[v != 0])
            lin = float(np.median(nz)) if len(nz) else 1.0
            m = float(np.max(np.abs(v))) * 1.3 if len(v) else 1.0
            # bins uniform in the symlog-like coordinate t = sign(x) log10(1 + |x|/lin)
            t = np.linspace(-np.log10(1 + m / lin), np.log10(1 + m / lin), nbins)
            bins = np.sign(t) * lin * (10 ** np.abs(t) - 1)
            return {'kind': 'symlog', 'linthresh': lin, 'lim': (-m, m), 'bins': bins, 'note': f'symlog, linear within +-{lin:.2g}'}
        pos = v[v > 0]
        lo, hi = float(pos.min()) / 1.5, float(pos.max()) * 1.5
        n_np = int((v <= 0).sum())
        return {'kind': 'log', 'lim': (lo, hi), 'bins': np.logspace(np.log10(lo), np.log10(hi), nbins),
                'note': 'log' + (f', {n_np} non-positive not shown' if n_np else '')}

    @staticmethod
    def _apply_flux_axis(ax, spec):
        if spec['kind'] == 'log':
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog', linthresh=spec['linthresh'])
            ax.axhline(0, color='k', lw=0.4)
        ax.set_ylim(*spec['lim'])

    def _color_spec(self, code, df, color_by):
        '''
        How to color the records of df: 'radius' (r/a groups), 'collisionality' (the code's own input, log
        scale; falls back to radius when the code has none), 'run', or a Series of category labels (e.g. settings)
        '''
        if isinstance(color_by, pd.Series):
            return {'kind': 'cat', 'labels': color_by, 'colors': self._color_by(color_by.to_frame('c'), 'c'), 'title': color_by.name or ''}
        if color_by == 'collisionality':
            from matplotlib.colors import LogNorm
            ccol, clabel = self._collisionality(code, df)
            vals = df[ccol].to_numpy(dtype=float) if ccol else np.array([])
            if np.any(vals > 0):
                pos = vals[vals > 0]
                return {'kind': 'cont', 'values': vals, 'norm': LogNorm(vmin=pos.min(), vmax=max(pos.max(), pos.min() * 1.0001)), 'title': clabel}
            color_by = 'radius'
        if color_by == 'radius':
            labels, colors = self._radial_bins(self.radius(code, df))
            return {'kind': 'cat', 'labels': labels, 'colors': colors, 'title': 'radius'}
        return {'kind': 'cat', 'labels': df[color_by].astype(str), 'colors': self._color_by(df.assign(_c=df[color_by].astype(str)), '_c'), 'title': color_by}

    @staticmethod
    def _scatter_spec(ax, x, y, spec, mask=None, s=6):
        '''
        Scatter y vs x colored by a _color_spec (mask restricts the records); returns the mappable of a continuous
        spec. Categories are drawn in one call in shuffled order, so no category hides behind the last one drawn.
        '''
        mask = np.ones(len(x), dtype=bool) if mask is None else np.asarray(mask)
        if spec['kind'] == 'cont':
            return ax.scatter(x[mask], y[mask], c=spec['values'][mask], norm=spec['norm'], cmap='viridis', s=s, alpha=0.8, zorder=2)
        idx = np.random.default_rng(0).permutation(np.flatnonzero(mask))
        colors = spec['labels'].map(spec['colors']).to_numpy()[idx]
        ax.scatter(x[idx], y[idx], s=s, c=list(colors), alpha=0.7, zorder=2)
        return None

    @staticmethod
    def _color_key(fig, spec, sc):
        '''Colorbar (continuous) or legend (categories) in the right margin that adjust_figure_layout leaves'''
        if spec['kind'] == 'cont':
            if sc is not None:
                fig.colorbar(sc, cax=fig.add_axes([0.925, 0.1, 0.012, 0.8])).set_label(spec['title'])
            return
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], ls='', marker='o', color=c, label=k) for k, c in spec['colors'].items()][:25]
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.905, 0.5), fontsize=7, title=spec['title'], title_fontsize=7, frameon=False)

    def plotFluxesVsDrives(self, code, fn=None, run=None, color_by='radius'):
        '''
        Every flux against every drive: rows Qe, Qi, Ge; columns a/LTe, a/LTi, a/Lne (electron and main-ion
        gradients resolved by charge), plus the distribution of each flux as a last column. Colored by radius
        (r/a groups, see _radial_bins) by default: the same gradient drives very different fluxes at different
        radii. color_by='collisionality' colors by the code's collisionality input (log), or 'run'.
        1-sigma bars when the flux has a stored std. Fluxes span orders of magnitude: heat fluxes on a log axis
        (non-positive values cannot be drawn and are counted in the label), particle flux on symlog (linear
        within +-median|Ge|). Drives stay linear. `run` restricts to one run id or a list of them.
        '''
        import matplotlib.pyplot as plt
        df = self.load(code, run=run)
        if len(df) == 0:
            return None
        fluxes = {k: self._first(df, 'out_', v) for k, v in self._FLUXES.items()}
        drives = self._drives(code, df)
        spec = self._color_spec(code, df, color_by)

        if fn is not None:
            fig = fn.add_figure(label=code.upper())
        else:
            fig = plt.figure(figsize=(18, 11))
        axs = fig.subplots(3, 4, gridspec_kw={'width_ratios': [1, 1, 1, 0.55]})

        sc = None
        for irow, (fname, fcol) in enumerate(fluxes.items()):
            stdcol = self._std_of(df, fcol)
            yscale = self._flux_axis(df[fcol].to_numpy(dtype=float), fname) if fcol is not None else None
            for icol, dname in enumerate(('Te', 'Ti', 'ne')):
                ax, dcol = axs[irow, icol], drives.get(dname)
                if fcol is None or dcol is None:
                    ax.text(0.5, 0.5, f'no {fname} or {self._DRIVE_LABELS[dname]} column', ha='center', va='center', transform=ax.transAxes)
                    continue
                if stdcol is not None:
                    ax.errorbar(df[dcol], df[fcol], yerr=df[stdcol], fmt='none', ecolor='gray', elinewidth=0.6, alpha=0.5, zorder=1)
                sc = self._scatter_spec(ax, df[dcol].to_numpy(dtype=float), df[fcol].to_numpy(dtype=float), spec) or sc
                if irow == 2:
                    ax.set_xlabel(f'{self._DRIVE_LABELS[dname]}  ({dcol[3:]})')
                if icol == 0:
                    ax.set_ylabel(f'{fname}  ({fcol[4:]})\n{yscale["note"]}' + ('\n1-sigma bars' if stdcol else ''), fontsize=9)
                self._apply_flux_axis(ax, yscale)
                ax.grid(True, alpha=0.3)
            # distribution of this flux on the same flux axis (log-spaced bins; log counts to show the tails)
            axh = axs[irow, 3]
            if fcol is not None:
                vals = df[fcol].dropna().to_numpy(dtype=float)
                if yscale['kind'] == 'log':
                    vals = vals[vals > 0]
                axh.hist(vals, bins=yscale['bins'], orientation='horizontal', color='gray')
                self._apply_flux_axis(axh, yscale)
                axh.set_xscale('log')
            axh.set_xlabel('records (log)' if irow == 2 else '')
            axh.tick_params(labelleft=False)
            axh.grid(True, alpha=0.3)
        axs[0, 1].set_title(f'{code.upper()}: {len(df)} records, {df["run"].nunique()} run(s)', fontsize=10)
        axs[0, 3].set_title('distribution', fontsize=9)

        GRAPHICStools.adjust_figure_layout(fig)
        self._color_key(fig, spec, sc)
        return fig

    # Each flux against the gradient that drives it
    _OWN_DRIVE = {'Qe': 'Te', 'Qi': 'Ti', 'Ge': 'ne'}

    def plotFluxesByRadius(self, code, fn=None, run=None, color_by='collisionality', label=None, fig=None):
        '''
        Trends at fixed radius: rows Qe, Qi, Ge against their own drive (a/LTe, a/LTi, a/Lne), one column per
        radial group (_radial_bins). The flux axis is shared along a row, so stiffness and threshold can be
        compared between radii. Colored by collisionality (log; by run when the code has no collisionality
        input), or by any color_by accepted by _color_spec (plotSettings passes the settings of each record).
        '''
        import matplotlib.pyplot as plt
        df = self.load(code, run=run)
        if len(df) == 0:
            return None
        groups, gcolors = self._radial_bins(self.radius(code, df))
        order = list(gcolors)
        if isinstance(color_by, str) and color_by == 'collisionality' and self._collisionality(code, df)[0] is None:
            color_by = 'run'
        spec = self._color_spec(code, df, color_by)
        fluxes = {k: self._first(df, 'out_', v) for k, v in self._FLUXES.items()}
        drives = self._drives(code, df)

        own = fig is None   # else a (sub)figure of the caller, which also draws the color key
        if own:
            fig = fn.add_figure(label=label or f'{code.upper()} by radius') if fn is not None else plt.figure(figsize=(18, 10))
        axs = np.atleast_2d(fig.subplots(3, len(order), sharey='row', squeeze=False))
        sc = None
        for irow, (fname, fcol) in enumerate(fluxes.items()):
            dcol = drives.get(self._OWN_DRIVE[fname])
            if fcol is None or dcol is None:
                axs[irow, 0].text(0.5, 0.5, f'no {fname} or its drive', ha='center', va='center', transform=axs[irow, 0].transAxes)
                continue
            stdcol = self._std_of(df, fcol)
            yscale = self._flux_axis(df[fcol].to_numpy(dtype=float), fname)
            x, y = df[dcol].to_numpy(dtype=float), df[fcol].to_numpy(dtype=float)
            for icol, g in enumerate(order):
                ax, m = axs[irow, icol], (groups == g).to_numpy()
                if stdcol is not None:
                    ax.errorbar(x[m], y[m], yerr=df[stdcol].to_numpy(dtype=float)[m], fmt='none', ecolor='gray', elinewidth=0.6, alpha=0.5, zorder=1)
                sc = self._scatter_spec(ax, x, y, spec, mask=m) or sc
                self._apply_flux_axis(ax, yscale)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
                if irow == 0:
                    ax.set_title(f'{g}  ({int(m.sum())})', fontsize=9)
                ax.set_xlabel(f'{self._DRIVE_LABELS[self._OWN_DRIVE[fname]]} ({dcol[3:]})', fontsize=8)
            axs[irow, 0].set_ylabel(f'{fname}  ({fcol[4:]})\n{yscale["note"]}' + ('\n1-sigma bars' if stdcol else ''), fontsize=8)
        if own:
            GRAPHICStools.adjust_figure_layout(fig)
            fig.subplots_adjust(wspace=0.08)
            self._color_key(fig, spec, sc)
        else:
            fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, wspace=0.08, hspace=0.35)
        return fig

    # -------------------------------------------------------------------------- averaging windows (CGYRO, GX)
    def plotWindows(self, code, fn=None, run=None):
        '''
        How long each time-averaged run was and which part of it the flux average used (codes that store
        avg_tmin/avg_tmax: CGYRO, GX). Left, one line per record grouped by radius, on the total simulated time
        (a/cs): time inherited from the restart chain (gray), this run (light), averaging window (dark, colored
        by radius), MAX_TIME requested (|), and x where the run stopped before MAX_TIME. Right: window length vs
        r/a (open = cold start), relative std of Qi/Qe vs window length, effective samples of Qi vs window
        length, and cost per a/cs.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        df = self.load(code, run=run)
        if len(df) == 0 or 'out_avg_tmin' not in df:
            return None
        get = lambda c, fill=np.nan: df[c].to_numpy(dtype=float) if c in df else np.full(len(df), fill)
        roa = self.radius(code, df)
        groups, colors = self._radial_bins(roa)
        t_inh, t_last = np.nan_to_num(get('out_restart_t_inherited', 0.0)), get('out_t_last')
        tmin, tmax, max_time = get('out_avg_tmin'), get('out_avg_tmax'), get('out_max_time')
        wlen = tmax - tmin
        warm = get('out_restart_warm', 0.0) > 0
        order = np.lexsort((np.arange(len(df)), roa.to_numpy(dtype=float)))

        fig = fn.add_figure(label=f'{code.upper()} windows') if fn is not None else plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1])
        ax = fig.add_subplot(gs[:, 0])
        y = np.empty(len(df))
        y[order] = np.arange(len(df))
        segs = [[(0, yi), (ti, yi)] for yi, ti in zip(y, t_inh) if ti > 0]
        ax.add_collection(LineCollection(segs, colors='gray', lw=2, alpha=0.5))
        ax.add_collection(LineCollection([[(a, yi), (a + b, yi)] for yi, a, b in zip(y, t_inh, t_last)], colors='lightsteelblue', lw=2))
        ax.add_collection(LineCollection([[(a + b, yi), (a + c, yi)] for yi, a, b, c in zip(y, t_inh, tmin, tmax)],
                                         colors=[colors[g] for g in groups], lw=3.5))
        ax.plot(t_inh + max_time, y, '|', color='k', ms=6)
        short = get('out_reached_max_time', 1.0) == 0
        ax.plot((t_inh + t_last)[short], y[short], 'x', color='r', ms=6, label=f'stopped before MAX_TIME ({int(short.sum())})')
        ticks = [np.mean(y[(groups == g).to_numpy()]) for g in colors if (groups == g).any()]
        ax.set_yticks(ticks, [g for g in colors if (groups == g).any()], fontsize=7)
        for g in colors:
            m = (groups == g).to_numpy()
            if m.any():
                ax.axhline(y[m].max() + 0.5, color='k', lw=0.3)
        ax.set_ylim(len(df) - 0.5, -0.5)
        ax.set_xlim(0, np.nanmax(np.concatenate([t_inh + t_last, t_inh + max_time])) * 1.02)
        ax.set_xlabel('total simulated time (a/cs): inherited (gray) + this run (light), window (dark)')
        ax.set_title(f'{code.upper()}: {len(df)} records; | = MAX_TIME requested', fontsize=9)
        if short.any():
            ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, axis='x', alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        for g, c in colors.items():
            m = (groups == g).to_numpy()
            ax.scatter(roa[m & warm], wlen[m & warm], color=c, s=20)
            ax.scatter(roa[m & ~warm], wlen[m & ~warm], facecolors='none', edgecolors=c, s=20)
        ax.set_xlabel('r/a'); ax.set_ylabel('window length (a/cs)')
        ax.set_title(f'median window {np.nanmedian(wlen):.0f} a/cs = {100 * np.nanmedian(wlen / t_last):.0f}% of the run; open = cold start', fontsize=8)

        ax = fig.add_subplot(gs[0, 2])
        for fname, mk in (('Qi', 'o'), ('Qe', 's')):
            mcol, scol = self._first(df, 'out_', [f'{fname}_mean']), self._first(df, 'out_', [f'{fname}_std'])
            if mcol and scol:
                rel = get(scol) / np.abs(get(mcol))
                ax.scatter(wlen, rel, c=[colors[g] for g in groups], marker=mk, s=16, label=fname)
        ax.set_yscale('log'); ax.set_xlabel('window length (a/cs)'); ax.set_ylabel('std / |mean|')
        ax.legend(fontsize=7, title='marker', title_fontsize=7)

        ax = fig.add_subplot(gs[1, 1])
        if 'out_Qi_ncorr' in df:
            ax.scatter(wlen, get('out_Qi_ncorr'), c=[colors[g] for g in groups], s=16)
            ax.set_ylabel('Qi effective samples (ncorr)')
        ax.set_xlabel('window length (a/cs)')

        ax = fig.add_subplot(gs[1, 2])
        if 'out_cost_s_per_acs' in df:
            ax.scatter(roa, get('out_cost_s_per_acs'), c=[colors[g] for g in groups], s=16)
            wall = np.nansum(get('out_wall_s', 0.0) * np.nan_to_num(get('out_n_nodes', 1.0), nan=1.0)) / 3600
            ax.set_title(f'total {wall:.0f} node-hours in this tab', fontsize=8)
            ax.set_ylabel('wall s per a/cs (per run)')
        ax.set_xlabel('r/a')
        for a in fig.axes[1:]:
            a.grid(True, alpha=0.3)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, wspace=0.3, hspace=0.3)
        return fig

    # -------------------------------------------------------------------------- code settings (numerics, model knobs)
    # Input keys that are code settings rather than plasma physics: resolution, model choices and knobs.
    # _RUN_CONTROL keys only steer the run (length, output cadence, parallelization, initial amplitude):
    # left out of the settings and of the physics signature
    _SETTINGS = {
        'tglf': r'^(SAT_RULE|NBASIS_M(IN|AX)|NKY|KY$|KYGRID_MODEL|NXGRID|N?WIDTH(_MIN)?$|FIND_WIDTH|USE_[A-Z_]+|WDIA_TRAPPED|IBRANCH|'
                r'NMODES|FILTER|THETA_TRAPPED|ETG_FACTOR|ALPHA_(ZF|QUENCH|E|P|MACH)$|XNU_(MODEL|FACTOR)|DEBYE_FACTOR|UNITS|'
                r'ADIABATIC_ELEC|NEW_EIKONAL|DAMP_(PSI|SIG)|LINSKER_FACTOR|GRADB_FACTOR|WD_ZERO|PARK|G(C)?HAT|RLNP_CUTOFF|[BF]T?_MODEL_SA)',
        'neo':  r'^(N_(RADIAL|THETA|ENERGY|XI)$|[A-Z0-9_]+_MODEL$)',
        'cgyro': r'^(N_TOROIDAL|KY$|N_RADIAL|BOX_SIZE|N_THETA|N_XI|N_ENERGY|E_MAX|N_FIELD|NONLINEAR_FLAG|[A-Z_]+_MODEL$|'
                 r'SHEAR_METHOD|N?UP_(THETA|RADIAL|ALPHA)|DELTA_T(_METHOD)?$|EXCH_FLAG|[A-Z_]+_SCALE$)',
        'gx':   r'^(nx|ny|ntheta|nperiod|nhermite|nlaguerre|y0|x0|jtwist|boundary|nonlinear_mode|scheme|cfl|dt|closure_model|'
                r'hypercollisions|hyper|HB_hyper|nu_hyper_[ml]|p_hyper(_[ml])?|D_hyper|D_H|p_HB|w_osc|ei_colls|fphi|fapar|fbpar|collisions_model)$',
        'qualikiz': r'^(kthetarhos|numsols|relacc\d|maxruns|maxpts|timeout|ETGmult|collmult|coll_flag|rot_flag|phys_meth|'
                    r'separateflux|typee|typei|verbose|el_type|set_qn_normni|set_qn_an|check_qn|x_eq_rho)',
    }
    _RUN_CONTROL = {
        'cgyro': r'^(MAX_TIME|PRINT_STEP|RESTART_STEP|TOROIDALS_PER_PROC|MOMENT_PRINT_FLAG|FIELD_PRINT_FLAG|AMP0?|FREQ_TOL|SILENT_FLAG)$',
        'neo':   r'^SILENT_FLAG$',
        'tglf':  r'^(WRITE_WAVEFUNCTION_FLAG|NN_MAX_ERROR)$',
        'gx':    r'^(t_max|nstep|nwrite|nsave|debug|save_for_restart|restart|append_on_restart|omega|fluxes|fields|moments|'
                 r'init_amp|init_field|gaussian_init|ikpar_init)$',
    }

    def _setting_columns(self, code, df):
        '''(settings columns, run-control columns) of this code present in df'''
        import re
        ins = [c for c in df.columns if c.startswith('in_')]
        rs, rc = self._SETTINGS.get(code), self._RUN_CONTROL.get(code)
        run_control = [c for c in ins if rc and re.match(rc, c[3:])]
        return [c for c in ins if rs and re.match(rs, c[3:]) and c not in run_control], run_control

    def settings(self, code, run=None, df=None):
        '''
        Code settings of every record: returns (df, settings columns that vary, label per record). The label
        names the values of the varying settings ('SAT_RULE=3, USE_BPER=1'); records with equal labels ran with
        the same settings.
        '''
        df = self.load(code, run=run) if df is None else df
        cols, _ = self._setting_columns(code, df)
        varying = [c for c in cols if df[c].nunique(dropna=False) > 1]
        fmt = lambda v: 'unset' if pd.isna(v) else (f'{v:g}' if isinstance(v, (int, float, np.number)) else str(v))
        labels = df[varying].apply(lambda r: ', '.join(f'{c[3:]}={fmt(v)}' for c, v in r.items()), axis=1) if varying else pd.Series('', index=df.index)
        return df, varying, labels.rename('settings')

    def _physics_signature(self, code, df, digits=5):
        '''Hash of every numeric input that is neither a setting nor run control, rounded to `digits` significant digits'''
        settings, run_control = self._setting_columns(code, df)
        cols = [c for c in df.columns if c.startswith('in_') and c not in settings + run_control and pd.api.types.is_numeric_dtype(df[c])]
        v = df[cols].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = 10 ** (np.floor(np.log10(np.abs(v))) - digits + 1)
            v = np.where(np.isfinite(mag) & (v != 0), np.round(v / mag) * mag, v)
        return pd.util.hash_pandas_object(pd.DataFrame(v).round(12), index=False).to_numpy()

    def match_settings(self, code, run=None):
        '''
        Records with the SAME physics inputs (every numeric input except settings and run control, to 5 significant
        digits) run with different settings: one row per (reference record, other record), reference = the most
        common settings. Columns: settings, r/a and the fluxes `<flux>_ref` / `<flux>` (+ stds when stored).
        '''
        df, varying, labels = self.settings(code, run=run)
        if not varying:
            return pd.DataFrame()
        ref = labels.value_counts().index[0]
        t = pd.DataFrame({'sig': self._physics_signature(code, df), 'settings': labels, 'roa': self.radius(code, df)})
        for name, cands in self._FLUX_PAIRS:
            col = self._first(df, 'out_', cands)
            if col is not None:
                t[name] = df[col]
                if self._std_of(df, col):
                    t[f'{name}_std'] = df[self._std_of(df, col)]
        a, b = t[t['settings'] == ref].drop_duplicates('sig'), t[t['settings'] != ref]
        m = b.merge(a.drop(columns=['settings', 'roa']), on='sig', suffixes=('', '_ref')).drop(columns='sig')
        m.attrs['reference'] = ref
        return m

    def plotSettings(self, code, fn=None, run=None):
        '''
        Effect of the code settings that vary in the database (e.g. TGLF SAT_RULE, CGYRO N_TOROIDAL/BOX_SIZE):
        one tab with the records per settings combination and the flux-vs-drive trends per radius colored by
        settings (differences there mix settings with physics); when some physics points were run with more
        than one setting, a second tab with those matched pairs: flux with each setting vs flux with the most
        common one (the clean comparison).
        '''
        import matplotlib.pyplot as plt
        df, varying, labels = self.settings(code, run=run)
        if not varying:
            print(f"\t- harvest: {code} settings are the same in every record; no settings tab", typeMsg='i')
            return None
        # settings combinations named A, B, ... by number of records; the left panel spells them out
        counts = labels.value_counts()
        letter = {k: chr(65 + i) if i < 26 else f'#{i}' for i, k in enumerate(counts.index)}
        letters = labels.map(letter).rename('settings')
        colors = self._color_by(letters.to_frame('c'), 'c')
        fig = fn.add_figure(label=f'{code.upper()} settings') if fn is not None else plt.figure(figsize=(18, 11))
        sub = fig.subfigures(1, 2, width_ratios=[1, 4])
        ax = sub[0].subplots()
        ax.axis('off')
        ypos = 0.98
        ax.text(0.0, ypos, f'{code.upper()}: {len(varying)} settings vary', fontsize=9, weight='bold', va='top', transform=ax.transAxes)
        for k, n in counts.head(12).items():
            ypos -= 0.035
            ax.text(0.0, ypos, f'{letter[k]}: {n} records', fontsize=8, weight='bold', color=colors[letter[k]], va='top', transform=ax.transAxes)
            for item in k.split(', '):
                ypos -= 0.025
                ax.text(0.06, ypos, item, fontsize=7, color=colors[letter[k]], va='top', transform=ax.transAxes)
        self.plotFluxesByRadius(code, run=run, color_by=letters, fig=sub[1])

        pairs = self.match_settings(code, run=run)
        if len(pairs) == 0:
            groups, _ = self._radial_bins(self.radius(code, df))
            per_radius = letters.groupby(groups).nunique().max() <= 1
            note = ('settings change only between radii\n(one combination per radial group)' if per_radius else
                    'no physics point was run with two settings:\nthe trends mix settings and physics')
            ax.text(0.0, 0.01, note, transform=ax.transAxes, fontsize=8, va='bottom', color='darkred')
            return fig
        pairs['settings'] = pairs['settings'].map(letter)
        pairs.attrs['reference'] = letter[pairs.attrs['reference']]
        fig2 = fn.add_figure(label=f'{code.upper()} settings parity') if fn is not None else plt.figure(figsize=(15, 5))
        axs = fig2.subplots(1, 3)
        for ax, (name, _) in zip(axs, self._FLUX_PAIRS):
            if name not in pairs:
                continue
            for k, g in pairs.groupby('settings'):
                ax.errorbar(g[f'{name}_ref'], g[name], xerr=g.get(f'{name}_std_ref'), yerr=g.get(f'{name}_std'),
                            fmt='o', ms=3, color=colors[k], alpha=0.7, elinewidth=0.6, label=k)
            lim = np.nanmax(np.abs(pairs[[f'{name}_ref', name]].to_numpy(dtype=float))) * 1.1
            ax.plot([-lim, lim], [-lim, lim], '--', color='gray', lw=1)
            ax.set_xlabel(f'{name} with settings {pairs.attrs["reference"]} (see {code.upper()} settings tab)'); ax.set_ylabel(f'{name} with the other settings')
            ax.set_title(f'{name}: {len(pairs)} matched physics points', fontsize=9)
            ax.grid(True, alpha=0.3)
        axs[0].legend(fontsize=6, loc='upper left')
        GRAPHICStools.adjust_figure_layout(fig2)
        return fig

    # -------------------------------------------------------------------------- input coverage
    # Categories of input-file keys (regex on the key, first match wins; unmatched -> 'other')
    _CATEGORIES = {
        'eped': [('engineering', r'^(ip|bt|r|a)$'), ('shape', r'^(kappa|delta|zeta|s_three|s_four|toq)'),
                 ('pedestal', r'^(neped|nesep|tesep|teped|betan|tewid|ptotwid)'), ('composition', r'^(zeffped|m|z|mi|zi)$'),
                 ('EPED settings', r'^(cfg_|stability)')],
        '_transport': [('drives', r'^(RLTS|RLNS|DLNTDR|DLNNDR|dlntdr|dlnndr|VEXB_SHEAR|VPAR_SHEAR|tprim|fprim|A[tn][ei])'),
                       ('collisions & beta', r'^(XNU|NU_|nu_|BETA|beta|DEBYE|RHO_STAR|rho|ZEFF|z_eff)'),
                       ('species', r'^(TAUS|AS_|ZS_|MASS|TEMP|DENS|Z_|temp|dens|z_|mass)'),
                       ('geometry', r'^(RMIN|RMAJ|ZMAJ|ZMAG|S_ZMAG|DRMAJDX|DZMAJDX|SHIFT|Q_|Q$|SHEAR|KAPPA|S_KAPPA|DELTA|S_DELTA|ZETA|'
                                    r'S_ZETA|SHAPE|P_PRIME|rmin|rmaj|q$|s$|kappa|delta|zeta|shift|shape)')],
    }
    # An input is flagged as discrete ("only a few values") when its distinct values are at most
    # max(_FEW_VALUES, _FEW_FRACTION * distinct values of the most-varied input of the same code): e.g.
    # geometry, which only changes between radii/evaluations while gradients change at every evaluation.
    # (Relative to the most-varied input, not to the record count: TGLF has ~13 records per plasma point
    # from the scan trick, NEO one.)
    _FEW_VALUES, _FEW_FRACTION = 10, 0.1

    def coverage(self, code, run=None):
        '''
        How much of each numeric input was explored: one row per input with its category, percentiles,
        full range, number of distinct values and `width` = (p95 - p5) / max(|p5|, |p95|) (0 = never varied,
        1 = spans from ~0 to its magnitude, 2 = symmetric about zero). Sorted by category, then width.
        '''
        import re
        df = self.load(code, run=run, with_run_info=False)
        rules = self._CATEGORIES.get(code, self._CATEGORIES['_transport'])
        if code in self._SETTINGS:
            rules = [('settings', self._SETTINGS[code])] + rules
        rows = []
        for c in df.columns:
            if not c.startswith('in_') or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            v = df[c].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            key = c[3:]
            cat = next((name for name, rx in rules if re.match(rx, key)), 'other')
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            scale = max(abs(p5), abs(p95), abs(v.min()), abs(v.max()))
            rows.append({'input': key, 'category': cat, 'min': v.min(), 'p5': p5, 'median': p50, 'p95': p95, 'max': v.max(),
                         'n_distinct': len(np.unique(np.round(v, 12))), 'scale': scale,
                         'width': (p95 - p5) / max(abs(p5), abs(p95)) if max(abs(p5), abs(p95)) > 0 else 0.0})
        cov = pd.DataFrame(rows)
        if len(cov) == 0:
            return cov
        cov.attrs['few'] = int(max(self._FEW_VALUES, self._FEW_FRACTION * cov['n_distinct'].max()))
        cov['discrete'] = (cov['n_distinct'] > 1) & (cov['n_distinct'] <= cov.attrs['few'])
        order = {name: i for i, (name, _) in enumerate(sorted(rules, key=lambda r: r[0] == 'settings'))}   # settings matched first, shown last
        cov['_o'] = cov['category'].map(lambda c: order.get(c, len(order)))
        return cov.sort_values(['_o', 'width'], ascending=[True, False]).drop(columns='_o').reset_index(drop=True)

    _CATEGORY_COLORS = {'drives': 'tab:blue', 'collisions & beta': 'tab:green', 'species': 'tab:purple', 'geometry': 'tab:brown',
                        'engineering': 'tab:blue', 'shape': 'tab:brown', 'pedestal': 'tab:green', 'composition': 'tab:purple',
                        'EPED settings': 'gray', 'settings': 'tab:red', 'other': 'gray'}

    def plotCoverage(self, code, fn=None, run=None, include_shape=False, ncols=6, max_panels=30):
        '''
        What was explored, in real units: one histogram per input that varies, titled with its [p5, p95] and
        number of distinct values; color = category; shaded background = only a few distinct values (e.g.
        geometry, which only changes between radii/evaluations). Inputs with identical values in every record
        (e.g. RLNS_1..4 under quasineutrality) share one panel; SHAPE_* Fourier coefficients are hidden unless
        include_shape=True. More than max_panels panels spill over into further tabs.
        '''
        import matplotlib.pyplot as plt
        df = self.load(code, run=run, with_run_info=False)
        cov = self.coverage(code, run=run)
        if len(cov) == 0:
            return []
        varying = cov[cov['n_distinct'] > 1]
        n_const = int((cov['n_distinct'] <= 1).sum())
        is_shape = varying['input'].str.upper().str.startswith('SHAPE')
        n_shape = int(is_shape.sum()) if not include_shape else 0
        if not include_shape:
            varying = varying[~is_shape]

        # merge inputs whose values are identical in every record
        panels, seen = [], {}
        for _, r in varying.iterrows():
            key = np.round(np.nan_to_num(df[f"in_{r['input']}"].to_numpy(dtype=float), nan=np.inf), 12).tobytes()
            if key in seen:
                seen[key]['aliases'].append(r['input'])
            else:
                seen[key] = {'row': r, 'aliases': []}
                panels.append(seen[key])
        few = cov.attrs['few']
        header = (f"{code.upper()}, {len(df)} records: {len(varying) + n_shape} inputs vary, {n_const} constant (not shown)"
                  + (f", {n_shape} SHAPE_* coefficients hidden (include_shape=True)" if n_shape else '')
                  + f". Real units; gray band and title: [p5, p95]; shaded panel = at most {few} distinct values. Color: "
                  + ", ".join(f"{c} ({self._CATEGORY_COLORS.get(c, 'gray').replace('tab:', '')})" for c in dict.fromkeys(p['row']['category'] for p in panels)))

        figs = []
        chunks = [panels[i:i + max_panels] for i in range(0, len(panels), max_panels)] or [[]]
        for ichunk, chunk in enumerate(chunks):
            label = f'{code.upper()} ranges' + (f' {ichunk + 1}/{len(chunks)}' if len(chunks) > 1 else '')
            fig = fn.add_figure(label=label) if fn is not None else plt.figure(figsize=(18, 11))
            figs.append(fig)
            if not chunk:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f'no input varies ({n_const} constant)', ha='center', va='center', transform=ax.transAxes)
                continue
            nrows = int(np.ceil(len(chunk) / ncols))
            axs = np.atleast_1d(fig.subplots(nrows, ncols)).flatten()
            for ax, p in zip(axs, chunk):
                r = p['row']
                v = df[f"in_{r['input']}"].dropna().to_numpy(dtype=float)
                ax.axvspan(r['p5'], r['p95'], color='gray', alpha=0.15, lw=0)
                ax.hist(v, bins=int(min(40, max(10, r['n_distinct']))), color=self._CATEGORY_COLORS.get(r['category'], 'gray'), alpha=0.85)
                if r['discrete']:
                    ax.set_facecolor('#fdf0dc')
                name = r['input'] + (f" (= {', '.join(p['aliases'])})" if p['aliases'] else '')
                ax.set_title(f"{name}\n[{r['p5']:.3g}, {r['p95']:.3g}]   {r['n_distinct']} values", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.set_yticks([])
                ax.grid(True, axis='x', alpha=0.3)
            for ax in axs[len(chunk):]:
                ax.axis('off')
            fig.text(0.01, 0.99, header, fontsize=8, va='top', wrap=True)
            fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.04, wspace=0.15, hspace=0.75)
        return figs

    # Key physics inputs for the pairwise coverage plot, in order ('Te'/'Ti'/'ne' = charge-resolved drives)
    _PAIRS = {
        'tglf':  ['Te', 'Ti', 'ne', 'TAUS_2', 'XNUE', 'BETAE', 'Q_LOC', 'RMIN_LOC'],
        'neo':   ['Te', 'Ti', 'ne', 'NU_1', 'RHO_STAR', 'Q', 'SHEAR', 'RMIN_OVER_A'],
        'cgyro': ['Te', 'Ti', 'ne', 'NU_EE', 'BETAE_UNIT', 'Q', 'S', 'RMIN', 'nu_ee', 'beta_star', 'q', 's', 'rmin'],
        'eped':  ['ip', 'bt', 'r', 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep'],
    }

    def plotPairs(self, code, variables=None, fn=None, run=None, max_vars=8):
        '''
        Corner plot of joint coverage for key physics inputs (histograms on the diagonal): shows whether inputs
        were varied together or only one at a time (the TGLF scan trick perturbs one input around each base point).
        `variables`: input names (without in_), or 'Te'/'Ti'/'ne' for the charge-resolved gradients; constants dropped.
        '''
        import matplotlib.pyplot as plt
        df = self.load(code, run=run, with_run_info=False)
        if len(df) == 0:
            return None
        drives = self._drives(code, df)
        names = variables or self._PAIRS.get(code)
        if names is None:
            cov = self.coverage(code, run=run)
            names = list(cov[(cov['n_distinct'] > 1) & (~cov['input'].str.startswith('SHAPE')) & (cov['category'] != 'other')]
                         .sort_values('width', ascending=False)['input'])
        cols, labels = [], []
        for n in names:
            col = drives.get(n) if n in ('Te', 'Ti', 'ne') else (f'in_{n}' if f'in_{n}' in df.columns else None)
            if col is None or col in cols or df[col].nunique() <= 1:
                continue
            cols.append(col)
            labels.append(f'{self._DRIVE_LABELS[n]} ({col[3:]})' if n in ('Te', 'Ti', 'ne') else col[3:])
        cols, labels = cols[:max_vars], labels[:max_vars]
        if fn is not None:
            fig = fn.add_figure(label=f'{code.upper()} pairs')
        else:
            fig = plt.figure(figsize=(15, 13))
        n = len(cols)
        if n < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'fewer than two varying inputs', ha='center', va='center', transform=ax.transAxes)
            return fig
        axs = fig.subplots(n, n)
        s, alpha = (2, 0.25) if len(df) > 2000 else (6, 0.6)
        for i in range(n):
            for j in range(n):
                ax = axs[i, j]
                if j > i:
                    ax.axis('off')
                    continue
                if i == j:
                    ax.hist(df[cols[i]].dropna(), bins=40, color='gray')
                    ax.tick_params(labelleft=False)
                else:
                    ax.scatter(df[cols[j]], df[cols[i]], s=s, alpha=alpha, color='tab:blue', rasterized=True)
                if i == n - 1:
                    ax.set_xlabel(labels[j], fontsize=7)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=7)
                elif j > 0:
                    ax.tick_params(labelleft=False)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)
        axs[0, 0].set_title(f'{code.upper()}: {len(df)} records, joint coverage of {n} inputs', fontsize=9, loc='left')
        fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07, wspace=0.08, hspace=0.08)
        return fig

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
        return

    fn = db.plotDatabase(codes=[args.code] if args.code else None)
    fn.show()

    # Interactive session, like the other mitim_plot_* commands: the database and one DataFrame per code are in scope
    frames = {c: db.load(c) for c in db.codes()}
    runs = db.runs()
    print("\n- Interactive session. In scope:", typeMsg='i')
    print("\t db       harvest_database  (db.load(code, columns=, run=), db.summary(), db.interpret(code), db.match_records(a, b))")
    print("\t          db.radius(code, frames[code]) r/a per record;  db.settings(code), db.match_settings(code) code settings per record")
    print("\t frames   {code: DataFrame} ->", {c: f.shape for c, f in frames.items()})
    print("\t runs     provenance table, one row per (run, code)")
    print("\t fn       the notebook;  db.plot(code, x, y, ax=fn.add_figure(label='mine').add_subplot(111)); fn.show()  adds a tab")
    from IPython import embed
    embed()


if __name__ == "__main__":
    # `python -m mitim_tools.harvest_tools.HARVESTtools push|plot ...` when the console scripts are not installed
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "plot"):
        print("usage: python -m mitim_tools.harvest_tools.HARVESTtools push <folders...> [--file F] | plot [file|folder] [--code C] [--x X --y Y] [--noplot]")
        sys.exit(2)
    cmd = sys.argv.pop(1)
    main_push() if cmd == "push" else main_plot()
