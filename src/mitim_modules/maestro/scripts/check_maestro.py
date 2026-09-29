from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
from datetime import datetime

from mitim_tools.misc_tools.IOtools import createTimeTXT

# Compiled once, reused for every folder
_RE_BEAT       = re.compile(r'Beat_(\d+)')
_RE_EVAL       = re.compile(r'Evaluation\.(\d+)')
_RE_SBATCH_JOB = re.compile(r'Submitted batch job (\S+)')
_RE_SLURM_JOB  = re.compile(r'SLURM job (\S+)')
_RE_TOOK       = re.compile(r'\* MAESTRO took(.+)')
# SLURM cancellation line written to the job's error file, e.g.
#   *** JOB 15880485 ON node2431 CANCELLED AT 2026-06-11T17:03:03 DUE TO PREEMPTION ***
# The "DUE TO <reason>" part is optional (plain scancel does not write it).
_RE_CANCELLED  = re.compile(r'JOB\s+(\S+)\s+ON\s+\S+\s+CANCELLED\s+AT\s+(\S+?)(?:\s+DUE TO\s+([^*]+?))?\s*\*')
# Strip trailing "(... ms)" milisec suffix that createTimeTXT always appends.
# The closing paren is optional because createTimeTXT itself has a `txt[:-1]`
# at the end that lops off the trailing ')' when no fractional unit was added.
_RE_MILISEC    = re.compile(r'\s*\([^)]*ms\)?\s*$')

# Tail chunk size for reading log files (bytes) — "MAESTRO took" is always near the end
_LOG_TAIL_BYTES = 4096

# Worker cap: more than this on a shared NFS login node tends to contend rather than help
_DEFAULT_WORKERS = 8

_COLORS = {
    "PORTALS":        "\033[31m",  # red
    "EPED":           "\033[35m",  # magenta
    "TRANSP":         "\033[33m",  # yellow
    "SHARPNESS":      "\033[95m",  # bright magenta
    "CONFINEMENT":    "\033[94m",  # bright blue
    "LENGYEL":        "\033[96m",  # bright cyan
    "PENDING":        "\033[36m",  # cyan
    "UNKNOWN":        "\033[34m",  # blue
    "FINISHED":       "\033[32m",  # green
    "POTENTIAL FAIL": "\033[91m",  # bright red
    "FAILED":         "\033[91m",  # bright red
}
_RESET = "\033[0m"


def clipstr(txt, chars=40):
    if not isinstance(txt, str):
        txt = f"{txt}"
    return f"{'...' if len(txt) > chars else ''}{txt[-chars:]}"


def _stat_or_none(path):
    try:
        return os.stat(path)
    except OSError:
        return None


def _scandir_names(path):
    """Return list of (name, is_dir) for entries in path. Empty on OSError."""
    try:
        with os.scandir(path) as it:
            return [(e.name, e.is_dir()) for e in it]
    except OSError:
        return []


def _read_first_line(path):
    try:
        with open(path, 'r') as f:
            return f.readline()
    except OSError:
        return ''


def _read_tail(path, nbytes=_LOG_TAIL_BYTES):
    """Return the last `nbytes` of a file as a string without reading the whole file."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def _strip_milisec(txt):
    """Drop the noisy '(NNN ms)' suffix createTimeTXT always appends."""
    return _RE_MILISEC.sub('', txt).strip()


def _extract_took(*candidate_paths):
    """Scan a few candidate log files for the 'MAESTRO took ...' line. First hit wins."""
    for p in candidate_paths:
        tail = _read_tail(p)
        if not tail:
            continue
        m = _RE_TOOK.search(tail)
        if m:
            return _strip_milisec(m.group(1))
    return ''


def _elapsed_from_timing_jsonl(timing_path):
    """Sum duration_s across all records in a mitim_timer JSONL ledger.

    Returns the formatted duration string, or '' if missing/unreadable.
    Note: accumulates across resumed runs."""
    try:
        total = 0.0
        with open(timing_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                dur = rec.get('duration_s')
                if isinstance(dur, (int, float)):
                    total += dur
    except OSError:
        return ''
    if total <= 0:
        return ''
    return _strip_milisec(createTimeTXT(total))


def _elapsed_text(folder_str):
    """Best-effort elapsed time for a finished MAESTRO run, or ''.

    Prefers the '* MAESTRO took' log line (the last invocation's wall time);
    falls back to a sum of Outputs/Performance/timing.jsonl (cumulative across
    resumed runs, so it's an upper bound when checkpoints were used)."""
    txt = _extract_took(
        folder_str + '/Outputs/maestro.log',
        folder_str + '/slurm_output.dat',
        folder_str + '/Outputs/beat_final',
    )
    if txt:
        return txt
    return _elapsed_from_timing_jsonl(folder_str + '/Outputs/Performance/timing.jsonl')


def get_squeue_by_jobid(user: str | None = None) -> dict[str, dict[str, str]]:
    """Return a mapping of SLURM jobid -> info by running `squeue` once."""
    if user is None:
        user = os.environ.get("USER")
    if not user:
        return {}

    try:
        squeue_out = subprocess.run(
            # -r: one row per job-array element, so PENDING elements report their
            # full <job>_<task> id instead of the aggregated <job>_[lo-hi%N] row,
            # which never matches the per-case stub ids from run_maestro_scan.
            # %j (job name) enables the fallback match for cases whose recorded
            # job id went stale (e.g. manual resubmissions).
            ["squeue", "-u", user, "-o", "%i|%T|%V|%S|%C|%P|%j", "-h", "-r"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if squeue_out.returncode != 0 or not squeue_out.stdout:
        return {}

    jobs: dict[str, dict[str, str]] = {}
    for raw_line in squeue_out.stdout.splitlines():
        parts = raw_line.strip().split("|")
        if len(parts) != 7:
            continue
        job_id, state, submit_time, start_time, cores, partition, name = (p.strip() for p in parts)
        if job_id:
            jobs[job_id] = {
                "state": state,
                "submit_time": submit_time,
                "start_time": start_time,
                "cores": cores,
                "partition": partition,
                "name": name,
            }
    return jobs


def _slurm_cancellation(folder_str, tracked_jobid=None):
    """Look for a SLURM cancellation notice (preemption, time limit, scancel)
    in the tail of slurm_error.dat. Returns {'jobid', 'when', 'reason'} for the
    LAST such notice, or None. When tracked_jobid is given, notices from a
    different (older) job id are ignored."""
    tail = _read_tail(folder_str + '/slurm_error.dat')
    if not tail:
        return None
    match = None
    for match in _RE_CANCELLED.finditer(tail):
        pass
    if match is None:
        return None
    jobid, when, reason = match.group(1), match.group(2), match.group(3)
    if tracked_jobid and jobid.split('_')[0] != tracked_jobid.split('_')[0]:
        return None
    return {
        'jobid': jobid,
        'when': when,
        'reason': reason.strip() if reason else None,
    }


def _cancellation_text(cancellation):
    txt = f"cancelled at {cancellation['when']}"
    if cancellation['reason']:
        txt += f" due to {cancellation['reason']}"
    return txt


def _job_status_from_squeue(folder_str, squeue_by_jobid):
    """Return (job_status_str, job_state, job_id) for the folder, or ('', None, job_id)
    if no live job (job_id may still be known from the submission logs)."""
    slurm_sbatch_path = folder_str + '/sbatch_submission.log'
    slurm_output_path = folder_str + '/slurm_output.dat'

    job_match = None
    sbatch_line = _read_first_line(slurm_sbatch_path)
    if sbatch_line:
        job_match = _RE_SBATCH_JOB.search(sbatch_line)
    else:
        output_line = _read_first_line(slurm_output_path)
        if output_line:
            job_match = _RE_SLURM_JOB.search(output_line)

    if not job_match:
        job_id = None
        job_info = None
    else:
        job_id = job_match.group(1)
        job_info = squeue_by_jobid.get(job_id)

    matched_by_name = False
    if not job_info:
        # Fallback: the recorded job id is stale (e.g. the case was manually
        # resubmitted, so a new id is in the queue but no log was refreshed).
        # MAESTRO jobs are deterministically named mitim_<folder name> (with an
        # optional _c<N> suffix for chained wall-time chunks), so match by name
        # in the same squeue snapshot; latest submission wins.
        target = f"mitim_{Path(folder_str).name}"
        candidates = [
            info for info in squeue_by_jobid.values()
            if info.get("name") == target or info.get("name", "").startswith(target + "_c")
        ]
        if candidates:
            job_info = max(candidates, key=lambda i: i.get("submit_time", ""))
            matched_by_name = True

    if not job_info:
        return '', None, job_id

    state = job_info["state"]
    submit_time = job_info["submit_time"]
    start_time = job_info["start_time"]
    cores = job_info["cores"]
    partition = job_info["partition"]

    # For RUNNING jobs, measure elapsed from StartTime; otherwise from SubmitTime
    # (so pending jobs still show queue wait). Slurm uses "N/A"/"Unknown" pre-start.
    use_start = state.upper() == "RUNNING" and start_time not in ("", "N/A", "Unknown")
    ref_time = start_time if use_start else submit_time

    name_note = " [recorded jobid stale; matched by job name]" if matched_by_name else ""

    try:
        ref_dt = datetime.strptime(ref_time, '%Y-%m-%dT%H:%M:%S')
        delta = datetime.now() - ref_dt
        hours = delta.days * 24 + delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        return f"{state} for {hours}h {minutes}m ({cores} cores, {partition}){name_note}", state, job_id
    except Exception:
        return f"{state} (submitted {submit_time}) ({cores} cores on {partition}){name_note}", state, job_id


def _classify_folder(folder, squeue_by_jobid, chars_folder_clip, show_full_path):
    """Inspect one MAESTRO folder. Return ('running'|'finished'|'failed', row_tuple)."""
    folder_str = str(folder)
    display_name = str(folder) if show_full_path else folder.name
    folder_display = clipstr(display_name, chars=chars_folder_clip)

    job_status, job_state, job_id = _job_status_from_squeue(folder_str, squeue_by_jobid)

    # Surface SLURM cancellations (preemption, time limit, scancel) recorded in
    # slurm_error.dat. On preemptable partitions the job is requeued under the
    # SAME id, so squeue alone shows an innocent PENDING/RUNNING — annotate it.
    cancellation = _slurm_cancellation(folder_str, tracked_jobid=job_id)
    if cancellation and job_status:
        job_status += f" [{_cancellation_text(cancellation)}; requeued]"

    # Last beat detected by scanning Beats/
    beats_folder_path = folder_str + '/Beats'
    beat_entries = [(name, isdir) for name, isdir in _scandir_names(beats_folder_path)
                    if isdir and _RE_BEAT.match(name)]
    last_beat = None
    run_names = []
    if beat_entries:
        beat_entries.sort(key=lambda t: int(_RE_BEAT.match(t[0]).group(1)))
        last_beat_name_only = beat_entries[-1][0]
        last_beat = Path(beats_folder_path) / last_beat_name_only
        run_names = [name for name, _ in _scandir_names(str(last_beat))]

    last_beat_name = last_beat.name if last_beat is not None else "NO BEATS"

    # ---- FINISHED: true success marker is Outputs/input.gacode_final (written
    # unconditionally by MAESTRO.finalize(), independent of --terminal).
    final_gacode_path = folder_str + '/Outputs/input.gacode_final'
    final_gacode_stat = _stat_or_none(final_gacode_path)
    if final_gacode_stat is not None:
        mod_time = datetime.fromtimestamp(final_gacode_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        took = _elapsed_text(folder_str)
        details = f"took {took}" if took else ''
        return 'finished', (folder_display, "FINISHED", last_beat_name, details, f"completed on {mod_time}")

    # ---- Determine beat type via scandir results (already in memory)
    txt  = ''
    beat = 'UNKNOWN'
    if last_beat is not None:
        if 'run_portals' in run_names:
            beat = 'PORTALS'
            exe_path = str(last_beat) + '/run_portals/Execution'
            eval_entries = [(name, isdir) for name, isdir in _scandir_names(exe_path)
                            if isdir and _RE_EVAL.match(name)]
            if not eval_entries:
                txt = 'no execution folder yet'
            else:
                eval_entries.sort(key=lambda t: int(_RE_EVAL.match(t[0]).group(1)))
                txt = f"last evaluation: {eval_entries[-1][0].split('.')[-1]}"
        elif 'run_eped' in run_names:
            beat = 'EPED'
        elif 'run_transp' in run_names:
            beat = 'TRANSP'
        elif 'run_lengyel' in run_names:
            beat = 'LENGYEL'
        elif 'run_sharpness' in run_names:   # legacy pre-'bc' folder naming
            beat = 'SHARPNESS'
        elif 'run_confinement' in run_names:  # legacy pre-'bc' folder naming
            beat = 'CONFINEMENT'
        else:
            bc_runs = [n for n in run_names if n.startswith('run_bc_')]
            if bc_runs:
                beat = bc_runs[0].removeprefix('run_').upper()   # e.g. BC_CONFINEMENT

    if job_state and job_state.upper() in {"PENDING", "PD"}:
        details = f"{beat}{(' - ' + txt) if txt else ''}" if (beat != 'UNKNOWN' or txt) else ''
        return 'running', (folder_display, last_beat_name, "PENDING", details, job_status or "PENDING")

    # No live job + an explicit SLURM cancellation on record -> definite failure with reason
    def _failure_reason(folder_str):
        if cancellation:
            return _cancellation_text(cancellation)
        mod_time = datetime.fromtimestamp(os.stat(folder_str).st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return f"failed on {mod_time}"

    if last_beat is None:
        if job_status:
            return 'running', (folder_display, "NO BEATS", "UNKNOWN", "", job_status)
        fail_type = "FAILED" if cancellation else "POTENTIAL FAIL"
        return 'failed', (folder_display, "NO BEATS", fail_type, "", _failure_reason(folder_str))

    if not job_status:
        fail_type = "FAILED" if cancellation else "POTENTIAL FAIL"
        return 'failed', (folder_display, last_beat_name, fail_type, txt, _failure_reason(folder_str))

    return 'running', (folder_display, last_beat_name, beat, txt, job_status)


def _expand_folders(folders):
    """Glob-expand patterns into a deduplicated list of directories."""
    expanded = []
    for pattern in folders:
        pattern_path = Path(pattern).expanduser()
        name_pattern = pattern_path.name

        has_glob = any(ch in name_pattern for ch in ("*", "?", "["))
        if has_glob:
            parent = pattern_path.parent
            parent = parent if parent != Path(".") else Path.cwd()
            expanded.extend([p for p in parent.glob(name_pattern) if p.is_dir()])
        else:
            candidate = pattern_path
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.is_dir():
                expanded.append(candidate)

    out, seen = [], set()
    for p in expanded:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def check_cases(folders, chars_folder_clip=500, max_workers=_DEFAULT_WORKERS, show_full_path=False, user=None):

    folders = _expand_folders(folders)
    if not folders:
        print("No MAESTRO folders matched.")
        return

    # One squeue call covers every folder
    squeue_by_jobid = get_squeue_by_jobid(user=user)

    rows_running, rows_finished, rows_failed = [], [], []
    bucket = {'running': rows_running, 'finished': rows_finished, 'failed': rows_failed}

    workers = max(1, min(max_workers, len(folders)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(
            lambda f: _classify_folder(f, squeue_by_jobid, chars_folder_clip, show_full_path),
            folders,
        )
        for category, row in results:
            bucket[category].append(row)

    header = ("Folder", "Last Beat", "Type", "Details", "Job Status")
    all_rows = rows_running + rows_finished + rows_failed
    col_widths = [max(len(row[i]) for row in all_rows + [header]) for i in range(5)]

    summary = (f"Total: {len(folders)}  "
               f"(running: {len(rows_running)}, "
               f"finished: {len(rows_finished)}, "
               f"failed: {len(rows_failed)})")
    print(summary)

    header_line = (f"{header[0]:<{col_widths[0]}} - {header[1]:<{col_widths[1]}} - "
                   f"{header[2]:<{col_widths[2]}} - {header[3]:<{col_widths[3]}} - "
                   f"{header[4]:<{col_widths[4]}}")
    print(header_line)
    print("-" * len(header_line))

    def print_block(rows, title):
        if not rows:
            return
        print(f"\n===== {title} ({len(rows)} cases) =====")
        for row in rows:
            beat_type = row[2] if row[2] else title
            if title in ("FINISHED", "FAILED"):
                beat_type = title
            color = _COLORS.get(beat_type, "")
            line = (f"{row[0]:<{col_widths[0]}} - {row[1]:<{col_widths[1]}} - "
                    f"{row[2]:<{col_widths[2]}} - {row[3]:<{col_widths[3]}} - "
                    f"{row[4]:<{col_widths[4]}}")
            print(f"{color}{line}{_RESET}")

    print_block(rows_running, "RUNNING")
    print_block(rows_finished, "FINISHED")
    print_block(rows_failed, "FAILED")
    print('')


def main():
    parser = argparse.ArgumentParser(description="Quick status check for MAESTRO run folders.")
    parser.add_argument("folders", type=str, nargs="*",
                        help="One or more MAESTRO folders (globs allowed).")
    parser.add_argument("--full-path", action="store_true",
                        help="Show full folder paths instead of basenames.")
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                        help=f"Parallel folder scans (default {_DEFAULT_WORKERS}).")
    parser.add_argument("--user", type=str, default=None,
                        help="SLURM user to query (default $USER).")
    parser.add_argument("--clip", type=int, default=50,
                        help="Max chars for the folder column (default 50).")

    args = parser.parse_args()

    check_cases(
        args.folders,
        chars_folder_clip=args.clip,
        max_workers=args.workers,
        show_full_path=args.full_path,
        user=args.user,
    )


if __name__ == "__main__":
    main()
