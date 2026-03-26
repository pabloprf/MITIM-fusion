# Script created by ChatGPT 4.5
from pathlib import Path
import argparse
import re
import os
import subprocess
from datetime import datetime
from IPython import embed
from mitim_tools.opt_tools.scripts import slurm

# Compiled once, reused for every folder
_RE_BEAT       = re.compile(r'Beat_(\d+)')
_RE_EVAL       = re.compile(r'Evaluation\.(\d+)')
_RE_SBATCH_JOB = re.compile(r'Submitted batch job (\S+)')
_RE_SLURM_JOB  = re.compile(r'SLURM job (\S+)')
_RE_TOOK       = re.compile(r'\* MAESTRO took(.+)')

# Tail chunk size for reading log files (bytes) — "MAESTRO took" is always near the end
_LOG_TAIL_BYTES = 4096

def clipstr(txt, chars=40):
    if not isinstance(txt, str):
        txt = f"{txt}"
    return f"{'...' if len(txt) > chars else ''}{txt[-chars:]}" if txt is not None else None

def _stat_or_none(path):
    """Single syscall: return os.stat result or None if path doesn't exist."""
    try:
        return os.stat(path)
    except OSError:
        return None

def _scandir_names(path):
    """Return list of (name, is_dir) for entries in path using scandir (one syscall per entry saved)."""
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

def get_squeue_by_jobid(user: str | None = None) -> dict[str, dict[str, str]]:
    """Return a mapping of SLURM jobid -> info by running `squeue` once."""
    if user is None:
        user = os.environ.get("USER")
    if not user:
        return {}

    try:
        squeue_out = subprocess.run(
            ["squeue", "-u", user, "-o", "%i|%T|%V|%C|%P", "-h"],
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
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        job_id, state, submit_time, cores, partition = parts
        job_id = job_id.strip()
        if not job_id:
            continue
        jobs[job_id] = {
            "state": state.strip(),
            "submit_time": submit_time.strip(),
            "cores": cores.strip(),
            "partition": partition.strip(),
        }
    return jobs

def check_cases(folders, chars_folder_clip=500):

    colors = {
        "PORTALS": "\033[31m",        # red
        "EPED": "\033[35m",           # magenta
        "TRANSP": "\033[33m",         # yellow
        "PENDING": "\033[36m",        # cyan
        "UNKNOWN": "\033[34m",        # blue
        "FINISHED": "\033[91m",       # bright red
        "POTENTIAL FAIL": "\033[91m", # bright red
        "FAILED": "\033[91m",         # bright red
    }
    RESET = "\033[0m"

    folders_clean = []
    for pattern in folders:
        pattern_path = Path(pattern).expanduser()
        name_pattern = pattern_path.name

        has_glob = any(ch in name_pattern for ch in ("*", "?", "["))
        if has_glob:
            parent = pattern_path.parent
            parent = parent if parent != Path(".") else Path.cwd()
            folders_clean.extend([p for p in parent.glob(name_pattern) if p.is_dir()])
        else:
            candidate = pattern_path
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.is_dir():
                folders_clean.append(candidate)

    # De-duplicate while preserving order
    folders = []
    seen: set[str] = set()
    for p in folders_clean:
        key = str(p)
        if key not in seen:
            seen.add(key)
            folders.append(p)

    rows_running = []
    rows_finished = []
    rows_failed = []

    header = ("Folder", "Last Beat", "Type", "Details", "Job Status")

    # One squeue call for all jobs
    squeue_by_jobid = get_squeue_by_jobid()

    for folder in folders:
        folder_str = str(folder)

        # --- Job ID from slurm files (read first line only) ---
        job_match = None
        slurm_output_path    = folder_str + '/slurm_output.dat'
        slurm_sbatch_path    = folder_str + '/sbatch_submission.log'
        slurm_output_stat    = None

        sbatch_line = _read_first_line(slurm_sbatch_path)
        if sbatch_line:
            job_match = _RE_SBATCH_JOB.search(sbatch_line)
        else:
            slurm_output_stat = _stat_or_none(slurm_output_path)   # stat cached for reuse below
            if slurm_output_stat:
                job_match = _RE_SLURM_JOB.search(_read_first_line(slurm_output_path))

        job_status = ''
        job_state  = None
        if job_match:
            job_id   = job_match.group(1)
            job_info = squeue_by_jobid.get(job_id)
            if job_info:
                state        = job_info["state"].strip()
                job_state    = state
                submit_time  = job_info["submit_time"].strip()
                cores        = job_info["cores"].strip()
                partition    = job_info["partition"].strip()
                try:
                    submit_dt      = datetime.strptime(submit_time, '%Y-%m-%dT%H:%M:%S')
                    time_in_queue  = datetime.now() - submit_dt
                    hours          = time_in_queue.days * 24 + time_in_queue.seconds // 3600
                    minutes        = (time_in_queue.seconds % 3600) // 60
                    job_status     = f"{state} for {hours}h {minutes}m ({cores} cores, {partition})"
                except Exception:
                    job_status = f"{state} (submitted {submit_time}) ({cores} cores on {partition})"

        # --- Last beat: use scandir so is_dir() costs no extra syscall ---
        beats_folder_path = folder_str + '/Beats'
        last_beat     = None
        run_names: list[str] = []

        beat_entries = [(name, isdir) for name, isdir in _scandir_names(beats_folder_path)
                        if isdir and _RE_BEAT.match(name)]
        if beat_entries:
            beat_entries.sort(key=lambda t: int(_RE_BEAT.match(t[0]).group(1)))
            last_beat_name_only = beat_entries[-1][0]
            last_beat = Path(beats_folder_path) / last_beat_name_only
            run_names = [name for name, _ in _scandir_names(str(last_beat))]

        last_beat_name = last_beat.name if last_beat is not None else "NO BEATS"
        folder_clip    = clipstr(folder, chars=chars_folder_clip)

        # --- Finished check: reuse slurm_output_stat if we already have it ---
        beat_final_path = folder_str + '/Outputs/beat_final'
        beat_final_stat = _stat_or_none(beat_final_path)

        if beat_final_stat is not None:
            if slurm_output_stat is None:
                slurm_output_stat = _stat_or_none(slurm_output_path)
            if slurm_output_stat is not None:
                mod_time = datetime.fromtimestamp(beat_final_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                # Read only the tail — "MAESTRO took" is always at the end
                txt = ''
                tail = _read_tail(slurm_output_path)
                m = _RE_TOOK.search(tail)
                if m:
                    txt = m.group(1).strip()
                rows_finished.append((folder_clip, "FINISHED", last_beat_name, txt, f"completed on {mod_time}"))
                continue

        # --- Determine beat type via scandir results (already in memory) ---
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

        if job_state and job_state.upper() in {"PENDING", "PD"}:
            details = f"{beat}{(' - ' + txt) if txt else ''}" if (beat != 'UNKNOWN' or txt) else ''
            rows_running.append((folder_clip, last_beat_name, "PENDING", details, job_status or "PENDING"))
            continue

        if last_beat is None:
            if job_status:
                rows_running.append((folder_clip, "NO BEATS", "UNKNOWN", "", job_status))
                continue
            mod_time = datetime.fromtimestamp(os.stat(folder_str).st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            rows_failed.append((folder_clip, "NO BEATS", "POTENTIAL FAIL", "", f"failed on {mod_time}"))
            continue

        if not job_status and beat_final_stat is None:
            mod_time = datetime.fromtimestamp(os.stat(folder_str).st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            rows_failed.append((folder_clip, last_beat_name, "POTENTIAL FAIL", txt, f"failed on {mod_time}"))
            continue

        rows_running.append((folder_clip, last_beat_name, beat, txt, job_status))

    # Recalculate col_widths after adding labels to ensure proper alignment
    all_rows = rows_running + rows_finished + rows_failed
    col_widths = [max(len(row[i]) for row in all_rows + [header]) for i in range(5)]

    header_line = f"{header[0]:<{col_widths[0]}} - {header[1]:<{col_widths[1]}} - {header[2]:<{col_widths[2]}} - {header[3]:<{col_widths[3]}} - {header[4]:<{col_widths[4]}}"
    print(header_line)
    print("-" * len(header_line))

    def print_block(rows, title):
        if rows:
            print(f"\n===== {title} ({len(rows)} cases) =====")
            for row in rows:
                beat_type = row[2] if row[2] else title
                if title in ["FINISHED", "FAILED"]:
                    beat_type = title
                color = colors.get(beat_type, "")
                line = f"{row[0]:<{col_widths[0]}} - {row[1]:<{col_widths[1]}} - {row[2]:<{col_widths[2]}} - {row[3]:<{col_widths[3]}} - {row[4]:<{col_widths[4]}}"
                print(f"{color}{line}{RESET}")

    print_block(rows_running, "RUNNING")
    print_block(rows_finished, "FINISHED")
    print_block(rows_failed, "FAILED")
    print('')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")

    args = parser.parse_args()

    check_cases(args.folders, chars_folder_clip=50)

if __name__ == "__main__":
    main()
