# Script created by ChatGPT 4.5
from pathlib import Path
import argparse
import re
import os
import time
import subprocess
from datetime import datetime
from IPython import embed

def clipstr(txt, chars=40):
    if not isinstance(txt, str):
        txt = f"{txt}"
    return f"{'...' if len(txt) > chars else ''}{txt[-chars:]}" if txt is not None else None

def get_squeue_by_jobid(user: str | None = None) -> dict[str, dict[str, str]]:
    """Return a mapping of SLURM jobid -> info by running `squeue` once.

    Keys in the returned dict include: state, submit_time, cores, partition.
    If `squeue` is unavailable or fails, returns an empty dict.
    """
    if user is None:
        user = os.environ.get("USER")
    if not user:
        return {}

    try:
        # %i: jobid, %T: state, %V: submit time, %C: CPUs, %P: partition
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

    # Cache SLURM queue info once (fast) instead of `squeue -j <id>` per folder (slow).
    squeue_by_jobid = get_squeue_by_jobid()

    for folder in folders:
        if not folder.is_dir():
            continue
        beats_folder = folder / 'Beats'

        if not beats_folder.exists(): #safe_exists(beats_folder):
            mod_time = datetime.fromtimestamp(folder.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            rows_failed.append((clipstr(folder, chars=chars_folder_clip), "NO BEATS", "POTENTIAL FAIL", "", f"failed on {mod_time}"))
            continue

        pattern = re.compile(r'Beat_(\d+)')
        subfolders = [d for d in beats_folder.iterdir() if d.is_dir() and pattern.match(d.name)]
        sorted_subfolders = sorted(subfolders, key=lambda d: int(pattern.match(d.name).group(1)))
        last_beat = sorted_subfolders[-1]
        run_folder = [n.name for n in last_beat.iterdir()]

        outputs_folder = folder / 'Outputs'

        txt = ''
        job_status = ''

        slurm_output = folder / 'slurm_output.dat'
        job_id = None
        if slurm_output.exists():
            with open(slurm_output, 'r') as f:
                first_line = f.readline()
                job_match = re.search(r'SLURM job (\d+)', first_line)
                if job_match:
                    job_id = job_match.group(1)
                    job_info = squeue_by_jobid.get(job_id)
                    if job_info:
                        state = job_info.get("state", "").strip()
                        submit_time = job_info.get("submit_time", "").strip()
                        cores = job_info.get("cores", "").strip()
                        partition = job_info.get("partition", "").strip()

                        # SLURM submit time formats vary by site/config; keep a safe fallback.
                        try:
                            submit_dt = datetime.strptime(submit_time, '%Y-%m-%dT%H:%M:%S')
                            time_in_queue = datetime.now() - submit_dt
                            hours = time_in_queue.days * 24 + time_in_queue.seconds // 3600
                            minutes = (time_in_queue.seconds % 3600) // 60
                            job_status = f"{state} for {hours}h {minutes}m ({cores} cores, {partition})"
                        except Exception:
                            job_status = f"{state} (submitted {submit_time}) ({cores} cores on {partition})"

        if (outputs_folder / 'beat_final').exists() and slurm_output.exists():
            mod_time = datetime.fromtimestamp((outputs_folder / 'beat_final').stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            with open(slurm_output, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if '* MAESTRO took' in line:
                    txt = line.split('* MAESTRO took')[-1].strip()
            rows_finished.append((clipstr(folder, chars=chars_folder_clip), "FINISHED", last_beat.name, txt, f"completed on {mod_time}"))
            continue

        if not job_status and not (outputs_folder / 'beat_final').exists():
            mod_time = datetime.fromtimestamp(folder.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            rows_failed.append((clipstr(folder, chars=chars_folder_clip), last_beat.name, "POTENTIAL FAIL", txt, f"failed on {mod_time}"))
            continue

        if 'run_portals' in run_folder:
            beat = 'PORTALS'
            exe_folder = last_beat / 'run_portals' / 'Execution'
            if not exe_folder.exists():
                txt = 'no execution folder yet'
            else:
                pattern_eval = re.compile(r'Evaluation.(\d+)')
                subfolders_eval = [d for d in exe_folder.iterdir() if d.is_dir() and pattern_eval.match(d.name)]
                sorted_subfolders_eval = sorted(subfolders_eval, key=lambda d: int(pattern_eval.match(d.name).group(1)))
                last_ev = sorted_subfolders_eval[-1].name.split('.')[-1]
                txt = f"last evaluation: {last_ev}"
        elif 'run_eped' in run_folder:
            beat = 'EPED'
        elif 'run_transp' in run_folder:
            beat = 'TRANSP'
        else:
            beat = 'UNKNOWN'

        rows_running.append((clipstr(folder, chars=chars_folder_clip), last_beat.name, beat, txt, job_status))

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

def safe_exists(path, retries=3, delay=0.5):
    path = Path(path)
    for _ in range(retries):
        if path.exists() or os.path.exists(str(path)):
            return True
        time.sleep(delay)
    return False

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=str, nargs="*")

    args = parser.parse_args()

    folders = args.folders

    check_cases(folders, chars_folder_clip=50)

if __name__ == "__main__":
    main()
