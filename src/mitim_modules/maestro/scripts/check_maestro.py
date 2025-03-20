# Script created by ChatGPT 4.5

from pathlib import Path
import argparse
import re
import subprocess
from datetime import datetime
import fnmatch
import os

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

parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="*")

args = parser.parse_args()

folders_clean = []
for pattern in args.folders:
    parent = Path(pattern).parent
    parent = parent if parent != Path('.') else Path.cwd()
    matched_folders = [f for f in parent.iterdir() if fnmatch.fnmatch(f.name, Path(pattern).name)]
    folders_clean.extend(matched_folders)

folders = folders_clean

rows_running = []
rows_finished = []
rows_failed = []

header = ("Folder", "Last Beat", "Type", "Details", "Job Status")

for folder in folders:
    beats_folder = folder / 'Beats'

    if not beats_folder.exists():
        mod_time = datetime.fromtimestamp(folder.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        rows_failed.append((str(folder), "NO BEATS", "POTENTIAL FAIL", "", f"failed on {mod_time}"))
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
                squeue_out = subprocess.run(["squeue", "-j", job_id, "-o", "%T|%V|%C|%P", "-h"],
                                            capture_output=True, text=True)
                if squeue_out.stdout.strip():
                    state, submit_time, cores, partition = squeue_out.stdout.strip().split('|')
                    submit_dt = datetime.strptime(submit_time, '%Y-%m-%dT%H:%M:%S')
                    time_in_queue = datetime.now() - submit_dt
                    hours = time_in_queue.days * 24 + time_in_queue.seconds // 3600
                    minutes = (time_in_queue.seconds % 3600) // 60
                    job_status = f"{state.strip()} for {hours}h {minutes}m ({cores} cores on {partition})"

    if (outputs_folder / 'beat_final').exists():
        mod_time = datetime.fromtimestamp((outputs_folder / 'beat_final').stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        with open(slurm_output, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if '* MAESTRO took' in line:
                txt = line.split('* MAESTRO took')[-1].strip()
        rows_finished.append((str(folder), "FINISHED", last_beat.name, txt, f"completed on {mod_time}"))
        continue

    if not job_status and not (outputs_folder / 'beat_final').exists():
        mod_time = datetime.fromtimestamp(folder.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        rows_failed.append((str(folder), last_beat.name, "POTENTIAL FAIL", txt, f"failed on {mod_time}"))
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

    rows_running.append((str(folder), last_beat.name, beat, txt, job_status))

# Recalculate col_widths after adding labels to ensure proper alignment
all_rows = rows_running + rows_finished + rows_failed
col_widths = [max(len(row[i]) for row in all_rows + [header]) for i in range(5)]

header_line = f"{header[0]:<{col_widths[0]}} - {header[1]:<{col_widths[1]}} - {header[2]:<{col_widths[2]}} - {header[3]:<{col_widths[3]}} - {header[4]:<{col_widths[4]}}"
print(header_line)
print("-" * len(header_line))

def print_block(rows, title):
    if rows:
        print(f"\n===== {title} =====")
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