"""
The SLURM wall of the driver itself (a PORTALS run launched with sbatch): when the driver will be killed
before its next poll while the GK job is still live, the radii it was waiting for would stay uncollected
and nobody would know. DriverWallMarker leaves <PORTALS run>/Outputs/MITIM_DRIVER_WALL_UNFINISHED.json
for the launcher, which resubmits the driver; the next driver re-attaches (check_existing_runs) and drops
the marker once an evaluation of that run is read.
"""

import datetime
import json
import os
import subprocess
from pathlib import Path

from mitim_tools.simulation_tools.SIMtools import JobStatus
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.status import TaskState
from mitim_tools.misc_tools.LOGtools import printMsg as print


# Seconds of driver wall kept on top of one poll interval before the marker is written
DRIVER_WALL_MARGIN = 300


def parse_slurm_time_left(text):
    '''
    Seconds in a squeue %L field ("D-HH:MM:SS", "HH:MM:SS", "MM:SS"); None for UNLIMITED, NOT_SET
    or anything else that is not a time.
    '''
    text = text.strip()
    days, _, clock = text.rpartition("-")
    try:
        fields = [int(x) for x in clock.split(":")]
        days = int(days) if days else 0
    except ValueError:
        return None
    if not 2 <= len(fields) <= 3:
        return None
    h, m, s = ([0] + fields)[-3:]
    return ((days * 24 + h) * 60 + m) * 60 + s


def driver_seconds_left():
    '''
    Wall seconds left to the SLURM job this process runs in (squeue -o %L, run locally); None outside
    SLURM, for an unlimited job, or when squeue fails: the caller then behaves as if time were plentiful.
    '''
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        res = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%L"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_slurm_time_left(res.stdout.strip().splitlines()[0]) if res.returncode == 0 and res.stdout.strip() else None


def portals_run_folder(folder):
    '''
    PORTALS run folder of a transport folder, <run>/Execution/Evaluation.N/transport_simulation_folder or
    <run>/Initialization/initialization_simple_relax/portals_sr_ev_N/transport_simulation_folder; None for
    any other layout.
    '''
    folder = Path(folder)
    if folder.name == "transport_simulation_folder":
        if folder.parent.parent.name == "Execution":
            return folder.parents[2]
        if folder.parent.parent.name == "initialization_simple_relax":
            return folder.parents[3]
    return None


class DriverWallMarker:
    '''
    One per GK submission. `poll` runs once per poll iteration (after the squeue of the GK job) and
    rewrites the marker while the driver has less than one poll interval + DRIVER_WALL_MARGIN left and
    the job or a child is live; `clear` removes it at the successful read of any evaluation of the same
    run (the marker is per run folder, and PORTALS evaluations of one run are sequential).
    Outside a PORTALS layout the marker sits next to the submission metadata.
    '''

    def __init__(self, sim, folder, base_subfolder):
        self.sim = sim
        self.folder = Path(folder)
        run = portals_run_folder(folder)
        self.path = (run / "Outputs" if run is not None else self.folder / base_subfolder) / markers.DRIVER_WALL_UNFINISHED
        self._warned = False

    def poll(self, every_n_minutes, rows=None):
        seconds_left = driver_seconds_left()
        if seconds_left is None or seconds_left >= every_n_minutes * 60 + DRIVER_WALL_MARGIN:
            return
        job = self.sim.simulation_job
        live_children = self.sim._live_child_jobids(alive_if_unreachable=True)
        if job.status == JobStatus.GONE and not live_children:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._payload(seconds_left, live_children, rows), indent=2))
        if not self._warned:
            self._warned = True
            print(f"\t* DRIVER WALL: {seconds_left} s left to driver job {os.environ.get('SLURM_JOB_ID')} while {self.sim.__class__.__name__} job "
                  f"{job.jobid} is live: unfinished radii will be left uncollected; wrote {self.path} for the launcher to resubmit the driver", typeMsg='w')

    def clear(self):
        if self.path.exists():
            print(f"\t- Evaluation read: removing driver-wall marker {self.path}", typeMsg='i')
        self.path.unlink(missing_ok=True)

    def _payload(self, seconds_left, live_children, rows):
        job = self.sim.simulation_job
        return {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "driver_seconds_left": seconds_left,
            "evaluation folder": str(self.folder),
            "array jobid": job.jobid,
            "array state": job.infoSLURM.get("STATE"),
            "child jobids": live_children,
            "unfinished radii": [{"folder": row["folder"], "state": row["effective"]}
                                 for row in (rows or []) if row["effective"] != TaskState.FINISHED.value],
            "note": "The driver hit its SLURM wall before this GK job finished. Resubmit the same driver with "
                    "check_existing_runs: True: it re-attaches to the job (or reads the finished radii), and "
                    "removes this file once the evaluation is read. 'unfinished radii' is empty when no per-radius "
                    "probe ran (non-CGYRO backend or probe failure).",
        }
