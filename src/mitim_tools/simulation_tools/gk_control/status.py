"""
Per-radius state of a running CGYRO submission: what one remote probe (templates/gk_control/cgyro_probe.sh)
says about every radius folder, classified into a TaskState (running, stalled, finished, ...).
"""
from dataclasses import dataclass
from enum import Enum
from mitim_tools.misc_tools.FARMINGtools import SlurmState
from mitim_tools.simulation_tools.gk_control.shell import _ShellTemplate, _PROBE_BASH
from mitim_tools.misc_tools.LOGtools import printMsg as print


def _format_wall_seconds(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


# What the coarse, one-row-per-job squeue STATE of the outer poller means for the array
# as a whole: the job is no longer in the queue as far as that poller can tell. Used only
# as a tiebreaker on top of the per-task filesystem signals.
_SLURM_JOB_GONE_STATES = frozenset({
    SlurmState.ABSENT, SlurmState.COMPLETED, SlurmState.TIMEOUT,
    SlurmState.FAILED, SlurmState.CANCELLED,
})


class TaskState(str, Enum):
    '''State of one radius, as the probe reports it and as the reclassification leaves it.'''
    NOT_STARTED = "NOT_STARTED"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    STALLED = "STALLED"
    STALLED_INIT = "STALLED_INIT"
    TIMED_OUT = "TIMED_OUT"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

    @classmethod
    def coerce(cls, token):
        try:
            return cls((token or "").strip().upper())
        except ValueError:
            return None


def seconds_remaining(cost, t, t0, max_time):
    '''
    Wall seconds a running launch still needs: the a/cs left, MAX_TIME - (t - t0), at `cost` seconds per
    a/cs (last TOTAL of out.cgyro.timing, one row per PRINT_STEP = 1 a/cs). t0 is the simulated time the
    launch started from (.mitim_t0): MAX_TIME counts from there, not from zero.
    '''
    if cost is None or t is None or max_time is None:
        return None
    return max(max_time - (t - (t0 or 0.0)), 0.0) * cost


def seconds_to_accept(cost, min_time):
    '''Wall seconds an extra case needs to reach min_time a/cs at `cost` seconds per a/cs, plus 5 min of start-up.'''
    return None if cost is None else float(min_time) * cost + 300.0


def _as_seconds(token):
    try:
        return max(0, int(token))
    except ValueError:
        return 0


def _as_float(token):
    try:
        return float(token)
    except ValueError:
        return None


@dataclass
class RadiusStatus:
    '''One radius as the remote probe saw it, reclassified against the job-wide slurm state.'''

    folder: str
    raw: TaskState
    state: TaskState
    seconds_since_update: int = 0
    seconds_since_init: int = 0
    seconds_per_step: float = None
    avg_text: str = "NA"
    steps: str = "0"
    tag: str = "-"
    exited: bool = False
    reason: str = ""
    tag_suffix: str = ""
    line: str = ""
    rescue_child: str = None
    seconds_per_step_last: float = None
    t: float = None
    t0: float = None
    max_time: float = None

    @classmethod
    def from_probe_line(cls, line, slurm_state=None, job_terminal=False, rescue_child=None):
        '''
        One `folder|state|avg|steps|wall|since_update|tag|exited[|last|t|t0|max_time]` probe line.

        Reclassification, in priority order:
            1. a terminal out.cgyro.tag token (FINISHED/TIMEOUT/ERROR), or CGYRO's EXIT line
            2. out.cgyro.timing has not been appended to for longer than stale_threshold -> STALLED
               (TIMED_OUT if the job-wide slurm state is also terminal). This is what catches slurm
               wall-clock kills: out.cgyro.info's mtime never moves after init, so "wall since init"
               alone made killed runs look identical to live ones.
            3. the job-wide slurm state is terminal while the radius still reads RUNNING -> TIMED_OUT
               (the job is already gone but the staleness window has not elapsed yet)
            4. otherwise what the probe reported.
        A radius already handed to a rescue job (`rescue_child`) no longer belongs to the array, so the
        array's slurm state (e.g. NOT FOUND after a re-attach) does not time it out.
        Returns None for a malformed line (a remote that could not stat, a truncated read).
        '''
        parts = line.strip().split("|")
        if len(parts) < 7:
            return None
        folder, state, avg, steps, wall, since_update, tag_token = parts[:7]
        raw = TaskState.coerce(state)
        status = cls(
            folder=folder,
            raw=raw,
            state=raw,
            seconds_since_update=_as_seconds(since_update),
            seconds_since_init=_as_seconds(wall),
            seconds_per_step=_as_float(avg),
            avg_text=avg,
            steps=steps,
            tag=tag_token,
            exited=(parts[7] if len(parts) > 7 else "0") == "1",
            line=line.strip(),
            rescue_child=rescue_child,
            **{name: _as_float(parts[i]) for i, name in enumerate(("seconds_per_step_last", "t", "t0", "max_time"), start=8) if len(parts) > i},
        )
        status._reclassify(slurm_state, job_terminal and rescue_child is None)
        return status

    @property
    def remaining_seconds(self):
        '''Wall seconds this launch still needs (see seconds_remaining); None while any piece is unknown.'''
        return seconds_remaining(self.seconds_per_step_last, self.t, self.t0, self.max_time)

    @property
    def stale_threshold(self):
        '''
        Seconds without an out.cgyro.timing append before this radius counts as stale: 3x its own
        step time, floored at 300 s so short I/O pauses, checkpoint writes and filesystem blips don't
        false-flag a healthy run, capped at 600 s. 180 s while the step time is still unknown.
        '''
        if self.seconds_per_step is not None and self.seconds_per_step > 0:
            return max(300, min(600, int(3 * self.seconds_per_step)))
        return 180

    def _reclassify(self, slurm_state, job_terminal):
        token = self.tag.upper() if (self.tag and self.tag != "-") else ""
        update_str = _format_wall_seconds(self.seconds_since_update)

        if token == "FINISHED" or self.exited:
            self.state = TaskState.FINISHED
            self.reason = " (out.cgyro.tag=FINISHED)" if token == "FINISHED" else " (EXIT line in out.cgyro.info)"
            return
        if token == "TIMEOUT":
            self.state, self.reason = TaskState.TIMED_OUT, " (out.cgyro.tag=TIMEOUT)"
            return
        if token == "ERROR":
            self.state, self.reason = TaskState.ERROR, " (out.cgyro.tag=ERROR)"
            return

        # Any other non-empty token (CGYRO phase indicators like "100"/"200") is informational:
        # surfaced as a suffix so the step/avg/wall detail survives instead of being replaced
        if token:
            self.tag_suffix = f" [out.cgyro.tag={self.tag}]"

        stale = self.seconds_since_update > self.stale_threshold
        if self.raw == TaskState.RUNNING and stale:
            if job_terminal:
                self.state = TaskState.TIMED_OUT
                self.reason = f" (no out.cgyro.timing update for {update_str}; slurm STATE={slurm_state})"
            else:
                self.state = TaskState.STALLED
                self.reason = (f" (no out.cgyro.timing update for {update_str}; threshold "
                               f"{self.stale_threshold}s — slurm wall-clock kill or rank crash likely)")
        elif self.raw == TaskState.INITIALIZED and stale:
            self.state = TaskState.STALLED_INIT
            self.reason = f" (no out.cgyro.timing after {update_str}; threshold {self.stale_threshold}s)"
        elif self.raw == TaskState.RUNNING and job_terminal:
            self.state = TaskState.TIMED_OUT
            self.reason = f" (slurm STATE={slurm_state})"

    def describe(self):
        '''(line, typeMsg) as the poll prints this radius.'''
        wall = _format_wall_seconds(self.seconds_since_init) if self.seconds_since_init > 0 else "—"
        update = _format_wall_seconds(self.seconds_since_update)
        detail = f"{self.steps} step(s), avg TOTAL/step = {self.avg_text}s"
        head = f"\t     {self.folder}: "
        tail = self.tag_suffix + (f" [rescue job {self.rescue_child}]" if self.rescue_child else "")
        lines = {
            TaskState.NOT_STARTED:  (f"pending — no out.cgyro.info on disk yet{tail}", ""),
            TaskState.INITIALIZED:  (f"initialized — out.cgyro.info present, awaiting out.cgyro.timing (wall since init: {wall}){tail}", ""),
            TaskState.RUNNING:      (f"running — {detail} (wall since init: {wall}, last update {update} ago){tail}", ""),
            TaskState.STALLED:      (f"stalled{self.reason} — {detail} (wall since init: {wall}){tail}", 'w'),
            TaskState.STALLED_INIT: (f"stalled at init{self.reason} — out.cgyro.timing never appeared (wall since init: {wall}){tail}", 'w'),
            TaskState.TIMED_OUT:    (f"timed out{self.reason} — {detail} (wall since init: {wall}, last update {update} ago){tail}", 'w'),
            TaskState.FINISHED:     (f"finished{self.reason} — {detail} (wall since init: {wall}){tail}", 'i'),
            TaskState.ERROR:        (f"ERROR{self.reason} — {detail} (wall since init: {wall}){tail}", 'w'),
        }
        text, type_msg = lines.get(self.state, (f"unknown state — raw='{self.line}'", ""))
        return head + text, type_msg

    def to_row(self):
        '''The plain-dict shape the rescuer and the submission metadata consume.'''
        return {
            "folder": self.folder,
            "raw_state": self.raw.value if self.raw is not None else "",
            "effective": self.state.value if self.state is not None else "",
            "since_update_i": self.seconds_since_update,
            "wall_i": self.seconds_since_init,
            "avg_f": self.seconds_per_step,
            "stale_threshold_warn": self.stale_threshold,
            "tag_token": self.tag,
            "cost_last": self.seconds_per_step_last,
            "remaining_s": self.remaining_seconds,
        }


class CgyroProbe:
    '''
    One remote pass over every radius folder of a submission (templates/gk_control/cgyro_probe.sh), in a single
    ssh round trip. The remote side averages the TOTAL column of out.cgyro.timing, counts its steps,
    and reports the mtime of out.cgyro.timing (the live-append signal) next to that of out.cgyro.info
    (the init timestamp, which never moves) plus the first token of out.cgyro.tag.
    '''

    def __init__(self, job, session=None):
        self.job = job
        self.session = session   # live ssh session to reuse; None -> one is opened for this probe

    def script(self, folders):
        return _ShellTemplate(_PROBE_BASH).substitute(
            exec_folder=self.job.folderExecution,
            folder_list=" ".join(f'"{f}"' for f in folders),
        )

    def run(self, folders):
        '''The probe's output lines, or None when the remote could not be reached.'''
        script = self.script(folders)
        try:
            if self.session is not None:
                out, _err = self.session.execute(script, printYN=False)
            else:
                with self.job.session() as session:
                    out, _err = session.execute(script, printYN=False)
        except Exception as e:
            print(f"\t- [per-task status] remote inspection failed ({e}); continuing", typeMsg='w')
            return None
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return (out or "").splitlines()
