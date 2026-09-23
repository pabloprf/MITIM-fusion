import os
import shutil
import string
import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as plt
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools.utils import GACODEdefaults, CGYROutils
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.utils import SIMplot
from mitim_tools.misc_tools import GRAPHICStools, CONFIGread
from mitim_tools.misc_tools.FARMINGtools import SlurmState
from mitim_tools.gacode_tools.utils import GACODEplotting
from mitim_tools.misc_tools.LOGtools import printMsg as print


def _annotate_missing(ax, reason):
    '''
    Stamp a small "data unavailable" note on an axes when the underlying
    CGYRO output files weren't written or weren't retrieved (e.g.
    MOMENT_PRINT_FLAG=0 drops kxky_n/e/v; FIELD_PRINT_FLAG=0 drops
    kxky_apar/bpar). Keeps the surrounding title/labels intact so the reader
    sees which panel was supposed to be there.
    '''
    ax.text(
        0.5, 0.5, f"Data unavailable\n({reason})",
        ha='center', va='center', transform=ax.transAxes,
        color='gray', style='italic', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.4),
    )


def _format_wall_seconds(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def body_keeping_exit_status(pre_cmd, main_cmd, cleanup_cmd, verdict_cmd=""):
    '''
    Shell body running pre_cmd, main_cmd, cleanup_cmd in that order, whose exit status is
    main_cmd's. Without this the trailing cleanup (an rm) set it, so a CGYRO killed by a
    signal or crashed mid-run was logged by the in-allocation scheduler as rc=0.
    `(exit $rc)` sets the status without ending the shell, so anything a launcher appends
    after this body (slurm_array script, bash `{ ... } &` group) still runs.
    Note: the status is only as good as what the launcher chain propagates, and the
    wall-budget watchdog stops runs on purpose - completion is judged from CGYRO's own EXIT
    line (run_specifications["completion_marker"]), not from this code.
    verdict_cmd runs right after main_cmd and may rewrite _mitim_rc (gacode's `cgyro` script
    exits 0 even when the executable crashed, see CgyroLaunchBody.exit_verdict).
    '''
    return (pre_cmd + "\n" + main_cmd.rstrip("\n") + "\n_mitim_rc=$?\n"
            + (verdict_cmd + "\n" if verdict_cmd else "")
            + cleanup_cmd + "\n(exit $_mitim_rc)\n")


# ----------------------------------------------------------------------------------------------------
# The bash MITIM wraps around every CGYRO launch and poll lives in templates/ so it can be read and
# linted as shell. `@{name}` is the placeholder delimiter, so the shell's own `$` survives verbatim.
# ----------------------------------------------------------------------------------------------------

class _ShellTemplate(string.Template):
    delimiter = "@"


def _shell_text(name):
    return (__mitimroot__ / "templates" / name).read_text()


_WATCHDOG_BASH = _shell_text("cgyro_watchdog.sh")
_PROBE_BASH = _shell_text("cgyro_probe.sh")


class Watchdog:
    '''
    Supervisor around one radial launch (templates/cgyro_watchdog.sh), in its own process group.
    Every launch honors a mitim_stop file in its folder (dropped by the in-allocation scheduler or by
    `mitim_kill_cgyro`): it waits for the next restart write, so the blob a later iteration warm-starts
    from is whole, leaves mitim_budget.tag (accepted as finished by radius_finished) and stops CGYRO.
        MANUAL (main radii, default): stop only on request, at any simulated time.
        BUDGET (main radii, load_balance strategy 'wall_budget'): also stop once the wall budget is
               spent AND out.cgyro.time shows >= min_time a/cs.
        STOP   (scheduler extras): a request below min_time discards the case (mitim_discard.tag)
               instead. Main radii are never discarded: the evaluation needs them.
    '''

    MANUAL, BUDGET, STOP = "manual", "budget", "stop"

    def __init__(self, rho_dir, mode=None, min_time=0.0, minutes_per_call=None, template=None):
        self.rho_dir = rho_dir
        self.mode = mode or self.MANUAL
        self.min_time = float(min_time)
        self.template = _WATCHDOG_BASH if template is None else template
        if self.mode == self.BUDGET and minutes_per_call is None:
            raise ValueError("[MITIM] load_balance strategy 'wall_budget' needs 'minutes_per_call' (the wall minutes each radial call gets)")
        self.budget_s = int(float(minutes_per_call) * 60) if self.mode == self.BUDGET else 0

    @classmethod
    def from_load_balance(cls, rho_dir, load_balance, mode=None, template=None):
        '''mode None picks BUDGET when the load_balance strategy asks for it, else MANUAL.'''
        lb = load_balance or {}
        if mode is None:
            mode = cls.BUDGET if lb.get("strategy") == "wall_budget" else cls.MANUAL
        return cls(rho_dir, mode=mode, min_time=lb.get("min_time", 0.0),
                   minutes_per_call=lb.get("minutes_per_call"), template=template)

    def wrap(self, cmd):
        return _ShellTemplate(self.template).substitute(
            rhodir=self.rho_dir,
            budget_s=self.budget_s,
            min_time=f"{self.min_time:g}",
            discard="1" if self.mode == self.STOP else "0",
            cgyro_cmd=cmd.rstrip("\n"),
        )


class CgyroLaunchBody:
    '''
    Bash body of one radial CGYRO call: the environment every MPI rank inherits, the launch shape the
    machine needs, and the marker/cleanup files the rest of MITIM reads back. `build(watchdog)`
    assembles them into the body SIMtools' JobScript builders stage.
    '''

    def __init__(self, folder, p, n=1, additional_command="", resolved=None, cpus_per_node=1):
        from mitim_tools.misc_tools import SLURMtools

        self.folder = folder
        self.p = p
        self.additional_command = additional_command
        self.cpus_per_node = cpus_per_node
        self.machine = CONFIGread.machineSettings(code='cgyro')

        # MPI layout is resolved centrally in SLURMtools so the invented knobs (full-node MPI on GPU
        # machines, MPS sharing) live in one place instead of here and in code_slurm_settings
        self.resolved = resolved if resolved is not None else SLURMtools.resolve(
            code='cgyro', allocation={'resources_per_call': int(n)}, verbose=False)
        self.mpi = self.resolved.mpi
        self.nodes = self.mpi.get("nodes", 1)   # >1 for multi-node radial calls (resources_per_call > gpus_per_node)

        # Bash mode inside an existing SLURM allocation (driver under salloc/sbatch)
        self.bash_mode = self.resolved.submission_type == "bash" and self.mpi.get("numa") is not None
        self.srun_wrap = bool(self.machine.get("srun_wrap_calls", False)) and self.bash_mode
        if self.srun_wrap and self.nodes > 1:
            raise ValueError("[MITIM] srun_wrap_calls supports single-node radial calls only (resources_per_call <= gpus_per_node)")
        self.hosts = SIMtools.slurm_allocation_hostnames() if self.bash_mode else []

    # ------------------------------------------------------------------
    def env_exports(self):
        '''
        What every rank must inherit. Without OMP_NUM_THREADS the OpenMPI launcher prints "could not
        find environment variable OMP_NUM_THREADS" and the CGYRO launcher's NUMA->GPU binding (driven
        by -numa/-mpinuma) silently collapses: all ranks end up on GPU 0 and OOM. OMP_STACKSIZE=1G is
        the GACODE-recommended default for GPU offload kernels; too small and the first large-grid
        kernel segfaults. OMPI_MCA_io=^ompio picks anything but OMPIO, which on NFS (engaging /orcd)
        spent 30-100 s per output step against <1 s for ROMIO; not a ROMIO name, since the component
        is romio321 in OpenMPI 4 and romio341 in OpenMPI 5 and naming one the build lacks leaves no
        MPI-IO at all (CGYRO dies in cgyro_write_hosts). Ignored by MPICH-based builds (Perlmutter GPU).

        In bash mode the srun-based launchers also need the step pinned to one node holding its own
        GPUs, so the concurrent calls the bash builder backgrounds land on different nodes (a shared
        node would map two calls onto the same GPUs via SLURM_LOCALID). srun honors these as input
        environment variables; the node count is read from SLURM_JOB_NUM_NODES (SLURM_NNODES alone is
        ignored), both are set for safety.
        '''
        exports = (
            f"export OMP_NUM_THREADS={self.mpi['nomp']}\n"
            f"export OMP_STACKSIZE=1G\n"
            "export OMPI_MCA_io=^ompio\n"
        )
        exports += self.host_selection()
        if self.bash_mode and not self.srun_wrap:
            exports += (
                f"export SLURM_JOB_NUM_NODES={self.nodes}\n"
                f"export SLURM_NNODES={self.nodes}\n"
                f"export SLURM_GPUS_PER_NODE={self.mpi['numa']}\n"
            )
            if self.hosts:
                exports += "export SLURM_JOB_NODELIST=$_sel; export SLURM_NODELIST=$_sel\n"
        return exports

    @property
    def calls_per_node(self):
        '''How many radial calls share one node: >1 when a call takes fewer GPUs than the node has.'''
        gpus_per_node = int(self.machine.get("gpus_per_node") or 0)
        return max(1, gpus_per_node // self.mpi['numa']) if gpus_per_node else 1

    def host_selection(self):
        '''
        Node choice happens in bash: the builder runs ONE body for every radius in a loop, SIMtools
        exports the allocation as MITIM_HOSTS and a 1-based MITIM_CALL counter, and call k takes
        hosts [(k-1)*nodes, k*nodes). When several calls share a node, call k takes host k // calls_per_node;
        which of that node's GPUs it gets is SLURM's choice (see _srun_step).
        '''
        if not self.hosts:
            return ""
        if self.calls_per_node > 1:
            return f"_cpn={self.calls_per_node}; _k=$((MITIM_CALL-1)); _nh=${{#MITIM_HOSTS[@]}}; _sel=${{MITIM_HOSTS[$(( (_k/_cpn) % _nh ))]}}\n"
        return (
            f"_npc={self.nodes}; _k=$((MITIM_CALL-1)); _nh=${{#MITIM_HOSTS[@]}}; _sel=\"\"\n"
            f"for _j in $(seq 0 $((_npc-1))); do _h=${{MITIM_HOSTS[$(( (_k*_npc+_j) % _nh ))]}}; _sel=\"${{_sel}}${{_sel:+,}}${{_h}}\"; done\n"
        )

    def _cgyro_invocation(self, folder=None, numa=True, trailing=None):
        '''The cgyro command line. `trailing` defaults to a space plus additional_command, empty or not.'''
        m = self.mpi
        layout = f"-numa {m['numa']} -mpinuma {m['mpinuma']} " if numa else ""
        tail = f" {self.additional_command}" if trailing is None else trailing
        return (f"cgyro -e {folder or self.folder} -n {m['n']} -nomp {m['nomp']} "
                f"{layout}-p {self.p}{tail}")

    def _srun_step(self):
        '''
        Machines whose gacode launcher is OpenMPI `mpirun` (engaging PSFCR8_GPU): inside a multi-node
        allocation mpirun launches its daemons wherever SLURM puts them and ignores hostfiles. Run each
        radial call as an srun step ON its node instead (numa tasks so the step owns the node's CPUs and
        GPUs; only task 0 runs mpirun) with the SLURM view narrowed to that node, so mpirun spawns its
        ranks locally with no daemons. Enabled per machine with `srun_wrap_calls: true` (single-node
        calls only). No SLURM_* exports outside the step: srun would read them as options. The step takes
        the node's whole CPU share of its GPUs (128 cores / 4 GPUs -> 32 per task on engaging R8), not
        just nomp: a whole-node cpuset is what lets the NUMA platform place ranks by rankfile.
        '''
        m = self.mpi
        inner = (
            "if [ \"$SLURM_PROCID\" != \"0\" ]; then exit 0; fi; "
            "export H=$(hostname); export SLURM_JOB_NODELIST=$H SLURM_NODELIST=$H SLURM_JOB_NUM_NODES=1 SLURM_NNODES=1 "
            f"SLURM_TASKS_PER_NODE={m['numa']} SLURM_NTASKS={m['numa']} SLURM_NPROCS={m['numa']} SLURM_JOB_CPUS_PER_NODE={m['nomp'] * m['numa']}; "
            f"export OMP_NUM_THREADS={m['nomp']} OMP_STACKSIZE=1G OMPI_MCA_io=^ompio; "
            + self._cgyro_invocation(folder='"$MITIM_FOLDER"', trailing="")
        )
        gpus_per_node = int(self.machine.get("gpus_per_node") or m['numa'])
        if self.calls_per_node > 1:
            # Calls sharing a node: each step owns its GPUs and cores exclusively (no --overlap, --exact),
            # so the gacode wrapper's CUDA_VISIBLE_DEVICES=<local rank> resolves inside the step's own
            # device cgroup. With --overlap every step was handed the node's first GPU. A step also
            # takes the job's whole --mem unless told its share, which serializes the calls.
            sharing = f"--exact ${{SLURM_MEM_PER_NODE:+--mem=$((SLURM_MEM_PER_NODE/{self.calls_per_node}))M}}"
            cpus_per_task = m['nomp']
        else:
            sharing, cpus_per_task = "--overlap", max(m['nomp'], self.cpus_per_node // max(gpus_per_node, 1))
        return (f"srun -N1 -n{m['numa']} -c{cpus_per_task} --gpus-per-node={m['numa']} --cpu-bind=none ${{_sel:+-w $_sel}} {sharing} --export=ALL "
                f"bash -c '{inner}' {self.additional_command}")

    def launch(self):
        '''Environment plus the cgyro invocation in the shape this machine needs.'''
        if self.srun_wrap:
            return self.env_exports() + f"export MITIM_FOLDER={self.folder}\n" + self._srun_step()
        return self.env_exports() + self._cgyro_invocation(numa=self.mpi.get("numa") is not None)

    def markers(self):
        '''
        (marker_cmd, cleanup_cmd) around the launch.

        .mitim_run_started is the baseline for "did this run write a restart?"; .mitim_t0 is the
        simulated time this launch starts from (0 for a fresh or warm start, the tag time for an
        in-place rescue), which the scheduler needs to estimate the remaining a/cs as
        MAX_TIME - (t - t0). t0 comes from out.cgyro.tag line 2, not from the tail of out.cgyro.time:
        an interrupted run wrote outputs past its last restart and CGYRO rewinds those on resume.

        The cleanup drops a warm-start bin.cgyro.restart the run never overwrote, so the retrieval
        tarball doesn't ferry back a blob we already have locally: a restart newer than the marker was
        written during the run (keep), older or equal was staged before it (delete). The baseline is
        the marker and not out.cgyro.info, whose EXIT line is appended at the END of the run, so that
        comparison would delete every legitimately fresh restart. No-op when either file is absent.
        The if-block keeps the slurm_array additional_command's trailing newline from breaking chaining.
        '''
        restart_path = f"{self.p}/{self.folder}/bin.cgyro.restart"
        marker_path = f"{self.p}/{self.folder}/.mitim_run_started"
        marker_cmd = (f'touch "{marker_path}"; _t0=$(sed -n 2p "{self.p}/{self.folder}/out.cgyro.tag" 2>/dev/null '
                      f'| awk \'{{print $1+0}}\'); echo "${{_t0:-0}}" > "{self.p}/{self.folder}/.mitim_t0"')
        cleanup_cmd = (
            f'if [ -f "{restart_path}" ] && [ -f "{marker_path}" ] && '
            f'[ ! "{restart_path}" -nt "{marker_path}" ]; then '
            f'rm -f "{restart_path}"; fi; rm -f "{marker_path}"'
        )
        return marker_cmd, cleanup_cmd

    def exit_verdict(self):
        '''
        Turn a 0 exit status into 1 when CGYRO did not finish. gacode's `cgyro` script ends with an
        if-block that returns 0 whatever the executable returned, so a CGYRO that crashed (e.g. disk
        quota exceeded writing out.cgyro.prec) was recorded by SLURM as COMPLETED 0:0. CGYRO appends
        "EXIT: (CGYRO) ..." to out.cgyro.info only on a clean end; a watchdog stop (mitim_budget.tag)
        or discard (mitim_discard.tag) is intentional and keeps its own status.
        '''
        run_dir = f"{self.p}/{self.folder}"
        return (f'if [ "$_mitim_rc" = 0 ] && ! grep -qs "^EXIT: (CGYRO)" "{run_dir}/out.cgyro.info" && '
                f'[ ! -f "{run_dir}/mitim_budget.tag" ] && [ ! -f "{run_dir}/mitim_discard.tag" ]; then '
                f'echo "MITIM: cgyro returned 0 but out.cgyro.info has no EXIT line; reporting rc=1" >&2; _mitim_rc=1; fi')

    def build(self, watchdog):
        marker_cmd, cleanup_cmd = self.markers()
        return body_keeping_exit_status(marker_cmd, watchdog.wrap(self.launch()), cleanup_cmd, self.exit_verdict())


# ----------------------------------------------------------------------------------------------------
# Per-task status of a submission, and the rescue of the radii it finds stalled
# ----------------------------------------------------------------------------------------------------

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

    @classmethod
    def from_probe_line(cls, line, slurm_state=None, job_terminal=False):
        '''
        One `folder|state|avg|steps|wall|since_update|tag|exited` probe line.

        Reclassification, in priority order:
            1. a terminal out.cgyro.tag token (FINISHED/TIMEOUT/ERROR), or CGYRO's EXIT line
            2. out.cgyro.timing has not been appended to for longer than stale_threshold -> STALLED
               (TIMED_OUT if the job-wide slurm state is also terminal). This is what catches slurm
               wall-clock kills: out.cgyro.info's mtime never moves after init, so "wall since init"
               alone made killed runs look identical to live ones.
            3. the job-wide slurm state is terminal while the radius still reads RUNNING -> TIMED_OUT
               (the job is already gone but the staleness window has not elapsed yet)
            4. otherwise what the probe reported.
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
        )
        status._reclassify(slurm_state, job_terminal)
        return status

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
        tail = self.tag_suffix
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
        }


class CgyroProbe:
    '''
    One remote pass over every radius folder of a submission (templates/cgyro_probe.sh), in a single
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


def cgyro_per_task_status(sim, session=None):
    '''
    Per-(subfolder, rho) status of a CGYRO submission: prints one line per radius and returns the
    rows the stall rescuer acts on. Used as the detector half of `check(custom_checker=...)`.
    '''
    job = getattr(sim, "simulation_job", None)
    kwargs_organize = getattr(sim, "kwargs_organize", None)
    if job is None or kwargs_organize is None or not getattr(job, "launchSlurm", False):
        return []

    # The same ordered execution-folder list SIMtools._run staged for the slurm array
    folders = SIMtools.WorkPlan.from_code_executor(kwargs_organize["code_executor"]).rel_paths
    if not folders:
        return []

    slurm_state = (getattr(job, "infoSLURM", None) or {}).get("STATE")
    job_terminal = SlurmState.from_token(slurm_state) in _SLURM_JOB_GONE_STATES

    lines = CgyroProbe(job, session=session).run(folders)
    if lines is None:
        return []

    print(f"\t- Per-task CGYRO status ({len(folders)} element(s)):")
    rows = []
    for line in lines:
        status = RadiusStatus.from_probe_line(line, slurm_state=slurm_state, job_terminal=job_terminal)
        if status is None:
            continue
        text, type_msg = status.describe()
        print(text, typeMsg=type_msg)
        rows.append(status.to_row())

    return rows


def _slurm_state_for_target(job, target):
    '''
    Return the slurm state for `target` (a "<jobid>" or "<jobid>_<idx>" string),
    queried via `sacct -X --format=State`. Returns the uppercased state token,
    or None if sacct produced no useful output (e.g. site without sacct, jobid
    too old to be in the accounting window). The caller treats None as "no
    signal -- continue with the existing decision tree" rather than as a
    terminal verdict.
    '''
    cmd = f'sacct -j {target} -X --format=State -n -P'
    try:
        out, _err = job.execute(cmd, printYN=False)
    except Exception as e:
        print(f"\t    * sacct query for {target} failed ({type(e).__name__}: {e}); proceeding without per-task slurm-state guard", typeMsg='w')
        return None
    if isinstance(out, bytes):
        out = out.decode(errors='replace')
    out = (out or "").strip()
    if not out:
        return None
    # Multiple lines possible (sacct emits one row per step). The first non-empty
    # line is the parent task state — what we actually care about.
    first = out.splitlines()[0].strip()
    if not first:
        return None
    # State strings can carry trailing markers like "CANCELLED+" or
    # "CANCELLED by 12345"; canonicalize to the leading word.
    return first.split()[0].rstrip("+").upper()


@dataclass
class LedgerEntry:
    '''
    One folder's auto-resubmit history, in the dict shape persisted in cgyro_submission.json and read
    back by SIMtools._child_jobids and the re-attach path.
    '''

    n_attempts: int = 0
    child_jobids: list = field(default_factory=list)
    last_action_at: str = None
    status: str = "active"

    @classmethod
    def from_json(cls, entry):
        entry = entry or {}
        return cls(
            n_attempts=int(entry.get("n_attempts", 0)),
            child_jobids=list(entry.get("child_jobids", []) or []),
            last_action_at=entry.get("last_action_at"),
            status=str(entry.get("status", "active")),
        )

    def to_json(self):
        return {
            "n_attempts": self.n_attempts,
            "child_jobids": self.child_jobids,
            "last_action_at": self.last_action_at,
            "status": self.status,
        }

    @property
    def terminal_no_rescue(self):
        '''slurm already answered for this folder: never rescue it, and never ask sacct again.'''
        return self.status.startswith("TERMINAL_NO_RESCUE")

    def is_closed(self, cap):
        '''No further rescue is owed: already resolved, or the per-rho retry cap is spent.'''
        return self.status.startswith(("EXHAUSTED", "TERMINAL_NO_RESCUE")) or self.n_attempts >= cap


class StallRescuer:
    '''
    Rescues the radii `cgyro_per_task_status` reports as stalled past their kill threshold: scancel that
    one array task, clear its stale signal files, and resubmit it alone with the node it died on excluded.
    bin.cgyro.restart is kept but out.cgyro.tag is not, so CGYRO starts from the blob with restart_flag 2:
    the simulated time restarts from 0, it is not continued from the dead task's time.

    Per-rho retries are capped at `max_resubmits_per_rho` (default 1). Once a rho is closed it is left
    alone: fetch notes the missing per-rho outputs and downstream PORTALS handles the gap (e.g. the TGLF
    fallback in transport_cgyro). The ledger is persisted into the submission metadata after every action
    so a re-attached PORTALS picks up the child jobids.
    '''

    CLEANUP_FILES = ["out.cgyro.timing", "out.cgyro.tag", "out.cgyro.info", "slurm_output.dat", "slurm_error.dat"]

    def __init__(self, sim, settings, session=None):
        self.sim = sim
        self.job = sim.simulation_job
        self.session = session   # live ssh session to reuse; None -> one is opened for this rescue
        self.init_kill_s = int(settings.get("stall_init_kill_seconds", 1800))
        self.run_kill_s = int(settings.get("stall_running_kill_seconds", 1800))
        self.cap = int(settings.get("max_resubmits_per_rho", 1))
        organize = getattr(sim, "kwargs_organize", None) or {}
        self.array_index_by_folder = organize.get("array_index_by_folder", {})
        self.per_folder_commands = organize.get("per_folder_commands", {})
        self.metadata_dirty = False

    @property
    def ledger(self):
        if getattr(self.sim, "_resubmit_ledger", None) is None:
            self.sim._resubmit_ledger = {}
        return self.sim._resubmit_ledger

    def rescue(self, rows):
        if not self.array_index_by_folder or not self.per_folder_commands:
            return   # not a slurm_array submission — single-task rescue does not apply

        candidates = self._candidates(rows)
        if not candidates:
            return
        print(f"\t- [auto-resubmit] {len(candidates)} stalled task(s) past the kill threshold; attempting rescue", typeMsg='w')

        if self.session is not None:
            self._rescue_all(candidates)
        else:
            try:
                self.job.connect()
            except Exception as e:
                print(f"\t- [auto-resubmit] could not open SSH connection ({type(e).__name__}: {e}); skipping rescue this poll", typeMsg='w')
                return
            try:
                self._rescue_all(candidates)
            finally:
                try:
                    self.job.close()
                except Exception:
                    pass

        if self.metadata_dirty:
            try:
                self.sim._write_submission_metadata(getattr(self.sim, "_base_subfolder", None))
            except Exception as e:
                print(f"\t- [auto-resubmit] metadata write failed ({type(e).__name__}: {e}); ledger held in-memory only", typeMsg='w')

    def _candidates(self, rows):
        '''(row, threshold) for every row stalled longer than its own kill threshold.'''
        candidates = []
        for row in rows:
            state, since = row["effective"], row["since_update_i"]
            if state == TaskState.STALLED and since > self.run_kill_s:
                candidates.append((row, self.run_kill_s))
            elif state == TaskState.STALLED_INIT and since > self.init_kill_s:
                candidates.append((row, self.init_kill_s))
        return candidates

    def _rescue_all(self, candidates):
        for row, threshold in candidates:
            entry = LedgerEntry.from_json(self.ledger.get(row["folder"]))
            try:
                self._rescue_one(row, threshold, entry)
            finally:
                self.ledger[row["folder"]] = entry.to_json()

    def _rescue_one(self, row, threshold, entry):
        folder, state, since = row["folder"], row["effective"], row["since_update_i"]

        if entry.terminal_no_rescue:
            return
        if entry.is_closed(self.cap):
            if entry.status != "EXHAUSTED":
                entry.status = "EXHAUSTED"
                self.metadata_dirty = True
            print(
                f"\t  - [auto-resubmit] {folder}: {state} {since}s (>{threshold}s) — "
                f"RESUBMIT_EXHAUSTED ({entry.n_attempts}/{self.cap}); leaving as-is",
                typeMsg='w',
            )
            return

        attempt = entry.n_attempts + 1
        print(
            f"\t  - [auto-resubmit] {folder}: {state} {since}s (>{threshold}s) — "
            f"attempting rescue {attempt}/{self.cap}",
            typeMsg='w',
        )

        target, rescuing_child = self._resolve_target(folder, entry)
        if target is None:
            return
        if self._slurm_says_dead(folder, target, entry):
            return
        if not self._scancel(folder, target):
            return
        if not self._clean_remote(folder):
            return
        self._resubmit(folder, entry, attempt, rescuing_child)

    def _resolve_target(self, folder, entry):
        '''
        (jobid to scancel, whether it is a rescue child). The latest child when this rho was already
        rescued on an earlier poll, otherwise the parent array's task at the stalled index.
        '''
        if entry.child_jobids:
            return entry.child_jobids[-1], True
        array_idx = self.array_index_by_folder.get(folder)
        if array_idx is not None and self.job.jobid is not None:
            return f"{self.job.jobid}_{array_idx}", False
        print(f"\t    * cannot resolve scancel target (jobid={self.job.jobid}, array_idx={array_idx}); skipping {folder}", typeMsg='w')
        return None, False

    def _slurm_says_dead(self, folder, target, entry):
        '''
        True when slurm's own answer forbids the rescue.

        A terminal state means the work unit is no longer in flight, whether it succeeded (CGYRO
        reached MAX_TIME and exited cleanly without writing out.cgyro.tag — the usual false positive
        behind a STALLED classification) or died (a rescue would just restart from the warm start the
        next BO iteration retries anyway). A not-yet-started state means whatever the probe saw in
        that folder predates this job, e.g. an interrupted run preserved for an in-place rescue, and
        cancelling the element would kill a pending rescued radius. None means sacct gave no signal:
        fall through to the rescue rather than block on an unsupported sacct setup.
        '''
        slurm_state = _slurm_state_for_target(self.job, target)
        state = SlurmState.from_token(slurm_state)
        if state.terminal:
            entry.status = f"TERMINAL_NO_RESCUE:{slurm_state}"
            self.metadata_dirty = True
            print(
                f"\t    * slurm reports {target} is already in terminal state "
                f"{slurm_state} (likely finished cleanly without writing out.cgyro.tag); "
                f"skipping rescue for {folder} -- detector classification was a false positive",
                typeMsg='i',
            )
            return True
        if state in (SlurmState.PENDING, SlurmState.CONFIGURING, SlurmState.REQUEUED, SlurmState.SUSPENDED):
            print(f"\t    * slurm reports {target} is {slurm_state} (not started); ignoring stale files, no rescue", typeMsg='i')
            return True
        if slurm_state is not None:
            print(f"\t    * slurm reports {target} is {slurm_state}; proceeding with rescue", typeMsg='i')
        return False

    def _scancel(self, folder, target):
        try:
            _, err = self.job.execute(f"scancel {target}", printYN=True)
            if isinstance(err, bytes):
                err = err.decode(errors='replace')
            if err and err.strip():
                print(f"\t    * scancel stderr (non-fatal): {err.strip()}", typeMsg='w')
        except Exception as e:
            print(f"\t    * scancel failed ({type(e).__name__}: {e}); skipping rescue for {folder}", typeMsg='w')
            return False
        return True

    def _clean_remote(self, folder):
        '''Clear the stale signal files, keeping bin.cgyro.restart as the new task's warm start.'''
        try:
            self.job.execute(f"cd {self.job.folderExecution}/{folder} && rm -f " + " ".join(self.CLEANUP_FILES), printYN=False)
        except Exception as e:
            print(f"\t    * remote cleanup failed ({type(e).__name__}: {e}); skipping rescue for {folder}", typeMsg='w')
            return False
        return True

    def _resubmit(self, folder, entry, attempt, rescuing_child):
        # Bad-node exclusion: the node of THIS array element, from the squeue rows of the last poll
        # (the whole array's node list would exclude healthy nodes too). Child rescues have no tracked
        # per-jobid node, so slurm places them freely; None too when the element had no node yet.
        bad_node = None if rescuing_child else self.job.node_of(self.array_index_by_folder.get(folder))

        code_call_str = self.per_folder_commands.get(folder)
        if not code_call_str:
            print(f"\t    * no stored bash body for {folder}; cannot resubmit", typeMsg='w')
            return

        label = f"_resubmit_{Path(folder).name}_a{attempt}"
        try:
            new_jobid = self.job.resubmit_single_task(code_call_str, label, exclude_node=bad_node)
        except Exception as e:
            print(f"\t    * resubmit_single_task raised ({type(e).__name__}: {e}); ledger NOT incremented (will retry next poll)", typeMsg='w')
            return
        if not new_jobid:
            print(f"\t    * resubmit_single_task returned no jobid; ledger NOT incremented (will retry next poll)", typeMsg='w')
            return

        entry.n_attempts = attempt
        entry.child_jobids.append(new_jobid)
        # naive UTC plus an explicit "Z", the format already stored in cgyro_submission.json
        entry.last_action_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        self.metadata_dirty = True
        print(f"\t    * rescued {folder}: new jobid={new_jobid} (attempt {attempt}/{self.cap}, exclude={bad_node})", typeMsg='i')


def _cgyro_handle_stalled_tasks(sim, rows, session=None):
    '''Stall-rescue entry point for one poll's rows (see StallRescuer). No-op unless the user opted in.'''
    settings = getattr(sim, "auto_resubmit_settings", None) or {}
    if not settings.get("enabled", False):
        return
    if getattr(sim, "simulation_job", None) is None:
        return
    StallRescuer(sim, settings, session=session).rescue(rows)


def cgyro_per_task_callback(sim):
    '''
    Bound as `_custom_check_callback` and called as `custom_checker(sim)` once per poll by
    `mitim_simulation.check`, which is why this stays a module function. One ssh session for the whole
    poll: the probe prints and returns the rows, the rescuer acts on the stalled ones. Users who
    haven't opted into auto_resubmit_settings see the historical detect-only behavior.
    '''
    job = getattr(sim, "simulation_job", None)
    kwargs_organize = getattr(sim, "kwargs_organize", None)
    if job is None or kwargs_organize is None or not getattr(job, "launchSlurm", False):
        return
    try:
        with job.session() as session:
            rows = cgyro_per_task_status(sim, session=session)
            if rows:
                _cgyro_handle_stalled_tasks(sim, rows, session=session)
    except Exception as e:
        print(f"\t- [per-task status] poll over ssh failed ({type(e).__name__}: {e}); continuing", typeMsg='w')


class _ResolvedControls:
    '''
    The controls `_run_prepare` will actually materialise, resolved once for the `_enforce_*` steps:
    the per-rho controls snapshot (input.cgyro.controls), replaced wholesale by the model-yaml block
    when a code_settings label is given — which is what SIMtools.modifyInputs does — with extraOptions
    read on top of it.
    '''

    def __init__(self, sim, extraOptions, code_settings=None):
        controls = {}
        if sim.rhos is not None and len(sim.rhos) > 0:
            controls = dict(sim.inputs_files[sim.rhos[0]].controls)
        if code_settings is not None:
            try:
                controls = GACODEdefaults.addCGYROcontrol(code_settings)
            except Exception as e:
                print(
                    f"\t- [preprocess] Could not resolve code_settings={code_settings!r} "
                    f"against input.cgyro.models.yaml ({e}); falling back to controls file only",
                    typeMsg="w",
                )
        self.controls = controls
        self.extraOptions = extraOptions

    def get(self, key, default=None):
        '''The value that will be written for `key`, scalar or per-rho list.'''
        return self.extraOptions.get(key, self.controls.get(key, default))

    def as_list(self, key, n=1, cast=float, default=None):
        '''(values broadcast to length n, the source value) for `key`; (None, None) when absent.'''
        source = self.get(key, default)
        if source is None:
            return None, None
        values = [cast(v) for v in source] if self.is_list(source) else [cast(source)]
        return self.broadcast(values, n), source

    @staticmethod
    def is_list(value):
        return isinstance(value, (list, np.ndarray))

    @staticmethod
    def broadcast(values, n):
        '''A single value serves every rho; anything else is already per-rho.'''
        return values * n if len(values) == 1 else values


def _restart_outputs(max_time, delta_t, print_step):
    '''
    Data outputs within the run: CGYRO's time loop is i_time = 1..nint(MAX_TIME/DELTA_T) and it writes
    a restart when mod(i_time, RESTART_STEP*PRINT_STEP) == 0, so RESTART_STEP*PRINT_STEP must not
    exceed n_time. A ceil(MAX_TIME/(DELTA_T*PRINT_STEP)) overshoots n_time whenever the ratio is
    non-integer (e.g. DELTA_T=0.006) and the restart then never fires.
    '''
    return max(1, int(round(max_time / delta_t)) // int(round(print_step)))


def _coerced_restart_step(restart_step, n_outputs):
    '''
    Keep the user/controls RESTART_STEP when it fires at least once with its last write aligned to the
    final output step; otherwise take n_outputs, which writes exactly one restart at that step.
    '''
    return restart_step if (0 < restart_step <= n_outputs and n_outputs % restart_step == 0) else n_outputs


class CGYRO(SIMtools.mitim_simulation, SIMplot.GKplotting):

    # Opts CGYRO into persisting slurm-submission metadata (jobid, remote
    # folder, retrieval plan) whenever run_type='submit' is used, so a later
    # PORTALS restart can re-attach to the in-flight job rather than resubmit.
    _submission_metadata_filename = "cgyro_submission.json"

    # Per-task inspection for `check(custom_checker=...)`. Picked up by
    # transport_cgyro.py via `getattr(gk_object, '_custom_check_callback', None)`
    # so the generic gyrokinetic_model evaluator stays code-agnostic (GX etc.
    # simply get None and skip). The wrapper runs the detector (prints + returns
    # structured rows) and then dispatches to the auto-resubmit orchestrator,
    # which is a no-op unless the user has opted in via `auto_resubmit_settings`.
    _custom_check_callback = staticmethod(cgyro_per_task_callback)

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Transient state used by run() to feed preprocess_options into _run_prepare()
        self._preprocess_options = None
        self._extra_point_n = None

        # On GPU machines, always use a job array so each radius gets its own GPU allocation.
        _cgyro_machine_settings = CONFIGread.machineSettings(code='cgyro')
        _force_submission_type = 'slurm_array' if (_cgyro_machine_settings.get('gpus_per_node') or 0) > 0 else None

        self.run_specifications = {
            'code': 'cgyro',
            'input_file': 'input.cgyro',
            'code_call': self.code_call,
            'control_function': GACODEdefaults.addCGYROcontrol,
            'controls_file': 'input.cgyro.controls',
            'state_converter': 'to_cgyro',
            'input_class': CGYROinput,
            'complete_variation': None,
            'default_cores': 16,  # Default cores to use in the simulation
            'output_class': CGYROutils.CGYROoutput,
            'force_submission_type': _force_submission_type,
            # Interrupted-run rescue (SIMtools._rescue_interrupted_runs): with both the
            # restart blob and out.cgyro.tag present, re-running `cgyro -e` in the same
            # folder continues the time integration (restart_flag=1) for MAX_TIME more
            # a/cs from the tag time (out.cgyro.tag: line 1 i_current, line 2 t_current).
            'rescue_spec': {'required': ['bin.cgyro.restart', 'out.cgyro.tag'], 'progress_file': 'out.cgyro.tag', 'progress_line': 2, 'time_key': 'MAX_TIME',
                            # RESTART_STEP was sized for the full MAX_TIME, so it has to follow the trim
                            # (and be excluded from the identity md5, like MAX_TIME, for the next rescue)
                            'after_trim': self._restart_step_after_trim,
                            'checksum_ignore': ['MAX_TIME', 'RESTART_STEP'],
                            'report_files': ['out.cgyro.time', 'bin.cgyro.ky_flux', 'bin.cgyro.restart', 'out.cgyro.tag']},
            # A radius is only 'done' if CGYRO wrote its EXIT line (files exist from step 1 on)...
            'completion_marker': ('out.cgyro.info', 'EXIT'),
            # ...or the watchdog stopped it past min_time. Optional, never mandatory: a run that ends
            # by itself never writes this file.
            'completion_alt_file': 'mitim_budget.tag',
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t CGYRO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.output_files_simulation["minimal_base"] = [
            "bin.cgyro.geo",
            "bin.cgyro.ky_cflux",
            "bin.cgyro.ky_flux",
            "input.cgyro.gen",
            "out.cgyro.egrid",
            "out.cgyro.equilibrium",
            "out.cgyro.grids",
            "out.cgyro.hosts",
            "out.cgyro.info",
            "out.cgyro.memory",
            "out.cgyro.mpi",
            "out.cgyro.prec",
            "out.cgyro.rotation",
            "out.cgyro.startups",
            "out.cgyro.time",
            "out.cgyro.timing",
            "out.cgyro.version",
        ]

        self.output_files_simulation["complete_base"] = list(self.output_files_simulation["minimal_base"])

        # Best-effort retrievals: tarred if present, absence logged once (no
        # 60s retry, no cold-start trigger). Two groups:
        #   - restart blobs + companion .flag/.tag that CGYRO may skip writing
        #     on short runs, crashes, or COMPLETING-timeouts.
        #   - large bin.cgyro.kxky_* dumps (tens to hundreds of MB each) that
        #     diagnostics can do without and whose retrieval over a slow
        #     shared filesystem was the dominant cost of fetch().
        self.output_files_simulation["optional_base"] = [
            "bin.cgyro.restart",
            "bin.cgyro.restart.flag",
            "out.cgyro.tag",
            "bin.cgyro.kxky_apar",
            "bin.cgyro.kxky_bpar",
            "bin.cgyro.kxky_e",
            "bin.cgyro.kxky_n",
            "bin.cgyro.kxky_phi",
            "bin.cgyro.kxky_v",
            "mitim_budget.tag",
        ]

        # Nonlinear sim
        for key in ['minimal', 'complete']:
            self.output_files_simulation[f"{key}_nonlinear"] = self.output_files_simulation[f"{key}_base"] + [
                "bin.cgyro.freq"
                ]

        # Linear sim
        for key in ['minimal', 'complete']:
            self.output_files_simulation[f"{key}_linear"] = self.output_files_simulation[f"{key}_base"] + [
                "out.cgyro.freq",
                "bin.cgyro.phib",
                "bin.cgyro.aparb",
                "bin.cgyro.bparb",
                ]

        # Make sure, just in case, that "complete" and "minimal" are populated from this __init__, even if it will be re-defined later
        self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_nonlinear"])
        self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_nonlinear"])
        self.output_files_simulation["optional"] = copy.deepcopy(self.output_files_simulation["optional_base"])

        # Primary/fallback pairs for the remote-prune step in retrieve().
        # bin.cgyro.restart.old is the previous-cycle checkpoint CGYRO keeps
        # under RESTART_PRESERVATION_MODE>=3 (default 3) as a durability net
        # while the next .restart is being written. If the job died mid-
        # write the only intact checkpoint on disk is .old — promote it to
        # .restart on the remote before tarring so we pull exactly one file
        # and still recover from the degenerate case. See
        # gacode/cgyro/src/cgyro_restart.F90:120-220.
        self.output_file_fallbacks = {
            "bin.cgyro.restart": "bin.cgyro.restart.old",
        }
        

    @staticmethod
    def _allocation_cpus_per_node():
        '''Per-node CPU count of the allocation (first entry of SLURM_JOB_CPUS_PER_NODE, e.g. "128(x5)" -> 128).'''
        raw = os.environ.get("SLURM_JOB_CPUS_PER_NODE", "")
        digits = "".join(ch for ch in raw.split("(")[0].split(",")[0] if ch.isdigit())
        return int(digits) if digits else 1

    # Thin wrapper: capture preprocess_options / load_balance and delegate to the generic run()
    def run(self, *args, preprocess_options=None, load_balance=None, **kwargs):
        '''
        load_balance: {'strategy': None | 'wall_budget' | 'extra_points', 'minutes_per_call': float, 'min_time': float}
        (namelist transport.options.cgyro.run.load_balance); see Watchdog and _extra_point_hooks.
        '''
        self._preprocess_options = preprocess_options
        self._load_balance = load_balance
        try:
            return super().run(*args, **kwargs)
        finally:
            self._preprocess_options = None
            self._load_balance = None
            self._extra_point_n = None

    def code_call(self, folder, p, n=1, additional_command="", watchdog=None, resolved=None):
        '''
        Bash body of one radial call, in the shape SIMtools' JobScript builders ask for
        (`code_call(folder=..., n=..., p=...)`).
        watchdog: Watchdog mode for this launch; None takes it from load_balance.
        resolved: an already-resolved SLURMtools allocation, when the caller has one.
        '''
        body = CgyroLaunchBody(folder, p, n=n, additional_command=additional_command,
                               resolved=resolved, cpus_per_node=self._allocation_cpus_per_node())
        return body.build(Watchdog.from_load_balance(
            f"{p}/{folder}", getattr(self, "_load_balance", None), mode=watchdog, template=self._WALL_BUDGET_WATCHDOG))

    # The bash itself lives in templates/cgyro_watchdog.sh; kept as an attribute so a caller can supply
    # its own template through `self`
    _WALL_BUDGET_WATCHDOG = _WATCHDOG_BASH

    def _wall_budget_wrap(self, cgyro_cmd, rho_dir, mode=None):
        '''The launch, wrapped in its Watchdog (see that class for what each mode does).'''
        return Watchdog.from_load_balance(
            rho_dir, getattr(self, "_load_balance", None), mode=mode, template=self._WALL_BUDGET_WATCHDOG).wrap(cgyro_cmd)

    # ------------------------------------------------------------------
    # load_balance strategy 'extra_points' (bash mode): hooks for the in-allocation
    # scheduler (SCHEDULERtools). The transport layer registers `extra_point_builder`,
    # a callable (rho, finished_scratch_dir, local_dir) -> path of a ready input.cgyro
    # for a perturbed case at that radius (or None); everything else is generic here.
    # ------------------------------------------------------------------
    extra_point_builder = None

    def _extra_point_hooks(self, resources_per_call):
        lb = getattr(self, "_load_balance", None) or {}
        if lb.get("strategy") != "extra_points":
            return None
        if self.extra_point_builder is None:
            print("\t- load_balance 'extra_points' requested but no extra_point_builder is registered (standalone CGYRO run?); waiting for the slowest radius instead", typeMsg="w")
            return None
        self._extra_point_n = int(resources_per_call)
        return {"on_call_finished": self._launch_extra_point,
                "estimate_remaining": self._estimate_remaining,
                "estimate_to_accept": self._estimate_to_accept,
                "idle_slot_sources": self._idle_slot_sources}

    def _scratch(self, rel):
        return Path(self.simulation_job.folderExecution) / rel

    @staticmethod
    def _last_col(path, col=None):
        try:
            row = path.read_text().strip().splitlines()[-1].split()
            return float(row[-1] if col is None else row[col])
        except (OSError, IndexError, ValueError):
            return None

    def _cost_per_acs(self, rel):
        '''Seconds per a/cs from the last TOTAL of out.cgyro.timing (one row per PRINT_STEP = 1 a/cs).'''
        return self._last_col(self._scratch(rel) / "out.cgyro.timing")

    def _estimate_remaining(self, rel):
        d = self._scratch(rel)
        cost, t = self._cost_per_acs(rel), self._last_col(d / "out.cgyro.time", col=0)
        if cost is None or t is None:
            return None
        try:
            t0 = float((d / ".mitim_t0").read_text().strip() or 0.0)
            max_time = float(next(l.split("=")[1] for l in (d / "input.cgyro").read_text().splitlines() if l.strip().startswith("MAX_TIME")))
        except (OSError, StopIteration, ValueError):
            return None
        return max(max_time - (t - t0), 0.0) * cost

    def _estimate_to_accept(self, rel):
        cost = self._cost_per_acs(rel)
        lb = getattr(self, "_load_balance", None) or {}
        return None if cost is None else float(lb.get("min_time", 0.0)) * cost + 300.0

    def _idle_slot_sources(self, main_rels):
        '''
        Scheduler hook: the radii of this evaluation that an earlier driver job already finished,
        so a relaunch that runs only the unfinished ones can still give its idle slots extras.
        Each is made to look like a radius that just finished in scratch: its stored `<file>_<rho>`
        outputs are symlinked into the scratch folder under their plain names (skipped when the
        folder is still there), which is all _launch_extra_point, _estimate_to_accept and the
        builder read. Radii whose extra is already done locally are left out.
        '''
        spec = SIMtools.CompletionSpec.from_run_specifications(self.run_specifications)
        extra_done = SIMtools.CompletionSpec.coerce(spec, alt_file="mitim_budget.tag")
        main = set(main_rels)
        sources = []
        for sub in sorted({rel.split("/")[0] for rel in main_rels}):
            local = Path(self.FolderGACODE) / sub
            for info in sorted(local.glob(f"{spec.marker_file}_*")):
                rho = float(info.name.rsplit("_", 1)[-1])
                rel = f"{sub}/{SIMtools.rho_folder(rho)}"
                if rel in main or not spec.finished(local, rho)[0]:
                    continue
                if extra_done.finished(Path(self.FolderGACODE) / "extra_cgyro" / SIMtools.rho_folder(rho))[0]:
                    continue
                self._stage_finished_radius(local, rho, self._scratch(rel))
                sources.append(rel)
        return sources

    @staticmethod
    def _stage_finished_radius(local, rho, scratch_dir):
        if (scratch_dir / "out.cgyro.info").exists():
            return
        scratch_dir.mkdir(parents=True, exist_ok=True)
        suffix = SIMtools.rho_suffix(rho)
        for f in local.glob(f"*{suffix}"):
            (scratch_dir / f.name[:-len(suffix)]).symlink_to(f.resolve())

    def _launch_extra_point(self, rel):
        '''Scheduler hook: prepare extra_cgyro/rho_<rho> in scratch (perturbed input.cgyro +
        the finished run's restart blob as warm start) and return (rel_extra, bash body).'''
        rho = float(rel.rsplit("rho_", 1)[-1])
        rel_extra = f"extra_cgyro/{SIMtools.rho_folder(rho)}"
        local_dir = Path(self.FolderGACODE) / rel_extra
        local_dir.mkdir(parents=True, exist_ok=True)
        input_cgyro = self.extra_point_builder(rho, self._scratch(rel), local_dir)
        if input_cgyro is None:
            return None
        dst = self._scratch(rel_extra)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_cgyro, dst / "input.cgyro")
        restart = self._scratch(rel) / "bin.cgyro.restart"
        if restart.exists():
            shutil.copy2(restart, dst / "bin.cgyro.restart")   # no out.cgyro.tag: warm start, t from 0
        body = self.run_specifications["code_call"](folder=rel_extra, p=str(self.simulation_job.folderExecution), n=self._extra_point_n, watchdog="stop")
        return rel_extra, body

    # Redefine to raise warning and allow selection of output files
    def _run_prepare(
        self,
        subfolder_simulation,
        extraOptions=None,
        multipliers=None,
        **kwargs,
    ):

        # ---------------------------------------------
        # Check if any *_SCALE_* variable is being used
        # ---------------------------------------------
        dictionary_check = {}
        if extraOptions is not None:
            if multipliers is not None:
                dictionary_check = {**extraOptions, **multipliers}
            else:
                dictionary_check = extraOptions
        elif multipliers is not None:
                dictionary_check = multipliers

        for key in dictionary_check:
            if '_SCALE_' in key:
                print("The use of *_SCALE_* is discouraged, please use the appropriate variable instead.", typeMsg='q')

        # ---------------------------------------------

        # Check if it's linear
        if 'Nonlinear' not in kwargs.get('code_settings', ''):
            self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_linear"])
            self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_linear"])
        else:
            self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_nonlinear"])
            self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_nonlinear"])
        # Optional-retrieval set is the same for linear and nonlinear.
        self.output_files_simulation["optional"] = copy.deepcopy(self.output_files_simulation["optional_base"])

        # Pre-process BOX_SIZE / N_RADIAL from local equilibrium if requested.
        # Model yaml (input.cgyro.models.yaml) can supply per-model defaults;
        # user-supplied self._preprocess_options override on a per-key basis.
        from mitim_tools.gacode_tools.utils import GACODEdefaults
        model_preprocess = GACODEdefaults.getCGYROpreprocessDefaults(kwargs.get("code_settings"))
        user_preprocess = getattr(self, "_preprocess_options", None) or {}
        merged_preprocess = {**model_preprocess, **user_preprocess}
        if merged_preprocess:
            saved_preprocess = getattr(self, "_preprocess_options", None)
            self._preprocess_options = merged_preprocess
            try:
                extraOptions = self._apply_cgyro_preprocessing(extraOptions or {})
            finally:
                self._preprocess_options = saved_preprocess

        # Enforce TOROIDALS_PER_PROC compatibility with N_TOROIDAL and MPI rank count.
        # Resolution order matches what _run_prepare will eventually write:
        #   input.cgyro.controls  ->  input.cgyro.models.yaml[code_settings]  ->  extraOptions
        extraOptions = self._enforce_toroidals_per_proc(
            extraOptions or {},
            kwargs.get("allocation"),
            code_settings=kwargs.get("code_settings"),
        )

        # Enforce PRINT_STEP so that DELTA_T * PRINT_STEP == 1.0 (integer PRINT_STEP).
        # Same resolution order as above: controls -> yaml -> extraOptions.
        extraOptions = self._enforce_print_step(
            extraOptions or {},
            code_settings=kwargs.get("code_settings"),
        )

        # Ensure RESTART_STEP >= total number of data outputs so the restart file
        # is written at least once at the end. One data output == DELTA_T*PRINT_STEP,
        # so n_outputs = ceil(MAX_TIME / (DELTA_T*PRINT_STEP)).
        extraOptions = self._enforce_restart_step(
            extraOptions or {},
            code_settings=kwargs.get("code_settings"),
        )

        return super()._run_prepare(
            subfolder_simulation,
            extraOptions=extraOptions,
            multipliers=multipliers,
            **kwargs,
        )

    def _enforce_toroidals_per_proc(self, extraOptions, allocation, code_settings=None):
        """
        CGYRO distributes N_TOROIDAL across MPI ranks (= resources_per_call GPUs).
        Each rank holds N_TOROIDAL/resources toroidal modes, so TOROIDALS_PER_PROC
        must be a multiple of that ratio. If the user's value is incompatible (or
        missing), coerce it to the preferred valid value and warn.
        """
        from mitim_tools.misc_tools import SLURMtools

        allocation = allocation or {}
        default_rpc = SLURMtools.CODE_HINTS.get('cgyro', {}).get("default_resources_per_call", 1)
        resources_per_call = int(allocation.get("resources_per_call", default_rpc))
        if resources_per_call <= 0:
            return extraOptions

        # Ranks per node (one MPI rank per GPU for CGYRO). Only used to keep the nonlinear
        # all-to-all on-node; if the machine block cannot be read we fall back to a value
        # that makes the locality preference a no-op.
        try:
            gpus_per_node = int(CONFIGread.machineSettings(code='cgyro').get("gpus_per_node") or 0)
        except Exception:
            gpus_per_node = 0
        ranks_per_node = min(resources_per_call, gpus_per_node) if gpus_per_node > 0 else resources_per_call

        resolved = _ResolvedControls(self, extraOptions, code_settings)
        n_tor_src = resolved.get('N_TOROIDAL', 1)
        n_tor_list = [int(v) for v in n_tor_src] if resolved.is_list(n_tor_src) else [int(n_tor_src)]

        # Single toroidal mode (e.g. linear single-ky runs): the only valid toroidal
        # split is one mode per process. This is a normal configuration, not a
        # misconfiguration, so set TOROIDALS_PER_PROC=1 and report it as info — the
        # remaining ranks parallelize the radial/velocity grid instead.
        if all(nt == 1 for nt in n_tor_list):
            if 'TOROIDALS_PER_PROC' in extraOptions:
                return extraOptions
            extraOptions = copy.deepcopy(extraOptions)
            extraOptions['TOROIDALS_PER_PROC'] = [1] * len(n_tor_list) if resolved.is_list(n_tor_src) else 1
            print(
                f"\t- [preprocess] N_TOROIDAL=1 (single toroidal mode); setting TOROIDALS_PER_PROC=1 "
                f"(resources_per_call={resources_per_call} rank(s) parallelize the radial/velocity grid)",
                typeMsg="i",
            )
            return extraOptions

        # CGYRO aborts at startup unless the MPI rank count (= resources_per_call) is a
        # multiple of the number of toroidal groups, N_TOROIDAL/TOROIDALS_PER_PROC (which
        # also requires TOROIDALS_PER_PROC to divide N_TOROIDAL). The smallest valid
        # TOROIDALS_PER_PROC maximizes toroidal parallelism (leftover rank multiplicity
        # goes to the velocity/radial decomposition); TOROIDALS_PER_PROC=N_TOROIDAL (a
        # single group) is always valid, so a solution always exists.
        def _is_valid(nt, tpp):
            return tpp > 0 and nt % tpp == 0 and resources_per_call % (nt // tpp) == 0

        # CGYRO's process grid is n_proc = n_proc_1 x n_toroidal_procs, where
        # n_toroidal_procs = N_TOROIDAL/TOROIDALS_PER_PROC and n_proc_1 splits nc and nv
        # (cgyro_mpi_grid.F90:57,233-236). n_toroidal_procs is the size of the nonlinear
        # all-to-all communicator, and MPI_RANK_ORDER=2 (CGYRO's default) makes that
        # communicator rank-contiguous -- so holding it to one node's worth of ranks keeps
        # the all-to-all on-node. Picking the smallest valid TOROIDALS_PER_PROC maximizes it
        # instead. This only ever bites on multi-node radial calls: when the call fits in one
        # node, validity already forces n_toroidal_procs <= resources_per_call, so every valid
        # value is on-node and this rule reduces to taking the smallest valid one.
        def _grid_allows(nt, tpp, n_radial):
            # n_proc_1 > 1 additionally requires n_proc_1 to divide nv and nc, or CGYRO aborts
            # (cgyro_mpi_grid.F90:240-248). N_SPECIES is not resolved at this point, so require
            # n_proc_1 | N_ENERGY*N_XI, which is sufficient for nv = N_ENERGY*N_XI*N_SPECIES.
            n_proc_1 = resources_per_call // (nt // tpp)
            if n_proc_1 == 1:
                return True
            if None in (n_energy, n_xi, n_theta, n_radial):
                return False
            return (n_energy * n_xi) % n_proc_1 == 0 and (n_radial * n_theta) % n_proc_1 == 0

        def _preferred_valid(nt, n_radial):
            valid = [tpp for tpp in range(1, nt + 1) if _is_valid(nt, tpp)]
            on_node = [tpp for tpp in valid
                       if (nt // tpp) <= ranks_per_node and _grid_allows(nt, tpp, n_radial)]
            return min(on_node) if on_node else min(valid)

        def _as_int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        n_energy = _as_int(resolved.get('N_ENERGY'))
        n_xi = _as_int(resolved.get('N_XI'))
        n_theta = _as_int(resolved.get('N_THETA'))

        if any(nt <= 0 for nt in n_tor_list):
            print(
                f"\t- [preprocess] Invalid N_TOROIDAL={n_tor_list}; leaving TOROIDALS_PER_PROC as-is",
                typeMsg="w",
            )
            return extraOptions

        # Respect an explicit user-supplied TOROIDALS_PER_PROC in extraOptions
        # (user is overriding on purpose), but warn if CGYRO will reject the combination.
        if 'TOROIDALS_PER_PROC' in extraOptions:
            tpp_user = extraOptions['TOROIDALS_PER_PROC']
            tpp_user_list = [int(v) for v in tpp_user] if resolved.is_list(tpp_user) else [int(tpp_user)] * len(n_tor_list)
            for nt, tpp in zip(n_tor_list, tpp_user_list):
                if not _is_valid(nt, tpp):
                    print(
                        f"\t- [preprocess] User-supplied TOROIDALS_PER_PROC={tpp} is incompatible with "
                        f"N_TOROIDAL={nt} and resources_per_call={resources_per_call} "
                        f"(MPI ranks must be a multiple of N_TOROIDAL/TOROIDALS_PER_PROC); CGYRO may abort",
                        typeMsg="w",
                    )
            return extraOptions

        extraOptions = copy.deepcopy(extraOptions)
        tpp_list, tpp_src = resolved.as_list('TOROIDALS_PER_PROC', n=len(n_tor_list), cast=int, default=1)

        # Broadcast a scalar N_TOROIDAL to match a per-rho TOROIDALS_PER_PROC list.
        n_tor_list = _ResolvedControls.broadcast(n_tor_list, len(tpp_list))
        if len(tpp_list) != len(n_tor_list):
            print(
                f"\t- [preprocess] TOROIDALS_PER_PROC length {len(tpp_list)} mismatches "
                f"N_TOROIDAL length {len(n_tor_list)}; leaving as-is",
                typeMsg="w",
            )
            return extraOptions

        # N_RADIAL is per-rho (set by _apply_cgyro_preprocessing, which runs before this),
        # while N_TOROIDAL is usually a scalar from the model yaml. Broadcast the length-1
        # lists up to the longest so a per-rho N_RADIAL still reaches _grid_allows; if the
        # lengths are genuinely incompatible, drop N_RADIAL rather than guess (which makes
        # _grid_allows conservative and falls back to the smallest valid value).
        nr_src = resolved.get('N_RADIAL')
        nr_list = [_as_int(v) for v in nr_src] if resolved.is_list(nr_src) else [_as_int(nr_src)]
        n_rho = max(len(n_tor_list), len(nr_list))
        if len(n_tor_list) == 1:
            n_tor_list = n_tor_list * n_rho
            tpp_list = _ResolvedControls.broadcast(tpp_list, n_rho)
        nr_list = _ResolvedControls.broadcast(nr_list, n_rho)
        if not (len(n_tor_list) == len(tpp_list) == len(nr_list) == n_rho):
            n_tor_list = n_tor_list[:1] * n_rho if len(n_tor_list) == 1 else n_tor_list
            nr_list = [None] * len(n_tor_list)

        coerced = [_preferred_valid(nt, nr) for nt, nr in zip(n_tor_list, nr_list)]

        if coerced != tpp_list:
            n_groups = [nt // tpp for nt, tpp in zip(n_tor_list, coerced)]
            print(
                f"\t- [preprocess] TOROIDALS_PER_PROC {tpp_list} -> {coerced} "
                f"(MPI ranks {resources_per_call}, {ranks_per_node} per node; toroidal groups "
                f"{n_groups}, which sets the size of the nonlinear all-to-all)",
                typeMsg="i",
            )

        # Preserve scalar shape if the caller originally gave a scalar and all coerced match.
        if not resolved.is_list(tpp_src) and len(set(coerced)) == 1:
            extraOptions['TOROIDALS_PER_PROC'] = coerced[0]
        else:
            extraOptions['TOROIDALS_PER_PROC'] = coerced

        return extraOptions

    def _enforce_print_step(self, extraOptions, code_settings=None):
        """
        Set PRINT_STEP so that DELTA_T * PRINT_STEP == 1.0 (PRINT_STEP integer).
        If the user explicitly sets PRINT_STEP in extraOptions, it is respected.
        """
        if 'PRINT_STEP' in extraOptions:
            return extraOptions

        resolved = _ResolvedControls(self, extraOptions, code_settings)
        dt_list, dt_src = resolved.as_list('DELTA_T')
        if dt_list is None:
            return extraOptions

        if any(dt <= 0 for dt in dt_list):
            print(
                f"\t- [preprocess] DELTA_T={dt_list} contains non-positive values; leaving PRINT_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        # PRINT_STEP * DELTA_T should equal 1.0 -> PRINT_STEP = round(1.0 / DELTA_T).
        print_steps = [max(1, int(round(1.0 / dt))) for dt in dt_list]

        extraOptions = copy.deepcopy(extraOptions)
        if not resolved.is_list(dt_src) and len(set(print_steps)) == 1:
            extraOptions['PRINT_STEP'] = print_steps[0]
        else:
            extraOptions['PRINT_STEP'] = print_steps

        print(
            f"\t- [preprocess] PRINT_STEP set to {extraOptions['PRINT_STEP']} "
            f"(DELTA_T={dt_src} -> DELTA_T*PRINT_STEP ~= 1.0)",
            typeMsg="i",
        )
        return extraOptions

    def _enforce_restart_step(self, extraOptions, code_settings=None):
        """
        Ensure CGYRO writes a restart by the end of the run: RESTART_STEP is coerced to the number of
        data outputs within MAX_TIME (see _restart_outputs / _coerced_restart_step), unless the
        controls-file / yaml value already divides it.
        If the user explicitly sets RESTART_STEP in extraOptions, it is respected.
        """
        if 'RESTART_STEP' in extraOptions:
            return extraOptions

        resolved = _ResolvedControls(self, extraOptions, code_settings)
        dt_list, dt_src = resolved.as_list('DELTA_T')
        ps_list, ps_src = resolved.as_list('PRINT_STEP')
        mt_list, mt_src = resolved.as_list('MAX_TIME')

        if dt_list is None or ps_list is None or mt_list is None:
            return extraOptions
        if any(v <= 0 for v in dt_list + ps_list + mt_list):
            print(
                f"\t- [preprocess] Non-positive DELTA_T/PRINT_STEP/MAX_TIME; leaving RESTART_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        n = max(len(dt_list), len(ps_list), len(mt_list))
        dt_list, ps_list, mt_list = (_ResolvedControls.broadcast(x, n) for x in (dt_list, ps_list, mt_list))
        if not (len(dt_list) == len(ps_list) == len(mt_list) == n):
            print(
                "\t- [preprocess] DELTA_T/PRINT_STEP/MAX_TIME length mismatch; leaving RESTART_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        n_outputs = [_restart_outputs(mt, dt, ps) for dt, ps, mt in zip(dt_list, ps_list, mt_list)]

        rs_list, rs_src = resolved.as_list('RESTART_STEP', n=n, cast=int, default=0)
        if rs_list is None:
            rs_list, rs_src = [0] * n, 0
        if len(rs_list) != n:
            rs_list = rs_list + [rs_list[-1]] * (n - len(rs_list)) if len(rs_list) < n else rs_list[:n]

        coerced = [_coerced_restart_step(rs, no) for rs, no in zip(rs_list, n_outputs)]

        extraOptions = copy.deepcopy(extraOptions)
        # Preserve scalar shape if inputs were all scalar and all coerced agree.
        scalars = not any(_ResolvedControls.is_list(x) for x in (dt_src, ps_src, mt_src, rs_src))
        if scalars and len(set(coerced)) == 1:
            extraOptions['RESTART_STEP'] = coerced[0]
        else:
            extraOptions['RESTART_STEP'] = coerced

        print(
            f"\t- [preprocess] RESTART_STEP set to {extraOptions['RESTART_STEP']} "
            f"(<= n_outputs = floor(nint(MAX_TIME/DELTA_T)/PRINT_STEP) = {n_outputs}; "
            f"restart fires when mod(i_time, RESTART_STEP*PRINT_STEP) == 0)",
            typeMsg="i",
        )
        return extraOptions

    @staticmethod
    def _restart_step_after_trim(text, remaining):
        """
        Re-derive RESTART_STEP for the shortened run of a rescued radius (rescue_spec
        'after_trim' hook). _enforce_restart_step sized RESTART_STEP from the FULL
        MAX_TIME, so once the rescue trims MAX_TIME to what is left the trigger
        mod(i_time, RESTART_STEP*PRINT_STEP) == 0 with i_time = 1..nint(MAX_TIME/DELTA_T)
        can no longer fire: the continuation writes no checkpoint and mitim_kill_cgyro's
        watchdog (which waits on out.cgyro.tag) blocks on that radius.

        Same coercion as _enforce_restart_step, on n_outputs of the remaining window.
        Returns (text, note appended to the [rescue] log line); unchanged when the keys
        are missing from the staged input.
        """
        import re

        def _value(key):
            m = re.search(rf"^{key}\s*=\s*(\S+)", text, flags=re.M)
            return m, (float(m.group(1)) if m else None)

        m_rs, rs = _value('RESTART_STEP')
        _, dt = _value('DELTA_T')
        _, ps = _value('PRINT_STEP')
        if m_rs is None or not dt or not ps:
            return text, ""

        rs = int(rs)
        new_rs = _coerced_restart_step(rs, _restart_outputs(remaining, dt, ps))
        if new_rs == rs:
            return text, ""
        return text[:m_rs.start(1)] + f"{new_rs}" + text[m_rs.end(1):], f", RESTART_STEP {rs} -> {new_rs}"

    def _apply_cgyro_preprocessing(self, extraOptions):
        """
        Compute BOX_SIZE and N_RADIAL per rho from the caller-provided
        ky_min plus Q, S, RMIN from self.inputs_files[rho], and inject them
        (along with KY=ky_min) into a copy of extraOptions as per-rho arrays.
        """

        allowed_keys = {'ky_min', 'L_x', 'N_radial', 'min_box_size'}
        opts = dict(self._preprocess_options) if self._preprocess_options else {}
        unknown = set(opts) - allowed_keys
        if unknown:
            raise ValueError(
                f"[MITIM] Unknown preprocess_options keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed_keys)}"
            )
        ky_min_opt    = opts.get('ky_min', 0.1)
        L_x           = opts.get('L_x', 90.0)
        N_radial      = opts.get('N_radial', 256)
        min_box_size  = opts.get('min_box_size', 100)

        extraOptions = copy.deepcopy(extraOptions)

        for conflict_key in ('KY', 'BOX_SIZE', 'N_RADIAL'):
            if conflict_key in extraOptions:
                print(
                    f"\t- [preprocess] {conflict_key} was set in extraOptions; "
                    f"it will be overwritten by the preprocessing result",
                    typeMsg="w",
                )

        box_sizes = []
        n_radials = []
        ky_mins   = []

        print("\t- [preprocess] Computing BOX_SIZE and N_RADIAL per rho:")
        for i, rho in enumerate(self.rhos):
            input_rho = self.inputs_files[rho]
            q     = float(input_rho.plasma['Q'])
            shear = float(input_rho.plasma['S'])
            rmin  = float(input_rho.plasma['RMIN'])

            if isinstance(ky_min_opt, (list, np.ndarray)):
                ky_min = float(ky_min_opt[i])
            else:
                ky_min = float(ky_min_opt)

            box_size, n_radial_val = CGYROutils.compute_box_and_nradial(
                q=q,
                shear=shear,
                rmin=rmin,
                ky_min=ky_min,
                L_x=L_x,
                N_radial=N_radial,
                min_box_size=min_box_size,
            )

            print(
                f"\t\t* rho={rho:.4f}: q={q:.3f} s={shear:.3f} r/a={rmin:.3f} "
                f"KY={ky_min:.4f} -> BOX_SIZE={box_size} N_RADIAL={n_radial_val}",
                typeMsg="i",
            )

            box_sizes.append(box_size)
            n_radials.append(n_radial_val)
            ky_mins.append(ky_min)

        extraOptions['KY']       = ky_mins
        extraOptions['BOX_SIZE'] = box_sizes
        extraOptions['N_RADIAL'] = n_radials
        return extraOptions

    # Re-defined to make specific arguments explicit
    def read(
        self,
        tmin = 0.0,
        tmin_is_rel = True,
        minimal = False,
        last_tmin_for_linear = True,
        **kwargs
    ):

        super().read(
            tmin = tmin,
            tmin_is_rel = tmin_is_rel,
            minimal = minimal,
            last_tmin_for_linear = last_tmin_for_linear,
            **kwargs)

    def read_linear_scan(
        self,
        folder=None,
        preffix="scan",
        store_as_label=None,
        irho = 0,
        **kwargs
    ):
        '''
        Useful utility for when a folder contains subfolders like... scan0, scan1, scan2... with different ky
        '''
        
        if folder is None:
            folder = self.FolderGACODE
        
        main_label = kwargs.get('label', 'run1')
        del kwargs['label']
        
        # Get all folders inside "folder" that start with "preffix"
        subfolders = [subfolder for subfolder in Path(folder).glob(f"*{preffix}*") if subfolder.is_dir()]

        # ----------------------------------------------------------
        # Store in resutls
        # ----------------------------------------------------------

        # Store results in the form of {main_label}_KY_{subfolder}
        
        labels_in_results = []
        if len(subfolders) == 0:
            print(f"No subfolders found in {folder} with preffix {preffix}. Reading the folder directly.")
            labels_in_results.append(f'{main_label}_KY_scan0')
            self.read(label=labels_in_results[-1], folder=folder, **kwargs)
        else:   
            for subfolder in subfolders:
                labels_in_results.append(f'{main_label}_KY_{subfolder.name}')
                self.read(label=labels_in_results[-1], folder=subfolder, **kwargs)        

        # ----------------------------------------------------------
        # Make it a linear scan for the main label
        # ----------------------------------------------------------
        
        # Store special linear scan class as {main_label}
        
        labelsD = []
        for label in labels_in_results:
            parts = label.split('_')
            if len(parts) >= 3 and parts[-2] == "KY":
                # Extract the base name (scan1) and middle value (0.3/0.4)
                base_name = '_'.join(parts[0:-2])               
                labelsD.append(label)

        if store_as_label is not None:
            main_label = store_as_label

        self.results[main_label] = CGYROutils.CGYROlinear_scan(labelsD, self.results, irho=irho)

    # Redefined to remove potential large objects
    def save_pickle(self, file, **kwargs):

        class_to_store = super().prepare_for_save()

        # cgyrodata carries _thread.lock through its internal HDF/file handles,
        # which breaks copy.deepcopy. Temporarily detach each cgyrodata from the
        # live object, deepcopy the rest, then restore them so the in-memory
        # object is unchanged after save.
        stashed = []
        for key in class_to_store.results:
            for irho in range(len(class_to_store.results[key]['output'])):
                out = class_to_store.results[key]['output'][irho]
                if 'cgyrodata' in out.__dict__:
                    stashed.append((out, out.cgyrodata))
                    out.cgyrodata = None
        try:
            class_to_store = copy.deepcopy(class_to_store)
        finally:
            for out, data in stashed:
                out.cgyrodata = data

        super().save_pickle(file, class_to_store = class_to_store, **kwargs)
                
                
    def plot(
        self,
        labels=[""],
        fn=None,
        include_2D=True,
        common_colorbar=True):
        
        # If it has radii, we need to correct the labels
        labels = self._correct_rhos_labels(labels)
    
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook
            self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")
        else:
            self.fn = fn

        fig = self.fn.add_figure(label="Fluxes (time)")
        axsFluxes_t = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Fluxes (ky)")
        axsFluxes_ky = fig.subplot_mosaic(
            """
            ACE
            BDF
            """
        )
        fig = self.fn.add_figure(label="Intensities (time)")
        axsIntensities = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (ky)")
        axsIntensities_ky = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (kx)")
        axsIntensities_kx = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Cross-phases (ky)")
        axsCrossPhases = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Turbulence (linear)")
        axsTurbulence = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )

        # One "Averaging" figure per case: window selection diagnostics of the primary fluxes
        for i, label in enumerate(labels):
            if hasattr(self.results[label], 'averaging'):
                fig = self.fn.add_figure(label=f"Averaging, {label}")
                self.results[label].averaging.plot(fig=fig, color=GRAPHICStools.listColors()[i % len(GRAPHICStools.listColors())], label_plot=label)

        create_ballooning = False
        for label in labels:
            if 'phi_ballooning' in self.results[label].__dict__:
                create_ballooning = True
            
        if create_ballooning:

            fig = self.fn.add_figure(label="Ballooning")
            axsBallooning = fig.subplot_mosaic(
                """
                135
                246
                """
                )
        else:
            axsBallooning = None
        
        
        if include_2D:
            axs2D = []
            for i in range(len(labels)):
                fig = self.fn.add_figure(label="Turbulence (2D), " + labels[i])
                
                mosaic = _2D_mosaic(4) # Plot 4 times by default
                
                axs2D.append(fig.subplot_mosaic(mosaic))
        
        fig = self.fn.add_figure(label="Timing")
        axsTiming = fig.subplot_mosaic(
            """
            AC
            BC
            """
        )

        fig = self.fn.add_figure(label="Inputs")
        axsInputs = fig.subplot_mosaic(
            """
            A
            B
            """
        )

        
        colors = GRAPHICStools.listColors()

        # Safety net: if one sub-plot method raises (e.g. a missing optional
        # CGYRO output file that the per-panel guards didn't cover), we do
        # NOT want it to abort the whole notebook build. Wrap each call in a
        # try/except that logs a warning and continues.
        def _safe_plot(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as _e:
                print(f"\t- {fn.__name__} failed ({_e}); skipping this figure and continuing", typeMsg='w')
                return None

        colorbars_all = []  # Store all colorbars for later use
        for j in range(len(labels)):

            _safe_plot(self.plot_fluxes,
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            _safe_plot(self.plot_fluxes_ky,
                axs=axsFluxes_ky,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            _safe_plot(self.plot_intensities_ky,
                axs=axsIntensities_ky,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,
            )
            _safe_plot(self.plot_intensities,
                axs=axsIntensities,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            _safe_plot(self.plot_intensities_kx,
                axs=axsIntensities_kx,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            _safe_plot(self.plot_turbulence,
                axs=axsTurbulence,
                label=labels[j],
                c=colors[j],
            )
            _safe_plot(self.plot_cross_phases,
                axs=axsCrossPhases,
                label=labels[j],
                c=colors[j],
            )
            if create_ballooning:
                _safe_plot(self.plot_ballooning,
                    axs=axsBallooning,
                    label=labels[j],
                    c=colors[j],
                )

            if include_2D:

                colorbars = _safe_plot(self.plot_2D,
                    axs=axs2D[j],
                    label=labels[j],
                )

                colorbars_all.append(colorbars)

            _safe_plot(self.plot_timing,
                axs=axsTiming,
                label=labels[j],
                c=colors[j],
            )

            _safe_plot(self.plot_inputs,
                ax=axsInputs["A"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
                normalization_label= labels[0],  # Normalize to the first label
                only_plot_differences=len(labels) > 1,  # Only plot differences if there are multiple labels
            )

            _safe_plot(self.plot_inputs,
                ax=axsInputs["B"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
            )
            
        axsInputs["A"].axhline(
            1.0,
            color="k",
            ls="--",
            lw=2.0
        )
        
        GRAPHICStools.adjust_subplots(axs=axsInputs, vertical=0.4, horizontal=0.3)
        
        # Modify the colorbars to have a common range. Skip label indices where
        # plot_2D was a no-op — e.g. when bin.cgyro.kxky_phi / kxky_n / kxky_e
        # aren't available (MOMENT_PRINT_FLAG / FIELD_PRINT_FLAG off). _safe_plot
        # returns None in that case, so colorbars_all can carry None entries
        # which would crash the common-range aggregation below.
        valid_cbs = [cbs for cbs in colorbars_all if cbs is not None]
        if include_2D and common_colorbar and len(valid_cbs) > 0:
            for var in ['phi', 'n', 'e']:
                min_val = np.inf
                max_val = -np.inf
                for cbs in valid_cbs:
                    cb = cbs[0][var]
                    vals = cb.mappable.get_clim()
                    min_val = min(min_val, vals[0])
                    max_val = max(max_val, vals[1])

                for cbs in valid_cbs:
                    for it in range(len(cbs)):
                        cb = cbs[it][var]
                        cb.mappable.set_clim(min_val, max_val)
                        cb.update_ticks()
                        #cb.set_label(f"{var} (common range)")

        # Back to the original labels before _correct_rhos_labels
        self.results = self.results_all

    def plot_timing(self, axs=None, label="", c="b"):
        """
        Characterize the computational cost of the run from out.cgyro.timing
        (wall-clock seconds spent in each code section, one row per data output):
            A: wall time per data output (TOTAL column)
            B: cumulative wall time (setup time included as offset)
            C: share of the total run time spent in each code section
        """
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(15, 8))
            axs = fig.subplot_mosaic(
                """
                AC
                BC
                """
            )

        data = self.results[label]
        if "timing" not in data.__dict__:
            print(f"\t- No timing information for {label}; skipping timing plot", typeMsg="w")
            return

        steps = np.arange(1, len(data.timing_total) + 1)
        setup_time = sum(getattr(data, "timing_setup", {}).values())
        total_time = setup_time + data.timing_total.sum()

        # A: cost of each data output
        ax = axs["A"]
        ax.plot(steps, data.timing_total, "-o", c=c, lw=1.0, markersize=3, label=label)
        ax.axhline(data.timing_total.mean(), c=c, ls="--", lw=0.5)
        ax.set_xlabel("Data output #")
        ax.set_ylabel("Wall time per output (s)")
        ax.set_title("Cost per data output (dashed: mean)")
        # Pin bottom at 0 but keep the top growing to fit every overlaid case: set_ylim
        # disables y-autoscale, so a plain set_ylim(bottom=0) on the first case would
        # freeze the top to that case's range and clip the others.
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], float(np.nanmax(data.timing_total)) * 1.05))
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc="best", prop={"size": 8})

        # B: cumulative cost
        ax = axs["B"]
        cumulative = (setup_time + np.cumsum(data.timing_total)) / 60.0
        ax.plot(
            steps,
            cumulative,
            "-o", c=c, lw=1.0, markersize=3,
            label=f"{label} (setup {setup_time:.1f}s, total {total_time/60.0:.1f}min)",
        )
        ax.set_xlabel("Data output #")
        ax.set_ylabel("Cumulative wall time (min)")
        ax.set_title("Cumulative cost (setup included)")
        # Same as A: expand the top across cases (cumulative is monotonic, so [-1] is the max)
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], float(cumulative[-1]) * 1.05))
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc="best", prop={"size": 8})

        # C: where the time goes, by code section
        ax = axs["C"]
        share = 100.0 * data.timing.sum(axis=0) / data.timing.sum()
        y = np.arange(len(data.timing_names))
        ax.plot(share, y, "o", c=c, markersize=8, label=label)
        ax.set_yticks(y)
        ax.set_yticklabels(data.timing_names)
        ax.invert_yaxis()
        ax.set_xlabel("Share of run time (%)")
        ax.set_title("Time per code section")
        ax.set_xlim(left=0)
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc="best", prop={"size": 8})

    def plot_inputs(self, ax = None, label="", c="b", ms = 10, normalization_label=None, only_plot_differences=False):

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(18, 9))

        rel_tol = 1e-2

        legadded = False
        for i, ikey in enumerate(self.results[label].params1D):
            
            z = self.results[label].params1D[ikey]
            
            if normalization_label is not None:
                z0 = self.results[normalization_label].params1D[ikey]
                zp = z/z0 if z0 != 0 else 0
                label_plot = f"{label} / {normalization_label}"
            else:
                label_plot = label
                zp = z

            if (not only_plot_differences) or (not np.isclose(z, z0, rtol=rel_tol)):
                ax.plot(ikey,zp,'o',markersize=ms,color=c,label=label_plot if not legadded else '')
                legadded = True

        if normalization_label is not None:
            if only_plot_differences:
                ylabel = f"Parameters (DIFFERENT by {rel_tol*100:.2f}%) relative to {normalization_label}"
            else:
                ylabel = f"Parameters relative to {normalization_label}"
        else:
            ylabel = "Parameters"

        ax.set_xlabel("Parameter")
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel(ylabel)
        GRAPHICStools.addDenseAxis(ax)
        if legadded:
            ax.legend(loc='best')

    def plot_intensities(self, axs = None, label= "cgyro1", c="b", addText=True):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
            
        ax = axs["A"]
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta \\phi/\\phi_0$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].t, self.results[label].apar_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}, $A_\\parallel$")
            ax.plot(self.results[label].t, self.results[label].bpar_rms_sumnr_sumn*100.0, '--', c=c, lw=2, label=f"{label}, $B_\\parallel$")
            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity fluctuations')
        

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["C"]
        try:
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_e/n_{e,0}/n_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Density intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["D"]
        try:
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_e/T_{e,0}/T_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Temperature intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))




        ax = axs["E"]
        try:
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Density intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["F"]
        try:
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/T_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Temperature intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].t, self.results[label].ni_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Density intensity fluctuations')


        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].t, self.results[label].Ti_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Temperature intensity fluctuations')


        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_ky(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].phi_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].phi_rms_sumnr_mean-self.results[label].phi_rms_sumnr_std, self.results[label].phi_rms_sumnr_mean+self.results[label].phi_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta\phi/\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].ky, self.results[label].apar_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].apar_rms_sumnr_mean-self.results[label].apar_rms_sumnr_std, self.results[label].apar_rms_sumnr_mean+self.results[label].apar_rms_sumnr_std, color=c, alpha=0.2)
            ax.plot(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean, '--', markersize=5, color=c, label=label+', $B_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean-self.results[label].bpar_rms_sumnr_std, self.results[label].bpar_rms_sumnr_mean+self.results[label].bpar_rms_sumnr_std, color=c, alpha=0.2)

            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta F_\parallel/F_{\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs. $k_\\theta\\rho_s$')
        
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["C"]
        try:
            ax.plot(self.results[label].ky, self.results[label].ne_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].ne_rms_sumnr_mean-self.results[label].ne_rms_sumnr_std, self.results[label].ne_rms_sumnr_mean+self.results[label].ne_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Electron temperature intensity
        ax = axs["D"]
        try:
            ax.plot(self.results[label].ky, self.results[label].Te_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].Te_rms_sumnr_mean-self.results[label].Te_rms_sumnr_std, self.results[label].Te_rms_sumnr_mean+self.results[label].Te_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["E"]
        try:
            ax.plot(self.results[label].ky, self.results[label].ni_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].ni_rms_sumnr_mean-self.results[label].ni_rms_sumnr_std, self.results[label].ni_rms_sumnr_mean+self.results[label].ni_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Ion temperature intensity
        ax = axs["F"]
        try:
            ax.plot(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean-self.results[label].Ti_rms_sumnr_std, self.results[label].Ti_rms_sumnr_mean+self.results[label].Ti_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].ni_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)


        # Ion temperature intensity
        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].Ti_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_kx(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta \\phi/\\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs kx')
        ax.legend(loc='best', prop={'size': 8},)
        ax.set_yscale('log')
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["C"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].kx, self.results[label].apar_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].bpar_rms_sumn_mean, '--', markersize=1.0, lw=1.0, color=c, label=label+', $B_\\parallel$ (mean)')

            ax.legend(loc='best', prop={'size': 8},)


        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs kx')
        ax.set_yscale('log')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["B"]
        try:
            ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
            ax.plot(self.results[label].kx, self.results[label].ne_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')
            ax.legend(loc='best', prop={'size': 8},)
            ax.set_yscale('log')
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs kx')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        # Electron temperature intensity
        ax = axs["D"]
        try:
            ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
            ax.plot(self.results[label].kx, self.results[label].Te_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')
            ax.legend(loc='best', prop={'size': 8},)
            ax.set_yscale('log')
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs kx')
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)


    def plot_cross_phases(self, axs = None, label= "cgyro1", c="b"):

        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
        m = GRAPHICStools.listmarkers()
            
        ax = axs["A"]
        try:
            ax.plot(self.results[label].ky, self.results[label].neTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].neTe_kx0_mean-self.results[label].neTe_kx0_std, self.results[label].neTe_kx0_mean+self.results[label].neTe_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n + kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_e-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_e-T_e$ cross-phase ($k_x=0$)')


        ax = axs["B"]
        try:
            ax.plot(self.results[label].ky, self.results[label].niTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].niTi_kx0_mean-self.results[label].niTi_kx0_std, self.results[label].niTi_kx0_mean+self.results[label].niTi_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n + kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_i-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_i-T_i$ cross-phase ($k_x=0$)')

        ax = axs["C"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phine_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phine_kx0_mean-self.results[label].phine_kx0_std, self.results[label].phine_kx0_mean+self.results[label].phine_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_e$ cross-phase ($k_x=0$)')

        ax = axs["D"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phini_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phini_kx0_mean-self.results[label].phini_kx0_std, self.results[label].phini_kx0_mean+self.results[label].phini_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ cross-phase ($k_x=0$)')


        ax = axs["E"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phiTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phiTe_kx0_mean-self.results[label].phiTe_kx0_std, self.results[label].phiTe_kx0_mean+self.results[label].phiTe_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_e$ cross-phase ($k_x=0$)')


        ax = axs["F"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phiTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phiTi_kx0_mean-self.results[label].phiTi_kx0_std, self.results[label].phiTi_kx0_mean+self.results[label].phiTi_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ cross-phase ($k_x=0$)')


        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].phiTi_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ (all) cross-phase ($k_x=0$)')


        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].phini_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ (all) cross-phase ($k_x=0$)')
        
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_ballooning(self, time = None, label="cgyro1", c="b", axs=None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                135
                246
                """
            )

        if time is None:
            time = np.min([self.results[label].tmin, self.results[label].tmax_fluct])
        
        it = np.argmin(np.abs(self.results[label].t - time))

        colorsC, _ = GRAPHICStools.colorTableFade(
            len(self.results[label].ky),
            startcolor=c,
            endcolor=c,
            alphalims=[1.0, 0.4],
        )

        ax = axs['1']
        for ky in range(len(self.results[label].ky)):
            for var, axsT in zip(
                ["phi_ballooning", "apar_ballooning", "bpar_ballooning"],
                [[axs['1'], axs['2']], [axs['3'], axs['4']], [axs['5'], axs['6']]],
            ):

                f = self.results[label].__dict__[var][:, it]
                y1 = np.real(f)
                y2 = np.imag(f)
                x = self.results[label].theta_ballooning / np.pi

                # Normalize
                y1_max = np.max(np.abs(y1))
                y2_max = np.max(np.abs(y2))
                y1 /= y1_max
                y2 /= y2_max

                ax = axsT[0]
                ax.plot(
                    x,
                    y1,
                    color=colorsC[ky],
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y1_max:.2e})",
                )
                ax = axsT[1]
                ax.plot(
                    x, 
                    y2, 
                    color=colorsC[ky], 
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y2_max:.2e})",
                )


        ax = axs['1']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta\\phi$)")
        ax.set_title("$\\delta\\phi$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([-2 * np.pi, 2 * np.pi])

        ax = axs['3']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta A\\parallel$)")
        ax.set_title("$\\delta A\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['5']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta B\\parallel$)")
        ax.set_title("$\\delta B\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['2']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta\\phi$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['4']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta A\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['6']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta B\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)


        for ax in [axs['1'], axs['3'], axs['5'], axs['2'], axs['4'], axs['6']]:
            ax.axvline(x=0, lw=0.5, ls="--", c="k")
            ax.axhline(y=0, lw=0.5, ls="--", c="k")
            
            
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_2D(self, label="cgyro1", axs=None, times = None):

        # plot_2D needs kxky_phi (always), kxky_n (MOMENT_PRINT_FLAG=1) and
        # kxky_e (MOMENT_PRINT_FLAG=1). If any of the underlying fluctuation
        # arrays is missing (typically because the user disabled the print
        # flags to save disk / retrieval time), skip cleanly instead of
        # aborting the whole plot chain.
        _res = self.results.get(label) if hasattr(self, 'results') else None
        if _res is None or not all(hasattr(_res, _a) for _a in ('phi', 'ne', 'Te')):
            print("\t- plot_2D skipped: needs phi/ne/Te (requires bin.cgyro.kxky_phi + kxky_n + kxky_e; enable MOMENT_PRINT_FLAG=1 / FIELD_PRINT_FLAG=1)", typeMsg='w')
            return

        if times is None:
            times = []
            
            number_times = len(axs)//3 if axs is not None else 4

            try:
                times = [self.results[label].t[-1-i*10] for i in range(number_times)]
            except IndexError:
                 times = [self.results[label].t[-1-i*1] for i in range(number_times)]

        if axs is None:

            mosaic = _2D_mosaic(len(times))

            plt.ion()
            fig = plt.figure(figsize=(18, 9))
            axs = fig.subplot_mosaic(mosaic)

        # Pre-calculate global min/max for each field type across all times
        phi_values = []
        n_values = []
        e_values = []
        
        for time in times:
            it = np.argmin(np.abs(self.results[label].t - time))
            
            # Get phi values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)
            phi_values.append(fp)
            
            # Get n values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)
            n_values.append(fp)
            
            # Get e values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)
            e_values.append(fp)
        
        # Calculate global ranges
        phi_max = np.max([np.max(np.abs(fp)) for fp in phi_values])
        phi_min, phi_max = -phi_max, +phi_max
        
        n_max = np.max([np.max(np.abs(fp)) for fp in n_values])
        n_min, n_max = -n_max, +n_max
        
        e_max = np.max([np.max(np.abs(fp)) for fp in e_values])
        e_min, e_max = -e_max, +e_max

        colorbars = []  # Store colorbar references
        # Now plot with consistent colorbar ranges
        for time_i, time in enumerate(times):
            
            print(f"\t- Plotting 2D turbulence for {label} at time {time}")
            
            it = np.argmin(np.abs(self.results[label].t - time))
            
            cfig = axs[str(time_i+1)].get_figure()
            
            # Phi plot
            ax = axs[str(time_i+1)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)

            cs1 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(phi_min,phi_max,(phi_max-phi_min)/256),cmap=plt.get_cmap('jet'))
            cphi = cfig.colorbar(cs1, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta\\phi/\\phi_0$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # N plot
            ax = axs[str(time_i+1+len(times))]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)

            cs2 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(n_min,n_max,(n_max-n_min)/256),cmap=plt.get_cmap('jet'))
            cn = cfig.colorbar(cs2, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta n_e/n_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # E plot
            ax = axs[str(time_i+1+len(times)*2)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)

            cs3 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(e_min,e_max,(e_max-e_min)/256),cmap=plt.get_cmap('jet'))
            ce = cfig.colorbar(cs3, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta E_e/E_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            # Store the colorbar objects with their associated contour plots
            colorbars.append({
                'phi': cphi,
                'n': cn,
                'e': ce
            })

        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.4, horizontal=0.3)

        return colorbars
        
    def _to_real_space(self, variable = 'kxky_phi', species = None, label="cgyro1", theta_plot = 0, it = -1):
        
        # from pygacode
        def maptoreal_fft(nr,nn,nx,ny,c):

            d = np.zeros([nx,nn],dtype=complex)
            for i in range(nr):
                p = i-nr//2
                if -p < 0:
                    k = -p+nx
                else:
                    k = -p
                d[k,0:nn] = np.conj(c[i,0:nn])
            f = np.fft.irfft2(d,s=[nx,ny],norm='forward')*0.5

            # Correct for half-sum
            f = 2*f

            return f

        # Real space
        nr = self.results[label].cgyrodata.n_radial
        nn = self.results[label].cgyrodata.n_n
        craw = self.results[label].cgyrodata.__dict__[variable]
        
        itheta = np.argmin(np.abs(self.results[label].theta_stored-theta_plot))
        if species is None:
            c = craw[:,itheta,:,it]
        else:
            c = craw[:,itheta,species,:,it]

        nx = self.results[label].cgyrodata.__dict__[variable].shape[0]
        ny = nx
        
        # Arrays
        x = np.arange(nx)*2*np.pi/nx
        y = np.arange(ny)*2*np.pi/ny
        f = maptoreal_fft(nr,nn,nx,ny,c)
        
        # Physical maxima
        ky1 = self.results[label].cgyrodata.ky[1] if len(self.results[label].cgyrodata.ky) > 1 else self.results[label].cgyrodata.ky[0]
        xmax = self.results[label].cgyrodata.length
        ymax = (2*np.pi)/np.abs(ky1)
        xp = x/(2*np.pi)*xmax
        yp = y/(2*np.pi)*ymax

        # Periodic extensions
        xp = np.append(xp,xmax)
        yp = np.append(yp,ymax)
        fp = np.zeros([nx+1,ny+1])
        fp[0:nx,0:ny] = f[:,:]
        fp[-1,:] = fp[0,:]
        fp[:,-1] = fp[:,0]
        
        return xp, yp, fp
        
    def plot_quick_linear(self, labels=["cgyro1"], fig=None):
 
        colors = GRAPHICStools.listColors()
        ls = GRAPHICStools.listLS()

        if fig is None:
            fig = plt.figure(figsize=(15,9))

        axs = fig.subplot_mosaic(
            """
            12
            34
            """
        )
            
        def _plot_linear_stability(axs, labels, label_base,col_lin ='b', start_cont=0):

            irho = self.results[label_base].irho

            for cont, label in enumerate(labels):
                c = self.results[label]['output'][irho]
                baseColor = colors[cont+start_cont+1]
                colorsC, _ = GRAPHICStools.colorTableFade(
                    len(c.ky),
                    startcolor=baseColor,
                    endcolor=baseColor,
                    alphalims=[1.0, 0.4],
                )

                ax = axs['1']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.g[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$, $r/a={c.roa:.2f}$",
                        ls = ls[irho]
                    )

                ax = axs['2']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.f[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$, $r/a={c.roa:.2f}$",
                        ls = ls[irho]
                    )

            roa = self.results[self.results[label_base].labels[0]]['output'][irho].roa

            GACODEplotting.plotTGLFspectrum(
                [axs['3'], axs['4']],
                abs(self.results[label_base].ky),
                self.results[label_base].g_mean,
                freq=self.results[label_base].f_mean,
                coeff=0.0,
                c=col_lin,
                ls="-",
                lw=1,
                label=f"r/a = {roa}",
                facecolors=colors,
                markersize=50,
                alpha=1.0,
                titles=["Growth Rate", "Real Frequency"],
                removeLow=1e-4,
                ylabel=True,
            )
            axs['3'].legend(loc='best', prop={'size': 8},)
            
            return cont

        co = -1
        for i,label0 in enumerate(labels):
            co = _plot_linear_stability(axs, self.results[label0].labels, label0, start_cont=co, col_lin=colors[i])

        ax = axs['1']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$\\gamma$ $(c_s/a)$")
        ax.set_title("Growth Rate")
        ax.set_xlim(left=0)
        ax.legend(loc='best', prop={'size': 8},)
        
        ax = axs['2']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.set_ylabel("$\\omega$ $(c_s/a)$")
        ax.set_title("Real Frequency")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_xlim(left=0)
        
        for ax in [axs['1'], axs['2'], axs['3'], axs['4']]:
            GRAPHICStools.addDenseAxis(ax)
        
        plt.tight_layout()

class CGYROinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.cgyro.controls",
            code="CGYRO",
            n_species='N_SPECIES',
        )

def _2D_mosaic(n_times):

    num_cols = n_times

    # Create the mosaic layout dynamically
    mosaic = []
    counter = 1
    for _ in range(3):
        row = []
        for _ in range(num_cols):
            row.append(str(counter))
            counter += 1
        mosaic.append(row)
        
    return mosaic