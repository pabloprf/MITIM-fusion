"""
Rescue of radii that stop progressing: a stalled radius of a SLURM submission is cancelled, cleaned
and resubmitted on its own (StallRescuer, with its ledger in cgyro_submission.json), a radius whose
element hit its wall clock is continued in place by a job of its own within the same submission
(StallRescuer, max_continuations_per_rho), and a radius a previous execution left unfinished in
scratch is continued in place (rescue_interrupted_runs).
"""
import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path
from mitim_tools.misc_tools.FARMINGtools import SlurmState
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.completion import CompletionSpec
from mitim_tools.simulation_tools.gk_control.status import TaskState
from mitim_tools.misc_tools.LOGtools import printMsg as print


def _slurm_elapsed_seconds(token):
    '''sacct Elapsed ("[D-]HH:MM:SS", or "MM:SS") in seconds; None when it cannot be parsed.'''
    try:
        days, _, clock = token.strip().rpartition("-")
        seconds = 0
        for part in clock.split(":"):
            seconds = 60 * seconds + int(part)
        return seconds + 86400 * int(days or 0)
    except ValueError:
        return None


def _slurm_state_for_target(job, target):
    '''
    (state, seconds running) for `target` (a "<jobid>" or "<jobid>_<idx>" string), queried via
    `sacct -X --format=State,Elapsed`, then `squeue -j -o "%T|%M"` when sacct has nothing: a job
    submitted seconds ago has no accounting record yet but is already in the queue, and without this
    fallback a PENDING element whose folder holds old files (an interrupted run kept for the in-place
    rescue, a requeue) was cancelled as stalled before it ever started. For a requeued element sacct
    reports only its latest instance, so the elapsed time counts from the latest (re)start. state is
    the uppercased token, or None if neither command produced useful output (e.g. site without sacct
    and a job that already left the queue); the caller treats None as "no signal -- continue with the
    existing decision tree" rather than as a terminal verdict. seconds is None unless the state is RUNNING.
    '''
    first = ""
    for cmd in (f'sacct -j {target} -X --format=State,Elapsed -n -P', f'squeue -h -j {target} -o "%T|%M"'):
        try:
            out, _err = job.execute(cmd, printYN=False)
        except Exception as e:
            print(f"\t    * {cmd.split()[0]} query for {target} failed ({type(e).__name__}: {e}); proceeding without per-task slurm-state guard", typeMsg='w')
            return None, None
        if isinstance(out, bytes):
            out = out.decode(errors='replace')
        # Multiple lines possible (sacct emits one row per step). The first non-empty
        # line is the parent task state — what we actually care about.
        first = next((l.strip() for l in (out or "").splitlines() if l.strip()), "")
        if first:
            break
    if not first:
        return None, None
    state_token, _, elapsed_token = first.partition("|")
    # State strings can carry trailing markers like "CANCELLED+" or
    # "CANCELLED by 12345"; canonicalize to the leading word.
    state = state_token.split()[0].rstrip("+").upper() if state_token.split() else None
    running_s = _slurm_elapsed_seconds(elapsed_token) if (state == "RUNNING" and elapsed_token) else None
    return state, running_s


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
    continuation_jobids: list = field(default_factory=list)   # in-place continuations, also in child_jobids

    @classmethod
    def from_json(cls, entry):
        entry = entry or {}
        return cls(
            n_attempts=int(entry.get("n_attempts", 0)),
            child_jobids=list(entry.get("child_jobids", []) or []),
            last_action_at=entry.get("last_action_at"),
            status=str(entry.get("status", "active")),
            continuation_jobids=list(entry.get("continuation_jobids", []) or []),
        )

    def to_json(self):
        entry = {
            "n_attempts": self.n_attempts,
            "child_jobids": self.child_jobids,
            "last_action_at": self.last_action_at,
            "status": self.status,
        }
        if self.continuation_jobids:
            entry["continuation_jobids"] = self.continuation_jobids
        return entry

    @property
    def n_continuations(self):
        return len(self.continuation_jobids)

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

    A radius whose element sacct reports ended by the wall clock, a preemption or a node failure, with a
    restart written (first token of out.cgyro.tag a positive output index), is instead continued in place
    as soon as its row goes stale or its element leaves the queue: nothing is deleted, the remote input's
    run length is trimmed so the continuation ends at the original t_end (.mitim_t_end) and it is
    resubmitted alone (restart_flag 1, t continues). Capped at `max_continuations_per_rho`, counted apart
    from the stall retries; 0 (no rescue_interrupted) turns it off.
    '''

    CLEANUP_FILES = ["out.cgyro.timing", "out.cgyro.tag", "out.cgyro.info", "slurm_output.dat", "slurm_error.dat"]

    # sacct end states that leave the run intact on disk, ready to be continued from its last restart
    CONTINUE_STATES = (SlurmState.TIMEOUT, SlurmState.PREEMPTED, SlurmState.NODE_FAIL)

    def __init__(self, sim, settings, session=None):
        self.sim = sim
        self.job = sim.simulation_job
        self.session = session   # live ssh session to reuse; None -> one is opened for this rescue
        self.init_kill_s = int(settings.get("stall_init_kill_seconds", 1800))
        self.run_kill_s = int(settings.get("stall_running_kill_seconds", 1800))
        self.cap = int(settings.get("max_resubmits_per_rho", 1))
        self.stall_enabled = bool(settings.get("enabled", False))
        self.continuation_cap = int(settings.get("max_continuations_per_rho", 0))
        organize = getattr(sim, "kwargs_organize", None) or {}
        self.array_index_by_folder = organize.get("array_index_by_folder", {})
        self.per_folder_commands = organize.get("per_folder_commands", {})
        self.metadata_dirty = False
        self._slurm_states = {}

    @property
    def ledger(self):
        if getattr(self.sim, "_resubmit_ledger", None) is None:
            self.sim._resubmit_ledger = {}
        return self.sim._resubmit_ledger

    def rescue(self, rows):
        if not self.array_index_by_folder or not self.per_folder_commands:
            return   # not a slurm_array submission — single-task rescue does not apply

        candidates = self._candidates(rows)
        continuations = self._continuation_candidates(rows)
        if not (candidates or continuations):
            return
        if candidates:
            print(f"\t- [auto-resubmit] {len(candidates)} stalled task(s) past the kill threshold; attempting rescue", typeMsg='w')

        if self.session is not None:
            self._act(candidates, continuations)
        else:
            try:
                self.job.connect()
            except Exception as e:
                print(f"\t- [auto-resubmit] could not open SSH connection ({type(e).__name__}: {e}); skipping rescue this poll", typeMsg='w')
                return
            try:
                self._act(candidates, continuations)
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

    def _act(self, candidates, continuations):
        '''Continue the wall-hit radii first; a continued radius is no stall to rescue.'''
        continued = self._continue_all(continuations)
        self._rescue_all([(row, threshold) for row, threshold in candidates if row["folder"] not in continued])

    def _candidates(self, rows):
        '''(row, threshold) for every row stalled longer than its own kill threshold.'''
        candidates = []
        if not self.stall_enabled:
            return candidates
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
            f"checking slurm before rescue {attempt}/{self.cap}",
            typeMsg='w',
        )

        target, rescuing_child = self._resolve_target(folder, entry)
        if target is None:
            return
        slurm_state, running_s = self._slurm_state(target)
        if self._slurm_says_dead(folder, target, entry, slurm_state):
            return
        if self._restarted_recently(target, slurm_state, running_s, threshold):
            return
        if not self._scancel(folder, target):
            return
        if not self._clean_remote(folder):
            return
        self._resubmit(folder, entry, attempt, rescuing_child, stalled_on_node=running_s is not None)

    def _slurm_state(self, target):
        '''_slurm_state_for_target, asked once per target and poll (the continuation and the stall pass share it).'''
        if target not in self._slurm_states:
            self._slurm_states[target] = _slurm_state_for_target(self.job, target)
        return self._slurm_states[target]

    def _continuation_candidates(self, rows):
        '''
        Rows that may be a wall-hit element to continue in place: stale or timed out, with a restart
        written, and continuations left. sacct decides in _continue_one; a row it finds still running
        goes on to the stall pass untouched.
        '''
        if not self.continuation_cap:
            return []
        candidates = []
        for row in rows:
            if row["effective"] not in (TaskState.STALLED.value, TaskState.TIMED_OUT.value) or _restart_output(row.get("tag_token")) is None:
                continue
            entry = LedgerEntry.from_json(self.ledger.get(row["folder"]))
            if entry.terminal_no_rescue or entry.n_continuations >= self.continuation_cap:
                continue
            candidates.append(row)
        return candidates

    def _continue_all(self, rows):
        '''The folders continued in place this poll.'''
        continued = set()
        for row in rows:
            entry = LedgerEntry.from_json(self.ledger.get(row["folder"]))
            if self._continue_one(row, entry):
                self.ledger[row["folder"]] = entry.to_json()
                continued.add(row["folder"])
        return continued

    def _continue_one(self, row, entry):
        folder = row["folder"]
        code_call_str = self.per_folder_commands.get(folder)
        target, rescuing_child = self._resolve_target(folder, entry)
        if target is None or not code_call_str:
            return False
        slurm_state, _ = self._slurm_state(target)
        state = SlurmState.from_token(slurm_state)
        if state not in self.CONTINUE_STATES:
            return False

        plan = self._continuation_input(folder)
        if plan is None:
            return False
        text, t_tag, t_end, note = plan
        if not self._write_remote_input(folder, text):
            return False

        n = entry.n_continuations + 1
        bad_node = self.job.node_of(self.array_index_by_folder.get(folder)) if (state is SlurmState.NODE_FAIL and not rescuing_child) else None
        try:
            new_jobid = self.job.resubmit_single_task(code_call_str, f"_continue_{Path(folder).name}_c{n}", exclude_node=bad_node)
        except Exception as e:
            print(f"\t  - [continue] {folder}: resubmit_single_task raised ({type(e).__name__}: {e}); will retry next poll", typeMsg='w')
            return False
        if not new_jobid:
            print(f"\t  - [continue] {folder}: resubmit_single_task returned no jobid; will retry next poll", typeMsg='w')
            return False

        entry.child_jobids.append(new_jobid)
        entry.continuation_jobids.append(new_jobid)
        entry.last_action_at = _utc_now()
        # a live child again: the radius is no longer closed for the extras that wait on it
        if entry.status.startswith("EXHAUSTED"):
            entry.status = "active"
        self.metadata_dirty = True
        print(
            f"\t  - [continue] {folder}: {target} ended {slurm_state} at t_tag={t_tag:g}; continuing in place to "
            f"t_end={t_end:g}{note}: jobid={new_jobid} (continuation {n}/{self.continuation_cap}, exclude={bad_node})",
            typeMsg='i',
        )
        return True

    def _continuation_input(self, folder):
        '''
        (input text trimmed to end at t_end, t_tag, t_end, log note) for the continuation of `folder`, or None.
        t_tag is the time CGYRO resumes from (rescue_spec progress line of out.cgyro.tag), t_end the time the
        launch had to reach (.mitim_t_end, written by cgyro_requeue_trim.sh; else .mitim_t0 + the time_key).
        The trim is _trim_time_key with total=t_end, the one the next-submission rescue applies with the
        original run length; the continuation's own launch then records t_tag + remaining = t_end again.
        '''
        spec = self.sim.run_specifications.get("rescue_spec") or {}
        input_file = self.sim.run_specifications.get("input_file")
        line = spec.get("progress_line")
        progress = f"sed -n {line}p" if line else "tail -n1"
        sep = markers.CONTINUE_SPLIT
        cmd = (f'cd "{self.job.folderExecution}/{folder}" && {progress} {spec.get("progress_file")} 2>/dev/null; echo {sep}; '
               f'cat {markers.T_END} 2>/dev/null; echo {sep}; cat {markers.T0} 2>/dev/null; echo {sep}; cat {input_file}')
        try:
            out, _ = self.job.execute(cmd, printYN=False)
        except Exception as e:
            print(f"\t  - [continue] {folder}: remote read failed ({type(e).__name__}: {e}); not continued this poll", typeMsg='w')
            return None
        if isinstance(out, bytes):
            out = out.decode(errors='replace')
        parts = (out or "").split(f"{sep}\n")
        if len(parts) != 4:
            print(f"\t  - [continue] {folder}: unexpected remote read ({len(parts)} part(s)); not continued", typeMsg='w')
            return None
        tag, t_end_line, t0, text = parts
        try:
            t_tag = float(tag.split()[0])
            m = re.search(rf"^{spec.get('time_key')}\s*=\s*(\S+)", text, flags=re.M)
            t_end = float(t_end_line.split()[1]) if len(t_end_line.split()) > 1 else float(t0.split()[0]) + float(m.group(1))
        except (IndexError, ValueError, AttributeError):
            print(f"\t  - [continue] {folder}: resume time or t_end unreadable; not continued", typeMsg='w')
            return None
        text, note = _trim_time_key(text, spec, t_tag, total=t_end)
        return text, t_tag, t_end, note

    def _write_remote_input(self, folder, text):
        input_file = self.sim.run_specifications.get("input_file")
        cmd = f'cat > "{self.job.folderExecution}/{folder}/{input_file}" <<\'MITIM_INPUT_EOF\'\n{text.rstrip(chr(10))}\nMITIM_INPUT_EOF'
        try:
            self.job.execute(cmd, printYN=False)
        except Exception as e:
            print(f"\t  - [continue] {folder}: remote {input_file} write failed ({type(e).__name__}: {e}); not continued this poll", typeMsg='w')
            return False
        return True

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

    def _slurm_says_dead(self, folder, target, entry, slurm_state):
        '''
        True when slurm's own answer (`slurm_state`, from sacct) forbids the rescue.

        A terminal state means the work unit is no longer in flight, whether it succeeded (CGYRO
        reached MAX_TIME and exited cleanly without writing out.cgyro.tag — the usual false positive
        behind a STALLED classification) or died (a rescue would just restart from the warm start the
        next BO iteration retries anyway). A not-yet-started state means whatever the probe saw in
        that folder predates this job, e.g. an interrupted run preserved for an in-place rescue, and
        cancelling the element would kill a pending rescued radius. None means sacct gave no signal:
        fall through to the rescue rather than block on an unsupported sacct setup.
        '''
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
        return False

    def _restarted_recently(self, target, slurm_state, running_s, threshold):
        '''
        True when slurm (re)started the element less than `threshold` seconds ago. A preempted element
        is requeued and later restarts from bin.cgyro.restart, but out.cgyro.timing keeps its
        pre-preemption mtime until the restarted CGYRO appends to it: the probe's stall clock then
        counts the preemption and the queue wait, not a hang. The stall clock restarts at the (re)start.
        '''
        if running_s is not None and running_s < threshold:
            print(
                f"\t    * slurm reports {target} RUNNING for only {running_s}s (<{threshold}s): (re)started after "
                f"the files were last written (requeue/preemption); stall clock restarts, no rescue",
                typeMsg='i',
            )
            return True
        if slurm_state is not None:
            running = f" for {running_s}s" if running_s is not None else ""
            print(f"\t    * slurm reports {target} is {slurm_state}{running}; proceeding with rescue", typeMsg='i')
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

    def _resubmit(self, folder, entry, attempt, rescuing_child, stalled_on_node=False):
        # Bad-node exclusion: the node of THIS array element, from the squeue rows of the last poll
        # (the whole array's node list would exclude healthy nodes too), and only when sacct confirmed
        # the element has been running there for longer than the kill threshold, i.e. the hang happened
        # on that node. Child rescues have no tracked per-jobid node, so slurm places them freely.
        bad_node = None
        if stalled_on_node and not rescuing_child:
            bad_node = self.job.node_of(self.array_index_by_folder.get(folder))

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
        entry.last_action_at = _utc_now()
        self.metadata_dirty = True
        print(f"\t    * rescued {folder}: new jobid={new_jobid} (attempt {attempt}/{self.cap}, exclude={bad_node})", typeMsg='i')


def _cgyro_handle_stalled_tasks(sim, rows, session=None):
    '''Stall-rescue entry point for one poll's rows (see StallRescuer). No-op unless the user opted in.'''
    settings = getattr(sim, "auto_resubmit_settings", None) or {}
    if not (settings.get("enabled", False) or settings.get("max_continuations_per_rho", 0)):
        return
    if getattr(sim, "simulation_job", None) is None:
        return
    StallRescuer(sim, settings, session=session).rescue(rows)


def _restart_output(token):
    '''The output index a run resumes from (first token of out.cgyro.tag), or None unless a positive integer.'''
    token = str(token or "").strip()
    return int(token) if token.isdigit() and int(token) > 0 else None


def _trim_time_key(text, spec, done, total=None):
    '''
    (text, log note) with spec["time_key"] rewritten to the time left, total - done (at least 1), then
    spec["after_trim"] applied (CGYRO: RESTART_STEP). total defaults to the time_key value in `text`, the
    run length of a launch that started at t=0; unchanged text and "" when the key is absent.
    '''
    time_key = spec.get("time_key")
    m = re.search(rf"^({time_key}\s*=\s*)(\S+)", text, flags=re.M) if time_key else None
    if m is None:
        return text, ""
    current = float(m.group(2))
    remaining = max((current if total is None else total) - done, 1.0)
    text = text[:m.start(2)] + f"{remaining:.5E}" + text[m.end(2):]
    note = f", {time_key} {current:g} -> {remaining:g} remaining"
    after_trim = spec.get("after_trim")
    if after_trim is not None:
        text, extra = after_trim(text, remaining)
        note += extra
    return text, note


def _continuations_in_queue(sim, rels):
    '''
    {rel: jobid} of the radii among `rels` whose in-place continuation (StallRescuer) of an earlier
    submission is still queued or running, from the ledger that submission left on `sim`. squeue giving
    no verdict counts as still there: a second CGYRO in the same folder is the outcome to avoid.
    '''
    ledger = getattr(sim, "_resubmit_ledger", None) or {}
    entries = {rel: LedgerEntry.from_json(ledger.get(rel)) for rel in rels}
    latest = {rel: e.child_jobids[-1] for rel, e in entries.items() if e.continuation_jobids}
    if not latest:
        return {}
    from mitim_tools.simulation_tools.SIMtools import jobids_in_queue
    job = sim.simulation_job
    try:
        job.connect()
        states, _ = jobids_in_queue(job, list(latest.values()))
        job.close()
    except Exception as e:
        print(f"\t- [rescue] squeue of the in-place continuations failed ({type(e).__name__}: {e}); treating them as alive", typeMsg="w")
        return latest
    if states is None:
        return latest
    return {rel: jid for rel, jid in latest.items() if jid in states}


def _utc_now():
    '''Naive UTC plus an explicit "Z", the format stored in cgyro_submission.json.'''
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def _seconds_since(stamp):
    try:
        then = datetime.datetime.fromisoformat(stamp.rstrip("Z")).replace(tzinfo=datetime.timezone.utc)
    except (AttributeError, ValueError):
        return 0.0
    return (datetime.datetime.now(datetime.timezone.utc) - then).total_seconds()


def rescue_interrupted_runs(sim, kwargs_run, folders, folders_red, input_file):
    '''
    Continue, in place, radial runs that a previous execution left unfinished in
    the scratch folder (driver killed, allocation expired). Enabled by
    run(rescue_interrupted=True) for codes declaring `run_specifications["rescue_spec"]`:
        {"required": [files that must exist, e.g. restart blob + tag],
         "progress_file": file holding the time the code will resume from,
         "progress_line": its 1-based line number (None = last line); first token is read,
         "time_key": input-file key holding the run length (e.g. "MAX_TIME"),
         "checksum_ignore": input-file key prefixes excluded from the identity md5
            (optional, defaults to [time_key]; list every key `after_trim` rewrites,
            otherwise the next rescue of the same folder fails the identity check),
         "after_trim": optional callable (text, remaining) -> (text, log_note) applied
            to the staged input after the time_key rewrite, for keys whose value must
            follow the trimmed run length (CGYRO: RESTART_STEP),
         "report_files": files whose sizes are logged for forensics (optional),
         "keep_staged": staged files a continued radius still receives (optional; CGYRO: the
            cost-guard files, which a launch made before the guard existed left no copy of)}
    A rho sub-folder is rescued only if the required files are there and its
    input file is byte-identical (md5) to the one just generated, so a changed
    namelist, preset or gradient never continues a stale run. Rescued folders
    are kept across the scratch wipe (mitim_job.preserve_subfolders) and receive
    no staged restart file, so the code's own continuation logic takes over
    (CGYRO: restart + out.cgyro.tag -> restart_flag=1, t continues). Codes take
    `time_key` as a number of steps to ADD from the restart point (CGYRO:
    n_time = MAX_TIME/DELTA_T), so for rescued radii that key is rewritten in the
    staged input to the remaining time (original minus the resume time) and
    excluded from the identity md5. The resume time must be the one the code
    restarts from (CGYRO: t_current in out.cgyro.tag, second line), NOT the last
    printed time: the restart blob predates the last output by up to one restart
    interval, and counting from the wrong point over- or under-shoots the target.
    Everything else is staged and run as usual.

    A sub-folder whose run already ENDED there (the code's completion marker, e.g. CGYRO's
    EXIT line or mitim_budget.tag) passes the same identity check but is not run again: in
    bash mode results come back only when the whole evaluation ends, so a driver killed
    after some radii finished leaves them finished in scratch only. Such folders are kept
    across the wipe like the rescued ones and returned, so `_run` launches the others and
    retrieves them all.
    Returns the scratch-relative folders that ended.
    '''
    spec = sim.run_specifications.get("rescue_spec")
    sim.simulation_job.preserve_subfolders = []
    sim._finished_in_scratch = []
    if not (kwargs_run.get("rescue_interrupted", False) and spec):
        return []
    if getattr(sim.simulation_job, "run_in_place", False):
        print("\t- rescue_interrupted requested but the run is in-place (no scratch folder): nothing to rescue", typeMsg="i")
        return []

    time_key = spec.get("time_key")
    ignore_prefixes = list(spec.get("checksum_ignore") or ([time_key] if time_key else []))
    completion = CompletionSpec.from_run_specifications(sim.run_specifications)
    found = sim.simulation_job.probe_interrupted_runs(
        folders_red, spec.get("required", []), input_file,
        progress_file=spec.get("progress_file"), progress_line=spec.get("progress_line"),
        checksum_ignore_prefix=ignore_prefixes, report_files=spec.get("report_files"),
        completion=None if completion is None else (completion.marker_file, completion.marker_text, completion.alt_file),
    )
    if not found:
        return []
    alive = _continuations_in_queue(sim, list(found))
    if alive:
        raise RuntimeError(
            "[MITIM] radii still being continued in place by an earlier submission of this evaluation: "
            + ", ".join(f"{rel} (job {jid})" for rel, jid in alive.items())
            + ". Re-attach to it (check_existing_runs: true) or scancel those jobs before running it again"
        )

    import hashlib
    rescued, finished = [], []
    for folder_sim_this, rel in zip(folders, folders_red):
        if rel not in found:
            continue
        md5_remote, progress, report, ended = found[rel]
        text = (folder_sim_this / input_file).read_text()
        kept = "".join(l for l in text.splitlines(keepends=True) if not any(l.startswith(p) for p in ignore_prefixes))
        if hashlib.md5(kept.encode()).hexdigest() != md5_remote:
            print(f"\t- [rescue] {rel}: {'finished' if ended else 'interrupted'} run found but its {input_file} differs from the new one; discarding it", typeMsg="w")
            continue
        if ended:
            # Staged restarts would overwrite the finished run's own files on extraction
            for f in folder_sim_this.iterdir():
                if f.name != input_file:
                    f.unlink()
            finished.append(rel)
            print(f"\t- [rescue] {rel}: run already finished in scratch (t={progress}); not relaunched, collected with the others [{report}]", typeMsg="i")
            continue
        # Without a resume time the trim below cannot happen and the radius would run
        # the full time_key again on top of the restart point, so it is not rescued
        try:
            done = float(progress)
        except (TypeError, ValueError):
            print(f"\t- [rescue] {rel}: interrupted run found but no resume time could be read ({progress!r}); discarding it", typeMsg="w")
            continue
        # Keep only the input file locally (plus what the code declares safe to send again):
        # staged restarts would overwrite the orphan's own (more advanced) restart on extraction
        keep_staged = {input_file, *spec.get("keep_staged", [])}
        for f in folder_sim_this.iterdir():
            if f.name not in keep_staged:
                f.unlink()
        # Trim the run length to what is left (the code counts steps from the restart point)
        text, remaining_msg = _trim_time_key(text, spec, done)
        if remaining_msg:
            (folder_sim_this / input_file).write_text(text)
        rescued.append(rel)
        print(f"\t- [rescue] {rel}: continuing interrupted run in place (resuming from t={progress}{remaining_msg}) [{report}]", typeMsg="i")

    sim.simulation_job.preserve_subfolders = rescued + finished
    sim._finished_in_scratch = finished
    return finished
