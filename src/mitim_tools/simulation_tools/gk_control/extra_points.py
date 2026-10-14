"""
load_balance strategy "extra_points": give the slot a radius frees when it finishes early to an extra
case at that radius (a perturbed input, warm-started from the finished run), in both submission modes.

InAllocationScheduler (bash mode, driver inside a SLURM allocation) replaces the bash `for folder ...
& / jobs -rp` loop of SIMtools when a run wants to react to calls finishing at different times: each
radial body runs as its own subprocess on its node slot; when a call finishes while others still run,
an optional hook may launch an "extra" call on the freed slot. Slots that no main call ever takes (a
rescue relaunching only the radii an earlier driver job left unfinished) are offered extras too, built
from the radii that did finish (idle_slot_sources). Extras are told to stop (a `mitim_stop` file in
their folder) as soon as the last main call ends, and the ones that finished (`completion_marker`, e.g.
CGYRO's EXIT line) or were stopped past min_time (`accepted_marker`) are reported as accepted so the
caller can retrieve them.

ArrayExtraPoints is the slurm-array counterpart, driven from the driver's poll.
"""
import os
import time
import signal
import subprocess
from pathlib import Path
from mitim_tools.misc_tools.FARMINGtools import SlurmState
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.status import (
    _SLURM_JOB_GONE_STATES, TaskState, RadiusStatus, CgyroProbe, seconds_remaining, seconds_to_accept)
from mitim_tools.simulation_tools.gk_control.rescue import LedgerEntry, _utc_now, _seconds_since
from mitim_tools.misc_tools.LOGtools import printMsg as print


class InAllocationScheduler:
    def __init__(
        self,
        bodies,                      # {rel_folder: bash body} for the main calls, launch order = dict order
        hosts,                       # allocation hostnames (MITIM_HOSTS); call k runs with MITIM_CALL=k
        concurrency,                 # max simultaneous calls
        on_call_finished=None,       # fn(rel_folder) -> (rel_extra, body) or None
        estimate_remaining=None,     # fn(rel_folder) -> seconds a running call still needs, or None
        estimate_to_accept=None,     # fn(rel_folder) -> seconds an extra started on this slot needs to become acceptable
        idle_slot_sources=None,      # fn([main rel_folder]) -> [rel_folder of calls that finished before this run], read once at start
        accepted_marker=markers.BUDGET_TAG,
        completion_marker=None,      # (file, substring) or SIMtools.CompletionSpec: an extra that ran to its end, e.g. ("out.cgyro.info", "EXIT")
        stop_file=markers.STOP_REQUEST,
        poll_seconds=30,
        stop_grace_seconds=1800,     # after the stop file, how long to wait for extras to wind down before killing
    ):
        self.bodies = dict(bodies)
        self.hosts = list(hosts)
        self.concurrency = max(1, int(concurrency))
        self.on_call_finished = on_call_finished
        self.estimate_remaining = estimate_remaining
        self.estimate_to_accept = estimate_to_accept
        self.idle_slot_sources = idle_slot_sources
        self.accepted_marker = accepted_marker
        self.completion_marker = completion_marker
        self.stop_file = stop_file
        self.poll_seconds = poll_seconds
        self.stop_grace_seconds = stop_grace_seconds

    # ------------------------------------------------------------------
    def _launch(self, cwd, prelude, log, rel, body, call_index):
        script = cwd / f"mitim_call_{rel.replace('/', '_')}.sh"
        hosts = ("MITIM_HOSTS=( " + " ".join(self.hosts) + " )\n") if self.hosts else ""
        script.write_text(f"#!/usr/bin/env bash\n{prelude}\ncd {cwd}\n{hosts}MITIM_CALL={call_index}\n{body}\n")
        script.chmod(0o755)
        # own process group so a stop can reach srun/mpirun and everything under them
        return subprocess.Popen(["bash", str(script)], cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # ------------------------------------------------------------------
    def run(self, cwd, prelude="", log_path=None):
        '''
        Blocks until every main call has ended and every extra has stopped.
        Returns {"accepted": [rel_extra, ...], "discarded": [rel_extra, ...]}.
        '''
        cwd = Path(cwd)
        log = open(log_path or (cwd / "mitim.out"), "a")
        pending = list(self.bodies.items())
        running, extras = {}, {}          # rel -> (proc, call_index)
        ended_extras = set()
        call_index = 0
        t_start = time.time()
        sources = self._idle_sources()
        try:
            while pending or running:
                while pending and len(running) + len(extras) < self.concurrency:
                    rel, body = pending.pop(0)
                    call_index += 1
                    running[rel] = (self._launch(cwd, prelude, log, rel, body, call_index), call_index)
                    print(f"\t- [scheduler] launched {rel} (call {call_index})")
                time.sleep(self.poll_seconds)
                if not pending and sources:
                    self._fill_idle_slots(cwd, prelude, log, sources, running, extras)
                for rel in [r for r, (p, _) in running.items() if p.poll() is not None]:
                    proc, k = running.pop(rel)
                    print(f"\t- [scheduler] {rel} ended (rc={proc.returncode}) after {(time.time()-t_start)/60:.0f} min")
                    if pending or self.on_call_finished is None:
                        continue
                    launched = self._maybe_extra(cwd, prelude, log, rel, k, running)
                    if launched is not None:
                        extras[launched[0]] = (launched[1], k)
                for rel in [r for r, (p, _) in extras.items() if p.poll() is not None and r not in ended_extras]:
                    ended_extras.add(rel)   # stays in extras for the final classification
                    print(f"\t- [scheduler] extra {rel} ended on its own (rc={extras[rel][0].returncode})")
        finally:
            # also on an exception or a KeyboardInterrupt: an extra left running holds the GPUs
            try:
                self._stop_extras(cwd, extras)
            finally:
                log.close()
        accepted = [r for r in extras if self._accepted(cwd / r)]
        discarded = [r for r in extras if r not in accepted]
        if extras:
            print(f"\t- [scheduler] extras accepted: {accepted or 'none'}; discarded: {discarded or 'none'}")
        return {"accepted": accepted, "discarded": discarded}

    def _accepted(self, folder):
        '''An extra is usable if it ran to its end or was stopped past min_time.'''
        from mitim_tools.simulation_tools.gk_control.completion import CompletionSpec
        return CompletionSpec.coerce(self.completion_marker, alt_file=self.accepted_marker).finished(folder)[0]

    def _idle_sources(self):
        '''Finished calls whose slot is free from the start, when an extra hook is set.'''
        if self.idle_slot_sources is None or self.on_call_finished is None:
            return []
        try:
            sources = list(self.idle_slot_sources(list(self.bodies)))
        except Exception as e:
            print(f"\t- [scheduler] idle-slot source hook failed ({type(e).__name__}: {e}); only slots freed during the run get extras", typeMsg="w")
            return []
        if sources:
            print(f"\t- [scheduler] {len(sources)} call(s) finished before this run ({', '.join(sources)}); their extras may use the slots no main call takes")
        return sources

    def _fill_idle_slots(self, cwd, prelude, log, sources, running, extras):
        '''
        Offer the slots no main call or live extra holds to extras of `sources` (consumed in order),
        under the same idle-window test as a freed slot. Call k runs on slot (k-1) % concurrency, the
        same mapping the launch body uses for its host. Waits until every running main call has a
        remaining-time estimate: at launch there is none and the test would pass blindly.
        '''
        occupied = {(k - 1) % self.concurrency for _, k in running.values()}
        occupied |= {(k - 1) % self.concurrency for p, k in extras.values() if p.poll() is None}
        for slot in (s for s in range(self.concurrency) if s not in occupied):
            if not sources:
                return
            if self.estimate_remaining and any(self.estimate_remaining(r) is None for r in running):
                return
            rel = sources.pop(0)
            launched = self._maybe_extra(cwd, prelude, log, rel, slot + 1, running, label=f"the idle slot {slot + 1}")
            if launched is not None:
                extras[launched[0]] = (launched[1], slot + 1)

    def _maybe_extra(self, cwd, prelude, log, rel, call_index, running, label=None):
        if not running:
            return None
        # The slot stays free until run() returns, which is when the LAST main call ends, so the
        # window is the longest remaining time over every still-running main call
        remaining = [self.estimate_remaining(r) for r in running] if self.estimate_remaining else [None]
        remaining = [x for x in remaining if x is not None]
        needed = self.estimate_to_accept(rel) if self.estimate_to_accept else None
        if remaining and needed is not None and max(remaining) < needed:
            print(f"\t- [scheduler] {rel} idle window ~{max(remaining)/60:.0f} min < {needed/60:.0f} min needed for an acceptable extra; leaving the node idle")
            return None
        try:
            extra = self.on_call_finished(rel)
        except Exception as e:
            print(f"\t- [scheduler] extra-point hook failed for {rel} ({type(e).__name__}: {e}); leaving the node idle", typeMsg="w")
            return None
        if extra is None:
            return None
        rel_extra, body = extra
        proc = self._launch(cwd, prelude, log, rel_extra, body, call_index)
        print(f"\t- [scheduler] launched extra {rel_extra} on {label or f'the slot of {rel}'} (idle window ~{(max(remaining)/60) if remaining else float('nan'):.0f} min)")
        return rel_extra, proc

    def _stop_extras(self, cwd, extras):
        live = {r: p for r, (p, _) in extras.items() if p.poll() is None}
        if not live:
            return
        for rel in live:
            (cwd / rel / self.stop_file).write_text("stop\n")
        print(f"\t- [scheduler] main calls done; asked {len(live)} extra(s) to stop (graceful if past min_time)")
        t0 = time.time()
        while live and time.time() - t0 < self.stop_grace_seconds:
            time.sleep(min(self.poll_seconds, 10))
            live = {r: p for r, p in live.items() if p.poll() is None}
        for rel, proc in live.items():
            print(f"\t- [scheduler] extra {rel} did not stop within {self.stop_grace_seconds/60:.0f} min; killing it", typeMsg="w")
            self._kill(proc)
            proc.wait()


class ArrayExtraPoints:
    '''
    load_balance strategy 'extra_points' when every radius is its own slurm element (slurm_array,
    run_type 'submit'): the array-mode counterpart of InAllocationScheduler. A finished
    element gives its GPUs back to the partition, so the extra case of that radius is submitted as a
    standalone single-radius job with the sbatch settings of one element (mitim_job.resubmit_single_task,
    the primitive of the stall rescue), from the driver's poll (`cgyro_per_task_callback`).

    Same rules as in bash mode:
        launch   one extra per finished radius, once every main radius has started and the slowest one
                 still needs longer than the extra needs to reach min_time (seconds_remaining against
                 seconds_to_accept); never while a main element is still waiting for a node.
        stop     when the last main radius ends: mitim_stop in the extra's folder (its STOP watchdog ends
                 it after a restart write past min_time, or discards it below), scancel after
                 stop_grace_seconds; an extra still pending in the queue is cancelled at once.
        accept   EXIT line or mitim_budget.tag; `collect` adds the accepted folders to the retrieval of
                 fetch(), which stores them under <FolderGACODE>/extra_cgyro/ like the bash-mode ones.

    The ledger {main folder: {folder, jobid, status, ...}} lives on the simulation object and in
    cgyro_submission.json, so a driver that re-attaches keeps polling, stopping and collecting the extras
    the previous one submitted. The launch bodies were built at submit time (kwargs_organize
    ["extra_points"]), with the allocation and the min_time of that submission.
    '''

    RUNNING, STOPPING, ACCEPTED, DISCARDED, SKIPPED = "RUNNING", "STOPPING", "ACCEPTED", "DISCARDED", "SKIPPED"
    IN_FLIGHT = (RUNNING, STOPPING)
    MAIN_OVER = (TaskState.FINISHED.value, TaskState.TIMED_OUT.value, TaskState.ERROR.value)

    def __init__(self, sim, session=None):
        self.sim = sim
        self.job = getattr(sim, "simulation_job", None)
        self.session = session   # live ssh session of the poll; None -> collect() opens its own
        plan = (getattr(sim, "kwargs_organize", None) or {}).get("extra_points") or {}
        self.commands = plan.get("commands") or {}
        self.min_time = float(plan.get("min_time", 0.0))
        self.stop_grace_s = float(plan.get("stop_grace_seconds", 1800))
        self.dirty = False

    @property
    def ledger(self):
        if getattr(self.sim, "_extra_ledger", None) is None:
            self.sim._extra_ledger = {}
        return self.sim._extra_ledger

    @property
    def exec_folder(self):
        return str(self.job.folderExecution)

    def _in_flight(self):
        return {rel: e for rel, e in self.ledger.items() if e.get("status") in self.IN_FLIGHT}

    @staticmethod
    def extra_folder(rel):
        return f"{markers.EXTRA_DIR}/{Path(rel).name}"

    # ------------------------------------------------------------------
    # Every poll
    # ------------------------------------------------------------------

    def poll(self, rows):
        if not (self.commands or self.ledger):
            return
        states = self._refresh()
        if self._mains_over(rows):
            self._stop(states)
        else:
            self._launch_due(rows)
        self._save()

    def _refresh(self):
        '''
        Print the extras in flight and classify the ones that left the queue. Returns {main folder:
        probe status} of those still in it; with no answer from slurm every one counts as still there.
        '''
        flying = self._in_flight()
        if not flying:
            return {}
        queued = SIMtools.jobids_in_queue(self.job, [e["jobid"] for e in flying.values()])[0]
        probed = self._probe(flying)

        print(f"\t- Extra case(s) of finished radii ({len(flying)} in flight):")
        states = {}
        for rel, entry in flying.items():
            status = probed.get(entry["folder"])
            if status is not None:
                print(status.describe()[0] + f" [extra job {entry['jobid']}]")
            if queued is None or entry["jobid"] in queued:
                states[rel] = status
            elif status is not None:
                self._classify(entry, status.exited, "left the queue")
        return states

    def _probe(self, entries):
        lines = CgyroProbe(self.job, session=self.session).run([e["folder"] for e in entries.values()]) or []
        return {st.folder: st for st in (RadiusStatus.from_probe_line(l) for l in lines) if st is not None}

    def _classify(self, entry, accepted, why):
        entry["status"] = self.ACCEPTED if accepted else self.DISCARDED
        self.dirty = True
        print(f"\t- [extra point] {entry['folder']} {why}: {'accepted' if accepted else 'discarded (it did not reach min_time)'}", typeMsg="i")

    def _mains_over(self, rows):
        '''
        True when no main radius is integrating any more. A radius handed to a rescue job follows that
        job (the array's own state says nothing about it) until the stall rescuer closes it.
        '''
        job_gone = SlurmState.from_token((getattr(self.job, "infoSLURM", None) or {}).get("STATE")) in _SLURM_JOB_GONE_STATES
        rescues = getattr(self.sim, "_resubmit_ledger", None) or {}
        for row in rows:
            if row["effective"] in self.MAIN_OVER:
                continue
            rescue = LedgerEntry.from_json(rescues.get(row["folder"]))
            if rescue.child_jobids:
                if rescue.status.startswith(("EXHAUSTED", "TERMINAL_NO_RESCUE")):
                    continue
                return False
            if not job_gone:
                return False
        return True

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def _launch_due(self, rows):
        if getattr(self.sim, "extra_point_builder", None) is None:
            return
        candidates = [r for r in rows if r["effective"] == TaskState.FINISHED.value
                      and r["folder"] in self.commands and r["folder"] not in self.ledger]
        running = [r for r in rows if r["effective"] not in self.MAIN_OVER]
        if not candidates or not running:
            return
        # A main element without an estimate has not started (or just did): it needs the nodes more
        # than an extra does, and without its remaining time the idle window would pass blindly
        remaining = [r.get("remaining_s") for r in running]
        if any(x is None for x in remaining):
            return
        window = max(remaining)
        limit = (getattr(self.job, "slurm_settings", None) or {}).get("array_limit")

        for row in candidates:
            rel = row["folder"]
            if limit is not None and len(running) + len(self._in_flight()) >= int(limit):
                return   # max_concurrent_calls also bounds mains + extras; asked again next poll
            needed = seconds_to_accept(row.get("cost_last"), self.min_time)
            if needed is not None and window < needed:
                self._skip(rel, f"idle window ~{window/60:.0f} min < {needed/60:.0f} min needed for an acceptable extra")
                continue
            self._launch(rel, window)

    def _skip(self, rel, why):
        self.ledger[rel] = {"folder": self.extra_folder(rel), "jobid": None, "status": self.SKIPPED, "reason": why}
        self.dirty = True
        print(f"\t- [extra point] {rel}: {why}; no extra case", typeMsg="i")

    def _launch(self, rel, window):
        rho = float(rel.rsplit("rho_", 1)[-1])
        rel_extra = self.extra_folder(rel)
        local_dir = Path(self.sim.FolderGACODE) / rel_extra
        spec = SIMtools.CompletionSpec.from_run_specifications(self.sim.run_specifications)
        if SIMtools.CompletionSpec.coerce(spec, alt_file=markers.BUDGET_TAG).finished(local_dir)[0]:
            return self._skip(rel, "its extra case is already stored locally")
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            input_cgyro = self.sim.extra_point_builder(rho, self._finished_outputs(rel), local_dir)
            if input_cgyro is None:
                return self._skip(rel, "the builder returned no case")
            jobid = self._stage_and_submit(rel, rel_extra, input_cgyro)
        except Exception as e:
            return self._skip(rel, f"extra-point launch failed ({type(e).__name__}: {e})")
        if not jobid:
            return self._skip(rel, "sbatch returned no jobid")
        self.ledger[rel] = {"folder": rel_extra, "jobid": jobid, "status": self.RUNNING, "submitted_at": _utc_now()}
        self.dirty = True
        print(f"\t- [extra point] {rel} finished: extra case {rel_extra} submitted as job {jobid} (idle window ~{window/60:.0f} min)", typeMsg="i")

    def _finished_outputs(self, rel):
        '''
        Folder holding the finished radius's outputs under their plain names, for the builder to read
        its fluxes: the scratch folder itself when the driver shares its filesystem, else a copy of the
        files a fetch brings (never the restart blob) pulled into the job's local folder.
        '''
        if self.job.ssh is None:
            return Path(self.exec_folder) / rel
        from mitim_tools.misc_tools.FARMINGtools import RetrievalSpec
        files = list(self.sim.kwargs_organize["filesToRetrieve"])
        # best_effort: a failed pull must not raise the interactive disk-full prompt inside the poll
        if not self.job.retrieve(check_if_files_received=False, best_effort=True, spec=RetrievalSpec(folders=[rel], selective={rel: files})):
            raise RuntimeError(f"could not pull the outputs of {rel}")
        return Path(self.job.folder_local) / rel

    def _stage_and_submit(self, rel, rel_extra, input_cgyro):
        '''The perturbed input.cgyro plus the finished run's restart blob as warm start (no out.cgyro.tag: t from 0).'''
        src, dst = f"{self.exec_folder}/{rel}", f"{self.exec_folder}/{rel_extra}"
        self.job.execute(f'mkdir -p "{dst}" && if [ -f "{src}/bin.cgyro.restart" ]; then cp "{src}/bin.cgyro.restart" "{dst}/"; fi', printYN=False)
        self.job.put_file(input_cgyro, f"{dst}/input.cgyro")
        return self.job.resubmit_single_task(self.commands[rel], f"_extra_{Path(rel_extra).name}")

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def _stop(self, states):
        '''`states`: {main folder: probe status} of the extras still in the queue.'''
        for rel, status in states.items():
            entry = self.ledger[rel]
            if status is not None and status.raw == TaskState.NOT_STARTED:
                # never started: a stop file would be removed by its watchdog at launch
                self._scancel(entry, "the main radii ended before it started")
                self._classify(entry, False, "cancelled")
                continue
            # dropped again every poll: a requeued extra clears the file when it restarts
            self.job.execute(f'echo stop > "{self.exec_folder}/{entry["folder"]}/{markers.STOP_REQUEST}"', printYN=False)
            if entry["status"] == self.RUNNING:
                entry["status"], entry["stop_requested_at"] = self.STOPPING, _utc_now()
                self.dirty = True
                print(f"\t- [extra point] main radii done; asked {entry['folder']} (job {entry['jobid']}) to stop (graceful if past min_time)", typeMsg="i")
            elif _seconds_since(entry.get("stop_requested_at")) > self.stop_grace_s:
                self._scancel(entry, f"it did not stop within {self.stop_grace_s/60:.0f} min")

    def _scancel(self, entry, why):
        print(f"\t- [extra point] {entry['folder']} (job {entry['jobid']}): {why}; cancelling it", typeMsg="w")
        self.job.execute(f"scancel {entry['jobid']}", printYN=False)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def collect(self):
        '''
        Before fetch(): settle every extra (one still in the queue is cancelled: the evaluation is
        being collected) and add the accepted ones to the retrieval, best-effort, with the file
        patterns of the main folders. _organize_results then moves them next to the main results.
        '''
        if not self.ledger or self.job is None:
            return
        flying = self._in_flight()
        if flying:
            with self.job.session() as self.session:
                queued = SIMtools.jobids_in_queue(self.job, [e["jobid"] for e in flying.values()])[0] or {}
                for entry in flying.values():
                    if entry["jobid"] in queued:
                        self._scancel(entry, "the evaluation is being collected")
                probed = self._probe(flying)
            for entry in flying.values():
                status = probed.get(entry["folder"])
                self._classify(entry, status is not None and status.exited, "ended")

        accepted = [e["folder"] for e in self.ledger.values() if e.get("status") == self.ACCEPTED]
        discarded = [e["folder"] for e in self.ledger.values() if e.get("status") == self.DISCARDED]
        selective = self.job.output_folders_selective
        patterns = next(iter(selective.values()), None) if selective else None
        for folder in accepted:
            if folder not in self.job.output_folders:
                self.job.output_folders.append(folder)
            if patterns is not None:
                selective[folder] = list(patterns)
        self.job.scheduler_result = {"accepted": accepted, "discarded": discarded}
        if accepted or discarded:
            print(f"\t- [extra point] extras accepted: {accepted or 'none'}; discarded: {discarded or 'none'}", typeMsg="i")
        self._save()

    def _save(self):
        if not self.dirty:
            return
        self.dirty = False
        try:
            self.sim._write_submission_metadata(getattr(self.sim, "_base_subfolder", None))
        except Exception as e:
            print(f"\t- [extra point] metadata write failed ({type(e).__name__}: {e}); ledger held in-memory only", typeMsg='w')
