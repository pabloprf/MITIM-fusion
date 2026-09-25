"""
In-allocation step scheduler for bash-mode runs (driver inside a SLURM allocation).

Replaces the bash `for folder ... & / jobs -rp` loop of SIMtools when a run wants to
react to calls finishing at different times: each radial body runs as its own
subprocess on its node slot; when a call finishes while others still run, an optional
hook may launch an "extra" call on the freed slot (load_balance strategy
"extra_points"). Slots that no main call ever takes (a rescue relaunching only the radii an
earlier driver job left unfinished) are offered extras too, built from the radii that did finish
(idle_slot_sources). Extras are told to stop (a `mitim_stop` file in their folder) as
soon as the last main call ends, and the ones that finished (`completion_marker`, e.g. CGYRO's
EXIT line) or were stopped past min_time (`accepted_marker`) are reported as accepted so the
caller can retrieve them.
"""
import os
import time
import signal
import subprocess
from pathlib import Path
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
        accepted_marker="mitim_budget.tag",
        completion_marker=None,      # (file, substring) or SIMtools.CompletionSpec: an extra that ran to its end, e.g. ("out.cgyro.info", "EXIT")
        stop_file="mitim_stop",
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
        from mitim_tools.simulation_tools.SIMtools import CompletionSpec
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
