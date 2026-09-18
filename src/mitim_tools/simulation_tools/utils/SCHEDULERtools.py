"""
In-allocation step scheduler for bash-mode runs (driver inside a SLURM allocation).

Replaces the bash `for folder ... & / jobs -rp` loop of SIMtools when a run wants to
react to calls finishing at different times: each radial body runs as its own
subprocess on its node slot; when a call finishes while others still run, an optional
hook may launch an "extra" call on the freed slot (load_balance strategy
"extra_points"). Extras are told to stop (a `mitim_stop` file in their folder) as
soon as the last main call ends, and the ones that left `accepted_marker` are reported
as accepted so the caller can retrieve them.
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
        accepted_marker="mitim_budget.tag",
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
        self.accepted_marker = accepted_marker
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
        call_index = 0
        t_start = time.time()
        try:
            while pending or running:
                while pending and len(running) + len(extras) < self.concurrency:
                    rel, body = pending.pop(0)
                    call_index += 1
                    running[rel] = (self._launch(cwd, prelude, log, rel, body, call_index), call_index)
                    print(f"\t- [scheduler] launched {rel} (call {call_index})")
                time.sleep(self.poll_seconds)
                for rel in [r for r, (p, _) in running.items() if p.poll() is not None]:
                    proc, k = running.pop(rel)
                    print(f"\t- [scheduler] {rel} ended (rc={proc.returncode}) after {(time.time()-t_start)/60:.0f} min")
                    if pending or self.on_call_finished is None:
                        continue
                    launched = self._maybe_extra(cwd, prelude, log, rel, k, running)
                    if launched is not None:
                        extras[launched[0]] = (launched[1], k)
                for rel in [r for r, (p, _) in extras.items() if p.poll() is not None]:
                    proc, _ = extras.pop(rel)
                    print(f"\t- [scheduler] extra {rel} ended on its own (rc={proc.returncode})")
                    extras[rel] = (proc, None)   # keep for the final classification
            self._stop_extras(cwd, extras)
        finally:
            log.close()
        accepted = [r for r in extras if (cwd / r / self.accepted_marker).exists()]
        discarded = [r for r in extras if r not in accepted]
        if extras:
            print(f"\t- [scheduler] extras accepted: {accepted or 'none'}; discarded: {discarded or 'none'}")
        return {"accepted": accepted, "discarded": discarded}

    def _maybe_extra(self, cwd, prelude, log, rel, call_index, running):
        if not running:
            return None
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
        print(f"\t- [scheduler] launched extra {rel_extra} on the slot of {rel} (idle window ~{(max(remaining)/60) if remaining else float('nan'):.0f} min)")
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
