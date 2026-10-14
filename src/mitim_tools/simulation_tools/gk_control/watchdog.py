"""
The supervisor around one radial CGYRO launch (templates/cgyro_watchdog.sh) and the shell that
runs right before and after it: launch markers, requeue trim, exit verdict.
"""
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.shell import _ShellTemplate, _WATCHDOG_BASH, _REQUEUE_TRIM_BASH


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

    def __init__(self, rho_dir, mode=None, min_time=0.0, minutes_per_call=None, template=None, guard=False):
        self.rho_dir = rho_dir
        self.mode = mode or self.MANUAL
        # cost guard (CostGuard): main radii only, an extra case has no target of its own
        self.guard = bool(guard) and self.mode != self.STOP
        self.min_time = float(min_time)
        self.template = _WATCHDOG_BASH if template is None else template
        if self.mode == self.BUDGET and minutes_per_call is None:
            raise ValueError("[MITIM] load_balance strategy 'wall_budget' needs 'minutes_per_call' (the wall minutes each radial call gets)")
        self.budget_s = int(float(minutes_per_call) * 60) if self.mode == self.BUDGET else 0

    @classmethod
    def from_load_balance(cls, rho_dir, load_balance, mode=None, template=None, guard=False):
        '''mode None picks BUDGET when the load_balance strategy asks for it, else MANUAL.'''
        lb = load_balance or {}
        if mode is None:
            mode = cls.BUDGET if lb.get("strategy") == "wall_budget" else cls.MANUAL
        return cls(rho_dir, mode=mode, min_time=lb.get("min_time", 0.0),
                   minutes_per_call=lb.get("minutes_per_call"), template=template, guard=guard)

    def wrap(self, cmd):
        return _ShellTemplate(self.template).substitute(
            rhodir=self.rho_dir,
            budget_s=self.budget_s,
            min_time=f"{self.min_time:g}",
            discard="1" if self.mode == self.STOP else "0",
            guard="1" if self.guard else "0",
            cgyro_cmd=cmd.rstrip("\n"),
        )


def launch_markers(run_dir):
    '''
    (marker_cmd, cleanup_cmd) around the launch in `run_dir`.

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

    After the markers, templates/cgyro_requeue_trim.sh trims MAX_TIME when SLURM requeues the same
    job (CGYRO would otherwise add the full MAX_TIME on top of the checkpoint it resumes from).
    '''
    restart_path = f"{run_dir}/bin.cgyro.restart"
    marker_path = f"{run_dir}/{markers.RUN_STARTED}"
    marker_cmd = (f'touch "{marker_path}"; _t0=$(sed -n 2p "{run_dir}/out.cgyro.tag" 2>/dev/null '
                  f'| awk \'{{print $1+0}}\'); echo "${{_t0:-0}}" > "{run_dir}/{markers.T0}"\n'
                  + _ShellTemplate(_REQUEUE_TRIM_BASH).substitute(run_dir=run_dir).rstrip("\n"))
    cleanup_cmd = (
        f'if [ -f "{restart_path}" ] && [ -f "{marker_path}" ] && '
        f'[ ! "{restart_path}" -nt "{marker_path}" ]; then '
        f'rm -f "{restart_path}"; fi; rm -f "{marker_path}"'
    )
    return marker_cmd, cleanup_cmd


def exit_verdict(run_dir):
    '''
    Turn a 0 exit status into 1 when CGYRO did not finish. gacode's `cgyro` script ends with an
    if-block that returns 0 whatever the executable returned, so a CGYRO that crashed (e.g. disk
    quota exceeded writing out.cgyro.prec) was recorded by SLURM as COMPLETED 0:0. CGYRO appends
    "EXIT: (CGYRO) ..." to out.cgyro.info only on a clean end.

    And the other way round: a run the watchdog stopped and accepted (mitim_budget.tag: wall budget,
    mitim_stop, cost guard) ends with the SIGTERM status 143, which SLURM showed as FAILED although
    the radius counts as finished; it is reported as 0. A discarded extra case (mitim_discard.tag)
    produced nothing usable and keeps its own status.
    '''
    info = f"{run_dir}/{markers.CGYRO_COMPLETION[0]}"
    budget, discard = f"{run_dir}/{markers.BUDGET_TAG}", f"{run_dir}/{markers.DISCARD_TAG}"
    return (f'if [ -f "{budget}" ]; then _mitim_rc=0; fi; '
            f'if [ "$_mitim_rc" = 0 ] && ! grep -qs "{markers.CGYRO_EXIT_PATTERN}" "{info}" && '
            f'[ ! -f "{budget}" ] && [ ! -f "{discard}" ]; then '
            f'echo "MITIM: cgyro returned 0 but out.cgyro.info has no EXIT line; reporting rc=1" >&2; _mitim_rc=1; fi')
