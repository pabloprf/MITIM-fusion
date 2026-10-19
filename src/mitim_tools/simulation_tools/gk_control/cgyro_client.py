"""
GKRunControl, the run-control half of the CGYRO class: cost guard and load_balance captured around
run(), the watchdog template, the per-poll callback, and the hooks of the extra-point schedulers (bash
and array modes). CGYRO inherits it first, so these override the generic mitim_simulation ones.
"""
import shutil
import tempfile
from pathlib import Path
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.shell import _WATCHDOG_BASH
from mitim_tools.simulation_tools.gk_control.watchdog import Watchdog
from mitim_tools.simulation_tools.gk_control.cost_guard import CostGuard
from mitim_tools.simulation_tools.gk_control.status import seconds_remaining, seconds_to_accept
from mitim_tools.simulation_tools.gk_control.extra_points import ArrayExtraPoints
from mitim_tools.simulation_tools.gk_control.submission import cgyro_per_task_callback
from mitim_tools.misc_tools.LOGtools import printMsg as print


class GKRunControl:

    # Per-task inspection for `check(custom_checker=...)`. Picked up by
    # transport_cgyro.py via `getattr(gk_object, '_custom_check_callback', None)`
    # so the generic gyrokinetic_model evaluator stays code-agnostic (GX etc.
    # simply get None and skip). The wrapper runs the detector (prints + returns
    # structured rows) and then dispatches to the auto-resubmit orchestrator,
    # which is a no-op unless the user has opted in via `auto_resubmit_settings`.
    _custom_check_callback = staticmethod(cgyro_per_task_callback)

    # Thin wrapper: capture preprocess_options / load_balance and delegate to the generic run()
    def run(self, *args, preprocess_options=None, load_balance=None, cost_guard=None, **kwargs):
        '''
        load_balance: {'strategy': None | 'wall_budget' | 'extra_points', 'minutes_per_call': float, 'min_time': float}
        (namelist transport.options.cgyro.run.load_balance); see Watchdog and _extra_point_hooks.
        cost_guard: namelist transport.options.cgyro.run.cost_guard plus 'targets_GB'; see CostGuard.
        '''
        self._preprocess_options = preprocess_options
        self._load_balance = load_balance
        self._cost_guard = CostGuard(cost_guard)
        # extra cases of a timed-out radius can wait for the next submission only if that one continues
        # it in place and reads this submission's ledger back (ArrayExtraPoints hand over)
        self._rescue_interrupted = bool(kwargs.get("rescue_interrupted", False))
        # code_call reads allocation['omp_threads_cpu'] from here (SIMtools' JobScript passes only n)
        self._allocation = kwargs.get("allocation")
        guard_staging = tempfile.TemporaryDirectory() if self._cost_guard.enabled else None
        try:
            if guard_staging is not None:
                kwargs["additional_files_to_send"] = self._cost_guard.stage(guard_staging.name, kwargs.get("additional_files_to_send"))
            return super().run(*args, **kwargs)
        finally:
            if guard_staging is not None:
                guard_staging.cleanup()
            self._preprocess_options = None
            self._load_balance = None
            self._cost_guard = None
            self._extra_point_n = None
            self._allocation = None
            self._rescue_interrupted = False

    # The bash itself lives in templates/gk_control/cgyro_watchdog.sh; kept as an attribute so a caller can supply
    # its own template through `self`
    _WALL_BUDGET_WATCHDOG = _WATCHDOG_BASH

    def _wall_budget_wrap(self, cgyro_cmd, rho_dir, mode=None):
        '''The launch, wrapped in its Watchdog (see that class for what each mode does).'''
        return Watchdog.from_load_balance(
            rho_dir, getattr(self, "_load_balance", None), mode=mode, template=self._WALL_BUDGET_WATCHDOG).wrap(cgyro_cmd)

    # ------------------------------------------------------------------
    # load_balance strategy 'extra_points': hooks for the in-allocation scheduler
    # (SCHEDULERtools, bash mode) and the plan for ArrayExtraPoints (slurm_array). The
    # transport layer registers `extra_point_builder`, a callable
    # (rho, finished_run_dir, local_dir) -> path of a ready input.cgyro for a perturbed
    # case at that radius (or None); everything else is generic here.
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

    def _extra_point_array_plan(self, script, settings):
        '''
        slurm_array counterpart of _extra_point_hooks: the launch body of every radius's extra case,
        built now, while the allocation and load_balance of this submission are on the object (both
        are gone once run() returns, and a re-attaching driver never had them). See ArrayExtraPoints.
        '''
        lb = getattr(self, "_load_balance", None) or {}
        if lb.get("strategy") != "extra_points":
            return None
        if settings.run_type is not SIMtools.RunType.SUBMIT:
            # TODO (Pablo, 2026-10-13, deferred): cover run_type 'normal' (the template default) by sending such an
            # evaluation through the submit / poll / fetch path; today its driver blocks in `sbatch --wait` and never polls
            print(f"\t- load_balance 'extra_points' with a slurm array needs run_type 'submit' (the driver polls the elements); with '{settings.run_type.value}' array elements just release their nodes", typeMsg="w")
            return None
        if self.extra_point_builder is None:
            print("\t- load_balance 'extra_points' requested but no extra_point_builder is registered (standalone CGYRO run?); array elements just release their nodes", typeMsg="w")
            return None
        commands = {}
        for folder in script.folders:
            rel_extra = ArrayExtraPoints.extra_folder(folder)
            commands[folder] = script._call(rel_extra, watchdog="stop", additional_command=script._redirect(rel_extra))
        print("\t- load_balance 'extra_points': a radius that finishes early gets one extra case as its own job, while the slowest radius leaves time for it", typeMsg="i")
        ep = lb.get("extra_points") or {}
        return {"commands": commands, "min_time": float(lb.get("min_time", 0.0)), "stop_grace_seconds": 1800,
                "nice": ep.get("nice", 100), "yield_to_pending_mains": bool(ep.get("yield_to_pending_mains", True)),
                "rescues_first": bool(ep.get("rescues_first", True)),
                "hand_over": bool(getattr(self, "_rescue_interrupted", False) and getattr(self, "_reattach_enabled", False))}

    def _extra_points_handed_over(self):
        '''
        Extra cases an earlier submission of this evaluation handed over (ArrayExtraPoints.take_over), as
        the extra ledger this submission starts with. Called before the job is sent: the radii continued in
        place are the rescued ones still listed in preserve_subfolders, and the extras' folders join that
        list so a scratch wipe going in keeps them.
        '''
        job = self.simulation_job
        preserve = list(getattr(job, "preserve_subfolders", None) or [])
        continued = [rel for rel in preserve if rel not in (getattr(self, "_finished_in_scratch", None) or [])]
        taken = ArrayExtraPoints.take_over(getattr(self, "_extra_ledger", None), continued)
        if taken:
            job.preserve_subfolders = preserve + [e["folder"] for e in taken.values() if e["folder"] not in preserve]
        return taken

    def fetch(self):
        if getattr(self, "simulation_job", None) is not None and self.simulation_job.launchSlurm:
            ArrayExtraPoints(self).collect()
        super().fetch()

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
            t0 = float((d / markers.T0).read_text().strip() or 0.0)
            max_time = float(next(l.split("=")[1] for l in (d / "input.cgyro").read_text().splitlines() if l.strip().startswith("MAX_TIME")))
        except (OSError, StopIteration, ValueError):
            return None
        return seconds_remaining(cost, t, t0, max_time)

    def _estimate_to_accept(self, rel):
        lb = getattr(self, "_load_balance", None) or {}
        return seconds_to_accept(self._cost_per_acs(rel), lb.get("min_time", 0.0))

    def _idle_slot_sources(self, main_rels):
        '''
        Scheduler hook: the radii of this evaluation that an earlier driver job already finished,
        so a relaunch that runs only the unfinished ones can still give its idle slots extras.
        Each is made to look like a radius that just finished in scratch: its stored `<file>_<rho>`
        outputs are symlinked into the scratch folder under their plain names (skipped when the
        folder is still there), which is all _launch_extra_point, _estimate_to_accept and the
        builder read. Radii that finished in scratch before this run (SIMtools._rescue_interrupted_runs
        keeps them there and does not relaunch them) are sources as they are. Radii whose extra is
        already done locally are left out.
        '''
        spec = SIMtools.CompletionSpec.from_run_specifications(self.run_specifications)
        extra_done = SIMtools.CompletionSpec.coerce(spec, alt_file=markers.BUDGET_TAG)
        main = set(main_rels)
        sources = []
        for sub in sorted({rel.split("/")[0] for rel in main_rels}):
            local = Path(self.FolderGACODE) / sub
            for info in sorted(local.glob(f"{spec.marker_file}_*")):
                rho = float(info.name.rsplit("_", 1)[-1])
                rel = f"{sub}/{SIMtools.rho_folder(rho)}"
                if rel in main or not spec.finished(local, rho)[0]:
                    continue
                if extra_done.finished(Path(self.FolderGACODE) / markers.EXTRA_DIR / SIMtools.rho_folder(rho))[0]:
                    continue
                self._stage_finished_radius(local, rho, self._scratch(rel))
                sources.append(rel)
        for rel in getattr(self, "_finished_in_scratch", []):
            rho = float(rel.rsplit("rho_", 1)[-1])
            if rel in main or rel in sources or extra_done.finished(Path(self.FolderGACODE) / markers.EXTRA_DIR / SIMtools.rho_folder(rho))[0]:
                continue
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
        rel_extra = f"{markers.EXTRA_DIR}/{SIMtools.rho_folder(rho)}"
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
