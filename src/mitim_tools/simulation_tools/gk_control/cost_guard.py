"""
Cost guard of a PORTALS-CGYRO radius (namelist transport.options.cgyro.run.cost_guard): stages the
judge (templates/gk_control/cgyro_guard.py) and its thresholds, recognises its stops, floors their std.
"""
import json
from pathlib import Path
import numpy as np
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.shell import TEMPLATES
from mitim_tools.misc_tools.LOGtools import printMsg as print


class CostGuard:
    '''
    Stops a main radius that is burning wall time far from its targets, the way a user watching the
    run would (namelist transport.options.cgyro.run.cost_guard; the knobs are documented there).
    The decision is taken on the compute node by templates/gk_control/cgyro_guard.py, which the Watchdog runs
    every couple of minutes: this class only stages, per radius, that script and the mitim_guard.json
    it reads (thresholds + the turbulent heat-flux targets of the radius, GB units). A stop leaves
    mitim_budget.tag starting with GUARD, so the radius counts as finished like any watchdog stop.
    The same tag (with reason=cash_karp_abort) accepts a CGYRO that ended on the Cash-Karp abort
    while over-driven, and the fluxes of every guard-stopped radius get a wide std (floored_std).

    targets_GB: {rho: {"QeGB": float, "QiGB": float}}, total target minus neoclassical, added by the
    transport layer (transport_cgyro._gk_cost_guard). Without targets there is nothing to judge.
    '''

    DEFAULTS = {"seconds_per_acs": {"gpu": 100.0, "cpu": 1000.0}, "flux_ratio": 5.0, "min_time": 250.0,
                "waive_min_time_hours": 12.0, "floor_time": 150.0, "window": 20.0, "rising_tolerance": 0.1}
    FILE, SCRIPT = markers.GUARD_JSON, markers.GUARD_SCRIPT
    TAG_TOKEN = markers.GUARD_TOKEN

    def __init__(self, options=None):
        options = options or {}
        self.enabled = bool(options.get("enabled", False))
        options = {**options, "seconds_per_acs": self.machine_cost(options.get("seconds_per_acs", self.DEFAULTS["seconds_per_acs"]))}
        self.thresholds = {k: float(options.get(k, v)) for k, v in self.DEFAULTS.items()}
        # Cash-Karp abort accepted as a guard stop (cgyro_guard.abort_verdict); null ratio = flux_ratio
        self.thresholds["accept_abort"] = bool(options.get("accept_abort", True))
        self.thresholds["abort_flux_ratio"] = float(options.get("abort_flux_ratio") or self.thresholds["flux_ratio"])
        self.std_floor = float(options.get("std_floor") or 0.0)
        self.targets_GB = options.get("targets_GB") or {}
        if self.enabled and not self.targets_GB:
            print("\t- cost_guard is enabled but no flux targets were given (standalone CGYRO run?); the guard stays off", typeMsg="w")
            self.enabled = False

    def stage(self, folder, additional_files_to_send=None):
        '''
        Writes mitim_guard_<rho>.json under `folder` and returns `additional_files_to_send` with the
        guard files added to each radius that has targets.
        '''
        files = {rho: list(entries or []) for rho, entries in (additional_files_to_send or {}).items()}
        for rho, targets in self.targets_GB.items():
            guard_file = Path(folder) / f"mitim_guard_{rho:.4f}.json"
            guard_file.write_text(json.dumps({**self.thresholds, "targets_GB": targets}, indent=2))
            files.setdefault(rho, []).extend([(guard_file, self.FILE), (TEMPLATES / "cgyro_guard.py", self.SCRIPT)])
        return files

    @staticmethod
    def machine_cost(seconds_per_acs):
        '''
        seconds_per_acs as one number, or per machine class {"gpu": .., "cpu": ..}: the class of the
        machine CGYRO runs on (config_user.json, gpus_per_node 0 -> "cpu") picks the number.
        '''
        if not isinstance(seconds_per_acs, dict):
            return seconds_per_acs
        from mitim_tools.misc_tools import CONFIGread
        machine_class = "gpu" if int(CONFIGread.machineSettings(code="cgyro").get("gpus_per_node") or 0) > 0 else "cpu"
        return seconds_per_acs[machine_class]

    @classmethod
    def stopped(cls, tag_text):
        '''True when a mitim_budget.tag was left by the cost guard.'''
        return bool(tag_text) and tag_text.split()[:1] == [cls.TAG_TOKEN]

    @staticmethod
    def floored_std(mean, std, std_floor):
        '''
        std of a guard-stopped radius: at least std_floor x |mean|, element-wise. Its trace is a short
        transient, so the standard error of the mean over it would tell the surrogates the magnitude
        is known; the floor leaves them only the direction.
        '''
        return np.maximum(std, std_floor * np.abs(mean))
