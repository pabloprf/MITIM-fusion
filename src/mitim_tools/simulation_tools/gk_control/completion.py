"""
"Did this radius run to completion?": CGYRO's EXIT line, or the tag a watchdog leaves when it stops
a run on purpose (markers.BUDGET_TAG).
"""
from dataclasses import dataclass
from pathlib import Path


def _simtools():
    # SIMtools imports this module; resolved at call time
    from mitim_tools.simulation_tools import SIMtools
    return SIMtools


@dataclass
class CompletionSpec:
    '''
    How a code says "this radius ran to completion": a substring that must appear in
    `marker_file`, or the presence of `alt_file` (e.g. the tag left when a watchdog
    stops a run on purpose). Names carry the per-radius suffix when a rho is given,
    and are plain when it is not (scratch-folder layout).
    '''
    marker_file: str = None
    marker_text: str = None
    alt_file: str = None

    @classmethod
    def from_run_specifications(cls, run_specifications):
        marker = (run_specifications or {}).get("completion_marker")
        return None if marker is None else cls(marker[0], marker[1], (run_specifications or {}).get("completion_alt_file"))

    @classmethod
    def coerce(cls, completion_marker, alt_file=None):
        '''Accept a CompletionSpec, a (file, substring) tuple or None.'''
        if isinstance(completion_marker, cls):
            return cls(completion_marker.marker_file, completion_marker.marker_text,
                       completion_marker.alt_file if alt_file is None else alt_file)
        if completion_marker is None:
            return cls(None, None, alt_file)
        return cls(completion_marker[0], completion_marker[1], alt_file)

    def _name(self, file, rho):
        return file if rho is None else f"{file}{_simtools().rho_suffix(rho)}"

    def finished(self, folder, rho=None):
        '''Returns (finished, marker_path).'''
        folder = Path(folder)
        mfile = folder / self._name(self.marker_file, rho) if self.marker_file is not None else folder
        finished = False
        if self.marker_file is not None:
            try:
                finished = self.marker_text in mfile.read_text(errors="ignore")
            except OSError:
                finished = False
        if not finished and self.alt_file is not None:
            finished = (folder / self._name(self.alt_file, rho)).exists()
        return finished, mfile

    def unfinished(self, plan):
        '''(call, marker_path) for every call of the plan that did not run to completion.'''
        WorkPlan = _simtools().WorkPlan
        if not isinstance(plan, WorkPlan):
            plan = WorkPlan.from_code_executor(plan)
        return [(call, mfile) for call, (ok, mfile) in ((c, self.finished(c.folder, c.rho)) for c in plan) if not ok]


def radius_finished(folder, rho, completion_marker, completion_alt_file=None):
    """
    Whether the run stored in `folder` as `<file>_<rho>` ran to completion.

    completion_marker: (file, substring); `<file>_<rho>` must CONTAIN the substring. For
    CGYRO that is ('out.cgyro.info', 'EXIT'), a line written only on an orderly finish
    (cgyro_final_kernel.F90) - a crash writes 'ERROR: (CGYRO)' and a signal writes nothing.
    completion_alt_file: file whose presence (`<file>_<rho>`) also counts as finished,
    e.g. the 'mitim_budget.tag' left when CGYRO's wall-budget watchdog stops a run on purpose.

    Returns (finished, marker_path).
    """
    return CompletionSpec.coerce(completion_marker, alt_file=completion_alt_file).finished(folder, rho)
