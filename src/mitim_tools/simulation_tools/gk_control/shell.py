"""
The bash MITIM wraps around every CGYRO launch and poll lives in templates/ so it can be read and
linted as shell. `@{name}` is the placeholder delimiter, so the shell's own `$` survives verbatim.
"""
import string
from mitim_tools import __mitimroot__

TEMPLATES = __mitimroot__ / "templates"


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


class _ShellTemplate(string.Template):
    delimiter = "@"


def _shell_text(name):
    return (TEMPLATES / name).read_text()


_WATCHDOG_BASH = _shell_text("cgyro_watchdog.sh")
_PROBE_BASH = _shell_text("cgyro_probe.sh")
_REQUEUE_TRIM_BASH = _shell_text("cgyro_requeue_trim.sh")
