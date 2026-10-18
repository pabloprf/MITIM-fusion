"""
Names of the files and tokens the gyrokinetic run-control machinery reads or writes in a radius
folder (or next to a run). One place, so a name cannot drift between the writer and the readers.
The bash templates under templates/ carry the same literals (they cannot import this module);
tests/dev_tests/test_gk_control_markers.py checks that they match.
"""

# Stops and their outcome
BUDGET_TAG = "mitim_budget.tag"          # watchdog stopped the radius on purpose: it counts as finished
DISCARD_TAG = "mitim_discard.tag"        # an extra case stopped below min_time: thrown away
STOP_REQUEST = "mitim_stop"              # dropped by the scheduler or mitim_kill_cgyro: stop at the next restart write
GUARD_TOKEN = "GUARD"                    # first word of a BUDGET_TAG written by the cost guard

# Cost guard files, staged per radius
GUARD_JSON = "mitim_guard.json"
GUARD_SCRIPT = "mitim_guard.py"
GUARD_STATUS = "mitim_guard.status"

# Launch markers
RUN_STARTED = ".mitim_run_started"       # baseline for "did this launch write a restart?"
T0 = ".mitim_t0"                         # simulated time this launch starts from [a/cs]
T_END = ".mitim_t_end"                   # "<job> <t_end>": time this submission must reach (requeue trim)

# CGYRO's own signals
CGYRO_COMPLETION = ("out.cgyro.info", "EXIT")         # written only on an orderly finish
CGYRO_EXIT_PATTERN = "^EXIT: (CGYRO)"                 # the same line, as a grep pattern
CASH_KARP_ABORT = "ERROR: (CGYRO) Cash-Carp step exceeded"

# Submission state, next to the run
CGYRO_SUBMISSION = "cgyro_submission.json"
GX_SUBMISSION = "gx_submission.json"
RESTART_SOURCES = "restart_sources.json"
DRIVER_WALL_UNFINISHED = "MITIM_DRIVER_WALL_UNFINISHED.json"   # <PORTALS run>/Outputs: driver hit its SLURM wall mid-poll

# Extra cases (load_balance 'extra_points')
EXTRA_DIR = "extra_cgyro"
EXTRA_POINT_JSON = "mitim_extra_point.json"
EXTRA_HARVESTED = "mitim_harvested"
SCHEDULER_CALL = "mitim_call_{}.sh"      # in-allocation scheduler, one per call (rel folder with / -> _)

# Interrupted-run probe output token
RESCUE_TOKEN = "MITIM_RESCUE"
