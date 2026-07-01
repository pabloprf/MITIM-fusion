"""
TRANSP debugging utilities.

A collection of tools for making a failed / opaque TRANSP run understandable. The
entry point is :func:`diagnose_transp_failure`, which turns a crashed run's log
into a single human-readable paragraph naming the actual cause (the signal and a
plain-language gloss, the simulation time / step, the physics event at the trap,
and the geometry / underflow breadcrumbs) instead of the bare "TRANSP stopped".

This is wired into ``TRANSPsingularity.interpretRun`` (so the exception MAESTRO
dies with is self-describing) but is also usable interactively on any tr.log via
:func:`diagnose_transp_logfile`.

Add further TRANSP debugging helpers here rather than scattering them through the
run wrappers.
"""

import re
from pathlib import Path
from mitim_tools.transp_tools.utils.TRANSPhelpers import CONTAINER_LAUNCH_ERRORS

# ---------------------------------------------------------------------------
# Log signatures
# ---------------------------------------------------------------------------

# InfiniBand / RDMA container-launch failure: the apptainer container is denied
# the mlx5 UD queue-pair ("Operation not permitted") and OpenMPI's mpirun then
# segfaults. Infrastructure/permission, not physics; node/config-dependent and
# usually succeeds on a retry elsewhere.
RDMA_LAUNCH_ERRORS = [
    "Failed to modify UD QP",
    "error initializing an OpenFabrics device",
]

# Other fatal TRANSP aborts (fortran-side), checked when no signal line is present.
# Kept in sync with the `hard_failure` set in TRANSPsingularity.interpretRun (that set
# decides status=-1; this one only labels the already-stopped run). The signal-adjacent
# entries ("Backtrace...", "Segmentation fault - invalid memory reference") are normally
# caught earlier by the `Program received signal` branch, but are listed for parity so a
# bare abort with no signal line still gets a clean "hard_abort" label, not "unclassified".
HARD_ABORT_SIGNATURES = [
    "TRANSP ABORTR SUBROUTINE CALLED",
    "Error termination",
    "%bad_exit:  generic f77 error exit call",
    "Backtrace for this error:",
    "Segmentation fault - invalid memory reference",
    "*** End of error message ***",
]

# GFortran prints "Program received signal <NAME>: <text>" on a hardware trap.
_SIGNAL_GLOSS = {
    "SIGFPE":  "floating-point exception — a divide-by-zero, overflow, or invalid "
               "operation (e.g. sqrt/log of a negative, or 0/0) produced a NaN or Inf",
    "SIGSEGV": "segmentation fault — an invalid memory access",
    "SIGABRT": "abort — an internal consistency check called abort()",
    "SIGKILL": "killed by the OS — frequently out-of-memory",
}

_RE_SIGNAL = re.compile(r"Program received signal (\w+)")
_RE_TA     = re.compile(r"TA=\s*([0-9.eE+-]+)")
_RE_NSTEP  = re.compile(r"NSTEP[= ]+\s*(\d+)")
_RE_CURV   = re.compile(r"curvature ratio.*?is:\s*([0-9.eE+-]+)")
_RE_GSERR  = re.compile(r"Avg\. GS error:\s*([0-9.eE+-]+)")
_RE_IPERR  = re.compile(r"Plasma Current:.*error:\s*([0-9.eE+-]+)%")
_RE_NODE   = re.compile(r"Local host:\s*(\S+)")
_RE_NODE2  = re.compile(r"(node\d+):rank")

# Lines that are pure boilerplate and carry no diagnostic value (skipped when
# picking a "last meaningful line" as fallback context).
_BOILERPLATE = (
    "get_rygrid", "check_save_state", "RESET TO ZERO", "plasma_hash",
    "MOMENTS CHECKSUM", "CPU TIME",
)

# How many lines before the trap to scan for the physics context / breadcrumbs.
_PRE_TRAP_WINDOW = 40


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _as_lines(log):
    """Accept a path, a single string, or a list of lines; return a list of lines."""
    if isinstance(log, Path):
        return log.expanduser().read_text(errors="replace").splitlines()
    if isinstance(log, str):
        # A str is either a filesystem path or the log text itself; only probe the
        # filesystem for a short, single-line string (a real path never has newlines).
        if "\n" not in log and len(log) < 4096:
            try:
                p = Path(log).expanduser()
                if p.exists():
                    return p.read_text(errors="replace").splitlines()
            except OSError:
                pass
        return log.splitlines()
    return list(log)


def _last_match(lines, regex):
    """Value of the last line matching `regex` (first capture group), or None."""
    val = None
    for ln in lines:
        m = regex.search(ln)
        if m:
            val = m.group(1)
    return val


def _signal_line_index(lines):
    for i, ln in enumerate(lines):
        if "Program received signal" in ln:
            return i
    return None


# ---------------------------------------------------------------------------
# Physics-context classification (what was TRANSP doing at the trap)
# ---------------------------------------------------------------------------

def _classify_context(pre_lines):
    """Human-readable description of what TRANSP was doing right before the trap."""
    text = "\n".join(pre_lines)
    if any(s in text for s in ("sawtooth_trigger", "SAWTOOTH EVENT", "Porcelli sawtooth crash")):
        ctx = "a sawtooth crash / reconnection event (Porcelli trigger)"
        if "INVALID INTERPOLATION" in text:
            ctx += ", with an invalid field interpolation flagged just before the trap"
        return ctx
    if any(s in text for s in ("MHD EQUILIBRIUM CALCULATED", "*** TEQ ***", "EQBDY_CHECK")):
        return "the MHD equilibrium / geometry update (TEQ)"
    if "NEUTRAL SOURCE" in text:
        return "the neutral-source / recycling calculation"
    # Fallback: last non-boilerplate, non-empty line.
    for ln in reversed(pre_lines):
        s = ln.strip()
        if s and not any(b in s for b in _BOILERPLATE):
            return f"an unclassified module (last log line: '{s}')"
    return "an unidentified TRANSP module"


def _breadcrumbs(pre_lines, context):
    """Numeric breadcrumbs from the last steps before the trap (curvature ratio,
    denormal-underflow resets, and — for an equilibrium trap — the convergence
    state, so an equilibrium FP trap is not misread as a convergence failure)."""
    parts = []
    curv = _last_match(pre_lines, _RE_CURV)
    if curv is not None:
        parts.append(
            f"boundary (separatrix) curvature ratio had climbed to {float(curv):.3g} "
            "(elevated vs typical healthy runs, ~0.06)"
        )
    n_reset = sum("RESET TO ZERO" in ln for ln in pre_lines)
    if n_reset:
        parts.append(f"{n_reset} denormal-underflow resets (%MFRCHK) immediately prior")

    if "equilibrium" in context:
        gs = _last_match(pre_lines, _RE_GSERR)
        ip = _last_match(pre_lines, _RE_IPERR)
        if gs is not None and ip is not None:
            parts.append(
                f"the equilibrium itself was still converging (GS error {float(gs):.2g}, "
                f"Ip error {float(ip):.2g}%), so this is a solver-internal FP trap rather "
                "than a convergence failure"
            )

    return ("Breadcrumbs before the trap: " + "; ".join(parts) + ".") if parts else ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def diagnose_transp_failure(log, logname=None):
    """Parse a (crashed) TRANSP run log into a human-readable one-paragraph diagnosis.

    Parameters
    ----------
    log : str | Path | list[str]
        A path to the tr.log, its full text, or a list of its lines.
    logname : str, optional
        A short name for the log file to append as a pointer (e.g. '89685P02tr.log').

    Returns
    -------
    dict with keys:
        category : one of 'container_rdma_launch', 'container_namespace_launch',
                   'physics_signal', 'hard_abort', 'unclassified'
        message  : the human-readable diagnosis (single paragraph)
        plus category-specific fields (signal, sim_time, nstep, context, node).
    """
    lines = _as_lines(log)
    text = "\n".join(lines)
    tag = f" See {logname}." if logname else ""

    # 1) Infrastructure: IB / RDMA container-launch failure ------------------
    if any(s in text for s in RDMA_LAUNCH_ERRORS):
        node = _last_match(lines, _RE_NODE) or _last_match(lines, _RE_NODE2)
        where = f" on {node}" if node else ""
        msg = (f"the MPI layer could not initialize the InfiniBand device (mlx5){where}: "
               "'Operation not permitted'. The apptainer container was denied the RDMA "
               "queue-pair and mpirun then segfaulted. This is an infrastructure / "
               "permission failure, NOT physics — node/config-dependent and usually "
               "succeeds on a retry elsewhere." + tag)
        return {"category": "container_rdma_launch", "message": msg, "node": node}

    # 2) Infrastructure: user-namespace container-launch failure -------------
    if any(s in text for s in CONTAINER_LAUNCH_ERRORS):
        msg = ("the apptainer container could not launch (user-namespace creation denied; "
               "check user.max_user_namespaces). TRANSP never ran — infrastructure, not "
               "physics." + tag)
        return {"category": "container_namespace_launch", "message": msg}

    # 3) Solver hardware signal (SIGFPE / SIGSEGV / ...) inside the binary ----
    m = _RE_SIGNAL.search(text)
    if m:
        signal = m.group(1)
        gloss = _SIGNAL_GLOSS.get(signal, "a fatal signal")
        idx = _signal_line_index(lines)
        pre = lines[max(0, idx - _PRE_TRAP_WINDOW):idx] if idx is not None else lines
        sim_time = _last_match(pre, _RE_TA)
        nstep = _last_match(pre, _RE_NSTEP)
        context = _classify_context(pre)
        when = f" at t≈{float(sim_time):.4g}s" if sim_time else ""
        step = f" (NSTEP {nstep})" if nstep else ""
        msg = (f"crashed inside the TRANSP solver with {signal} ({gloss}){when}{step}, "
               f"during {context}.")
        crumbs = _breadcrumbs(pre, context)
        if crumbs:
            msg += " " + crumbs
        msg += (" Driven by the physics inputs (not the node), so a requeue almost "
                "certainly reproduces it." + tag)
        return {"category": "physics_signal", "message": msg, "signal": signal,
                "sim_time": sim_time, "nstep": nstep, "context": context}

    # 4) Other fatal TRANSP aborts -------------------------------------------
    for sig in HARD_ABORT_SIGNATURES:
        if sig in text:
            msg = f"TRANSP aborted ('{sig.strip()}'); see the log tail." + tag
            return {"category": "hard_abort", "message": msg, "signature": sig}

    # 5) Unclassified --------------------------------------------------------
    tail = "\n".join(lines[-15:]).strip()
    msg = ("terminated without a NORMAL EXIT and with no recognized error signature. "
           f"Last log lines below.{tag}\n{tail}")
    return {"category": "unclassified", "message": msg}


def diagnose_transp_logfile(path):
    """Interactive convenience: diagnose a tr.log on disk and print the result."""
    path = Path(path).expanduser()
    diag = diagnose_transp_failure(path, logname=path.name)
    print(f"[MITIM] TRANSP failure diagnosis ({diag['category']}):\n  {diag['message']}")
    return diag
