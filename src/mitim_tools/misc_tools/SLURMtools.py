"""
SLURMtools — centralized resource resolution for MITIM-fusion.

This module resolves three inputs into a single allocation description:

    (user `allocation` dict) + (machine config) + (code hints)
                              │
                              ▼
                      ResolvedAllocation
                      ├── use_slurm: bool
                      ├── sbatch:  dict of LITERAL sbatch flag names
                      │            (e.g. 'cpus-per-task', 'gpus-per-node',
                      │             'mem', 'time', 'array', 'nodes',
                      │             'ntasks', 'ntasks-per-node', 'exclusive')
                      ├── mpi:     {'n', 'nomp', 'numa', 'mpinuma'} for code_call
                      ├── concurrency: max parallel radii in bash mode
                      └── submission_type: bash | slurm_standard | slurm_array

The resolver is the single place where:
    - default cores per code live (TGLF=4, NEO=1, CGYRO=16)
    - GPU-vs-CPU layout logic lives (previously only inside CGYROtools)
    - memory hierarchy is applied (user > code > machine)
    - non-SLURM bash-mode concurrency is computed

Callers pass the user-facing `allocation` dict:

    {'resources_per_call': int,   # parallelism unit per radial call
                                  # (CPU cores for TGLF/NEO, GPUs for CGYRO/GX)
     'minutes': int,              # wall-clock
     'mem': str|None}             # sbatch --mem string, optional
"""

from dataclasses import dataclass, field
from typing import Optional

from mitim_tools.misc_tools import CONFIGread


# ---------------------------------------------------------------------------
# Code hints: the small per-code table that replaces each tool's
# `code_slurm_settings` function.
# ---------------------------------------------------------------------------

CODE_HINTS = {
    "tglf":  {"default_resources_per_call": 4,  "uses_gpu": False, "full_node_mpi": False},
    "neo":   {"default_resources_per_call": 1,  "uses_gpu": False, "full_node_mpi": False},
    "cgyro": {"default_resources_per_call": 16, "uses_gpu": True,  "full_node_mpi": True,
              "cores_per_mpi": 16},  # OMP threads per MPI rank. With 32 cores/GPU on engaging
                                     # this gives 2 ranks/GPU via MPS — the layout that's been
                                     # validated by Nathan's runs. Combined with the cores-per-call
                                     # rank scaling in _resolve_mpi_layout, resources_per_call=2
                                     # → mpi.n=4 / nomp=16 / numa=4 / mpinuma=1.
    "gx":    {"default_resources_per_call": 4,  "uses_gpu": True,  "full_node_mpi": False,
              "fixed_mem": "100GB", "gpus_per_task": 1, "requires_gpu": True},
    # Add others (tgyro, ...) here as they adopt the resolver.
}


# ---------------------------------------------------------------------------
# ResolvedAllocation — what the resolver returns.
# ---------------------------------------------------------------------------

@dataclass
class ResolvedAllocation:
    use_slurm: bool
    sbatch: dict = field(default_factory=dict)     # native sbatch flag names
    mpi: dict = field(default_factory=dict)        # {n, nomp, numa, mpinuma}
    concurrency: int = 1                           # bash mode only
    submission_type: str = "bash"                  # bash | slurm_standard | slurm_array
    resources_per_call: int = 1                    # echoed for code_call(n=...)


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve(
    code,
    allocation=None,
    n_rhos=1,
    n_subfolders=1,
    machine_settings=None,
    launch_slurm=True,
    force_submission_type=None,
    job_name="mitim_job",
    array_list=None,
    exclusive=None,
):
    """
    Resolve user `allocation` + machine config + code hints → ResolvedAllocation.

    Parameters
    ----------
    code : str
        Code name ('tglf', 'neo', 'cgyro', ...). Used to look up CODE_HINTS.
    allocation : dict | None
        User-facing knobs: {'resources_per_call', 'minutes', 'mem'}. `None` → defaults.
        `resources_per_call` is the unit of parallelism per radial call — CPU cores
        for CPU codes (TGLF/NEO), GPUs for GPU codes (CGYRO/GX). The resolver maps
        it to the right sbatch flag and logs the mapping.
    n_rhos : int
        Number of radial locations per subfolder.
    n_subfolders : int
        Number of subfolders (typically 1; >1 when multiple plasmas batched).
    machine_settings : dict | None
        Output of CONFIGread.machineSettings(code=code). Fetched if None.
    launch_slurm : bool
        If False, force bash mode regardless of machine config.
    force_submission_type : str | None
        'slurm_standard' | 'slurm_array' | 'bash' to override the heuristic.
    job_name : str
        SLURM job name.
    array_list : list[str] | None
        Array indices for slurm_array submissions (e.g. ['0','1','2']).
    exclusive : bool | None
        If True, force `--exclusive` on the sbatch (whole-node reservation —
        useful with slurm_array on clusters that don't enforce per-job GPU
        isolation). If None, defer to the machine config's slurm.exclusive.
    """
    hints = CODE_HINTS.get(code, {"default_resources_per_call": 1, "uses_gpu": False, "full_node_mpi": False})
    allocation = dict(allocation or {})

    # Hard GPU requirement (e.g. GX): fail early on CPU-only machines.
    if hints.get("requires_gpu"):
        gpus_per_node_probe = (machine_settings or CONFIGread.machineSettings(code=code)).get("gpus_per_node", 0) or 0
        if gpus_per_node_probe == 0:
            raise Exception(
                f"[MITIM] {code} requires GPUs; selected machine has gpus_per_node=0. "
                "Choose a machine with GPUs in the config file."
            )

    # --- Fetch machine settings once (no per-code duplicated reads) ---------
    if machine_settings is None:
        machine_settings = CONFIGread.machineSettings(code=code)

    # MachineConfig subclasses dict, so plain `.get(...)` works whether the
    # caller passed a raw dict (tests) or a typed MachineConfig (real runs).
    cores_per_node = int(machine_settings.get("cores_per_node") or 0)
    gpus_per_node = int(machine_settings.get("gpus_per_node") or 0)
    code_cores_per_mpi = hints.get("cores_per_mpi")
    machine_slurm = machine_settings.get("slurm", {}) or {}
    has_slurm = bool(machine_slurm.get("partition"))

    # --- User knobs ---------------------------------------------------------
    resources_per_call = int(allocation.get("resources_per_call", hints["default_resources_per_call"]))
    minutes = int(allocation.get("minutes", 10))
    user_mem = allocation.get("mem", None)     # None → fall through to machine

    # --- Decide submission type --------------------------------------------
    use_slurm = launch_slurm and has_slurm
    total_cores_required = resources_per_call * n_rhos * n_subfolders

    if not use_slurm:
        submission_type = "bash"
    elif force_submission_type is not None:
        submission_type = force_submission_type
    else:
        # Heuristic: if all radii fit in one node's capacity, use standard;
        # otherwise use an array. For GPU codes the capacity is GPUs/node.
        capacity = gpus_per_node if (hints["uses_gpu"] and gpus_per_node > 0) else cores_per_node
        if capacity > 0 and total_cores_required < capacity:
            submission_type = "slurm_standard"
        else:
            submission_type = "slurm_array"

    # --- MPI layout (consumed by code_call) --------------------------------
    mpi = _resolve_mpi_layout(hints, resources_per_call, cores_per_node, gpus_per_node, code_cores_per_mpi)

    # --- Bash concurrency (non-SLURM mode) ---------------------------------
    concurrency = 1
    if submission_type == "bash":
        # Cap based on local cores (or GPU count for GPU codes).
        local_capacity = gpus_per_node if (hints["uses_gpu"] and gpus_per_node > 0) else cores_per_node
        if local_capacity <= 0:
            local_capacity = resources_per_call  # last resort: one at a time
        concurrency = max(1, local_capacity // max(1, resources_per_call))

    # --- Build sbatch dict (LITERAL sbatch flag names) ---------------------
    sbatch = {}
    if submission_type != "bash":
        sbatch["job-name"] = job_name
        sbatch["time"] = format_time(minutes)

        # Memory hierarchy: user allocation > code hint (fixed_mem) > machine config default
        resolved_mem = user_mem if user_mem is not None else hints.get("fixed_mem")
        if resolved_mem is None:
            resolved_mem = machine_slurm.get("mem")
        if resolved_mem is not None:
            sbatch["mem"] = resolved_mem

        # Code-level fixed gpus-per-task hint (GX: one GPU per MPI task)
        if hints.get("gpus_per_task") is not None and submission_type != "bash":
            sbatch["gpus-per-task"] = hints["gpus_per_task"]

        _fill_sbatch_layout(
            sbatch,
            submission_type=submission_type,
            hints=hints,
            resources_per_call=resources_per_call,
            cores_per_node=cores_per_node,
            gpus_per_node=gpus_per_node,
            code_cores_per_mpi=code_cores_per_mpi,
            n_rhos=n_rhos,
            n_subfolders=n_subfolders,
            array_list=array_list,
        )

        # Explicit user override for --exclusive (e.g. force whole-node
        # ownership for each slurm_array element on clusters that don't
        # enforce per-job GPU isolation). None = defer to machine config.
        if exclusive is True:
            sbatch["exclusive"] = True
        elif exclusive is False:
            sbatch["exclusive"] = False

    # --- One log line documenting the abstract → native mapping ------------
    _log_mapping(code, hints, resources_per_call, submission_type, sbatch)

    return ResolvedAllocation(
        use_slurm=use_slurm,
        sbatch=sbatch,
        mpi=mpi,
        concurrency=concurrency,
        submission_type=submission_type,
        resources_per_call=resources_per_call,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_mpi_layout(hints, resources_per_call, cores_per_node, gpus_per_node, code_cores_per_mpi=None):
    """MPI parameters that the per-code `code_call` needs.

    GPU/full-node-MPI scaling (CGYRO): rank count scales with the *cores* the
    call is entitled to, not just the GPU count, so we actually use the node's
    CPU side instead of leaving 87% idle.

        cores_per_call = cores_per_node × (resources_per_call / gpus_per_node)
        n_ranks        = cores_per_call / cores_per_mpi
        nomp           = cores_per_mpi
        numa = n_ranks, mpinuma = 1   (1 rank per NUMA, fine up to a node's NUMA count)

    Concrete on engaging_rpp_gpu_nathan (cores=128, gpus=4, cores_per_mpi=16):
        rpc=1 -> n=2  / nomp=16 / numa=2  (¼ node)
        rpc=2 -> n=4  / nomp=16 / numa=4  (½ node, 2 ranks/GPU via MPS)
        rpc=4 -> n=8  / nomp=16 / numa=8  (full node — numa>physical NUMA, see TODO)

    TODO: when `n_ranks` exceeds physical NUMA-per-node (typically 4 on these
    GPU nodes), the right move is mpinuma>1 rather than overstating numa.
    Currently no caller hits this since rpc≤2 is what's used in practice.
    """
    if hints["uses_gpu"] and hints["full_node_mpi"] and gpus_per_node > 0 and cores_per_node > 0:
        nomp = int(code_cores_per_mpi) if code_cores_per_mpi else max(1, cores_per_node // gpus_per_node)
        cores_per_call = max(1, (cores_per_node * int(resources_per_call)) // gpus_per_node)
        n_ranks = max(1, cores_per_call // nomp)
        return {
            "n": n_ranks,
            "nomp": nomp,
            "numa": n_ranks,
            "mpinuma": 1,
        }
    return {"n": resources_per_call, "nomp": 1, "numa": None, "mpinuma": None}


def _fill_sbatch_layout(sbatch, *, submission_type, hints, resources_per_call,
                        cores_per_node, gpus_per_node, code_cores_per_mpi,
                        n_rhos, n_subfolders, array_list):
    """Populate nodes/ntasks/cpus-per-task/gpus-per-node etc. using native names."""
    if hints["uses_gpu"] and hints["full_node_mpi"] and gpus_per_node > 0:
        # CGYRO GPU path. n_ranks scales with cores-per-call (cores_per_node ×
        # rpc/gpus_per_node) so MPS-shared layouts like Nathan's (4 ranks on 2
        # GPUs with 16 OMP each) emerge naturally from `resources_per_call=2`.
        # `gpus-per-node` still tracks the GPU slice so the scheduler honors
        # GPU partitioning.
        n_gpus_requested = max(1, min(int(resources_per_call), gpus_per_node))
        omp_per_task = int(code_cores_per_mpi) if code_cores_per_mpi else (
            cores_per_node // gpus_per_node if cores_per_node else 1)
        cores_per_call = max(1, (int(cores_per_node or 0) * int(resources_per_call)) // gpus_per_node) \
            if cores_per_node else (n_gpus_requested * omp_per_task)
        n_ranks = max(1, cores_per_call // omp_per_task)

        if submission_type == "slurm_standard":
            n_radii = n_rhos * n_subfolders
            sbatch["ntasks"] = n_ranks * n_radii
        elif submission_type == "slurm_array":
            sbatch["nodes"] = 1
            sbatch["ntasks-per-node"] = n_ranks
            sbatch["array"] = ",".join(array_list or [])

        sbatch["cpus-per-task"] = omp_per_task
        sbatch["gpus-per-node"] = n_gpus_requested

    elif hints["uses_gpu"]:
        # GX-style GPU codes: one MPI rank per GPU; resources_per_call = GPUs per
        # radial call. `gpus-per-task` is set from the hints table, so we just
        # need ntasks = resources_per_call × n_radii. Do NOT set cpus-per-task —
        # GX doesn't multi-thread and a stray --cpus-per-task would confuse the
        # MPI launcher slot count.
        if submission_type == "slurm_standard":
            sbatch["ntasks"] = resources_per_call * n_rhos * n_subfolders
        elif submission_type == "slurm_array":
            sbatch["ntasks"] = resources_per_call
            sbatch["array"] = ",".join(array_list or [])

    else:
        # CPU codes (TGLF / NEO / CPU-CGYRO). One resource == one CPU core.
        if submission_type == "slurm_standard":
            sbatch["ntasks"] = n_rhos * n_subfolders
        elif submission_type == "slurm_array":
            sbatch["ntasks"] = 1
            sbatch["array"] = ",".join(array_list or [])
        sbatch["cpus-per-task"] = resources_per_call


def _log_mapping(code, hints, resources_per_call, submission_type, sbatch):
    """One-line explainer at resolve time — makes the CPU-vs-GPU unit obvious."""
    try:
        from mitim_tools.misc_tools.LOGtools import printMsg as _print
    except Exception:
        return
    if submission_type == "bash":
        unit = "GPU" if hints.get("uses_gpu") else "CPU core"
        _print(f"\t- {code}: resources_per_call={resources_per_call} → {resources_per_call} {unit}(s) per radial call (bash)", typeMsg="i")
        return
    if hints.get("uses_gpu") and hints.get("full_node_mpi"):
        gpn = sbatch.get("gpus-per-node")
        _print(f"\t- {code}: resources_per_call={resources_per_call} → --gpus-per-node={gpn} (GPUs per radial call)", typeMsg="i")
    elif hints.get("uses_gpu"):
        gpt = sbatch.get("gpus-per-task")
        _print(f"\t- {code}: resources_per_call={resources_per_call} → --gpus-per-task={gpt} × ntasks={sbatch.get('ntasks')}", typeMsg="i")
    else:
        cpt = sbatch.get("cpus-per-task")
        _print(f"\t- {code}: resources_per_call={resources_per_call} → --cpus-per-task={cpt} (CPU cores per radial call)", typeMsg="i")


def format_time(minutes):
    minutes = int(minutes)
    if minutes >= 60:
        h, m = divmod(minutes, 60)
        return f"{h:02d}:{m:02d}:00"
    return f"{minutes:02d}:00"
