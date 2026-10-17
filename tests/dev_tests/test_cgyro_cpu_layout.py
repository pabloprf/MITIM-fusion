"""
test_cgyro_cpu_layout.py
========================
Nonlinear CGYRO on CPU-only machines (gpus_per_node 0): resources_per_call is CPU cores per radial
call, split into MPI ranks x OMP threads by allocation['omp_threads_cpu'] (default 1), over whole
nodes when the call exceeds one node.

Checked here, with raw machine dicts (no config file, no ssh):
    - SLURMtools MPI layout and sbatch dicts on engaging CPU (64 cores/node) and Perlmutter CPU
      (128 cores/node), array and standard, plus the error cases;
    - the GPU layouts/sbatch (engaging 128 cores + 4 GPUs, Perlmutter 4 GPUs/node) equal the
      dicts origin/development (637fd85f) produced, hard-coded below;
    - the cgyro command line CgyroLaunchBody writes on CPU, and that code_call carries the OMP knob;
    - TOROIDALS_PER_PROC on CPU (rank count = cores / threads) and GPU, and the error when no value fits the grid.

Run as:

    python tests/dev_tests/test_cgyro_cpu_layout.py

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
mitim_root = repo_root / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools import CONFIGread, SLURMtools

ENGAGING_CPU = {"machine": "x", "cores_per_node": 64, "gpus_per_node": 0,
                "slurm": {"partition": "sched_mit_psfc_r8,mit_preemptable", "exclusive": True, "mem": "230G"}}
PERLMUTTER_CPU = {"machine": "x", "cores_per_node": 128, "gpus_per_node": 0,
                  "slurm": {"account": "m3195", "constraint": "cpu", "qos": "regular"}}
ENGAGING_GPU = {"machine": "x", "cores_per_node": 128, "gpus_per_node": 4, "slurm": {"partition": "p", "mem": "500GB"}}
PERLMUTTER_GPU = {"machine": "x", "cores_per_node": 64, "gpus_per_node": 4, "slurm": {"account": "m3195", "constraint": "gpu"}}
LOCAL_CPU = {"machine": "local", "cores_per_node": 8, "gpus_per_node": 0}

# Grid of the ARC reduced nonlinear case (engaging_scaling/input.cgyro.base): nv = 8*16 (x4 species), nc = 270*16
GRID = {"N_TOROIDAL": 16, "N_ENERGY": 8, "N_XI": 16, "N_THETA": 16, "N_RADIAL": 270}


def resolve(machine, rpc, submission_type, omp=None, n_rhos=3):
    allocation = {"resources_per_call": rpc, "minutes": 60}
    if omp is not None:
        allocation["omp_threads_cpu"] = omp
    return SLURMtools.resolve("cgyro", allocation, n_rhos=n_rhos, machine_settings=machine, force_submission_type=submission_type,
                              job_name="j", array_list=[str(i) for i in range(n_rhos)], verbose=False)


def raises(fn, *args, match="", **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        assert match in str(e), str(e)
        return str(e)
    raise AssertionError(f"{fn.__name__}{args} did not raise")


@contextlib.contextmanager
def machine(block):
    '''CONFIGread.machineSettings returns `block` for every code (CGYROtools reads it through the module).'''
    keep = CONFIGread.machineSettings
    CONFIGread.machineSettings = lambda *a, **k: dict(block)
    try:
        yield
    finally:
        CONFIGread.machineSettings = keep


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cpu_mpi_layouts():
    '''(n, nomp, nodes) for the CPU sizes we run; numa/mpinuma stay None so cgyro gets no -numa.'''
    cases = [
        (ENGAGING_CPU, 64, 1, (64, 1, 1)), (ENGAGING_CPU, 64, 2, (32, 2, 1)),
        (ENGAGING_CPU, 128, 1, (128, 1, 2)), (ENGAGING_CPU, 128, 2, (64, 2, 2)),
        (ENGAGING_CPU, 512, 1, (512, 1, 8)), (ENGAGING_CPU, 512, 2, (256, 2, 8)),
        (PERLMUTTER_CPU, 128, 1, (128, 1, 1)), (PERLMUTTER_CPU, 128, 4, (32, 4, 1)),
        (PERLMUTTER_CPU, 2048, 4, (512, 4, 16)),
    ]
    for m, rpc, omp, (n, nomp, nodes) in cases:
        mpi = resolve(m, rpc, "slurm_array", omp=omp).mpi
        assert mpi == {"n": n, "nomp": nomp, "numa": None, "mpinuma": None, "nodes": nodes}, (m["cores_per_node"], rpc, omp, mpi)
    # Knob omitted -> 1 thread per rank, never the GPU hint's 16
    assert resolve(ENGAGING_CPU, 128, "slurm_array").mpi["nomp"] == 1
    print("PASS: CPU MPI layouts (engaging 64/128/512, Perlmutter 128/2048, nomp 1/2/4)")


def test_cpu_layout_errors():
    raises(resolve, ENGAGING_CPU, 96, "slurm_array", match="not a multiple of it")            # 1.5 nodes
    raises(resolve, ENGAGING_CPU, 65, "slurm_array", omp=2, match="not a multiple of omp_threads_cpu")
    raises(resolve, PERLMUTTER_CPU, 2000, "slurm_array", omp=4, match="not a multiple of it")
    raises(resolve, ENGAGING_CPU, 192, "slurm_array", omp=3, match="does not divide cores_per_node")  # 21.3 ranks/node
    # Bash mode (local runs, driver inside an allocation) never spans nodes: an oversized call only
    # oversubscribes, as before, instead of raising
    for rpc in (16, 12):
        r = resolve(LOCAL_CPU, rpc, None, n_rhos=1)
        assert r.submission_type == "bash" and r.mpi == {"n": rpc, "nomp": 1, "numa": None, "mpinuma": None, "nodes": 1}, r
    raises(resolve, LOCAL_CPU, 9, None, omp=2, n_rhos=1, match="not a multiple of omp_threads_cpu")
    print("PASS: CPU layout errors (partial node, rpc % omp, cores % omp) and bash tolerance")


def test_cpu_sbatch_array_and_standard():
    base = {"job-name": "j", "time": "01:00:00"}
    r = resolve(ENGAGING_CPU, 128, "slurm_array", omp=2)
    assert r.sbatch == {**base, "mem": "230G", "nodes": 2, "ntasks-per-node": 32, "cpus-per-task": 2, "array": "0,1,2"}, r.sbatch
    r = resolve(ENGAGING_CPU, 64, "slurm_array")
    assert r.sbatch == {**base, "mem": "230G", "nodes": 1, "ntasks-per-node": 64, "cpus-per-task": 1, "array": "0,1,2"}, r.sbatch
    r = resolve(PERLMUTTER_CPU, 2048, "slurm_array", omp=4)
    assert r.sbatch == {**base, "nodes": 16, "ntasks-per-node": 32, "cpus-per-task": 4, "array": "0,1,2"}, r.sbatch
    r = SLURMtools.resolve("cgyro", {"resources_per_call": 128, "minutes": 60, "max_concurrent_calls": 2}, n_rhos=3,
                           machine_settings=ENGAGING_CPU, force_submission_type="slurm_array", job_name="j",
                           array_list=["0", "1", "2"], verbose=False)
    assert r.sbatch["array_limit"] == 2 and "gpus-per-node" not in r.sbatch
    # Standard: one allocation for all radii
    r = resolve(ENGAGING_CPU, 128, "slurm_standard", omp=2)
    assert r.sbatch == {**base, "mem": "230G", "ntasks": 192, "cpus-per-task": 2, "nodes": 6}, r.sbatch
    r = resolve(ENGAGING_CPU, 32, "slurm_standard")
    assert r.sbatch == {**base, "mem": "230G", "ntasks": 96, "cpus-per-task": 1}, r.sbatch
    r = resolve(PERLMUTTER_CPU, 2048, "slurm_standard", omp=4)
    assert r.sbatch == {**base, "ntasks": 1536, "cpus-per-task": 4, "nodes": 48}, r.sbatch
    print("PASS: CPU sbatch dicts (array + standard, no GPU flags)")


def test_gpu_layouts_unchanged():
    '''Dicts produced by origin/development 637fd85f for the same inputs (captured before the change).'''
    expected = {
        (id(ENGAGING_GPU), 4, "slurm_array"): (
            {"mpinuma": 1, "n": 4, "nodes": 1, "nomp": 16, "numa": 4},
            {"array": "0,1,2", "cpus-per-task": 16, "gpus-per-node": 4, "job-name": "j", "mem": "500GB", "nodes": 1,
             "ntasks-per-node": 4, "time": "01:00:00"}),
        (id(ENGAGING_GPU), 4, "slurm_standard"): (
            {"mpinuma": 1, "n": 4, "nodes": 1, "nomp": 16, "numa": 4},
            {"cpus-per-task": 16, "gpus-per-node": 4, "job-name": "j", "mem": "500GB", "ntasks": 12, "time": "01:00:00"}),
        (id(PERLMUTTER_GPU), 8, "slurm_array"): (
            {"mpinuma": 1, "n": 8, "nodes": 2, "nomp": 16, "numa": 4},
            {"array": "0,1,2", "cpus-per-task": 16, "gpus-per-node": 4, "job-name": "j", "nodes": 2, "ntasks-per-node": 4,
             "time": "01:00:00"}),
        (id(PERLMUTTER_GPU), 8, "slurm_standard"): (
            {"mpinuma": 1, "n": 8, "nodes": 2, "nomp": 16, "numa": 4},
            {"cpus-per-task": 16, "gpus-per-node": 4, "job-name": "j", "nodes": 6, "ntasks": 24, "time": "01:00:00"}),
    }
    for m in (ENGAGING_GPU, PERLMUTTER_GPU):
        for (mid, rpc, st), (mpi, sbatch) in expected.items():
            if mid != id(m):
                continue
            for omp in (None, 4):   # the CPU knob must not touch the GPU path
                r = resolve(m, rpc, st, omp=omp)
                assert r.mpi == mpi and r.sbatch == sbatch, (rpc, st, omp, r.mpi, r.sbatch)
    print("PASS: GPU layouts and sbatch identical to origin/development (engaging rpc 4, Perlmutter rpc 8)")


def test_cpu_launch_line():
    with machine(ENGAGING_CPU):
        body = CGYROtools.CgyroLaunchBody("base_cgyro/rho_0.5000", "/scratch/x", n=128, omp_threads_cpu=2)
        txt = body.launch()
        assert not body.bash_mode and body.nodes == 2
        assert txt == ("export OMP_NUM_THREADS=2\nexport OMP_STACKSIZE=1G\nexport OMPI_MCA_io=^ompio\n"
                       "cgyro -e base_cgyro/rho_0.5000 -n 64 -nomp 2 -p /scratch/x "), txt
        # code_call takes the knob from the allocation the run was called with
        cg = quiet(CGYROtools.CGYRO)
        assert cg.run_specifications["force_submission_type"] == "slurm_array"
        cg._allocation = {"resources_per_call": 128, "omp_threads_cpu": 2}
        assert "cgyro -e base_cgyro/rho_0.5000 -n 64 -nomp 2 -p /scratch/x" in cg.code_call("base_cgyro/rho_0.5000", "/scratch/x", n=128)
    with machine(LOCAL_CPU):
        # Local bash (no SLURM block): same plain line, no srun wrapping, no host selection
        body = CGYROtools.CgyroLaunchBody("base_cgyro/rho_0.5000", "/scratch/x", n=8)
        assert body.resolved.submission_type == "bash" and not body.bash_mode and body.hosts == []
        assert body.launch().endswith("cgyro -e base_cgyro/rho_0.5000 -n 8 -nomp 1 -p /scratch/x ")
    print("PASS: CPU cgyro line (-n ranks -nomp threads, no -numa) in array and local bash")


def test_cpu_toroidals_per_proc():
    with machine(ENGAGING_CPU):
        cg = quiet(CGYROtools.CGYRO)
        # 128 ranks (2 nodes): n_toroidal_procs 16 <= 64 ranks/node, n_proc_1 = 8 divides nv and nc
        assert quiet(cg._enforce_toroidals_per_proc, dict(GRID), {"resources_per_call": 128})["TOROIDALS_PER_PROC"] == 1
        # 128 cores / 2 threads = 64 ranks: n_proc_1 = 4
        assert quiet(cg._enforce_toroidals_per_proc, dict(GRID), {"resources_per_call": 128, "omp_threads_cpu": 2})["TOROIDALS_PER_PROC"] == 1
    with machine(PERLMUTTER_CPU):
        cg = quiet(CGYROtools.CGYRO)
        # 2048 cores / 4 threads = 512 ranks over 16 nodes: n_proc_1 = 32 divides nv = 128 and nc = 4320
        assert quiet(cg._enforce_toroidals_per_proc, dict(GRID), {"resources_per_call": 2048, "omp_threads_cpu": 4})["TOROIDALS_PER_PROC"] == 1
        # 96 ranks: n_proc_1 in {96, 48, 24, 12, 6}, none divides nv = 128 -> refuse, naming the grid
        msg = raises(quiet, cg._enforce_toroidals_per_proc, dict(GRID), {"resources_per_call": 96}, match="N_TOROIDAL=16")
        for piece in ("nv = N_ENERGY*N_XI = 8*16 = 128", "nc = N_RADIAL*N_THETA = 270*16 = 4320", "96 MPI ranks", "[1, 2, 4, 8, 16, 32, 64, 128]"):
            assert piece in msg, (piece, msg)
        # An explicit TOROIDALS_PER_PROC bypasses the check (warning only), and an unresolved grid only warns
        assert quiet(cg._enforce_toroidals_per_proc, {**GRID, "TOROIDALS_PER_PROC": 4}, {"resources_per_call": 96})["TOROIDALS_PER_PROC"] == 4
        partial = {k: v for k, v in GRID.items() if k != "N_ENERGY"}
        assert quiet(cg._enforce_toroidals_per_proc, partial, {"resources_per_call": 96})["TOROIDALS_PER_PROC"] == 1
    print("PASS: CPU TOROIDALS_PER_PROC (ranks = cores/threads) and its grid error")


def test_gpu_toroidals_per_proc():
    # Every GPU choice that passes the grid rule equals origin/development (637fd85f)
    for block, expected in ((ENGAGING_GPU, {1: 16, 2: 8, 4: 4, 8: 4}), (PERLMUTTER_GPU, {4: 4, 8: 4})):
        with machine(block):
            cg = quiet(CGYROtools.CGYRO)
            for rpc, tpp in expected.items():
                assert quiet(cg._enforce_toroidals_per_proc, dict(GRID), {"resources_per_call": rpc})["TOROIDALS_PER_PROC"] == tpp, (block, rpc)
    with machine(PERLMUTTER_GPU):
        cg = quiet(CGYROtools.CGYRO)
        # 48 GPUs: origin/development silently took TOROIDALS_PER_PROC 1 -> n_proc_1 = 3, which CGYRO rejects at startup
        msg = raises(quiet, cg._enforce_toroidals_per_proc, dict(GRID), {"resources_per_call": 48}, match="N_TOROIDAL=16")
        for piece in ("resources_per_call=48 (n_proc)", "nv = N_ENERGY*N_XI = 8*16 = 128", "resources_per_call = GPUs = ranks"):
            assert piece in msg, (piece, msg)
        # explicit TOROIDALS_PER_PROC still bypasses it; an unresolved grid still only warns
        assert quiet(cg._enforce_toroidals_per_proc, {**GRID, "TOROIDALS_PER_PROC": 1}, {"resources_per_call": 48})["TOROIDALS_PER_PROC"] == 1
        partial = {k: v for k, v in GRID.items() if k != "N_XI"}
        assert quiet(cg._enforce_toroidals_per_proc, partial, {"resources_per_call": 48})["TOROIDALS_PER_PROC"] == 1
    print("PASS: GPU TOROIDALS_PER_PROC unchanged where the grid fits (engaging rpc 1/2/4/8, Perlmutter 4/8); rpc 48 raises")


if __name__ == "__main__":
    test_cpu_mpi_layouts()
    test_cpu_layout_errors()
    test_cpu_sbatch_array_and_standard()
    test_gpu_layouts_unchanged()
    test_cpu_launch_line()
    test_cpu_toroidals_per_proc()
    test_gpu_toroidals_per_proc()
    print("\nALL PASS")
