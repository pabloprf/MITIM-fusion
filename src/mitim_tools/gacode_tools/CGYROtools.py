import os
import json
import shutil
import tempfile
import datetime
import functools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as plt
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools.utils import GACODEdefaults, CGYROutils
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.utils import SIMplot
from mitim_tools.misc_tools import GRAPHICStools, CONFIGread
from mitim_tools.misc_tools.FARMINGtools import SlurmState
from mitim_tools.gacode_tools.utils import GACODEplotting
from mitim_tools.misc_tools.LOGtools import printMsg as print


def _annotate_missing(ax, reason):
    '''
    Stamp a small "data unavailable" note on an axes when the underlying
    CGYRO output files weren't written or weren't retrieved (e.g.
    MOMENT_PRINT_FLAG=0 drops kxky_n/e/v; FIELD_PRINT_FLAG=0 drops
    kxky_apar/bpar). Keeps the surrounding title/labels intact so the reader
    sees which panel was supposed to be there.
    '''
    ax.text(
        0.5, 0.5, f"Data unavailable\n({reason})",
        ha='center', va='center', transform=ax.transAxes,
        color='gray', style='italic', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.4),
    )


# Watchdog, CostGuard, the shell templates and body_keeping_exit_status live in simulation_tools/gk_control
# (aliases kept as CGYROtools attributes)
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.simulation_tools.gk_control.shell import (
    body_keeping_exit_status, _ShellTemplate, _shell_text, _WATCHDOG_BASH, _PROBE_BASH, _REQUEUE_TRIM_BASH)
from mitim_tools.simulation_tools.gk_control.watchdog import Watchdog
from mitim_tools.simulation_tools.gk_control import watchdog as _watchdog
from mitim_tools.simulation_tools.gk_control.cost_guard import CostGuard
from mitim_tools.misc_tools import FARMINGtools as _FARMINGtools

# Before retrieve() drops bin.cgyro.restart.old next to a bin.cgyro.restart, repair the pair as a launch would
# (cgyro_restart_validate.sh): after a failed restart write the .old is the only intact checkpoint
_FARMINGtools.REMOTE_PAIR_REPAIRS["bin.cgyro.restart"] = functools.partial(_watchdog.restart_validate, apply=True)


class CgyroLaunchBody:
    '''
    Bash body of one radial CGYRO call: the environment every MPI rank inherits, the launch shape the
    machine needs, and the marker/cleanup files the rest of MITIM reads back. `build(watchdog)`
    assembles them into the body SIMtools' JobScript builders stage.
    '''

    def __init__(self, folder, p, n=1, additional_command="", resolved=None, cpus_per_node=1, omp_threads_cpu=None):
        from mitim_tools.misc_tools import SLURMtools

        self.folder = folder
        self.p = p
        self.additional_command = additional_command
        self.cpus_per_node = cpus_per_node
        self.machine = CONFIGread.machineSettings(code='cgyro')

        # MPI layout is resolved centrally in SLURMtools so the invented knobs (full-node MPI on GPU
        # machines, MPS sharing) live in one place instead of here and in code_slurm_settings.
        # omp_threads_cpu (allocation knob, CPU-only machines) must reach this resolve too: it sets
        # the -n/-nomp written below, while SIMtools' own resolve only sizes the sbatch
        allocation = {'resources_per_call': int(n)}
        if omp_threads_cpu is not None:
            allocation['omp_threads_cpu'] = omp_threads_cpu
        self.resolved = resolved if resolved is not None else SLURMtools.resolve(
            code='cgyro', allocation=allocation, verbose=False)
        self.mpi = self.resolved.mpi
        self.nodes = self.mpi.get("nodes", 1)   # >1 for multi-node radial calls (resources_per_call > gpus_per_node)

        # Bash mode inside an existing SLURM allocation (driver under salloc/sbatch)
        self.bash_mode = self.resolved.submission_type == "bash" and self.mpi.get("numa") is not None
        self.srun_wrap = bool(self.machine.get("srun_wrap_calls", False)) and self.bash_mode
        if self.srun_wrap and self.nodes > 1:
            raise ValueError("[MITIM] srun_wrap_calls supports single-node radial calls only (resources_per_call <= gpus_per_node)")
        self.hosts = SIMtools.slurm_allocation_hostnames() if self.bash_mode else []

    # ------------------------------------------------------------------
    def env_exports(self):
        '''
        What every rank must inherit. Without OMP_NUM_THREADS the OpenMPI launcher prints "could not
        find environment variable OMP_NUM_THREADS" and the CGYRO launcher's NUMA->GPU binding (driven
        by -numa/-mpinuma) silently collapses: all ranks end up on GPU 0 and OOM. OMP_STACKSIZE=1G is
        the GACODE-recommended default for GPU offload kernels; too small and the first large-grid
        kernel segfaults. OMPI_MCA_io=^ompio picks anything but OMPIO, which on NFS (engaging /orcd)
        spent 30-100 s per output step against <1 s for ROMIO; not a ROMIO name, since the component
        is romio321 in OpenMPI 4 and romio341 in OpenMPI 5 and naming one the build lacks leaves no
        MPI-IO at all (CGYRO dies in cgyro_write_hosts). Ignored by MPICH-based builds (Perlmutter GPU).

        In bash mode the srun-based launchers also need the step pinned to one node holding its own
        GPUs, so the concurrent calls the bash builder backgrounds land on different nodes (a shared
        node would map two calls onto the same GPUs via SLURM_LOCALID). srun honors these as input
        environment variables; the node count is read from SLURM_JOB_NUM_NODES (SLURM_NNODES alone is
        ignored), both are set for safety.
        '''
        exports = (
            f"export OMP_NUM_THREADS={self.mpi['nomp']}\n"
            f"export OMP_STACKSIZE=1G\n"
            "export OMPI_MCA_io=^ompio\n"
        )
        exports += self.host_selection()
        if self.bash_mode and not self.srun_wrap:
            exports += (
                f"export SLURM_JOB_NUM_NODES={self.nodes}\n"
                f"export SLURM_NNODES={self.nodes}\n"
                f"export SLURM_GPUS_PER_NODE={self.mpi['numa']}\n"
            )
            if self.hosts:
                exports += "export SLURM_JOB_NODELIST=$_sel; export SLURM_NODELIST=$_sel\n"
        return exports

    @property
    def calls_per_node(self):
        '''How many radial calls share one node: >1 when a call takes fewer GPUs than the node has.'''
        gpus_per_node = int(self.machine.get("gpus_per_node") or 0)
        return max(1, gpus_per_node // self.mpi['numa']) if gpus_per_node else 1

    def host_selection(self):
        '''
        Node choice happens in bash: the builder runs ONE body for every radius in a loop, SIMtools
        exports the allocation as MITIM_HOSTS and a 1-based MITIM_CALL counter, and call k takes
        hosts [(k-1)*nodes, k*nodes). When several calls share a node, call k takes host k // calls_per_node;
        which of that node's GPUs it gets is SLURM's choice (see _srun_step).
        '''
        if not self.hosts:
            return ""
        if self.calls_per_node > 1:
            return f"_cpn={self.calls_per_node}; _k=$((MITIM_CALL-1)); _nh=${{#MITIM_HOSTS[@]}}; _sel=${{MITIM_HOSTS[$(( (_k/_cpn) % _nh ))]}}\n"
        return (
            f"_npc={self.nodes}; _k=$((MITIM_CALL-1)); _nh=${{#MITIM_HOSTS[@]}}; _sel=\"\"\n"
            f"for _j in $(seq 0 $((_npc-1))); do _h=${{MITIM_HOSTS[$(( (_k*_npc+_j) % _nh ))]}}; _sel=\"${{_sel}}${{_sel:+,}}${{_h}}\"; done\n"
        )

    def _cgyro_invocation(self, folder=None, numa=True, trailing=None):
        '''The cgyro command line. `trailing` defaults to a space plus additional_command, empty or not.'''
        m = self.mpi
        layout = f"-numa {m['numa']} -mpinuma {m['mpinuma']} " if numa else ""
        tail = f" {self.additional_command}" if trailing is None else trailing
        return (f"cgyro -e {folder or self.folder} -n {m['n']} -nomp {m['nomp']} "
                f"{layout}-p {self.p}{tail}")

    def _srun_step(self):
        '''
        Machines whose gacode launcher is OpenMPI `mpirun` (engaging PSFCR8_GPU): inside a multi-node
        allocation mpirun launches its daemons wherever SLURM puts them and ignores hostfiles. Run each
        radial call as an srun step ON its node instead (numa tasks so the step owns the node's CPUs and
        GPUs; only task 0 runs mpirun) with the SLURM view narrowed to that node, so mpirun spawns its
        ranks locally with no daemons. Enabled per machine with `srun_wrap_calls: true` (single-node
        calls only). No SLURM_* exports outside the step: srun would read them as options. The step takes
        the node's whole CPU share of its GPUs (128 cores / 4 GPUs -> 32 per task on engaging R8), not
        just nomp: a whole-node cpuset is what lets the NUMA platform place ranks by rankfile.
        '''
        m = self.mpi
        inner = (
            "if [ \"$SLURM_PROCID\" != \"0\" ]; then exit 0; fi; "
            "export H=$(hostname); export SLURM_JOB_NODELIST=$H SLURM_NODELIST=$H SLURM_JOB_NUM_NODES=1 SLURM_NNODES=1 "
            f"SLURM_TASKS_PER_NODE={m['numa']} SLURM_NTASKS={m['numa']} SLURM_NPROCS={m['numa']} SLURM_JOB_CPUS_PER_NODE={m['nomp'] * m['numa']}; "
            f"export OMP_NUM_THREADS={m['nomp']} OMP_STACKSIZE=1G OMPI_MCA_io=^ompio; "
            + self._cgyro_invocation(folder='"$MITIM_FOLDER"', trailing="")
        )
        gpus_per_node = int(self.machine.get("gpus_per_node") or m['numa'])
        if self.calls_per_node > 1:
            # Calls sharing a node: each step owns its GPUs and cores exclusively (no --overlap, --exact),
            # so the gacode wrapper's CUDA_VISIBLE_DEVICES=<local rank> resolves inside the step's own
            # device cgroup. With --overlap every step was handed the node's first GPU. A step also
            # takes the job's whole --mem unless told its share, which serializes the calls.
            sharing = f"--exact ${{SLURM_MEM_PER_NODE:+--mem=$((SLURM_MEM_PER_NODE/{self.calls_per_node}))M}}"
            cpus_per_task = m['nomp']
        else:
            sharing, cpus_per_task = "--overlap", max(m['nomp'], self.cpus_per_node // max(gpus_per_node, 1))
        return (f"srun -N1 -n{m['numa']} -c{cpus_per_task} --gpus-per-node={m['numa']} --cpu-bind=none ${{_sel:+-w $_sel}} {sharing} --export=ALL "
                f"bash -c '{inner}' {self.additional_command}")

    def launch(self):
        '''Environment plus the cgyro invocation in the shape this machine needs.'''
        if self.srun_wrap:
            return self.env_exports() + f"export MITIM_FOLDER={self.folder}\n" + self._srun_step()
        return self.env_exports() + self._cgyro_invocation(numa=self.mpi.get("numa") is not None)

    def markers(self):
        '''(marker_cmd, cleanup_cmd) around the launch; see gk_control.watchdog.launch_markers.'''
        return _watchdog.launch_markers(f"{self.p}/{self.folder}")

    def exit_verdict(self):
        '''0 -> 1 when CGYRO did not finish, accepted stop -> 0; see gk_control.watchdog.exit_verdict.'''
        return _watchdog.exit_verdict(f"{self.p}/{self.folder}")

    def build(self, watchdog):
        marker_cmd, cleanup_cmd = self.markers()
        return body_keeping_exit_status(marker_cmd, watchdog.wrap(self.launch()), cleanup_cmd, self.exit_verdict())


# ----------------------------------------------------------------------------------------------------
# Run control (per-task status, stall rescue, submission lifecycle, extra points, the GKRunControl half of
# the CGYRO class) lives in simulation_tools/gk_control; these names stay reachable as CGYROtools attributes
# ----------------------------------------------------------------------------------------------------
from mitim_tools.simulation_tools.gk_control.status import (
    _format_wall_seconds, _SLURM_JOB_GONE_STATES, TaskState, seconds_remaining, seconds_to_accept, _as_seconds, _as_float,
    RadiusStatus, CgyroProbe)
from mitim_tools.simulation_tools.gk_control.rescue import (
    _slurm_elapsed_seconds, _slurm_state_for_target, LedgerEntry, StallRescuer, _cgyro_handle_stalled_tasks,
    _utc_now, _seconds_since)
from mitim_tools.simulation_tools.gk_control.submission import cgyro_per_task_status, cgyro_per_task_callback
from mitim_tools.simulation_tools.gk_control.extra_points import ArrayExtraPoints
from mitim_tools.simulation_tools.gk_control.cgyro_client import GKRunControl

class _ResolvedControls:
    '''
    The controls `_run_prepare` will actually materialise, resolved once for the `_enforce_*` steps:
    the per-rho controls snapshot (input.cgyro.controls), replaced wholesale by the model-yaml block
    when a code_settings label is given — which is what SIMtools.modifyInputs does — with extraOptions
    read on top of it.
    '''

    def __init__(self, sim, extraOptions, code_settings=None):
        controls = {}
        if sim.rhos is not None and len(sim.rhos) > 0:
            controls = dict(sim.inputs_files[sim.rhos[0]].controls)
        if code_settings is not None:
            try:
                controls = GACODEdefaults.addCGYROcontrol(code_settings)
            except Exception as e:
                print(
                    f"\t- [preprocess] Could not resolve code_settings={code_settings!r} "
                    f"against input.cgyro.models.yaml ({e}); falling back to controls file only",
                    typeMsg="w",
                )
        self.controls = controls
        self.extraOptions = extraOptions

    def get(self, key, default=None):
        '''The value that will be written for `key`, scalar or per-rho list.'''
        return self.extraOptions.get(key, self.controls.get(key, default))

    def as_list(self, key, n=1, cast=float, default=None):
        '''(values broadcast to length n, the source value) for `key`; (None, None) when absent.'''
        source = self.get(key, default)
        if source is None:
            return None, None
        values = [cast(v) for v in source] if self.is_list(source) else [cast(source)]
        return self.broadcast(values, n), source

    @staticmethod
    def is_list(value):
        return isinstance(value, (list, np.ndarray))

    @staticmethod
    def broadcast(values, n):
        '''A single value serves every rho; anything else is already per-rho.'''
        return values * n if len(values) == 1 else values


def _restart_outputs(max_time, delta_t, print_step):
    '''
    Data outputs within the run: CGYRO's time loop is i_time = 1..nint(MAX_TIME/DELTA_T) and it writes
    a restart when mod(i_time, RESTART_STEP*PRINT_STEP) == 0, so RESTART_STEP*PRINT_STEP must not
    exceed n_time. A ceil(MAX_TIME/(DELTA_T*PRINT_STEP)) overshoots n_time whenever the ratio is
    non-integer (e.g. DELTA_T=0.006) and the restart then never fires.
    '''
    return max(1, int(round(max_time / delta_t)) // int(round(print_step)))


def _coerced_restart_step(restart_step, n_outputs):
    '''
    Keep the user/controls RESTART_STEP when it fires at least once with its last write aligned to the
    final output step; otherwise take n_outputs, which writes exactly one restart at that step.
    '''
    return restart_step if (0 < restart_step <= n_outputs and n_outputs % restart_step == 0) else n_outputs


class CGYRO(GKRunControl, SIMtools.mitim_simulation, SIMplot.GKplotting):

    # Opts CGYRO into persisting slurm-submission metadata (jobid, remote
    # folder, retrieval plan) whenever run_type='submit' is used, so a later
    # PORTALS restart can re-attach to the in-flight job rather than resubmit.
    _submission_metadata_filename = markers.CGYRO_SUBMISSION

    # Restart blob the PORTALS warm-start chain (restart_from_cases) stages, retrieved per rho as
    # bin.cgyro.restart_<rho:.4f> (cgyro_restart.RestartChain)
    _warm_start_file = "bin.cgyro.restart"
    # Set by the PORTALS evaluator when restart_from_cases reads this run's restart files back (_restart_retrieval)
    keep_warm_start_file = False

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Transient state used by run() to feed preprocess_options into _run_prepare()
        self._preprocess_options = None
        self._extra_point_n = None

        # On SLURM machines, always use a job array so each radius gets its own allocation: its GPUs on
        # GPU machines, its whole nodes on CPU-only machines (a nonlinear CPU radius takes one or more
        # full nodes). allocation['submission_type'] still overrides it, and machines without SLURM
        # run bash regardless (SLURMtools.resolve).
        _force_submission_type = 'slurm_array'

        self.run_specifications = {
            'code': 'cgyro',
            'input_file': 'input.cgyro',
            'code_call': self.code_call,
            'control_function': GACODEdefaults.addCGYROcontrol,
            'controls_file': 'input.cgyro.controls',
            'state_converter': 'to_cgyro',
            'input_class': CGYROinput,
            'complete_variation': None,
            'default_cores': 16,  # Default cores to use in the simulation
            'output_class': CGYROutils.CGYROoutput,
            'force_submission_type': _force_submission_type,
            # Interrupted-run rescue (SIMtools._rescue_interrupted_runs): with both the
            # restart blob and out.cgyro.tag present, re-running `cgyro -e` in the same
            # folder continues the time integration (restart_flag=1) for MAX_TIME more
            # a/cs from the tag time (out.cgyro.tag: line 1 i_current, line 2 t_current).
            'rescue_spec': {'required': ['bin.cgyro.restart', 'out.cgyro.tag'], 'progress_file': 'out.cgyro.tag', 'progress_line': 2, 'time_key': 'MAX_TIME',
                            # RESTART_STEP was sized for the full MAX_TIME, so it has to follow the trim
                            # (and be excluded from the identity md5, like MAX_TIME, for the next rescue)
                            'after_trim': self._restart_step_after_trim,
                            'checksum_ignore': ['MAX_TIME', 'RESTART_STEP'],
                            'report_files': ['out.cgyro.time', 'bin.cgyro.ky_flux', 'bin.cgyro.restart', 'out.cgyro.tag'],
                            # a radius continued in place is guarded like a fresh one
                            'keep_staged': [CostGuard.FILE, CostGuard.SCRIPT],
                            # a tag or blob broken by a failed restart write is repaired at launch
                            # (cgyro_restart_validate.sh): read the time that repair resumes from
                            'progress_cmd': functools.partial(_watchdog.restart_validate, apply=False)},
            # A radius is only 'done' if CGYRO wrote its EXIT line (files exist from step 1 on)...
            'completion_marker': markers.CGYRO_COMPLETION,
            # ...or the watchdog stopped it past min_time. Optional, never mandatory: a run that ends
            # by itself never writes this file.
            'completion_alt_file': markers.BUDGET_TAG,
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t CGYRO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.output_files_simulation["minimal_base"] = [
            "bin.cgyro.geo",
            "bin.cgyro.ky_cflux",
            "bin.cgyro.ky_flux",
            "input.cgyro.gen",
            "out.cgyro.egrid",
            "out.cgyro.equilibrium",
            "out.cgyro.grids",
            "out.cgyro.hosts",
            "out.cgyro.info",
            "out.cgyro.memory",
            "out.cgyro.mpi",
            "out.cgyro.prec",
            "out.cgyro.rotation",
            "out.cgyro.startups",
            "out.cgyro.time",
            "out.cgyro.timing",
            "out.cgyro.version",
        ]

        self.output_files_simulation["complete_base"] = list(self.output_files_simulation["minimal_base"])

        # Best-effort retrievals: tarred if present, absence logged once (no
        # 60s retry, no cold-start trigger). Two groups:
        #   - restart blobs + companion .flag/.tag that CGYRO may skip writing
        #     on short runs, crashes, or COMPLETING-timeouts.
        #   - large bin.cgyro.kxky_* dumps (tens to hundreds of MB each) that
        #     diagnostics can do without and whose retrieval over a slow
        #     shared filesystem was the dominant cost of fetch().
        self.output_files_simulation["optional_base"] = [
            "bin.cgyro.restart",
            "bin.cgyro.restart.flag",
            "out.cgyro.tag",
            "bin.cgyro.kxky_apar",
            "bin.cgyro.kxky_bpar",
            "bin.cgyro.kxky_e",
            "bin.cgyro.kxky_n",
            "bin.cgyro.kxky_phi",
            "bin.cgyro.kxky_v",
            markers.BUDGET_TAG,
        ]

        # Nonlinear sim
        for key in ['minimal', 'complete']:
            self.output_files_simulation[f"{key}_nonlinear"] = self.output_files_simulation[f"{key}_base"] + [
                "bin.cgyro.freq"
                ]

        # Linear sim
        for key in ['minimal', 'complete']:
            self.output_files_simulation[f"{key}_linear"] = self.output_files_simulation[f"{key}_base"] + [
                "out.cgyro.freq",
                "bin.cgyro.phib",
                "bin.cgyro.aparb",
                "bin.cgyro.bparb",
                ]

        # Make sure, just in case, that "complete" and "minimal" are populated from this __init__, even if it will be re-defined later
        self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_nonlinear"])
        self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_nonlinear"])
        self.output_files_simulation["optional"] = copy.deepcopy(self.output_files_simulation["optional_base"])

        # Primary/fallback pairs for the remote-prune step in retrieve().
        # bin.cgyro.restart.old is the previous-cycle checkpoint CGYRO keeps
        # under RESTART_PRESERVATION_MODE>=3 (default 3) as a durability net
        # while the next .restart is being written. If the job died mid-
        # write the only intact checkpoint on disk is .old — promote it to
        # .restart on the remote before tarring so we pull exactly one file
        # and still recover from the degenerate case. See
        # gacode/cgyro/src/cgyro_restart.F90:120-220.
        self.output_file_fallbacks = {
            "bin.cgyro.restart": "bin.cgyro.restart.old",
        }
        

    @staticmethod
    def _allocation_cpus_per_node():
        '''Per-node CPU count of the allocation (first entry of SLURM_JOB_CPUS_PER_NODE, e.g. "128(x5)" -> 128).'''
        raw = os.environ.get("SLURM_JOB_CPUS_PER_NODE", "")
        digits = "".join(ch for ch in raw.split("(")[0].split(",")[0] if ch.isdigit())
        return int(digits) if digits else 1

    def code_call(self, folder, p, n=1, additional_command="", watchdog=None, resolved=None):
        '''
        Bash body of one radial call, in the shape SIMtools' JobScript builders ask for
        (`code_call(folder=..., n=..., p=...)`).
        watchdog: Watchdog mode for this launch; None takes it from load_balance.
        resolved: an already-resolved SLURMtools allocation, when the caller has one.
        '''
        body = CgyroLaunchBody(folder, p, n=n, additional_command=additional_command,
                               resolved=resolved, cpus_per_node=self._allocation_cpus_per_node(),
                               omp_threads_cpu=(getattr(self, "_allocation", None) or {}).get("omp_threads_cpu"))
        guard = getattr(self, "_cost_guard", None)
        return body.build(Watchdog.from_load_balance(
            f"{p}/{folder}", getattr(self, "_load_balance", None), mode=watchdog, template=self._WALL_BUDGET_WATCHDOG,
            guard=guard is not None and guard.enabled))

    def _restart_retrieval(self, only_minimal_files):
        '''
        A minimal retrieval (keep_files "none"/"pickle") skips the restart blob (~0.7 GB per radius on Perlmutter
        grids) unless a PORTALS warm-start chain reads it back locally (keep_warm_start_file). Rescue, requeue
        resumes and mitim_kill_cgyro only use the copy in the remote scratch.
        '''
        if only_minimal_files and not self.keep_warm_start_file:
            restart = (self._warm_start_file, f"{self._warm_start_file}.flag")
            self.output_files_simulation["optional"] = [f for f in self.output_files_simulation["optional"] if f not in restart]

    # Redefine to raise warning and allow selection of output files
    def _run_prepare(
        self,
        subfolder_simulation,
        extraOptions=None,
        multipliers=None,
        **kwargs,
    ):

        # ---------------------------------------------
        # Check if any *_SCALE_* variable is being used
        # ---------------------------------------------
        dictionary_check = {}
        if extraOptions is not None:
            if multipliers is not None:
                dictionary_check = {**extraOptions, **multipliers}
            else:
                dictionary_check = extraOptions
        elif multipliers is not None:
                dictionary_check = multipliers

        for key in dictionary_check:
            if '_SCALE_' in key:
                print("The use of *_SCALE_* is discouraged, please use the appropriate variable instead.", typeMsg='q')

        # ---------------------------------------------

        # Check if it's linear
        if 'Nonlinear' not in kwargs.get('code_settings', ''):
            self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_linear"])
            self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_linear"])
        else:
            self.output_files_simulation["complete"] = copy.deepcopy(self.output_files_simulation["complete_nonlinear"])
            self.output_files_simulation["minimal"] = copy.deepcopy(self.output_files_simulation["minimal_nonlinear"])
        # Optional-retrieval set is the same for linear and nonlinear.
        self.output_files_simulation["optional"] = copy.deepcopy(self.output_files_simulation["optional_base"])
        self._restart_retrieval(kwargs.get("only_minimal_files", False))

        # Pre-process BOX_SIZE / N_RADIAL from local equilibrium if requested.
        # Model yaml (input.cgyro.models.yaml) can supply per-model defaults;
        # user-supplied self._preprocess_options override on a per-key basis.
        # fft_friendly / fft_tol only tune the preprocessing: alone they do not switch it on
        # (a model without preprocess_options, e.g. "Linear", keeps its KY / BOX_SIZE / N_RADIAL).
        from mitim_tools.gacode_tools.utils import GACODEdefaults
        model_preprocess = GACODEdefaults.getCGYROpreprocessDefaults(kwargs.get("code_settings"))
        user_preprocess = getattr(self, "_preprocess_options", None) or {}
        merged_preprocess = {**model_preprocess, **user_preprocess}
        if model_preprocess or set(user_preprocess) - {'fft_friendly', 'fft_tol'}:
            saved_preprocess = getattr(self, "_preprocess_options", None)
            self._preprocess_options = merged_preprocess
            try:
                extraOptions = self._apply_cgyro_preprocessing(extraOptions or {})
            finally:
                self._preprocess_options = saved_preprocess

        # Enforce TOROIDALS_PER_PROC compatibility with N_TOROIDAL and MPI rank count.
        # Resolution order matches what _run_prepare will eventually write:
        #   input.cgyro.controls  ->  input.cgyro.models.yaml[code_settings]  ->  extraOptions
        extraOptions = self._enforce_toroidals_per_proc(
            extraOptions or {},
            kwargs.get("allocation"),
            code_settings=kwargs.get("code_settings"),
        )

        # Enforce PRINT_STEP so that DELTA_T * PRINT_STEP == 1.0 (integer PRINT_STEP).
        # Same resolution order as above: controls -> yaml -> extraOptions.
        extraOptions = self._enforce_print_step(
            extraOptions or {},
            code_settings=kwargs.get("code_settings"),
        )

        # Ensure RESTART_STEP >= total number of data outputs so the restart file
        # is written at least once at the end. One data output == DELTA_T*PRINT_STEP,
        # so n_outputs = ceil(MAX_TIME / (DELTA_T*PRINT_STEP)).
        extraOptions = self._enforce_restart_step(
            extraOptions or {},
            code_settings=kwargs.get("code_settings"),
        )

        return super()._run_prepare(
            subfolder_simulation,
            extraOptions=extraOptions,
            multipliers=multipliers,
            **kwargs,
        )

    def _enforce_toroidals_per_proc(self, extraOptions, allocation, code_settings=None):
        """
        CGYRO distributes N_TOROIDAL across MPI ranks (= resources_per_call GPUs on GPU
        machines; resources_per_call / omp_threads_cpu on CPU-only machines).
        Each rank holds N_TOROIDAL/resources toroidal modes, so TOROIDALS_PER_PROC
        must be a multiple of that ratio. If the user's value is incompatible (or
        missing), coerce it to the preferred valid value and warn. A rank count that no
        TOROIDALS_PER_PROC can fit to the velocity/configuration grid raises instead (CPU and GPU).
        """
        from mitim_tools.misc_tools import SLURMtools

        allocation = allocation or {}
        default_rpc = SLURMtools.CODE_HINTS.get('cgyro', {}).get("default_resources_per_call", 1)
        resources_per_call = int(allocation.get("resources_per_call", default_rpc))
        if resources_per_call <= 0:
            return extraOptions

        # Ranks per node (one MPI rank per GPU for CGYRO). Only used to keep the nonlinear
        # all-to-all on-node; if the machine block cannot be read we fall back to a value
        # that makes the locality preference a no-op.
        try:
            machine = CONFIGread.machineSettings(code='cgyro')
            gpus_per_node = int(machine.get("gpus_per_node") or 0)
            cores_per_node = int(machine.get("cores_per_node") or 0)
            on_cpu = gpus_per_node == 0
        except Exception:
            gpus_per_node, cores_per_node, on_cpu = 0, 0, False
        if on_cpu:
            # CPU-only machine: resources_per_call is cores, split into ranks x omp_threads_cpu threads
            nomp = max(1, int(allocation.get("omp_threads_cpu") or 1))
            n_ranks = resources_per_call // nomp
            if n_ranks <= 0:
                return extraOptions   # SLURMtools.resolve rejects this layout with its own error
            ranks_per_node = min(n_ranks, cores_per_node // nomp) if cores_per_node >= nomp else n_ranks
            ranks_txt = f"resources_per_call={resources_per_call} / omp_threads_cpu={nomp} = {n_ranks} MPI ranks"
        else:
            n_ranks = resources_per_call
            ranks_per_node = min(resources_per_call, gpus_per_node) if gpus_per_node > 0 else resources_per_call
            ranks_txt = f"resources_per_call={resources_per_call}"

        resolved = _ResolvedControls(self, extraOptions, code_settings)
        n_tor_src = resolved.get('N_TOROIDAL', 1)
        n_tor_list = [int(v) for v in n_tor_src] if resolved.is_list(n_tor_src) else [int(n_tor_src)]

        # Single toroidal mode (e.g. linear single-ky runs): the only valid toroidal
        # split is one mode per process. This is a normal configuration, not a
        # misconfiguration, so set TOROIDALS_PER_PROC=1 and report it as info — the
        # remaining ranks parallelize the radial/velocity grid instead.
        if all(nt == 1 for nt in n_tor_list):
            if 'TOROIDALS_PER_PROC' in extraOptions:
                return extraOptions
            extraOptions = copy.deepcopy(extraOptions)
            extraOptions['TOROIDALS_PER_PROC'] = [1] * len(n_tor_list) if resolved.is_list(n_tor_src) else 1
            print(
                f"\t- [preprocess] N_TOROIDAL=1 (single toroidal mode); setting TOROIDALS_PER_PROC=1 "
                f"(resources_per_call={resources_per_call} rank(s) parallelize the radial/velocity grid)",
                typeMsg="i",
            )
            return extraOptions

        self._warn_self_connected_modes(resolved, n_tor_list)

        # CGYRO aborts at startup unless the MPI rank count (= resources_per_call) is a
        # multiple of the number of toroidal groups, N_TOROIDAL/TOROIDALS_PER_PROC (which
        # also requires TOROIDALS_PER_PROC to divide N_TOROIDAL). The smallest valid
        # TOROIDALS_PER_PROC maximizes toroidal parallelism (leftover rank multiplicity
        # goes to the velocity/radial decomposition); TOROIDALS_PER_PROC=N_TOROIDAL (a
        # single group) is always valid, so a solution always exists.
        def _is_valid(nt, tpp, ranks=n_ranks):
            return tpp > 0 and nt % tpp == 0 and ranks % (nt // tpp) == 0

        # CGYRO's process grid is n_proc = n_proc_1 x n_toroidal_procs, where
        # n_toroidal_procs = N_TOROIDAL/TOROIDALS_PER_PROC and n_proc_1 splits nc and nv
        # (cgyro_mpi_grid.F90:57,233-236). n_toroidal_procs is the size of the nonlinear
        # all-to-all communicator, and MPI_RANK_ORDER=2 (CGYRO's default) makes that
        # communicator rank-contiguous -- so holding it to one node's worth of ranks keeps
        # the all-to-all on-node. Picking the smallest valid TOROIDALS_PER_PROC maximizes it
        # instead. This only ever bites on multi-node radial calls: when the call fits in one
        # node, validity already forces n_toroidal_procs <= resources_per_call, so every valid
        # value is on-node and this rule reduces to taking the smallest valid one.
        def _grid_allows(nt, tpp, n_radial, ranks=n_ranks):
            # n_proc_1 > 1 additionally requires n_proc_1 to divide nv and nc, or CGYRO aborts
            # (cgyro_mpi_grid.F90:240-248). N_SPECIES is not resolved at this point, so require
            # n_proc_1 | N_ENERGY*N_XI, which is sufficient for nv = N_ENERGY*N_XI*N_SPECIES.
            n_proc_1 = ranks // (nt // tpp)
            if n_proc_1 == 1:
                return True
            if None in (n_energy, n_xi, n_theta, n_radial):
                return False
            return (n_energy * n_xi) % n_proc_1 == 0 and (n_radial * n_theta) % n_proc_1 == 0

        def _preferred_valid(nt, n_radial):
            valid = [tpp for tpp in range(1, nt + 1) if _is_valid(nt, tpp)]
            on_node = [tpp for tpp in valid
                       if (nt // tpp) <= ranks_per_node and _grid_allows(nt, tpp, n_radial)]
            if on_node:
                return min(on_node)
            return _off_node_fallback(nt, n_radial, valid)

        def _off_node_fallback(nt, n_radial, valid):
            # No on-node candidate. A candidate that passes the grid rule off-node is still fine;
            # one that fails it makes CGYRO abort at startup, so refuse instead of guessing
            grid_ok = [tpp for tpp in valid if _grid_allows(nt, tpp, n_radial)]
            if grid_ok:
                return min(grid_ok)
            if None in (n_energy, n_xi, n_theta, n_radial):
                print(
                    f"\t- [preprocess] N_ENERGY/N_XI/N_THETA/N_RADIAL not all resolved; cannot check the CGYRO "
                    f"grid rule for {ranks_txt}, keeping the smallest valid TOROIDALS_PER_PROC={min(valid)}",
                    typeMsg="w",
                )
                return min(valid)
            nv, nc = n_energy * n_xi, n_radial * n_theta
            fits = [r for r in range(1, 2 * n_ranks + 1)
                    if any(_is_valid(nt, t, ranks=r) and _grid_allows(nt, t, n_radial, ranks=r) for t in range(1, nt + 1))]
            raise ValueError(
                f"[MITIM] CGYRO cannot run N_TOROIDAL={nt} on {ranks_txt} (n_proc): for every TOROIDALS_PER_PROC, "
                f"n_proc_1 = n_proc / (N_TOROIDAL/TOROIDALS_PER_PROC) fails to divide nv = N_ENERGY*N_XI = {n_energy}*{n_xi} = {nv} "
                f"(N_SPECIES not counted) or nc = N_RADIAL*N_THETA = {n_radial}*{n_theta} = {nc}, and CGYRO aborts at startup. "
                f"Rank counts that fit this grid up to {2 * n_ranks}: {fits} "
                f"({'resources_per_call = ranks x omp_threads_cpu' if on_cpu else 'resources_per_call = GPUs = ranks'}). "
                f"Set TOROIDALS_PER_PROC in extraOptions to bypass this check."
            )

        def _as_int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        n_energy = _as_int(resolved.get('N_ENERGY'))
        n_xi = _as_int(resolved.get('N_XI'))
        n_theta = _as_int(resolved.get('N_THETA'))

        if any(nt <= 0 for nt in n_tor_list):
            print(
                f"\t- [preprocess] Invalid N_TOROIDAL={n_tor_list}; leaving TOROIDALS_PER_PROC as-is",
                typeMsg="w",
            )
            return extraOptions

        # Respect an explicit user-supplied TOROIDALS_PER_PROC in extraOptions
        # (user is overriding on purpose), but warn if CGYRO will reject the combination.
        if 'TOROIDALS_PER_PROC' in extraOptions:
            tpp_user = extraOptions['TOROIDALS_PER_PROC']
            tpp_user_list = [int(v) for v in tpp_user] if resolved.is_list(tpp_user) else [int(tpp_user)] * len(n_tor_list)
            for nt, tpp in zip(n_tor_list, tpp_user_list):
                if not _is_valid(nt, tpp):
                    print(
                        f"\t- [preprocess] User-supplied TOROIDALS_PER_PROC={tpp} is incompatible with "
                        f"N_TOROIDAL={nt} and {ranks_txt} "
                        f"(MPI ranks must be a multiple of N_TOROIDAL/TOROIDALS_PER_PROC); CGYRO may abort",
                        typeMsg="w",
                    )
            return extraOptions

        extraOptions = copy.deepcopy(extraOptions)
        tpp_list, tpp_src = resolved.as_list('TOROIDALS_PER_PROC', n=len(n_tor_list), cast=int, default=1)

        # Broadcast a scalar N_TOROIDAL to match a per-rho TOROIDALS_PER_PROC list.
        n_tor_list = _ResolvedControls.broadcast(n_tor_list, len(tpp_list))
        if len(tpp_list) != len(n_tor_list):
            print(
                f"\t- [preprocess] TOROIDALS_PER_PROC length {len(tpp_list)} mismatches "
                f"N_TOROIDAL length {len(n_tor_list)}; leaving as-is",
                typeMsg="w",
            )
            return extraOptions

        # N_RADIAL is per-rho (set by _apply_cgyro_preprocessing, which runs before this),
        # while N_TOROIDAL is usually a scalar from the model yaml. Broadcast the length-1
        # lists up to the longest so a per-rho N_RADIAL still reaches _grid_allows; if the
        # lengths are genuinely incompatible, drop N_RADIAL rather than guess (which makes
        # _grid_allows conservative and falls back to the smallest valid value).
        nr_src = resolved.get('N_RADIAL')
        nr_list = [_as_int(v) for v in nr_src] if resolved.is_list(nr_src) else [_as_int(nr_src)]
        n_rho = max(len(n_tor_list), len(nr_list))
        if len(n_tor_list) == 1:
            n_tor_list = n_tor_list * n_rho
            tpp_list = _ResolvedControls.broadcast(tpp_list, n_rho)
        nr_list = _ResolvedControls.broadcast(nr_list, n_rho)
        if not (len(n_tor_list) == len(tpp_list) == len(nr_list) == n_rho):
            n_tor_list = n_tor_list[:1] * n_rho if len(n_tor_list) == 1 else n_tor_list
            nr_list = [None] * len(n_tor_list)

        coerced = [_preferred_valid(nt, nr) for nt, nr in zip(n_tor_list, nr_list)]

        if coerced != tpp_list:
            n_groups = [nt // tpp for nt, tpp in zip(n_tor_list, coerced)]
            print(
                f"\t- [preprocess] TOROIDALS_PER_PROC {tpp_list} -> {coerced} "
                f"(MPI ranks {n_ranks}, {ranks_per_node} per node; toroidal groups "
                f"{n_groups}, which sets the size of the nonlinear all-to-all)",
                typeMsg="i",
            )

        # Preserve scalar shape if the caller originally gave a scalar and all coerced match.
        if not resolved.is_list(tpp_src) and len(set(coerced)) == 1:
            extraOptions['TOROIDALS_PER_PROC'] = coerced[0]
        else:
            extraOptions['TOROIDALS_PER_PROC'] = coerced

        return extraOptions

    def _warn_self_connected_modes(self, resolved, n_tor_list):
        """
        Nonlinear multi-mode runs: warn, per radius, when a simulated toroidal mode connects to itself after
        one poloidal turn (CGYROutils.self_connected_modes) on the BOX_SIZE / N_RADIAL / N_TOROIDAL that will
        be written (preprocessing result or values set by hand in extraOptions). Changes nothing.
        """
        box_list, _ = resolved.as_list('BOX_SIZE', cast=int)
        nr_list, _ = resolved.as_list('N_RADIAL', cast=int)
        if box_list is None or nr_list is None or int(resolved.get('NONLINEAR_FLAG', 0)) != 1:
            return
        n_rho = max(len(self.rhos), len(box_list), len(nr_list), len(n_tor_list))
        grids = zip(*(resolved.broadcast(v, n_rho) for v in (box_list, nr_list, n_tor_list)))
        for i, (box, nr, nt) in enumerate(grids):
            closed = CGYROutils.self_connected_modes(box, nr, nt)
            if not closed:
                continue
            where = f"rho={self.rhos[i]:.4f}" if len(self.rhos) == n_rho else f"radius #{i}"
            print(
                f"\t- [preprocess] {where}: BOX_SIZE={box} and N_RADIAL={nr} (N_RADIAL/BOX_SIZE={nr / box:g}) make toroidal "
                f"mode(s) n={closed} of N_TOROIDAL={nt} (ky = n*KY) connect to themselves after one poloidal turn (CGYRO "
                f"links radial index ir to ir +- n*BOX_SIZE modulo N_RADIAL at the ends of the field line), so the flux of "
                f"those modes can be set by the grid and not by the plasma. No simulated mode closes on itself once "
                f"N_RADIAL/BOX_SIZE >= N_TOROIDAL.",
                typeMsg="w",
            )

    def _enforce_print_step(self, extraOptions, code_settings=None):
        """
        Set PRINT_STEP so that DELTA_T * PRINT_STEP == 1.0 (PRINT_STEP integer).
        If the user explicitly sets PRINT_STEP in extraOptions, it is respected.
        """
        if 'PRINT_STEP' in extraOptions:
            return extraOptions

        resolved = _ResolvedControls(self, extraOptions, code_settings)
        dt_list, dt_src = resolved.as_list('DELTA_T')
        if dt_list is None:
            return extraOptions

        if any(dt <= 0 for dt in dt_list):
            print(
                f"\t- [preprocess] DELTA_T={dt_list} contains non-positive values; leaving PRINT_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        # PRINT_STEP * DELTA_T should equal 1.0 -> PRINT_STEP = round(1.0 / DELTA_T).
        print_steps = [max(1, int(round(1.0 / dt))) for dt in dt_list]

        extraOptions = copy.deepcopy(extraOptions)
        if not resolved.is_list(dt_src) and len(set(print_steps)) == 1:
            extraOptions['PRINT_STEP'] = print_steps[0]
        else:
            extraOptions['PRINT_STEP'] = print_steps

        print(
            f"\t- [preprocess] PRINT_STEP set to {extraOptions['PRINT_STEP']} "
            f"(DELTA_T={dt_src} -> DELTA_T*PRINT_STEP ~= 1.0)",
            typeMsg="i",
        )
        return extraOptions

    def _enforce_restart_step(self, extraOptions, code_settings=None):
        """
        Ensure CGYRO writes a restart by the end of the run: RESTART_STEP is coerced to the number of
        data outputs within MAX_TIME (see _restart_outputs / _coerced_restart_step), unless the
        controls-file / yaml value already divides it.
        If the user explicitly sets RESTART_STEP in extraOptions, it is respected.
        """
        if 'RESTART_STEP' in extraOptions:
            return extraOptions

        resolved = _ResolvedControls(self, extraOptions, code_settings)
        dt_list, dt_src = resolved.as_list('DELTA_T')
        ps_list, ps_src = resolved.as_list('PRINT_STEP')
        mt_list, mt_src = resolved.as_list('MAX_TIME')

        if dt_list is None or ps_list is None or mt_list is None:
            return extraOptions
        if any(v <= 0 for v in dt_list + ps_list + mt_list):
            print(
                f"\t- [preprocess] Non-positive DELTA_T/PRINT_STEP/MAX_TIME; leaving RESTART_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        n = max(len(dt_list), len(ps_list), len(mt_list))
        dt_list, ps_list, mt_list = (_ResolvedControls.broadcast(x, n) for x in (dt_list, ps_list, mt_list))
        if not (len(dt_list) == len(ps_list) == len(mt_list) == n):
            print(
                "\t- [preprocess] DELTA_T/PRINT_STEP/MAX_TIME length mismatch; leaving RESTART_STEP as-is",
                typeMsg="w",
            )
            return extraOptions

        n_outputs = [_restart_outputs(mt, dt, ps) for dt, ps, mt in zip(dt_list, ps_list, mt_list)]

        rs_list, rs_src = resolved.as_list('RESTART_STEP', n=n, cast=int, default=0)
        if rs_list is None:
            rs_list, rs_src = [0] * n, 0
        if len(rs_list) != n:
            rs_list = rs_list + [rs_list[-1]] * (n - len(rs_list)) if len(rs_list) < n else rs_list[:n]

        coerced = [_coerced_restart_step(rs, no) for rs, no in zip(rs_list, n_outputs)]

        extraOptions = copy.deepcopy(extraOptions)
        # Preserve scalar shape if inputs were all scalar and all coerced agree.
        scalars = not any(_ResolvedControls.is_list(x) for x in (dt_src, ps_src, mt_src, rs_src))
        if scalars and len(set(coerced)) == 1:
            extraOptions['RESTART_STEP'] = coerced[0]
        else:
            extraOptions['RESTART_STEP'] = coerced

        print(
            f"\t- [preprocess] RESTART_STEP set to {extraOptions['RESTART_STEP']} "
            f"(<= n_outputs = floor(nint(MAX_TIME/DELTA_T)/PRINT_STEP) = {n_outputs}; "
            f"restart fires when mod(i_time, RESTART_STEP*PRINT_STEP) == 0)",
            typeMsg="i",
        )
        return extraOptions

    @staticmethod
    def _restart_step_after_trim(text, remaining):
        """
        Re-derive RESTART_STEP for the shortened run of a rescued radius (rescue_spec
        'after_trim' hook). _enforce_restart_step sized RESTART_STEP from the FULL
        MAX_TIME, so once the rescue trims MAX_TIME to what is left the trigger
        mod(i_time, RESTART_STEP*PRINT_STEP) == 0 with i_time = 1..nint(MAX_TIME/DELTA_T)
        can no longer fire: the continuation writes no checkpoint and mitim_kill_cgyro's
        watchdog (which waits on out.cgyro.tag) blocks on that radius.

        Same coercion as _enforce_restart_step, on n_outputs of the remaining window.
        Returns (text, note appended to the [rescue] log line); unchanged when the keys
        are missing from the staged input.
        """
        import re

        def _value(key):
            m = re.search(rf"^{key}\s*=\s*(\S+)", text, flags=re.M)
            return m, (float(m.group(1)) if m else None)

        m_rs, rs = _value('RESTART_STEP')
        _, dt = _value('DELTA_T')
        _, ps = _value('PRINT_STEP')
        if m_rs is None or not dt or not ps:
            return text, ""

        rs = int(rs)
        new_rs = _coerced_restart_step(rs, _restart_outputs(remaining, dt, ps))
        if new_rs == rs:
            return text, ""
        return text[:m_rs.start(1)] + f"{new_rs}" + text[m_rs.end(1):], f", RESTART_STEP {rs} -> {new_rs}"

    def _apply_cgyro_preprocessing(self, extraOptions):
        """
        Compute BOX_SIZE and N_RADIAL per rho from the caller-provided
        ky_min plus Q, S, RMIN from self.inputs_files[rho], and inject them
        (along with KY=ky_min) into a copy of extraOptions as per-rho arrays.
        fft_friendly is off unless given (the PORTALS namelist template sets it to true).
        """

        allowed_keys = {'ky_min', 'L_x', 'N_radial', 'min_box_size', 'fft_friendly', 'fft_tol'}
        opts = dict(self._preprocess_options) if self._preprocess_options else {}
        unknown = set(opts) - allowed_keys
        if unknown:
            raise ValueError(
                f"[MITIM] Unknown preprocess_options keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed_keys)}"
            )
        ky_min_opt    = opts.get('ky_min', 0.1)
        L_x           = opts.get('L_x', 90.0)
        N_radial      = opts.get('N_radial', 256)
        min_box_size  = opts.get('min_box_size', 100)
        fft_friendly  = opts.get('fft_friendly', False)
        fft_tol       = opts.get('fft_tol', 0.06)

        extraOptions = copy.deepcopy(extraOptions)

        for conflict_key in ('KY', 'BOX_SIZE', 'N_RADIAL'):
            if conflict_key in extraOptions:
                print(
                    f"\t- [preprocess] {conflict_key} was set in extraOptions; "
                    f"it will be overwritten by the preprocessing result",
                    typeMsg="w",
                )

        box_sizes = []
        n_radials = []
        ky_mins   = []

        print("\t- [preprocess] Computing BOX_SIZE and N_RADIAL per rho:")
        for i, rho in enumerate(self.rhos):
            input_rho = self.inputs_files[rho]
            q     = float(input_rho.plasma['Q'])
            shear = float(input_rho.plasma['S'])
            rmin  = float(input_rho.plasma['RMIN'])

            if isinstance(ky_min_opt, (list, np.ndarray)):
                ky_min = float(ky_min_opt[i])
            else:
                ky_min = float(ky_min_opt)

            box_size, n_radial_val = CGYROutils.compute_box_and_nradial(
                q=q,
                shear=shear,
                rmin=rmin,
                ky_min=ky_min,
                L_x=L_x,
                N_radial=N_radial,
                min_box_size=min_box_size,
                fft_friendly=fft_friendly,
                fft_tol=fft_tol,
            )

            print(
                f"\t\t* rho={rho:.4f}: q={q:.3f} s={shear:.3f} r/a={rmin:.3f} "
                f"KY={ky_min:.4f} -> BOX_SIZE={box_size} N_RADIAL={n_radial_val}",
                typeMsg="i",
            )

            box_sizes.append(box_size)
            n_radials.append(n_radial_val)
            ky_mins.append(ky_min)

        extraOptions['KY']       = ky_mins
        extraOptions['BOX_SIZE'] = box_sizes
        extraOptions['N_RADIAL'] = n_radials
        return extraOptions

    # Re-defined to make specific arguments explicit
    def read(
        self,
        tmin = 0.0,
        tmin_is_rel = True,
        minimal = False,
        last_tmin_for_linear = True,
        input_gacode = None,  # State to take the physical units from when CGYRO did not write them (PROFILE_MODEL=1)
        **kwargs
    ):

        super().read(
            tmin = tmin,
            tmin_is_rel = tmin_is_rel,
            minimal = minimal,
            last_tmin_for_linear = last_tmin_for_linear,
            input_gacode = input_gacode,
            state_for_units = input_gacode,
            **kwargs)

    def read_linear_scan(
        self,
        folder=None,
        preffix="scan",
        store_as_label=None,
        irho = 0,
        **kwargs
    ):
        '''
        Useful utility for when a folder contains subfolders like... scan0, scan1, scan2... with different ky
        '''
        
        if folder is None:
            folder = self.FolderGACODE
        
        main_label = kwargs.get('label', 'run1')
        del kwargs['label']
        
        # Get all folders inside "folder" that start with "preffix"
        subfolders = [subfolder for subfolder in Path(folder).glob(f"*{preffix}*") if subfolder.is_dir()]

        # ----------------------------------------------------------
        # Store in resutls
        # ----------------------------------------------------------

        # Store results in the form of {main_label}_KY_{subfolder}
        
        labels_in_results = []
        if len(subfolders) == 0:
            print(f"No subfolders found in {folder} with preffix {preffix}. Reading the folder directly.")
            labels_in_results.append(f'{main_label}_KY_scan0')
            self.read(label=labels_in_results[-1], folder=folder, **kwargs)
        else:   
            for subfolder in subfolders:
                labels_in_results.append(f'{main_label}_KY_{subfolder.name}')
                self.read(label=labels_in_results[-1], folder=subfolder, **kwargs)        

        # ----------------------------------------------------------
        # Make it a linear scan for the main label
        # ----------------------------------------------------------
        
        # Store special linear scan class as {main_label}
        
        labelsD = []
        for label in labels_in_results:
            parts = label.split('_')
            if len(parts) >= 3 and parts[-2] == "KY":
                # Extract the base name (scan1) and middle value (0.3/0.4)
                base_name = '_'.join(parts[0:-2])               
                labelsD.append(label)

        if store_as_label is not None:
            main_label = store_as_label

        self.results[main_label] = CGYROutils.CGYROlinear_scan(labelsD, self.results, irho=irho)

    # Redefined to remove potential large objects
    def save_pickle(self, file, **kwargs):

        class_to_store = super().prepare_for_save()

        # cgyrodata carries _thread.lock through its internal HDF/file handles,
        # which breaks copy.deepcopy. Temporarily detach each cgyrodata from the
        # live object, deepcopy the rest, then restore them so the in-memory
        # object is unchanged after save.
        stashed = []
        for key in class_to_store.results:
            for irho in range(len(class_to_store.results[key]['output'])):
                out = class_to_store.results[key]['output'][irho]
                if 'cgyrodata' in out.__dict__:
                    stashed.append((out, out.cgyrodata))
                    out.cgyrodata = None
        try:
            class_to_store = copy.deepcopy(class_to_store)
        finally:
            for out, data in stashed:
                out.cgyrodata = data

        super().save_pickle(file, class_to_store = class_to_store, **kwargs)
                
                
    def plot(
        self,
        labels=[""],
        fn=None,
        include_2D=True,
        common_colorbar=True):
        
        # If it has radii, we need to correct the labels
        labels = self._correct_rhos_labels(labels)
    
        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook
            self.fn = FigureNotebook("CGYRO Notebook", geometry="1600x1000")
        else:
            self.fn = fn

        fig = self.fn.add_figure(label="Fluxes (time)")
        axsFluxes_t = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Fluxes (ky)")
        axsFluxes_ky = fig.subplot_mosaic(
            """
            ACE
            BDF
            """
        )
        fig = self.fn.add_figure(label="Intensities (time)")
        axsIntensities = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (ky)")
        axsIntensities_ky = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Intensities (kx)")
        axsIntensities_kx = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )
        fig = self.fn.add_figure(label="Cross-phases (ky)")
        axsCrossPhases = fig.subplot_mosaic(
            """
            ACEG
            BDFH
            """
        )
        fig = self.fn.add_figure(label="Turbulence (linear)")
        axsTurbulence = fig.subplot_mosaic(
            """
            AC
            BD
            """
        )

        # One "Averaging" figure per case: window selection diagnostics of the primary fluxes
        for i, label in enumerate(labels):
            if hasattr(self.results[label], 'averaging'):
                fig = self.fn.add_figure(label=f"Averaging, {label}")
                self.results[label].averaging.plot(fig=fig, color=GRAPHICStools.listColors()[i % len(GRAPHICStools.listColors())], label_plot=label)

        create_ballooning = False
        for label in labels:
            if 'phi_ballooning' in self.results[label].__dict__:
                create_ballooning = True
            
        if create_ballooning:

            fig = self.fn.add_figure(label="Ballooning")
            axsBallooning = fig.subplot_mosaic(
                """
                135
                246
                """
                )
        else:
            axsBallooning = None
        
        
        if include_2D:
            axs2D = []
            for i in range(len(labels)):
                fig = self.fn.add_figure(label="Turbulence (2D), " + labels[i])
                
                mosaic = _2D_mosaic(4) # Plot 4 times by default
                
                axs2D.append(fig.subplot_mosaic(mosaic))
        
        fig = self.fn.add_figure(label="Timing")
        axsTiming = fig.subplot_mosaic(
            """
            AC
            BC
            """
        )

        fig = self.fn.add_figure(label="Inputs")
        axsInputs = fig.subplot_mosaic(
            """
            A
            B
            """
        )

        
        colors = GRAPHICStools.listColors()

        # Safety net: if one sub-plot method raises (e.g. a missing optional
        # CGYRO output file that the per-panel guards didn't cover), we do
        # NOT want it to abort the whole notebook build. Wrap each call in a
        # try/except that logs a warning and continues.
        def _safe_plot(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as _e:
                print(f"\t- {fn.__name__} failed ({_e}); skipping this figure and continuing", typeMsg='w')
                return None

        colorbars_all = []  # Store all colorbars for later use
        for j in range(len(labels)):

            _safe_plot(self.plot_fluxes,
                axs=axsFluxes_t,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            _safe_plot(self.plot_fluxes_ky,
                axs=axsFluxes_ky,
                label=labels[j],
                c=colors[j],
                plotLegend=j == len(labels) - 1,
            )
            _safe_plot(self.plot_intensities_ky,
                axs=axsIntensities_ky,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,
            )
            _safe_plot(self.plot_intensities,
                axs=axsIntensities,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            _safe_plot(self.plot_intensities_kx,
                axs=axsIntensities_kx,
                label=labels[j],
                c=colors[j],
                addText=j == len(labels) - 1,  # Add text only for the last label
            )
            _safe_plot(self.plot_turbulence,
                axs=axsTurbulence,
                label=labels[j],
                c=colors[j],
            )
            _safe_plot(self.plot_cross_phases,
                axs=axsCrossPhases,
                label=labels[j],
                c=colors[j],
            )
            if create_ballooning:
                _safe_plot(self.plot_ballooning,
                    axs=axsBallooning,
                    label=labels[j],
                    c=colors[j],
                )

            if include_2D:

                colorbars = _safe_plot(self.plot_2D,
                    axs=axs2D[j],
                    label=labels[j],
                )

                colorbars_all.append(colorbars)

            _safe_plot(self.plot_timing,
                axs=axsTiming,
                label=labels[j],
                c=colors[j],
            )

            _safe_plot(self.plot_inputs,
                ax=axsInputs["A"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
                normalization_label= labels[0],  # Normalize to the first label
                only_plot_differences=len(labels) > 1,  # Only plot differences if there are multiple labels
            )

            _safe_plot(self.plot_inputs,
                ax=axsInputs["B"],
                label=labels[j],
                c=colors[j],
                ms= 10-j*0.5,  # Decrease marker size for each label
            )
            
        axsInputs["A"].axhline(
            1.0,
            color="k",
            ls="--",
            lw=2.0
        )
        
        GRAPHICStools.adjust_subplots(axs=axsInputs, vertical=0.4, horizontal=0.3)
        
        # Modify the colorbars to have a common range. Skip label indices where
        # plot_2D was a no-op — e.g. when bin.cgyro.kxky_phi / kxky_n / kxky_e
        # aren't available (MOMENT_PRINT_FLAG / FIELD_PRINT_FLAG off). _safe_plot
        # returns None in that case, so colorbars_all can carry None entries
        # which would crash the common-range aggregation below.
        valid_cbs = [cbs for cbs in colorbars_all if cbs is not None]
        if include_2D and common_colorbar and len(valid_cbs) > 0:
            for var in ['phi', 'n', 'e']:
                min_val = np.inf
                max_val = -np.inf
                for cbs in valid_cbs:
                    cb = cbs[0][var]
                    vals = cb.mappable.get_clim()
                    min_val = min(min_val, vals[0])
                    max_val = max(max_val, vals[1])

                for cbs in valid_cbs:
                    for it in range(len(cbs)):
                        cb = cbs[it][var]
                        cb.mappable.set_clim(min_val, max_val)
                        cb.update_ticks()
                        #cb.set_label(f"{var} (common range)")

        # Back to the original labels before _correct_rhos_labels
        self.results = self.results_all

    def plot_timing(self, axs=None, label="", c="b"):
        """
        Characterize the computational cost of the run from out.cgyro.timing
        (wall-clock seconds spent in each code section, one row per data output):
            A: wall time per data output (TOTAL column)
            B: cumulative wall time (setup time included as offset)
            C: share of the total run time spent in each code section
        """
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(15, 8))
            axs = fig.subplot_mosaic(
                """
                AC
                BC
                """
            )

        data = self.results[label]
        if "timing" not in data.__dict__:
            print(f"\t- No timing information for {label}; skipping timing plot", typeMsg="w")
            return

        steps = np.arange(1, len(data.timing_total) + 1)
        setup_time = sum(getattr(data, "timing_setup", {}).values())
        total_time = setup_time + data.timing_total.sum()

        # A: cost of each data output
        ax = axs["A"]
        ax.plot(steps, data.timing_total, "-o", c=c, lw=1.0, markersize=3, label=label)
        ax.axhline(data.timing_total.mean(), c=c, ls="--", lw=0.5)
        ax.set_xlabel("Data output #")
        ax.set_ylabel("Wall time per output (s)")
        ax.set_title("Cost per data output (dashed: mean)")
        # Pin bottom at 0 but keep the top growing to fit every overlaid case: set_ylim
        # disables y-autoscale, so a plain set_ylim(bottom=0) on the first case would
        # freeze the top to that case's range and clip the others.
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], float(np.nanmax(data.timing_total)) * 1.05))
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc="best", prop={"size": 8})

        # B: cumulative cost
        ax = axs["B"]
        cumulative = (setup_time + np.cumsum(data.timing_total)) / 60.0
        ax.plot(
            steps,
            cumulative,
            "-o", c=c, lw=1.0, markersize=3,
            label=f"{label} (setup {setup_time:.1f}s, total {total_time/60.0:.1f}min)",
        )
        ax.set_xlabel("Data output #")
        ax.set_ylabel("Cumulative wall time (min)")
        ax.set_title("Cumulative cost (setup included)")
        # Same as A: expand the top across cases (cumulative is monotonic, so [-1] is the max)
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], float(cumulative[-1]) * 1.05))
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc="best", prop={"size": 8})

        # C: where the time goes, by code section
        ax = axs["C"]
        share = 100.0 * data.timing.sum(axis=0) / data.timing.sum()
        y = np.arange(len(data.timing_names))
        ax.plot(share, y, "o", c=c, markersize=8, label=label)
        ax.set_yticks(y)
        ax.set_yticklabels(data.timing_names)
        ax.invert_yaxis()
        ax.set_xlabel("Share of run time (%)")
        ax.set_title("Time per code section")
        ax.set_xlim(left=0)
        GRAPHICStools.addDenseAxis(ax)
        ax.legend(loc="best", prop={"size": 8})

    def plot_inputs(self, ax = None, label="", c="b", ms = 10, normalization_label=None, only_plot_differences=False):

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(18, 9))

        rel_tol = 1e-2

        legadded = False
        for i, ikey in enumerate(self.results[label].params1D):
            
            z = self.results[label].params1D[ikey]
            
            if normalization_label is not None:
                z0 = self.results[normalization_label].params1D[ikey]
                zp = z/z0 if z0 != 0 else 0
                label_plot = f"{label} / {normalization_label}"
            else:
                label_plot = label
                zp = z

            if (not only_plot_differences) or (not np.isclose(z, z0, rtol=rel_tol)):
                ax.plot(ikey,zp,'o',markersize=ms,color=c,label=label_plot if not legadded else '')
                legadded = True

        if normalization_label is not None:
            if only_plot_differences:
                ylabel = f"Parameters (DIFFERENT by {rel_tol*100:.2f}%) relative to {normalization_label}"
            else:
                ylabel = f"Parameters relative to {normalization_label}"
        else:
            ylabel = "Parameters"

        ax.set_xlabel("Parameter")
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel(ylabel)
        GRAPHICStools.addDenseAxis(ax)
        if legadded:
            ax.legend(loc='best')

    def plot_intensities(self, axs = None, label= "cgyro1", c="b", addText=True):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
            
        ax = axs["A"]
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
        ax.plot(self.results[label].t, self.results[label].phi_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
  
        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta \\phi/\\phi_0$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity fluctuations')
        ax.legend(loc='best', prop={'size': 8},)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].t, self.results[label].apar_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}, $A_\\parallel$")
            ax.plot(self.results[label].t, self.results[label].bpar_rms_sumnr_sumn*100.0, '--', c=c, lw=2, label=f"{label}, $B_\\parallel$")
            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity fluctuations')
        

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["C"]
        try:
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].ne_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_e/n_{e,0}/n_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Density intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["D"]
        try:
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].Te_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_e/T_{e,0}/T_{e0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron Temperature intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))




        ax = axs["E"]
        try:
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].ni_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Density intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



        ax = axs["F"]
        try:
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn*100.0, '-', c=c, lw=2, label=f"{label}")
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_n0*100.0, '-.', c=c, lw=0.5, label=f"{label}, $n=0$")
            ax.plot(self.results[label].t, self.results[label].Ti_rms_sumnr_sumn1*100.0, '--', c=c, lw=0.5, label=f"{label}, $n>0$")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/T_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion Temperature intensity fluctuations')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].t, self.results[label].ni_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta n_i/n_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Density intensity fluctuations')


        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].t, self.results[label].Ti_all_rms_sumnr_sumn[ion]*100.0, ls[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]}")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$t$ ($a/c_s$)"); #ax.set_xlim(left=0.0)
        ax.set_ylabel("$\\delta T_i/T_{i,0}/n_{i0}$ (%)")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) Temperature intensity fluctuations')


        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_ky(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].ky, self.results[label].phi_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
        ax.fill_between(self.results[label].ky, self.results[label].phi_rms_sumnr_mean-self.results[label].phi_rms_sumnr_std, self.results[label].phi_rms_sumnr_mean+self.results[label].phi_rms_sumnr_std, color=c, alpha=0.2)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta\phi/\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs. $k_\\theta\\rho_s$')
        ax.legend(loc='best', prop={'size': 8},)
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["B"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].ky, self.results[label].apar_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].apar_rms_sumnr_mean-self.results[label].apar_rms_sumnr_std, self.results[label].apar_rms_sumnr_mean+self.results[label].apar_rms_sumnr_std, color=c, alpha=0.2)
            ax.plot(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean, '--', markersize=5, color=c, label=label+', $B_\\parallel$ (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].bpar_rms_sumnr_mean-self.results[label].bpar_rms_sumnr_std, self.results[label].bpar_rms_sumnr_mean+self.results[label].bpar_rms_sumnr_std, color=c, alpha=0.2)

            ax.legend(loc='best', prop={'size': 8},)

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel(r"$\delta F_\parallel/F_{\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs. $k_\\theta\\rho_s$')
        
        ax.axhline(0.0, color='k', ls='--', lw=1)

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["C"]
        try:
            ax.plot(self.results[label].ky, self.results[label].ne_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].ne_rms_sumnr_mean-self.results[label].ne_rms_sumnr_std, self.results[label].ne_rms_sumnr_mean+self.results[label].ne_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Electron temperature intensity
        ax = axs["D"]
        try:
            ax.plot(self.results[label].ky, self.results[label].Te_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].Te_rms_sumnr_mean-self.results[label].Te_rms_sumnr_std, self.results[label].Te_rms_sumnr_mean+self.results[label].Te_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["E"]
        try:
            ax.plot(self.results[label].ky, self.results[label].ni_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].ni_rms_sumnr_mean-self.results[label].ni_rms_sumnr_std, self.results[label].ni_rms_sumnr_mean+self.results[label].ni_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta n_i/n_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Ion temperature intensity
        ax = axs["F"]
        try:
            ax.plot(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean, '-o', markersize=5, color=c, label=label+' (mean)')
            ax.fill_between(self.results[label].ky, self.results[label].Ti_rms_sumnr_mean-self.results[label].Ti_rms_sumnr_std, self.results[label].Ti_rms_sumnr_mean+self.results[label].Ti_rms_sumnr_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ion temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n_r}|\delta T_i/T_{i,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
        # Ion particle intensity
        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].ni_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta n_i/n_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) particle intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)


        # Ion temperature intensity
        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].Ti_all_rms_sumnr_mean[ion], ls[ion]+'o', markersize=5, color=c, label=f"{label}, {self.results[label].all_names[ion]} (mean)")
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\delta T_i/T_{i,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Ions (all) temperature intensity vs. $k_\\theta\\rho_s$')
        ax.axhline(0.0, color='k', ls='--', lw=1)
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_intensities_kx(self, axs=None, label="", c="b", addText=True):
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                AC
                BD
                """
            )

        # Potential intensity
        ax = axs["A"]
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
        ax.plot(self.results[label].kx, self.results[label].phi_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta \\phi/\\phi_0$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Potential intensity vs kx')
        ax.legend(loc='best', prop={'size': 8},)
        ax.set_yscale('log')
        
        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta\phi/\phi_0|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # EM potential intensity
        ax = axs["C"]
        if 'apar' in self.results[label].__dict__:
            ax.plot(self.results[label].kx, self.results[label].apar_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+', $A_\\parallel$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].bpar_rms_sumn_mean, '--', markersize=1.0, lw=1.0, color=c, label=label+', $B_\\parallel$ (mean)')

            ax.legend(loc='best', prop={'size': 8},)


        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta F_\\parallel/F_{\\parallel,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('EM potential intensity vs kx')
        ax.set_yscale('log')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta F_\parallel/F_{\parallel,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        # Electron particle intensity
        ax = axs["B"]
        try:
            ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
            ax.plot(self.results[label].kx, self.results[label].ne_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].ne_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')
            ax.legend(loc='best', prop={'size': 8},)
            ax.set_yscale('log')
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta n_e/n_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron particle intensity vs kx')

        # Add mathematical definitions text
        if addText:
            ax.text(0.02, 0.95,
                    r'$\sqrt{\langle\sum_{n}|\delta n_e/n_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        # Electron temperature intensity
        ax = axs["D"]
        try:
            ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn_mean, '-o', markersize=1.0, lw=1.0, color=c, label=label+' (mean)')
            ax.plot(self.results[label].kx, self.results[label].Te_rms_n0_mean, '-.', markersize=0.5, lw=0.5, color=c, label=label+', $n=0$ (mean)')
            ax.plot(self.results[label].kx, self.results[label].Te_rms_sumn1_mean, '--', markersize=0.5, lw=0.5, color=c, label=label+', $n>0$ (mean)')
            ax.legend(loc='best', prop={'size': 8},)
            ax.set_yscale('log')
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{x}$")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$")
        GRAPHICStools.addDenseAxis(ax)
        ax.set_title('Electron temperature intensity vs kx')
        
        if addText:
            ax.text(0.02, 0.95, 
                    r'$\sqrt{\langle\sum_{n}|\delta T_e/T_{e,0}|^2\rangle}$',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)


    def plot_cross_phases(self, axs = None, label= "cgyro1", c="b"):

        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                ACEG
                BDFH
                """
            )
            
        ls = GRAPHICStools.listLS()
        m = GRAPHICStools.listmarkers()
            
        ax = axs["A"]
        try:
            ax.plot(self.results[label].ky, self.results[label].neTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].neTe_kx0_mean-self.results[label].neTe_kx0_std, self.results[label].neTe_kx0_mean+self.results[label].neTe_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n + kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_e-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_e-T_e$ cross-phase ($k_x=0$)')


        ax = axs["B"]
        try:
            ax.plot(self.results[label].ky, self.results[label].niTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].niTi_kx0_mean-self.results[label].niTi_kx0_std, self.results[label].niTi_kx0_mean+self.results[label].niTi_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n + kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$n_i-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$n_i-T_i$ cross-phase ($k_x=0$)')

        ax = axs["C"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phine_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phine_kx0_mean-self.results[label].phine_kx0_std, self.results[label].phine_kx0_mean+self.results[label].phine_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_e$ cross-phase ($k_x=0$)')

        ax = axs["D"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phini_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phini_kx0_mean-self.results[label].phini_kx0_std, self.results[label].phini_kx0_mean+self.results[label].phini_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ cross-phase ($k_x=0$)')


        ax = axs["E"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phiTe_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phiTe_kx0_mean-self.results[label].phiTe_kx0_std, self.results[label].phiTe_kx0_mean+self.results[label].phiTe_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_e$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_e$ cross-phase ($k_x=0$)')


        ax = axs["F"]
        try:
            ax.plot(self.results[label].ky, self.results[label].phiTi_kx0_mean, '-o', c=c, lw=2, label=f"{label} (mean)")
            ax.fill_between(self.results[label].ky, self.results[label].phiTi_kx0_mean-self.results[label].phiTi_kx0_std, self.results[label].phiTi_kx0_mean+self.results[label].phiTi_kx0_std, color=c, alpha=0.2)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ cross-phase ($k_x=0$)')


        ax = axs["G"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].phiTi_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_e (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-T_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-T_i$ (all) cross-phase ($k_x=0$)')


        ax = axs["H"]
        try:
            for ion in self.results[label].ions_flags:
                ax.plot(self.results[label].ky, self.results[label].phini_all_kx0_mean[ion], ls[ion]+m[ion], c=c, lw=1, label=f"{label}, {self.results[label].all_names[ion]} (mean)", markersize=4)
            ax.legend(loc='best', prop={'size': 8},)
        except AttributeError:
            _annotate_missing(ax, "needs bin.cgyro.kxky_n (MOMENT_PRINT_FLAG=1)")

        ax.set_xlabel("$k_{\\theta} \\rho_s$")
        ax.set_ylabel("$\\phi-n_i$ cross-phase (degrees)"); ax.set_ylim([-180, 180])
        GRAPHICStools.addDenseAxis(ax)
        ax.axhline(0.0, color='k', ls='--', lw=1)
        ax.set_title('$\\phi-n_i$ (all) cross-phase ($k_x=0$)')
        
        
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_ballooning(self, time = None, label="cgyro1", c="b", axs=None):
        
        if axs is None:
            plt.ion()
            fig = plt.figure(figsize=(18, 9))

            axs = fig.subplot_mosaic(
                """
                135
                246
                """
            )

        if time is None:
            time = np.min([self.results[label].tmin, self.results[label].tmax_fluct])
        
        it = np.argmin(np.abs(self.results[label].t - time))

        colorsC, _ = GRAPHICStools.colorTableFade(
            len(self.results[label].ky),
            startcolor=c,
            endcolor=c,
            alphalims=[1.0, 0.4],
        )

        ax = axs['1']
        for ky in range(len(self.results[label].ky)):
            for var, axsT in zip(
                ["phi_ballooning", "apar_ballooning", "bpar_ballooning"],
                [[axs['1'], axs['2']], [axs['3'], axs['4']], [axs['5'], axs['6']]],
            ):

                f = self.results[label].__dict__[var][:, it]
                y1 = np.real(f)
                y2 = np.imag(f)
                x = self.results[label].theta_ballooning / np.pi

                # Normalize
                y1_max = np.max(np.abs(y1))
                y2_max = np.max(np.abs(y2))
                y1 /= y1_max
                y2 /= y2_max

                ax = axsT[0]
                ax.plot(
                    x,
                    y1,
                    color=colorsC[ky],
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y1_max:.2e})",
                )
                ax = axsT[1]
                ax.plot(
                    x, 
                    y2, 
                    color=colorsC[ky], 
                    ls="-",
                    label=f"$k_{{\\theta}}\\rho_s={np.abs( self.results[label].ky[ky]):.2f}$ (max {y2_max:.2e})",
                )


        ax = axs['1']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta\\phi$)")
        ax.set_title("$\\delta\\phi$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([-2 * np.pi, 2 * np.pi])

        ax = axs['3']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta A\\parallel$)")
        ax.set_title("$\\delta A\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['5']
        ax.set_xlabel("$\\theta/\\pi$ (normalized to maximum)")
        ax.set_ylabel("Re($\\delta B\\parallel$)")
        ax.set_title("$\\delta B\\parallel$")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['2']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta\\phi$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['4']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta A\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['6']
        ax.set_xlabel("$\\theta/\\pi$")
        ax.set_ylabel("Im($\\delta B\\parallel$)")
        ax.legend(loc="best", prop={"size": 8})
        GRAPHICStools.addDenseAxis(ax)


        for ax in [axs['1'], axs['3'], axs['5'], axs['2'], axs['4'], axs['6']]:
            ax.axvline(x=0, lw=0.5, ls="--", c="k")
            ax.axhline(y=0, lw=0.5, ls="--", c="k")
            
            
        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.3, horizontal=0.3)

    def plot_2D(self, label="cgyro1", axs=None, times = None):

        # plot_2D needs kxky_phi (always), kxky_n (MOMENT_PRINT_FLAG=1) and
        # kxky_e (MOMENT_PRINT_FLAG=1). If any of the underlying fluctuation
        # arrays is missing (typically because the user disabled the print
        # flags to save disk / retrieval time), skip cleanly instead of
        # aborting the whole plot chain.
        _res = self.results.get(label) if hasattr(self, 'results') else None
        if _res is None or not all(hasattr(_res, _a) for _a in ('phi', 'ne', 'Te')):
            print("\t- plot_2D skipped: needs phi/ne/Te (requires bin.cgyro.kxky_phi + kxky_n + kxky_e; enable MOMENT_PRINT_FLAG=1 / FIELD_PRINT_FLAG=1)", typeMsg='w')
            return

        if times is None:
            times = []
            
            number_times = len(axs)//3 if axs is not None else 4

            try:
                times = [self.results[label].t[-1-i*10] for i in range(number_times)]
            except IndexError:
                 times = [self.results[label].t[-1-i*1] for i in range(number_times)]

        if axs is None:

            mosaic = _2D_mosaic(len(times))

            plt.ion()
            fig = plt.figure(figsize=(18, 9))
            axs = fig.subplot_mosaic(mosaic)

        # Pre-calculate global min/max for each field type across all times
        phi_values = []
        n_values = []
        e_values = []
        
        for time in times:
            it = np.argmin(np.abs(self.results[label].t - time))
            
            # Get phi values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)
            phi_values.append(fp)
            
            # Get n values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)
            n_values.append(fp)
            
            # Get e values
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)
            e_values.append(fp)
        
        # Calculate global ranges
        phi_max = np.max([np.max(np.abs(fp)) for fp in phi_values])
        phi_min, phi_max = -phi_max, +phi_max
        
        n_max = np.max([np.max(np.abs(fp)) for fp in n_values])
        n_min, n_max = -n_max, +n_max
        
        e_max = np.max([np.max(np.abs(fp)) for fp in e_values])
        e_min, e_max = -e_max, +e_max

        colorbars = []  # Store colorbar references
        # Now plot with consistent colorbar ranges
        for time_i, time in enumerate(times):
            
            print(f"\t- Plotting 2D turbulence for {label} at time {time}")
            
            it = np.argmin(np.abs(self.results[label].t - time))
            
            cfig = axs[str(time_i+1)].get_figure()
            
            # Phi plot
            ax = axs[str(time_i+1)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_phi', it = it)

            cs1 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(phi_min,phi_max,(phi_max-phi_min)/256),cmap=plt.get_cmap('jet'))
            cphi = cfig.colorbar(cs1, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta\\phi/\\phi_0$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # N plot
            ax = axs[str(time_i+1+len(times))]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_n',species = self.results[label].electron_flag, it = it)

            cs2 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(n_min,n_max,(n_max-n_min)/256),cmap=plt.get_cmap('jet'))
            cn = cfig.colorbar(cs2, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta n_e/n_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')

            # E plot
            ax = axs[str(time_i+1+len(times)*2)]
            xp, yp, fp = self._to_real_space(label=label, variable = 'kxky_e',species = self.results[label].electron_flag, it = it)

            cs3 = ax.contourf(xp,yp,np.transpose(fp),levels=np.arange(e_min,e_max,(e_max-e_min)/256),cmap=plt.get_cmap('jet'))
            ce = cfig.colorbar(cs3, ax=ax)

            ax.set_xlabel("$x/\\rho_s$")
            ax.set_ylabel("$y/\\rho_s$")
            ax.set_title(f"$\\delta E_e/E_{{e,0}}$ (t={self.results[label].t[it]} $a/c_s$)")
            ax.set_aspect('equal')
            
            # Store the colorbar objects with their associated contour plots
            colorbars.append({
                'phi': cphi,
                'n': cn,
                'e': ce
            })

        GRAPHICStools.adjust_subplots(axs=axs, vertical=0.4, horizontal=0.3)

        return colorbars
        
    def _to_real_space(self, variable = 'kxky_phi', species = None, label="cgyro1", theta_plot = 0, it = -1):
        
        # from pygacode
        def maptoreal_fft(nr,nn,nx,ny,c):

            d = np.zeros([nx,nn],dtype=complex)
            for i in range(nr):
                p = i-nr//2
                if -p < 0:
                    k = -p+nx
                else:
                    k = -p
                d[k,0:nn] = np.conj(c[i,0:nn])
            f = np.fft.irfft2(d,s=[nx,ny],norm='forward')*0.5

            # Correct for half-sum
            f = 2*f

            return f

        # Real space
        nr = self.results[label].cgyrodata.n_radial
        nn = self.results[label].cgyrodata.n_n
        craw = self.results[label].cgyrodata.__dict__[variable]
        
        itheta = np.argmin(np.abs(self.results[label].theta_stored-theta_plot))
        if species is None:
            c = craw[:,itheta,:,it]
        else:
            c = craw[:,itheta,species,:,it]

        nx = self.results[label].cgyrodata.__dict__[variable].shape[0]
        ny = nx
        
        # Arrays
        x = np.arange(nx)*2*np.pi/nx
        y = np.arange(ny)*2*np.pi/ny
        f = maptoreal_fft(nr,nn,nx,ny,c)
        
        # Physical maxima
        ky1 = self.results[label].cgyrodata.ky[1] if len(self.results[label].cgyrodata.ky) > 1 else self.results[label].cgyrodata.ky[0]
        xmax = self.results[label].cgyrodata.length
        ymax = (2*np.pi)/np.abs(ky1)
        xp = x/(2*np.pi)*xmax
        yp = y/(2*np.pi)*ymax

        # Periodic extensions
        xp = np.append(xp,xmax)
        yp = np.append(yp,ymax)
        fp = np.zeros([nx+1,ny+1])
        fp[0:nx,0:ny] = f[:,:]
        fp[-1,:] = fp[0,:]
        fp[:,-1] = fp[:,0]
        
        return xp, yp, fp
        
    def plot_quick_linear(self, labels=["cgyro1"], fig=None):
 
        colors = GRAPHICStools.listColors()
        ls = GRAPHICStools.listLS()

        if fig is None:
            fig = plt.figure(figsize=(15,9))

        axs = fig.subplot_mosaic(
            """
            12
            34
            """
        )
            
        def _plot_linear_stability(axs, labels, label_base,col_lin ='b', start_cont=0):

            irho = self.results[label_base].irho

            for cont, label in enumerate(labels):
                c = self.results[label]['output'][irho]
                baseColor = colors[cont+start_cont+1]
                colorsC, _ = GRAPHICStools.colorTableFade(
                    len(c.ky),
                    startcolor=baseColor,
                    endcolor=baseColor,
                    alphalims=[1.0, 0.4],
                )

                ax = axs['1']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.g[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$, $r/a={c.roa:.2f}$",
                        ls = ls[irho]
                    )

                ax = axs['2']
                for ky in range(len(c.ky)):
                    ax.plot(
                        c.t,
                        c.f[ky,:],
                        color=colorsC[ky],
                        label=f"$k_{{\\theta}}\\rho_s={np.abs(c.ky[ky]):.2f}$, $r/a={c.roa:.2f}$",
                        ls = ls[irho]
                    )

            roa = self.results[self.results[label_base].labels[0]]['output'][irho].roa

            GACODEplotting.plotTGLFspectrum(
                [axs['3'], axs['4']],
                abs(self.results[label_base].ky),
                self.results[label_base].g_mean,
                freq=self.results[label_base].f_mean,
                coeff=0.0,
                c=col_lin,
                ls="-",
                lw=1,
                label=f"r/a = {roa}",
                facecolors=colors,
                markersize=50,
                alpha=1.0,
                titles=["Growth Rate", "Real Frequency"],
                removeLow=1e-4,
                ylabel=True,
            )
            axs['3'].legend(loc='best', prop={'size': 8},)
            
            return cont

        co = -1
        for i,label0 in enumerate(labels):
            co = _plot_linear_stability(axs, self.results[label0].labels, label0, start_cont=co, col_lin=colors[i])

        ax = axs['1']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_ylabel("$\\gamma$ $(c_s/a)$")
        ax.set_title("Growth Rate")
        ax.set_xlim(left=0)
        ax.legend(loc='best', prop={'size': 8},)
        
        ax = axs['2']
        ax.set_xlabel("Time $(a/c_s)$")
        ax.set_ylabel("$\\omega$ $(c_s/a)$")
        ax.set_title("Real Frequency")
        ax.axhline(y=0, lw=0.5, ls="--", c="k")
        ax.set_xlim(left=0)
        
        for ax in [axs['1'], axs['2'], axs['3'], axs['4']]:
            GRAPHICStools.addDenseAxis(ax)
        
        plt.tight_layout()

class CGYROinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.cgyro.controls",
            code="CGYRO",
            n_species='N_SPECIES',
        )

def _2D_mosaic(n_times):

    num_cols = n_times

    # Create the mosaic layout dynamically
    mosaic = []
    counter = 1
    for _ in range(3):
        row = []
        for _ in range(num_cols):
            row.append(str(counter))
            counter += 1
        mosaic.append(row)
        
    return mosaic