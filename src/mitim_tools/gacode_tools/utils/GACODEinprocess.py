"""
GACODEinprocess.py
==================
Pure in-process logic for the GACODE-family simulation wrappers.

This module contains **only** the in-process (ctypes) execution path —
prep, run-prepare, run, read.  None of the methods here check
``self.in_process`` or fall back to a subprocess engine; that decision
lives in ``TGLFtools.TGLF`` / ``NEOtools.NEO``, which dispatch to the
methods defined here when their ``in_process`` flag is set, and to the
inherited ``mitim_simulation`` (subprocess) implementation otherwise.

The classes are mixins — they do **not** inherit from
``mitim_simulation``.  ``TGLF`` / ``NEO`` use multiple inheritance to
combine the subprocess engine with the in-process mixin::

    class TGLF(SIMtools.mitim_simulation, GACODEinprocess.TGLFInProcess):
        def prep(self, ...):
            if self.in_process:
                return self.prep_inprocess(...)
            return super().prep(...)
        ...

Method names are suffixed with ``_inprocess`` to make name collisions
with the subprocess parent impossible.

Class layout
------------
::

    _GACODEInProcessMixin                 (shared in-process plumbing)
            ▲
            │       ┌──────────────┐
            ├───────┤ TGLFInProcess │   (TGLF-specific hooks + read)
            │       └──────────────┘
            │       ┌──────────────┐
            └───────┤  NEOInProcess │   (NEO-specific hooks + read)
                    └──────────────┘

Hooks subclasses customise:
* ``_inprocess_load_worker()``        — return the per-code worker callable
* ``_inprocess_postprocess_input()``  — apply per-rho input corrections
* ``_inprocess_build_flat_dict()``    — build the flat input dict
* ``_inprocess_print_per_rho()``      — per-rho post-run logging

…and the class attribute ``_inprocess_code_name`` (used for tempdir
prefixes and log lines).
"""

from __future__ import annotations

import copy
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from mitim_tools.simulation_tools.SIMtools import modifyInputs
from mitim_tools.gacode_tools.utils import NORMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.PLASMAtools import md_u


# ---------------------------------------------------------------------------
# BLAS thread budget control (runtime API, works after dlopen)
#
# When the in-process driver fans N codes out across N Python worker threads,
# each worker enters Fortran code that calls into the underlying BLAS/LAPACK
# library.  That BLAS library has its own thread pool and, by default, sizes
# it to the entire node — so 18 workers on a 64-core node would each spin up
# 64 BLAS threads, giving 18*64 ≈ 1152 OS threads competing for 64 cores.
#
# Setting OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS in the
# process environment is unreliable: env vars are read at library init time
# (i.e. at dlopen) and pinning every BLAS to 1 thread globally also serialises
# unrelated single-call paths that legitimately want multi-threaded BLAS
# (laptop runs of TGLF visibly slow down 5-10x, since the per-call wins from
# many BLAS threads dominate when there's no oversubscription pressure).
#
# Instead we use the BLAS library's own runtime API to set the per-pool
# thread budget JUST BEFORE submitting a worker pool, sized to
#     max(1, n_cpus // n_workers)
# This is best-effort: we try every BLAS we know about (openblas, MKL,
# Apple Accelerate / vecLib).  If none of them are loadable we print what
# happened and carry on — no failure.
# ---------------------------------------------------------------------------

# Module-level cache so we don't keep re-dlopen'ing the same BLAS library.
_BLAS_HANDLES: dict = {}     # name -> (cdll, setter_callable)


def _try_load(libname: str):
    """Best-effort dlopen.  Returns the CDLL handle or None."""
    try:
        return ctypes.CDLL(libname)
    except OSError:
        return None


def _discover_blas_setters() -> list[tuple[str, callable]]:
    """
    Discover BLAS thread-count setters available in this process.

    Returns a list of (name, setter) where setter(n: int) caps the BLAS
    library at *n* threads.  Best effort: we don't error if a particular
    BLAS isn't installed — most processes only have one anyway.
    """
    if _BLAS_HANDLES:
        return [(name, setter) for name, (_lib, setter) in _BLAS_HANDLES.items()]

    candidates: list[tuple[str, list[str], list[str]]] = []
    # (display name, dlopen candidates, setter symbol candidates)
    if sys.platform == "darwin":
        candidates += [
            ("openblas",   ["libopenblas.dylib", "libopenblas.0.dylib"],
                           ["openblas_set_num_threads", "openblas_set_num_threads64_"]),
            ("Accelerate", ["/System/Library/Frameworks/Accelerate.framework/Accelerate"],
                           ["BLASSetThreading"]),  # rarely present, harmless if absent
        ]
    else:
        candidates += [
            ("openblas",   ["libopenblas.so.0", "libopenblas.so"],
                           ["openblas_set_num_threads", "openblas_set_num_threads64_"]),
            ("mkl",        ["libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so"],
                           ["MKL_Set_Num_Threads", "mkl_set_num_threads_"]),
        ]

    found: list[tuple[str, callable]] = []
    for name, libnames, syms in candidates:
        for ln in libnames:
            lib = _try_load(ln)
            if lib is None:
                continue
            for sym in syms:
                fn = getattr(lib, sym, None)
                if fn is None:
                    continue
                fn.argtypes = [ctypes.c_int]
                fn.restype  = None

                def _setter(n: int, _fn=fn):
                    _fn(ctypes.c_int(int(n)))

                _BLAS_HANDLES[name] = (lib, _setter)
                found.append((name, _setter))
                break
            if name in _BLAS_HANDLES:
                break

    return found


def _set_blas_threads(n: int) -> list[str]:
    """
    Best-effort: cap every BLAS we can reach at *n* threads.

    Returns the list of BLAS names we successfully pinned, for logging.
    """
    n = max(1, int(n))
    pinned = []
    for name, setter in _discover_blas_setters():
        try:
            setter(n)
            pinned.append(name)
        except Exception:  # noqa: BLE001 — best-effort, never fail the run
            pass
    return pinned


# ===========================================================================
# Shared in-process mixin
# ===========================================================================

class _GACODEInProcessMixin:
    """
    Shared in-process plumbing for the GACODE-family wrappers.

    This is a **mixin** — it does not inherit from any simulation base
    class.  It assumes the host instance provides the following
    attributes (which ``mitim_simulation.__init__`` and the concrete
    subclass set up):

    * ``self.rhos``                — list of rho values
    * ``self.run_specifications``  — populated by the concrete subclass
    * ``self.FolderGACODE``        — set by ``prep_inprocess``
    * ``self.FolderSimLast``       — set by ``_run_prepare_inprocess``
    * ``self.NormalizationSets``   — populated by ``prep_inprocess``
    * ``self.results``             — created by ``mitim_simulation.__init__``
    * ``self.inputs_files``        — populated by ``prep_inprocess``
    * ``self._inprocess_cache``    — created by ``_init_inprocess``

    Concrete subclasses (``TGLFInProcess`` / ``NEOInProcess``) override
    the small ``_inprocess_*`` hooks below to plug in code-specific
    behaviour.
    """

    # Subclasses override.  Used for tempdir prefix + log lines.
    _inprocess_code_name: str = "gacode"

    # ------------------------------------------------------------------
    # State initializer (call from the concrete class __init__)
    # ------------------------------------------------------------------

    def _init_inprocess(self) -> None:
        """Initialise the in-process result cache.  Idempotent."""
        if not hasattr(self, "_inprocess_cache"):
            # {str(folder_sim): {"raw": {rho: outputs}, "inputs": {rho: input_obj}}}
            self._inprocess_cache: dict = {}

    # ------------------------------------------------------------------
    # Hooks subclasses override
    # ------------------------------------------------------------------

    def _inprocess_load_worker(self):
        """Return the module-level ``_parallel_worker`` callable for this code."""
        raise NotImplementedError

    def _inprocess_postprocess_input(self, input_sim_rho, code_settings, kwargs_control):
        """
        Apply per-rho corrections after ``modifyInputs()``.  Default is just
        to call ``anticipate_problems``; TGLF additionally removes
        low-density / fast species and optionally enforces quasineutrality.
        """
        input_sim_rho.anticipate_problems()

    def _inprocess_build_flat_dict(self, input_obj) -> dict:
        """
        Convert an input object into the flat dict consumed by the
        in-process worker.  Default merges ``controls`` and ``plasma``
        (NEO species fields live in ``plasma`` already).  TGLF overrides
        to also unpack the per-species sub-dicts.
        """
        flat: dict = {}
        flat.update(input_obj.controls)
        flat.update(input_obj.plasma)
        return flat

    def _inprocess_blas_threads_per_worker(self, n_cpus: int, n_workers: int) -> int:
        """
        How many BLAS threads each pool worker is allowed to use.

        Default policy: divide cores evenly across workers, so the total
        BLAS thread count never exceeds the available cores.  TGLF overrides
        this to return 1 unconditionally because its eigensolves don't
        amortise openblas's per-call thread sync (measured: 1 thread is
        meaningfully faster than 8 for the full TGLF run).
        """
        return max(1, n_cpus // max(1, n_workers))

    def _inprocess_print_per_rho(self, rho, outputs):
        """Hook for per-rho post-run logging.  Default is silent."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inprocess_get_cached(self, folder):
        """
        Look up the cached in-process result entry for ``folder`` (or for
        ``self.FolderSimLast`` when *folder* is None).  Raises a clear
        RuntimeError if nothing is cached.
        """
        subfolder = folder if folder is not None else self.FolderSimLast
        cache_key = str(subfolder)
        if cache_key not in self._inprocess_cache:
            raise RuntimeError(
                f"No in-process results cached for key '{cache_key}'. "
                "Call run() first."
            )
        cached = self._inprocess_cache[cache_key]
        return cached, cached["raw"], cached["inputs"]

    # ------------------------------------------------------------------
    # In-process implementations of prep / _run_prepare / _run
    # No fallback to subprocess — that decision is made by the host class.
    # ------------------------------------------------------------------

    def prep_inprocess(self, mitim_state):
        """
        Prepare the run from a MITIM state (input.gacode path or object)
        with zero file I/O — no folder is needed, no folder is created.

        ``self.FolderGACODE`` is set to a synthetic in-memory ``Path``
        like ``<tglf_inprocess>/`` that is **never** touched on disk; it
        only serves as a logical base for cache keys + ``inp.file``
        attributes that downstream code may inspect.
        """
        from mitim_tools.gacode_tools import PROFILEStools

        self._init_inprocess()

        # Synthetic, never-created logical base path. No tempfile.mkdtemp,
        # no real directory anywhere on disk — purely an identifier used
        # to construct cache keys (str(folder_sim)) further down.
        self.FolderGACODE = Path(f"<{self._inprocess_code_name}_inprocess>")

        # Load profiles
        if isinstance(mitim_state, (str, Path)):
            self.profiles = PROFILEStools.gacode_state(str(mitim_state))
        else:
            self.profiles = mitim_state

        self.profiles.derive_quantities(mi_ref=md_u)

        # Build per-rho input objects in memory — no files written
        state_converter = self.run_specifications['state_converter']    # 'to_tglf' / 'to_neo'
        input_class     = self.run_specifications['input_class']        # TGLFinput / NEOinput
        input_file_stem = self.run_specifications['input_file']         # 'input.tglf' / 'input.neo'

        raw = getattr(self.profiles, state_converter)(r=self.rhos, r_is_rho=True)
        self.inputs_files = {}
        for rho, d in raw.items():
            inp = input_class.initialize_in_memory(d)
            # Set a logical (never-created) file path so any later code
            # that inspects inp.file does not hit AttributeError.
            inp.file = self.FolderGACODE / f"{input_file_stem}_{float(rho):.4f}"
            self.inputs_files[rho] = inp

        # Normalizations (pure in-memory)
        print("> Setting up normalizations")
        self.NormalizationSets, cdf = NORMtools.normalizations(self.profiles)
        return cdf

    def _run_prepare_inprocess(self, subfolder_simulation,
                               code_executor=None, code_executor_full=None,
                               code_settings=None, extraOptions={},
                               multipliers={}, minimum_delta_abs={},
                               **kwargs_control):
        """
        Build per-rho input objects in memory via ``modifyInputs()`` —
        no folder creation, no file writes.

        ``kwargs_control`` accepts and silently ignores any subprocess-only
        keyword arguments (cold_start, launchSlurm, slurm_setup, etc.) so
        the host class can simply forward ``**kwargs`` without filtering.
        """
        if code_executor is None:
            code_executor = {}
        if code_executor_full is None:
            code_executor_full = {}

        # Compute the logical folder path — purely an in-memory cache
        # key, never created on disk.  No `.resolve()` because the base
        # path is synthetic (e.g. `<tglf_inprocess>`) and we want it to
        # stay that way.
        Folder_sim = self.FolderGACODE / subfolder_simulation

        inputs = copy.deepcopy(self.inputs_files)

        mod_input_file = {}
        for i, rho in enumerate(self.rhos):
            print(f"\t- [in-process] Preparing input for rho={rho:.4f}")
            input_sim_rho = modifyInputs(
                inputs[rho],
                code_settings=code_settings,
                extraOptions=extraOptions,
                multipliers=multipliers,
                minimum_delta_abs=minimum_delta_abs if minimum_delta_abs else {},
                position_change=i,
                addControlFunction=self.run_specifications["control_function"],
                controls_file=self.run_specifications["controls_file"],
                NS=inputs[rho].num_recorded,
            )
            self._inprocess_postprocess_input(input_sim_rho, code_settings, kwargs_control)
            mod_input_file[rho] = input_sim_rho

        code_executor_full[subfolder_simulation] = {}
        code_executor[subfolder_simulation] = {}
        for irho in self.rhos:
            entry = {
                "folder":     Folder_sim,
                "dictionary": mod_input_file[irho],
                "inputs":     None,
                "extraOptions": extraOptions,
                "multipliers":  multipliers,
                "additional_files_to_send": None,
            }
            code_executor_full[subfolder_simulation][irho] = entry
            code_executor[subfolder_simulation][irho]      = entry

        self.FolderSimLast = Folder_sim
        return code_executor, code_executor_full

    def _run_inprocess(self, code_executor, **kwargs_run):
        """
        Execute the in-process ``ctypes`` engine for every ``(subfolder,
        rho)`` in *code_executor*.

        Each (subfolder, rho) is dispatched to a ``ThreadPoolExecutor``
        running the per-code worker; ctypes releases the GIL during the
        Fortran call, and each thread dlopens its own private copy of
        the shared library so Fortran globals are independent — true
        parallelism without macOS spawn/fork issues.

        Results are stored in ``self._inprocess_cache[str(folder_sim)]``.
        ``kwargs_run`` is accepted but ignored (subprocess-only options).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        self._init_inprocess()
        worker = self._inprocess_load_worker()

        # Build work items — one per (subfolder_variation, rho)
        work_items = []   # (subfolder_sim, rho, flat, input_obj, folder_sim)
        for subfolder_sim, rho_dict in code_executor.items():
            for rho, rho_info in rho_dict.items():
                folder_sim = rho_info["folder"]
                input_obj  = rho_info["dictionary"]
                flat = self._inprocess_build_flat_dict(input_obj)
                work_items.append((subfolder_sim, rho, flat, input_obj, folder_sim))

        n_jobs = len(work_items)
        if n_jobs == 0:
            self.simulation_job = None
            return

        n_cpus    = os.cpu_count() or 1
        n_workers = min(n_cpus, n_jobs)
        code_name = self._inprocess_code_name.upper()

        # --- per-pool BLAS thread budget --------------------------------------
        # Each worker thread will enter Fortran code that calls into BLAS.
        # We pin the BLAS thread pool just before submitting so that
        # n_workers * blas_threads_per_worker stays ≤ n_cpus, avoiding the
        # 18*64-on-a-64-core-node oversubscription pattern that hammered
        # the cluster previously.  Per-code subclasses override
        # _inprocess_blas_threads_per_worker() — TGLF returns 1 unconditionally
        # because its eigensolve sub-blocks don't amortise openblas thread
        # sync (measured: 1 thread is faster than 8 threads per call).
        blas_threads_per_worker = self._inprocess_blas_threads_per_worker(
            n_cpus, n_workers
        )
        blas_pinned = _set_blas_threads(blas_threads_per_worker)

        thread_env = {
            "MKL":      os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP":      os.environ.get("OMP_NUM_THREADS"),
        }
        print(
            f"\t- [in-process] Submitting {n_jobs} {code_name} cases "
            f"across {n_workers} workers "
            f"(BLAS threads/worker={blas_threads_per_worker}, "
            f"runtime-pinned={blas_pinned or 'none'}; "
            f"env MKL={thread_env['MKL']} OPENBLAS={thread_env['OPENBLAS']} "
            f"OMP={thread_env['OMP']}; cpu_count={n_cpus})"
        )

        # Per-job timing wrapper — measures wall time spent inside the Fortran
        # call for each (subfolder, rho), so we can spot stragglers.
        def _timed(flat):
            t = time.perf_counter()
            out = worker(flat)
            return out, time.perf_counter() - t

        results_by_folder: dict = {}
        per_job_times: list[float] = []
        t_pool_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(_timed, flat): (subfolder_sim, rho, input_obj, folder_sim)
                for subfolder_sim, rho, flat, input_obj, folder_sim in work_items
            }
            for future in as_completed(future_map):
                subfolder_sim, rho, input_obj, folder_sim = future_map[future]
                outputs, dt = future.result()
                per_job_times.append(dt)
                key = str(folder_sim)
                if key not in results_by_folder:
                    results_by_folder[key] = {"raw": {}, "inputs": {}}
                results_by_folder[key]["raw"][float(rho)]    = outputs
                results_by_folder[key]["inputs"][float(rho)] = input_obj
                self._inprocess_print_per_rho(rho, outputs)
                print(
                    f"\t  [in-process] {code_name} done  "
                    f"subfolder={subfolder_sim!s:<24s} rho={float(rho):.3f}  "
                    f"wall={dt:6.2f}s"
                )

        wall = time.perf_counter() - t_pool_start
        if per_job_times:
            t_min = min(per_job_times)
            t_max = max(per_job_times)
            t_sum = sum(per_job_times)
            t_avg = t_sum / len(per_job_times)
            speedup = t_sum / wall if wall > 0 else float("nan")
            print(
                f"\t- [in-process] {code_name} pool finished: "
                f"{n_jobs} jobs in {wall:.2f}s wall  "
                f"(per-job min/avg/max = {t_min:.2f}/{t_avg:.2f}/{t_max:.2f}s, "
                f"sum={t_sum:.2f}s, parallel speedup={speedup:.1f}x / ideal {n_workers}x)"
            )

        self._inprocess_cache.update(results_by_folder)
        self.simulation_job = None  # keep attribute consistent with mitim_simulation


# ===========================================================================
# TGLF in-process mixin
# ===========================================================================

class TGLFInProcess(_GACODEInProcessMixin):
    """
    TGLF-specific in-process mixin.  Provides ``prep_inprocess``,
    ``_run_prepare_inprocess``, ``_run_inprocess`` (inherited from the
    base mixin) plus ``read_inprocess`` which builds ``TGLFoutput``
    instances from the cached in-process results.
    """

    _inprocess_code_name = "tglf"

    def _inprocess_load_worker(self):
        from mitim_tools.simulation_tools.interfaces import tglf_inprocess as _tip
        return _tip._parallel_worker

    def _inprocess_blas_threads_per_worker(self, n_cpus: int, n_workers: int) -> int:
        # TGLF eigensolves are too small to benefit from multi-threaded BLAS:
        # measured 1 thread = ~700 ms/call, 8 threads = ~1280 ms/call.  This
        # mirrors what gacode's `tglf` shell wrapper does (it exports
        # OMP_NUM_THREADS=1 before invoking the binary).
        return 1

    def _inprocess_postprocess_input(self, input_sim_rho, code_settings, kwargs_control):
        # TGLF: drop low-density / fast species (gated by ApplyCorrections,
        # mirroring SIMtools.change_and_write_code) and optionally enforce
        # quasineutrality before passing the input to the Fortran engine.
        # PORTALS' transport_tglf passes ApplyCorrections=False — without
        # this gate the in-process path would silently strip species the
        # subprocess path keeps and produce different fluxes.
        if code_settings is not None and kwargs_control.get("ApplyCorrections", True):
            input_sim_rho.removeLowDensitySpecie()
            input_sim_rho.remove_fast()
        if kwargs_control.get("Quasineutral", False):
            input_sim_rho.ensureQuasineutrality()
        input_sim_rho.anticipate_problems()

    def _inprocess_build_flat_dict(self, input_obj):
        # TGLFinput stores species in a dedicated `species` dict, so we
        # have to unpack it into KEY_<i> form for the worker.
        flat: dict = {}
        flat.update(input_obj.controls)
        flat.update(input_obj.plasma)
        for i, sp_data in input_obj.species.items():
            for var, val in sp_data.items():
                flat[f"{var}_{i}"] = val
        return flat

    def _inprocess_print_per_rho(self, rho, outputs):
        print(
            f"\t- [in-process] rho={rho:.4f}  "
            f"Qe={outputs['elec_eflux']:.4e}  "
            f"Qi[0]={outputs['ion_eflux'][0]:.4e}"
        )

    def read_inprocess(self, label="run1", folder=None):
        """Build TGLF results from the in-process cache — pure in-process."""
        # Local import avoids a circular dependency at module load time
        from mitim_tools.gacode_tools.TGLFtools import TGLFoutput

        cached, raw, inp_files = self._inprocess_get_cached(folder)

        if "d_perp_dict" not in self.__dict__:
            self.d_perp_dict = None

        # `updateConvolution` and the convolution attributes are defined
        # on the concrete TGLF subclass; the mixin assumes the host
        # provides them.
        self.updateConvolution()

        output_list, inputclasses, parsed = [], [], []
        for rho in self.rhos:
            inputclass = inp_files.get(rho) or inp_files.get(float(rho))
            outputs    = raw.get(rho)    or raw.get(float(rho))
            out = TGLFoutput.from_inprocess(inputclass, outputs)
            out.unnormalize(
                self.NormalizationSets["SELECTED"],
                rho=rho,
                convolution_fun_fluct=self.convolution_fun_fluct,
                factorTot_to_Perp=self.factorTot_to_Perp,
            )
            output_list.append(out)
            inputclasses.append(inputclass)
            parsed.append(
                self._inprocess_build_flat_dict(inputclass) if inputclass is not None else {}
            )

        self.results[label] = {
            "output":      output_list,
            "inputclasses": inputclasses,
            "parsed":      parsed,
            "x":           np.array(self.rhos),
            "convolution_fun_fluct": self.convolution_fun_fluct,
            "DRMAJDX_LOC": self.DRMAJDX_LOC,
            "profiles":    self.NormalizationSets.get("input_gacode"),
            "wavefunction": {},
        }
        print(f"\t- [in-process] Read TGLF results for label '{label}' "
              f"({len(self.rhos)} rho values)")


# ===========================================================================
# NEO in-process mixin
# ===========================================================================

class NEOInProcess(_GACODEInProcessMixin):
    """
    NEO-specific in-process mixin.  Provides ``prep_inprocess``,
    ``_run_prepare_inprocess``, ``_run_inprocess`` (inherited from the
    base mixin) plus ``read_inprocess`` which builds ``NEOoutput``
    instances from the cached in-process results.
    """

    _inprocess_code_name = "neo"

    def _inprocess_load_worker(self):
        from mitim_tools.simulation_tools.interfaces import neo_inprocess as _nip
        return _nip._parallel_worker

    def _inprocess_print_per_rho(self, rho, outputs):
        Qe = outputs.get("efluxtot_dke", [0])[0] + outputs.get("efluxtot_gv", [0])[0]
        Qi0 = (outputs.get("efluxtot_dke", [0, 0])[1] + outputs.get("efluxtot_gv", [0, 0])[1]) if outputs.get("ns", 0) > 1 else 0.0
        print(
            f"\t- [in-process] rho={rho:.4f}  "
            f"Qe={Qe:.4e}  Qi[0]={Qi0:.4e}"
        )
        if outputs.get("error_status", 0) != 0:
            print(
                f"\t- [in-process] rho={rho:.4f}  "
                f"WARNING error_status={outputs['error_status']}",
                typeMsg="w",
            )

    def read_inprocess(self, label="run1", folder=None):
        """Build NEO results from the in-process cache — pure in-process."""
        # Local import avoids a circular dependency at module load time
        from mitim_tools.gacode_tools.NEOtools import NEOoutput

        cached, raw, inp_files = self._inprocess_get_cached(folder)

        output_list = []
        parsed_list = []
        for rho in self.rhos:
            inputclass = inp_files.get(rho) or inp_files.get(float(rho))
            outputs    = raw.get(rho)    or raw.get(float(rho))
            out = NEOoutput.from_inprocess(inputclass, outputs)
            if 'NormalizationSets' in self.__dict__:
                out.unnormalize(self.NormalizationSets["SELECTED"], rho=rho)
            output_list.append(out)
            parsed_list.append(
                self._inprocess_build_flat_dict(inputclass) if inputclass is not None else {}
            )

        self.results[label] = {
            "output": output_list,
            "parsed": parsed_list,
            "x":      np.array(self.rhos),
        }
        print(f"\t- [in-process] Read NEO results for label '{label}' "
              f"({len(self.rhos)} rho values)")
