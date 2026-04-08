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
import os
from pathlib import Path

import numpy as np

from mitim_tools.simulation_tools.SIMtools import modifyInputs
from mitim_tools.gacode_tools.utils import NORMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.PLASMAtools import md_u


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

        n_workers = os.cpu_count() or 1
        print(
            f"\t- [in-process] Submitting {n_jobs} {self._inprocess_code_name.upper()} "
            f"cases across {min(n_workers, n_jobs)} workers"
        )

        results_by_folder: dict = {}
        with ThreadPoolExecutor(max_workers=min(n_workers, n_jobs)) as pool:
            future_map = {
                pool.submit(worker, flat): (subfolder_sim, rho, input_obj, folder_sim)
                for subfolder_sim, rho, flat, input_obj, folder_sim in work_items
            }
            for future in as_completed(future_map):
                subfolder_sim, rho, input_obj, folder_sim = future_map[future]
                outputs = future.result()
                key = str(folder_sim)
                if key not in results_by_folder:
                    results_by_folder[key] = {"raw": {}, "inputs": {}}
                results_by_folder[key]["raw"][float(rho)]    = outputs
                results_by_folder[key]["inputs"][float(rho)] = input_obj
                self._inprocess_print_per_rho(rho, outputs)

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

    def _inprocess_postprocess_input(self, input_sim_rho, code_settings, kwargs_control):
        # TGLF: drop low-density / fast species and optionally enforce
        # quasineutrality before passing the input to the Fortran engine.
        if code_settings is not None:
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
            f"Qe={outputs['elec_eflux']:.4f}  "
            f"Qi[0]={outputs['ion_eflux'][0]:.4f}"
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
