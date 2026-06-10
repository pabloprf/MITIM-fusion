"""
tglf_inprocess.py
=================
In-process TGLF execution via a ctypes-loaded shared library (libtglf_serial.so).

Three entry points are provided:

1. ``runner.prepare(profiles_or_path, rhos, code_settings, ...)`` +
   ``runner.run_all()`` / ``runner.run_rho(rho)``
   **Standalone fully in-memory path** — pass an ``input.gacode`` file path or
   a live ``PROFILES_GACODE`` object; inputs are built via ``to_tglf()`` and
   cached in memory.  All subsequent run calls are zero file I/O.

2. ``runner.run_from_dict(input_dict)``
   Sets all Fortran module variables directly from a flat Python dict — zero
   file I/O.  The dict is exactly what ``PROFILES_GACODE.to_tglf()`` returns
   for one rho value.

3. ``runner.run(gen_file)``
   Legacy path — reads an ``input.tglf.gen`` file from disk.

All paths call ``c_tglf_run()`` in-process and return the same output dict.

API
---
    from mitim_tools.simulation_tools.interfaces.tglf_inprocess import (
        TGLFInProcess,
        generate_input_gen,
        write_gbflux,
    )

    # --- standalone in-memory path (preferred) ---
    runner = TGLFInProcess()
    runner.prepare("path/to/input.gacode", rhos=[0.4, 0.6], code_settings="SAT1")
    results = runner.run_all()        # {rho: output_dict}
    out     = runner.run_rho(0.4)

    # --- dict-based path ---
    runner  = TGLFInProcess()
    outputs = runner.run_from_dict(input_dict)  # dict from to_tglf()

    # --- file-based path (requires GACODE_ROOT) ---
    gen_file = generate_input_gen("/path/to/input.tglf")
    outputs  = runner.run(gen_file)

    write_gbflux(outputs, "/path/to/out.tglf.gbflux")

Prerequisites
-------------
    Build the shared library once per machine::

        cd <MITIM-fusion>/src/mitim_tools/simulation_tools/interfaces
        bash build_tglf_lib.sh

Thread safety
-------------
The Fortran library uses global module variables.  A single loaded instance
is NOT safe to call from multiple threads simultaneously.

For parallel in-process runs we copy the .so to unique temporary paths so
each worker thread gets its own ``dlopen`` handle with independent Fortran
globals.  ``ctypes`` releases the GIL during Fortran calls, so threads give
true parallelism without any of the macOS multiprocessing pitfalls (no
spawn/fork/forkserver issues, no ``if __name__ == '__main__':`` requirement).
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import tempfile
import threading
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# All build artefacts live in tglf_build/ (gitignored) inside this directory.
# ---------------------------------------------------------------------------
_INTERFACES_DIR = Path(__file__).parent
_LIB_PATH = _INTERFACES_DIR / "tglf_build" / "libtglf_serial.so"

# ---------------------------------------------------------------------------
# Lazy singleton: load libtglf_serial.so once per process (sequential use)
# ---------------------------------------------------------------------------
_lib: Any = None

# ---------------------------------------------------------------------------
# Thread-parallel support: each worker thread gets its own private copy of
# the .so so Fortran module-level globals are truly independent.
# ---------------------------------------------------------------------------
_thread_local = threading.local()
_temp_lib_paths: list[str] = []
_temp_lib_lock  = threading.Lock()


@atexit.register
def _cleanup_temp_libs() -> None:
    """Remove per-thread .so copies created for parallel runs."""
    for p in _temp_lib_paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OpenBLAS thread pinning
#
# WHY:
# Empirically (measured on a 16-core M-series Mac with conda-forge openblas),
# TGLF in-process is FASTEST when openblas is pinned to 1 thread:
#   1 thread :  ~700 ms / call
#   8 threads: ~1280 ms / call
# The TGLF eigensolve sub-blocks are too small to amortise openblas's
# per-call thread sync cost — multi-threaded openblas is pure overhead here.
# This is exactly why the gacode `tglf` shell wrapper exports
# OMP_NUM_THREADS=1 before invoking the binary; we mirror that behaviour for
# the in-process path via the openblas runtime API instead of an env var,
# so the user's other Python libraries (numpy etc.) are not affected.
#
# We do NOT do the same for NEO — its dense linear solves are larger and
# may legitimately benefit from multi-threaded BLAS.  See neo_inprocess.py
# if a future measurement says otherwise.
# ---------------------------------------------------------------------------
# Cached setter callables — populated by _discover_thread_setters() on first
# use, then re-applied via _pin_threads() before every c_tglf_run call.
_THREAD_SETTERS: list[tuple[str, "callable"]] = []
_THREAD_SETTERS_INIT = False


def _discover_thread_setters() -> list[tuple[str, "callable"]]:
    """
    Discover every "set num threads" entry point reachable in this process.

    On macOS conda-forge, libopenblas is built with USE_OPENMP=1, which
    means its actual worker pool is controlled by libomp (LLVM OpenMP),
    not by openblas's internal thread state.  Calling
    ``openblas_set_num_threads()`` works *until* the first OpenMP parallel
    region runs, which silently resets the worker count to libomp's
    OMP_NUM_THREADS (16 → 12 here).  So we have to pin BOTH openblas AND
    libomp on every call to be sure.

    We cache the discovered setters so we only dlopen once per process.
    Best-effort: missing libraries are ignored silently.
    """
    global _THREAD_SETTERS_INIT
    if _THREAD_SETTERS_INIT:
        return _THREAD_SETTERS

    candidates = [
        # (display name, libnames to try, symbol to look up)
        ("openblas", ["libopenblas.0.dylib", "libopenblas.dylib",
                      "libopenblas.so.0", "libopenblas.so"],
                     "openblas_set_num_threads"),
        ("omp",      ["libomp.dylib",
                      "libomp.so", "libgomp.so.1", "libgomp.so"],
                     "omp_set_num_threads"),
    ]
    for name, libnames, sym in candidates:
        for ln in libnames:
            try:
                lib = ctypes.CDLL(ln)
            except OSError:
                continue
            fn = getattr(lib, sym, None)
            if fn is None:
                continue
            fn.argtypes = [ctypes.c_int]
            fn.restype  = None

            def _setter(n: int, _fn=fn):
                _fn(ctypes.c_int(int(n)))

            _THREAD_SETTERS.append((name, _setter))
            break  # found this candidate; move to next family

    _THREAD_SETTERS_INIT = True
    return _THREAD_SETTERS


def _pin_threads(n: int = 1) -> list[str]:
    """
    Cap openblas + libomp at *n* threads.  Returns the names actually pinned
    (for logging).  Cheap: ~10 µs once setters are cached, so safe to call
    immediately before every c_tglf_run.
    """
    pinned = []
    for name, setter in _discover_thread_setters():
        try:
            setter(int(n))
            pinned.append(name)
        except Exception:  # noqa: BLE001 — best-effort, never fail
            pass
    return pinned


def _setup_lib_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach ctypes argument/return-type annotations to *lib*. Returns *lib*."""
    lib.c_tglf_set_path.restype  = None
    lib.c_tglf_set_path.argtypes = [ctypes.c_char_p]

    lib.c_tglf_read_input.restype  = None
    lib.c_tglf_read_input.argtypes = []

    lib.c_tglf_run.restype  = None
    lib.c_tglf_run.argtypes = []

    _double11 = ctypes.c_double * 11
    lib.c_tglf_get_outputs.restype = None
    lib.c_tglf_get_outputs.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(_double11),
        ctypes.POINTER(_double11),
        ctypes.POINTER(_double11),
        ctypes.POINTER(_double11),
        ctypes.POINTER(_double11),
    ]
    return lib


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"libtglf_serial.so not found at {_LIB_PATH}\n"
            "  Build it once with:\n"
            f"    cd {_INTERFACES_DIR} && bash build_tglf_lib.sh"
        )

    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load {_LIB_PATH}: {exc}\n"
            "Check that the library was compiled for the current platform."
        ) from exc

    # Pin openblas + libomp to 1 thread for the TGLF code path. We re-pin
    # before every c_tglf_run via _set_inputs_from_dict, since the call
    # itself can reset the count. See _discover_thread_setters() for why.
    pinned = _pin_threads(1)
    if pinned:
        print(f"[tglf_inprocess] pinned {', '.join(pinned)} to 1 thread "
              f"(TGLF eigensolve is faster single-threaded)")

    _lib = _setup_lib_signatures(lib)
    return _lib


def _load_unique_lib() -> ctypes.CDLL:
    """
    Load a PRIVATE copy of the shared library.

    ``dlopen`` (and ctypes.CDLL) caches handles by realpath, so calling
    ``CDLL(same_path)`` from multiple threads returns the same instance —
    and the same Fortran module-level globals.  Copying the file to a unique
    temp path forces a new ``dlopen`` handle with independent globals, making
    truly parallel ctypes calls safe.

    The temp file is registered for deletion at process exit.
    """
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"libtglf_serial.so not found at {_LIB_PATH}\n"
            "  Build it once with:\n"
            f"    cd {_INTERFACES_DIR} && bash build_tglf_lib.sh"
        )
    fd, tmp_path = tempfile.mkstemp(suffix="_tglf_thread.so")
    os.close(fd)
    shutil.copy(str(_LIB_PATH), tmp_path)
    with _temp_lib_lock:
        _temp_lib_paths.append(tmp_path)
    try:
        lib = ctypes.CDLL(tmp_path)
    except OSError as exc:
        raise RuntimeError(f"Failed to load private lib copy {tmp_path}: {exc}") from exc
    # Pin openblas + libomp to 1 thread (idempotent across thread libs).
    _pin_threads(1)
    return _setup_lib_signatures(lib)


def _get_thread_lib() -> ctypes.CDLL:
    """Return the calling thread's private library instance, creating it on first use."""
    if not hasattr(_thread_local, "lib"):
        _thread_local.lib = _load_unique_lib()
    return _thread_local.lib


# ---------------------------------------------------------------------------
# Direct module-variable access — maps dict keys → Fortran tglf_interface vars
# ---------------------------------------------------------------------------

# gfortran mangles   module::var   →   __module_MOD_var
# On macOS the linker adds one more leading _ (seen in `nm` as ___module_MOD_var),
# but ctypes/dlsym uses the name without that extra leading _.
_MOD = "__tglf_interface_MOD_tglf_{}_in"

# nsm = 12 (from tglf_max_dimensions); species arrays have 12 elements
_NSM = 12

# Base names of per-species array variables (without the _N index suffix)
_SPECIES_BASES = frozenset([
    "zs", "mass", "rlns", "rlts", "taus", "as", "vpar",
    "vpar_shear", "vns_shear", "vts_shear",
])

# Fortran LOGICAL variables → stored as 4-byte integer (1 = .true.)
_LOGICALS = frozenset([
    "use_transport_model", "dump_flag", "quiet_flag",
    "iflux", "use_bper", "use_bpar", "use_mhd_rule", "use_bisection",
    "use_inboard_detrapped", "use_ave_ion_grid", "adiabatic_elec",
    "find_width", "new_eikonal",
])

# Fortran INTEGER variables → stored as 4-byte integer
_INTEGERS = frozenset([
    "test_flag", "geometry_flag", "write_wavefunction_flag",
    "ibranch", "nmodes", "nbasis_max", "nbasis_min",
    "nxgrid", "nky", "sat_rule", "kygrid_model", "xnu_model",
    "vpar_model", "vpar_shear_model", "ns", "nwidth",
    "b_model_sa", "ft_model_sa",
])


def _set_inputs_from_dict(lib: ctypes.CDLL, input_dict: dict) -> None:
    """
    Write every entry in *input_dict* directly into the tglf_interface Fortran
    module variables.  This replaces the ``tglf_read_input()`` file-read.

    Key conventions (matching ``PROFILES_GACODE.to_tglf()`` output):
    - Scalar controls / plasma:  ``"BETAE"``, ``"RMIN_LOC"``, ``"SAT_RULE"`` …
    - Species arrays:            ``"ZS_1"``, ``"RLTS_2"`` … (1-based index)
    - Shape harmonics:           ``"SHAPE_COS0"``, ``"SHAPE_S_SIN3"`` …
      (Fortran names append ``_loc``: ``tglf_shape_cos0_loc_in``)
    - Units string:              ``"UNITS"``  → ``character(len=8)``
    """
    # Always suppress output — these are not part of the physics dict
    ctypes.c_int32.in_dll(lib, _MOD.format("quiet_flag")).value = 1
    ctypes.c_int32.in_dll(lib, _MOD.format("dump_flag")).value  = 0

    # Zero all species arrays so stale values from a prior call never persist
    for base in _SPECIES_BASES:
        arr = (ctypes.c_double * _NSM).in_dll(lib, _MOD.format(base))
        ctypes.memset(arr, 0, ctypes.sizeof(arr))

    for key, val in input_dict.items():
        ku = key.upper()

        # ---- species array: KEY_N (e.g. ZS_1, VPAR_SHEAR_2) ----
        head, _, tail = ku.rpartition("_")
        if tail.isdigit() and head.lower() in _SPECIES_BASES:
            idx = int(tail) - 1          # Fortran 1-based → 0-based
            arr = (ctypes.c_double * _NSM).in_dll(lib, _MOD.format(head.lower()))
            arr[idx] = float(val)
            continue

        # ---- shape harmonics: SHAPE_* → tglf_shape_*_loc_in ----
        kl = ku.lower()
        var = (kl + "_loc") if kl.startswith("shape_") else kl

        sym = _MOD.format(var)

        if ku == "UNITS":
            CharT = ctypes.c_char * 8
            s = CharT.in_dll(lib, sym)
            enc = str(val).encode("ascii")[:8].ljust(8)
            ctypes.memmove(s, enc, 8)

        elif var in _LOGICALS:
            ctypes.c_int32.in_dll(lib, sym).value = 1 if val else 0

        elif var in _INTEGERS:
            ctypes.c_int.in_dll(lib, sym).value = int(val)

        else:
            # real — c_double because the library was built with -fdefault-real-8
            try:
                ctypes.c_double.in_dll(lib, sym).value = float(val)
            except OSError:
                pass   # unknown key (e.g. a MITIM-only field) — skip silently


def _collect_outputs(lib: ctypes.CDLL, double11: type) -> dict:
    """Read tglf_*_out variables via c_tglf_get_outputs and return a dict."""
    ns_out         = ctypes.c_int(0)
    elec_pflux     = ctypes.c_double(0.0)
    elec_eflux     = ctypes.c_double(0.0)
    elec_eflux_low = ctypes.c_double(0.0)
    elec_mflux     = ctypes.c_double(0.0)
    elec_expwd     = ctypes.c_double(0.0)
    ion_pflux      = double11(*([0.0] * 11))
    ion_eflux      = double11(*([0.0] * 11))
    ion_eflux_low  = double11(*([0.0] * 11))
    ion_mflux      = double11(*([0.0] * 11))
    ion_expwd      = double11(*([0.0] * 11))

    lib.c_tglf_get_outputs(
        ctypes.byref(ns_out),
        ctypes.byref(elec_pflux),
        ctypes.byref(elec_eflux),
        ctypes.byref(elec_eflux_low),
        ctypes.byref(elec_mflux),
        ctypes.byref(elec_expwd),
        ctypes.byref(ion_pflux),
        ctypes.byref(ion_eflux),
        ctypes.byref(ion_eflux_low),
        ctypes.byref(ion_mflux),
        ctypes.byref(ion_expwd),
    )

    ns = int(ns_out.value)
    ni = ns - 1
    return {
        "ns":             ns,
        "elec_pflux":     float(elec_pflux.value),
        "elec_eflux":     float(elec_eflux.value),
        "elec_eflux_low": float(elec_eflux_low.value),
        "elec_mflux":     float(elec_mflux.value),
        "elec_expwd":     float(elec_expwd.value),
        "ion_pflux":      list(ion_pflux[:ni]),
        "ion_eflux":      list(ion_eflux[:ni]),
        "ion_eflux_low":  list(ion_eflux_low[:ni]),
        "ion_mflux":      list(ion_mflux[:ni]),
        "ion_expwd":      list(ion_expwd[:ni]),
    }


# ---------------------------------------------------------------------------
# Input preprocessing: input.tglf → input.tglf.gen  (file-based path)
# ---------------------------------------------------------------------------

def generate_input_gen(input_tglf_path: str | Path) -> Path:
    """
    Convert an ``input.tglf`` file to ``input.tglf.gen`` using GACODE's
    ``tglf_parse.py`` logic.  Requires ``GACODE_ROOT`` to be set.
    """
    input_tglf_path = Path(input_tglf_path).resolve()
    if not input_tglf_path.exists():
        raise FileNotFoundError(f"input.tglf not found: {input_tglf_path}")

    gacode_root = os.environ.get("GACODE_ROOT")
    if not gacode_root:
        raise RuntimeError(
            "GACODE_ROOT environment variable is not set. "
            "Source your gacode_setup script before using tglf_inprocess."
        )
    tglf_bin = Path(gacode_root) / "tglf" / "bin"

    def _load_from_bin(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_tglf_gacode_{name}", tglf_bin / f"{name}.py"
        )
        if spec is None:
            raise ImportError(f"Cannot locate {name}.py in {tglf_bin}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tglf_defaults_mod = _load_from_bin("tglf_defaults")
    x = tglf_defaults_mod.set_defaults()
    x.set_extension(".gen")
    x.read_input(str(input_tglf_path))

    if x.error:
        raise RuntimeError(
            f"TGLF input parsing failed for {input_tglf_path}:\n{x.error_msg}"
        )

    gen_path = Path(str(input_tglf_path) + ".gen")
    if not gen_path.exists():
        raise RuntimeError(f"tglf_parse produced no output — expected {gen_path}")
    return gen_path


# ---------------------------------------------------------------------------
# Output writer: outputs dict → out.tglf.gbflux
# ---------------------------------------------------------------------------

def write_gbflux(outputs: dict, filepath: str | Path) -> None:
    """
    Write ``out.tglf.gbflux`` from the output dict returned by
    :meth:`TGLFInProcess.run` or :meth:`TGLFInProcess.run_from_dict`.

    Format: one ASCII line — ``Ge Gi[0]…Gi[ni-1]  Qe Qi…  Me Mi…  Se Si…``
    """
    ni = outputs["ns"] - 1
    values: list[float] = []
    values.append(outputs["elec_pflux"])
    values.extend(outputs["ion_pflux"][:ni])
    values.append(outputs["elec_eflux"])
    values.extend(outputs["ion_eflux"][:ni])
    values.append(outputs["elec_mflux"])
    values.extend(outputs["ion_mflux"][:ni])
    values.append(outputs["elec_expwd"])
    values.extend(outputs["ion_expwd"][:ni])
    Path(filepath).write_text(" ".join(f"{v:11.4e}" for v in values) + "\n")


# ---------------------------------------------------------------------------
# Main in-process runner
# ---------------------------------------------------------------------------

class TGLFInProcess:
    """
    Run TGLF in-process via ctypes — no subprocess fork, no physics file I/O.

    Three usage modes:

    **Standalone / fully in-memory** (preferred)::

        runner = TGLFInProcess()
        runner.prepare("path/to/input.gacode", rhos=[0.4, 0.6], code_settings="SAT1")
        results = runner.run_all()   # {rho: output_dict} — zero file I/O
        out     = runner.run_rho(0.4)

    **Dict-based** (when you already have a flat input dict)::

        tglf_inputs = profiles.to_tglf(r=rhos, code_settings="SAT1")
        out = runner.run_from_dict(tglf_inputs[0.5])

    **File-based** (legacy, requires GACODE_ROOT)::

        gen = generate_input_gen("/work/run1/input.tglf")
        out = runner.run(gen)

    Sequential calls are safe; for parallel use ``multiprocessing`` so each
    worker has its own independent copy of the library globals.
    """

    def __init__(self) -> None:
        self._lib      = _load_lib()
        self._double11 = ctypes.c_double * 11
        self._inputs: dict = {}  # {float(rho): flat_dict} — populated by prepare()

    # ------------------------------------------------------------------
    # Standalone in-memory path
    # ------------------------------------------------------------------

    def prepare(
        self,
        profiles,
        rhos: list = [0.5],
        code_settings = "SAT1",
        extraOptions: dict = {},
        multipliers: dict = {},
    ) -> None:
        """
        Build and cache TGLF input dicts in memory from a PROFILES_GACODE object
        or an ``input.gacode`` file path.

        Calls ``profiles.to_tglf()`` once, applies *multipliers* and
        *extraOptions*, and stores the flat dicts in ``self._inputs``.
        After this call, :meth:`run_all` and :meth:`run_rho` are zero file I/O.

        Parameters
        ----------
        profiles:
            Path to an ``input.gacode`` file (str or Path), **or** a live
            ``PROFILES_GACODE`` / ``gacode_state`` object already in memory.
        rhos:
            Radial locations (rho_tor) to prepare.
        code_settings:
            TGLF control-parameter preset passed to ``to_tglf()``
            (e.g. ``"SAT0"``, ``"SAT1"``, ``"SAT2"``).
        extraOptions:
            Additional key/value overrides applied on top.  Keys are
            case-insensitive TGLF parameter names.
        multipliers:
            ``{parameter: factor}`` — each named parameter is scaled by the
            given factor after ``to_tglf()`` populates it.
        """
        if isinstance(profiles, (str, Path)):
            from mitim_tools.gacode_tools.PROFILEStools import gacode_state
            profiles = gacode_state(str(profiles))

        raw = profiles.to_tglf(r=list(rhos), code_settings=code_settings)

        self._inputs = {}
        for rho, d in raw.items():
            flat = dict(d)
            for key, factor in multipliers.items():
                ku = key.upper()
                if ku in flat:
                    flat[ku] = flat[ku] * factor
            flat.update({k.upper(): v for k, v in extraOptions.items()})
            self._inputs[float(rho)] = flat

    def run_all(self) -> dict:
        """
        Run TGLF for every rho prepared by :meth:`prepare`.

        Returns ``{rho: output_dict}`` — same structure as :meth:`run_from_dict`.
        """
        if not self._inputs:
            raise RuntimeError("No inputs prepared — call prepare() first.")
        return {rho: self.run_from_dict(d) for rho, d in self._inputs.items()}

    def run_rho(self, rho: float) -> dict:
        """Run TGLF for a single rho prepared by :meth:`prepare`."""
        if rho not in self._inputs:
            raise KeyError(
                f"rho={rho} not in prepared inputs. "
                f"Available: {sorted(self._inputs)}"
            )
        return self.run_from_dict(self._inputs[rho])

    # ------------------------------------------------------------------
    # File-based path
    # ------------------------------------------------------------------

    def run(self, gen_file_path: str | Path) -> dict:
        """
        Execute TGLF reading inputs from ``input.tglf.gen`` on disk.

        Parameters
        ----------
        gen_file_path:
            Path to ``input.tglf.gen`` produced by :func:`generate_input_gen`.
        """
        gen_file_path = Path(gen_file_path).resolve()
        if not gen_file_path.exists():
            raise FileNotFoundError(f"input.tglf.gen not found: {gen_file_path}")

        path_str = str(gen_file_path.parent) + "/"
        self._lib.c_tglf_set_path(path_str.encode("ascii"))
        self._lib.c_tglf_read_input()
        _pin_threads(1)            # see _discover_thread_setters() — must be re-applied per call
        self._lib.c_tglf_run()
        return _collect_outputs(self._lib, self._double11)

    # ------------------------------------------------------------------
    # In-memory path
    # ------------------------------------------------------------------

    def run_from_dict(self, input_dict: dict) -> dict:
        """
        Execute TGLF setting all inputs directly from *input_dict* — no file I/O.

        *input_dict* is the value for one rho from ``PROFILES_GACODE.to_tglf()``,
        optionally with ``extra_options`` applied on top::

            tglf_inputs = profiles.to_tglf(r=[0.5], code_settings="SAT1")
            d = tglf_inputs[0.5]
            d.update({"XNU_FACTOR": 1.5})   # optional override
            out = runner.run_from_dict(d)

        Parameters
        ----------
        input_dict:
            Flat dict whose keys follow the TGLF parameter naming convention
            (e.g. ``"BETAE"``, ``"RMIN_LOC"``, ``"ZS_1"``, ``"SAT_RULE"``).
            Species arrays use 1-based index suffixes (``"ZS_1"``, ``"ZS_2"``…).
            Shape harmonics omit ``_LOC`` (``"SHAPE_COS0"`` maps to
            ``tglf_shape_cos0_loc_in``).
        """
        _set_inputs_from_dict(self._lib, input_dict)
        _pin_threads(1)            # see _discover_thread_setters() — must be re-applied per call
        self._lib.c_tglf_run()
        return _collect_outputs(self._lib, self._double11)


# ------------------------------------------------------------------
# Module-level worker — must be at module scope to be picklable
# for ProcessPoolExecutor / multiprocessing.
# ------------------------------------------------------------------

def _parallel_worker(flat: dict) -> dict:
    """
    Run one TGLF case on the calling thread's private library instance.

    Each worker thread in the pool loads its own independent copy of the
    shared library (via ``_get_thread_lib()``), giving isolated Fortran
    module-level globals.  Because ``ctypes`` releases the GIL during the
    Fortran call, multiple threads execute the physics truly in parallel.
    """
    lib = _get_thread_lib()
    double11 = ctypes.c_double * 11
    _set_inputs_from_dict(lib, flat)
    _pin_threads(1)            # see _discover_thread_setters() — must be re-applied per call
    lib.c_tglf_run()
    return _collect_outputs(lib, double11)
