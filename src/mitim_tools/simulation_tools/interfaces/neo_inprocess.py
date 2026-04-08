"""
neo_inprocess.py
================
In-process NEO execution via a ctypes-loaded shared library (libneo_serial.so).

Three entry points are provided:

1. ``runner.prepare(profiles_or_path, rhos, code_settings, ...)`` +
   ``runner.run_all()`` / ``runner.run_rho(rho)``
   **Standalone fully in-memory path** — pass an ``input.gacode`` file path or
   a live ``PROFILES_GACODE`` object; inputs are built via ``to_neo()`` and
   cached in memory.  All subsequent run calls are zero file I/O.

2. ``runner.run_from_dict(input_dict)``
   Sets all Fortran module variables directly from a flat Python dict — zero
   file I/O.  The dict is exactly what ``PROFILES_GACODE.to_neo()`` returns
   for one rho value.

3. ``runner.run(gen_file_dir)``
   Legacy path — reads an ``input.neo.gen`` file from disk.

All paths call ``c_neo_run()`` in-process and return the same output dict.

API
---
    from mitim_tools.simulation_tools.interfaces.neo_inprocess import (
        NEOInProcess,
        generate_input_gen,
        write_transport_flux,
    )

    # --- standalone in-memory path (preferred) ---
    runner = NEOInProcess()
    runner.prepare("path/to/input.gacode", rhos=[0.4, 0.6], code_settings="Sonic")
    results = runner.run_all()        # {rho: output_dict}
    out     = runner.run_rho(0.4)

    # --- dict-based path ---
    runner  = NEOInProcess()
    outputs = runner.run_from_dict(input_dict)  # dict from to_neo()

    # --- file-based path (requires GACODE_ROOT) ---
    gen_file = generate_input_gen("/path/to/input.neo")
    outputs  = runner.run(gen_file)

Prerequisites
-------------
    Build the shared library once per machine::

        cd <MITIM-fusion>/src/mitim_tools/simulation_tools/interfaces
        bash build_neo_lib.sh

Thread safety
-------------
The Fortran library uses global module variables.  A single loaded instance
is NOT safe to call from multiple threads simultaneously.

For parallel in-process runs we copy the .so to unique temporary paths so
each worker thread gets its own ``dlopen`` handle with independent Fortran
globals.  ``ctypes`` releases the GIL during Fortran calls, so threads give
true parallelism without any of the macOS multiprocessing pitfalls.
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
# All build artefacts live in neo_build/ (gitignored) inside this directory.
# ---------------------------------------------------------------------------
_INTERFACES_DIR = Path(__file__).parent
_LIB_PATH = _INTERFACES_DIR / "neo_build" / "libneo_serial.so"

# ---------------------------------------------------------------------------
# Lazy singleton: load libneo_serial.so once per process (sequential use)
# ---------------------------------------------------------------------------
_lib: Any = None

# ---------------------------------------------------------------------------
# Thread-parallel support: each worker thread gets its own private copy of
# the .so so Fortran module-level globals are truly independent.
# ---------------------------------------------------------------------------
_thread_local = threading.local()
_temp_lib_paths: list[str] = []
_temp_lib_lock = threading.Lock()


@atexit.register
def _cleanup_temp_libs() -> None:
    """Remove per-thread .so copies created for parallel runs."""
    for p in _temp_lib_paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# NEO output array sizes
# ---------------------------------------------------------------------------
_NSM = 11           # n_species_max — per-species arrays are dimension(11)
_NGEO = 5           # neo_geoparams_out is dimension(5)
_double11 = ctypes.c_double * _NSM
_double5  = ctypes.c_double * _NGEO


def _setup_lib_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach ctypes argument/return-type annotations to *lib*. Returns *lib*."""
    lib.c_neo_set_path.restype  = None
    lib.c_neo_set_path.argtypes = [ctypes.c_char_p]

    lib.c_neo_read_input.restype  = None
    lib.c_neo_read_input.argtypes = []

    lib.c_neo_run.restype  = None
    lib.c_neo_run.argtypes = []

    lib.c_neo_get_outputs.restype = None
    lib.c_neo_get_outputs.argtypes = [
        ctypes.POINTER(ctypes.c_int),         # ns_out
        # 10 scalar theory outputs
        ctypes.POINTER(ctypes.c_double),      # pflux_thHH
        ctypes.POINTER(ctypes.c_double),      # eflux_thHHi
        ctypes.POINTER(ctypes.c_double),      # eflux_thHHe
        ctypes.POINTER(ctypes.c_double),      # eflux_thCHi
        ctypes.POINTER(ctypes.c_double),      # jpar_thS
        ctypes.POINTER(ctypes.c_double),      # jpar_thK
        ctypes.POINTER(ctypes.c_double),      # jpar_thN
        ctypes.POINTER(ctypes.c_double),      # jtor_thS
        ctypes.POINTER(ctypes.c_double),      # jpar_thSmod
        ctypes.POINTER(ctypes.c_double),      # jtor_thSmod
        # Hirshman-Sigmar
        ctypes.POINTER(_double11),            # pflux_thHS
        ctypes.POINTER(_double11),            # eflux_thHS
        # DKE
        ctypes.POINTER(_double11),            # pflux_dke
        ctypes.POINTER(_double11),            # efluxtot_dke
        ctypes.POINTER(_double11),            # efluxncv_dke
        ctypes.POINTER(_double11),            # mflux_dke
        ctypes.POINTER(_double11),            # vpol_dke
        ctypes.POINTER(_double11),            # vtor_dke
        ctypes.POINTER(ctypes.c_double),      # jpar_dke
        ctypes.POINTER(ctypes.c_double),      # jtor_dke
        # Gyro-viscosity
        ctypes.POINTER(_double11),            # pflux_gv
        ctypes.POINTER(_double11),            # efluxtot_gv
        ctypes.POINTER(_double11),            # efluxncv_gv
        ctypes.POINTER(_double11),            # mflux_gv
        # NCLASS
        ctypes.POINTER(_double11),            # nclassvis
        ctypes.POINTER(_double11),            # pflux_nclass
        ctypes.POINTER(_double11),            # efluxtot_nclass
        ctypes.POINTER(_double11),            # vpol_nclass
        ctypes.POINTER(_double11),            # vtor_nclass
        ctypes.POINTER(ctypes.c_double),      # jpar_nclass
        # Geometry
        ctypes.POINTER(_double5),             # geoparams
        # Error status
        ctypes.POINTER(ctypes.c_int),         # error_status
    ]
    return lib


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"libneo_serial.so not found at {_LIB_PATH}\n"
            "  Build it once with:\n"
            f"    cd {_INTERFACES_DIR} && bash build_neo_lib.sh"
        )

    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load {_LIB_PATH}: {exc}\n"
            "Check that the library was compiled for the current platform."
        ) from exc

    _lib = _setup_lib_signatures(lib)
    return _lib


def _load_unique_lib() -> ctypes.CDLL:
    """
    Load a PRIVATE copy of the shared library so each thread gets independent
    Fortran module-level globals.  See tglf_inprocess.py for rationale.
    """
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"libneo_serial.so not found at {_LIB_PATH}\n"
            "  Build it once with:\n"
            f"    cd {_INTERFACES_DIR} && bash build_neo_lib.sh"
        )
    fd, tmp_path = tempfile.mkstemp(suffix="_neo_thread.so")
    os.close(fd)
    shutil.copy(str(_LIB_PATH), tmp_path)
    with _temp_lib_lock:
        _temp_lib_paths.append(tmp_path)
    try:
        lib = ctypes.CDLL(tmp_path)
    except OSError as exc:
        raise RuntimeError(f"Failed to load private lib copy {tmp_path}: {exc}") from exc
    return _setup_lib_signatures(lib)


def _get_thread_lib() -> ctypes.CDLL:
    """Return the calling thread's private library instance, creating it on first use."""
    if not hasattr(_thread_local, "lib"):
        _thread_local.lib = _load_unique_lib()
    return _thread_local.lib


# ---------------------------------------------------------------------------
# Direct module-variable access — maps dict keys → Fortran neo_interface vars
# ---------------------------------------------------------------------------
#
# gfortran mangles  module::var  →  __module_MOD_var
# All neo_interface inputs are named  neo_<var>_in  (one exception:
# neo_subroutine_flag has no `_in` suffix).
#
def _sym(var: str) -> str:
    if var == "subroutine_flag":
        return "__neo_interface_MOD_neo_subroutine_flag"
    return f"__neo_interface_MOD_neo_{var}_in"


# Per-species real arrays (dim 11)
_SPECIES_REAL_BASES = frozenset([
    "z", "mass", "dens", "temp", "dlnndr", "dlntdr",
    "temp_para", "dlntdr_para", "temp_perp", "dlntdr_perp",
    "profile_dlnndr_scale", "profile_dlntdr_scale",
])

# Per-species integer arrays (dim 11)
_SPECIES_INT_BASES = frozenset([
    "aniso_model",
])

# Scalar Fortran INTEGER variables (stored as 4-byte integer)
_INTEGERS = frozenset([
    "n_energy", "n_xi", "n_theta", "n_radial", "matsz_scalefac",
    "silent_flag", "sim_model", "equilibrium_model", "collision_model",
    "profile_model", "profile_erad0_model", "ipccw", "btccw",
    "rotation_model", "spitzer_model",
    "coll_uncoupledei_model", "coll_uncoupledaniso_model",
    "ae_flag", "n_species", "threed_model", "threed_exb_model",
    "threed_drift_model", "laguerre_method", "write_cmoments_flag",
    "subroutine_flag", "test_flag",
])


def _zero_species_arrays(lib: ctypes.CDLL) -> None:
    """
    Zero all per-species arrays so stale values from a prior call don't
    persist.  Integer arrays with required default values (e.g.
    aniso_model = 1 = isotropic) are reset to that default rather than 0,
    so the input dict is allowed to omit them.
    """
    for base in _SPECIES_REAL_BASES:
        arr = (ctypes.c_double * _NSM).in_dll(lib, _sym(base))
        ctypes.memset(arr, 0, ctypes.sizeof(arr))
    # Integer arrays — reset to NEO defaults (aniso_model = 1 means
    # isotropic single-temperature; 0 is invalid).
    arr = (ctypes.c_int * _NSM).in_dll(lib, _sym("aniso_model"))
    for k in range(_NSM):
        arr[k] = 1


def _set_inputs_from_dict(lib: ctypes.CDLL, input_dict: dict) -> None:
    """
    Write every entry in *input_dict* directly into the neo_interface Fortran
    module variables.  This replaces the ``neo_read_input()`` file-read.

    Key conventions (matching ``PROFILES_GACODE.to_neo()`` output):
    - Scalar controls / plasma:  ``"NU_1"``, ``"RMIN_OVER_A"``, ``"SIM_MODEL"`` …
    - Species arrays:            ``"Z_1"``, ``"DLNTDR_2"`` … (1-based index)
    - Profile-scale arrays:      ``"PROFILE_DLNNDR_1_SCALE"`` (index in the middle)
    - Shape harmonics:           ``"SHAPE_COS0"``, ``"SHAPE_S_SIN3"`` …
    - rbf_dir:                   ``"RBF_DIR"`` → ``character(len=16)``
    """
    _zero_species_arrays(lib)

    for key, val in input_dict.items():
        ku = key.upper()
        kl = ku.lower()

        # ---- profile_<base>_N_scale → profile_<base>_scale[N-1] ----
        if kl.startswith("profile_") and kl.endswith("_scale"):
            mid = kl[len("profile_"):-len("_scale")]
            head, _, tail = mid.rpartition("_")
            if tail.isdigit():
                base = f"profile_{head}_scale"
                if base in _SPECIES_REAL_BASES:
                    idx = int(tail) - 1
                    arr = (ctypes.c_double * _NSM).in_dll(lib, _sym(base))
                    arr[idx] = float(val)
                    continue

        # ---- species array: KEY_N (e.g. Z_1, DLNTDR_2) ----
        head, _, tail = kl.rpartition("_")
        if tail.isdigit():
            idx = int(tail) - 1
            if head in _SPECIES_REAL_BASES:
                arr = (ctypes.c_double * _NSM).in_dll(lib, _sym(head))
                arr[idx] = float(val)
                continue
            if head in _SPECIES_INT_BASES:
                arr = (ctypes.c_int * _NSM).in_dll(lib, _sym(head))
                arr[idx] = int(val)
                continue

        # ---- scalar variable ----
        var = kl
        # alias: NEO uses RMIN_OVER_A_2 in the .gen file → neo_rmin_over_a_2_in
        # all other names map directly.

        sym = _sym(var)

        if var == "rbf_dir":
            CharT = ctypes.c_char * 16
            s = CharT.in_dll(lib, sym)
            enc = str(val).encode("ascii")[:16].ljust(16)
            ctypes.memmove(s, enc, 16)

        elif var in _INTEGERS:
            ctypes.c_int.in_dll(lib, sym).value = int(val)

        else:
            # real — c_double because the library was built with -fdefault-real-8
            try:
                ctypes.c_double.in_dll(lib, sym).value = float(val)
            except OSError:
                pass   # unknown key (e.g. a MITIM-only field) — skip silently

    # Force quiet AFTER applying the dict, so an explicit SILENT_FLAG=0
    # in the input does not re-enable file output (interfacelocaldump
    # would otherwise try to open out.neo.localdump in cwd).
    ctypes.c_int.in_dll(lib, _sym("silent_flag")).value = 1


def _collect_outputs(lib: ctypes.CDLL) -> dict:
    """Read neo_*_out variables via c_neo_get_outputs and return a dict."""
    ns_out         = ctypes.c_int(0)

    pflux_thHH     = ctypes.c_double(0.0)
    eflux_thHHi    = ctypes.c_double(0.0)
    eflux_thHHe    = ctypes.c_double(0.0)
    eflux_thCHi    = ctypes.c_double(0.0)
    jpar_thS       = ctypes.c_double(0.0)
    jpar_thK       = ctypes.c_double(0.0)
    jpar_thN       = ctypes.c_double(0.0)
    jtor_thS       = ctypes.c_double(0.0)
    jpar_thSmod    = ctypes.c_double(0.0)
    jtor_thSmod    = ctypes.c_double(0.0)

    z11 = lambda: _double11(*([0.0] * _NSM))
    pflux_thHS     = z11()
    eflux_thHS     = z11()

    pflux_dke      = z11()
    efluxtot_dke   = z11()
    efluxncv_dke   = z11()
    mflux_dke      = z11()
    vpol_dke       = z11()
    vtor_dke       = z11()
    jpar_dke       = ctypes.c_double(0.0)
    jtor_dke       = ctypes.c_double(0.0)

    pflux_gv       = z11()
    efluxtot_gv    = z11()
    efluxncv_gv    = z11()
    mflux_gv       = z11()

    nclassvis      = z11()
    pflux_nclass   = z11()
    efluxtot_nclass= z11()
    vpol_nclass    = z11()
    vtor_nclass    = z11()
    jpar_nclass    = ctypes.c_double(0.0)

    geoparams      = _double5(*([0.0] * _NGEO))
    error_status   = ctypes.c_int(0)

    lib.c_neo_get_outputs(
        ctypes.byref(ns_out),
        ctypes.byref(pflux_thHH),
        ctypes.byref(eflux_thHHi),
        ctypes.byref(eflux_thHHe),
        ctypes.byref(eflux_thCHi),
        ctypes.byref(jpar_thS),
        ctypes.byref(jpar_thK),
        ctypes.byref(jpar_thN),
        ctypes.byref(jtor_thS),
        ctypes.byref(jpar_thSmod),
        ctypes.byref(jtor_thSmod),
        ctypes.byref(pflux_thHS),
        ctypes.byref(eflux_thHS),
        ctypes.byref(pflux_dke),
        ctypes.byref(efluxtot_dke),
        ctypes.byref(efluxncv_dke),
        ctypes.byref(mflux_dke),
        ctypes.byref(vpol_dke),
        ctypes.byref(vtor_dke),
        ctypes.byref(jpar_dke),
        ctypes.byref(jtor_dke),
        ctypes.byref(pflux_gv),
        ctypes.byref(efluxtot_gv),
        ctypes.byref(efluxncv_gv),
        ctypes.byref(mflux_gv),
        ctypes.byref(nclassvis),
        ctypes.byref(pflux_nclass),
        ctypes.byref(efluxtot_nclass),
        ctypes.byref(vpol_nclass),
        ctypes.byref(vtor_nclass),
        ctypes.byref(jpar_nclass),
        ctypes.byref(geoparams),
        ctypes.byref(error_status),
    )

    ns = int(ns_out.value)
    return {
        "ns":              ns,
        # ---- theory ----
        "pflux_thHH":      float(pflux_thHH.value),
        "eflux_thHHi":     float(eflux_thHHi.value),
        "eflux_thHHe":     float(eflux_thHHe.value),
        "eflux_thCHi":     float(eflux_thCHi.value),
        "jpar_thS":        float(jpar_thS.value),
        "jpar_thK":        float(jpar_thK.value),
        "jpar_thN":        float(jpar_thN.value),
        "jtor_thS":        float(jtor_thS.value),
        "jpar_thSmod":     float(jpar_thSmod.value),
        "jtor_thSmod":     float(jtor_thSmod.value),
        "pflux_thHS":      list(pflux_thHS[:ns]),
        "eflux_thHS":      list(eflux_thHS[:ns]),
        # ---- DKE ----
        "pflux_dke":       list(pflux_dke[:ns]),
        "efluxtot_dke":    list(efluxtot_dke[:ns]),
        "efluxncv_dke":    list(efluxncv_dke[:ns]),
        "mflux_dke":       list(mflux_dke[:ns]),
        "vpol_dke":        list(vpol_dke[:ns]),
        "vtor_dke":        list(vtor_dke[:ns]),
        "jpar_dke":        float(jpar_dke.value),
        "jtor_dke":        float(jtor_dke.value),
        # ---- gyro-viscosity ----
        "pflux_gv":        list(pflux_gv[:ns]),
        "efluxtot_gv":     list(efluxtot_gv[:ns]),
        "efluxncv_gv":     list(efluxncv_gv[:ns]),
        "mflux_gv":        list(mflux_gv[:ns]),
        # ---- NCLASS ----
        "nclassvis":       list(nclassvis[:ns]),
        "pflux_nclass":    list(pflux_nclass[:ns]),
        "efluxtot_nclass": list(efluxtot_nclass[:ns]),
        "vpol_nclass":     list(vpol_nclass[:ns]),
        "vtor_nclass":     list(vtor_nclass[:ns]),
        "jpar_nclass":     float(jpar_nclass.value),
        # ---- geometry ----
        "geoparams":       list(geoparams[:]),
        # ---- status ----
        "error_status":    int(error_status.value),
    }


# ---------------------------------------------------------------------------
# Input preprocessing: input.neo → input.neo.gen  (file-based path)
# ---------------------------------------------------------------------------

def generate_input_gen(input_neo_path: str | Path) -> Path:
    """
    Convert an ``input.neo`` file to ``input.neo.gen`` using GACODE's
    ``neo_parse.py`` logic.  Requires ``GACODE_ROOT`` to be set.
    """
    input_neo_path = Path(input_neo_path).resolve()
    if not input_neo_path.exists():
        raise FileNotFoundError(f"input.neo not found: {input_neo_path}")

    gacode_root = os.environ.get("GACODE_ROOT")
    if not gacode_root:
        raise RuntimeError(
            "GACODE_ROOT environment variable is not set. "
            "Source your gacode_setup script before using neo_inprocess."
        )

    # neo_parse.py uses argv[0] dirname-style discovery; the simplest way to
    # invoke it correctly is to import it from neo/bin and run it from a cwd
    # that contains the input.neo file (it hardcodes 'input.neo').
    neo_bin = Path(gacode_root) / "neo" / "bin"
    parser_path = neo_bin / "neo_parse.py"
    if not parser_path.exists():
        raise FileNotFoundError(f"neo_parse.py not found: {parser_path}")

    # neo_parse.py reads 'input.neo' from cwd and writes 'input.neo.gen'.
    # Use a subprocess so its sys.exit() doesn't kill the host process, and
    # so its hard-coded relative paths work.
    import subprocess
    result = subprocess.run(
        ["python", str(parser_path)],
        cwd=str(input_neo_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"NEO input parsing failed for {input_neo_path}:\n"
            f"  stdout: {result.stdout}\n  stderr: {result.stderr}"
        )

    gen_path = input_neo_path.with_name("input.neo.gen")
    if not gen_path.exists():
        raise RuntimeError(f"neo_parse produced no output — expected {gen_path}")
    return gen_path


# ---------------------------------------------------------------------------
# Output writer: outputs dict → out.neo.transport_flux (subset)
# ---------------------------------------------------------------------------

def write_transport_flux(outputs: dict, filepath: str | Path, roa: float = 0.5) -> None:
    """
    Write a minimal ``out.neo.transport_flux`` from the output dict returned
    by :meth:`NEOInProcess.run` or :meth:`NEOInProcess.run_from_dict`.

    The on-disk format produced by NEO is structured into three sections —
    DKE, GV, and TGYRO (= DKE + GV).  Each section starts with a comment
    line and contains one row per species: ``Z  Gamma  Q  Pi``.

    This is intentionally a small writer; full output reconstruction would
    require regenerating all the per-radius/per-theta files.
    """
    ns = outputs["ns"]

    # We only know per-species DKE and GV; reconstruct Z from neo input by
    # asking the caller to pass it in via outputs (or default to integer
    # placeholder).  For the typical use of this writer (smoke-tests), the
    # caller passes a dict that came from to_neo() and Z is not retained
    # here, so we just emit zeros for the Z column.
    Z = [0.0] * ns

    def _section(header: str, G, Q, M):
        out = [f"#  {header}"]
        for k in range(ns):
            out.append(f" {Z[k]:6.3f}  {G[k]:13.5e}  {Q[k]:13.5e}  {M[k]:13.5e}")
        return "\n".join(out)

    G_dke = outputs["pflux_dke"]
    Q_dke = outputs["efluxtot_dke"]
    M_dke = outputs["mflux_dke"]
    G_gv  = outputs["pflux_gv"]
    Q_gv  = outputs["efluxtot_gv"]
    M_gv  = outputs["mflux_gv"]
    G_tg  = [a + b for a, b in zip(G_dke, G_gv)]
    Q_tg  = [a + b for a, b in zip(Q_dke, Q_gv)]
    M_tg  = [a + b for a, b in zip(M_dke, M_gv)]

    text = "\n".join([
        f" r/a {roa:13.5e}",
        _section("pflux_dke",   G_dke, Q_dke, M_dke),
        _section("pflux_gv",    G_gv,  Q_gv,  M_gv),
        _section("pflux_tgyro", G_tg,  Q_tg,  M_tg),
        "",
    ])
    Path(filepath).write_text(text)


# ---------------------------------------------------------------------------
# Main in-process runner
# ---------------------------------------------------------------------------

class NEOInProcess:
    """
    Run NEO in-process via ctypes — no subprocess fork, no physics file I/O.

    Three usage modes:

    **Standalone / fully in-memory** (preferred)::

        runner = NEOInProcess()
        runner.prepare("path/to/input.gacode", rhos=[0.4, 0.6], code_settings="Sonic")
        results = runner.run_all()   # {rho: output_dict} — zero file I/O
        out     = runner.run_rho(0.4)

    **Dict-based** (when you already have a flat input dict)::

        neo_inputs = profiles.to_neo(r=rhos, code_settings="Sonic")
        out = runner.run_from_dict(neo_inputs[0.5])

    **File-based** (legacy, requires GACODE_ROOT)::

        gen = generate_input_gen("/work/run1/input.neo")
        out = runner.run(gen)

    Sequential calls are safe; for parallel use thread-private library copies
    via ``_get_thread_lib()`` (see ``_parallel_worker``).
    """

    def __init__(self) -> None:
        self._lib = _load_lib()
        self._inputs: dict = {}  # {float(rho): flat_dict} — populated by prepare()

    # ------------------------------------------------------------------
    # Standalone in-memory path
    # ------------------------------------------------------------------

    def prepare(
        self,
        profiles,
        rhos: list = [0.5],
        code_settings = "Sonic",
        extraOptions: dict = {},
        multipliers: dict = {},
    ) -> None:
        """
        Build and cache NEO input dicts in memory from a PROFILES_GACODE object
        or an ``input.gacode`` file path.

        Calls ``profiles.to_neo()`` once, applies *multipliers* and
        *extraOptions*, and stores the flat dicts in ``self._inputs``.
        After this call, :meth:`run_all` and :meth:`run_rho` are zero file I/O.
        """
        if isinstance(profiles, (str, Path)):
            from mitim_tools.gacode_tools.PROFILEStools import gacode_state
            profiles = gacode_state(str(profiles))

        raw = profiles.to_neo(r=list(rhos), code_settings=code_settings)

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
        """Run NEO for every rho prepared by :meth:`prepare`."""
        if not self._inputs:
            raise RuntimeError("No inputs prepared — call prepare() first.")
        return {rho: self.run_from_dict(d) for rho, d in self._inputs.items()}

    def run_rho(self, rho: float) -> dict:
        """Run NEO for a single rho prepared by :meth:`prepare`."""
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
        Execute NEO reading inputs from ``input.neo.gen`` on disk.

        Parameters
        ----------
        gen_file_path:
            Path to ``input.neo.gen`` produced by :func:`generate_input_gen`,
            **or** the directory containing it.
        """
        gen_file_path = Path(gen_file_path).resolve()
        if gen_file_path.is_dir():
            gen_dir = gen_file_path
        else:
            gen_dir = gen_file_path.parent
        if not (gen_dir / "input.neo.gen").exists():
            raise FileNotFoundError(f"input.neo.gen not found in {gen_dir}")

        # NEO's `path` global is character(len=80), so we cannot store an
        # absolute path longer than that.  Instead, chdir into the run
        # directory and pass "./" as the path — NEO then opens
        # "./input.neo.gen" which is always short enough.
        prev_cwd = os.getcwd()
        try:
            os.chdir(str(gen_dir))
            self._lib.c_neo_set_path(b"./")
            self._lib.c_neo_read_input()
            self._lib.c_neo_run()
            return _collect_outputs(self._lib)
        finally:
            os.chdir(prev_cwd)

    # ------------------------------------------------------------------
    # In-memory path
    # ------------------------------------------------------------------

    def run_from_dict(self, input_dict: dict) -> dict:
        """
        Execute NEO setting all inputs directly from *input_dict* — no file I/O.

        *input_dict* is the value for one rho from ``PROFILES_GACODE.to_neo()``.
        Species arrays use 1-based index suffixes (``"Z_1"``, ``"DLNTDR_2"``…).
        """
        # Set a benign path so any internal file paths NEO might touch are
        # written to /tmp rather than the user's cwd.  This is mostly defensive.
        if not getattr(self, "_path_set", False):
            tmp = tempfile.mkdtemp(prefix="neo_inproc_")
            self._lib.c_neo_set_path((tmp + "/").encode("ascii"))
            self._path_set = True

        _set_inputs_from_dict(self._lib, input_dict)
        self._lib.c_neo_run()
        return _collect_outputs(self._lib)


# ------------------------------------------------------------------
# Module-level worker — picklable / thread-safe entry point.
# ------------------------------------------------------------------

def _parallel_worker(flat: dict) -> dict:
    """
    Run one NEO case on the calling thread's private library instance.

    Each worker thread loads its own independent copy of the shared library
    (via ``_get_thread_lib()``), giving isolated Fortran module-level globals.
    """
    lib = _get_thread_lib()
    _set_inputs_from_dict(lib, flat)
    lib.c_neo_run()
    return _collect_outputs(lib)
