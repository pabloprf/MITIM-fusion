"""
vgen_inprocess.py
=================
In-process VGEN execution via a ctypes-loaded shared library
(``libvgen_serial.so``).

The standard PORTALS path runs ``profiles_gen -vgen`` as a subprocess via
``mitim_job`` (SLURM submission, tarballing, etc.).  The in-process path
loads the same Fortran physics directly into the Python process and
runs every radial surface sequentially in a single thread — no SLURM,
no tarball/transfer overhead, no shell scripts.

VGEN still needs an ``input.gacode`` file on disk because gacode's
``expro_read`` reads from a file path; the wrapper does NOT change that.
What it eliminates is the job-manager wrapping around the call.

API
---
::

    from mitim_tools.simulation_tools.interfaces.vgen_inprocess import VGENInProcess

    runner = VGENInProcess()
    runner.run(
        folder         = "/path/to/work_dir",   # must contain input.gacode
        er_method      = 2,
        vel_method     = 1,
        erspecies_indx = 1,
        nth_min        = 17,
        nth_max        = 39,
        n_species      = 5,
    )
    # ⇒ folder/vgen/input.gacode now exists with NEO-computed w0 populated

Prerequisites
-------------
Build the shared library once per machine::

    cd <MITIM-fusion>/src/mitim_tools/simulation_tools/interfaces
    bash build_vgen_lib.sh
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# All build artefacts live in vgen_build/ (gitignored) inside this directory.
# ---------------------------------------------------------------------------
_INTERFACES_DIR = Path(__file__).parent
_LIB_PATH = _INTERFACES_DIR / "vgen_build" / "libvgen_serial.so"

# ---------------------------------------------------------------------------
# Lazy singleton: load libvgen_serial.so once per process.
# VGEN runs sequentially over surfaces and PORTALS calls it once per
# iteration, so a single shared instance is enough — no thread-private
# copies like the TGLF / NEO wrappers need.
# ---------------------------------------------------------------------------
_lib: Any = None


def _setup_lib_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.c_vgen_set_path.restype  = None
    lib.c_vgen_set_path.argtypes = [ctypes.c_char_p]

    lib.c_vgen_run.restype  = None
    lib.c_vgen_run.argtypes = [
        ctypes.c_int,   # er_method
        ctypes.c_int,   # vel_method
        ctypes.c_int,   # erspecies_indx
        ctypes.c_int,   # nth_min
        ctypes.c_int,   # nth_max
        ctypes.c_int,   # n_species
    ]
    return lib


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"libvgen_serial.so not found at {_LIB_PATH}\n"
            "  Build it once with:\n"
            f"    cd {_INTERFACES_DIR} && bash build_vgen_lib.sh"
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class VGENInProcess:
    """
    Run gacode's ``vgen`` (velocity generation) workflow in-process.

    Usage::

        runner = VGENInProcess()
        runner.run(folder, er_method=2, vel_method=1, erspecies_indx=1,
                   nth_min=17, nth_max=39, n_species=5)
        # → <folder>/vgen/input.gacode now exists with NEO-computed w0
    """

    def __init__(self) -> None:
        self._lib = _load_lib()

    def run(
        self,
        folder,
        er_method: int      = 2,
        vel_method: int     = 1,
        erspecies_indx: int = 1,
        nth_min: int        = 17,
        nth_max: int        = 39,
        n_species: int      = 5,
    ) -> Path:
        """
        Run vgen on ``<folder>/input.gacode`` and write
        ``<folder>/vgen/input.gacode`` with the populated w0(rad/s).

        Parameters
        ----------
        folder:
            Working directory.  Must already contain ``input.gacode``.
        er_method:
            How vgen computes Er.  Currently only ``2`` (NEO weak rotation
            limit) is wired through; this is the option PORTALS uses.
        vel_method:
            Velocity method.  Currently only ``1`` (NEO weak rotation) is
            wired through, matching er_method=2; ``2`` (strong rotation)
            is not implemented in-process.
        erspecies_indx:
            1-based index of the ion species to match for Er computation.
        nth_min, nth_max:
            Min / max poloidal theta resolution.
        n_species:
            Number of NEO species (sets ``neo_n_species_in``).

        Returns
        -------
        Path to the generated ``vgen/input.gacode`` file.
        """
        folder = Path(folder).resolve()
        if not (folder / "input.gacode").exists():
            raise FileNotFoundError(f"input.gacode not found in {folder}")

        # Guard here as well as in the Fortran wrapper so an already-built
        # (older) libvgen_serial.so cannot silently return weak-rotation
        # results for an unsupported request.
        if int(er_method) != 2 or int(vel_method) != 1:
            raise NotImplementedError(
                f"In-process VGEN supports only er_method=2 / vel_method=1 (NEO weak rotation); "
                f"got er_method={er_method}, vel_method={vel_method}. Use in_process=False for other methods."
            )

        # vgen always writes its output into a `vgen/` subdirectory of the
        # cwd.  Ensure it exists before calling the Fortran routine.
        (folder / "vgen").mkdir(parents=True, exist_ok=True)

        # The Fortran wrapper opens files relative to cwd (and stores the
        # path in neo/vgen globals for any code that consults it).  chdir
        # so expro_read('input.gacode') and expro_write('vgen/input.gacode')
        # land in the right place.
        prev_cwd = os.getcwd()
        try:
            os.chdir(str(folder))
            self._lib.c_vgen_set_path(b"./")
            self._lib.c_vgen_run(
                ctypes.c_int(int(er_method)),
                ctypes.c_int(int(vel_method)),
                ctypes.c_int(int(erspecies_indx)),
                ctypes.c_int(int(nth_min)),
                ctypes.c_int(int(nth_max)),
                ctypes.c_int(int(n_species)),
            )
        finally:
            os.chdir(prev_cwd)

        out = folder / "vgen" / "input.gacode"
        if not out.exists():
            raise RuntimeError(
                f"vgen completed but {out} was not produced — check NEO error output."
            )
        return out
