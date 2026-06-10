#!/usr/bin/env bash
# =============================================================================
# build_neo_lib.sh
#
# Build libneo_serial.so — a shared library for in-process NEO execution.
#
# Strategy
# --------
# NEO is already compiled by the GACODE build system in ${GACODE_ROOT}/neo/src/.
# We reuse those pre-built object files (which include -fPIC from the GACODE
# platform config) and compile only our thin C-binding wrapper neo_c_api.f90.
#
# Excluded objects (they call MPI primitives):
#   neo.o          — main program
#   neo_init.o     — parallel initializer (uses MPI_COMM_RANK / SIZE)
#
# We keep neo_init_serial.o so the wrapper can set i_proc/n_proc/path.
# Several other NEO files `use mpi` (neo_do, neo_neural.fann) for the module
# only — they don't actually call any MPI routines, so linking against an MPI
# wrapper is enough to satisfy any residual symbols.
#
# Prerequisites
# -------------
#   - GACODE_ROOT set and gacode already built (neo/src/*.o must exist)
#   - mpifort / mpif90 (provided by openmpi-mpifort in the pixi env)
#   - liblapack + libblas + libfftw3
#
# Usage
# -----
#   cd <this directory>
#   bash build_neo_lib.sh            # build
#   bash build_neo_lib.sh --clean    # remove build artefacts and library
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- validate environment ----------------------------------------------------
if [[ -z "${GACODE_ROOT:-}" ]]; then
    echo "ERROR: GACODE_ROOT is not set. Source your gacode_setup script first."
    exit 1
fi

NEO_SRC="${GACODE_ROOT}/neo/src"
GACODE_MOD="${GACODE_ROOT}/modules"

if [[ ! -d "${NEO_SRC}" ]]; then
    echo "ERROR: NEO source directory not found: ${NEO_SRC}"
    exit 1
fi

REQUIRED_OBJ="${NEO_SRC}/neo_run.o"
if [[ ! -f "${REQUIRED_OBJ}" ]]; then
    echo "ERROR: ${REQUIRED_OBJ} not found."
    echo "       Build GACODE first: cd \${GACODE_ROOT} && make"
    exit 1
fi

# --- optional clean ----------------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning build artefacts (neo_build/)..."
    rm -rf "${SCRIPT_DIR}/neo_build"
    echo "Done."
    exit 0
fi

# --- detect Fortran compiler -------------------------------------------------
# Prefer the MPI wrapper because several NEO objects `use mpi` and a few of
# the gacode shared libs (expro_comm.o) actually call mpi_bcast / mpi_comm_*
# at link time.  We force the MPI wrapper unless the user explicitly opts
# out via NEO_FC=...
if [[ -n "${NEO_FC:-}" ]]; then
    FC="${NEO_FC}"
elif command -v mpifort &>/dev/null; then
    FC=mpifort
elif command -v mpif90 &>/dev/null; then
    FC=mpif90
else
    echo "ERROR: no MPI Fortran wrapper found (mpifort / mpif90)."
    echo "       In a pixi env, ensure openmpi-mpifort is installed."
    exit 1
fi

# --- detect library prefix ---------------------------------------------------
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    LIB_DIRS=(-L"${CONDA_PREFIX}/lib")
elif [[ -n "${PREFIX:-}" ]]; then
    LIB_DIRS=(-L"${PREFIX}/lib")
else
    LIB_DIRS=(-L/usr/local/lib -L/usr/lib)
fi

# All build artefacts (including the .so) live in neo_build/ — gitignored.
BUILD_DIR="${SCRIPT_DIR}/neo_build"
mkdir -p "${BUILD_DIR}"

echo "Building libneo_serial.so"
echo "  FC         : $FC"
echo "  NEO_SRC    : $NEO_SRC  (pre-built .o files)"
echo "  OUTPUT     : ${BUILD_DIR}/libneo_serial.so"

# ---------------------------------------------------------------------------
# Compile neo_c_api.f90  (only this file needs compiling)
# Flags must match the GACODE build: -fdefault-real-8 makes REAL == c_double.
# -I${GACODE_MOD} provides pre-built .mod files (neo_interface.mod etc.)
# ---------------------------------------------------------------------------
echo "  Compiling neo_c_api.f90 ..."
"$FC" -O2 -fPIC \
    -fdefault-real-8 -fdefault-double-8 \
    -fallow-argument-mismatch \
    -fall-intrinsics \
    -I"${GACODE_MOD}" \
    -J"${BUILD_DIR}" \
    -c "${SCRIPT_DIR}/neo_c_api.f90" \
    -o "${BUILD_DIR}/neo_c_api.o"

# ---------------------------------------------------------------------------
# Collect the pre-built NEO object files
# Excluded: neo.o (main program), neo_init.o (calls MPI_COMM_RANK/SIZE)
# ---------------------------------------------------------------------------
OBJECTS=()
for name in \
    neo_globals \
    neo_energy_grid \
    neo_interface \
    neo_allocate_profile \
    neo_umfpack \
    neo_sparse_solve \
    neo_equilibrium \
    neo_g_velocitygrids \
    neo_rotation \
    neo_nclass_dr \
    neo_theory \
    neo_transport \
    neo_3d_driver \
    neo_check \
    neo_compute_fcoll \
    neo_neural \
    neo_do \
    neo_error \
    neo_init_serial \
    neo_make_profiles \
    neo_read_input \
    neo_run \
    neo_spitzer \
    matconv; do

    obj="${NEO_SRC}/${name}.o"
    if [[ ! -f "$obj" ]]; then
        echo "ERROR: pre-built object not found: $obj"
        echo "       Rebuild GACODE: cd \${GACODE_ROOT} && make"
        exit 1
    fi
    OBJECTS+=("$obj")
done
OBJECTS+=("${BUILD_DIR}/neo_c_api.o")

# ---------------------------------------------------------------------------
# Pull in the gacode shared static archives that NEO depends on.
# These mirror EXTRA_LIBS from gacode/neo/Makefile.
# ---------------------------------------------------------------------------
EXTRA_LIBS=(
    "${GACODE_ROOT}/f2py/expro/expro_lib.a"
    "${GACODE_ROOT}/f2py/geo/geo_lib.a"
    "${GACODE_ROOT}/shared/math/math_lib.a"
    "${GACODE_ROOT}/shared/UMFPACK/UMFPACK_lib.a"
    "${GACODE_ROOT}/shared/nclass/nclass_lib.a"
)
for a in "${EXTRA_LIBS[@]}"; do
    if [[ ! -f "$a" ]]; then
        echo "ERROR: required gacode archive not found: $a"
        echo "       Build GACODE first: cd \${GACODE_ROOT} && make"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Link shared library
# ---------------------------------------------------------------------------
echo "  Linking libneo_serial.so ..."
"$FC" -shared -fPIC \
    "${OBJECTS[@]}" \
    "${EXTRA_LIBS[@]}" \
    "${LIB_DIRS[@]}" \
    -llapack -lblas \
    -o "${BUILD_DIR}/libneo_serial.so"

echo ""
echo "SUCCESS: ${BUILD_DIR}/libneo_serial.so"
ls -lh "${BUILD_DIR}/libneo_serial.so"
