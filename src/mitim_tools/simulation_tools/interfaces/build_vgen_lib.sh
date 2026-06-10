#!/usr/bin/env bash
# =============================================================================
# build_vgen_lib.sh
#
# Build libvgen_serial.so — a shared library that runs the gacode "vgen"
# velocity-generation workflow in-process via ctypes (vgen_inprocess.py).
#
# Strategy
# --------
# Compile only the thin C-binding wrapper (vgen_c_api.f90) and link it
# against the prebuilt vgen objects (vgen_globals.o, vgen_compute_neo.o,
# vgen_getgeo.o) plus the same gacode static archives the standard vgen
# binary uses (neo_lib.a, expro_lib.a, geo_lib.a, math_lib.a, UMFPACK_lib.a,
# nclass_lib.a).
#
# Excluded objects
# ----------------
#   vgen.o         — main program (calls MPI_INIT directly)
#   vgen_init.o    — uses MPI_finalize on errors and hard-codes MPI_COMM_WORLD;
#                    its body is reproduced inside vgen_c_api.f90 with comm=0
#   vgen_reduce.o  — calls MPI_ALLREDUCE; with one rank we just skip it
#
# vgen_compute_neo.o uses MPI_Wtime() but with timing_flag=0 (set in
# vgen_globals) the timer is read but never written, so a stub library
# isn't needed.  We still link via the MPI Fortran wrapper to satisfy the
# residual `use mpi` references that the gacode shared libs carry.
#
# Prerequisites
# -------------
#   - GACODE_ROOT set and gacode already built (vgen/src/*.o + neo/src/*.o)
#   - mpifort / mpif90 (provided by openmpi-mpifort in the pixi env)
#   - liblapack + libblas
#
# Usage
# -----
#   bash build_vgen_lib.sh            # build
#   bash build_vgen_lib.sh --clean    # remove build artefacts and library
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- validate environment ----------------------------------------------------
if [[ -z "${GACODE_ROOT:-}" ]]; then
    echo "ERROR: GACODE_ROOT is not set. Source your gacode_setup script first."
    exit 1
fi

VGEN_SRC="${GACODE_ROOT}/vgen/src"
GACODE_MOD="${GACODE_ROOT}/modules"

if [[ ! -d "${VGEN_SRC}" ]]; then
    echo "ERROR: vgen source directory not found: ${VGEN_SRC}"
    exit 1
fi

REQUIRED_OBJ="${VGEN_SRC}/vgen_compute_neo.o"
if [[ ! -f "${REQUIRED_OBJ}" ]]; then
    echo "ERROR: ${REQUIRED_OBJ} not found."
    echo "       Build GACODE first: cd \${GACODE_ROOT} && make"
    exit 1
fi

# --- optional clean ----------------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning build artefacts (vgen_build/)..."
    rm -rf "${SCRIPT_DIR}/vgen_build"
    echo "Done."
    exit 0
fi

# --- detect Fortran compiler -------------------------------------------------
# Force the MPI wrapper unless the user explicitly opts out via VGEN_FC=...
if [[ -n "${VGEN_FC:-}" ]]; then
    FC="${VGEN_FC}"
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

# All build artefacts (including the .so) live in vgen_build/ — gitignored.
BUILD_DIR="${SCRIPT_DIR}/vgen_build"
mkdir -p "${BUILD_DIR}"

echo "Building libvgen_serial.so"
echo "  FC         : $FC"
echo "  VGEN_SRC   : $VGEN_SRC  (pre-built .o files)"
echo "  OUTPUT     : ${BUILD_DIR}/libvgen_serial.so"

# ---------------------------------------------------------------------------
# Compile vgen_c_api.f90
# ---------------------------------------------------------------------------
echo "  Compiling vgen_c_api.f90 ..."
"$FC" -O2 -fPIC \
    -fdefault-real-8 -fdefault-double-8 \
    -fallow-argument-mismatch \
    -fall-intrinsics \
    -I"${GACODE_MOD}" \
    -J"${BUILD_DIR}" \
    -c "${SCRIPT_DIR}/vgen_c_api.f90" \
    -o "${BUILD_DIR}/vgen_c_api.o"

# ---------------------------------------------------------------------------
# Collect the prebuilt vgen object files we need
# Excluded: vgen.o, vgen_init.o, vgen_reduce.o (see header comment)
# ---------------------------------------------------------------------------
OBJECTS=()
for name in vgen_globals vgen_compute_neo vgen_getgeo; do
    obj="${VGEN_SRC}/${name}.o"
    if [[ ! -f "$obj" ]]; then
        echo "ERROR: pre-built object not found: $obj"
        echo "       Rebuild GACODE: cd \${GACODE_ROOT} && make"
        exit 1
    fi
    OBJECTS+=("$obj")
done
OBJECTS+=("${BUILD_DIR}/vgen_c_api.o")

# ---------------------------------------------------------------------------
# Pull in the gacode shared static archives that vgen depends on.
# Mirrors EXTRA_LIBS from gacode/vgen/Makefile.
# ---------------------------------------------------------------------------
EXTRA_LIBS=(
    "${GACODE_ROOT}/neo/src/neo_lib.a"
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
echo "  Linking libvgen_serial.so ..."
"$FC" -shared -fPIC \
    "${OBJECTS[@]}" \
    "${EXTRA_LIBS[@]}" \
    "${LIB_DIRS[@]}" \
    -llapack -lblas \
    -o "${BUILD_DIR}/libvgen_serial.so"

echo ""
echo "SUCCESS: ${BUILD_DIR}/libvgen_serial.so"
ls -lh "${BUILD_DIR}/libvgen_serial.so"
