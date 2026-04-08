#!/usr/bin/env bash
# =============================================================================
# build_tglf_lib.sh
#
# Build libtglf_serial.so — a shared library for in-process TGLF execution.
#
# Strategy
# --------
# TGLF is already compiled by the GACODE build system in ${GACODE_ROOT}/tglf/src/.
# We reuse those pre-built object files (which include -fPIC from the GACODE
# platform config) and compile only our thin C-binding wrapper tglf_c_api.f90.
# This avoids re-fighting GACODE's legacy Fortran compiler quirks.
#
# MPI files excluded (they reference `use mpi` or `use tglf_mpi`):
#   tglf_init_mpi.o  tglf_run_mpi.o  tglf_TM_mpi.o
#
# The tglf_run.o in GACODE's src was compiled WITHOUT -DMPI_TGLF, so it
# provides the serial tglf_run() subroutine we need.
#
# Prerequisites
# -------------
#   - GACODE_ROOT set and gacode already built (tglf/src/*.o must exist)
#   - gfortran (for compiling tglf_c_api.f90 only)
#   - liblapack + libblas
#
# Usage
# -----
#   cd <this directory>
#   bash build_tglf_lib.sh            # build
#   bash build_tglf_lib.sh --clean    # remove build artefacts and library
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- validate environment ----------------------------------------------------
if [[ -z "${GACODE_ROOT:-}" ]]; then
    echo "ERROR: GACODE_ROOT is not set. Source your gacode_setup script first."
    exit 1
fi

TGLF_SRC="${GACODE_ROOT}/tglf/src"
GACODE_MOD="${GACODE_ROOT}/modules"

if [[ ! -d "${TGLF_SRC}" ]]; then
    echo "ERROR: TGLF source directory not found: ${TGLF_SRC}"
    exit 1
fi

# Quick sanity check: ensure the gacode build produced the required .o files
REQUIRED_OBJ="${TGLF_SRC}/tglf_run.o"
if [[ ! -f "${REQUIRED_OBJ}" ]]; then
    echo "ERROR: ${REQUIRED_OBJ} not found."
    echo "       Build GACODE first: cd \${GACODE_ROOT} && make"
    exit 1
fi

# --- optional clean ----------------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning build artefacts (tglf_build/)..."
    rm -rf "${SCRIPT_DIR}/tglf_build"
    echo "Done."
    exit 0
fi

# --- detect Fortran compiler -------------------------------------------------
FC="${FC:-gfortran}"
if ! command -v "$FC" &>/dev/null; then
    echo "ERROR: Fortran compiler '$FC' not found. Install gfortran or set FC."
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

# All build artefacts (including the .so) live in tglf_build/ — gitignored.
BUILD_DIR="${SCRIPT_DIR}/tglf_build"
mkdir -p "${BUILD_DIR}"

echo "Building libtglf_serial.so"
echo "  FC         : $FC"
echo "  TGLF_SRC   : $TGLF_SRC  (pre-built .o files)"
echo "  OUTPUT     : ${BUILD_DIR}/libtglf_serial.so"

# ---------------------------------------------------------------------------
# Compile tglf_c_api.f90  (only this file needs compiling)
# Flags must match the GACODE build: -fdefault-real-8 makes REAL == c_double.
# -I${GACODE_MOD} provides pre-built .mod files (tglf_interface.mod etc.)
# ---------------------------------------------------------------------------
echo "  Compiling tglf_c_api.f90 ..."
"$FC" -O2 -fPIC \
    -fdefault-real-8 -fdefault-double-8 \
    -fallow-argument-mismatch \
    -fall-intrinsics \
    -I"${GACODE_MOD}" \
    -J"${BUILD_DIR}" \
    -c "${SCRIPT_DIR}/tglf_c_api.f90" \
    -o "${BUILD_DIR}/tglf_c_api.o"

# ---------------------------------------------------------------------------
# Collect the pre-built non-MPI TGLF object files
# MPI files excluded: tglf_init_mpi.o  tglf_run_mpi.o  tglf_TM_mpi.o
# ---------------------------------------------------------------------------
OBJECTS=()
for name in \
    tglf_isnan \
    tglf_isinf \
    tglf_modules \
    tglf_pkg \
    tglf_allocate \
    tglf_deallocate \
    tglf_startup \
    tglf_hermite \
    tglf_inout \
    tglf_setup_geometry \
    tglf_LS \
    tglf_eigensolver \
    tglf_geometry \
    tglf_matrix \
    tglf_max \
    tglf_interface \
    tglf_error \
    tglf_shutdown \
    tglf_read_input \
    tglf_multiscale_spectrum \
    tglf_kygrid \
    tglf_run \
    tglf_TM \
    tglf_nn_TM; do

    obj="${TGLF_SRC}/${name}.o"
    if [[ ! -f "$obj" ]]; then
        echo "ERROR: pre-built object not found: $obj"
        echo "       Rebuild GACODE: cd \${GACODE_ROOT} && make"
        exit 1
    fi
    OBJECTS+=("$obj")
done
OBJECTS+=("${BUILD_DIR}/tglf_c_api.o")

# ---------------------------------------------------------------------------
# Link shared library
# ---------------------------------------------------------------------------
echo "  Linking libtglf_serial.so ..."
"$FC" -shared -fPIC \
    "${OBJECTS[@]}" \
    "${LIB_DIRS[@]}" \
    -llapack -lblas \
    -o "${BUILD_DIR}/libtglf_serial.so"

echo ""
echo "SUCCESS: ${BUILD_DIR}/libtglf_serial.so"
ls -lh "${BUILD_DIR}/libtglf_serial.so"
