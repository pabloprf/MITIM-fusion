#!/usr/bin/env bash
# =============================================================================
# build_tglf_lib.sh
#
# Build libtglf_serial.so — a shared library for in-process TGLF execution.
#
# Strategy
# --------
# We do NOT reuse GACODE's pre-built tglf .o files: most GACODE platform
# configs (e.g. PIXI_OPENMP) build without -fPIC, which is fine on macOS
# (PIC implicit) but breaks shared-library linking on Linux x86_64 with:
#   "relocation R_X86_64_PC32 ... can not be used when making a shared object;
#    recompile with -fPIC"
#
# Instead we compile the non-MPI TGLF .f90/.F90 sources ourselves with -fPIC
# into BUILD_DIR, then link them together with our thin C-binding wrapper
# tglf_c_api.f90 to produce libtglf_serial.so. This is fully self-contained
# and platform-agnostic.
#
# MPI files excluded (they `use mpi` or `use tglf_mpi`):
#   tglf_init_mpi  tglf_run_mpi  tglf_TM_mpi
# tglf_run.F90 is compiled WITHOUT -DMPI_TGLF, giving the serial tglf_run().
#
# Prerequisites
# -------------
#   - GACODE_ROOT set, with tglf/src/*.f90 sources present (no need to have
#     run `make` in gacode beforehand)
#   - gfortran (or any Fortran compiler exposing the same flags)
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

if [[ ! -d "${TGLF_SRC}" ]]; then
    echo "ERROR: TGLF source directory not found: ${TGLF_SRC}"
    exit 1
fi

# Sanity check: ensure the .f90 sources exist (we compile from source, so we
# do NOT require gacode itself to have been built).
REQUIRED_SRC="${TGLF_SRC}/tglf_modules.f90"
if [[ ! -f "${REQUIRED_SRC}" ]]; then
    echo "ERROR: ${REQUIRED_SRC} not found. Is GACODE_ROOT correct?"
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
echo "  TGLF_SRC   : $TGLF_SRC  (compiling from source with -fPIC)"
echo "  OUTPUT     : ${BUILD_DIR}/libtglf_serial.so"

# ---------------------------------------------------------------------------
# Common flags. -fdefault-real-8/-fdefault-double-8 must match the GACODE
# build so REAL == c_double on the C-binding side. -fPIC is mandatory on
# Linux for shared-library linking; harmless on macOS.
# ---------------------------------------------------------------------------
FFLAGS=(-O2 -fPIC
        -fdefault-real-8 -fdefault-double-8
        -fallow-argument-mismatch
        -fall-intrinsics)

# Order matters: module-providing files must precede their users. This list
# mirrors the OBJECTS order in ${TGLF_SRC}/Makefile, with the *_mpi files
# omitted and tglf_run.F90 (capital F) handled explicitly.
TGLF_SOURCES_F90=(
    tglf_isnan
    tglf_modules
    tglf_pkg
    tglf_allocate
    tglf_deallocate
    tglf_startup
    tglf_hermite
    tglf_inout
    tglf_setup_geometry
    tglf_LS
    tglf_eigensolver
    tglf_geometry
    tglf_matrix
    tglf_max
    tglf_interface
    tglf_error
    tglf_isinf
    tglf_shutdown
    tglf_read_input
    tglf_multiscale_spectrum
    tglf_kygrid
    tglf_TM
    tglf_nn_TM
)

# Stage .f90/.F90 sources into BUILD_DIR before compiling. gfortran always
# searches the source file's directory for .mod files, and ${TGLF_SRC} may
# contain stale .mod files left over from a prior gacode build (possibly
# from a different gfortran version), which would otherwise be picked up
# and cause "Cannot read module file ... different version" errors.
# Compiling out of BUILD_DIR isolates us from those stale artefacts.
echo "  Staging sources into ${BUILD_DIR} ..."
for name in "${TGLF_SOURCES_F90[@]}"; do
    src="${TGLF_SRC}/${name}.f90"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: source file not found: $src"
        exit 1
    fi
    cp -f "$src" "${BUILD_DIR}/${name}.f90"
done
cp -f "${TGLF_SRC}/tglf_run.F90" "${BUILD_DIR}/tglf_run.F90"

OBJECTS=()
for name in "${TGLF_SOURCES_F90[@]}"; do
    obj="${BUILD_DIR}/${name}.o"
    echo "  Compiling ${name}.f90 ..."
    ( cd "${BUILD_DIR}" && "$FC" "${FFLAGS[@]}" \
        -I"${BUILD_DIR}" -J"${BUILD_DIR}" \
        -c "${name}.f90" -o "${name}.o" )
    OBJECTS+=("$obj")
done

# tglf_run is .F90 (preprocessed). Build the SERIAL variant: do NOT define
# MPI_TGLF.
echo "  Compiling tglf_run.F90 (serial) ..."
( cd "${BUILD_DIR}" && "$FC" "${FFLAGS[@]}" \
    -I"${BUILD_DIR}" -J"${BUILD_DIR}" \
    -c "tglf_run.F90" -o "tglf_run.o" )
OBJECTS+=("${BUILD_DIR}/tglf_run.o")

# Finally compile our thin C-binding wrapper. -I${BUILD_DIR} picks up the
# tglf_interface.mod we just produced.
echo "  Compiling tglf_c_api.f90 ..."
"$FC" "${FFLAGS[@]}" \
    -I"${BUILD_DIR}" -J"${BUILD_DIR}" \
    -c "${SCRIPT_DIR}/tglf_c_api.f90" \
    -o "${BUILD_DIR}/tglf_c_api.o"
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
