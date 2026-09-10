#!/bin/bash
# CP2K launcher wrapper — sets up library paths for the manually
# installed CP2K 2024.1 (from conda-forge binaries extracted under .conda/).
# Usage: cp2k-run.sh [cp2k args...]  (or source it: source cp2k-run.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.conda/deps/mkl-link:${PROJECT_ROOT}/.conda/deps/extract/lib:${PROJECT_ROOT}/.conda/pkgs/libgfortran5-16.1.0-h79bb938_3/lib:/opt/intel/oneapi/2024.1/lib:/opt/intel/oneapi/2024.1/mkl/2024.1/lib:${LD_LIBRARY_PATH}"

CP2K_BIN="${CP2K_BIN:-${PROJECT_ROOT}/.conda/cp2k-extract/bin/cp2k.ssmp}"

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  # sourced — just export env
  return 0
fi

exec "$CP2K_BIN" "$@"