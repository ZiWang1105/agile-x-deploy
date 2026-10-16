#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-official/bin/agilex-interactive ]]; then
  echo 'Run ./scripts/setup_official_ik.sh first.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec .venv-official/bin/agilex-interactive "$@"
