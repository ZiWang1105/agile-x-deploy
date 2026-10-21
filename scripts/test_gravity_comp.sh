#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-official/bin/python ]]; then
  echo 'The existing .venv-official environment is required.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec .venv-official/bin/python -m agilex_deploy.gravity_comp_test "$@"
