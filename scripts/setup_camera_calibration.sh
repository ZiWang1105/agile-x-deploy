#!/usr/bin/env bash
# Reuse official IK/FK; add the pinned image-processing dependency for calibration.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-official/bin/python ]]; then
  echo 'Run ./scripts/setup_official_ik.sh first.' >&2
  exit 1
fi
.venv-official/bin/python -m pip install -r requirements-camera.txt
.venv-official/bin/python scripts/calibrate_camera.py --help
