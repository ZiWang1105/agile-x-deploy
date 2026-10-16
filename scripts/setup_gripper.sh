#!/usr/bin/env bash
# Install protocol dependencies only. This script never opens CAN.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-official/bin/python ]]; then
  echo 'Run ./scripts/setup_official_ik.sh first.' >&2
  exit 1
fi
.venv-official/bin/python -m pip install -r requirements-gripper.txt
.venv-official/bin/python -c 'from agilex_deploy.gripper import official_codec; official_codec(); print("Official gripper codec ready; no CAN opened")'
