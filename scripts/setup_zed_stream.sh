#!/usr/bin/env bash
# Install camera dependencies without modifying the IK environments.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv-camera/bin/python ]]; then
    uv venv .venv-camera --python 3.10
  fi
  uv pip install --python .venv-camera/bin/python -r requirements-camera.txt
else
  if [[ ! -x .venv-camera/bin/python ]]; then
    python3 -m venv .venv-camera
  fi
  .venv-camera/bin/python -m pip install -r requirements-camera.txt
fi

.venv-camera/bin/python -c 'import cv2; print("Camera environment ready; OpenCV", cv2.__version__)'
