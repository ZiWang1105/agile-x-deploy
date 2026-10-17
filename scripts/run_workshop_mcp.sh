#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-official/bin/python -m agilex_deploy.workshop_mcp "$@"
