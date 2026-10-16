#!/usr/bin/env bash
# Install the official solver's CasADi-enabled Pinocchio in an isolated prefix.
# This script builds models and imports dependencies; it never opens CAN.
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$PWD"
if [[ ! -x .tools/bin/micromamba ]]; then
  mkdir -p .tools
  python3 - <<'PY'
from pathlib import Path
import hashlib
import tarfile
import urllib.request
archive = Path('.tools/micromamba-2.3.2.tar.bz2')
url = 'https://micro.mamba.pm/api/micromamba/linux-64/2.3.2'
with urllib.request.urlopen(url, timeout=60) as response:
    archive.write_bytes(response.read())
expected = '5512233cdd8564a671626081026dc861537a963baa06706baab08fac6f3bb9d2'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Micromamba archive checksum mismatch')
with tarfile.open(archive) as package:
    member = package.getmember('bin/micromamba')
    if not member.isfile():
        raise RuntimeError('Unexpected micromamba archive member')
    binary = Path('.tools/bin/micromamba')
    binary.parent.mkdir(exist_ok=True)
    binary.write_bytes(package.extractfile(member).read())
    binary.chmod(0o755)
PY
fi
operation=create
if [[ -d .venv-official/conda-meta ]]; then operation=install; fi
.tools/bin/micromamba "$operation" -y -p "$project_dir/.venv-official" -f environment.official.yml
.venv-official/bin/python -m pip install -r requirements.txt -e .
.venv-official/bin/python scripts/build_assembly.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-official/bin/python - <<'PY'
import pinocchio, pinocchio.casadi, casadi
from agilex_deploy.model import build_sim_model, Kinematics, initial_positions
from agilex_deploy.official_ik import OfficialDualIK
build_sim_model()
ik = OfficialDualIK(Kinematics(), initial_positions())
print('Official IK ready:', pinocchio.__version__, casadi.__version__, ik.last_status)
PY
