#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Install project dependencies without replacing the site-provided PyTorch.
uv pip install --no-cache --link-mode=copy --no-deps -r requirements-alcf-ace2.txt
uv pip install --no-cache --link-mode=copy ruff pytest

make check PYTHON=python3

# The ACE2 wrapper adapts FME's device selector when auto resolves to XPU.
# On CUDA and CPU systems, FME keeps its normal device selection.
export LW4DVAR_FIXTURE_CACHE="${LW4DVAR_FIXTURE_CACHE:-$HOME/datascience/lw4dvar-fixtures}"
LW4DVAR_DEVICE="${LW4DVAR_DEVICE:-auto}" python3 -m pytest -q -m ace2
python3 test_scripts/smoke_sunspot_xpu.py
