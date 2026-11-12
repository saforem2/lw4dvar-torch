#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Install project dependencies without replacing the site-provided PyTorch.
uv pip install --no-cache --link-mode=copy --no-deps -r requirements-alcf-ace2.txt
uv pip install --no-cache --link-mode=copy ruff pytest

make check PYTHON=python3

# FME 2026.5.1 selects CPU on XPU systems, so keep ACE2 on CPU. The final
# command separately checks Sunspot XPU selection and autograd.
export LW4DVAR_FIXTURE_CACHE="${LW4DVAR_FIXTURE_CACHE:-$HOME/datascience/lw4dvar-fixtures}"
LW4DVAR_DEVICE=cpu python3 -m pytest -q -m ace2
python3 test_scripts/smoke_sunspot_xpu.py
