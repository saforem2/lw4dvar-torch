#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Keep the site-provided PyTorch build and install the ACE2 runtime dependencies.
uv pip install --no-cache --link-mode=copy --no-deps -r requirements-alcf-ace2.txt

LW4DVAR_DEVICE=xpu python3 backends/ace2/smoke_test_ace2.py
