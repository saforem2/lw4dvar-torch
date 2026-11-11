#!/usr/bin/env bash
# ezpz setup for the one-node Sirius CUDA device smoke test.
set -eo pipefail

# Sirius exposes the current conda module, while ezpz's generic Sirius
# fallback may select an older module or a PolarisAT micromamba path. Select
# the verified site module, then let ezpz_setup_env create/activate the
# system-site-packages venv with uv and configure the PBS job.
source <(curl -fsSL https://ezpz.cool/utils.sh)
module use /soft/modulefiles
module load conda/2026-10-09
conda activate base
ezpz_setup_env

command -v uv >/dev/null
[[ -n "${VIRTUAL_ENV:-}" ]]
python3 --version
python3 -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
