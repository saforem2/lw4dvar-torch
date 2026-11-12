#!/bin/bash --login
# Target-side setup for `ezpz submit --env-setup-script`.
# Keep the site's PyTorch accelerator build; install project dependencies with uv.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")"
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
