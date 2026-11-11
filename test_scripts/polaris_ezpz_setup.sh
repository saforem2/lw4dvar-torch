#!/usr/bin/env bash
# ezpz submit --env-setup-script entry point for the Polaris CUDA smoke test.
set -eo pipefail

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

# Pin the ezpz site setup helpers used for this validation.
EZPZ_UTILS_REV=4a2aba376eb855004baeaebc5f953cc2a95b1850
EZPZ_UTILS_FILE="${TMPDIR:-/tmp}/ezpz-utils-${EZPZ_UTILS_REV}.sh"
if [[ ! -s "${EZPZ_UTILS_FILE}" ]]; then
	curl --fail --silent --show-error --location \
		"https://raw.githubusercontent.com/saforem2/ezpz/${EZPZ_UTILS_REV}/src/ezpz/bin/utils.sh" \
		-o "${EZPZ_UTILS_FILE}"
fi
source "${EZPZ_UTILS_FILE}"

# Use ezpz for Polaris PBS metadata and the site CUDA/compiler module stack.
ezpz_setup_job
ezpz_load_modules

# This prebuilt conda prefix contains the CUDA PyTorch used by the smoke.
module load conda/2025-09-25
