#!/usr/bin/env bash
# ezpz submit --env-setup-script: build the ACE2 runtime inside the job env.
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
uv pip install --no-cache --link-mode=copy --no-deps \
	"git+https://github.com/saforem2/ezpz" \
	-r requirements-alcf-ace2.txt
