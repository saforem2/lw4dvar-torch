# `lw4dvar-torch` hardware status

## Git

- Fork: [`saforem2/lw4dvar-torch`](https://github.com/saforem2/lw4dvar-torch)
- Branch: [`feature/ezpz-device-agnostic`](https://github.com/saforem2/lw4dvar-torch/tree/feature/ezpz-device-agnostic)
- Sunspot-tested code: [`ddcc279`](https://github.com/saforem2/lw4dvar-torch/commit/ddcc279bf803803a991ff4deaeddab1a7c39a027), “Generalize device selection with ezpz”. SHA-256 checks of all 50 changed paths matched between the tested Sunspot staging tree and that commit.
- Current branch includes [`6a987e5`](https://github.com/saforem2/lw4dvar-torch/commit/6a987e5c4fbcd9784b53ed795df06b3fdad0285e), which adds the Sirius setup used below.
- This report now records both Sirius and Polaris CUDA smoke results.
- No pull request has been opened.

## Changes

- Added `ezpz`-based device resolution in `forecast_model.py`, including `TORCH_DEVICE` overrides.
- Removed hardcoded CUDA defaults from backend constructors and YAML run configurations; configurations now use `auto`.
- Generalized synchronization and memory helpers and updated the JAX bridge.
- Added resolver unit coverage and `test_scripts/smoke_sunspot_xpu.py`.
- CUDA-specific tuning remains behind device checks.

## Sunspot proof

Both records are from the Sunspot PBS scheduler. Each completed in `workq` with `job_state = F` and `Exit_status = 0`.

| Job | Compute node | Wall time | Application output |
|---|---|---:|---|
| `12483304` | `x1922c4s3b0n0` | `00:00:17` | `device=xpu torch=2.13.0.dev20260519+xpu loss=0.491877 grad_norm=0.176259` |
| `12483310` | `x1922c5s6b0n0` | `00:02:48` | `1 passed in 156.04s (0:02:36)` |

Job `12483304` ran the XPU forward/backward smoke from `test_scripts/smoke_sunspot_xpu.py`. Job `12483310` ran the mini ACE2 end-to-end solver test on a compute node. The test config deliberately uses CPU; the XPU run separately verifies device selection and autograd on Sunspot’s XPU stack.

Raw PBS output is saved at `$HOME/lw4dvar-xpu-final.o12483304` and `$HOME/lw4dvar-torch-sunspot-port/lw4dvar-mini-ace2.o12483310`. The scheduler records can be checked with `qstat -xf 12483304` and `qstat -xf 12483310`.

The mini ACE2 environment has `fme==2026.5.1`, `torch-harmonics==0.8.0`, and their dependencies. It reuses the installed XPU Torch by path and keeps the ACE2 packages in a separate venv. The test checkpoint was cached at `~/.cache/lw4dvar/mini-ace2-crtm-land-16layer-2p5deg-e64-l8.tar`; SHA-256: `c3d6818df5478ea5efb53b72b38c371a8da7a90fb8aa7108a796b8edebab0fe3`.

## Reproduce on Sunspot

Run submission commands from a Sunspot login shell with `qsub` available. The commands below use the checkout and environments from the successful jobs.

### 1. Get the tested branch

```bash
git clone --branch feature/ezpz-device-agnostic \
  https://github.com/saforem2/lw4dvar-torch.git
cd lw4dvar-torch
```

### 2. Prepare the mini ACE2 environment

The standard ALCF setup is `source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env`. This run preserves the existing XPU Torch environment, so it uses `ezpz_ensure_uv` to create a separate venv and `ezpz_setup "$TEST"` in the test job.

The recorded install used `ensurepip` and `pip` because `uv` was not on `PATH` in that shell. The commands below use `uv` for environment creation and package installation.

```bash
BASE="$HOME/.ezpz-venvs/ezpz-xpu-py312"
TEST="$HOME/.ezpz-venvs/lw4dvar-mini-ace2-uv-py312"
BASE_SITE="$($BASE/bin/python -c 'import site; print(site.getsitepackages()[0])')"

source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_ensure_uv
uv venv --python="$BASE/bin/python" --system-site-packages "$TEST"
PYTHONPATH="$BASE_SITE" uv pip install --python "$TEST/bin/python" \
  fme==2026.5.1 torch-harmonics==0.8.0
"$TEST/bin/python" -c 'import torch, fme, torch_harmonics; print(torch.__version__)'
TEST_SITE="$($TEST/bin/python -c 'import site; print(site.getsitepackages()[0])')"
```

The mini test downloads the public fixture checkpoint into `~/.cache/lw4dvar` when it is absent and validates its manifest SHA-256.

### 3. Submit the XPU smoke

```bash
BASE="$HOME/.ezpz-venvs/ezpz-xpu-py312"
XPU_SETUP="cd $PWD && source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup $BASE"
"$BASE/bin/ezpz" submit \
  -N 1 -t 00:10:00 -q workq -A datascience \
  --filesystems tegu:home --job-name lw4dvar-xpu-final \
  --no-launch --env "$XPU_SETUP" -- \
  python test_scripts/smoke_sunspot_xpu.py
```

Expected output includes `device=xpu`, the XPU Torch version, a finite loss, and a finite gradient norm.

### 4. Submit the mini ACE2 solver test

```bash
BASE="$HOME/.ezpz-venvs/ezpz-xpu-py312"
TEST="$HOME/.ezpz-venvs/lw4dvar-mini-ace2-uv-py312"
BASE_SITE="$($BASE/bin/python -c 'import site; print(site.getsitepackages()[0])')"
TEST_SITE="$($TEST/bin/python -c 'import site; print(site.getsitepackages()[0])')"
ACE2_SETUP="source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup $TEST && export PYTHONPATH=$TEST_SITE:$BASE_SITE"
"$BASE/bin/ezpz" submit \
  -N 1 -t 00:30:00 -q workq -A datascience \
  --filesystems tegu:home --job-name lw4dvar-mini-ace2 \
  --no-launch --env "$ACE2_SETUP" -- \
  "$TEST/bin/python" -m pytest -q -m ace2 tests/ace2/test_ace2_mini.py
```

Expected result: `1 passed` in about 2–3 minutes. Check the scheduler result and the test log, not only the submission response:

```bash
qstat -xf JOB_ID | egrep 'Job Id:|job_state =|Exit_status =|resources_used.walltime'
```

The earlier general unit run reported `101 passed` and one ACE2 skip before the ACE2 dependencies were installed.

## Sirius CUDA proof

The hardware smoke also completed on Sirius. PBS recorded job `33440.sirius-pbs-0001.hsn.cm.sirius.alcf.anl.gov` as `job_state = F`, `Exit_status = 0`, and `resources_used.walltime = 00:00:17`. It ran on `x3200c0s13b0n0`; PBS identifies that host as a Sirius node with four A100 GPUs.

The application output was:

```text
Python 3.13.15
torch=2.14.0 cuda=13.0
device=cuda torch=2.14.0 cuda=13.0 loss=0.491877 grad_norm=0.176259
```

This run exercised `resolve_device`, CUDA forward/backward, finite gradients, and device synchronization. It did not run the ACE2 solver on Sirius. The job used the pushed branch at commit [`6a987e5`](https://github.com/saforem2/lw4dvar-torch/commit/6a987e5c4fbcd9784b53ed795df06b3fdad0285e), with `ezpz` submit client commit `3aac71153d96c1ec48bd0e7b96814481f29752ef`. Its log is `$HOME/tmp/lw4dvar-torch-sirius-6a987e5/lw4dvar-sirius-smoke.o33440` on Sirius.

## Reproduce the Sirius CUDA smoke

Clone the branch from a Sirius login shell:

```bash
git clone --branch feature/ezpz-device-agnostic \
  https://github.com/saforem2/lw4dvar-torch.git \
  "$HOME/tmp/lw4dvar-torch-sirius"
```

Then submit from a machine with the `ezpz` checkout available to `uv` and an SSH alias named `sirius`. Set `SIRIUS_REPO` to the matching absolute path on Sirius:

```bash
SIRIUS_REPO="/home/foremans/tmp/lw4dvar-torch-sirius"
EZPZ_REPO="/path/to/saforem2/ezpz"

uv run --python 3.12 --project "$EZPZ_REPO" ezpz submit \
  --remote sirius --backend ssh --workdir "$SIRIUS_REPO" \
  --job-name lw4dvar-sirius-smoke --nodes 1 --queue workq \
  --time 00:10:00 --account datascience --filesystems home \
  --env-setup-script "$SIRIUS_REPO/test_scripts/sirius_ezpz_setup.sh" \
  --no-launch -- python3 test_scripts/smoke_polaris_cuda.py
```

`test_scripts/sirius_ezpz_setup.sh` loads Sirius's `conda/2026-10-09` module, then runs `ezpz_setup_env`. That setup creates its venv with `uv` and inherits the site Torch package; it does not install packages with `pip`. The explicit conda module selects the current Sirius Python/Torch stack. PBS requires `--account datascience` for this queue.

## Polaris CUDA proof

The corrected Polaris smoke completed as job `7838743.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. PBS recorded `job_state = F`, `Exit_status = 0`, and `resources_used.walltime = 00:01:04`. It ran on `x3005c0s19b1n0`; PBS identifies that Polaris node as having four A100 GPUs.

The application output was:

```text
device=cuda torch=2.8.0+cu128 cuda=12.8 loss=0.491877 grad_norm=0.176259
```

The run used the pushed branch at commit [`6a987e5`](https://github.com/saforem2/lw4dvar-torch/commit/6a987e5c4fbcd9784b53ed795df06b3fdad0285e). It exercised the CUDA device resolver, forward/backward, finite gradients, and synchronization. The first attempt exited before the test because its working directory did not contain `test_scripts/`; the successful retry submitted from the repository root. The output log is `$HOME/tmp/lw4dvar-torch-polaris-6a987e5/lw4dvar-cuda-smoke.o7838743` on Polaris.

## Reproduce the Polaris CUDA smoke

Clone the tested branch from a Polaris login shell:

```bash
git clone --branch feature/ezpz-device-agnostic \
  https://github.com/saforem2/lw4dvar-torch.git \
  "$HOME/tmp/lw4dvar-torch-polaris"
```

Then submit from a machine with an `ezpz` source checkout available to `uv` and an SSH alias named `polaris`. Set `POLARIS_REPO` to the matching absolute path on Polaris:

```bash
POLARIS_REPO="/home/foremans/tmp/lw4dvar-torch-polaris"
EZPZ_REPO="/path/to/saforem2/ezpz"

uv run --python 3.12 --project "$EZPZ_REPO" ezpz submit \
  --remote polaris --backend ssh --workdir "$POLARIS_REPO" \
  --job-name lw4dvar-cuda-smoke --nodes 1 --queue capacity \
  --time 00:10:00 --account AuroraGPT --filesystems home:eagle \
  --env-setup-script "$POLARIS_REPO/test_scripts/polaris_ezpz_setup.sh" \
  --no-launch -- \
  /opt/cray/pe/python/3.12.12/bin/python3.12 test_scripts/smoke_polaris_cuda.py
```

`test_scripts/polaris_ezpz_setup.sh` uses the Cray Python 3.12 runtime with the existing CUDA Torch site-packages on `PYTHONPATH`. This smoke installs no Python packages. The caller-side `ezpz submit` client runs under `uv`.

## Scope

The Sunspot XPU smoke, Sunspot mini ACE2 solver, Sirius CUDA smoke, and Polaris CUDA smoke passed. A production-scale 4D-Var workload and full backend runs on XPU or CUDA have not been validated.
