# `lw4dvar-torch` Sunspot status

## Git

- Fork: [`saforem2/lw4dvar-torch`](https://github.com/saforem2/lw4dvar-torch)
- Branch: [`feature/ezpz-device-agnostic`](https://github.com/saforem2/lw4dvar-torch/tree/feature/ezpz-device-agnostic)
- Commit: [`ddcc279`](https://github.com/saforem2/lw4dvar-torch/commit/ddcc279bf803803a991ff4deaeddab1a7c39a027), “Generalize device selection with ezpz”
- The working tree is clean. SHA-256 checks of all 50 changed paths matched between the tested Sunspot staging tree and commit `ddcc279`.
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

The existing XPU environment supplies Torch. The new venv contains FME and ACE2 test dependencies.

```bash
BASE="$HOME/.ezpz-venvs/ezpz-xpu-py312"
TEST="$HOME/.ezpz-venvs/lw4dvar-mini-ace2-py312"
BASE_SITE="$($BASE/bin/python -c 'import site; print(site.getsitepackages()[0])')"

"$BASE/bin/python" -m ensurepip --upgrade
"$BASE/bin/python" -m venv --system-site-packages "$TEST"
PYTHONPATH="$BASE_SITE" "$TEST/bin/python" -m pip install \
  fme==2026.5.1 torch-harmonics==0.8.0

TEST_SITE="$($TEST/bin/python -c 'import site; print(site.getsitepackages()[0])')"
```

The mini test downloads the public fixture checkpoint into `~/.cache/lw4dvar` when it is absent and validates its manifest SHA-256.

### 3. Submit the XPU smoke

```bash
XPU_SETUP="cd $PWD && source $BASE/bin/activate"
"$BASE/bin/ezpz" submit \
  -N 1 -t 00:10:00 -q workq -A datascience \
  --filesystems tegu:home --job-name lw4dvar-xpu-final \
  --no-launch --env "$XPU_SETUP" -- \
  python test_scripts/smoke_sunspot_xpu.py
```

Expected output includes `device=xpu`, the XPU Torch version, a finite loss, and a finite gradient norm.

### 4. Submit the mini ACE2 solver test

```bash
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

## Scope

The XPU smoke and mini ACE2 test passed. A production-scale 4D-Var workload and full backend runs on XPU have not been validated.
