# `lw4dvar-torch` ALCF status

## Code

- Fork: [`saforem2/lw4dvar-torch`](https://github.com/saforem2/lw4dvar-torch)
- Branch: [`feature/ezpz-device-agnostic`](https://github.com/saforem2/lw4dvar-torch/tree/feature/ezpz-device-agnostic)
- The original Sunspot-tested device changes are at [`ddcc279`](https://github.com/saforem2/lw4dvar-torch/commit/ddcc279bf803803a991ff4deaeddab1a7c39a027). SHA-256 checks of all 50 changed paths matched the tested Sunspot tree.
- No pull request has been opened for `lw4dvar-torch`.

The device resolver uses `ezpz`, backend defaults use `auto`, and CUDA-specific tuning remains behind device checks. The test and run commands are in [`ALCF_QUICKSTART.md`](ALCF_QUICKSTART.md).

## Shared ALCF instructions

The clone, setup, dependency install, and mini-case command are the same on Aurora, Sunspot, Polaris, and Sirius. `ezpz_setup_env` selects the site stack. `uv pip install --no-deps` keeps that PyTorch build in place.

Run `ezpz launch` inside a compute allocation. It launches the workload; it does not request nodes. From a login node, use an interactive allocation or `ezpz submit`; PBS queue, account, and filesystem values depend on the system and project.

Sirius needs [ezpz PR #297](https://github.com/saforem2/ezpz/pull/297) merged for the standard setup to select the site-matched CUDA 13 stack. The successful Sirius run used `conda/2026-10-09`.

## Hardware results

| System | Hardware | Result | Evidence |
|---|---|---|---|
| Sunspot | Intel XPU | XPU forward/backward smoke passed. Mini ACE2 solver test passed with its CPU test config. | PBS jobs `12483304`, `12483310` |
| Sirius | 4× A100, CUDA 13.0 | CUDA forward/backward smoke and mini ACE2 solver test passed. | PBS jobs `33440`, `33446` |
| Polaris | 4× A100, CUDA 12.8 | CUDA forward/backward smoke passed. Mini ACE2 solver test has not run; the per-user queue limit blocked submission. | PBS job `7838743` |
| Aurora | Intel XPU | Not validated in this effort. | — |

These are hardware smoke and mini-fixture results. The full-sized 240-cycle ACE2 run has not run on any system.

## Run evidence

Each completed PBS job ended with `job_state = F` and `Exit_status = 0`.

| Job | System / node | Wall time | Output | Tested code |
|---|---|---:|---|---|
| `12483304` | Sunspot / `x1922c4s3b0n0` | `00:00:17` | `device=xpu torch=2.13.0.dev20260519+xpu loss=0.491877 grad_norm=0.176259` | `ddcc279` |
| `12483310` | Sunspot / `x1922c5s6b0n0` | `00:02:48` | `1 passed in 156.04s (0:02:36)` | `ddcc279` |
| `33440.sirius-pbs-0001.hsn.cm.sirius.alcf.anl.gov` | Sirius / `x3200c0s13b0n0` | `00:00:17` | `torch=2.14.0 cuda=13.0`, finite loss and gradient | `6a987e5` |
| `33446.sirius-pbs-0001.hsn.cm.sirius.alcf.anl.gov` | Sirius / `x3200c0s13b0n0` | `00:01:05` | `torch 2.14.0 cuda 13.0 cudnn 92600 available True`; `1 passed in 33.62s` | `29507ce` |
| `7838743.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` | Polaris / `x3005c0s19b1n0` | `00:01:04` | `device=cuda torch=2.8.0+cu128 cuda=12.8`, finite loss and gradient | `6a987e5` |

The Sunspot mini solver job wrote `$HOME/lw4dvar-torch-sunspot-port/lw4dvar-mini-ace2.o12483310`; its checkpoint SHA-256 was `c3d6818df5478ea5efb53b72b38c371a8da7a90fb8aa7108a796b8edebab0fe3`. Sirius job `33446` ran the mini fixture with one 24-hour window and 50 epochs. It checked that the cost decreased and verified analysis and forecast outputs. The fixture is not the full ACE2 dataset.

Logs remain on their respective systems:

- Sunspot XPU smoke: `$HOME/lw4dvar-xpu-final.o12483304`
- Sirius CUDA smoke: `$HOME/tmp/lw4dvar-torch-sirius-6a987e5/lw4dvar-sirius-smoke.o33440`
- Polaris CUDA smoke: `$HOME/tmp/lw4dvar-torch-polaris-6a987e5/lw4dvar-cuda-smoke.o7838743`

## Full-model pilot

The historical 240-cycle configuration depends on a 2015 surface-pressure observation archive under another project's storage. This account cannot read that path, so the data must be staged at a readable location before the full run. The Polaris mini-case submission also needs a queue slot; the last attempt was rejected at the per-user queued-job limit.
