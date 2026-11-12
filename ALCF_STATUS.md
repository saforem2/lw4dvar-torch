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

## Tests and model coverage

- The checked-in `test_scripts/run_sunspot_validation.sh` ran as PBS job `12483373` at commit `791c29a238285f6c0debf730c65cae5a7043b224` on `x1921c0s6b0n0`: Ruff passed; `102 passed` in 16.12 s; mini ACE2 solver `1 passed` in 117.75 s; XPU forward/backward passed with finite loss and gradient. PBS reported `job_state = F`, `Exit_status = 0`, walltime `00:02:33`.
- Sunspot PBS job `12483369`, branch commit `bae663c3c8b77f09d58444ec7ed32cbfa938ad13`, ran on compute node `x1921c0s6b0n0` with the ALCF Python 3.12.12 / PyTorch `2.13.0a0+gitcf30153` environment: Ruff passed; `102 passed` in 34.43 s; mini ACE2 solver `1 passed` in 119.54 s; XPU forward/backward passed with finite loss and gradient. PBS reported `job_state = F`, `Exit_status = 0`, walltime `00:03:50`.
- Sunspot job `12483370` separately tried the mini ACE2 solver with `LW4DVAR_DEVICE=auto`. Device resolution selected `xpu`, but installed `fme 2026.5.1` selected CPU, so `ACE2Model` stopped with `device=xpu but fme will place the model on cpu` (`Exit_status = 1`). The installed FME selector supports CUDA, MPS, or CPU; ACE2-on-XPU is not validated. Keep ACE2 on CPU on Sunspot and use the standalone XPU smoke for XPU coverage.
- The tracked configs cover AIFS 1.1/2.0/3, FCN3, ACE2, SFNO, Aurora, GraphCast, and NGCM. In the validations recorded here, ACE2 is the only backend with a checkpoint-backed forward/solver run. The other backends have config and shared-grid coverage, not forward runs.

The model submodules are not initialized in this checkout. The full test configs also point at external checkpoint and cache paths, so additional backend runs need checkpoints and caches staged in readable storage first. The test-suite result above is from an ALCF compute node; the earlier macOS run is not accelerator validation.

## Run evidence

Each completed successful PBS job ended with `job_state = F` and `Exit_status = 0`. The ACE2/XPU attempt above ended with `Exit_status = 1` for the documented FME device mismatch.

| Job | System / node | Wall time | Output | Tested code |
|---|---|---:|---|---|
| `12483304` | Sunspot / `x1922c4s3b0n0` | `00:00:17` | `device=xpu torch=2.13.0.dev20260519+xpu loss=0.491877 grad_norm=0.176259` | `ddcc279` |
| `12483310` | Sunspot / `x1922c5s6b0n0` | `00:02:48` | `1 passed in 156.04s (0:02:36)` | `ddcc279` |
| `33440.sirius-pbs-0001.hsn.cm.sirius.alcf.anl.gov` | Sirius / `x3200c0s13b0n0` | `00:00:17` | `torch=2.14.0 cuda=13.0`, finite loss and gradient | `6a987e5` |
| `33446.sirius-pbs-0001.hsn.cm.sirius.alcf.anl.gov` | Sirius / `x3200c0s13b0n0` | `00:01:05` | `torch 2.14.0 cuda 13.0 cudnn 92600 available True`; `1 passed in 33.62s` | `29507ce` |
| `7838743.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` | Polaris / `x3005c0s19b1n0` | `00:01:04` | `device=cuda torch=2.8.0+cu128 cuda=12.8`, finite loss and gradient | `6a987e5` |
| `12483369.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov` | Sunspot / `x1921c0s6b0n0` | `00:03:50` | `102` unit tests; mini ACE2 solver pass; XPU forward/backward finite loss and gradient | `bae663c` |
| `12483370.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov` | Sunspot / `x1921c0s6b0n0` | `00:00:29` | ACE2 with auto device stopped because FME chose CPU while resolver chose XPU | `bae663c` |
| `12483373.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov` | Sunspot / `x1921c0s6b0n0` | `00:02:33` | Checked-in recipe: `102` unit tests; mini ACE2 solver pass; XPU forward/backward finite loss and gradient | `791c29a` |

The Sunspot mini solver job wrote `$HOME/lw4dvar-torch-sunspot-port/lw4dvar-mini-ace2.o12483310`; its checkpoint SHA-256 was `c3d6818df5478ea5efb53b72b38c371a8da7a90fb8aa7108a796b8edebab0fe3`. Sirius job `33446` ran the mini fixture with one 24-hour window and 50 epochs. It checked that the cost decreased and verified analysis and forecast outputs. The fixture is not the full ACE2 dataset.

## Checkpoint and project storage

The repository estimates each AIFS checkpoint at about `1 GB`, FCN3 at
`2.8 GB`, and the Aurora checkpoint at about `4.9 GB` plus a static-field
file. SFNO's NGC package is `6.9 GB` and converts to `2.1 GB` of weights. The
mini ACE2 fixture used here is `22 MB`. Sunspot's `datascience` project quota
after the run was `21.05 TB / 30 TB` used, about `8.95 TB` available.

Logs remain on their respective systems:

- Sunspot XPU smoke: `$HOME/lw4dvar-xpu-final.o12483304`
- Sirius CUDA smoke: `$HOME/tmp/lw4dvar-torch-sirius-6a987e5/lw4dvar-sirius-smoke.o33440`
- Polaris CUDA smoke: `$HOME/tmp/lw4dvar-torch-polaris-6a987e5/lw4dvar-cuda-smoke.o7838743`
- New Sunspot validation logs: `/tegu/datascience/foremans/lw4dvar-validation-logs/` (`12483369`, `12483370`, `12483373`).

## Full-model pilot

The historical 240-cycle configuration depends on a 2015 surface-pressure observation archive under another project's storage. A fresh Sunspot check found `/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs` unreadable, so the data must be staged at a readable location before the full run. The Polaris mini-case submission also needs a queue slot; the last attempt was rejected at the per-user queued-job limit.
