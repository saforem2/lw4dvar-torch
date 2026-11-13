# Running on a non-ALCF CUDA system

The model code supports `device: auto` on CUDA through `ezpz`. This guide uses
a local Linux CUDA environment; it does not use `ezpz_setup_env`, ALCF modules,
PBS, or ALCF project paths.

> **Validation status:** The CUDA code path has been exercised on ALCF GPUs.
> This environment recipe and full ACE2 run have not yet been tested outside
> ALCF.

## Environment

Create a Python 3.12 environment with `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install a CUDA-enabled PyTorch build that matches the host's NVIDIA driver.
Choose Linux, Pip, Python, and the appropriate CUDA platform in the
[PyTorch install selector](https://pytorch.org/get-started/locally/), then use
`uv pip` in place of `pip3` in its command. Keep the selected PyTorch build in
place when installing the solver lock:

```bash
uv pip install --no-cache --link-mode=copy --no-deps \
  "git+https://github.com/saforem2/ezpz" \
  -r requirements-alcf-ace2.txt
```

The dependency lock was assembled for the ALCF ACE2 environment. It omits
PyTorch because ALCF setup supplies that package. Outside ALCF, install PyTorch
first and check that the lock's FME and `torch_harmonics` versions work with
that build.

Confirm CUDA is visible and `auto` resolves to CUDA:

```bash
python - <<'PY'
import torch
import forecast_model

device = forecast_model.resolve_device("auto")
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={device}")
assert torch.cuda.is_available()
assert device.type == "cuda"
PY
```

Run the fixture-backed ACE2 solver test:

```bash
LW4DVAR_DEVICE=auto python -m pytest -q tests/ace2/test_ace2_mini.py
```

For a full-checkpoint run, stage the ACE2 checkpoint and forcing files, initial
and verification cache, and `psobs` data on the machine. The download and
checksum commands in [ALCF quick start](alcf-quickstart.md#full-ace2-checkpoint-and-sample-inputs)
are portable; its PBS submission commands are Sunspot-specific. Run the solver
as a single process with a config whose asset paths point to those staged files:

```bash
LW4DVAR_DEVICE=auto python -u long_window_4dvar.py path/to/ace2_config.yml
```

The CUDA hardware checks recorded in [ALCF status](alcf-status.md) ran on ALCF
systems; the full ACE2 checkpoint assimilation run is validated on Sunspot
XPU.
