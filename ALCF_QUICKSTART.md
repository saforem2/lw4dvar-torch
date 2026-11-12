# ALCF quick start

The repo entry point is `long_window_4dvar.py`. The default CI ACE2 test uses
the bundled mini checkpoint and data, so it needs no model submodule or
external observation cache.

```bash
git clone --branch feature/ezpz-device-agnostic --single-branch \
  https://github.com/saforem2/lw4dvar-torch.git
cd lw4dvar-torch
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
uv pip install --no-cache --link-mode=copy \
  "git+https://github.com/saforem2/ezpz" \
  -r requirements-alcf-ace2.txt
```

Run the mini ACE2 4D-Var test on the machine's detected accelerator:

```bash
LW4DVAR_DEVICE=auto ezpz launch --nproc 1 --nproc_per_node 1 --nhosts 1 -- \
  python3 -m pytest -q tests/ace2/test_ace2_mini.py
```

`ezpz launch` runs one process because the optimizer is not distributed. The
test downloads the 22 MB mini checkpoint to the user cache if needed, runs one
24-hour window for 50 epochs, checks that the cost decreases, and verifies the
analysis/forecast output files. Omit `LW4DVAR_DEVICE=auto` to keep the CI
default (`cpu`).

For the full ACE2 checkpoint, use the same launcher with a config that points
to the checkpoint, yearly forcing, initial-condition/verification cache, and
surface-pressure observations:

```bash
ezpz launch --nproc 1 --nproc_per_node 1 --nhosts 1 -- \
  python3 -u long_window_4dvar.py my_ace2_config.yml
```

The 240-cycle experiment needs its matching surface-pressure observation
archive. The paths in historical configs point to another project's storage
and are not portable. That archive must be staged in a location this account
can read before launching the full-model pilot.
