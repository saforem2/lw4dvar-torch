# ALCF quick start

Use the same setup on Aurora, Sunspot, Polaris, and Sirius. Sirius needs
[ezpz PR #297](https://github.com/saforem2/ezpz/pull/297) merged first so setup
selects its CUDA 13 site stack.

```bash
git clone --branch feature/ezpz-device-agnostic --single-branch \
  https://github.com/saforem2/lw4dvar-torch.git
cd lw4dvar-torch
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
uv pip install --no-cache --link-mode=copy --no-deps \
  "git+https://github.com/saforem2/ezpz" -r requirements-alcf-ace2.txt
```

`ezpz_setup_env` loads the machine's accelerator stack and creates the Python
environment. The install uses `uv`; `--no-deps` keeps it from replacing the
site-provided PyTorch build. `requirements-alcf-ace2.txt` supplies the solver
and mini-test dependencies.

Run the bundled mini ACE2 4D-Var case inside a one-node compute allocation.
`ezpz launch` handles the process launch; it does not request nodes. From a
login shell, request an allocation with `ezpz submit` or the site's PBS
workflow. Queue, account, and filesystem values depend on the machine and
project.

```bash
LW4DVAR_DEVICE=auto ezpz launch --nproc 1 --nproc_per_node 1 --nhosts 1 -- \
  python3 -m pytest -q tests/ace2/test_ace2_mini.py
```

The test downloads a 22 MB fixture checkpoint when needed, runs one 24-hour
window for 50 epochs, checks that the cost decreases, and verifies the
analysis and forecast files. It is a mini fixture, not the full ACE2 workload.

The full solver entry point is `long_window_4dvar.py`. Run it with a config
and the full checkpoint, forcing, initial-condition and verification cache,
and surface-pressure observations staged in readable storage:

```bash
ezpz launch --nproc 1 --nproc_per_node 1 --nhosts 1 -- \
  python3 -u long_window_4dvar.py path/to/ace2_config.yml
```

The historical 240-cycle config refers to another project's observation
archive. That data must be staged somewhere this account can read before the
full-model pilot can run.
