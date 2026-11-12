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

Run the bundled mini ACE2 4D-Var case inside a compute allocation. `auto`
selects the active accelerator through `ezpz`; the ACE2 wrapper adapts FME's
device selector on XPU systems.

```bash
LW4DVAR_DEVICE=auto ezpz launch --nproc 1 --nproc_per_node 1 --nhosts 1 -- \
  python3 -m pytest -q tests/ace2/test_ace2_mini.py
```

The test downloads a 22 MB fixture checkpoint when needed, runs one 24-hour
window for 50 epochs, checks that the cost decreases, and verifies the
analysis and forecast files. It is a mini fixture, not the full ACE2 workload.

## Full ACE2 checkpoint on Sunspot XPU

The released 1° ACE2 checkpoint, its 2020 initial condition, and matching
forcing were exercised on a Sunspot XPU compute node. The two 2020 inputs are
public LFS files from `allenai/ACE2-ERA5`; their SHA-256 values are recorded in
[`ALCF_STATUS.md`](ALCF_STATUS.md). With those files present, submit the full
checkpoint forecast/gradient smoke from a Sunspot login node:

```bash
uvx --from git+https://github.com/saforem2/ezpz ezpz submit \
  --scheduler PBS --no-launch \
  --env-setup-script "$PWD/test_scripts/alcf_env_setup.sh" \
  -N 1 -q workq --time 00:30:00 -A datascience --filesystems tegu:home \
  --job-name lw4dvar-ace2-xpu --workdir "$PWD" \
  -- bash "$PWD/test_scripts/run_ace2_xpu_smoke.sh"
```

The smoke checks real initial-state decoding, a 4-step forecast, finite and
matching latent gradients with and without checkpointing, an AdamW update,
and 4- and 20-step forward/backward timings. Job `12483387` passed on
`x1922c5s4b0n0`: 4-step forward/backward took 1.64 s, 20 steps took 9.27 s,
with 10.08 GiB peak XPU allocation. This is model-level validation; the full
240-cycle observation-assimilation run still needs `ic_cache_ace2`.

## Sunspot compute-node validation

From a shared `datascience` checkout on Sunspot, submit the repo checks, mini
ACE2 solver, and XPU smoke as one one-node job:

```bash
uvx --from git+https://github.com/saforem2/ezpz ezpz submit --remote sunspot --no-launch \
  --env-setup-script /tegu/datascience/foremans/lw4dvar-torch/test_scripts/alcf_env_setup.sh \
  -N 1 -q backfill --time 02:00:00 -A datascience --filesystems tegu:home \
  --workdir /tegu/datascience/foremans --job-name lw4dvar-checks \
  -- bash /tegu/datascience/foremans/lw4dvar-torch/test_scripts/run_sunspot_validation.sh
```

The setup script uses `ezpz_setup_env`; the test script installs dependencies
with `uv` and keeps the site's PyTorch build. The validation recipe now runs
mini ACE2 with automatic accelerator selection. Recent XPU ACE2 results and
PBS logs are recorded in [`ALCF_STATUS.md`](ALCF_STATUS.md).

The full solver entry point is `long_window_4dvar.py`. Run it with a config
and the full checkpoint, forcing, initial-condition and verification cache,
and surface-pressure observations staged in readable storage:

```bash
ezpz launch --nproc 1 --nproc_per_node 1 --nhosts 1 -- \
  python3 -u long_window_4dvar.py path/to/ace2_config.yml
```

On Sunspot, the public ISPD v4.7 observations are staged at
`/tegu/datascience/foremans/lw4dvar-observations/ispdv4.7/`. The raw files are
hourly HDF5; the solver-format conversion is reproducible with:

```bash
uv run --no-project --with h5py python test_scripts/convert_ispd_hdf5.py \
  --download-missing \
  --input /tegu/datascience/foremans/lw4dvar-observations/ispdv4.7/raw \
  --output /tegu/datascience/foremans/lw4dvar-observations/ispdv4.7/psobs
```

This fetches any missing source files directly from NCAR GDEX and writes the
solver-format files in the same command. `config_test_ace2.yml` reads the
repo-relative `./psobs` path; from the Sunspot checkout root, link it to the
staged output:

```bash
ln -s /tegu/datascience/foremans/lw4dvar-observations/ispdv4.7/psobs psobs
```

Use the converted `psobs` directory as `obspath`. This source has 9,968
observations at `2015-01-15T00`, versus 10,821 in the historical archive, so
results are not an exact reproduction. The original Polaris path is
`/lus/eagle/projects/moonshot-reanalysis/jwhitaker/psobs`; it is not readable
by this account. See [`ALCF_STATUS.md`](ALCF_STATUS.md) for the current cache
requirements and hardware results.
