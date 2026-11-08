# RSRS experiments

This repository is the Python-facing experiment layer for RSRS. It prepares
and validates samples, renders reproducible experiment bundles, runs
factorisations, computes errors and solves, and produces plots and reports.

The refactoring separates four responsibilities:

- `so-kernels` (a sibling repository) provides Python operators backed by
  Bempp-cl, Bempp-rs, and KiFMM.
- `bempp-rsrs` (the sibling `../rsrs` repository) owns the Rust
  factorisation and factor application.
- `python/rsrs_experiments` owns sampling, validation, orchestration, GMRES,
  reporting, and plotting.
- `bempp_rsrs` exposes sample-fed factorisation, factor application, ranks,
  timings, effort, and memory statistics from the sibling Rust repository.

This repository is Python-only. Rust factorisation code and its PyO3 binding
live in `../rsrs`; new experiments use the versioned JSON/Python configuration
layer rather than adding one-off drivers.

## Development setup

The reproducible Linux environment is described by `Dockerfile` and
`compose.yaml`:

```bash
docker compose build
docker compose run --rm dev
```

For a native environment, use:

```bash
bash scripts/setup_local.sh
source .venv/bin/activate
source .rsrs-env.sh
```

On systems with multiple BLAS implementations, select the LP64 BLAS used to
build RSRS before setup. The generated environment file preserves the same
runtime search path:

```bash
RSRS_BLAS_LIB_DIR=/path/to/lp64-openblas/lib bash scripts/setup_local.sh
```

The Bempp-cl, ExaFMM, and KiFMM revisions used by the existing compatibility
path remain pinned as submodules under `external/`. Development versions of
RSRS, Bempp-rs, KiFMM, and `so-kernels` live as sibling repositories.

## Create experiments

Copy `configs/experiment_bundle_template.json`, fill in its `DEFAULTS`,
`TESTS`, `BUNDLES`, and `GENERATE` sections, then run:

```bash
rsrs-experiment build configs/my_experiments.json
```

This creates a dated `.sh` or `.sbatch` file. An individual test and a bundle
use the same configuration format. Print the numbered option catalog with:

```bash
rsrs-experiment options
```

See [Experiment bundles](docs/EXPERIMENT_BUNDLES.md) for configuration and
GMRES modes, including reuse of one unpreconditioned solve across a bundle.

## Sampling

Sampling is target-based and resumable. Existing compatible rows are reused;
only missing rows are generated:

```bash
rsrs-experiment sample \
  --operator biegrid_complex_nonsymmetric \
  --parameters '{"cells_per_axis": 16}' \
  --samples 4000 \
  --workers 8 \
  --batch-size 10 \
  --validate sampled \
  --output results/sample_cache/my_case
```

Validation may be disabled, sampled quickly, or applied to every row. Python
never asks RSRS to sample implicitly: insufficient rows produce a structured
error so the user can choose whether to extend the cache and rerun.
In JSON cases, `sampling.save_samples=false` keeps the generated samples only
for the duration of that experiment; `true` retains the resumable cache.

See [Python experiment pipeline](docs/PYTHON_EXPERIMENT_PIPELINE.md) and
[Python-first RSRS architecture](docs/PYTHON_RSRS_ARCHITECTURE.md) for the
sample layout and ownership boundaries.

## Tests

With the development dependencies installed:

```bash
python -m pytest tests -q
```

The current implementation checklist is maintained in
[Refactor requirements](docs/REFACTOR_REQUIREMENTS.md).
