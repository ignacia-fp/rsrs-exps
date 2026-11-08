# Python experiment pipeline

`rsrs-experiments` owns orchestration around the Rust RSRS factorisation:

1. `so_kernels.OperatorSpec` constructs the requested operator independently in
   each spawned Python process.
2. Sampling workers process deterministic Gaussian samples in row batches. The
   default batch size is 10. Operators are built and warmed once per worker.
   Every worker is constructed in a fresh unique working/temporary directory;
   this prevents ExaFMM workers from sharing fixed-name runtime files, even if
   a scheduler later reuses the same process identifier.
3. Batch files are assembled by global row range into memory-mapped arrays.
4. Optional validation uses a separate pool of fresh isolated operators. It can
   check the first/middle/last validation batches (`sampled`) or every row
   (`all`), with independent worker count and validation batch size.
5. Rust receives the completed samples and remains solely responsible for the
   RSRS tree, ranks, sample requirements, and factorisation.
6. Python computes operator, solve, residual, and plotting diagnostics.

## Sample and validate

```bash
rsrs-experiment sample \
  --operator biegrid_complex_nonsymmetric \
  --parameters '{"cells_per_axis": 22}' \
  --samples 256 \
  --workers 4 \
  --batch-size 10 \
  --validate all \
  --validation-workers 2 \
  --validation-batch-size 30 \
  --output results/biegrid_samples
```

Each sample row is generated from `SeedSequence([seed, family, global_row])`.
The sequence therefore does not depend on worker scheduling, and different
workers cannot accidentally restart the same random stream. An interrupted run
can reuse completed batches only when their configuration fingerprint matches.

The validator recomputes natural samples

```text
y_sketch = A   @ y_test
z_sketch = A.T @ z_test
```

using fresh operators. Conversion to RSRS's conjugated internal storage for a
complex transpose family occurs only inside the native RSRS boundary.

## Run RSRS

```bash
rsrs-experiment run \
  --cache results/biegrid_samples \
  --options configs/biegrid_rsrs.json \
  --output results/biegrid_factorisation
```

If Rust reports insufficient samples, the command stops and records the exact
deficit. It never enters an automatic resampling loop.

## Plot an error report

```bash
rsrs-experiment plot-errors \
  --report results/biegrid_factorisation/rsrs_report.json \
  --output results/biegrid_factorisation/errors.png
```

Each completed experiment writes focused `factorization_report.json`,
`errors_report.json`, and `solves_report.json` files. The factorization report
contains the RSRS ranks, box statistics, intrinsic per-stage timings, sample
consumption, effort, and memory statistics. The backward-compatible combined
`rsrs_report.json` is retained for plotting and existing analysis scripts.

## Thread ownership

The JSON case has two independent controls:

- `rsrs.factorization_threads` selects the size of RSRS's private Rayon pool;
- `environment.OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS`
  are applied before Python starts and control native linear algebra and
  OpenMP-backed sampling/target operators.

`execution.factorization_blas_threads` limits BLAS inside the Rayon-parallel
factorization, and `execution.rsrs_apply_blas_threads` does the same inside
RSRS factor and inverse applications (both default to one). Outside those
scoped RSRS calls, Python target operators retain their requested native thread
settings. Sampling uses process workers, so reserve approximately
`sampling.workers` times the per-worker native thread count and avoid exceeding
the CPUs allocated by the scheduler.
