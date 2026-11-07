# RSRS Python refactor requirements

This is the consolidated implementation checklist from the refactoring
discussion. It is deliberately separate from live Cardinal/Pitzer experiment
management.

## Kernel package

- [~] Provide one installable Python package containing Bempp-rs operator
  construction and the matching KiFMM interface.
- [~] Expose reusable Laplace particle-FMM and full Bempp single-layer
  operators, including vector and block application.
- [ ] Extend the same interface to the other kernels required by the existing
  experiment configurations.
- [ ] Provide explicit native-thread controls suitable for process-based
  parallel sampling.

## Sampling and caches

- [~] Generate tests and sketches in Python with an explicit row-oriented
  sample convention.
- [~] Validate shape, dtype, and transpose convention before factorisation.
- [ ] Provide the general parallel sampler, cache validator, repair tool, and
  deterministic consolidation tool as reusable modules rather than
  case-specific scripts.
- [ ] Record seeds/sample identifiers so independent workers cannot generate
  duplicate tests.
- [ ] Support streaming or chunked transfer to avoid duplicating complete
  sample matrices in memory.

## RSRS factorisation paths

- [x] Preserve native Rust-operator sampling: Rust operators may be sampled by
  RSRS internally and may load/extend a cache.
- [ ] Add a precomputed-sample path for Python arrays.
- [ ] Add a precomputed-cache path for Python-generated on-disk samples.
- [ ] Never call a Python operator from Rust to obtain missing samples.
- [ ] Introduce a resumable factorisation session. When Python samples are
  insufficient, retain factors/tree state, return a structured request, accept
  new rows, and continue without restarting.
- [ ] Return required Y and Z row counts, level, and limiting box in an
  `InsufficientSamples`/`NeedsSamples` status.

## Python factor interface

- [~] Define a SciPy `LinearOperator` wrapper for the approximate matrix and
  its inverse.
- [ ] Expose factor `matvec`, `matmat`, transpose, conjugate-transpose, and
  inverse application through PyO3.
- [ ] Release the GIL during factorisation and factor application.
- [ ] Keep factors and resumable state as persistent Rust-owned Python objects.

## Configuration and resources

- [~] Expose every supported factorisation option through the versioned Python case schema.
- [~] Use a versioned configuration schema and keep defaults in Rust.
- [~] Support optional `leaf_size`; `None` preserves the existing automatic
  octree logic and an explicit positive value overrides it.
- [~] Separate `factorization_threads` and `apply_threads` while accepting the
  legacy `num_threads` field.
- [ ] Allow Python sampling workers and threads-per-worker to be configured
  independently and prevent native-thread oversubscription.
- [x] Replace the legacy configuration generator with the shared Python case schema.

## Results and numerical experiments

- [ ] Rust reports factorisation ranks, intrinsic timings, sample consumption,
  tree statistics, and memory usage as structured data.
- [ ] Python performs sample consistency checks, norm/error estimation, GMRES,
  residual histories, and comparisons with the original operator.
- [ ] Python owns RCS/field plotting and publication-table generation.
- [ ] Permit unpreconditioned GMRES to be skipped or reused independently of
  RSRS factorisation runs.

## Reproducibility and validation

- [x] Bring in a genuine, current clone of upstream RSRS while preserving the
  Cardinal-derived source snapshot for comparison.
- [ ] Add numerical parity tests between native Rust sampling and precomputed
  Python samples.
- [ ] Add complex nonsymmetric tests that distinguish transpose from conjugate
  transpose.
- [ ] Provide a reproducible x86-64 Linux/Docker build and installation guide.
- [ ] Retire superseded case-specific scripts only after the replacement
  workflow passes end-to-end tests.

Legend: `[x]` complete, `[~]` initial implementation exists, `[ ]` pending.
