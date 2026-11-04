# Python sampling workflow

This branch contains only the reusable Python sampling workflow on top of
commit `71ce5394`. It does not change RSRS, Bempp-CL, ExaFMM, or dependency
versions.

## Safety model

- Worker processes use multiprocessing `spawn`.
- Each worker enters its own directory and sets its own `TMPDIR` before
  constructing a fresh BEM++/ExaFMM operator.
- Gaussian vectors are determined by `(seed, stream, global_row)`, so changing
  worker count or repairing a row cannot duplicate or reorder samples.
- Worker chunks are associated with explicit global row intervals.
- Validation constructs another fresh isolated operator.
- Assembly refuses a workspace without a passing validation report and never
  overwrites an existing Rust cache.

The dedicated Laplace production script also checkpoints every ten-row batch
atomically. Its validator checks every checksum and deterministic test vector,
one seeded random linear combination per contiguous group of at most 30 rows,
and 60 distributed individual rows.

## Environment

Use the repository's existing pinned environment. Do not update dependencies.
The scripts require NumPy, h5py, BEM++ and the already configured ExaFMM
backend.

## Generic CLI

```bash
python scripts/rsrs_samples.py list-kernels
python scripts/rsrs_samples.py generate \
  --case configs/sampling/laplace_sphere.json \
  --output results/sample_workspaces/laplace_pilot_30 \
  --samples 30 --workers 2 --threads-per-worker 8
```

Commands are `generate`, `validate`, `repair`, and `assemble`. Generation uses
full independent validation by default. Start with 30 samples before a
production run.

The F16 example case requires the historical `.msh` file. Update its `mesh`
entry to the local path and set `kappa`, mesh width, and expected DOFs for the
quasi-static, intermediate, or oscillatory case.

## Proven resumable Laplace command

```bash
python scripts/generate_laplace_single_layer_parallel.py \
  --geometry sphere --h 0.005 --samples 6040 \
  --workers 8 --batch-size 10 --assembler exafmm --resume
```

Choose workers so `workers * threads_per_process` does not exceed the Slurm
allocation. Do not run two ExaFMM workers from the same working directory.

## Sharing

Commit and share scripts, case descriptions, validation reports and small
manifests. Do not commit sample arrays, HDF5 caches, meshes, logs, environments,
or generated results.
