# Saved Accuracy Metrics

Each `error_stats_*.json` contains an `accuracy_metrics` object with these four
headline estimates. Here `A` is the reference operator, `A_tilde` the RSRS forward
approximation, `B` the computed approximate inverse, and `n = dim`.

| JSON key | Definition | Interpretation |
| --- | --- | --- |
| `compression_relerr_fro` | `||A - A_tilde||_F / ||A||_F` | Relative aggregate compression error |
| `compression_relerr_2` | `||A - A_tilde||_2 / ||A||_2` | Relative spectral compression error |
| `solve_residual_relerr_fro` | `||I - B A||_F / sqrt(n)` | RMS relative solution error over isotropic unit solutions, with `b = A x` |
| `solve_residual_norm_2` | `||I - B A||_2` | Estimated worst-direction relative solution error |

All four are dimensionless. The solve residual is **`I - B A`**, not `I - A B`.
The normalized Frobenius solve metric is not an average over arbitrary
right-hand sides. The spectral solve residual needs no division by `sqrt(n)`
because `||I||_2 = 1`. Relative spectral compression error is scaled by `||A||_2`,
not by `||A x||_2`, so it is not a uniform relative matvec-error bound.

The object records definitions, interpretations, estimator settings and seeds,
and `values_are_estimates: true`, `certified_bounds: false`. Current estimates
use 20 real Gaussian test vectors for Frobenius metrics and 50 power iterations
for the spectral metrics. The Frobenius estimates are `||(A-A_tilde) G||_F / ||A G||_F`
and `||(I-B A) H||_F / ||H||_F`. These use fresh, independent-of-training
evaluation probes with separate fixed seeds for `G` and `H`, not columns reused
from saved `Omega` or `Psi`. Adjoint consistency and estimator convergence need
checking before claims of spectral accuracy. These are algebraic errors against
the reference operator (live FMM for these sweeps), not PDE discretization errors.

Raw fields are retained unchanged for compatibility and auditing:

- `norm_apply_fro / norm_a_fro` gives `compression_relerr_fro`.
- `norm_apply_2 / norm_a_2` gives `compression_relerr_2`.
- `err_solve_fro / sqrt(dim)` gives `solve_residual_relerr_fro`.
- `err_solve_2` gives `solve_residual_norm_2` directly.
- `solve_error_rhs` is a separate sampled solution error, not a matrix norm.

Null headline values indicate invalid/nonfinite raw inputs or undefined
normalization, not zero error.

## Existing and In-Progress Sweeps

The current u192 queue uses already-running executables. The following annotator
updates completed results and their per-run status entries atomically, preserves
raw measurements, and writes `accuracy_summary.json`, grouped by family and rank:

```bash
python scripts/saved_accuracy_metrics.py --queue-root /path/to/queue
```

For an active queue, `--launch-watch` starts a detached, single-instance annotator
that checks every 60 seconds. It does not run any compression or sample generation.
It leaves running result/status files alone and updates aggregate queue statuses
only after the drivers finish. The watcher exits after the main queue and added
rank queues finish and all completed results have been annotated.

```bash
python scripts/saved_accuracy_metrics.py --queue-root /path/to/queue --launch-watch
```

Check `accuracy_metrics_status.json` and `accuracy_metrics.log` in the queue
directory. This backfill is for the current sweeps whose raw-field definitions
and estimator settings match this source, not unrelated historical result formats.
Newly built Rust executables write the same metric object directly.
