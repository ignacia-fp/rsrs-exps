# Numbered RSRS experiment bundles

Experiment definitions now live in small Python dictionaries. They are kept
separate from sampling, factorisation, and result analysis, so preparing a job
does not execute or modify an experiment.

For normal use, copy the fully documented JSON template
[`configs/experiment_bundle_template.json`](../configs/experiment_bundle_template.json).
You do not need to remember command flags: edit the JSON and run
`rsrs-experiment build your_file.json`.

Start from [`configs/experiment_bundle_example.py`](../configs/experiment_bundle_example.py).
Its four top-level dictionaries are:

- `DEFAULTS`: options shared by every test in the file;
- `TESTS`: named individual tests;
- `BUNDLES`: ordered lists of test names;
- `SLURM`: resource directives used only for `.sbatch` output.

`SHARE_UNPRECONDITIONED=True` controls baseline GMRES reuse across a bundle.

## Individual tests and bundles in the same JSON

`TESTS` is the library of reusable experiments. `BUNDLES` contains ordered
lists of names from `TESTS`. A generation request can select either kind:

```json
"GENERATE": {
  "selection": "intermediate",
  "name": "intermediate_only",
  "kind": "sh"
}
```

or:

```json
"GENERATE": {
  "selection": "f16_all",
  "name": "complete_f16_bundle",
  "kind": "sbatch"
}
```

To generate several scripts at once, make `GENERATE` a list:

```json
"GENERATE": [
  {"selection": "quasi", "name": "quasi_check", "kind": "sh"},
  {"selection": "f16_all", "name": "f16_all", "kind": "sbatch"}
]
```

All three forms reuse the same `DEFAULTS`, `TESTS`, and `BUNDLES`; no test
parameters need to be copied.

## Numbered options

Print the complete, documented catalog with:

```bash
rsrs-experiment options
```

The documented numbers are resolved into the Python-native case schema before
execution. For example, `operator=3` selects the complex nonsymmetric BIEGrid,
`symmetry=0` selects `NoSymm`, and `nonsymmetric_id_combination=1` selects
`Concat`. Run `rsrs-experiment options` for the complete current catalog.

Ordinary numerical values—tolerances, mesh width, kappa, thread counts, and
tree depth—remain explicit. `leaf_size=None` preserves the normal automatic
leaf-size logic.

## GMRES modes

Set `diagnostics.gmres.mode` to one of:

- `0`: no GMRES;
- `1`: unpreconditioned GMRES only;
- `2`: RSRS-preconditioned GMRES only;
- `3`: both.

Mode `1` is a genuine baseline-only path: Python constructs the physical
operator, runs GMRES, stores the complete residual histories in
`rsrs_report.json`, and exits without loading sample arrays, building an
octree, or constructing RSRS factors.

By default, if a bundle contains the same physical operator and right-hand
sides with different RSRS parameters, the first mode-`3` command remains
“both” and later equivalent commands are rendered as mode `2`. Thus the
unpreconditioned system is solved once. Set `SHARE_UNPRECONDITIONED=False` to
disable this behaviour.

## Matrix-free spectral error report

`diagnostics.spectral` controls iterative 2-norm estimates for operators that
cannot be materialised densely:

```json
"spectral": {
  "enabled": true,
  "iterations": 10,
  "block_size": 2,
  "tolerance": 0.001,
  "seed": 2701
}
```

The report includes `norm_a_2`, `norm_factor_error_2` (also recorded under the
historical name `norm_apply_2`), `factor_relative_2`, `inverse_residual_2`
(historically `err_solve_2`), `norm_rsrs_2`, `norm_rsrs_inverse_2`, and
two distinct conditioning estimates:

- `condition_target_via_rsrs_estimate_2 = ||A||_2 ||F^{-1}||_2` estimates the
  target operator's condition number using the RSRS approximate inverse;
- `condition_rsrs_estimate_2 = ||F||_2 ||F^{-1}||_2` estimates the condition
  number of the operator represented by the RSRS factors themselves.

The probe report also retains inexpensive Frobenius-norm estimates. Each
iterative 2-norm estimate records its iteration count, convergence flag, and
wall time under `error_estimation`.

The estimator applies block power iteration to the corresponding normal
operator. `block_size` processes search directions together with
matrix-matrix products, allowing the FMM backend to exploit internal
parallelism without duplicating the target operator or native RSRS factors in
separate processes.

## Sample targets and optional validation

## Separate sampling and target operators

`operator` and `operator_parameters` describe the target operator used for
error estimates, GMRES, and the system being preconditioned. By default,
sampling uses that same operator. To build the RSRS factors from a cheaper
approximation, provide `sampling_operator_parameters` (and optionally
`sampling_operator` when the backend itself differs):

```json
"operator": 5,
"operator_parameters": {
  "grid": "mesh.msh",
  "kappa": 2.0,
  "expansion_order": 8,
  "ncrit": 400
},
"sampling_operator": null,
"sampling_operator_parameters": {
  "expansion_order": 4
}
```

When `sampling_operator` is null, these parameters override the corresponding
target parameters and all other values—including the grid and kappa—are
inherited. The cache records the resolved sampling operator. The final report records both. The
runner rejects mismatched dimensions or differently ordered geometry before
factorization diagnostics are computed.

Set `sampling.save_samples` to `true` to retain a reusable cache. Set it to
`false` to generate an isolated temporary cache for this factorisation only;
the samples are removed after errors and solves finish. Ephemeral sampling
does not require a `sample_cache` path.

The sampling command treats `--samples` as a target, not as an instruction to
regenerate everything:

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

- If the compatible cache has fewer than 4,000 rows, only the missing batches
  are generated.
- If it already has 4,000 or more rows, no sampling is performed.
- A smaller later target never truncates the existing cache.
- `--validate none` performs no validation, `--validate sampled` performs the
  quick first/middle/last-batch check, and `--validate all` checks every row.
- Validation happens before factorisation when this command precedes the run.

Sample identity is determined by the full operator specification, RNG seed,
and symmetry convention. The target count is deliberately not part of the
identity, which makes safe extension possible.

## Tests and sweeps

A test may be a plain option dictionary:

```python
TESTS = {
    "intermediate": {
        "operator": 5,
        "operator_parameters": {
            "grid": "extra_meshes/f16_mesh_width_2.00e-2_495380.msh",
            "kappa": 2.6666667,
            "expansion_order": 8,
        },
        "sample_cache": "results/sample_cache/intermediate",
        "rsrs": {"tol_id": 1e-2, "tol_ext_near": 1e-16},
    },
}
```

Or it may contain `options` and `sweep`. A sweep forms the Cartesian product
of its lists and writes one command per combination:

```python
"quasi": {
    "options": {"rsrs": {"tol_id": 1e-2}},
    "sweep": {
        "rsrs.tol_id": [1e-2, 1e-4],
        "rsrs.tol_ext_near": [1e-8, 1e-16],
    },
}
```

## Generate scripts

Generate a directly runnable shell script:

```bash
rsrs-experiment render \
  --config configs/my_f16_tests.py \
  --selection f16_all \
  --name f16_comparison \
  --kind sh
```

Generate a Slurm submission script from the same definition:

```bash
rsrs-experiment render \
  --config configs/my_f16_tests.py \
  --selection f16_all \
  --name f16_comparison \
  --kind sbatch
```

The default output directory is `generated_jobs`. On 7 November 2026 these
commands produce `f16_comparison_2026-11-07.sh` or
`f16_comparison_2026-11-07.sbatch`. Generation refuses to overwrite a file
with the same name and date. `--date YYYY-MM-DD` can reproduce an earlier
name, and `--output-dir` selects another destination.

Every generated command contains one complete, versioned JSON case consumed by
`rsrs-experiment run-case`. Therefore the script itself is an exact record of
what will run and does not depend on later edits to the dictionary file.
