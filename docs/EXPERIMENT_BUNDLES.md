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

The numbers retain the historical `rsrs_config.py` meanings. Important
examples are `operator=19` for `BemppClHelmholtzCombined`, `geometry=6` for
F16, `assembler=1` for ExaFMM, `symmetry=0` for `NoSymm`, and
`nonsymmetric_combination=1` for `Concat`.

Ordinary numerical values—tolerances, mesh width, kappa, thread counts, and
tree depth—remain explicit. `leaf_size=None` preserves the normal automatic
leaf-size logic.

## GMRES modes

Set `gmres` to one of:

- `0`: no GMRES;
- `1`: unpreconditioned GMRES only;
- `2`: RSRS-preconditioned GMRES only;
- `3`: both.

Mode `1` is a genuine baseline-only path: the Rust driver constructs the
physical operator and right-hand sides, writes `unpreconditioned_gmres.json`
and its solution-vector sidecar, and exits without building an octree, loading
samples, or constructing RSRS factors.

By default, if a bundle contains the same physical operator and right-hand
sides with different RSRS parameters, the first mode-`3` command remains
“both” and later equivalent commands are rendered as mode `2`. Thus the
unpreconditioned system is solved once. Set `SHARE_UNPRECONDITIONED=False` to
disable this behaviour.

## Sample targets and optional validation

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
        "h": 0.03,
        "kappa": 2.6666667,
        "id_tols": [1e-2, 1e-4],
        "tol_ext_near": 1e-16,
    },
}
```

Or it may contain `options` and `sweep`. A sweep forms the Cartesian product
of its lists and writes one command per combination:

```python
"quasi": {
    "options": {"h": 0.046, "kappa": 1.73913},
    "sweep": {
        "id_tols": [[1e-2], [1e-4]],
        "tol_ext_near": [1e-8, 1e-16],
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

Every generated command contains the four complete JSON blocks expected by
the existing Rust driver. Therefore the script itself is an exact record of
what will run and does not depend on later edits to the dictionary file.
