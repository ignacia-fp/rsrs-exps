"""Small local BIEGrid bundle using the Python-only experiment path."""

DEFAULTS = {
    "operator": 3,
    "operator_parameters": {"cells_per_axis": 8},
    "sample_cache": "results/sample_cache/biegrid_complex",
    "sampling": {
        "target": 200,
        "workers": 4,
        "batch_size": 10,
        "symmetry": 2,
        "validation": "sampled",
    },
    "rsrs": {
        "tol_id": 1.0e-2,
        "max_tree_depth": 6,
        "symmetry": 0,
        "nonsymmetric_id_combination": 1,
    },
    "diagnostics": {
        "gmres": {"mode": 3, "tolerance": 1.0e-7, "rhs_count": 3}
    },
}

TESTS = {
    "baseline": {},
    "near_tolerance": {
        "options": {},
        "sweep": {"rsrs.tol_ext_near": [1.0e-8, 1.0e-16]},
    },
}

BUNDLES = {"all": ["baseline", "near_tolerance"]}
SHARE_UNPRECONDITIONED = True
SLURM = {"job_name": "biegrid", "cpus_per_task": 4, "mem": "8G", "time": "01:00:00"}
GENERATE = {"selection": "all", "name": "biegrid_example", "kind": "sh"}
