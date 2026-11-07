"""Copy this file and edit its dictionaries to define experiment bundles.

Run ``rsrs-experiment options`` for the complete numbered option guide.
Only values that differ from DEFAULTS need to be specified in each test.
"""

DEFAULTS = {
    "operator": 19,                 # BemppClHelmholtzCombined
    "precision": 1,                # Double
    "dimension": 1,                # KappaAndMeshwidth
    "geometry": 6,                 # F16
    "assembler": 1,                # ExaFMM
    "symmetry": 0,                 # NoSymm
    "nonsymmetric_combination": 1, # Concat
    "fixed_rank_sampling": 0,      # PerLevel
    "factorization": 1,            # Joint
    "max_tree_depth": 16,
    "num_threads": 56,
    "load_samples": True,
    "save_samples": False,
    "solve_tol": 1e-7,
    "gmres": 3,                     # Both; equivalent bundle tests share baseline GMRES
}

# When true, the first equivalent test runs both GMRES variants and later
# RSRS-parameter variants automatically use preconditioned-only GMRES.
SHARE_UNPRECONDITIONED = True

TESTS = {
    "quasi": {
        "options": {
            "h": 0.046,
            "kappa": 1.73913,
            "id_tols": [1e-2, 1e-4],
        },
        # A Cartesian sweep: this produces two commands.
        "sweep": {"tol_ext_near": [1e-8, 1e-16]},
    },
    "intermediate": {
        "h": 0.03,
        "kappa": 2.6666667,
        "id_tols": [1e-2, 1e-4],
        "tol_ext_near": 1e-16,
    },
}

BUNDLES = {
    "f16_all": ["quasi", "intermediate"],
}

# Used only when --kind sbatch is selected. Keys become #SBATCH --key=value.
SLURM = {
    "job_name": "f16_rsrs",
    "account": "PAS3325",
    "time": "2-00:00:00",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 56,
    "mem": "200G",
}
