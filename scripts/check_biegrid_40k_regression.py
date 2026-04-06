#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ERROR_STATS = REPO_ROOT / (
    "results/"
    "square_BIEGrid_precision_double_mesh_width_5.03e-3_od_10_0.00_num_threads_4/"
    "rsrs_null_Projection_toln_1e-16_os_40_osdiag_40_initsam_0_mrnk_1_mlvl_1_"
    "Symmetric_rpick_Min_next_LuLstSq_tolextn_1e-16_db_ext_LuLstSq_tol_lstsq_1e-16_rrqr/"
    "error_stats_4e1.json"
)

STRUCTURED_OPERATOR_ARGS = {
    "structured_operator_type": "BIEGrid",
    "precision": "Double",
}
TEST_GEOMETRY_ARGS = {
    "id_tols": [40],
    "dim_args": [{"MeshWidth": 0.005025125628140704}],
    "geometry_type": "Square",
    "max_tree_depth": 10,
    "n_sources": 1,
    "assembler": "Dense",
}
RSRS_ARGS = {
    "oversampling": 40,
    "oversampling_diag_blocks": 40,
    "min_num_samples": 0,
    "initial_num_samples": 0,
    "run_seed": 12345,
    "shift": {"type": "False"},
    "null_method": "Projection",
    "qr_method": "RRQR",
    "near_block_extraction_method": "LuLstSq",
    "diag_block_extraction_method": "LuLstSq",
    "lu_pivot_method": {"type": "Lu", "value": 0},
    "diag_pivot_method": {"type": "Lu", "value": 0},
    "tol_null": 1e-16,
    "tol_id": 40,
    "tol_ext_near": 1e-16,
    "tol_diag_ext": 1e-16,
    "min_rank": 1,
    "min_level": 1,
    "symmetry": "Symmetric",
    "rank_picking": "Min",
    "fact_type": "Joint",
    "save_samples": False,
    "load_samples": False,
    "num_threads": 4,
    "flush_factors": False,
    "store_far": False,
    "symmetric": None,
}
OUTPUT_ARGS = {
    "solve": {"False": None},
    "plot": False,
    "dense_errors": False,
    "factors_cn": False,
    "results_output": "All",
}

THRESHOLDS = {
    "tot_num_samples": 1520,
    "adjoint_consistency_error": 1.0e-6,
    "norm_2_error": 5.0e-6,
    "solve_error": 2.0e-3,
}


def main() -> int:
    start_time = time.time()
    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MKL_DOMAIN_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "GOTO_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    env.pop("RAYON_NUM_THREADS", None)

    command = [
        "cargo",
        "run",
        "--release",
        json.dumps(STRUCTURED_OPERATOR_ARGS),
        json.dumps(TEST_GEOMETRY_ARGS),
        json.dumps(RSRS_ARGS),
        json.dumps(OUTPUT_ARGS),
    ]
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    if not ERROR_STATS.exists():
        raise FileNotFoundError(f"Missing regression output: {ERROR_STATS}")
    if ERROR_STATS.stat().st_mtime + 1.0 < start_time:
        raise RuntimeError(f"Regression output was not refreshed: {ERROR_STATS}")

    metrics = json.loads(ERROR_STATS.read_text())

    print("BIEGrid 40k regression metrics:")
    for key in (
        "tot_num_samples",
        "norm_2_error",
        "norm_fro_error",
        "adjoint_consistency_error",
        "self_adjoint_apply_error",
        "solve_error",
    ):
        print(f"  {key} = {metrics.get(key)}")

    failures = []
    if metrics.get("tot_num_samples") != THRESHOLDS["tot_num_samples"]:
        failures.append(
            f"tot_num_samples expected {THRESHOLDS['tot_num_samples']} "
            f"got {metrics.get('tot_num_samples')}"
        )
    for key in ("adjoint_consistency_error", "norm_2_error", "solve_error"):
        value = metrics.get(key)
        if value is None or value >= THRESHOLDS[key]:
            failures.append(f"{key} expected < {THRESHOLDS[key]} got {value}")

    if failures:
        print("Regression check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
