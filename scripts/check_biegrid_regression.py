#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

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
    "num_threads": 1,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-type", default="BIEGrid")
    parser.add_argument("--symmetry", default="Symmetric")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--mesh-width", type=float, default=0.005025125628140704)
    parser.add_argument("--suite-name", default="BIEGrid regression")
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--load-samples", action="store_true")
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--max-adjoint-consistency-error", type=float, default=1.0e-6)
    parser.add_argument("--max-adjoint-consistency-error-inv", type=float, default=None)
    parser.add_argument("--max-norm-2-error", type=float, default=5.0e-6)
    parser.add_argument("--max-solve-error", type=float, default=2.0e-3)
    return parser.parse_args()


def mesh_width_slug(mesh_width: float) -> str:
    base, exp = f"{mesh_width:.2e}".split("e")
    exp = str(int(exp))
    return f"{base}e{exp}"


def error_stats_path(operator_type: str, symmetry: str, mesh_width: float, num_threads: int) -> Path:
    mesh_width_str = mesh_width_slug(mesh_width)
    return REPO_ROOT / (
        "results/"
        f"square_{operator_type}_precision_double_mesh_width_{mesh_width_str}_od_10_0.00_num_threads_{num_threads}/"
        "rsrs_null_Projection_toln_1e-16_os_40_osdiag_40_initsam_0_mrnk_1_mlvl_1_"
        f"{symmetry}_rpick_Min_next_LuLstSq_tolextn_1e-16_db_ext_LuLstSq_tol_lstsq_1e-16_rrqr/"
        "error_stats_4e1.json"
    )


def main() -> int:
    args = parse_args()
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
    env["RAYON_NUM_THREADS"] = str(args.num_threads)
    env["RSRS_RUN_SEED"] = "12345"
    test_geometry_args = dict(TEST_GEOMETRY_ARGS)
    test_geometry_args["dim_args"] = [{"MeshWidth": args.mesh_width}]
    rsrs_args = dict(RSRS_ARGS)
    rsrs_args["num_threads"] = args.num_threads
    rsrs_args["symmetry"] = args.symmetry
    rsrs_args["save_samples"] = args.save_samples
    rsrs_args["load_samples"] = args.load_samples
    structured_operator_args = {
        "structured_operator_type": args.operator_type,
        "precision": "Double",
    }
    error_stats = error_stats_path(args.operator_type, args.symmetry, args.mesh_width, args.num_threads)

    command = [
        "cargo",
        "run",
        "--release",
        json.dumps(structured_operator_args),
        json.dumps(test_geometry_args),
        json.dumps(rsrs_args),
        json.dumps(OUTPUT_ARGS),
    ]
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    if not error_stats.exists():
        raise FileNotFoundError(f"Missing regression output: {error_stats}")
    if error_stats.stat().st_mtime + 1.0 < start_time:
        raise RuntimeError(f"Regression output was not refreshed: {error_stats}")

    metrics = json.loads(error_stats.read_text())

    print(f"{args.suite_name} ({args.run_id}) metrics:")
    for key in (
        "tot_num_samples",
        "norm_2_error",
        "norm_fro_error",
        "adjoint_consistency_error",
        "adjoint_consistency_error_inv",
        "self_adjoint_apply_error",
        "solve_error",
    ):
        print(f"  {key} = {metrics.get(key)}")

    failures = []
    if args.expected_samples is not None and metrics.get("tot_num_samples") != args.expected_samples:
        failures.append(
            f"tot_num_samples expected {args.expected_samples} "
            f"got {metrics.get('tot_num_samples')}"
        )
    metric_thresholds = {
        "adjoint_consistency_error": args.max_adjoint_consistency_error,
        "adjoint_consistency_error_inv": args.max_adjoint_consistency_error_inv,
        "norm_2_error": args.max_norm_2_error,
        "solve_error": args.max_solve_error,
    }
    for key, threshold in metric_thresholds.items():
        if threshold is None:
            continue
        value = metrics.get(key)
        if value is None or value >= threshold:
            failures.append(f"{key} expected < {threshold} got {value}")

    if failures:
        print(f"{args.suite_name} ({args.run_id}) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"{args.suite_name} ({args.run_id}) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
