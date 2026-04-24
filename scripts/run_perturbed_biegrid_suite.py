#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MESH_WIDTH = 0.01
NUM_THREADS = 1
MODE = "Constant"

CASES = [
    {
        "label": "Real symmetric perturbed BIEGrid, 10k serial",
        "operator_type": "BIEGridRealSymmetricPerturbed",
        "symmetry": "Symmetric",
        "max_adjoint": "1.0e-12",
        "max_adjoint_inv": "5.0e-13",
        "max_norm_2": "5.0e-7",
        "max_solve": "1.0e-5",
    },
    {
        "label": "Real nonsymmetric perturbed BIEGrid, 10k serial",
        "operator_type": "BIEGridRealPerturbed",
        "symmetry": "NoSymm",
        "max_adjoint": "1.0e-12",
        "max_adjoint_inv": "5.0e-13",
        "max_norm_2": "5.0e-7",
        "max_solve": "1.5e-4",
    },
    {
        "label": "Complex symmetric perturbed BIEGrid, 10k serial",
        "operator_type": "BIEGridComplexSymmetricPerturbed",
        "symmetry": "Symmetric",
        "max_adjoint": "1.0e-12",
        "max_adjoint_inv": "5.0e-13",
        "max_norm_2": "5.0e-7",
        "max_solve": "5.0e-5",
    },
    {
        "label": "Complex nonsymmetric perturbed BIEGrid, 10k serial",
        "operator_type": "BIEGridComplexPerturbed",
        "symmetry": "NoSymm",
        "max_adjoint": "1.0e-12",
        "max_adjoint_inv": "5.0e-13",
        "max_norm_2": "5.0e-7",
        "max_solve": "5.0e-5",
    },
]


def mesh_width_slug(mesh_width: float) -> str:
    base, exp = f"{mesh_width:.2e}".split("e")
    exp = str(int(exp))
    return f"{base}e{exp}"


def error_stats_path(operator_type: str, symmetry: str) -> Path:
    mesh_width_str = mesh_width_slug(MESH_WIDTH)
    return REPO_ROOT / (
        "results/"
        f"square_{operator_type}_precision_double_mesh_width_{mesh_width_str}_od_10_0.00_num_threads_{NUM_THREADS}/"
        "rsrs_null_Projection_toln_1e-16_os_40_osdiag_40_initsam_0_"
        f"fsamp_{MODE}_mrnk_1_mlvl_1_"
        f"{symmetry}_rpick_Min_next_LuLstSq_tolextn_1e-16_db_ext_LuLstSq_tol_lstsq_1e-16_rrqr/"
        "error_stats_4e1.json"
    )


def fmt(v):
    if v is None:
        return "NA"
    return f"{float(v):.3e}"


def run_case(case):
    cmd = [
        sys.executable,
        "scripts/check_biegrid_regression.py",
        "--operator-type",
        case["operator_type"],
        "--symmetry",
        case["symmetry"],
        "--fixed-rank-sampling-mode",
        MODE,
        "--num-threads",
        str(NUM_THREADS),
        "--mesh-width",
        str(MESH_WIDTH),
        "--expected-samples",
        "1520",
        "--suite-name",
        case["label"],
        "--run-id",
        "serial",
        "--max-adjoint-consistency-error",
        case["max_adjoint"],
        "--max-adjoint-consistency-error-inv",
        case["max_adjoint_inv"],
        "--max-norm-2-error",
        case["max_norm_2"],
        "--max-solve-error",
        case["max_solve"],
    ]

    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    stats_path = error_stats_path(case["operator_type"], case["symmetry"])

    metrics = {}
    if stats_path.exists():
        metrics = json.loads(stats_path.read_text())

    return {
        "label": case["label"],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": proc.returncode,
        "stats_path": str(stats_path),
        "err_solve_2": metrics.get("err_solve_2"),
        "err_solve_fro": metrics.get("err_solve_fro"),
        "solve_error_rhs": metrics.get("solve_error_rhs"),
        "adjoint": metrics.get("adjoint_consistency_error"),
        "adjoint_inv": metrics.get("adjoint_consistency_error_inv"),
    }


def print_summary(rows):
    print("\n==================== Suite Summary (10k serial, Constant) ====================")
    print(
        f"{'Case':50} {'Status':6} {'err_solve_2':>12} {'err_solve_fro':>14} "
        f"{'solve_rhs':>12} {'adj':>12} {'adj_inv':>12}"
    )
    for row in rows:
        print(
            f"{row['label'][:50]:50} {row['status']:6} {fmt(row['err_solve_2']):>12} "
            f"{fmt(row['err_solve_fro']):>14} {fmt(row['solve_error_rhs']):>12} "
            f"{fmt(row['adjoint']):>12} {fmt(row['adjoint_inv']):>12}"
        )
    print("===============================================================================")


def main() -> int:
    rows = []
    for case in CASES:
        print(f"\n--- Running: {case['label']} ---")
        rows.append(run_case(case))

    print_summary(rows)

    failed = [row for row in rows if row["status"] != "PASS"]
    if failed:
        print("\nOne or more perturbed regressions failed.")
        return 1

    print("\nAll perturbed regressions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
