#!/usr/bin/env python3
"""Small end-to-end Python/RSRS comparison for the three local backends.

This is an integration benchmark, not a performance claim. Bempp-rs and
Bempp-cl represent Laplace single-layer boundary operators. The KiFMM case is
the underlying Laplace particle kernel and is therefore reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from so_kernels import (
    BemppRsLaplaceSingleLayer,
    bempp_cl_laplace_single_layer,
    generate_samples,
    laplace_particle_fmm,
    load_bempp_cl,
)
from bempp_rsrs import (
    InsufficientSamplesError,
    RsrsOptions,
    RsrsSession,
    SampleBundle,
    ThreadOptions,
    TreeOptions,
)


def fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(actual - reference) / max(denominator, np.finfo(float).tiny))


def timed(callable_, repeats: int = 1):
    start = perf_counter()
    value = None
    for _ in range(repeats):
        value = callable_()
    return value, (perf_counter() - start) / repeats


def build_backends(refinement: int, expansion_order: int) -> list[tuple[str, str, Any, Any]]:
    backends: list[tuple[str, str, Any, Any]] = []

    start = perf_counter()
    bempp_rs = BemppRsLaplaceSingleLayer(
        refinement,
        function_space_degree=0,
        expansion_order=expansion_order,
    )
    backends.append(("bempp_rs", "boundary_laplace_single_layer", bempp_rs, perf_counter() - start))

    api = load_bempp_cl().api
    grid = api.shapes.regular_sphere(refinement)
    start = perf_counter()
    bempp_cl = bempp_cl_laplace_single_layer(
        grid,
        assembler="dense",
        expansion_order=expansion_order,
    )
    backends.append(("bempp_cl", "boundary_laplace_single_layer", bempp_cl, perf_counter() - start))

    particle_points = fibonacci_sphere(bempp_rs.dimension)
    start = perf_counter()
    kifmm = laplace_particle_fmm(
        particle_points,
        particle_points,
        expansion_order=expansion_order,
        n_crit=max(8, bempp_rs.dimension // 4),
    )
    # Construction is lazy; include the first evaluation in build/warm-up time.
    kifmm.matvec(np.ones(kifmm.shape[1]))
    backends.append(("kifmm", "particle_laplace_kernel", kifmm, perf_counter() - start))
    return backends


def transpose_of(name: str, operator: Any) -> Any:
    if name == "bempp_cl":
        return operator.transpose_operator
    if name == "bempp_rs":
        return _Transpose(operator)
    # Same source and target points make this real Laplace particle matrix
    # symmetric. This does not imply equivalence to the boundary operators.
    return operator


class _Transpose:
    def __init__(self, operator: Any):
        self.operator = operator
        self.shape = operator.shape

    def matvec(self, values):
        return self.operator.transpose_matvec(values)

    def matmat(self, values):
        return self.operator.transpose_matmat(values)


def run_backend(
    name: str,
    family: str,
    operator: Any,
    build_seconds: float,
    *,
    sample_count: int,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    dimension = int(operator.shape[0])
    transpose = transpose_of(name, operator)
    rng = np.random.default_rng(seed + 1000)
    probe = rng.standard_normal(dimension)
    dual_probe = rng.standard_normal(dimension)

    _, warmup_seconds = timed(lambda: operator.matvec(probe))
    reference, matvec_seconds = timed(lambda: operator.matvec(probe), repeats=3)
    trans_reference, transpose_seconds = timed(lambda: transpose.matvec(dual_probe), repeats=3)
    bilinear_left = float(np.dot(dual_probe, reference))
    bilinear_right = float(np.dot(trans_reference, probe))
    transpose_consistency = abs(bilinear_left - bilinear_right) / max(
        abs(bilinear_left), abs(bilinear_right), np.finfo(float).tiny
    )

    samples, sample_seconds = timed(
        lambda: generate_samples(
            operator,
            sample_count=sample_count,
            seed=seed,
            dtype=np.float64,
        )
    )
    sample_recheck = operator.matmat(samples.y_test[: min(4, sample_count)].T).T
    sample_validation_error = relative_error(
        sample_recheck, samples.y_sketch[: min(4, sample_count)]
    )
    bundle = SampleBundle(samples.y_test, samples.y_sketch)

    options = RsrsOptions(
        tol_id=1e-3,
        tol_null=1e-14,
        tol_ext_near=1e-12,
        tol_diag_ext=1e-12,
        symmetry="Symmetric",
        load_samples=False,
        save_samples=False,
        tree=TreeOptions(max_tree_depth=4, leaf_size=4),
        threads=ThreadOptions(threads, threads),
    )
    points = fibonacci_sphere(dimension)
    session = RsrsSession(points, options)
    factor_start = perf_counter()
    try:
        factors = session.factorize(bundle)
    except InsufficientSamplesError as error:
        request = error.request
        return {
            "backend": name,
            "operator_family": family,
            "dimension": dimension,
            "status": "insufficient_samples",
            "sample_count": sample_count,
            "required_total": request.total_required,
            "additional_y_rows": request.y_rows,
            "additional_z_rows": request.z_rows,
            "build_seconds": build_seconds,
            "warmup_seconds": warmup_seconds,
            "matvec_seconds": matvec_seconds,
            "transpose_matvec_seconds": transpose_seconds,
            "sample_seconds": sample_seconds,
            "sample_validation_error": sample_validation_error,
            "transpose_consistency_error": transpose_consistency,
        }
    factor_seconds = perf_counter() - factor_start

    dense, dense_seconds = timed(lambda: operator.matmat(np.eye(dimension)))
    factor_dense, factor_dense_seconds = timed(lambda: factors.matmat(np.eye(dimension)))
    inverse_dense, inverse_dense_seconds = timed(
        lambda: factors.inverse_matmat(np.eye(dimension))
    )
    factor_error_fro = relative_error(factor_dense, dense)

    rhs = rng.standard_normal(dimension)
    exact_solution = np.linalg.solve(dense, rhs)
    approximate_solution, inverse_apply_seconds = timed(lambda: factors.inverse_matvec(rhs))
    solve_error = relative_error(approximate_solution, exact_solution)
    solve_residual = relative_error(operator.matvec(approximate_solution), rhs)

    iterations: dict[str, int] = {}

    def solve(label: str, preconditioner: LinearOperator | None):
        count = 0

        def callback(_):
            nonlocal count
            count += 1

        start = perf_counter()
        solution, info = gmres(
            LinearOperator(operator.shape, matvec=operator.matvec, dtype=np.float64),
            rhs,
            M=preconditioner,
            rtol=1e-8,
            atol=0.0,
            restart=min(20, dimension),
            maxiter=10 * dimension,
            callback=callback,
            callback_type="pr_norm",
        )
        iterations[label] = count
        return info, solution, perf_counter() - start

    no_prec_info, no_prec_solution, no_prec_seconds = solve("no_preconditioner", None)
    preconditioner = LinearOperator(
        operator.shape, matvec=factors.inverse_matvec, dtype=np.float64
    )
    prec_info, prec_solution, prec_seconds = solve("rsrs", preconditioner)

    return {
        "backend": name,
        "operator_family": family,
        "dimension": dimension,
        "status": "complete",
        "sample_count": sample_count,
        "build_seconds": build_seconds,
        "warmup_seconds": warmup_seconds,
        "matvec_seconds": matvec_seconds,
        "transpose_matvec_seconds": transpose_seconds,
        "sample_seconds": sample_seconds,
        "sample_seconds_per_row": sample_seconds / sample_count,
        "factor_seconds": factor_seconds,
        "factor_apply_matrix_seconds": factor_dense_seconds,
        "inverse_apply_matrix_seconds": inverse_dense_seconds,
        "inverse_apply_vector_seconds": inverse_apply_seconds,
        "dense_reference_seconds": dense_seconds,
        "sample_validation_error": sample_validation_error,
        "transpose_consistency_error": transpose_consistency,
        "factor_error_fro": factor_error_fro,
        "solve_error_rhs": solve_error,
        "solve_relative_residual": solve_residual,
        "gmres_no_prec_iterations": iterations["no_preconditioner"],
        "gmres_no_prec_seconds": no_prec_seconds,
        "gmres_no_prec_info": int(no_prec_info),
        "gmres_no_prec_solution_error": relative_error(no_prec_solution, exact_solution),
        "gmres_rsrs_iterations": iterations["rsrs"],
        "gmres_rsrs_seconds": prec_seconds,
        "gmres_rsrs_info": int(prec_info),
        "gmres_rsrs_solution_error": relative_error(prec_solution, exact_solution),
    }


def write_reports(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps({"config": config, "results": rows}, indent=2) + "\n"
    )
    columns = sorted({key for row in rows for key in row})
    with (output / "report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Small Python backend integration report",
        "",
        "Bempp-rs and Bempp-cl are boundary single-layer operators. KiFMM is a particle-kernel test and is not a numerical equivalence comparison.",
        "",
        "| backend | family | n | status | samples | factor s | factor error | solve error | GMRES no prec | GMRES RSRS |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        display = dict(row)
        display.update(
            factor_seconds=f"{row.get('factor_seconds', float('nan')):.4g}",
            factor_error_fro=f"{row.get('factor_error_fro', float('nan')):.3e}",
            solve_error_rhs=f"{row.get('solve_error_rhs', float('nan')):.3e}",
            gmres_no_prec_iterations=row.get("gmres_no_prec_iterations", "-"),
            gmres_rsrs_iterations=row.get("gmres_rsrs_iterations", "-"),
        )
        lines.append(
            "| {backend} | {operator_family} | {dimension} | {status} | {sample_count} | {factor_seconds} | {factor_error_fro} | {solve_error_rhs} | {gmres_no_prec_iterations} | {gmres_rsrs_iterations} |".format(
                **display,
            )
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--expansion-order", type=int, default=6)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    config = vars(args).copy()
    config["output"] = str(config["output"])
    rows = []
    for name, family, operator, build_seconds in build_backends(
        args.refinement, args.expansion_order
    ):
        print(f"[small-backend-test] running {name} ({family}), n={operator.shape[0]}")
        row = run_backend(
            name,
            family,
            operator,
            build_seconds,
            sample_count=args.samples,
            seed=args.seed,
            threads=args.threads,
        )
        rows.append(row)
        write_reports(args.output, rows, config)
        print(f"[small-backend-test] {name}: {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
