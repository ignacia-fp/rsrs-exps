"""Run sample-fed RSRS and extract Python-owned numerical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from threadpoolctl import threadpool_limits

from .cache import SampleCache
from .sampling import ParallelSamplingConfig, sample_operator


@dataclass(frozen=True)
class RsrsRunResult:
    factors: Any
    report: dict[str, Any]


class _SingleBlasRsrsFactors:
    """Keep BLAS serial inside Rayon-parallel RSRS factor applications."""

    def __init__(self, factors: Any, *, blas_threads: int = 1):
        self._factors = factors
        self._blas_threads = int(blas_threads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._factors, name)

    def apply(self, values, **options):
        with threadpool_limits(limits=self._blas_threads, user_api="blas"):
            return self._factors.apply(values, **options)

    def matvec(self, values):
        return self.apply(values)

    def matmat(self, values):
        return self.apply(values)

    def inverse_matvec(self, values):
        return self.apply(values, inverse=True)

    def inverse_matmat(self, values):
        return self.apply(values, inverse=True)

    def aslinearoperator(self, *, inverse: bool = False) -> LinearOperator:
        dimension = int(self.dimension)
        dtype = np.dtype(self.dtype)
        return LinearOperator(
            (dimension, dimension),
            dtype=dtype,
            matvec=lambda values: self.apply(values, inverse=inverse),
            matmat=lambda values: self.apply(values, inverse=inverse),
            rmatvec=lambda values: self.apply(
                values, transpose=True, conjugate=True, inverse=inverse
            ),
            rmatmat=lambda values: self.apply(
                values, transpose=True, conjugate=True, inverse=inverse
            ),
        )


def _write_report_files(output: Path, report: Mapping[str, Any]) -> None:
    """Write the combined report plus focused error and solve reports."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "rsrs_report.json").write_text(json.dumps(report, indent=2) + "\n")
    common = {
        key: report[key]
        for key in (
            "created_utc",
            "status",
            "cache",
            "operator",
            "sampling_operator",
            "dimension",
            "dtype",
            "sample_count",
            "configuration",
            "execution",
        )
        if key in report
    }
    if "errors" in report:
        error_report = {
            **common,
            "errors": report["errors"],
            "error_estimation": report.get("error_estimation", {}),
            "error_computation_seconds": report.get("timings_seconds", {}).get(
                "error_computation"
            ),
        }
        (output / "errors_report.json").write_text(
            json.dumps(error_report, indent=2) + "\n"
        )
        print(f"[report] errors={output / 'errors_report.json'}", flush=True)
    if "rsrs" in report:
        factorization_report = {
            **common,
            "factorization_seconds": report.get("timings_seconds", {}).get(
                "factorization"
            ),
            "rsrs": report["rsrs"],
        }
        (output / "factorization_report.json").write_text(
            json.dumps(factorization_report, indent=2) + "\n"
        )
        print(
            f"[report] factorization={output / 'factorization_report.json'}",
            flush=True,
        )
    if "gmres" in report:
        solves_report = {**common, "gmres": report["gmres"]}
        (output / "solves_report.json").write_text(
            json.dumps(solves_report, indent=2) + "\n"
        )
        print(f"[report] solves={output / 'solves_report.json'}", flush=True)
    print(f"[report] combined={output / 'rsrs_report.json'}", flush=True)


def _linear_operator(operator: Any) -> LinearOperator:
    if hasattr(operator, "aslinearoperator"):
        return operator.aslinearoperator()
    return LinearOperator(
        operator.shape,
        dtype=np.dtype(operator.dtype),
        matvec=operator.matvec,
        matmat=getattr(operator, "matmat", None),
    )


def _right_hand_sides(operator: Any, *, count: int, seed: int) -> np.ndarray:
    provider = getattr(operator, "right_hand_sides", None)
    if callable(provider):
        try:
            values = np.asarray(provider(count=count))
        except (AttributeError, NotImplementedError):
            pass
        else:
            return values.reshape(operator.shape[0], count)
    dtype = np.dtype(operator.dtype)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((operator.shape[0], count)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * rng.standard_normal(values.shape)
    return values


def _gmres_report(
    operator: Any,
    factors: Any | None,
    *,
    mode: str,
    tolerance: float,
    max_iterations: int,
    restart: int | None,
    rhs_count: int,
    seed: int,
    log_every: int = 1,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": mode, "solves": {}}
    if mode not in {"unpreconditioned", "preconditioned", "both"}:
        raise ValueError(f"unknown GMRES mode {mode!r}")
    if log_every < 0:
        raise ValueError("GMRES log_every must be nonnegative")
    if mode in {"preconditioned", "both"} and factors is None:
        raise ValueError("preconditioned GMRES requires RSRS factors")
    matrix = _linear_operator(operator)
    rhs = _right_hand_sides(operator, count=rhs_count, seed=seed)
    variants = []
    if mode in {"unpreconditioned", "both"}:
        variants.append(("unpreconditioned", None))
    if mode in {"preconditioned", "both"}:
        variants.append(("preconditioned", factors.aslinearoperator(inverse=True)))
    report: dict[str, Any] = {
        "mode": mode,
        "tolerance": tolerance,
        "solves": {name: [] for name, _ in variants},
    }
    direct = []
    for column in range(rhs_count):
        if factors is not None:
            print(
                f"[gmres][direct-rsrs] rhs={column + 1}/{rhs_count} applying inverse",
                flush=True,
            )
            started = time.perf_counter()
            solution = factors.inverse_matvec(rhs[:, column])
            denominator = max(
                float(np.linalg.norm(rhs[:, column])), np.finfo(float).tiny
            )
            direct.append(
                {
                    "rhs": column,
                    "final_true_relative_residual": float(
                        np.linalg.norm(_apply(operator, solution) - rhs[:, column])
                        / denominator
                    ),
                    "seconds": time.perf_counter() - started,
                }
            )
            print(
                f"[gmres][direct-rsrs] rhs={column + 1}/{rhs_count} "
                f"true_relative_residual={direct[-1]['final_true_relative_residual']:.6e} "
                f"seconds={direct[-1]['seconds']:.3f}",
                flush=True,
            )

        for name, preconditioner in variants:
            history: list[float] = []
            print(
                f"[gmres][{name}] rhs={column + 1}/{rhs_count} starting "
                f"rtol={tolerance:.3e} max_iterations={max_iterations} restart={restart}",
                flush=True,
            )

            def record_residual(value: float) -> None:
                residual = float(value)
                history.append(residual)
                iteration = len(history)
                if log_every and (iteration == 1 or iteration % log_every == 0):
                    print(
                        f"[gmres][{name}] rhs={column + 1}/{rhs_count} "
                        f"iteration={iteration} callback_residual={residual:.6e}",
                        flush=True,
                    )

            started = time.perf_counter()
            solution, info = gmres(
                matrix,
                rhs[:, column],
                M=preconditioner,
                rtol=tolerance,
                atol=0.0,
                restart=restart,
                maxiter=max_iterations,
                callback=record_residual,
                # ``legacy`` reports the preconditioned residual norm while
                # interpreting maxiter as inner iterations, matching the
                # historical RSRS experiment driver and its 500-iteration cap.
                callback_type="legacy",
            )
            denominator = max(float(np.linalg.norm(rhs[:, column])), np.finfo(float).tiny)
            true_residual = float(
                np.linalg.norm(_apply(operator, solution) - rhs[:, column]) / denominator
            )
            report["solves"][name].append(
                {
                    "rhs": column,
                    "info": int(info),
                    "iterations": len(history),
                    "recorded_residuals": history,
                    "final_true_relative_residual": true_residual,
                    "seconds": time.perf_counter() - started,
                }
            )
            print(
                f"[gmres][{name}] rhs={column + 1}/{rhs_count} complete "
                f"iterations={len(history)} info={int(info)} "
                f"true_relative_residual={true_residual:.6e} "
                f"seconds={report['solves'][name][-1]['seconds']:.3f}",
                flush=True,
            )
    if factors is not None:
        report["direct_rsrs_inverse"] = direct
        report["rel_err_approx_inv"] = [
            item["final_true_relative_residual"] for item in direct
        ]
    return report


MAX_OPERATOR_COLUMNS = 30


def _chunked_matrix_call(method, values: np.ndarray) -> np.ndarray:
    """Apply a matrix method without giving a backend more than 30 RHS."""

    if values.ndim != 2 or values.shape[1] <= MAX_OPERATOR_COLUMNS:
        return np.asarray(method(values))
    return np.concatenate(
        [
            np.asarray(method(values[:, start : start + MAX_OPERATOR_COLUMNS]))
            for start in range(0, values.shape[1], MAX_OPERATOR_COLUMNS)
        ],
        axis=1,
    )


def _apply(operator: Any, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1 and hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(values))
    if values.ndim == 2 and hasattr(operator, "matmat"):
        return _chunked_matrix_call(operator.matmat, values)
    return _chunked_matrix_call(operator.__matmul__, values)


def _progress_apply(
    operator: Any,
    values: np.ndarray,
    *,
    label: str,
    batch_size: int = 10,
) -> np.ndarray:
    """Apply a block while reporting progress through expensive RHS columns."""

    if values.ndim != 2:
        return _apply(operator, values)
    columns = values.shape[1]
    outputs = []
    for start in range(0, columns, batch_size):
        stop = min(start + batch_size, columns)
        print(
            f"[errors] operation={label} columns={start + 1}:{stop}/{columns} starting",
            flush=True,
        )
        started = time.perf_counter()
        outputs.append(_apply(operator, values[:, start:stop]))
        print(
            f"[errors] operation={label} columns={start + 1}:{stop}/{columns} "
            f"complete seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return np.concatenate(outputs, axis=1)


def _adjoint_apply(operator: Any, values: np.ndarray) -> np.ndarray:
    """Apply the Hermitian adjoint using an explicit algebraic transpose."""

    # Some symmetric matrix-free backends deliberately do not implement a
    # separate transpose traversal.  For A^T=A, A^*x=conj(A conj(x)).
    if bool(getattr(operator, "symmetric", False)):
        return np.conjugate(_apply(operator, np.conjugate(values)))
    transpose = getattr(operator, "transpose_operator", None)
    if transpose is not None:
        return np.conjugate(_apply(transpose, np.conjugate(values)))
    method = getattr(
        operator,
        "transpose_matvec" if values.ndim == 1 else "transpose_matmat",
        None,
    )
    if callable(method):
        return np.conjugate(_chunked_matrix_call(method, np.conjugate(values)))
    linear = operator.aslinearoperator() if hasattr(operator, "aslinearoperator") else None
    if linear is not None:
        if values.ndim == 1:
            return np.asarray(linear.rmatvec(values))
        return _chunked_matrix_call(linear.rmatmat, values)
    raise TypeError("spectral errors require an explicit transpose/adjoint application")


def _transpose_apply(operator: Any, values: np.ndarray) -> np.ndarray:
    """Apply the algebraic transpose without conjugating complex values."""

    if bool(getattr(operator, "symmetric", False)):
        return _apply(operator, values)
    transpose = getattr(operator, "transpose_operator", None)
    if transpose is not None:
        return _apply(transpose, values)
    method = getattr(
        operator,
        "transpose_matvec" if values.ndim == 1 else "transpose_matmat",
        None,
    )
    if callable(method):
        return _chunked_matrix_call(method, values)
    raise TypeError("historical right-application errors require an algebraic transpose")


def _column_max_relative(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = np.linalg.norm(actual - expected, axis=0)
    denominator = np.maximum(np.linalg.norm(expected, axis=0), np.finfo(float).tiny)
    return float(np.max(numerator / denominator))


def _adjoint_consistency(forward, adjoint, probes: np.ndarray) -> float:
    """Return max |<x,Fy>-<F*x,y>| relative to either inner product."""

    split = probes.shape[1] // 2
    x = probes[:, :split]
    y = probes[:, split : 2 * split]
    fy = np.asarray(forward(y))
    f_star_x = np.asarray(adjoint(x))
    lhs = np.sum(np.conjugate(x) * fy, axis=0)
    rhs = np.sum(np.conjugate(f_star_x) * y, axis=0)
    denominator = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0e-14)
    return float(np.max(np.abs(lhs - rhs) / denominator))


def _block_spectral_norm(
    apply,
    adjoint,
    *,
    dimension: int,
    dtype: np.dtype,
    seed: int,
    iterations: int,
    block_size: int,
    tolerance: float,
    progress_label: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Estimate a largest singular value through block power iteration."""

    width = min(max(int(block_size), 1), dimension, MAX_OPERATOR_COLUMNS)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((dimension, width)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        q += 1j * rng.standard_normal(q.shape)
    q, _ = np.linalg.qr(q, mode="reduced")
    previous = 0.0
    converged = False
    completed = 0
    for completed in range(1, max(int(iterations), 1) + 1):
        q, _ = np.linalg.qr(np.asarray(adjoint(np.asarray(apply(q)))), mode="reduced")
        image = np.asarray(apply(q))
        estimate = float(np.linalg.svd(image, compute_uv=False)[0])
        if progress_label is not None:
            print(
                f"[errors][spectral] estimate={progress_label} "
                f"iteration={completed}/{max(int(iterations), 1)} value={estimate:.6e}",
                flush=True,
            )
        if previous > 0.0 and abs(estimate - previous) <= tolerance * estimate:
            converged = True
            break
        previous = estimate
    return estimate, {
        "iterations": completed,
        "block_size": width,
        "converged": converged,
        "relative_tolerance": tolerance,
    }


def _relerr(actual: np.ndarray, expected: np.ndarray, order: Any = 2) -> float:
    denominator = max(float(np.linalg.norm(expected, ord=order)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected, ord=order) / denominator)


def _diagnostic_cost_estimate(
    operator: Any,
    factors: Any,
    *,
    seed: int,
    probes: int,
    spectral_options: Mapping[str, Any] | None,
    historical_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Time representative matvecs and estimate target-operator diagnostic work."""

    dimension = int(operator.shape[0])
    dtype = np.dtype(getattr(operator, "dtype", factors.dtype))
    rng = np.random.default_rng(seed + 7919)
    vector = rng.standard_normal(dimension).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        vector += 1j * rng.standard_normal(dimension)

    def timed(label: str, apply) -> float:
        print(f"[errors][estimate] timing={label} starting", flush=True)
        started = time.perf_counter()
        apply(vector)
        seconds = time.perf_counter() - started
        print(
            f"[errors][estimate] timing={label} seconds={seconds:.3f}",
            flush=True,
        )
        return seconds

    forward_seconds = timed("target_forward_matvec", lambda x: _apply(operator, x))
    adjoint_seconds = timed(
        "target_adjoint_matvec", lambda x: _adjoint_apply(operator, x)
    )
    rsrs_apply_seconds = timed("rsrs_apply", factors.matvec)
    rsrs_inverse_seconds = timed("rsrs_inverse_apply", factors.inverse_matvec)

    historical = dict(historical_options or {})
    application_samples = max(int(historical.get("application_samples", 10)), 1)
    spectral = dict(spectral_options or {})
    spectral_enabled = bool(spectral.get("enabled", True))
    iterations = max(int(spectral.get("iterations", 10)), 1) if spectral_enabled else 0
    block_size = min(
        max(int(spectral.get("block_size", 2)), 1), dimension, MAX_OPERATOR_COLUMNS
    )

    # The non-spectral diagnostics use two forward probe applications,
    # one forward and two transpose historical applications, plus one known
    # solution application. Each spectral iteration uses six forward and
    # three adjoint block applications across the three target-dependent
    # estimates. Early convergence can reduce this upper bound.
    forward_equivalent = (
        2 * max(int(probes), 1)
        + application_samples
        + 1
        + 6 * iterations * block_size
    )
    adjoint_equivalent = 2 * application_samples + 3 * iterations * block_size
    target_upper_seconds = (
        forward_equivalent * forward_seconds
        + adjoint_equivalent * adjoint_seconds
    )
    print(
        "[errors][estimate] configured target-operator work "
        f"forward_matvec_equivalents<={forward_equivalent} "
        f"adjoint_matvec_equivalents<={adjoint_equivalent} "
        f"target_operator_seconds<={target_upper_seconds:.1f} "
        "(upper bound; spectral estimates may converge early; excludes RSRS and QR/SVD work)",
        flush=True,
    )
    return {
        "target_forward_matvec_seconds": forward_seconds,
        "target_adjoint_matvec_seconds": adjoint_seconds,
        "rsrs_apply_seconds": rsrs_apply_seconds,
        "rsrs_inverse_apply_seconds": rsrs_inverse_seconds,
        "forward_matvec_equivalents_upper_bound": forward_equivalent,
        "adjoint_matvec_equivalents_upper_bound": adjoint_equivalent,
        "target_operator_seconds_upper_bound": target_upper_seconds,
    }


def _errors(
    operator: Any,
    factors: Any,
    *,
    seed: int,
    probes: int,
    spectral_options: Mapping[str, Any] | None = None,
    historical_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    dimension = int(operator.shape[0])
    dtype = np.dtype(getattr(operator, "dtype", factors.dtype))
    step_timings: dict[str, float] = {}

    def begin_step(name: str) -> float:
        print(f"[errors] step={name} starting", flush=True)
        return time.perf_counter()

    def end_step(name: str, started: float) -> None:
        seconds = time.perf_counter() - started
        step_timings[name] = seconds
        print(f"[errors] step={name} complete seconds={seconds:.3f}", flush=True)

    probe_started = begin_step("frobenius_probes")
    rng = np.random.default_rng(seed)
    test = rng.standard_normal((dimension, probes)).astype(dtype)
    rhs = rng.standard_normal((dimension, probes)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        test += 1j * rng.standard_normal(test.shape)
        rhs += 1j * rng.standard_normal(rhs.shape)

    true_apply = _progress_apply(
        operator, test, label="frobenius_target_apply"
    )
    approximate_apply = factors.matmat(test)
    approximate_solution = factors.inverse_matmat(rhs)
    inverse_image = _progress_apply(
        operator, approximate_solution, label="frobenius_inverse_residual"
    )
    probe_scale = np.sqrt(max(probes, 1))
    metrics = {
        "probe_apply_relative_fro": _relerr(approximate_apply, true_apply, "fro"),
        "inverse_residual_relative_fro": _relerr(inverse_image, rhs, "fro"),
        "norm_a_fro_estimate": float(np.linalg.norm(true_apply, ord="fro") / probe_scale),
        "norm_factor_error_fro_estimate": float(
            np.linalg.norm(approximate_apply - true_apply, ord="fro") / probe_scale
        ),
        "inverse_residual_fro_estimate": float(
            np.linalg.norm(inverse_image - rhs, ord="fro")
            / probe_scale
        ),
    }
    metrics["norm_a_fro"] = metrics["norm_a_fro_estimate"]
    metrics["norm_apply_fro"] = metrics["norm_factor_error_fro_estimate"]
    metrics["err_solve_fro"] = metrics["inverse_residual_fro_estimate"]
    end_step("frobenius_probes", probe_started)

    application_started = begin_step("historical_application_errors")
    historical = dict(historical_options or {})
    app_samples = int(historical.pop("application_samples", 10))
    adjoint_samples = int(historical.pop("adjoint_samples", 8))
    historical_seed = int(historical.pop("seed", seed + 2000))
    if historical:
        raise ValueError(f"unknown historical-diagnostic options: {sorted(historical)}")
    sample_width = max(app_samples, 1)
    app_rng = np.random.default_rng(historical_seed)
    app_test = app_rng.standard_normal((dimension, sample_width)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        app_test += 1j * app_rng.standard_normal(app_test.shape)
    target_forward = _apply(operator, app_test)
    factor_forward = factors.matmat(app_test)
    target_transpose = _transpose_apply(operator, app_test)
    factor_transpose = factors.apply(app_test, transpose=True)
    metrics.update(
        {
            "app_err_left": _column_max_relative(factor_forward, target_forward),
            "app_err_right": _column_max_relative(factor_transpose, target_transpose),
            "app_inv_err_left": _column_max_relative(
                factors.inverse_matmat(target_forward), app_test
            ),
            "app_inv_err_right": _column_max_relative(
                _transpose_apply(operator, factors.apply(app_test, transpose=True, inverse=True)),
                app_test,
            ),
        }
    )
    end_step("historical_application_errors", application_started)

    adjoint_started = begin_step("adjoint_consistency")
    adjoint_width = max(adjoint_samples, 1)
    adjoint_rng = np.random.default_rng(historical_seed + 1)
    adjoint_test = adjoint_rng.standard_normal((dimension, 2 * adjoint_width)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        adjoint_test += 1j * adjoint_rng.standard_normal(adjoint_test.shape)
    factor_adjoint = lambda x: factors.apply(x, transpose=True, conjugate=True)
    inverse_adjoint = lambda x: factors.apply(
        x, transpose=True, conjugate=True, inverse=True
    )
    metrics["adjoint_consistency_error"] = _adjoint_consistency(
        factors.matmat, factor_adjoint, adjoint_test
    )
    metrics["adjoint_consistency_error_inv"] = _adjoint_consistency(
        factors.inverse_matmat, inverse_adjoint, adjoint_test
    )
    self_test = adjoint_test[:, :adjoint_width]
    self_forward = factors.matmat(self_test)
    metrics["self_adjoint_apply_error"] = _column_max_relative(
        factor_adjoint(self_test), self_forward
    )
    end_step("adjoint_consistency", adjoint_started)

    # Match the old sampled solve check: draw a known real-valued solution,
    # form b=A x, then compare F^{-1}b with x.
    solve_started = begin_step("known_solution_check")
    solve_rng = np.random.default_rng(historical_seed + 2)
    known_solution = solve_rng.standard_normal(dimension).astype(dtype)
    known_rhs = _apply(operator, known_solution)
    metrics["solve_error_rhs"] = _relerr(
        factors.inverse_matvec(known_rhs), known_solution
    )
    end_step("known_solution_check", solve_started)

    dense_builder = getattr(operator, "dense", None)
    estimator_report: dict[str, Any] = {"method": "disabled"}
    if callable(dense_builder) and dimension <= 2048:
        dense_started = begin_step("exact_dense_spectral_errors")
        dense = np.asarray(dense_builder())
        identity = np.eye(dimension, dtype=dtype)
        approximate_dense = factors.matmat(identity)
        approximate_inverse = factors.inverse_matmat(identity)
        exact_solution = np.linalg.solve(dense, rhs)
        norm_a = float(np.linalg.norm(dense, ord=2))
        norm_error = float(np.linalg.norm(approximate_dense - dense, ord=2))
        inverse_residual = float(
            np.linalg.norm(dense @ approximate_inverse - identity, ord=2)
        )
        norm_factor = float(np.linalg.norm(approximate_dense, ord=2))
        norm_inverse = float(np.linalg.norm(approximate_inverse, ord=2))
        metrics.update(
            {
                "condition_number_2": float(np.linalg.cond(dense)),
                "norm_a_2": norm_a,
                "norm_factor_error_2": norm_error,
                "norm_apply_2": norm_error,
                "factor_relative_2": norm_error / max(norm_a, np.finfo(float).tiny),
                "factor_relative_fro": _relerr(approximate_dense, dense, "fro"),
                "solve_relative_fro": _relerr(approximate_solution, exact_solution, "fro"),
                "inverse_residual_2": inverse_residual,
                "err_solve_2": inverse_residual,
                "norm_rsrs_2": norm_factor,
                "norm_rsrs_inverse_2": norm_inverse,
                "condition_rsrs_estimate_2": norm_factor * norm_inverse,
                "condition_target_via_rsrs_estimate_2": norm_a * norm_inverse,
            }
        )
        estimator_report = {"method": "exact_dense"}
        end_step("exact_dense_spectral_errors", dense_started)
    else:
        settings = dict(spectral_options or {})
        enabled = bool(settings.pop("enabled", True))
        iterations = int(settings.pop("iterations", 10))
        block_size = int(settings.pop("block_size", 2))
        tolerance = float(settings.pop("tolerance", 1.0e-3))
        spectral_seed = int(settings.pop("seed", seed + 1000))
        if settings:
            raise ValueError(f"unknown spectral-error options: {sorted(settings)}")
        estimator_report = {
            "method": "block_power_normal_operator" if enabled else "disabled",
            "configuration": {
                "iterations": iterations,
                "block_size": block_size,
                "tolerance": tolerance,
                "seed": spectral_seed,
            },
            "estimates": {},
        }
        if enabled:
            def estimate(name, forward, adjoint, offset):
                print(
                    f"[errors][spectral] estimate={name} starting "
                    f"iterations={iterations} block_size={min(block_size, MAX_OPERATOR_COLUMNS)}",
                    flush=True,
                )
                started = time.perf_counter()
                value, details = _block_spectral_norm(
                    forward,
                    adjoint,
                    dimension=dimension,
                    dtype=dtype,
                    seed=spectral_seed + offset,
                    iterations=iterations,
                    block_size=block_size,
                    tolerance=tolerance,
                    progress_label=name,
                )
                details["seconds"] = time.perf_counter() - started
                step_timings[f"spectral_{name}"] = details["seconds"]
                estimator_report["estimates"][name] = details
                print(
                    f"[errors][spectral] estimate={name} complete value={value:.6e} "
                    f"iterations={details['iterations']} converged={details['converged']} "
                    f"seconds={details['seconds']:.3f}",
                    flush=True,
                )
                return value

            factor_apply = lambda x: factors.matmat(x)
            factor_adjoint = lambda x: factors.apply(x, transpose=True, conjugate=True)
            inverse_apply = lambda x: factors.inverse_matmat(x)
            inverse_adjoint = lambda x: factors.apply(
                x, transpose=True, conjugate=True, inverse=True
            )
            operator_apply = lambda x: _apply(operator, x)
            operator_adjoint = lambda x: _adjoint_apply(operator, x)
            difference_apply = lambda x: factor_apply(x) - operator_apply(x)
            difference_adjoint = lambda x: factor_adjoint(x) - operator_adjoint(x)
            residual_apply = lambda x: operator_apply(inverse_apply(x)) - x
            residual_adjoint = lambda x: inverse_adjoint(operator_adjoint(x)) - x

            norm_a = estimate("target_operator", operator_apply, operator_adjoint, 1)
            norm_error = estimate("factor_error", difference_apply, difference_adjoint, 2)
            inverse_residual = estimate(
                "inverse_residual", residual_apply, residual_adjoint, 3
            )
            norm_factor = estimate("rsrs_operator", factor_apply, factor_adjoint, 4)
            norm_inverse = estimate("rsrs_inverse", inverse_apply, inverse_adjoint, 5)
            metrics.update(
                {
                    "norm_a_2": norm_a,
                    "norm_factor_error_2": norm_error,
                    "norm_apply_2": norm_error,
                    "factor_relative_2": norm_error / max(norm_a, np.finfo(float).tiny),
                    "inverse_residual_2": inverse_residual,
                    "err_solve_2": inverse_residual,
                    "norm_rsrs_2": norm_factor,
                    "norm_rsrs_inverse_2": norm_inverse,
                    "condition_rsrs_estimate_2": norm_factor * norm_inverse,
                    "condition_target_via_rsrs_estimate_2": norm_a * norm_inverse,
                }
            )
    if "condition_rsrs_estimate_2" in metrics:
        metrics["cond_rsrs_estimate"] = metrics["condition_rsrs_estimate_2"]
    estimator_report["step_timings_seconds"] = step_timings
    return metrics, estimator_report


def run_rsrs(
    cache: SampleCache | Path,
    options: Mapping[str, Any] | Any,
    *,
    output: Path,
    error_probes: int = 4,
    error_seed: int = 1701,
    points: np.ndarray | Path | None = None,
    gmres_options: Mapping[str, Any] | None = None,
    operator_spec: Mapping[str, Any] | None = None,
    sampling_operator_spec: Mapping[str, Any] | None = None,
    spectral_options: Mapping[str, Any] | None = None,
    historical_options: Mapping[str, Any] | None = None,
    execution_options: Mapping[str, Any] | None = None,
) -> RsrsRunResult:
    from bempp_rsrs import InsufficientSamplesError, RsrsOptions, RsrsSession
    from so_kernels import create_operator

    cache = cache if isinstance(cache, SampleCache) else SampleCache(Path(cache))
    manifest = cache.manifest
    sampling_operator = None
    cached_operator_spec = manifest.get("operator")
    if operator_spec is None and cached_operator_spec is None:
        raise ValueError(
            "legacy sample caches do not identify their physical operator; "
            "provide operator/operator_parameters in the experiment JSON"
        )
    target_operator_spec = dict(operator_spec or cached_operator_spec)
    reported_sampling_operator_spec = dict(
        sampling_operator_spec or cached_operator_spec or target_operator_spec
    )
    if points is None:
        cached_points = cache.path / "points.npy"
        if cached_points.is_file():
            points = np.load(cached_points)
        else:
            sampling_operator = create_operator(target_operator_spec)
            points = getattr(sampling_operator, "points", None)
    elif isinstance(points, Path):
        points = np.load(points)
    if points is None:
        raise ValueError(
            "RSRS requires point coordinates; expose operator.points, save points.npy "
            "with the cache, or pass points explicitly"
        )
    if not isinstance(options, RsrsOptions):
        options = RsrsOptions.from_mapping(dict(options))
    # Initialise RSRS before a backend operator whenever cached points are
    # available.  Bempp-rs and RSRS both use MPI internally, and constructing
    # the operator first can leave the second extension unable to initialise
    # its MPI datatype state in the same Python process.
    dtype = np.dtype(manifest["dtype"])
    session = RsrsSession(points, options, dtype=dtype)
    operator = sampling_operator or create_operator(target_operator_spec)
    expected_shape = (int(manifest["dimension"]), int(manifest["dimension"]))
    if tuple(operator.shape) != expected_shape:
        raise ValueError(
            f"target operator shape {operator.shape} does not match sampled "
            f"operator shape {expected_shape}"
        )
    target_points = getattr(operator, "points", None)
    if target_points is not None:
        target_points = np.asarray(target_points)
        if target_points.shape != np.asarray(points).shape or not np.allclose(
            target_points, points, rtol=1.0e-12, atol=1.0e-13
        ):
            raise ValueError(
                "target and sampling operators do not use the same ordered geometry"
            )
    print(
        f"[experiment] loading samples cache={cache.path} "
        f"count={manifest['sample_count']}",
        flush=True,
    )
    samples = cache.sample_bundle()
    execution = dict(execution_options or {})
    factorization_blas_threads = int(
        execution.pop("factorization_blas_threads", 1)
    )
    rsrs_apply_blas_threads = int(execution.pop("rsrs_apply_blas_threads", 1))
    if factorization_blas_threads < 1 or rsrs_apply_blas_threads < 1:
        raise ValueError("RSRS BLAS thread counts must be positive")
    if execution:
        raise ValueError(f"unknown execution options: {sorted(execution)}")
    print(
        "[experiment] starting RSRS factorization "
        f"rayon_threads={options.threads.factorization_threads} "
        f"blas_threads={factorization_blas_threads}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        # RSRS owns its Rayon pool through ``factorization_threads``. Limit
        # BLAS/OpenMP libraries only for this section, then restore their
        # process settings before target-operator errors and GMRES begin.
        with threadpool_limits(
            limits=factorization_blas_threads, user_api="blas"
        ):
            factors = session.factorize(samples)
    except InsufficientSamplesError as error:
        print(
            "[experiment] RSRS stopped: insufficient samples "
            f"request={vars(error.request)}",
            flush=True,
        )
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_samples",
            "cache": str(cache.path),
            "sample_request": vars(error.request),
            "factorization_seconds": time.perf_counter() - started,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "rsrs_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    factorization_seconds = time.perf_counter() - started
    diagnostic_factors = _SingleBlasRsrsFactors(
        factors, blas_threads=rsrs_apply_blas_threads
    )
    print(
        f"[experiment] RSRS factorization complete seconds={factorization_seconds:.3f}",
        flush=True,
    )
    gmres_options = dict(gmres_options or {})
    print(
        "[experiment] starting early approximate-inverse and GMRES checks "
        f"rsrs_apply_rayon_threads={options.threads.apply_threads} "
        f"rsrs_apply_blas_threads={rsrs_apply_blas_threads}",
        flush=True,
    )
    gmres_started = time.perf_counter()
    gmres_result = _gmres_report(
        operator,
        diagnostic_factors,
        mode=str(gmres_options.pop("mode", "none")),
        tolerance=float(gmres_options.pop("tolerance", 1.0e-7)),
        max_iterations=int(gmres_options.pop("max_iterations", 500)),
        restart=gmres_options.pop("restart", None),
        rhs_count=int(gmres_options.pop("rhs_count", 1)),
        seed=int(gmres_options.pop("seed", 1701)),
        log_every=int(gmres_options.pop("log_every", 1)),
    )
    if gmres_options:
        raise ValueError(f"unknown GMRES options: {sorted(gmres_options)}")
    gmres_seconds = time.perf_counter() - gmres_started
    print(
        f"[experiment] approximate-inverse and GMRES checks complete "
        f"seconds={gmres_seconds:.3f}",
        flush=True,
    )
    diagnostic_cost_estimate = _diagnostic_cost_estimate(
        operator,
        diagnostic_factors,
        seed=error_seed,
        probes=error_probes,
        spectral_options=spectral_options,
        historical_options=historical_options,
    )
    print(
        "[experiment] starting remaining error diagnostics "
        f"rsrs_apply_rayon_threads={options.threads.apply_threads} "
        f"rsrs_apply_blas_threads={rsrs_apply_blas_threads} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'inherited')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'inherited')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'inherited')}",
        flush=True,
    )
    error_started = time.perf_counter()
    errors, error_estimation = _errors(
        operator,
        diagnostic_factors,
        seed=error_seed,
        probes=error_probes,
        spectral_options=spectral_options,
        historical_options=historical_options,
    )
    errors["tot_num_samples"] = int(manifest["sample_count"])
    errors["residual_size"] = int(factors.report.get("residual_size", 0))
    error_seconds = time.perf_counter() - error_started
    for name, value in errors.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"[errors][result] {name}={value:.6e}", flush=True)
    print(
        f"[experiment] error diagnostics complete seconds={error_seconds:.3f}",
        flush=True,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "cache": str(cache.path),
        "operator": target_operator_spec,
        "sampling_operator": reported_sampling_operator_spec,
        "dimension": factors.dimension,
        "dtype": str(factors.dtype),
        "sample_count": manifest["sample_count"],
        "configuration": options.to_mapping(),
        "execution": {
            "factorization_blas_threads": factorization_blas_threads,
            "rsrs_apply_blas_threads": rsrs_apply_blas_threads,
        },
        "rsrs": factors.report,
        "timings_seconds": {
            "factorization": factorization_seconds,
            "gmres": gmres_seconds,
            "error_computation": error_seconds,
        },
        "errors": errors,
        "error_estimation": error_estimation,
        "diagnostic_cost_estimate": diagnostic_cost_estimate,
        "gmres": gmres_result,
    }
    _write_report_files(output, report)
    return RsrsRunResult(diagnostic_factors, report)


def run_case(case: Mapping[str, Any]) -> RsrsRunResult | dict[str, Any]:
    """Execute one complete Python-owned experiment definition."""
    from so_kernels import create_operator

    definition = dict(case)
    environment = {
        str(key): str(value) for key, value in dict(definition.get("environment", {})).items()
    }
    operator_spec = dict(definition["operator"])
    sampling_operator_spec = dict(definition.get("sampling_operator", operator_spec))
    output = Path(definition["output"])
    diagnostics = dict(definition.get("diagnostics", {}))
    gmres_options = dict(diagnostics.pop("gmres", {}))
    spectral_options = dict(diagnostics.pop("spectral", {}))
    historical_options = dict(diagnostics.pop("historical_errors", {}))
    gmres_mode = str(gmres_options.get("mode", "none"))
    if gmres_mode == "unpreconditioned":
        operator = create_operator(operator_spec)
        gmres_result = _gmres_report(
            operator,
            None,
            mode=gmres_mode,
            tolerance=float(gmres_options.get("tolerance", 1.0e-7)),
            max_iterations=int(gmres_options.get("max_iterations", 500)),
            restart=gmres_options.get("restart"),
            rhs_count=int(gmres_options.get("rhs_count", 1)),
            seed=int(gmres_options.get("seed", 1701)),
            log_every=int(gmres_options.get("log_every", 1)),
        )
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "operator": operator_spec,
            "gmres": gmres_result,
        }
        _write_report_files(output, report)
        return report

    sampling = dict(definition.get("sampling", {}))
    target = int(sampling.pop("target", 0))
    save_samples = bool(sampling.pop("save_samples", True))
    sampling_workers = int(sampling.pop("workers", 1))
    sampling_batch_size = int(sampling.pop("batch_size", 10))
    sampling_seed = int(sampling.pop("seed", 0))
    sampling_symmetry = str(sampling.pop("symmetry", "auto"))
    sampling_validation = str(sampling.pop("validation", "none"))
    sampling_validation_workers = sampling.pop("validation_workers", None)
    sampling_validation_batch_size = sampling.pop(
        "validation_batch_size", None
    )
    sampling_validation_tolerance = float(
        sampling.pop("validation_tolerance", 5.0e-4)
    )
    cache_value = definition.get("sample_cache")
    temporary_cache = None
    if target > 0 and not save_samples:
        temporary_cache = tempfile.TemporaryDirectory(prefix="rsrs-samples-")
        cache_path = Path(temporary_cache.name)
        print(
            "[experiment] samples are ephemeral and will be removed after this run",
            flush=True,
        )
    elif cache_value:
        cache_path = Path(cache_value)
    else:
        raise ValueError("sample_cache is required when no ephemeral samples are generated")
    try:
        if target > 0:
            sample_operator(
                ParallelSamplingConfig(
                    operator=sampling_operator_spec,
                    sample_count=target,
                    output=cache_path,
                    workers=sampling_workers,
                    batch_size=sampling_batch_size,
                    seed=sampling_seed,
                    symmetry=sampling_symmetry,
                    validation=sampling_validation,
                    validation_workers=sampling_validation_workers,
                    validation_batch_size=sampling_validation_batch_size,
                    validation_tolerance=sampling_validation_tolerance,
                    environment=environment,
                )
            )
        if sampling:
            raise ValueError(f"unknown sampling options: {sorted(sampling)}")
        return run_rsrs(
            cache_path,
            definition.get("rsrs", {}),
            output=output,
            error_probes=int(diagnostics.pop("error_probes", 4)),
            error_seed=int(diagnostics.pop("error_seed", 1701)),
            gmres_options=gmres_options,
            operator_spec=operator_spec,
            sampling_operator_spec=sampling_operator_spec,
            spectral_options=spectral_options,
            historical_options=historical_options,
            execution_options=definition.get("execution", {}),
        )
    finally:
        if temporary_cache is not None:
            temporary_cache.cleanup()
