"""Run sample-fed RSRS and extract Python-owned numerical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .cache import SampleCache
from .sampling import ParallelSamplingConfig, sample_operator


@dataclass(frozen=True)
class RsrsRunResult:
    factors: Any
    report: dict[str, Any]


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
        values = np.asarray(provider(count=count))
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
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": mode, "solves": {}}
    if mode not in {"unpreconditioned", "preconditioned", "both"}:
        raise ValueError(f"unknown GMRES mode {mode!r}")
    if mode in {"preconditioned", "both"} and factors is None:
        raise ValueError("preconditioned GMRES requires RSRS factors")
    matrix = _linear_operator(operator)
    rhs = _right_hand_sides(operator, count=rhs_count, seed=seed)
    variants = []
    if mode in {"unpreconditioned", "both"}:
        variants.append(("unpreconditioned", None))
    if mode in {"preconditioned", "both"}:
        variants.append(("preconditioned", factors.aslinearoperator(inverse=True)))
    report: dict[str, Any] = {"mode": mode, "tolerance": tolerance, "solves": {}}
    for name, preconditioner in variants:
        solves = []
        for column in range(rhs_count):
            history: list[float] = []
            started = time.perf_counter()
            solution, info = gmres(
                matrix,
                rhs[:, column],
                M=preconditioner,
                rtol=tolerance,
                atol=0.0,
                restart=restart,
                maxiter=max_iterations,
                callback=lambda value: history.append(float(value)),
                # ``legacy`` reports the preconditioned residual norm while
                # interpreting maxiter as inner iterations, matching the
                # historical RSRS experiment driver and its 500-iteration cap.
                callback_type="legacy",
            )
            denominator = max(float(np.linalg.norm(rhs[:, column])), np.finfo(float).tiny)
            true_residual = float(
                np.linalg.norm(_apply(operator, solution) - rhs[:, column]) / denominator
            )
            solves.append(
                {
                    "rhs": column,
                    "info": int(info),
                    "iterations": len(history),
                    "recorded_residuals": history,
                    "final_true_relative_residual": true_residual,
                    "seconds": time.perf_counter() - started,
                }
            )
        report["solves"][name] = solves
    return report


def _apply(operator: Any, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1 and hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(values))
    if values.ndim == 2 and hasattr(operator, "matmat"):
        return np.asarray(operator.matmat(values))
    return np.asarray(operator @ values)


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
        return np.conjugate(np.asarray(method(np.conjugate(values))))
    linear = operator.aslinearoperator() if hasattr(operator, "aslinearoperator") else None
    if linear is not None:
        return np.asarray(linear.rmatvec(values) if values.ndim == 1 else linear.rmatmat(values))
    raise TypeError("spectral errors require an explicit transpose/adjoint application")


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
) -> tuple[float, dict[str, Any]]:
    """Estimate a largest singular value through block power iteration."""

    width = min(max(int(block_size), 1), dimension)
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


def _errors(
    operator: Any,
    factors: Any,
    *,
    seed: int,
    probes: int,
    spectral_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    dimension = int(operator.shape[0])
    dtype = np.dtype(getattr(operator, "dtype", factors.dtype))
    rng = np.random.default_rng(seed)
    test = rng.standard_normal((dimension, probes)).astype(dtype)
    rhs = rng.standard_normal((dimension, probes)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        test += 1j * rng.standard_normal(test.shape)
        rhs += 1j * rng.standard_normal(rhs.shape)

    true_apply = _apply(operator, test)
    approximate_apply = factors.matmat(test)
    approximate_solution = factors.inverse_matmat(rhs)
    probe_scale = np.sqrt(max(probes, 1))
    metrics = {
        "probe_apply_relative_fro": _relerr(approximate_apply, true_apply, "fro"),
        "inverse_residual_relative_fro": _relerr(_apply(operator, approximate_solution), rhs, "fro"),
        "norm_a_fro_estimate": float(np.linalg.norm(true_apply, ord="fro") / probe_scale),
        "norm_factor_error_fro_estimate": float(
            np.linalg.norm(approximate_apply - true_apply, ord="fro") / probe_scale
        ),
        "inverse_residual_fro_estimate": float(
            np.linalg.norm(_apply(operator, approximate_solution) - rhs, ord="fro")
            / probe_scale
        ),
    }

    dense_builder = getattr(operator, "dense", None)
    estimator_report: dict[str, Any] = {"method": "disabled"}
    if callable(dense_builder) and dimension <= 2048:
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
                )
                details["seconds"] = time.perf_counter() - started
                estimator_report["estimates"][name] = details
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
    spectral_options: Mapping[str, Any] | None = None,
) -> RsrsRunResult:
    from bempp_rsrs import InsufficientSamplesError, RsrsOptions, RsrsSession
    from so_kernels import create_operator

    cache = cache if isinstance(cache, SampleCache) else SampleCache(Path(cache))
    manifest = cache.manifest
    sampling_operator = None
    target_operator_spec = dict(operator_spec or manifest["operator"])
    if points is None:
        cached_points = cache.path / "points.npy"
        if cached_points.is_file():
            points = np.load(cached_points)
        else:
            sampling_operator = create_operator(manifest["operator"])
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
    operator = create_operator(target_operator_spec)
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
    samples = cache.sample_bundle()
    started = time.perf_counter()
    try:
        factors = session.factorize(samples)
    except InsufficientSamplesError as error:
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
    error_started = time.perf_counter()
    errors, error_estimation = _errors(
        operator,
        factors,
        seed=error_seed,
        probes=error_probes,
        spectral_options=spectral_options,
    )
    error_seconds = time.perf_counter() - error_started
    gmres_options = dict(gmres_options or {})
    gmres_result = _gmres_report(
        operator,
        factors,
        mode=str(gmres_options.pop("mode", "none")),
        tolerance=float(gmres_options.pop("tolerance", 1.0e-7)),
        max_iterations=int(gmres_options.pop("max_iterations", 500)),
        restart=gmres_options.pop("restart", None),
        rhs_count=int(gmres_options.pop("rhs_count", 1)),
        seed=int(gmres_options.pop("seed", 1701)),
    )
    if gmres_options:
        raise ValueError(f"unknown GMRES options: {sorted(gmres_options)}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "cache": str(cache.path),
        "operator": target_operator_spec,
        "sampling_operator": manifest["operator"],
        "dimension": factors.dimension,
        "dtype": str(factors.dtype),
        "sample_count": manifest["sample_count"],
        "configuration": options.to_mapping(),
        "rsrs": factors.report,
        "timings_seconds": {
            "factorization": factorization_seconds,
            "error_computation": error_seconds,
        },
        "errors": errors,
        "error_estimation": error_estimation,
        "gmres": gmres_result,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "rsrs_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return RsrsRunResult(factors, report)


def run_case(case: Mapping[str, Any]) -> RsrsRunResult | dict[str, Any]:
    """Execute one complete Python-owned experiment definition."""
    from so_kernels import create_operator

    definition = dict(case)
    operator_spec = dict(definition["operator"])
    sampling_operator_spec = dict(definition.get("sampling_operator", operator_spec))
    output = Path(definition["output"])
    diagnostics = dict(definition.get("diagnostics", {}))
    gmres_options = dict(diagnostics.pop("gmres", {}))
    spectral_options = dict(diagnostics.pop("spectral", {}))
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
        )
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "operator": operator_spec,
            "gmres": gmres_result,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "rsrs_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    sampling = dict(definition.get("sampling", {}))
    target = int(sampling.pop("target", 0))
    save_samples = bool(sampling.pop("save_samples", True))
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
                    workers=int(sampling.pop("workers", 1)),
                    batch_size=int(sampling.pop("batch_size", 10)),
                    seed=int(sampling.pop("seed", 0)),
                    symmetry=str(sampling.pop("symmetry", "auto")),
                    validation=str(sampling.pop("validation", "none")),
                    validation_workers=sampling.pop("validation_workers", None),
                    validation_batch_size=sampling.pop("validation_batch_size", None),
                    validation_tolerance=float(
                        sampling.pop("validation_tolerance", 5.0e-4)
                    ),
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
            spectral_options=spectral_options,
        )
    finally:
        if temporary_cache is not None:
            temporary_cache.cleanup()
