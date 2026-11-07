"""Run sample-fed RSRS and extract Python-owned numerical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .cache import SampleCache


@dataclass(frozen=True)
class RsrsRunResult:
    factors: Any
    report: dict[str, Any]


def _apply(operator: Any, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1 and hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(values))
    if values.ndim == 2 and hasattr(operator, "matmat"):
        return np.asarray(operator.matmat(values))
    return np.asarray(operator @ values)


def _relerr(actual: np.ndarray, expected: np.ndarray, order: Any = 2) -> float:
    denominator = max(float(np.linalg.norm(expected, ord=order)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected, ord=order) / denominator)


def _errors(operator: Any, factors: Any, *, seed: int, probes: int) -> dict[str, float]:
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
    metrics = {
        "probe_apply_relative_fro": _relerr(approximate_apply, true_apply, "fro"),
        "inverse_residual_relative_fro": _relerr(_apply(operator, approximate_solution), rhs, "fro"),
    }

    dense_builder = getattr(operator, "dense", None)
    if callable(dense_builder) and dimension <= 2048:
        dense = np.asarray(dense_builder())
        approximate_dense = factors.matmat(np.eye(dimension, dtype=dtype))
        exact_solution = np.linalg.solve(dense, rhs)
        metrics.update(
            {
                "condition_number_2": float(np.linalg.cond(dense)),
                "factor_relative_2": _relerr(approximate_dense, dense, 2),
                "factor_relative_fro": _relerr(approximate_dense, dense, "fro"),
                "solve_relative_fro": _relerr(approximate_solution, exact_solution, "fro"),
            }
        )
    return metrics


def run_rsrs(
    cache: SampleCache | Path,
    options: Mapping[str, Any] | Any,
    *,
    output: Path,
    error_probes: int = 4,
    error_seed: int = 1701,
    points: np.ndarray | Path | None = None,
) -> RsrsRunResult:
    from bempp_rsrs import InsufficientSamplesError, RsrsOptions, RsrsSession
    from so_kernels import create_operator

    cache = cache if isinstance(cache, SampleCache) else SampleCache(Path(cache))
    manifest = cache.manifest
    operator = create_operator(manifest["operator"])
    if points is None:
        cached_points = cache.path / "points.npy"
        points = np.load(cached_points) if cached_points.is_file() else getattr(operator, "points", None)
    elif isinstance(points, Path):
        points = np.load(points)
    if points is None:
        raise ValueError(
            "RSRS requires point coordinates; expose operator.points, save points.npy "
            "with the cache, or pass points explicitly"
        )
    if not isinstance(options, RsrsOptions):
        options = RsrsOptions.from_mapping(dict(options))
    session = RsrsSession(points, options, dtype=operator.dtype)
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
    errors = _errors(operator, factors, seed=error_seed, probes=error_probes)
    error_seconds = time.perf_counter() - error_started
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "cache": str(cache.path),
        "operator": manifest["operator"],
        "dimension": factors.dimension,
        "dtype": str(factors.dtype),
        "sample_count": manifest["sample_count"],
        "configuration": options.to_mapping(),
        "timings_seconds": {
            "factorization": factorization_seconds,
            "error_computation": error_seconds,
        },
        "errors": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "rsrs_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return RsrsRunResult(factors, report)
