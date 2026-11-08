"""Process-isolated, deterministic, batched sampling and validation."""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Literal, Mapping

import numpy as np

from .cache import SampleCache, assemble_cache


ValidationMode = Literal["none", "sampled", "all"]
MAX_OPERATOR_COLUMNS = 30


@dataclass(frozen=True)
class ParallelSamplingConfig:
    operator: Mapping[str, Any]
    sample_count: int
    output: Path
    workers: int = 1
    batch_size: int = 10
    seed: int = 0
    symmetry: Literal["auto", "symmetric", "nonsymmetric"] = "auto"
    warmup: bool = True
    resume: bool = True
    validation: ValidationMode = "none"
    validation_workers: int | None = None
    validation_batch_size: int | None = None
    validation_tolerance: float = 5.0e-4
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_count < 1 or self.workers < 1 or self.batch_size < 1:
            raise ValueError("sample_count, workers, and batch_size must be positive")
        if self.validation_workers is not None and self.validation_workers < 1:
            raise ValueError("validation_workers must be positive")
        if self.validation_batch_size is not None and self.validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive")


@dataclass(frozen=True)
class ParallelSamplingResult:
    cache: SampleCache
    report: dict[str, Any]


_OPERATOR: Any | None = None
_WORKER_DIRECTORY: Path | None = None
_WARMED_Y = False
_WARMED_Z = False


def _apply(operator: Any, vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 2 and vectors.shape[1] > MAX_OPERATOR_COLUMNS:
        result = np.concatenate(
            [
                _apply(operator, vectors[:, start : start + MAX_OPERATOR_COLUMNS])
                for start in range(0, vectors.shape[1], MAX_OPERATOR_COLUMNS)
            ],
            axis=1,
        )
    elif hasattr(operator, "matmat"):
        result = np.asarray(operator.matmat(vectors))
    else:
        result = np.asarray(operator @ vectors)
    if result.size != vectors.size:
        raise RuntimeError(f"operator returned {result.size} entries, expected {vectors.size}")
    return result.reshape(vectors.shape)


def _transpose(operator: Any) -> Any:
    candidate = getattr(operator, "transpose_operator", None)
    if candidate is not None:
        return candidate
    if hasattr(operator, "transpose_matmat"):
        class View:
            shape = operator.shape
            dtype = operator.dtype

            @staticmethod
            def matmat(values):
                return operator.transpose_matmat(values)

        return View()
    raise TypeError("nonsymmetric sampling requires an explicit algebraic transpose")


def _worker_init(operator_spec: Mapping[str, Any], root: str, environment: Mapping[str, str]):
    global _OPERATOR, _WORKER_DIRECTORY, _WARMED_Y, _WARMED_Z
    from so_kernels import create_operator

    for key, value in environment.items():
        os.environ[str(key)] = str(value)
    workers_root = Path(root)
    workers_root.mkdir(parents=True, exist_ok=True)
    # Do not rely only on a PID: schedulers and repeated runs may reuse process
    # identifiers while stale ExaFMM files still exist.  A fresh directory for
    # every spawned process isolates all fixed-name runtime files as well as
    # ordinary temporary and cache files.
    worker_root = Path(
        tempfile.mkdtemp(prefix=f"worker_{os.getpid()}_", dir=workers_root)
    )
    os.chdir(worker_root)
    for variable in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        os.environ[variable] = str(worker_root)
    tempfile.tempdir = str(worker_root)
    _WORKER_DIRECTORY = worker_root
    _OPERATOR = create_operator(operator_spec)
    _WARMED_Y = False
    _WARMED_Z = False


def _gaussian_rows(
    dimension: int, row_start: int, row_stop: int, seed: int, family: int, dtype: np.dtype
) -> np.ndarray:
    rows = []
    complex_values = np.issubdtype(dtype, np.complexfloating)
    for row in range(row_start, row_stop):
        rng = np.random.default_rng(np.random.SeedSequence([seed, family, row]))
        values = rng.standard_normal(dimension)
        if complex_values:
            values = values + 1j * rng.standard_normal(dimension)
        rows.append(np.asarray(values, dtype=dtype))
    return np.ascontiguousarray(np.stack(rows, axis=1))


def _has_z(operator: Any, symmetry: str) -> bool:
    if symmetry == "symmetric":
        return False
    if symmetry == "nonsymmetric":
        return True
    return not bool(getattr(operator, "symmetric", False))


def _sample_batch(task: Mapping[str, Any]) -> dict[str, Any]:
    global _WARMED_Y, _WARMED_Z
    assert _OPERATOR is not None
    operator = _OPERATOR
    row_start, row_stop = int(task["row_start"]), int(task["row_stop"])
    dimension = int(operator.shape[0])
    if operator.shape != (dimension, dimension):
        raise ValueError(f"RSRS requires a square operator, got {operator.shape}")
    dtype = np.dtype(getattr(operator, "dtype", np.float64))
    has_z = _has_z(operator, str(task["symmetry"]))
    if row_start == 0 and hasattr(operator, "points"):
        points = np.asarray(operator.points, dtype=np.float64)
        if points.shape != (dimension, 3):
            raise ValueError(
                f"operator points must have shape ({dimension}, 3), got {points.shape}"
            )
        points_path = Path(task["cache"]) / "points.npy"
        temporary_points = points_path.with_suffix(".tmp.npy")
        np.save(temporary_points, points)
        temporary_points.replace(points_path)
    started = time.perf_counter()
    omega_y = _gaussian_rows(dimension, row_start, row_stop, task["seed"], 0, dtype)
    if task["warmup"] and not _WARMED_Y:
        _apply(operator, omega_y[:, :1])
        _WARMED_Y = True
    y_started = time.perf_counter()
    y = _apply(operator, omega_y)
    y_seconds = time.perf_counter() - y_started
    values = {"y_test": omega_y.T, "y_sketch": y.T}
    z_seconds = 0.0
    if has_z:
        omega_z = _gaussian_rows(dimension, row_start, row_stop, task["seed"], 1, dtype)
        transpose = _transpose(operator)
        if task["warmup"] and not _WARMED_Z:
            _apply(transpose, omega_z[:, :1])
            _WARMED_Z = True
        z_started = time.perf_counter()
        z = _apply(transpose, omega_z)
        z_seconds = time.perf_counter() - z_started
        values.update({"z_test": omega_z.T, "z_sketch": z.T})

    chunk = Path(task["chunk"])
    temporary = chunk.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        row_start=row_start,
        row_stop=row_stop,
        fingerprint=np.asarray(task["fingerprint"]),
        **values,
    )
    temporary.replace(chunk)
    result = {
        "row_start": row_start,
        "row_stop": row_stop,
        "count": row_stop - row_start,
        "dimension": dimension,
        "dtype": dtype.str,
        "has_z": has_z,
        "worker_pid": os.getpid(),
        "worker_directory": str(_WORKER_DIRECTORY),
        "chunk": str(chunk),
        "y_apply_seconds": y_seconds,
        "z_apply_seconds": z_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[sampling-worker] pid={os.getpid()} rows={row_start}:{row_stop} "
        f"elapsed={result['elapsed_seconds']:.3f}s",
        flush=True,
    )
    return result


def _validate_batch(task: Mapping[str, Any]) -> dict[str, Any]:
    assert _OPERATOR is not None
    operator = _OPERATOR
    row_start, row_stop = int(task["row_start"]), int(task["row_stop"])
    arrays = Path(task["cache"]) / "arrays"
    y_test = np.asarray(np.load(arrays / "y_test.npy", mmap_mode="r")[row_start:row_stop])
    y_sketch = np.asarray(np.load(arrays / "y_sketch.npy", mmap_mode="r")[row_start:row_stop])
    actual_y = _apply(operator, y_test.T).T
    y_row_errors = np.linalg.norm(actual_y - y_sketch, axis=1) / np.maximum(
        np.linalg.norm(y_sketch, axis=1), np.finfo(float).tiny
    )
    errors = {"y": float(y_row_errors.max())}
    z_test_path = arrays / "z_test.npy"
    if z_test_path.exists():
        z_test = np.asarray(np.load(z_test_path, mmap_mode="r")[row_start:row_stop])
        z_sketch = np.asarray(
            np.load(arrays / "z_sketch.npy", mmap_mode="r")[row_start:row_stop]
        )
        actual_z = _apply(_transpose(operator), z_test.T).T
        z_row_errors = np.linalg.norm(actual_z - z_sketch, axis=1) / np.maximum(
            np.linalg.norm(z_sketch, axis=1), np.finfo(float).tiny
        )
        errors["z"] = float(z_row_errors.max())
    result = {
        "row_start": row_start,
        "row_stop": row_stop,
        "worker_pid": os.getpid(),
        "errors": errors,
        "max_error": max(errors.values()),
    }
    print(
        f"[validation-worker] pid={os.getpid()} rows={result['row_start']}:"
        f"{result['row_stop']} max_error={result['max_error']:.3e}",
        flush=True,
    )
    return result


def _ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def _existing_result(
    chunk: Path, row_start: int, row_stop: int, fingerprint: str
) -> dict | None:
    if not chunk.exists():
        return None
    try:
        with np.load(chunk) as values:
            if int(values["row_start"]) != row_start or int(values["row_stop"]) != row_stop:
                return None
            if str(values["fingerprint"].item()) != fingerprint:
                return None
            y = values["y_test"]
            return {
                "row_start": row_start,
                "row_stop": row_stop,
                "count": row_stop - row_start,
                "dimension": int(y.shape[1]),
                "dtype": y.dtype.str,
                "has_z": "z_test" in values,
                "worker_pid": None,
                "worker_directory": None,
                "chunk": str(chunk),
                "y_apply_seconds": 0.0,
                "z_apply_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "resumed": True,
            }
    except (OSError, ValueError, KeyError):
        return None


def _selected_validation(
    total: int, batch_size: int, mode: ValidationMode
) -> list[tuple[int, int]]:
    ranges = _ranges(total, batch_size)
    if mode == "all" or len(ranges) <= 3:
        return ranges
    indices = sorted({0, len(ranges) // 2, len(ranges) - 1})
    return [ranges[index] for index in indices]


def sample_operator(config: ParallelSamplingConfig) -> ParallelSamplingResult:
    output = Path(config.output).resolve()
    chunks = output / "chunks"
    workers_root = output / "workers"
    chunks.mkdir(parents=True, exist_ok=True)
    workers_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    completed = []
    # A target increase must not invalidate already generated rows.  Sample
    # identity depends on the operator, RNG seed and algebraic convention, not
    # on how many rows the user ultimately requests.
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "operator": dict(config.operator),
                "seed": config.seed,
                "symmetry": config.symmetry,
            },
            sort_keys=True,
        ).encode("utf8")
    ).hexdigest()
    effective_sample_count = config.sample_count
    manifest_path = output / "manifest.json"
    if config.resume and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            if previous.get("fingerprint") == fingerprint:
                effective_sample_count = max(
                    effective_sample_count, int(previous.get("sample_count", 0))
                )
        except (OSError, ValueError, TypeError):
            pass
    for row_start, row_stop in _ranges(effective_sample_count, config.batch_size):
        chunk = chunks / f"batch_{row_start:09d}_{row_stop:09d}.npz"
        existing = (
            _existing_result(chunk, row_start, row_stop, fingerprint)
            if config.resume
            else None
        )
        if existing is not None:
            completed.append(existing)
        else:
            tasks.append(
                {
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "seed": config.seed,
                    "symmetry": config.symmetry,
                    "warmup": config.warmup,
                    "fingerprint": fingerprint,
                    "chunk": str(chunk),
                    "cache": str(output),
                }
            )

    started = time.perf_counter()
    total_batches = len(tasks) + len(completed)
    print(
        f"[sampling] target_rows={effective_sample_count} batch_size={config.batch_size} "
        f"workers={min(config.workers, max(len(tasks), 1))} "
        f"new_batches={len(tasks)} resumed_batches={len(completed)}",
        flush=True,
    )
    if tasks:
        context = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(
            max_workers=min(config.workers, len(tasks)),
            mp_context=context,
            initializer=_worker_init,
            initargs=(dict(config.operator), str(workers_root / "sampling"), dict(config.environment)),
        ) as executor:
            futures = {executor.submit(_sample_batch, task): task for task in tasks}
            for future in cf.as_completed(futures):
                completed.append(future.result())
                elapsed = time.perf_counter() - started
                generated = sum(
                    item["count"] for item in completed if not item.get("resumed", False)
                )
                remaining = sum(
                    task["row_stop"] - task["row_start"]
                    for pending, task in futures.items()
                    if not pending.done()
                )
                rate = generated / elapsed if elapsed > 0.0 else 0.0
                eta = remaining / rate if rate > 0.0 else 0.0
                print(
                    f"[sampling] completed_batches={len(completed)}/{total_batches} "
                    f"generated_rows={generated} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    sampling_seconds = time.perf_counter() - started
    completed.sort(key=lambda item: item["row_start"])
    print("[sampling] assembling row-ordered sample arrays", flush=True)
    cache = assemble_cache(output, completed)
    print(
        f"[sampling] assembled rows={effective_sample_count} elapsed={sampling_seconds:.1f}s",
        flush=True,
    )

    validation_results: list[dict] = []
    validation_seconds = 0.0
    if config.validation != "none":
        validation_batch_size = config.validation_batch_size or config.batch_size
        selected = _selected_validation(
            effective_sample_count, validation_batch_size, config.validation
        )
        validation_workers = config.validation_workers or config.workers
        validation_tasks = [
            {
                "row_start": row_start,
                "row_stop": row_stop,
                "cache": str(output),
            }
            for row_start, row_stop in selected
        ]
        validation_started = time.perf_counter()
        print(
            f"[validation] mode={config.validation} batches={len(validation_tasks)} "
            f"workers={min(validation_workers, len(validation_tasks))}",
            flush=True,
        )
        context = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(
            max_workers=min(validation_workers, len(validation_tasks)),
            mp_context=context,
            initializer=_worker_init,
            initargs=(
                dict(config.operator),
                str(workers_root / "validation"),
                dict(config.environment),
            ),
        ) as executor:
            validation_results = list(executor.map(_validate_batch, validation_tasks))
        validation_seconds = time.perf_counter() - validation_started
        worst = max(item["max_error"] for item in validation_results)
        print(
            f"[validation] complete worst_error={worst:.3e} "
            f"elapsed={validation_seconds:.1f}s",
            flush=True,
        )
        if worst > config.validation_tolerance:
            raise RuntimeError(
                f"sample validation failed: {worst:.3e} > {config.validation_tolerance:.3e}"
            )

    first = completed[0]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "operator": dict(config.operator),
        "configuration": {
            **asdict(config),
            "output": str(output),
        },
        "requested_sample_count": config.sample_count,
        "effective_sample_count": effective_sample_count,
        "dimension": first["dimension"],
        "dtype": first["dtype"],
        "has_z": first["has_z"],
        "batch_results": completed,
        "sampling_seconds": sampling_seconds,
        "validation_seconds": validation_seconds,
        "validation_results": validation_results,
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "sample_count": effective_sample_count,
                "dimension": first["dimension"],
                "dtype": first["dtype"],
                "has_z": first["has_z"],
                "operator": dict(config.operator),
                "seed": config.seed,
                "batch_size": config.batch_size,
                "fingerprint": fingerprint,
                "points_file": "points.npy" if (output / "points.npy").is_file() else None,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "sampling_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return ParallelSamplingResult(cache, report)
