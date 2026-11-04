from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .kernels import build_kernel
from .random import test_block
from .spec import CaseSpec
from .storage import atomic_json, sha256, utc_now


def log(message: str) -> None:
    print(f"[rsrs-samples] {message}", flush=True)


def generate_chunk(task: dict) -> dict:
    case = CaseSpec(**task["case"])
    root = Path(task["workspace"]) / "chunks" / f"chunk_{task['worker']:03d}"
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    os.environ["TMPDIR"] = str(root)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = str(task["threads_per_worker"])
    start, stop = task["row_start"], task["row_stop"]
    log(f"worker={task['worker']} rows={start}:{stop} building {case.kernel}")
    built = build_kernel(case)
    warm = test_block(task["seed"], "y", [start], built.dimension)
    built.sketch("y", warm)
    families = ["y_test", "y_sketch"] + ([] if built.symmetric else ["z_test", "z_sketch"])
    shapes = {name: (stop - start, built.dimension) for name in families}
    arrays = {
        name: np.lib.format.open_memmap(
            root / f"{name}.npy", mode="w+", dtype=(np.float64 if name.endswith("test") else built.dtype), shape=shape
        )
        for name, shape in shapes.items()
    }
    timings = {"y": 0.0, "z": 0.0}
    for stream in ("y",) if built.symmetric else ("y", "z"):
        for lo in range(start, stop, task["batch_size"]):
            hi = min(lo + task["batch_size"], stop)
            tests = test_block(task["seed"], stream, list(range(lo, hi)), built.dimension)
            tic = time.perf_counter()
            sketches = built.sketch(stream, tests)
            elapsed = time.perf_counter() - tic
            arrays[f"{stream}_test"][lo - start : hi - start] = tests.T
            arrays[f"{stream}_sketch"][lo - start : hi - start] = sketches.T
            timings[stream] += elapsed
            completed = hi - start
            eta = timings[stream] / completed * (stop - hi)
            log(f"worker={task['worker']} {stream.upper()} rows={lo}:{hi} batch={elapsed:.1f}s done={completed}/{stop-start} eta={eta/60:.1f}m")
    for array in arrays.values():
        array.flush()
    meta = {
        "worker": task["worker"], "row_start": start, "row_stop": stop,
        "dimension": built.dimension, "symmetric": built.symmetric, "families": families,
        "timings": timings, "completed_utc": utc_now(),
    }
    meta["sha256"] = {name: sha256(root / f"{name}.npy") for name in families}
    atomic_json(root / "chunk.json", meta)
    return meta


def generate_repair(task: dict) -> dict:
    case = CaseSpec(**task["case"])
    root = Path(task["output"])
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    os.environ["TMPDIR"] = str(root)
    built = build_kernel(case)
    rows = task["rows"]
    families = ["y_test", "y_sketch"] + ([] if built.symmetric else ["z_test", "z_sketch"])
    data = {name: [] for name in families}
    for offset in range(0, len(rows), task["batch_size"]):
        selected = rows[offset : offset + task["batch_size"]]
        for stream in ("y",) if built.symmetric else ("y", "z"):
            tests = test_block(task["seed"], stream, selected, built.dimension)
            data[f"{stream}_test"].append(tests.T)
            data[f"{stream}_sketch"].append(built.sketch(stream, tests).T)
        log(f"repair rows complete={min(offset+task['batch_size'], len(rows))}/{len(rows)}")
    np.save(root / "rows.npy", np.asarray(rows, dtype=np.int64))
    for name, pieces in data.items():
        np.save(root / f"{name}.npy", np.concatenate(pieces, axis=0))
    meta = {"rows": rows, "families": families, "dimension": built.dimension, "completed_utc": utc_now()}
    atomic_json(root / "repair.json", meta)
    return meta
