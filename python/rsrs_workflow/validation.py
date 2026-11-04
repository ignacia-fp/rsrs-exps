from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .kernels import build_kernel
from .spec import CaseSpec
from .storage import atomic_json, read_chunk, sha256, utc_now, verify_contiguous


def relative_errors(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(actual - expected, axis=0)
    denominator = np.maximum(np.linalg.norm(expected, axis=0), np.finfo(float).tiny)
    return numerator / denominator


def validate_workspace(workspace: Path, mode: str = "full", tolerance: float = 5e-7) -> dict:
    manifest = json.loads((workspace / "manifest.json").read_text())
    case = CaseSpec(**manifest["case"])
    fingerprint = manifest.get("mesh_fingerprint")
    if fingerprint and sha256(Path(fingerprint["path"])) != fingerprint["sha256"]:
        raise RuntimeError("mesh fingerprint differs from the grid used for generation")
    total = int(manifest["sampling"]["samples"])
    chunks = verify_contiguous(workspace, total)
    isolated = workspace / "validation_operator"
    isolated.mkdir(exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(isolated)
    os.environ["TMPDIR"] = str(isolated)
    try:
        built = build_kernel(case)
        repair_index = {}
        for directory in sorted((workspace / "repairs").glob("repair_*")) if (workspace / "repairs").exists() else []:
            repair_rows = np.load(directory / "rows.npy")
            for local, row in enumerate(repair_rows.tolist()):
                repair_index[int(row)] = (directory, local)
        results, bad_rows = [], []
        batch_size = int(manifest["sampling"].get("batch_size", 10))
        for meta, path in chunks:
            _, arrays = read_chunk(path)
            local_rows = list(range(meta["row_stop"] - meta["row_start"]))
            if mode == "spot" and len(local_rows) > 3:
                local_rows = sorted({0, len(local_rows) // 2, len(local_rows) - 1})
            for stream in ("y",) if built.symmetric else ("y", "z"):
                for offset in range(0, len(local_rows), batch_size):
                    selected = local_rows[offset : offset + batch_size]
                    tests = np.array(arrays[f"{stream}_test"][selected], copy=True)
                    expected = np.array(arrays[f"{stream}_sketch"][selected], copy=True)
                    for position, local in enumerate(selected):
                        row = meta["row_start"] + local
                        replacement = repair_index.get(row)
                        if replacement:
                            directory, repair_local = replacement
                            tests[position] = np.load(directory / f"{stream}_test.npy", mmap_mode="r")[repair_local]
                            expected[position] = np.load(directory / f"{stream}_sketch.npy", mmap_mode="r")[repair_local]
                    errors = relative_errors(built.sketch(stream, tests.T), expected.T)
                    for local, error in zip(selected, errors):
                        row = meta["row_start"] + local
                        results.append({"row": row, "stream": stream, "relative_error": float(error)})
                        if not np.isfinite(error) or error > tolerance:
                            bad_rows.append(row)
            print(f"[rsrs-validate] checked through row {meta['row_stop']} bad={len(set(bad_rows))}", flush=True)
    finally:
        os.chdir(old_cwd)
    report = {
        "status": "passed" if not bad_rows else "failed", "mode": mode, "tolerance": tolerance,
        "checked": len(results), "bad_rows": sorted(set(bad_rows)),
        "max_error": max((item["relative_error"] for item in results), default=0.0),
        "completed_utc": utc_now(), "results": results,
    }
    atomic_json(workspace / "validation_report.json", report)
    manifest["validation"] = {key: report[key] for key in ("status", "mode", "tolerance", "checked", "bad_rows", "max_error", "completed_utc")}
    manifest["status"] = "validated" if not bad_rows else "validation_failed"
    manifest["updated_utc"] = utc_now()
    atomic_json(workspace / "manifest.json", manifest)
    return report
