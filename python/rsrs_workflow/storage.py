from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .spec import CaseSpec, SamplingSpec

FAMILIES = ("y_test", "y_sketch", "z_test", "z_sketch")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def new_manifest(case: CaseSpec, sampling: SamplingSpec) -> dict:
    mesh_fingerprint = None
    if case.mesh is not None:
        mesh_path = Path(case.mesh)
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh_fingerprint = {"path": str(mesh_path), "sha256": sha256(mesh_path), "bytes": mesh_path.stat().st_size}
    return {
        "format": "rsrs-python-samples-v1",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "generating",
        "case": asdict(case),
        "mesh_fingerprint": mesh_fingerprint,
        "sampling": asdict(sampling),
        "dimension": None,
        "symmetric": None,
        "chunks": [],
        "repairs": [],
        "validation": None,
        "assembled_cache": None,
    }


def chunk_directories(workspace: Path) -> list[Path]:
    return sorted((workspace / "chunks").glob("chunk_*"))


def read_chunk(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = json.loads((path / "chunk.json").read_text())
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in meta["families"]}
    expected = meta["row_stop"] - meta["row_start"]
    for name, array in arrays.items():
        if array.ndim != 2 or array.shape[0] != expected:
            raise RuntimeError(f"invalid {name} shape in {path}: {array.shape}")
    return meta, arrays


def verify_contiguous(workspace: Path, expected_rows: int) -> list[tuple[dict, Path]]:
    result, position = [], 0
    for path in chunk_directories(workspace):
        meta, _ = read_chunk(path)
        if meta["row_start"] != position:
            raise RuntimeError(f"gap or overlap before {path}: expected row {position}")
        position = meta["row_stop"]
        result.append((meta, path))
    if position != expected_rows:
        raise RuntimeError(f"chunks cover {position} rows, expected {expected_rows}")
    return result


def parse_rows(expression: str, limit: int) -> list[int]:
    rows: set[int] = set()
    for token in expression.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            fields = [int(value) if value else None for value in token.split(":")]
            if len(fields) not in (2, 3):
                raise ValueError(f"invalid row range {token!r}")
            rows.update(range(*slice(*fields).indices(limit)))
        else:
            rows.add(int(token))
    if not rows or min(rows) < 0 or max(rows) >= limit:
        raise ValueError(f"rows must be within [0, {limit})")
    return sorted(rows)
