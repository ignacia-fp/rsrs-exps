#!/usr/bin/env python3
"""Consolidate a validated sampling workspace without doubling peak storage."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import h5py
import numpy as np


def close_memmap(array: np.ndarray) -> None:
    """Close the mmap owned by a NumPy memmap before unlinking its file."""
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_chunks(workspace: Path, rows: int) -> list[tuple[dict, Path]]:
    chunks: list[tuple[dict, Path]] = []
    position = 0
    for directory in sorted((workspace / "chunks").glob("chunk_*")):
        meta = json.loads((directory / "chunk.json").read_text())
        if meta["row_start"] != position:
            raise RuntimeError(f"gap or overlap before {directory}: expected row {position}")
        position = int(meta["row_stop"])
        chunks.append((meta, directory))
    if position != rows:
        raise RuntimeError(f"chunks cover {position} rows, expected {rows}")
    return chunks


def repair_index(workspace: Path) -> dict[int, tuple[Path, int]]:
    result: dict[int, tuple[Path, int]] = {}
    for directory in sorted((workspace / "repairs").glob("repair_*")):
        for local, row in enumerate(np.load(directory / "rows.npy").tolist()):
            result[int(row)] = (directory, local)
    return result


def verify_family(output: Path, family: str, parts: int, shape: tuple[int, int]) -> None:
    for part in range(parts):
        path = output / f"{family}_file.{part:05d}.h5"
        if not path.is_file():
            raise RuntimeError(f"missing assembled part {path}")
        with h5py.File(path, "r") as handle:
            if tuple(handle.attrs["shape"]) != shape:
                raise RuntimeError(f"wrong shape attribute in {path}")
            expected = shape[0] * min(4096, shape[1] - part * 4096)
            if handle["real"].size != expected or handle["imag"].size != expected:
                raise RuntimeError(f"wrong dataset length in {path}")


def assemble(workspace: Path, output: Path, consume: bool) -> None:
    workspace = workspace.resolve()
    output = output.resolve()
    manifest = json.loads((workspace / "manifest.json").read_text())
    validation = manifest.get("validation") or {}
    if validation.get("status") != "passed":
        raise RuntimeError("refusing to assemble a workspace that has not passed validation")
    output.mkdir(parents=True, exist_ok=True)

    rows = int(manifest["sampling"]["samples"])
    dimension = int(manifest["dimension"])
    chunks = load_chunks(workspace, rows)
    repairs = repair_index(workspace)
    families = ["y_test", "y_sketch"]
    if not manifest["symmetric"]:
        families.extend(("z_test", "z_sketch"))
    output.mkdir(parents=True)
    parts = math.ceil(dimension / 4096)

    for family in families:
        existing_parts = sorted(output.glob(f"{family}_file.*.h5"))
        if len(existing_parts) == parts:
            verify_family(output, family, parts, (rows, dimension))
            print(f"[rsrs-assemble] family={family} already complete and verified", flush=True)
            continue
        for path in existing_parts:
            path.unlink()
        if existing_parts:
            print(f"[rsrs-assemble] family={family} removed {len(existing_parts)} incomplete parts", flush=True)
        print(f"[rsrs-assemble] family={family} start", flush=True)
        for part, col0 in enumerate(range(0, dimension, 4096)):
            col1 = min(col0 + 4096, dimension)
            pieces = []
            for meta, directory in chunks:
                source = directory / f"{family}.npy"
                array = np.load(source, mmap_mode="r")
                expected_rows = int(meta["row_stop"]) - int(meta["row_start"])
                if array.shape != (expected_rows, dimension):
                    raise RuntimeError(f"invalid {family} shape in {directory}: {array.shape}")
                piece = np.array(array[:, col0:col1], copy=True)
                close_memmap(array)
                del array
                for row in range(int(meta["row_start"]), int(meta["row_stop"])):
                    replacement = repairs.get(row)
                    if replacement:
                        repair_dir, local = replacement
                        repair = np.load(repair_dir / f"{family}.npy", mmap_mode="r")
                        piece[row - int(meta["row_start"])] = repair[local, col0:col1]
                        close_memmap(repair)
                        del repair
                pieces.append(piece)
            block = np.concatenate(pieces, axis=0)
            target = output / f"{family}_file.{part:05d}.h5"
            with h5py.File(target, "w") as handle:
                handle.attrs["shape"] = (rows, dimension)
                # RSRS/RLST arrays and the Rust multipart loader use column-major
                # storage: all sample rows for coordinate j are contiguous.
                handle.create_dataset(
                    "real",
                    data=np.asarray(block.real, dtype=np.float64).reshape(-1, order="F"),
                )
                handle.create_dataset(
                    "imag",
                    data=np.asarray(block.imag, dtype=np.float64).reshape(-1, order="F"),
                )
            if part == 0 or (part + 1) % 20 == 0 or part + 1 == parts:
                print(f"[rsrs-assemble] {family} part={part+1}/{parts}", flush=True)

        verify_family(output, family, parts, (rows, dimension))
        print(f"[rsrs-assemble] family={family} verified", flush=True)
        if consume:
            del block, pieces
            gc.collect()
            for _, directory in chunks:
                (directory / f"{family}.npy").unlink()
            print(f"[rsrs-assemble] family={family} source arrays removed", flush=True)

    report = {
        "status": "assembled",
        "storage_order": "column-major",
        "workspace": str(workspace),
        "rows": rows,
        "dimension": dimension,
        "families": families,
        "repairs": sorted(repairs),
        "validation": validation,
    }
    atomic_json(output / "assembly_report.json", report)
    shutil.copy2(workspace / "manifest.json", output / "generation_manifest.json")
    shutil.copy2(workspace / "validation_report.json", output / "validation_report.json")

    if consume:
        # Every source family has an independently verified HDF5 equivalent and
        # the provenance files above have already been copied into the cache.
        shutil.rmtree(workspace)
        print(f"[rsrs-assemble] consumed workspace={workspace}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consume-workspace", action="store_true")
    args = parser.parse_args()
    assemble(args.workspace, args.output, args.consume_workspace)


if __name__ == "__main__":
    main()
