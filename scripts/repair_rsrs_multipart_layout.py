#!/usr/bin/env python3
"""Repair RSRS multipart files accidentally flattened in NumPy C order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


FAMILIES = ("y_test", "y_sketch", "z_test", "z_sketch")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def repair_file(path: Path, rows: int) -> dict:
    temporary = path.with_name(path.name + ".layout-fix.tmp")
    if temporary.exists():
        temporary.unlink()

    with h5py.File(path, "r") as source:
        shape = tuple(int(value) for value in source.attrs["shape"])
        if shape[0] != rows:
            raise RuntimeError(f"{path}: shape rows {shape[0]} != expected {rows}")
        names = tuple(name for name in ("real", "imag") if name in source)
        if not names:
            raise RuntimeError(f"{path}: neither real nor imag dataset exists")
        size = int(source[names[0]].size)
        if size % rows:
            raise RuntimeError(f"{path}: dataset length {size} is not divisible by {rows}")
        width = size // rows
        logical = {}
        chunks = {}
        for name in names:
            flat = source[name][:]
            if flat.size != size:
                raise RuntimeError(f"{path}: inconsistent {name} length")
            logical[name] = flat.reshape((rows, width), order="C")
            chunks[name] = source[name].chunks

    before = {name: digest(value) for name, value in logical.items()}
    with h5py.File(temporary, "w") as target:
        target.attrs["shape"] = shape
        for name, value in logical.items():
            flat = value.reshape(-1, order="F")
            kwargs = {"chunks": chunks[name]} if chunks[name] is not None else {}
            target.create_dataset(name, data=flat, **kwargs)

    with h5py.File(temporary, "r") as check:
        after = {
            name: digest(check[name][:].reshape((rows, width), order="F"))
            for name in logical
        }
    if before != after:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: logical values changed during layout repair")

    os.replace(temporary, path)
    return {"file": path.name, "width": width, "logical_sha256": before}


def repair_cache(sampling: Path) -> None:
    sampling = sampling.resolve()
    report_path = sampling / "layout_repair_report.json"
    if report_path.exists():
        raise RuntimeError(f"refusing to repair an already marked cache: {report_path}")
    assembly_path = sampling / "assembly_report.json"
    if not assembly_path.is_file():
        raise RuntimeError(f"missing Python assembly provenance: {assembly_path}")
    assembly = json.loads(assembly_path.read_text())
    rows = int(assembly["rows"])

    repaired = []
    for family in FAMILIES:
        parts = sorted(sampling.glob(f"{family}_file.*.h5"))
        if not parts:
            raise RuntimeError(f"no parts found for {family} in {sampling}")
        for index, path in enumerate(parts, 1):
            repaired.append(repair_file(path, rows))
            if index == 1 or index % 20 == 0 or index == len(parts):
                print(f"[layout-repair] {family} part={index}/{len(parts)}", flush=True)

    report = {
        "status": "repaired",
        "repair": "c-order-flat-to-rust-column-major-v1",
        "sampling": str(sampling),
        "rows": rows,
        "files": repaired,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    print(f"[layout-repair] complete report={report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sampling", type=Path, nargs="+")
    args = parser.parse_args()
    for sampling in args.sampling:
        repair_cache(sampling)


if __name__ == "__main__":
    main()
