#!/usr/bin/env python3
"""Append a validated workspace to an RSRS multipart cache, without mutation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from python.rsrs_workflow.storage import atomic_json, read_chunk, utc_now, verify_contiguous


FAMILIES = ("y_test", "y_sketch", "z_test", "z_sketch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-cache", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prefix-rows", type=int, required=True)
    args = parser.parse_args()

    prefix = args.prefix_cache.resolve()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((workspace / "manifest.json").read_text())
    validation = manifest.get("validation") or {}
    if validation.get("status") != "passed" or validation.get("covered_rows") != manifest["sampling"]["samples"]:
        raise RuntimeError("workspace has not passed all-row combination validation")
    tail_rows = int(manifest["sampling"]["samples"])
    dimension = int(manifest["dimension"])
    chunks = verify_contiguous(workspace, tail_rows)
    output.mkdir(parents=True)

    try:
        for family in FAMILIES:
            base = f"{family}_file"
            source_files = sorted(prefix.glob(f"{base}.*.h5"))
            expected_parts = math.ceil(dimension / 4096)
            if len(source_files) != expected_parts:
                raise RuntimeError(f"{base}: expected {expected_parts} parts, found {len(source_files)}")
            for part, source_path in enumerate(source_files):
                with h5py.File(source_path, "r") as source:
                    old_shape = tuple(int(v) for v in source.attrs["shape"])
                    if old_shape != (args.expected_prefix_rows, dimension):
                        raise RuntimeError(f"unexpected prefix shape in {source_path}: {old_shape}")
                    old_rows = old_shape[0]
                    width = len(source["real"]) // old_rows
                    new_rows = old_rows + tail_rows
                    real = np.empty((width, new_rows), dtype=np.float64)
                    imag = np.empty((width, new_rows), dtype=np.float64)
                    real[:, :old_rows] = source["real"][:].reshape(width, old_rows)
                    imag[:, :old_rows] = source["imag"][:].reshape(width, old_rows)
                col0 = part * 4096
                col1 = col0 + width
                for meta, directory in chunks:
                    _, arrays = read_chunk(directory)
                    row0 = old_rows + int(meta["row_start"])
                    row1 = old_rows + int(meta["row_stop"])
                    block = np.asarray(arrays[family][:, col0:col1])
                    real[:, row0:row1] = block.real.T
                    imag[:, row0:row1] = block.imag.T
                target = output / source_path.name
                with h5py.File(target, "w") as handle:
                    handle.attrs["shape"] = np.asarray((new_rows, dimension), dtype=np.uint64)
                    handle.create_dataset("real", data=real.ravel(order="C"))
                    handle.create_dataset("imag", data=imag.ravel(order="C"))
                print(
                    f"[cache-extension] {family} part={part+1}/{len(source_files)} "
                    f"rows={new_rows} cols={col0}:{col1}",
                    flush=True,
                )
        report = {
            "status": "complete",
            "prefix_cache": str(prefix),
            "workspace": str(workspace),
            "prefix_rows": args.expected_prefix_rows,
            "tail_rows": tail_rows,
            "rows": args.expected_prefix_rows + tail_rows,
            "dimension": dimension,
            "layout": "RSRS column-major multipart HDF5",
            "completed_utc": utc_now(),
        }
        atomic_json(output / "extension_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        (output / "INCOMPLETE").write_text("cache extension did not complete\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
