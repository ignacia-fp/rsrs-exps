from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .storage import atomic_json, read_chunk, utc_now, verify_contiguous


def _repair_index(workspace: Path) -> dict[int, tuple[Path, int]]:
    index: dict[int, tuple[Path, int]] = {}
    repairs = workspace / "repairs"
    if not repairs.exists():
        return index
    for directory in sorted(repairs.glob("repair_*")):
        rows = np.load(directory / "rows.npy")
        for local, row in enumerate(rows.tolist()):
            index[int(row)] = (directory, local)
    return index


def assemble(workspace: Path, output: Path, column_block: int = 4096) -> dict:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("assembly requires h5py in the active RSRS environment") from exc
    manifest = json.loads((workspace / "manifest.json").read_text())
    validation = manifest.get("validation") or {}
    if validation.get("status") != "passed":
        raise RuntimeError("refusing to assemble: workspace has not passed validation")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = int(manifest["sampling"]["samples"])
    chunks = verify_contiguous(workspace, rows)
    dimension = int(manifest["dimension"])
    families = ["y_test", "y_sketch"] + ([] if manifest["symmetric"] else ["z_test", "z_sketch"])
    repairs = _repair_index(workspace)
    base_names = {name: f"{name}_file" for name in families}
    for family in families:
        for part, col0 in enumerate(range(0, dimension, column_block)):
            col1 = min(col0 + column_block, dimension)
            pieces = []
            for meta, directory in chunks:
                _, arrays = read_chunk(directory)
                piece = np.array(arrays[family][:, col0:col1], copy=True)
                for global_row in range(meta["row_start"], meta["row_stop"]):
                    replacement = repairs.get(global_row)
                    if replacement:
                        repair_dir, local = replacement
                        repair = np.load(repair_dir / f"{family}.npy", mmap_mode="r")
                        piece[global_row - meta["row_start"]] = repair[local, col0:col1]
                pieces.append(piece)
            block = np.concatenate(pieces, axis=0)
            with h5py.File(output / f"{base_names[family]}.{part:05d}.h5", "w") as handle:
                handle.attrs["shape"] = (rows, dimension)
                handle.create_dataset("real", data=np.asarray(block.real, dtype=np.float64).ravel())
                handle.create_dataset("imag", data=np.asarray(block.imag, dtype=np.float64).ravel())
            if part == 0 or (part + 1) % 20 == 0 or col1 == dimension:
                print(f"[rsrs-assemble] {family} part={part+1}/{math.ceil(dimension/column_block)}", flush=True)
    report = {
        "status": "assembled", "workspace": str(workspace), "rows": rows, "dimension": dimension,
        "families": families, "repair_rows": sorted(repairs), "completed_utc": utc_now(),
    }
    atomic_json(output / "assembly_report.json", report)
    manifest["assembled_cache"] = {"path": str(output), "completed_utc": report["completed_utc"]}
    manifest["updated_utc"] = utc_now()
    atomic_json(workspace / "manifest.json", manifest)
    return report
