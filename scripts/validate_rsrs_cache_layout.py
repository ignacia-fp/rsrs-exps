#!/usr/bin/env python3
"""Validate multipart layout against deterministic RSRS Python test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


STREAM_TAGS = {"y": 0x59, "z": 0x5A}


def expected_test(seed: int, stream: str, row: int, dimension: int) -> np.ndarray:
    sequence = np.random.SeedSequence([seed, STREAM_TAGS[stream], row])
    return np.random.default_rng(sequence).standard_normal(dimension)


def load_row(sampling: Path, family: str, row: int, rows: int, dimension: int) -> np.ndarray:
    pieces = []
    parts = sorted(sampling.glob(f"{family}_file.*.h5"))
    if not parts:
        raise RuntimeError(f"missing {family} parts in {sampling}")
    for expected_part, path in enumerate(parts):
        if path.name != f"{family}_file.{expected_part:05d}.h5":
            raise RuntimeError(f"non-contiguous part sequence at {path}")
        with h5py.File(path, "r") as handle:
            if tuple(handle.attrs["shape"]) != (rows, dimension):
                raise RuntimeError(f"wrong shape in {path}: {tuple(handle.attrs['shape'])}")
            real = handle["real"][:]
            if real.size % rows:
                raise RuntimeError(f"invalid real length in {path}")
            width = real.size // rows
            value = real.reshape((rows, width), order="F")[row]
            if "imag" in handle:
                imag = handle["imag"][:]
                if imag.size != real.size:
                    raise RuntimeError(f"real/imag length mismatch in {path}")
                value = value + 1j * imag.reshape((rows, width), order="F")[row]
            pieces.append(value)
    result = np.concatenate(pieces)
    if result.size != dimension:
        raise RuntimeError(f"{family}: reconstructed {result.size} coordinates, expected {dimension}")
    return result


def validate(sampling: Path) -> None:
    sampling = sampling.resolve()
    repair = sampling / "layout_repair_report.json"
    assembly_path = sampling / "assembly_report.json"
    assembly = json.loads(assembly_path.read_text()) if assembly_path.is_file() else {}
    if not repair.is_file() and assembly.get("storage_order") != "column-major":
        raise RuntimeError(
            f"cache has neither a layout repair report nor column-major assembly provenance: {sampling}"
        )
    manifest = json.loads((sampling / "generation_manifest.json").read_text())
    rows = int(manifest["sampling"]["samples"])
    dimension = int(manifest["dimension"])
    seed = int(manifest["sampling"]["seed"])
    selected = sorted({0, rows // 2, rows - 1})

    for stream in ("y", "z"):
        for row in selected:
            actual = load_row(sampling, f"{stream}_test", row, rows, dimension)
            expected = expected_test(seed, stream, row, dimension)
            error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            print(f"[layout-validation] cache={sampling.parent.name} stream={stream} row={row} error={error:.3e}", flush=True)
            if not np.isfinite(error) or error > 1e-14:
                raise RuntimeError(f"{stream} row {row} does not match deterministic test: {error:.3e}")

    # Force complete multipart traversal for sketches as well; their numerical
    # operator relation was established by pre-consolidation spot validation.
    for family in ("y_sketch", "z_sketch"):
        for row in selected:
            value = load_row(sampling, family, row, rows, dimension)
            if not np.all(np.isfinite(value)) or np.linalg.norm(value) == 0:
                raise RuntimeError(f"invalid {family} row {row}")
    print(f"[layout-validation] passed cache={sampling}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sampling", type=Path, nargs="+")
    args = parser.parse_args()
    for sampling in args.sampling:
        validate(sampling)


if __name__ == "__main__":
    main()
