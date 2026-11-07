#!/usr/bin/env python3
"""Copy a contiguous row interval from an RSRS multipart HDF5 sample cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

BASES = ("y_test_file", "y_sketch_file", "z_test_file", "z_sketch_file")


def files(cache: Path, base: str) -> list[Path]:
    result = sorted(cache.glob(f"{base}.*.h5"))
    if not result:
        raise FileNotFoundError(f"no {base} parts in {cache}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True, help="zero-based inclusive row")
    parser.add_argument("--stop", type=int, required=True, help="zero-based exclusive row")
    args = parser.parse_args()
    if args.start < 0 or args.stop <= args.start:
        parser.error("require 0 <= start < stop")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    output_rows = args.stop - args.start
    report = {"source": str(args.source.resolve()), "output": str(args.output.resolve()),
              "source_rows": None, "rows": [args.start, args.stop],
              "output_rows": output_rows, "dimension": None, "parts": {}}
    try:
        for base in BASES:
            source_parts = files(args.source, base)
            report["parts"][base] = len(source_parts)
            for part_index, source_path in enumerate(source_parts):
                destination_path = args.output / source_path.name
                with h5py.File(source_path, "r") as source:
                    shape = tuple(int(x) for x in source.attrs["shape"])
                    source_rows, dimension = shape
                    if args.stop > source_rows:
                        raise ValueError(f"row interval exceeds {source_path}: {shape}")
                    if report["source_rows"] is None:
                        report["source_rows"], report["dimension"] = shape
                    elif (report["source_rows"], report["dimension"]) != shape:
                        raise ValueError(f"inconsistent cache shape in {source_path}: {shape}")
                    width, remainder = divmod(source["real"].size, source_rows)
                    if remainder or source["imag"].size != source["real"].size:
                        raise ValueError(f"invalid multipart data size in {source_path}")
                    with h5py.File(destination_path, "w") as destination:
                        for key, value in source.attrs.items():
                            destination.attrs[key] = value
                        destination.attrs["shape"] = np.array(
                            (output_rows, dimension), dtype=np.int64
                        )
                        for component in ("real", "imag"):
                            source_matrix = np.asarray(source[component]).reshape(
                                (source_rows, width), order="F"
                            )
                            selected = np.ascontiguousarray(
                                source_matrix[args.start:args.stop, :]
                            )
                            flat = selected.reshape(-1, order="F")
                            destination.create_dataset(component, data=flat)
                            if not np.array_equal(destination[component][:], flat):
                                raise RuntimeError(f"write verification failed: {destination_path}")
                print(
                    f"[trim-cache] {base} part={part_index + 1}/{len(source_parts)}",
                    flush=True,
                )
        report["status"] = "complete"
        (args.output / "trim_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[trim-cache] complete output={args.output}", flush=True)
        return 0
    except BaseException:
        report["status"] = "failed"
        (args.output / "trim_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
