#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("laplace_sampler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    module = load_module(args.repository / "scripts/generate_laplace_single_layer_parallel.py")
    config = json.loads((args.staging / "config.json").read_text())
    dimension = json.loads((args.staging / "mesh_info.json").read_text())["dimension"]
    batch_size = int(config["batch_size"])
    all_ranges = list(module.batch_ranges(int(config["samples"]), batch_size))
    available = {
        (int(path.name.split("_")[1]), int(path.name.split("_")[2]))
        for path in (args.staging / "_batches").glob("batch_*_*")
        if ".tmp." not in path.name
    }
    prefix = []
    for pair in all_ranges:
        if pair not in available:
            break
        prefix.append(pair)
    if not available:
        raise RuntimeError("no completed batches")
    completed = sorted(available)
    rows = prefix[-1][1]
    inventory = {
        "completed_directories": len(available),
        "contiguous_batches": len(prefix),
        "contiguous_rows": rows,
        "all_completed_rows": sum(b - a for a, b in completed),
        "first_missing_range": list(all_ranges[len(prefix)]) if len(prefix) < len(all_ranges) else None,
    }
    report_dir = args.staging / "_partial_validation_combination30"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"[partial-validation] inventory={json.dumps(inventory)}", flush=True)
    report = module.validate_batches(
        completed, args.staging / "_batches", report_dir / "receipts",
        str(args.staging / "grid.msh"), config["validation_assembler"],
        int(config["seed"]), int(dimension), float(config["tolerance"]),
    )
    report["inventory"] = inventory
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[partial-validation] complete report={report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
