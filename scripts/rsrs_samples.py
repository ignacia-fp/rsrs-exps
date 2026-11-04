#!/usr/bin/env python3
"""Generate, validate, repair, and assemble RSRS sample caches."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from python.rsrs_workflow.assembly import assemble
from python.rsrs_workflow.kernels import REGISTRY
from python.rsrs_workflow.random import contiguous_ranges
from python.rsrs_workflow.sampling import generate_chunk, generate_repair
from python.rsrs_workflow.spec import CaseSpec, SamplingSpec
from python.rsrs_workflow.storage import atomic_json, new_manifest, parse_rows, utc_now
from python.rsrs_workflow.validation import validate_workspace


def load_case(path: Path) -> CaseSpec:
    return CaseSpec(**json.loads(path.read_text())).resolved(path.parent)


def generate(args) -> None:
    case = load_case(args.case)
    sampling = SamplingSpec(args.samples, args.workers, args.threads_per_worker, args.batch_size, args.seed, args.validation, args.tolerance)
    workspace = args.output.resolve()
    if workspace.exists():
        raise FileExistsError(f"refusing to overwrite {workspace}")
    (workspace / "chunks").mkdir(parents=True)
    manifest = new_manifest(case, sampling)
    atomic_json(workspace / "manifest.json", manifest)
    # Set thread limits in the parent before spawn imports NumPy/BEM++ in a
    # child. Setting these only inside the worker is too late for some BLAS and
    # OpenMP runtimes.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = str(sampling.threads_per_worker)
    tasks = [
        {"worker": worker, "row_start": start, "row_stop": stop, "workspace": str(workspace),
         "case": asdict(case), "seed": sampling.seed, "batch_size": sampling.batch_size,
         "threads_per_worker": sampling.threads_per_worker}
        for worker, (start, stop) in enumerate(contiguous_ranges(sampling.samples, sampling.workers)) if start < stop
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=sampling.workers, mp_context=context) as pool:
        chunks = list(pool.map(generate_chunk, tasks))
    manifest = json.loads((workspace / "manifest.json").read_text())
    manifest["chunks"] = [{key: chunk[key] for key in ("worker", "row_start", "row_stop", "sha256")} for chunk in chunks]
    manifest["dimension"] = chunks[0]["dimension"]
    manifest["symmetric"] = chunks[0]["symmetric"]
    manifest["status"] = "generated"
    manifest["updated_utc"] = utc_now()
    atomic_json(workspace / "manifest.json", manifest)
    if sampling.validation != "none":
        report = validate_workspace(workspace, sampling.validation, sampling.validation_tolerance)
        if report["status"] != "passed":
            raise SystemExit(2)


def validate(args) -> None:
    report = validate_workspace(args.workspace.resolve(), args.mode, args.tolerance)
    print(json.dumps({key: report[key] for key in ("status", "checked", "bad_rows", "max_error")}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


def repair(args) -> None:
    workspace = args.workspace.resolve()
    manifest = json.loads((workspace / "manifest.json").read_text())
    rows = parse_rows(args.rows, int(manifest["sampling"]["samples"]))
    repairs = workspace / "repairs"
    repairs.mkdir(exist_ok=True)
    output = repairs / f"repair_{len(list(repairs.glob('repair_*'))):03d}"
    meta = generate_repair({"case": manifest["case"], "output": str(output), "rows": rows,
                            "seed": manifest["sampling"]["seed"], "batch_size": args.batch_size})
    manifest["repairs"].append({"path": str(output), "rows": rows, "completed_utc": meta["completed_utc"]})
    manifest["status"] = "repaired_needs_validation"
    manifest["validation"] = None
    manifest["updated_utc"] = utc_now()
    atomic_json(workspace / "manifest.json", manifest)
    print(f"repair written to {output}; run validate before assemble")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    listed = commands.add_parser("list-kernels"); listed.set_defaults(func=lambda _: print("\n".join(sorted(REGISTRY))))
    generated = commands.add_parser("generate")
    generated.add_argument("--case", type=Path, required=True); generated.add_argument("--output", type=Path, required=True)
    generated.add_argument("--samples", type=int, required=True); generated.add_argument("--workers", type=int, default=1)
    generated.add_argument("--threads-per-worker", type=int, default=1); generated.add_argument("--batch-size", type=int, default=10)
    generated.add_argument("--seed", type=int, default=20261104); generated.add_argument("--validation", choices=("full", "spot", "none"), default="full")
    generated.add_argument("--tolerance", type=float, default=5e-7); generated.set_defaults(func=generate)
    checked = commands.add_parser("validate"); checked.add_argument("--workspace", type=Path, required=True)
    checked.add_argument("--mode", choices=("full", "spot"), default="full"); checked.add_argument("--tolerance", type=float, default=5e-7); checked.set_defaults(func=validate)
    repaired = commands.add_parser("repair"); repaired.add_argument("--workspace", type=Path, required=True)
    repaired.add_argument("--rows", required=True, help="zero-based rows, e.g. 4,9,20:30"); repaired.add_argument("--batch-size", type=int, default=10); repaired.set_defaults(func=repair)
    assembled = commands.add_parser("assemble"); assembled.add_argument("--workspace", type=Path, required=True)
    assembled.add_argument("--output", type=Path, required=True); assembled.add_argument("--column-block", type=int, default=4096)
    assembled.set_defaults(func=lambda a: assemble(a.workspace.resolve(), a.output.resolve(), a.column_block))
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
