#!/usr/bin/env python3
"""Validate every workspace row probabilistically in fixed-size combinations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from python.rsrs_workflow.kernels import build_kernel
from python.rsrs_workflow.spec import CaseSpec
from python.rsrs_workflow.storage import atomic_json, read_chunk, sha256, utc_now, verify_contiguous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=5e-7)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    manifest = json.loads((workspace / "manifest.json").read_text())
    total = int(manifest["sampling"]["samples"])
    seed = int(manifest["sampling"]["seed"])
    chunks = verify_contiguous(workspace, total)
    fingerprint = manifest.get("mesh_fingerprint")
    if fingerprint and sha256(Path(fingerprint["path"])) != fingerprint["sha256"]:
        raise RuntimeError("mesh fingerprint differs from generation mesh")

    isolated = workspace / "validation_operator_combinations"
    isolated.mkdir(exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(isolated)
    os.environ["TMPDIR"] = str(isolated)
    try:
        operator = build_kernel(CaseSpec(**manifest["case"]))
        streams = ("y",) if operator.symmetric else ("y", "z")
        checked_groups = 0
        covered_rows = 0
        max_error = 0.0
        worst = None
        for meta, directory in chunks:
            _, arrays = read_chunk(directory)
            count = meta["row_stop"] - meta["row_start"]
            for local0 in range(0, count, args.group_size):
                local1 = min(count, local0 + args.group_size)
                global0 = meta["row_start"] + local0
                global1 = meta["row_start"] + local1
                coefficients = np.random.default_rng(
                    np.random.SeedSequence([seed, 0xC030, global0, global1])
                ).standard_normal(local1 - local0)
                coefficients /= np.linalg.norm(coefficients)
                for stream in streams:
                    tests = np.asarray(arrays[f"{stream}_test"][local0:local1])
                    sketches = np.asarray(arrays[f"{stream}_sketch"][local0:local1])
                    combined_test = coefficients @ tests
                    observed = coefficients @ sketches
                    expected = np.asarray(operator.sketch(stream, combined_test[:, None]))[:, 0]
                    denominator = max(np.linalg.norm(expected), np.linalg.norm(observed), np.finfo(float).tiny)
                    error = float(np.linalg.norm(expected - observed) / denominator)
                    checked_groups += 1
                    if error > max_error:
                        max_error, worst = error, {"stream": stream, "rows": [global0, global1]}
                    if not np.isfinite(error) or error > args.tolerance:
                        raise RuntimeError(
                            f"validation failed stream={stream} rows={global0}:{global1} "
                            f"error={error:.6e} tolerance={args.tolerance:.6e}"
                        )
                covered_rows += local1 - local0
                print(
                    f"[combination-validation] rows={global0}:{global1} "
                    f"covered={covered_rows}/{total} max_error={max_error:.3e}",
                    flush=True,
                )
    finally:
        os.chdir(old_cwd)

    report = {
        "status": "passed",
        "mode": f"combination-{args.group_size}",
        "tolerance": args.tolerance,
        "covered_rows": covered_rows,
        "checked_groups": checked_groups,
        "max_error": max_error,
        "worst": worst,
        "completed_utc": utc_now(),
    }
    atomic_json(workspace / "combination_validation_report.json", report)
    manifest["validation"] = report
    manifest["status"] = "validated"
    manifest["updated_utc"] = utc_now()
    atomic_json(workspace / "manifest.json", manifest)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
