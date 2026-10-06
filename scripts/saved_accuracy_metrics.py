#!/usr/bin/env python3
"""Label saved RSRS accuracy metrics without recomputing or changing raw results."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time


SCHEMA = Path(__file__).with_name("accuracy_metrics_schema.json")
METRIC_NAMES = ("compression_relerr_fro", "compression_relerr_2", "solve_residual_relerr_fro", "solve_residual_norm_2")
TERMINAL_QUEUE_STATES = {"completed", "completed_with_failures"}


def read_json(path):
    return json.loads(Path(path).read_text())


def accuracy_metrics(record):
    dim = record["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError("dim must be a nonnegative integer")

    def finite_ratio(numerator, denominator):
        if not isinstance(numerator, (int, float)) or isinstance(numerator, bool):
            return None
        if not math.isfinite(numerator) or numerator < 0:
            return None
        if not isinstance(denominator, (int, float)) or isinstance(denominator, bool):
            return None
        if not math.isfinite(denominator) or denominator <= 0:
            return None
        value = numerator / denominator
        return value if math.isfinite(value) else None

    metrics = read_json(SCHEMA)
    metrics.update(
        compression_relerr_fro=finite_ratio(record["norm_apply_fro"], record["norm_a_fro"]),
        compression_relerr_2=finite_ratio(record["norm_apply_2"], record["norm_a_2"]),
        solve_residual_relerr_fro=finite_ratio(record["err_solve_fro"], math.sqrt(dim)),
        solve_residual_norm_2=finite_ratio(record["err_solve_2"], 1.0),
    )
    return metrics


def atomic_json(path, record):
    path = Path(path)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), mode)
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_result(path):
    record = read_json(path)
    metrics = accuracy_metrics(record)
    changed = record.get("accuracy_metrics") != metrics
    if changed:
        record["accuracy_metrics"] = metrics
        atomic_json(path, record)
    return record, changed


def backfill_queue(root):
    root = Path(root).resolve()
    rows, errors, lookup = [], [], {}
    updated = 0
    for family in ("sphere", "jet"):
        for status_path in sorted((root / family).glob("n*/k*/status.json")):
            try:
                run = read_json(status_path)
                if run.get("state") != "completed" or run.get("returncode") != 0:
                    continue
                result_path = Path(run["stats"]["error_stats"]).resolve()
                if not result_path.is_relative_to((status_path.parent / "results").resolve()):
                    raise ValueError("result path is outside this run's results directory")
                raw = read_json(result_path)
                if raw["dim"] != run["n_dofs"]:
                    raise ValueError("result and run dimensions differ")
                record, changed = update_result(result_path)
                updated += int(changed)
                metrics = record["accuracy_metrics"]
                if run.get("accuracy_metrics") != metrics or run["stats"].get("error_stats_data") != record:
                    run["accuracy_metrics"] = metrics
                    run["stats"]["error_stats_data"] = record
                    atomic_json(status_path, run)
                row = {key: run.get(key) for key in (
                    "n_dofs", "rank", "p", "leaf_size", "elapsed_seconds", "samples_required", "sample_root",
                )}
                row.update(family=family, result_file=str(result_path),
                           samples_consumed=record.get("tot_num_samples"),
                           accuracy_metrics=metrics)
                rows.append(row)
                lookup[(run["n_dofs"], run["rank"])] = metrics
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(dict(path=str(status_path), error=str(error)))

    # A live driver owns its aggregate status file. Only update it after it stops.
    queue_paths = [root / "status.json"] + sorted(root.glob("sphere_rank*_queue/status.json"))
    for path in queue_paths:
        if not path.exists():
            continue
        aggregate = read_json(path)
        if aggregate.get("state") not in TERMINAL_QUEUE_STATES:
            continue
        changed = False
        for entry in aggregate.get("results", []):
            metrics = lookup.get((entry.get("n_dofs"), entry.get("rank")))
            if metrics is not None and entry.get("accuracy_metrics") != metrics:
                entry["accuracy_metrics"] = metrics
                if "error_stats_data" in entry.get("stats", {}):
                    entry["stats"]["error_stats_data"]["accuracy_metrics"] = metrics
                changed = True
        if changed:
            atomic_json(path, aggregate)

    rows.sort(key=lambda row: (row["family"], row["rank"], row["n_dofs"]))
    summary = dict(schema_version="rsrs_accuracy_summary_v1", results=rows, errors=errors)
    summary_path = root / "accuracy_summary.json"
    if not summary_path.exists() or read_json(summary_path) != summary:
        atomic_json(summary_path, summary)
    return dict(completed_results=len(rows), result_files_updated=updated, errors=errors)


def watch(root, interval):
    root = Path(root).resolve()
    with (root / "accuracy_metrics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = backfill_queue(root)
            queues = [root / "status.json"] + sorted(root.glob("sphere_rank*_queue/status.json"))
            done = all(p.exists() and read_json(p).get("state") in TERMINAL_QUEUE_STATES for p in queues)
            phase = "completed" if done and not result["errors"] else "watching"
            atomic_json(root / "accuracy_metrics_status.json", dict(
                state=phase, pid=os.getpid(), updated_unix=time.time(), **result,
            ))
            if phase == "completed":
                return
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--launch-watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not args.queue_root.is_dir() or args.interval <= 0:
        parser.error("queue-root must exist and interval must be positive")
    if args.launch_watch:
        with (args.queue_root / "accuracy_metrics.log").open("a") as log:
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--queue-root", str(args.queue_root.resolve()),
                "--watch", "--interval", str(args.interval),
            ], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps(dict(watcher_pid=process.pid)))
    elif args.watch:
        watch(args.queue_root, args.interval)
    else:
        print(json.dumps(backfill_queue(args.queue_root)))


if __name__ == "__main__":
    main()
