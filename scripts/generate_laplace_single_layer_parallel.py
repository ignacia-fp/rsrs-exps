#!/usr/bin/env python3
"""Parallel Y-only sampling of the Laplace single-layer Galerkin matrix on a sphere.

Each subprocess builds its own Bempp operator ONCE, then processes many small
batches (10 samples by default). A progress/elapsed/ETA line is emitted after
EACH completed batch. The main process independently builds a fresh operator,
checks every stored test vector and checksum, validates one seeded random
combination per 30 rows, and validates 60 distributed rows individually.

Intermediate per-batch .npy files make runs resumable with --resume. Only after
all samples validate are the original RSRS-compatible multipart HDF5 files
written to OUTPUT/sampling/; the saved sphere mesh is OUTPUT/grid.msh.

Requirements: numpy, h5py, bempp-cl, sphere meshing dependencies (e.g. gmsh),
and whichever Bempp assembler backend is selected.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np

PREFIX = "[laplace-samples]"
BLOCK_COLS = 4096
_WORKER_OPERATOR = None
_WORKER_DIM = None
_WORKER_ID = None
_WORKER_CONFIG = None


def log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


def duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    sec = int(seconds)
    hours, remainder = divmod(sec, 3600)
    mins, sec = divmod(remainder, 60)
    return f"{hours:02d}:{mins:02d}:{sec:02d}" if hours else f"{mins:02d}:{sec:02d}"


def progress(stage: str, done: int, total: int, *, started: float,
             new_done: int, new_total: int, detail: str = "",
             eta_override: float | None = None) -> None:
    """ETA is based on batches finished in THIS invocation; no timer polling."""
    elapsed = time.perf_counter() - started
    remaining = (elapsed * (new_total - new_done) / new_done
                 if new_done else float("nan"))
    if eta_override is not None:
        remaining = eta_override
    log(f"{stage}: progress={done}/{total} ({100 * done / total:.1f}%) "
        f"elapsed={duration(elapsed)} eta={duration(remaining)}"
        + (f" {detail}" if detail else ""))


def build_operator(mesh_path: str, assembler: str):
    import bempp_cl.api as bempp
    bempp.DEFAULT_PRECISION = "double"
    grid = bempp.import_grid(mesh_path)
    space = bempp.function_space(grid, "DP", 0)
    op = bempp.operators.boundary.laplace.single_layer(
        space, space, space, assembler=assembler
    ).weak_form()
    return op, int(space.global_dof_count)


def matvec(operator, vector: np.ndarray) -> np.ndarray:
    result = np.asarray(operator @ vector).reshape(-1)
    if np.iscomplexobj(result):
        if np.any(np.abs(result.imag) > 1e-12 * np.maximum(1, np.abs(result.real))):
            raise ValueError("Laplace matrix produced non-negligible imaginary entries")
        result = result.real
    return np.asarray(result, dtype=np.float64)


def sample_vector(seed: int, index: int, dimension: int) -> np.ndarray:
    """Independent reproducible stream per global sample index, not per worker."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    return rng.standard_normal(dimension)


def batch_ranges(samples: int, batch_size: int):
    for row0 in range(0, samples, batch_size):
        yield row0, min(row0 + batch_size, samples)


def batch_name(row0: int, row1: int) -> str:
    return f"batch_{row0:08d}_{row1:08d}"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def check_batch(folder: Path, row0: int, row1: int, dimension: int) -> dict:
    """Check the saved batch, including checksums, before trusting a resumed run."""
    manifest = json.loads((folder / "meta.json").read_text())
    if [row0, row1] != manifest["rows"] or manifest["dimension"] != dimension:
        raise ValueError(f"invalid batch metadata: {folder}")
    for key in ("y_test", "y_sketch"):
        path = folder / f"{key}.npy"
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape != (row1 - row0, dimension) or data.dtype != np.float64:
            raise ValueError(f"invalid shape or dtype in {path}: {data.shape} {data.dtype}")
        if digest(path) != manifest["sha256"][key]:
            raise ValueError(f"checksum mismatch: {path}")
    return manifest


def init_worker(config: dict, counter) -> None:
    """Called ONCE per process; the operator is reused for all its batches."""
    global _WORKER_OPERATOR, _WORKER_DIM, _WORKER_ID, _WORKER_CONFIG
    with counter.get_lock():
        _WORKER_ID = counter.value
        counter.value += 1
    _WORKER_CONFIG = config
    log(f"worker={_WORKER_ID} pid={os.getpid()} building operator")
    started = time.perf_counter()
    _WORKER_OPERATOR, _WORKER_DIM = build_operator(
        config["mesh_path"], config["assembler"]
    )
    if _WORKER_DIM != config["dimension"]:
        raise ValueError(f"worker={_WORKER_ID}: grid dimension mismatch")
    log(f"worker={_WORKER_ID} operator ready elapsed={time.perf_counter()-started:.3f}s")


def generate_batch(task: tuple[int, int]) -> dict:
    """Compute exactly one batch with the worker's already-built operator."""
    row0, row1 = task
    started = time.perf_counter()
    count = row1 - row0
    ys = np.empty((count, _WORKER_DIM), dtype=np.float64)
    tests = np.empty_like(ys)
    for local, index in enumerate(range(row0, row1)):
        omega = sample_vector(_WORKER_CONFIG["seed"], index, _WORKER_DIM)
        tests[local, :] = omega
        ys[local, :] = matvec(_WORKER_OPERATOR, omega)

    apply_seconds = time.perf_counter() - started
    folder = Path(_WORKER_CONFIG["batch_dir"]) / batch_name(row0, row1)
    temporary = folder.with_name(folder.name + f".tmp.{os.getpid()}")
    temporary.mkdir()
    try:
        np.save(temporary / "y_test.npy", tests, allow_pickle=False)
        np.save(temporary / "y_sketch.npy", ys, allow_pickle=False)
        record = {
            "rows": [row0, row1], "dimension": _WORKER_DIM,
            "worker_id": _WORKER_ID, "count": count,
            "apply_seconds": apply_seconds,
            "sha256": {
                key: digest(temporary / f"{key}.npy")
                for key in ("y_test", "y_sketch")
            },
        }
        (temporary / "meta.json").write_text(json.dumps(record, indent=2) + "\n")
        temporary.rename(folder)  # Atomic complete-batch checkpoint.
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    record["elapsed_seconds"] = time.perf_counter() - started
    log(f"worker={_WORKER_ID} complete rows={row0}:{row1} "
        f"count={count} apply={apply_seconds:.3f}s "
        f"elapsed={record['elapsed_seconds']:.3f}s")
    return record


def sample_parallel(pending: list[tuple[int, int]], workers: int,
                    config: dict, total: int, already_done: int) -> float:
    started = time.perf_counter()
    if not pending:
        log(f"sampling: already complete {already_done}/{total} (resumed)")
        return 0.0
    pending_total = sum(b - a for a, b in pending)
    completed_new = 0
    batch_compute_seconds = 0.0
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    log(f"sampling: workers={workers} batch_size={config['batch_size']} "
        f"pending={pending_total}/{total}")
    with cf.ProcessPoolExecutor(
        max_workers=min(workers, len(pending)),
        mp_context=ctx, initializer=init_worker, initargs=(config, counter),
    ) as pool:
        futures = [pool.submit(generate_batch, rng) for rng in pending]
        for future in cf.as_completed(futures):
            result = future.result()
            completed_new += result["count"]
            batch_compute_seconds += result["elapsed_seconds"]
            # Estimate using observed batch computation time across parallel
            # workers; exclude their one-time operator construction overhead.
            estimated_eta = (batch_compute_seconds / completed_new
                             * (pending_total - completed_new)
                             / min(workers, len(pending)))
            progress("sampling", already_done + completed_new, total,
                     started=started, new_done=completed_new,
                     new_total=pending_total, eta_override=estimated_eta,
                     detail=f"worker={result['worker_id']} "
                            f"rows={result['rows'][0]}:{result['rows'][1]} "
                            f"batch_elapsed={result['elapsed_seconds']:.3f}s")
    return time.perf_counter() - started


def validate_batches(batches: list[tuple[int, int]], batch_dir: Path,
                     receipt_dir: Path, mesh_path: str, assembler: str,
                     seed: int, dimension: int, tolerance: float) -> dict:
    """Validate every row structurally and every 30 rows by a random combination.

    Linearity makes one seeded combination a probabilistic check of every
    sample relation in the group. Sixty distributed rows are also checked
    individually to retain sensitivity to small isolated errors.
    """
    started = time.perf_counter()
    receipt_dir.mkdir(exist_ok=True)
    validation_version = "combination30-plus-60-individual-v1"
    group_size = 30
    total = sum(b - a for a, b in batches)
    available_rows = [row for a, b in batches for row in range(a, b)]
    exact_positions = np.linspace(0, total - 1, min(60, total), dtype=int)
    exact_rows = {available_rows[position] for position in exact_positions}
    groups = []
    current = []
    current_count = 0
    for pair in batches:
        count = pair[1] - pair[0]
        if current and (pair[0] != current[-1][1] or current_count + count > group_size):
            groups.append(current)
            current, current_count = [], 0
        current.append(pair)
        current_count += count
    if current:
        groups.append(current)
    checked_prior = 0
    max_error = 0.0
    worst_row = None
    todo = []
    for group in groups:
        manifests = [check_batch(batch_dir / batch_name(a, b), a, b, dimension)
                     for a, b in group]
        row0, row1 = group[0][0], group[-1][1]
        fingerprints = [manifest["sha256"] for manifest in manifests]
        receipt_path = receipt_dir / f"group_{row0:08d}_{row1:08d}.json"
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if (previous.get("validation_version") == validation_version
                    and previous.get("sha256") == fingerprints
                    and previous.get("assembler") == assembler
                    and previous.get("tolerance") == tolerance
                    and previous.get("rows") == [row0, row1]
                    and previous["max_relative_error"] <= tolerance):
                checked_prior += row1 - row0
                if previous["max_relative_error"] > max_error:
                    max_error = previous["max_relative_error"]
                    worst_row = previous["worst_row"]
                continue
        todo.append((group, manifests, receipt_path))

    pending_total = total - checked_prior
    if not todo:
        log(f"validation: {total}/{total} samples already independently validated "
            f"in a previous invocation")
        return {"checked_samples": total, "max_relative_error": max_error,
                "worst_row": worst_row, "tolerance": tolerance,
                "assembler": assembler, "reused_validated_samples": checked_prior,
                "validation_version": validation_version,
                "operator_construction_seconds": 0.0,
                "apply_seconds": 0.0, "elapsed_seconds": time.perf_counter() - started}

    log(f"validation: building fresh operator assembler={assembler}")
    build_started = time.perf_counter()
    independent, actual_dim = build_operator(mesh_path, assembler)
    build_seconds = time.perf_counter() - build_started
    if actual_dim != dimension:
        raise RuntimeError("validator: mesh DOFs do not match generation")
    log(f"validation: operator ready elapsed={build_seconds:.3f}s")

    apply_started = time.perf_counter()
    checked_new = 0
    exact_checked = 0
    combinations_checked = 0
    for group, manifests, receipt_path in todo:
        row0, row1 = group[0][0], group[-1][1]
        batch_started = time.perf_counter()
        tests = np.concatenate([
            np.load(batch_dir / batch_name(a, b) / "y_test.npy", mmap_mode="r", allow_pickle=False)
            for a, b in group], axis=0)
        sketches = np.concatenate([
            np.load(batch_dir / batch_name(a, b) / "y_sketch.npy", mmap_mode="r", allow_pickle=False)
            for a, b in group], axis=0)
        local_max = 0.0
        local_worst = None
        indices = [index for a, b in group for index in range(a, b)]
        for local, index in enumerate(indices):
            omega = np.asarray(tests[local])
            if not np.array_equal(omega, sample_vector(seed, index, dimension)):
                raise ValueError(f"random test vector mismatch at sample={index}")
            if not (np.all(np.isfinite(omega)) and np.all(np.isfinite(sketches[local]))):
                raise ValueError(f"nonfinite values at sample={index}")

        coefficients = np.random.default_rng(
            np.random.SeedSequence([seed, 0xC030, row0, row1])
        ).standard_normal(len(indices))
        coefficients /= np.linalg.norm(coefficients)
        combined_test = coefficients @ tests
        observed = coefficients @ sketches
        expected = matvec(independent, combined_test)
        denominator = max(np.linalg.norm(expected), np.linalg.norm(observed),
                          np.finfo(float).tiny)
        error = float(np.linalg.norm(expected - observed) / denominator)
        local_max, local_worst = error, f"combination:{row0}:{row1}"
        combinations_checked += 1
        if error > tolerance:
            raise ValueError(f"validation FAILED combination rows={row0}:{row1} "
                             f"relative_error={error:.6e} tolerance={tolerance:.6e}")

        local_by_index = {index: local for local, index in enumerate(indices)}
        for index in sorted(exact_rows.intersection(indices)):
            local = local_by_index[index]
            expected = matvec(independent, np.asarray(tests[local]))
            observed = np.asarray(sketches[local])
            denominator = max(np.linalg.norm(expected), np.linalg.norm(observed),
                              np.finfo(float).tiny)
            error = float(np.linalg.norm(expected - observed) / denominator)
            exact_checked += 1
            if error > local_max:
                local_max, local_worst = error, index
            if error > tolerance:
                raise ValueError(f"validation FAILED sample={index} "
                                 f"relative_error={error:.6e} tolerance={tolerance:.6e}")
        if local_max > max_error:
            max_error, worst_row = local_max, local_worst
        batch_elapsed = time.perf_counter() - batch_started
        write_json_atomic(receipt_path, {
            "validation_version": validation_version,
            "rows": [row0, row1], "sha256": [m["sha256"] for m in manifests],
            "assembler": assembler, "tolerance": tolerance,
            "max_relative_error": local_max, "worst_row": local_worst,
            "batch_elapsed_seconds": batch_elapsed,
        })
        checked_new += len(indices)
        progress("validation", checked_prior + checked_new, total,
                 started=apply_started, new_done=checked_new,
                 new_total=pending_total,
                 detail=f"rows={row0}:{row1} "
                        f"batch_elapsed={batch_elapsed:.3f}s "
                        f"max_rel_error={max_error:.3e}")
    return {
        "checked_samples": total,
        "max_relative_error": max_error, "worst_row": worst_row,
        "tolerance": tolerance, "assembler": assembler,
        "reused_validated_samples": checked_prior,
        "validation_version": validation_version,
        "combination_group_size": group_size,
        "combinations_checked": combinations_checked,
        "individual_rows_planned": sorted(exact_rows),
        "individual_rows_checked_this_invocation": exact_checked,
        "operator_construction_seconds": build_seconds,
        "apply_seconds": time.perf_counter() - apply_started,
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_hdf5(batches: list[tuple[int, int]], batch_dir: Path,
               sampling_dir: Path, samples: int, dim: int) -> dict:
    """Write original 4096-column HDF5 format, using bounded extra memory."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("final cache assembly requires h5py") from exc
    started = time.perf_counter()
    parts = {}
    for source_key, name in (("y_test", "y_test_file"),
                             ("y_sketch", "y_sketch_file")):
        parts[name] = 0
        number_of_parts = (dim + BLOCK_COLS - 1) // BLOCK_COLS
        for part, col0 in enumerate(range(0, dim, BLOCK_COLS)):
            col1 = min(dim, col0 + BLOCK_COLS)
            width = col1 - col0
            # This layout is exactly Fortran flattening of (samples, width).
            block = np.empty((width, samples), dtype=np.float64)
            for row0, row1 in batches:
                source = np.load(batch_dir / batch_name(row0, row1) /
                                 f"{source_key}.npy", mmap_mode="r", allow_pickle=False)
                block[:, row0:row1] = source[:, col0:col1].T
            flat = block.ravel(order="C")
            path = sampling_dir / f"{name}.{part:05d}.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["shape"] = np.array((samples, dim), dtype=np.int64)
                handle.create_dataset("real", data=flat)
                handle.create_dataset("imag", shape=(samples * width,),
                                      dtype="float64", fillvalue=0.0)
            # Verify actual stored format and values before declaring success.
            with h5py.File(path, "r") as handle:
                if (tuple(handle.attrs["shape"]) != (samples, dim)
                        or set(handle.keys()) != {"real", "imag"}
                        or handle["real"].shape != (samples * width,)
                        or handle["imag"].shape != (samples * width,)):
                    raise RuntimeError(f"invalid HDF5 layout in {path}")
                for offset in range(0, flat.size, 1_000_000):
                    stop = min(offset + 1_000_000, flat.size)
                    if not np.array_equal(handle["real"][offset:stop], flat[offset:stop]):
                        raise RuntimeError(f"HDF5 real data mismatch: {path}")
                    if np.any(handle["imag"][offset:stop]):
                        raise RuntimeError(f"HDF5 imaginary data mismatch: {path}")
            parts[name] += 1
            log(f"cache: wrote {name} part={part+1}/{number_of_parts} "
                f"cols={col0}:{col1} elapsed={duration(time.perf_counter()-started)}")
    return {"shape": [samples, dim], "block_cols": BLOCK_COLS,
            "parts": parts, "force_complex": True,
            "write_and_verify_seconds": time.perf_counter() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("sphere",), default="sphere")
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="samples per task AND per validation update (default: 10)")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--assembler", choices=("dense", "exafmm", "kifmm"),
                        default="dense")
    parser.add_argument("--validation-assembler", choices=("dense", "exafmm", "kifmm"))
    parser.add_argument("--tolerance", type=float, default=5e-7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue a matching run from its .incomplete directory")
    args = parser.parse_args()
    if not np.isfinite(args.h) or args.h <= 0:
        parser.error("--h must be positive and finite")
    if args.samples <= 0 or args.workers <= 0 or args.batch_size <= 0 or args.seed < 0:
        parser.error("--samples, --workers, --batch-size must be positive; seed >= 0")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive and finite")

    name = (f"sphere_h_{args.h:.2e}_samples_{args.samples}_seed_{args.seed}_batch_{args.batch_size}")
    output = (args.output or Path("results/laplace_single_layer") / name).resolve()
    staging = output.with_name(output.name + ".incomplete")
    validation_assembler = args.validation_assembler or args.assembler
    config = {
        "format_version": 2, "geometry": args.geometry,
        "h": args.h, "samples": args.samples, "workers": args.workers,
        "batch_size": args.batch_size, "seed": args.seed,
        "assembler": args.assembler,
        "validation_assembler": validation_assembler, "tolerance": args.tolerance,
    }
    if output.exists():
        raise FileExistsError(f"completed output already exists: {output}")
    if args.resume:
        if not staging.is_dir():
            raise FileNotFoundError(f"no resumable run at: {staging}")
        previous = json.loads((staging / "config.json").read_text())
        if previous != config:
            raise ValueError("--resume arguments differ from existing config.json")
        if not (staging / "grid.msh").is_file():
            raise FileNotFoundError("resume requires saved grid.msh")
        log(f"resuming output={output}")
    else:
        if staging.exists():
            raise FileExistsError(f"unfinished run exists: {staging}; use --resume")
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        write_json_atomic(staging / "config.json", config)

    started = time.perf_counter()
    mesh_seconds = 0.0
    batches = list(batch_ranges(args.samples, args.batch_size))
    batch_dir = staging / "_batches"
    batch_dir.mkdir(exist_ok=True)
    log(f"start geometry=sphere h={args.h:g} samples={args.samples} "
        f"workers={args.workers} batch_size={args.batch_size} "
        f"assembler={args.assembler}")
    log(f"output={output}")
    try:
        mesh_path = staging / "grid.msh"
        if not args.resume:
            import bempp_cl.api as bempp
            log("mesh: generating unit sphere")
            mesh_started = time.perf_counter()
            grid = bempp.shapes.sphere(h=args.h)
            bempp.export(str(mesh_path), grid=grid)
            dim = int(bempp.function_space(grid, "DP", 0).global_dof_count)
            elements = int(grid.number_of_elements)
            mesh_seconds = time.perf_counter() - mesh_started
            write_json_atomic(staging / "mesh_info.json",
                              {"dimension": dim, "triangles": elements})
            log(f"mesh: elements={elements} dofs={dim} elapsed={mesh_seconds:.3f}s")
        else:
            info = json.loads((staging / "mesh_info.json").read_text())
            dim, elements = info["dimension"], info["triangles"]
            log(f"mesh: reusing grid.msh elements={elements} dofs={dim}")

        # A stopped task may leave a temporary batch directory; never trust it.
        for orphan in batch_dir.glob("batch_*.tmp.*"):
            if orphan.is_dir():
                shutil.rmtree(orphan)
        pending = []
        done_before = 0
        for row0, row1 in batches:
            folder = batch_dir / batch_name(row0, row1)
            if folder.is_dir():
                check_batch(folder, row0, row1, dim)
                done_before += row1 - row0
            else:
                pending.append((row0, row1))

        worker_config = {"mesh_path": str(mesh_path), "assembler": args.assembler,
                         "seed": args.seed, "dimension": dim,
                         "batch_dir": str(batch_dir), "batch_size": args.batch_size}
        sampling_seconds = sample_parallel(pending, args.workers, worker_config,
                                           args.samples, done_before)
        log("sampling: complete; beginning independent validation")
        validation = validate_batches(
            batches, batch_dir, staging / "_validation_receipts",
            str(mesh_path), validation_assembler, args.seed, dim, args.tolerance,
        )
        log("cache: writing multipart HDF5 Y samples")
        temporary_sampling = staging / "sampling.incomplete"
        if temporary_sampling.exists():
            shutil.rmtree(temporary_sampling)
        temporary_sampling.mkdir()
        cache = write_hdf5(batches, batch_dir, temporary_sampling,
                           args.samples, dim)
        final_sampling = staging / "sampling"
        if final_sampling.exists():
            shutil.rmtree(final_sampling)
        temporary_sampling.rename(final_sampling)
        total = time.perf_counter() - started
        report = {
            "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "geometry": "sphere", "h_requested": args.h,
            "samples": args.samples, "workers": args.workers,
            "batch_size": args.batch_size, "batches": len(batches),
            "seed": args.seed, "operator": "Laplace single-layer weak form",
            "discretization": "DP0", "dofs": dim, "triangles": elements,
            "generation_assembler": args.assembler,
            "validation": validation, "cache": cache,
            "resumed": bool(args.resume),
            "timings_seconds_this_invocation": {
                "mesh": mesh_seconds, "parallel_sampling": sampling_seconds,
                "validation": validation["elapsed_seconds"],
                "hdf5_write_and_verify": cache["write_and_verify_seconds"],
                "total": total,
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)  # Commit only after EVERY sample is validated.
        for temporary in (output / "_batches", output / "_validation_receipts"):
            try:
                shutil.rmtree(temporary)
            except OSError as exc:
                log(f"WARNING: could not remove intermediate {temporary}: {exc}")
        log(f"complete samples={args.samples} mesh={mesh_seconds:.3f}s "
            f"sampling={sampling_seconds:.3f}s "
            f"validation={validation['elapsed_seconds']:.3f}s "
            f"hdf5={cache['write_and_verify_seconds']:.3f}s "
            f"total={total:.3f}s max_rel_error={validation['max_relative_error']:.3e}")
        log(f"complete output={output} report={output / 'report.json'}")
        return 0
    except BaseException as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        log(f"FAILED: partial run retained at {staging} (retry with --resume)")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
