#!/usr/bin/env python3
"""Sample the four historical BIEGrid variants and test sample-fed RSRS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from so_kernels import BIEGridOperator, generate_samples
from bempp_rsrs import InsufficientSamplesError, RsrsOptions, RsrsSession, SampleBundle
from bempp_rsrs.config import ThreadOptions, TreeOptions


VARIANTS = (
    "real_symmetric",
    "real_nonsymmetric",
    "complex_symmetric",
    "complex_nonsymmetric",
)


def relative_norm(numerator, denominator, ord_value=2):
    return float(np.linalg.norm(numerator, ord=ord_value) / np.linalg.norm(denominator, ord=ord_value))


def run_case(variant: str, cells: int, sample_count: int, seed: int, rank: int) -> dict:
    operator = BIEGridOperator(
        cells_per_axis=cells,
        variant=variant,
        perturbation_scale=1.0e-2,
        perturbation_seed=12345,
    )
    symmetric = operator.symmetric
    started = time.perf_counter()
    generated = generate_samples(
        operator,
        transpose_operator=None if symmetric else operator.transpose_operator,
        sample_count=sample_count,
        seed=seed,
        dtype=operator.dtype,
    )
    sampling_seconds = time.perf_counter() - started
    samples = SampleBundle(
        generated.y_test,
        generated.y_sketch,
        generated.z_test,
        generated.z_sketch,
    )
    options = RsrsOptions(
        oversampling=8,
        oversampling_diag_blocks=8,
        min_num_samples=0,
        initial_num_samples=0,
        fixed_rank_sampling_mode="Constant",
        nonsymmetric_id_combination="Concat",
        tol_id=float(rank),
        min_rank=1,
        min_level=1,
        symmetry="Symmetric" if symmetric else "NoSymm",
        rank_picking="Min",
        save_samples=False,
        load_samples=True,
        tree=TreeOptions(max_tree_depth=4, leaf_size=8),
        threads=ThreadOptions(factorization_threads=1, apply_threads=1),
    )
    session = RsrsSession(operator.points, options, dtype=operator.dtype)
    started = time.perf_counter()
    try:
        factors = session.factorize(samples)
    except InsufficientSamplesError as error:
        return {
            "variant": variant,
            "dimension": operator.dimension,
            "dtype": str(operator.dtype),
            "status": "insufficient_samples",
            "provided_samples": sample_count,
            "sample_request": vars(error.request),
            "sampling_seconds": sampling_seconds,
        }
    factorization_seconds = time.perf_counter() - started

    dense = operator.dense()
    identity = np.eye(operator.dimension, dtype=operator.dtype)
    started = time.perf_counter()
    approximate = factors.matmat(identity)
    approximate_transpose = factors.transpose_matmat(identity)
    apply_seconds = time.perf_counter() - started

    rng = np.random.default_rng(seed + 1000)
    rhs = rng.standard_normal((operator.dimension, 3)).astype(operator.dtype)
    if np.issubdtype(operator.dtype, np.complexfloating):
        rhs += 1j * rng.standard_normal(rhs.shape)
    exact_solution = np.linalg.solve(dense, rhs)
    approximate_solution = factors.inverse_matmat(rhs)
    exact_transpose_solution = np.linalg.solve(dense.T, rhs)
    approximate_transpose_solution = factors.inverse_transpose_matmat(rhs)

    return {
        "variant": variant,
        "dimension": operator.dimension,
        "dtype": str(operator.dtype),
        "status": "complete",
        "provided_samples": sample_count,
        "condition_number_2": float(np.linalg.cond(dense)),
        "sampling_seconds": sampling_seconds,
        "factorization_seconds": factorization_seconds,
        "factor_apply_seconds": apply_seconds,
        "relative_factor_2": relative_norm(approximate - dense, dense, 2),
        "relative_factor_fro": relative_norm(approximate - dense, dense, "fro"),
        "relative_transpose_2": relative_norm(approximate_transpose - dense.T, dense.T, 2),
        "relative_solve_2": relative_norm(approximate_solution - exact_solution, exact_solution, 2),
        "relative_residual_2": relative_norm(dense @ approximate_solution - rhs, rhs, 2),
        "relative_transpose_solve_2": relative_norm(
            approximate_transpose_solution - exact_transpose_solution,
            exact_transpose_solution,
            2,
        ),
        "relative_transpose_residual_2": relative_norm(
            dense.T @ approximate_transpose_solution - rhs,
            rhs,
            2,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = {
        "configuration": {
            "cells_per_axis": args.cells,
            "dimension": (args.cells + 1) ** 2,
            "sample_count": args.samples,
            "sample_distribution": "Gaussian",
            "sample_seed": args.seed,
            "perturbation_seed": 12345,
            "perturbation_scale": 1.0e-2,
            "fixed_rank": args.rank,
        },
        "cases": [],
    }
    for variant in VARIANTS:
        print(f"[biegrid-rsrs] starting {variant}", flush=True)
        result = run_case(variant, args.cells, args.samples, args.seed, args.rank)
        report["cases"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[biegrid-rsrs] {variant}: {result['status']}", flush=True)
        if result["status"] != "complete":
            print(json.dumps(result["sample_request"], indent=2), flush=True)
            return 2

    lines = [
        "# Small sample-fed BIEGrid RSRS test",
        "",
        "| Variant | rel. factor 2 | rel. factor Fro. | solve error | residual | transpose solve |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['variant']} | {row['relative_factor_2']:.3e} | "
            f"{row['relative_factor_fro']:.3e} | {row['relative_solve_2']:.3e} | "
            f"{row['relative_residual_2']:.3e} | {row['relative_transpose_solve_2']:.3e} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
