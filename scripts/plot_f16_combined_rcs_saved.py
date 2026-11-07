#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_directions(n_dirs: int) -> np.ndarray:
    if n_dirs <= 1:
        return np.array([[1.0, 0.0, 0.0]], dtype=float)

    indices = np.arange(0, n_dirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_dirs)
    theta = np.pi * (1 + 5**0.5) * indices

    dirs = np.stack(
        (
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ),
        axis=1,
    )
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return dirs


def build_config(tol: float, n_sources: int, num_threads: int, h: float, kappa: float, max_tree_depth: int):
    from rsrs_config import RSRSBenchmarkConfig

    return RSRSBenchmarkConfig(
        operator_type=19,                  # BemppClHelmholtzCombined
        precision=1,                       # Double
        id_tols=[tol],
        dim_arg_type=1,                    # KappaAndMeshwidth
        geometry=6,                        # F16
        h=h,
        kappa=kappa,
        max_tree_depth=max_tree_depth,
        n_sources=n_sources,
        assembler=1,                       # ExaFMM
        oversampling_near=8,
        oversampling_diag=16,
        min_num_samples=0,
        initial_num_samples=0,
        fixed_rank_sampling_mode=0,        # PerLevel
        nonsymmetric_id_combination=1,     # Concat
        null_method=0,                     # Projection
        block_extraction_method=0,         # LuLstSq
        pivot_method=2,                    # LuHybrid
        rrqr=0,                            # RRQR
        min_rank=1,
        min_level=1,
        tol_ext_near=1e-10,
        tol_ext_diag=1e-16,
        fact_type=1,                       # Joint
        symmetry=0,                        # NoSymm
        rank_picking=0,                    # Min
        save_samples=True,
        load_samples=True,
        num_threads=num_threads,
        solve=True,
        solve_tol=1e-10,
        plot=True,
        dense_errors=False,
        factors_cn=False,
        results_output=0,                  # All
    )


def plane_to_points(plane: str, angles: np.ndarray) -> np.ndarray:
    z = np.zeros_like(angles)

    if plane == "xy":
        return np.vstack([-np.cos(angles), -np.sin(angles), z])
    if plane == "xz":
        return np.vstack([-np.cos(angles), z, -np.sin(angles)])
    if plane == "yz":
        return np.vstack([z, -np.cos(angles), -np.sin(angles)])

    raise ValueError(f"Unknown plane '{plane}'")


def rcs_db(field: np.ndarray) -> np.ndarray:
    power = 4.0 * np.pi * np.abs(field) ** 2
    power = np.maximum(power, np.finfo(np.float64).tiny)
    return 10.0 * np.log10(power)


def closed_curve(angles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([angles, [angles[0]]]), np.concatenate([values, [values[0]]])


def write_directions(out_path: Path, directions: np.ndarray) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for i, d in enumerate(directions):
            fh.write(f"source_{i:03d}: {d[0]: .16e} {d[1]: .16e} {d[2]: .16e}\n")


def load_sols_from_h5(vector_path: Path, sol_kind: str) -> list[np.ndarray]:
    import h5py

    group_name = f"sols_{sol_kind}"
    with h5py.File(vector_path, "r") as h5f:
        if group_name not in h5f:
            raise KeyError(f"Solution group '{group_name}' not found in {vector_path}")

        group = h5f[group_name]
        real = np.asarray(group["real"])
        if "imag" in group:
            data = real + 1j * np.asarray(group["imag"])
        else:
            data = real

    return [data[i].copy() for i in range(data.shape[0])]


def write_csv(
    out_path: Path,
    angles: np.ndarray,
    source_fields: list[np.ndarray],
    sum_field: np.ndarray,
) -> None:
    header = ["angle_rad", "angle_deg"]
    for i in range(len(source_fields)):
        header.extend(
            [
                f"source_{i:03d}_real",
                f"source_{i:03d}_imag",
                f"source_{i:03d}_rcs_db",
            ]
        )
    header.extend(["sum_real", "sum_imag", "sum_rcs_db"])

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        source_db = [rcs_db(f) for f in source_fields]
        sum_db = rcs_db(sum_field)

        for j, angle in enumerate(angles):
            row = [angle, np.degrees(angle)]
            for i in range(len(source_fields)):
                row.extend(
                    [
                        np.real(source_fields[i][j]),
                        np.imag(source_fields[i][j]),
                        source_db[i][j],
                    ]
                )
            row.extend([np.real(sum_field[j]), np.imag(sum_field[j]), sum_db[j]])
            writer.writerow(row)


def plot_plane(
    out_path: Path,
    plane: str,
    angles: np.ndarray,
    source_fields: list[np.ndarray],
    polar: bool,
) -> None:
    db_curves = [rcs_db(f) for f in source_fields]
    sum_field = np.sum(source_fields, axis=0)
    sum_db = rcs_db(sum_field)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(source_fields), 3)))
    linestyles = ["-", "--", ":", "-."]

    if polar:
        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = plt.figure(figsize=(9.0, 5.5))
        ax = fig.add_subplot(111)

    for i, db in enumerate(db_curves):
        label = f"Source {i}"
        ls = linestyles[i % len(linestyles)]
        color = colors[i % len(colors)]

        if polar:
            a_plot, db_plot = closed_curve(angles, db)
            ax.plot(a_plot, db_plot, lw=1.8, ls=ls, color=color, label=label)
        else:
            ax.plot(np.degrees(angles), db, lw=1.8, ls=ls, color=color, label=label)

    if polar:
        a_plot, db_plot = closed_curve(angles, sum_db)
        ax.plot(a_plot, db_plot, lw=2.8, color="black", label="Coherent sum")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"RCS ({plane.upper()} plane)", va="bottom")
    else:
        ax.plot(np.degrees(angles), sum_db, lw=2.8, color="black", label="Coherent sum")
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("RCS (dB)")
        ax.set_title(f"RCS ({plane.upper()} plane)")
        ax.grid(True, alpha=0.3)

    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-source RCS for the saved F16 Helmholtz-combined run."
    )
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--sol-kind",
        choices=["prec", "no_prec", "approx_inv"],
        default="prec",
    )
    parser.add_argument(
        "--planes",
        nargs="+",
        choices=["xy", "xz", "yz"],
        default=["xy", "xz"],
        help="Angular far-field cuts to plot.",
    )
    parser.add_argument("--angles", type=int, default=400)
    parser.add_argument("--n-sources", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=56)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--max-tree-depth", type=int, default=16)
    parser.add_argument("--cartesian", action="store_true")
    parser.add_argument(
        "--solve-vectors",
        type=Path,
        default=None,
        help="Load solutions directly from this solve_vectors_*.h5 file.",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=None,
        help="Use this grid.msh instead of locating or relocating a generated grid.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    import bempp_cl.api

    config = None
    if args.solve_vectors is not None:
        vector_path = args.solve_vectors.expanduser().resolve()
        if not vector_path.exists():
            raise FileNotFoundError(f"Could not find solve vectors at {vector_path}")
        sols = load_sols_from_h5(vector_path, args.sol_kind)
        scenario_dir = vector_path.parent
    else:
        config = build_config(
            tol=args.tol,
            n_sources=args.n_sources,
            num_threads=args.num_threads,
            h=args.h,
            kappa=args.kappa,
            max_tree_depth=args.max_tree_depth,
        )
        sols = config.load_sols(args.tol, sol_kind=args.sol_kind)
        scenario_dir = config.results_path()

    if len(sols) == 0:
        raise RuntimeError("No saved solutions were found for this configuration.")

    if len(sols) != args.n_sources:
        print(
            f"Warning: loaded {len(sols)} solution vector(s), "
            f"but --n-sources={args.n_sources}. Proceeding with loaded solutions."
        )

    scenario_dir.mkdir(parents=True, exist_ok=True)

    tol_tag = f"{args.tol:.0e}".replace("+0", "").replace("+", "")
    out_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else scenario_dir / f"rcs_multisource_{args.sol_kind}_{tol_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid is not None:
        grid_path = args.grid.expanduser().resolve()
    elif config is not None:
        grid_path = (
            repo_root
            / "results"
            / config.generate_folder_name()
            / config.generate_sub_folder_name()
            / "grid.msh"
        )
        if not grid_path.exists():
            config.relocate_grid()
    else:
        grid_path = scenario_dir / "grid.msh"

    if not grid_path.exists():
        raise FileNotFoundError(f"Could not find grid.msh at {grid_path}")

    print(f"Using results folder: {scenario_dir}")
    print(f"Using grid: {grid_path}")
    print(f"Writing RCS outputs to: {out_dir}")

    directions = generate_directions(len(sols))
    for i, d in enumerate(directions):
        print(f"source {i}: d = [{d[0]: .6f}, {d[1]: .6f}, {d[2]: .6f}]")
    write_directions(out_dir / "source_directions.txt", directions)

    grid = bempp_cl.api.import_grid(str(grid_path))
    space = bempp_cl.api.function_space(grid, "DP", 0)
    dual_space = space

    angles = np.linspace(0.0, 2.0 * np.pi, args.angles, endpoint=False)

    for plane in args.planes:
        points = plane_to_points(plane, angles)
        ff_op = bempp_cl.api.operators.far_field.helmholtz.single_layer(
            space, points, args.kappa
        )

        source_fields = []
        for sol in sols:
            grid_fun = bempp_cl.api.GridFunction(
                space, projections=np.asarray(sol), dual_space=dual_space
            )
            field = np.asarray(ff_op * grid_fun).reshape(-1)
            source_fields.append(field)

        sum_field = np.sum(source_fields, axis=0)

        csv_path = out_dir / f"rcs_{plane}_{args.sol_kind}.csv"
        png_path = out_dir / f"rcs_{plane}_{args.sol_kind}.png"

        write_csv(csv_path, angles, source_fields, sum_field)
        plot_plane(
            png_path,
            plane,
            angles,
            source_fields,
            polar=not args.cartesian,
        )

        print(f"Wrote {png_path}")
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
