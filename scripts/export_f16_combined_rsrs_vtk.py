#!/usr/bin/env python3
"""Export VTK files for a saved F16 Helmholtz-combined RSRS run.

This script rebuilds the matching ``RSRSBenchmarkConfig``, resolves the
corresponding ``solve_vectors_*.h5`` file from ``rsrs_config``, and exports:

1. ``geometry.vtk``
2. ``surface_density_XXX.vtk`` for each selected source
3. optionally ``field_<kind>_XXX.vtu`` for each selected source
4. optionally ``field_<kind>_sum.vtu`` for the coherent sum over the selected sources

Notes
-----
- If you omit ``--source-index``, all stored sources are exported.
- The summed field is the complex coherent sum of the selected fields.
- For ``field-kind total``, this sums total fields.
- For ``field-kind scattered``, this sums scattered fields.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import meshio
import numpy as np

import bempp_cl.api as bempp


def build_config(
    tol: float,
    n_sources: int,
    h: float,
    kappa: float,
    max_tree_depth: int,
    num_threads: int,
):
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
        tol_ext_near=1e-8,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rsrs-exps-root",
        type=Path,
        default=None,
        help="Path to the rsrs-exps checkout containing the saved results. Default: infer it from the script location.",
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance whose saved vectors should be exported.")
    parser.add_argument("--h", type=float, default=0.02, help="Mesh width used in the saved run.")
    parser.add_argument("--kappa", type=float, default=4.0, help="Wavenumber used in the saved run.")
    parser.add_argument("--max-tree-depth", type=int, default=16, help="max_tree_depth used in the saved run.")
    parser.add_argument("--num-threads", type=int, default=56, help="num_threads used in the saved run.")
    parser.add_argument(
        "--sol-kind",
        choices=("prec", "no_prec", "approx_inv"),
        default="prec",
        help="Which saved solution family to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Default: next to solve_vectors_*.h5.",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        action="append",
        default=None,
        help="Specific source index or indices to export. Default: export all rows.",
    )
    parser.add_argument(
        "--n-sources",
        type=int,
        default=3,
        help="Number of incident directions used in the saved run. Set this to 1 for one-source results.",
    )
    parser.add_argument(
        "--export-field-slice",
        action="store_true",
        help="Also export a sampled scattered/total field slice.",
    )
    parser.add_argument(
        "--field-kind",
        choices=("scattered", "total"),
        default="total",
        help="Which field to export on the slice.",
    )
    parser.add_argument(
        "--plane",
        choices=("xy", "xz", "yz"),
        default="xy",
        help="Slice orientation. 'xy' fixes z, 'xz' fixes y, and 'yz' fixes x.",
    )
    parser.add_argument(
        "--plane-offset",
        "--slice-z",
        dest="plane_offset",
        type=float,
        default=0.0,
        help="Fixed coordinate normal to the slice plane.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=-3.0,
        help="First in-plane coordinate minimum. For xy/xz this is x; for yz this is y.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=3.0,
        help="First in-plane coordinate maximum. For xy/xz this is x; for yz this is y.",
    )
    parser.add_argument(
        "--ymin",
        "--zmin",
        dest="ymin",
        type=float,
        default=-3.0,
        help="Second in-plane coordinate minimum. For xy this is y; for xz/yz this is z.",
    )
    parser.add_argument(
        "--ymax",
        "--zmax",
        dest="ymax",
        type=float,
        default=3.0,
        help="Second in-plane coordinate maximum. For xy this is y; for xz/yz this is z.",
    )
    parser.add_argument("--nx", type=int, default=200, help="Number of first-axis sample points.")
    parser.add_argument("--ny", "--nz", dest="ny", type=int, default=200, help="Number of second-axis sample points.")
    return parser.parse_args()


def _add_rsrs_paths(rsrs_exps_root: Path) -> None:
    root = rsrs_exps_root.expanduser().resolve()
    for candidate in (root, root / "python"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def load_rsrs_helpers(rsrs_exps_root: Path):
    _add_rsrs_paths(rsrs_exps_root)
    import geometry as rsrs_geometry
    import right_hand_sides as rsrs_rhs
    return rsrs_geometry, rsrs_rhs


def reconstruct_grid(grid_file: Path | None, rsrs_exps_root: Path, geometry_type: str, dim_param: float):
    if grid_file is not None and grid_file.exists():
        return bempp.import_grid(str(grid_file))

    rsrs_geometry, _ = load_rsrs_helpers(rsrs_exps_root)
    grid = rsrs_geometry.get_geometry(geometry_type.lower(), dim_param)
    if not hasattr(grid, "vertices") or not hasattr(grid, "elements"):
        raise TypeError(
            f"geometry '{geometry_type}' with dim_param={dim_param} did not return a Bempp grid."
        )
    return grid


def reconstruct_directions(rsrs_exps_root: Path, n_sources: int) -> np.ndarray:
    _, rsrs_rhs = load_rsrs_helpers(rsrs_exps_root)
    return np.asarray(rsrs_rhs.generate_directions(n_sources), dtype=np.float64)


def load_solve_vectors(path: Path, group_name: str) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as h5f:
        if group_name == "auto":
            if "sols_prec" in h5f:
                group_name = "sols_prec"
            elif "sols_approx_inv" in h5f:
                group_name = "sols_approx_inv"
            elif "sols_no_prec" in h5f:
                group_name = "sols_no_prec"
            else:
                raise KeyError(
                    "None of 'sols_prec', 'sols_approx_inv', or 'sols_no_prec' were found in the HDF5 file."
                )
        group = h5f[group_name]
        if "real" not in group:
            raise KeyError(f"Group '{group_name}' does not contain a 'real' dataset.")
        coeffs = np.asarray(group["real"][()])
        if "imag" in group:
            coeffs = coeffs + 1j * np.asarray(group["imag"][()])
    if coeffs.ndim != 2:
        raise ValueError(f"Expected a rank-2 coefficient matrix, found shape {coeffs.shape}.")
    return coeffs, group_name


def select_source_indices(source_index: list[int] | None, n_rows: int) -> list[int]:
    if source_index is None:
        return list(range(n_rows))
    bad = [idx for idx in source_index if idx < 0 or idx >= n_rows]
    if bad:
        raise IndexError(f"Source indices out of range for {n_rows} stored vectors: {bad}")
    return source_index


def make_plane_mesh(
    plane: str,
    coord1_min: float,
    coord1_max: float,
    coord2_min: float,
    coord2_max: float,
    offset: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]], np.ndarray]:
    xs = np.linspace(coord1_min, coord1_max, nx)
    ys = np.linspace(coord2_min, coord2_max, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    if plane == "xy":
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, offset)))
        points_3xn = np.vstack((xx.ravel(), yy.ravel(), np.full(xx.size, offset)))
    elif plane == "xz":
        points = np.column_stack((xx.ravel(), np.full(xx.size, offset), yy.ravel()))
        points_3xn = np.vstack((xx.ravel(), np.full(xx.size, offset), yy.ravel()))
    else:
        points = np.column_stack((np.full(xx.size, offset), xx.ravel(), yy.ravel()))
        points_3xn = np.vstack((np.full(xx.size, offset), xx.ravel(), yy.ravel()))

    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n0 = i * ny + j
            n1 = (i + 1) * ny + j
            n2 = i * ny + (j + 1)
            n3 = (i + 1) * ny + (j + 1)
            triangles.append([n0, n1, n2])
            triangles.append([n1, n3, n2])

    cells = [("triangle", np.asarray(triangles, dtype=np.int32))]
    return points, cells, points_3xn


def evaluate_combined_field(
    slp_pot,
    space,
    coefficients: np.ndarray,
    kappa: float,
    direction: np.ndarray,
    points_3xn: np.ndarray,
    field_kind: str,
) -> np.ndarray:
    grid_fun = bempp.GridFunction(space, coefficients=coefficients)
    scattered = -np.asarray(slp_pot.evaluate(grid_fun)).reshape(-1)

    if field_kind == "scattered":
        return scattered

    incident = np.exp(1j * kappa * (direction @ points_3xn))
    return incident + scattered


def export_field_slice(
    out_path: Path,
    values: np.ndarray,
    points: np.ndarray,
    cells: list[tuple[str, np.ndarray]],
) -> None:
    point_data = {
        "real": np.real(values),
        "imag": np.imag(values),
        "abs": np.abs(values),
    }
    meshio.write_points_cells(str(out_path), points, cells, point_data=point_data)


def export_surface_density(out_path: Path, grid, grid_fun) -> None:
    data = np.asarray(grid_fun.evaluate_on_element_centers()).T
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]

    cell_data = {"domain_index": [grid.domain_indices.astype(np.int32)]}
    if np.iscomplexobj(data):
        cell_data["real"] = [np.real(data)]
        cell_data["imag"] = [np.imag(data)]
        cell_data["abs"] = [np.abs(data)]
    else:
        cell_data["data"] = [data]

    meshio.write_points_cells(
        str(out_path),
        grid.vertices.T,
        [("triangle", grid.elements.T.astype(np.int32))],
        cell_data=cell_data,
    )


def export_geometry(out_path: Path, grid) -> None:
    meshio.write_points_cells(
        str(out_path),
        grid.vertices.T,
        [("triangle", grid.elements.T.astype(np.int32))],
    )


def main() -> None:
    args = parse_args()

    if args.rsrs_exps_root is None:
        rsrs_exps_root = Path(__file__).resolve().parents[1]
    else:
        rsrs_exps_root = args.rsrs_exps_root.expanduser().resolve()

    _add_rsrs_paths(rsrs_exps_root)
    os.chdir(rsrs_exps_root)

    config = build_config(
        tol=args.tol,
        n_sources=args.n_sources,
        h=args.h,
        kappa=args.kappa,
        max_tree_depth=args.max_tree_depth,
        num_threads=args.num_threads,
    )

    solve_vectors_path = config.solve_vectors_path(tol=args.tol)
    if solve_vectors_path is None:
        raise FileNotFoundError("Could not resolve a solve_vectors path from the configuration.")

    group_map = {
        "prec": "sols_prec",
        "no_prec": "sols_no_prec",
        "approx_inv": "sols_approx_inv",
    }

    export_args = argparse.Namespace(
        solve_vectors=solve_vectors_path,
        group=group_map[args.sol_kind],
        output_dir=args.output_dir.expanduser() if args.output_dir is not None else None,
        source_index=args.source_index,
        grid_file=config.grid_path(),
        rsrs_exps_root=rsrs_exps_root,
        geometry_type="f16",
        dim_param=args.h,
        kappa=args.kappa,
        n_sources=args.n_sources,
        export_field_slice=args.export_field_slice,
        field_kind=args.field_kind,
        plane=args.plane,
        plane_offset=args.plane_offset,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        nx=args.nx,
        ny=args.ny,
    )

    solve_vectors_path = export_args.solve_vectors.expanduser().resolve()
    if not solve_vectors_path.exists():
        raise FileNotFoundError(f"solve-vectors file not found: {solve_vectors_path}")

    output_dir = export_args.output_dir
    if output_dir is None:
        output_dir = solve_vectors_path.with_suffix("")
        output_dir = output_dir.parent / f"{output_dir.name}_{args.sol_kind}_vtk"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coeff_matrix, used_group = load_solve_vectors(solve_vectors_path, export_args.group)
    grid = reconstruct_grid(
        export_args.grid_file,
        rsrs_exps_root,
        export_args.geometry_type,
        export_args.dim_param,
    )
    space = bempp.function_space(grid, "DP", 0)

    if coeff_matrix.shape[1] != space.global_dof_count:
        raise ValueError(
            "Stored vector length does not match the reconstructed DP0 space: "
            f"{coeff_matrix.shape[1]} vs {space.global_dof_count}."
        )

    source_indices = select_source_indices(export_args.source_index, coeff_matrix.shape[0])

    directions = None
    slice_points = None
    slice_cells = None
    slice_points_3xn = None
    slp_pot = None

    if export_args.export_field_slice:
        directions = reconstruct_directions(rsrs_exps_root, export_args.n_sources)
        if len(directions) < coeff_matrix.shape[0]:
            raise ValueError(
                f"Need at least {coeff_matrix.shape[0]} directions, but only {len(directions)} were reconstructed."
            )

        slice_points, slice_cells, slice_points_3xn = make_plane_mesh(
            export_args.plane,
            export_args.xmin,
            export_args.xmax,
            export_args.ymin,
            export_args.ymax,
            export_args.plane_offset,
            export_args.nx,
            export_args.ny,
        )

        print(
            f"Building Helmholtz potential operator on {export_args.nx * export_args.ny} slice points "
            f"with assembler='dense'..."
        )
        slp_pot = bempp.operators.potential.helmholtz.single_layer(
            space,
            slice_points_3xn,
            export_args.kappa,
            assembler="dense",
        )
        print("Finished building potential operator.")

    print(f"Using results folder: {config.results_path()}")
    print(f"Loaded {coeff_matrix.shape[0]} solve vector(s) from group '{used_group}'.")
    print(
        f"Reconstructed grid with {grid.number_of_elements} elements and "
        f"DP0 space with {space.global_dof_count} dofs."
    )

    export_geometry(output_dir / "geometry.vtk", grid)

    field_sum = None

    for idx in source_indices:
        coeffs = coeff_matrix[idx]
        grid_fun = bempp.GridFunction(space, coefficients=coeffs)

        surface_out = output_dir / f"surface_density_{idx:03d}.vtk"
        export_surface_density(surface_out, grid, grid_fun)
        print(f"Wrote {surface_out}")

        if export_args.export_field_slice:
            print(f"Evaluating field for source {idx}...")
            field_values = evaluate_combined_field(
                slp_pot,
                space,
                coeffs,
                export_args.kappa,
                directions[idx],
                slice_points_3xn,
                export_args.field_kind,
            )
            print(f"Finished field for source {idx}.")

            field_out = output_dir / f"field_{export_args.field_kind}_{idx:03d}.vtu"
            export_field_slice(field_out, field_values, slice_points, slice_cells)
            print(f"Wrote {field_out}")

            if field_sum is None:
                field_sum = field_values.copy()
            else:
                field_sum += field_values

    if export_args.export_field_slice and field_sum is not None:
        print("Writing coherent summed field...")
        sum_out = output_dir / f"field_{export_args.field_kind}_sum.vtu"
        export_field_slice(sum_out, field_sum, slice_points, slice_cells)
        print(f"Wrote {sum_out}")

    print(f"All VTK outputs written to: {output_dir}")


if __name__ == "__main__":
    main()