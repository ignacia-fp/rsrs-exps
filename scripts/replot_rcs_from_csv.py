#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


SOURCE_STYLES = ["-", "--", ":"]
SOURCE_COLORS = ["tab:blue", "tab:brown", "tab:cyan", "tab:orange", "tab:green"]


def load_rcs_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    names = data.dtype.names or ()
    theta = np.asarray(data["angle_rad"], dtype=float)

    pat = re.compile(r"source_(\d{3})_real$")
    source_ids = sorted(int(m.group(1)) for name in names if (m := pat.match(name)))

    source_fields = []
    for sid in source_ids:
        tag = f"source_{sid:03d}"
        field = np.asarray(data[f"{tag}_real"], dtype=float) + 1j * np.asarray(
            data[f"{tag}_imag"], dtype=float
        )
        source_fields.append(field)

    return theta, source_ids, source_fields


def rcs_db_from_field(field: np.ndarray) -> np.ndarray:
    power = np.abs(field) ** 2
    power = np.maximum(power, np.finfo(float).tiny)
    return 10.0 * np.log10(4.0 * np.pi * power)


def aggregate_rcs_db(source_fields: list[np.ndarray], mode: str) -> np.ndarray:
    fields = np.vstack(source_fields)

    if mode == "coherent":
        field = np.sum(fields, axis=0)
        power = np.abs(field) ** 2
    elif mode == "power_sum":
        power = np.sum(np.abs(fields) ** 2, axis=0)
    elif mode == "power_avg":
        power = np.mean(np.abs(fields) ** 2, axis=0)
    else:
        raise ValueError(f"Unknown aggregate mode: {mode}")

    power = np.maximum(power, np.finfo(float).tiny)
    return 10.0 * np.log10(4.0 * np.pi * power)


def close_polar_curve(theta: np.ndarray, values: np.ndarray):
    return np.concatenate([theta, [theta[0]]]), np.concatenate([values, [values[0]]])


def setup_polar(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


def infer_plane_from_csv_name(csv_path: Path) -> str | None:
    m = re.search(r"rcs_(xy|xz|yz)_", csv_path.name.lower())
    return m.group(1) if m else None


def infer_mesh_path(csv_path: Path) -> Path | None:
    candidate = csv_path.parent.parent / "grid.msh"
    if candidate.exists():
        return candidate
    return None


def _rot2(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def load_projected_mesh_triangles(
    mesh_path: Path,
    view: str,
    nose_polar_deg: float,
) -> np.ndarray:
    mesh = meshio.read(str(mesh_path))

    tri_blocks = [block.data for block in mesh.cells if block.type == "triangle"]
    if not tri_blocks:
        raise ValueError(f"No triangle cells found in mesh: {mesh_path}")
    tri = np.vstack(tri_blocks)

    pts3 = np.asarray(mesh.points, dtype=float)[:, :3]

    if view == "xy":
        pts2 = pts3[:, [0, 1]]
    elif view == "xz":
        pts2 = pts3[:, [0, 2]]
    elif view == "yz":
        pts2 = pts3[:, [1, 2]]
    else:
        raise ValueError("view must be one of: 'xy', 'xz', 'yz'")

    pts2 = pts2 - pts2.mean(axis=0, keepdims=True)

    scale = max(np.ptp(pts2[:, 0]), np.ptp(pts2[:, 1]))
    if scale <= 0.0:
        scale = 1.0
    pts2 = pts2 / scale

    target_cart_deg = 90.0 - nose_polar_deg
    rot = _rot2(np.deg2rad(target_cart_deg))
    pts2 = pts2 @ rot.T

    return pts2[tri]


def add_mesh_silhouette(
    ax,
    mesh_path: Path,
    view: str,
    nose_polar_deg: float = 180.0,
    facecolor: str = "black",
    alpha: float = 0.18,
    inset=(0.395, 0.395, 0.21, 0.21),
):
    tri_pts = load_projected_mesh_triangles(mesh_path, view, nose_polar_deg)

    ax_in = ax.inset_axes(inset, transform=ax.transAxes)
    ax_in.set_aspect("equal")
    ax_in.axis("off")

    coll = PolyCollection(
        tri_pts,
        facecolors=facecolor,
        edgecolors="none",
        linewidths=0.0,
        alpha=alpha,
        rasterized=True,
        antialiaseds=False,
    )
    ax_in.add_collection(coll)

    xy = tri_pts.reshape(-1, 2)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    pad = 0.08 * max(xmax - xmin, ymax - ymin)

    ax_in.set_xlim(xmin - pad, xmax + pad)
    ax_in.set_ylim(ymin - pad, ymax + pad)


def project_direction_to_plane(
    direction: np.ndarray,
    view: str,
    mode: str = "incoming",
):
    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.size != 3:
        raise ValueError(f"Direction must have 3 entries, got {d.size}")

    vec = -d if mode == "incoming" else d

    if view == "xy":
        proj = np.array([vec[0], vec[1]], dtype=float)
        axis_labels = ("x", "y")
    elif view == "xz":
        proj = np.array([vec[0], vec[2]], dtype=float)
        axis_labels = ("x", "z")
    elif view == "yz":
        proj = np.array([vec[1], vec[2]], dtype=float)
        axis_labels = ("y", "z")
    else:
        raise ValueError("view must be one of: 'xy', 'xz', 'yz'")

    norm = float(np.linalg.norm(proj))
    if norm > 0.0:
        proj = proj / norm

    return proj, axis_labels, norm


def add_direction_projection_inset(
    ax,
    directions: list[np.ndarray],
    source_ids: list[int],
    view: str,
    mode: str = "incoming",
    inset=(0.075, 0.74, 0.19, 0.19),
    colors: list[str] | None = None,
):
    if not directions:
        return

    ax_in = ax.inset_axes(inset, transform=ax.transAxes)
    ax_in.set_aspect("equal")
    ax_in.axis("off")

    circ = plt.Circle(
        (0.0, 0.0),
        1.0,
        edgecolor="0.75",
        facecolor="white",
        lw=0.8,
        alpha=0.9,
    )
    ax_in.add_patch(circ)
    ax_in.axhline(0.0, color="0.88", lw=0.8, zorder=0)
    ax_in.axvline(0.0, color="0.88", lw=0.8, zorder=0)

    colors = colors or SOURCE_COLORS
    title = "Inc. dir." if mode == "incoming" else "Prop. dir."

    out_of_plane = []
    axis_labels = None

    n = min(len(directions), len(source_ids))
    for i in range(n):
        sid = source_ids[i]
        color = colors[i % len(colors)]

        proj, axis_labels, proj_norm = project_direction_to_plane(
            directions[i], view, mode=mode
        )

        if proj_norm < 1.0e-14:
            out_of_plane.append((sid, color))
            continue

        end = 0.82 * proj
        ax_in.arrow(
            0.0,
            0.0,
            end[0],
            end[1],
            width=0.025,
            head_width=0.14,
            head_length=0.14,
            length_includes_head=True,
            color=color,
            alpha=0.95,
            zorder=3,
        )

        label_pos = 0.98 * proj
        ax_in.text(
            label_pos[0],
            label_pos[1],
            str(sid),
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )

    if out_of_plane:
        y0 = -0.78
        for k, (sid, color) in enumerate(out_of_plane):
            yy = y0 + 0.18 * k
            ax_in.plot(0.0, yy, marker="o", ms=5, mec=color, mfc="white", mew=1.4)
            ax_in.text(0.12, yy, f"{sid}", color=color, fontsize=8, ha="left", va="center")

    ax_in.set_xlim(-1.12, 1.12)
    ax_in.set_ylim(-1.12, 1.12)
    ax_in.set_title(title, fontsize=8, pad=2)

    if axis_labels is not None:
        ax_in.text(1.04, -0.03, axis_labels[0], fontsize=8, ha="left", va="center")
        ax_in.text(0.03, 1.04, axis_labels[1], fontsize=8, ha="left", va="bottom")


def plot_sources(
    out_path: Path,
    theta: np.ndarray,
    source_ids: list[int],
    source_fields: list[np.ndarray],
    plane_label: str,
    plot_view: str,
    mesh_path: Path | None,
    silhouette_view: str,
    nose_angle_deg: float,
    directions: list[np.ndarray] | None = None,
    direction_mode: str = "incoming",
    direction_inset=(0.075, 0.74, 0.19, 0.19),
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    setup_polar(ax)

    for i, (sid, field) in enumerate(zip(source_ids, source_fields)):
        db = rcs_db_from_field(field)
        th, rr = close_polar_curve(theta, db)
        ax.plot(
            th,
            rr,
            lw=2.0,
            ls=SOURCE_STYLES[i % len(SOURCE_STYLES)],
            color=SOURCE_COLORS[i % len(SOURCE_COLORS)],
            label=f"Source {sid}",
        )

    ax.set_title(f"RCS by Source ({plane_label} plane)", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0))

    if mesh_path is not None:
        add_mesh_silhouette(
            ax,
            mesh_path=mesh_path,
            view=silhouette_view,
            nose_polar_deg=nose_angle_deg,
            alpha=0.18,
        )

    if directions:
        add_direction_projection_inset(
            ax,
            directions=directions,
            source_ids=source_ids,
            view=plot_view,
            mode=direction_mode,
            inset=direction_inset,
            colors=SOURCE_COLORS,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    out_path: Path,
    theta: np.ndarray,
    source_ids: list[int],
    source_fields: list[np.ndarray],
    plane_label: str,
    plot_view: str,
    mode: str,
    label: str,
    mesh_path: Path | None,
    silhouette_view: str,
    nose_angle_deg: float,
    directions: list[np.ndarray] | None = None,
    direction_mode: str = "incoming",
    direction_inset=(0.075, 0.74, 0.19, 0.19),
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    setup_polar(ax)

    db = aggregate_rcs_db(source_fields, mode)
    th, rr = close_polar_curve(theta, db)
    ax.plot(th, rr, lw=2.8, color="black")

    ax.set_title(f"{label} ({plane_label} plane)", va="bottom")

    if mesh_path is not None:
        add_mesh_silhouette(
            ax,
            mesh_path=mesh_path,
            view=silhouette_view,
            nose_polar_deg=nose_angle_deg,
            alpha=0.18,
        )

    if directions:
        add_direction_projection_inset(
            ax,
            directions=directions,
            source_ids=source_ids,
            view=plot_view,
            mode=direction_mode,
            inset=direction_inset,
            colors=SOURCE_COLORS,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--plane-label", default=None)
    parser.add_argument(
        "--aggregate-mode",
        choices=["coherent", "power_sum", "power_avg"],
        default="coherent",
    )
    parser.add_argument(
        "--aggregate-label",
        default="Combined illumination (coherent)",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Optional mesh path for the central silhouette. Default: infer from the CSV location.",
    )
    parser.add_argument(
        "--silhouette-view",
        choices=["xy", "xz", "yz", "auto"],
        default="auto",
        help="Projection used for the central silhouette.",
    )
    parser.add_argument(
        "--nose-angle-deg",
        type=float,
        default=180.0,
        help="Polar angle where the aircraft nose should point.",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs=3,
        action="append",
        metavar=("DX", "DY", "DZ"),
        help=(
            "Incident plane-wave propagation direction. Repeat once per source. "
            "By default the inset shows the incoming direction (-d)."
        ),
    )
    parser.add_argument(
        "--direction-mode",
        choices=["incoming", "propagation"],
        default="incoming",
        help="Show incoming direction (-d) or propagation direction (d) in the inset.",
    )
    parser.add_argument(
        "--direction-inset",
        type=float,
        nargs=4,
        metavar=("LEFT", "BOTTOM", "WIDTH", "HEIGHT"),
        default=(0.075, 0.74, 0.19, 0.19),
        help="Inset box for the direction projection in axes coordinates.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    theta, source_ids, source_fields = load_rcs_csv(csv_path)

    inferred_plane = infer_plane_from_csv_name(csv_path)
    if args.plane_label is None:
        plane_label = inferred_plane.upper() if inferred_plane is not None else "XY"
    else:
        plane_label = args.plane_label

    if args.mesh is not None:
        mesh_path = args.mesh.expanduser().resolve()
    else:
        mesh_path = infer_mesh_path(csv_path)

    if args.silhouette_view == "auto":
        silhouette_view = inferred_plane if inferred_plane is not None else "xy"
    else:
        silhouette_view = args.silhouette_view

    plot_view = inferred_plane if inferred_plane is not None else silhouette_view

    if mesh_path is not None and not mesh_path.exists():
        print(f"Warning: mesh path not found: {mesh_path}")
        mesh_path = None

    directions = [np.asarray(v, dtype=float) for v in (args.direction or [])]
    if directions and len(directions) != len(source_ids):
        nuse = min(len(directions), len(source_ids))
        print(
            f"Warning: got {len(directions)} direction vector(s) for "
            f"{len(source_ids)} source curve(s); using first {nuse}."
        )
        directions = directions[:nuse]

    base = csv_path.with_suffix("")
    out_sources = base.parent / f"{base.name}_sources.png"
    out_combined = base.parent / f"{base.name}_{args.aggregate_mode}.png"

    plot_sources(
        out_sources,
        theta,
        source_ids,
        source_fields,
        plane_label,
        plot_view,
        mesh_path,
        silhouette_view,
        args.nose_angle_deg,
        directions=directions,
        direction_mode=args.direction_mode,
        direction_inset=tuple(args.direction_inset),
    )

    plot_combined(
        out_combined,
        theta,
        source_ids,
        source_fields,
        plane_label,
        plot_view,
        args.aggregate_mode,
        args.aggregate_label,
        mesh_path,
        silhouette_view,
        args.nose_angle_deg,
        directions=directions,
        direction_mode=args.direction_mode,
        direction_inset=tuple(args.direction_inset),
    )

    print(f"Wrote {out_sources}")
    print(f"Wrote {out_combined}")


if __name__ == "__main__":
    main()