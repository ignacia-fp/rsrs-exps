#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


SOURCE_COLORS = ["tab:blue", "tab:brown", "tab:cyan", "tab:orange", "tab:green"]


def configure_jcp_style() -> None:
    """Use a conservative monochrome, journal-ready Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_rcs_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    names = data.dtype.names or ()
    theta = np.asarray(data["angle_rad"], dtype=float)

    pat = re.compile(r"source_(\d{3})_real$")
    source_ids = sorted(int(m.group(1)) for name in names if (m := pat.match(name)))

    source_fields = {}
    for sid in source_ids:
        tag = f"source_{sid:03d}"
        field = np.asarray(data[f"{tag}_real"], dtype=float) + 1j * np.asarray(
            data[f"{tag}_imag"], dtype=float
        )
        source_fields[sid] = field

    return theta, source_ids, source_fields


def infer_plane_label(path: Path) -> str:
    m = re.search(r"rcs_(xy|xz|yz)_", path.name.lower())
    return m.group(1).upper() if m else "XY"


def validate_inputs(
    theta_ref: np.ndarray,
    theta_cmp: np.ndarray,
    ids_ref: list[int],
    ids_cmp: list[int],
):
    if set(ids_ref) != set(ids_cmp):
        raise ValueError(
            f"Source id mismatch.\n"
            f"Reference ids: {ids_ref}\n"
            f"Compare ids:   {ids_cmp}"
        )

    if theta_ref.shape != theta_cmp.shape or not np.allclose(theta_ref, theta_cmp):
        raise ValueError("The two CSV files do not have matching theta samples.")


def rcs_power_from_field(field: np.ndarray) -> np.ndarray:
    power = 4.0 * np.pi * np.abs(field) ** 2
    return np.maximum(power, np.finfo(float).tiny)


def rcs_db_from_field(field: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(rcs_power_from_field(field))


def aggregate_power(source_fields: list[np.ndarray], mode: str) -> np.ndarray:
    fields = np.vstack(source_fields)

    if mode == "coherent":
        power = np.abs(np.sum(fields, axis=0)) ** 2
    elif mode == "power_sum":
        power = np.sum(np.abs(fields) ** 2, axis=0)
    elif mode == "power_avg":
        power = np.mean(np.abs(fields) ** 2, axis=0)
    else:
        raise ValueError(f"Unknown aggregate mode: {mode}")

    power = 4.0 * np.pi * power
    return np.maximum(power, np.finfo(float).tiny)


def aggregate_db(source_fields: list[np.ndarray], mode: str) -> np.ndarray:
    return 10.0 * np.log10(aggregate_power(source_fields, mode))


def sort_theta_and_values(theta_rad: np.ndarray, values: np.ndarray):
    theta_deg = np.mod(np.rad2deg(theta_rad), 360.0)
    order = np.argsort(theta_deg)
    return theta_deg[order], np.asarray(values)[order]


def panel_series(theta_rad: np.ndarray, ref_values: np.ndarray, cmp_values: np.ndarray):
    theta_deg, ref_sorted = sort_theta_and_values(theta_rad, ref_values)
    _, cmp_sorted = sort_theta_and_values(theta_rad, cmp_values)
    return theta_deg, ref_sorted, cmp_sorted


def style_axis(ax, monochrome: bool = False):
    grid_kwargs = {"color": "0.82"} if monochrome else {}
    ax.grid(True, which="major", alpha=0.55 if monochrome else 0.35, lw=0.55, **grid_kwargs)
    ax.grid(True, which="minor", alpha=0.28 if monochrome else 0.15, lw=0.4, **grid_kwargs)
    ax.set_xlim(0.0, 360.0)
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.xaxis.set_minor_locator(MultipleLocator(15))


def make_compare_figure(
    nrows: int,
    title: str,
    reference_label: str,
    compare_label: str,
):
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(11.8, 2.15 * nrows + 0.6),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.09, right=0.83, top=0.93, bottom=0.07, hspace=0.08)
    fig.suptitle(title, y=0.98, fontsize=15)

    legend_handles = [
        Line2D([0], [0], color="0.05", lw=2.2, ls="-", label=reference_label),
        Line2D([0], [0], color="0.05", lw=2.2, ls="--", label=compare_label),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.845, 0.965),
        frameon=True,
        borderaxespad=0.0,
        fontsize=11,
    )

    return fig, axes


def make_single_legend_figure(nrows: int, title: str):
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(11.8, 2.15 * nrows + 0.6),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.07, hspace=0.08)
    fig.suptitle(title, y=0.98, fontsize=15)
    return fig, axes


def max_abs_diff_text(a: np.ndarray, b: np.ndarray) -> str:
    return f"max |Δ| = {np.max(np.abs(a - b)):.2f}"


def rms_diff_text(a: np.ndarray, b: np.ndarray) -> str:
    return f"RMS = {np.sqrt(np.mean((a - b) ** 2)):.2f}"


def max_rel_to_peak_text(a: np.ndarray, b: np.ndarray) -> str:
    denom = max(np.max(np.abs(a)), np.finfo(float).tiny)
    rel = np.max(np.abs(a - b)) / denom
    return f"max |Δ| / peak = {100.0 * rel:.1f}%"


def build_rows(
    theta_rad: np.ndarray,
    source_ids: list[int],
    ref_fields: dict[int, np.ndarray],
    cmp_fields: dict[int, np.ndarray],
    aggregate_mode: str,
):
    rows = []

    ref_agg_db = aggregate_db([ref_fields[sid] for sid in source_ids], aggregate_mode)
    cmp_agg_db = aggregate_db([cmp_fields[sid] for sid in source_ids], aggregate_mode)
    ref_agg_lin = aggregate_power([ref_fields[sid] for sid in source_ids], aggregate_mode)
    cmp_agg_lin = aggregate_power([cmp_fields[sid] for sid in source_ids], aggregate_mode)

    rows.append(
        {
            "label_db": "Sum (dB)",
            "label_lin": "Sum",
            "color": "black",
            "ref_db": ref_agg_db,
            "cmp_db": cmp_agg_db,
            "ref_lin": ref_agg_lin,
            "cmp_lin": cmp_agg_lin,
        }
    )

    for i, sid in enumerate(source_ids):
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        rows.append(
            {
                "label_db": f"S{sid} (dB)",
                "label_lin": f"S{sid}",
                "color": color,
                "ref_db": rcs_db_from_field(ref_fields[sid]),
                "cmp_db": rcs_db_from_field(cmp_fields[sid]),
                "ref_lin": rcs_power_from_field(ref_fields[sid]),
                "cmp_lin": rcs_power_from_field(cmp_fields[sid]),
            }
        )

    return rows


def plot_db_compare(
    out_path: Path,
    theta_rad: np.ndarray,
    rows: list[dict],
    plane_label: str,
    aggregate_mode: str,
    reference_label: str,
    compare_label: str,
    monochrome: bool = False,
):
    title = f"RCS Comparison ({plane_label} plane)"
    if aggregate_mode:
        title = f"RCS Comparison ({plane_label} plane, {aggregate_mode})"
    fig, axes = make_compare_figure(
        len(rows),
        title,
        reference_label,
        compare_label,
    )

    for ax, row in zip(axes, rows):
        theta_deg, ref_db, cmp_db = panel_series(theta_rad, row["ref_db"], row["cmp_db"])
        color = "black" if monochrome else row["color"]
        ax.plot(theta_deg, ref_db, color=color, lw=1.6, ls="-")
        ax.plot(theta_deg, cmp_db, color=color, lw=1.6, ls="--")
        ax.set_ylabel(row["label_db"], fontsize=11)
        style_axis(ax, monochrome=monochrome)

        box_text = f"{max_abs_diff_text(ref_db, cmp_db)} dB"
        ax.text(
            0.995,
            0.035,
            box_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92, boxstyle="round,pad=0.22"),
        )

    axes[-1].set_xlabel(r"Angle $\theta$ (deg)", fontsize=12)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_db_difference(
    out_path: Path,
    theta_rad: np.ndarray,
    rows: list[dict],
    plane_label: str,
    reference_label: str,
    compare_label: str,
):
    fig, axes = make_single_legend_figure(
        len(rows),
        f"RCS Difference ({plane_label} plane): {compare_label} - {reference_label}",
    )

    for ax, row in zip(axes, rows):
        theta_deg, ref_db, cmp_db = panel_series(theta_rad, row["ref_db"], row["cmp_db"])
        diff = cmp_db - ref_db

        ax.plot(theta_deg, diff, color=row["color"], lw=2.0)
        ax.axhline(0.0, color="0.2", lw=1.0, alpha=0.7)
        ax.set_ylabel(f"{row['label_lin']} Δ", fontsize=11)
        style_axis(ax)

        ymax = max(np.max(np.abs(diff)), 1.0)
        ax.set_ylim(-1.12 * ymax, 1.12 * ymax)

        box_text = f"{max_abs_diff_text(cmp_db, ref_db)} dB, {rms_diff_text(cmp_db, ref_db)} dB"
        ax.text(
            0.995,
            0.035,
            box_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92, boxstyle="round,pad=0.22"),
        )

    axes[-1].set_xlabel(r"Angle $\theta$ (deg)", fontsize=12)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_linear_compare(
    out_path: Path,
    theta_rad: np.ndarray,
    rows: list[dict],
    plane_label: str,
    aggregate_mode: str,
    reference_label: str,
    compare_label: str,
    yscale: str,
):
    fig, axes = make_compare_figure(
        len(rows),
        f"Linear RCS Comparison ({plane_label} plane, {aggregate_mode})",
        reference_label,
        compare_label,
    )

    for ax, row in zip(axes, rows):
        theta_deg, ref_lin, cmp_lin = panel_series(theta_rad, row["ref_lin"], row["cmp_lin"])
        ax.plot(theta_deg, ref_lin, color=row["color"], lw=2.2, ls="-")
        ax.plot(theta_deg, cmp_lin, color=row["color"], lw=2.2, ls="--")
        ax.set_ylabel(row["label_lin"], fontsize=11)
        style_axis(ax)

        if yscale == "log":
            ax.set_yscale("log")

        box_text = max_rel_to_peak_text(ref_lin, cmp_lin)
        ax.text(
            0.995,
            0.035,
            box_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92, boxstyle="round,pad=0.22"),
        )

    axes[-1].set_xlabel(r"Angle $\theta$ (deg)", fontsize=12)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_csv", type=Path, help="Reference CSV, e.g. rcs_xy_prec.csv")
    parser.add_argument("compare_csv", type=Path, help="Comparison CSV, e.g. rcs_xy_approx_inv.csv")
    parser.add_argument(
        "--aggregate-mode",
        choices=["coherent", "power_sum", "power_avg"],
        default="coherent",
        help="How to combine the source fields for the aggregate row.",
    )
    parser.add_argument(
        "--jcp-style",
        action="store_true",
        help="Use monochrome curves, Times-like serif type, and also write vector PDF.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Omit the aggregate row and plot only the individual sources.",
    )
    parser.add_argument(
        "--reference-label",
        default="Preconditioned GMRES",
        help="Legend label for the reference curves.",
    )
    parser.add_argument(
        "--compare-label",
        default="Approx. inverse",
        help="Legend label for the comparison curves.",
    )
    parser.add_argument(
        "--plane-label",
        default=None,
        help="Override plane label in the title, e.g. XY or XZ.",
    )
    parser.add_argument(
        "--linear-yscale",
        choices=["linear", "log"],
        default="log",
        help="Y scale for the linear RCS figure. Default: log.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help=(
            "Output stem without extension. Default: beside compare_csv using its stem. "
            "Example: --output-stem plots/xy_compare"
        ),
    )
    args = parser.parse_args()

    if args.jcp_style:
        configure_jcp_style()

    ref_csv = args.reference_csv.expanduser().resolve()
    cmp_csv = args.compare_csv.expanduser().resolve()

    theta_ref, ids_ref, ref_fields = load_rcs_csv(ref_csv)
    theta_cmp, ids_cmp, cmp_fields = load_rcs_csv(cmp_csv)

    validate_inputs(theta_ref, theta_cmp, ids_ref, ids_cmp)

    source_ids = sorted(set(ids_ref) & set(ids_cmp))
    plane_label = args.plane_label or infer_plane_label(ref_csv)

    rows = build_rows(
        theta_rad=theta_ref,
        source_ids=source_ids,
        ref_fields=ref_fields,
        cmp_fields=cmp_fields,
        aggregate_mode=args.aggregate_mode,
    )
    if args.no_aggregate:
        rows = rows[1:]

    if args.output_stem is None:
        stem = cmp_csv.parent / cmp_csv.stem
    else:
        stem = args.output_stem.expanduser()
        if not stem.is_absolute():
            stem = (Path.cwd() / stem).resolve()

    out_db_compare = stem.parent / f"{stem.name}_db_compare.png"
    out_db_diff = stem.parent / f"{stem.name}_db_diff.png"
    out_linear_compare = stem.parent / f"{stem.name}_linear_compare.png"

    plot_db_compare(
        out_path=out_db_compare,
        theta_rad=theta_ref,
        rows=rows,
        plane_label=plane_label,
        aggregate_mode="" if args.no_aggregate else args.aggregate_mode,
        reference_label=args.reference_label,
        compare_label=args.compare_label,
        monochrome=args.jcp_style,
    )

    plot_db_difference(
        out_path=out_db_diff,
        theta_rad=theta_ref,
        rows=rows,
        plane_label=plane_label,
        reference_label=args.reference_label,
        compare_label=args.compare_label,
    )

    plot_linear_compare(
        out_path=out_linear_compare,
        theta_rad=theta_ref,
        rows=rows,
        plane_label=plane_label,
        aggregate_mode=args.aggregate_mode,
        reference_label=args.reference_label,
        compare_label=args.compare_label,
        yscale=args.linear_yscale,
    )

    print(f"Wrote {out_db_compare}")
    print(f"Wrote {out_db_compare.with_suffix('.pdf')}")
    print(f"Wrote {out_db_diff}")
    print(f"Wrote {out_linear_compare}")


if __name__ == "__main__":
    main()
