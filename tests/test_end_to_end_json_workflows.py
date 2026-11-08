"""Black-box tests for JSON -> generated script -> numerical report workflows."""

from __future__ import annotations

from datetime import date
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


CREATION_DATE = date(2026, 11, 7)


def _fibonacci_sphere(count: int) -> list[list[float]]:
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)).tolist()


def _write_octahedron(path: Path) -> None:
    """Write a twice-refined octahedral sphere as a Gmsh v2 surface mesh."""

    points = [
        np.array(value, dtype=float)
        for value in (
            (0, 0, 1), (1, 0, 0), (0, 1, 0),
            (-1, 0, 0), (0, -1, 0), (0, 0, -1),
        )
    ]
    cells = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4),
    ]
    for _ in range(2):
        edges: dict[tuple[int, int], int] = {}
        refined = []
        for a, b, c in cells:
            mids = []
            for u, v in ((b, c), (a, c), (a, b)):
                edge = tuple(sorted((u, v)))
                if edge not in edges:
                    midpoint = points[u] + points[v]
                    midpoint /= np.linalg.norm(midpoint)
                    edges[edge] = len(points)
                    points.append(midpoint)
                mids.append(edges[edge])
            refined.extend(
                ((a, mids[2], mids[1]), (b, mids[0], mids[2]),
                 (c, mids[1], mids[0]), tuple(mids))
            )
        cells = refined
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(points))]
    lines.extend(
        f"{index} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
        for index, point in enumerate(points, start=1)
    )
    lines.extend(("$EndNodes", "$Elements", str(len(cells))))
    lines.extend(
        f"{index} 2 2 1 1 {a + 1} {b + 1} {c + 1}"
        for index, (a, b, c) in enumerate(cells, start=1)
    )
    lines.extend(("$EndElements", ""))
    path.write_text("\n".join(lines))


CASES = (
    pytest.param(0, {"cells_per_axis": 6}, 1, (), id="biegrid-real-symmetric"),
    pytest.param(1, {"cells_per_axis": 6}, 0, (), id="biegrid-real-nonsymmetric"),
    pytest.param(2, {"cells_per_axis": 6}, 1, (), id="biegrid-complex-symmetric"),
    pytest.param(3, {"cells_per_axis": 6}, 0, (), id="biegrid-complex-nonsymmetric"),
    pytest.param(
        4,
        {
            "grid": "{mesh}",
            "assembler": "dense",
            "precision": "double",
            "expansion_order": 4,
        },
        1,
        ("bempp_cl.api",),
        id="bempp-cl-laplace-sphere",
    ),
    pytest.param(
        6,
        {
            "refinement_level": 2,
            "function_space_degree": 0,
            "quadrature_degree": 6,
            "local_tree_depth": 1,
            "global_tree_depth": 2,
            "expansion_order": 4,
        },
        1,
        ("bempp_rs_operators",),
        id="bempp-rs-laplace-sphere",
    ),
    pytest.param(
        7,
        {
            "sources": "{points}",
            "targets": "{points}",
            "expansion_order": 4,
            "n_crit": 8,
            "block_size": 8,
            "diagonal_shift": 2.0,
        },
        1,
        ("kifmm_py",),
        id="kifmm-laplace-particles",
    ),
)


def _require_modules(names: tuple[str, ...]) -> None:
    missing = []
    for name in ("bempp_rsrs", *names):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        pytest.skip("native integration dependencies are not installed: " + ", ".join(missing))


@pytest.mark.parametrize("operator,parameters,symmetry,modules", CASES)
def test_json_template_generates_and_runs_correct_case(
    tmp_path: Path,
    operator: int,
    parameters: dict,
    symmetry: int,
    modules: tuple[str, ...],
) -> None:
    """Exercise the same public JSON and generated-shell path used by users."""

    _require_modules(modules)
    mesh = tmp_path / "octahedron.msh"
    _write_octahedron(mesh)
    points = _fibonacci_sphere(128)
    parameters = json.loads(
        json.dumps(parameters).replace('"{mesh}"', json.dumps(str(mesh))).replace(
            '"{points}"', json.dumps(points)
        )
    )
    cache = tmp_path / "sample_cache"
    output = tmp_path / "result"
    generated = tmp_path / "generated"
    definition = {
        "GENERATE": {
            "selection": "case",
            "name": "end_to_end",
            "kind": "sh",
            "output_dir": str(generated),
            "date": CREATION_DATE.isoformat(),
            "executable": f"{sys.executable} -m rsrs_experiments run-case",
        },
        "SHARE_UNPRECONDITIONED": True,
        "DEFAULTS": {
            "operator": operator,
            "operator_parameters": parameters,
            "sample_cache": str(cache),
            "output": str(output),
            "sampling": {
                "target": 160,
                "save_samples": True,
                "workers": 2,
                "batch_size": 10,
                "seed": 1801,
                "symmetry": 0,
                "validation": "all",
                "validation_workers": 2,
                "validation_batch_size": 16,
                "validation_tolerance": 1.0e-10,
            },
            "rsrs": {
                "tol_id": 1.0e-4,
                "oversampling": 8,
                "oversampling_diag_blocks": 8,
                "max_tree_depth": 4,
                "leaf_size": 8,
                "symmetry": symmetry,
                "fixed_rank_sampling_mode": 0,
                "nonsymmetric_id_combination": 1,
                "factorization_threads": 1,
                "apply_threads": 1,
            },
            "diagnostics": {
                "error_probes": 3,
                "error_seed": 1901,
                "gmres": {
                    "mode": 3,
                    "tolerance": 1.0e-7,
                    "max_iterations": 100,
                    "restart": None,
                    "rhs_count": 1,
                    "seed": 2001,
                },
            },
        },
        "TESTS": {"case": {}},
        "BUNDLES": {"case": ["case"]},
    }
    config = tmp_path / "case.json"
    config.write_text(json.dumps(definition, indent=2))
    environment = os.environ.copy()

    subprocess.run(
        [sys.executable, "-m", "rsrs_experiments", "build", str(config)],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
    )
    script = generated / f"end_to_end_{CREATION_DATE.isoformat()}.sh"
    assert script.is_file() and os.access(script, os.X_OK)
    subprocess.run(
        [str(script)],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=300,
    )

    report = json.loads((output / "rsrs_report.json").read_text())
    sampling_report = json.loads((cache / "sampling_report.json").read_text())
    assert report["status"] == "complete"
    assert report["operator"]["operator_type"] in script.read_text()
    assert report["sample_count"] == 160
    worker_directories: dict[int, str] = {}
    for batch in sampling_report["batch_results"]:
        if batch["worker_pid"] is not None:
            worker_directories.setdefault(batch["worker_pid"], batch["worker_directory"])
            assert worker_directories[batch["worker_pid"]] == batch["worker_directory"]
            assert "/workers/sampling/worker_" in batch["worker_directory"]
    assert worker_directories
    assert report["rsrs"]["active_samples"] > 0
    assert report["rsrs"]["ranks"]
    assert report["errors"]["probe_apply_relative_fro"] < 5.0e-2
    assert report["errors"]["inverse_residual_relative_fro"] < 5.0e-2
    assert set(report["gmres"]["solves"]) == {"unpreconditioned", "preconditioned"}
    for solves in report["gmres"]["solves"].values():
        assert len(solves) == 1
        assert solves[0]["info"] == 0
        assert solves[0]["recorded_residuals"]
        assert solves[0]["final_true_relative_residual"] < 2.0e-6

    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["sample_count"] == 160
    assert (cache / "points.npy").is_file()
