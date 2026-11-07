"""Declarative, numbered experiment configurations and dated script rendering.

This module deliberately contains no numerical work.  It turns small Python
dictionaries into the four JSON argument blocks understood by the historical
``bempp-rsrs-exps`` driver and writes reproducible shell or Slurm scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import itertools
import json
from pathlib import Path
import re
import runpy
import shlex
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Choice:
    value: str
    description: str


CATALOGS: dict[str, dict[int, Choice]] = {
    "operator": {
        0: Choice("BasicStructuredOperator", "small built-in test operator"),
        1: Choice("BemppClLaplaceSingleLayer", "Bempp-cl Laplace single-layer operator"),
        2: Choice("BemppClHelmholtzSingleLayer", "Bempp-cl Helmholtz single-layer operator"),
        3: Choice("KiFMMLaplaceOperator", "KiFMM Laplace operator"),
        4: Choice("KiFMMHelmholtzOperator", "KiFMM Helmholtz operator"),
        5: Choice("BemppRsLaplaceOperator", "Bempp-rs Laplace operator"),
        6: Choice("BemppClLaplaceCombined", "Bempp-cl combined Laplace operator"),
        7: Choice("BemppClLaplaceSingleLayerCP", "Laplace single-layer CP variant"),
        8: Choice("BemppClLaplaceSingleLayerMM", "Laplace single-layer MM variant"),
        9: Choice("BemppClHelmholtzSingleLayerCP", "Helmholtz single-layer CP variant"),
        10: Choice("BemppClLaplaceSingleLayerCPID", "Laplace CP-ID variant"),
        11: Choice("BemppClLaplaceSingleLayerP1", "Laplace P1 variant"),
        12: Choice("KiFMMLaplaceOperatorV", "vector KiFMM Laplace variant"),
        13: Choice("BemppClLaplaceCombinedP1", "combined Laplace P1 variant"),
        14: Choice("BemppClLaplaceSingleLayerCPIDP1", "Laplace CP-ID P1 variant"),
        15: Choice("BemppClHelmholtzSingleLayerCPID", "Helmholtz CP-ID variant"),
        16: Choice("BemppClMaxwellEfie", "Bempp-cl Maxwell EFIE"),
        17: Choice("BemppClHelmholtzSingleLayerP1", "Helmholtz P1 variant"),
        18: Choice("BemppClBurtonMiller", "Bempp-cl Burton-Miller operator"),
        19: Choice("BemppClHelmholtzCombined", "Bempp-cl combined Helmholtz operator"),
        20: Choice("BemppClLaplaceSecond", "second Laplace formulation"),
        21: Choice("BIEGrid", "historical BIEGrid test operator"),
    },
    "precision": {0: Choice("Single", "32-bit"), 1: Choice("Double", "64-bit")},
    "dimension": {
        0: Choice("Kappa", "derive h from kappa"),
        1: Choice("KappaAndMeshwidth", "provide kappa and h independently"),
        2: Choice("Meshwidth", "provide h only"),
        3: Choice("RefinementLevelAndDepth", "provide Bempp-rs refinement level and depth"),
    },
    "geometry": {
        i: Choice(value, value.replace("Surface", " surface"))
        for i, value in enumerate(("SphereSurface", "CubeSurface", "CylinderSurface", "EllipsoidSurface", "Dihedral", "Device", "F16", "RidgedHorn", "EMCCAlmond", "FrigateHull", "Plane", "Square", "Cube"))
    },
    "results": {0: Choice("All", "all reports"), 1: Choice("Rank", "rank reports"), 2: Choice("Time", "timing reports")},
    "null_method": {0: Choice("Projection", "projector I - Omega+ Omega"), 1: Choice("Svd", "SVD null space"), 2: Choice("Qr", "QR null space")},
    "block_extraction": {0: Choice("LuLstSq", "normal equations with LU"), 1: Choice("Svd", "SVD least squares")},
    "pivot": {0: Choice("Lu", "LU"), 1: Choice("DirectInversion", "direct inversion"), 2: Choice("LuHybrid", "hybrid LU")},
    "qr": {0: Choice("RRQR", "rank-revealing QR"), 1: Choice("SRRQR", "strong RRQR")},
    "rank_picking": {i: Choice(v, v) for i, v in enumerate(("Min", "Max", "Avg", "Mid", "DoubleMin", "Tol"))},
    "factorization": {0: Choice("Split", "split factorisation"), 1: Choice("Joint", "joint factorisation")},
    "fixed_rank_sampling": {0: Choice("PerLevel", "adapt active samples at every level"), 1: Choice("Constant", "constant active sample budget")},
    "nonsymmetric_combination": {0: Choice("Sum", "common ID from Y + Z"), 1: Choice("Concat", "common ID from [Y, Z]")},
    "assembler": {0: Choice("Dense", "dense assembly"), 1: Choice("ExaFMM", "ExaFMM"), 2: Choice("KiFMM", "KiFMM")},
    "symmetry": {0: Choice("NoSymm", "general nonsymmetric"), 1: Choice("Symmetric", "transpose symmetric"), 2: Choice("Hermitian", "conjugate-transpose symmetric")},
    "gmres": {
        0: Choice("False", "do not run GMRES"),
        1: Choice("UnprecOnly", "unpreconditioned GMRES only; do not load samples or factorize"),
        2: Choice("PrecOnly", "RSRS-preconditioned GMRES only"),
        3: Choice("True", "both unpreconditioned and RSRS-preconditioned GMRES"),
    },
}


DEFAULTS: dict[str, Any] = {
    "operator": 0, "precision": 1, "dimension": 2, "geometry": 0,
    "h": 0.1, "kappa": None, "ref_level": None, "depth": None,
    "id_tols": [1e-2], "max_tree_depth": 6, "leaf_size": None,
    "n_sources": 1, "assembler": 0, "results": 0,
    "gmres": 3, "solve_tol": 1e-7, "plot": True,
    "dense_errors": False, "factors_cn": False,
    "oversampling": 8, "oversampling_diag": 16,
    "min_num_samples": 0, "initial_num_samples": 0,
    "fixed_rank_sampling": 0, "nonsymmetric_combination": 1,
    "run_seed": None, "shift": 0.0, "null_method": 0, "qr": 0,
    "srrqr_f": 1.01, "block_extraction": 0, "pivot": 2,
    "lu_stab": 0.0, "diag_stab": 0.0, "tol_null": 1e-16,
    "tol_ext_near": 1e-16, "tol_ext_diag": 1e-16,
    "min_rank": 1, "min_level": 1, "symmetry": 0,
    "rank_picking": 0, "factorization": 1,
    "save_samples": False, "load_samples": True,
    "num_threads": 1, "apply_threads": None,
    "flush_factors": False, "store_far": False,
}


def _choice(catalog: str, index: Any) -> str:
    try:
        return CATALOGS[catalog][int(index)].value
    except (KeyError, TypeError, ValueError) as exc:
        allowed = ", ".join(map(str, CATALOGS[catalog]))
        raise ValueError(f"invalid {catalog} option {index!r}; choose one of {allowed}") from exc


def option_guide() -> str:
    lines = ["RSRS numbered option guide", "============================"]
    for name, choices in CATALOGS.items():
        lines.extend(("", name + ":"))
        lines.extend(f"  {number}: {item.value} — {item.description}" for number, item in choices.items())
    return "\n".join(lines) + "\n"


def _method(kind: str, value: float) -> dict[str, Any]:
    return {"type": kind, "value": value} if kind in {"Lu", "LuHybrid"} else {"type": kind}


def compile_test(values: Mapping[str, Any]) -> tuple[dict, dict, dict, dict]:
    """Validate and compile one compact numbered test dictionary."""
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown test keys: {unknown}")
    cfg = {**DEFAULTS, **values}
    if not cfg["id_tols"]:
        raise ValueError("id_tols must contain at least one tolerance")
    dim = _choice("dimension", cfg["dimension"])
    if dim == "Kappa":
        if cfg["kappa"] is None: raise ValueError("dimension=0 requires kappa")
        dim_arg = {"Kappa": cfg["kappa"]}
    elif dim == "KappaAndMeshwidth":
        if cfg["kappa"] is None: raise ValueError("dimension=1 requires kappa")
        dim_arg = {"KappaAndMeshwidth": [cfg["kappa"], cfg["h"]]}
    elif dim == "Meshwidth":
        dim_arg = {"MeshWidth": cfg["h"]}
    else:
        if cfg["ref_level"] is None or cfg["depth"] is None:
            raise ValueError("dimension=3 requires ref_level and depth")
        dim_arg = {"RefinementLevelAndDepth": [cfg["ref_level"], cfg["depth"]]}

    data = {"structured_operator_type": _choice("operator", cfg["operator"]), "precision": _choice("precision", cfg["precision"])}
    scenario = {
        "id_tols": list(cfg["id_tols"]), "dim_args": [dim_arg],
        "geometry_type": _choice("geometry", cfg["geometry"]),
        "max_tree_depth": cfg["max_tree_depth"], "n_sources": cfg["n_sources"],
        "assembler": _choice("assembler", cfg["assembler"]),
    }
    if cfg["leaf_size"] is not None: scenario["leaf_size"] = cfg["leaf_size"]
    qr = _choice("qr", cfg["qr"])
    pivot = _choice("pivot", cfg["pivot"])
    rsrs = {
        "oversampling": cfg["oversampling"], "oversampling_diag_blocks": cfg["oversampling_diag"],
        "min_num_samples": cfg["min_num_samples"], "initial_num_samples": cfg["initial_num_samples"],
        "fixed_rank_sampling_mode": _choice("fixed_rank_sampling", cfg["fixed_rank_sampling"]),
        "nonsymmetric_id_combination": _choice("nonsymmetric_combination", cfg["nonsymmetric_combination"]),
        "run_seed": cfg["run_seed"], "shift": {"type": "False"} if cfg["shift"] == 0 else {"type": "True", "value": cfg["shift"]},
        "null_method": _choice("null_method", cfg["null_method"]),
        "qr_method": "RRQR" if qr == "RRQR" else {"SRRQR": cfg["srrqr_f"]},
        "near_block_extraction_method": _choice("block_extraction", cfg["block_extraction"]),
        "diag_block_extraction_method": _choice("block_extraction", cfg["block_extraction"]),
        "lu_pivot_method": _method(pivot, cfg["lu_stab"]), "diag_pivot_method": _method(pivot, cfg["diag_stab"]),
        "tol_null": cfg["tol_null"], "tol_id": list(cfg["id_tols"])[0],
        "tol_ext_near": cfg["tol_ext_near"], "tol_diag_ext": cfg["tol_ext_diag"],
        "min_rank": cfg["min_rank"], "min_level": cfg["min_level"],
        "symmetry": _choice("symmetry", cfg["symmetry"]), "rank_picking": _choice("rank_picking", cfg["rank_picking"]),
        "fact_type": _choice("factorization", cfg["factorization"]), "save_samples": cfg["save_samples"],
        "load_samples": cfg["load_samples"], "num_threads": cfg["num_threads"],
        "flush_factors": cfg["flush_factors"], "store_far": cfg["store_far"], "symmetric": None,
    }
    if cfg["apply_threads"] is not None: rsrs["apply_threads"] = cfg["apply_threads"]
    gmres = _choice("gmres", cfg["gmres"])
    output = {
        "solve": {gmres: cfg["solve_tol"]} if gmres != "False" else {"False": None},
        "plot": cfg["plot"], "dense_errors": cfg["dense_errors"],
        "factors_cn": cfg["factors_cn"], "results_output": _choice("results", cfg["results"]),
    }
    return data, scenario, rsrs, output


def _expanded(test: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    base = dict(test.get("options", test))
    sweep = dict(test.get("sweep", {})) if "options" in test else {}
    if not sweep:
        yield base
        return
    for keys in itertools.product(*(sweep[key] for key in sweep)):
        yield {**base, **dict(zip(sweep, keys))}


def load_definition(path: Path) -> dict[str, Any]:
    if path.suffix == ".json": return json.loads(path.read_text())
    namespace = runpy.run_path(str(path))
    return {
        key: namespace.get(key, True if key == "SHARE_UNPRECONDITIONED" else {})
        for key in (
            "DEFAULTS", "TESTS", "BUNDLES", "SLURM",
            "SHARE_UNPRECONDITIONED", "GENERATE",
        )
    }


def select_tests(definition: Mapping[str, Any], name: str) -> list[tuple[str, dict[str, Any]]]:
    common = dict(definition.get("DEFAULTS", {}))
    tests = definition.get("TESTS", {})
    selected = definition.get("BUNDLES", {}).get(name, [name])
    if name not in definition.get("BUNDLES", {}) and name not in tests:
        raise ValueError(f"unknown test or bundle {name!r}")
    answer = []
    for test_name in selected:
        if test_name not in tests: raise ValueError(f"bundle {name!r} refers to unknown test {test_name!r}")
        for number, options in enumerate(_expanded(tests[test_name]), start=1):
            answer.append((test_name if number == 1 else f"{test_name}_{number}", {**common, **options}))
    return answer


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not slug: raise ValueError("script name must contain a letter or number")
    return slug


def render_script(definition: Mapping[str, Any], selection: str, *, output_dir: Path, script_name: str,
                  kind: str = "sh", creation_date: date | None = None, executable: str = "cargo run --release") -> Path:
    tests = select_tests(definition, selection)
    stamp = (creation_date or date.today()).isoformat()
    extension = "sbatch" if kind == "sbatch" else "sh"
    path = output_dir / f"{_slug(script_name)}_{stamp}.{extension}"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing script: {path}")
    lines = ["#!/bin/bash"]
    if kind == "sbatch":
        slurm = {**definition.get("SLURM", {})}
        lines.extend(f"#SBATCH --{key.replace('_', '-')}={value}" for key, value in slurm.items() if value is not None)
    lines += ["", "set -euo pipefail", "", "export OPENBLAS_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "export OMP_NUM_THREADS=1", "export OMP_DYNAMIC=FALSE", "unset RAYON_NUM_THREADS", ""]
    baseline_experiments: set[str] = set()
    share_baseline = bool(definition.get("SHARE_UNPRECONDITIONED", True))
    for test_name, original_options in tests:
        options = dict(original_options)
        shared_note = None
        if share_baseline and int(options.get("gmres", DEFAULTS["gmres"])) == 3:
            identity_options = {**options, "gmres": 0}
            data_identity, scenario_identity, _, _ = compile_test(identity_options)
            scenario_identity = dict(scenario_identity)
            # These affect RSRS, but not the physical unpreconditioned system.
            scenario_identity.pop("id_tols", None)
            scenario_identity.pop("max_tree_depth", None)
            scenario_identity.pop("leaf_size", None)
            identity = json.dumps(
                [data_identity, scenario_identity, options.get("solve_tol", DEFAULTS["solve_tol"])],
                sort_keys=True,
            )
            if identity in baseline_experiments:
                options["gmres"] = 2
                shared_note = "reusing unpreconditioned GMRES from the first equivalent experiment"
            else:
                baseline_experiments.add(identity)
        blocks = compile_test(options)
        command = " ".join([executable, *(shlex.quote(json.dumps(block, separators=(",", ":"))) for block in blocks)])
        lines.append(f"echo {shlex.quote('Starting ' + test_name)}")
        if shared_note:
            lines.append(f"echo {shlex.quote(shared_note)}")
        lines += [command, ""]
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    return path
