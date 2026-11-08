"""Declarative Python-native RSRS cases and dated script rendering."""

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
        0: Choice("biegrid_real_symmetric", "real transpose-symmetric BIEGrid"),
        1: Choice("biegrid_real_nonsymmetric", "real nonsymmetric BIEGrid"),
        2: Choice("biegrid_complex_symmetric", "complex transpose-symmetric BIEGrid"),
        3: Choice("biegrid_complex_nonsymmetric", "complex nonsymmetric BIEGrid"),
        4: Choice("bempp_cl_laplace_single_layer", "Bempp-cl Laplace single layer"),
        5: Choice("bempp_cl_helmholtz_combined", "Bempp-cl combined Helmholtz"),
        6: Choice("bempp_rs_laplace_single_layer", "Bempp-rs Laplace single layer"),
        7: Choice("laplace_particle_fmm", "KiFMM Laplace particle operator"),
    },
    "symmetry": {
        0: Choice("NoSymm", "general nonsymmetric"),
        1: Choice("Symmetric", "algebraic-transpose symmetric"),
        2: Choice("Hermitian", "conjugate-transpose symmetric"),
    },
    "sampling_symmetry": {
        0: Choice("auto", "infer from the operator"),
        1: Choice("symmetric", "generate Y only"),
        2: Choice("nonsymmetric", "generate Y and Z"),
    },
    "gmres": {
        0: Choice("none", "do not run GMRES"),
        1: Choice("unpreconditioned", "unpreconditioned only; no samples or factors"),
        2: Choice("preconditioned", "RSRS-preconditioned only"),
        3: Choice("both", "unpreconditioned and RSRS-preconditioned"),
    },
    "null_method": {
        0: Choice("Projection", "projector nullification"),
        1: Choice("Svd", "SVD null space"),
        2: Choice("Qr", "QR null space"),
    },
    "block_extraction": {
        0: Choice("LuLstSq", "LU regularised least squares"),
        1: Choice("Svd", "SVD least squares"),
    },
    "qr": {0: Choice("RRQR", "rank-revealing QR"), 1: Choice("SRRQR", "strong RRQR")},
    "rank_picking": {
        i: Choice(value, value)
        for i, value in enumerate(("Min", "Max", "Avg", "Mid", "DoubleMin", "Tol"))
    },
    "factorization": {
        0: Choice("Split", "split factorisation"),
        1: Choice("Joint", "joint factorisation"),
    },
    "fixed_rank_sampling": {
        0: Choice("PerLevel", "per-level sample budget"),
        1: Choice("Constant", "constant sample budget"),
    },
    "nonsymmetric_combination": {
        0: Choice("Sum", "combine Y and Z by summation"),
        1: Choice("Concat", "combine Y and Z by concatenation"),
    },
}


DEFAULTS: dict[str, Any] = {
    "operator": 0,
    "operator_parameters": {"cells_per_axis": 8},
    "sampling_operator": None,
    "sampling_operator_parameters": None,
    "sample_cache": None,
    "output": None,
    "sampling": {
        "target": 0,
        "save_samples": True,
        "workers": 1,
        "batch_size": 10,
        "seed": 0,
        "symmetry": 0,
        "validation": "none",
    },
    "rsrs": {
        "oversampling": 8,
        "oversampling_diag_blocks": 16,
        "min_num_samples": 0,
        "initial_num_samples": 0,
        "tol_null": 1.0e-16,
        "tol_id": 1.0e-2,
        "tol_ext_near": 1.0e-16,
        "tol_diag_ext": 1.0e-16,
        "min_rank": 1,
        "min_level": 1,
        "max_tree_depth": 6,
        "leaf_size": None,
        "symmetry": 0,
        "fixed_rank_sampling_mode": 0,
        "nonsymmetric_id_combination": 1,
        "null_method": 0,
        "qr_method": 0,
        "near_block_extraction_method": 0,
        "diag_block_extraction_method": 0,
        "rank_picking": 0,
        "fact_type": 1,
        "factorization_threads": 1,
        "apply_threads": 1,
        "flush_factors": False,
        "store_far": False,
    },
    "diagnostics": {
        "error_probes": 4,
        "error_seed": 1701,
        "gmres": {
            "mode": 0,
            "tolerance": 1.0e-7,
            "max_iterations": 500,
            "restart": None,
            "rhs_count": 1,
            "seed": 1701,
        },
    },
}


def _choice(catalog: str, value: Any) -> str:
    if isinstance(value, str):
        allowed_values = {item.value for item in CATALOGS[catalog].values()}
        if value in allowed_values:
            return value
    try:
        return CATALOGS[catalog][int(value)].value
    except (KeyError, TypeError, ValueError) as exc:
        allowed = ", ".join(f"{key}={item.value}" for key, item in CATALOGS[catalog].items())
        raise ValueError(f"invalid {catalog} option {value!r}; choose {allowed}") from exc


def option_guide() -> str:
    lines = ["RSRS numbered option guide", "============================"]
    for name, choices in CATALOGS.items():
        lines.extend(("", name + ":"))
        lines.extend(
            f"  {number}: {item.value} — {item.description}"
            for number, item in choices.items()
        )
    return "\n".join(lines) + "\n"


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def compile_case(values: Mapping[str, Any], *, name: str = "case") -> dict[str, Any]:
    """Resolve numbered choices into one self-contained Python experiment case."""
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown case keys: {unknown}")
    case = _deep_merge(DEFAULTS, values)
    # Operator constructors have unrelated parameter vocabularies.  Replacing
    # this mapping avoids leaking BIEGrid's default ``cells_per_axis`` into a
    # Bempp or KiFMM constructor.
    if "operator_parameters" in values:
        case["operator_parameters"] = dict(values["operator_parameters"])
    if "sampling_operator_parameters" in values:
        parameters = values["sampling_operator_parameters"]
        case["sampling_operator_parameters"] = (
            None if parameters is None else dict(parameters)
        )
    operator = case["operator"]
    operator_type = _choice("operator", operator) if not isinstance(operator, Mapping) else None
    operator_spec = (
        dict(operator)
        if isinstance(operator, Mapping)
        else {"operator_type": operator_type, "parameters": dict(case["operator_parameters"])}
    )
    sampling_operator = case["sampling_operator"]
    sampling_parameters = case["sampling_operator_parameters"]
    if sampling_operator is None and sampling_parameters is None:
        sampling_operator_spec = dict(operator_spec)
    else:
        selected = case["operator"] if sampling_operator is None else sampling_operator
        if isinstance(selected, Mapping):
            if sampling_parameters is not None:
                raise ValueError(
                    "sampling_operator_parameters cannot accompany a complete "
                    "sampling_operator mapping"
                )
            sampling_operator_spec = dict(selected)
        else:
            sampling_type = _choice("operator", selected)
            parameters = (
                dict(case["operator_parameters"])
                if sampling_parameters is None
                else dict(sampling_parameters)
            )
            sampling_operator_spec = {
                "operator_type": sampling_type,
                "parameters": parameters,
            }
    rsrs = dict(case["rsrs"])
    for key, catalog in (
        ("symmetry", "symmetry"),
        ("fixed_rank_sampling_mode", "fixed_rank_sampling"),
        ("nonsymmetric_id_combination", "nonsymmetric_combination"),
        ("null_method", "null_method"),
        ("qr_method", "qr"),
        ("near_block_extraction_method", "block_extraction"),
        ("diag_block_extraction_method", "block_extraction"),
        ("rank_picking", "rank_picking"),
        ("fact_type", "factorization"),
    ):
        rsrs[key] = _choice(catalog, rsrs[key])
    tree = {
        "max_tree_depth": int(rsrs.pop("max_tree_depth")),
        "leaf_size": rsrs.pop("leaf_size"),
    }
    threads = {
        "factorization_threads": int(rsrs.pop("factorization_threads")),
        "apply_threads": int(rsrs.pop("apply_threads")),
    }
    rsrs.update({"schema_version": 1, "tree": tree, "threads": threads})
    sampling = dict(case["sampling"])
    sampling["symmetry"] = _choice("sampling_symmetry", sampling["symmetry"])
    diagnostics = dict(case["diagnostics"])
    diagnostics["gmres"] = dict(diagnostics["gmres"])
    diagnostics["gmres"]["mode"] = _choice("gmres", diagnostics["gmres"]["mode"])
    mode = diagnostics["gmres"]["mode"]
    ephemeral_samples = (
        int(sampling.get("target", 0)) > 0
        and not bool(sampling.get("save_samples", True))
    )
    if mode != "unpreconditioned" and not case["sample_cache"] and not ephemeral_samples:
        raise ValueError(
            "sample_cache is required unless GMRES is unpreconditioned or newly "
            "generated samples are ephemeral"
        )
    return {
        "schema_version": 1,
        "name": name,
        "operator": operator_spec,
        "sampling_operator": sampling_operator_spec,
        "sample_cache": case["sample_cache"],
        "output": case["output"] or f"results/experiments/{name}",
        "sampling": sampling,
        "rsrs": rsrs,
        "diagnostics": diagnostics,
    }


def _expanded(test: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    base = dict(test.get("options", test))
    sweep = dict(test.get("sweep", {})) if "options" in test else {}
    if not sweep:
        yield base
        return
    for values in itertools.product(*(sweep[key] for key in sweep)):
        update: dict[str, Any] = {}
        for dotted, value in zip(sweep, values):
            target = update
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        yield _deep_merge(base, update)


def load_definition(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        values = json.loads(path.read_text())
        return {key: value for key, value in values.items() if not key.startswith("_")}
    namespace = runpy.run_path(str(path))
    return {
        key: namespace.get(key, True if key == "SHARE_UNPRECONDITIONED" else {})
        for key in ("DEFAULTS", "TESTS", "BUNDLES", "SLURM", "SHARE_UNPRECONDITIONED", "GENERATE")
    }


def select_tests(definition: Mapping[str, Any], name: str) -> list[tuple[str, dict[str, Any]]]:
    common = dict(definition.get("DEFAULTS", {}))
    tests = definition.get("TESTS", {})
    selected = definition.get("BUNDLES", {}).get(name, [name])
    if name not in definition.get("BUNDLES", {}) and name not in tests:
        raise ValueError(f"unknown test or bundle {name!r}")
    answer = []
    for test_name in selected:
        if test_name not in tests:
            raise ValueError(f"bundle {name!r} refers to unknown test {test_name!r}")
        for number, options in enumerate(_expanded(tests[test_name]), start=1):
            expanded_name = test_name if number == 1 else f"{test_name}_{number}"
            answer.append((expanded_name, _deep_merge(common, options)))
    return answer


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not slug:
        raise ValueError("script name must contain a letter or number")
    return slug


def render_script(
    definition: Mapping[str, Any],
    selection: str,
    *,
    output_dir: Path,
    script_name: str,
    kind: str = "sh",
    creation_date: date | None = None,
    executable: str = "rsrs-experiment run-case",
) -> Path:
    tests = select_tests(definition, selection)
    stamp = (creation_date or date.today()).isoformat()
    extension = "sbatch" if kind == "sbatch" else "sh"
    path = output_dir / f"{_slug(script_name)}_{stamp}.{extension}"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing script: {path}")
    lines = ["#!/bin/bash"]
    if kind == "sbatch":
        lines.extend(
            f"#SBATCH --{key.replace('_', '-')}={value}"
            for key, value in definition.get("SLURM", {}).items()
            if value is not None
        )
    lines += ["", "set -euo pipefail", "", "export OPENBLAS_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "export OMP_NUM_THREADS=1", ""]
    baseline_operators: set[str] = set()
    share_baseline = bool(definition.get("SHARE_UNPRECONDITIONED", True))
    for test_name, options in tests:
        case = compile_case(options, name=test_name)
        gmres = case["diagnostics"]["gmres"]
        identity = json.dumps(
            [case["operator"], gmres["tolerance"], gmres["rhs_count"], gmres["seed"]],
            sort_keys=True,
        )
        if share_baseline and gmres["mode"] == "both":
            if identity in baseline_operators:
                gmres["mode"] = "preconditioned"
            else:
                baseline_operators.add(identity)
        command = f"{executable} {shlex.quote(json.dumps(case, separators=(',', ':')))}"
        lines.extend((f"echo {shlex.quote('Starting ' + test_name)}", command, ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    return path
