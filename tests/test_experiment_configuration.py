from datetime import date
import json

import pytest

from rsrs_experiments.configuration import compile_case, render_script, select_tests
from rsrs_experiments.cli import main


def base_case(**updates):
    values = {
        "operator": 3,
        "operator_parameters": {"cells_per_axis": 4},
        "sample_cache": "results/sample_cache/biegrid",
        "sampling": {"target": 100, "workers": 2, "validation": "sampled"},
        "rsrs": {"symmetry": 0, "nonsymmetric_id_combination": 1},
        "diagnostics": {"gmres": {"mode": 3, "tolerance": 1.0e-7}},
    }
    values.update(updates)
    return values


def test_numbered_options_compile_to_python_case():
    case = compile_case(base_case(), name="complex")
    assert case["operator"]["operator_type"] == "biegrid_complex_nonsymmetric"
    assert case["rsrs"]["nonsymmetric_id_combination"] == "Concat"
    assert case["rsrs"]["symmetry"] == "NoSymm"
    assert case["diagnostics"]["gmres"]["mode"] == "both"
    assert case["sampling"]["target"] == 100


def test_unpreconditioned_case_needs_no_cache_or_sampling():
    case = compile_case(
        {
            "operator": 0,
            "operator_parameters": {"cells_per_axis": 4},
            "sample_cache": None,
            "diagnostics": {"gmres": {"mode": 1}},
        },
        name="baseline",
    )
    assert case["diagnostics"]["gmres"]["mode"] == "unpreconditioned"


def test_bundle_sweep_and_dated_slurm_render(tmp_path):
    definition = {
        "DEFAULTS": base_case(),
        "TESTS": {
            "a": {
                "options": {},
                "sweep": {"rsrs.tol_ext_near": [1.0e-8, 1.0e-16]},
            }
        },
        "BUNDLES": {"all": ["a"]},
        "SLURM": {"job_name": "check", "cpus_per_task": 4},
    }
    assert [name for name, _ in select_tests(definition, "all")] == ["a", "a_2"]
    path = render_script(
        definition,
        "all",
        output_dir=tmp_path,
        script_name="my run",
        kind="sbatch",
        creation_date=date(2026, 11, 7),
    )
    text = path.read_text()
    assert "#SBATCH --cpus-per-task=4" in text
    assert text.count("rsrs-experiment run-case") == 2
    cases = [
        json.loads(line.split("run-case ", 1)[1].strip("'"))
        for line in text.splitlines()
        if "rsrs-experiment run-case" in line
    ]
    assert cases[0]["diagnostics"]["gmres"]["mode"] == "both"
    assert cases[1]["diagnostics"]["gmres"]["mode"] == "preconditioned"
    with pytest.raises(FileExistsError):
        render_script(
            definition,
            "all",
            output_dir=tmp_path,
            script_name="my run",
            kind="sbatch",
            creation_date=date(2026, 11, 7),
        )


def test_unknown_numeric_choice_is_rejected():
    with pytest.raises(ValueError, match="invalid symmetry option"):
        compile_case(base_case(rsrs={"symmetry": 99}))


def test_json_build_generates_bundle(tmp_path, monkeypatch):
    config = tmp_path / "jobs.json"
    config.write_text(
        json.dumps(
            {
                "GENERATE": {
                    "selection": "all",
                    "name": "bundle",
                    "kind": "sh",
                    "output_dir": str(tmp_path),
                },
                "DEFAULTS": base_case(),
                "TESTS": {"a": {}, "b": {"rsrs": {"tol_id": 1.0e-4}}},
                "BUNDLES": {"all": ["a", "b"]},
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["rsrs-experiment", "build", str(config)])
    assert main() == 0
    bundle = next(tmp_path.glob("bundle_*.sh"))
    assert bundle.read_text().count("rsrs-experiment run-case") == 2
