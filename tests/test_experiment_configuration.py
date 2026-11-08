from datetime import date
import json
from pathlib import Path

import pytest

from rsrs_experiments.configuration import compile_case, render_script, select_tests
from rsrs_experiments.cli import main
import rsrs_experiments.runner as runner


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


def test_operator_parameters_replace_unrelated_defaults():
    case = compile_case(
        base_case(
            operator=6,
            operator_parameters={"refinement_level": 0, "expansion_order": 4},
        ),
        name="bempp-rs",
    )
    assert case["operator"]["parameters"] == {
        "refinement_level": 0,
        "expansion_order": 4,
    }


def test_sampling_and_target_operators_can_use_different_parameters():
    case = compile_case(
        base_case(
            operator=7,
            operator_parameters={
                "sources": [[0.0, 0.0, 0.0]],
                "targets": [[0.0, 0.0, 0.0]],
                "expansion_order": 8,
            },
            sampling_operator_parameters={
                "sources": [[0.0, 0.0, 0.0]],
                "targets": [[0.0, 0.0, 0.0]],
                "expansion_order": 4,
            },
        ),
        name="mixed-order",
    )
    assert case["operator"]["parameters"]["expansion_order"] == 8
    assert case["sampling_operator"]["parameters"]["expansion_order"] == 4


def test_ephemeral_sampling_does_not_require_a_cache_path():
    case = compile_case(
        base_case(
            sample_cache=None,
            sampling={"target": 128, "save_samples": False},
        ),
        name="ephemeral",
    )
    assert case["sample_cache"] is None
    assert case["sampling"]["save_samples"] is False


def test_ephemeral_samples_are_removed_after_the_run(monkeypatch, tmp_path):
    observed: dict[str, Path] = {}
    sentinel = object()

    def fake_sample(config):
        observed["cache"] = Path(config.output)
        assert observed["cache"].is_dir()
        (observed["cache"] / "generated").write_text("temporary")

    def fake_run(cache, *_args, **_kwargs):
        assert Path(cache) == observed["cache"]
        assert (Path(cache) / "generated").is_file()
        return sentinel

    monkeypatch.setattr(runner, "sample_operator", fake_sample)
    monkeypatch.setattr(runner, "run_rsrs", fake_run)
    result = runner.run_case(
        compile_case(
            base_case(
                sample_cache=None,
                output=str(tmp_path / "result"),
                sampling={"target": 128, "save_samples": False},
            ),
            name="ephemeral",
        )
    )
    assert result is sentinel
    assert not observed["cache"].exists()


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
