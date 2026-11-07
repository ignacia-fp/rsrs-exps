from datetime import date

import pytest

from rsrs_experiments.configuration import compile_test, render_script, select_tests
from rsrs_experiments.cli import main


def test_numbered_options_compile_to_legacy_json_blocks():
    data, scenario, rsrs, output = compile_test({
        "operator": 19, "dimension": 1, "geometry": 6,
        "assembler": 1, "symmetry": 0, "nonsymmetric_combination": 1,
        "kappa": 2.0, "h": 0.03, "id_tols": [1e-2, 1e-4],
    })
    assert data["structured_operator_type"] == "BemppClHelmholtzCombined"
    assert scenario["dim_args"] == [{"KappaAndMeshwidth": [2.0, 0.03]}]
    assert rsrs["nonsymmetric_id_combination"] == "Concat"
    assert rsrs["symmetry"] == "NoSymm"
    assert output["solve"] == {"True": 1e-7}

    _, _, _, unprec = compile_test({"gmres": 1})
    assert unprec["solve"] == {"UnprecOnly": 1e-7}


def test_bundle_sweep_and_dated_slurm_render(tmp_path):
    definition = {
        "DEFAULTS": {"dimension": 2, "h": 0.1},
        "TESTS": {"a": {"options": {"id_tols": [1e-2]}, "sweep": {"tol_ext_near": [1e-8, 1e-16]}}},
        "BUNDLES": {"all": ["a"]},
        "SLURM": {"job_name": "check", "cpus_per_task": 4},
    }
    assert [name for name, _ in select_tests(definition, "all")] == ["a", "a_2"]
    path = render_script(definition, "all", output_dir=tmp_path, script_name="my run", kind="sbatch", creation_date=date(2026, 11, 7))
    assert path.name == "my_run_2026-11-07.sbatch"
    text = path.read_text()
    assert "#SBATCH --cpus-per-task=4" in text
    assert text.count("cargo run --release") == 2
    assert text.count('"True":1e-07') == 1
    assert text.count('"PrecOnly":1e-07') == 1
    with pytest.raises(FileExistsError):
        render_script(definition, "all", output_dir=tmp_path, script_name="my run", kind="sbatch", creation_date=date(2026, 11, 7))


def test_unknown_numeric_choice_is_rejected():
    with pytest.raises(ValueError, match="invalid symmetry option"):
        compile_test({"symmetry": 99})


def test_json_build_can_generate_an_individual_test_and_bundle(tmp_path, monkeypatch):
    config = tmp_path / "jobs.json"
    config.write_text(__import__("json").dumps({
        "GENERATE": [
            {"selection": "a", "name": "one", "kind": "sh", "output_dir": str(tmp_path)},
            {"selection": "all", "name": "bundle", "kind": "sbatch", "output_dir": str(tmp_path)},
        ],
        "DEFAULTS": {"dimension": 2, "h": 0.1},
        "TESTS": {"a": {"id_tols": [1e-2]}, "b": {"id_tols": [1e-4]}},
        "BUNDLES": {"all": ["a", "b"]},
        "SLURM": {"job_name": "bundle"},
    }))
    monkeypatch.setattr("sys.argv", ["rsrs-experiment", "build", str(config)])
    assert main() == 0
    assert len(list(tmp_path.glob("one_*.sh"))) == 1
    bundle = next(tmp_path.glob("bundle_*.sbatch"))
    assert bundle.read_text().count("cargo run --release") == 2
