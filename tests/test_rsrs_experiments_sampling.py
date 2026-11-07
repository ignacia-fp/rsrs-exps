import json

import numpy as np

from rsrs_experiments import ParallelSamplingConfig, run_rsrs, sample_operator
from so_kernels import create_operator


def test_parallel_batches_are_ordered_independent_and_validated(tmp_path):
    specification = {
        "operator_type": "biegrid_complex_nonsymmetric",
        "parameters": {"cells_per_axis": 6},
    }
    result = sample_operator(
        ParallelSamplingConfig(
            operator=specification,
            sample_count=23,
            output=tmp_path / "samples",
            workers=2,
            batch_size=10,
            seed=91,
            validation="all",
            validation_workers=2,
            validation_batch_size=7,
        )
    )
    arrays = result.cache.arrays(mmap_mode=None)
    assert arrays["y_test"].shape == (23, 49)
    assert arrays["z_test"].shape == (23, 49)
    assert len(result.report["batch_results"]) == 3
    assert len(result.report["validation_results"]) == 4
    assert max(item["max_error"] for item in result.report["validation_results"]) < 1e-13

    operator = create_operator(specification)
    np.testing.assert_allclose(
        arrays["y_sketch"], (operator.matmat(arrays["y_test"].T)).T, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        arrays["z_sketch"],
        (operator.transpose_operator.matmat(arrays["z_test"].T)).T,
        rtol=2e-13,
        atol=2e-13,
    )
    normalized = arrays["y_test"] / np.linalg.norm(arrays["y_test"], axis=1)[:, None]
    correlations = np.abs(normalized @ normalized.conj().T)
    np.fill_diagonal(correlations, 0.0)
    assert correlations.max() < 0.8

    repeated = sample_operator(
        ParallelSamplingConfig(
            operator=specification,
            sample_count=23,
            output=tmp_path / "samples",
            workers=2,
            batch_size=10,
            seed=91,
            validation="none",
        )
    )
    assert all(item.get("resumed") for item in repeated.report["batch_results"])
    np.testing.assert_array_equal(repeated.cache.arrays()["y_test"], arrays["y_test"])


def test_sampling_manifest_records_reproducibility_inputs(tmp_path):
    specification = {
        "operator_type": "biegrid_real_symmetric",
        "parameters": {"cells_per_axis": 4},
    }
    result = sample_operator(
        ParallelSamplingConfig(
            operator=specification,
            sample_count=10,
            output=tmp_path / "real",
            workers=1,
            seed=12,
        )
    )
    manifest = json.loads((result.cache.path / "manifest.json").read_text())
    assert manifest["operator"] == specification
    assert manifest["sample_count"] == 10
    assert manifest["batch_size"] == 10
    assert manifest["has_z"] is False
    assert manifest["points_file"] == "points.npy"
    assert np.load(result.cache.path / "points.npy").shape == (25, 3)


def test_larger_target_extends_cache_and_smaller_target_never_truncates(tmp_path):
    specification = {
        "operator_type": "biegrid_real_symmetric",
        "parameters": {"cells_per_axis": 4},
    }
    first = sample_operator(ParallelSamplingConfig(
        operator=specification, sample_count=10, output=tmp_path / "extend",
        workers=1, batch_size=10, seed=44,
    ))
    first_rows = first.cache.arrays(mmap_mode=None)["y_test"].copy()
    extended = sample_operator(ParallelSamplingConfig(
        operator=specification, sample_count=30, output=tmp_path / "extend",
        workers=2, batch_size=10, seed=44, validation="sampled",
    ))
    assert extended.report["requested_sample_count"] == 30
    assert extended.report["effective_sample_count"] == 30
    assert sum(not item.get("resumed", False) for item in extended.report["batch_results"]) == 2
    np.testing.assert_array_equal(extended.cache.arrays()["y_test"][:10], first_rows)

    retained = sample_operator(ParallelSamplingConfig(
        operator=specification, sample_count=10, output=tmp_path / "extend",
        workers=1, batch_size=10, seed=44,
    ))
    assert retained.report["requested_sample_count"] == 10
    assert retained.report["effective_sample_count"] == 30
    assert retained.cache.manifest["sample_count"] == 30
    assert all(item.get("resumed") for item in retained.report["batch_results"])


def test_sample_cache_runs_rsrs_and_extracts_errors(tmp_path):
    specification = {
        "operator_type": "biegrid_real_nonsymmetric",
        "parameters": {"cells_per_axis": 6},
    }
    sampled = sample_operator(
        ParallelSamplingConfig(
            operator=specification,
            sample_count=128,
            output=tmp_path / "rsrs_samples",
            workers=2,
            batch_size=10,
            seed=27,
            validation="sampled",
            validation_workers=2,
        )
    )
    result = run_rsrs(
        sampled.cache,
        {
            "oversampling": 8,
            "oversampling_diag_blocks": 8,
            "fixed_rank_sampling_mode": "Constant",
            "tol_id": 8.0,
            "symmetry": "NoSymm",
            "rank_picking": "Min",
            "load_samples": True,
            "save_samples": False,
            "tree": {"max_tree_depth": 4, "leaf_size": 8},
            "threads": {"factorization_threads": 1, "apply_threads": 1},
        },
        output=tmp_path / "rsrs_result",
    )
    assert result.report["status"] == "complete"
    assert result.report["errors"]["factor_relative_2"] < 1e-2
    assert result.report["errors"]["inverse_residual_relative_fro"] < 1e-2
    assert (tmp_path / "rsrs_result" / "rsrs_report.json").is_file()
