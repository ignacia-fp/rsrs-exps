import json

import numpy as np
import pytest

from rsrs_experiments import ParallelSamplingConfig, run_rsrs, sample_operator
from rsrs_experiments.cache import SampleCache
from rsrs_experiments.runner import _apply as runner_apply
from rsrs_experiments.runner import _block_spectral_norm, _errors
from rsrs_experiments import runner
from rsrs_experiments.sampling import _apply as sampling_apply
from so_kernels import create_operator


@pytest.mark.parametrize("apply", [runner_apply, sampling_apply])
def test_operator_matrix_calls_are_limited_to_thirty_columns(apply):
    class RecordingOperator:
        def __init__(self):
            self.widths = []

        def matmat(self, values):
            self.widths.append(values.shape[1])
            return 2.0 * values

    operator = RecordingOperator()
    values = np.arange(4 * 71, dtype=float).reshape(4, 71)
    np.testing.assert_array_equal(apply(operator, values), 2.0 * values)
    assert operator.widths == [30, 30, 11]


def test_block_spectral_norm_matches_dense_singular_value():
    matrix = np.array([[1.0 + 2.0j, 3.0], [-2.0j, 4.0 - 1.0j]])
    estimate, report = _block_spectral_norm(
        lambda values: matrix @ values,
        lambda values: matrix.conj().T @ values,
        dimension=2,
        dtype=np.dtype(np.complex128),
        seed=3,
        iterations=20,
        block_size=2,
        tolerance=1.0e-12,
    )
    np.testing.assert_allclose(estimate, np.linalg.norm(matrix, ord=2), rtol=1.0e-12)
    assert report["iterations"] >= 1


def test_historical_complex_diagnostics_cover_adjoint_and_inverse_routes():
    matrix = np.array([[2.0 + 1.0j, -0.5j], [1.5 - 0.25j, 3.0 - 2.0j]])

    class Operator:
        shape = matrix.shape
        dtype = matrix.dtype
        symmetric = False

        def matvec(self, values):
            return matrix @ values

        def matmat(self, values):
            return matrix @ values

        def transpose_matvec(self, values):
            return matrix.T @ values

        def transpose_matmat(self, values):
            return matrix.T @ values

        def dense(self):
            return matrix

    class Factors:
        dtype = matrix.dtype

        def apply(self, values, *, transpose=False, conjugate=False, inverse=False):
            selected = np.linalg.inv(matrix) if inverse else matrix
            if transpose:
                selected = selected.T
            if conjugate:
                return np.conjugate(selected @ np.conjugate(values))
            return selected @ values

        def matmat(self, values):
            return matrix @ values

        def inverse_matvec(self, values):
            return np.linalg.solve(matrix, values)

        def inverse_matmat(self, values):
            return np.linalg.solve(matrix, values)

    errors, report = _errors(Operator(), Factors(), seed=9, probes=4)
    for name in (
        "app_err_left",
        "app_err_right",
        "app_inv_err_left",
        "app_inv_err_right",
        "adjoint_consistency_error",
        "adjoint_consistency_error_inv",
        "solve_error_rhs",
    ):
        assert errors[name] < 1.0e-12
    assert errors["self_adjoint_apply_error"] > 1.0e-3
    assert report["method"] == "exact_dense"


def test_legacy_multipart_hdf5_cache_is_detected_and_loaded_read_only(tmp_path):
    h5py = pytest.importorskip("h5py")
    sampling = tmp_path / "legacy" / "sampling"
    sampling.mkdir(parents=True)
    rows, dimension = 3, 5
    matrices = {
        name: (
            np.arange(rows * dimension).reshape(rows, dimension)
            + offset
            + 1j * (offset + 0.5)
        ).astype(np.complex128)
        for offset, name in enumerate(("y_test", "y_sketch", "z_test", "z_sketch"))
    }
    for name, matrix in matrices.items():
        for part, columns in enumerate((slice(0, 2), slice(2, 5))):
            values = matrix[:, columns].T.reshape(-1)
            with h5py.File(sampling / f"{name}_file.{part:05d}.h5", "w") as file:
                file.attrs["shape"] = (rows, dimension)
                file.create_dataset("real", data=values.real)
                file.create_dataset("imag", data=values.imag)

    cache = SampleCache(tmp_path / "legacy")
    assert cache.manifest["format"] == "legacy_multipart_hdf5"
    assert cache.manifest["sample_count"] == rows
    bundle = cache.sample_bundle()
    for name, expected in matrices.items():
        np.testing.assert_array_equal(getattr(bundle, name), expected)
    assert not (cache.path / "manifest.json").exists()


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


def test_zero_sampling_target_accepts_dormant_sampling_controls(
    tmp_path, monkeypatch
):
    cache = tmp_path / "existing_samples"
    observed = {}

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("target=0 must not invoke sampling")

    def record_rsrs(cache_path, options, **kwargs):
        observed["cache"] = cache_path
        observed["options"] = options
        return "factorization-result"

    monkeypatch.setattr(runner, "sample_operator", unexpected_sampling)
    monkeypatch.setattr(runner, "run_rsrs", record_rsrs)

    result = runner.run_case(
        {
            "operator": {
                "operator_type": "biegrid_real_symmetric",
                "parameters": {"cells_per_axis": 4},
            },
            "sample_cache": str(cache),
            "sampling": {
                "target": 0,
                "save_samples": True,
                "workers": 8,
                "batch_size": 10,
                "seed": 17,
                "symmetry": "symmetric",
                "validation": "sampled",
                "validation_workers": 4,
                "validation_batch_size": 20,
                "validation_tolerance": 1.0e-6,
            },
            "rsrs": {"tol_id": 1.0e-2},
            "diagnostics": {"gmres": {"mode": "none"}},
            "output": str(tmp_path / "output"),
        }
    )

    assert result == "factorization-result"
    assert observed["cache"] == cache
    assert observed["options"] == {"tol_id": 1.0e-2}


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
            "fixed_rank_sampling_mode": "PerLevel",
            "tol_id": 0.01,
            "symmetry": "NoSymm",
            "rank_picking": "Min",
            "load_samples": True,
            "save_samples": False,
            "tree": {"max_tree_depth": 4, "leaf_size": 8},
            "threads": {"factorization_threads": 1, "apply_threads": 1},
        },
        output=tmp_path / "rsrs_result",
        gmres_options={
            "mode": "both",
            "tolerance": 1.0e-7,
            "max_iterations": 100,
            "rhs_count": 2,
        },
    )
    assert result.report["status"] == "complete"
    assert result.report["errors"]["factor_relative_2"] < 1e-2
    assert result.report["errors"]["inverse_residual_relative_fro"] < 1e-2
    for name in (
        "app_err_left",
        "app_err_right",
        "app_inv_err_left",
        "app_inv_err_right",
        "adjoint_consistency_error",
        "adjoint_consistency_error_inv",
        "self_adjoint_apply_error",
        "solve_error_rhs",
        "cond_rsrs_estimate",
        "tot_num_samples",
        "residual_size",
    ):
        assert name in result.report["errors"]
    assert result.report["rsrs"]["ranks"]
    assert result.report["rsrs"]["active_samples"] > 0
    assert set(result.report["gmres"]["solves"]) == {
        "unpreconditioned",
        "preconditioned",
    }
    assert all(
        solve["recorded_residuals"]
        for solves in result.report["gmres"]["solves"].values()
        for solve in solves
    )
    assert len(result.report["gmres"]["direct_rsrs_inverse"]) == 2
    assert len(result.report["gmres"]["rel_err_approx_inv"]) == 2
    assert (tmp_path / "rsrs_result" / "rsrs_report.json").is_file()
    assert (tmp_path / "rsrs_result" / "errors_report.json").is_file()
    assert (tmp_path / "rsrs_result" / "factorization_report.json").is_file()
    assert (tmp_path / "rsrs_result" / "solves_report.json").is_file()
