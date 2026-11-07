import numpy as np
import pytest

from python.rsrs_runtime import RsrsFactorOperator, RsrsFactorizationOptions, SampleBundle


class DenseFactors:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.dimension = self.matrix.shape[0]
        self.dtype = self.matrix.dtype

    def apply(self, values, *, transpose, conjugate, inverse):
        matrix = self.matrix
        if transpose:
            matrix = matrix.T
        if conjugate:
            matrix = matrix.conj()
        if inverse:
            return np.linalg.solve(matrix, values)
        return matrix @ values


def test_sample_bundle_accepts_consistent_complex_nonsymmetric_samples():
    tests = np.eye(3, dtype=np.complex128)
    matrix = np.array([[1, 2j, 0], [3, 4, 5j], [0, 6, 7]], dtype=np.complex128)
    bundle = SampleBundle(
        y_test=tests,
        y_sketch=(matrix @ tests.T).T,
        z_test=tests,
        z_sketch=(matrix.T @ tests.T).T,
    )
    assert bundle.dimension == 3
    assert bundle.sample_count == 3
    assert bundle.has_z


def test_sample_bundle_rejects_partial_z_family():
    with pytest.raises(ValueError, match="both be present"):
        SampleBundle(np.ones((2, 3)), np.ones((2, 3)), z_test=np.ones((2, 3)))


def test_factor_operator_matches_dense_matrix_inverse_and_adjoint():
    matrix = np.array([[2 + 1j, 1], [3j, 4 - 2j]], dtype=np.complex128)
    vector = np.array([1 - 1j, 2 + 0.5j])
    factors = DenseFactors(matrix)

    operator = RsrsFactorOperator(factors)
    inverse = RsrsFactorOperator(factors, inverse=True)

    np.testing.assert_allclose(operator @ vector, matrix @ vector)
    np.testing.assert_allclose(operator.H @ vector, matrix.conj().T @ vector)
    np.testing.assert_allclose(inverse @ vector, np.linalg.solve(matrix, vector))


def test_legacy_configuration_maps_threads_and_automatic_leaf_size():
    options = RsrsFactorizationOptions.from_mapping(
        {
            "oversampling": 8,
            "oversampling_diag_blocks": 16,
            "min_num_samples": 0,
            "initial_num_samples": 0,
            "fixed_rank_sampling_mode": "PerLevel",
            "nonsymmetric_id_combination": "Concat",
            "shift": {"type": "False"},
            "null_method": "Projection",
            "qr_method": "RRQR",
            "near_block_extraction_method": "LuLstSq",
            "diag_block_extraction_method": "LuLstSq",
            "lu_pivot_method": {"type": "LuHybrid", "value": 0},
            "diag_pivot_method": {"type": "LuHybrid", "value": 0},
            "tol_null": 1e-16,
            "tol_id": 1e-2,
            "tol_ext_near": 1e-10,
            "tol_diag_ext": 1e-16,
            "min_rank": 1,
            "min_level": 1,
            "symmetry": "NoSymm",
            "rank_picking": "Min",
            "fact_type": "Joint",
            "save_samples": True,
            "load_samples": True,
            "num_threads": 56,
            "flush_factors": False,
            "store_far": False,
            "symmetric": None,
        },
        scenario={"max_tree_depth": 16},
    )

    assert options.tree.max_tree_depth == 16
    assert options.tree.leaf_size is None
    assert options.threads.factorization_threads == 56
    assert options.threads.apply_threads == 56


def test_explicit_leaf_size_and_split_thread_controls_override_legacy_value():
    options = RsrsFactorizationOptions.from_mapping(
        {
            "num_threads": 56,
            "factorization_threads": 1,
            "apply_threads": 32,
            "leaf_size": 400,
        }
    )
    assert options.tree.leaf_size == 400
    assert options.threads.factorization_threads == 1
    assert options.threads.apply_threads == 32
