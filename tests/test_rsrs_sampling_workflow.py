from pathlib import Path

import numpy as np

from python.rsrs_workflow.random import contiguous_ranges, test_vector as gaussian_test_vector
from python.rsrs_workflow.storage import parse_rows


def test_global_row_rng_is_worker_independent():
    direct = gaussian_test_vector(17, "y", 12, 8)
    after_other_rows = [gaussian_test_vector(17, "y", row, 8) for row in range(20)][12]
    assert np.array_equal(direct, after_other_rows)
    assert not np.array_equal(direct, gaussian_test_vector(17, "z", 12, 8))


def test_ranges_are_contiguous():
    assert contiguous_ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_parse_sparse_rows():
    assert parse_rows("1,3:6,9", 10) == [1, 3, 4, 5, 9]
