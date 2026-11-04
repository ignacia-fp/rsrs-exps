from __future__ import annotations

import numpy as np

_STREAMS = {"y": 0x59, "z": 0x5A}


def test_vector(seed: int, stream: str, row: int, dimension: int, dtype=np.float64) -> np.ndarray:
    """Return a deterministic Gaussian vector for one global sample row.

    The value is independent of worker count, chunking, and execution order.
    """
    if stream not in _STREAMS:
        raise ValueError(f"unknown stream {stream!r}")
    seq = np.random.SeedSequence([int(seed), _STREAMS[stream], int(row)])
    return np.random.default_rng(seq).standard_normal(dimension).astype(dtype, copy=False)


def test_block(seed: int, stream: str, rows: range | list[int], dimension: int, dtype=np.float64) -> np.ndarray:
    return np.column_stack([test_vector(seed, stream, r, dimension, dtype) for r in rows])


def contiguous_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    quotient, remainder = divmod(total, workers)
    result, start = [], 0
    for worker in range(workers):
        stop = start + quotient + int(worker < remainder)
        result.append((start, stop))
        start = stop
    return result
