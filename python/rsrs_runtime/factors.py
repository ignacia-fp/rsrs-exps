"""SciPy-facing wrapper around native RSRS factors."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator


class NativeFactors(Protocol):
    """Minimal interface that the PyO3 RSRS factor class must implement."""

    @property
    def dimension(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...

    def apply(
        self,
        values: NDArray[np.generic],
        *,
        transpose: bool,
        conjugate: bool,
        inverse: bool,
    ) -> NDArray[np.generic]: ...


class RsrsFactorOperator(LinearOperator):
    """Approximate matrix or inverse exposed as a SciPy LinearOperator."""

    def __init__(self, factors: NativeFactors, *, inverse: bool = False):
        self._factors = factors
        self._inverse = inverse
        dimension = int(factors.dimension)
        super().__init__(dtype=np.dtype(factors.dtype), shape=(dimension, dimension))

    def _apply(
        self,
        values: NDArray[np.generic],
        *,
        transpose: bool,
        conjugate: bool,
    ) -> NDArray[np.generic]:
        array = np.asarray(values, dtype=self.dtype)
        result = self._factors.apply(
            np.ascontiguousarray(array),
            transpose=transpose,
            conjugate=conjugate,
            inverse=self._inverse,
        )
        return np.asarray(result, dtype=self.dtype)

    def _matvec(self, vector: NDArray[np.generic]) -> NDArray[np.generic]:
        return self._apply(vector, transpose=False, conjugate=False)

    def _matmat(self, matrix: NDArray[np.generic]) -> NDArray[np.generic]:
        return self._apply(matrix, transpose=False, conjugate=False)

    def _rmatvec(self, vector: NDArray[np.generic]) -> NDArray[np.generic]:
        return self._apply(vector, transpose=True, conjugate=True)

    def _adjoint(self) -> LinearOperator:
        parent = self

        def matvec(vector: NDArray[np.generic]) -> NDArray[np.generic]:
            return parent._apply(vector, transpose=True, conjugate=True)

        def rmatvec(vector: NDArray[np.generic]) -> NDArray[np.generic]:
            return parent._apply(vector, transpose=False, conjugate=False)

        return LinearOperator(
            shape=self.shape,
            dtype=self.dtype,
            matvec=matvec,
            rmatvec=rmatvec,
        )

