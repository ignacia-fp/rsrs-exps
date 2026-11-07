"""Validated in-memory sample bundles passed to the native RSRS wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SampleConvention(str, Enum):
    """Meaning of the second (``z``) sketch family."""

    TRANSPOSE = "transpose"


def _sample_matrix(name: str, value: ArrayLike) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {array.dtype}")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class SampleBundle:
    """RSRS test/sketch arrays with one sample per row.

    For nonsymmetric operators all four arrays are required. Symmetric and
    Hermitian configurations can omit both ``z`` arrays.
    """

    y_test: ArrayLike
    y_sketch: ArrayLike
    z_test: ArrayLike | None = None
    z_sketch: ArrayLike | None = None
    z_convention: SampleConvention = SampleConvention.TRANSPOSE

    def __post_init__(self) -> None:
        object.__setattr__(self, "y_test", _sample_matrix("y_test", self.y_test))
        object.__setattr__(self, "y_sketch", _sample_matrix("y_sketch", self.y_sketch))
        if self.z_test is not None:
            object.__setattr__(self, "z_test", _sample_matrix("z_test", self.z_test))
        if self.z_sketch is not None:
            object.__setattr__(self, "z_sketch", _sample_matrix("z_sketch", self.z_sketch))
        self.validate()

    @property
    def dimension(self) -> int:
        return int(self.y_test.shape[1])

    @property
    def sample_count(self) -> int:
        return int(self.y_test.shape[0])

    @property
    def has_z(self) -> bool:
        return self.z_test is not None

    def validate(self, *, require_z: bool = False) -> None:
        if self.y_test.shape != self.y_sketch.shape:
            raise ValueError(
                "y_test and y_sketch must have identical shapes; "
                f"got {self.y_test.shape} and {self.y_sketch.shape}"
            )
        if self.y_test.dtype != self.y_sketch.dtype:
            raise TypeError(
                "y_test and y_sketch must have identical dtypes; "
                f"got {self.y_test.dtype} and {self.y_sketch.dtype}"
            )

        one_z_missing = (self.z_test is None) != (self.z_sketch is None)
        if one_z_missing:
            raise ValueError("z_test and z_sketch must either both be present or both be absent")
        if require_z and self.z_test is None:
            raise ValueError("nonsymmetric RSRS factorisation requires z samples")

        if self.z_test is not None and self.z_sketch is not None:
            if self.z_test.shape != self.z_sketch.shape:
                raise ValueError(
                    "z_test and z_sketch must have identical shapes; "
                    f"got {self.z_test.shape} and {self.z_sketch.shape}"
                )
            if self.z_test.shape != self.y_test.shape:
                raise ValueError(
                    "y and z families must have identical shapes; "
                    f"got {self.y_test.shape} and {self.z_test.shape}"
                )
            if self.z_test.dtype != self.y_test.dtype or self.z_sketch.dtype != self.y_test.dtype:
                raise TypeError("all sample arrays must have the same dtype")

