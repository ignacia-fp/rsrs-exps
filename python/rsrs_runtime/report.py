"""Structured reports returned by the future native RSRS binding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleUsage:
    y_rows: int
    z_rows: int | None
    rows_required: int
    rows_loaded: int = 0
    rows_generated_in_rust: int = 0


@dataclass(frozen=True)
class TimingReport:
    total_ms: float
    factorization_ms: float
    sample_loading_ms: float = 0.0
    native_sampling_ms: float = 0.0
    id_ms: float = 0.0
    lu_ms: float = 0.0
    diagonal_extraction_ms: float = 0.0
    factor_apply_ms: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class FactorizationReport:
    """Stable Python representation of intrinsic RSRS statistics."""

    configuration: dict[str, Any]
    effective_leaf_size: int
    ranks: list[int]
    box_sizes: list[int]
    near_field_sizes: list[int]
    sample_usage: SampleUsage
    timings: TimingReport
    level_effort: list[dict[str, Any]] = field(default_factory=list)
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    limiting_factors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

