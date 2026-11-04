from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    kernel: str
    geometry: str
    mesh: str | None = None
    mesh_width: float | None = None
    refinement_level: int | None = None
    kappa: float | None = None
    precision: str = "double"
    assembler: str = "exafmm"
    expected_dofs: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def resolved(self, base: Path | None = None) -> "CaseSpec":
        if self.mesh is None:
            return self
        path = Path(self.mesh).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        values = asdict(self)
        values["mesh"] = str(path.resolve())
        return CaseSpec(**values)


@dataclass(frozen=True)
class SamplingSpec:
    samples: int
    workers: int = 1
    threads_per_worker: int = 1
    batch_size: int = 10
    seed: int = 20261104
    validation: str = "full"
    validation_tolerance: float = 5e-7

    def __post_init__(self) -> None:
        if self.samples < 1 or self.workers < 1 or self.batch_size < 1:
            raise ValueError("samples, workers, and batch_size must be positive")
        if self.validation not in {"full", "spot", "none"}:
            raise ValueError("validation must be full, spot, or none")
