"""Versioned Python configuration for native and sample-fed RSRS runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ThreadOptions:
    """Independent resource settings for factor construction and application."""

    factorization_threads: int = 1
    apply_threads: int = 1

    def __post_init__(self) -> None:
        if self.factorization_threads < 1:
            raise ValueError("factorization_threads must be positive")
        if self.apply_threads < 1:
            raise ValueError("apply_threads must be positive")


@dataclass(frozen=True)
class TreeOptions:
    """Octree controls shared by native and Python-driven factorisations."""

    max_tree_depth: int = 6
    leaf_size: int | None = None

    def __post_init__(self) -> None:
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be positive")
        if self.leaf_size is not None and self.leaf_size < 1:
            raise ValueError("leaf_size must be positive when provided")


@dataclass(frozen=True)
class RsrsFactorizationOptions:
    """Complete factorisation options accepted by the Python RSRS API.

    Structured method settings such as ``shift`` and ``qr_method`` retain the
    JSON representation already emitted by :mod:`rsrs_config`. This keeps the
    compatibility adapter lossless while the native PyO3 enums are introduced.
    """

    schema_version: int = SCHEMA_VERSION
    oversampling: int = 8
    oversampling_diag_blocks: int = 16
    min_num_samples: int = 0
    initial_num_samples: int = 0
    fixed_rank_sampling_mode: str = "PerLevel"
    nonsymmetric_id_combination: str = "Concat"
    run_seed: int | None = None
    shift: Any = field(default_factory=lambda: {"type": "False"})
    null_method: str = "Projection"
    qr_method: Any = "RRQR"
    near_block_extraction_method: str = "LuLstSq"
    diag_block_extraction_method: str = "LuLstSq"
    lu_pivot_method: Any = field(default_factory=lambda: {"type": "LuHybrid", "value": 0})
    diag_pivot_method: Any = field(default_factory=lambda: {"type": "LuHybrid", "value": 0})
    tol_null: float = 1e-16
    tol_id: float = 1e-2
    tol_ext_near: float = 1e-16
    tol_diag_ext: float = 1e-16
    min_rank: int = 1
    min_level: int = 1
    symmetry: str = "NoSymm"
    rank_picking: str = "Min"
    fact_type: str = "Joint"
    save_samples: bool = False
    load_samples: bool = True
    sample_storage_dir: str | None = None
    flush_factors: bool = False
    store_far: bool = False
    tree: TreeOptions = field(default_factory=TreeOptions)
    threads: ThreadOptions = field(default_factory=ThreadOptions)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RSRS configuration schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        for name in (
            "oversampling",
            "oversampling_diag_blocks",
            "min_num_samples",
            "initial_num_samples",
            "min_rank",
            "min_level",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        scenario: Mapping[str, Any] | None = None,
    ) -> "RsrsFactorizationOptions":
        """Read the shared schema or a legacy ``rsrs_config.rsrs_args`` map.

        Legacy ``num_threads`` configures both new thread settings. A caller
        can override either one with ``factorization_threads`` or
        ``apply_threads``. Tree depth is accepted from the scenario map because
        that is where the current experiment configuration stores it.
        """

        data = dict(values)
        schema_version = int(data.pop("schema_version", SCHEMA_VERSION))

        legacy_threads = int(data.pop("num_threads", 1))
        thread_data = dict(data.pop("threads", {}))
        factorization_threads = int(
            data.pop(
                "factorization_threads",
                thread_data.pop("factorization_threads", legacy_threads),
            )
        )
        apply_threads = int(
            data.pop("apply_threads", thread_data.pop("apply_threads", legacy_threads))
        )
        if thread_data:
            raise ValueError(f"unknown thread options: {sorted(thread_data)}")

        scenario_data = dict(scenario or {})
        tree_data = dict(data.pop("tree", {}))
        max_tree_depth = int(
            data.pop(
                "max_tree_depth",
                tree_data.pop("max_tree_depth", scenario_data.get("max_tree_depth", 6)),
            )
        )
        leaf_size = data.pop(
            "leaf_size",
            tree_data.pop("leaf_size", scenario_data.get("leaf_size")),
        )
        if tree_data:
            raise ValueError(f"unknown tree options: {sorted(tree_data)}")

        # Historical experiment-only fields are not factorisation options.
        data.pop("symmetric", None)

        known = {item.name for item in __import__("dataclasses").fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown RSRS options: {unknown}")

        return cls(
            schema_version=schema_version,
            tree=TreeOptions(max_tree_depth=max_tree_depth, leaf_size=leaf_size),
            threads=ThreadOptions(
                factorization_threads=factorization_threads,
                apply_threads=apply_threads,
            ),
            **data,
        )

    def to_mapping(self, *, legacy: bool = False) -> dict[str, Any]:
        """Serialize the shared schema or the current legacy argument shape."""

        data = asdict(self)
        if not legacy:
            return data

        tree = data.pop("tree")
        threads = data.pop("threads")
        data.pop("schema_version")
        data["num_threads"] = threads["factorization_threads"]
        data["apply_threads"] = threads["apply_threads"]
        data["max_tree_depth"] = tree["max_tree_depth"]
        if tree["leaf_size"] is not None:
            data["leaf_size"] = tree["leaf_size"]
        return data

