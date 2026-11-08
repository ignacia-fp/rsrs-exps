"""On-disk, row-ordered sample cache assembled from worker batches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np


FAMILIES = ("y_test", "y_sketch", "z_test", "z_sketch")


@dataclass(frozen=True)
class SampleCache:
    path: Path

    @property
    def legacy_directory(self) -> Path | None:
        """Return a historical multipart HDF5 directory, if present."""

        candidates = (self.path, self.path / "sampling")
        for candidate in candidates:
            if next(candidate.glob("y_test_file.*.h5"), None) is not None:
                return candidate
        return None

    @property
    def manifest(self) -> dict:
        manifest = self.path / "manifest.json"
        if manifest.is_file():
            return json.loads(manifest.read_text())
        legacy = self.legacy_directory
        if legacy is None:
            raise FileNotFoundError(f"no sample-cache manifest or legacy HDF5 parts in {self.path}")
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("legacy RSRS sample caches require h5py") from exc
        first = sorted(legacy.glob("y_test_file.*.h5"))[0]
        with h5py.File(first, "r") as file:
            shape = tuple(int(value) for value in np.asarray(file.attrs["shape"]).reshape(-1))
            if len(shape) != 2:
                raise ValueError(f"invalid legacy sample shape {shape} in {first}")
            dtype = np.dtype(np.complex128 if "imag" in file else np.float64)
        return {
            "format": "legacy_multipart_hdf5",
            "dimension": shape[1],
            "sample_count": shape[0],
            "dtype": str(dtype),
            "has_z": bool(list(legacy.glob("z_test_file.*.h5"))),
            "legacy_directory": str(legacy),
        }

    @property
    def chunks_dir(self) -> Path:
        return self.path / "chunks"

    def chunk_paths(self) -> list[Path]:
        return sorted(self.chunks_dir.glob("batch_*.npz"))

    def arrays(self, *, mmap_mode: str | None = "r") -> dict[str, np.ndarray]:
        manifest = self.manifest
        if manifest.get("format") == "legacy_multipart_hdf5":
            from bempp_rsrs.cache import load_family

            legacy = Path(manifest["legacy_directory"])
            names = FAMILIES if manifest["has_z"] else FAMILIES[:2]
            return {name: load_family(legacy, f"{name}_file") for name in names}
        names = FAMILIES if manifest["has_z"] else FAMILIES[:2]
        return {
            name: np.load(self.path / "arrays" / f"{name}.npy", mmap_mode=mmap_mode)
            for name in names
        }

    def sample_bundle(self):
        from bempp_rsrs import SampleBundle, load_sample_bundle

        legacy = self.legacy_directory
        if legacy is not None and not (self.path / "manifest.json").is_file():
            return load_sample_bundle(legacy, require_z=self.manifest["has_z"])

        values = self.arrays()
        return SampleBundle(
            values["y_test"],
            values["y_sketch"],
            values.get("z_test"),
            values.get("z_sketch"),
        )

    def iter_chunks(self) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
        for path in self.chunk_paths():
            with np.load(path) as data:
                yield path, {name: np.asarray(data[name]) for name in data.files}


def assemble_cache(path: Path, batch_results: list[dict]) -> SampleCache:
    ordered = sorted(batch_results, key=lambda item: item["row_start"])
    if not ordered or ordered[0]["row_start"] != 0:
        raise RuntimeError("sample batches do not start at row zero")
    expected = 0
    for item in ordered:
        if item["row_start"] != expected:
            raise RuntimeError(
                f"sample batch gap or overlap at row {expected}: {item['row_start']}"
            )
        expected = item["row_stop"]

    total = expected
    dimension = int(ordered[0]["dimension"])
    dtype = np.dtype(ordered[0]["dtype"])
    has_z = bool(ordered[0]["has_z"])
    names = FAMILIES if has_z else FAMILIES[:2]
    arrays_dir = path / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: np.lib.format.open_memmap(
            arrays_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(total, dimension)
        )
        for name in names
    }
    for item in ordered:
        with np.load(item["chunk"]) as chunk:
            row_slice = slice(item["row_start"], item["row_stop"])
            for name in names:
                values = np.asarray(chunk[name])
                if values.shape != (item["row_stop"] - item["row_start"], dimension):
                    raise RuntimeError(f"{item['chunk']}: invalid {name} shape {values.shape}")
                outputs[name][row_slice] = values
    for output in outputs.values():
        output.flush()
    return SampleCache(path)
