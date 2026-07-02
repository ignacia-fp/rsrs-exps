import json
import os
import re
from pathlib import Path

import numpy as np


def _rust_sci_format(x):
    base, exp = f"{x:e}".split("e")
    base = base.rstrip("0").rstrip(".")
    return f"{base}e{int(exp)}"


def results_base_path(config):
    folder = config.generate_folder_name()
    subfolder = config.generate_sub_folder_name()
    return Path(os.getcwd()) / "results" / folder / subfolder


def grid_path(config):
    return results_base_path(config) / "grid.msh"


def expected_solve_vectors_path(config, tol):
    if tol is None:
        raise ValueError("tol must be provided when constructing an expected solve-vectors path.")
    return results_base_path(config) / f"solve_vectors_{_rust_sci_format(tol)}.h5"


def load_all_stats(config, kind="error"):
    """
    Load all JSON stats files of a given kind from the scenario's result directory.

    Parameters
    ----------
    kind : str
        Type of statistics to load. Must be one of: 'error', 'time', 'rank',
        or 'condition_number'. Corresponds to files named like
        'error_stats_{tol}.json', etc.

    Returns
    -------
    List[dict]
        List of dictionaries containing parsed JSON data, each augmented with a
        'tolerance' key.
    """
    valid_kinds = {"error", "time", "rank", "condition_number"}
    if kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {valid_kinds}.")

    base_path = results_base_path(config)

    stats = []
    pattern = re.compile(fr"{kind}_stats_(.+)\.json")

    for file in base_path.iterdir():
        if file.is_file() and file.name.startswith(f"{kind}_stats_") and file.suffix == ".json":
            match = pattern.match(file.name)
            if match:
                tol_str = match.group(1)
                try:
                    tolerance = float(tol_str)
                except ValueError:
                    continue

                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["tolerance"] = tolerance
                    stats.append(data)

    return stats


def select_error_stat(config, tol=None):
    all_stats = load_all_stats(config, kind="error")
    all_stats.sort(key=lambda d: float(d["tolerance"]))

    if not all_stats:
        raise ValueError("No error stats loaded.")

    if tol is None:
        return all_stats[0]

    for data in all_stats:
        if data is None:
            continue
        try:
            if np.isclose(float(data["tolerance"]), float(tol)):
                return data
        except (KeyError, TypeError, ValueError):
            continue

    raise ValueError(f"No error stats found for tolerance {tol}.")


def solve_vectors_path(config, tol=None):
    base_path = results_base_path(config)
    if not base_path.exists():
        return expected_solve_vectors_path(config, tol) if tol is not None else None

    stat = select_error_stat(config, tol=tol)
    solves = stat.get("solves", {})
    vectors_file = solves.get("vectors_file")
    if not vectors_file:
        return expected_solve_vectors_path(config, tol) if tol is not None else None
    return results_base_path(config) / vectors_file


def decode_legacy_vectors(vectors):
    decoded = []
    for vec in vectors or []:
        arr = np.asarray(vec)
        if arr.ndim == 1:
            decoded.append(arr.astype(float))
        elif arr.ndim == 2 and arr.shape[-1] == 2:
            decoded.append(arr[:, 0] + 1j * arr[:, 1])
        else:
            raise ValueError(f"Unexpected solution shape {arr.shape}")
    return decoded


def load_solution_group(config, stat, group_name):
    solves = stat.get("solves", {})
    vectors_file = solves.get("vectors_file")

    if vectors_file:
        import h5py

        vector_path = results_base_path(config) / vectors_file
        if not vector_path.exists():
            raise FileNotFoundError(
                f"Expected solve-vector file at {vector_path}, but it doesn't exist."
            )

        with h5py.File(vector_path, "r") as h5f:
            if group_name not in h5f:
                return []

            group = h5f[group_name]
            real = np.array(group["real"][:], copy=True)
            if "imag" in group:
                imag = np.array(group["imag"][:], copy=True)
                data = real + 1j * imag
            else:
                data = real

        return [data[i].copy() for i in range(data.shape[0])]

    return decode_legacy_vectors(solves.get(group_name, []))
