"""Optional publication-neutral plots generated from JSON experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Any


def _report(value: Path | Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(value).read_text()) if isinstance(value, Path) else dict(value)


def plot_error_summary(report: Path | Mapping[str, Any], output: Path) -> Path:
    import matplotlib.pyplot as plt

    values = _report(report)
    errors = values.get("errors", {})
    names = list(errors)
    data = [errors[name] for name in names]
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.bar(range(len(names)), data, color="black")
    axis.set_yscale("log")
    axis.set_ylabel("Relative error")
    axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names])
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output
