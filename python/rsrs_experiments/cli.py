"""Command-line interface for the reusable RSRS experiment pipeline."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from .cache import SampleCache
from .configuration import load_definition, option_guide, render_script
from .plotting import plot_error_summary
from .runner import run_case, run_rsrs
from .sampling import ParallelSamplingConfig, sample_operator


def _json(value: str) -> dict:
    if value.lstrip().startswith(("{", "[")):
        return json.loads(value)
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    return json.loads(path.read_text()) if is_file else json.loads(value)


def _sample(args: argparse.Namespace) -> int:
    spec = {"operator_type": args.operator, "parameters": _json(args.parameters)}
    result = sample_operator(
        ParallelSamplingConfig(
            operator=spec,
            sample_count=args.samples,
            output=args.output,
            workers=args.workers,
            batch_size=args.batch_size,
            seed=args.seed,
            symmetry=args.symmetry,
            resume=not args.no_resume,
            validation=args.validate,
            validation_workers=args.validation_workers,
            validation_batch_size=args.validation_batch_size,
            validation_tolerance=args.validation_tolerance,
        )
    )
    print(result.cache.path)
    return 0


def _run(args: argparse.Namespace) -> int:
    result = run_rsrs(
        SampleCache(args.cache),
        _json(args.options),
        output=args.output,
        error_probes=args.error_probes,
        error_seed=args.error_seed,
        points=args.points,
    )
    print(json.dumps(result.report["errors"], indent=2))
    return 0


def _run_case(args: argparse.Namespace) -> int:
    result = run_case(_json(args.case))
    report = result.report if hasattr(result, "report") else result
    print(json.dumps(report, indent=2))
    return 0


def _plot(args: argparse.Namespace) -> int:
    print(plot_error_summary(args.report, args.output))
    return 0


def _options(args: argparse.Namespace) -> int:
    print(option_guide(), end="")
    return 0


def _render(args: argparse.Namespace) -> int:
    created = date.fromisoformat(args.date) if args.date else None
    path = render_script(
        load_definition(args.config),
        args.selection,
        output_dir=args.output_dir,
        script_name=args.name,
        kind=args.kind,
        creation_date=created,
        executable=args.executable,
    )
    print(path)
    return 0


def _build(args: argparse.Namespace) -> int:
    """Render using generation settings stored inside one JSON/Python file."""
    definition = load_definition(args.config)
    raw = definition.get("GENERATE", {})
    jobs = raw if isinstance(raw, list) else [raw]
    if not jobs:
        raise ValueError("GENERATE must contain at least one generation request")
    for item in jobs:
        settings = dict(item)
        required = [key for key in ("selection", "name") if not settings.get(key)]
        if required:
            raise ValueError(
                f"GENERATE is missing required field(s): {', '.join(required)}"
            )
        created = date.fromisoformat(settings["date"]) if settings.get("date") else None
        path = render_script(
            definition,
            settings["selection"],
            output_dir=Path(settings.get("output_dir", "generated_jobs")),
            script_name=settings["name"],
            kind=settings.get("kind", "sh"),
            creation_date=created,
            executable=settings.get("executable", "rsrs-experiment run-case"),
        )
        print(path)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rsrs-experiment")
    commands = root.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="generate and optionally validate samples")
    sample.add_argument("--operator", required=True)
    sample.add_argument("--parameters", default="{}", help="JSON object or JSON file")
    sample.add_argument("--samples", type=int, required=True)
    sample.add_argument("--workers", type=int, default=1)
    sample.add_argument("--batch-size", type=int, default=10)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--symmetry", choices=("auto", "symmetric", "nonsymmetric"), default="auto")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--no-resume", action="store_true")
    sample.add_argument("--validate", choices=("none", "sampled", "all"), default="none")
    sample.add_argument("--validation-workers", type=int)
    sample.add_argument("--validation-batch-size", type=int)
    sample.add_argument("--validation-tolerance", type=float, default=5.0e-4)
    sample.set_defaults(handler=_sample)
    run = commands.add_parser("run", help="factorize a saved sample cache and report errors")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--options", required=True, help="RSRS options JSON object or file")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--error-probes", type=int, default=4)
    run.add_argument("--error-seed", type=int, default=1701)
    run.add_argument("--points", type=Path)
    run.set_defaults(handler=_run)
    case = commands.add_parser(
        "run-case", help="run one JSON-defined sampling/factorisation/solve case"
    )
    case.add_argument("case", help="JSON object or path to a JSON case")
    case.set_defaults(handler=_run_case)
    plot = commands.add_parser("plot-errors", help="plot errors from an RSRS JSON report")
    plot.add_argument("--report", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.set_defaults(handler=_plot)
    options = commands.add_parser("options", help="print the numbered option guide")
    options.set_defaults(handler=_options)
    build = commands.add_parser(
        "build", help="build a script using settings contained in one configuration file"
    )
    build.add_argument("config", type=Path, help="fill-in JSON or Python configuration")
    build.set_defaults(handler=_build)
    render = commands.add_parser("render", help="render a named test or bundle as a dated script")
    render.add_argument("--config", type=Path, required=True, help="Python or JSON experiment dictionary")
    render.add_argument("--selection", required=True, help="name in TESTS or BUNDLES")
    render.add_argument("--name", required=True, help="user-facing filename prefix")
    render.add_argument("--kind", choices=("sh", "sbatch"), default="sh")
    render.add_argument("--output-dir", type=Path, default=Path("generated_jobs"))
    render.add_argument("--date", help="override creation date (YYYY-MM-DD)")
    render.add_argument("--executable", default="rsrs-experiment run-case", help="driver command")
    render.set_defaults(handler=_render)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.handler(args))
