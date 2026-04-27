#!/usr/bin/env python3

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsrs_config import RSRSBenchmarkConfig


def harden_runtime_resolution(script_path: Path) -> None:
    original_lines = script_path.read_text().splitlines()

    pinned_runtime = [
        'REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"',
        'export PYENV_VERSION="${PYENV_VERSION:-kifmm-bempp-env}"',
        'if command -v pyenv >/dev/null 2>&1; then',
        '  export RSRS_EXPS_PYTHON="${RSRS_EXPS_PYTHON:-$(pyenv which python)}"',
        "fi",
        'export PYTHONPATH="$REPO_ROOT/external/bempp-cl:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
        'if [ -z "${BEMPP_KIFMM_LIBRARY:-}" ]; then',
        '  for candidate in \\',
        '    "$REPO_ROOT/external/kifmm-for-rsrs/target/release/libkifmm.dylib" \\',
        '    "$REPO_ROOT/external/kifmm-for-rsrs/target/release/libkifmm.so" \\',
        '    "$REPO_ROOT/external/kifmm-for-rsrs/target/maturin/libkifmm.dylib" \\',
        '    "$REPO_ROOT/external/kifmm-for-rsrs/target/maturin/libkifmm.so"; do',
        '    if [ -f "$candidate" ]; then',
        '      export BEMPP_KIFMM_LIBRARY="$candidate"',
        "      break",
        "    fi",
        "  done",
        "fi",
        'if [ -z "${BEMPP_KIFMM_LIBRARY:-}" ]; then',
        '  echo "Could not find a local KiFMM shared library under external/kifmm-for-rsrs/target." >&2',
        "  exit 1",
        "fi",
        'KIFMM_LIB_DIR="$(dirname "$BEMPP_KIFMM_LIBRARY")"',
        'export DYLD_LIBRARY_PATH="$KIFMM_LIB_DIR:$KIFMM_LIB_DIR/deps${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"',
        'export LD_LIBRARY_PATH="$KIFMM_LIB_DIR:$KIFMM_LIB_DIR/deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        'echo "RSRS_EXPS_PYTHON=${RSRS_EXPS_PYTHON:-<unset>}"',
        'echo "BEMPP_KIFMM_LIBRARY=$BEMPP_KIFMM_LIBRARY"',
        '"${RSRS_EXPS_PYTHON:-python}" - <<\'PY\'',
        "import sys",
        "import bempp_cl",
        'print(f"Python executable: {sys.executable}")',
        'print(f"bempp_cl module: {bempp_cl.__file__}")',
        "PY",
    ]

    script_path.write_text(
        "\n".join([original_lines[0], *pinned_runtime, *original_lines[1:]]) + "\n"
    )


def main() -> None:
    output_script = REPO_ROOT / "run_test_bempp_kifmm_sphere_5k.sh"

    config = RSRSBenchmarkConfig(
        operator_type=1,  # BemppClLaplaceSingleLayer
        precision=1,  # Double
        h=0.079,  # ~4990 DP0 dofs on the sphere with the current bempp-cl mesher
        dim_arg_type=2,  # Meshwidth
        geometry=0,  # SphereSurface
        id_tols=[8],  # fixed-rank smoke test
        solve=False,
        plot=False,
        dense_errors=False,
        results_output=0,  # All
        rank_picking=0,  # Min
        fact_type=1,  # Joint
        max_tree_depth=2,
        assembler=2,  # KiFMM
        symmetry=1,  # Symmetric
        save_samples=False,
        load_samples=False,
        num_threads=8,
    )

    config.generate_bash_script(str(output_script))
    harden_runtime_resolution(output_script)
    print(f"Wrote {output_script}")
    print(f"Sample cache path: {config.sample_storage_path()}")


if __name__ == "__main__":
    main()
