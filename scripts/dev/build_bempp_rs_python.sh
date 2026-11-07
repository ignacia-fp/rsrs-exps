#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bempp_root="${BEMPP_RS_ROOT:-${repo_root}/../bempp-rs}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This script must run inside the Linux development container." >&2
    exit 2
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "Expected x86_64 Linux to match Cardinal; found $(uname -m)." >&2
    exit 2
fi

if [[ ! -d "${bempp_root}" ]]; then
    echo "Missing standalone Bempp-rs checkout at ${bempp_root}." >&2
    echo "Set BEMPP_RS_ROOT to override the default sibling checkout." >&2
    exit 2
fi

export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/rsrs-venv}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/opt/cargo-target}"
export RUSTUP_TOOLCHAIN="${BEMPP_CFFI_TOOLCHAIN:-nightly-2026-10-15}"

cd "${bempp_root}"

cargo check --locked --lib
maturin develop --release --locked
python -m pytest \
    python/test/test_fmm_laplace.py \
    python/test/test_fmm_single_layer.py \
    -q
