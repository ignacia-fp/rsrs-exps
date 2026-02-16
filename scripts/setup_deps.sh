#!/usr/bin/env bash
set -euo pipefail

DEPS_DIR="${DEPS_DIR:-/deps}"
WORKSPACE="${WORKSPACE:-/workspace}"

echo "[setup] pwd before cd: $(pwd)"
echo "[setup] Using WORKSPACE=$WORKSPACE"
echo "[setup] Using DEPS_DIR=$DEPS_DIR"

mkdir -p "$DEPS_DIR"
cd "$WORKSPACE"
echo "[setup] pwd after cd: $(pwd)"

# --- Python venv bootstrap ---
if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] Creating Python venv in $WORKSPACE/.venv"
  rm -rf .venv
  python3.10 -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
  echo "[setup] ERROR: venv activate script not found at $WORKSPACE/.venv/bin/activate"
  ls -la .venv || true
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Python: $(python --version)"
python -m pip install -U pip >/dev/null
python -m pip install -U uv maturin >/dev/null

export PYTHON_INCLUDE_DIR="$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
export PYTHON_LIB_DIR="$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')")"

echo "[setup] PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR"
echo "[setup] PYTHON_LIB_DIR=$PYTHON_LIB_DIR"

# --- Ensure gmsh in venv points to system gmsh (idempotent) ---
SYS_GMSH="$(command -v gmsh || true)"
if [ -z "$SYS_GMSH" ]; then
  echo "[setup] ERROR: gmsh not found in PATH (install via apt in Dockerfile: gmsh + runtime libs)"
  exit 1
fi

TARGET_GMSH="$WORKSPACE/.venv/bin/gmsh"
if [ "$(readlink -f "$TARGET_GMSH" 2>/dev/null || true)" != "$SYS_GMSH" ]; then
  echo "[setup] Linking system gmsh into venv: $SYS_GMSH"
  rm -f "$TARGET_GMSH"
  ln -s "$SYS_GMSH" "$TARGET_GMSH"
fi

# --- exafmm-t ---
if [ ! -d "$DEPS_DIR/exafmm-t" ]; then
  echo "[setup] Cloning exafmm-t into $DEPS_DIR"
  git clone https://github.com/exafmm/exafmm-t.git "$DEPS_DIR/exafmm-t"
fi

if [ ! -f "$DEPS_DIR/exafmm-t/.installed.ok" ]; then
  echo "[setup] Building & installing exafmm-t"
  pushd "$DEPS_DIR/exafmm-t" >/dev/null
  CFLAGS="-O3" CXXFLAGS="-O3" ./configure
  make -j"$(nproc)"
  make install
  python setup.py install
  popd >/dev/null
  touch "$DEPS_DIR/exafmm-t/.installed.ok"
fi

# ---- bempp-cl (requires exafmm installed first) ----
if [ ! -f "$DEPS_DIR/.bempp_installed.ok" ]; then
  echo "[setup] Installing bempp-cl into venv"
  uv pip install bempp-cl
  touch "$DEPS_DIR/.bempp_installed.ok"
fi

# ---- kifmm-for-rsrs ----
if [ ! -d "$DEPS_DIR/kifmm-for-rsrs" ]; then
  echo "[setup] Cloning kifmm-for-rsrs into $DEPS_DIR"
  git clone https://github.com/ignacia-fp/kifmm-for-rsrs.git "$DEPS_DIR/kifmm-for-rsrs"
fi

if [ ! -f "$DEPS_DIR/kifmm-for-rsrs/.installed.ok" ]; then
  echo "[setup] Building & installing kifmm-for-rsrs via maturin"
  pushd "$DEPS_DIR/kifmm-for-rsrs/kifmm" >/dev/null
  maturin develop --release
  popd >/dev/null
  touch "$DEPS_DIR/kifmm-for-rsrs/.installed.ok"
fi

# ---- Build main Rust project ----
echo "[setup] Building Rust project"
cargo build

# If no command was provided (common with some compose setups), default to bash.
if [ "$#" -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
