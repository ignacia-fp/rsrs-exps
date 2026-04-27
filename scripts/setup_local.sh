#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKSPACE="${WORKSPACE:-$REPO_ROOT}"
DEPS_DIR="${DEPS_DIR:-$WORKSPACE/.deps}"
BUILD_RUST_PROJECT="${BUILD_RUST_PROJECT:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_SYSTEM_PACKAGES=0
INSTALL_RUST=0

APT_PACKAGES=(
  ca-certificates
  curl
  git
  pkg-config
  build-essential
  cmake
  clang
  lld
  gfortran
  libssl-dev
  openmpi-bin
  libopenmpi-dev
  libhdf5-dev
  libopenblas-dev
  liblapack-dev
  libfftw3-dev
  python3.10
  python3.10-dev
  python3.10-venv
  python3-pip
  patchelf
  libfontconfig1-dev
  libfreetype6-dev
  gmsh
)

print_manual_package_hints() {
  case "$(uname -s)" in
    Darwin)
      cat <<'EOF'
[setup-local] macOS hint:
  Use Homebrew (or another package manager) for the native prerequisites, then rerun:
    bash scripts/setup_local.sh --python "$(command -v python3)"

  A typical Homebrew baseline is:
    brew install git pkg-config cmake llvm open-mpi hdf5 openblas fftw gmsh

  Notes:
    - ExaFMM is skipped automatically on macOS.
    - KiFMM and dense assembly are still supported locally.
EOF
      ;;
    *)
      cat <<'EOF'
[setup-local] Non-apt Linux hint:
  Make sure the following prerequisites are installed or loaded before rerunning:
    - Python 3 with venv support (or pip + virtualenv)
    - git
    - pkg-config
    - cmake
    - a C/C++ toolchain
    - MPI (mpicc plus pkg-config metadata)
    - BLAS / LAPACK
    - FFTW
    - HDF5
    - gmsh

  Then rerun:
    bash scripts/setup_local.sh --python "$(command -v python3)"
EOF
      ;;
  esac
}

print_help() {
  cat <<EOF
Usage: scripts/setup_local.sh [options]

Run the non-Docker rsrs-exps setup on the host system.

Options:
  --install-system-packages  Install the Ubuntu/Debian package prerequisites with apt-get.
  --install-rust             Install Rust with rustup if cargo is not already available.
  --skip-rust-build          Skip the final top-level cargo build.
  --python BIN              Use BIN to create the virtual environment.
  --deps-dir PATH           Store dependency build caches in PATH.
  -h, --help                Show this help text.

Examples:
  bash scripts/setup_local.sh --install-system-packages --install-rust
  bash scripts/setup_local.sh --python python3.11

Notes:
  On macOS, this setup skips ExaFMM automatically because ExaFMM is not
  supported there. KiFMM and non-ExaFMM workflows can still be installed.
  On systems without sudo, install or load the required dependencies first and
  pass the chosen interpreter with --python.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-system-packages)
      INSTALL_SYSTEM_PACKAGES=1
      ;;
    --install-rust)
      INSTALL_RUST=1
      ;;
    --skip-rust-build)
      BUILD_RUST_PROJECT=0
      ;;
    --python)
      shift
      if [ "$#" -eq 0 ]; then
        echo "[setup-local] ERROR: --python requires an argument"
        exit 1
      fi
      PYTHON_BIN="$1"
      ;;
    --deps-dir)
      shift
      if [ "$#" -eq 0 ]; then
        echo "[setup-local] ERROR: --deps-dir requires an argument"
        exit 1
      fi
      DEPS_DIR="$1"
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "[setup-local] ERROR: unknown option '$1'"
      print_help
      exit 1
      ;;
  esac
  shift
done

run_with_optional_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[setup-local] ERROR: sudo is required to run: $*"
    exit 1
  fi
}

install_system_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[setup-local] ERROR: --install-system-packages currently supports apt-get based systems only"
    print_manual_package_hints
    exit 1
  fi

  echo "[setup-local] Installing system packages with apt-get"
  run_with_optional_sudo apt-get update
  run_with_optional_sudo apt-get install -y "${APT_PACKAGES[@]}"
}

install_rust_toolchain() {
  if command -v cargo >/dev/null 2>&1; then
    echo "[setup-local] cargo already available; skipping Rust installation"
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "[setup-local] ERROR: curl is required to install Rust via rustup"
    exit 1
  fi

  echo "[setup-local] Installing Rust with rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
}

if [ "$INSTALL_SYSTEM_PACKAGES" = "1" ]; then
  install_system_packages
fi

if [ "$INSTALL_SYSTEM_PACKAGES" = "0" ] && ! command -v apt-get >/dev/null 2>&1; then
  echo "[setup-local] Note: automatic system package installation is unavailable on this host."
  print_manual_package_hints
fi

if [ "$INSTALL_RUST" = "1" ]; then
  install_rust_toolchain
fi

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[setup-local] ERROR: cargo is not available on PATH"
  echo "[setup-local] Install Rust first, or rerun with --install-rust"
  exit 1
fi

echo "[setup-local] Ensuring git submodules are present"
git -C "$REPO_ROOT" submodule update --init --recursive

WORKSPACE="$WORKSPACE" \
DEPS_DIR="$DEPS_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
BUILD_RUST_PROJECT="$BUILD_RUST_PROJECT" \
"$REPO_ROOT/scripts/setup_deps.sh" true

cat <<EOF
[setup-local] Setup complete.

[setup-local] To use this environment in a new shell, run:
  source "$WORKSPACE/.venv/bin/activate"
  source "$WORKSPACE/.rsrs-env.sh"
EOF
