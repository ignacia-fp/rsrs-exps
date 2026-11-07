# Bempp-rs Python/FMM development

This environment builds the local Bempp-rs fork on x86-64 Linux, matching the
architecture used on Cardinal. It avoids macOS `libffi` failures and the ARM
ARPACK `c_char` mismatch encountered during local development.

## Start the environment

From the root of `rsrs-exps`, run:

```bash
bash scripts/dev/run_bempp_rs_python_container.sh
```

The first build downloads and compiles the native dependency stack. Later
builds reuse Docker and Cargo caches. The repository is mounted at
`/workspace`, so edits remain in the host checkout.

## Build and test

Inside the container, run:

```bash
bash scripts/dev/build_bempp_rs_python.sh
```

This performs three checks in order:

1. compile-check the locked Bempp-rs Rust library;
2. install its mixed Rust/CFFI Python package in the container virtualenv;
3. run the particle-FMM and full single-layer composition tests.

The build script uses pinned `nightly-2026-10-15` for both checks so Cargo can
reuse one dependency cache. Nightly is required because Maturin's
CFFI/cbindgen backend requests expanded Rust source via
`-Zunpretty=expanded`.

The dependency lockfile is intentional. In particular,
`c-api-tools-macros 0.1.0` must use `syn 2`; resolving it against `syn 3`
breaks its procedural macros.

## Operator boundary

`LaplaceParticleFmm` evaluates only the Laplace Green kernel between particle
locations. `LaplaceSingleLayerFmm` represents the complete matrix-free BEM
composition

```text
point_to_space @ (kernel - near_correction) @ space_to_point + singular
```

Sampling code must use the latter. Using the particle evaluator directly would
omit basis/quadrature mappings, near-field correction, and singular integrals.

## Cardinal

The Docker environment is for development and reproducibility. Cardinal should
build the same locked sources in its native x86-64 environment rather than run
an emulated container. No script in this directory submits or modifies cluster
jobs.
