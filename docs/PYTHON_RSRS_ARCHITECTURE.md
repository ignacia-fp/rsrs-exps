# Python-first RSRS architecture

## Goal

The experiment driver should not pass a Python matrix-vector callback into
Rust. Python constructs the physical operator, generates and validates random
samples, and passes the resulting arrays (or a cache containing those arrays)
to RSRS. Rust performs the factorisation and exposes the factors back to
Python.

This restriction applies only to the Python-facing API. When an operator is
implemented in Rust, RSRS retains its native sampling path and may generate
additional test/sketch rows internally.

This gives three independently testable layers:

1. **`so-kernels`**: installable Python bindings for the Bempp-rs and KiFMM
   kernels. It owns geometry/operator construction and matrix-vector products.
2. **`bempp-rsrs`**: the RSRS Rust crate plus a PyO3 extension. The native Rust
   API can accept a Rust operator and sample it; the Python extension accepts
   points, options, and precomputed sample arrays or a cache path. It owns
   factorisation, factor application, ranks, and timings.
3. **`rsrs-exps`**: Python experiment orchestration. It owns sampling policy,
   cache validation/repair, norm and error estimation, GMRES, plots, and final
   experiment reports.

## Sample convention

Samples are stored row-wise so one row is one random test vector:

- `y_test[i, :] = omega_i`
- `y_sketch[i, :] = A @ omega_i`
- `z_test[i, :] = eta_i`
- `z_sketch[i, :] = A.T @ eta_i`

The `z` convention deliberately uses the non-conjugating transpose. For a
complex operator this is different from `A.conj().T`. The cache manifest must
record this convention explicitly. Symmetric and Hermitian modes may omit the
second sample family when RSRS does not require it.

Python arrays use C-contiguous `(samples, dimension)` storage at the API
boundary. The native layer is responsible for converting that representation
to the internal RLST layout exactly once.

## Public Python API

The intended stable API is:

```python
from bempp_rsrs import SampleBundle, factorize

samples = SampleBundle(
    y_test=omega_y,
    y_sketch=y,
    z_test=omega_z,
    z_sketch=z,
)
samples.validate()

factors, report = factorize(
    points=points,
    samples=samples,
    options=options,
)

A_approx = factors.aslinearoperator(inverse=False)
M = factors.aslinearoperator(inverse=True)
```

`factorize` also accepts a cache directory. Cache loading is performed in
Python so cache format and validation can evolve without coupling experiment
logic to the factorisation crate.

The factor object must provide:

- `matvec` and `matmat` for the approximate matrix;
- `rmatvec` using the conjugate transpose expected by SciPy;
- inverse versions of those operations for preconditioning;
- `aslinearoperator(inverse=...)`;
- factorisation ranks and timing data.

## Two sample providers

The factorisation core consumes a sample-provider abstraction with two
implementations:

- `RustOperatorSamples`: retains the current `run(operator)` behaviour. It
  loads compatible cached rows when requested and applies the native Rust
  operator whenever more rows are required.
- `PrecomputedSamples`: used by Python. It serves rows supplied as NumPy arrays
  or loaded from a cache and never calls back into Python for a matrix-vector
  product.

Both providers produce the same internal `SketchData` representation, after
which tree traversal, skeletonisation, factor construction, statistics, and
factor application are identical. This avoids maintaining two numerical RSRS
implementations.

The intended Rust entry points are therefore conceptually:

```rust,ignore
rsrs.run(operator);                 // native operator; may sample in Rust
rsrs.run_with_samples(samples);     // precomputed rows; never samples
rsrs.run_with_cache(cache_path);    // precomputed rows on disk; never samples
```

The latter two are the only entry points exposed through PyO3.

## Configuration

The Python API must expose every RSRS option currently selectable through
the factorisation API, rather than offering a reduced set of convenient presets. This
includes:

- ID, nullification, extraction, and diagonal tolerances;
- adaptive-tolerance and fixed-rank modes;
- rank picking and nonsymmetric ID combination;
- oversampling, diagonal oversampling, initial/minimum samples, and per-level
  versus constant fixed-rank sampling;
- symmetry and factor-storage modes;
- null-space, QR, pivoting, and least-squares extraction methods;
- minimum rank, minimum level, maximum tree depth, and factor flushing;
- independent factorisation and factor-application thread counts;
- rank, timing, effort, and memory-statistics controls.

`rsrs-experiment` is the high-level experiment/configuration generator. It
emits the same versioned configuration object accepted by the Python API and
the native Rust API and does not maintain a separate numerical implementation.
Missing fields are resolved by the `bempp_rsrs` options object.

Tree leaf size is represented as an optional setting:

```python
RsrsOptions(
    leaf_size=None,  # use the existing automatic/default octree logic
)
```

When the user provides a positive `leaf_size`, it is passed to octree
construction explicitly. When it is absent or `None`, the current operator and
geometry-specific leaf-size selection is left unchanged. Configuration reports
must record both the requested value and the effective value chosen at runtime.

Factorisation results return a structured report rather than relying on parsed
terminal output. At minimum it contains:

- effective configuration and tree parameters;
- ranks and box sizes per level;
- sample requirements and rows consumed per family;
- loading, update, ID, extraction, and total factorisation timings;
- factor application timings collected separately from factorisation;
- memory statistics and limiting levels/boxes.

The report can be converted to JSON by Python and incorporated directly into
the existing experiment result format.

## Insufficient samples in the Python path

The precomputed-sample provider must never generate samples. If a level needs
more rows than were supplied, it returns a structured `InsufficientSamples`
error with:

- required total rows;
- supplied rows for each family;
- level and limiting box, when available.

The Python driver may then generate the missing independent rows, append and
validate them, and retry factorisation. By contrast, the native Rust provider
may satisfy the same request immediately by applying its Rust operator.

## Reporting ownership

Rust reports only quantities intrinsic to RSRS:

- factorisation time, excluding sampling;
- per-stage and per-level timings;
- ranks, box sizes, and memory statistics;
- factor application time.

Python reports quantities that require the original operator or experiment
policy:

- sample consistency errors;
- approximate-inverse and solve errors;
- operator norm estimates;
- GMRES histories and iteration counts;
- RCS/field plots and publication tables.

## Implementation sequence

1. Stabilise `SampleBundle`, cache manifests, and SciPy factor wrappers in
   Python.
2. Introduce the common sample-provider abstraction while preserving the
   existing Rust-operator sampling behaviour.
3. Add array-fed and cache-fed RSRS entry points backed by the precomputed
   provider.
4. Expose complex128 factors through PyO3, followed by float64 and other
   supported scalar types.
5. Move existing error, GMRES, and plotting paths from the Rust experiment
   binary into Python modules using the stable factor wrapper.
6. Retire only the Python-to-Rust operator-callback experiment path after
   numerical parity tests. Keep native Rust operator sampling supported.
