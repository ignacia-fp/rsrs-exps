import os
from pathlib import Path

import numpy as np
import bempp_cl.api
import h5py

from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh  # (still imported; some users keep it around)
from scipy.sparse.linalg import LinearOperator

from .geometry import get_geometry, get_barycenters, get_edges_centres
from .righ_hand_sides import right_hand_side

from kifmm_py import (
    KiFmm,
    LaplaceKernel,
    HelmholtzKernel,
    SingleNodeTree,
    EvalType,
    FftFieldTranslation,
)
import time
from threadpoolctl import threadpool_limits, threadpool_info

bempp_cl.api.GLOBAL_PARAMETERS.fmm.expansion_order = 3
# bempp_cl.api.GLOBAL_PARAMETERS.quadrature.regular = 1
# bempp_cl.api.GLOBAL_PARAMETERS.quadrature.singular = 1

bempp_cl.api.DEFAULT_PRECISION = "single"
# -------------------------
# Small shared helpers (no functional change)
# -------------------------
def _set_bempp_precision(precision: str) -> None:
    bempp_cl.api.DEFAULT_PRECISION = "single" if precision == "single" else "double"


import time
import numpy as np

NPROC=32
ASSEMBLER = "dense"

def apply(op, X):
    """
    Apply operator/matrix op to X.

    Supports:
      - 1D vector (n,)
      - 2D block (n, m)
    Returns output with matching dimensionality.
    """

    X = np.asarray(X)

    is_vector = (X.ndim == 1)

    # Normalize vector to (n,1) for matmul if needed
    if is_vector:
        X_mat = X[:, None]
    else:
        X_mat = X

    # NumPy dense
    if isinstance(op, (np.ndarray, np.matrix)):
        Y = op @ X_mat
    # SciPy sparse or objects supporting @
    elif hasattr(op, "__matmul__"):
        try:
            Y = op @ X_mat
        except Exception:
            Y = op * X_mat
    else:
        # BEM++ weak form operators
        Y = op * X_mat

    Y = np.asarray(Y)

    # Return shape consistent with input
    if is_vector:
        return Y.ravel()
    return Y

def _maybe_generate_samples(
    op,
    op_T,
    precision,
    n: int,
    init_samples: int,
    *,
    prefix_y: str = "y",
    prefix_z: str = "z",
    base_seed: int | None = None,
    # Fast-path decomposition: op = common + nonsym, op_T = common + nonsymT
    common=None,
    nonsym=None,
    nonsymT=None,
    # If True and op and op_T are effectively the same, only store y_* (otherwise store both y_* and z_*)
    save_only_y_if_symmetric: bool = False,
    transposable = False
):
    print(threadpool_info())
    """
    Ensure at least init_samples are present on disk (in sampling/).
    If fewer exist, append the missing number.

    If (common, nonsym, nonsymT) are provided:
        op   = common + nonsym
        op_T = common + nonsymT
    and we compute common*[Omega_y Omega_z] in one multiply.
    """

    with threadpool_limits(limits=NPROC):
        if op is None:
            test_mat = np.ones((n, 1))
            t0 = time.perf_counter()
            apply(nonsym, test_mat)
            t_test = time.perf_counter() - t0
            print(f"sampling test: {t_test:.3f}s")
            t0 = time.perf_counter()
            apply(nonsym, test_mat)
            t_test = time.perf_counter() - t0
            print(f"sampling test 2: {t_test:.3f}s")
            t_total0 = time.perf_counter()
        else: 
            test_mat = np.ones((n, 1))
            t0 = time.perf_counter()
            apply(op, test_mat)
            t_test = time.perf_counter() - t0
            print(f"sampling test: {t_test:.3f}s")
            t0 = time.perf_counter()
            apply(op, test_mat)
            t_test = time.perf_counter() - t0
            print(f"sampling test 2: {t_test:.3f}s")
            t_total0 = time.perf_counter()

        y_test = f"{prefix_y}_test_file"
        y_sk   = f"{prefix_y}_sketch_file"
        z_test = f"{prefix_z}_test_file"
        z_sk   = f"{prefix_z}_sketch_file"

        # Count existing samples (rows)
        t0 = time.perf_counter()
        m_old = existing_num_rows(y_test) or 0
        t_count = time.perf_counter() - t0

        if init_samples <= m_old:
            t_total = time.perf_counter() - t_total0
            print(
                f"[sampling] already have {m_old} samples (target={init_samples}); "
                f"nothing to do. (count={t_count:.3f}s, total={t_total:.3f}s)"
            )
            return

        n_add = init_samples - m_old
        print(f"[sampling] have {m_old}, target {init_samples} -> appending {n_add} samples")

        # RNG
        t0 = time.perf_counter()
        rng_y = make_rng(base_seed, stream_tag="Omega_y", m_existing=m_old)
        rng_z = make_rng(base_seed, stream_tag="Omega_z", m_existing=m_old)
       
        
        if op is None:
            symmetric_mode = False
        else:
            symmetric_mode = (op_T is op)
        
        if transposable:
            symmetric_mode = False

        if precision == "single":
            rdtype = np.float32
        else:
            rdtype = np.float64
        Omega_y = rng_y.standard_normal((n, n_add), dtype=rdtype)
        
        if not symmetric_mode:
            Omega_z = rng_z.standard_normal((n, n_add), dtype=rdtype)
        t_rng = time.perf_counter() - t0

        t0 = time.perf_counter()

        # Apply
        if transposable:
            Y = np.asarray(apply(op, Omega_y))
            Z = np.asarray(apply(op.T, Omega_z))
        
        else:
            if symmetric_mode:
                Y = np.asarray(apply(op, Omega_y))

                if save_only_y_if_symmetric:
                    t_apply = time.perf_counter() - t0

                    # I/O
                    t1 = time.perf_counter()
                    store_complex = np.iscomplexobj(Y)
                    append_samples_multipart(Omega_y.T, y_test, store_complex)
                    append_samples_multipart(Y.T,       y_sk,   store_complex)
                    t_io = time.perf_counter() - t1

                    m_new = existing_num_rows(y_test) or (m_old + n_add)
                    t_total = time.perf_counter() - t_total0
                    print(
                        "[sampling] symmetric: saved only y_* | "
                        f"added={n_add}, total={m_new} | "
                        f"count={t_count:.3f}s, rng={t_rng:.3f}s, apply={t_apply:.3f}s, io={t_io:.3f}s, total={t_total:.3f}s"
                    )
                    return

                # Otherwise: still save z_* but avoid extra multiply by reusing Y
                Z = Y
                Omega_z = Omega_y

            elif common is not None and nonsym is not None and nonsymT is not None:
                # Fast split path: common apply once on concatenated Omegas
                Omega_big = np.concatenate([Omega_y, Omega_z], axis=1)  # (n, 2*n_add)
                C_big = np.asarray(apply(common, Omega_big))                  # (n, 2*n_add)
                C_y = C_big[:, :n_add]
                C_z = C_big[:, n_add:]

                N_y  = np.asarray(apply(nonsym, Omega_y))
                NT_z = np.asarray(apply(nonsymT, Omega_z))

                Y = C_y + N_y
                Z = C_z + NT_z

            else:
                # Default path: two multiplies
                Y = np.asarray(apply(op, Omega_y))
                Z = np.asarray(apply(op_T, Omega_z))

        t_apply = time.perf_counter() - t0

        # I/O
        t0 = time.perf_counter()
        store_complex = np.iscomplexobj(Y) or np.iscomplexobj(Z)
        append_samples_multipart(Omega_y.T,          y_test, store_complex)
        append_samples_multipart(Y.T,               y_sk,   store_complex)
        append_samples_multipart(np.conj(Omega_z).T, z_test, store_complex)
        append_samples_multipart(np.conj(Z).T,       z_sk,   store_complex)
        t_io = time.perf_counter() - t0

        m_new = existing_num_rows(y_test) or (m_old + n_add)
        t_total = time.perf_counter() - t_total0
        print(
            "[sampling] done | "
            f"added={n_add}, total={m_new} | "
            f"count={t_count:.3f}s, rng={t_rng:.3f}s, apply={t_apply:.3f}s, io={t_io:.3f}s, total={t_total:.3f}s"
        )


# -----------------Sampling------------------------- #

# Match Rust constants
BLOCK_COLS = 4096
CHUNK_ROWS = 256
SAMPLING_DIR = "sampling"


def _ensure_sampling_dir() -> None:
    Path(SAMPLING_DIR).mkdir(parents=True, exist_ok=True)


def _canonical_base(input: str) -> str:
    s = str(input)
    # strip ".00000.h5"
    if s.endswith(".h5") and len(s) >= 9:
        tail = s[-9:]  # ".00000.h5"
        if tail[0] == "." and tail[6] == "." and tail[7:] == "h5" and tail[1:6].isdigit():
            return s[:-9]
    # strip ".h5"
    if s.endswith(".h5"):
        return s[:-3]
    return s


def _part_path(base: str, b: int) -> str:
    b0 = _canonical_base(base)
    return f"{SAMPLING_DIR}/{b0}.{b:05d}.h5"


def _find_part_files(base: str) -> list[tuple[int, str]]:
    """
    Mirror Rust find_part_files(): scan sampling/ for stem.{00000..}.h5
    Require contiguous indices 0..K-1.
    """
    stem = _canonical_base(base)
    d = Path(SAMPLING_DIR)
    if not d.exists():
        return []

    parts: list[tuple[int, str]] = []
    prefix = f"{stem}."
    for p in d.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith(prefix) and name.endswith(".h5")):
            continue
        mid = name[len(prefix):-3]  # between '.' and '.h5'
        if len(mid) != 5 or not mid.isdigit():
            continue
        idx = int(mid)
        parts.append((idx, str(p)))

    parts.sort(key=lambda t: t[0])
    # contiguous check
    for expected, (idx, _) in enumerate(parts):
        if idx != expected:
            raise RuntimeError(
                f"missing part file index {expected} (found {idx}) in {SAMPLING_DIR}"
            )
    return parts


def _nblocks(ncols: int) -> int:
    return (ncols + BLOCK_COLS - 1) // BLOCK_COLS


def _block_width(ncols: int, b: int) -> int:
    col0 = b * BLOCK_COLS
    return min(ncols - col0, BLOCK_COLS)


def _write_shape_attr(f: h5py.File, m: int, ncols: int) -> None:
    f.attrs["shape"] = np.array([m, ncols], dtype=np.uint64)


def _read_shape_attr(f: h5py.File) -> tuple[int, int]:
    sh = f.attrs["shape"]
    if len(sh) != 2:
        raise RuntimeError("shape attr must have length 2")
    return int(sh[0]), int(sh[1])


def _chunk_len_for(total_len: int, wk: int) -> int:
    """
    h5py requirement: chunk dimension <= data dimension.
    We use Rust-like chunk heuristic CHUNK_ROWS*wk, but clamp to total_len.
    """
    return min(int(total_len), max(1, CHUNK_ROWS * wk))


def save_samples_multipart_overwrite(A: np.ndarray, base: str, force_complex: bool = False) -> None:
    """
    Overwrite multipart files with A in Rust-compatible layout.
    If force_complex=True, write both datasets "real" and "imag" (imag zeros if A is real).
    """
    _ensure_sampling_dir()

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    m, ncols = A.shape
    AF = np.asfortranarray(A)

    # Remove existing parts
    parts = _find_part_files(base)
    for _, p in parts:
        Path(p).unlink()

    is_complex = force_complex or np.iscomplexobj(AF)

    for b in range(_nblocks(ncols)):
        wk = _block_width(ncols, b)
        col0 = b * BLOCK_COLS
        blk = AF[:, col0:col0 + wk]
        flat = blk.reshape(-1, order="F")

        p = _part_path(base, b)
        with h5py.File(p, "w") as f:
            _write_shape_attr(f, m, ncols)

            if not is_complex:
                chunk_len = _chunk_len_for(flat.size, wk)
                f.create_dataset("real", data=np.asarray(flat), shape=(flat.size,), chunks=(chunk_len,))
            else:
                re = np.asarray(flat.real)
                im = np.asarray(flat.imag) if np.iscomplexobj(flat) else np.zeros_like(re)
                chunk_len = _chunk_len_for(re.size, wk)
                f.create_dataset("real", data=re, shape=(re.size,), chunks=(chunk_len,))
                f.create_dataset("imag", data=im, shape=(im.size,), chunks=(chunk_len,))


def append_samples_multipart(A_extra: np.ndarray, base: str, force_complex: bool = False) -> None:
    """
    Append rows of A_extra to existing multipart data for `base`.
    If no existing data, this becomes a fresh write.

    If force_complex=True, always write both "real" and "imag" datasets (imag zeros if input is real).
    """
    _ensure_sampling_dir()

    A_extra = np.asarray(A_extra)
    if A_extra.ndim != 2:
        raise ValueError(f"A_extra must be 2D, got shape {A_extra.shape}")
    A_extraF = np.asfortranarray(A_extra)
    m_add, ncols = A_extraF.shape

    is_complex = force_complex or np.iscomplexobj(A_extraF)

    parts = _find_part_files(base)
    if not parts:
        save_samples_multipart_overwrite(A_extraF, base, force_complex=force_complex)
        return

    with h5py.File(parts[0][1], "r") as f0:
        m_old, ncols_old = _read_shape_attr(f0)
        stored_is_complex = ("imag" in f0)

    if ncols_old != ncols:
        raise RuntimeError(f"ncols mismatch for base '{base}': stored {ncols_old}, extra {ncols}")

    if stored_is_complex != is_complex:
        raise RuntimeError(
            f"format mismatch for base '{base}': stored complex={stored_is_complex}, "
            f"but requested complex={is_complex}. Delete existing sampling/{_canonical_base(base)}.*.h5 to reset."
        )

    m_new = m_old + m_add

    for b, p in parts:
        wk = _block_width(ncols, b)
        col0 = b * BLOCK_COLS

        blk_extra = A_extraF[:, col0:col0 + wk]
        flat_extra = blk_extra.reshape(-1, order="F")

        with h5py.File(p, "r") as f:
            old_re = np.array(f["real"][:], copy=False)
            if old_re.size != m_old * wk:
                raise RuntimeError(f"len mismatch in {p}::real: got {old_re.size}, expected {m_old*wk}")

            if not is_complex:
                new_re = np.concatenate([old_re, np.asarray(flat_extra)], axis=0)
            else:
                old_im = np.array(f["imag"][:], copy=False)
                if old_im.size != m_old * wk:
                    raise RuntimeError(f"len mismatch in {p}::imag: got {old_im.size}, expected {m_old*wk}")

                extra_re = np.asarray(flat_extra.real)
                extra_im = np.asarray(flat_extra.imag) if np.iscomplexobj(flat_extra) else np.zeros_like(extra_re)

                new_re = np.concatenate([old_re, extra_re], axis=0)
                new_im = np.concatenate([old_im, extra_im], axis=0)

        with h5py.File(p, "w") as f:
            _write_shape_attr(f, m_new, ncols)
            chunk_len = _chunk_len_for(new_re.size, wk)
            f.create_dataset("real", data=new_re, shape=(new_re.size,), chunks=(chunk_len,))
            if is_complex:
                f.create_dataset("imag", data=new_im, shape=(new_im.size,), chunks=(chunk_len,))


def existing_num_rows(base: str) -> int | None:
    parts = _find_part_files(base)
    if not parts:
        return None
    with h5py.File(parts[0][1], "r") as f0:
        m, _ = _read_shape_attr(f0)
    return m


def make_rng(base_seed: int | None, stream_tag: str, m_existing: int) -> np.random.Generator:
    """
    base_seed=None => OS entropy (independent each run)
    base_seed=int  => deterministic independent streams per (m_existing, stream_tag)
    """
    if base_seed is None:
        return np.random.default_rng()

    tag_int = abs(hash(stream_tag)) % (2**32)
    ss = np.random.SeedSequence(base_seed, spawn_key=(m_existing, tag_int))
    return np.random.default_rng(ss)


def generate_and_append_test_and_sketches(
    op,
    op_T,
    n,
    n_samples_target: int,          # <-- interpret this as "desired total samples"
    base_seed: int | None = None,
    prefix_y: str = "y",
    prefix_z: str = "z",
):
    """
    Ensure there are at least `n_samples_target` samples stored on disk.
    If fewer exist, append the missing number (independent new samples).
    If enough exist, do nothing.

    Storage matches Rust: we store transposed (samples are rows).
    """

    t_total0 = time.perf_counter()

    y_test = f"{prefix_y}_test_file"
    y_sk   = f"{prefix_y}_sketch_file"
    z_test = f"{prefix_z}_test_file"
    z_sk   = f"{prefix_z}_sketch_file"

    # How many samples (rows) already exist?
    m_old = existing_num_rows(y_test) or 0

    # If we already have enough, don't add anything
    if n_samples_target <= m_old:
        # Optional: sanity check other files exist / same row count
        print(f"[sampling] already have {m_old} samples (target={n_samples_target}); not appending.")
        return

    n_add = n_samples_target - m_old
    print(f"[sampling] have {m_old}, target {n_samples_target} -> appending {n_add} samples")

    # Independent RNG streams; deterministic if base_seed set, otherwise OS entropy.
    rng_y = make_rng(base_seed, stream_tag="Omega_y", m_existing=m_old)
    rng_z = make_rng(base_seed, stream_tag="Omega_z", m_existing=m_old)

    # Generate tests as (n, n_add) for applying operator; store transposed as (n_add, n)
    Omega_y = rng_y.standard_normal((n, n_add))
    Omega_z = rng_z.standard_normal((n, n_add))

    # Sketches
    Y = np.asarray(apply(op, Omega_y))
    Z = np.asarray(apply(op_T, Omega_z))

    # IMPORTANT: real operators should be stored as real-on-disk (no imag dataset)
    # complex operators should be stored as complex-on-disk (real+imag datasets)
    # We decide per-array based on dtype/object, no forcing.
    store_complex = np.iscomplexobj(Y) or np.iscomplexobj(Z)

    append_samples_multipart(Omega_y.T,          y_test, store_complex)
    append_samples_multipart(Y.T,               y_sk,   store_complex)
    append_samples_multipart(np.conj(Omega_z).T, z_test, store_complex)
    append_samples_multipart(np.conj(Z).T,       z_sk,   store_complex)

    m_new = existing_num_rows(y_test)
    print(f"[sampling] appended {n_add}. Total stored samples now: {m_new}.")

    t_total = time.perf_counter() - t_total0

    print(
        "[sampling] done. "
        f"Total time ={t_total:.3f}s"
    )


# -----------------End of sampling------------------------- #


class BaseStructuredOperator(ABC):
    def __init__(
        self,
        dim_param,
        kappa,
        geometry_type='sphere_surface',
        precision='double',
        n_sources=1,
        init_samples=0
    ):
        try:
            print(dim_param)
            if dim_param < 1:
                self.grid = get_geometry(geometry_type, dim_param)
                path = Path(os.getcwd()) / "results"
                os.makedirs(path, exist_ok=True)
                bempp_cl.api.export('results/current_grid.msh', grid=self.grid)
                self.points = get_barycenters(self.grid)
            else:
                self.grid = None
                self.points = get_geometry(geometry_type, dim_param)

            self.init_samples = init_samples
            self.precision = precision
            self.n_points = len(self.points)
            self.n_sources = n_sources
            self.kappa = kappa
            self.form = 'weak'
            self.mat = None  # Set in subclass
            self.mat_T = None  # Set in subclass when needed
            self.rhs = None

            # scalar_type: 'real' or 'complex' (formerly overwritten in operator_type)
            self.scalar_type = None

            # operator_type: keep for operator name string (as requested)
            self.operator_type = None

            self.rhs_data_type = None
        except Exception as e:
            print("Error initializing BaseStructuredOperator:", e)
            raise

    @abstractmethod
    def mv(self, v):
        pass

    @abstractmethod
    def mv_trans(self, v):
        pass

    @property
    @abstractmethod
    def cond(self):
        pass

    @property
    @abstractmethod
    def dense(self):
        pass


class BasicStructuredOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        self.scalar_type = 'real'
        self.mat = (1.0/self.n_points) * np.exp(-cdist(self.points, self.points) ** 2)
        self.mat[np.diag_indices_from(self.mat)] = 1
        self.operator_type = 'BasicStructuredOperator'
        if self.precision == 'single':
            self.rhs_data_type = np.float32
        else:
            self.rhs_data_type = np.float64

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat @ v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat.T @ v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented (unused).")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return self.mat

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, None, n_sources)


class BemppClLaplaceSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            _maybe_generate_samples(self.mat, self.mat, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayer:", e)
            raise

    def mv(self, v):
        return self.mat * v

    def mv_trans(self, v):
        return self.mat.T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = self.domain
            with threadpool_limits(limits=NPROC):
                self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                    self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER).weak_form()
            _maybe_generate_samples(self.mat, self.mat, self.precision, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayer:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat.T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class KiFMMLaplaceOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            print("Precision:", self.rhs_data_type)
            self.scalar_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperator'
            points = self.points.ravel()
            print("Number of dofs:", self.n_points)
            sources = points.astype(self.rhs_data_type)
            targets = points.astype(self.rhs_data_type)
            expansion_order = np.array([6], np.uint64)
            n_vec = 1
            n_crit = 150
            prune_empty = True
            eval_type = EvalType.Value

            structured_operator = LaplaceKernel(self.rhs_data_type, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(self.rhs_data_type)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(structured_operator, block_size=32)

            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()

            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing KiFMMLaplaceOperator:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            self.mat.clear()
            charges = v.astype(self.rhs_data_type)
            self.mat.attach_charges_unordered(charges)
            self.mat.evaluate()
            res = np.copy(charges) + (1/self.n_points) * self.mat.all_potentials_u.reshape(-1)
            return res

    def mv_trans(self, v):
        return self.mv(v)

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        raise ValueError("There is not a dense representation implemented for this operator.")

    def get_rhs(self):
        return right_hand_side(self, None)


class KiFMMHelmholtzOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            if self.precision == 'single':
                dtype = np.float32
                self.rhs_data_type = np.complex64
            else:
                dtype = np.float64
                self.rhs_data_type = np.complex128

            self.scalar_type = 'complex'
            self.kappa = dtype(self.kappa)
            self.operator_type = 'KiFMMHelmholtzOperator'
            points = self.points.ravel()
            print("Number of dofs:", self.n_points)
            sources = points.astype(dtype)
            targets = points.astype(dtype)
            expansion_order = np.array([6], np.uint64)
            n_vec = 1
            n_crit = 150
            prune_empty = True
            eval_type = EvalType.Value

            structured_operator = HelmholtzKernel(dtype, self.kappa, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(self.rhs_data_type)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(structured_operator, block_size=32)

            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing KiFMMHelmholtzOperator:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            self.mat.clear()
            charges = v.astype(self.rhs_data_type)
            self.mat.attach_charges_unordered(charges)
            self.mat.evaluate()
            res = np.copy(charges) + (1/self.n_points) * self.mat.all_potentials_u.reshape(-1)
            return res

    def mv_trans(self, v):
        return self.mv(v)

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        raise ValueError("There is not a dense representation implemented for this operator.")

    def get_rhs(self):
        return right_hand_side(self, None)


class BemppRsLaplaceOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        self.scalar_type = 'real'
        self.operator_type = 'BemppRsLaplaceOperator'
        print("Number of dofs:", self.n_points)
        if self.precision == 'single':
            self.rhs_data_type = np.float32
        else:
            self.rhs_data_type = np.float64

    def mv(self, v):
        raise ValueError("Mv implemented in rust.")

    def mv_trans(self, v):
        raise ValueError("Mv trans implemented in rust.")

    @property
    def cond(self):
        raise ValueError("Matrix not initialized.")

    @property
    def dense(self):
        raise ValueError("Matrix not initialized.")

    def get_rhs(self):
        raise ValueError("Rhs implemented in rust.")


class BemppClLaplaceCombined(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceCombined'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            sl = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            dl = bempp_cl.api.operators.boundary.laplace.double_layer(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            self.mat = sl + 0.5 * identity + dl
            self.mat_T = sl + 0.5 * identity + dl.T

            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceCombined:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSecond(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSecond'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            dl = bempp_cl.api.operators.boundary.laplace.double_layer(
                self.domain, self.range, self.dual_to_range, assembler="fmm"
            ).weak_form()
            adl = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, assembler="fmm"
            ).weak_form()
            self.mat = 0.5 * identity + dl
            self.mat_T = 0.5 * identity + adl
            self.rhs_data_type = self.mat.dtype
            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.form = 'weak'
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceSecond:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerCP(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCP'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            single_layer = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range
            ).weak_form()

            @bempp_cl.api.real_callable
            def constant(x, n, domain_index, result):
                result[0] = 1.0

            rank_1_fun = bempp_cl.api.GridFunction(self.domain, fun=constant).projections()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            hypersingular = bempp_cl.api.operators.boundary.laplace.hypersingular(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            h_shape = hypersingular.shape

            def mv(v):
                return hypersingular * v + rank_1_fun * (rank_1_fun @ v)

            def rmv(v):
                return hypersingular.T * v + rank_1_fun * (rank_1_fun @ v)

            prec = LinearOperator(h_shape, matvec=mv, rmatvec=rmv)
            self.mat = g_inv * prec * g_inv * single_layer
            self.mat_T = single_layer.T * g_inv * prec.T * g_inv

            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.form = 'strong'
            self.rhs = self.get_rhs(n_sources=self.n_sources)
            self.n_points = self.rhs[0].shape[0]
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCP:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerMM(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerMM'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.range
            sl = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range, assembler="fmm"
            )
            self.mat = sl.strong_form()
            _maybe_generate_samples(self.mat, self.mat, self.n_points, self.init_samples)
            self.form = 'strong'
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerMM:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat.T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayerCP(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerCP'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = self.domain
            hypersingular = bempp_cl.api.operators.boundary.laplace.hypersingular(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            single_layer = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa
            ).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            self.mat = g_inv * hypersingular * g_inv * single_layer
            self.mat_T = single_layer.T * g_inv * hypersingular.T * g_inv
            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.form = 'strong'
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
            self.n_points = self.rhs[0].shape[0]
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerCP:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerCPID(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCPID'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()

            adjoint_double_layer = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, assembler="fmm"
            ).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            adjoint_double_layer_T = bempp_cl.api.operators.boundary.laplace.double_layer(
                self.domain, self.range, self.dual_to_range, assembler="fmm"
            ).weak_form()
            self.mat = g_inv * (0.25 * identity + adjoint_double_layer * g_inv * adjoint_double_layer)
            self.mat_T = (0.25 * identity + adjoint_double_layer_T * g_inv * adjoint_double_layer_T) * g_inv
            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCPID:", e)
            raise

    def mv(self, v):
        return self.mat * v

    def mv_trans(self, v):
        return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            _maybe_generate_samples(self.mat, self.mat, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerP1:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat.T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class KiFMMLaplaceOperatorV(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperatorV'
            points = self.points.ravel()
            sources = points.astype(self.rhs_data_type)
            targets = points.astype(self.rhs_data_type)
            expansion_order = np.array([6], np.uint64)
            n_vec = 1
            n_crit = 150
            prune_empty = True
            eval_type = EvalType.Value

            structured_operator = LaplaceKernel(self.rhs_data_type, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(self.rhs_data_type)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(structured_operator, block_size=32)

            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()

            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing KiFMMLaplaceOperator:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            self.mat.clear()
            charges = v.astype(self.rhs_data_type)
            self.mat.attach_charges_unordered(charges)
            self.mat.evaluate()
            res = np.copy(charges) + (1/self.n_points) * self.mat.all_potentials_u.reshape(-1)
            return res

    def mv_trans(self, v):
        return self.mv(v)

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        raise ValueError("There is not a dense representation implemented for this operator.")

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, None, n_sources)


class BemppClLaplaceCombinedP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceCombinedP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            )
            self.mat = (
                bempp_cl.api.operators.boundary.laplace.single_layer(
                    self.domain, self.range, self.dual_to_range, assembler="fmm"
                ).weak_form()
                + 0.5 * identity.weak_form()
            )
            _maybe_generate_samples(self.mat, self.mat, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClLaplaceCombined:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat.T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerCPIDP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCPIDP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()

            adjoint_double_layer = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            adjoint_double_layer_T = adjoint_double_layer.T
            self.mat = g_inv * (0.25 * identity + adjoint_double_layer * g_inv * adjoint_double_layer)
            self.mat_T = (0.25 * identity + adjoint_double_layer_T * g_inv * adjoint_double_layer_T) * g_inv
            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)

        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCPIDP1:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayerCPID(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerCPID'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()

            adjoint_double_layer = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, kappa
            ).weak_form()
            dl = bempp_cl.api.operators.boundary.helmholtz.double_layer(
                self.domain, self.range, self.dual_to_range, kappa, assembler="fmm"
            ).weak_form()
            dlt = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, kappa, assembler="fmm"
            ).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            self.mat = g_inv * (0.25 * identity + dlt * g_inv * adjoint_double_layer)
            self.mat_T = (0.25 * identity + dl * g_inv * dl) * g_inv
            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.form = 'strong'
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerCPID:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClMaxwellEfie(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = get_edges_centres(self.grid)
            self.n_points = len(self.points)
            print("Approximate number of dofs:", self.n_points)
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClMaxwellEfie'
            self.domain = bempp_cl.api.function_space(self.grid, "RWG", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "SNC", 0)
            self.mat = bempp_cl.api.operators.boundary.maxwell.electric_field(
                self.domain, self.dual_to_range, self.range, kappa
            ).weak_form()
            self.mat_T = self.mat.T
            self.n_points = self.mat.shape[0]
            _maybe_generate_samples(self.mat, self.mat_T, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClMaxwellEfie:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayerP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa
            ).weak_form()
            _maybe_generate_samples(self.mat, self.mat, self.n_points, self.init_samples)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerP1:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat.T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClBurtonMiller(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClBurtonMiller'
            print("Number of dofs:", self.n_points)
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.range = self.domain
            eta = 1j * kappa
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            dl = bempp_cl.api.operators.boundary.helmholtz.double_layer(
                self.domain, self.range, self.dual_to_range, kappa, assembler="fmm"
            ).weak_form()
            dlt = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, kappa, assembler="fmm"
            ).weak_form()
            hl = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa, assembler="fmm"
            ).weak_form()
            self.mat = 0.5 * identity - dl + (1j / kappa) * hl
            self.mat_T = 0.5 * identity - dlt + (1j / kappa) * hl
            self.form = 'weak'
            self.rhs_data_type = self.mat.dtype
            self.n_points = self.mat.shape[0]
            common  = 0.5*identity + (1j / kappa) * hl
            nonsym  = -dl
            nonsymT = -dlt
            _maybe_generate_samples(
                self.mat, self.mat_T, self.n_points, self.init_samples,
                common=common,
                nonsym=nonsym,
                nonsymT=nonsymT
            )
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClBurtonMiller:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat * v

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return self.mat_T * v

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Neumann', n_sources)


class BemppClHelmholtzCombined(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1, init_samples=0):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources, init_samples)
        try:
            _set_bempp_precision(self.precision)
            print("Number of dofs:", self.n_points)
            print("kappa: ", kappa)
            self.scalar_type = 'complex'
            self.operator_type = 'BemppClHelmholtzCombined'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            '''with threadpool_limits(limits=NPROC):
                identity = bempp_cl.api.operators.boundary.sparse.identity(
                    self.domain, self.range, self.dual_to_range
                ).weak_form()
                sl = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                    self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER
                ).weak_form()
                dl = bempp_cl.api.operators.boundary.helmholtz.double_layer(
                    self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER
                ).weak_form()
                dlt = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                    self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER
                ).weak_form()'''

            with threadpool_limits(limits=NPROC):
                r_dtype = np.float32 if self.precision == "single" else np.float64
                c_dtype = np.complex64 if self.precision == "single" else np.complex128

                half = r_dtype(0.5)
                j = c_dtype(1j)
                k = r_dtype(kappa)
                jk = c_dtype(j * c_dtype(k))  # typed scalar, avoids complex128 promotion

                # --- Build common = 0.5*I - i*k*SL without keeping I and SL afterwards ---

                # I (real dense) -> start common as complex dense
                I = (bempp_cl.api.operators.boundary.sparse.identity(
                        self.domain, self.range, self.dual_to_range
                    ).weak_form()
                    .to_dense()
                    .astype(r_dtype, copy=False)
                )
                common = (half * I).astype(c_dtype, copy=False)
                del I

                # SL (complex dense) -> common -= jk * SL, then delete SL
                SL = (bempp_cl.api.operators.boundary.helmholtz.single_layer(
                        self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER
                    ).weak_form()
                    .to_dense()
                    .astype(c_dtype, copy=False)
                )
                common -= jk * SL
                del SL

                # --- Build nonsymmetric pieces (must keep them) ---
                nonsym = (bempp_cl.api.operators.boundary.helmholtz.double_layer(
                        self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER
                    ).weak_form()
                    .to_dense()
                    .astype(c_dtype, copy=False)
                )


                '''nonsymT = (bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                        self.domain, self.range, self.dual_to_range, kappa, assembler=ASSEMBLER
                    ).weak_form()
                    .to_dense()
                    .astype(c_dtype, copy=False)
                )'''

                self.mat  = nonsym + common

                del nonsym
                del common

                _maybe_generate_samples(
                    self.mat, 
                    None, 
                    self.precision, 
                    self.n_points, 
                    self.init_samples,
                    transposable=True
                    #common=common,
                    #nonsym=nonsym,
                    #nonsymT=nonsymT
                )
                # --- Build mat/mat_T from required pieces ---
                # mat   = common + nonsym
                # mat_T = common + nonsymT
                #del nonsym

                #self.mat_T = nonsymT + common
                #del nonsym
                #del common
            self.rhs = self.get_rhs(n_sources=self.n_sources)
        except Exception as e:
            print("Error initializing BemppClHelmholtzCombined:", e)
            raise

    def mv(self, v):
        with threadpool_limits(limits=NPROC):
            return apply(self.mat, v)

    def mv_trans(self, v):
        with threadpool_limits(limits=NPROC):
            return apply(self.mat.T, v)

    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")

    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)
