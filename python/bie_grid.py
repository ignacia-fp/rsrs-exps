import numpy as np
from scipy.integrate import dblquad, tplquad
from scipy.sparse import eye as speye
from scipy.sparse.linalg import LinearOperator


def get_vectorized_kernel_func(scalar_func):

    def vec_kernel_func(XX, YY):

        assert XX.shape[-1] == YY.shape[-1]
        ndim = XX.shape[-1]
        assert (ndim == 2) or (ndim == 3)

        m = XX.shape[0]
        n = YY.shape[0]
        pt1 = np.ones(ndim,)
        dtype = np.asarray(scalar_func(*pt1)).dtype
        result = np.zeros((m, n), dtype=dtype)

        for i in range(XX.shape[0]):
            for j in range(YY.shape[0]):
                diff = XX[i] - YY[j]
                if np.linalg.norm(diff, ord=2) == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = scalar_func(*diff)
        return result

    return vec_kernel_func


def get_regular_grid(n, ndim):

    pts = np.arange(0, n + 1) / n
    if ndim == 2:
        xx0, xx1 = np.meshgrid(pts, pts, indexing='ij')

        XX = np.vstack((xx0.flatten(), xx1.flatten()))
        return XX.T
    else:
        xx0, xx1, xx2 = np.meshgrid(pts, pts, pts, indexing='ij')

        XX = np.vstack((xx0.flatten(), xx1.flatten(), xx2.flatten()))
        return XX.T


def build_rank_one_box_perturbations(
    n_points,
    ndim,
    scale=3.0e-3,
    seed=12345,
):

    indices = np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    left = rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)
    right = rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)

    left /= max(np.linalg.norm(left), 1.0e-30)
    right /= max(np.linalg.norm(right), 1.0e-30)

    terms = [
        {
            "target_indices": indices,
            "left": left.astype(np.complex128, copy=False),
            "right": right.astype(np.complex128, copy=False),
            "alpha": np.complex128(scale * (1.0 + 2.0j) / np.sqrt(5.0)),
            "label": "global",
        },
    ]

    return {
        "box_indices": indices,
        "terms": terms,
    }


def build_real_rank_one_box_perturbations(
    n_points,
    ndim,
    scale=3.0e-3,
    seed=12345,
):

    indices = np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    left = rng.standard_normal(n_points)
    right = rng.standard_normal(n_points)

    left /= max(np.linalg.norm(left), 1.0e-30)
    right /= max(np.linalg.norm(right), 1.0e-30)

    terms = [
        {
            "target_indices": indices,
            "left": left.astype(np.float64, copy=False),
            "right": right.astype(np.float64, copy=False),
            "alpha": np.float64(scale),
            "label": "global",
        },
    ]

    return {
        "box_indices": indices,
        "terms": terms,
    }


def filter_perturbation_terms(perturbation, mode):

    mode = str(mode).strip().lower()
    if mode in {"both", "global"}:
        return perturbation

    selected = [term for term in perturbation["terms"] if term["label"].lower() == mode]
    if not selected:
        raise ValueError(f"Unknown perturbation mode '{mode}', expected one of: both, global")

    return {
        "box_indices": perturbation["box_indices"],
        "terms": selected,
    }


def perturbation_dense_matrix(n_points, perturbation, symmetry_mode="none"):

    dense = np.zeros((n_points, n_points), dtype=np.complex128)
    box = perturbation["box_indices"]
    for term in perturbation["terms"]:
        target = term["target_indices"]
        left = term["left"]
        right = term["right"]
        alpha = term["alpha"]

        if symmetry_mode == "real_symmetric":
            dense[np.ix_(box, box)] += alpha * np.outer(left, left)
        elif symmetry_mode == "complex_symmetric":
            dense[np.ix_(box, box)] += alpha * np.outer(left, left)
        else:
            dense[np.ix_(box, target)] += alpha * np.outer(left, np.conjugate(right))
        if symmetry_mode == "hermitian":
            dense[np.ix_(target, box)] += np.conjugate(alpha) * np.outer(right, np.conjugate(left))

    return dense


def make_complex_wrapped_operator(base_op, n_points, perturbation=None, symmetry_mode="none"):

    def _normalize_input(x):
        x = np.asarray(x)
        is_vector = x.ndim == 1
        x_mat = x.reshape(n_points, 1) if is_vector else x
        return is_vector, np.asarray(x_mat, dtype=np.complex128)

    def _base_apply(x_mat):
        real_part = np.asarray(base_op.matmat(np.asarray(x_mat.real, dtype=np.float64)))
        imag_part = np.asarray(base_op.matmat(np.asarray(x_mat.imag, dtype=np.float64)))
        return real_part.astype(np.complex128, copy=False) + 1j * imag_part.astype(
            np.complex128,
            copy=False,
        )

    def _forward_perturb(x_mat):
        out = np.zeros((n_points, x_mat.shape[1]), dtype=np.complex128)
        if perturbation is None:
            return out

        box = perturbation["box_indices"]
        for term in perturbation["terms"]:
            target = term["target_indices"]
            left = term["left"]
            right = term["right"]
            alpha = term["alpha"]

            if symmetry_mode == "complex_symmetric":
                coeff = left @ x_mat[box, :]
                out[box, :] += alpha * np.outer(left, coeff)
            else:
                coeff = np.conjugate(right) @ x_mat[target, :]
                out[box, :] += alpha * np.outer(left, coeff)

            if symmetry_mode == "hermitian":
                coeff_adj = np.conjugate(left) @ x_mat[box, :]
                out[target, :] += np.conjugate(alpha) * np.outer(right, coeff_adj)

        return out

    def _adjoint_perturb(x_mat):
        out = np.zeros((n_points, x_mat.shape[1]), dtype=np.complex128)
        if perturbation is None:
            return out

        box = perturbation["box_indices"]
        for term in perturbation["terms"]:
            target = term["target_indices"]
            left = term["left"]
            right = term["right"]
            alpha = term["alpha"]

            if symmetry_mode == "complex_symmetric":
                coeff = np.conjugate(left) @ x_mat[box, :]
                out[box, :] += np.conjugate(alpha) * np.outer(np.conjugate(left), coeff)
            else:
                coeff_adj = np.conjugate(left) @ x_mat[box, :]
                out[target, :] += np.conjugate(alpha) * np.outer(right, coeff_adj)

            if symmetry_mode == "hermitian":
                coeff = np.conjugate(right) @ x_mat[target, :]
                out[box, :] += alpha * np.outer(left, coeff)

        return out

    def _transpose_perturb(x_mat):
        out = np.zeros((n_points, x_mat.shape[1]), dtype=np.complex128)
        if perturbation is None:
            return out

        box = perturbation["box_indices"]
        for term in perturbation["terms"]:
            target = term["target_indices"]
            left = term["left"]
            right = term["right"]
            alpha = term["alpha"]

            if symmetry_mode == "complex_symmetric":
                coeff = left @ x_mat[box, :]
                out[box, :] += alpha * np.outer(left, coeff)
            else:
                coeff = left @ x_mat[box, :]
                out[target, :] += alpha * np.outer(np.conjugate(right), coeff)

            if symmetry_mode == "hermitian":
                coeff_t = right @ x_mat[target, :]
                out[box, :] += np.conjugate(alpha) * np.outer(np.conjugate(left), coeff_t)

        return out

    def _apply(x):
        is_vector, x_mat = _normalize_input(x)
        result = _base_apply(x_mat) + _forward_perturb(x_mat)
        if is_vector:
            return result.reshape(n_points,)
        return result

    def _apply_transpose(x):
        is_vector, x_mat = _normalize_input(x)
        result = _base_apply(x_mat) + _transpose_perturb(x_mat)
        if is_vector:
            return result.reshape(n_points,)
        return result

    def _apply_adjoint(x):
        is_vector, x_mat = _normalize_input(x)
        result = _base_apply(x_mat) + _adjoint_perturb(x_mat)
        if is_vector:
            return result.reshape(n_points,)
        return result

    forward_op = LinearOperator(
        shape=(n_points, n_points),
        dtype=np.complex128,
        matvec=_apply,
        rmatvec=_apply_adjoint,
        matmat=_apply,
        rmatmat=_apply_adjoint,
    )

    transpose_op = LinearOperator(
        shape=(n_points, n_points),
        dtype=np.complex128,
        matvec=_apply_transpose,
        rmatvec=_apply,
        matmat=_apply_transpose,
        rmatmat=_apply,
    )

    return forward_op, transpose_op


def make_real_wrapped_operator(base_op, n_points, perturbation=None, symmetry_mode="none"):

    def _normalize_input(x):
        x = np.asarray(x)
        is_vector = x.ndim == 1
        x_mat = x.reshape(n_points, 1) if is_vector else x
        return is_vector, np.asarray(x_mat, dtype=np.float64)

    def _base_apply(x_mat):
        return np.asarray(base_op.matmat(np.asarray(x_mat, dtype=np.float64)), dtype=np.float64)

    def _forward_perturb(x_mat):
        out = np.zeros((n_points, x_mat.shape[1]), dtype=np.float64)
        if perturbation is None:
            return out

        box = perturbation["box_indices"]
        for term in perturbation["terms"]:
            target = term["target_indices"]
            left = term["left"]
            right = term["right"]
            alpha = term["alpha"]

            if symmetry_mode == "real_symmetric":
                coeff = left @ x_mat[box, :]
                out[box, :] += alpha * np.outer(left, coeff)
                continue

            coeff = right @ x_mat[target, :]
            out[box, :] += alpha * np.outer(left, coeff)

        return out

    def _transpose_perturb(x_mat):
        out = np.zeros((n_points, x_mat.shape[1]), dtype=np.float64)
        if perturbation is None:
            return out

        box = perturbation["box_indices"]
        for term in perturbation["terms"]:
            target = term["target_indices"]
            left = term["left"]
            right = term["right"]
            alpha = term["alpha"]

            if symmetry_mode == "real_symmetric":
                coeff = left @ x_mat[box, :]
                out[box, :] += alpha * np.outer(left, coeff)
                continue

            coeff = left @ x_mat[box, :]
            out[target, :] += alpha * np.outer(right, coeff)

        return out

    def _apply(x):
        is_vector, x_mat = _normalize_input(x)
        result = _base_apply(x_mat) + _forward_perturb(x_mat)
        if is_vector:
            return result.reshape(n_points,)
        return result

    def _apply_transpose(x):
        is_vector, x_mat = _normalize_input(x)
        result = _base_apply(x_mat) + _transpose_perturb(x_mat)
        if is_vector:
            return result.reshape(n_points,)
        return result

    return LinearOperator(
        shape=(n_points, n_points),
        dtype=np.float64,
        matvec=_apply,
        rmatvec=_apply_transpose,
        matmat=_apply,
        rmatmat=_apply_transpose,
    )


def get_regular_grid_linear_operator(XX, scalar_func):

    N, ndim = XX.shape
    n = int(np.round(N ** (1. / ndim))) - 1
    assert np.abs((n + 1) ** ndim - N) < 1e-14

    def vec_func(XX, YY):
        return get_vectorized_kernel_func(scalar_func)(XX, YY) / N

    if ndim == 2:

        h = 1 / n
        dii = 4 * dblquad(scalar_func, 0, h / 2, 0, h / 2)[0]

        a = vec_func(XX, XX[0].reshape(1, 2))
        a = a.reshape(n + 1, n + 1)

        B = np.zeros((2 * n + 1, 2 * n + 1))
        B[:n + 1, :n + 1] = a
        B[:n + 1, n + 1:] = a[:, 1:n + 1]
        B[n + 1:, :n + 1] = a[1:n + 1, :]
        B[n + 1:, n + 1:] = a[1:n + 1, 1:n + 1]

        B[:, n + 1:] = np.fliplr(B[:, n + 1:])
        B[n + 1:, :] = np.flipud(B[n + 1:, :])

        G = np.fft.fft2(B)
        S = speye(N) * dii

        def matmat_fft(x):
            is_vector = x.ndim == 1
            x_mat = x.reshape(N, 1) if is_vector else x
            nrhs = x_mat.shape[-1]
            x_tmp = x_mat.copy().reshape(n + 1, n + 1, nrhs)

            tmp_fft = G.reshape(2 * n + 1, 2 * n + 1, 1) * np.fft.fft2(
                x_tmp,
                s=(2 * n + 1, 2 * n + 1),
                axes=(0, 1),
            )

            tmp = np.fft.ifft2(tmp_fft, axes=(0, 1))
            result = tmp[:n + 1, :n + 1, :].copy()
            result = np.real(result.reshape(N, nrhs)) + S @ x_mat

            if is_vector:
                result = result.reshape(N,)
            return result

        Aop_fast = LinearOperator(
            shape=(N, N),
            matvec=matmat_fft,
            rmatvec=matmat_fft,
            matmat=matmat_fft,
            rmatmat=matmat_fft,
        )

    elif ndim == 3:

        h = 1 / n
        dii = 8 * tplquad(scalar_func, 0, h / 2, 0, h / 2, 0, h / 2)[0]

        a = vec_func(XX, XX[0].reshape(1, 3))
        a = a.reshape(n + 1, n + 1, n + 1)

        B = np.zeros((2 * n + 1, 2 * n + 1, 2 * n + 1))
        B[:n + 1, :n + 1, :n + 1] = a
        B[:n + 1, :n + 1, n + 1:] = a[:, :, 1:n + 1]
        B[:n + 1, n + 1:, :n + 1] = a[:, 1:n + 1, :]
        B[:n + 1, n + 1:, n + 1:] = a[:, 1:n + 1, 1:n + 1]
        B[n + 1:, :n + 1, :n + 1] = a[1:n + 1, :, :]
        B[n + 1:, :n + 1, n + 1:] = a[1:n + 1, :, 1:n + 1]
        B[n + 1:, n + 1:, :n + 1] = a[1:n + 1, 1:n + 1, :]
        B[n + 1:, n + 1:, n + 1:] = a[1:n + 1, 1:n + 1, 1:n + 1]

        B[:, :, n + 1:] = np.flip(B[:, :, n + 1:], axis=2)
        B[:, n + 1:, :] = np.flip(B[:, n + 1:, :], axis=1)
        B[n + 1:, :, :] = np.flip(B[n + 1:, :, :], axis=0)

        G = np.fft.fftn(B)
        S = speye(N) * dii

        def matmat_fft(x):
            is_vector = x.ndim == 1
            x_mat = x.reshape(N, 1) if is_vector else x
            nrhs = x_mat.shape[-1]
            x_tmp = x_mat.copy().reshape(n + 1, n + 1, n + 1, nrhs)

            tmp_fft = G.reshape(2 * n + 1, 2 * n + 1, 2 * n + 1, 1) * np.fft.fftn(
                x_tmp,
                s=(2 * n + 1, 2 * n + 1, 2 * n + 1),
                axes=(0, 1, 2),
            )

            tmp = np.fft.ifftn(tmp_fft, axes=(0, 1, 2))
            result = tmp[:n + 1, :n + 1, :n + 1, :].copy()
            result = np.real(result.reshape(N, nrhs)) + S @ x_mat

            if is_vector:
                result = result.reshape(N,)
            return result

        Aop_fast = LinearOperator(
            shape=(N, N),
            matvec=matmat_fft,
            rmatvec=matmat_fft,
            matmat=matmat_fft,
            rmatmat=matmat_fft,
        )

    else:
        raise ValueError

    return Aop_fast, vec_func, S


class BIEGrid:

    def __init__(self, N, ndim):

        self.n = int(np.round(N ** (1 / ndim)))

        if ndim == 2:

            self.n = int(np.round(np.sqrt(N)))
            self.xx = get_regular_grid(self.n, 2)

            def scalar_func(x, y):
                return -(1 / (2 * np.pi)) * np.log(np.sqrt(x ** 2 + y ** 2))

        elif ndim == 3:

            self.n = int(np.round(N ** (1 / 3)))
            self.xx = get_regular_grid(self.n, 3)

            def scalar_func(x, y, z):
                return (1 / (4 * np.pi)) / np.sqrt(x ** 2 + y ** 2 + z ** 2)
        else:
            raise ValueError

        self.Aop_fast, self.vec_kernel_func, self.S = get_regular_grid_linear_operator(
            self.xx,
            scalar_func,
        )
        self.scalar_func = scalar_func
        self.N = self.xx.shape[0]

    @property
    def fast_apply_op(self):
        return self.Aop_fast

    @property
    def Amat(self):
        return self.fast_apply_op.matmat(np.eye(self.N))

    @property
    def XX(self):
        return self.xx
