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
