
import numpy as np
import bempp_cl.api
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from .geometry import get_geometry, get_barycenters, get_edges_centres
from .righ_hand_sides import right_hand_side
from scipy.sparse.linalg import LinearOperator
bempp_cl.api.GLOBAL_PARAMETERS.fmm.expansion_order = 3
#bempp_cl.api.GLOBAL_PARAMETERS.quadrature.regular = 1
#bempp_cl.api.GLOBAL_PARAMETERS.quadrature.singular = 1
from kifmm_py import (
    KiFmm,
    LaplaceKernel,
    HelmholtzKernel,
    SingleNodeTree,
    EvalType,
    FftFieldTranslation,
)
################################################################################
import warnings
import numpy as np
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
     IdentityOperator
from numpy import asanyarray, asarray, array, zeros
from scipy.linalg import get_lapack_funcs

_coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
                 ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
                 ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
                 ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
                 ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
                 ('D','D'):'D'}


def coerce(x,y):
    if x not in 'fdFD':
        x = 'd'
    if y not in 'fdFD':
        y = 'd'
    return _coerce_rules[x,y]

def _get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol


def make_system(A, M, x0, b):
    """Make a linear system Ax=b

    Parameters
    ----------
    A : LinearOperator
        sparse or dense matrix (or any valid input to aslinearoperator)
    M : {LinearOperator, Nones}
        preconditioner
        sparse or dense matrix (or any valid input to aslinearoperator)
    x0 : {array_like, str, None}
        initial guess to iterative method.
        ``x0 = 'Mb'`` means using the nonzero initial guess ``M @ b``.
        Default is `None`, which means using the zero initial guess.
    b : array_like
        right hand side

    Returns
    -------
    (A, M, x, b, postprocess)
        A : LinearOperator
            matrix of the linear system
        M : LinearOperator
            preconditioner
        x : rank 1 ndarray
            initial guess
        b : rank 1 ndarray
            right hand side
        postprocess : function
            converts the solution vector to the appropriate
            type and dimensions (e.g. (N,1) matrix)

    """
    A_ = A
    A = aslinearoperator(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix, but got shape={(A.shape,)}')

    N = A.shape[0]

    b = asanyarray(b)

    if not (b.shape == (N,1) or b.shape == (N,)):
        raise ValueError(f'shapes of A {A.shape} and b {b.shape} are '
                         'incompatible')

    if b.dtype.char not in 'fdFD':
        b = b.astype('d')  # upcast non-FP types to double

    def postprocess(x):
        return x

    if hasattr(A,'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)

    b = asarray(b,dtype=xtype)  # make b the same type as x
    b = b.ravel()

    # process preconditioner
    if M is None:
        if hasattr(A_,'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_,'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                               dtype=A.dtype)
    else:
        M = aslinearoperator(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    # set initial guess
    if x0 is None:
        x = zeros(N, dtype=xtype)
    elif isinstance(x0, str):
        if x0 == 'Mb':  # use nonzero initial guess ``M @ b``
            bCopy = b.copy()
            x = M.matvec(bCopy)
    else:
        x = array(x0, dtype=xtype)
        if not (x.shape == (N, 1) or x.shape == (N,)):
            raise ValueError(f'shapes of A {A.shape} and '
                             f'x0 {x.shape} are incompatible')
        x = x.ravel()

    return A, M, x, b, postprocess


def gmres(A, b, x0=None, *, rtol=1e-5, atol=0., restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
    """
    Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution (a vector of zeros by default).
    atol, rtol : float
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    restart : int, optional
        Number of iterations between restarts. Larger values increase
        iteration cost, but may be necessary for convergence.
        If omitted, ``min(20, n)`` is used.
    maxiter : int, optional
        Maximum number of iterations (restart cycles).  Iteration will stop
        after maxiter steps even if the specified tolerance has not been
        achieved. See `callback_type`.
    M : {sparse array, ndarray, LinearOperator}
        Inverse of the preconditioner of `A`.  `M` should approximate the
        inverse of `A` and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
        In this implementation, left preconditioning is used,
        and the preconditioned residual is minimized. However, the final
        convergence is tested with respect to the ``b - A @ x`` residual.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(args)``, where ``args`` are selected by `callback_type`.
    callback_type : {'x', 'pr_norm', 'legacy'}, optional
        Callback function argument requested:
          - ``x``: current iterate (ndarray), called on every restart
          - ``pr_norm``: relative (preconditioned) residual norm (float),
            called on every inner iteration
          - ``legacy`` (default): same as ``pr_norm``, but also changes the
            meaning of `maxiter` to count inner iterations instead of restart
            cycles.

        This keyword has no effect if `callback` is not set.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations

    See Also
    --------
    LinearOperator

    Notes
    -----
    A preconditioner, P, is chosen such that P is close to A but easy to solve
    for. The preconditioner parameter required by this routine is
    ``M = P^-1``. The inverse should preferably not be calculated
    explicitly.  Rather, use the following template to produce M::

      # Construct a linear operator that computes P^-1 @ x.
      import scipy.sparse.linalg as spla
      M_x = lambda x: spla.spsolve(P, x)
      M = spla.LinearOperator((n, n), M_x)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import gmres
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = gmres(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    if callback is not None and callback_type is None:
        # Warn about 'callback_type' semantic changes.
        # Probably should be removed only in far future, Scipy 2.0 or so.
        msg = ("scipy.sparse.linalg.gmres called without specifying "
               "`callback_type`. The default value will be changed in"
               " a future release. For compatibility, specify a value "
               "for `callback_type` explicitly, e.g., "
               "``gmres(..., callback_type='pr_norm')``, or to retain the "
               "old behavior ``gmres(..., callback_type='legacy')``"
               )
        warnings.warn(msg, category=DeprecationWarning, stacklevel=3)

    if callback_type is None:
        callback_type = 'legacy'

    if callback_type not in ('x', 'pr_norm', 'legacy'):
        raise ValueError(f"Unknown callback_type: {callback_type!r}")

    if callback is None:
        callback_type = None

    A, M, x, b, postprocess = make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = len(b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('gmres', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    eps = np.finfo(x.dtype.char).eps

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    if maxiter is None:
        maxiter = n*10

    if restart is None:
        restart = 20
    restart = min(restart, n)

    Mb_nrm2 = np.linalg.norm(psolve(b))

    # ====================================================
    # =========== Tolerance control from gh-8400 =========
    # ====================================================
    # Tolerance passed to GMRESREVCOM applies to the inner
    # iteration and deals with the left-preconditioned
    # residual.
    ptol_max_factor = 1.
    ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
    presid = 0.
    # ====================================================
    lartg = get_lapack_funcs('lartg', dtype=x.dtype)

    # allocate internal variables
    v = np.empty([restart+1, n], dtype=x.dtype)
    h = np.zeros([restart, restart+1], dtype=x.dtype)
    givens = np.zeros([restart, 2], dtype=x.dtype)

    # legacy iteration count
    inner_iter = 0

    for iteration in range(maxiter):
        if iteration == 0:
            r = b - matvec(x) if x.any() else b.copy()
            if np.linalg.norm(r) < atol:  # Are we done?
                return postprocess(x), 0

        v[0, :] = psolve(r)
        tmp = np.linalg.norm(v[0, :])
        v[0, :] *= (1 / tmp)
        # RHS of the Hessenberg problem
        S = np.zeros(restart+1, dtype=x.dtype)
        S[0] = tmp
        print(iteration)
        breakdown = False
        for col in range(restart):
            av = matvec(v[col, :])
            w = psolve(av)

            # Modified Gram-Schmidt
            h0 = np.linalg.norm(w)
            for k in range(col+1):
                tmp = dotprod(v[k, :], w)
                h[col, k] = tmp
                w -= tmp*v[k, :]

            h1 = np.linalg.norm(w)
            h[col, col + 1] = h1
            v[col + 1, :] = w[:]

            # Exact solution indicator
            if h1 <= eps*h0:
                h[col, col + 1] = 0
                breakdown = True
            else:
                v[col + 1, :] *= (1 / h1)

            # apply past Givens rotations to current h column
            for k in range(col):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = h[col, [k, k+1]]
                h[col, [k, k + 1]] = [c*n0 + s*n1, -s.conj()*n0 + c*n1]

            # get and apply current rotation to h and S
            c, s, mag = lartg(h[col, col], h[col, col+1])
            givens[col, :] = [c, s]
            h[col, [col, col+1]] = mag, 0

            # S[col+1] component is always 0
            tmp = -np.conjugate(s)*S[col]
            S[[col, col + 1]] = [c*S[col], tmp]
            presid = np.abs(tmp)
            inner_iter += 1
            print("res:", presid / bnrm2)
            if callback_type in ('legacy', 'pr_norm'):
                callback(presid / bnrm2)
            # Legacy behavior
            if callback_type == 'legacy' and inner_iter == maxiter:
                break
            if presid <= ptol or breakdown:
                break

        # Solve h(col, col) upper triangular system and allow pseudo-solve
        # singular cases as in (but without the f2py copies):
        # y = trsv(h[:col+1, :col+1].T, S[:col+1])

        if h[col, col] == 0:
            S[col] = 0

        y = np.zeros([col+1], dtype=x.dtype)
        y[:] = S[:col+1]
        for k in range(col, 0, -1):
            if y[k] != 0:
                y[k] /= h[k, k]
                tmp = y[k]
                y[:k] -= tmp*h[k, :k]
        if y[0] != 0:
            y[0] /= h[0, 0]
        print("G-Solve y:", y)
        x += y @ v[:col+1, :]

        r = b - matvec(x)
        rnorm = np.linalg.norm(r)

        # Legacy exit
        if callback_type == 'legacy' and inner_iter == maxiter:
            return postprocess(x), 0 if rnorm <= atol else maxiter

        if callback_type == 'x':
            callback(x)

        if rnorm <= atol:
            break
        elif breakdown:
            # Reached breakdown (= exact solution), but the external
            # tolerance check failed. Bail out with failure.
            break
        elif presid <= ptol:
            # Inner loop passed but outer didn't
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

        ptol = presid * min(ptol_max_factor, atol / rnorm)

    info = 0 if (rnorm <= atol) else maxiter
    return postprocess(x), info
################################################################################

def get_cond(A):
    ATA = A.H @ A # A^T A as a LinearOperator
    vals_max = eigsh(ATA, k=1, which='LM', return_eigenvectors=False)
    vals_min = eigsh(ATA, k=1, which='SM', return_eigenvectors=False)
    return np.sqrt(vals_max[0] / vals_min[0])


class BaseStructuredOperator(ABC):
    def __init__(self, dim_param, kappa, geometry_type='sphere_surface', precision='double'):
        try:
            if dim_param < 1:
                self.grid = get_geometry(geometry_type, dim_param)
                self.points = get_barycenters(self.grid)
            else:
                self.grid = None
                self.points = get_geometry(geometry_type, dim_param)
            self.precision = precision
            self.n_points = len(self.points)
            print("Number of dofs:", self.n_points)
            self.kappa = kappa
            self.mat = None  # Set in subclass
            self.rhs = None
            self.operator_type = None
            self.rhs_data_type = None
            self.operator_type = None
        except Exception as e:
            print("Error initializing BaseStructuredOperator:", e)
            raise

    @abstractmethod
    def mv(self, v):
        pass
    
    @property
    @abstractmethod
    def cond(self):
        pass
    
    @property
    @abstractmethod
    def dense(self):
        pass
    
    def rhs(self):
        pass


class BasicStructuredOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        self.operator_type = 'real'
        self.mat = (1.0/self.n_points)* np.exp(-cdist(self.points, self.points) ** 2)
        self.mat[np.diag_indices_from(self.mat)] = 1
        self.operator_type = 'BasicStructuredOperator'
        if self.precision == 'single':
            self.rhs_data_type = np.float32
        else:
            self.rhs_data_type = np.float64

    def mv(self, v):
        return self.mat @ v
    
    @property
    def cond(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return get_cond(self.mat)
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return self.mat
    
    def get_rhs(self):
        return right_hand_side(self, None)


class BemppClLaplaceSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayer:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class BemppClHelmholtzSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = self.domain
            op = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa)
            self.mat = op.weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
            
            self.rhs = self.rhs.projections(self.dual_to_range)


            x, info = gmres(self.mat, self.rhs, restart=100)
            #sol_bm, info, it_count_bm = gmres(op, self.rhs, use_strong_form=True, return_iteration_count=True)
            #print(f"GMRES converged in {it_count_bm} iterations for BemppClCombinedHelmholtz operator.")
            
            print(self.rhs[0])

        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayer:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')


class KiFMMLaplaceOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            
            self.operator_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperator'
            points = self.points.ravel()
            sources = points.astype(self.rhs_data_type)
            targets = points.astype(self.rhs_data_type)
            expansion_order = np.array([6], np.uint64)  # Single expansion order as using n_crit
            n_vec = 1
            n_crit = 150
            prune_empty = True  
            eval_type = EvalType.Value

            # EvalType computes either potentials (EvalType.Value) or potentials + derivatives (EvalType.ValueDeriv)
            structured_operator = LaplaceKernel(self.rhs_data_type, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(self.rhs_data_type)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(structured_operator, block_size=32)

            # Create FMM runtime object
            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()

            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing KiFMMLaplaceOperator:", e)
            raise

    def mv(self, v):
        self.mat.clear()
        charges = v.astype(self.rhs_data_type)
        self.mat.attach_charges_unordered(charges)
        self.mat.evaluate()
        res = np.copy(charges) + (1/self.n_points)*self.mat.all_potentials_u.reshape(-1)
        return res
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        raise ValueError("There is not a dense representation implemented for this operator.")
    
    def get_rhs(self):
        return right_hand_side(self, None)


class KiFMMHelmholtzOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                dtype = np.float32
                self.rhs_data_type = np.complex64
            else:
                dtype = np.float64
                self.rhs_data_type = np.complex128
            
            self.operator_type = 'complex'
            self.kappa = dtype(self.kappa)
            self.operator_type = 'KiFMMHelmholtzOperator'
            points = self.points.ravel()
            sources = points.astype(dtype)
            targets = points.astype(dtype)
            expansion_order = np.array([6], np.uint64)  # Single expansion order as using n_crit
            n_vec = 1
            n_crit = 150
            prune_empty = True  
            eval_type = EvalType.Value
            # EvalType computes either potentials (EvalType.Value) or potentials + derivatives (EvalType.ValueDeriv)
            structured_operator = HelmholtzKernel(dtype, self.kappa, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(self.rhs_data_type)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(structured_operator, block_size=32)

            # Create FMM runtime object
            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing KiFMMHelmholtzOperator:", e)
            raise

    def mv(self, v):
        self.mat.clear()
        charges = v.astype(self.rhs_data_type)
        self.mat.attach_charges_unordered(charges)
        self.mat.evaluate()
        res = np.copy(charges) + (1/self.n_points)*self.mat.all_potentials_u.reshape(-1)
        return res
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        raise ValueError("There is not a dense representation implemented for this operator.")
    
    def get_rhs(self):
        return right_hand_side(self, None)
    


class BemppRsLaplaceOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        self.operator_type = 'real'
        self.operator_type = 'BemppRsLaplaceOperator'
        if self.precision == 'single':
            self.rhs_data_type = np.float32
        else:
            self.rhs_data_type = np.float64

    def mv(self, v):
        raise ValueError("Mv implemented in rust.")
    
    @property
    def cond(self):
        raise ValueError("Matrix not initialized.")
    
    @property
    def dense(self):
        raise ValueError("Matrix not initialized.")
    
    def get_rhs(self):
        raise ValueError("Rhs implemented in rust.")


class BemppClLaplaceSingleLayerModified(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerModified'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range, assembler = "fmm").weak_form() + 0.5*identity.weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerModified:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class BemppClLaplaceSingleLayerCP(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCP'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            single_layer = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range, assembler = "fmm").weak_form()
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            @bempp_cl.api.real_callable
            def constant(x, n, domain_index, result):
                result[0] = 1.0

            rank_1_fun = bempp_cl.api.GridFunction(self.domain, fun=constant).projections()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            hypersingular = bempp_cl.api.operators.boundary.laplace.hypersingular(self.domain, self.range, self.dual_to_range, assembler = "fmm").weak_form()
            h_shape = hypersingular.shape

            def mv(v):
                return hypersingular*v + rank_1_fun * (rank_1_fun @ v)
            prec = LinearOperator(h_shape, matvec=mv)
            self.mat = g_inv*prec*g_inv*single_layer
            self.rhs_data_type = self.mat.dtype
            rhs = self.get_rhs()
            self.rhs = g_inv*prec*g_inv*rhs
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCP:", e)
            raise

    def mv(self, v):
        return self.mat.matvec(v)
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class BemppClLaplaceSingleLayerMM(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerMM'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.range
            sl = bempp_cl.api.operators.boundary.laplace.single_layer(self.domain, self.range, self.dual_to_range, assembler = "fmm")
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            self.mat = sl.strong_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerMM:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
 

class BemppClHelmholtzSingleLayerCP(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerCP'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = self.domain
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            hypersingular = bempp_cl.api.operators.boundary.laplace.hypersingular(self.domain, self.range, self.dual_to_range, assembler = "fmm").strong_form()
            single_layer = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa, assembler = "fmm").strong_form()
            self.mat = hypersingular*single_layer 
            self.rhs_data_type = self.mat.dtype
            self.rhs = hypersingular*self.get_rhs()
            

        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerCP:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')


class BemppClLaplaceSingleLayerCPID(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCPID'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range).weak_form()
            
            adjoint_double_layer = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            adjoint_double_layer_T = bempp_cl.api.operators.boundary.laplace.double_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            self.mat = g_inv*(0.25*identity + adjoint_double_layer*g_inv*adjoint_double_layer)
            self.mat_T = (0.25*identity + adjoint_double_layer_T*g_inv*adjoint_double_layer_T)*g_inv
            self.rhs_data_type = self.mat.dtype
            rhs = self.get_rhs()
            self.rhs = self.mat_T*rhs
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCPID:", e)
            raise

    def mv(self, v):
        return self.mat_T*(self.mat * v)
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class BemppClLaplaceSingleLayerP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()#, assembler = "fmm").weak_form()
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerP1:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class KiFMMLaplaceOperatorV(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            
            self.operator_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperatorV'
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            points = self.points.ravel()
            sources = points.astype(self.rhs_data_type)
            targets = points.astype(self.rhs_data_type)
            expansion_order = np.array([6], np.uint64)  # Single expansion order as using n_crit
            n_vec = 1
            n_crit = 150
            prune_empty = True  
            eval_type = EvalType.Value

            # EvalType computes either potentials (EvalType.Value) or potentials + derivatives (EvalType.ValueDeriv)
            structured_operator = LaplaceKernel(self.rhs_data_type, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(self.rhs_data_type)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(structured_operator, block_size=32)

            # Create FMM runtime object
            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()

            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing KiFMMLaplaceOperator:", e)
            raise

    def mv(self, v):
        self.mat.clear()
        charges = v.astype(self.rhs_data_type)
        self.mat.attach_charges_unordered(charges)
        self.mat.evaluate()
        res = np.copy(charges) + (1/self.n_points)*self.mat.all_potentials_u.reshape(-1)
        return res
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        raise ValueError("There is not a dense representation implemented for this operator.")
    
    def get_rhs(self):
        return right_hand_side(self, None)


class BemppClLaplaceSingleLayerModifiedP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerModifiedP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range, assembler = "fmm").weak_form() + 0.5*identity.weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerModified:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
  

class BemppClLaplaceSingleLayerCPIDP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCPIDP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range).weak_form()
            
            adjoint_double_layer = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            adjoint_double_layer_T = adjoint_double_layer.transpose()
            self.mat = 0.25*identity + adjoint_double_layer*g_inv*adjoint_double_layer
            self.mat_T = (0.25*identity + adjoint_double_layer_T*g_inv*adjoint_double_layer_T)*g_inv
            self.rhs_data_type = self.mat.dtype
            rhs = self.get_rhs()
            self.rhs = self.mat_T*rhs
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCPID:", e)
            raise

    def mv(self, v):
        return self.mat_T*(self.mat * v)
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')


class BemppClHelmholtzSingleLayerCPID(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerCPID'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range).weak_form()
            
            adjoint_double_layer = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            adjoint_double_layer_T = bempp_cl.api.operators.boundary.helmholtz.double_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()
            self.mat = g_inv*(0.25*identity + adjoint_double_layer*g_inv*adjoint_double_layer)
            self.mat_T = (0.25*identity + adjoint_double_layer_T*g_inv*adjoint_double_layer_T)*g_inv
            self.rhs_data_type = self.mat.dtype
            rhs = self.get_rhs()
            self.rhs = self.mat_T*rhs
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerCPID:", e)
            raise

    def mv(self, v):
        return self.mat_T*(self.mat * v)
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
   

class BemppClMaxwellEfie(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'complex'
            self.operator_type = 'BemppClMaxwellEfie'
            self.domain = bempp_cl.api.function_space(self.grid, "RWG", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "SNC", 0)
            self.mat = bempp_cl.api.operators.boundary.maxwell.electric_field(self.domain, self.dual_to_range, self.range, kappa).weak_form()
            self.points = get_edges_centres(self.grid)   
            self.n_points = len(self.points)                   
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClMaxwellEfie:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
   

class BemppClHelmholtzSingleLayerP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClHelmholtzSingleLayerP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()#, assembler = "fmm").weak_form()
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerP1:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class BemppClCombinedHelmholtz(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClCombinedHelmholtz'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = self.domain
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            )
            adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, kappa
            )
            slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa
            )
            op = (0.5 * identity + adlp - 1j * kappa * slp)
            self.mat = op.weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()

            #from bempp_cl.api.linalg import gmres

            #sol_bm, info, it_count_bm = gmres(op, self.rhs, use_strong_form=True, return_iteration_count=True)
            #print(f"GMRES converged in {it_count_bm} iterations for BemppClCombinedHelmholtz operator.")
        except Exception as e:
            print("Error initializing BemppClCombinedHelmholtz:", e)
            raise

    def mv(self, v):
        return self.mat * v
    
    @property
    def cond(self):
        raise ValueError("Condition number not implemented yet for this operator.")
    
    @property
    def dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

