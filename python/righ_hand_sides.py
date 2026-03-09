
from unittest import result
import numpy as np
import bempp_cl.api

def generate_directions(n_dirs):
    """Generate n_dirs approximately uniform unit directions on the sphere."""
    if n_dirs <= 1:
        return np.array([[1.0, 0.0, 0.0]])

    indices = np.arange(0, n_dirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_dirs)
    theta = np.pi * (1 + 5**0.5) * indices  # golden angle

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    dirs = np.stack((x, y, z), axis=1)

    # 🔒 Explicit normalization (for safety)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return dirs

import numpy as np

def generate_sources(grid, n_sources, padding=1.2):
    """Generate monopole source positions around a geometry.
    
    Parameters
    ----------
    grid : bempp.api.Grid
        The geometry grid of the scatterer.
    n_sources : int
        Number of monopole sources to place.
    padding : float
        Factor to expand the bounding radius ( >1 places sources outside ).
    """
    # Compute approximate bounding sphere
    vertices = np.array(grid.vertices)
    center = np.mean(vertices, axis=1)
    max_radius = np.max(np.linalg.norm(vertices.T - center, axis=1))
    radius = padding * max_radius

    if n_sources <= 1:
        return np.array([center + np.array([radius, 0.0, 0.0])])

    # Fibonacci sphere for uniform angular distribution
    indices = np.arange(0, n_sources, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_sources)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    directions = np.stack((x, y, z), axis=1)

    sources = center + radius * directions
    return sources


def l_dirichlet_data(s):
    s = np.asarray(s, dtype=np.float64)

    @bempp_cl.api.real_callable
    def dirichlet_data(x, n, domain_index, result):
        r = np.linalg.norm(x - s)
        result[0] = 1.0 / (4 * np.pi * r)

    return dirichlet_data

def h_dirichlet_data(d, kappa):
    d = np.asarray(d, dtype=np.float64)

    @bempp_cl.api.complex_callable
    def dirichlet_data(x, n, domain_index, result):
        result[0] = -np.exp(1j * kappa * np.dot(d, x))

    return dirichlet_data

def h_neumann_data(d, kappa):
    d = np.asarray(d, dtype=np.float64)

    @bempp_cl.api.complex_callable
    def neumann_data(x, n, domain_index, result):
        result[0] = -1j * kappa * np.exp(1j * kappa * np.dot(d, x)) * np.dot(n, d)

    return neumann_data

def m_dirichlet_data(d, kappa):
    d = np.asarray(d, dtype=np.float64)
    polarization = np.array([-1.0, 0, 0])
    @bempp_cl.api.complex_callable
    def dirichlet_data(x, n, domain_index, result):
        incident_field = polarization * np.exp(1j * kappa * np.dot(d, x))
        result[:] = np.cross(incident_field, n)

    return dirichlet_data

def h_combined_data(d, kappa):
    d = np.asarray(d, dtype=np.float64)

    @bempp_cl.api.complex_callable
    def combined_data(x, n, domain_index, result):
        result[0] = 1j * kappa * np.exp(1j * kappa * np.dot(d, x)) * (np.dot(n, d) - 1)

    return combined_data


def h_bm_data(d, k, normalise=True):
    d = np.asarray(d, dtype=np.float64)

    @bempp_cl.api.complex_callable
    def bm_rhs(x, n, domain_index, result):
        phase = np.exp(1j * k * np.dot(d, x))
        result[0] = phase * (1.0 - np.dot(n, d))

    return bm_rhs


def right_hand_side(operator, problem_type, n_sources=1):
    """
    Construct RHS vector(s) for a given operator and problem type.
    Always returns a list of np.ndarray (one per source).

    If operator.precision == "single":
        downcast returned vectors to float32 (Laplace) or complex64 (Helmholtz/Maxwell).
    If operator.precision == "double":
        leave as-is.
    """

    # ----------------------------
    # Downcast policy
    # ----------------------------
    prec = operator.precision
    if prec not in ("single", "double"):
        raise ValueError("operator.precision must be 'single' or 'double'")

    # choose target dtype for returned numpy vectors
    if prec == "single":
        # Laplace -> real32, everything else here -> complex64
        target_dtype = np.float32 if "Laplace" in operator.operator_type else np.complex64
    else:
        target_dtype = None  # no casting

    def _maybe_cast(v):
        """Flatten and optionally downcast to operator precision."""
        a = np.asarray(v).ravel()
        if target_dtype is not None and a.dtype != np.dtype(target_dtype):
            a = a.astype(target_dtype, copy=False)
        return a

    def _gridfun_to_vec(gfun):
        """Extract coefficients/projections and apply casting."""
        v = gfun.projections(operator.dual_to_range) if operator.form == "weak" else gfun.coefficients
        return _maybe_cast(v)

    undefined_rhs = {
        'BasicStructuredOperator', 'KiFMMLaplaceOperator', 'KiFMMHelmholtzOperator',
        'KiFMMLaplaceOperatorV', 'BemppClLaplaceSingleLayerCPID',
        'BemppClLaplaceSingleLayerCPIDP1', 'BemppClHelmholtzSingleLayerCPID'
    }

    # ---------------------------------------------------------------------
    # Fallback operator: random RHS
    # ---------------------------------------------------------------------
    if operator.operator_type in undefined_rhs:
        print("Warning: undefined problem type for this operator. Returning random vector(s).")
        n = operator.n_points
        if "Laplace" in operator.operator_type:
            gen = lambda: np.random.rand(n)
        else:
            gen = lambda: np.random.rand(n) + 1j * np.random.rand(n)
        return [_maybe_cast(gen()) for _ in range(n_sources)]

    # ---------------------------------------------------------------------
    # LAPLACE SINGLE LAYER
    # ---------------------------------------------------------------------
    elif operator.operator_type.startswith("BemppClLaplaceSingleLayer"):
        identity = bempp_cl.api.operators.boundary.sparse.identity(
            operator.domain, operator.range, operator.dual_to_range)
        dlp = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
            operator.domain, operator.range, operator.domain, assembler=operator.assembler)

        if problem_type == 'Dirichlet':
            rhs_list = []
            sources = generate_sources(operator.domain.grid, n_sources)

            for s in sources:
                d_data = l_dirichlet_data(s)  # keep your existing callable
                gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
                rhs_gfun = (dlp - 0.5 * identity) * gfun
                rhs_list.append(_gridfun_to_vec(rhs_gfun))

            return rhs_list

        elif problem_type == 'Neumann':
            raise ValueError("Neumann problem not implemented for Laplace.")

    # ---------------------------------------------------------------------
    # HELMHOLTZ SINGLE LAYER
    # ---------------------------------------------------------------------
    elif operator.operator_type.startswith("BemppClHelmholtzSingleLayer"):
        kappa = operator.kappa

        if problem_type == 'Dirichlet':
            rhs_list = []
            directions = generate_directions(n_sources)

            for d in directions:
                d_data = h_dirichlet_data(d, kappa)  # keep your existing callable
                gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
                rhs_list.append(_gridfun_to_vec(gfun))

            return rhs_list

        elif problem_type == 'Neumann':
            raise ValueError("Neumann problem not implemented for Helmholtz.")

    # ---------------------------------------------------------------------
    # LAPLACE COMBINED
    # ---------------------------------------------------------------------
    elif operator.operator_type == 'BemppClLaplaceCombined':
        rhs_list = []
        sources = generate_sources(operator.domain.grid, n_sources)

        for s in sources:
            d_data = l_dirichlet_data(s)
            gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
            rhs_list.append(_gridfun_to_vec(gfun))

        return rhs_list

    # ---------------------------------------------------------------------
    # LAPLACE SECOND
    # ---------------------------------------------------------------------
    elif operator.operator_type == 'BemppClLaplaceSecond':
        rhs_list = []
        sources = generate_sources(operator.domain.grid, n_sources)

        for s in sources:
            d_data = l_dirichlet_data(s)
            gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
            rhs_list.append(_gridfun_to_vec(gfun))

        return rhs_list

    # ---------------------------------------------------------------------
    # HELMHOLTZ COMBINED
    # ---------------------------------------------------------------------
    elif operator.operator_type == 'BemppClHelmholtzCombined':
        kappa = operator.kappa

        if problem_type == 'Dirichlet':
            rhs_list = []
            directions = generate_directions(n_sources)

            for d in directions:
                d_data = h_combined_data(d, kappa)
                gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
                rhs_list.append(_gridfun_to_vec(gfun))

            return rhs_list

        elif problem_type == 'Neumann':
            raise ValueError("Neumann problem not implemented for Helmholtz.")

    # ---------------------------------------------------------------------
    # MAXWELL EFIE
    # ---------------------------------------------------------------------
    elif operator.operator_type == 'BemppClMaxwellEfie':
        kappa = operator.kappa
        rhs_list = []
        directions = generate_directions(n_sources)

        for d in directions:
            d_data = m_dirichlet_data(d, kappa)
            gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
            rhs_list.append(_gridfun_to_vec(gfun))

        return rhs_list

    # ---------------------------------------------------------------------
    # BURTON–MILLER
    # ---------------------------------------------------------------------
    elif operator.operator_type == 'BemppClBurtonMiller':
        kappa = operator.kappa
        rhs_list = []
        directions = generate_directions(n_sources)

        for d in directions:
            d_data = h_bm_data(d, kappa)
            gfun = bempp_cl.api.GridFunction(operator.domain, fun=d_data)
            rhs_list.append(_gridfun_to_vec(gfun))

        return rhs_list

    # ---------------------------------------------------------------------
    # Catch-all
    # ---------------------------------------------------------------------
    else:
        raise ValueError(f"Unsupported operator type: {operator.operator_type}")