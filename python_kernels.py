from scipy.spatial.distance import cdist
import numpy as np
import bempp_cl.api
from scipy.sparse.linalg import eigsh
import time
from kifmm_py import (
    KiFmm,
    LaplaceKernel,
    HelmholtzKernel,
    SingleNodeTree,
    EvalType,
    BlasFieldTranslation,
    FftFieldTranslation,
)

def generate_random_points_on_sphere(n_points, radius=1):
    # Generate random polar and azimuthal angles
    theta = np.arccos(2 * np.random.rand(n_points) - 1)  # Polar angle
    phi = 2 * np.pi * np.random.rand(n_points)            # Azimuthal angle

    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Stack into a list of points
    points = np.column_stack((x, y, z))
    
    return points


def random_points_in_sphere(n_points, radius=1.0, center=(0.0, 0.0, 0.0)):
    # Generate random directions
    vec = np.random.normal(size=(n_points, 3))
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    
    # Generate radii with cube root to ensure uniform volume sampling
    r = radius * np.random.random(n_points) ** (1/3)
    
    # Scale unit vectors by radii
    points = vec * r[:, np.newaxis]
    
    # Shift by the sphere's center
    return points + np.array(center)


def random_points_in_cube(n_points, side_length=1.0, center=(0.0, 0.0, 0.0)):
    half = side_length / 2
    points = np.random.uniform(
        low=-half, high=half, size=(n_points, 3)
    )
    return points + np.array(center)


def random_points_in_trefoil_knot(n_points=1000, noise=0.0):
    # Parametric equations for a trefoil knot
    t = np.random.uniform(0, 2 * np.pi, n_points)
    x = (2 + np.cos(3 * t)) * np.cos(2 * t)
    y = (2 + np.cos(3 * t)) * np.sin(2 * t)
    z = np.sin(3 * t)

    # Add isotropic Gaussian noise if specified
    if noise > 0:
        x += np.random.normal(0, noise, n_points)
        y += np.random.normal(0, noise, n_points)
        z += np.random.normal(0, noise, n_points)

    return np.stack((x, y, z), axis=1)

class BaseKernel:
    def __init__(self, dim_param, geometry_type='sphere_surface'):
        try:
            if dim_param < 1:
                self.grid = get_geometry(geometry_type, dim_param)
                self.points = get_barycenters(self.grid)
            else:
                self.grid = None
                self.points = get_geometry(geometry_type, dim_param)
            self.n_points = len(self.points)
            self.mat = None  # Set in subclass
            self.rhs = None
        except Exception as e:
            print("Error initializing BaseKernel:", e)
            raise

    def mv(self, v):
        return self.mat @ v
    
    def get_cond(self):
        ATA = self.mat.H @ self.mat # A^T A as a LinearOperator
        vals_max = eigsh(ATA, k=1, which='LM', return_eigenvectors=False)
        vals_min = eigsh(ATA, k=1, which='SM', return_eigenvectors=False)
        return np.sqrt(vals_max[0] / vals_min[0])
    
    def get_dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return self.mat
    
    def get_rhs(self):
        raise ValueError("get_rhs not implemented for this kernel class.")


class Kernel(BaseKernel):
    def __init__(self, dim_param, geometry_type, _kappa):
        super().__init__(dim_param, geometry_type)
        self.mat = (1.0/self.n_points)* np.exp(-cdist(self.points, self.points) ** 2)
        self.mat[np.diag_indices_from(self.mat)] = 1


class HelmholtzKernel(BaseKernel):
    def __init__(self, dim_param, geometry_type, kappa):
        super().__init__(dim_param, geometry_type)
        try:
            self.mat = cdist(self.points, self.points)
            self.mat = self.mat.astype(np.complex128) 
            kappa = 1.0j * kappa
            self.mat = (1.0 / self.n_points) * np.exp(kappa * self.mat) / (4.0 * np.pi * self.mat)
            self.mat[np.diag_indices_from(self.mat)] = 1
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing HelmholtzKernel:", e)
            raise 

    def get_rhs(self):
        np.random.rand(self.n_points)


class SimplePythonLaplaceKernel(BaseKernel):
    def __init__(self, dim_param, geometry_type, _kappa):
        super().__init__(dim_param, geometry_type)
        try:
            self.mat = cdist(self.points, self.points)
            self.mat = (1.0 / self.n_points) * 1.0 / (4 * np.pi * self.mat)
            self.mat[np.diag_indices_from(self.mat)] = 1
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing LaplaceKernel:", e)
            raise  

    def get_rhs(self):
        np.random.rand(self.n_points)


class BemLaplaceKernel(BaseKernel):
    def __init__(self, dim_param, geometry_type, _kappa):
        super().__init__(dim_param, geometry_type)
        try:
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            print("Number of dofs:", self.n_points)
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemLaplaceKernel:", e)
            raise

    def mv(self, v):
        start_time = time.time()
        res = self.mat * v
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Mv time: ", elapsed_time) 
        return res
    
    def get_dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        @bempp_cl.api.real_callable
        def dirichlet_data(x, n, domain_index, result):
            result[0] = 1./(4 * np.pi *((x[0] - .9)**2 + x[1]**2 + x[2]**2)**(0.5))
            
        dirichlet_fun = bempp_cl.api.GridFunction(self.domain, fun=dirichlet_data)

        identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                self.range,
                                                                self.dual_to_range)
        dlp = bempp_cl.api.operators.boundary.laplace.double_layer(self.domain,
                                                                self.range,
                                                                self.domain)

        rhs = (.5 * identity + dlp) * dirichlet_fun

        return rhs.projections()
    

class BemHelmholtzKernel(BaseKernel):
    def __init__(self, dim_param, geometry_type, kappa):
        super().__init__(dim_param, geometry_type)
        try:
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()
            self.kappa = kappa
            self.rhs = self.get_rhs()
            print("Number of dofs:", self.n_points, kappa)

        except Exception as e:
            print("Error initializing BemHelmholtzKernel:", e)
            raise
    def mv(self, v):
        return self.mat * v
    
    def get_dense(self):
        if self.mat is None:
            raise ValueError("Matrix not initialized.")
        return bempp_cl.api.as_matrix(self.mat)
    
    def get_rhs(self):
        kappa = self.kappa
        @bempp_cl.api.complex_callable
        def dirichlet_data(x, n, domain_index, result):
            result[0] = np.exp(1j * kappa * x[0])

        dirichlet_fun = bempp_cl.api.GridFunction(self.domain, fun=dirichlet_data)

        identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                self.range,
                                                                self.dual_to_range)
        dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(self.domain,
                                                                self.range,
                                                                self.domain, self.kappa)

        rhs = (.5 * identity + dlp) * dirichlet_fun

        return rhs.projections()


dtype = np.float64

class KiFMMLaplaceKernel(BaseKernel):
    def __init__(self, dim_param, geometry_type, _kappa):
        super().__init__(dim_param, geometry_type)
        try:
            
            points = self.points.ravel()
            sources = points.astype(dtype)
            targets = points.astype(dtype)
            expansion_order = np.array([6], np.uint64)  # Single expansion order as using n_crit
            n_vec = 1
            n_crit = 150
            prune_empty = True  
            eval_type = EvalType.Value

            # EvalType computes either potentials (EvalType.Value) or potentials + derivatives (EvalType.ValueDeriv)
            kernel = LaplaceKernel(dtype, eval_type)
            charges = np.zeros(self.n_points * n_vec).astype(dtype)
            tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)
            field_translation = FftFieldTranslation(kernel, block_size=32)
            # Create FMM runtime object
            self.mat = KiFmm(expansion_order, tree, field_translation, timed=True)
            self.mat.clear()
            print("Number of dofs:", self.n_points)
            self.rhs = self.get_rhs()
        except Exception as e:
            print("Error initializing BemLaplaceKernel:", e)
            raise

    def mv(self, v):
        self.mat.clear()
        charges = v.astype(dtype)
        self.mat.attach_charges_unordered(charges)
        self.mat.evaluate()
        res = np.copy(charges) + (1/self.n_points)*self.mat.all_potentials_u.reshape(-1)
        return res
    
    def get_dense(self):
        raise ValueError("Dense is not implemented.")
    
    def get_rhs(self):
        return np.random.rand(self.n_points).astype(dtype)
    


def get_barycenters(grid):
    barycenters = []
    for el in grid.elements.T:
        v1 = grid.vertices.T[el[0]]
        v2 = grid.vertices.T[el[1]]
        v3 = grid.vertices.T[el[2]]
        x = (v1[0] + v2[0] + v3[0])/3
        y = (v1[1] + v2[1] + v3[1])/3
        z = (v1[2] + v2[2] + v3[2])/3
        barycenters.append([x, y, z])
    return np.array(barycenters)


def get_geometry(geometry_type, dim_param):
    if geometry_type == 'sphere_surface':
        if dim_param < 1:
            return bempp_cl.api.shapes.sphere(h=dim_param)
        else:
            return generate_random_points_on_sphere(dim_param)
    elif geometry_type == 'cube_surface':
        return bempp_cl.api.shapes.cube(h=dim_param)
    elif geometry_type == 'cylinder_surface':   
        return bempp_cl.api.shapes.cylinders(h=dim_param, r=[0.5])
    elif geometry_type == 'ellipsoid_surface':
        return bempp_cl.api.shapes.ellipsoid(h=dim_param)
    elif geometry_type == 'trefoil_knot':
        random_points_in_trefoil_knot(n_points=dim_param)
    elif geometry_type == 'sphere': 
        random_points_in_sphere(n_points=dim_param)
    elif geometry_type == 'cube': 
        random_points_in_cube(n_points=dim_param)
    elif geometry_type == 'sheets':
        corners1 = np.array([[-.5, -1, 0],
                             [-.5, 1, 0],
                             [-2, 1, 2],
                             [-2, -1, 2]])
        corners2 = np.array([[.5, -1, 0],
                            [.5, 1, 0],
                            [2, 1, 2],
                            [2, -1, 2]])
        corners3 = np.array([[-1, -1, -1],
                            [1, -1, -1],
                            [1, 1, -1],
                            [-1, 1, -1]])

        grid1 = bempp_cl.api.shapes.screen(corners1)
        grid2 = bempp_cl.api.shapes.screen(corners2)
        grid3 = bempp_cl.api.shapes.screen(corners3)
        return bempp_cl.api.grid.union([grid1, grid2, grid3])