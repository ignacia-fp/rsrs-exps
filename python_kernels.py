from scipy.spatial.distance import cdist
import numpy as np
import bempp_cl.api
from scipy.sparse.linalg import LinearOperator, eigsh

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


class BaseKernel:
    def __init__(self, dim_param):
        self.n_points = None
        self.points = None
        self.mat = None  # Set in subclass

    def mv(self, v):
        return self.mat @ v
    
    def get_cond(self):
        ATA = self.mat.H @ self.mat # A^T A as a LinearOperator
        vals_max = eigsh(ATA, k=1, which='LM', return_eigenvectors=False)
        vals_min = eigsh(ATA, k=1, which='SM', return_eigenvectors=False)
        return np.sqrt(vals_max[0] / vals_min[0])


class Kernel(BaseKernel):
    def __init__(self, dim_param, _kappa):
        super().__init__(dim_param)
        self.n_points = dim_param
        self.points = generate_random_points_on_sphere(self.n_points)
        self.mat = (1.0/dim_param)* np.exp(-cdist(self.points, self.points) ** 2)
        self.mat[np.diag_indices_from(self.mat)] = 1


class HelmholtzKernel(BaseKernel):
    def __init__(self, dim_param, kappa):
        super().__init__(dim_param)
        self.n_points = dim_param
        self.points = generate_random_points_on_sphere(self.n_points)
        try:
            self.mat = cdist(self.points, self.points)
            kappa = 1j * kappa
            self.mat = (1.0 / dim_param) * np.exp(kappa * self.mat) / (4 * np.pi * self.mat)
            self.mat[np.diag_indices_from(self.mat)] = 1
        except Exception as e:
            print("Error initializing HelmholtzKernel:", e)
            raise  # Let the caller handle the failure


class LaplaceKernel(BaseKernel):
    def __init__(self, dim_param, _kappa):
        super().__init__(dim_param)
        self.n_points = dim_param
        self.points = generate_random_points_on_sphere(self.n_points)
        try:
            self.mat = cdist(self.points, self.points)
            self.mat = (1.0 / dim_param) * 1.0 / (4 * np.pi * self.mat)
            self.mat[np.diag_indices_from(self.mat)] = 1
        except Exception as e:
            print("Error initializing LaplaceKernel:", e)
            raise  # Let the caller handle the failure


class BemLaplaceKernel(BaseKernel):
    def __init__(self, dim_param, _kappa):
        super().__init__(dim_param)
        try:
            grid = get_geometry('sphere', dim_param)
            dp0_space = bempp_cl.api.function_space(grid, "DP", 0)
            p1_space = bempp_cl.api.function_space(grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                dp0_space, p1_space, dp0_space).weak_form()
            self.points = get_barycenters(grid)
            self.n_points = len(self.points)
            print("Number of dofs:", self.n_points)
        except Exception as e:
            print("Error initializing BemLaplaceKernel:", e)
            raise
    def mv(self, v):
        return self.mat * v
    

class BemHelmholtzKernel(BaseKernel):
    def __init__(self, dim_param, kappa):
        super().__init__(dim_param)
        try:
            grid = get_geometry('sphere', dim_param)
            dp0_space = bempp_cl.api.function_space(grid, "DP", 0)
            p1_space = bempp_cl.api.function_space(grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                dp0_space, p1_space, dp0_space, kappa).weak_form()
            self.points = get_barycenters(grid)
            self.n_points = len(self.points)
            print("Number of dofs:", self.n_points)
        except Exception as e:
            print("Error initializing BemHelmholtzKernel:", e)
            raise
    def mv(self, v):
        return self.mat * v


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


def get_geometry(geometry_type, h):
    if geometry_type == 'sphere':
        return bempp_cl.api.shapes.sphere(h=h)
    elif geometry_type == 'sheet':
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