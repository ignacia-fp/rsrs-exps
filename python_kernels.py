from scipy.spatial.distance import cdist
import numpy as np


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
    def __init__(self, n):
        self.dim = n
        self.points = generate_random_points_on_sphere(self.dim)
        self.mat = None  # Set in subclass

    def mv(self, v):
        return self.mat @ v

class Kernel(BaseKernel):
    def __init__(self, n, _kappa):
        super().__init__(n)
        self.mat = (1.0/n)* np.exp(-cdist(self.points, self.points) ** 2)
        self.mat[np.diag_indices_from(self.mat)] = 1


class HelmholtzKernel(BaseKernel):
    def __init__(self, n, kappa):
        super().__init__(n)
        try:
            self.mat = cdist(self.points, self.points)
            kappa = 1j * kappa
            self.mat = (1.0 / n) * np.exp(kappa * self.mat) / (4 * np.pi * self.mat)
            self.mat[np.diag_indices_from(self.mat)] = 1
        except Exception as e:
            print("Error initializing HelmholtzKernel:", e)
            raise  # Let the caller handle the failure


class LaplaceKernel(BaseKernel):
    def __init__(self, n, _kappa):
        super().__init__(n)
        try:
            self.mat = cdist(self.points, self.points)
            self.mat = (1.0 / n) * 1.0 / (4 * np.pi * self.mat)
            self.mat[np.diag_indices_from(self.mat)] = 1
        except Exception as e:
            print("Error initializing LaplaceKernel:", e)
            raise  # Let the caller handle the failure