import numpy as np
import bempp_cl.api

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


def get_edges_centres(grid):
    barycenters = []
    for el in grid.edges.T:
        v1 = grid.vertices.T[el[0]]
        v2 = grid.vertices.T[el[1]]
        barycenters.append(0.5*(v1+v2))
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
    elif geometry_type == 'satellite1':
        return bempp_cl.api.import_grid('./extra_meshes/satellite_1.msh')