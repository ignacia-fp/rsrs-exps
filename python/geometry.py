import numpy as np
import bempp_cl.api

def generate_random_points_on_sphere(n_points, radius=1):
    rng = np.random.default_rng(25)
    u = rng.random(n_points)
    v = rng.random(n_points)

    theta = np.arccos(2 * u - 1)
    phi = 2 * np.pi * v

    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Stack into a list of points
    points = np.column_stack((x, y, z))
    
    return points


import numpy as np

def generate_random_points_on_square(
    n_points,
    origin=np.array([0.0, 0.0, 0.0]),
    u_vec=np.array([1.0, 0.0, 0.0]),
    v_vec=np.array([0.0, 1.0, 0.0]),
    centered=True
):
    """
    Generate uniform random points on a 1x1 square in 3D.

    Default square:
        - lies in the XY plane
        - centered at the origin
        - spans x,y ∈ [-0.5, 0.5], z = 0

    Parameters
    ----------
    n_points : int
        Number of points.
    origin : array-like, shape (3,), optional
        Center of the square if centered=True, otherwise a corner.
    u_vec, v_vec : array-like, shape (3,), optional
        Edge direction vectors spanning the square (not necessarily unit).
    centered : bool
        If True, origin is interpreted as the *center* of the square.
        If False, origin is interpreted as the *lower-left corner*.

    Returns
    -------
    points : ndarray of shape (n_points, 3)
    """
    '''rng = np.random.default_rng(25)
    origin = np.asarray(origin, dtype=float)
    u_vec = np.asarray(u_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    # Normalize spans so that the final square is 1×1
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)

    rng = np.random.default_rng(25)

    # Sample s,t ∈ [0,1]
    s = rng.random(n_points)
    t = rng.random(n_points)

    # If centered, shift square so that origin is at its center
    if centered:
        # Shift so that (s,t) ∈ [-0.5, 0.5]
        s = s - 0.5
        t = t - 0.5

    points = origin + np.outer(s, u_vec) + np.outer(t, v_vec)
    return points'''

    return sample_points_on_square(level=6, origin=np.array([0.0, 0.0, 0.000001]), total_points=n_points)


def sample_points_on_square(
    level: int,
    total_points: int,
    origin=np.array([0.0, 0.0, 0.0]),
    u_vec=np.array([1.0, 0.0, 0.0]),
    v_vec=np.array([0.0, 1.0, 0.0]),
    centered: bool = True,
    seed: int = 25,
):
    """
    Generate `total_points` uniformly on a flat 1x1 square in 3D,
    distributed evenly across a uniform level-L quadtree on the square.

    Returns
    -------
    points : (total_points, 3)
        3D coordinates of points on the square.
    """
    if level < 0:
        raise ValueError("level must be >= 0")
    if total_points <= 0:
        raise ValueError("total_points must be >= 1")

    rng = np.random.default_rng(seed)

    origin = np.asarray(origin, float)
    u_vec = np.asarray(u_vec, float)
    v_vec = np.asarray(v_vec, float)

    # Normalize directions so square is 1x1
    u_hat = u_vec / np.linalg.norm(u_vec)
    v_hat = v_vec / np.linalg.norm(v_vec)

    n = 2 ** level
    n_leaves = n * n

    if total_points < n_leaves:
        raise ValueError(
            "total_points < number of leaves: cannot place at least one point per leaf"
        )

    q, r = divmod(total_points, n_leaves)

    # enumerate leaf indices (i,j)
    ii, jj = np.meshgrid(
        np.arange(n, dtype=int),
        np.arange(n, dtype=int),
        indexing="ij",
    )
    leaf_ij = np.stack([ii.ravel(), jj.ravel()], axis=1)

    counts = np.full(n_leaves, q, dtype=int)
    counts[:r] += 1

    leaf_ij_rep = np.repeat(leaf_ij, counts, axis=0)

    cell = 1.0 / n
    uv = rng.random((total_points, 2))

    s = leaf_ij_rep[:, 0] * cell + uv[:, 0] * cell
    t = leaf_ij_rep[:, 1] * cell + uv[:, 1] * cell

    if centered:
        s -= 0.5
        t -= 0.5

    points = origin + s[:, None] * u_hat + t[:, None] * v_hat
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
    from extra_meshes import shapes as sh
    if geometry_type == 'sphere_surface':
        if dim_param < 1:
            return bempp_cl.api.shapes.sphere(h=dim_param)
        else:
            return generate_random_points_on_sphere(dim_param)
    elif geometry_type =='square':
        return generate_random_points_on_square(n_points=dim_param)
    elif geometry_type == 'cube_surface':
        return bempp_cl.api.shapes.cube(h=dim_param)
    elif geometry_type == 'cylinder_surface':   
        return bempp_cl.api.shapes.cylinders(h=dim_param, r=[0.5])
    elif geometry_type == 'ellipsoid_surface':
        return bempp_cl.api.shapes.ellipsoid(h=dim_param)
    elif geometry_type == 'trefoil_knot':
        return random_points_in_trefoil_knot(n_points=dim_param)
    elif geometry_type == 'sphere': 
        return random_points_in_sphere(n_points=dim_param)
    elif geometry_type == 'cube': 
        return random_points_in_cube(n_points=dim_param)
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
    elif geometry_type == 'dihedral':
        return sh.dihedral_with_y(h=dim_param)
    elif geometry_type == 'device':
        return sh.device(h=dim_param)
    elif geometry_type == 'f16':
        return sh.f16(h=dim_param)
    elif geometry_type == 'ridged_horn':
        return sh.ridged_horn_tem_antenna(h=dim_param)
    elif geometry_type == 'emcc_almond':
        return sh.emcc_almond(h=dim_param)
    elif geometry_type == 'frigate_hull':
        return sh.frigate_hull(h=dim_param)
    elif geometry_type == 'plane':
        return sh.plane(h=dim_param)