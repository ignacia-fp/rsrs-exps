import bempp_cl.api
import numpy as np


grid = bempp_cl.api.shapes.sphere(h=0.5)
space = bempp_cl.api.function_space(grid, "P", 1)

print("Grid has", grid.entity_count(0), "elements")
dp0_space = bempp_cl.api.function_space(grid, "DP", 0)
p1_space = bempp_cl.api.function_space(grid, "P", 1)
mat = bempp_cl.api.operators.boundary.laplace.single_layer(
    dp0_space, p1_space, dp0_space).weak_form()


from kifmm_py import (
    KiFmm,
    LaplaceKernel,
    HelmholtzKernel,
    SingleNodeTree,
    EvalType,
    BlasFieldTranslation,
    FftFieldTranslation,
)


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

points = get_barycenters(grid)
n_sources = len(points)
print('points', points)
points = points.ravel()
print('points2', points)
np.random.seed(0)

dim = 3
dtype = np.float32

# Set FMM Parameters
expansion_order = np.array([6], np.uint64)  # Single expansion order as using n_crit
n_vec = 1
n_crit = 150
#n_sources = 1000
#n_targets = 2000
prune_empty = True  # Optionally remove empty leaf boxes, their siblings, and ancestors, from the Tree

# Setup source/target/charge data in Fortran order
sources = points.astype(dtype)#np.random.rand(n_sources * dim).astype(dtype)
targets = points.astype(dtype)#np.random.rand(n_targets * dim).astype(dtype)
charges = np.random.rand(n_sources * n_vec).astype(dtype)

print(np.random.rand(n_sources * dim).astype(dtype))
print(sources)
print(charges)

eval_type = EvalType.Value

# EvalType computes either potentials (EvalType.Value) or potentials + derivatives (EvalType.ValueDeriv)
kernel = LaplaceKernel(dtype, eval_type)

tree = SingleNodeTree(sources, targets, charges, n_crit=n_crit, prune_empty=prune_empty)

field_translation = FftFieldTranslation(kernel, block_size=32)

# Create FMM runtime object
fmm = KiFmm(expansion_order, tree, field_translation, timed=True)

# Evaluate potentials
fmm.evaluate()

# Examine potentials
fmm.all_potentials

# Examine operator times rounded in milliseconds
fmm.operator_times()

print(fmm.all_potentials.reshape(-1))

fmm.clear()

fmm.attach_charges_unordered(charges)

print('gi', fmm.target_global_indices)#.source_tree().global_indices.len())

id = np.eye(n_sources)

def mv(v):
    fmm.clear()
    charges = v.astype(dtype)
    fmm.attach_charges_unordered(charges)
    fmm.evaluate()
    res = fmm.all_potentials_u.reshape(-1)
    return res

mat = np.zeros((n_sources, n_sources))
for i in range(n_sources):
    col = id[:, i]
    mat[:, i] = mv(col)

print(mat)


import matplotlib.pyplot as plt

plt.imshow(mat, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')
plt.title('Matrix Heatmap')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

print(mat[54, 54])