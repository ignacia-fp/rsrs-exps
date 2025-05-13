import bempp_cl.api
import numpy as np


grid = bempp_cl.api.shapes.sphere(h=0.5)
space = bempp_cl.api.function_space(grid, "P", 1)

print("Grid has", grid.entity_count(0), "elements")
dp0_space = bempp_cl.api.function_space(grid, "DP", 0)
p1_space = bempp_cl.api.function_space(grid, "P", 1)
mat = bempp_cl.api.operators.boundary.laplace.single_layer(
    dp0_space, p1_space, dp0_space).weak_form()
