from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .spec import CaseSpec


def apply(operator, vectors: np.ndarray) -> np.ndarray:
    try:
        result = operator @ vectors
    except Exception:
        result = operator * vectors
    result = np.asarray(result)
    if result.size != vectors.size:
        raise RuntimeError(f"operator returned {result.shape}, expected {vectors.shape}")
    return result.reshape(vectors.shape)


@dataclass
class BuiltKernel:
    dimension: int
    dtype: np.dtype
    symmetric: bool
    forward: Callable[[np.ndarray], np.ndarray]
    adjoint: Callable[[np.ndarray], np.ndarray] | None
    metadata: dict

    def sketch(self, stream: str, vectors: np.ndarray) -> np.ndarray:
        if stream == "y":
            return self.forward(vectors)
        if self.symmetric:
            return self.forward(vectors)
        if self.adjoint is None:
            raise RuntimeError("kernel has no adjoint application")
        return self.adjoint(vectors)


def _bempp_setup(spec: CaseSpec):
    import bempp_cl.api

    bempp_cl.api.DEFAULT_PRECISION = spec.precision
    bempp_cl.api.GLOBAL_PARAMETERS.fmm.expansion_order = int(spec.parameters.get("fmm_expansion_order", 3))
    bempp_cl.api.GLOBAL_PARAMETERS.quadrature.regular = int(spec.parameters.get("quadrature_regular", 2))
    return bempp_cl.api


def _grid(spec: CaseSpec, api):
    if spec.mesh:
        path = Path(spec.mesh)
        if not path.is_file():
            raise FileNotFoundError(path)
        return api.import_grid(str(path))
    if spec.geometry == "sphere" and spec.refinement_level is not None:
        return api.shapes.regular_sphere(spec.refinement_level)
    raise ValueError("provide a mesh, or geometry=sphere with refinement_level")


def build_f16_helmholtz_combined(spec: CaseSpec) -> BuiltKernel:
    if spec.kappa is None:
        raise ValueError("f16_helmholtz_combined requires kappa")
    api = _bempp_setup(spec)
    grid = _grid(spec, api)
    space = api.function_space(grid, "DP", 0)
    dimension = int(space.global_dof_count)
    if spec.expected_dofs is not None and dimension != spec.expected_dofs:
        raise RuntimeError(f"grid has {dimension} DOFs, expected {spec.expected_dofs}")
    identity = api.operators.boundary.sparse.identity(space, space, space).weak_form()
    single = api.operators.boundary.helmholtz.single_layer(
        space, space, space, spec.kappa, assembler=spec.assembler
    ).weak_form()
    adj_double = api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, spec.kappa, assembler=spec.assembler
    ).weak_form()
    operator = 0.5 * identity + adj_double - 1j * spec.kappa * single
    transpose = 0.5 * identity + adj_double.T - 1j * spec.kappa * single.T
    return BuiltKernel(
        dimension=dimension,
        dtype=np.dtype(np.complex128 if spec.precision == "double" else np.complex64),
        symmetric=False,
        forward=lambda x: apply(operator, x),
        # SketchData with TransMode::Trans conjugates both the test and sketch
        # while placing vectors into row-oriented storage.  Therefore the
        # canonical stored relation is z_sketch = conj(A^T conj(z_test)).
        # Current Gaussian tests are real, but retaining both conjugations is
        # important for an unambiguous complex-data contract.
        adjoint=lambda x: np.conj(apply(transpose, np.conj(x))),
        metadata={"dofs": dimension, "trans_mode": "Trans", "grid": spec.mesh},
    )


def build_laplace_single_layer(spec: CaseSpec) -> BuiltKernel:
    api = _bempp_setup(spec)
    grid = _grid(spec, api)
    space = api.function_space(grid, "DP", 0)
    dimension = int(space.global_dof_count)
    if spec.expected_dofs is not None and dimension != spec.expected_dofs:
        raise RuntimeError(f"grid has {dimension} DOFs, expected {spec.expected_dofs}")
    operator = api.operators.boundary.laplace.single_layer(
        space, space, space, assembler=spec.assembler
    ).weak_form()
    return BuiltKernel(
        dimension=dimension,
        dtype=np.dtype(np.float64 if spec.precision == "double" else np.float32),
        symmetric=True,
        forward=lambda x: apply(operator, x),
        adjoint=None,
        metadata={"dofs": dimension, "trans_mode": "Symmetric", "grid": spec.mesh},
    )


REGISTRY = {
    "f16_helmholtz_combined": build_f16_helmholtz_combined,
    "laplace_single_layer": build_laplace_single_layer,
}


def build_kernel(spec: CaseSpec) -> BuiltKernel:
    try:
        builder = REGISTRY[spec.kernel]
    except KeyError as exc:
        raise ValueError(f"unknown kernel {spec.kernel!r}; choose from {sorted(REGISTRY)}") from exc
    return builder(spec)
