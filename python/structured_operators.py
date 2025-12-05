
import os
import numpy as np
import bempp_cl.api
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from .geometry import get_geometry, get_barycenters, get_edges_centres
from .righ_hand_sides import right_hand_side
from scipy.sparse.linalg import LinearOperator
from pathlib import Path
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

def get_cond(A):
    ATA = A.H @ A # A^T A as a LinearOperator
    vals_max = eigsh(ATA, k=1, which='LM', return_eigenvectors=False)
    vals_min = eigsh(ATA, k=1, which='SM', return_eigenvectors=False)
    return np.sqrt(vals_max[0] / vals_min[0])


class BaseStructuredOperator(ABC):
    def __init__(self, dim_param, kappa, geometry_type='sphere_surface', precision='double', n_sources=1):
        try:
            print(dim_param)
            if dim_param < 1:
                self.grid = get_geometry(geometry_type, dim_param)
                path = Path(os.getcwd()) / "results"
                os.makedirs(path, exist_ok=True)
                bempp_cl.api.export('results/current_grid.msh', grid=self.grid)
                self.points = get_barycenters(self.grid)
            else:
                self.grid = None
                self.points = get_geometry(geometry_type, dim_param)
            self.precision = precision
            self.n_points = len(self.points)
            self.n_sources = n_sources
            self.kappa = kappa
            self.form = 'weak'
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
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
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
    
    def get_rhs(self, n_sources=1):
        return right_hand_side(self, None, n_sources)


class BemppClLaplaceSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            print("Number of dofs:", self.n_points)
            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = self.domain
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)

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
    
    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class KiFMMLaplaceOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            print("Precision:", self.rhs_data_type)
            self.operator_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperator'
            points = self.points.ravel()
            print("Number of dofs:", self.n_points)
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
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
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
            print("Number of dofs:", self.n_points)
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
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        self.operator_type = 'real'
        self.operator_type = 'BemppRsLaplaceOperator'
        print("Number of dofs:", self.n_points)
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
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            print("Number of dofs:", self.n_points)
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
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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
    
    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerCP(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCP'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            single_layer = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            @bempp_cl.api.real_callable
            def constant(x, n, domain_index, result):
                result[0] = 1.0

            rank_1_fun = bempp_cl.api.GridFunction(self.domain, fun=constant).projections()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            hypersingular = bempp_cl.api.operators.boundary.laplace.hypersingular(self.domain, self.range, self.dual_to_range).weak_form()
            h_shape = hypersingular.shape

            def mv(v):
                return hypersingular*v + rank_1_fun * (rank_1_fun @ v)
            
            def rmv(v):
                return hypersingular.T*v + rank_1_fun * (rank_1_fun @ v)
            prec = LinearOperator(h_shape, matvec=mv, rmatvec=rmv)
            self.mat = g_inv*prec*g_inv*single_layer
            self.mat_T = single_layer.T*g_inv*prec.T*g_inv
            self.rhs_data_type = self.mat.dtype
            self.form = 'strong'
            rhs = self.get_rhs(n_sources=self.n_sources)
            self.rhs = [self.mat_T*(g_inv*prec*g_inv*r) for r in rhs]
            self.n_points = self.rhs[0].shape[0]
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCP:", e)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerMM(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerMM'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.range
            sl = bempp_cl.api.operators.boundary.laplace.single_layer(self.domain, self.range, self.dual_to_range, assembler = "fmm")
            self.mat = sl.strong_form()
            self.form = 'strong'
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayerCP(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerCP'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = self.domain
            hypersingular = bempp_cl.api.operators.boundary.laplace.hypersingular(self.domain, self.range, self.dual_to_range).weak_form()
            single_layer = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            self.mat = g_inv*hypersingular*g_inv*single_layer 
            self.mat_T = single_layer.T*g_inv*hypersingular.T*g_inv
            self.form = 'strong'
            self.rhs_data_type = self.mat.dtype
            self.rhs = [self.mat_T*(g_inv*(hypersingular*r)) for r in self.get_rhs(n_sources=self.n_sources)]
            self.n_points = self.rhs[0].shape[0]
        except Exception as e:
            print("Error initializing BemppClHelmholtzSingleLayerCP:", e)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerCPID(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            print("Number of dofs:", self.n_points)
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
            adjoint_double_layer_T = adjoint_double_layer.T#bempp_cl.api.operators.boundary.laplace.double_layer(
                #self.domain, self.range, self.dual_to_range).weak_form()
            self.mat = g_inv*(0.25*identity + adjoint_double_layer*g_inv*adjoint_double_layer)
            self.mat_T = (0.25*identity + adjoint_double_layer_T*g_inv*adjoint_double_layer_T)*g_inv
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class KiFMMLaplaceOperatorV(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperatorV'
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, None, n_sources)


class BemppClLaplaceSingleLayerModifiedP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerModifiedP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range, assembler = "fmm").weak_form() + 0.5*identity.weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClLaplaceSingleLayerCPIDP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerCPIDP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)

            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range).weak_form()
            
            adjoint_double_layer = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range).weak_form()
            g_inv = get_inverse_mass_matrix(self.range, self.dual_to_range)
            adjoint_double_layer_T = adjoint_double_layer.transpose()
            self.mat = 0.25*identity + adjoint_double_layer*g_inv*adjoint_double_layer
            self.mat_T = (0.25*identity + adjoint_double_layer_T*g_inv*adjoint_double_layer_T)*g_inv
            self.n_points = self.mat.shape[1]
            self.rhs_data_type = self.mat.dtype
            rhs = self.get_rhs(n_sources=self.n_sources)
            self.rhs = [self.mat_T*r for r in rhs]
            
        except Exception as e:
            print("Error initializing BemppClLaplaceSingleLayerCPIDP1:", e)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayerCPID(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            print("Number of dofs:", self.n_points)
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
            #adjoint_double_layer_T = bempp_cl.api.operators.boundary.helmholtz.double_layer(
            #    self.domain, self.range, self.dual_to_range, kappa).weak_form()
            self.mat = g_inv*(0.25*identity + adjoint_double_layer*g_inv*adjoint_double_layer)
            self.mat_T = (0.25*identity.T + adjoint_double_layer.T*g_inv.T*adjoint_double_layer.T)*g_inv.T
            self.form = 'strong'
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
            self.rhs = [self.mat_T*r for r in self.rhs]
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
    
    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClMaxwellEfie(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = get_edges_centres(self.grid)   
            self.n_points = len(self.points) 
            print("Approximate number of dofs:", self.n_points)  
            self.operator_type = 'complex'
            self.operator_type = 'BemppClMaxwellEfie'
            self.domain = bempp_cl.api.function_space(self.grid, "RWG", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "SNC", 0)
            self.mat = bempp_cl.api.operators.boundary.maxwell.electric_field(self.domain, self.dual_to_range, self.range, kappa).weak_form()   
            #identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
            #                                                        self.range,
            #                                                        self.dual_to_range).weak_form()
            #magnetic = bempp_cl.api.operators.boundary.maxwell.magnetic_field(self.domain, self.dual_to_range, self.range, kappa).weak_form() 
            #electric = bempp_cl.api.operators.boundary.maxwell.electric_field(self.domain, self.dual_to_range, self.range, kappa).weak_form() 
            #self.mat = magnetic - 0.5*identity              
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
            self.n_points = self.mat.shape[0]
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClHelmholtzSingleLayerP1(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.points = self.grid.vertices.T
            self.n_points = self.points.shape[0]
            print("Number of dofs:", self.n_points)
            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayerP1'
            self.domain = bempp_cl.api.function_space(self.grid, "P", 1)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


class BemppClCombinedHelmholtz(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision, n_sources=1):
        super().__init__(dim_param, kappa, geometry_type, precision, n_sources)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"
            self.operator_type = 'complex'
            self.operator_type = 'BemppClCombinedHelmholtz'
            print("Number of dofs:", self.n_points)
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.range = self.domain
            identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.domain, self.range, self.dual_to_range
            ).weak_form()
            adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.domain, self.range, self.dual_to_range, kappa
            ).weak_form()
            slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa
            ).weak_form()
            self.mat = 0.5 * identity + adlp - 1j * kappa * slp#, assembler = "fmm").weak_form()
            '''identity = bempp_cl.api.operators.boundary.sparse.identity(
                self.dual_to_range, self.dual_to_range, self.domain
            ).weak_form()
            dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(
                self.dual_to_range, self.dual_to_range, self.domain, kappa
            ).weak_form()
            slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.dual_to_range, self.dual_to_range, self.domain, kappa
            ).weak_form()'''
            self.mat_T = 0.5 * identity.T + adlp.T - 1j * kappa * slp.T#(0.5 * identity + dlp - 1j * kappa * slp)
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs(n_sources=self.n_sources)
            self.rhs = [self.mat_T*r for r in self.rhs]
            self.n_points = self.mat.shape[1]
        except Exception as e:
            print("Error initializing BemppClCombinedHelmholtz:", e)
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

    def get_rhs(self, n_sources=1):
        return right_hand_side(self, 'Dirichlet', n_sources)


