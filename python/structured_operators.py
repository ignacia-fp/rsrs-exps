
import numpy as np
import bempp_cl.api
import time
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from .geometry import get_geometry, get_barycenters
from .righ_hand_sides import right_hand_side

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
    def __init__(self, dim_param, kappa, geometry_type='sphere_surface', precision='double'):
        try:
            if dim_param < 1:
                self.grid = get_geometry(geometry_type, dim_param)
                self.points = get_barycenters(self.grid)
            else:
                self.grid = None
                self.points = get_geometry(geometry_type, dim_param)
            self.precision = precision
            self.n_points = len(self.points)
            print("Number of dofs:", self.n_points)
            self.kappa = kappa
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
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
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
    
    def get_rhs(self):
        return right_hand_side(self, None)


class BemppClLaplaceSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
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
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')
    

class BemppClHelmholtzSingleLayer(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'complex'
            self.operator_type = 'BemppClHelmholtzSingleLayer'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            self.mat = bempp_cl.api.operators.boundary.helmholtz.single_layer(
                self.domain, self.range, self.dual_to_range, kappa).weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
            

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
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')


class KiFMMLaplaceOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                self.rhs_data_type = np.float32
            else:
                self.rhs_data_type = np.float64
            
            self.operator_type = 'real'
            self.operator_type = 'KiFMMLaplaceOperator'
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
    
    def get_rhs(self):
        return right_hand_side(self, None)


class KiFMMHelmholtzOperator(BaseStructuredOperator):
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
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
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        self.operator_type = 'real'
        self.operator_type = 'BemppRsLaplaceOperator'
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
    def __init__(self, dim_param, kappa, geometry_type, precision):
        super().__init__(dim_param, kappa, geometry_type, precision)
        try:
            if self.precision == 'single':
                bempp_cl.api.DEFAULT_PRECISION = "single"
            else:
                bempp_cl.api.DEFAULT_PRECISION = "double"

            self.operator_type = 'real'
            self.operator_type = 'BemppClLaplaceSingleLayerModified'
            self.domain = bempp_cl.api.function_space(self.grid, "DP", 0)
            self.dual_to_range = self.domain
            self.range = bempp_cl.api.function_space(self.grid, "P", 1)
            identity = bempp_cl.api.operators.boundary.sparse.identity(self.domain,
                                                                    self.range,
                                                                    self.dual_to_range)
            self.mat = bempp_cl.api.operators.boundary.laplace.single_layer(
                self.domain, self.range, self.dual_to_range).weak_form() + 0.5*identity.weak_form()#, assembler = "fmm").weak_form()
            self.rhs_data_type = self.mat.dtype
            self.rhs = self.get_rhs()
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
    
    def get_rhs(self):
        return right_hand_side(self, 'Dirichlet')