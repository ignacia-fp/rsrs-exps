
import numpy as np
import bempp_cl.api

def right_hand_side(operator, problem_type):
    undefined_rhs = {'BasicStructuredOperator', 'KiFMMLaplaceOperator', 'KiFMMHelmholtzOperator', 'BemppClLaplaceSingleLayerCPID', 'KiFMMLaplaceOperatorV', 'BemppClLaplaceSingleLayerCPIDP1'}
    if operator.operator_type in undefined_rhs:
        print("Warning: problem types for this operator have not been defined, so a random vector is returned.")
        if operator.rhs_data_type == np.complex64 or operator.rhs_data_type == np.complex128:
            real_parts = np.random.rand(operator.n_points)
            imag_parts = np.random.rand(operator.n_points)
            rhs = real_parts + 1j*imag_parts
            rhs = rhs.astype(operator.rhs_data_type)
        else:
            rhs = np.random.rand(operator.n_points).astype(operator.rhs_data_type)
        return rhs
    elif operator.operator_type == 'BemppClLaplaceSingleLayer' or operator.operator_type == 'BemppClLaplaceSingleLayerModified' or operator.operator_type == 'BemppClLaplaceSingleLayerCP' or operator.operator_type =='BemppClLaplaceSingleLayerMM' or operator.operator_type == 'BemppClLaplaceSingleLayerP1' or operator.operator_type == 'BemppClLaplaceSingleLayerModifiedP1':
        if problem_type == 'Dirichlet':
            @bempp_cl.api.real_callable
            def dirichlet_data(x, n, domain_index, result):
                result[0] = 1./(4 * np.pi *((x[0] - .9)**2 + x[1]**2 + x[2]**2)**(0.5))
                
            dirichlet_fun = bempp_cl.api.GridFunction(operator.domain, fun=dirichlet_data)

            identity = bempp_cl.api.operators.boundary.sparse.identity(operator.domain,
                                                                    operator.range,
                                                                    operator.dual_to_range)
            dlp = bempp_cl.api.operators.boundary.laplace.double_layer(operator.domain,
                                                                    operator.range,
                                                                    operator.domain, assembler = "fmm")#, assembler = "fmm")#,
                                                                    #assembler="fmm")

            rhs = (dlp - 0.5 * identity) * dirichlet_fun

            if operator.operator_type == 'BemppClLaplaceSingleLayer' or operator.operator_type == 'BemppClLaplaceSingleLayerModified':
                return rhs.projections()
            else:
                return rhs.coefficients
        
        elif problem_type == 'Neumann':
            raise ValueError("Neumann problem not implemented.")

    elif operator.operator_type == 'BemppClHelmholtzSingleLayer' or operator.operator_type == 'BemppClHelmholtzSingleLayerCP':
        kappa = operator.kappa
        if problem_type == 'Dirichlet':
            @bempp_cl.api.complex_callable
            def dirichlet_data(x, n, domain_index, result):
                result[0] = -1j * kappa * np.exp(1j * kappa * x[0]) * n[0]

            dirichlet_fun = bempp_cl.api.GridFunction(operator.domain, fun=dirichlet_data)

            identity = bempp_cl.api.operators.boundary.sparse.identity(operator.domain,
                                                                    operator.range,
                                                                    operator.dual_to_range)
            dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(operator.domain,
                                                                    operator.range,
                                                                    operator.domain, 
                                                                    kappa, assembler = "fmm")#,
                                                                    #assembler="fmm")

            rhs = (dlp - 0.5 * identity) * dirichlet_fun
            if operator.operator_type == 'BemppClHelmholtzSingleLayer' or operator.operator_type == 'BemppClHelmholtzSingleLayerCP':
                return rhs.projections()
            else:
                return rhs.coefficients
        
        elif problem_type == 'Neumann':
            raise ValueError("Neumann problem not implemented.")


