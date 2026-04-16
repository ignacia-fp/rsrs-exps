from rsrs_config import RSRSBenchmarkConfig
import os

for var in [
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "MKL_DYNAMIC",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
]:
    os.environ.pop(var, None)


''''config = RSRSBenchmarkConfig(operator_type=19, 
        dim_arg_type=1,  
        precision = 0, 
        initial_num_samples = 7000, 
        id_tols = [20, 40, 60], 
        geometry=6, 
        min_level=1, 
        h=0.055, 
        depth = 2, 
        kappa = 9.0, 
        factors_cn = False, 
        dense_errors =False, 
        max_tree_depth=16, 
        rrqr=0, 
        f=1.0, 
        n_sources=5, 
        solve = True, 
        symmetric=0, 
        save_samples=False)'''

'''config = RSRSBenchmarkConfig(operator_type=19,
        dim_arg_type=1,
        precision = 0,
        initial_num_samples = 1000,
        id_tols = [5, 10, 15, 20],
        geometry=6,
        min_level=1,
        h=0.046,
        depth = 2,
        kappa = 10.0,
        factors_cn = False,
        dense_errors =False,
        max_tree_depth=16,
        rrqr=0,
        f=1.0,
        n_sources=5,
        solve = True,
        symmetry=0,
        save_samples=False)'''

config = RSRSBenchmarkConfig(operator_type=19, 
        dim_arg_type=1,  
        precision = 1, 
        initial_num_samples = 0, 
        id_tols = [20], 
        geometry=8, 
        min_level=1, 
        h=0.01,#016, 
        depth = 2, 
        kappa = 3.0, 
        factors_cn = False, 
        dense_errors =False, 
        max_tree_depth=16, 
        rrqr=0, 
        f=1.0, 
        n_sources=1, 
        solve = True, 
        symmetry=0, 
        save_samples=False,
        load_samples=False,
        assembler=0)


#config.sample_with_python(10000)

#config.assembler_index = 1
config.generate_bash_script("run_test.sh")
