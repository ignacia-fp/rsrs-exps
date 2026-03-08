from rsrs_config import RSRSBenchmarkConfig
import numpy as np
print(RSRSBenchmarkConfig.__init__.__doc__)

#print(RSRSBenchmarkConfig.__init__.__doc__)

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
        symmetric=False, 
        save_samples=False)'''

'''config = RSRSBenchmarkConfig(operator_type=16, 
        dim_arg_type=1,  
        precision = 1, 
        initial_num_samples = 7000, 
        id_tols = [120], 
        geometry=1, 
        min_level=1, 
        h=0.025,#016, 
        depth = 2, 
        kappa = 2.0, 
        factors_cn = False, 
        dense_errors =False, 
        max_tree_depth=16, 
        rrqr=0, 
        f=1.0, 
        n_sources=1, 
        solve = True, 
        symmetric=False, 
        save_samples=False)'''

config = RSRSBenchmarkConfig(operator_type=19,
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
        symmetric=False,
        save_samples=False)

#config = RSRSBenchmarkConfig(operator_type=16, dim_arg_type=1, id_tols = [100, 120], geometry=0, min_level=1, h=0.02, depth = 2, kappa = 0.1, factors_cn = False, dense_errors =False, max_tree_depth=4, rrqr=0, f=1.0, n_sources=1, solve = True, symmetric=False, save_samples=False)
config.generate_bash_script("run_test.sh")
