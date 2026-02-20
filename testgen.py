from rsrs_config import RSRSBenchmarkConfig
import numpy as np
print(RSRSBenchmarkConfig.__init__.__doc__)

#print(RSRSBenchmarkConfig.__init__.__doc__)

config = RSRSBenchmarkConfig(operator_type=16, dim_arg_type=0, id_tols = [80], geometry=7, min_level=1, h=0.1, depth = 2, kappa = 2.0, factors_cn = False, dense_errors =False, max_tree_depth=16, rrqr=0, f=1.0, n_sources=1, solve = True, symmetric=False, save_samples=False)

#config = RSRSBenchmarkConfig(operator_type=16, dim_arg_type=1, id_tols = [100, 120], geometry=0, min_level=1, h=0.02, depth = 2, kappa = 0.1, factors_cn = False, dense_errors =False, max_tree_depth=4, rrqr=0, f=1.0, n_sources=1, solve = True, symmetric=False, save_samples=False)
config.generate_bash_script("run_test.sh")