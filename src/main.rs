use bempp_rsrs::utils::low_rank_matrices::KernelMatrix;
use rsrs_exps::profiling::TestFramework;
use rlst::c64;

fn main() {
    let geometry = "sphere";
    let kernel = "helmholtz";
    let npoints = [1000];//[500, 1000, 3000, 5000, 10000, 20000];

    
    if kernel == "standard_real"{
        <f64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_exp_real_kernel_matrix, &npoints, 0.0);
    }
    else if kernel == "standard_complex"{
        <c64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_exp_complex_kernel_matrix, &npoints, 0.0);
    }
    else if kernel == "laplace"{
        <f64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_laplace_matrix, &npoints, 0.0);
    }
    else{
        let pi =std::f64::consts::PI;
        <c64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_helmholtz_matrix, &npoints, pi);
    }
}
