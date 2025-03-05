use bempp_rsrs::utils::low_rank_matrices::KernelMatrix;
use rsrs_exps::{io::plot_results::plot_stats, profiling::TestFramework};
use rlst::c64;

fn run_kernel_test(geometry: &str, kernel: &str, npoints: &[usize], kappa: f64, id_tols: &[f64]){
    if kernel == "standard_real"{
        <f64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_exp_real_kernel_matrix, &npoints, kappa, id_tols);
    }
    else if kernel == "standard_complex"{
        <c64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_exp_complex_kernel_matrix, &npoints, kappa, id_tols);
    }
    else if kernel == "laplace"{
        <f64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_laplace_matrix, &npoints, kappa, id_tols);
    }
    else{
        <c64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_helmholtz_matrix, &npoints, kappa, id_tols);
    }

}


fn main() {
    let geometry = "sphere";
    let kernel = "helmholtz";
    let npoints = [500, 1000, 3000, 5000, 10000, 20000];
    let id_tols = [1e-2, 1e-4, 1e-6, 1e-8];
    let pi =std::f64::consts::PI;
    let kappa = pi;
    run_kernel_test(geometry, kernel, &npoints, kappa, &id_tols);
    plot_stats(geometry, kernel, &npoints, kappa, &id_tols);
}
