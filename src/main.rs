use rlst::c64;
use rlst::prelude::*;
use rsrs_exps::io::python_kernel::KernelAttr;
use rsrs_exps::{
    io::{
        low_rank_matrices::KernelMatrix,
        python_kernel::{Kernel, LocalFrom},
    }, //, plot_results::get_time_piecharts},
    // profiling_options::add_samples_test::SampleTestFramework,
    test_prep_ops::{TestFramework, TestOptions},
};

/*fn run(
    geometry: &str,
    kernel: &str,
    npoints: &[usize],
    mut kappa: f64,
    version: f64,
    id_tols: &[f64],
    options: TestOptions,
) {
    if kernel == "standard_real" {
        kappa = 0.0;
        <f64 as TestFramework>::select_and_run_test(
            geometry,
            kernel,
            KernelMatrix::get_exp_real_kernel_matrix,
            &npoints,
            kappa,
            version,
            id_tols,
            &options,
        );
    } else if kernel == "standard_complex" {
        <c64 as TestFramework>::select_and_run_test(
            geometry,
            kernel,
            KernelMatrix::get_exp_complex_kernel_matrix,
            &npoints,
            kappa,
            version,
            id_tols,
            &options,
        );
    } else if kernel == "laplace" {
        kappa = 0.0;
        <f64 as TestFramework>::select_and_run_test(
            geometry,
            kernel,
            KernelMatrix::get_laplace_matrix,
            &npoints,
            kappa,
            version,
            id_tols,
            &options,
        );
    } else {
        <c64 as TestFramework>::select_and_run_test(
            geometry,
            kernel,
            KernelMatrix::get_helmholtz_matrix,
            &npoints,
            kappa,
            version,
            id_tols,
            &options,
        );
    }
}*/

fn run_ops(
    geometry: &str,
    kernel: &str,
    npoints: &[usize],
    mut kappa: f64,
    version: f64,
    id_tols: &[f64],
    options: TestOptions,
) {
    kappa = 0.0;
    <f64 as TestFramework>::select_and_run_test(
        geometry,
        kernel,
        KernelMatrix::get_exp_real_kernel_matrix,
        &npoints,
        kappa,
        version,
        id_tols,
        &options,
    );

}

fn main() {
    let geometry = "sphere";
    let kernel = "laplace";
    let npoints = [5000]; //[20000, 50000, 70000, 90000];//, 100000, 150000, 170000];//, 180000];
    let id_tols = [1e-6];//[1e-2, 1e-4, 1e-6]; //1e-4, 1e-6];
    let pi = std::f64::consts::PI;
    let kappa = 0.0;
    let version = 0.0;

    run_ops(
        geometry,
        kernel,
        &npoints,
        kappa,
        version,
        &id_tols,
        TestOptions {
            plot: false,
            test_type: "all".to_owned(),
        },
    );

    //get_time_piecharts(geometry, kernel, &npoints, kappa, version, &id_tols);

    //<f64 as SampleTestFramework>::test::<f64>(geometry, kernel, &npoints);
}
