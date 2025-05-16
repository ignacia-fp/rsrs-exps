use rlst::prelude::*;
use rsrs_exps::io::python_kernel::KernelType;
use rsrs_exps::test_prep_ops::Results;
use rsrs_exps::test_prep_ops::TestFrameworkImpl;
use rsrs_exps::test_prep_ops::{DimArg, TestFramework};

fn set_and_run_tests(
    geometry: String,
    kernel_type: &KernelType,
    dim_args: &[DimArg],
    mut kappa: f64,
    version: f64,
    id_tols: &[f64],
) {
    if matches!(kernel_type, KernelType::BemHelmholtz)
        || matches!(kernel_type, KernelType::Helmholtz)
    {
        let test_framework = TestFramework::<c64>::new(
            geometry.to_string(),
            kernel_type,
            &dim_args,
            kappa,
            version,
            &id_tols,
            Results::All,
            false,
        );
        test_framework.run_tests();
    } else {
        kappa = 0.0;
        let test_framework = TestFramework::<f64>::new(
            geometry.to_string(),
            kernel_type,
            &dim_args,
            kappa,
            version,
            &id_tols,
            Results::All,
            false,
        );
        test_framework.run_tests();
    }

    
}

fn main() {
    //Usual test for numpoints: [20000, 50000, 70000, 90000];
    let geometry = "sphere";
    //let dim_args = [DimArg::NumPoints(5000)];
    let dim_args = [DimArg::MeshWidth(1e-1)]; 
    let id_tols = [1e-2, 1e-4, 1e-6]; //1e-4, 1e-6];
    let pi = std::f64::consts::PI;
    let kappa = pi;
    let version = 0.0;
    let kernel_type = KernelType::BemHelmholtz;

    set_and_run_tests(geometry.to_string(), &kernel_type, &dim_args, kappa, version, &id_tols);
}
