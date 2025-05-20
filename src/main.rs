use bempp_rsrs::rsrs::rsrs_cycle::RankPicking;
use bempp_rsrs::rsrs::rsrs_cycle::RsrsOptions;
use bempp_rsrs::utils::least_squares_and_null::NullMethod;
use rlst::prelude::*;
use rsrs_exps::io::python_kernel::GeometryType;
use rsrs_exps::io::python_kernel::KernelType;
use rsrs_exps::test_prep::Results;
use rsrs_exps::test_prep::TestFrameworkImpl;
use rsrs_exps::test_prep::{DimArg, TestFramework};

fn set_and_run_tests(
    geometry_type: GeometryType,
    kernel_type: &KernelType,
    dim_args: &[DimArg],
    mut kappa: f64,
    id_tols: &[f64],
) {
    if matches!(kernel_type, KernelType::BemHelmholtz)
        || matches!(kernel_type, KernelType::Helmholtz)
    {
        let options = RsrsOptions::new(
            8,
            16,
            420,
            NullMethod::Projection,
            1e-10,
            1e-2,
            1e-10,
            1e-10,
            1e-10,
            4,
            true,
            RankPicking::Tol,
        );
        let mut test_framework = TestFramework::<c64>::new(
            kernel_type,
            geometry_type,
            &dim_args,
            kappa,
            options,
            &id_tols,
            Results::All,
            false,
        );
        test_framework.run_tests();
    } else {
        kappa = 0.0;
        let options = RsrsOptions::new(
            8,
            16,
            420,
            NullMethod::Projection,
            1e-10,
            1e-2,
            1e-10,
            1e-10,
            1e-10,
            4,
            true,
            RankPicking::Tol,
        );
        let mut test_framework = TestFramework::<f64>::new(
            kernel_type,
            geometry_type,
            &dim_args,
            kappa,
            options,
            &id_tols,
            Results::All,
            false,
        );
        test_framework.run_tests();
    }
}

fn main() {
    let dim_args = [DimArg::MeshWidth(1e-1)];
    let id_tols = [1e-2, 1e-4, 1e-6];
    let pi = std::f64::consts::PI;
    let kappa = pi;
    let kernel_type = KernelType::BemHelmholtz;

    set_and_run_tests(
        GeometryType::SphereSurface,
        &kernel_type,
        &dim_args,
        kappa,
        &id_tols,
    );
}
