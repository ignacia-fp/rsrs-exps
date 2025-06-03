use bempp_rsrs::rsrs::rsrs_cycle::RankPicking;
use bempp_rsrs::rsrs::rsrs_cycle::RsrsOptions;
use bempp_rsrs::rsrs::rsrs_factors::PivotMethod;
use bempp_rsrs::utils::least_squares_and_null::{BlockExtractionMethod, NullMethod};
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
            BlockExtractionMethod::LuLstSq,
            BlockExtractionMethod::LuLstSq,
            PivotMethod::Lu,
            PivotMethod::Lu,
            1e-10,
            1e-2,
            1e-10,
            1e-10,
            4,
            true,
            RankPicking::Min,
        );
        let mut test_framework = TestFramework::<c64>::new(
            kernel_type,
            geometry_type,
            &dim_args,
            kappa,
            options,
            &id_tols,
            Results::All,
            true,
            1e-5,
            true,
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
            BlockExtractionMethod::LuLstSq,
            BlockExtractionMethod::LuLstSq,
            PivotMethod::Lu,
            PivotMethod::Lu,
            1e-10,
            1e-2,
            1e-10,
            1e-10,
            4,
            true,
            RankPicking::Min,
        );
        let mut test_framework = TestFramework::<f64>::new(
            kernel_type,
            geometry_type,
            &dim_args,
            kappa,
            options,
            &id_tols,
            Results::All,
            true,
            1e-5,
            false,
            false,
        );
        test_framework.run_tests();
    }
}

fn main() {
    let id_tols = [1e-2, 1e-4, 1e-6]; //[1e-2, 1e-4, 1e-6];
    let pi = std::f64::consts::PI;
    let kappa = 8.0 * pi;
    let kernel_type = KernelType::KiFmmLaplace;
    let h = 2.0 * pi / (8.0 * kappa);
    println!("Meshwidth: {}", h);
    let dim_args = [DimArg::MeshWidth(h)];
    set_and_run_tests(
        GeometryType::SphereSurface,
        &kernel_type,
        &dim_args,
        kappa,
        &id_tols,
    );
}
