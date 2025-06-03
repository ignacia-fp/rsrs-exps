use crate::io::plot_results::time_piechart;
use crate::io::python_kernel::{
    get_bempp_points, GeometryType, Kernel, KernelImpl, KernelOperator, KernelParams, KernelType,
    LocalFrom,
};
use crate::io::read_and_write::Iterations;
use crate::io::read_and_write::{save_error_stats, save_rank_stats, save_time_stats};
use crate::io::solve::{solve_prec_system, solve_system};
use bempp_octree::Octree;
use bempp_rsrs::rsrs::rsrs_cycle::{Rsrs, RsrsOptions};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rlst::prelude::*;
type Real<T> = <T as rlst::RlstScalar>::Real;
pub enum Results {
    All,
    Rank,
    Time,
}
pub struct TestOptions {
    pub plot: bool,
    pub results: Results,
    pub dense_errors: bool,
    pub solve: bool,
    pub solve_tol: f64,
}

pub struct TestParams<Item: RlstScalar> {
    geometry_type: GeometryType,
    kernel_type: KernelType,
    dim_args: Vec<DimArg>,
    kappa: Real<Item>,
    rsrs_params: RsrsOptions<Item>,
    id_tols: Vec<Real<Item>>,
}

impl<Item: RlstScalar> TestParams<Item> {
    fn new(
        geometry_type: GeometryType,
        kernel_type: &KernelType,
        dim_args: &[DimArg],
        kappa: Real<Item>,
        rsrs_params: RsrsOptions<Item>,
        id_tols: &[Real<Item>],
    ) -> Self {
        Self {
            geometry_type,
            kernel_type: kernel_type.clone(),
            dim_args: dim_args.to_vec(),
            kappa,
            rsrs_params,
            id_tols: id_tols.to_vec(),
        }
    }

    fn get_kernel_name(&self) -> &str {
        let kernel_name = match self.kernel_type {
            KernelType::Laplace => "laplace",
            KernelType::Helmholtz => "helmholtz",
            KernelType::Exp => "exponential",
            KernelType::BemLaplace => "bem_laplace",
            KernelType::BemHelmholtz => "bem_helmholtz",
            KernelType::KiFmmLaplace => "kifmm_laplace",
        };

        kernel_name
    }

    fn get_test_dir(&self, dim_num: usize) -> String {
        let geometry = match self.geometry_type {
            GeometryType::SphereSurface => "sphere_surface",
            GeometryType::CubeSurface => "cube_surface",
            GeometryType::CylinderSurface => "cylinder_surface",
            GeometryType::EllipsoidSurface => "ellipsoid_surface",
            GeometryType::TrefoilKnot => "trefoil_knot",
            GeometryType::Sphere => "sphere",
            GeometryType::Cube => "cube",
        }
        .to_string();
        let kernel = self.get_kernel_name();
        let kappa = format!("{:.2}", self.kappa);
        let version = self.rsrs_params.to_identifier();

        let dim_part = match self.dim_args[dim_num] {
            DimArg::NumPoints(n) => format!("n_points_{}", n),
            DimArg::MeshWidth(h) => format!("mesh_width_{:e}", h),
        };

        let filename = format!(
            "{}_{}_{}_{}",
            geometry,
            kernel,
            dim_part,
            kappa, // temporarily omit version
        );

        // Use `format!` again to prepend "results/" and append version
        let path_str = format!("results/{}/{}.json", filename, version);
        path_str
    }
}

pub struct TestFramework<Item: RlstScalar> {
    test_options: TestOptions,
    test_params: TestParams<Item>,
}

#[derive(Debug, Clone)]
pub enum DimArg {
    NumPoints(usize),
    MeshWidth(f64),
}

pub trait TestFrameworkImpl<Item: RlstScalar> {
    type Item: RlstScalar;
    fn new(
        kernel_type: &KernelType,
        geometry_type: GeometryType,
        dim_args: &[DimArg],
        kappa: f64,
        rsrs_options: RsrsOptions<Self::Item>,
        id_tols: &[f64],
        results: Results,
        solve: bool,
        solve_tol: f64,
        plot: bool,
        dense_errors: bool,
    ) -> Self;

    fn run_tests(&mut self);
}

macro_rules! implement_test_framework {
    ($scalar:ty) => {
        impl TestFrameworkImpl<$scalar> for TestFramework<$scalar> {
            type Item = $scalar;
            fn new(
                kernel_type: &KernelType,
                geometry_type: GeometryType,
                dim_args: &[DimArg],
                kappa: f64,
                rsrs_options: RsrsOptions<$scalar>,
                id_tols: &[f64],
                results: Results,
                solve: bool,
                solve_tol: f64,
                plot: bool,
                dense_errors: bool,
            ) -> Self {
                let test_params = TestParams::new(
                    geometry_type,
                    kernel_type,
                    dim_args,
                    kappa,
                    rsrs_options,
                    id_tols,
                );
                let test_options = TestOptions {
                    plot,
                    results,
                    dense_errors,
                    solve,
                    solve_tol,
                };
                Self {
                    test_params,
                    test_options,
                }
            }

            fn run_tests(&mut self) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();

                for (dim_num, dim_arg) in self.test_params.dim_args.iter().enumerate() {
                    let kernel_params = KernelParams::new(
                        self.test_params.kernel_type.clone(),
                        self.test_params.geometry_type.clone(),
                        dim_arg.clone(),
                        self.test_params.kappa,
                    );
                    let kernel: Kernel = <Kernel as KernelImpl<$scalar>>::new(kernel_params);
                    let dim = kernel.n_points;
                    let operator = KernelOperator::from_local(&kernel);
                    let points: Vec<bempp_octree::Point> = get_bempp_points(&kernel).unwrap();

                    let max_level: usize = 6;
                    let max_leaf_points: usize = 50;
                    let tree: Octree<'_, SimpleCommunicator> =
                        Octree::new(&points, max_level, max_leaf_points, &comm);
                    let global_number_of_points: usize = tree.global_number_of_points();
                    let global_max_level: usize = tree.global_max_level();

                    let mut iterations = Iterations {
                        no_prec: None,
                        prec: None,
                    };

                    if self.test_options.solve {
                        iterations.no_prec =
                            Some(solve_system(operator.clone(), self.test_options.solve_tol));
                    }

                    if comm.rank() == 0 {
                        println!(
                            "Setup octree with {} points and maximum level {}",
                            global_number_of_points, global_max_level
                        );
                    }

                    for &id_tol in self.test_params.id_tols.iter() {
                        let mut iterations = iterations.clone();

                        println!("Test: {} points, tol:{}", dim, id_tol);

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(dim, &tree, self.test_params.rsrs_params.clone());

                        let mut rsrs_factors = rsrs_algo.run(&operator);

                        let path_str = self.test_params.get_test_dir(dim_num);

                        if self.test_options.solve {
                            iterations.prec = Some(solve_prec_system(
                                operator.clone(),
                                &mut rsrs_factors,
                                self.test_options.solve_tol,
                            ));
                        }

                        match self.test_options.results {
                            Results::All => {
                                save_error_stats::<$scalar>(
                                    &operator,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    iterations,
                                    id_tol,
                                    &path_str,
                                    self.test_options.dense_errors,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);

                                if self.test_options.plot {
                                    time_piechart(id_tol, &path_str);
                                }
                            }
                            Results::Rank => {
                                save_error_stats::<$scalar>(
                                    &operator,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    iterations,
                                    id_tol,
                                    &path_str,
                                    self.test_options.dense_errors,
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                            }
                            Results::Time => {
                                save_error_stats::<$scalar>(
                                    &operator,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    iterations,
                                    id_tol,
                                    &path_str,
                                    self.test_options.dense_errors,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                if self.test_options.plot {
                                    time_piechart(id_tol, &path_str);
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

implement_test_framework!(f64);
implement_test_framework!(c64);
