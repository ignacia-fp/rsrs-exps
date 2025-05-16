use crate::io::plot_results::time_piechart;
use crate::io::python_kernel::{get_bempp_points, KernelOperator, Kernel, KernelImpl, LocalFrom, KernelParams, KernelType};
use crate::io::read_and_write::{save_error_stats, save_rank_stats, save_time_stats};
use bempp_octree::Octree;
use bempp_rsrs::rsrs::box_skeletonisation::Tols;
use bempp_rsrs::rsrs::rsrs_cycle::{RankPicking, Rsrs, RsrsOptions};
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
}

pub struct TestParams<Item: RlstScalar> {
    geometry: String,
    kernel_type: KernelType,
    dim_args: Vec<DimArg>,
    kappa: Real<Item>,
    version: Real<Item>,
    id_tols: Vec<Real<Item>>,
}

impl<Item: RlstScalar> TestParams<Item> {
    fn new(
        geometry: String,
        kernel_type: &KernelType,
        dim_args: &[DimArg],
        kappa: Real<Item>,
        version: Real<Item>,
        id_tols: &[Real<Item>],
    ) -> Self {
        Self {
            geometry,
            kernel_type: kernel_type.clone(),
            dim_args: dim_args.to_vec(),
            kappa,
            version,
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
        };

        kernel_name
    }

    fn get_test_dir(&self, dim_num: usize) -> String {
        let mut path_str = "results/".to_string();
        let mut geometry_and_points = self.geometry.to_string();
        let kappa_string = format!("{:.2}", self.kappa);
        let version_string = format!("{:.2}", self.version);
        geometry_and_points.push('_');
        geometry_and_points.push_str(self.get_kernel_name());
        geometry_and_points.push('_');

        match self.dim_args[dim_num] {
            DimArg::NumPoints(num_points) => {
                geometry_and_points.push_str("n_points_");
                geometry_and_points.push_str(&num_points.to_string());
            }
            DimArg::MeshWidth(h) => {
                geometry_and_points.push_str("mesh_width_");
                let h_string = format!("{:.2}", h);
                geometry_and_points.push_str(&h_string);
            }
        };

        geometry_and_points.push('_');
        geometry_and_points.push_str(&kappa_string);
        geometry_and_points.push('_');
        geometry_and_points.push_str(&version_string);
        path_str.push_str(&geometry_and_points);
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

fn compute_dense_kernel<Item: RlstScalar>(
    kernel: &Operator<KernelOperator<'_, Item, Kernel>>,
) -> DynamicArray<Item, 2>
where
    Kernel: KernelImpl<Item>,
{
    let dim = kernel.domain().dimension();
    let mut dense_kernel = rlst_dynamic_array2!(Item, [dim, dim]);
    for i in 0..dim {
        let mut el_vec = ArrayVectorSpace::zero(kernel.domain());
        el_vec.view_mut()[[i]] = num::One::one();
        let res = kernel.apply(el_vec.r_mut(), TransMode::NoTrans);
        dense_kernel.r_mut().slice(1, i).fill_from(res.view());
    }

    return dense_kernel;
}

pub trait TestFrameworkImpl {
    type Item: RlstScalar;
    fn new(
        geometry: String,
        kernel_type: &KernelType,
        dim_args: &[DimArg],
        kappa: f64,
        version: f64,
        id_tols: &[f64],
        results: Results,
        plot: bool,
    ) -> Self;

    fn run_tests(&self);
}

macro_rules! implement_test_framework {
    ($scalar:ty) => {
        impl TestFrameworkImpl for TestFramework<$scalar> {
            type Item = $scalar;
            fn new(
                geometry: String,
                kernel_type: &KernelType,
                dim_args: &[DimArg],
                kappa: f64,
                version: f64,
                id_tols: &[f64],
                results: Results,
                plot: bool,
            ) -> Self {
                let test_params =
                    TestParams::new(geometry, kernel_type, dim_args, kappa, version, id_tols);
                let test_options = TestOptions { plot, results };
                Self {
                    test_params,
                    test_options,
                }
            }

            fn run_tests(&self) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();

                for (dim_num, dim_arg) in self.test_params.dim_args.iter().enumerate() {
                    let kernel_params = KernelParams::new(
                        self.test_params.kernel_type.clone(),
                        dim_arg.clone(),
                        0.0,
                    );
                    let kernel: Kernel = <Kernel as KernelImpl<Self::Item>>::new(kernel_params);
                    let operator = Operator::from_local(&kernel);
                    let points: Vec<bempp_octree::Point> = get_bempp_points(&kernel).unwrap();

                    let max_level: usize = 6;
                    let max_leaf_points: usize = 50;
                    let tree: Octree<'_, SimpleCommunicator> =
                        Octree::new(&points, max_level, max_leaf_points, &comm);
                    let global_number_of_points: usize = tree.global_number_of_points();
                    let global_max_level: usize = tree.global_max_level();

                    if comm.rank() == 0 {
                        println!(
                            "Setup octree with {} points and maximum level {}",
                            global_number_of_points, global_max_level
                        );
                    }

                    for &id_tol in self.test_params.id_tols.iter(){
                        println!(
                            "Test: {} points, tol:{}",
                            operator.domain().dimension(),
                            id_tol
                        );
                        let tols: Tols<Self::Item> = Tols {
                            id: id_tol,
                            null: 1e-10,
                            lstq: 1e-10,
                        };

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(operator.domain().dimension(), tols, &tree, true);
                        let options = RsrsOptions {
                            oversampling_diag_blocks: 16,
                            rank_picking: RankPicking::Tol,
                            oversampling: 8,
                            initial_num_samples: 420,
                        };
                        let mut rsrs_factors = rsrs_algo.run(&operator, &options);
                        let mut dense_kernel = compute_dense_kernel(&operator);

                        let path_str = self.test_params.get_test_dir(dim_num);

                        match self.test_options.results {
                            Results::All => {
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_error_stats(
                                    &mut dense_kernel,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    id_tol,
                                    &path_str,
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);

                                if self.test_options.plot {
                                    time_piechart(
                                        &self.test_params.geometry,
                                        self.test_params.get_kernel_name(),
                                        dim_arg,
                                        self.test_params.kappa,
                                        self.test_params.version,
                                        id_tol,
                                    );
                                }
                            }
                            Results::Rank => {
                                save_error_stats(
                                    &mut dense_kernel,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    id_tol,
                                    &path_str,
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                            }
                            Results::Time => {
                                save_error_stats(
                                    &mut dense_kernel,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    id_tol,
                                    &path_str,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                if self.test_options.plot {
                                    time_piechart(
                                        &self.test_params.geometry,
                                        self.test_params.get_kernel_name(),
                                        dim_arg,
                                        self.test_params.kappa,
                                        self.test_params.version,
                                        id_tol,
                                    );
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
