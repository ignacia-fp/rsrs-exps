use crate::io::geometries::{cube_surface, randomly_distributed, sphere_surface};
use crate::io::plot_results::time_piechart;
use crate::io::python_kernel::{Kernel, KernelAttr, LocalFrom};
use crate::io::read_and_write::{save_error_stats, save_rank_stats, save_time_stats};
use bempp_octree::Octree;
use crate::io::python_kernel::KernelOperator;
use bempp_rsrs::rsrs::box_skeletonisation::Tols;
use bempp_rsrs::rsrs::rsrs_cycle::{RankPicking, Rsrs, RsrsOptions};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rlst::prelude::*;
use crate::io::python_kernel::{KernelType, KernelParams};

pub struct TestOptions {
    pub plot: bool,
    pub test_type: String,
}

pub trait TestFramework: RlstScalar {
    fn run_test(
        geometry: &str,
        kernel: &str,
        geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point>,
        kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>,
        npoints: usize,
        kappa: <Self as RlstScalar>::Real,
        version: <Self as RlstScalar>::Real,
        id_tols: &[<Self as RlstScalar>::Real],
        comm: &SimpleCommunicator,
        options: &TestOptions,
    );

    fn select_and_run_test(
        geometry: &str,
        kernel: &str,
        kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>,
        npoints: &[usize],
        kappa: <Self as RlstScalar>::Real,
        version: <Self as RlstScalar>::Real,
        id_tols: &[<Self as RlstScalar>::Real],
        options: &TestOptions,
    );
}

fn compute_dense_kernel(kernel: &Operator<KernelOperator<'_, f64, Kernel>>) -> DynamicArray<f64, 2>
{
    let dim = kernel.domain().dimension();
    let mut dense_kernel = rlst_dynamic_array2!(f64, [dim, dim]);
    for i in 0..dim
    {
         let mut el_vec = ArrayVectorSpace::zero(kernel.domain());
         el_vec.view_mut()[[i]] = 1.0;
         let res = kernel.apply(el_vec.r_mut());
         dense_kernel.r_mut().slice(1, i).fill_from(res.view());
    }

    return dense_kernel
}

macro_rules! implement_test_framework {
    ($scalar:ty) => {
        impl TestFramework for $scalar {
            fn run_test(
                geometry: &str,
                kernel_name: &str,
                _geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point>,
                _kernel_fn: fn(
                    &[bempp_octree::Point],
                    <$scalar as RlstScalar>::Real,
                ) -> DynamicArray<$scalar, 2>,
                npoints: usize,
                kappa: <$scalar as RlstScalar>::Real,
                version: <$scalar as RlstScalar>::Real,
                id_tols: &[<$scalar as RlstScalar>::Real],
                comm: &SimpleCommunicator,
                test_options: &TestOptions,
            ) {
                let h = 1.0/npoints as f64;
                let kernel_params = KernelParams::new(KernelType::BemLaplace, npoints, 0.0, h);
                let kernel = Kernel::new(kernel_params);
                let operator = Operator::from_local(&kernel);
                
                let points: Vec<bempp_octree::Point> = kernel.get_bempp_points();
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

                let mut path_str = "results/".to_string();
                let mut geometry_and_points = geometry.to_string();
                let kappa_string = format!("{:.2}", kappa);
                let version_string = format!("{:.2}", version);
                geometry_and_points.push('_');
                geometry_and_points.push_str(kernel_name);
                geometry_and_points.push('_');
                geometry_and_points.push_str(&npoints.to_string());
                geometry_and_points.push('_');
                geometry_and_points.push_str(&kappa_string);
                geometry_and_points.push('_');
                geometry_and_points.push_str(&version_string);
                path_str.push_str(&geometry_and_points);

                for &id_tol in id_tols.iter() {
                    println!("Test: {} points, tol:{}", operator.domain().dimension(), id_tol);
                    let tols: Tols<$scalar> = Tols {
                        id: id_tol,
                        null: num::Zero::zero(),
                        lstq: num::Zero::zero(),
                    };

                    
                    let mut rsrs_algo: Rsrs<$scalar> =
                        Rsrs::new(operator.domain().dimension(), tols, &tree, true);
                    let options = RsrsOptions {
                        oversampling_diag_blocks: 16,
                        rank_picking: RankPicking::Mid,
                        oversampling: 8,
                        initial_num_samples: 420,
                    };
                    let mut rsrs_factors = rsrs_algo.run(&operator, &options);
                    let mut dense_kernel = compute_dense_kernel(&operator);

                    if test_options.test_type == "all" {
                        save_time_stats(&rsrs_algo, id_tol, &path_str);
                        save_error_stats(
                            &mut dense_kernel,
                            &mut rsrs_factors,
                            &rsrs_algo,
                            id_tol,
                            &path_str,
                        );
                        save_rank_stats(
                            &mut dense_kernel,
                            &rsrs_factors,
                            &rsrs_algo,
                            id_tol,
                            &path_str,
                        );

                        if test_options.plot {
                            time_piechart(geometry, kernel_name, npoints, kappa, version, id_tol);
                        }
                    } else if test_options.test_type == "rank" {
                        save_error_stats(
                            &mut dense_kernel,
                            &mut rsrs_factors,
                            &rsrs_algo,
                            id_tol,
                            &path_str,
                        );
                        save_rank_stats(
                            &mut dense_kernel,
                            &rsrs_factors,
                            &rsrs_algo,
                            id_tol,
                            &path_str,
                        );
                    } else {
                        save_error_stats(
                            &mut dense_kernel,
                            &mut rsrs_factors,
                            &rsrs_algo,
                            id_tol,
                            &path_str,
                        );
                        save_time_stats(&rsrs_algo, id_tol, &path_str);
                        if test_options.plot {
                            time_piechart(geometry, kernel_name, npoints, kappa, version, id_tol);
                        }
                    }
                }
            }

            fn select_and_run_test(
                geometry: &str,
                kernel: &str,
                kernel_fn: fn(
                    &[bempp_octree::Point],
                    <Self as RlstScalar>::Real,
                ) -> DynamicArray<Self, 2>,
                npoints: &[usize],
                kappa: <Self as RlstScalar>::Real,
                version: <Self as RlstScalar>::Real,
                id_tols: &[<Self as RlstScalar>::Real],
                options: &TestOptions,
            ) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();
                for &n in npoints {
                    let geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> =
                        if geometry == "cube" {
                            cube_surface
                        } else if geometry == "sphere" {
                            sphere_surface
                        } else {
                            randomly_distributed
                        };
                    Self::run_test(
                        geometry,
                        kernel,
                        geometry_fn,
                        kernel_fn,
                        n,
                        kappa,
                        version,
                        &id_tols,
                        &comm,
                        options,
                    );
                }
            }
        }
    };
}

implement_test_framework!(f64);
//implement_test_framework!(c64);
