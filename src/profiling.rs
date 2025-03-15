use mpi::{topology::SimpleCommunicator, traits::Communicator};
use bempp_rsrs::rsrs::rsrs_cycle::{Rsrs, RsrsData, RsrsOptions, Termination};
use crate::io::geometries::{cube_surface, sphere_surface};
use bempp_rsrs::rsrs::box_skeletonisation::Tols;
use crate::io::read_and_write::save_stats;
use bempp_octree::Octree;
use rlst::prelude::*;

pub trait TestFramework: RlstScalar {
    fn test_rsrs_geometry(geometry: &str, kernel: &str, geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> , kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real)-> DynamicArray<Self, 2>, npoints: usize, kappa: <Self as RlstScalar>::Real, id_tols: &[<Self as RlstScalar>::Real], comm: &SimpleCommunicator);
    fn run_test(geometry: &str, kernel: &str, kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>, npoints: &[usize], kappa: <Self as RlstScalar>::Real, id_tols: &[<Self as RlstScalar>::Real]);
}


macro_rules! implement_test_framework{
    ($scalar:ty) => {
            impl TestFramework for $scalar {
                fn test_rsrs_geometry(geometry: &str, kernel: &str, geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> , kernel_fn: fn(&[bempp_octree::Point], <$scalar as RlstScalar>::Real)-> DynamicArray<$scalar, 2>, npoints: usize, kappa: <$scalar as RlstScalar>::Real, id_tols: &[<$scalar as RlstScalar>::Real], comm: &SimpleCommunicator)
                {
                    let points: Vec<bempp_octree::Point> = geometry_fn(npoints, &comm);
                    let max_level: usize = 16;
                    let max_leaf_points: usize = 50;
                    let tree: Octree<'_, SimpleCommunicator>  = Octree::new(&points, max_level, max_leaf_points, &comm);
                    let global_number_of_points: usize = tree.global_number_of_points();
                    let global_max_level: usize = tree.global_max_level();
                    if comm.rank() == 0 {
                        println!(
                            "Setup octree with {} points and maximum level {}",
                            global_number_of_points,
                            global_max_level
                        );
                    }
  
                    let mut path_str = "results/".to_string();
                    let mut geometry_and_points = geometry.to_string();
                    let kappa_string = format!("{:.2}", kappa);
                    geometry_and_points.push('_');
                    geometry_and_points.push_str(kernel);
                    geometry_and_points.push('_');
                    geometry_and_points.push_str(&npoints.to_string());
                    geometry_and_points.push('_');
                    geometry_and_points.push_str(&kappa_string);
                    path_str.push_str(&geometry_and_points);

                    for &id_tol in id_tols.iter(){
                        println!("Test: {} points, tol:{}", npoints, id_tol);
                        let tols : Tols<$scalar> = Tols{id: id_tol, null: num::Zero::zero(), lstq: num::Zero::zero()};
                        let mut kernel_mat: DynamicArray<$scalar, 2> = kernel_fn(&points, kappa);
                        let mut rsrs_algo: RsrsData<$scalar> = <RsrsData<$scalar> as Rsrs>::new(&kernel_mat, tols, &tree);
                        let options = RsrsOptions{ hermitian: true, silent: true, split: true, termination: Termination::ReachRoot, oversampling: 8};
                        let rsrs_factors = rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat, options);
                        save_stats(&mut kernel_mat, &rsrs_factors, &rsrs_algo, id_tol, &path_str);
                    }

                }
            
                fn run_test(geometry: &str, kernel: &str, kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>, npoints: &[usize], kappa:<Self as RlstScalar>::Real, id_tols: &[<Self as RlstScalar>::Real]){
                    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                    let comm: SimpleCommunicator = universe.world();
                    for &n in npoints{
                        let mut geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> = sphere_surface;
                        if geometry == "cube"{
                            geometry_fn = cube_surface;
                        }
                        Self::test_rsrs_geometry(geometry, kernel, geometry_fn, kernel_fn, n, kappa, &id_tols, &comm);
                    }
                }
            }
    }
}

implement_test_framework!(f64);
implement_test_framework!(c64);

