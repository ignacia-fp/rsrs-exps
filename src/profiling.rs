use crate::io::outputs::{write_vec_to_new_file, write_multi_vec_to_new_file, save_stats};
use crate::io::geometries::{cube_surface, sphere_surface};
use bempp_rsrs::rsrs::rsrs_factors::{RsrsFactors, RsrsFactorsOps};
use bempp_rsrs::rsrs::rsrs_cycle::{Rsrs, RsrsData};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use bempp_rsrs::rsrs::box_skeletonisation::Tols;
use bempp_rsrs::rsrs::rsrs_cycle::RsrsOptions;
use bempp_octree::Octree;
use rlst::prelude::*;
use std::path::Path;
use num::NumCast;
use std::fs;



fn get_box_errors<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(kernel_mat: &mut DynamicArray<Item, 2>, rsrs_factors: &RsrsFactors<Item>, tol: <Item as RlstScalar>::Real, path_str: &str)->(<Item as RlstScalar>::Real, <Item as RlstScalar>::Real, <Item as RlstScalar>::Real)
where <Item as RlstScalar>::Real: for<'a> std::iter::Sum<&'a <Item as RlstScalar>::Real>
{
    type Real<Item> = <Item as RlstScalar>::Real;
    let npoints = kernel_mat.shape()[0];
    let (rel_errs, abs_errs) = rsrs_factors.get_boxes_errors(kernel_mat, true);
    let diag_ae = rsrs_factors.get_diag_errors(kernel_mat);

    let string_tol = format!("{:e}", tol);
    let mut rel_errors_path = path_str.to_string();
    rel_errors_path.push_str("/relative_errors_");
    rel_errors_path.push_str(&string_tol);
    rel_errors_path.push('/');

    fs::create_dir_all(Path::new(&rel_errors_path)).unwrap();

    let mut abs_errors_path =  path_str.to_string();
    abs_errors_path.push_str("/absolute_errors_");
    abs_errors_path.push_str(&string_tol);
    abs_errors_path.push('/');

    fs::create_dir_all(Path::new(&abs_errors_path)).unwrap();

    write_multi_vec_to_new_file(&rel_errors_path, &rel_errs);
    write_multi_vec_to_new_file(&abs_errors_path, &abs_errs);

    let diag_ae_r;
    let diag_ae_s;
    
    if diag_ae.len() >1{
        diag_ae_r = &diag_ae[0..diag_ae.len()-2];
        diag_ae_s = diag_ae[diag_ae.len()-1];
    }
    else{
        diag_ae_r = &[];
        diag_ae_s = diag_ae[0];
    }

    let diag_ae_r_sum = diag_ae_r.iter().sum::<<Item as rlst::RlstScalar>::Real>();
    let len: Real<Item> = NumCast::from(diag_ae_r.len()).unwrap();
    let diag_ae_r_mean = diag_ae_r_sum/len;

    let mut ident = rlst_dynamic_array2!(Item, [npoints, npoints]);
    ident.set_identity();

    let zero = ident - kernel_mat.view();

    (zero.view().norm_1(), diag_ae_r_mean, diag_ae_s)

}
//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.
pub trait TestFramework: RlstScalar {
    fn test_rsrs_geometry(geometry: &str, kernel: &str, geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> , kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real)-> DynamicArray<Self, 2>, npoints: usize, kappa: <Self as RlstScalar>::Real, id_tols: &[<Self as RlstScalar>::Real], comm: &SimpleCommunicator);
    fn run_test(geometry: &str, kernel: &str, kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>, npoints: &[usize], kappa: <Self as RlstScalar>::Real);
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
                    let mut app_inv = Vec::new();
                    let mut diag_errs = Vec::new();
                    let mut skel_errs = Vec::new();
                    let mut tot_num_samples = Vec::new();
                    let mut path_str = "results/".to_string();
                    let mut geometry_and_points = geometry.to_string();
                    geometry_and_points.push('_');
                    geometry_and_points.push_str(kernel);
                    geometry_and_points.push('_');
                    geometry_and_points.push_str(&npoints.to_string());
                    path_str.push_str(&geometry_and_points);
                    for &id_tol in id_tols.iter(){
                        println!("Test: {} points, tol:{}", npoints, id_tol);
                        let tols : Tols<$scalar> = Tols{id: id_tol, null: num::Zero::zero(), lstq: num::Zero::zero()};
                        let mut kernel_mat: DynamicArray<$scalar, 2> = kernel_fn(&points, kappa);
                        let mut rsrs_algo: RsrsData<$scalar> = <RsrsData<$scalar> as Rsrs>::new(&kernel_mat, tols, &tree);
                        let options = RsrsOptions{ hermitian: false, silent: true };
                        let rsrs_factors = rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat, options);
                        save_stats(&rsrs_algo, id_tol, &path_str);
                        let (norm_app_inv, diag_ae_mean, skel_ae) = get_box_errors(&mut kernel_mat, &rsrs_factors,  id_tol, &path_str);
                        app_inv.push(norm_app_inv);
                        if !diag_ae_mean.is_nan(){
                            diag_errs.push(diag_ae_mean);
                        }
                        skel_errs.push(skel_ae);
                        tot_num_samples.push(rsrs_algo.y_data.num_samples as f64);
                    }

                    let mut app_id_path = path_str.clone();
                    let mut diag_errs_path = path_str.clone();
                    let mut skel_errs_path = path_str.clone();
                    let mut num_samples_path = path_str.clone();

                    app_id_path.push_str("/app_id.json");
                    diag_errs_path.push_str("/diag_errs.json");
                    skel_errs_path.push_str("/skel_errs.json");
                    num_samples_path.push_str("/num_samples.json");

                    let _= write_vec_to_new_file(&app_id_path, &app_inv);
                    let _= write_vec_to_new_file(&diag_errs_path, &diag_errs);
                    let _= write_vec_to_new_file(&skel_errs_path, &skel_errs);
                    let _= write_vec_to_new_file(&num_samples_path, &tot_num_samples);

                }
            
                fn run_test(geometry: &str, kernel: &str, kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>, npoints: &[usize], kappa:<Self as RlstScalar>::Real){
                    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                    let comm: SimpleCommunicator = universe.world();
                    for &n in npoints{
                        let id_tols = [1e-2, 1e-4, 1e-6, 1e-8];
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

