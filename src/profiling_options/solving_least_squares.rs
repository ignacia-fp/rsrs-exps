use super::add_samples_test::add_samples_parallel;
use crate::io::{
    geometries::{cube_surface, randomly_distributed, sphere_surface},
    low_rank_matrices::KernelMatrix,
};
use bempp_rsrs::rsrs::sketch::SketchData;
use mpi::topology::SimpleCommunicator;
use rlst::{dense::linalg::lu::MatrixLu, prelude::*};
use std::time::Instant;

fn solve_svd<
    Item: RlstScalar + MatrixPseudoInverse,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    test_mat: &Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2> {
    let shape = test_mat.shape();
    let mut test_mat_copy = empty_array();
    test_mat_copy.fill_from_resize(test_mat.r());
    let mut pinv = rlst_dynamic_array2!(Item, [shape[1], shape[0]]); // Avoid extra allocation
    test_mat_copy
        .r_mut()
        .into_pseudo_inverse_alloc(pinv.r_mut(), tol_lstq)
        .unwrap();
    let mut sol: DynamicArray<Item, 2> = empty_array();
    sol.r_mut()
        .simple_mult_into_resize(sketch_mat.r(), pinv.r());

    sol
}

fn solve_lu<
    Item: RlstScalar + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    test_mat: &Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let test_shape = test_mat.shape();
    let sketch_shape = sketch_mat.shape();
    if test_shape[0] < test_shape[1] {
        let mut normal = rlst_dynamic_array2!(Item, [test_shape[0], test_shape[0]]);
        let mut id: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, normal.shape());
        id.set_identity();
        id.scale_inplace(Item::from_real(tol_lstq));
        normal.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::Trans,
            num::One::one(),
            test_mat.r(),
            test_mat.r(),
            num::Zero::zero(),
        );

        normal.sum_into(id); //Regularisation

        let mut rhs = rlst_dynamic_array2!(Item, [test_shape[0], test_shape[0]]);
        rhs.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::Trans,
            num::One::one(),
            test_mat.r(),
            sketch_mat.r(),
            num::Zero::zero(),
        );
        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();
        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &lu,
            TransMode::NoTrans,
            rhs.r_mut(),
        );
        let mut sol = rlst_dynamic_array2!(Item, [sketch_shape[0], test_shape[0]]);

        sol.fill_from(rhs.r().transpose());

        sol
    } else {
        let mut test_mat_trans = empty_array();
        test_mat_trans.fill_from_resize(test_mat.r().transpose());
        let mut normal = rlst_dynamic_array2!(Item, [test_shape[1], test_shape[1]]);
        let mut id: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, normal.shape());
        id.set_identity();
        id.scale_inplace(Item::from_real(tol_lstq));

        normal
            .r_mut()
            .simple_mult_into(test_mat_trans.r(), test_mat.r());
        normal.sum_into(id); //Regularisation

        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();
        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &lu,
            TransMode::NoTrans,
            test_mat_trans.r_mut(),
        );

        let mut sol = rlst_dynamic_array2!(Item, [sketch_shape[0], test_shape[0]]);
        sol.r_mut()
            .simple_mult_into(sketch_mat.r(), test_mat_trans.r());

        sol
    }
}

pub fn compare_lu_and_svd(geometry: &str, kernel: &str, npoints: &[usize], tol_lstq: f64) {
    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();

    let kernel_fn: fn(&[bempp_octree::Point], f64) -> DynamicArray<f64, 2> =
        if kernel == "standard_real" {
            KernelMatrix::get_exp_real_kernel_matrix
        } else {
            KernelMatrix::get_laplace_matrix
        };

    let geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> =
        if geometry == "cube" {
            cube_surface
        } else if geometry == "sphere" {
            sphere_surface
        } else {
            randomly_distributed
        };

    let num_samples = [50, 100, 200, 500, 800, 1000, 2000];
    let mut results = rlst_dynamic_array2!(f32, [num_samples.len(), npoints.len()]);
    let mut view = results.r_mut();
    for (i, &n) in npoints.into_iter().enumerate() {
        for (j, n_samples) in num_samples.into_iter().enumerate() {
            println!("\nNum points and samples: {}, {}", n, n_samples);
            let points: Vec<bempp_octree::Point> = geometry_fn(n, &comm);
            let arr: DynamicArray<f64, 2> = kernel_fn(&points, num::Zero::zero());
            let test: DynamicArray<f64, 2> = empty_array();
            let sketch: DynamicArray<f64, 2> = empty_array();
            let mut sketch_data: SketchData<f64> = SketchData {
                sketch,
                test,
                dim: arr.shape()[0],
                num_samples: 0,
                trans: false,
            };

            add_samples_parallel::<f64>(&mut sketch_data, n_samples, &arr, true, 0);

            let start: Instant = Instant::now();
            let sol_svd = solve_svd(&sketch_data.sketch, &sketch_data.test, tol_lstq);
            let time_svd = start.elapsed();
            println!("SVD time: {:?}", time_svd);

            let start: Instant = Instant::now();
            let sol_lu = solve_lu(&sketch_data.sketch, &sketch_data.test, tol_lstq);
            let time_lu = start.elapsed();
            println!("LU time: {:?}", time_lu);

            let mut app_sketch_svd = empty_array();
            app_sketch_svd
                .r_mut()
                .simple_mult_into_resize(sol_svd.r(), sketch_data.test.r());

            let error_svd = (app_sketch_svd.r() - sketch_data.sketch.r()).norm_1()
                / (sketch_data.sketch.r()).norm_1();
            println!("Error SVD: {}", error_svd);

            let mut app_sketch_lu = empty_array();
            app_sketch_lu
                .r_mut()
                .simple_mult_into_resize(sol_lu.r(), sketch_data.test.r());
            let error_lu = (app_sketch_lu.r() - sketch_data.sketch.r()).norm_1()
                / (sketch_data.sketch.r()).norm_1();
            println!("Error LU: {}", error_lu);

            let diff: f32 = time_lu.as_micros() as f32 / time_svd.as_micros() as f32;
            println!("Difference: {:?}", diff);
            view[[j, i]] = diff;
        }
    }
}
