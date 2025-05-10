use bempp_rsrs::rsrs::rsrs_factors::{FactorOptions, RsrsFactors, RsrsFactorsOps, RsrsSide};
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
type Real<T> = <T as rlst::RlstScalar>::Real;

pub fn app_inv_error<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    target_arr: &DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
    side: RsrsSide,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let dim = target_arr.shape()[1];
    let mut sample_mat_1 = empty_array();
    let mut sample_mat_2 = empty_array();
    let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let factor_options = FactorOptions {
        inv: true,
        trans: false,
    };

    let view_shape;
    let view_offset = match side {
        RsrsSide::Left => |ind| [0, ind],
        RsrsSide::Right => |ind| [ind, 0],
        RsrsSide::Squeeze => |_ind| [0, 0],
    };

    match side {
        RsrsSide::Left => {
            sample_mat_1.resize_in_place([dim, sample_size]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(target_arr.r(), sample_mat_1.r());
            view_shape = [dim, 1];
        }
        RsrsSide::Right => {
            sample_mat_1.resize_in_place([sample_size, dim]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(sample_mat_1.r(), target_arr.r());
            view_shape = [1, dim];
        }
        RsrsSide::Squeeze => {
            view_shape = [0, 0];
        }
    }

    rsrs_factors.mul(&mut sample_mat_2, side, &factor_options);

    let mut res = empty_array();
    res.fill_from_resize(sample_mat_2.r() - sample_mat_1.r());

    let max_err = (0..sample_size)
        .into_iter()
        .map(|sample_ind| {
            let binding = res.r().into_subview(view_offset(sample_ind), view_shape);
            let res_view = binding.view_flat();
            let binding = sample_mat_1
                .r()
                .into_subview(view_offset(sample_ind), view_shape);
            let sample_vec = binding.view_flat();
            res_view.norm_2() / sample_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err.unwrap()
}

pub fn app_error<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    target_arr: &DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
    side: RsrsSide,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let dim = target_arr.shape()[1];

    let mut sample_mat_1 = empty_array();
    let mut sample_mat_2 = empty_array();
    let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let factor_options = FactorOptions {
        inv: false,
        trans: false,
    };

    let view_shape;
    let view_offset = match side {
        RsrsSide::Left => |ind| [0, ind],
        RsrsSide::Right => |ind| [ind, 0],
        RsrsSide::Squeeze => |_ind| [0, 0],
    };

    match side {
        RsrsSide::Left => {
            sample_mat_1.resize_in_place([dim, sample_size]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(target_arr.r(), sample_mat_1.r());
            view_shape = [dim, 1];
        }
        RsrsSide::Right => {
            sample_mat_1.resize_in_place([sample_size, dim]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(sample_mat_1.r(), target_arr.r());
            view_shape = [1, dim];
        }
        RsrsSide::Squeeze => {
            view_shape = [0, 0];
        }
    }

    rsrs_factors.mul(&mut sample_mat_1, side, &factor_options);

    let mut res = empty_array();
    res.fill_from_resize(sample_mat_2.r() - sample_mat_1.r());

    let max_err = (0..sample_size)
        .into_iter()
        .map(|sample_ind| {
            let binding = res.r().into_subview(view_offset(sample_ind), view_shape);
            let res_r = binding.view_flat();
            let binding = sample_mat_2
                .r()
                .into_subview(view_offset(sample_ind), view_shape);
            let sample_vec = binding.view_flat();
            res_r.norm_2() / sample_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err.unwrap()
}

pub fn rsrs_error_estimator<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    target_arr: &DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
) -> (Real<Item>, Real<Item>, Real<Item>, Real<Item>, Real<Item>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let app_inv_err_left = app_inv_error(target_arr, rsrs_factors, sample_size, RsrsSide::Left);
    let app_inv_err_right = app_inv_error(target_arr, rsrs_factors, sample_size, RsrsSide::Right);
    let app_err_left = app_error(target_arr, rsrs_factors, sample_size, RsrsSide::Left);
    let app_err_right = app_error(target_arr, rsrs_factors, sample_size, RsrsSide::Right);
    let cond = num::One::one(); //condition_number_estimator(target_arr, 100);

    //println!("Estimated condition number: {:?}", cond);

    (
        app_inv_err_left,
        app_inv_err_right,
        app_err_left,
        app_err_right,
        cond,
    )
}

pub fn power_method<Item: RlstScalar + RandScalar>(
    arr: &DynamicArray<Item, 2>,
    _max_iters: usize,
    tol: Item::Real,
) -> Item
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let n = arr.shape()[0];
    let mut x = rlst_dynamic_array2!(Item, [n, 1]);
    let mut y = rlst_dynamic_array2!(Item, [n, 1]);

    let mut normal = empty_array();
    normal.r_mut().mult_into_resize(
        TransMode::ConjTrans,
        TransMode::NoTrans,
        num::One::one(),
        arr.r(),
        arr.r(),
        num::Zero::zero(),
    );

    let mut rng = rand::thread_rng();

    x.fill_from_standard_normal(&mut rng);
    let norm = x.view_flat().norm_2();
    x.r_mut()
        .scale_inplace(Item::from_f64(1.0).unwrap() / Item::from_real(norm));

    let mut lambda_old: Item = num::Zero::zero();
    let mut lambda: Item = num::One::one();

    while (lambda - lambda_old).abs() > tol {
        lambda_old = lambda;
        y.r_mut().simple_mult_into(normal.r(), x.r());
        let norm = y.r().view_flat().norm_2();
        x.r_mut().fill_from(y.r());
        x.r_mut()
            .scale_inplace(Item::from_f64(1.0).unwrap() / Item::from_real(norm));
        let mut aux = empty_array();
        aux.r_mut().simple_mult_into_resize(normal.r(), x.r());
        let mut res = empty_array();
        res.r_mut().mult_into_resize(
            TransMode::ConjTrans,
            TransMode::NoTrans,
            num::One::one(),
            x.r(),
            aux.r(),
            num::Zero::zero(),
        );
        lambda = res[[0, 0]];
    }

    lambda.sqrt()
}

pub fn condition_number_estimator<Item: RlstScalar + RandScalar>(
    arr: &DynamicArray<Item, 2>,
    nsamples: usize,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let n = arr.shape()[0];

    let mut normal = empty_array();
    normal.r_mut().mult_into_resize(
        TransMode::ConjTrans,
        TransMode::NoTrans,
        num::One::one(),
        arr.r(),
        arr.r(),
        num::Zero::zero(),
    );

    let raleigh_coeffs: Vec<_> = (0..nsamples)
        .into_par_iter()
        .map(|_| {
            let mut x = rlst_dynamic_array2!(Item, [n, 1]);
            let mut rng = rand::thread_rng();
            x.fill_from_standard_normal(&mut rng);
            let norm = x.view_flat().norm_2();
            x.r_mut()
                .scale_inplace(Item::from_f64(1.0).unwrap() / Item::from_real(norm));
            let mut aux = empty_array();
            aux.r_mut().simple_mult_into_resize(normal.r(), x.r());
            let mut rayleigh = empty_array();
            rayleigh.r_mut().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                num::One::one(),
                x.r(),
                aux.r(),
                num::Zero::zero(),
            );
            rayleigh[[0, 0]].abs()
        })
        .collect();

    let min = raleigh_coeffs
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .sqrt();
    let max = raleigh_coeffs
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .sqrt();

    max / min
}
