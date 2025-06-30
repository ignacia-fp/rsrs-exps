use std::sync::{Arc, Mutex};

/*use super::structured_operator::{
    StructuredOperator, StructuredOperatorImpl, StructuredOperatorInterface,
};*/
use bempp_rsrs::rsrs::rsrs_factors::Inv;
use bempp_rsrs::rsrs::rsrs_factors::{
    CommutativeFactors, Factor, FactorMulType, FactorOperations, FactorOptions, FactorType,
    IdFactor, LuFactor,
};
use bempp_rsrs::rsrs::rsrs_factors::{RsrsFactors, RsrsSide};
use bempp_rsrs::rsrs::sketch::{SampleType, SamplingSpace};
use bempp_rsrs::utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction};
use num::NumCast;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};

type Real<T> = <T as rlst::RlstScalar>::Real;

fn _gen_sample_frame<Item: RlstScalar + RandScalar, Space: SamplingSpace<F = Item> + Clone>(
    sample_size: usize,
    space: Space,
) -> VectorFrame<<Space as rlst::LinearSpace>::E>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let mut frame = VectorFrame::default();
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();

    for _ in 0..sample_size {
        let mut sample_vec = SamplingSpace::zero(std::rc::Rc::new(space.clone()));
        space.sampling(&mut sample_vec, &mut rng, SampleType::StandardNormal);
        frame.push(sample_vec);
    }
    frame
}

fn mul_op<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    operator: &OpImpl,
    sample_frame: &VectorFrame<<Space as rlst::LinearSpace>::E>,
    trans_mode: TransMode,
) -> VectorFrame<<Space as rlst::LinearSpace>::E> {
    let mut frame = VectorFrame::default();
    for sample_vec in sample_frame.iter() {
        frame.push(operator.apply(sample_vec.r(), trans_mode));
    }

    frame
}

pub fn app_error<
    'a,
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_op: &OpImpl2,
    sample_size: usize,
    side: RsrsSide,
    inv: bool,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let trans_mode = if matches!(side, RsrsSide::Left) {
        TransMode::NoTrans
    } else {
        TransMode::Trans
    };

    let mut sample_frame = VectorFrame::default();
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let space = target_op.range();

    for _ in 0..sample_size {
        let mut sample_vec = SamplingSpace::zero(space.clone());
        space.sampling(&mut sample_vec, &mut rng, SampleType::StandardNormal);
        sample_frame.push(sample_vec);
    }

    let mut mod_sample_frame = mul_op(target_op, &sample_frame, TransMode::NoTrans);

    let max_err = if inv {
        mod_sample_frame = mul_op(rsrs_op, &mod_sample_frame, trans_mode);
        mod_sample_frame
            .iter_mut()
            .zip(sample_frame.iter())
            .map(|(approx_vec, ref_vec)| {
                approx_vec.sub_inplace(ref_vec.r());
                if approx_vec.r().norm() == num::Zero::zero() {
                    num::One::one()
                } else {
                    approx_vec.r().norm() / ref_vec.r().norm()
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    } else {
        sample_frame = mul_op(rsrs_op, &sample_frame, trans_mode);
        sample_frame
            .iter_mut()
            .zip(mod_sample_frame.iter())
            .map(|(approx_vec, ref_vec)| {
                approx_vec.sub_inplace(ref_vec.r());
                if ref_vec.r().norm() == num::Zero::zero() {
                    num::One::one()
                } else {
                    approx_vec.r().norm() / ref_vec.r().norm()
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    };

    max_err.unwrap()
}

pub fn rsrs_error_estimator<
    'a,
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_operator: &OpImpl2,
    sample_size: usize,
    inv: bool,
) -> (Real<Item>, Real<Item>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let app_err_left = app_error(target_op, rsrs_operator, sample_size, RsrsSide::Left, inv);

    let app_err_right = app_error(target_op, rsrs_operator, sample_size, RsrsSide::Right, inv);

    (app_err_left, app_err_right)
}

/////////////Dense matrix error extraction

type Errors = (f64, f64);
type ErrorStats = (f64, f64, f64, f64);

// Error functions
pub fn spectral_norm_estimator<Item: RlstScalar + RandScalar>(
    arr: &DynamicArray<Item, 2>,
    sample_size: usize,
) -> std::option::Option<f64>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let dim = arr.shape()[1];

    let max_err = (0..sample_size)
        .into_iter()
        .map(|_sample_ind| {
            let mut test_vec = rlst_dynamic_array1!(Item, [dim]);
            let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
            test_vec.fill_from_standard_normal(&mut local_rng);
            let mut res_vec = empty_array();
            res_vec
                .r_mut()
                .simple_mult_into_resize(arr.r(), test_vec.r());
            res_vec.norm_2() / test_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err.map(|val| num::NumCast::from(val).unwrap())
}

fn box_errors_id<Item: RlstScalar + RandScalar>(
    id_factor: &IdFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Errors
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let ind_r = &id_factor.ind_r;
    let far_indices = &id_factor.ind_f;

    let arr_rf = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_r.clone(), far_indices.clone()),
    )
    .unwrap()
    .ext;
    let arr_fr = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(far_indices.clone(), ind_r.clone()),
    )
    .unwrap()
    .ext;

    let arr_rf = spectral_norm_estimator(&arr_rf, 10).unwrap();
    let arr_fr = spectral_norm_estimator(&arr_fr, 10).unwrap();

    (arr_rf, arr_fr)
}

fn box_errors_lu<Item: RlstScalar + RandScalar>(
    lu_factor: &LuFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Errors
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let ind_r = &lu_factor.ind_r;
    let ind_t = &lu_factor.ind_t;

    let arr_rt = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_r.clone(), ind_t.clone()),
    )
    .unwrap()
    .ext;
    let arr_tr = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_t.clone(), ind_r.clone()),
    )
    .unwrap()
    .ext;

    let arr_rt = spectral_norm_estimator(&arr_rt, 10).unwrap();
    let arr_tr = spectral_norm_estimator(&arr_tr, 10).unwrap();

    (arr_rt, arr_tr)
}

fn commutative_factors_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse + MatrixLu + MatrixId + MatrixQr,
>(
    factors: &CommutativeFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
) -> Vec<Errors>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let target_arr = Arc::new(Mutex::new(target_arr));
    let mul_type_left = FactorMulType {
        side: Side::Left,
        factor_type: FactorType::F,
        right_trans: false,
    };
    let mul_type_right = FactorMulType {
        side: Side::Right,
        factor_type: FactorType::S,
        right_trans: false,
    };

    let factor_options = FactorOptions {
        inv: true,
        trans: false,
    };

    let errors: Vec<_> = factors
        .par_iter()
        .map(|factor| {
            let mut target_arr = target_arr.lock().unwrap();
            match factor {
                Factor::Lu(lu_factor) => {
                    let (arr_rt, arr_tr) = box_errors_lu(lu_factor, &mut target_arr);
                    lu_factor.mul(&mut target_arr, &factor_options, &mul_type_left);
                    lu_factor.mul(&mut target_arr, &factor_options, &mul_type_right);
                    let (arr_rt_ae, arr_tr_ae) = box_errors_lu(lu_factor, &mut target_arr);
                    let rel_errs: Errors = (arr_rt_ae / arr_rt, arr_tr_ae / arr_tr);
                    rel_errs
                }
                Factor::Id(id_factor) => {
                    let (arr_rf, arr_fr) = box_errors_id(id_factor, &mut target_arr);
                    id_factor.mul(&mut target_arr, &factor_options, &mul_type_left);
                    id_factor.mul(&mut target_arr, &factor_options, &mul_type_right);
                    let (arr_rf_ae, arr_fr_ae) = box_errors_id(id_factor, &mut target_arr);
                    let rel_errs: Errors = (arr_rf_ae / arr_rf, arr_fr_ae / arr_fr);
                    rel_errs
                }
                Factor::Diag(diag_box_factor) => {
                    let mut exact_diag_box = <Extraction<Item> as MatrixExtraction>::new(
                        &mut target_arr,
                        ExtInsType::Cross(
                            diag_box_factor.inds.clone(),
                            diag_box_factor.inds.clone(),
                        ),
                    )
                    .unwrap()
                    .ext;

                    let shape = exact_diag_box.shape();

                    let mut app_dbox = rlst_dynamic_array2!(Item, shape);
                    app_dbox.set_identity();

                    let options = FactorOptions {
                        inv: false,
                        trans: false,
                    };
                    diag_box_factor.arr.mul(&mut app_dbox, Side::Left, &options);

                    let mut res: DynamicArray<Item, 2> = empty_array();
                    res.fill_from_resize(exact_diag_box.r() - app_dbox.r());

                    let err_diag = spectral_norm_estimator(&res, 10).unwrap()
                        / spectral_norm_estimator(&exact_diag_box, 10).unwrap();

                    let mut app_inv_dbox = rlst_dynamic_array2!(Item, shape);
                    app_inv_dbox.set_identity();

                    let options = FactorOptions {
                        inv: true,
                        trans: false,
                    };

                    diag_box_factor
                        .arr
                        .mul(&mut app_inv_dbox, Side::Left, &options);

                    let _ = exact_diag_box.r_mut().into_inverse_alloc().unwrap();

                    let mut res: DynamicArray<Item, 2> = empty_array();
                    res.fill_from_resize(exact_diag_box.r() - app_inv_dbox.r());

                    let err_inv_diag = spectral_norm_estimator(&res, 10).unwrap()
                        / spectral_norm_estimator(&exact_diag_box, 10).unwrap();

                    let errors: Errors = (err_diag, err_inv_diag);
                    errors
                }
            }
        })
        .collect();

    errors
}

fn el_factors_inv_mul_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
) -> (Vec<ErrorStats>, Vec<ErrorStats>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let errors: Vec<(Vec<Errors>, Vec<Errors>)> = (0..rsrs_factors.num_levels)
        .map(|level_it| {
            let factors = &rsrs_factors.id_factors[level_it];
            let id_errors = commutative_factors_errors(&factors, target_arr);
            let lu_errors = rsrs_factors.lu_factors[level_it]
                .iter()
                .map(|lu_batch| commutative_factors_errors(&lu_batch, target_arr))
                .flatten()
                .collect();
            (id_errors, lu_errors)
        })
        .collect();

    let stats = |errors_vec: Vec<Errors>| {
        let mut mu_1 = 0.0;
        let mut mu_2 = 0.0;
        let mut std_dev_1 = 0.0;
        let mut std_dev_2 = 0.0;

        errors_vec.iter().for_each(|(errors_1, errors_2)| {
            mu_1 += *errors_1;
            mu_2 += *errors_2;
        });

        let len: f64 = NumCast::from(errors_vec.len()).unwrap();
        mu_1 /= len;
        mu_2 /= len;

        errors_vec.iter().for_each(|(errors_1, errors_2)| {
            std_dev_1 += (*errors_1 - mu_1).powi(2);
            std_dev_2 += (*errors_2 - mu_2).powi(2);
        });

        std_dev_1 /= len;
        std_dev_2 /= len;

        (mu_1, mu_2, std_dev_1.sqrt(), std_dev_2.sqrt())
    };

    let mut id_stats = Vec::new();
    let mut lu_stats = Vec::new();
    errors
        .iter()
        .for_each(|(id_level_errors, lu_level_errors)| {
            if !id_level_errors.is_empty() {
                id_stats.push(stats(id_level_errors.to_vec()));
                lu_stats.push(stats(lu_level_errors.to_vec()));
            }
        });
    (id_stats, lu_stats)
}

pub fn get_boxes_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse + MatrixId + MatrixLu + MatrixQr,
>(
    structured_operator_mat: &mut DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    _tol: f64,
) where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let (id_error_stats, lu_error_stats) =
        &el_factors_inv_mul_errors(rsrs_factors, structured_operator_mat);

    id_error_stats
        .iter()
        .enumerate()
        .for_each(|(level, stats)| {
            let (mu_1, mu_2, std_dev_1, std_dev_2) = stats;
            println!(
                "Errors ID, level {} : ({} +/- {}, {} +/- {})",
                level, mu_1, std_dev_1, mu_2, std_dev_2
            );
        });

    lu_error_stats
        .iter()
        .enumerate()
        .for_each(|(level, stats)| {
            let (mu_1, mu_2, std_dev_1, std_dev_2) = stats;
            println!(
                "Errors LU, level {} : ({} +/- {}, {} +/- {})",
                level, mu_1, std_dev_1, mu_2, std_dev_2
            );
            //assert!(*mu_1 <= tol && *mu_2 <= tol);
        });

    println!("\n");

    let diag_re =
        commutative_factors_errors(&rsrs_factors.diag_box_factors, structured_operator_mat);

    let diag_re_r;
    let diag_re_s;

    if diag_re.len() > 1 {
        diag_re_r = &diag_re[0..diag_re.len() - 2];
        diag_re_s = diag_re[diag_re.len() - 1];
    } else {
        diag_re_r = &[];
        diag_re_s = diag_re[0];
    }

    let diag_re_r_sum = diag_re_r
        .into_iter()
        .fold((0.0, 0.0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

    let len: f64 = NumCast::from(diag_re_r.len()).unwrap();
    let diag_re_r_mean = (diag_re_r_sum.0 / len, diag_re_r_sum.1 / len);

    println!(
        "Mean residual diagonal blocks errors : {:?}, sketch block error: {:?}",
        diag_re_r_mean, diag_re_s
    );

    /*assert!(
        diag_re_r_mean.0 <= tol
            && diag_re_r_mean.1 <= tol
            && diag_re_s.0 <= tol
            && diag_re_s.1 <= tol
    );*/
}
