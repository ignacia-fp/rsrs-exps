use super::python_kernel::{Kernel, KernelImpl, KernelOperator};
use bempp_rsrs::rsrs::rsrs_factors::LocalFrom;
use bempp_rsrs::rsrs::rsrs_factors::{RsrsFactors, RsrsFactorsImpl, RsrsOperator, RsrsSide};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
type Real<T> = <T as rlst::RlstScalar>::Real;

fn gen_sample_frame<Item: RlstScalar + RandScalar>(
    sample_size: usize,
    space: std::rc::Rc<ArrayVectorSpace<Item>>,
) -> VectorFrame<ArrayVectorSpaceElement<Item>>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let mut frame = VectorFrame::default();
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();

    for _ in 0..sample_size {
        let mut sample_vec = ArrayVectorSpace::zero(space.clone());
        sample_vec.view_mut().fill_from_standard_normal(&mut rng);
        frame.push(sample_vec);
    }
    frame
}

fn mul_op<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
>(
    operator: &OpImpl,
    sample_frame: &VectorFrame<ArrayVectorSpaceElement<Item>>,
    trans_mode: TransMode,
) -> VectorFrame<ArrayVectorSpaceElement<Item>> {
    let mut frame = VectorFrame::default();
    for sample_vec in sample_frame.iter() {
        frame.push(operator.apply(sample_vec.r(), trans_mode));
    }

    frame
}

pub fn app_error<
    'a,
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    Op1: KernelImpl<Item> + Shape<2>,
    Op2: RsrsFactorsImpl<Item> + Shape<2>,
>(
    target_op: &KernelOperator<'a, Item, Op1>,
    rsrs_op: &mut RsrsOperator<'a, Item, Op2>,
    sample_size: usize,
    side: RsrsSide,
    inv: bool,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let trans_mode = if matches!(side, RsrsSide::Left) {
        TransMode::NoTrans
    } else {
        TransMode::Trans
    };

    let mut sample_frame = gen_sample_frame(sample_size, target_op.range());
    let mut mod_sample_frame = mul_op(target_op, &sample_frame, trans_mode);

    rsrs_op.set_inv(inv);

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
>(
    target_op: &KernelOperator<'a, Item, Kernel>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
) -> (Real<Item>, Real<Item>, Real<Item>, Real<Item>, Real<Item>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    Kernel: KernelImpl<Item>,
{
    let mut rsrs_operator = RsrsOperator::from_local(rsrs_factors);

    let app_err_left = app_error(
        target_op,
        &mut rsrs_operator,
        sample_size,
        RsrsSide::Left,
        false,
    );

    let app_err_right = app_error(
        target_op,
        &mut rsrs_operator,
        sample_size,
        RsrsSide::Right,
        false,
    );

    let app_inv_err_left = app_error(
        target_op,
        &mut rsrs_operator,
        sample_size,
        RsrsSide::Left,
        true,
    );

    let app_inv_err_right = app_error(
        target_op,
        &mut rsrs_operator,
        sample_size,
        RsrsSide::Right,
        true,
    );

    let cond = num::One::one();

    (
        app_inv_err_left,
        app_inv_err_right,
        app_err_left,
        app_err_right,
        cond,
    )
}
