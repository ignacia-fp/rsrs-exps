use bempp_rsrs::rsrs::rsrs_factors::base_factors::{BaseFactorOptions, FactorData};
use bempp_rsrs::rsrs::rsrs_factors::commutative_factors::MultiLevelIdFactors;
use bempp_rsrs::rsrs::rsrs_factors::commutative_factors::RsrsFactors;
use bempp_rsrs::rsrs::rsrs_factors::commutative_factors::{
    CommutativeFactors, Factor, FactorOperations, FactorType, IdFactor, LuFactor, MulOptions,
};
use bempp_rsrs::rsrs::rsrs_factors::rsrs_operator::Inv;
use bempp_rsrs::rsrs::sketch::{SampleType, SamplingSpace};
use bempp_rsrs::utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction};
use mpi::traits::Communicator;
use mpi::traits::Equivalence;
use num::{FromPrimitive, NumCast};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::{
    dense::{
        linalg::{interpolative_decomposition::MatrixIdNoSkel, lu::MatrixLu},
        tools::RandScalar,
    },
    prelude::*,
};
use serde::Serialize;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

type Real<T> = <T as rlst::RlstScalar>::Real;

pub struct IdOperator<
    Item: RlstScalar + RandScalar,
    Space: SamplingSpace<F = Item> + IndexableSpace,
> {
    domain: Rc<Space>,
    range: Rc<Space>,
}

impl<Item: RlstScalar + RandScalar, Space: SamplingSpace<F = Item> + IndexableSpace> std::fmt::Debug
    for IdOperator<Item, Space>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim_1 = self.domain().dimension();
        let dim_2 = self.range().dimension();
        write!(f, "Id Operator: [{}x{}]", dim_1, dim_2).unwrap();
        Ok(())
    }
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + LinearSpace + IndexableSpace,
    > OperatorBase for IdOperator<Item, Space>
{
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<Item: RlstScalar + RandScalar> AsApply for IdOperator<Item, ArrayVectorSpace<Item>>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        _trans_mode: TransMode,
    ) {
        y.r_mut().fill_inplace(x);
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

impl<C: Communicator, Item: RlstScalar + Equivalence + RandScalar> AsApply
    for IdOperator<Item, DistributedArrayVectorSpace<'_, C, Item>>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        _trans_mode: TransMode,
    ) {
        y.r_mut().fill_inplace(x);
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

impl<Item: RlstScalar + RandScalar, Space: SamplingSpace<F = Item> + IndexableSpace>
    IdOperator<Item, Space>
{
    pub fn new(domain: Rc<Space>, range: Rc<Space>) -> Self {
        IdOperator { domain, range }
    }
}

pub struct NormalOperator<
    Item: RlstScalar + RandScalar,
    Space: SamplingSpace<F = Item> + IndexableSpace,
    Op1: OperatorBase<Domain = Space, Range = Space>,
> {
    pub op: Op1,
    pub transpose_matches_apply: bool,
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + IndexableSpace,
        Op1: OperatorBase<Domain = Space, Range = Space>,
    > NormalOperator<Item, Space, Op1>
{
    pub fn new(op: Op1, transpose_matches_apply: bool) -> Self {
        Self {
            op,
            transpose_matches_apply,
        }
    }
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + IndexableSpace,
        Op1: OperatorBase<Domain = Space, Range = Space>,
    > std::fmt::Debug for NormalOperator<Item, Space, Op1>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim_1 = self.op.domain().dimension();
        write!(f, "Id Operator: [{}x{}]", dim_1, dim_1).unwrap();
        Ok(())
    }
}

pub struct DiffOperator<
    Item: RlstScalar + RandScalar,
    Space: SamplingSpace<F = Item> + IndexableSpace,
    Op1: OperatorBase<Domain = Space, Range = Space>,
    Op2: OperatorBase<Domain = Space, Range = Space>,
>(pub Op1, pub Op2);

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + IndexableSpace,
        Op1: OperatorBase<Domain = Space, Range = Space>,
        Op2: OperatorBase<Domain = Space, Range = Space>,
    > std::fmt::Debug for DiffOperator<Item, Space, Op1, Op2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim_1 = self.0.domain().dimension();
        let dim_2 = self.0.range().dimension();
        write!(f, "Id Operator: [{}x{}]", dim_1, dim_2).unwrap();
        Ok(())
    }
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + LinearSpace + IndexableSpace,
        Op1: OperatorBase<Domain = Space, Range = Space>,
        Op2: OperatorBase<Domain = Space, Range = Space>,
    > OperatorBase for DiffOperator<Item, Space, Op1, Op2>
{
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.0.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.0.range().clone()
    }
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + LinearSpace + IndexableSpace,
        Op1: OperatorBase<Domain = Space, Range = Space>,
    > OperatorBase for NormalOperator<Item, Space, Op1>
{
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.op.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.op.range().clone()
    }
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + LinearSpace + IndexableSpace,
        Op1: AsApply<Domain = Space, Range = Space>,
        Op2: AsApply<Domain = Space, Range = Space>,
    > AsApply for DiffOperator<Item, Space, Op1, Op2>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        trans_mode: TransMode,
    ) {
        let mut y_aux = zero_element(self.range());

        self.0
            .apply_extended(alpha, x.r(), beta, y_aux.r_mut(), trans_mode);
        self.1
            .apply_extended(alpha, x.r(), beta, y.r_mut(), trans_mode);

        y.r_mut().sub_inplace(y_aux.r());
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + LinearSpace + IndexableSpace,
        Op1: AsApply<Domain = Space, Range = Space>,
    > AsApply for NormalOperator<Item, Space, Op1>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        _trans_mode: TransMode,
    ) {
        let mut y_aux_1 = zero_element(self.range());
        let mut y_aux_2 = zero_element(self.range());
        self.op
            .apply_extended(alpha, x.r(), beta, y_aux_1.r_mut(), TransMode::NoTrans);
        let y_aux_1_conj = self.domain().conj_vec(&y_aux_1);

        if self.transpose_matches_apply {
            self.op.apply_extended(
                alpha,
                y_aux_1_conj,
                beta,
                y_aux_2.r_mut(),
                TransMode::NoTrans,
            );
        } else {
            self.op
                .apply_extended(alpha, y_aux_1_conj, beta, y_aux_2.r_mut(), TransMode::Trans);
        }

        let y_aux_2_conj = self.domain().conj_vec(&y_aux_2);
        y.r_mut().fill_inplace(y_aux_2_conj);
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

fn _gen_sample_frame<Item: RlstScalar + RandScalar, Space: SamplingSpace<F = Item>>(
    sample_size: usize,
    space: Rc<Space>,
    seed: u64,
) -> VectorFrame<<Space as rlst::LinearSpace>::E>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let mut frame = VectorFrame::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..sample_size {
        let mut sample_vec = SamplingSpace::zero(space.clone());
        space.sampling(&mut sample_vec, &mut rng, SampleType::StandardNormal);
        frame.push(sample_vec);
    }
    frame
}

fn _gen_real_gaussian_sample_frame<Item: RlstScalar + RandScalar, Space: SamplingSpace<F = Item>>(
    sample_size: usize,
    space: Rc<Space>,
    seed: u64,
) -> VectorFrame<<Space as rlst::LinearSpace>::E>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let mut frame = VectorFrame::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..sample_size {
        let mut sample_vec = SamplingSpace::zero(space.clone());
        space.sampling(&mut sample_vec, &mut rng, SampleType::RealStandardNormal);
        frame.push(sample_vec);
    }
    frame
}

fn mul_op<
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
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

pub fn app_error_right<
    'a,
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_op: &OpImpl2,
    sample_size: usize,
    inv: bool,
    seed: u64,
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
    let trans_mode = TransMode::Trans;

    let mut sample_frame = VectorFrame::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let space = target_op.range();

    for _ in 0..sample_size {
        let mut sample_vec = SamplingSpace::zero(space.clone());
        space.sampling(&mut sample_vec, &mut rng, SampleType::StandardNormal);
        sample_frame.push(sample_vec);
    }

    let mut mod_sample_frame = mul_op(rsrs_op, &sample_frame, trans_mode);

    let max_err = if inv {
        mod_sample_frame = mul_op(target_op, &mod_sample_frame, trans_mode);
        mod_sample_frame
            .iter_mut()
            .zip(sample_frame.iter())
            .map(|(approx_vec, ref_vec)| {
                approx_vec.sub_inplace(ref_vec.r());
                if ref_vec.r().norm() == num::Zero::zero() {
                    approx_vec.r().norm()
                } else {
                    approx_vec.r().norm() / ref_vec.r().norm()
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    } else {
        sample_frame = mul_op(target_op, &sample_frame, trans_mode);
        sample_frame
            .iter_mut()
            .zip(mod_sample_frame.iter_mut())
            .map(|(ref_vec, approx_vec)| {
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

pub fn app_error_left<
    'a,
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_op: &OpImpl2,
    sample_size: usize,
    inv: bool,
    seed: u64,
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
    let trans_mode = TransMode::NoTrans;

    let mut sample_frame = VectorFrame::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let space = target_op.domain();

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
                if ref_vec.r().norm() == num::Zero::zero() {
                    approx_vec.r().norm()
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
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_operator: &OpImpl2,
    sample_size: usize,
    inv: bool,
    left_seed: u64,
    right_seed: u64,
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
    let app_err_left = app_error_left(target_op, rsrs_operator, sample_size, inv, left_seed);

    let app_err_right = app_error_right(target_op, rsrs_operator, sample_size, inv, right_seed);

    (app_err_left, app_err_right)
}

pub fn frobenius_diff_and_reference_norm<
    Item: RlstScalar,
    Space: SamplingSpace<F = Item> + IndexableSpace,
    OpRef: AsApply<Domain = Space, Range = Space>,
    OpApprox: AsApply<Domain = Space, Range = Space>,
>(
    reference_op: &OpRef,
    approx_op: &OpApprox,
    sample_size: usize,
    seed: u64,
) -> (Real<Item>, Real<Item>)
where
    Item: RandScalar
        + MatrixInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let sample_size = sample_size.max(1);
    let sample_frame = _gen_real_gaussian_sample_frame(sample_size, reference_op.domain(), seed);
    let reference_frame = mul_op(reference_op, &sample_frame, TransMode::NoTrans);
    let mut approx_frame = mul_op(approx_op, &sample_frame, TransMode::NoTrans);

    let mut reference_sq = 0.0f64;
    let mut diff_sq = 0.0f64;
    let sample_scale = (sample_size as f64).sqrt();

    for (reference_col, approx_col) in reference_frame.iter().zip(approx_frame.iter_mut()) {
        approx_col.sub_inplace(reference_col.r());

        let reference_norm: f64 = NumCast::from(reference_col.norm()).unwrap();
        let diff_norm: f64 = NumCast::from(approx_col.norm()).unwrap();

        reference_sq += reference_norm * reference_norm;
        diff_sq += diff_norm * diff_norm;
    }

    let reference_fro = reference_sq.sqrt() / sample_scale;
    let diff_fro = diff_sq.sqrt() / sample_scale;
    let rel_diff = diff_fro / reference_fro.max(1.0e-14);

    (
        Real::<Item>::from_f64(rel_diff).unwrap(),
        Real::<Item>::from_f64(reference_fro).unwrap(),
    )
}

/////////////Dense matrix error extraction
type ErrorsFactor = (f64, f64, f64, f64);
type ErrorStats = (f64, f64, f64, f64, f64, f64, f64, f64);

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
) -> ErrorsFactor
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

    let mut res = empty_array();
    res.fill_from_resize(arr_rf.r() - arr_fr.r().transpose());

    let arr_rf = arr_rf.r().norm_fro();
    let arr_fr = arr_fr.r().transpose().norm_fro();

    (
        num::NumCast::from(arr_rf).unwrap(),
        num::NumCast::from(arr_fr).unwrap(),
        0.0,
        0.0,
    )
}

fn box_errors_lu<Item: RlstScalar + RandScalar + MatrixLu + MatrixInverse + MatrixPseudoInverse>(
    lu_factor: &LuFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
    get_factors_errors: bool,
) -> ErrorsFactor
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
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

    let base_options = BaseFactorOptions {
        inv: false,
        trans: TransMode::NoTrans,
        trans_target: false,
    };

    if get_factors_errors {
        let (err_rect, err_sq) = match &lu_factor.u_arr {
            FactorData::Comp(composed_factor_data) => {
                let mut err_rect = empty_array();
                let mut err_sq = empty_array();
                err_rect
                    .r_mut()
                    .fill_from_resize(composed_factor_data.rectg.arr.r() - arr_rt.r());

                let mut app_lu = rlst_dynamic_array2!(Item, [ind_r.len(), ind_r.len()]);
                app_lu.set_identity();

                let b_options = base_options.clone();
                composed_factor_data
                    .sq
                    .mul(&mut app_lu, Side::Left, &b_options);

                let arr_rr = <Extraction<Item> as MatrixExtraction>::new(
                    arr,
                    ExtInsType::Cross(ind_r.clone(), ind_r.clone()),
                )
                .unwrap()
                .ext;

                err_sq.r_mut().fill_from_resize(app_lu.r() - arr_rr.r());

                (
                    err_rect.norm_fro() / arr_rt.r().norm_fro(),
                    err_sq.norm_fro() / arr_rr.norm_fro(),
                )
            }
            FactorData::Reg(_array) => todo!(),
        };

        let arr_rt = arr_rt.r().norm_fro();
        let arr_tr = arr_tr.r().transpose().norm_fro();

        (
            num::NumCast::from(arr_rt).unwrap(),
            num::NumCast::from(arr_tr).unwrap(),
            num::NumCast::from(err_rect).unwrap(),
            num::NumCast::from(err_sq).unwrap(),
        )
    } else {
        let arr_rt = arr_rt.r().norm_fro();
        let arr_tr = arr_tr.r().transpose().norm_fro();
        (
            num::NumCast::from(arr_rt).unwrap(),
            num::NumCast::from(arr_tr).unwrap(),
            0.0,
            0.0,
        )
    }

    //(arr_rt, arr_tr)
}

fn commutative_factors_errors<
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixId
        + MatrixIdNoSkel
        + MatrixQr,
>(
    factors: &CommutativeFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
) -> Vec<ErrorsFactor>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let target_arr = Arc::new(Mutex::new(target_arr));

    let base_options = BaseFactorOptions {
        inv: true,
        trans: TransMode::NoTrans,
        trans_target: false,
    };
    let factor_options_left = MulOptions {
        base_options: base_options.clone(),
        side: Side::Left,
        factor_type: FactorType::F,
    };

    let factor_options_right = MulOptions {
        base_options: base_options.clone(),
        side: Side::Right,
        factor_type: FactorType::S,
    };

    let errors: Vec<_> = factors
        .par_iter()
        .map(|factor| {
            let mut target_arr = target_arr.lock().unwrap();
            match factor {
                Factor::Lu(lu_factor) => {
                    let (arr_rt, arr_tr, err_rect, err_sq) =
                        box_errors_lu(lu_factor, &mut target_arr, true);
                    lu_factor.mul(&mut target_arr, &factor_options_left);
                    lu_factor.mul(&mut target_arr, &factor_options_right);
                    let (arr_rt_ae, arr_tr_ae, _, _) =
                        box_errors_lu(lu_factor, &mut target_arr, false);
                    let rel_errs: ErrorsFactor =
                        (arr_rt_ae / arr_rt, arr_tr_ae / arr_tr, err_rect, err_sq);
                    rel_errs
                }
                Factor::Id(id_factor) => {
                    let (arr_rf, arr_fr, _, _) = box_errors_id(id_factor, &mut target_arr);
                    id_factor.mul(&mut target_arr, &factor_options_left);
                    id_factor.mul(&mut target_arr, &factor_options_right);
                    let (arr_rf_ae, arr_fr_ae, _, _) = box_errors_id(id_factor, &mut target_arr);
                    let rel_errs: ErrorsFactor = (arr_rf_ae / arr_rf, arr_fr_ae / arr_fr, 0.0, 0.0);
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

                    let base_options = BaseFactorOptions {
                        inv: false,
                        trans: TransMode::NoTrans,
                        trans_target: false,
                    };

                    diag_box_factor
                        .arr
                        .mul(&mut app_dbox, &Side::Left, &base_options);

                    let mut res: DynamicArray<Item, 2> = empty_array();
                    res.fill_from_resize(exact_diag_box.r() - app_dbox.r());

                    let err_diag = spectral_norm_estimator(&res, 10).unwrap()
                        / spectral_norm_estimator(&exact_diag_box, 10).unwrap();

                    let mut identity = rlst_dynamic_array2!(Item, shape);
                    identity.set_identity();

                    let base_options = BaseFactorOptions {
                        inv: true,
                        trans: TransMode::NoTrans,
                        trans_target: false,
                    };

                    diag_box_factor
                        .arr
                        .mul(&mut exact_diag_box, &Side::Left, &base_options);

                    let mut res: DynamicArray<Item, 2> = empty_array();
                    res.fill_from_resize(exact_diag_box.r() - identity.r());

                    let err_inv_diag = spectral_norm_estimator(&res, 10).unwrap();

                    let errors: ErrorsFactor = (err_diag, err_inv_diag, 0.0, 0.0);
                    errors
                }
            }
        })
        .collect();

    errors
}

fn el_factors_inv_mul_errors<
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
) -> (Vec<Vec<ErrorStats>>, Vec<Vec<ErrorStats>>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let errors: Vec<(Vec<Vec<ErrorsFactor>>, Vec<Vec<ErrorsFactor>>)> = (0..rsrs_factors
        .num_levels)
        .map(|level_it| {
            let (id_errors, lu_errors) = match &rsrs_factors.id_factors {
                MultiLevelIdFactors::Single(id_factors) => {
                    let factors = &id_factors[level_it];
                    let id_errors = commutative_factors_errors(factors, target_arr);
                    let lu_errors = rsrs_factors.lu_factors[level_it]
                        .iter()
                        .map(|lu_batch| {
                            println!("lu batch len: {}", lu_batch.len());
                            commutative_factors_errors(&lu_batch, target_arr)
                        })
                        .collect();
                    (vec![id_errors], lu_errors)
                }
                MultiLevelIdFactors::Batched(id_factors) => {
                    let mut id_errors = Vec::new();
                    let mut lu_errors = Vec::new();

                    for (ind_batch, id_batch) in id_factors[level_it].iter().enumerate() {
                        // apply ID
                        let id_err = commutative_factors_errors(id_batch, target_arr);
                        id_errors.push(id_err);

                        // apply LU for *this batch*
                        let lu_batch = &rsrs_factors.lu_factors[level_it][ind_batch];
                        let lu_err = commutative_factors_errors(lu_batch, target_arr);
                        lu_errors.push(lu_err);
                    }

                    (id_errors, lu_errors)
                }
            };

            (id_errors, lu_errors)
        })
        .collect();

    let stats = |errors_vec: Vec<ErrorsFactor>| {
        let mut mu_1 = 0.0;
        let mut mu_2 = 0.0;
        let mut mu_3 = 0.0;
        let mut mu_4 = 0.0;
        let mut std_dev_1 = 0.0;
        let mut std_dev_2 = 0.0;
        let mut std_dev_3 = 0.0;
        let mut std_dev_4 = 0.0;

        errors_vec
            .iter()
            .for_each(|(errors_1, errors_2, errors_3, errors_4)| {
                mu_1 += *errors_1;
                mu_2 += *errors_2;
                mu_3 += *errors_3;
                mu_4 += *errors_4;
            });

        let len: f64 = NumCast::from(errors_vec.len()).unwrap();
        mu_1 /= len;
        mu_2 /= len;
        mu_3 /= len;
        mu_4 /= len;

        errors_vec
            .iter()
            .for_each(|(errors_1, errors_2, errors_3, errors_4)| {
                std_dev_1 += (*errors_1 - mu_1).powi(2);
                std_dev_2 += (*errors_2 - mu_2).powi(2);
                std_dev_3 += (*errors_3 - mu_3).powi(2);
                std_dev_4 += (*errors_4 - mu_4).powi(2);
            });

        std_dev_1 /= len;
        std_dev_2 /= len;
        std_dev_3 /= len;
        std_dev_4 /= len;

        (
            mu_1,
            mu_2,
            mu_3,
            mu_4,
            std_dev_1.sqrt(),
            std_dev_2.sqrt(),
            std_dev_3.sqrt(),
            std_dev_4.sqrt(),
        )
    };

    let mut id_stats = Vec::new();
    let mut lu_stats = Vec::new(); // This will be Vec<Vec<(mu1, mu2, std1, std2)>>

    errors
        .iter()
        .for_each(|(id_level_errors, lu_level_errors)| {
            if !id_level_errors.is_empty() {
                // Stats for ID errors at this level
                //id_stats.push(stats(id_level_errors.to_vec()));
                let id_level_stats: Vec<_> = id_level_errors
                    .iter()
                    .map(|batch| stats(batch.to_vec()))
                    .collect();
                id_stats.push(id_level_stats);

                // Stats for each LU batch at this level
                let lu_level_stats: Vec<_> = lu_level_errors
                    .iter()
                    .map(|batch| stats(batch.to_vec()))
                    .collect();

                lu_stats.push(lu_level_stats);
            }
        });
    (id_stats, lu_stats)
}

#[derive(Serialize)]
struct ErrorStatsContainer {
    id_error_stats: Vec<Vec<(f64, f64, f64, f64, f64, f64, f64, f64)>>,
    lu_error_stats: Vec<Vec<(f64, f64, f64, f64, f64, f64, f64, f64)>>,
    diag_error_stats: (f64, f64),
}

pub fn get_boxes_errors<
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixLu
        + MatrixQr,
>(
    structured_operator_mat: &mut DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    tol: f64,
    path_str: &str,
) where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let (id_error_stats, lu_error_stats) =
        el_factors_inv_mul_errors(rsrs_factors, structured_operator_mat);

    for (level, level_stats) in id_error_stats.iter().enumerate() {
        for (batch, s) in level_stats.iter().enumerate() {
            println!(
                "Errors ID, level {}, batch {} : ({} +/- {}, {} +/- {})",
                level, batch, s.0, s.4, s.1, s.5
            );
        }
    }

    // Print LU stats per batch in each level
    for (level, level_stats) in lu_error_stats.iter().enumerate() {
        for (batch, s) in level_stats.iter().enumerate() {
            println!(
                "Errors LU, level {}, batch {} : ({} +/- {}, {} +/- {})",
                level, batch, s.0, s.4, s.1, s.5
            );
        }
    }

    println!("\n");

    // Diagonal box factor stats
    let diag_re =
        commutative_factors_errors(&rsrs_factors.diag_box_factors, structured_operator_mat);
    let (diag_re_r, diag_re_s) = if diag_re.len() > 1 {
        (&diag_re[..diag_re.len() - 1], diag_re[diag_re.len() - 1])
    } else {
        (&[][..], diag_re[0])
    };

    let diag_re_r_sum = diag_re_r
        .iter()
        .fold((0.0, 0.0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

    let len: f64 = NumCast::from(diag_re_r.len()).unwrap_or(1.0);
    let diag_re_r_mean = (diag_re_r_sum.0 / len, diag_re_r_sum.1 / len);

    println!(
        "Mean residual diagonal blocks errors : {:?}, sketch block error: {:?}",
        diag_re_r_mean, diag_re_s
    );

    let diag_error_stats = (diag_re_r_mean.0, diag_re_r_mean.1);

    // Flatten LU stats to save to JSON as a flat list
    //let lu_error_stats_flat: Vec<ErrorStats> = lu_error_stats.into_iter().flatten().collect();

    let error_stats_container = ErrorStatsContainer {
        id_error_stats,
        lu_error_stats, //: lu_error_stats_flat,
        diag_error_stats,
    };

    // Serialize and write to file
    fs::create_dir_all(Path::new(path_str)).unwrap();
    let stats_path = format!("{}/block_error_stats_{:e}.json", path_str, tol);

    let json_string = serde_json::to_string_pretty(&error_stats_container)
        .expect("Failed to serialize error stats");

    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::Complex;

    struct DenseMatrixOperator<Item: RlstScalar> {
        matrix: DynamicArray<Item, 2>,
        domain: Rc<ArrayVectorSpace<Item>>,
        range: Rc<ArrayVectorSpace<Item>>,
    }

    impl<Item: RlstScalar> DenseMatrixOperator<Item> {
        fn new(matrix: DynamicArray<Item, 2>) -> Self {
            let shape = matrix.shape();
            Self {
                matrix,
                domain: ArrayVectorSpace::from_dimension(shape[1]),
                range: ArrayVectorSpace::from_dimension(shape[0]),
            }
        }

        fn entry(&self, row: usize, col: usize, trans_mode: TransMode) -> Item {
            match trans_mode {
                TransMode::NoTrans => self.matrix[[row, col]],
                TransMode::Trans => self.matrix[[col, row]],
                TransMode::ConjNoTrans => self.matrix[[row, col]].conj(),
                TransMode::ConjTrans => self.matrix[[col, row]].conj(),
            }
        }
    }

    impl<Item: RlstScalar> std::fmt::Debug for DenseMatrixOperator<Item> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "DenseMatrixOperator([{}x{}])",
                self.range.dimension(),
                self.domain.dimension()
            )
        }
    }

    impl<Item: RlstScalar> OperatorBase for DenseMatrixOperator<Item> {
        type Domain = ArrayVectorSpace<Item>;
        type Range = ArrayVectorSpace<Item>;

        fn domain(&self) -> Rc<Self::Domain> {
            self.domain.clone()
        }

        fn range(&self) -> Rc<Self::Range> {
            self.range.clone()
        }
    }

    impl<Item: RlstScalar> AsApply for DenseMatrixOperator<Item> {
        fn apply_extended<
            ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
            ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
        >(
            &self,
            alpha: <Self::Range as LinearSpace>::F,
            x: Element<ContainerIn>,
            beta: <Self::Range as LinearSpace>::F,
            mut y: Element<ContainerOut>,
            trans_mode: TransMode,
        ) {
            let row_dim = self.range.dimension();
            let col_dim = self.domain.dimension();
            let input = x.imp().view();
            let prev = y.imp().view().iter().collect::<Vec<_>>();
            let mut out = vec![Item::from_real(Item::real(0.0)); row_dim];

            for row in 0..row_dim {
                let mut accum = Item::from_real(Item::real(0.0));
                for col in 0..col_dim {
                    accum = accum + self.entry(row, col, trans_mode) * input[[col]];
                }
                out[row] = alpha * accum + beta * prev[row];
            }

            y.imp_mut().fill_inplace_raw(&out);
        }

        fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
            &self,
            x: Element<ContainerIn>,
            trans_mode: TransMode,
        ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
            let mut y = zero_element(self.range());
            self.apply_extended(
                <<Self::Range as LinearSpace>::F as num::One>::one(),
                x,
                <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
                y.r_mut(),
                trans_mode,
            );
            y
        }
    }

    struct ToggleInverseOperator<Item: RlstScalar> {
        forward: DenseMatrixOperator<Item>,
        inverse: DenseMatrixOperator<Item>,
        inv: bool,
    }

    impl<Item: RlstScalar> ToggleInverseOperator<Item> {
        fn new(forward: DynamicArray<Item, 2>, inverse: DynamicArray<Item, 2>) -> Self {
            Self {
                forward: DenseMatrixOperator::new(forward),
                inverse: DenseMatrixOperator::new(inverse),
                inv: false,
            }
        }

        fn active(&self) -> &DenseMatrixOperator<Item> {
            if self.inv {
                &self.inverse
            } else {
                &self.forward
            }
        }
    }

    impl<Item: RlstScalar> std::fmt::Debug for ToggleInverseOperator<Item> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "ToggleInverseOperator(inv={})", self.inv)
        }
    }

    impl<Item: RlstScalar> OperatorBase for ToggleInverseOperator<Item> {
        type Domain = ArrayVectorSpace<Item>;
        type Range = ArrayVectorSpace<Item>;

        fn domain(&self) -> Rc<Self::Domain> {
            self.active().domain()
        }

        fn range(&self) -> Rc<Self::Range> {
            self.active().range()
        }
    }

    impl<Item: RlstScalar> AsApply for ToggleInverseOperator<Item> {
        fn apply_extended<
            ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
            ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
        >(
            &self,
            alpha: <Self::Range as LinearSpace>::F,
            x: Element<ContainerIn>,
            beta: <Self::Range as LinearSpace>::F,
            y: Element<ContainerOut>,
            trans_mode: TransMode,
        ) {
            self.active().apply_extended(alpha, x, beta, y, trans_mode);
        }

        fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
            &self,
            x: Element<ContainerIn>,
            trans_mode: TransMode,
        ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
            self.active().apply(x, trans_mode)
        }
    }

    impl<Item: RlstScalar> Inv for ToggleInverseOperator<Item> {
        fn inv(&mut self, inv: bool) {
            self.inv = inv;
        }
    }

    fn rel_l2_error<Item: RlstScalar>(actual: &[Item], expected: &[Item]) -> f64 {
        let mut actual_arr = rlst_dynamic_array1!(Item, [actual.len()]);
        let mut expected_arr = rlst_dynamic_array1!(Item, [expected.len()]);
        actual_arr.fill_from_raw_data(actual);
        expected_arr.fill_from_raw_data(expected);

        let mut diff = empty_array();
        diff.fill_from_resize(actual_arr.r() - expected_arr.r());

        let num: f64 = NumCast::from(diff.norm_2()).unwrap();
        let den: f64 = NumCast::from(expected_arr.norm_2()).unwrap();
        num / den.max(1.0e-14)
    }

    fn apply_operator<Item, Op>(op: &Op, input: &[Item], trans_mode: TransMode) -> Vec<Item>
    where
        Item: RlstScalar,
        Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    {
        let mut x = zero_element(op.domain());
        x.imp_mut().fill_inplace_raw(input);
        op.apply(x.r(), trans_mode).view().iter().collect()
    }

    fn copy_array2<Item: RlstScalar>(arr: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
        let shape = arr.shape();
        let mut out = rlst_dynamic_array2!(Item, [shape[0], shape[1]]);
        out.fill_from(arr.r());
        out
    }

    fn exact_frobenius_norm_complex(arr: &DynamicArray<Complex<f64>, 2>) -> f64 {
        arr.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
    }

    fn populate_complex_test_matrix(arr: &mut DynamicArray<Complex<f64>, 2>) {
        let rows = arr.shape()[0];
        let cols = arr.shape()[1];
        for row in 0..rows {
            for col in 0..cols {
                let u = (row as f64 + 1.0) / rows as f64;
                let v = (col as f64 + 1.0) / cols as f64;
                let re = 3.2 * u * v
                    + 0.07 * ((row as f64 + 1.0) * (col as f64 + 2.0)).sin()
                    + 0.02 * ((row + 3 * col + 1) as f64).cos();
                let im = -1.4 * u * v + 0.05 * ((2 * row + col + 1) as f64).cos()
                    - 0.03 * ((row + col + 2) as f64).sin();
                arr[[row, col]] = Complex::new(re, im);
            }
        }
    }

    #[test]
    fn app_error_estimators_are_zero_for_exact_operator_and_inverse() {
        let mut matrix = rlst_dynamic_array2!(f64, [2, 2]);
        matrix[[0, 0]] = 1.0;
        matrix[[0, 1]] = 2.0;
        matrix[[1, 0]] = 0.0;
        matrix[[1, 1]] = 1.0;

        let mut inverse = rlst_dynamic_array2!(f64, [2, 2]);
        inverse[[0, 0]] = 1.0;
        inverse[[0, 1]] = -2.0;
        inverse[[1, 0]] = 0.0;
        inverse[[1, 1]] = 1.0;

        let target = DenseMatrixOperator::new(copy_array2(&matrix));
        let mut rsrs_like = ToggleInverseOperator::new(matrix, inverse);

        let left_err = app_error_left(&target, &rsrs_like, 8, false, 1);
        let right_err = app_error_right(&target, &rsrs_like, 8, false, 2);
        let left_err: f64 = NumCast::from(left_err).unwrap();
        let right_err: f64 = NumCast::from(right_err).unwrap();

        assert!(left_err <= 1.0e-12, "left error too large: {left_err}");
        assert!(right_err <= 1.0e-12, "right error too large: {right_err}");

        rsrs_like.inv(true);
        let left_inv_err = app_error_left(&target, &rsrs_like, 8, true, 3);
        let right_inv_err = app_error_right(&target, &rsrs_like, 8, true, 4);
        let left_inv_err: f64 = NumCast::from(left_inv_err).unwrap();
        let right_inv_err: f64 = NumCast::from(right_inv_err).unwrap();

        assert!(
            left_inv_err <= 1.0e-12,
            "left inverse error too large: {left_inv_err}"
        );
        assert!(
            right_inv_err <= 1.0e-12,
            "right inverse error too large: {right_inv_err}"
        );

        rsrs_like.inv(false);
        let exact_target_fro = (6.0f64).sqrt();
        let (rel_fro_error, norm_fro_target) =
            frobenius_diff_and_reference_norm(&target, &rsrs_like, 1024, 5);
        let rel_fro_error: f64 = NumCast::from(rel_fro_error).unwrap();
        let norm_fro_target: f64 = NumCast::from(norm_fro_target).unwrap();
        let target_fro_rel_err = (norm_fro_target - exact_target_fro).abs() / exact_target_fro;

        assert!(
            rel_fro_error <= 1.0e-12,
            "forward Frobenius error too large: {rel_fro_error}"
        );
        assert!(
            target_fro_rel_err <= 0.12,
            "unexpected forward Frobenius norm estimate: estimated={norm_fro_target}, exact={exact_target_fro}, rel_err={target_fro_rel_err}"
        );

        rsrs_like.inv(true);
        let product = rsrs_like.r().product(target.r());
        let id_op = IdOperator::new(target.domain(), target.range());
        let (rel_fro_inverse_error, norm_fro_identity) =
            frobenius_diff_and_reference_norm(&id_op, &product, 1024, 6);
        let rel_fro_inverse_error: f64 = NumCast::from(rel_fro_inverse_error).unwrap();
        let norm_fro_identity: f64 = NumCast::from(norm_fro_identity).unwrap();
        let exact_identity_fro = (2.0f64).sqrt();
        let identity_fro_rel_err =
            (norm_fro_identity - exact_identity_fro).abs() / exact_identity_fro;

        assert!(
            rel_fro_inverse_error <= 1.0e-12,
            "inverse Frobenius error too large: {rel_fro_inverse_error}"
        );
        assert!(
            identity_fro_rel_err <= 0.12,
            "unexpected identity Frobenius norm estimate: estimated={norm_fro_identity}, exact={exact_identity_fro}, rel_err={identity_fro_rel_err}"
        );
    }

    #[test]
    fn normal_operator_matches_dense_adjoint_product() {
        let mut matrix = rlst_dynamic_array2!(Complex<f64>, [2, 2]);
        matrix[[0, 0]] = Complex::new(1.0, 0.5);
        matrix[[0, 1]] = Complex::new(-0.25, 1.0);
        matrix[[1, 0]] = Complex::new(0.75, -0.5);
        matrix[[1, 1]] = Complex::new(2.0, 0.25);

        let op = DenseMatrixOperator::new(copy_array2(&matrix));
        let normal = NormalOperator::new(op, false);
        let x = vec![Complex::new(1.0, -0.5), Complex::new(-0.25, 0.75)];

        let actual = apply_operator(&normal, &x, TransMode::NoTrans);
        let tmp = apply_operator(
            &DenseMatrixOperator::new(copy_array2(&matrix)),
            &x,
            TransMode::NoTrans,
        );
        let tmp_conj: Vec<_> = tmp.iter().map(|value| value.conj()).collect();
        let second = apply_operator(
            &DenseMatrixOperator::new(matrix),
            &tmp_conj,
            TransMode::Trans,
        );
        let expected: Vec<_> = second.iter().map(|value| value.conj()).collect();

        let err = rel_l2_error(&actual, &expected);
        assert!(err <= 1.0e-12, "normal operator error too large: {err}");
    }

    #[test]
    fn randomized_frobenius_estimator_matches_exact_dense_norms_complex() {
        let mut matrix = rlst_dynamic_array2!(Complex<f64>, [10, 10]);
        populate_complex_test_matrix(&mut matrix);

        let zero = rlst_dynamic_array2!(Complex<f64>, [10, 10]);
        let target = DenseMatrixOperator::new(copy_array2(&matrix));
        let zero_op = DenseMatrixOperator::new(zero);

        let (rel_fro_error, estimated_fro_norm) =
            frobenius_diff_and_reference_norm(&target, &zero_op, 1024, 7);
        let rel_fro_error: f64 = NumCast::from(rel_fro_error).unwrap();
        let estimated_fro_norm: f64 = NumCast::from(estimated_fro_norm).unwrap();
        let exact_fro_norm = exact_frobenius_norm_complex(&matrix);

        let norm_rel_err = (estimated_fro_norm - exact_fro_norm).abs() / exact_fro_norm;

        assert!(
            norm_rel_err <= 0.12,
            "estimated complex Frobenius norm too far from exact: estimated={estimated_fro_norm}, exact={exact_fro_norm}, rel_err={norm_rel_err}"
        );
        assert!(
            (rel_fro_error - 1.0).abs() <= 0.12,
            "relative complex Frobenius error should be close to 1 for zero approximation: got {rel_fro_error}"
        );
    }
}
