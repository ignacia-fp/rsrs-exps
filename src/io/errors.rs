use bempp_rsrs::rsrs::rsrs_factors::Inv;
use bempp_rsrs::rsrs::rsrs_factors::RsrsFactors;
use bempp_rsrs::rsrs::rsrs_factors::{
    CommutativeFactors, Factor, FactorOperations, FactorType, IdFactor, LuFactor, MulOptions,
};
use bempp_rsrs::rsrs::sketch::{SampleType, SamplingSpace};
use bempp_rsrs::utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction};
use mpi::traits::Communicator;
use mpi::traits::Equivalence;
use num::NumCast;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
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
>(pub Op1);

impl<
        Item: RlstScalar + RandScalar,
        Space: SamplingSpace<F = Item> + IndexableSpace,
        Op1: OperatorBase<Domain = Space, Range = Space>,
    > std::fmt::Debug for NormalOperator<Item, Space, Op1>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim_1 = self.0.domain().dimension();
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
        self.0.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.0.range().clone()
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
        self.0
            .apply_extended(alpha, x.r(), beta, y_aux_1.r_mut(), TransMode::NoTrans);
        let y_aux_1_conj = self.domain().conj_vec(&y_aux_1);
        self.0.apply_extended(
            alpha,
            y_aux_1_conj,
            beta,
            y_aux_2.r_mut(),
            TransMode::NoTrans,
        );
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

pub fn app_error_right<
    'a,
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_op: &OpImpl2,
    sample_size: usize,
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
    let trans_mode = TransMode::Trans;

    let mut sample_frame = VectorFrame::default();
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let space = target_op.range();

    for _ in 0..sample_size {
        let mut sample_vec = SamplingSpace::zero(space.clone());
        space.sampling(&mut sample_vec, &mut rng, SampleType::StandardNormal);
        sample_frame.push(sample_vec);
    }

    let mut mod_sample_frame = mul_op(rsrs_op, &sample_frame, trans_mode);

    let max_err = if inv {
        mod_sample_frame = mul_op(target_op, &mod_sample_frame, TransMode::NoTrans);
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
        sample_frame = mul_op(target_op, &sample_frame, TransMode::NoTrans);
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
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + MatrixQr,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    target_op: &OpImpl,
    rsrs_op: &OpImpl2,
    sample_size: usize,
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
    let trans_mode = TransMode::NoTrans;

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
    let app_err_left = app_error_left(target_op, rsrs_operator, sample_size, inv);

    let app_err_right = app_error_right(target_op, rsrs_operator, sample_size, inv);

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
    /*let mul_type_left = FactorMulType {
        side: Side::Left,
        factor_type: FactorType::F,
        right_trans: false,
    };
    let mul_type_right = FactorMulType {
        side: Side::Right,
        factor_type: FactorType::S,
        right_trans: false,
    };*/

    let factor_options_left = MulOptions {
        inv: true,
        trans: false,
        side: Side::Left,
        factor_type: FactorType::F,
        t_trans: false,
    };

    let factor_options_right = MulOptions {
        inv: true,
        trans: false,
        side: Side::Right,
        factor_type: FactorType::S,
        t_trans: false,
    };

    let errors: Vec<_> = factors
        .par_iter()
        .map(|factor| {
            let mut target_arr = target_arr.lock().unwrap();
            match factor {
                Factor::Lu(lu_factor) => {
                    let (arr_rt, arr_tr) = box_errors_lu(lu_factor, &mut target_arr);
                    lu_factor.mul(&mut target_arr, &factor_options_left);
                    lu_factor.mul(&mut target_arr, &factor_options_right);
                    let (arr_rt_ae, arr_tr_ae) = box_errors_lu(lu_factor, &mut target_arr);
                    let rel_errs: Errors = (arr_rt_ae / arr_rt, arr_tr_ae / arr_tr);
                    rel_errs
                }
                Factor::Id(id_factor) => {
                    let (arr_rf, arr_fr) = box_errors_id(id_factor, &mut target_arr);
                    id_factor.mul(&mut target_arr, &factor_options_left);
                    id_factor.mul(&mut target_arr, &factor_options_right);
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

                    let options = MulOptions {
                        inv: false,
                        trans: false,
                        side: Side::Left,
                        factor_type: FactorType::F,
                        t_trans: false,
                    };
                    diag_box_factor.arr.mul(&mut app_dbox, Side::Left, &options);

                    let mut res: DynamicArray<Item, 2> = empty_array();
                    res.fill_from_resize(exact_diag_box.r() - app_dbox.r());

                    let err_diag = spectral_norm_estimator(&res, 10).unwrap()
                        / spectral_norm_estimator(&exact_diag_box, 10).unwrap();

                    let mut app_inv_dbox = rlst_dynamic_array2!(Item, shape);
                    app_inv_dbox.set_identity();

                    let options = MulOptions {
                        inv: true,
                        trans: false,
                        side: Side::Left,
                        factor_type: FactorType::F,
                        t_trans: false,
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
) -> (Vec<ErrorStats>, Vec<Vec<ErrorStats>>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    let errors: Vec<(Vec<Errors>, Vec<Vec<Errors>>)> = (0..rsrs_factors.num_levels)
        .map(|level_it| {
            let factors = &rsrs_factors.id_factors[level_it];
            let id_errors = commutative_factors_errors(&factors, target_arr);

            let lu_errors = rsrs_factors.lu_factors[level_it]
                .iter()
                .map(|lu_batch| commutative_factors_errors(&lu_batch, target_arr))
                .collect(); // no .flatten()

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
    let mut lu_stats = Vec::new(); // This will be Vec<Vec<(mu1, mu2, std1, std2)>>

    errors
        .iter()
        .for_each(|(id_level_errors, lu_level_errors)| {
            if !id_level_errors.is_empty() {
                // Stats for ID errors at this level
                id_stats.push(stats(id_level_errors.to_vec()));

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
    id_error_stats: Vec<(f64, f64, f64, f64)>,
    lu_error_stats: Vec<(f64, f64, f64, f64)>,
    diag_error_stats: (f64, f64),
}

pub fn get_boxes_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse + MatrixId + MatrixLu + MatrixQr,
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

    // Print ID stats per level
    id_error_stats.iter().enumerate().for_each(|(level, s)| {
        println!(
            "Errors ID, level {} : ({} +/- {}, {} +/- {})",
            level, s.0, s.2, s.1, s.3
        );
    });

    // Print LU stats per batch in each level
    for (level, level_stats) in lu_error_stats.iter().enumerate() {
        for (batch, s) in level_stats.iter().enumerate() {
            println!(
                "Errors LU, level {}, batch {} : ({} +/- {}, {} +/- {})",
                level, batch, s.0, s.2, s.1, s.3
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
    let lu_error_stats_flat: Vec<ErrorStats> = lu_error_stats.into_iter().flatten().collect();

    let error_stats_container = ErrorStatsContainer {
        id_error_stats,
        lu_error_stats: lu_error_stats_flat,
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
