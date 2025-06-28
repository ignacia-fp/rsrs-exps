//use super::structured_operator::{
//    Attr, StructuredOperator, StructuredOperatorImpl, StructuredOperatorInterface,
//};
use bempp_rsrs::rsrs::{
    //rsrs_factors::{LocalFrom, RsrsFactors, RsrsOperator},
    sketch::SamplingSpace,
};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};

pub fn solve_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
    Space: SamplingSpace<F = Item> + IndexableSpace + rlst::InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    target_op: &OpImpl,
    rhs: &Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>>,
    tol: <Item as rlst::RlstScalar>::Real,
) -> usize
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationsData<Item>: GivensRotations<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let dim = target_op.domain().dimension();
    //let rhs = target_op.get_rhs();
    let mut residuals = Vec::<<Item as rlst::RlstScalar>::Real>::new();
    let gmres = GmresIteration::new(target_op.r(), rhs.r(), dim)
        .set_callable(|_, res| {
            residuals.push(res);
            println!("res: {}, {:?}", residuals.len(), res);
        })
        .set_tol(tol)
        .set_max_iter(150);
    let (_sol, _res) = gmres.run();

    residuals.len()
}

/*
pub fn solve_prec_system_so<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
    Space: SamplingSpace<F = Item> + IndexableSpace + rlst::InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space> + Clone,
>(
    target_op: &OpImpl,
    rsrs_operator: &OpImpl,
    mut rhs: Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>>,
    tol: <Item as rlst::RlstScalar>::Real,
) -> usize
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationsData<Item>: GivensRotations<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace
{
    let dim = target_op.domain().dimension();
    //let mut rhs: Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>> = target_op.get_rhs();
    //let mut rsrs_operator = RsrsOperator::from_local(rsrs_factors);
    //rsrs_operator.set_inv(true);
    rhs = rsrs_operator.apply(rhs, TransMode::NoTrans);

    let op = rsrs_operator.clone().product(target_op.clone());

    let mut residuals = Vec::<<Item as rlst::RlstScalar>::Real>::new();
    let gmres = GmresIteration::new(op.r(), rhs.r(), dim)
        .set_callable(|_, res| {
            residuals.push(res);
        })
        .set_tol(tol)
        .set_max_iter(100);
    let (_sol, _res) = gmres.run();

    residuals.len()
}
*/
