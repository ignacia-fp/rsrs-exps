use super::structured_operator::{
    Attr, StructuredOperator, StructuredOperatorImpl, StructuredOperatorOperator,
};
use bempp_rsrs::rsrs::rsrs_factors::{LocalFrom, RsrsFactors, RsrsOperator};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};

pub fn solve_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
>(
    target_op: StructuredOperatorOperator<'a, Item, StructuredOperator>,
    tol: <Item as rlst::RlstScalar>::Real,
) -> usize
where
    StructuredOperator: StructuredOperatorImpl<Item, Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationsData<Item>: GivensRotations<Item>,
{
    let dim = target_op.domain().dimension();
    let rhs = target_op.get_rhs();
    let mut residuals = Vec::<<Item as rlst::RlstScalar>::Real>::new();
    let gmres = GmresIteration::new(target_op.r(), rhs.r(), dim)
        .set_callable(|_, res| {
            residuals.push(res);
        })
        .set_tol(tol).set_max_iter(150);
    let (_sol, _res) = gmres.run();

    residuals.len()
}

pub fn solve_prec_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
>(
    target_op: StructuredOperatorOperator<'a, Item, StructuredOperator>,
    rsrs_factors: &mut RsrsFactors<Item>,
    tol: <Item as rlst::RlstScalar>::Real,
) -> usize
where
    StructuredOperator: StructuredOperatorImpl<Item, Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationsData<Item>: GivensRotations<Item>,
{
    let dim = target_op.domain().dimension();
    let mut rhs = target_op.get_rhs();
    let mut rsrs_operator = RsrsOperator::from_local(rsrs_factors);
    rsrs_operator.set_inv(true);
    rhs = rsrs_operator.apply(rhs, TransMode::NoTrans);

    let op = rsrs_operator.product(target_op);

    let mut residuals = Vec::<<Item as rlst::RlstScalar>::Real>::new();
    let gmres = GmresIteration::new(op.r(), rhs.r(), dim)
        .set_callable(|_, res| {
            residuals.push(res);
        })
        .set_tol(tol).set_max_iter(100);
    let (_sol, _res) = gmres.run();

    residuals.len()
}
