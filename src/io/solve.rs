use bempp_rsrs::rsrs::sketch::SamplingSpace;
use rlst::operator::RlstOperatorReference;
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
use rlst::operator::ConcreteElementContainerRef;

pub fn solve_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
    Space: SamplingSpace<F = Item> + IndexableSpace + rlst::InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    target_op: &OpImpl,
    rhs: &Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>>,
    tol: <Item as rlst::RlstScalar>::Real,
) -> (
    Vec<<Item as rlst::RlstScalar>::Real>,
    <Item as RlstScalar>::Real,
)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationsData<Item>: GivensRotations<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let dim = target_op.domain().dimension();
    let mut residuals = Vec::<<Item as rlst::RlstScalar>::Real>::new();
    let gmres: GmresIteration<
        '_,                                                         // Lifetime
        Space,                                                      // Vector space
        RlstOperatorReference<'_, OpImpl>,                          // Main operator
        RlstOperatorReference<'_, OpImpl>,                          // Identity preconditioner
        ConcreteElementContainerRef<'_, <Space as LinearSpace>::E>, // RHS container
    > = GmresIteration::new(target_op.r(), rhs.r(), dim)
        .set_callable(|_, res| {
            residuals.push(res);
            println!("res: {}, {:?}", residuals.len(), res);
        })
        .set_tol(tol)
        .set_max_iter(500);
    let (sol, _res) = gmres.run();

    let mut diff = rhs.duplicate();
    diff -= target_op.apply(sol, TransMode::NoTrans);

    let rel_norm = diff.norm() / rhs.norm();
    println!("Rel norm: {}", rel_norm);

    (residuals, rel_norm)
}

pub fn solve_prec_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
    Space: SamplingSpace<F = Item> + IndexableSpace + rlst::InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space>,
>(
    target_op: &OpImpl,
    rsrs_operator: &OpImpl2,
    rhs: &Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>>,
    tol: <Item as rlst::RlstScalar>::Real,
) -> (
    Vec<<Item as rlst::RlstScalar>::Real>,
    <Item as RlstScalar>::Real,
)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationsData<Item>: GivensRotations<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let dim = target_op.domain().dimension();

    //let rhs_prec = rsrs_operator.apply(rhs.r(), TransMode::NoTrans);
    //let prec_operator = rsrs_operator.r().product(target_op.r());
    let mut residuals = Vec::<<Item as rlst::RlstScalar>::Real>::new();
    let gmres: GmresIteration<
        '_,                                                         // lifetime
        Space,                                                      // your vector space
        RlstOperatorReference<'_, OpImpl>,                          // type of target_op.r()
        RlstOperatorReference<'_, OpImpl2>,                         // type of rsrs_operator.r()
        ConcreteElementContainerRef<'_, <Space as LinearSpace>::E>, // container for rhs.r()
    > = GmresIteration::new(target_op.r(), rhs.r(), dim)
        .set_callable(|_, res| {
            residuals.push(res);
            println!("res: {}, {:?}", residuals.len(), res);
        })
        .set_tol(tol)
        .set_max_iter(500)
        .set_preconditioner(rsrs_operator.r());
    let (sol, _res) = gmres.run();

    let mut diff = rhs.duplicate();
    diff -= target_op.apply(sol, TransMode::NoTrans);

    let rel_norm = diff.norm() / rhs.norm();
    println!("Rel norm: {}", rel_norm);

    (residuals, rel_norm)
}
