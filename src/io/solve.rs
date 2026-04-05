use bempp_rsrs::rsrs::sketch::SamplingSpace;
use rlst::operator::ConcreteElementContainerRef;
use rlst::operator::RlstOperatorReference;
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
use std::time::Instant;

const GMRES_RESTART: usize = 20;
const GMRES_MAX_ITER: usize = 5000;

/// Solve a system for multiple RHS vectors
pub fn solve_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
    Space: SamplingSpace<F = Item> + IndexableSpace + rlst::InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    target_op: &OpImpl,
    rhs_list: &[Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>>],
    tol: f64,
) -> (
    Vec<Vec<<Item as RlstScalar>::Real>>, // GMRES residuals
    Vec<<Item as RlstScalar>::Real>,      // relative norm
    Vec<Vec<Item>>, //Vec<Element<ConcreteElementContainer<<Space as LinearSpace>::E>>>, // solution
)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationData<Item>: rlst::GivensRotation<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let dim = target_op.r().domain().dimension();
    let mut res_vec = Vec::with_capacity(rhs_list.len());
    let mut res_norm_vec = Vec::with_capacity(rhs_list.len());
    let mut sols_vec = Vec::with_capacity(rhs_list.len());
    println!(
        "[rsrs-exps][solve] starting unpreconditioned GMRES: rhs_count={}, dim={}, tol={:.3e}, restart={}, max_iter={}",
        rhs_list.len(),
        dim,
        tol,
        GMRES_RESTART,
        GMRES_MAX_ITER
    );

    for (rhs_idx, rhs) in rhs_list.iter().enumerate() {
        let mut residuals = Vec::new();
        let stage = Instant::now();
        println!(
            "[rsrs-exps][solve] rhs {}/{}: unpreconditioned GMRES start",
            rhs_idx + 1,
            rhs_list.len()
        );

        let gmres: GmresIteration<
            '_,                                                         // Lifetime
            Space,                                                      // Vector space
            RlstOperatorReference<'_, OpImpl>,                          // Main operator
            ConcreteElementContainerRef<'_, <Space as LinearSpace>::E>, // RHS container
            RlstOperatorReference<'_, OpImpl>,                          // Identity preconditioner
        > = GmresIteration::new(target_op.r(), rhs.r(), dim)
            .set_callable(|_, res| {
                residuals.push(res);
                println!("res: {}, {:?}", residuals.len(), res);
            })
            .set_tol(tol)
            .set_restart(GMRES_RESTART)
            .set_max_iter(GMRES_MAX_ITER);

        let (sol, _res) = gmres.run();

        let mut diff = rhs.duplicate();
        diff -= target_op.apply(sol.r(), TransMode::NoTrans);
        let rel_norm = diff.norm() / rhs.norm();
        println!("Rel norm: {}", rel_norm);
        println!(
            "[rsrs-exps][solve] rhs {}/{}: unpreconditioned GMRES done in {:.3}s with {} recorded residual(s)",
            rhs_idx + 1,
            rhs_list.len(),
            stage.elapsed().as_secs_f64(),
            residuals.len()
        );

        res_vec.push(residuals);
        res_norm_vec.push(rel_norm);
        let mut sol_vec = rlst_dynamic_array2!(Item, [1, dim]);
        target_op
            .r()
            .domain()
            .fill_array(&sol, &mut sol_vec, 0, TransMode::NoTrans);
        sols_vec.push(sol_vec.data().to_vec());
    }

    println!("[rsrs-exps][solve] unpreconditioned GMRES finished");
    (res_vec, res_norm_vec, sols_vec)
}

/// Solve a preconditioned system for multiple RHS vectors
pub fn solve_prec_system<
    'a,
    Item: RlstScalar + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr + MatrixInverse,
    Space: SamplingSpace<F = Item> + IndexableSpace + rlst::InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space>,
>(
    target_op: &OpImpl,
    rsrs_operator: &OpImpl2,
    rhs_list: &[Element<rlst::operator::ConcreteElementContainer<<Space as LinearSpace>::E>>],
    tol: f64,
) -> (
    Vec<Vec<<Item as RlstScalar>::Real>>, // GMRES residuals
    Vec<<Item as RlstScalar>::Real>,      // relative norm
    Vec<Vec<Item>>, //Vec<Element<ConcreteElementContainer<<Space as LinearSpace>::E>>>, // solution
)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    GivensRotationData<Item>: rlst::GivensRotation<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: rlst::InnerProductSpace,
{
    let dim = target_op.domain().dimension();
    let mut res_vec = Vec::with_capacity(rhs_list.len());
    let mut res_norm_vec = Vec::with_capacity(rhs_list.len());
    let mut sols_vec = Vec::with_capacity(rhs_list.len());
    println!(
        "[rsrs-exps][solve] starting preconditioned GMRES: rhs_count={}, dim={}, tol={:.3e}, restart={}, max_iter={}",
        rhs_list.len(),
        dim,
        tol,
        GMRES_RESTART,
        GMRES_MAX_ITER
    );

    for (rhs_idx, rhs) in rhs_list.iter().enumerate() {
        let mut residuals = Vec::new();
        let stage = Instant::now();
        println!(
            "[rsrs-exps][solve] rhs {}/{}: preconditioned GMRES start",
            rhs_idx + 1,
            rhs_list.len()
        );

        let gmres: GmresIteration<
            '_,                                                         // lifetime
            Space,                                                      // vector space
            RlstOperatorReference<'_, OpImpl>,                          // target_op
            ConcreteElementContainerRef<'_, <Space as LinearSpace>::E>, // RHS container
            RlstOperatorReference<'_, OpImpl2>,                         // preconditioner
        > = GmresIteration::new(target_op.r(), rhs.r(), dim)
            .set_callable(|_, res| {
                residuals.push(res);
                println!("res: {}, {:?}", residuals.len(), res);
            })
            .set_tol(tol)
            .set_max_iter(GMRES_MAX_ITER)
            .set_restart(GMRES_RESTART)
            .set_preconditioner(rsrs_operator.r());

        let (sol, _res) = gmres.run();

        let mut diff = rhs.duplicate();
        diff -= target_op.apply(sol.r(), TransMode::NoTrans);
        let rel_norm = diff.norm() / rhs.norm();
        println!("Rel norm: {}", rel_norm);
        println!(
            "[rsrs-exps][solve] rhs {}/{}: preconditioned GMRES done in {:.3}s with {} recorded residual(s)",
            rhs_idx + 1,
            rhs_list.len(),
            stage.elapsed().as_secs_f64(),
            residuals.len()
        );

        res_vec.push(residuals);
        res_norm_vec.push(rel_norm);
        let mut sol_vec = rlst_dynamic_array2!(Item, [1, dim]);
        target_op
            .r()
            .domain()
            .fill_array(&sol, &mut sol_vec, 0, TransMode::NoTrans);
        sols_vec.push(sol_vec.data().to_vec());
    }

    println!("[rsrs-exps][solve] preconditioned GMRES finished");
    (res_vec, res_norm_vec, sols_vec)
}
