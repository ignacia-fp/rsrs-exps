use bempp_octree::Octree;
use bempp_rsrs::{
    rsrs::{
        args::{RankPicking, RsrsArgs, RsrsOptions, Symmetry},
        rsrs_cycle::Rsrs,
        rsrs_factors::{
            base_factors::BaseFactorOptions,
            commutative_factors::{
                Factor, FactorOperations, FactorType, MulOptions, MultiLevelIdFactors,
            },
            null_and_extract::PivotMethod,
            rsrs_operator::{FactType, Inv, LocalFromSpaces, RsrsFactorsImpl, RsrsOperator},
        },
        sketch::{RandScalar, SampleType, SamplingSpace, Shift},
    },
    utils::linear_algebra::{BlockExtractionMethod, NullMethod},
};
use mpi::topology::SimpleCommunicator;
use num::{Complex, FromPrimitive, NumCast};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{
    dense::linalg::{naupd::NonSymmetricArnoldiUpdate, neupd::NonSymmetricArnoldiExtract},
    prelude::*,
};
use rsrs_exps::{
    io::{
        errors::{
            frobenius_diff_and_reference_norm, frobenius_diff_and_reference_norm_with_mode,
            rsrs_error_estimator, DiffOperator, NormalOperator,
        },
        structured_operator::{
            get_bempp_points, Assembler, GeometryType, LocalFrom, StructuredOperator,
            StructuredOperatorImpl, StructuredOperatorInterface, StructuredOperatorParams,
        },
        structured_operators_types::StructuredOperatorType,
    },
    test_prep::Precision,
};

type ComplexItem = Complex<f64>;
type RealItem = f64;

const SMALL_MESH_WIDTH: f64 = 0.05;
const SMALL_FIXED_RANK: usize = 16;
const SMALL_MAX_LEAF_POINTS: usize = 8;
const EIGS_TOL: f64 = 1.0e-10;
const POWER_ITERATIONS: usize = 20;
const FROBENIUS_ESTIMATION_SAMPLES: usize = 20;
const ADJOINT_CHECK_SAMPLES: usize = 8;
const APP_ERR_LEFT_SEED: u64 = 0xA001_E110_0000_0001;
const APP_ERR_RIGHT_SEED: u64 = 0xA001_E110_0000_0002;
const ADJOINT_CHECK_SEED: u64 = 0xA31D_0C11_5EED_1234;
const INV_ADJOINT_CHECK_SEED: u64 = 0xA31D_0C11_5EED_5678;
const FROB_FORWARD_SEED: u64 = 0xF20B_0000_0000_0001;
const POWER_DIFF_SEED: u64 = 0x5000_0000_0000_0001;
const POWER_STRUCTURED_SEED: u64 = 0x5000_0000_0000_0002;
const SOLVE_CHECK_SEED: u64 = 0x501E_0000_0000_0001;

#[derive(Debug)]
struct RsrsMetrics {
    label: &'static str,
    dim: usize,
    rel_err_fro: f64,
    rel_adjoint_route: f64,
    rel_conj_adjoint_route: f64,
    rel_apply_adjoint: f64,
    rel_inv_adjoint_route: f64,
    rel_inv_conj_adjoint_route: f64,
    rel_inv_apply_adjoint: f64,
    rel_inverse_residual_fro: f64,
    norm_2_error: f64,
    norm_fro_error: f64,
    norm_fro_error_transpose: f64,
    adjoint_consistency_error: f64,
    adjoint_consistency_error_inv: f64,
    solve_error: f64,
    exact_manual_conjtrans: f64,
    exact_manual_conjtrans_supported: bool,
    rsrs_manual_conjtrans: f64,
    rsrs_inv_manual_conjtrans: f64,
}

#[derive(Debug)]
struct ComplexFactorRouteSummary {
    case_label: &'static str,
    family: &'static str,
    location: String,
    worst_transpose: f64,
    worst_transpose_mode: String,
    worst_manual_adjoint: f64,
    worst_manual_adjoint_mode: String,
}

#[derive(Debug)]
struct RealFactorRouteSummary {
    case_label: &'static str,
    family: &'static str,
    location: String,
    worst_transpose: f64,
    worst_transpose_mode: String,
}

fn square_leaf_points(_rank: usize) -> usize {
    SMALL_MAX_LEAF_POINTS
}

fn basis_vector_complex(dim: usize, index: usize) -> Vec<ComplexItem> {
    let mut basis = vec![ComplexItem::new(0.0, 0.0); dim];
    basis[index] = ComplexItem::new(1.0, 0.0);
    basis
}

fn basis_vector_real(dim: usize, index: usize) -> Vec<RealItem> {
    let mut basis = vec![0.0; dim];
    basis[index] = 1.0;
    basis
}

fn apply_operator_complex<Op>(
    op: &Op,
    input: &[ComplexItem],
    trans_mode: TransMode,
) -> Vec<ComplexItem>
where
    Op: AsApply<Domain = ArrayVectorSpace<ComplexItem>, Range = ArrayVectorSpace<ComplexItem>>,
{
    let mut x = zero_element(op.domain());
    x.imp_mut().fill_inplace_raw(input);
    let y = op.apply(x.r(), trans_mode);
    y.view().iter().collect()
}

fn apply_operator_real<Op>(op: &Op, input: &[RealItem], trans_mode: TransMode) -> Vec<RealItem>
where
    Op: AsApply<Domain = ArrayVectorSpace<RealItem>, Range = ArrayVectorSpace<RealItem>>,
{
    let mut x = zero_element(op.domain());
    x.imp_mut().fill_inplace_raw(input);
    let y = op.apply(x.r(), trans_mode);
    y.view().iter().collect()
}

fn assemble_matrix_complex<Op>(
    op: &Op,
    dim: usize,
    trans_mode: TransMode,
) -> DynamicArray<ComplexItem, 2>
where
    Op: AsApply<Domain = ArrayVectorSpace<ComplexItem>, Range = ArrayVectorSpace<ComplexItem>>,
{
    let mut matrix = rlst_dynamic_array2!(ComplexItem, [dim, dim]);
    for col in 0..dim {
        let basis = basis_vector_complex(dim, col);
        let column = apply_operator_complex(op, &basis, trans_mode);
        for (row, value) in column.into_iter().enumerate() {
            matrix[[row, col]] = value;
        }
    }
    matrix
}

fn assemble_matrix_real<Op>(op: &Op, dim: usize, trans_mode: TransMode) -> DynamicArray<RealItem, 2>
where
    Op: AsApply<Domain = ArrayVectorSpace<RealItem>, Range = ArrayVectorSpace<RealItem>>,
{
    let mut matrix = rlst_dynamic_array2!(RealItem, [dim, dim]);
    for col in 0..dim {
        let basis = basis_vector_real(dim, col);
        let column = apply_operator_real(op, &basis, trans_mode);
        for (row, value) in column.into_iter().enumerate() {
            matrix[[row, col]] = value;
        }
    }
    matrix
}

fn adjoint(matrix: &DynamicArray<ComplexItem, 2>) -> DynamicArray<ComplexItem, 2> {
    let shape = matrix.shape();
    let mut out = rlst_dynamic_array2!(ComplexItem, [shape[1], shape[0]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[col, row]] = matrix[[row, col]].conj();
        }
    }
    out
}

fn transpose_complex(matrix: &DynamicArray<ComplexItem, 2>) -> DynamicArray<ComplexItem, 2> {
    let shape = matrix.shape();
    let mut out = rlst_dynamic_array2!(ComplexItem, [shape[1], shape[0]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[col, row]] = matrix[[row, col]];
        }
    }
    out
}

fn transpose_real(matrix: &DynamicArray<RealItem, 2>) -> DynamicArray<RealItem, 2> {
    let shape = matrix.shape();
    let mut out = rlst_dynamic_array2!(RealItem, [shape[1], shape[0]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[col, row]] = matrix[[row, col]];
        }
    }
    out
}

fn diff_complex(
    lhs: &DynamicArray<ComplexItem, 2>,
    rhs: &DynamicArray<ComplexItem, 2>,
) -> DynamicArray<ComplexItem, 2> {
    let shape = lhs.shape();
    let mut out = rlst_dynamic_array2!(ComplexItem, [shape[0], shape[1]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[row, col]] = lhs[[row, col]] - rhs[[row, col]];
        }
    }
    out
}

fn diff_real(
    lhs: &DynamicArray<RealItem, 2>,
    rhs: &DynamicArray<RealItem, 2>,
) -> DynamicArray<RealItem, 2> {
    let shape = lhs.shape();
    let mut out = rlst_dynamic_array2!(RealItem, [shape[0], shape[1]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[row, col]] = lhs[[row, col]] - rhs[[row, col]];
        }
    }
    out
}

fn frobenius_norm_complex(matrix: &DynamicArray<ComplexItem, 2>) -> f64 {
    matrix
        .r()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn frobenius_norm_real(matrix: &DynamicArray<RealItem, 2>) -> f64 {
    matrix
        .r()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn complex_test_vector(dim: usize, offset: f64) -> Vec<ComplexItem> {
    (0..dim)
        .map(|idx| {
            let t = idx as f64 + 1.0 + offset;
            ComplexItem::new((0.31 * t).sin() + 0.2 * (0.13 * t).cos(), (0.17 * t).cos())
        })
        .collect()
}

fn real_test_vector(dim: usize, offset: f64) -> Vec<RealItem> {
    (0..dim)
        .map(|idx| {
            let t = idx as f64 + 1.0 + offset;
            (0.29 * t).sin() + 0.25 * (0.11 * t).cos()
        })
        .collect()
}

fn complex_inner(lhs: &[ComplexItem], rhs: &[ComplexItem]) -> ComplexItem {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l.conj() * *r).sum()
}

fn real_inner(lhs: &[RealItem], rhs: &[RealItem]) -> RealItem {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}

fn relative_complex_adjoint_defect<Op>(op: &Op, dim: usize) -> f64
where
    Op: AsApply<Domain = ArrayVectorSpace<ComplexItem>, Range = ArrayVectorSpace<ComplexItem>>,
{
    let x = complex_test_vector(dim, 0.0);
    let y = complex_test_vector(dim, 0.5);
    let ay = apply_operator_complex(op, &y, TransMode::NoTrans);
    let ahx = apply_operator_complex(op, &x, TransMode::Trans);
    let left = complex_inner(&x, &ay);
    let right = complex_inner(&ahx, &y);
    (left - right).norm() / left.norm().max(right.norm()).max(1.0e-14)
}

fn relative_real_adjoint_defect<Op>(op: &Op, dim: usize) -> f64
where
    Op: AsApply<Domain = ArrayVectorSpace<RealItem>, Range = ArrayVectorSpace<RealItem>>,
{
    let x = real_test_vector(dim, 0.0);
    let y = real_test_vector(dim, 0.5);
    let ay = apply_operator_real(op, &y, TransMode::NoTrans);
    let atx = apply_operator_real(op, &x, TransMode::Trans);
    let left = real_inner(&x, &ay);
    let right = real_inner(&atx, &y);
    (left - right).abs() / left.abs().max(right.abs()).max(1.0e-14)
}

fn relative_manual_conjtrans_identity_complex<Op>(op: &Op, dim: usize) -> f64
where
    Op: AsApply<Domain = ArrayVectorSpace<ComplexItem>, Range = ArrayVectorSpace<ComplexItem>>,
{
    let x = complex_test_vector(dim, 1.25);
    let x_conj = x.iter().map(|value| value.conj()).collect::<Vec<_>>();

    let direct = apply_operator_complex(op, &x, TransMode::ConjTrans);
    let mut manual = apply_operator_complex(op, &x_conj, TransMode::Trans);
    manual.iter_mut().for_each(|value| *value = value.conj());

    let numerator = direct
        .iter()
        .zip(manual.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let direct_norm = direct
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let manual_norm = manual
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();

    numerator / direct_norm.max(manual_norm).max(1.0e-14)
}

fn try_relative_manual_conjtrans_identity_complex<Op>(op: &Op, dim: usize) -> Option<f64>
where
    Op: AsApply<Domain = ArrayVectorSpace<ComplexItem>, Range = ArrayVectorSpace<ComplexItem>>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        relative_manual_conjtrans_identity_complex(op, dim)
    }))
    .ok()
}

fn estimate_normal_operator_norm_sq_like_save<
    Item: RlstScalar + RandScalar,
    Space: bempp_rsrs::rsrs::sketch::SamplingSpace<F = Item> + IndexableSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    operator: &OpImpl,
    _iterations: usize,
    _seed: u64,
) -> <Item as RlstScalar>::Real
where
    StandardNormal: Distribution<<Item as RlstScalar>::Real>,
    Standard: Distribution<<Item as RlstScalar>::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    Item: NonSymmetricArnoldiUpdate + NonSymmetricArnoldiExtract,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: InnerProductSpace,
{
    let tol_eigs = <Item as RlstScalar>::Real::from_f64(EIGS_TOL).unwrap();
    let mut eigs = Eigs::new(operator.r(), tol_eigs, None, None, None);
    let (sigma, _) = eigs.run(None, 1, None, false);
    sigma[0].abs()
}

fn estimate_adjoint_consistency_error_like_save<
    Item: RlstScalar + RandScalar,
    Space: bempp_rsrs::rsrs::sketch::SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    operator: &OpImpl,
    sample_size: usize,
    seed: u64,
) -> <Item as RlstScalar>::Real
where
    StandardNormal: Distribution<<Item as RlstScalar>::Real>,
    Standard: Distribution<<Item as RlstScalar>::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: InnerProductSpace,
{
    let mut max_rel = 0.0f64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..sample_size.max(1) {
        let mut x = <Space as SamplingSpace>::zero(operator.range());
        let mut y = <Space as SamplingSpace>::zero(operator.domain());
        operator
            .range()
            .sampling(&mut x, &mut rng, SampleType::StandardNormal);
        operator
            .domain()
            .sampling(&mut y, &mut rng, SampleType::StandardNormal);

        let ay = operator.apply(y.r(), TransMode::NoTrans);
        let a_star_x = operator.apply(x.r(), TransMode::ConjTrans);

        let lhs = x.inner_product(ay.r());
        let rhs = a_star_x.inner_product(y.r());

        let lhs_abs: f64 = NumCast::from(lhs.abs()).unwrap();
        let rhs_abs: f64 = NumCast::from(rhs.abs()).unwrap();
        let defect_abs: f64 = NumCast::from((lhs - rhs).abs()).unwrap();
        let denom = lhs_abs.max(rhs_abs).max(1.0e-14);
        max_rel = max_rel.max(defect_abs / denom);
    }

    <Item as RlstScalar>::Real::from_f64(max_rel).unwrap()
}

fn matmul_complex(
    lhs: &DynamicArray<ComplexItem, 2>,
    rhs: &DynamicArray<ComplexItem, 2>,
) -> DynamicArray<ComplexItem, 2> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    assert_eq!(lhs_shape[1], rhs_shape[0]);
    let mut out = rlst_dynamic_array2!(ComplexItem, [lhs_shape[0], rhs_shape[1]]);
    for row in 0..lhs_shape[0] {
        for col in 0..rhs_shape[1] {
            let mut acc = ComplexItem::new(0.0, 0.0);
            for inner in 0..lhs_shape[1] {
                acc += lhs[[row, inner]] * rhs[[inner, col]];
            }
            out[[row, col]] = acc;
        }
    }
    out
}

fn matmul_real(
    lhs: &DynamicArray<RealItem, 2>,
    rhs: &DynamicArray<RealItem, 2>,
) -> DynamicArray<RealItem, 2> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    assert_eq!(lhs_shape[1], rhs_shape[0]);
    let mut out = rlst_dynamic_array2!(RealItem, [lhs_shape[0], rhs_shape[1]]);
    for row in 0..lhs_shape[0] {
        for col in 0..rhs_shape[1] {
            let mut acc = 0.0;
            for inner in 0..lhs_shape[1] {
                acc += lhs[[row, inner]] * rhs[[inner, col]];
            }
            out[[row, col]] = acc;
        }
    }
    out
}

fn identity_complex(dim: usize) -> DynamicArray<ComplexItem, 2> {
    let mut out = rlst_dynamic_array2!(ComplexItem, [dim, dim]);
    for idx in 0..dim {
        out[[idx, idx]] = ComplexItem::new(1.0, 0.0);
    }
    out
}

fn identity_real(dim: usize) -> DynamicArray<RealItem, 2> {
    let mut out = rlst_dynamic_array2!(RealItem, [dim, dim]);
    for idx in 0..dim {
        out[[idx, idx]] = 1.0;
    }
    out
}

fn factor_family_label<Item: RlstScalar>(factor: &Factor<Item>) -> &'static str {
    match factor {
        Factor::Id(_) => "id",
        Factor::Lu(_) => "lu",
        Factor::Diag(_) => "diag",
    }
}

fn factor_location<Item: RlstScalar>(
    factor: &Factor<Item>,
    level: Option<usize>,
    batch: usize,
    index: usize,
) -> String {
    match factor {
        Factor::Id(id) => match level {
            Some(level) => format!(
                "level={level} batch={batch} index={index} r={} s={}",
                id.ind_r.len(),
                id.ind_s.len()
            ),
            None => format!(
                "batch={batch} index={index} r={} s={}",
                id.ind_r.len(),
                id.ind_s.len()
            ),
        },
        Factor::Lu(lu) => match level {
            Some(level) => format!(
                "level={level} batch={batch} index={index} r={} t={}",
                lu.ind_r.len(),
                lu.ind_t.len()
            ),
            None => format!(
                "batch={batch} index={index} r={} t={}",
                lu.ind_r.len(),
                lu.ind_t.len()
            ),
        },
        Factor::Diag(diag) => format!("diag index={index} n={}", diag.inds.len()),
    }
}

fn factor_branches<Item: RlstScalar>(factor: &Factor<Item>) -> &'static [FactorType] {
    match factor {
        Factor::Diag(_) => &[FactorType::F],
        Factor::Id(_) | Factor::Lu(_) => &[FactorType::F, FactorType::S],
    }
}

fn side_label(side: Side) -> &'static str {
    match side {
        Side::Left => "Left",
        Side::Right => "Right",
    }
}

fn apply_factor_complex(
    factor: &Factor<ComplexItem>,
    dim: usize,
    side: Side,
    factor_type: FactorType,
    base_options: BaseFactorOptions,
) -> DynamicArray<ComplexItem, 2> {
    let mut target = identity_complex(dim);
    let mut target_view = target.r_mut();
    let options = MulOptions {
        base_options,
        side,
        factor_type,
    };
    match factor {
        Factor::Id(id_factor) => id_factor.mul(&mut target_view, &options),
        Factor::Lu(lu_factor) => lu_factor.mul(&mut target_view, &options),
        Factor::Diag(diag_factor) => diag_factor.mul(&mut target_view, &options),
    }
    target
}

fn apply_factor_real(
    factor: &Factor<RealItem>,
    dim: usize,
    side: Side,
    factor_type: FactorType,
    base_options: BaseFactorOptions,
) -> DynamicArray<RealItem, 2> {
    let mut target = identity_real(dim);
    let mut target_view = target.r_mut();
    let options = MulOptions {
        base_options,
        side,
        factor_type,
    };
    match factor {
        Factor::Id(id_factor) => id_factor.mul(&mut target_view, &options),
        Factor::Lu(lu_factor) => lu_factor.mul(&mut target_view, &options),
        Factor::Diag(diag_factor) => diag_factor.mul(&mut target_view, &options),
    }
    target
}

fn summarize_complex_factor_routes(
    case_label: &'static str,
    factor: &Factor<ComplexItem>,
    dim: usize,
    level: Option<usize>,
    batch: usize,
    index: usize,
) -> ComplexFactorRouteSummary {
    let mut worst_transpose = 0.0;
    let mut worst_transpose_mode = String::new();
    let mut worst_manual_adjoint = 0.0;
    let mut worst_manual_adjoint_mode = String::new();

    for branch in factor_branches(factor) {
        for side in [Side::Left, Side::Right] {
            for inv in [false, true] {
                let no_trans = apply_factor_complex(
                    factor,
                    dim,
                    side,
                    branch.clone(),
                    BaseFactorOptions {
                        inv,
                        trans: TransMode::NoTrans,
                        trans_target: false,
                    },
                );
                let trans = apply_factor_complex(
                    factor,
                    dim,
                    side,
                    branch.clone(),
                    BaseFactorOptions {
                        inv,
                        trans: TransMode::Trans,
                        trans_target: false,
                    },
                );
                let denom = frobenius_norm_complex(&no_trans).max(1.0e-14);
                let rel_transpose =
                    frobenius_norm_complex(&diff_complex(&trans, &transpose_complex(&no_trans)))
                        / denom;
                let mut manual_adjoint = empty_array();
                manual_adjoint.fill_from_resize(trans.r());
                manual_adjoint
                    .data_mut()
                    .iter_mut()
                    .for_each(|value| *value = value.conj());
                let rel_manual_adjoint =
                    frobenius_norm_complex(&diff_complex(&manual_adjoint, &adjoint(&no_trans)))
                        / denom;

                let mode_label = format!("side={} branch={:?} inv={inv}", side_label(side), branch);

                if rel_transpose > worst_transpose {
                    worst_transpose = rel_transpose;
                    worst_transpose_mode = mode_label.clone();
                }
                if rel_manual_adjoint > worst_manual_adjoint {
                    worst_manual_adjoint = rel_manual_adjoint;
                    worst_manual_adjoint_mode = mode_label;
                }
            }
        }
    }

    ComplexFactorRouteSummary {
        case_label,
        family: factor_family_label(factor),
        location: factor_location(factor, level, batch, index),
        worst_transpose,
        worst_transpose_mode,
        worst_manual_adjoint,
        worst_manual_adjoint_mode,
    }
}

fn summarize_real_factor_routes(
    case_label: &'static str,
    factor: &Factor<RealItem>,
    dim: usize,
    level: Option<usize>,
    batch: usize,
    index: usize,
) -> RealFactorRouteSummary {
    let mut worst_transpose = 0.0;
    let mut worst_transpose_mode = String::new();

    for branch in factor_branches(factor) {
        for side in [Side::Left, Side::Right] {
            for inv in [false, true] {
                let no_trans = apply_factor_real(
                    factor,
                    dim,
                    side,
                    branch.clone(),
                    BaseFactorOptions {
                        inv,
                        trans: TransMode::NoTrans,
                        trans_target: false,
                    },
                );
                let trans = apply_factor_real(
                    factor,
                    dim,
                    side,
                    branch.clone(),
                    BaseFactorOptions {
                        inv,
                        trans: TransMode::Trans,
                        trans_target: false,
                    },
                );

                let denom = frobenius_norm_real(&no_trans).max(1.0e-14);
                let rel_transpose =
                    frobenius_norm_real(&diff_real(&trans, &transpose_real(&no_trans))) / denom;

                if rel_transpose > worst_transpose {
                    worst_transpose = rel_transpose;
                    worst_transpose_mode =
                        format!("side={} branch={:?} inv={inv}", side_label(side), branch);
                }
            }
        }
    }

    RealFactorRouteSummary {
        case_label,
        family: factor_family_label(factor),
        location: factor_location(factor, level, batch, index),
        worst_transpose,
        worst_transpose_mode,
    }
}

fn collect_complex_factor_route_summaries(
    case_label: &'static str,
    rsrs_operator: &RsrsOperator<
        '_,
        ComplexItem,
        ArrayVectorSpace<ComplexItem>,
        impl RsrsFactorsImpl<ComplexItem> + Shape<2>,
    >,
) -> Vec<ComplexFactorRouteSummary> {
    let factors = rsrs_operator.get_factors();
    let dim = factors.dim;
    let mut out = Vec::new();

    match &factors.id_factors {
        MultiLevelIdFactors::Single(levels) => {
            for (level, batch) in levels.iter().enumerate() {
                for (index, factor) in batch.iter().enumerate() {
                    out.push(summarize_complex_factor_routes(
                        case_label,
                        factor,
                        dim,
                        Some(level),
                        0,
                        index,
                    ));
                }
            }
        }
        MultiLevelIdFactors::Batched(levels) => {
            for (level, batches) in levels.iter().enumerate() {
                for (batch_index, batch) in batches.iter().enumerate() {
                    for (index, factor) in batch.iter().enumerate() {
                        out.push(summarize_complex_factor_routes(
                            case_label,
                            factor,
                            dim,
                            Some(level),
                            batch_index,
                            index,
                        ));
                    }
                }
            }
        }
    }

    for (level, batches) in factors.lu_factors.iter().enumerate() {
        for (batch_index, batch) in batches.iter().enumerate() {
            for (index, factor) in batch.iter().enumerate() {
                out.push(summarize_complex_factor_routes(
                    case_label,
                    factor,
                    dim,
                    Some(level),
                    batch_index,
                    index,
                ));
            }
        }
    }

    for (index, factor) in factors.diag_box_factors.iter().enumerate() {
        out.push(summarize_complex_factor_routes(
            case_label, factor, dim, None, 0, index,
        ));
    }

    out
}

fn collect_real_factor_route_summaries(
    case_label: &'static str,
    rsrs_operator: &RsrsOperator<
        '_,
        RealItem,
        ArrayVectorSpace<RealItem>,
        impl RsrsFactorsImpl<RealItem> + Shape<2>,
    >,
) -> Vec<RealFactorRouteSummary> {
    let factors = rsrs_operator.get_factors();
    let dim = factors.dim;
    let mut out = Vec::new();

    match &factors.id_factors {
        MultiLevelIdFactors::Single(levels) => {
            for (level, batch) in levels.iter().enumerate() {
                for (index, factor) in batch.iter().enumerate() {
                    out.push(summarize_real_factor_routes(
                        case_label,
                        factor,
                        dim,
                        Some(level),
                        0,
                        index,
                    ));
                }
            }
        }
        MultiLevelIdFactors::Batched(levels) => {
            for (level, batches) in levels.iter().enumerate() {
                for (batch_index, batch) in batches.iter().enumerate() {
                    for (index, factor) in batch.iter().enumerate() {
                        out.push(summarize_real_factor_routes(
                            case_label,
                            factor,
                            dim,
                            Some(level),
                            batch_index,
                            index,
                        ));
                    }
                }
            }
        }
    }

    for (level, batches) in factors.lu_factors.iter().enumerate() {
        for (batch_index, batch) in batches.iter().enumerate() {
            for (index, factor) in batch.iter().enumerate() {
                out.push(summarize_real_factor_routes(
                    case_label,
                    factor,
                    dim,
                    Some(level),
                    batch_index,
                    index,
                ));
            }
        }
    }

    for (index, factor) in factors.diag_box_factors.iter().enumerate() {
        out.push(summarize_real_factor_routes(
            case_label, factor, dim, None, 0, index,
        ));
    }

    out
}

fn small_rsrs_args<Item: RlstScalar<Real = f64>>(symmetry: Symmetry) -> RsrsArgs<Item> {
    small_rsrs_args_with_pivot(symmetry, PivotMethod::Lu(0.0))
}

fn small_rsrs_args_with_pivot<Item: RlstScalar<Real = f64>>(
    symmetry: Symmetry,
    pivot_method: PivotMethod,
) -> RsrsArgs<Item> {
    RsrsArgs::new(
        SMALL_FIXED_RANK,
        SMALL_FIXED_RANK,
        0,
        0,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        pivot_method.clone(),
        pivot_method,
        1.0e-16,
        SMALL_FIXED_RANK as f64,
        1.0e-16,
        1.0e-16,
        1,
        1,
        symmetry,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        false,
    )
}

fn small_tol_rsrs_args_with_rank_picking<Item: RlstScalar<Real = f64>>(
    symmetry: Symmetry,
    tol_id: f64,
    rank_picking: RankPicking,
) -> RsrsArgs<Item> {
    small_tol_rsrs_args_with_rank_picking_and_sampling(symmetry, tol_id, rank_picking, 0, 0)
}

fn small_tol_rsrs_args_with_rank_picking_and_sampling<Item: RlstScalar<Real = f64>>(
    symmetry: Symmetry,
    tol_id: f64,
    rank_picking: RankPicking,
    min_num_samples: usize,
    initial_num_samples: usize,
) -> RsrsArgs<Item> {
    RsrsArgs::new(
        8,
        16,
        min_num_samples,
        initial_num_samples,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(0.0),
        PivotMethod::Lu(0.0),
        1.0e-16,
        tol_id,
        1.0e-16,
        1.0e-16,
        1,
        1,
        symmetry,
        rank_picking,
        FactType::Joint,
        false,
        1,
        false,
        false,
    )
}

fn small_tol_rsrs_args<Item: RlstScalar<Real = f64>>(
    symmetry: Symmetry,
    tol_id: f64,
) -> RsrsArgs<Item> {
    small_tol_rsrs_args_with_rank_picking(symmetry, tol_id, RankPicking::Tol)
}

fn run_complex_rsrs_case(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    symmetry: Symmetry,
    label: &'static str,
) -> RsrsMetrics {
    run_complex_rsrs_case_with_args_and_leaf_points(
        comm,
        operator_type,
        label,
        small_rsrs_args(symmetry),
        square_leaf_points(SMALL_FIXED_RANK),
    )
}

fn run_complex_rsrs_case_with_args_and_leaf_points(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    label: &'static str,
    args: RsrsArgs<ComplexItem>,
    max_leaf_points: usize,
) -> RsrsMetrics {
    let params = StructuredOperatorParams::new(
        operator_type,
        Precision::Double,
        GeometryType::Square,
        SMALL_MESH_WIDTH,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface =
        <StructuredOperatorInterface as StructuredOperatorImpl<ComplexItem>>::new(&params);
    let points = get_bempp_points(&interface).unwrap();
    let operator =
        StructuredOperator::<ComplexItem, StructuredOperatorInterface>::from_local(interface);
    let dim = operator.domain().dimension();

    let options = RsrsOptions::<ComplexItem>::new(Some(args));
    let tree = Octree::new(&points, 8, max_leaf_points, comm);
    let mut rsrs = Rsrs::new(&tree, options, dim);
    let mut rsrs_factors = rsrs.run_with_seed(operator.r(), 7);
    let domain = std::rc::Rc::clone(&operator.domain());
    let range = std::rc::Rc::clone(&operator.range());
    let mut rsrs_operator = RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);

    let exact = assemble_matrix_complex(&operator, dim, TransMode::NoTrans);
    let approx = assemble_matrix_complex(&rsrs_operator, dim, TransMode::NoTrans);
    let approx_trans = assemble_matrix_complex(&rsrs_operator, dim, TransMode::Trans);
    let approx_conj_trans = assemble_matrix_complex(&rsrs_operator, dim, TransMode::ConjTrans);
    let exact_manual_conjtrans = try_relative_manual_conjtrans_identity_complex(&operator, dim);
    let rsrs_manual_conjtrans = relative_manual_conjtrans_identity_complex(&rsrs_operator, dim);

    let rel_err_fro = frobenius_norm_complex(&diff_complex(&exact, &approx))
        / frobenius_norm_complex(&exact).max(1.0e-14);
    let rel_adjoint_route = frobenius_norm_complex(&diff_complex(&approx_trans, &adjoint(&approx)))
        / frobenius_norm_complex(&approx).max(1.0e-14);
    let rel_conj_adjoint_route =
        frobenius_norm_complex(&diff_complex(&approx_conj_trans, &adjoint(&approx)))
            / frobenius_norm_complex(&approx).max(1.0e-14);
    let rel_apply_adjoint = relative_complex_adjoint_defect(&rsrs_operator, dim);

    rsrs_operator.inv(true);
    let approx_inv = assemble_matrix_complex(&rsrs_operator, dim, TransMode::NoTrans);
    let approx_inv_trans = assemble_matrix_complex(&rsrs_operator, dim, TransMode::Trans);
    let approx_inv_conj_trans = assemble_matrix_complex(&rsrs_operator, dim, TransMode::ConjTrans);
    let rel_inv_adjoint_route =
        frobenius_norm_complex(&diff_complex(&approx_inv_trans, &adjoint(&approx_inv)))
            / frobenius_norm_complex(&approx_inv).max(1.0e-14);
    let rel_inv_conj_adjoint_route =
        frobenius_norm_complex(&diff_complex(&approx_inv_conj_trans, &adjoint(&approx_inv)))
            / frobenius_norm_complex(&approx_inv).max(1.0e-14);
    let rel_inv_apply_adjoint = relative_complex_adjoint_defect(&rsrs_operator, dim);
    let rsrs_inv_manual_conjtrans = relative_manual_conjtrans_identity_complex(&rsrs_operator, dim);
    let inverse_residual =
        diff_complex(&matmul_complex(&approx_inv, &exact), &identity_complex(dim));
    let rel_inverse_residual_fro = frobenius_norm_complex(&inverse_residual)
        / frobenius_norm_complex(&identity_complex(dim)).max(1.0e-14);

    rsrs_operator.inv(false);
    let (_app_err_left, _app_err_right) = rsrs_error_estimator(
        &operator,
        &rsrs_operator,
        10,
        false,
        APP_ERR_LEFT_SEED,
        APP_ERR_RIGHT_SEED,
    );
    let adjoint_consistency_error: f64 =
        NumCast::from(estimate_adjoint_consistency_error_like_save(
            &rsrs_operator,
            ADJOINT_CHECK_SAMPLES,
            ADJOINT_CHECK_SEED,
        ))
        .unwrap();
    let (norm_fro_error_raw, _norm_fro_operator) = frobenius_diff_and_reference_norm(
        &operator,
        &rsrs_operator,
        FROBENIUS_ESTIMATION_SAMPLES,
        FROB_FORWARD_SEED,
    );
    let (norm_fro_error_transpose_raw, _norm_fro_operator_transpose) =
        frobenius_diff_and_reference_norm_with_mode(
            &operator,
            &rsrs_operator,
            FROBENIUS_ESTIMATION_SAMPLES,
            FROB_FORWARD_SEED ^ 0x55AA_55AA_55AA_55AA,
            TransMode::Trans,
        );
    let diff = DiffOperator(operator.r(), rsrs_operator.r());
    let normal = NormalOperator {
        op: diff.r(),
        transpose_matches_apply: false,
    };
    let sigma_1: f64 = NumCast::from(estimate_normal_operator_norm_sq_like_save(
        &normal.r(),
        POWER_ITERATIONS,
        POWER_DIFF_SEED,
    ))
    .unwrap();
    let normal_structured = NormalOperator {
        op: operator.r(),
        transpose_matches_apply: false,
    };
    let sigma_2: f64 = NumCast::from(estimate_normal_operator_norm_sq_like_save(
        &normal_structured.r(),
        POWER_ITERATIONS,
        POWER_STRUCTURED_SEED,
    ))
    .unwrap();
    let norm_2_error = sigma_1.abs().sqrt() / sigma_2.abs().sqrt();
    let norm_fro_error: f64 = NumCast::from(norm_fro_error_raw).unwrap();
    let norm_fro_error_transpose: f64 = NumCast::from(norm_fro_error_transpose_raw).unwrap();

    rsrs_operator.inv(true);
    let adjoint_consistency_error_inv: f64 =
        NumCast::from(estimate_adjoint_consistency_error_like_save(
            &rsrs_operator,
            ADJOINT_CHECK_SAMPLES,
            INV_ADJOINT_CHECK_SEED,
        ))
        .unwrap();
    let mut sol = <ArrayVectorSpace<ComplexItem> as SamplingSpace>::zero(operator.domain());
    let mut solve_rng = ChaCha8Rng::seed_from_u64(SOLVE_CHECK_SEED);
    operator
        .domain()
        .sampling(&mut sol, &mut solve_rng, SampleType::RealStandardNormal);
    let rhs = operator.apply(sol.r(), TransMode::NoTrans);
    let mut sol_app = rsrs_operator.apply(rhs.r(), TransMode::NoTrans);
    sol_app.sub_inplace(sol.r());
    let solve_error: f64 = NumCast::from(sol_app.norm() / sol.norm()).unwrap();

    std::mem::forget(operator);

    RsrsMetrics {
        label,
        dim,
        rel_err_fro,
        rel_adjoint_route,
        rel_conj_adjoint_route,
        rel_apply_adjoint,
        rel_inv_adjoint_route,
        rel_inv_conj_adjoint_route,
        rel_inv_apply_adjoint,
        rel_inverse_residual_fro,
        norm_2_error,
        norm_fro_error,
        norm_fro_error_transpose,
        adjoint_consistency_error,
        adjoint_consistency_error_inv,
        solve_error,
        exact_manual_conjtrans: exact_manual_conjtrans.unwrap_or(0.0),
        exact_manual_conjtrans_supported: exact_manual_conjtrans.is_some(),
        rsrs_manual_conjtrans,
        rsrs_inv_manual_conjtrans,
    }
}

fn run_real_rsrs_case(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    symmetry: Symmetry,
    label: &'static str,
) -> RsrsMetrics {
    run_real_rsrs_case_with_args(comm, operator_type, label, small_rsrs_args(symmetry))
}

fn run_real_rsrs_case_with_args(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    label: &'static str,
    args: RsrsArgs<RealItem>,
) -> RsrsMetrics {
    let params = StructuredOperatorParams::new(
        operator_type,
        Precision::Double,
        GeometryType::Square,
        SMALL_MESH_WIDTH,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<RealItem>>::new(&params);
    let points = get_bempp_points(&interface).unwrap();
    let operator =
        StructuredOperator::<RealItem, StructuredOperatorInterface>::from_local(interface);
    let dim = operator.domain().dimension();

    let options = RsrsOptions::<RealItem>::new(Some(args));
    let tree = Octree::new(&points, 8, square_leaf_points(SMALL_FIXED_RANK), comm);
    let mut rsrs = Rsrs::new(&tree, options, dim);
    let mut rsrs_factors = rsrs.run_with_seed(operator.r(), 7);
    let domain = std::rc::Rc::clone(&operator.domain());
    let range = std::rc::Rc::clone(&operator.range());
    let mut rsrs_operator = RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);

    let exact = assemble_matrix_real(&operator, dim, TransMode::NoTrans);
    let approx = assemble_matrix_real(&rsrs_operator, dim, TransMode::NoTrans);
    let approx_trans = assemble_matrix_real(&rsrs_operator, dim, TransMode::Trans);

    let rel_err_fro =
        frobenius_norm_real(&diff_real(&exact, &approx)) / frobenius_norm_real(&exact).max(1.0e-14);
    let rel_adjoint_route =
        frobenius_norm_real(&diff_real(&approx_trans, &transpose_real(&approx)))
            / frobenius_norm_real(&approx).max(1.0e-14);
    let rel_apply_adjoint = relative_real_adjoint_defect(&rsrs_operator, dim);

    rsrs_operator.inv(true);
    let approx_inv = assemble_matrix_real(&rsrs_operator, dim, TransMode::NoTrans);
    let approx_inv_trans = assemble_matrix_real(&rsrs_operator, dim, TransMode::Trans);
    let rel_inv_adjoint_route =
        frobenius_norm_real(&diff_real(&approx_inv_trans, &transpose_real(&approx_inv)))
            / frobenius_norm_real(&approx_inv).max(1.0e-14);
    let rel_inv_apply_adjoint = relative_real_adjoint_defect(&rsrs_operator, dim);
    let inverse_residual = diff_real(&matmul_real(&approx_inv, &exact), &identity_real(dim));
    let rel_inverse_residual_fro = frobenius_norm_real(&inverse_residual)
        / frobenius_norm_real(&identity_real(dim)).max(1.0e-14);

    rsrs_operator.inv(false);
    let (_app_err_left, _app_err_right) = rsrs_error_estimator(
        &operator,
        &rsrs_operator,
        10,
        false,
        APP_ERR_LEFT_SEED,
        APP_ERR_RIGHT_SEED,
    );
    let adjoint_consistency_error: f64 =
        NumCast::from(estimate_adjoint_consistency_error_like_save(
            &rsrs_operator,
            ADJOINT_CHECK_SAMPLES,
            ADJOINT_CHECK_SEED,
        ))
        .unwrap();
    let (norm_fro_error_raw, _norm_fro_operator) = frobenius_diff_and_reference_norm(
        &operator,
        &rsrs_operator,
        FROBENIUS_ESTIMATION_SAMPLES,
        FROB_FORWARD_SEED,
    );
    let (norm_fro_error_transpose_raw, _norm_fro_operator_transpose) =
        frobenius_diff_and_reference_norm_with_mode(
            &operator,
            &rsrs_operator,
            FROBENIUS_ESTIMATION_SAMPLES,
            FROB_FORWARD_SEED ^ 0x55AA_55AA_55AA_55AA,
            TransMode::Trans,
        );
    let diff = DiffOperator(operator.r(), rsrs_operator.r());
    let normal = NormalOperator {
        op: diff.r(),
        transpose_matches_apply: false,
    };
    let sigma_1: f64 = NumCast::from(estimate_normal_operator_norm_sq_like_save(
        &normal.r(),
        POWER_ITERATIONS,
        POWER_DIFF_SEED,
    ))
    .unwrap();
    let normal_structured = NormalOperator {
        op: operator.r(),
        transpose_matches_apply: false,
    };
    let sigma_2: f64 = NumCast::from(estimate_normal_operator_norm_sq_like_save(
        &normal_structured.r(),
        POWER_ITERATIONS,
        POWER_STRUCTURED_SEED,
    ))
    .unwrap();
    let norm_2_error = sigma_1.abs().sqrt() / sigma_2.abs().sqrt();
    let norm_fro_error: f64 = NumCast::from(norm_fro_error_raw).unwrap();
    let norm_fro_error_transpose: f64 = NumCast::from(norm_fro_error_transpose_raw).unwrap();

    rsrs_operator.inv(true);
    let adjoint_consistency_error_inv: f64 =
        NumCast::from(estimate_adjoint_consistency_error_like_save(
            &rsrs_operator,
            ADJOINT_CHECK_SAMPLES,
            INV_ADJOINT_CHECK_SEED,
        ))
        .unwrap();
    let mut sol = <ArrayVectorSpace<RealItem> as SamplingSpace>::zero(operator.domain());
    let mut solve_rng = ChaCha8Rng::seed_from_u64(SOLVE_CHECK_SEED);
    operator
        .domain()
        .sampling(&mut sol, &mut solve_rng, SampleType::RealStandardNormal);
    let rhs = operator.apply(sol.r(), TransMode::NoTrans);
    let mut sol_app = rsrs_operator.apply(rhs.r(), TransMode::NoTrans);
    sol_app.sub_inplace(sol.r());
    let solve_error: f64 = NumCast::from(sol_app.norm() / sol.norm()).unwrap();

    std::mem::forget(operator);

    RsrsMetrics {
        label,
        dim,
        rel_err_fro,
        rel_adjoint_route,
        rel_conj_adjoint_route: rel_adjoint_route,
        rel_apply_adjoint,
        rel_inv_adjoint_route,
        rel_inv_conj_adjoint_route: rel_inv_adjoint_route,
        rel_inv_apply_adjoint,
        rel_inverse_residual_fro,
        norm_2_error,
        norm_fro_error,
        norm_fro_error_transpose,
        adjoint_consistency_error,
        adjoint_consistency_error_inv,
        solve_error,
        exact_manual_conjtrans: 0.0,
        exact_manual_conjtrans_supported: true,
        rsrs_manual_conjtrans: 0.0,
        rsrs_inv_manual_conjtrans: 0.0,
    }
}

fn run_complex_factor_diagnostics_case(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    symmetry: Symmetry,
    label: &'static str,
) -> Vec<ComplexFactorRouteSummary> {
    run_complex_factor_diagnostics_case_with_args(
        comm,
        operator_type,
        label,
        small_rsrs_args(symmetry),
    )
}

fn run_complex_factor_diagnostics_case_with_args(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    label: &'static str,
    args: RsrsArgs<ComplexItem>,
) -> Vec<ComplexFactorRouteSummary> {
    let params = StructuredOperatorParams::new(
        operator_type,
        Precision::Double,
        GeometryType::Square,
        SMALL_MESH_WIDTH,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface =
        <StructuredOperatorInterface as StructuredOperatorImpl<ComplexItem>>::new(&params);
    let points = get_bempp_points(&interface).unwrap();
    let operator =
        StructuredOperator::<ComplexItem, StructuredOperatorInterface>::from_local(interface);
    let dim = operator.domain().dimension();

    let options = RsrsOptions::<ComplexItem>::new(Some(args));
    let tree = Octree::new(&points, 8, square_leaf_points(SMALL_FIXED_RANK), comm);
    let mut rsrs = Rsrs::new(&tree, options, dim);
    let mut rsrs_factors = rsrs.run_with_seed(operator.r(), 7);
    let domain = std::rc::Rc::clone(&operator.domain());
    let range = std::rc::Rc::clone(&operator.range());
    let rsrs_operator = RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);

    collect_complex_factor_route_summaries(label, &rsrs_operator)
}

fn run_real_factor_diagnostics_case(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    symmetry: Symmetry,
    label: &'static str,
) -> Vec<RealFactorRouteSummary> {
    run_real_factor_diagnostics_case_with_args(
        comm,
        operator_type,
        label,
        small_rsrs_args(symmetry),
    )
}

fn run_real_factor_diagnostics_case_with_args(
    comm: &SimpleCommunicator,
    operator_type: StructuredOperatorType,
    label: &'static str,
    args: RsrsArgs<RealItem>,
) -> Vec<RealFactorRouteSummary> {
    let params = StructuredOperatorParams::new(
        operator_type,
        Precision::Double,
        GeometryType::Square,
        SMALL_MESH_WIDTH,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<RealItem>>::new(&params);
    let points = get_bempp_points(&interface).unwrap();
    let operator =
        StructuredOperator::<RealItem, StructuredOperatorInterface>::from_local(interface);
    let dim = operator.domain().dimension();

    let options = RsrsOptions::<RealItem>::new(Some(args));
    let tree = Octree::new(&points, 8, square_leaf_points(SMALL_FIXED_RANK), comm);
    let mut rsrs = Rsrs::new(&tree, options, dim);
    let mut rsrs_factors = rsrs.run_with_seed(operator.r(), 7);
    let domain = std::rc::Rc::clone(&operator.domain());
    let range = std::rc::Rc::clone(&operator.range());
    let rsrs_operator = RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);

    collect_real_factor_route_summaries(label, &rsrs_operator)
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_small_regression() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let metrics = vec![
        run_real_rsrs_case(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            Symmetry::Symmetric,
            "real symmetric",
        ),
        run_real_rsrs_case(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            Symmetry::NoSymm,
            "real nonsymmetric",
        ),
        run_complex_rsrs_case(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            Symmetry::Symmetric,
            "complex symmetric",
        ),
        run_complex_rsrs_case(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            Symmetry::NoSymm,
            "complex nonsymmetric",
        ),
    ];

    println!("small perturbed BIEGrid RSRS metrics (mesh_width={SMALL_MESH_WIDTH}, rank={SMALL_FIXED_RANK})");
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>12} {:>10} {:>16} {:>16} {:>12}",
        "case",
        "dim",
        "rel_2_norm",
        "rel_fro",
        "rel_fro_T",
        "n2<=fro",
        "adjoint_check",
        "inverse_adj",
        "solve_err"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>5} {:>12.3e} {:>12.3e} {:>12.3e} {:>10} {:>16.3e} {:>16.3e} {:>12.3e}",
            row.label,
            row.dim,
            row.norm_2_error,
            row.norm_fro_error,
            row.norm_fro_error_transpose,
            if row.norm_2_error <= row.norm_fro_error {
                "yes"
            } else {
                "no"
            },
            row.adjoint_consistency_error,
            row.adjoint_consistency_error_inv,
            row.solve_error
        );
    }

    println!();
    println!("debug adjoint-route metrics");
    println!(
        "{:<22} {:>12} {:>14} {:>14} {:>14} {:>18} {:>18} {:>18} {:>18}",
        "case",
        "apply_fro",
        "adj(T)",
        "adj(H)",
        "adj_apply",
        "inv_adj(T)",
        "inv_adj(H)",
        "inv_adj_apply",
        "inv_residual_fro"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>12.3e} {:>14.3e} {:>14.3e} {:>14.3e} {:>18.3e} {:>18.3e} {:>18.3e} {:>18.3e}",
            row.label,
            row.rel_err_fro,
            row.rel_adjoint_route,
            row.rel_conj_adjoint_route,
            row.rel_apply_adjoint,
            row.rel_inv_adjoint_route,
            row.rel_inv_conj_adjoint_route,
            row.rel_inv_apply_adjoint,
            row.rel_inverse_residual_fro
        );
    }

    println!();
    println!("manual conj(trans(conj(x))) vs direct conjtrans");
    println!(
        "{:<22} {:>18} {:>18} {:>18}",
        "case", "exact_manual_H", "rsrs_manual_H", "rsrs_inv_manual_H"
    );
    for row in metrics.iter().filter(|row| row.label.contains("complex")) {
        let exact_manual = if row.exact_manual_conjtrans_supported {
            format!("{:>18.3e}", row.exact_manual_conjtrans)
        } else {
            format!("{:>18}", "unsupported")
        };
        println!(
            "{:<22} {} {:>18.3e} {:>18.3e}",
            row.label, exact_manual, row.rsrs_manual_conjtrans, row.rsrs_inv_manual_conjtrans
        );
    }

    assert!(
        metrics.iter().all(|row| {
            row.rel_err_fro.is_finite()
                && row.rel_adjoint_route.is_finite()
                && row.rel_apply_adjoint.is_finite()
                && row.rel_inv_adjoint_route.is_finite()
                && row.rel_inv_apply_adjoint.is_finite()
                && row.rel_inverse_residual_fro.is_finite()
                && row.norm_2_error.is_finite()
                && row.norm_fro_error.is_finite()
                && row.adjoint_consistency_error.is_finite()
                && row.adjoint_consistency_error_inv.is_finite()
                && row.solve_error.is_finite()
                && row.rsrs_manual_conjtrans.is_finite()
                && row.rsrs_inv_manual_conjtrans.is_finite()
        }),
        "small perturbed BIEGrid regression produced non-finite metrics"
    );
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_small_regression_luhybrid -- --exact --ignored --nocapture"]
fn biegrid_perturbed_rsrs_small_regression_luhybrid() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let args_symm_real =
        small_rsrs_args_with_pivot::<RealItem>(Symmetry::Symmetric, PivotMethod::LuHybrid(0.0));
    let args_nosymm_real =
        small_rsrs_args_with_pivot::<RealItem>(Symmetry::NoSymm, PivotMethod::LuHybrid(0.0));
    let args_symm_complex =
        small_rsrs_args_with_pivot::<ComplexItem>(Symmetry::Symmetric, PivotMethod::LuHybrid(0.0));
    let args_nosymm_complex =
        small_rsrs_args_with_pivot::<ComplexItem>(Symmetry::NoSymm, PivotMethod::LuHybrid(0.0));

    let metrics = vec![
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            "real symmetric",
            args_symm_real,
        ),
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            "real nonsymmetric",
            args_nosymm_real,
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            "complex symmetric",
            args_symm_complex,
            square_leaf_points(SMALL_FIXED_RANK),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            "complex nonsymmetric",
            args_nosymm_complex,
            square_leaf_points(SMALL_FIXED_RANK),
        ),
    ];

    println!(
        "small perturbed BIEGrid RSRS metrics (mesh_width={SMALL_MESH_WIDTH}, rank={SMALL_FIXED_RANK}, rank_picking=Min, pivot=LuHybrid)"
    );
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>12} {:>10} {:>16} {:>16} {:>12}",
        "case",
        "dim",
        "rel_2_norm",
        "rel_fro",
        "rel_fro_T",
        "n2<=fro",
        "adjoint_check",
        "inverse_adj",
        "solve_err"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>5} {:>12.3e} {:>12.3e} {:>12.3e} {:>10} {:>16.3e} {:>16.3e} {:>12.3e}",
            row.label,
            row.dim,
            row.norm_2_error,
            row.norm_fro_error,
            row.norm_fro_error_transpose,
            if row.norm_2_error <= row.norm_fro_error {
                "yes"
            } else {
                "no"
            },
            row.adjoint_consistency_error,
            row.adjoint_consistency_error_inv,
            row.solve_error
        );
    }

    assert!(
        metrics.iter().all(|row| {
            row.norm_2_error.is_finite()
                && row.norm_fro_error.is_finite()
                && row.norm_fro_error_transpose.is_finite()
                && row.adjoint_consistency_error.is_finite()
                && row.adjoint_consistency_error_inv.is_finite()
                && row.solve_error.is_finite()
        }),
        "small perturbed BIEGrid LuHybrid regression produced non-finite metrics"
    );
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_small_regression_tol1e12 -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_small_regression_tol1e12() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let metrics = vec![
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            "real symmetric",
            small_tol_rsrs_args(Symmetry::Symmetric, 1.0e-12),
        ),
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            "real nonsymmetric",
            small_tol_rsrs_args(Symmetry::NoSymm, 1.0e-12),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            "complex symmetric",
            small_tol_rsrs_args(Symmetry::Symmetric, 1.0e-12),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            "complex nonsymmetric",
            small_tol_rsrs_args(Symmetry::NoSymm, 1.0e-12),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
    ];

    println!("small perturbed BIEGrid RSRS metrics (mesh_width={SMALL_MESH_WIDTH}, tol_id=1e-12)");
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>10} {:>16} {:>16} {:>12}",
        "case",
        "dim",
        "rel_2_norm",
        "rel_fro",
        "n2<=fro",
        "adjoint_check",
        "inverse_adj",
        "solve_err"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>5} {:>12.3e} {:>12.3e} {:>10} {:>16.3e} {:>16.3e} {:>12.3e}",
            row.label,
            row.dim,
            row.norm_2_error,
            row.norm_fro_error,
            if row.norm_2_error <= row.norm_fro_error {
                "yes"
            } else {
                "no"
            },
            row.adjoint_consistency_error,
            row.adjoint_consistency_error_inv,
            row.solve_error
        );
    }
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_small_regression_tol1e12_rankpicking_tol -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_small_regression_tol1e12_rankpicking_tol() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let metrics = vec![
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            "real symmetric",
            small_tol_rsrs_args_with_rank_picking(Symmetry::Symmetric, 1.0e-12, RankPicking::Tol),
        ),
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            "real nonsymmetric",
            small_tol_rsrs_args_with_rank_picking(Symmetry::NoSymm, 1.0e-12, RankPicking::Tol),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            "complex symmetric",
            small_tol_rsrs_args_with_rank_picking(Symmetry::Symmetric, 1.0e-12, RankPicking::Tol),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            "complex nonsymmetric",
            small_tol_rsrs_args_with_rank_picking(Symmetry::NoSymm, 1.0e-12, RankPicking::Tol),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
    ];

    println!(
        "small perturbed BIEGrid RSRS metrics (mesh_width={SMALL_MESH_WIDTH}, tol_id=1e-12, rank_picking=Tol)"
    );
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>12} {:>10} {:>16} {:>16} {:>12}",
        "case",
        "dim",
        "rel_2_norm",
        "rel_fro",
        "rel_fro_T",
        "n2<=fro",
        "adjoint_check",
        "inverse_adj",
        "solve_err"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>5} {:>12.3e} {:>12.3e} {:>12.3e} {:>10} {:>16.3e} {:>16.3e} {:>12.3e}",
            row.label,
            row.dim,
            row.norm_2_error,
            row.norm_fro_error,
            row.norm_fro_error_transpose,
            row.norm_2_error <= row.norm_fro_error,
            row.adjoint_consistency_error,
            row.adjoint_consistency_error_inv,
            row.solve_error,
        );
    }

    assert!(
        metrics.iter().all(|row| {
            row.norm_2_error.is_finite()
                && row.norm_fro_error.is_finite()
                && row.adjoint_consistency_error.is_finite()
                && row.adjoint_consistency_error_inv.is_finite()
                && row.solve_error.is_finite()
        }),
        "small perturbed BIEGrid regression with rank_picking=Tol produced non-finite metrics"
    );
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_small_regression_tol1e12_rankpicking_tol_min1024 -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_small_regression_tol1e12_rankpicking_tol_min1024() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let metrics = vec![
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            "real symmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::Symmetric,
                1.0e-12,
                RankPicking::Tol,
                1024,
                1024,
            ),
        ),
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            "real nonsymmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::NoSymm,
                1.0e-12,
                RankPicking::Tol,
                1024,
                1024,
            ),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            "complex symmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::Symmetric,
                1.0e-12,
                RankPicking::Tol,
                1024,
                1024,
            ),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            "complex nonsymmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::NoSymm,
                1.0e-12,
                RankPicking::Tol,
                1024,
                1024,
            ),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
    ];

    println!(
        "small perturbed BIEGrid RSRS metrics (mesh_width={SMALL_MESH_WIDTH}, tol_id=1e-12, rank_picking=Tol, min_samples=1024)"
    );
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>12} {:>10} {:>16} {:>16} {:>12}",
        "case",
        "dim",
        "rel_2_norm",
        "rel_fro",
        "rel_fro_T",
        "n2<=fro",
        "adjoint_check",
        "inverse_adj",
        "solve_err"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>5} {:>12.3e} {:>12.3e} {:>12.3e} {:>10} {:>16.3e} {:>16.3e} {:>12.3e}",
            row.label,
            row.dim,
            row.norm_2_error,
            row.norm_fro_error,
            row.norm_fro_error_transpose,
            if row.norm_2_error <= row.norm_fro_error {
                "yes"
            } else {
                "no"
            },
            row.adjoint_consistency_error,
            row.adjoint_consistency_error_inv,
            row.solve_error
        );
    }
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_small_regression_tol1e9_rankpicking_tol_min1048 -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_small_regression_tol1e9_rankpicking_tol_min1048() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let metrics = vec![
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            "real symmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::Symmetric,
                1.0e-9,
                RankPicking::Tol,
                1048,
                1048,
            ),
        ),
        run_real_rsrs_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            "real nonsymmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::NoSymm,
                1.0e-9,
                RankPicking::Tol,
                1048,
                1048,
            ),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            "complex symmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::Symmetric,
                1.0e-9,
                RankPicking::Tol,
                1048,
                1048,
            ),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
        run_complex_rsrs_case_with_args_and_leaf_points(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            "complex nonsymmetric",
            small_tol_rsrs_args_with_rank_picking_and_sampling(
                Symmetry::NoSymm,
                1.0e-9,
                RankPicking::Tol,
                1048,
                1048,
            ),
            square_leaf_points(SMALL_FIXED_RANK),
        ),
    ];

    println!(
        "small perturbed BIEGrid RSRS metrics (mesh_width={SMALL_MESH_WIDTH}, tol_id=1e-9, rank_picking=Tol, min_samples=1048)"
    );
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>12} {:>10} {:>16} {:>16} {:>12}",
        "case",
        "dim",
        "rel_2_norm",
        "rel_fro",
        "rel_fro_T",
        "n2<=fro",
        "adjoint_check",
        "inverse_adj",
        "solve_err"
    );
    for row in &metrics {
        println!(
            "{:<22} {:>5} {:>12.3e} {:>12.3e} {:>12.3e} {:>10} {:>16.3e} {:>16.3e} {:>12.3e}",
            row.label,
            row.dim,
            row.norm_2_error,
            row.norm_fro_error,
            row.norm_fro_error_transpose,
            if row.norm_2_error <= row.norm_fro_error {
                "yes"
            } else {
                "no"
            },
            row.adjoint_consistency_error,
            row.adjoint_consistency_error_inv,
            row.solve_error
        );
    }
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_single_diag_complex_nonsymmetric -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_single_diag_complex_nonsymmetric() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let params = StructuredOperatorParams::new(
        StructuredOperatorType::BIEGridComplexPerturbed,
        Precision::Double,
        GeometryType::Square,
        SMALL_MESH_WIDTH,
        0.0,
        1,
        0,
        Assembler::Dense,
    );
    let interface =
        <StructuredOperatorInterface as StructuredOperatorImpl<ComplexItem>>::new(&params);
    let points = get_bempp_points(&interface).unwrap();
    let single_box_leaf_points = points.len();

    let single_diag_args = RsrsArgs::new(
        8,
        16,
        0,
        0,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(0.0),
        PivotMethod::Lu(0.0),
        1.0e-16,
        1.0e-15,
        1.0e-16,
        1.0e-16,
        1,
        1,
        Symmetry::NoSymm,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        false,
    );

    let metrics = run_complex_rsrs_case_with_args_and_leaf_points(
        &comm,
        StructuredOperatorType::BIEGridComplexPerturbed,
        "complex nonsymmetric single diag",
        single_diag_args,
        single_box_leaf_points,
    );

    println!(
        "single-diag complex nonsymmetric metrics (mesh_width={SMALL_MESH_WIDTH}, rank={SMALL_FIXED_RANK}, max_leaf_points={single_box_leaf_points})"
    );
    println!(
        "rel_2={:.3e} rel_fro={:.3e} rel_fro_T={:.3e} adjoint={:.3e} inverse_adj={:.3e} solve={:.3e}",
        metrics.norm_2_error,
        metrics.norm_fro_error,
        metrics.norm_fro_error_transpose,
        metrics.adjoint_consistency_error,
        metrics.adjoint_consistency_error_inv,
        metrics.solve_error
    );
    println!(
        "debug: apply_fro={:.3e} adj(T)={:.3e} adj(H)={:.3e} inv_adj(T)={:.3e} inv_adj(H)={:.3e} inv_residual_fro={:.3e}",
        metrics.rel_err_fro,
        metrics.rel_adjoint_route,
        metrics.rel_conj_adjoint_route,
        metrics.rel_inv_adjoint_route,
        metrics.rel_inv_conj_adjoint_route,
        metrics.rel_inverse_residual_fro
    );
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs biegrid_perturbed_rsrs_tol15_complex_nonsymmetric -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_tol15_complex_nonsymmetric() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let tol15_args = RsrsArgs::new(
        8,
        16,
        0,
        0,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(0.0),
        PivotMethod::Lu(0.0),
        1.0e-16,
        1.0e-15,
        1.0e-16,
        1.0e-16,
        1,
        1,
        Symmetry::NoSymm,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        false,
    );

    let metrics = run_complex_rsrs_case_with_args_and_leaf_points(
        &comm,
        StructuredOperatorType::BIEGridComplexPerturbed,
        "complex nonsymmetric tol1e-15",
        tol15_args,
        square_leaf_points(SMALL_FIXED_RANK),
    );

    println!(
        "tol15 complex nonsymmetric metrics (mesh_width={SMALL_MESH_WIDTH}, dim≈441, max_leaf_points={})",
        square_leaf_points(SMALL_FIXED_RANK)
    );
    println!(
        "rel_2={:.3e} rel_fro={:.3e} rel_fro_T={:.3e} adjoint={:.3e} inverse_adj={:.3e} solve={:.3e}",
        metrics.norm_2_error,
        metrics.norm_fro_error,
        metrics.norm_fro_error_transpose,
        metrics.adjoint_consistency_error,
        metrics.adjoint_consistency_error_inv,
        metrics.solve_error
    );
    println!(
        "debug: apply_fro={:.3e} adj(T)={:.3e} adj(H)={:.3e} inv_adj(T)={:.3e} inv_adj(H)={:.3e} inv_residual_fro={:.3e}",
        metrics.rel_err_fro,
        metrics.rel_adjoint_route,
        metrics.rel_conj_adjoint_route,
        metrics.rel_inv_adjoint_route,
        metrics.rel_inv_conj_adjoint_route,
        metrics.rel_inverse_residual_fro
    );
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs factor_route -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_factor_route_diagnostics() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let mut complex_rows = Vec::new();
    complex_rows.extend(run_complex_factor_diagnostics_case(
        &comm,
        StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
        Symmetry::Symmetric,
        "complex symmetric",
    ));
    complex_rows.extend(run_complex_factor_diagnostics_case(
        &comm,
        StructuredOperatorType::BIEGridComplexPerturbed,
        Symmetry::NoSymm,
        "complex nonsymmetric",
    ));

    complex_rows.sort_by(|lhs, rhs| {
        rhs.worst_manual_adjoint
            .max(rhs.worst_transpose)
            .total_cmp(&lhs.worst_manual_adjoint.max(lhs.worst_transpose))
    });

    println!(
        "small perturbed BIEGrid factor-route diagnostics (mesh_width={SMALL_MESH_WIDTH}, rank={SMALL_FIXED_RANK})"
    );
    println!();
    println!("worst complex factor transpose/manual-adjoint mismatches");
    println!(
        "{:<18} {:<6} {:>12} {:>12} {:<28} {:<28} {}",
        "case", "kind", "worst_T", "worst_H", "mode_T", "mode_H", "location"
    );
    for row in complex_rows.iter().take(24) {
        println!(
            "{:<18} {:<6} {:>12.3e} {:>12.3e} {:<28} {:<28} {}",
            row.case_label,
            row.family,
            row.worst_transpose,
            row.worst_manual_adjoint,
            row.worst_transpose_mode,
            row.worst_manual_adjoint_mode,
            row.location
        );
    }

    assert!(complex_rows
        .iter()
        .all(|row| { row.worst_transpose.is_finite() && row.worst_manual_adjoint.is_finite() }));
}

#[test]
#[ignore = "diagnostic regression; run manually with cargo test --test biegrid_perturbed_rsrs diag_factor_tol1e12 -- --ignored --nocapture"]
fn biegrid_perturbed_rsrs_diag_factor_tol1e12() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("RSRS_BIEGRID_PERTURB_SCALE", "1e-2");

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let mut real_rows = Vec::new();
    real_rows.extend(
        run_real_factor_diagnostics_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealSymmetricPerturbed,
            "real symmetric",
            small_tol_rsrs_args(Symmetry::Symmetric, 1.0e-12),
        )
        .into_iter()
        .filter(|row| row.family == "diag"),
    );
    real_rows.extend(
        run_real_factor_diagnostics_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridRealPerturbed,
            "real nonsymmetric",
            small_tol_rsrs_args(Symmetry::NoSymm, 1.0e-12),
        )
        .into_iter()
        .filter(|row| row.family == "diag"),
    );

    let mut complex_rows = Vec::new();
    complex_rows.extend(
        run_complex_factor_diagnostics_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
            "complex symmetric",
            small_tol_rsrs_args(Symmetry::Symmetric, 1.0e-12),
        )
        .into_iter()
        .filter(|row| row.family == "diag"),
    );
    complex_rows.extend(
        run_complex_factor_diagnostics_case_with_args(
            &comm,
            StructuredOperatorType::BIEGridComplexPerturbed,
            "complex nonsymmetric",
            small_tol_rsrs_args(Symmetry::NoSymm, 1.0e-12),
        )
        .into_iter()
        .filter(|row| row.family == "diag"),
    );

    real_rows.sort_by(|lhs, rhs| rhs.worst_transpose.total_cmp(&lhs.worst_transpose));
    complex_rows.sort_by(|lhs, rhs| {
        rhs.worst_manual_adjoint
            .max(rhs.worst_transpose)
            .total_cmp(&lhs.worst_manual_adjoint.max(lhs.worst_transpose))
    });

    println!(
        "small perturbed BIEGrid diagonal-factor diagnostics (mesh_width={SMALL_MESH_WIDTH}, tol_id=1e-12, rank_picking=Tol)"
    );
    println!();
    println!("real diag factors");
    println!(
        "{:<18} {:>12} {:<28} {}",
        "case", "worst_T", "mode_T", "location"
    );
    for row in &real_rows {
        println!(
            "{:<18} {:>12.3e} {:<28} {}",
            row.case_label, row.worst_transpose, row.worst_transpose_mode, row.location
        );
    }

    println!();
    println!("complex diag factors");
    println!(
        "{:<18} {:>12} {:>12} {:<28} {:<28} {}",
        "case", "worst_T", "worst_H", "mode_T", "mode_H", "location"
    );
    for row in &complex_rows {
        println!(
            "{:<18} {:>12.3e} {:>12.3e} {:<28} {:<28} {}",
            row.case_label,
            row.worst_transpose,
            row.worst_manual_adjoint,
            row.worst_transpose_mode,
            row.worst_manual_adjoint_mode,
            row.location
        );
    }

    assert!(real_rows.iter().all(|row| row.worst_transpose.is_finite()));
    assert!(complex_rows
        .iter()
        .all(|row| row.worst_transpose.is_finite() && row.worst_manual_adjoint.is_finite()));
}
