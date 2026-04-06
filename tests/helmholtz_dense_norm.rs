use bempp_octree::{MortonKey, Octree};
use bempp_rsrs::{
    rsrs::{
        args::{RankPicking, RsrsArgs, RsrsOptions, Symmetry},
        rsrs_cycle::Rsrs,
        rsrs_factors::rsrs_operator::FactType,
        rsrs_factors::{
            base_factors::BaseFactorOptions,
            commutative_factors::{
                CommutativeFactorsOperations, FactorType, MulOptions, MultiLevelIdFactors,
                RsrsFactors,
            },
        },
        sketch::Shift,
    },
    utils::linear_algebra::{BlockExtractionMethod, NullMethod},
};
use mpi::topology::SimpleCommunicator;
use num::Complex;
use rayon::ThreadPoolBuilder;
use rlst::prelude::*;
use rsrs_exps::{
    io::{
        structured_operator::{
            get_bempp_points, Assembler, GeometryType, LocalFrom, StructuredOperator,
            StructuredOperatorImpl, StructuredOperatorInterface, StructuredOperatorParams,
        },
        structured_operators_types::StructuredOperatorType,
    },
    test_prep::Precision,
};

type Item = Complex<f64>;

struct DenseMetrics {
    symmetry_label: &'static str,
    rel_err_2: f64,
    rel_err_fro: f64,
    trans_rel_err_2: f64,
    trans_route_rel_err_2: f64,
    exact_symmetry_rel_err_2: f64,
    diag_only_rel_err_2: f64,
    diag_id_rel_err_2: f64,
    manual_full_rel_err_2: f64,
    manual_full_vs_operator_rel_err_2: f64,
    left_top10_energy: f64,
    left_top25_energy: f64,
    right_top10_energy: f64,
    right_top25_energy: f64,
    left_top_indices: Vec<(usize, f64)>,
    right_top_indices: Vec<(usize, f64)>,
    left_top_leaves: Vec<LeafEnergy>,
    right_top_leaves: Vec<LeafEnergy>,
}

#[derive(Clone)]
struct LeafEnergy {
    key: MortonKey,
    energy: f64,
    occupancy: usize,
}

#[derive(Clone, Copy)]
enum StageApproximation {
    DiagOnly,
    DiagId,
    Full,
}

fn sphere_surface_leaf_points(rank: usize) -> usize {
    6 * rank
}

fn basis_vector(dim: usize, index: usize) -> Vec<Item> {
    let mut basis = vec![Item::new(0.0, 0.0); dim];
    basis[index] = Item::new(1.0, 0.0);
    basis
}

fn apply_operator<Op>(op: &Op, input: &[Item], trans_mode: TransMode) -> Vec<Item>
where
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let mut x = zero_element(op.domain());
    x.imp_mut().fill_inplace_raw(input);
    let y = op.apply(x.r(), trans_mode);
    y.view().iter().collect()
}

fn assemble_matrix<Op>(op: &Op, dim: usize, trans_mode: TransMode) -> DynamicArray<Item, 2>
where
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let mut matrix = rlst_dynamic_array2!(Item, [dim, dim]);

    for col in 0..dim {
        let basis = basis_vector(dim, col);
        let column = apply_operator(op, &basis, trans_mode);
        for (row, value) in column.into_iter().enumerate() {
            matrix[[row, col]] = value;
        }
    }

    matrix
}

fn spectral_norm(matrix: &DynamicArray<Item, 2>) -> f64 {
    let shape = matrix.shape();
    let reduced_dim = shape[0].min(shape[1]);
    let mut singular_values = rlst_dynamic_array1!(f64, [reduced_dim]);
    let mut u = rlst_dynamic_array2!(Item, [shape[0], reduced_dim]);
    let mut vt = rlst_dynamic_array2!(Item, [reduced_dim, shape[1]]);
    let mut aux = empty_array();
    aux.fill_from_resize(matrix.r());
    aux.r_mut()
        .into_svd_alloc(
            u.r_mut(),
            vt.r_mut(),
            singular_values.data_mut(),
            SvdMode::Reduced,
        )
        .unwrap();
    singular_values[[0]]
}

fn leading_svd(matrix: &DynamicArray<Item, 2>) -> (f64, Vec<Item>, Vec<Item>) {
    let shape = matrix.shape();
    let reduced_dim = shape[0].min(shape[1]);
    let mut singular_values = rlst_dynamic_array1!(f64, [reduced_dim]);
    let mut u = rlst_dynamic_array2!(Item, [shape[0], reduced_dim]);
    let mut vt = rlst_dynamic_array2!(Item, [reduced_dim, shape[1]]);
    let mut aux = empty_array();
    aux.fill_from_resize(matrix.r());
    aux.r_mut()
        .into_svd_alloc(
            u.r_mut(),
            vt.r_mut(),
            singular_values.data_mut(),
            SvdMode::Reduced,
        )
        .unwrap();

    let left = (0..shape[0]).map(|row| u[[row, 0]]).collect::<Vec<_>>();
    let right = (0..shape[1])
        .map(|col| vt[[0, col]].conj())
        .collect::<Vec<_>>();

    (singular_values[[0]], left, right)
}

fn frobenius_norm(matrix: &DynamicArray<Item, 2>) -> f64 {
    matrix
        .r()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn rel_norm_2(numerator: &DynamicArray<Item, 2>, denominator_norm: f64) -> f64 {
    spectral_norm(numerator) / denominator_norm.max(1.0e-14)
}

fn rel_fro(numerator: &DynamicArray<Item, 2>, denominator_norm: f64) -> f64 {
    frobenius_norm(numerator) / denominator_norm.max(1.0e-14)
}

fn diff(lhs: &DynamicArray<Item, 2>, rhs: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
    let mut out = empty_array();
    out.fill_from_resize(lhs.r() - rhs.r());
    out
}

fn identity_matrix(dim: usize) -> DynamicArray<Item, 2> {
    let mut identity = rlst_dynamic_array2!(Item, [dim, dim]);
    for index in 0..dim {
        identity[[index, index]] = Item::new(1.0, 0.0);
    }
    identity
}

fn transpose(matrix: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
    let mut out = empty_array();
    out.fill_from_resize(matrix.r().transpose());
    out
}

fn normalized_index_energy(values: &[Item]) -> Vec<(usize, f64)> {
    let total = values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .max(1.0e-30);
    let mut energies = values
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.norm_sqr() / total))
        .collect::<Vec<_>>();
    energies.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());
    energies
}

fn cumulative_energy(values: &[(usize, f64)], count: usize) -> f64 {
    values.iter().take(count).map(|(_, energy)| *energy).sum()
}

fn top_leaf_energy<C>(tree: &Octree<'_, C>, values: &[Item], count: usize) -> Vec<LeafEnergy>
where
    C: mpi::traits::Communicator,
{
    let total = values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .max(1.0e-30);
    let leaf_map = tree.leaf_keys_to_local_point_indices();
    let mut energies = tree
        .leaf_keys()
        .iter()
        .filter_map(|key| {
            leaf_map.get(key).map(|indices| LeafEnergy {
                key: *key,
                energy: indices
                    .iter()
                    .map(|&index| values[index].norm_sqr())
                    .sum::<f64>()
                    / total,
                occupancy: indices.len(),
            })
        })
        .collect::<Vec<_>>();

    energies.sort_by(|lhs, rhs| rhs.energy.partial_cmp(&lhs.energy).unwrap());
    energies.truncate(count);
    energies
}

fn format_index_energies(energies: &[(usize, f64)]) -> String {
    energies
        .iter()
        .map(|(index, energy)| format!("{index}:{energy:.3e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_leaf_energies(energies: &[LeafEnergy]) -> String {
    energies
        .iter()
        .map(|leaf| {
            let (_, xyz) = leaf.key.decode();
            format!(
                "L{}{:?}:e={:.3e},n={}",
                leaf.key.level(),
                xyz,
                leaf.energy,
                leaf.occupancy
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn replay_joint_left_no_trans(
    factors: &RsrsFactors<Item>,
    stage: StageApproximation,
) -> DynamicArray<Item, 2> {
    assert!(
        matches!(factors.fact_type, FactType::Joint),
        "stage replay assumes joint factor storage"
    );
    let id_levels = match &factors.id_factors {
        MultiLevelIdFactors::Batched(levels) => levels,
        MultiLevelIdFactors::Single(_) => {
            panic!("stage replay for split factors is not implemented")
        }
    };
    let mut target = identity_matrix(factors.dim);
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(factors.num_threads)
        .build()
        .unwrap();
    let base_options = BaseFactorOptions {
        inv: false,
        trans: TransMode::NoTrans,
        trans_target: false,
    };
    let s_options = MulOptions {
        base_options: base_options.clone(),
        side: Side::Left,
        factor_type: FactorType::S,
    };
    let f_options = MulOptions {
        base_options: base_options.clone(),
        side: Side::Left,
        factor_type: FactorType::F,
    };

    if !matches!(stage, StageApproximation::DiagOnly) {
        for level_it in 0..factors.num_levels {
            let id_batches = &id_levels[level_it];
            for batch_ind in 0..id_batches.len() {
                id_batches[batch_ind].mul(
                    &mut target,
                    &thread_pool,
                    factors.num_threads,
                    &s_options,
                );
                if matches!(stage, StageApproximation::Full) {
                    factors.lu_factors[level_it][batch_ind].mul(
                        &mut target,
                        &thread_pool,
                        factors.num_threads,
                        &s_options,
                    );
                }
            }
        }
    }

    factors.diag_box_factors.mul(
        &mut target,
        &thread_pool,
        factors.num_threads,
        &MulOptions {
            base_options: base_options.clone(),
            side: Side::Left,
            factor_type: FactorType::F,
        },
    );

    if !matches!(stage, StageApproximation::DiagOnly) {
        for level_it in (0..factors.num_levels).rev() {
            let id_batches = &id_levels[level_it];
            for batch_ind in (0..id_batches.len()).rev() {
                if matches!(stage, StageApproximation::Full) {
                    factors.lu_factors[level_it][batch_ind].mul(
                        &mut target,
                        &thread_pool,
                        factors.num_threads,
                        &f_options,
                    );
                }
                id_batches[batch_ind].mul(
                    &mut target,
                    &thread_pool,
                    factors.num_threads,
                    &f_options,
                );
            }
        }
    }

    target
}

fn run_case(
    comm: &SimpleCommunicator,
    points: &[bempp_octree::Point],
    operator: &StructuredOperator<Item, StructuredOperatorInterface>,
    symmetry: Symmetry,
    label: &'static str,
) -> DenseMetrics {
    let dim = points.len();
    let tol_id = 8.0;
    let args = RsrsArgs::new(
        8,
        16,
        0,
        0,
        None,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        bempp_rsrs::rsrs::rsrs_factors::null_and_extract::PivotMethod::Lu(0.0),
        bempp_rsrs::rsrs::rsrs_factors::null_and_extract::PivotMethod::Lu(0.0),
        1.0e-16,
        tol_id,
        1.0e-16,
        1.0e-16,
        1,
        1,
        symmetry,
        RankPicking::Min,
        FactType::Joint,
        false,
        8,
        false,
        true,
    );
    let options = RsrsOptions::<Item>::new(Some(args));
    let tree = Octree::new(points, 2, sphere_surface_leaf_points(tol_id as usize), comm);
    let mut rsrs = Rsrs::new(&tree, options, dim);
    let rsrs_operator = rsrs.get_rsrs_operator(operator.r());
    let factors = rsrs_operator.get_factors();

    let exact = assemble_matrix(operator, dim, TransMode::NoTrans);
    let exact_trans = assemble_matrix(operator, dim, TransMode::Trans);
    let approx = assemble_matrix(&rsrs_operator, dim, TransMode::NoTrans);
    let approx_trans = assemble_matrix(&rsrs_operator, dim, TransMode::Trans);
    let approx_diag_only = replay_joint_left_no_trans(factors, StageApproximation::DiagOnly);
    let approx_diag_id = replay_joint_left_no_trans(factors, StageApproximation::DiagId);
    let approx_full_manual = replay_joint_left_no_trans(factors, StageApproximation::Full);
    let exact_t = transpose(&exact);
    let approx_t = transpose(&approx);

    let exact_norm_2 = spectral_norm(&exact);
    let exact_fro = frobenius_norm(&exact);

    let err = diff(&exact, &approx);
    let trans_err = diff(&exact_trans, &approx_trans);
    let trans_route_err = diff(&approx_trans, &approx_t);
    let exact_symmetry_err = diff(&exact_trans, &exact_t);
    let diag_only_err = diff(&exact, &approx_diag_only);
    let diag_id_err = diff(&exact, &approx_diag_id);
    let manual_full_err = diff(&exact, &approx_full_manual);
    let manual_full_vs_operator = diff(&approx_full_manual, &approx);
    let (_sigma_err, left_singular, right_singular) = leading_svd(&err);
    let left_index_energy = normalized_index_energy(&left_singular);
    let right_index_energy = normalized_index_energy(&right_singular);
    let left_top_leaves = top_leaf_energy(&tree, &left_singular, 5);
    let right_top_leaves = top_leaf_energy(&tree, &right_singular, 5);

    DenseMetrics {
        symmetry_label: label,
        rel_err_2: rel_norm_2(&err, exact_norm_2),
        rel_err_fro: rel_fro(&err, exact_fro),
        trans_rel_err_2: rel_norm_2(&trans_err, exact_norm_2),
        trans_route_rel_err_2: rel_norm_2(&trans_route_err, exact_norm_2),
        exact_symmetry_rel_err_2: rel_norm_2(&exact_symmetry_err, exact_norm_2),
        diag_only_rel_err_2: rel_norm_2(&diag_only_err, exact_norm_2),
        diag_id_rel_err_2: rel_norm_2(&diag_id_err, exact_norm_2),
        manual_full_rel_err_2: rel_norm_2(&manual_full_err, exact_norm_2),
        manual_full_vs_operator_rel_err_2: rel_norm_2(&manual_full_vs_operator, exact_norm_2),
        left_top10_energy: cumulative_energy(&left_index_energy, 10),
        left_top25_energy: cumulative_energy(&left_index_energy, 25),
        right_top10_energy: cumulative_energy(&right_index_energy, 10),
        right_top25_energy: cumulative_energy(&right_index_energy, 25),
        left_top_indices: left_index_energy.into_iter().take(10).collect(),
        right_top_indices: right_index_energy.into_iter().take(10).collect(),
        left_top_leaves,
        right_top_leaves,
    }
}

#[test]
#[ignore = "diagnostic test; run manually with cargo test --test helmholtz_dense_norm -- --ignored --nocapture"]
fn helmholtz_dense_norm_diagnostic() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();

    let structured_operator_params = StructuredOperatorParams::new(
        StructuredOperatorType::KiFMMHelmholtzOperator,
        Precision::Double,
        GeometryType::SphereSurface,
        0.25,
        std::f64::consts::PI,
        1,
        0,
        Assembler::Dense,
    );

    let structured_operator: StructuredOperatorInterface =
        <StructuredOperatorInterface as StructuredOperatorImpl<Item>>::new(
            &structured_operator_params,
        );
    let points = get_bempp_points(&structured_operator).unwrap();
    let operator = StructuredOperator::from_local(structured_operator);

    let symmetric = run_case(&comm, &points, &operator, Symmetry::Symmetric, "Symmetric");
    let no_symm = run_case(&comm, &points, &operator, Symmetry::NoSymm, "NoSymm");

    for metrics in [symmetric, no_symm] {
        println!(
            "{} dense metrics: rel_err_2={:.6e}, rel_err_fro={:.6e}, trans_rel_err_2={:.6e}, trans_route_rel_err_2={:.6e}, exact_symmetry_rel_err_2={:.6e}, diag_only_rel_err_2={:.6e}, diag_id_rel_err_2={:.6e}, manual_full_rel_err_2={:.6e}, manual_full_vs_operator_rel_err_2={:.6e}",
            metrics.symmetry_label,
            metrics.rel_err_2,
            metrics.rel_err_fro,
            metrics.trans_rel_err_2,
            metrics.trans_route_rel_err_2,
            metrics.exact_symmetry_rel_err_2,
            metrics.diag_only_rel_err_2,
            metrics.diag_id_rel_err_2,
            metrics.manual_full_rel_err_2,
            metrics.manual_full_vs_operator_rel_err_2
        );
        println!(
            "{} leading left mode energy: top10={:.3e}, top25={:.3e}, top_indices=[{}], top_leaves=[{}]",
            metrics.symmetry_label,
            metrics.left_top10_energy,
            metrics.left_top25_energy,
            format_index_energies(&metrics.left_top_indices),
            format_leaf_energies(&metrics.left_top_leaves)
        );
        println!(
            "{} leading right mode energy: top10={:.3e}, top25={:.3e}, top_indices=[{}], top_leaves=[{}]",
            metrics.symmetry_label,
            metrics.right_top10_energy,
            metrics.right_top25_energy,
            format_index_energies(&metrics.right_top_indices),
            format_leaf_energies(&metrics.right_top_leaves)
        );
    }
}
