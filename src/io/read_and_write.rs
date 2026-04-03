use super::errors::rsrs_error_estimator;
use crate::io::errors::frobenius_diff_and_reference_norm;
use crate::io::errors::{DiffOperator, IdOperator, NormalOperator};
use bempp_rsrs::rsrs::rsrs_cycle::Rsrs;
use bempp_rsrs::rsrs::rsrs_factors::rsrs_operator::Inv;
use bempp_rsrs::rsrs::sketch::SampleType;
use bempp_rsrs::rsrs::sketch::SamplingSpace;
use bempp_rsrs::rsrs::statistics::LevelEffort;
use bempp_rsrs::rsrs::statistics::{IdTimes, LuTimes, UpdateTimes};
use hdf5::{File as Hdf5File, Group as Hdf5Group};
use num::complex::Complex;
use num::{FromPrimitive, NumCast};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{
    dense::{
        linalg::{interpolative_decomposition::MatrixIdNoSkel, lu::MatrixLu},
        tools::RandScalar,
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

type Real<T> = <T as rlst::RlstScalar>::Real;
const POWER_ITERATIONS: usize = 20;
const FROBENIUS_ESTIMATION_SAMPLES: usize = 20;
const ADJOINT_CHECK_SAMPLES: usize = 8;

#[derive(Serialize, Clone)]
pub struct Solves<Item: RlstScalar> {
    pub no_prec: Option<Vec<Vec<Real<Item>>>>,
    pub prec: Option<Vec<Vec<Real<Item>>>>,
    pub rel_err_no_prec: Option<Vec<Real<Item>>>,
    pub rel_err_prec: Option<Vec<Real<Item>>>,
    #[serde(skip_serializing)]
    pub sols_no_prec: Option<Vec<Vec<Item>>>,
    #[serde(skip_serializing)]
    pub sols_prec: Option<Vec<Vec<Item>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vectors_file: Option<String>,
}

#[derive(Serialize)]
pub struct ErrorStatsOutput<Item: RlstScalar> {
    app_inv_err_left: Real<Item>,
    app_inv_err_right: Real<Item>,
    app_err_left: Real<Item>,
    app_err_right: Real<Item>,
    adjoint_consistency_error: Real<Item>,
    norm_fro_error: Real<Item>,
    norm_fro_error_inv: Real<Item>,
    norm_fro_operator: Real<Item>,
    norm_2_error: Real<Item>,
    norm_2_error_inv: Real<Item>,
    norm_2_operator: Real<Item>,
    solve_error: Real<Item>,
    cond_rsrs_estimate: Real<Item>,
    tot_num_samples: usize,
    residual_size: usize,
    solves: Solves<Item>,
}
type CNTuple<T> = (Real<T>, Real<T>);
type CondType<T> = (CNTuple<T>, Option<(CNTuple<T>, CNTuple<T>)>);

#[derive(Serialize)]
pub struct ConditionNumberOutput<Item: RlstScalar> {
    id: Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
    lu: Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
    dfactors: Vec<(CondType<Item>, Option<CondType<Item>>)>,
}

impl<Item: RlstScalar> ConditionNumberOutput<Item> {
    pub fn new(
        res: (
            Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
            Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
            Vec<(CondType<Item>, Option<CondType<Item>>)>,
        ),
    ) -> Self {
        let (id, lu, dfactors) = res;
        Self { id, lu, dfactors }
    }
    pub fn save(&self, path_str: &str, tol: Real<Item>) {
        fs::create_dir_all(Path::new(&path_str)).unwrap();
        let string_tol = format!("{:e}", tol);
        let mut stats_path = path_str.to_string();
        stats_path.push_str("/condition_number_stats_");
        stats_path.push_str(&string_tol);
        stats_path.push_str(".json");

        let json_string = serde_json::to_string_pretty(&self).expect("Failed to serialize");
        let mut file = File::create(stats_path).unwrap();
        file.write_all(json_string.as_bytes()).unwrap();
    }
}

#[derive(Serialize)]
struct TimeStatsOutput {
    tot_num_samples: usize,
    min_samples: usize,
    max_level: usize,
    limiting_level: usize,
    leaf_count: usize,
    dim: usize,
    max_boxes: usize,
    max_points: usize,
    elapsed_time_at_limiting: u128,
    total_elapsed_time: u128,
    total_elapsed_time_wo_sampling: u128,
    sampling_extraction_time: u128,
    extraction_time: u128,
    sampling_time: Vec<u128>,
    id_times: Vec<IdTimes>,
    tot_id_time: u128,
    tot_lu_time: u128,
    lu_times: Vec<LuTimes>,
    update_times: Vec<UpdateTimes>,
    index_calculation: u128,
    sorting_near_field: u128,
    residual_calculation: u128,
    level_effort: Vec<LevelEffort>,
    mv_avg_time: Vec<u128>,
    run_start_rss_bytes: Option<u64>,
    max_sample_buffer_bytes: u64,
    max_factor_bytes: u64,
    max_accounted_factorization_bytes: u64,
    max_estimated_temporary_runtime_bytes: Option<u64>,
    memory_snapshots: Vec<MemorySnapshotOutput>,
}

#[derive(Serialize)]
struct RankStatsOutput {
    residual_size: usize,
    ranks: Vec<usize>,
    compression: Vec<f64>,
    box_sizes: Vec<usize>,
    near_field_sizes: Vec<usize>,
    dec_boxes_per_level: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct LuTimesOutput {
    pub extraction: u128,
    pub lu: u128,
}

#[derive(Debug, Deserialize)]
pub struct IdTimesOutput {
    pub nullification: u128,
    pub id: u128,
}

#[derive(Debug, Deserialize)]
pub struct UpdateTimesOutput {
    pub id: u128,
    pub lu: u128,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct RankStatsInput {
    residual_size: usize,
    ranks: Vec<usize>,
    compression: Vec<f64>,
    box_sizes: Vec<usize>,
    near_field_sizes: Vec<usize>,
    pub dec_boxes_per_level: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct FactorMemoryStatsOutput {
    total_bytes: u64,
    id_bytes: u64,
    lu_bytes: u64,
    diag_bytes: u64,
    perm_bytes: u64,
    id_count: usize,
    lu_count: usize,
    diag_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MemorySnapshotOutput {
    label: String,
    rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    baseline_rss_bytes: Option<u64>,
    sample_buffer_bytes: u64,
    factor_memory: FactorMemoryStatsOutput,
    accounted_factorization_bytes: u64,
    estimated_temporary_runtime_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct TimeStatsInput {
    total_elapsed_time: f64,
    pub extraction_time: f64,
    pub sampling_time: Vec<f64>,
    pub sampling_extraction_time: f64,
    pub id_times: Vec<IdTimesOutput>,
    pub tot_id_time: f64,
    pub tot_lu_time: f64,
    pub lu_times: Vec<LuTimesOutput>,
    pub update_times: Vec<UpdateTimesOutput>,
    pub index_calculation: f64,
    pub sorting_near_field: f64,
    pub residual_calculation: f64,
    #[serde(default)]
    pub run_start_rss_bytes: Option<u64>,
    #[serde(default)]
    pub max_sample_buffer_bytes: u64,
    #[serde(default)]
    pub max_factor_bytes: u64,
    #[serde(default)]
    pub max_accounted_factorization_bytes: u64,
    #[serde(default)]
    pub max_estimated_temporary_runtime_bytes: Option<u64>,
    #[serde(default)]
    pub memory_snapshots: Vec<MemorySnapshotOutput>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ErrorsInput {
    rel_errors: Vec<Vec<f64>>,
    abs_errors: Vec<Vec<f64>>,
}

pub fn save_time_stats<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(
    rsrs_data: &Rsrs<Item>,
    tol: Real<Item>,
    path_str: &str,
) where
    <Item as RlstScalar>::Real: for<'a> std::iter::Sum<&'a <Item as RlstScalar>::Real>,
{
    fs::create_dir_all(Path::new(&path_str)).unwrap();
    let string_tol = format!("{:e}", tol);
    let mut stats_path = path_str.to_string();
    stats_path.push_str("/time_stats_");
    stats_path.push_str(&string_tol);
    stats_path.push_str(".json");

    let stats = TimeStatsOutput {
        tot_num_samples: rsrs_data.active_samples,
        total_elapsed_time: rsrs_data.stats.total_elapsed_time,
        total_elapsed_time_wo_sampling: rsrs_data.stats.total_elapsed_time_wo_sampling,
        dim: rsrs_data.stats.dim,
        extraction_time: rsrs_data.stats.extraction_time,
        sampling_extraction_time: rsrs_data.stats.sampling_extraction_time,
        sampling_time: rsrs_data.stats.sampling_time.clone(),
        id_times: rsrs_data.stats.id_times.clone(),
        tot_id_time: rsrs_data.stats.tot_id_time,
        tot_lu_time: rsrs_data.stats.tot_lu_time,
        lu_times: rsrs_data.stats.lu_times.clone(),
        update_times: rsrs_data.stats.update_times.clone(),
        index_calculation: rsrs_data.stats.index_calculation,
        sorting_near_field: rsrs_data.stats.sorting_near_field,
        residual_calculation: rsrs_data.stats.residual_calculation,
        max_level: rsrs_data.stats.limiting_factors.max_level,
        min_samples: rsrs_data.stats.limiting_factors.min_samples,
        max_boxes: rsrs_data.stats.limiting_factors.limiting_level.num_boxes,
        max_points: rsrs_data
            .stats
            .limiting_factors
            .limiting_level
            .active_points,
        limiting_level: rsrs_data.stats.limiting_factors.limiting_level.level,
        leaf_count: rsrs_data.stats.limiting_factors.leaf_count,
        elapsed_time_at_limiting: rsrs_data.stats.limiting_factors.limiting_level.elapsed_time,
        level_effort: rsrs_data.stats.level_effort.clone(),
        mv_avg_time: rsrs_data.stats.mv_avg_time.clone(),
        run_start_rss_bytes: rsrs_data.stats.run_start_rss_bytes,
        max_sample_buffer_bytes: rsrs_data.stats.max_sample_buffer_bytes,
        max_factor_bytes: rsrs_data.stats.max_factor_bytes,
        max_accounted_factorization_bytes: rsrs_data.stats.max_accounted_factorization_bytes,
        max_estimated_temporary_runtime_bytes: rsrs_data
            .stats
            .max_estimated_temporary_runtime_bytes,
        memory_snapshots: rsrs_data
            .stats
            .memory_snapshots
            .iter()
            .map(|snapshot| MemorySnapshotOutput {
                label: snapshot.label.clone(),
                rss_bytes: snapshot.rss_bytes,
                peak_rss_bytes: snapshot.peak_rss_bytes,
                baseline_rss_bytes: snapshot.baseline_rss_bytes,
                sample_buffer_bytes: snapshot.sample_buffer_bytes,
                factor_memory: FactorMemoryStatsOutput {
                    total_bytes: snapshot.factor_memory.total_bytes,
                    id_bytes: snapshot.factor_memory.id_bytes,
                    lu_bytes: snapshot.factor_memory.lu_bytes,
                    diag_bytes: snapshot.factor_memory.diag_bytes,
                    perm_bytes: snapshot.factor_memory.perm_bytes,
                    id_count: snapshot.factor_memory.id_count,
                    lu_count: snapshot.factor_memory.lu_count,
                    diag_count: snapshot.factor_memory.diag_count,
                },
                accounted_factorization_bytes: snapshot.accounted_factorization_bytes,
                estimated_temporary_runtime_bytes: snapshot.estimated_temporary_runtime_bytes,
            })
            .collect(),
    };

    let json_string = serde_json::to_string_pretty(&stats).expect("Failed to serialize");
    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();
}

thread_local! {
    static THREAD_RNG: RefCell<ChaCha8Rng> = RefCell::new(init_rng());
}

fn init_rng() -> ChaCha8Rng {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seed = (time_seed as u64).wrapping_mul(0x9E3779B97F4A7C15); // or add thread ID if needed
    ChaCha8Rng::seed_from_u64(seed)
}

pub fn with_thread_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut ChaCha8Rng) -> R,
{
    THREAD_RNG.with(|rng_cell| {
        let mut rng = rng_cell.borrow_mut();
        f(&mut rng)
    })
}

fn estimate_normal_operator_norm_sq<
    Item: RlstScalar + RandScalar,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    operator: &OpImpl,
    iterations: usize,
) -> Real<Item>
where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: InnerProductSpace,
{
    let mut iterate = SamplingSpace::zero(operator.domain());
    with_thread_rng(|rng| {
        operator
            .domain()
            .sampling(&mut iterate, rng, SampleType::RealStandardNormal);
    });

    let mut iterate_norm = iterate.norm();
    if iterate_norm == num::Zero::zero() {
        return num::Zero::zero();
    }
    let inv_norm: Real<Item> = <Real<Item> as num::One>::one() / iterate_norm;
    iterate.scale_inplace(Item::from_real(inv_norm));

    let mut estimate = num::Zero::zero();
    for _ in 0..iterations {
        let next = operator.apply(iterate.r(), TransMode::NoTrans);
        estimate = next.norm();
        if estimate == num::Zero::zero() {
            return num::Zero::zero();
        }

        iterate = operator.domain().clone_vec(&next);
        iterate_norm = iterate.norm();
        if iterate_norm == num::Zero::zero() {
            return num::Zero::zero();
        }
        let inv_norm: Real<Item> = <Real<Item> as num::One>::one() / iterate_norm;
        iterate.scale_inplace(Item::from_real(inv_norm));
    }

    estimate
}

fn estimate_adjoint_consistency_error<
    Item: RlstScalar + RandScalar,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    operator: &OpImpl,
    sample_size: usize,
) -> Real<Item>
where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: InnerProductSpace,
{
    let mut max_rel = 0.0f64;

    for _ in 0..sample_size.max(1) {
        let mut x = SamplingSpace::zero(operator.range());
        let mut y = SamplingSpace::zero(operator.domain());
        with_thread_rng(|rng| {
            operator
                .range()
                .sampling(&mut x, rng, SampleType::StandardNormal);
            operator
                .domain()
                .sampling(&mut y, rng, SampleType::StandardNormal);
        });

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

    Real::<Item>::from_f64(max_rel).unwrap()
}

pub(crate) trait SolveVectorArchive<T: RlstScalar> {
    fn write_matrix(group: &Hdf5Group, data: &[Vec<T>]) -> hdf5::Result<()>;
}

fn matrix_shape<T>(data: &[Vec<T>]) -> hdf5::Result<(usize, usize)> {
    let rows = data.len();
    let cols = data.first().map_or(0, Vec::len);

    if data.iter().any(|row| row.len() != cols) {
        return Err(hdf5::Error::Internal(
            "all archived vectors must have the same length".into(),
        ));
    }

    Ok((rows, cols))
}

macro_rules! implement_solve_vector_archive_real {
    ($scalar:ty) => {
        impl SolveVectorArchive<$scalar> for $scalar {
            fn write_matrix(group: &Hdf5Group, data: &[Vec<$scalar>]) -> hdf5::Result<()> {
                let (rows, cols) = matrix_shape(data)?;
                let flat: Vec<$scalar> = data.iter().flat_map(|row| row.iter().copied()).collect();

                group
                    .new_dataset::<$scalar>()
                    .shape((rows, cols))
                    .create("real")?
                    .write_raw(&flat)?;

                Ok(())
            }
        }
    };
}

macro_rules! implement_solve_vector_archive_complex {
    ($scalar:ty) => {
        impl SolveVectorArchive<Complex<$scalar>> for Complex<$scalar> {
            fn write_matrix(group: &Hdf5Group, data: &[Vec<Complex<$scalar>>]) -> hdf5::Result<()> {
                let (rows, cols) = matrix_shape(data)?;
                let mut real = Vec::with_capacity(rows * cols);
                let mut imag = Vec::with_capacity(rows * cols);

                for row in data {
                    for value in row {
                        real.push(value.re);
                        imag.push(value.im);
                    }
                }

                group
                    .new_dataset::<$scalar>()
                    .shape((rows, cols))
                    .create("real")?
                    .write_raw(&real)?;
                group
                    .new_dataset::<$scalar>()
                    .shape((rows, cols))
                    .create("imag")?
                    .write_raw(&imag)?;

                Ok(())
            }
        }
    };
}

implement_solve_vector_archive_real!(f32);
implement_solve_vector_archive_real!(f64);
implement_solve_vector_archive_complex!(f32);
implement_solve_vector_archive_complex!(f64);

fn write_optional_vector_group<Item>(
    file: &Hdf5File,
    name: &str,
    data: Option<&[Vec<Item>]>,
) -> hdf5::Result<()>
where
    Item: RlstScalar + SolveVectorArchive<Item>,
{
    let Some(data) = data else {
        return Ok(());
    };

    if data.is_empty() {
        return Ok(());
    }

    let group = file.create_group(name)?;
    <Item as SolveVectorArchive<Item>>::write_matrix(&group, data)
}

fn save_solve_vectors<Item>(
    solves: &Solves<Item>,
    tol: Real<Item>,
    path_str: &str,
) -> Option<String>
where
    Item: RlstScalar + SolveVectorArchive<Item>,
{
    let has_vectors = solves.sols_no_prec.is_some() || solves.sols_prec.is_some();

    if !has_vectors {
        return None;
    }

    let file_name = format!("solve_vectors_{:e}.h5", tol);
    let file_path = Path::new(path_str).join(&file_name);
    let file = Hdf5File::create(&file_path).unwrap();

    write_optional_vector_group(&file, "sols_no_prec", solves.sols_no_prec.as_deref()).unwrap();
    write_optional_vector_group(&file, "sols_prec", solves.sols_prec.as_deref()).unwrap();

    Some(file_name)
}

pub(crate) fn save_error_stats<
    'a,
    Item: RlstScalar
        + RandScalar
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixId
        + MatrixIdNoSkel
        + MatrixLu
        + MatrixQr,
    Space: SamplingSpace<F = Item> + IndexableSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    OpImpl2: AsApply<Domain = Space, Range = Space> + Inv,
>(
    structured_operator_op: &OpImpl,
    rsrs_operator: &mut OpImpl2,
    rsrs_data: &Rsrs<Item>,
    solves: Solves<Item>,
    tol: Real<Item>,
    path_str: &str,
    transpose_matches_apply: bool,
    _complex_symmetric_rsrs: bool,
) where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    Item: SolveVectorArchive<Item>,
    <<Space as rlst::LinearSpace>::E as rlst::ElementImpl>::Space: InnerProductSpace,
    IdOperator<Item, Space>: rlst::AsApply,
    IdOperator<Item, Space>: OperatorBase<Domain = Space, Range = Space>,
{
    fs::create_dir_all(Path::new(&path_str)).unwrap();
    let string_tol = format!("{:e}", tol);
    let mut stats_path = path_str.to_string();
    stats_path.push_str("/error_stats_");
    stats_path.push_str(&string_tol);
    stats_path.push_str(".json");
    let mut solves = solves;
    solves.vectors_file = save_solve_vectors(&solves, tol, path_str);

    rsrs_operator.inv(false);
    let (app_err_left, app_err_right) =
        rsrs_error_estimator(structured_operator_op, rsrs_operator, 10, false);
    let adjoint_consistency_error =
        estimate_adjoint_consistency_error(rsrs_operator, ADJOINT_CHECK_SAMPLES);
    // Treat the approximation and error operators as general operators for norm
    // estimation, even when the target operator is symmetric. The RSRS
    // approximation is not guaranteed to preserve symmetry exactly.
    let approx_transpose_matches_apply = false;
    let (norm_fro_error, norm_fro_operator) =
        frobenius_diff_and_reference_norm(
            structured_operator_op,
            rsrs_operator,
            FROBENIUS_ESTIMATION_SAMPLES,
        );

    let diff = DiffOperator(structured_operator_op.r(), rsrs_operator.r());
    let normal = NormalOperator {
        op: diff.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };
    let sigma_1: Real<Item> =
        estimate_normal_operator_norm_sq(&normal.r(), POWER_ITERATIONS);

    let normal_structured = NormalOperator {
        op: structured_operator_op.r(),
        transpose_matches_apply,
    };
    let sigma_2: Real<Item> =
        estimate_normal_operator_norm_sq(&normal_structured.r(), POWER_ITERATIONS);

    let normal_rsrs = NormalOperator {
        op: rsrs_operator.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };
    let c_1: Real<Item> =
        estimate_normal_operator_norm_sq(&normal_rsrs.r(), POWER_ITERATIONS);

    let norm_2_error = sigma_1.abs().sqrt() / sigma_2.abs().sqrt();

    rsrs_operator.inv(true);

    let domain = std::rc::Rc::clone(&structured_operator_op.domain());
    let range = std::rc::Rc::clone(&structured_operator_op.range());
    let id_op = IdOperator::new(domain, range);

    let prod1 = rsrs_operator.r().product(structured_operator_op.r());
    let (norm_fro_error_inv, _) =
        frobenius_diff_and_reference_norm(&id_op, &prod1, FROBENIUS_ESTIMATION_SAMPLES);
    let diff = DiffOperator(prod1.r(), id_op.r());
    let normal = NormalOperator {
        op: diff.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };

    let sigma_1: Real<Item> =
        estimate_normal_operator_norm_sq(&normal.r(), POWER_ITERATIONS);

    let normal_rsrs = NormalOperator {
        op: rsrs_operator.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };
    let c_2: Real<Item> =
        estimate_normal_operator_norm_sq(&normal_rsrs.r(), POWER_ITERATIONS);

    let norm_2_error_inv = sigma_1.abs().sqrt();

    let condition_number = c_2.abs().sqrt() * c_1.abs().sqrt();

    let (app_inv_err_left, app_inv_err_right) =
        rsrs_error_estimator(structured_operator_op, rsrs_operator, 10, true);

    let mut sol = SamplingSpace::zero(structured_operator_op.r().domain());

    with_thread_rng(|rng| {
        structured_operator_op
            .r()
            .domain()
            .sampling(&mut sol, rng, SampleType::RealStandardNormal);
    });

    let rhs = structured_operator_op.apply(sol.r(), TransMode::NoTrans);

    let mut sol_app = rsrs_operator.apply(rhs, TransMode::NoTrans);

    sol_app.sub_inplace(sol.r());

    let solve_error = sol_app.norm() / sol.norm();

    let stats = ErrorStatsOutput::<Item> {
        app_inv_err_left,
        app_inv_err_right,
        app_err_left,
        app_err_right,
        adjoint_consistency_error,
        cond_rsrs_estimate: condition_number,
        norm_fro_error,
        norm_fro_error_inv,
        norm_fro_operator,
        norm_2_operator: sigma_2.abs().sqrt(),
        norm_2_error,
        norm_2_error_inv,
        solve_error,
        tot_num_samples: rsrs_data.y_data.num_samples,
        residual_size: rsrs_data.stats.residual_size,
        solves,
    };

    let json_string = serde_json::to_string_pretty(&stats).expect("Failed to serialize");
    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();
}

pub fn save_rank_stats<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(
    rsrs_data: &Rsrs<Item>,
    tol: Real<Item>,
    path_str: &str,
) where
    <Item as RlstScalar>::Real: for<'a> std::iter::Sum<&'a <Item as RlstScalar>::Real>,
{
    fs::create_dir_all(Path::new(&path_str)).unwrap();
    let string_tol = format!("{:e}", tol);
    let mut stats_path = path_str.to_string();
    stats_path.push_str("/rank_stats_");
    stats_path.push_str(&string_tol);
    stats_path.push_str(".json");

    let compression = rsrs_data
        .stats
        .box_sizes
        .iter()
        .zip(rsrs_data.stats.ranks.iter())
        .map(|(x, y)| 1.0 - (*y as f64 / *x as f64))
        .collect();

    let stats = RankStatsOutput {
        residual_size: rsrs_data.stats.residual_size,
        ranks: rsrs_data.stats.ranks.clone(),
        compression,
        box_sizes: rsrs_data.stats.box_sizes.clone(),
        near_field_sizes: rsrs_data.stats.near_field_sizes.clone(),
        dec_boxes_per_level: rsrs_data.stats.dec_boxes_per_level.clone(),
    };

    let json_string = serde_json::to_string_pretty(&stats).expect("Failed to serialize");
    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();
}

pub enum FileContent {
    RankStats(RankStatsInput),
    TimeStats(TimeStatsInput),
    Errors(ErrorsInput),
}

pub fn read_file<Item: RlstScalar>(
    path_str: &str,
    file_type: &str,
    tol: Item,
) -> std::io::Result<FileContent> {
    let tol_string = format!("{:e}", tol);
    let mut path_str = path_str.to_string();
    path_str.push_str("/");
    path_str.push_str(file_type);
    path_str.push_str("_");
    path_str.push_str(&tol_string);
    path_str.push_str(".json");

    // Open the JSON file
    let mut file = File::open(path_str)?;
    let mut contents = String::new();

    // Read the file content into a string
    file.read_to_string(&mut contents)?;

    match file_type {
        "time_stats" => {
            // Deserialize stats JSON into Rust struct
            let stats: TimeStatsInput =
                serde_json::from_str(&contents).expect("Failed to deserialize stats");
            Ok(FileContent::TimeStats(stats))
        }
        "rank_stats" => {
            // Deserialize stats JSON into Rust struct
            let stats: RankStatsInput =
                serde_json::from_str(&contents).expect("Failed to deserialize stats");
            Ok(FileContent::RankStats(stats))
        }
        "errors" => {
            let errors: ErrorsInput =
                serde_json::from_str(&contents).expect("Failed to deserialize errors");
            Ok(FileContent::Errors(errors))
        }
        _ => {
            println!("Invalid file type");
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid file type",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::errors::NormalOperator;
    use num::NumCast;
    use std::rc::Rc;

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

    fn jacobi_largest_eigenvalue_real(matrix: &[Vec<f64>]) -> f64 {
        let n = matrix.len();
        let mut a = matrix.to_vec();

        for _ in 0..200 {
            let mut p = 0usize;
            let mut q = 1usize;
            let mut max_offdiag = 0.0f64;

            for row in 0..n {
                for col in (row + 1)..n {
                    let value = a[row][col].abs();
                    if value > max_offdiag {
                        max_offdiag = value;
                        p = row;
                        q = col;
                    }
                }
            }

            if max_offdiag <= 1.0e-12 {
                break;
            }

            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            for k in 0..n {
                if k != p && k != q {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[p][k] = a[k][p];
                    a[k][q] = s * akp + c * akq;
                    a[q][k] = a[k][q];
                }
            }

            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }

        a.iter()
            .enumerate()
            .map(|(idx, row)| row[idx])
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn exact_spectral_norm_complex(arr: &DynamicArray<Complex<f64>, 2>) -> f64 {
        let rows = arr.shape()[0];
        let cols = arr.shape()[1];
        let mut gram = vec![vec![Complex::<f64>::new(0.0, 0.0); cols]; cols];

        for i in 0..cols {
            for j in i..cols {
                let mut value = Complex::<f64>::new(0.0, 0.0);
                for row in 0..rows {
                    value += arr[[row, i]].conj() * arr[[row, j]];
                }
                gram[i][j] = value;
                gram[j][i] = value.conj();
            }
        }

        let mut realified = vec![vec![0.0f64; 2 * cols]; 2 * cols];
        for i in 0..cols {
            for j in 0..cols {
                let value = gram[i][j];
                realified[i][j] = value.re;
                realified[i][j + cols] = -value.im;
                realified[i + cols][j] = value.im;
                realified[i + cols][j + cols] = value.re;
            }
        }

        jacobi_largest_eigenvalue_real(&realified).sqrt()
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
                let im = -1.4 * u * v
                    + 0.05 * ((2 * row + col + 1) as f64).cos()
                    - 0.03 * ((row + col + 2) as f64).sin();
                arr[[row, col]] = Complex::new(re, im);
            }
        }
    }

    #[test]
    fn adjoint_consistency_error_is_small_for_dense_complex_operator() {
        let mut matrix = rlst_dynamic_array2!(Complex<f64>, [10, 10]);
        populate_complex_test_matrix(&mut matrix);

        let op = DenseMatrixOperator::new(matrix);
        let adjoint_error: f64 =
            NumCast::from(estimate_adjoint_consistency_error(&op, 32)).unwrap();

        assert!(
            adjoint_error <= 1.0e-12,
            "dense operator adjoint consistency check too large: {adjoint_error}"
        );
    }

    #[test]
    fn power_iteration_normal_estimator_matches_dense_spectral_norm() {
        let mut matrix = rlst_dynamic_array2!(Complex<f64>, [10, 10]);
        populate_complex_test_matrix(&mut matrix);

        let exact_norm = exact_spectral_norm_complex(&matrix);
        let op = DenseMatrixOperator::new(matrix);
        let normal = NormalOperator {
            op: op.r(),
            transpose_matches_apply: false,
        };

        let estimated_norm_sq =
            estimate_normal_operator_norm_sq(&normal.r(), 40);
        let estimated_norm: f64 = NumCast::from(estimated_norm_sq.sqrt()).unwrap();
        let rel_err = (estimated_norm - exact_norm).abs() / exact_norm;

        assert!(
            rel_err <= 0.05,
            "power-iteration 2-norm estimate too far from exact: estimated={estimated_norm}, exact={exact_norm}, rel_err={rel_err}"
        );
    }
}
