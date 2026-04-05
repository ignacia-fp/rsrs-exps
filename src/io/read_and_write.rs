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
use num::FromPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::linalg::naupd::NonSymmetricArnoldiUpdate;
use rlst::dense::linalg::neupd::NonSymmetricArnoldiExtract;
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
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

type Real<T> = <T as rlst::RlstScalar>::Real;

fn start_save_stage(label: &str) -> Instant {
    println!("[rsrs-exps][save] {label}...");
    Instant::now()
}

fn finish_save_stage(label: &str, start: Instant) {
    println!(
        "[rsrs-exps][save] {label} done in {:.3}s",
        start.elapsed().as_secs_f64()
    );
}

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
        + MatrixQr
        + NonSymmetricArnoldiUpdate
        + NonSymmetricArnoldiExtract,
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
    complex_symmetric_rsrs: bool,
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
    println!(
        "[rsrs-exps][save] save_error_stats start: tol={:e}, path='{}'",
        tol, path_str
    );
    fs::create_dir_all(Path::new(&path_str)).unwrap();
    let string_tol = format!("{:e}", tol);
    let mut stats_path = path_str.to_string();
    stats_path.push_str("/error_stats_");
    stats_path.push_str(&string_tol);
    stats_path.push_str(".json");
    let mut solves = solves;
    let stage = start_save_stage("write solve-vector sidecar");
    solves.vectors_file = save_solve_vectors(&solves, tol, path_str);
    finish_save_stage("write solve-vector sidecar", stage);

    rsrs_operator.inv(false);
    let tol_eigs = <Space::F as RlstScalar>::Real::from_f64(1e-10).unwrap();
    let stage = start_save_stage("application error estimate (non-inverse)");
    let (app_err_left, app_err_right) =
        rsrs_error_estimator(structured_operator_op, rsrs_operator, 10, false);
    finish_save_stage("application error estimate (non-inverse)", stage);
    let approx_transpose_matches_apply = transpose_matches_apply && !complex_symmetric_rsrs;
    let stage = start_save_stage("frobenius norm comparison (non-inverse)");
    let (norm_fro_error, norm_fro_operator) =
        frobenius_diff_and_reference_norm(structured_operator_op, rsrs_operator);
    finish_save_stage("frobenius norm comparison (non-inverse)", stage);

    let diff = DiffOperator(structured_operator_op.r(), rsrs_operator.r());
    let normal = NormalOperator {
        op: diff.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };
    let stage = start_save_stage("largest singular value of operator difference");
    let mut eigs1 = Eigs::new(normal.r(), tol_eigs, None, None, None);
    let (sigma_1, _) = eigs1.run(None, 1, None, false);
    finish_save_stage("largest singular value of operator difference", stage);

    let normal_structured = NormalOperator {
        op: structured_operator_op.r(),
        transpose_matches_apply,
    };
    let stage = start_save_stage("largest singular value of original operator");
    let mut eigs2 = Eigs::new(normal_structured.r(), tol_eigs, None, None, None);
    let (sigma_2, _) = eigs2.run(None, 1, None, false);
    finish_save_stage("largest singular value of original operator", stage);

    let normal_rsrs = NormalOperator {
        op: rsrs_operator.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };
    let stage = start_save_stage("largest singular value of RSRS operator");
    let mut eigs3 = Eigs::new(normal_rsrs.r(), tol_eigs, None, None, None);
    let (c_1, _) = eigs3.run(None, 1, None, false);
    finish_save_stage("largest singular value of RSRS operator", stage);

    let norm_2_error = sigma_1[0].abs().sqrt() / sigma_2[0].abs().sqrt();

    rsrs_operator.inv(true);

    let domain = std::rc::Rc::clone(&structured_operator_op.domain());
    let range = std::rc::Rc::clone(&structured_operator_op.range());
    let id_op = IdOperator::new(domain, range);

    let prod1 = rsrs_operator.r().product(structured_operator_op.r());
    let stage = start_save_stage("frobenius norm comparison (inverse mode)");
    let (norm_fro_error_inv, _) = frobenius_diff_and_reference_norm(&id_op, &prod1);
    finish_save_stage("frobenius norm comparison (inverse mode)", stage);
    let diff = DiffOperator(prod1.r(), id_op.r());
    let normal = NormalOperator {
        op: diff.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };

    let stage = start_save_stage("largest singular value of inverse residual");
    let mut eigs1 = Eigs::new(normal.r(), tol_eigs, None, None, None);
    let (sigma_1, _) = eigs1.run(None, 1, None, false);
    finish_save_stage("largest singular value of inverse residual", stage);

    let normal_rsrs = NormalOperator {
        op: rsrs_operator.r(),
        transpose_matches_apply: approx_transpose_matches_apply,
    };
    let stage = start_save_stage("largest singular value of inverse RSRS operator");
    let mut eigs2 = Eigs::new(normal_rsrs.r(), tol_eigs, None, None, None);
    let (c_2, _) = eigs2.run(None, 1, None, false);
    finish_save_stage("largest singular value of inverse RSRS operator", stage);

    let norm_2_error_inv = sigma_1[0].abs().sqrt();

    let condition_number = c_2[0].abs().sqrt() * c_1[0].abs().sqrt();

    let (app_inv_err_left, app_inv_err_right) = {
        let stage = start_save_stage("application error estimate (inverse)");
        let res = rsrs_error_estimator(structured_operator_op, rsrs_operator, 10, true);
        finish_save_stage("application error estimate (inverse)", stage);
        res
    };

    let mut sol = SamplingSpace::zero(structured_operator_op.r().domain());

    with_thread_rng(|rng| {
        structured_operator_op
            .r()
            .domain()
            .sampling(&mut sol, rng, SampleType::RealStandardNormal);
    });

    let stage = start_save_stage("sampled solve error check");
    let rhs = structured_operator_op.apply(sol.r(), TransMode::NoTrans);

    let mut sol_app = rsrs_operator.apply(rhs, TransMode::NoTrans);

    sol_app.sub_inplace(sol.r());

    let solve_error = sol_app.norm() / sol.norm();
    finish_save_stage("sampled solve error check", stage);

    let stats = ErrorStatsOutput::<Item> {
        app_inv_err_left,
        app_inv_err_right,
        app_err_left,
        app_err_right,
        cond_rsrs_estimate: condition_number,
        norm_fro_error,
        norm_fro_error_inv,
        norm_fro_operator,
        norm_2_operator: sigma_2[0].abs().sqrt(),
        norm_2_error,
        norm_2_error_inv,
        solve_error,
        tot_num_samples: rsrs_data.y_data.num_samples,
        residual_size: rsrs_data.stats.residual_size,
        solves,
    };

    let stage = start_save_stage("serialize and write error_stats json");
    let json_string = serde_json::to_string_pretty(&stats).expect("Failed to serialize");
    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();
    finish_save_stage("serialize and write error_stats json", stage);
    println!("[rsrs-exps][save] save_error_stats finished");
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
