use crate::io::errors::{DiffOperator, IdOperator};

use super::errors::rsrs_error_estimator;
use bempp_rsrs::rsrs::rsrs_factors::Inv;
use bempp_rsrs::rsrs::sketch::SamplingSpace;
use bempp_rsrs::rsrs::{
    box_skeletonisation::UpdateTimes,
    rsrs_cycle::Rsrs,
    rsrs_factors::{IdTimes, LuTimes},
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::linalg::naupd::NonSymmetricArnoldiUpdate;
use rlst::dense::linalg::neupd::NonSymmetricArnoldiExtract;
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

type Real<T> = <T as rlst::RlstScalar>::Real;

#[derive(Serialize, Clone)]
pub struct Iterations<Item: RlstScalar> {
    pub no_prec: Option<Vec<Real<Item>>>,
    pub prec: Option<Vec<Real<Item>>>,
    pub rel_err_no_prec: Option<Real<Item>>,
    pub rel_err_prec: Option<Real<Item>>,
}

#[derive(Serialize)]
pub struct ErrorStatsOutput<Item: RlstScalar> {
    app_inv_err_left: Real<Item>,
    app_inv_err_right: Real<Item>,
    app_err_left: Real<Item>,
    app_err_right: Real<Item>,
    norm_2_error: Real<Item>,
    norm_2_error_inv: Real<Item>,
    app_condition_number: Real<Item>,
    tot_num_samples: usize,
    residual_size: usize,
    iterations: Iterations<Item>,
}

type CondType<T> = (Real<T>, Option<(Real<T>, Real<T>)>);

#[derive(Serialize)]
pub struct ConditionNumberOutput<Item: RlstScalar> {
    id: Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
    lu: Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
    dfactors: Vec<(CondType<Item>, Option<CondType<Item>>)>,
}

impl<Item: RlstScalar> ConditionNumberOutput<Item> {
    pub fn new(res: (
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<(CondType<Item>, Option<CondType<Item>>)>,
    )) -> Self {
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
        tot_num_samples: rsrs_data.y_data.num_samples,
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
        elapsed_time_at_limiting: rsrs_data.stats.limiting_factors.limiting_level.elapsed_time,
    };

    let json_string = serde_json::to_string_pretty(&stats).expect("Failed to serialize");
    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();
}

pub fn save_error_stats<
    'a,
    Item: RlstScalar<Real = f64>
        + RandScalar
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixId
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
    iterations: Iterations<Item>,
    tol: Real<Item>,
    path_str: &str,
) where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
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

    rsrs_operator.inv(false);

    let (app_err_left, app_err_right) =
        rsrs_error_estimator(structured_operator_op, rsrs_operator, 10, false);

    let diff = DiffOperator(structured_operator_op.r(), rsrs_operator.r());

    let mut eigs1 = Eigs::new(diff.r(), 1e-10, None, None, None);
    let (sigma_1, _) = eigs1.run(None, 1, None, false);
    
    let mut eigs2 = Eigs::new(structured_operator_op.r(), 1e-10, None, None, None);
    let (sigma_2, _) = eigs2.run(None, 1, None, false);

    let mut eigs3 = Eigs::new(rsrs_operator.r(), 1e-10, None, None, None);
    let (c_1, _) = eigs3.run(None, 1, None, false);

    let norm_2_error = sigma_1[0].abs() / sigma_2[0].abs();

    rsrs_operator.inv(true);

    let domain = std::rc::Rc::clone(&structured_operator_op.domain());
    let range = std::rc::Rc::clone(&structured_operator_op.range());
    let id_op = IdOperator::new(domain, range);

    let prod1 = rsrs_operator.r().product(structured_operator_op.r());
    let diff = DiffOperator(prod1.r(), id_op.r());

    let mut eigs1 = Eigs::new(diff.r(), 1e-10, None, None, None);
    let (sigma_1, _) = eigs1.run(None, 1, None, false);

    let mut eigs2 = Eigs::new(rsrs_operator.r(), 1e-10, None, None, None);
    let (c_2, _) = eigs2.run(None, 1, None, false);

    let norm_2_error_inv = sigma_1[0].abs();

    let condition_number = c_2[0].abs() * c_1[0].abs();


    let (app_inv_err_left, app_inv_err_right) =
        rsrs_error_estimator(structured_operator_op, rsrs_operator, 10, true);

    let stats = ErrorStatsOutput::<Item> {
        app_inv_err_left,
        app_inv_err_right,
        app_err_left,
        app_err_right,
        app_condition_number: condition_number,
        norm_2_error,
        norm_2_error_inv,
        tot_num_samples: rsrs_data.y_data.num_samples,
        residual_size: rsrs_data.stats.residual_size,
        iterations,
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
