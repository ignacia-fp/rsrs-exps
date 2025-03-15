use std::{fs::{self, File}, io::Read, path::Path};
use bempp_rsrs::rsrs::{box_skeletonisation::{IdTimes, UpdateTimes}, rsrs_cycle::RsrsData, rsrs_factors::{LuTimes, RsrsFactors, RsrsFactorsOps}};
use num::NumCast;
use serde::{Deserialize, Serialize};
use rlst::prelude::*;
use std::io::Write;

type Real<T> = <T as rlst::RlstScalar>::Real;

#[derive(Serialize)]
struct StatsOutput<Item: RlstScalar>{
    app_inv_norm: Real<Item>,
    diag_ae_mean: Real<Item>,
    skel_ae: Real<Item>,
    tot_num_samples: usize,
    total_elapsed_time: u64,
    extraction_time: u128,
    residual_size: usize,
    sampling_time: Vec<u128>,
    id_times: Vec<IdTimes>,
    lu_times: Vec<LuTimes>,
    update_times: Vec<UpdateTimes>
}

#[derive(Serialize)]
struct ErrorsOutput<Item: RlstScalar>{
    rel_errors: Vec<Vec<Real<Item>>>,
    abs_errors: Vec<Vec<Real<Item>>>
}

#[derive(Debug, Deserialize)]
pub struct LuTimesOutput{
    pub io: u128,
    pub extraction: u128,
    pub assembly: u128
}

#[derive(Debug, Deserialize)]
pub struct IdTimesOutput{
    pub nullification: u128,
    pub id: u128
}

#[derive(Debug, Deserialize)]
pub struct UpdateTimesOutput{
    pub id: u128,
    pub lu: u128
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct StatsInput{
    app_inv_norm: f64,
    diag_ae_mean: f64,
    skel_ae: f64,
    tot_num_samples: usize,
    residual_size: usize,
    total_elapsed_time: f64,
    pub extraction_time: f64,
    pub sampling_time: Vec<f64>,
    pub id_times: Vec<IdTimesOutput>,
    pub lu_times: Vec<LuTimesOutput>,
    pub update_times: Vec<UpdateTimesOutput>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ErrorsInput{
    rel_errors: Vec<Vec<f64>>,
    abs_errors: Vec<Vec<f64>>
}

pub fn save_stats<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(_kernel_mat: &mut DynamicArray<Item, 2>, _rsrs_factors: &RsrsFactors<Item>, rsrs_data: &RsrsData<Item>, tol: Real<Item>, path_str: &str)
where <Item as RlstScalar>::Real: for<'a> std::iter::Sum<&'a <Item as RlstScalar>::Real>{    
    fs::create_dir_all(Path::new(&path_str)).unwrap();
    let string_tol = format!("{:e}", tol);
    let mut stats_path = path_str.to_string();
    stats_path.push_str("/stats_");
    stats_path.push_str(&string_tol);
    stats_path.push_str(".json");

    //let norm_app_inv = num::Zero::zero();
    //let diag_ae_mean = num::Zero::zero();
    //let skel_ae = num::Zero::zero();

    let (norm_app_inv, diag_ae_mean, skel_ae) = write_box_errors(_kernel_mat, _rsrs_factors,  tol, &path_str);

    let stats = StatsOutput::<Item>{
        app_inv_norm: norm_app_inv,
        diag_ae_mean: diag_ae_mean,
        skel_ae: skel_ae,
        tot_num_samples: rsrs_data.y_data.num_samples,
        total_elapsed_time: rsrs_data.stats.total_elapsed_time,
        extraction_time: rsrs_data.stats.extraction_time,
        residual_size: rsrs_data.stats.residual_size,
        sampling_time: rsrs_data.stats.sampling_time.clone(),
        id_times: rsrs_data.stats.id_times.clone(),
        lu_times: rsrs_data.stats.lu_times.clone(),
        update_times: rsrs_data.stats.update_times.clone(),
    };
    
    let json_string = serde_json::to_string_pretty(&stats).expect("Failed to serialize");
    let mut file = File::create(stats_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();

}

pub fn write_box_errors<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(kernel_mat: &mut DynamicArray<Item, 2>, rsrs_factors: &RsrsFactors<Item>, tol: <Item as RlstScalar>::Real, path_str: &str)->(<Item as RlstScalar>::Real, <Item as RlstScalar>::Real, <Item as RlstScalar>::Real)
where <Item as RlstScalar>::Real: for<'a> std::iter::Sum<&'a <Item as RlstScalar>::Real>
{
    type Real<Item> = <Item as RlstScalar>::Real;
    let npoints = kernel_mat.shape()[0];
    let (rel_errs, abs_errs) = rsrs_factors.get_boxes_errors(kernel_mat, true);
    let diag_ae = rsrs_factors.get_diag_errors(kernel_mat);

    let errors = ErrorsOutput::<Item>{
        rel_errors: rel_errs,
        abs_errors: abs_errs
    };

    let string_tol = format!("{:e}", tol);
    let mut errors_path = path_str.to_string();
    errors_path.push_str("/errors_");
    errors_path.push_str(&string_tol);
    errors_path.push_str(".json");

    let json_string = serde_json::to_string_pretty(&errors).expect("Failed to serialize");
    let mut file = File::create(errors_path).unwrap();
    file.write_all(json_string.as_bytes()).unwrap();

    let diag_ae_r;
    let diag_ae_s;
    
    if diag_ae.len() >1{
        diag_ae_r = &diag_ae[0..diag_ae.len()-2];
        diag_ae_s = diag_ae[diag_ae.len()-1];
    }
    else{
        diag_ae_r = &[];
        diag_ae_s = diag_ae[0];
    }

    let diag_ae_r_sum = diag_ae_r.iter().sum::<<Item as rlst::RlstScalar>::Real>();
    let len: Real<Item> = NumCast::from(diag_ae_r.len()).unwrap();
    let diag_ae_r_mean = diag_ae_r_sum/len;

    let mut ident = rlst_dynamic_array2!(Item, [npoints, npoints]);
    ident.set_identity();

    let zero = ident - kernel_mat.r();

    (zero.r().norm_1(), diag_ae_r_mean, diag_ae_s)

}

pub enum FileContent {
    Stats(StatsInput),
    Errors(ErrorsInput),
}

pub fn read_file<Item: RlstScalar>(file_type: &str, geometry: &str, kernel: &str, npoints: usize, kappa: Item, tol: Item) -> std::io::Result<FileContent> {
    let mut path_str = "results/".to_string();
    let mut geometry_and_points = geometry.to_string();
    let kappa_string = format!("{:.2}", kappa);
    let tol_string = format!("{:e}", tol);
    geometry_and_points.push('_');
    geometry_and_points.push_str(kernel);
    geometry_and_points.push('_');
    geometry_and_points.push_str(&npoints.to_string());
    geometry_and_points.push('_');
    geometry_and_points.push_str(&kappa_string);
    path_str.push_str(&geometry_and_points);
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
        "stats" => {
            // Deserialize stats JSON into Rust struct
            let stats: StatsInput = serde_json::from_str(&contents).expect("Failed to deserialize stats");
            Ok(FileContent::Stats(stats))
        }
        "errors" => {
            // Deserialize errors JSON into Rust struct
            let errors: ErrorsInput = serde_json::from_str(&contents).expect("Failed to deserialize errors");
            Ok(FileContent::Errors(errors))
        }
        _ => {
            println!("Invalid file type");
            Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid file type"))
        }
    }
}
