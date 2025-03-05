use std::{fs::{self, File}, path::Path};
use bempp_rsrs::rsrs::{rsrs_cycle::RsrsData, rsrs_factors::{RsrsFactors, RsrsFactorsOps}};
use num::NumCast;
use serde::Serialize;
use rlst::prelude::*;
use std::io::Write;

type Real<T> = <T as rlst::RlstScalar>::Real;


#[derive(Serialize)]
struct Stats<Item: RlstScalar>{
    app_inv_norm: Real<Item>,
    diag_ae_mean: Real<Item>,
    skel_ae: Real<Item>,
    tot_num_samples: usize,
    total_elapsed_time: u64,
    extraction_time: u64,
    residual_size: usize,
    sampling_time: Vec<u128>,
    nullification_time: Vec<u128>,
    id_time: Vec<u128>,
    lu_time: Vec<u128>,
    update_id_time: Vec<u128>,
    update_lu_time: Vec<u128>,
}

#[derive(Serialize)]
struct Errors<Item: RlstScalar>{
    rel_errors: Vec<Vec<Real<Item>>>,
    abs_errors: Vec<Vec<Real<Item>>>
}

pub fn save_stats<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(kernel_mat: &mut DynamicArray<Item, 2>, rsrs_factors: &RsrsFactors<Item>, rsrs_data: &RsrsData<Item>, tol: Real<Item>, path_str: &str)
where <Item as RlstScalar>::Real: for<'a> std::iter::Sum<&'a <Item as RlstScalar>::Real>{    
    fs::create_dir_all(Path::new(&path_str)).unwrap();
    let string_tol = format!("{:e}", tol);
    let mut stats_path = path_str.to_string();
    stats_path.push_str("/stats_");
    stats_path.push_str(&string_tol);
    stats_path.push_str(".json");

    let (norm_app_inv, diag_ae_mean, skel_ae) = write_box_errors(kernel_mat, rsrs_factors,  tol, &path_str);

    let stats = Stats::<Item>{
        app_inv_norm: norm_app_inv,
        diag_ae_mean: diag_ae_mean,
        skel_ae: skel_ae,
        tot_num_samples: rsrs_data.y_data.num_samples,
        total_elapsed_time: rsrs_data.stats.total_elapsed_time,
        extraction_time: rsrs_data.stats.extraction_time,
        residual_size: rsrs_data.stats.residual_size,
        sampling_time: rsrs_data.stats.sampling_time.clone(),
        nullification_time: rsrs_data.stats.nullification_time.clone(),
        id_time: rsrs_data.stats.id_time.clone(),
        lu_time: rsrs_data.stats.lu_time.clone(),
        update_id_time: rsrs_data.stats.update_id_time.clone(),
        update_lu_time: rsrs_data.stats.update_lu_time.clone(),
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

    let errors = Errors::<Item>{
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

    let zero = ident - kernel_mat.view();

    (zero.view().norm_1(), diag_ae_r_mean, diag_ae_s)

}
