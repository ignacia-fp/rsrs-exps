use std::{error::Error, fs::{self, File}, io::BufWriter, path::Path};
use bempp_rsrs::rsrs::rsrs_cycle::RsrsData;
use rlst::RlstScalar;
use std::io::Write;

type Real<T> = <T as rlst::RlstScalar>::Real;


pub fn write_vec_to_new_file_u128(path: impl AsRef<Path>, value: &[u128]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    for x in value {
        writeln!(writer, "{x}")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_vec_to_new_file<Item: RlstScalar>(path: impl AsRef<Path>, value: &[Item]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    for x in value {
        writeln!(writer, "{x}")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_multi_vec_to_new_file<Item: RlstScalar>(path: impl AsRef<Path>, values: &[Vec<Item>]){
    for (id, val) in values.iter().enumerate(){
        let mut new_path = path.as_ref().to_str().unwrap().to_string();
        new_path.push_str(&id.to_string());
        new_path.push_str(".json");
        let _ = write_vec_to_new_file(new_path,val);
    }
}

pub fn save_stats<Item: RlstScalar>(rsrs_data: &RsrsData<Item>, tol: Real<Item>, path_str: &str){
    let string_tol = format!("{:e}", tol);

    let mut times_path = path_str.to_string();
    times_path.push_str("/times_");
    times_path.push_str(&string_tol);
    times_path.push('/');

    fs::create_dir_all(Path::new(&times_path)).unwrap();

    let mut sampling_path = times_path.clone();
    sampling_path.push_str("sampling.json");

    let mut nullification_path = times_path.clone();
    nullification_path.push_str("nullification.json");

    let mut id_time_path = times_path.clone();
    id_time_path.push_str("id.json");

    let mut lu_time_path = times_path.clone();
    lu_time_path.push_str("lu.json");

    let mut update_id_time_path = times_path.clone();
    update_id_time_path.push_str("update_id.json");

    let mut update_lu_time_path = times_path.clone();
    update_lu_time_path.push_str("update_lu.json");

    let mut mixed_path = times_path.clone();
    mixed_path.push_str("mixed.json");

    let mixed_res = [rsrs_data.stats.total_elapsed_time as u128, rsrs_data.stats.extraction_time as u128, rsrs_data.stats.residual_size as u128];    

    let _ = write_vec_to_new_file_u128(sampling_path, &rsrs_data.stats.sampling_time);
    let _ = write_vec_to_new_file_u128(nullification_path, &rsrs_data.stats.nullification_time);
    let _ = write_vec_to_new_file_u128(id_time_path, &rsrs_data.stats.id_time);
    let _ = write_vec_to_new_file_u128(lu_time_path, &rsrs_data.stats.lu_time);
    let _ = write_vec_to_new_file_u128(update_id_time_path, &rsrs_data.stats.update_id_time);
    let _ = write_vec_to_new_file_u128(update_lu_time_path, &rsrs_data.stats.update_lu_time);
    let _ = write_vec_to_new_file_u128(mixed_path, &mixed_res);
    

}
