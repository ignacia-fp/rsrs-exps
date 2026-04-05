use super::read_and_write::{read_file, FileContent};
use piechart::{Chart, Color, Data};

pub fn time_piechart(tol: f64, path_str: &str) {
    match read_file(path_str, "time_stats", tol).unwrap() {
        FileContent::TimeStats(stats) => {
            let tot_id_time = stats.tot_id_time as f32;
            let tot_sampling_time: f32 = stats.sampling_time.iter().map(|&x| x as f32).sum();
            let sample_loading_time = stats.sample_loading_time as f32;
            let extraction_sampling_time = stats.sampling_extraction_time as f32;
            let tot_update_id_time: f32 = stats.update_times.iter().map(|x| x.id as f32).sum();
            let tot_update_lu_time: f32 = stats.update_times.iter().map(|x| x.lu as f32).sum();
            let tot_lu_time = stats.tot_lu_time as f32;
            let tot_extraction_time = stats.extraction_time as f32;
            let total_rsrs_time = stats.total_elapsed_time_wo_sampling as f32;
            let other = (stats.sorting_near_field
                + stats.index_calculation
                + stats.residual_calculation) as f32;
            let untracked_rsrs_time: f32 = if stats.untracked_rsrs_time > 0.0 {
                stats.untracked_rsrs_time as f32
            } else {
                (total_rsrs_time
                    - (tot_lu_time
                        + tot_id_time
                        + other
                        + tot_update_id_time
                        + tot_update_lu_time
                        + tot_extraction_time))
                    .max(0.0_f32)
            };

            let data = vec![
                Data {
                    label: "LU".into(),
                    value: tot_lu_time,
                    color: Some(piechart::Color::Red.into()),
                    fill: '•',
                },
                Data {
                    label: "ID".into(),
                    value: tot_id_time,
                    color: Some(Color::Green.into()),
                    fill: '•',
                },
                Data {
                    label: "Other".into(),
                    value: other,
                    color: Some(Color::Blue.into()),
                    fill: '•',
                },
                Data {
                    label: "Untracked RSRS".into(),
                    value: untracked_rsrs_time,
                    color: Some(Color::RGB(255, 165, 0).into()),
                    fill: '•',
                },
                Data {
                    label: "Update ID".into(),
                    value: tot_update_id_time,
                    color: Some(Color::Yellow.into()),
                    fill: '•',
                },
                Data {
                    label: "Update LU".into(),
                    value: tot_update_lu_time,
                    color: Some(Color::Cyan.into()),
                    fill: '•',
                },
                Data {
                    label: "Sample Load".into(),
                    value: sample_loading_time,
                    color: Some(Color::RGB(160, 160, 160).into()),
                    fill: '•',
                },
                Data {
                    label: "Sampling".into(),
                    value: tot_sampling_time,
                    color: Some(Color::White.into()),
                    fill: '•',
                },
                Data {
                    label: "Extraction Sampling".into(),
                    value: extraction_sampling_time,
                    color: Some(Color::RGB(128, 128, 128).into()),
                    fill: '•',
                },
                Data {
                    label: "Extraction".into(),
                    value: tot_extraction_time,
                    color: Some(Color::Purple.into()),
                    fill: '•',
                },
            ];

            println!("LU: {}", tot_lu_time);
            println!("ID: {}", tot_id_time);
            println!("Other: {}", other);
            println!("Untracked RSRS: {}", untracked_rsrs_time);
            println!("Update ID: {}", tot_update_id_time);
            println!("Update LU: {}", tot_update_lu_time);
            println!("Sample Load: {}", sample_loading_time);
            println!("Extraction: {}", tot_extraction_time);
            println!("Total RSRS: {}", total_rsrs_time);
            println!(
                "Total Sampling: {}",
                sample_loading_time + tot_sampling_time + extraction_sampling_time
            );

            Chart::new()
                .radius(9)
                .aspect_ratio(2)
                .legend(true)
                .draw(&data);
        }
        _ => panic!("Expected FileContent::Stats variant"),
    };
}

pub fn get_time_piecharts(path_str: &str, tols: &[f64]) {
    for tol in tols {
        time_piechart(*tol, path_str);
    }
}
