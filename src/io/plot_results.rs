//use full_palette::ORANGE;
//use plotters::prelude::full_palette::PURPLE;
//use rlst::RlstScalar;
//use std::error::Error;
//use std::f64::consts::PI;
//use std::fs::File;
//use std::io::BufReader;
//use std::io::BufRead;
//use std::path::Path;
use super::read_and_write::{read_file, FileContent};
use piechart::{Chart, Data, Color};
use plotters::prelude::*;
/* 
pub fn read_vec_from_file(path: impl AsRef<Path>) -> Result<Vec<f64>, Box<dyn Error>> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut v = vec![];

    let mut buf = String::new();

    while reader.read_line(&mut buf)? != 0 {
        v.push(buf.trim_end().parse()?);
        buf.clear();
    }

    Ok(v)
}

pub fn plot_log_multi_series(x_data: &[f64], y_data: &Vec<Vec<f64>>, out_name_file: String, title: String){

    let root_area = BitMapBackend::new(&out_name_file, (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut max_series = Vec::new();

    for series in y_data.iter(){
        max_series.push(*series.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
    }

    let max_y = *max_series.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x = *x_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
    .set_label_area_size(LabelAreaPosition::Left, 40)
    .set_label_area_size(LabelAreaPosition::Bottom, 40)
    .caption(title, ("sans-serif", 40))
    .build_cartesian_2d((0.0..max_x).log_scale(), (0.0..max_y).log_scale())
    .unwrap();

    for series in y_data.iter(){
        let data = series.iter().enumerate().map(|el| (x_data[el.0], *el.1));
        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data, &RED))
            .unwrap()
            .label("rf")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    }
    ctx.configure_series_labels()
        .position(SeriesLabelPosition::UpperMiddle)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .unwrap();
    root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

}

pub fn plot_multi_series(x_data: &[f64], y_data: &Vec<Vec<f64>>, out_name_file: String, title: String){

    let root_area = BitMapBackend::new(&out_name_file, (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut max_series = Vec::new();

    for series in y_data.iter(){
        max_series.push(*series.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
    }

    let max_y = *max_series.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x = *x_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
    .set_label_area_size(LabelAreaPosition::Left, 40)
    .set_label_area_size(LabelAreaPosition::Bottom, 40)
    .caption(title, ("sans-serif", 40))
    .build_cartesian_2d((0.0..max_x).log_scale(), 0.0..max_y)
    .unwrap();

    for series in y_data.iter(){
        let data = series.iter().enumerate().map(|el| (x_data[el.0], *el.1));
        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data, &RED))
            .unwrap()
            .label("rf")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    }
    ctx.configure_series_labels()
        .position(SeriesLabelPosition::UpperMiddle)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .unwrap();
    root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

}

pub fn plot_series(x_data: &[f64], y_data: &[f64], out_name_file: String, title: String, log: bool){

    let root_area = BitMapBackend::new(&out_name_file, (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let data = y_data.iter().enumerate().map(|el| (x_data[el.0], *el.1));
    let max_y = *y_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x = *x_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    if log{
        let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d((0.0..max_x).log_scale(), (0.0..max_y).log_scale())
        .unwrap();
        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data, &RED))
            .unwrap()
            .label("rf")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
        ctx.configure_series_labels()
            .position(SeriesLabelPosition::UpperMiddle)
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()
            .unwrap();
        root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    }
    else{

        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption(title, ("sans-serif", 40))
            .build_cartesian_2d((0.0..max_x).log_scale(), 0.0..max_y)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data, &RED))
            .unwrap()
            .label("rf")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
        ctx.configure_series_labels()
            .position(SeriesLabelPosition::UpperMiddle)
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()
            .unwrap();
        root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    }

}

pub fn plot_errors(errors_data: (&Vec<f64>, &Vec<f64>, &Vec<f64>, &Vec<f64>), name_file: String, title: String){
  
    let (rf, fr, rt, tr) = errors_data;

    let root_area = BitMapBackend::new(&name_file, (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let n = (rf.len() +1) as f64;
    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0.0..n, (0.0..0.1).log_scale())
        .unwrap();


    ctx.configure_mesh().draw().unwrap();
    ctx.draw_series(LineSeries::new(rf.iter().enumerate().map(|x| (x.0 as f64, *x.1)), &RED))
        .unwrap()
        .label("rf")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    ctx.draw_series(LineSeries::new(fr.iter().enumerate().map(|x| (x.0 as f64, *x.1)), &BLUE))
        .unwrap()
        .label("fr")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    ctx.draw_series(LineSeries::new(rt.iter().enumerate().map(|x| (x.0 as f64, *x.1)), &GREEN))
        .unwrap()
        .label("rt")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));
    ctx.draw_series(LineSeries::new(tr.iter().enumerate().map(|x| (x.0 as f64, *x.1)), &ORANGE))
        .unwrap()
        .label("tr")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ORANGE));
    ctx.configure_series_labels()
        .position(SeriesLabelPosition::UpperMiddle)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .unwrap();
    root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

}

*/
pub fn plot_stats(geometry : &str, kernel: &str, npoints: &[usize], kappa: f64, tols: &[f64]){
    for &n in npoints{
        for tol in tols{
            match read_file("stats", geometry, kernel, n, kappa, *tol).unwrap() {
                FileContent::Stats(stats) => {
                    let tot_lu_time = stats.lu_time.iter().map(|&x| x as f32).sum();
                    let tot_id_time = stats.id_time.iter().map(|&x| x as f32).sum();
                    let tot_null_time = stats.nullification_time.iter().map(|&x| x as f32).sum();
                    let tot_sampling_time = stats.sampling_time.iter().map(|&x| x as f32).sum();
                    let tot_update_id_time = stats.update_id_time.iter().map(|&x| x as f32).sum();
                    let tot_update_lu_time = stats.update_lu_time.iter().map(|&x| x as f32).sum();
                    let tot_extraction_time = stats.extraction_time as f32;

                    println!("Total LU time: {} ms", tot_lu_time);
                    println!("Total ID time: {} ms", tot_id_time);
                    println!("Total Nullification time: {} ms", tot_null_time);
                    println!("Total Sampling time: {} ms", tot_sampling_time);
                    println!("Total Update ID time: {} ms", tot_update_id_time);
                    println!("Total Update LU time: {} ms", tot_update_lu_time);
                    println!("Total Extraction time: {} ms", tot_extraction_time);

                
                    let root = BitMapBackend::new("pie_chart.png", (500, 500)).into_drawing_area();
                    root.fill(&WHITE).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)).unwrap();

                    let data = vec![
                        Data { label: "LU".into(), value: tot_lu_time, color: Some(piechart::Color::Red.into()), fill: '•' },
                        Data { label: "ID".into(), value: tot_id_time, color: Some(Color::Green.into()), fill: '•' },
                        Data { label: "Nullification".into(), value: tot_null_time, color: Some(Color::Blue.into()), fill: '•' },
                        Data { label: "Update ID".into(), value: tot_update_id_time, color: Some(Color::Yellow.into()), fill: '•' },
                        Data { label: "Update LU".into(), value: tot_update_lu_time, color: Some(Color::White.into()), fill: '•' },
                        Data { label: "Sampling".into(), value: tot_sampling_time, color: Some(Color::Cyan.into()), fill: '•' },
                        Data { label: "Extraction".into(), value: tot_extraction_time, color: Some(Color::Purple.into()), fill: '•' },
                    ];

                    Chart::new()
                        .radius(9)
                        .aspect_ratio(2)
                        .legend(true)
                        .draw(&data);
                },
                _ => panic!("Expected FileContent::Stats variant"),
            };   
        }
    }

}