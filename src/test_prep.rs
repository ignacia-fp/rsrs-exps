use crate::io::plot_results::time_piechart;
use crate::io::read_and_write::{save_error_stats, save_rank_stats, save_time_stats, Solves};
use crate::io::solve::solve_system;
use crate::io::structured_operator::{
    get_bempp_points, GeometryType, StructuredOperator, StructuredOperatorImpl,
    StructuredOperatorInterface, StructuredOperatorParams,
};
use crate::io::structured_operator::{Assembler, LocalFrom};
use crate::io::structured_operators_types::StructuredOperatorType;
use bempp_octree::Octree;
use bempp_rsrs::rsrs::args::RsrsOptions;
use bempp_rsrs::rsrs::rsrs_cycle::Rsrs;
use bempp_rsrs::rsrs::rsrs_factors::rsrs_operator::{LocalFromSpaces, RsrsOperator};
use bempp_rsrs::rsrs::sketch::SamplingSpace;
use bempp_rsrs::utils::io::IOData;
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rlst::prelude::*;
use serde::{Deserialize, Serialize};
type Real<T> = <T as rlst::RlstScalar>::Real;
use crate::io::errors::get_boxes_errors;
use crate::io::read_and_write::ConditionNumberOutput;
use crate::io::solve::solve_prec_system;
use crate::io::structured_operator::Attr;
use bempp::boundary_assemblers::BoundaryAssemblerOptions;
use bempp_rsrs::rsrs::rsrs_factors::rsrs_operator::Inv;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::{Entity, Geometry, Grid, ParallelGrid, Point};
use num::ToPrimitive;
use rlst::tracing::trace_call;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub enum Results {
    All,
    Rank,
    Time,
}

#[derive(Debug, Clone, Deserialize)]
pub enum Precision {
    Single,
    Double,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "Real<Item>: Deserialize<'de>")]
pub enum DimArg<Item: RlstScalar> {
    Kappa(Real<Item>),
    KappaAndMeshwidth(Real<Item>, Real<Item>),
    MeshWidth(Real<Item>),
    RefinementLevelAndDepth(Real<Item>, Real<Item>),
}

#[derive(Debug, Clone, Deserialize)]
pub enum Solve {
    True(f64),
    False,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "Real<Item>: Deserialize<'de>")]
pub struct ScenarioArgs<Item: RlstScalar> {
    id_tols: Vec<Real<Item>>,
    dim_args: Vec<DimArg<Item>>,
    geometry_type: GeometryType,
    max_tree_depth: usize,
    n_sources: i32,
    assembler: Assembler,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataType {
    pub structured_operator_type: StructuredOperatorType,
    pub precision: Precision,
}

fn transpose_matches_apply(structured_operator_type: &StructuredOperatorType) -> bool {
    matches!(
        structured_operator_type,
        StructuredOperatorType::BemppRsLaplaceOperator
            | StructuredOperatorType::KiFMMLaplaceOperator
            | StructuredOperatorType::KiFMMHelmholtzOperator
            | StructuredOperatorType::KiFMMLaplaceOperatorV
    )
}

#[derive(Debug)]
pub struct ScenarioOptions<Item: RlstScalar> {
    id_tols: Vec<Real<Item>>,
    dim_args: Vec<(Real<Item>, Real<Item>)>,
    pub structured_operator_type: StructuredOperatorType,
    geometry_type: GeometryType,
    pub precision: Precision,
    max_tree_depth: usize,
    n_sources: i32,
    assembler: Assembler,
}

pub struct TestParams<Item: RlstScalar> {
    scenario_params: ScenarioOptions<Item>,
    rsrs_params: RsrsOptions<Item>,
}

pub struct TestFramework<Item: RlstScalar> {
    output_options: OutputOptions,
    test_params: TestParams<Item>,
}

fn sample_storage_dir(path_str: &str) -> PathBuf {
    Path::new(path_str).join("sampling")
}

fn sampling_seed_dir(path_str: &str) -> PathBuf {
    Path::new(path_str).join(".sampling_seed")
}

fn sampling_dir_has_presaved_sketches(sampling_dir: &Path) -> bool {
    let y_pair = sampling_dir.join("y_test_file.00000.h5").exists()
        && sampling_dir.join("y_sketch_file.00000.h5").exists();
    let z_pair = sampling_dir.join("z_test_file.00000.h5").exists()
        && sampling_dir.join("z_sketch_file.00000.h5").exists();
    y_pair || z_pair
}

fn existing_sampling_dir(path_str: &str) -> Option<PathBuf> {
    let preferred = sample_storage_dir(path_str);
    if sampling_dir_has_presaved_sketches(&preferred) {
        Some(preferred)
    } else {
        let legacy = PathBuf::from("sampling");
        sampling_dir_has_presaved_sketches(&legacy).then_some(legacy)
    }
}

fn clear_sampling_dir(path_str: &str) -> io::Result<()> {
    let sampling_dir = sample_storage_dir(path_str);
    if sampling_dir.exists() {
        fs::remove_dir_all(sampling_dir)?;
    }
    Ok(())
}

fn clear_sampling_seed_dir(path_str: &str) -> io::Result<()> {
    let sampling_seed_dir = sampling_seed_dir(path_str);
    if sampling_seed_dir.exists() {
        fs::remove_dir_all(sampling_seed_dir)?;
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?;
        } else {
            fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}

fn sampling_has_presaved_sketches(path_str: &str) -> bool {
    existing_sampling_dir(path_str).is_some()
}

fn snapshot_sampling_dir(path_str: &str) -> io::Result<()> {
    clear_sampling_seed_dir(path_str)?;
    if let Some(sampling_dir) = existing_sampling_dir(path_str) {
        copy_dir_all(&sampling_dir, &sampling_seed_dir(path_str))?;
    }
    Ok(())
}

fn restore_sampling_seed_dir(path_str: &str) -> io::Result<()> {
    clear_sampling_dir(path_str)?;
    let sampling_seed_dir = sampling_seed_dir(path_str);
    if sampling_seed_dir.exists() {
        copy_dir_all(&sampling_seed_dir, &sample_storage_dir(path_str))?;
    }
    Ok(())
}

#[derive(Serialize)]
struct SketchPairCheckOutput {
    total_samples: usize,
    checked_samples: usize,
    relerr: f64,
    max_abs_err: f64,
}

#[derive(Serialize)]
struct PresavedSketchCheckOutput {
    threshold: f64,
    y: Option<SketchPairCheckOutput>,
    z: Option<SketchPairCheckOutput>,
}

fn check_saved_sketch_pair<Item: RlstScalar + IOData<Item, Item = Item> + Default + Copy>(
    structured_operator: &StructuredOperatorInterface,
    sampling_dir: &Path,
    test_base: &str,
    sketch_base: &str,
    transposed: bool,
) -> Option<SketchPairCheckOutput>
where
    StructuredOperatorInterface: StructuredOperatorImpl<Item>,
    Real<Item>: ToPrimitive,
{
    let test_flat = <Item as IOData<Item>>::load_in_dir(test_base, Some(sampling_dir)).ok()?;
    let sketch_flat = <Item as IOData<Item>>::load_in_dir(sketch_base, Some(sampling_dir)).ok()?;
    let n_points = structured_operator.n_points;

    assert!(n_points > 0, "Structured operator has zero points.");
    assert!(
        test_flat.len() % n_points == 0,
        "Sample matrix '{}' has invalid length {} for n_points={}.",
        test_base,
        test_flat.len(),
        n_points
    );
    assert_eq!(
        test_flat.len(),
        sketch_flat.len(),
        "Sample matrix '{}' and sketch '{}' have incompatible sizes.",
        test_base,
        sketch_base
    );

    let total_rows = test_flat.len() / n_points;
    let mut input = vec![Item::default(); n_points];
    let mut output = vec![Item::default(); n_points];
    let mut rel_num = 0.0_f64;
    let mut rel_den = 0.0_f64;
    let mut max_abs_err = 0.0_f64;
    let row = 0;

    for col in 0..n_points {
        input[col] = test_flat[col * total_rows + row];
    }

    if transposed {
        <StructuredOperatorInterface as StructuredOperatorImpl<Item>>::mv_trans(
            structured_operator,
            &input,
            &mut output,
        );
    } else {
        <StructuredOperatorInterface as StructuredOperatorImpl<Item>>::mv(
            structured_operator,
            &input,
            &mut output,
        );
    }

    for col in 0..n_points {
        let expected = sketch_flat[col * total_rows + row];
        let diff_abs = (output[col] - expected).abs().to_f64().unwrap();
        let expected_abs = expected.abs().to_f64().unwrap();
        rel_num += diff_abs * diff_abs;
        rel_den += expected_abs * expected_abs;
        max_abs_err = max_abs_err.max(diff_abs);
    }

    let relerr = if rel_den > 0.0 {
        (rel_num / rel_den).sqrt()
    } else {
        rel_num.sqrt()
    };

    Some(SketchPairCheckOutput {
        total_samples: total_rows,
        checked_samples: 1,
        relerr,
        max_abs_err,
    })
}

fn save_presaved_sketch_check(path_str: &str, stats: &PresavedSketchCheckOutput) -> io::Result<()> {
    fs::create_dir_all(Path::new(path_str))?;
    let stats_path = Path::new(path_str).join("presaved_sketch_check.json");
    let json_string = serde_json::to_string_pretty(stats).expect("Failed to serialize");
    fs::write(stats_path, json_string)
}

fn validate_presaved_sketches<Item: RlstScalar + IOData<Item, Item = Item> + Default + Copy>(
    structured_operator: &StructuredOperatorInterface,
    path_str: &str,
) where
    StructuredOperatorInterface: StructuredOperatorImpl<Item>,
    Real<Item>: ToPrimitive,
{
    let Some(sampling_dir) = existing_sampling_dir(path_str) else {
        return;
    };

    let threshold = match std::mem::size_of::<Real<Item>>() {
        4 => 1e-4,
        _ => 1e-10,
    };

    let stats = PresavedSketchCheckOutput {
        threshold,
        y: check_saved_sketch_pair::<Item>(
            structured_operator,
            &sampling_dir,
            "y_test_file",
            "y_sketch_file",
            false,
        ),
        z: check_saved_sketch_pair::<Item>(
            structured_operator,
            &sampling_dir,
            "z_test_file",
            "z_sketch_file",
            true,
        ),
    };

    let _ = save_presaved_sketch_check(path_str, &stats);

    if let Some(y_stats) = &stats.y {
        assert!(
            y_stats.relerr <= threshold,
            "Presaved y-sketch check failed: relerr={} exceeds threshold={}.",
            y_stats.relerr,
            threshold
        );
    }
    if let Some(z_stats) = &stats.z {
        assert!(
            z_stats.relerr <= threshold,
            "Presaved z-sketch check failed: relerr={} exceeds threshold={}.",
            z_stats.relerr,
            threshold
        );
    }
}

fn default_max_leaf_points(id_tol: f64, geometry_type: &GeometryType) -> usize {
    if id_tol < 1.0 {
        return 50;
    }

    let rank = id_tol.to_usize().unwrap();
    match geometry_type {
        GeometryType::Square => 4 * rank,
        GeometryType::SphereSurface
        | GeometryType::CubeSurface
        | GeometryType::CylinderSurface
        | GeometryType::EllipsoidSurface
        | GeometryType::Dihedral
        | GeometryType::Device
        | GeometryType::F16
        | GeometryType::RidgedHorn
        | GeometryType::EMCCAlmond
        | GeometryType::FrigateHull
        | GeometryType::Plane => 6 * rank,
        GeometryType::Sphere | GeometryType::Cube => 8 * rank,
        GeometryType::TrefoilKnot => {
            panic!(
                "Leaf-size policy is undefined for geometry {:?}. \
Please classify it explicitly as 2D, 3D surface, or 3D volume.",
                geometry_type
            )
        }
    }
}

impl<Item: RlstScalar> TestParams<Item> {
    fn new(scenario_args: ScenarioOptions<Item>, rsrs_params: RsrsOptions<Item>) -> Self {
        Self {
            scenario_params: scenario_args,
            rsrs_params,
        }
    }

    fn get_structured_operator_name(&self) -> &str {
        let structured_operator_name = self.scenario_params.structured_operator_type.as_ref();
        structured_operator_name
    }

    fn get_test_dir(&self, dim_num: usize) -> String {
        let geometry = match self.scenario_params.geometry_type {
            GeometryType::SphereSurface => "sphere_surface",
            GeometryType::CubeSurface => "cube_surface",
            GeometryType::CylinderSurface => "cylinder_surface",
            GeometryType::EllipsoidSurface => "ellipsoid_surface",
            GeometryType::TrefoilKnot => "trefoil_knot",
            GeometryType::Sphere => "sphere",
            GeometryType::Cube => "cube",
            GeometryType::Dihedral => "dihedral",
            GeometryType::Device => "device",
            GeometryType::F16 => "f16",
            GeometryType::RidgedHorn => "ridged_horn",
            GeometryType::EMCCAlmond => "emcc_almond",
            GeometryType::FrigateHull => "frigate_hull",
            GeometryType::Plane => "plane",
            GeometryType::Square => "square",
        }
        .to_string();
        let structured_operator = self.get_structured_operator_name();
        let version = self.rsrs_params.to_identifier();

        let filename = if matches!(
            self.scenario_params.structured_operator_type,
            StructuredOperatorType::BemppRsLaplaceOperator
        ) {
            let (ref_level, depth): (Real<Item>, Real<Item>) =
                self.scenario_params.dim_args[dim_num];
            let depth = depth.to_usize().unwrap();
            let dim_pred = if ref_level < num::One::one() {
                format!(
                    "mesh_width_{:.2e}_od_{}",
                    ref_level, self.scenario_params.max_tree_depth
                )
            } else {
                let ref_level = ref_level.to_usize().unwrap();
                format!(
                    "ref_level_{}_depth_{}_od_{}",
                    ref_level, depth, self.scenario_params.max_tree_depth
                )
            };

            let threads = format!("num_threads_{}", self.rsrs_params.num_threads);

            format!(
                "{}_{}_{}_{}",
                geometry, structured_operator, dim_pred, threads
            )
        } else {
            let (h, kappa) = self.scenario_params.dim_args[dim_num];
            let dim_pred = format!(
                "mesh_width_{:.2e}_od_{}",
                h, self.scenario_params.max_tree_depth
            );
            let kappa = format!("{:.2}", kappa);
            let threads = format!("num_threads_{}", self.rsrs_params.num_threads);
            format!(
                "{}_{}_{}_{}_{}",
                geometry, structured_operator, dim_pred, kappa, threads
            )
        };

        // Use `format!` again to prepend "results/" and append version
        let path_str = format!("results/{}/{}", filename, version);
        path_str
    }
}

impl<Item: RlstScalar> ScenarioArgs<Item> {
    pub fn new(
        id_tols: Vec<Real<Item>>,
        dim_args: Vec<DimArg<Item>>,
        geometry_type: GeometryType,
        max_tree_depth: usize,
        n_sources: i32,
        assembler: Assembler,
    ) -> Self {
        Self {
            id_tols,
            dim_args,
            geometry_type,
            max_tree_depth,
            n_sources,
            assembler,
        }
    }
}

impl<Item: RlstScalar> ScenarioOptions<Item> {
    pub fn new(args: Option<ScenarioArgs<Item>>, data_type: DataType) -> Self {
        let args = match args {
            Some(input) => input,
            None => ScenarioArgs::new(
                vec![Item::real(1e-2)],
                vec![DimArg::Kappa(Item::real(std::f64::consts::PI))],
                GeometryType::SphereSurface,
                6,
                0,
                Assembler::Dense,
            ),
        };

        let dim_args: Vec<_> = args
            .dim_args
            .iter()
            .map(|val| match val {
                DimArg::Kappa(kappa) => {
                    let pi = std::f64::consts::PI;
                    let h = Item::real(2.0 * pi) / (Item::real(8.0) * *kappa);
                    (h, *kappa)
                }
                DimArg::KappaAndMeshwidth(kappa, h) => (*h, *kappa),
                DimArg::MeshWidth(h) => (*h, num::Zero::zero()),
                DimArg::RefinementLevelAndDepth(ref_level, depth) => (*ref_level, *depth),
            })
            .collect();
        Self {
            id_tols: args.id_tols,
            dim_args,
            structured_operator_type: data_type.structured_operator_type,
            geometry_type: args.geometry_type,
            precision: data_type.precision,
            max_tree_depth: args.max_tree_depth,
            n_sources: args.n_sources,
            assembler: args.assembler,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputOptions {
    solve: Solve,
    plot: bool,
    dense_errors: bool,
    factors_cn: bool,
    results_output: Results,
}

impl OutputOptions {
    pub fn new(
        solve: Solve,
        plot: bool,
        dense_errors: bool,
        factors_cn: bool,
        results_output: Results,
    ) -> Self {
        Self {
            solve,
            plot,
            dense_errors,
            factors_cn,
            results_output,
        }
    }
}

pub trait TestFrameworkImpl<'a, Item: RlstScalar, Space: SamplingSpace<F = Item> + IndexableSpace> {
    type Item: RlstScalar;

    fn new(
        scenario_args: ScenarioOptions<Item>,
        rsrs_args: RsrsOptions<Item>,
        output_args: OutputOptions,
    ) -> Self;

    fn run_tests(&mut self);
}

macro_rules! implement_test_framework {
    ($scalar:ty) => {
        impl<'a> TestFrameworkImpl<'a, $scalar, ArrayVectorSpace<$scalar>>
            for TestFramework<$scalar>
        {
            type Item = $scalar;
            fn new(
                scenario_args: ScenarioOptions<Self::Item>,
                rsrs_options: RsrsOptions<Self::Item>,
                output_args: OutputOptions,
            ) -> Self {
                let test_params = TestParams::new(scenario_args, rsrs_options);

                Self {
                    test_params,
                    output_options: output_args,
                }
            }

            fn run_tests(&mut self) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();
                for (dim_num, dim_arg) in
                    self.test_params.scenario_params.dim_args.iter().enumerate()
                {
                    let path_str = self.test_params.get_test_dir(dim_num);
                    let preferred_sampling_dir = sample_storage_dir(&path_str);
                    let preferred_sampling_dir_str =
                        preferred_sampling_dir.to_string_lossy().into_owned();
                    self.test_params.rsrs_params.sketching.sample_storage_dir =
                        Some(preferred_sampling_dir_str.clone());
                    let _ = clear_sampling_seed_dir(&path_str);
                    let structured_operator_params = StructuredOperatorParams::new(
                        self.test_params
                            .scenario_params
                            .structured_operator_type
                            .clone(),
                        self.test_params.scenario_params.precision.clone(),
                        self.test_params.scenario_params.geometry_type.clone(),
                        dim_arg.0.into(),
                        dim_arg.1.into(),
                        self.test_params.scenario_params.n_sources,
                        self.test_params.rsrs_params.sketching.initial_num_samples as i32,
                        self.test_params.scenario_params.assembler.clone(),
                    )
                    .with_sample_storage_dir(preferred_sampling_dir_str);

                    let structured_operator: StructuredOperatorInterface =
                        <StructuredOperatorInterface as StructuredOperatorImpl<Self::Item>>::new(
                            &structured_operator_params,
                        );
                    let has_presaved_sketches = sampling_has_presaved_sketches(&path_str);
                    if self.test_params.rsrs_params.sketching.load_samples && has_presaved_sketches
                    {
                        validate_presaved_sketches::<Self::Item>(&structured_operator, &path_str);
                        let _ = snapshot_sampling_dir(&path_str);
                    }
                    let points: Vec<bempp_octree::Point> =
                        get_bempp_points(&structured_operator).unwrap();
                    let operator = StructuredOperator::from_local(structured_operator);
                    let rhs = operator.get_rhs();
                    let dim = points.len();

                    let mut solves = Solves {
                        no_prec: None,
                        prec: None,
                        rel_err_no_prec: None,
                        rel_err_prec: None,
                        sols_no_prec: None,
                        sols_prec: None,
                        vectors_file: None,
                    };

                    match self.output_options.solve {
                        Solve::True(tol) => {
                            let (its, rel_err, sols) = solve_system(&operator, &rhs, tol);
                            solves.no_prec = Some(its);
                            solves.rel_err_no_prec = Some(rel_err);
                            solves.sols_no_prec = Some(sols);
                        }
                        Solve::False => {}
                    };
                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                        if self.test_params.rsrs_params.sketching.load_samples {
                            if has_presaved_sketches {
                                let _ = restore_sampling_seed_dir(&path_str);
                            } else {
                                let _ = clear_sampling_dir(&path_str);
                            }
                        }
                        let max_leaf_points = default_max_leaf_points(
                            id_tol.to_f64().unwrap(),
                            &self.test_params.scenario_params.geometry_type,
                        );
                        let tree: Octree<'_, SimpleCommunicator> = Octree::new(
                            &points,
                            self.test_params.scenario_params.max_tree_depth,
                            max_leaf_points,
                            &comm,
                        );
                        let global_number_of_points: usize = tree.global_number_of_points();
                        let global_max_level: usize = tree.global_max_level();

                        if comm.rank() == 0 {
                            println!(
                                "Setup octree with {} points and maximum level {}",
                                global_number_of_points, global_max_level
                            );
                        }

                        let mut solves = solves.clone();

                        println!("Test: {} points, tol:{}", dim, id_tol);
                        if comm.rank() == 0 {
                            let start_level = global_max_level;
                            let end_level = self.test_params.rsrs_params.min_level;
                            let total_levels = start_level.saturating_sub(end_level) + 1;
                            println!(
                                "RSRS factorization starting: levels {} down to {} ({} total levels)",
                                start_level, end_level, total_levels
                            );
                        }

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(&tree, self.test_params.rsrs_params.clone(), dim);

                        let domain = std::rc::Rc::clone(&operator.domain());
                        let range = std::rc::Rc::clone(&operator.range());

                        let mut rsrs_factors = rsrs_algo.run(operator.r());

                        let mut rsrs_operator =
                            RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);
                        let transpose_matches_apply = transpose_matches_apply(
                            &self.test_params.scenario_params.structured_operator_type,
                        );
                        match self.output_options.solve {
                            Solve::True(tol) => {
                                rsrs_operator.inv(true);
                                let (its, rel_err, sols) =
                                    solve_prec_system(&operator, &rsrs_operator, &rhs, tol);
                                rsrs_operator.inv(false);
                                solves.prec = Some(its);
                                solves.rel_err_prec = Some(rel_err);
                                solves.sols_prec = Some(sols);
                            }
                            Solve::False => {}
                        };

                        match self.output_options.results_output {
                            Results::All => {
                                save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                    transpose_matches_apply,
                                    self.test_params
                                        .rsrs_params
                                        .symmetry
                                        .complex_symmetric_val::<Self::Item>(),
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.plot {
                                    time_piechart(id_tol.into(), &path_str);
                                }

                                if self.output_options.factors_cn {
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                }

                                if self.output_options.dense_errors {
                                    let mut dense_structured_operator =
                                        rlst_dynamic_array2!($scalar, [dim, dim]);
                                    let domain = std::rc::Rc::clone(&operator.domain());
                                    for i in 0..dim {
                                        let mut el_vec =
                                            <rlst::ArrayVectorSpace<_> as SamplingSpace>::zero(
                                                domain.clone(),
                                            );
                                        el_vec.view_mut()[[i]] = num::One::one();
                                        let res =
                                            operator.apply(el_vec.r_mut(), TransMode::NoTrans);
                                        dense_structured_operator
                                            .r_mut()
                                            .slice(1, i)
                                            .fill_from(res.view());
                                    }
                                    get_boxes_errors(
                                        &mut dense_structured_operator,
                                        &mut rsrs_factors,
                                        num::NumCast::from(id_tol).unwrap(),
                                        path_str.as_str(),
                                    );
                                }
                            }
                            Results::Rank => {
                                save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                    transpose_matches_apply,
                                    self.test_params
                                        .rsrs_params
                                        .symmetry
                                        .complex_symmetric_val::<Self::Item>(),
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);

                                if self.output_options.factors_cn {
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                }
                            }
                            Results::Time => {
                                /*save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                );*/
                                save_time_stats(&rsrs_algo, id_tol, &path_str);

                                if self.output_options.factors_cn {
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                }

                                if self.output_options.plot {
                                    time_piechart(id_tol.into(), &path_str);
                                }
                            }
                        }
                    }

                }
            }
        }
    };
}
implement_test_framework!(f32);
implement_test_framework!(f64);
implement_test_framework!(c64);
implement_test_framework!(c32);

macro_rules! implement_distributed_test_framework {
    ($scalar:ty) => {
        impl<'a> TestFrameworkImpl<'a, $scalar, DistributedArrayVectorSpace<'a, SimpleCommunicator, $scalar>>
            for TestFramework<$scalar>
        {
            type Item = $scalar;

            fn new(
                scenario_args: ScenarioOptions<Self::Item>,
                rsrs_options: RsrsOptions<Self::Item>,
                output_args: OutputOptions,
            ) -> Self {
                let test_params = TestParams::new(scenario_args, rsrs_options);

                Self {
                    test_params,
                    output_options: output_args,
                }
            }

            fn run_tests(&mut self) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();
                env_logger::init();
                for (dim_num, dim_arg) in self.test_params.scenario_params.dim_args.iter().enumerate() {

                    let rank = comm.rank();


                    let grid = if dim_arg.0 > 1.0{
                        let refinement_level = dim_arg.0 as usize;
                        bempp::shapes::regular_sphere::<Self::Item, _>(refinement_level as u32, 1, &comm)

                    }
                    else{
                        bempp::shapes::sphere(1.0, (0.0, 0.0, 0.0), dim_arg.0, &comm).unwrap()
                    };


                    let local_tree_depth = 1;
                    let global_tree_depth = dim_arg.1 as usize;
                    let expansion_order = 6;

                    let quad_degree = 6;

                    let space = trace_call("instantiate_space", || {bempp::function::FunctionSpace::new(
                        &grid,
                        &LagrangeElementFamily::<Self::Item>::new(0, ndelement::types::Continuity::Discontinuous),
                        )
                    });

                    let mut options = BoundaryAssemblerOptions::default();
                    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quad_degree);

                    let qrule = options.get_regular_quadrature_rule(ReferenceCellType::Triangle);
                    let qpoints = qrule
                        .points
                        .iter()
                        .map(|&point| num::cast(point).unwrap())
                        .collect::<Vec<Real<Self::Item>>>();

                    let kifmm_evaluator =
                        bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator::from_spaces(
                            &space,
                            &space,
                            green_kernels::types::GreenKernelEvalType::Value,
                            local_tree_depth,
                            global_tree_depth,
                            expansion_order,
                            &qpoints,
                        );

                    let operator = bempp::laplace::evaluator::single_layer(
                        &space,
                        &space,
                        kifmm_evaluator.r(),
                        &options,
                    );

                    let mut x = zero_element(operator.domain());

                    x.view_mut()
                        .local_mut()
                        .fill_from_seed_equally_distributed(rank as usize);
                    let _res = trace_call("apply_laplace", || operator.apply(x.r(), TransMode::NoTrans));

                    let mut points = Vec::new();
                    let mut p = vec![0.0; 3];
                    for cell in grid.local_grid().entity_iter(2) {
                        let mut barycentre = [0.0f64; 3];
                        for point in cell.geometry().points() {
                            point.coords(&mut p);
                            barycentre[0] += num::cast::<_, f64>(p[0]).unwrap() / 3.0;
                            barycentre[1] += num::cast::<_, f64>(p[1]).unwrap() / 3.0;
                            barycentre[2] += num::cast::<_, f64>(p[2]).unwrap() / 3.0;
                        }
                        let point = bempp_octree::Point::new(barycentre, 000);
                        points.push(point);
                    }


                    let mut single_rhs: Element<rlst::operator::ConcreteElementContainer<_>> =
                        zero_element(operator.domain());
                    single_rhs.view_mut()
                        .local_mut()
                        .fill_from_seed_equally_distributed(rank as usize);
                    let rhs = vec![single_rhs];

                    let dim = points.len();

                    println!("{} degrees of freedom", dim);

                    let mut solves = Solves {
                        no_prec: None,
                        prec: None,
                        rel_err_prec: None,
                        rel_err_no_prec: None,
                        sols_no_prec: None,
                        sols_prec: None,
                        vectors_file: None,
                    };

                    match self.output_options.solve {
                        Solve::True(tol) => {
                                let (its, rel_err, sols) = solve_system(&operator, &rhs, tol);
                                solves.no_prec = Some(its);
                                solves.rel_err_no_prec = Some(rel_err);
                                solves.sols_no_prec = Some(sols);
                            },
                        Solve::False => {},
                    };

                    let path_str = self.test_params.get_test_dir(dim_num);
                    let preferred_sampling_dir = sample_storage_dir(&path_str);
                    self.test_params.rsrs_params.sketching.sample_storage_dir =
                        Some(preferred_sampling_dir.to_string_lossy().into_owned());
                    let _ = clear_sampling_seed_dir(&path_str);
                    let has_presaved_sketches = sampling_has_presaved_sketches(&path_str);
                    if self.test_params.rsrs_params.sketching.load_samples && has_presaved_sketches
                    {
                        let _ = snapshot_sampling_dir(&path_str);
                    }

                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                        if self.test_params.rsrs_params.sketching.load_samples {
                            if has_presaved_sketches {
                                let _ = restore_sampling_seed_dir(&path_str);
                            } else {
                                let _ = clear_sampling_dir(&path_str);
                            }
                        }

                        let max_leaf_points = default_max_leaf_points(
                            id_tol.to_f64().unwrap(),
                            &self.test_params.scenario_params.geometry_type,
                        );

                        let tree: Octree<'_, SimpleCommunicator> =
                            Octree::new(&points, self.test_params.scenario_params.max_tree_depth, max_leaf_points, &comm);
                        let global_number_of_points: usize = tree.global_number_of_points();
                        let global_max_level: usize = tree.global_max_level();


                        if comm.rank() == 0 {
                            println!(
                                "Setup octree with {} points and maximum level {}",
                                global_number_of_points, global_max_level
                            );
                        }

                        let mut solves = solves.clone();

                        println!("Test: {} points, tol:{}", dim, id_tol);
                        if comm.rank() == 0 {
                            let start_level = global_max_level;
                            let end_level = self.test_params.rsrs_params.min_level;
                            let total_levels = start_level.saturating_sub(end_level) + 1;
                            println!(
                                "RSRS factorization starting: levels {} down to {} ({} total levels)",
                                start_level, end_level, total_levels
                            );
                        }

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(&tree, self.test_params.rsrs_params.clone(), dim);
                        let mut rsrs_operator = rsrs_algo.get_rsrs_operator(operator.r());
                        let transpose_matches_apply = transpose_matches_apply(
                            &self.test_params.scenario_params.structured_operator_type,
                        );



                        match self.output_options.solve {
                            Solve::True(tol) => {
                                rsrs_operator.inv(true);
                                let (its, rel_err, sols) = solve_prec_system(&operator, &rsrs_operator, &rhs, tol);
                                rsrs_operator.inv(false);
                                solves.prec = Some(its);
                                solves.rel_err_prec = Some(rel_err);
                                solves.sols_prec = Some(sols);
                            }
                            Solve::False => {},
                        };

                        match self.output_options.results_output {
                            Results::All => {
                                save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                    transpose_matches_apply,
                                    self.test_params
                                        .rsrs_params
                                        .symmetry
                                        .complex_symmetric_val::<Self::Item>(),
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.plot {
                                    time_piechart(id_tol.into(), &path_str);
                                }

                                if self.output_options.factors_cn{
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                }

                                if self.output_options.dense_errors {
                                    panic!("Not implemented yet");
                                    /*let mut dense_structured_operator =
                                        rlst_dynamic_array2!($scalar, [dim, dim]);
                                    let domain = std::rc::Rc::clone(&operator.domain());
                                    for i in 0..dim {
                                        let mut el_vec =
                                            <rlst::ArrayVectorSpace<_> as SamplingSpace>::zero(
                                                domain.clone(),
                                            );
                                        el_vec.view_mut()[[i]] = num::One::one();
                                        let res =
                                            operator.apply(el_vec.r_mut(), TransMode::NoTrans);
                                        dense_structured_operator
                                            .r_mut()
                                            .slice(1, i)
                                            .fill_from(res.view());
                                    }
                                    get_boxes_errors(
                                        &mut dense_structured_operator,
                                        &mut rsrs_factors,
                                        num::NumCast::from(id_tol).unwrap(),
                                    );*/
                                }
                            }
                            Results::Rank => {
                                save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                    transpose_matches_apply,
                                    self.test_params
                                        .rsrs_params
                                        .symmetry
                                        .complex_symmetric_val::<Self::Item>(),
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.factors_cn{
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                }
                            }
                            Results::Time => {
                                /*save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                );*/
                                save_time_stats(&rsrs_algo, id_tol, &path_str);

                                if self.output_options.factors_cn{
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                }

                                if self.output_options.plot {
                                    time_piechart(id_tol.into(), &path_str);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

implement_distributed_test_framework!(f32);
implement_distributed_test_framework!(f64);
