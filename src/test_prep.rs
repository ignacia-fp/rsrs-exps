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
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rlst::prelude::*;
use serde::Deserialize;
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
use std::path::{Path, PathBuf};
use std::time::Instant;

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

fn log_root(rank: i32, message: &str) {
    if rank == 0 {
        println!("[rsrs-exps] {message}");
    }
}

fn start_root_stage(rank: i32, label: &str) -> Instant {
    if rank == 0 {
        println!("[rsrs-exps] {label}...");
    }
    Instant::now()
}

fn finish_root_stage(rank: i32, label: &str, start: Instant) {
    if rank == 0 {
        println!(
            "[rsrs-exps] {label} done in {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }
}

fn configured_run_seed() -> Option<u64> {
    std::env::var("RSRS_RUN_SEED")
        .ok()
        .and_then(|seed| seed.parse::<u64>().ok())
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

    fn geometry_slug(&self) -> &'static str {
        match self.scenario_params.geometry_type {
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
    }

    fn precision_slug(&self) -> &'static str {
        match self.scenario_params.precision {
            Precision::Single => "single",
            Precision::Double => "double",
        }
    }

    fn get_structured_operator_name(&self) -> &str {
        let structured_operator_name = self.scenario_params.structured_operator_type.as_ref();
        structured_operator_name
    }

    fn sample_cache_key(&self, dim_num: usize) -> String {
        let geometry = self.geometry_slug();
        let structured_operator = self.get_structured_operator_name();
        let precision = self.precision_slug();

        let dim_pred = if matches!(
            self.scenario_params.structured_operator_type,
            StructuredOperatorType::BemppRsLaplaceOperator
        ) {
            let (ref_level, depth): (Real<Item>, Real<Item>) =
                self.scenario_params.dim_args[dim_num];
            if ref_level < num::One::one() {
                format!("mesh_width_{:.2e}", ref_level)
            } else {
                let ref_level = ref_level.to_usize().unwrap();
                let depth = depth.to_usize().unwrap();
                format!("ref_level_{}_depth_{}", ref_level, depth)
            }
        } else {
            let (h, kappa) = self.scenario_params.dim_args[dim_num];
            if kappa == num::Zero::zero() {
                format!("mesh_width_{:.2e}", h)
            } else {
                format!("mesh_width_{:.2e}_kappa_{:.2}", h, kappa)
            }
        };

        format!(
            "{}_{}_precision_{}_{}",
            geometry, structured_operator, precision, dim_pred
        )
    }

    fn get_sample_storage_dir(&self, dim_num: usize) -> PathBuf {
        Path::new("results")
            .join("sample_cache")
            .join(self.sample_cache_key(dim_num))
            .join("sampling")
    }

    fn get_test_dir(&self, dim_num: usize) -> String {
        let geometry = self.geometry_slug().to_string();
        let structured_operator = self.get_structured_operator_name();
        let precision = self.precision_slug();
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
                "{}_{}_precision_{}_{}_{}",
                geometry, structured_operator, precision, dim_pred, threads
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
                "{}_{}_precision_{}_{}_{}_{}",
                geometry, structured_operator, precision, dim_pred, kappa, threads
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
                let rank = comm.rank();
                for (dim_num, dim_arg) in
                    self.test_params.scenario_params.dim_args.iter().enumerate()
                {
                    let path_str = self.test_params.get_test_dir(dim_num);
                    let preferred_sampling_dir = self.test_params.get_sample_storage_dir(dim_num);
                    let preferred_sampling_dir_str =
                        preferred_sampling_dir.to_string_lossy().into_owned();
                    self.test_params.rsrs_params.sketching.sample_storage_dir =
                        Some(preferred_sampling_dir_str.clone());
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

                    log_root(
                        rank,
                        &format!(
                            "scenario start: out_dir='{}', sample_dir='{}', operator='{}', load_samples={}, save_samples={}, init_samples={}",
                            path_str,
                            preferred_sampling_dir.display(),
                            self.test_params.scenario_params.structured_operator_type.as_ref(),
                            self.test_params.rsrs_params.sketching.load_samples,
                            self.test_params.rsrs_params.sketching.save_samples,
                            self.test_params.rsrs_params.sketching.initial_num_samples,
                        ),
                    );

                    let stage = start_root_stage(rank, "initialize structured operator");
                    let structured_operator: StructuredOperatorInterface =
                        <StructuredOperatorInterface as StructuredOperatorImpl<Self::Item>>::new(
                            &structured_operator_params,
                        );
                    finish_root_stage(rank, "initialize structured operator", stage);

                    let stage = start_root_stage(rank, "fetch geometry points");
                    let points: Vec<bempp_octree::Point> =
                        get_bempp_points(&structured_operator).unwrap();
                    finish_root_stage(rank, "fetch geometry points", stage);

                    let stage = start_root_stage(rank, "wrap structured operator");
                    let operator = StructuredOperator::from_local(structured_operator);
                    finish_root_stage(rank, "wrap structured operator", stage);

                    let stage = start_root_stage(rank, "fetch right-hand sides");
                    let rhs = operator.get_rhs();
                    finish_root_stage(rank, "fetch right-hand sides", stage);
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
                            let label = format!(
                                "solve without preconditioner ({} rhs, tol={:.3e})",
                                rhs.len(),
                                tol
                            );
                            let stage = start_root_stage(rank, &label);
                            let (its, rel_err, sols) = solve_system(&operator, &rhs, tol);
                            finish_root_stage(rank, &label, stage);
                            solves.no_prec = Some(its);
                            solves.rel_err_no_prec = Some(rel_err);
                            solves.sols_no_prec = Some(sols);
                        }
                        Solve::False => {}
                    };
                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                        let max_leaf_points = default_max_leaf_points(
                            id_tol.to_f64().unwrap(),
                            &self.test_params.scenario_params.geometry_type,
                        );
                        let label =
                            format!("build octree for id_tol={} (max_leaf_points={})", id_tol, max_leaf_points);
                        let stage = start_root_stage(rank, &label);
                        let tree: Octree<'_, SimpleCommunicator> = Octree::new(
                            &points,
                            self.test_params.scenario_params.max_tree_depth,
                            max_leaf_points,
                            &comm,
                        );
                        finish_root_stage(rank, &label, stage);
                        let global_number_of_points: usize = tree.global_number_of_points();
                        let global_max_level: usize = tree.global_max_level();

                        if rank == 0 {
                            println!(
                                "Setup octree with {} points and maximum level {}",
                                global_number_of_points, global_max_level
                            );
                        }

                        let mut solves = solves.clone();

                        println!("Test: {} points, tol:{}", dim, id_tol);
                        if rank == 0 {
                            let start_level = global_max_level;
                            let end_level = self.test_params.rsrs_params.min_level;
                            let total_levels = start_level.saturating_sub(end_level) + 1;
                            println!(
                                "RSRS factorization starting: levels {} down to {} ({} total levels)",
                                start_level, end_level, total_levels
                            );
                        }

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let stage = start_root_stage(rank, "construct RSRS state");
                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(&tree, self.test_params.rsrs_params.clone(), dim);
                        finish_root_stage(rank, "construct RSRS state", stage);

                        let domain = std::rc::Rc::clone(&operator.domain());
                        let range = std::rc::Rc::clone(&operator.range());

                        let stage = start_root_stage(rank, "run RSRS factorization");
                        let mut rsrs_factors = if let Some(seed) = configured_run_seed() {
                            if rank == 0 {
                                println!("[rsrs-exps] using deterministic RSRS seed: {seed}");
                            }
                            rsrs_algo.run_with_seed(operator.r(), seed)
                        } else {
                            rsrs_algo.run(operator.r())
                        };
                        finish_root_stage(rank, "run RSRS factorization", stage);

                        let stage = start_root_stage(rank, "build RSRS operator");
                        let mut rsrs_operator =
                            RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);
                        finish_root_stage(rank, "build RSRS operator", stage);
                        let transpose_matches_apply = transpose_matches_apply(
                            &self.test_params.scenario_params.structured_operator_type,
                        );
                        match self.output_options.solve {
                            Solve::True(tol) => {
                                let label = format!(
                                    "solve with RSRS preconditioner ({} rhs, tol={:.3e})",
                                    rhs.len(),
                                    tol
                                );
                                let stage = start_root_stage(rank, &label);
                                rsrs_operator.inv(true);
                                let (its, rel_err, sols) =
                                    solve_prec_system(&operator, &rsrs_operator, &rhs, tol);
                                rsrs_operator.inv(false);
                                finish_root_stage(rank, &label, stage);
                                solves.prec = Some(its);
                                solves.rel_err_prec = Some(rel_err);
                                solves.sols_prec = Some(sols);
                            }
                            Solve::False => {}
                        };

                        match self.output_options.results_output {
                            Results::All => {
                                let stage = start_root_stage(rank, "save error statistics");
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
                                finish_root_stage(rank, "save error statistics", stage);
                                let stage = start_root_stage(rank, "save time statistics");
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save time statistics", stage);
                                let stage = start_root_stage(rank, "save rank statistics");
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save rank statistics", stage);
                                if self.output_options.plot {
                                    let stage = start_root_stage(rank, "render time piechart");
                                    time_piechart(id_tol.into(), &path_str);
                                    finish_root_stage(rank, "render time piechart", stage);
                                }

                                if self.output_options.factors_cn {
                                    let stage =
                                        start_root_stage(rank, "save factor condition numbers");
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                    finish_root_stage(rank, "save factor condition numbers", stage);
                                }

                                if self.output_options.dense_errors {
                                    let stage = start_root_stage(rank, "compute dense/block errors");
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
                                    finish_root_stage(rank, "compute dense/block errors", stage);
                                }
                            }
                            Results::Rank => {
                                let stage = start_root_stage(rank, "save error statistics");
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
                                finish_root_stage(rank, "save error statistics", stage);
                                let stage = start_root_stage(rank, "save rank statistics");
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save rank statistics", stage);

                                if self.output_options.factors_cn {
                                    let stage =
                                        start_root_stage(rank, "save factor condition numbers");
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                    finish_root_stage(rank, "save factor condition numbers", stage);
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
                                let stage = start_root_stage(rank, "save time statistics");
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save time statistics", stage);

                                if self.output_options.factors_cn {
                                    let stage =
                                        start_root_stage(rank, "save factor condition numbers");
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                    finish_root_stage(rank, "save factor condition numbers", stage);
                                }

                                if self.output_options.plot {
                                    let stage = start_root_stage(rank, "render time piechart");
                                    time_piechart(id_tol.into(), &path_str);
                                    finish_root_stage(rank, "render time piechart", stage);
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

                    log_root(
                        rank,
                        &format!(
                            "distributed scenario start: out_dir='{}', sample_dir='{}', load_samples={}, save_samples={}, init_samples={}",
                            self.test_params.get_test_dir(dim_num),
                            self.test_params.get_sample_storage_dir(dim_num).display(),
                            self.test_params.rsrs_params.sketching.load_samples,
                            self.test_params.rsrs_params.sketching.save_samples,
                            self.test_params.rsrs_params.sketching.initial_num_samples,
                        ),
                    );
                    log_root(rank, &format!("{dim} local degrees of freedom"));

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
                                let label = format!(
                                    "solve without preconditioner ({} rhs, tol={:.3e})",
                                    rhs.len(),
                                    tol
                                );
                                let stage = start_root_stage(rank, &label);
                                let (its, rel_err, sols) = solve_system(&operator, &rhs, tol);
                                finish_root_stage(rank, &label, stage);
                                solves.no_prec = Some(its);
                                solves.rel_err_no_prec = Some(rel_err);
                                solves.sols_no_prec = Some(sols);
                            },
                        Solve::False => {},
                    };

                    let path_str = self.test_params.get_test_dir(dim_num);
                    let preferred_sampling_dir = self.test_params.get_sample_storage_dir(dim_num);
                    self.test_params.rsrs_params.sketching.sample_storage_dir =
                        Some(preferred_sampling_dir.to_string_lossy().into_owned());

                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                        let max_leaf_points = default_max_leaf_points(
                            id_tol.to_f64().unwrap(),
                            &self.test_params.scenario_params.geometry_type,
                        );

                        let label =
                            format!("build octree for id_tol={} (max_leaf_points={})", id_tol, max_leaf_points);
                        let stage = start_root_stage(rank, &label);
                        let tree: Octree<'_, SimpleCommunicator> =
                            Octree::new(&points, self.test_params.scenario_params.max_tree_depth, max_leaf_points, &comm);
                        finish_root_stage(rank, &label, stage);
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

                        let stage = start_root_stage(rank, "construct RSRS state");
                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(&tree, self.test_params.rsrs_params.clone(), dim);
                        finish_root_stage(rank, "construct RSRS state", stage);
                        let stage = start_root_stage(rank, "build RSRS operator");
                        let mut rsrs_operator = rsrs_algo.get_rsrs_operator(operator.r());
                        finish_root_stage(rank, "build RSRS operator", stage);
                        let transpose_matches_apply = transpose_matches_apply(
                            &self.test_params.scenario_params.structured_operator_type,
                        );



                        match self.output_options.solve {
                            Solve::True(tol) => {
                                let label = format!(
                                    "solve with RSRS preconditioner ({} rhs, tol={:.3e})",
                                    rhs.len(),
                                    tol
                                );
                                let stage = start_root_stage(rank, &label);
                                rsrs_operator.inv(true);
                                let (its, rel_err, sols) = solve_prec_system(&operator, &rsrs_operator, &rhs, tol);
                                rsrs_operator.inv(false);
                                finish_root_stage(rank, &label, stage);
                                solves.prec = Some(its);
                                solves.rel_err_prec = Some(rel_err);
                                solves.sols_prec = Some(sols);
                            }
                            Solve::False => {},
                        };

                        match self.output_options.results_output {
                            Results::All => {
                                let stage = start_root_stage(rank, "save error statistics");
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
                                finish_root_stage(rank, "save error statistics", stage);
                                let stage = start_root_stage(rank, "save time statistics");
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save time statistics", stage);
                                let stage = start_root_stage(rank, "save rank statistics");
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save rank statistics", stage);
                                if self.output_options.plot {
                                    let stage = start_root_stage(rank, "render time piechart");
                                    time_piechart(id_tol.into(), &path_str);
                                    finish_root_stage(rank, "render time piechart", stage);
                                }

                                if self.output_options.factors_cn{
                                    let stage =
                                        start_root_stage(rank, "save factor condition numbers");
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                    finish_root_stage(rank, "save factor condition numbers", stage);
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
                                let stage = start_root_stage(rank, "save error statistics");
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
                                finish_root_stage(rank, "save error statistics", stage);
                                let stage = start_root_stage(rank, "save rank statistics");
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save rank statistics", stage);
                                if self.output_options.factors_cn{
                                    let stage =
                                        start_root_stage(rank, "save factor condition numbers");
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                    finish_root_stage(rank, "save factor condition numbers", stage);
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
                                let stage = start_root_stage(rank, "save time statistics");
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                finish_root_stage(rank, "save time statistics", stage);

                                if self.output_options.factors_cn{
                                    let stage =
                                        start_root_stage(rank, "save factor condition numbers");
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                    finish_root_stage(rank, "save factor condition numbers", stage);
                                }

                                if self.output_options.plot {
                                    let stage = start_root_stage(rank, "render time piechart");
                                    time_piechart(id_tol.into(), &path_str);
                                    finish_root_stage(rank, "render time piechart", stage);
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
