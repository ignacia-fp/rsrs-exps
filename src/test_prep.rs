use crate::io::plot_results::time_piechart;
use crate::io::read_and_write::{save_error_stats, save_rank_stats, save_time_stats, Solves};
use crate::io::solve::solve_system;
use crate::io::structured_operator::LocalFrom;
use crate::io::structured_operator::{
    get_bempp_points, GeometryType, StructuredOperator, StructuredOperatorImpl,
    StructuredOperatorInterface, StructuredOperatorParams,
};
use crate::io::structured_operators_types::StructuredOperatorType;
use bempp_octree::Octree;
use bempp_rsrs::rsrs::rsrs_cycle::{Rsrs, RsrsOptions};
use bempp_rsrs::rsrs::rsrs_factors::{LocalFromSpaces, RsrsOperator};
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
use bempp_rsrs::rsrs::rsrs_factors::Inv;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::{Entity, Geometry, Grid, ParallelGrid, Point};
use num::ToPrimitive;
use rlst::tracing::trace_call;
use std::fs;
use std::io;
use std::path::Path;

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
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataType {
    pub structured_operator_type: StructuredOperatorType,
    pub precision: Precision,
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
}

pub struct TestParams<Item: RlstScalar> {
    scenario_params: ScenarioOptions<Item>,
    rsrs_params: RsrsOptions<Item>,
}

pub struct TestFramework<Item: RlstScalar> {
    output_options: OutputOptions,
    test_params: TestParams<Item>,
}

fn move_if_exists<P: AsRef<Path>>(src: P, dst_dir: P) -> io::Result<()> {
    let src = src.as_ref();
    let dst_dir = dst_dir.as_ref();

    if src.exists() {
        // Create destination directory if it doesn't exist
        if !dst_dir.exists() {
            fs::create_dir_all(dst_dir)?;
        }

        // Build the destination path (preserve filename)
        let file_name = src
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Source has no filename"))?;
        let dst_path = dst_dir.join(file_name);

        // Move the file (rename = move)
        fs::rename(src, &dst_path)?;
    }

    Ok(())
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

            format!("{}_{}_{}", geometry, structured_operator, dim_pred)
        } else {
            let (h, kappa) = self.scenario_params.dim_args[dim_num];
            let dim_pred = format!(
                "mesh_width_{:.2e}_od_{}",
                h, self.scenario_params.max_tree_depth
            );
            let kappa = format!("{:.2}", kappa);
            format!(
                "{}_{}_{}_{}",
                geometry, structured_operator, dim_pred, kappa
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
    ) -> Self {
        Self {
            id_tols,
            dim_args,
            geometry_type,
            max_tree_depth,
            n_sources,
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
                DimArg::MeshWidth(h) => (num::Zero::zero(), *h),
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
                    let structured_operator_params = StructuredOperatorParams::new(
                        self.test_params
                            .scenario_params
                            .structured_operator_type
                            .clone(),
                        self.test_params.scenario_params.precision.clone(),
                        self.test_params.scenario_params.geometry_type.clone(),
                        dim_arg.0,
                        dim_arg.1,
                        self.test_params.scenario_params.n_sources,
                    );

                    let structured_operator: StructuredOperatorInterface =
                        <StructuredOperatorInterface as StructuredOperatorImpl<Self::Item>>::new(
                            &structured_operator_params,
                        );
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

                    let path_str = self.test_params.get_test_dir(dim_num);

                    let test_path = Path::new(&path_str).join("test_file.h5");
                    let sketch_path = Path::new(&path_str).join("sketch_file.h5");
                    let _ = move_if_exists(test_path.to_str().unwrap(), ".");
                    let _ = move_if_exists(sketch_path.to_str().unwrap(), ".");

                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                        let max_leaf_points = if id_tol < 1.0 {
                            50
                        } else {
                            2 * id_tol.to_usize().unwrap()
                        };
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

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(&tree, self.test_params.rsrs_params.clone(), dim);

                        let domain = std::rc::Rc::clone(&operator.domain());
                        let range = std::rc::Rc::clone(&operator.range());

                        let mut rsrs_factors = rsrs_algo.run(operator.r());

                        let mut rsrs_operator =
                            RsrsOperator::from_local_spaces(&mut rsrs_factors, domain, range);

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
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.plot {
                                    time_piechart(id_tol, &path_str);
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
                                save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);

                                if self.output_options.factors_cn {
                                    let cn: ConditionNumberOutput<$scalar> =
                                        ConditionNumberOutput::new(
                                            rsrs_operator.get_condition_numbers(),
                                        );
                                    cn.save(&path_str, id_tol);
                                }

                                if self.output_options.plot {
                                    time_piechart(id_tol, &path_str);
                                }
                            }
                        }
                    }

                    let _ = move_if_exists("test_file.h5", &path_str);
                    let _ = move_if_exists("sketch_file.h5", &path_str);
                }
            }
        }
    };
}

implement_test_framework!(f64);
implement_test_framework!(c64);

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

                    let kifmm_evaluator =
                        bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator::from_spaces(
                            &space,
                            &space,
                            green_kernels::types::GreenKernelEvalType::Value,
                            local_tree_depth,
                            global_tree_depth,
                            expansion_order,
                            &qrule.points,
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
                        let mut barycentre = [0.0; 3];
                        for point in cell.geometry().points() {
                            point.coords(&mut p);
                            barycentre[0] += p[0]/3.0;
                            barycentre[1] += p[1]/3.0;
                            barycentre[2] += p[2]/3.0;
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

                    let test_path = Path::new(&path_str).join("test_file.h5");
                    let sketch_path = Path::new(&path_str).join("sketch_file.h5");
                    let _ = move_if_exists(test_path.to_str().unwrap(), ".");
                    let _ = move_if_exists(sketch_path.to_str().unwrap(), ".");

                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {

                        let max_leaf_points = if id_tol < 1.0 {
                            50
                        } else {
                            2 * id_tol.to_usize().unwrap()
                        };

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

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(&tree, self.test_params.rsrs_params.clone(), dim);
                        let mut rsrs_operator = rsrs_algo.get_rsrs_operator(operator.r());



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
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.plot {
                                    time_piechart(id_tol, &path_str);
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
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.factors_cn{
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                }
                            }
                            Results::Time => {
                                save_error_stats(
                                    &operator,
                                    &mut rsrs_operator,
                                    &rsrs_algo,
                                    solves,
                                    id_tol,
                                    &path_str,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);

                                if self.output_options.factors_cn{
                                    let cn: ConditionNumberOutput<$scalar> = ConditionNumberOutput::new(rsrs_operator.get_condition_numbers());
                                    cn.save(&path_str, id_tol);
                                }

                                if self.output_options.plot {
                                    time_piechart(id_tol, &path_str);
                                }
                            }
                        }
                    }
                    let _ = move_if_exists("test_file.h5", &path_str);
                    let _ = move_if_exists("sketch_file.h5", &path_str);
                }
            }
        }
    }
}

implement_distributed_test_framework!(f64);
