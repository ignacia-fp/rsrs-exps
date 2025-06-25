use crate::io::plot_results::time_piechart;
use crate::io::read_and_write::{save_error_stats, save_rank_stats, save_time_stats, Iterations};
use crate::io::solve::{solve_system_so, solve_prec_system_so};
use bempp_rsrs::rsrs::rsrs_factors::{LocalFrom, RsrsOperator};
use crate::io::structured_operator::one_dim_to_bempp_points;
use crate::io::structured_operator::{
    get_bempp_points, GeometryType, StructuredOperator, StructuredOperatorImpl,
    StructuredOperatorInterface, StructuredOperatorParams,
};
use crate::io::structured_operators_types::StructuredOperatorType;
use bempp::boundary_assemblers::BoundaryAssemblerOptions;
use bempp::evaluator_tools::grid_points_from_space;
use bempp_octree::Octree;
use bempp_rsrs::rsrs::rsrs_cycle::{Rsrs, RsrsOptions};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rlst::prelude::*;
use serde::Deserialize;
type Real<T> = <T as rlst::RlstScalar>::Real;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::ReferenceCellType;

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
}

pub struct TestParams<Item: RlstScalar> {
    scenario_params: ScenarioOptions<Item>,
    rsrs_params: RsrsOptions<Item>,
}

pub struct TestFramework<Item: RlstScalar> {
    output_options: OutputOptions,
    test_params: TestParams<Item>,
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
        }
        .to_string();
        let structured_operator = self.get_structured_operator_name();
        let version = self.rsrs_params.to_identifier();

        let (h, kappa) = self.scenario_params.dim_args[dim_num];

        let dim_part = format!("mesh_width_{:e}", h);
        let kappa = format!("{:.2}", kappa);

        let filename = format!(
            "{}_{}_{}_{}",
            geometry,
            structured_operator,
            dim_part,
            kappa, // temporarily omit version
        );

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
    ) -> Self {
        Self {
            id_tols,
            dim_args,
            geometry_type,
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
            })
            .collect();
        Self {
            id_tols: args.id_tols,
            dim_args,
            structured_operator_type: data_type.structured_operator_type,
            geometry_type: args.geometry_type,
            precision: data_type.precision,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputOptions {
    solve: Solve,
    plot: bool,
    dense_errors: bool,
    results_output: Results,
}

impl OutputOptions {
    pub fn new(solve: Solve, plot: bool, dense_errors: bool, results_output: Results) -> Self {
        Self {
            solve,
            plot,
            dense_errors,
            results_output,
        }
    }
}

pub trait TestFrameworkImpl<Item: RlstScalar> {
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
        impl TestFrameworkImpl<$scalar> for TestFramework<$scalar> {
            type Item = $scalar;
            fn new(
                scenario_args: ScenarioOptions<$scalar>,
                rsrs_options: RsrsOptions<$scalar>,
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
                    if matches!(
                        self.test_params
                            .scenario_params
                            .structured_operator_type
                            .clone(),
                        StructuredOperatorType::BemppRsLaplaceOperator
                    ) {
                        let universe = mpi::initialize().unwrap();
                        let world = universe.world();
                        let rank = world.rank();

                        let refinement_level = 3;
                        let local_tree_depth = 4;
                        let global_tree_depth = 4;
                        let expansion_order = 6;
                        let grid = bempp::shapes::regular_sphere::<f64, _>(
                            refinement_level as u32,
                            1,
                            &world,
                        );


                        let quad_degree = 6;
                        // Get the number of cells in the grid.

                        let space = bempp::function::FunctionSpace::new(
                                &grid,
                                &LagrangeElementFamily::<f64>::new(
                                    1,
                                    ndelement::types::Continuity::Standard,
                                ),
                            );


                        let mut options = BoundaryAssemblerOptions::default();
                        options.set_regular_quadrature_degree(
                            ReferenceCellType::Triangle,
                            quad_degree,
                        );

                        let qrule =
                            options.get_regular_quadrature_rule(ReferenceCellType::Triangle);

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
                        let qrule = bempp_quadrature::simplex_rules::simplex_rule_triangle(1).unwrap();

                        let points = one_dim_to_bempp_points(&grid_points_from_space(&space, &qrule.points));

                        let dim = points.len();
                        let max_level: usize = 6;
                        let max_leaf_points: usize = 50;
                        let tree: Octree<'_, SimpleCommunicator> =
                            Octree::new(&points, max_level, max_leaf_points, &comm);
                        let global_number_of_points: usize = tree.global_number_of_points();
                        let global_max_level: usize = tree.global_max_level();

                        /*let iterations = Iterations {
                            no_prec: None,
                            prec: None,
                        };*/

                        let rhs = zero_element(kifmm_evaluator.domain());
                        rhs.view_mut().local_mut().fill_from_seed_equally_distributed(rank as usize);

                        let mut iterations = Iterations {
                            no_prec: None,
                            prec: None,
                        };

                        iterations.no_prec = match self.output_options.solve {
                            Solve::True(tol) => {
                                let mut residuals = Vec::<$scalar>::new();
                                let gmres = GmresIteration::new(operator.r(), rhs.r(), dim)
                                    .set_callable(|_, res| {
                                        residuals.push(res.into());
                                    })
                                    .set_tol(tol)
                                    .set_max_iter(150);
                                let (_sol, _res) = gmres.run();
                                Some(residuals.len())},
                            Solve::False => None,
                        };

                        if comm.rank() == 0 {
                            println!(
                                "Setup octree with {} points and maximum level {}",
                                global_number_of_points, global_max_level
                            );
                        }

                        for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                            let mut iterations = iterations.clone();
                            //let iterations = iterations.clone();

                            println!("Test: {} points, tol:{}", dim, id_tol);

                            self.test_params.rsrs_params.id_options.tol_id = id_tol;

                            let mut rsrs_algo: Rsrs<Self::Item> =
                                Rsrs::new(dim, &tree, self.test_params.rsrs_params.clone());

                            let mut rsrs_factors = rsrs_algo.run(&operator);

                            let path_str = self.test_params.get_test_dir(dim_num);

                            iterations.prec = match self.output_options.solve {
                                Solve::True(tol) => {
                                    let mut rsrs_operator = RsrsOperator::from_local(&mut rsrs_factors);
                                    rsrs_operator.set_inv(true);
                                    rhs = rsrs_operator.apply(rhs, TransMode::NoTrans);

                                    let op = rsrs_operator.product(operator);

                                    let mut residuals = Vec::<$scalar>::new();
                                    let gmres = GmresIteration::new(op.r(), rhs.r(), dim)
                                        .set_callable(|_, res| {
                                            residuals.push(res);
                                        })
                                        .set_tol(tol)
                                        .set_max_iter(100);
                                    let (_sol, _res) = gmres.run();
                                    
                                    Some(residuals.len())
                                }
                                Solve::False => None,
                            };

                            match self.output_options.results_output {
                                Results::All => {
                                    save_error_stats::<$scalar>(
                                        &operator,
                                        &mut rsrs_factors,
                                        &rsrs_algo,
                                        iterations,
                                        id_tol,
                                        &path_str,
                                        self.output_options.dense_errors,
                                    );
                                    save_time_stats(&rsrs_algo, id_tol, &path_str);
                                    save_rank_stats(&rsrs_algo, id_tol, &path_str);

                                    if self.output_options.plot {
                                        time_piechart(id_tol, &path_str);
                                    }
                                }
                                Results::Rank => {
                                    save_error_stats::<$scalar>(
                                        &operator,
                                        &mut rsrs_factors,
                                        &rsrs_algo,
                                        iterations,
                                        id_tol,
                                        &path_str,
                                        self.output_options.dense_errors,
                                    );
                                    save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                }
                                Results::Time => {
                                    save_error_stats::<$scalar>(
                                        &operator,
                                        &mut rsrs_factors,
                                        &rsrs_algo,
                                        iterations,
                                        id_tol,
                                        &path_str,
                                        self.output_options.dense_errors,
                                    );
                                    save_time_stats(&rsrs_algo, id_tol, &path_str);
                                    if self.output_options.plot {
                                        time_piechart(id_tol, &path_str);
                                    }
                                }
                            }
                        }

                    } else {
                        let structured_operator_params = StructuredOperatorParams::new(
                            self.test_params
                                .scenario_params
                                .structured_operator_type
                                .clone(),
                            self.test_params.scenario_params.precision.clone(),
                            self.test_params.scenario_params.geometry_type.clone(),
                            dim_arg.0,
                            dim_arg.1,
                        );
                        let structured_operator: StructuredOperatorInterface =
                            <StructuredOperatorInterface as StructuredOperatorImpl<$scalar>>::new(
                                structured_operator_params,
                            );
                        let dim = structured_operator.n_points;
                        let operator = StructuredOperator::from_local(&structured_operator);
                        let points: Vec<bempp_octree::Point> =
                            get_bempp_points(&structured_operator).unwrap();


                        let max_level: usize = 6;
                        let max_leaf_points: usize = 50;
                        let tree: Octree<'_, SimpleCommunicator> =
                            Octree::new(&points, max_level, max_leaf_points, &comm);
                        let global_number_of_points: usize = tree.global_number_of_points();
                        let global_max_level: usize = tree.global_max_level();

                        /*let iterations = Iterations {
                            no_prec: None,
                            prec: None,
                        };*/

                        let mut iterations = Iterations {
                            no_prec: None,
                            prec: None,
                        };

                        iterations.no_prec = match self.output_options.solve {
                            Solve::True(tol) => Some(solve_system_so(operator.clone(), tol)),
                            Solve::False => None,
                        };

                        if comm.rank() == 0 {
                            println!(
                                "Setup octree with {} points and maximum level {}",
                                global_number_of_points, global_max_level
                            );
                        }

                        for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                            let mut iterations = iterations.clone();
                            //let iterations = iterations.clone();

                            println!("Test: {} points, tol:{}", dim, id_tol);

                            self.test_params.rsrs_params.id_options.tol_id = id_tol;

                            let mut rsrs_algo: Rsrs<Self::Item> =
                                Rsrs::new(dim, &tree, self.test_params.rsrs_params.clone());

                            let mut rsrs_factors = rsrs_algo.run(&operator);

                            let path_str = self.test_params.get_test_dir(dim_num);

                            iterations.prec = match self.output_options.solve {
                                Solve::True(tol) => {
                                    Some(solve_prec_system_so(operator.clone(), &mut rsrs_factors, tol))
                                }
                                Solve::False => None,
                            };

                            match self.output_options.results_output {
                                Results::All => {
                                    save_error_stats::<$scalar>(
                                        &operator,
                                        &mut rsrs_factors,
                                        &rsrs_algo,
                                        iterations,
                                        id_tol,
                                        &path_str,
                                        self.output_options.dense_errors,
                                    );
                                    save_time_stats(&rsrs_algo, id_tol, &path_str);
                                    save_rank_stats(&rsrs_algo, id_tol, &path_str);

                                    if self.output_options.plot {
                                        time_piechart(id_tol, &path_str);
                                    }
                                }
                                Results::Rank => {
                                    save_error_stats::<$scalar>(
                                        &operator,
                                        &mut rsrs_factors,
                                        &rsrs_algo,
                                        iterations,
                                        id_tol,
                                        &path_str,
                                        self.output_options.dense_errors,
                                    );
                                    save_rank_stats(&rsrs_algo, id_tol, &path_str);
                                }
                                Results::Time => {
                                    save_error_stats::<$scalar>(
                                        &operator,
                                        &mut rsrs_factors,
                                        &rsrs_algo,
                                        iterations,
                                        id_tol,
                                        &path_str,
                                        self.output_options.dense_errors,
                                    );
                                    save_time_stats(&rsrs_algo, id_tol, &path_str);
                                    if self.output_options.plot {
                                        time_piechart(id_tol, &path_str);
                                    }
                                }
                            }
                        }

                    };

                    /*let structured_operator_params = StructuredOperatorParams::new(
                        self.test_params
                            .scenario_params
                            .structured_operator_type
                            .clone(),
                        self.test_params.scenario_params.precision.clone(),
                        self.test_params.scenario_params.geometry_type.clone(),
                        dim_arg.0,
                        dim_arg.1,
                    );
                    let structured_operator: StructuredOperator =
                        <StructuredOperator as StructuredOperatorImpl<$scalar>>::new(
                            structured_operator_params,
                        );
                    let dim = structured_operator.n_points;
                    let operator = StructuredOperatorOperator::from_local(&structured_operator);
                    let points: Vec<bempp_octree::Point> =
                        get_bempp_points(&structured_operator).unwrap();

                    let max_level: usize = 6;
                    let max_leaf_points: usize = 50;
                    let tree: Octree<'_, SimpleCommunicator> =
                        Octree::new(&points, max_level, max_leaf_points, &comm);
                    let global_number_of_points: usize = tree.global_number_of_points();
                    let global_max_level: usize = tree.global_max_level();

                    /*let iterations = Iterations {
                        no_prec: None,
                        prec: None,
                    };*/

                    let mut iterations = Iterations {
                        no_prec: None,
                        prec: None,
                    };

                    iterations.no_prec = match self.output_options.solve {
                        Solve::True(tol) => Some(solve_system(operator.clone(), tol)),
                        Solve::False => None,
                    };

                    if comm.rank() == 0 {
                        println!(
                            "Setup octree with {} points and maximum level {}",
                            global_number_of_points, global_max_level
                        );
                    }

                    for &id_tol in self.test_params.scenario_params.id_tols.iter() {
                        let mut iterations = iterations.clone();
                        //let iterations = iterations.clone();

                        println!("Test: {} points, tol:{}", dim, id_tol);

                        self.test_params.rsrs_params.id_options.tol_id = id_tol;

                        let mut rsrs_algo: Rsrs<Self::Item> =
                            Rsrs::new(dim, &tree, self.test_params.rsrs_params.clone());

                        let mut rsrs_factors = rsrs_algo.run(&operator);

                        let path_str = self.test_params.get_test_dir(dim_num);

                        iterations.prec = match self.output_options.solve {
                            Solve::True(tol) => {
                                Some(solve_prec_system(operator.clone(), &mut rsrs_factors, tol))
                            }
                            Solve::False => None,
                        };

                        match self.output_options.results_output {
                            Results::All => {
                                save_error_stats::<$scalar>(
                                    &operator,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    iterations,
                                    id_tol,
                                    &path_str,
                                    self.output_options.dense_errors,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);

                                if self.output_options.plot {
                                    time_piechart(id_tol, &path_str);
                                }
                            }
                            Results::Rank => {
                                save_error_stats::<$scalar>(
                                    &operator,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    iterations,
                                    id_tol,
                                    &path_str,
                                    self.output_options.dense_errors,
                                );
                                save_rank_stats(&rsrs_algo, id_tol, &path_str);
                            }
                            Results::Time => {
                                save_error_stats::<$scalar>(
                                    &operator,
                                    &mut rsrs_factors,
                                    &rsrs_algo,
                                    iterations,
                                    id_tol,
                                    &path_str,
                                    self.output_options.dense_errors,
                                );
                                save_time_stats(&rsrs_algo, id_tol, &path_str);
                                if self.output_options.plot {
                                    time_piechart(id_tol, &path_str);
                                }
                            }
                        }
                    }*/
                }
            }
        }
    };
}

implement_test_framework!(f64);
implement_test_framework!(c64);
