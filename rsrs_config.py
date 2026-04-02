import json
import numpy as np
import subprocess
from typing import List, Dict, Union
import re
from pathlib import Path
import matplotlib.pyplot as plt
import os
import importlib
import _rsrs_config_results as _config_results

pi = np.pi

def camel_to_snake(name):
    # Insert underscore before each uppercase letter, except at the start
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Handle cases where multiple uppercase letters are consecutive (e.g., 'getHTTPResponse')
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

def sci_no_padding(x):
    base, exp = f"{x:.1e}".split("e")  # use more precision
    base = base.rstrip("0").rstrip(".")  # remove trailing zeros and dot
    return f"{base}e{int(exp)}"

def sci_no_padding2(x):
    base, exp = f"{x:.2e}".split("e")  # use more precision
    base = base.rstrip("0").rstrip(".")  # remove trailing zeros and dot
    return f"{base}e{int(exp)}"

def rust_float_format(x: float, precision: int = 2) -> str:
    s = f"{x:.{precision}e}"
    # Remove the leading zero in the exponent (e.g. e-01 -> e-1)
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s

def pivot_method(kind, value=0.0):
    if kind == "Lu":
        return {"type": kind, "value": value}
    else:
        return {"type": kind}
    
def qr_method(kind, value=0.0):
    if kind == "SRRQR":
        return {"SRRQR": value}
    else:
        return "RRQR"

def stab(value=0.0):
    if value != 0.0:
        return {"type": "True", "value": value}
    else:
        return {"type": "False"}

class RSRSBenchmarkConfig:
    def __init__(
        self,
        operator_type: int = 0,
        precision: int = 1,
        h: float = 0.1,
        ref_level: int = None,
        depth: int = None,
        kappa: float = None,
        id_tols: List[float] = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
        f: float = 1.01,
        dim_arg_type: int = 2,
        geometry: int = 0,
        solve_tol: float = 1e-10,
        solve: bool = True,
        plot: bool = True,
        dense_errors: bool = False,
        factors_cn: bool = False,
        results_output: int = 0,
        null_method: int = 0,
        block_extraction_method: int = 0,
        pivot_method: int = 0,
        rrqr: int = 0,
        rank_picking: int = 0,
        min_rank: int = 1,
        min_level: int = 1,
        max_tree_depth: int = 6,
        lu_stab = 0,
        diag_stab = 0,
        op_shift = 0,
        oversampling_near = 8,
        oversampling_diag = 16,
        tol_ext_near = 1e-16,
        tol_ext_diag = 1e-16,
        fact_type = 1,
        n_sources: int = 1,
        save_samples: bool = True,
        load_samples: bool = True,
        num_threads: int = 32,
        min_num_samples: int = 0,
        symmetry: int = 0,
        flush_factors: bool = False,
        store_far: bool = False,
        initial_num_samples = 0,
        assembler = 0,
        symmetric = None
    ):
        
        """
        Initialize RSRS Benchmark Configuration.

        Parameters
        ----------
        operator_type : int, optional
            Index selecting the structured operator type to use.
            Options include:
            0: BasicStructuredOperator (default)
            1: BemppClLaplaceSingleLayer
            2: BemppClHelmholtzSingleLayer
            3: KiFMMLaplaceOperator
            4: KiFMMHelmholtzOperator
            5: BemppRsLaplaceOperator
            6: BemppClLaplaceCombined
            7: BemppClLaplaceSingleLayerCP
            8: BemppClLaplaceSingleLayerMM
            9: BemppClHelmholtzSingleLayerCP
            10: BemppClLaplaceSingleLayerCPID
            11: BemppClLaplaceSingleLayerP1,
            12: KiFMMLaplaceOperatorV,
            13: BemppClLaplaceCombinedP1,
            14: BemppClLaplaceSingleLayerCPIDP1,
            15: BemppClHelmholtzSingleLayerCPID,
            16: BemppClMaxwellEfie
            17: BemppClHelmholtzSingleLayerP1
            18: BemppClBurtonMiller
            19: BemppCLHelmholtzCombined
            20: BemppClLaplaceSecond
            The choice affects the problem type and required parameters and more kernels can be addded in python/structured_operators.py

        precision : int, optional
            Index for numerical precision:
            0: Single precision (not fully enabled)
            1: Double precision (default)

        h : float, optional
            Characteristic meshwidth used in spatial discretization.
            For some dimension argument types, this can be computed internally from `kappa`.
            Default is 0.1.

        ref_level : int or None, optional
            Refinement level, used only if `dim_arg_type` corresponds to the refinement level that is 
            internally used by bempp-rs on a unit sphere ("RefinementLevelAndDepth"). Both `ref_level` and `depth`
            must be provided in this case. Default is None.

        depth : int or None, optional
            Depth of the fmm tree when using BemppRs operators.
            Required if `dim_arg_type` is "RefinementLevelAndDepth". Default is None.

        kappa : float or None, optional
            Wavenumber for oscillatory problems (Helmholtz type).
            Used to compute meshwidth if `dim_arg_type` is "Kappa".
            Required for Helmholtz operators.
            Default is None.

        id_tols : List[float], optional
            List of interpolation decomposition tolerances (ID tolerances).
            One test run will be executed for each tolerance value.
            Defaults to [1e-2, 1e-3, 1e-4, 1e-6, 1e-8].

        dim_arg_type : int, optional
            Index specifying the way spatial discretization parameters are provided:
            0: "Kappa" — compute h internally as 2π / (8 * kappa)
            1: "KappaAndMeshwidth" — provide both kappa and h explicitly
            2: "Meshwidth" — provide h only, no kappa
            3: "RefinementLevelAndDepth" — use multilevel structure parameters ref_level and depth
            Default is 2 ("Meshwidth").

        geometry : int, optional
            Index specifying the geometry shape used in the test.
            Options include:
            0: SphereSurface (default)
            1: CubeSurface
            2: CylinderSurface
            3: EllipsoidSurface
            4: Dihedral
            5: Device
            6: F16
            7: RidgedHorn
            8: EMCCAlmond
            9: FrigateHull
            10: Plane
            11: Square
            12: Cube
            Note: For Bempp-rs operators, only SphereSurface is supported.

        solve_tol : float, optional
            Tolerance for the GMRES linear solver if `solve` is True.
            Default is 1e-5.

        solve : bool, optional
            Whether to solve the linear system Ax = b using GMRES.
            Default is True.

        plot : bool, optional
            Whether to generate a pie chart showing execution time breakdown after tests.
            Default is True.

        dense_errors : bool, optional
            If True, compute RSRS errors on the dense matrix form of blocks.
            This is memory and time intensive; recommended only for small problems.
            Default is False.
        
        factors_cn: bool, optional
            If True, the algorithm will compute and save the condition numbers of all the small
            factors in the rsrs operator.

        results_output : int, optional
            Index controlling what results are output from the test:
            0: "All" — output all available information (default)
            1: "Rank" — output compression ranks and errors
            2: "Time" — output timing information and errors

        null_method : int, optional
            Index selecting method to nullify matrices:
            0: "Projection" (fastest, using I - Ω⁺Ω)
            1: "Svd"
            2: "Qr"

        block_extraction_method : int, optional
            Index selecting solver for least squares problems during block extraction:
            0: "LuLstSq" (solves normal equations with LU, fastest)
            1: "Svd"

        pivot_method : int, optional
            Index selecting pivot strategy for LU or matrix inversion:
            0: "Lu"
            1: "DirectInversion"

        rank_picking : int, optional
            Index selecting fixed-rank merging strategy when combining blocks:
            0: "Min" — use smallest skeleton size
            1: "Max" — use largest skeleton size
            2: "Avg" — use average skeleton size
            3: "Mid" — use median skeleton size
            4: "DoubleMin" — use double the minimum rank of previous level
            5: "Tol" — use the defined tolerance as rank

        min_rank: int, optional,
            Minimum rank that a block should have to be processed for RSRS.
            Default is 1.

        min_level: int, optional
            Minimum octree level.
            Default is 1.

        max_tree_depth: int, optional
            Maximum tree depth for the octree.
            Default is 6.

        fact_type: int, optional
            Type of factorisation:
            0: Split
            1: Joint
            Default is 1.

        n_sources : int, optional
            The number of distinct sources to consider during the computation.
            Default is 1.

        assembler: int, optional
            Type of assembler
            0: Dense
            1: FMM

        symmetry: int, optional
            Type of symmetry
            0: NoSymm
            1: Symmetric
            2: Hermitian

        Raises
        ------
        ValueError
            If required parameters for certain configurations are missing or inconsistent, such as:
            - `ref_level` or `depth` missing when `dim_arg_type` is "RefinementLevelAndDepth".
            - `kappa` missing when required for Helmholtz operators.
            - Invalid geometry for Bempp-rs operators.
            - `dense_errors` enabled for Bempp-rs operator which has no dense form.

        Notes
        -----
        - The parameter indices correspond to internal lists of options documented above.
        - For most operators, `kappa` is required if `dim_arg_type` is "Kappa" or "KappaAndMeshwidth".
        - This class constructs shell script commands to run benchmarks with the configured parameters.
        """

        ##### Construction of the test


        ## Data Type Arguments (what operator, which precision):
        self.structured_operator_types = [
            "BasicStructuredOperator", "BemppClLaplaceSingleLayer", "BemppClHelmholtzSingleLayer",
            "KiFMMLaplaceOperator", "KiFMMHelmholtzOperator", "BemppRsLaplaceOperator", "BemppClLaplaceCombined",
            "BemppClLaplaceSingleLayerCP", "BemppClLaplaceSingleLayerMM", "BemppClHelmholtzSingleLayerCP", "BemppClLaplaceSingleLayerCPID",
            "BemppClLaplaceSingleLayerP1", "KiFMMLaplaceOperatorV", "BemppClLaplaceCombinedP1", "BemppClLaplaceSingleLayerCPIDP1",
            "BemppClHelmholtzSingleLayerCPID", "BemppClMaxwellEfie", "BemppClHelmholtzSingleLayerP1", "BemppClBurtonMiller", "BemppClHelmholtzCombined",
            "BemppClLaplaceSecond", "BIEGrid"
        ]
        self.precision_types = ["Single", "Double"] # Single precision methods have not been enabled yet

        ## Scenario Arguments:
        # dim_args options:
        # - Kappa: algorithm will take kappa and set the characteristic meshwidth to be h = 2.0 * pi / (8.0 * kappa) (8 points per wavelength)
        # - KappaAndMeshwidth: receives a tuple, (kappa, h), so kappa is independent of h.
        # - MeshWidth: In case there is not an associated wavenumber and the user wants to set h.
        # - RefinementLevelAndDepth: both must be provided, otherwise a ValueError will be raised
        self.dim_arg_types = ["Kappa", "KappaAndMeshwidth", "Meshwidth", "RefinementLevelAndDepth"]

        ## Geometry Type:
        ## !!! For Bempp-Rs tests we only have a sphere for now, so this parameter is irrelevant
        self.geometry_types = ["SphereSurface", "CubeSurface", "CylinderSurface", "EllipsoidSurface", "Dihedral", "Device", "F16", "RidgedHorn", "EMCCAlmond", "FrigateHull", "Plane", "Square", "Cube"] ## We can generate more with Gmsh and link them

        ## Output Arguments: (outputs that the test the test will return)
        # In all cases, it returns the errors ||A_app^{-1}A - I||_2 and ||A_app - A||_2/||A||_2,
        # but we can also either ask for the compression results (Rank) or time that it takes
        # to complete each RSRS step.
        self.results_outputs = ["All", "Rank", "Time"]

        ## Run Type: 
        ## RSRS can either run all the ID steps in parallel and batch LU or batch both ID and LU and run one after the other
        self.fact_types = ["Split", "Joint"]

        ## RSRS arguments:
        # Method used to nullifying matrix.
        # The cheapest way is to build the projector: I - \Omega^+\Omega,
        # but SVD and QR are also allowed.
        self.null_methods = ["Projection", "Svd", "Qr"]

        # Method used to solve the least squares problem.
        # The cheapest is to solve normal equations through LU.
        self.block_extraction_methods = ["LuLstSq", "Svd"]

        # Way to compute the pivot or the inverse of a squared matrix.
        self.pivot_methods = ["Lu", "DirectInversion"]

        # Rank picking strategy:
        self.rank_pickings = ["Min", ## After the leaf level, when merging boxes, it looks for the smallest skeleton size and assigns it as the new box's fixed rank.
                "Max", ## Same as before, but it takes the largest skeleton size.
                "Avg", ## Same as before, but it averages the skeleton sizes.
                "Mid", ## Same as before, but it takes the skeleton size that is in the middle of the set.
                "DoubleMin", ## Takes double the minimum rank of the previous level as a fixed rank.
                "Tol", ## Always takes the defined tolerance.
                ]
        
        self.rrqr_keys = ["RRQR", "SRRQR"]

        self.assembler_types = ["Dense", "FMM"]

        self.symmetry_type = ["NoSymm", "Symmetric", "Hermitian"]

        # Store index-based options
        self.operator_type_index = operator_type
        self.precision_index = precision
        self.dim_arg_type_index = dim_arg_type
        self.geometry = geometry
        self.results_output = results_output
        self.null_method_index = null_method
        self.block_extraction_method_index = block_extraction_method
        self.pivot_method_index = pivot_method
        self.rank_picking_index = rank_picking
        self.rrqr_index = rrqr
        self.lu_stab = lu_stab
        self.diag_stab = diag_stab
        self.op_shift = op_shift
        self.fact_type = fact_type
        self.min_num_samples = min_num_samples
        self.assembler_index = assembler
        self.symmetry_index = symmetry

        # Store other parameters
        self.h = h ## Characteristic meshwidth
        self.kappa = kappa ## Wavenumber of oscillatory problems. It must be defined for these cases. 
                           ## It also serves as a parameter to compute h when provided.

        self.ref_level = ref_level
        self.depth = depth
        self.f = f
        self.id_tols = id_tols
        self.solve_tol = solve_tol
        self.solve = solve
        self.plot = plot
        self.dense_errors = dense_errors
        self.factors_cn = factors_cn
        self.min_rank = min_rank
        self.min_level = min_level
        self.max_tree_depth = max_tree_depth
        self.oversampling_near = oversampling_near
        self.oversampling_diag = oversampling_diag
        self.tol_ext_near = tol_ext_near
        self.tol_ext_diag = tol_ext_diag
        self.n_sources = n_sources
        self.save_samples = save_samples
        self.load_samples = load_samples
        self.num_threads = num_threads
        self.flush_factors = flush_factors
        self.store_far = store_far
        self.initial_num_samples = initial_num_samples
        self.symmetric = symmetric
        if self.dim_arg_types[self.dim_arg_type_index] == "RefinementLevelAndDepth":
            if self.ref_level is None or self.depth is None:
                raise ValueError("Both `ref_level` and `depth` must be specified for 'RefinementLevelAndDepth' dim_arg_type.")
        
        if self.dense_errors:
            print("WARNING: computing dense errors is time and memory intensive. Use only for small test cases")

        if self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayer" or self.structured_operator_types[self.operator_type_index] == "KiFMMHelmholtzOperator" or self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayerCP":
            if self.kappa == None:
                raise ValueError("A wavenumber must be provided.")

        #if self.structured_operator_types[self.operator_type_index] != "BemppClHelmholtzSingleLayer" and self.structured_operator_types[self.operator_type_index] != "KiFMMHelmholtzOperator" and self.structured_operator_types[self.operator_type_index] != "BemppRsLaplaceOperator" and self.structured_operator_types[self.operator_type_index] != "BemppClHelmholtzSingleLayerCP" and  self.structured_operator_types[self.operator_type_index] != "BemppClHelmholtzSingleLayerCPID" and self.structured_operator_types[self.operator_type_index] != "BemppClMaxwellEfie":
        if self.dim_arg_types[self.dim_arg_type_index] == "Kappa":

            if self.kappa != None:
                print("WARNING: Computing h from given kappa.")
                self.h = 2.0 * pi / (8.0 * self.kappa)
                print("h is " + str(self.h))
            else:
                raise ValueError("You must provide kappa for this option.")

        if self.structured_operator_types[self.operator_type_index] == "BemppRsLaplaceOperator":
            if self.geometry_types[self.geometry] != "SphereSurface":
                print("WARNING: For Bempp-rs we only have spherical surfaces for now.")
            if self.dim_arg_types[self.dim_arg_type_index] != "RefinementLevelAndDepth":
                raise ValueError("For Bempp-rs, 'RefinementLevelAndDepth' must be used to define the mesh width.")
            if self.dense_errors:
                raise ValueError("There is no dense form of this operator.")
            
        if self.rrqr_keys[self.rrqr_index] == "SRRQR":
            print("Using SRRQR with f = " + str(self.f))
        

    def data_type_args(self) -> Dict[str, str]:
        return {
            "structured_operator_type": self.structured_operator_types[self.operator_type_index],
            "precision": self.precision_types[self.precision_index],
        }

    def scenario_args(self) -> Dict[str, Union[List[float], List[Dict], str, int]]:
        dim_type = self.dim_arg_types[self.dim_arg_type_index]
        dim_args_map = {
            "Kappa": {"Kappa": self.kappa},
            "KappaAndMeshwidth": {"KappaAndMeshwidth": (self.kappa, self.h)},
            "Meshwidth": {"MeshWidth": self.h},
            "RefinementLevelAndDepth": {"RefinementLevelAndDepth": (self.ref_level, self.depth)},
        }
        return {
            "id_tols": self.id_tols,
            "dim_args": [dim_args_map[dim_type]],
            "geometry_type": self.geometry_types[self.geometry],
            "max_tree_depth": self.max_tree_depth,
            "n_sources": self.n_sources,
            "assembler": self.assembler_types[self.assembler_index]
        }

    def output_args(self) -> Dict:
        return {
            "solve": {"True": self.solve_tol} if self.solve else {"False": None},
            "plot": self.plot,
            "dense_errors": self.dense_errors,
            "factors_cn": self.factors_cn,
            "results_output": self.results_outputs[self.results_output],
        }

    def rsrs_args(self) -> Dict:
        return {
            "oversampling": self.oversampling_near,  ## Oversampling for each individual block
            "oversampling_diag_blocks": self.oversampling_diag,  ## Oversampling used when extracting diagonal blocks when RSRS finishes
            "min_num_samples": self.min_num_samples,
            "initial_num_samples": self.initial_num_samples,  ## Initial num samples: useful only when sampling is done in parallel way (not active yet)
            "shift": stab(self.op_shift),
            "null_method": self.null_methods[self.null_method_index],
            "qr_method": qr_method(self.rrqr_keys[self.rrqr_index], value=self.f),
            "near_block_extraction_method": self.block_extraction_methods[self.block_extraction_method_index],
            "diag_block_extraction_method": self.block_extraction_methods[self.block_extraction_method_index],
            "lu_pivot_method": pivot_method(self.pivot_methods[self.pivot_method_index], value=self.lu_stab),
            "diag_pivot_method": pivot_method(self.pivot_methods[self.pivot_method_index], value=self.diag_stab),
            "tol_null": 1e-16,  ## Tolerance when nullifying blocks
            "tol_id": self.id_tols[0],  ## ID tolerance (Irrelevant, since it is set with the scenario arguments)
            "tol_ext_near": self.tol_ext_near,  ## Tolerance used to compute pseudo inverses when extracting near field.
            "tol_diag_ext": self.tol_ext_diag,  ## Tolerance used to compute pseudo inverses when extracting diagonal blocks.
            "min_rank": self.min_rank,  ## Minimum size of the box. If the box is smaller, it will be saved for the next level.
            "min_level": self.min_level, ## Level at which the algorithm stops
            "symmetry": self.symmetry_type[self.symmetry_index],  ## Indicates if we should run RSRS for symmetry matrices (half the time and memory)
            "rank_picking": self.rank_pickings[self.rank_picking_index],
            "fact_type": self.fact_types[self.fact_type],
            "save_samples": self.save_samples,
            "load_samples": self.load_samples,
            "num_threads": self.num_threads,
            "flush_factors": self.flush_factors,
            "store_far": self.store_far,
            "symmetric": self.symmetric,
        }

    def as_dict(self) -> Dict[str, Dict]:
        return {
            "data_type_args": self.data_type_args(),
            "scenario_args": self.scenario_args(),
            "output_args": self.output_args(),
            "rsrs_args": self.rsrs_args(),
        }
    
    def generate_bash_script(self, filename: str = "run_test.sh", rayon_threads: int | None = None):
        def json_for_bash(obj):
            return json.dumps(obj).replace("'", "'\"'\"'")

        data_type_args_json = json_for_bash(self.data_type_args())
        scenario_args_json = json_for_bash(self.scenario_args())
        output_args_json = json_for_bash(self.output_args())
        rsrs_args_json = json_for_bash(self.rsrs_args())

        bash_lines = [
            "#!/bin/bash",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export MKL_DOMAIN_NUM_THREADS=1",
            "export MKL_DYNAMIC=FALSE",
            "export GOTO_NUM_THREADS=1",
            "export BLIS_NUM_THREADS=1",
            "export VECLIB_MAXIMUM_THREADS=1",
            "export OMP_NUM_THREADS=1",
            "export OMP_DYNAMIC=FALSE",
        ]

        #if rayon_threads is not None:
        #    bash_lines.append(f"export RAYON_NUM_THREADS={rayon_threads}")
        #else:
            # Explicitly clear RAYON_NUM_THREADS to ensure all cores are used
        bash_lines.append("unset RAYON_NUM_THREADS")

        bash_lines.append(
            f"cargo run --release '{data_type_args_json}' '{scenario_args_json}' '{rsrs_args_json}' '{output_args_json}'"
        )

        with open(filename, "w") as f:
            f.write("\n".join(bash_lines) + "\n")

        subprocess.run(["chmod", "+x", filename], check=True)


    def generate_bash_script_multi(
        self,
        filename: str = "run_test.sh",
        json_num_threads: list[int] | None = None,
        refinement_levels: list[int] | None = None,
    ):
        import json
        import subprocess
        import shlex

        # Base JSON blocks
        data_type_args = self.data_type_args()
        scenario_args_base = self.scenario_args()
        output_args = self.output_args()

        def esc(x):
            """Shell-escape a JSON string safely using shlex."""
            return shlex.quote(json.dumps(x))

        bash_lines = [
            "#!/bin/bash",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export MKL_DOMAIN_NUM_THREADS=1",
            "export MKL_DYNAMIC=FALSE",
            "export GOTO_NUM_THREADS=1",
            "export BLIS_NUM_THREADS=1",
            "export VECLIB_MAXIMUM_THREADS=1",
            "export OMP_NUM_THREADS=1",
            "export OMP_DYNAMIC=FALSE",
            "unset RAYON_NUM_THREADS",
            "",
        ]

        if refinement_levels is None:
            refinement_levels = [
                scenario_args_base["dim_args"][0]["RefinementLevelAndDepth"][0]
            ]

        if json_num_threads is None:
            json_num_threads = [self.rsrs_args().get("num_threads", 1)]

        bash_lines.append("echo \"Running RSRS benchmarks\"")
        bash_lines.append("")

        for rl in refinement_levels:
            bash_lines.append(f"echo \"RefinementLevel = {rl}\"")

            # Inject refinement level
            scenario_args = dict(scenario_args_base)
            scenario_args["dim_args"] = [{"RefinementLevelAndDepth": [rl, 2]}]

            # Escape static JSON
            data_json = esc(data_type_args)
            scenario_json = esc(scenario_args)
            output_json = esc(output_args)

            for nt in json_num_threads:
                rsrs_args = self.rsrs_args()
                rsrs_args["num_threads"] = nt
                rsrs_json = esc(rsrs_args)

                bash_lines.append(f"  echo \"  num_threads = {nt}\"")
                bash_lines.append(
                    f"  cargo run --release {data_json} {scenario_json} {rsrs_json} {output_json}"
                )

            bash_lines.append("")

        with open(filename, "w") as f:
            f.write("\n".join(bash_lines) + "\n")

        subprocess.run(["chmod", "+x", filename], check=True)
        
    def generate_folder_name(self) -> str:
        geom = camel_to_snake(self.geometry_types[self.geometry])
        op = self.structured_operator_types[self.operator_type_index]
        dim_key = self.dim_arg_types[self.dim_arg_type_index]

        if dim_key == "RefinementLevelAndDepth":
            if self.ref_level > 1:
                return f"{geom}_{op}_ref_level_{self.ref_level}_depth_{self.depth}_od_{self.max_tree_depth}_num_threads_{self.num_threads}"
            else:
                return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}_num_threads_{self.num_threads}"
        elif dim_key == "Meshwidth":
            return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}_num_threads_{self.num_threads}"
        elif dim_key == "Kappa":
            return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}_{self.kappa:.2f}_num_threads_{self.num_threads}"
        elif dim_key == "KappaAndMeshwidth":
            return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}_{self.kappa:.2f}_num_threads_{self.num_threads}"
        else:
            raise ValueError("Invalid dim_arg_type")

    def generate_sub_folder_name(self) -> str:
        args = self.rsrs_args()

        if self.rrqr_keys[self.rrqr_index] == "SRRQR":
            rrqr_pred = f"srrqr_" + sci_no_padding2(self.f)
        else:
            rrqr_pred = "rrqr"

        if self.symmetric is None:
            sym = "symmetry"
        else:
            sym = "symmetric"

        if self.op_shift == 0:
            return (
                f"rsrs_null_{args['null_method']}"
                f"_toln_{args['tol_null']:.0e}"
                f"_os_{args['oversampling']}"
                f"_osdiag_{args['oversampling_diag_blocks']}"
                f"_initsam_{args['initial_num_samples']}"
                f"_mrnk_{args['min_rank']}"
                f"_mlvl_{args['min_level']}"
                f"_herm_{camel_to_snake(str(args[sym]))}"
                f"_rpick_{args['rank_picking']}"
                f"_next_{args['near_block_extraction_method']}"
                f"_tolextn_{args['tol_ext_near']:.0e}"
                f"_db_ext_{args['diag_block_extraction_method']}"
                f"_tol_lstsq_{args['tol_diag_ext']:.0e}"
                f"_{rrqr_pred}"
            )
        else:
            return (
                f"rsrs_null_{args['null_method']}"
                f"_toln_{args['tol_null']:.0e}"
                f"_os_{args['oversampling']}"
                f"_osdiag_{args['oversampling_diag_blocks']}"
                f"_initsam_{args['initial_num_samples']}"
                f"_stabilised_{sci_no_padding(self.op_shift)}"
                f"_mrnk_{args['min_rank']}"
                f"_mlvl_{args['min_level']}"
                f"_herm_{camel_to_snake(str(args[sym]))}"
                f"_rpick_{args['rank_picking']}"
                f"_next_{args['near_block_extraction_method']}"
                f"_tolextn_{args['tol_ext_near']:.0e}"
                f"_db_ext_{args['diag_block_extraction_method']}"
                f"_tol_lstsq_{args['tol_diag_ext']:.0e}"
                f"_{rrqr_pred}"
            )

    def load_all_stats(self, kind="error"):
        return _config_results.load_all_stats(self, kind=kind)

    def _results_base_path(self):
        return _config_results.results_base_path(self)

    def _select_error_stat(self, tol=None):
        return _config_results.select_error_stat(self, tol=tol)

    @staticmethod
    def _decode_legacy_vectors(vectors):
        return _config_results.decode_legacy_vectors(vectors)

    def _load_solution_group(self, stat, group_name):
        return _config_results.load_solution_group(self, stat, group_name)

    @classmethod
    def generate_table_a(
        cls,
        series_specs,
        caption="Accuracy and solver effectiveness.",
        label="tab:accuracy",
        table_only=False,
        out_path=None,
    ):
        from results_tables import generate_table_a

        return generate_table_a(
            series_specs,
            caption=caption,
            label=label,
            table_only=table_only,
            out_path=out_path,
        )

    @classmethod
    def generate_table_b(
        cls,
        series_specs,
        caption="Preconditioner performance.",
        label="tab:preconditioner",
        table_only=False,
        out_path=None,
    ):
        from results_tables import generate_table_b

        return generate_table_b(
            series_specs,
            caption=caption,
            label=label,
            table_only=table_only,
            out_path=out_path,
        )

    
    def plot_errors_vs_tolerance(self, metric_index=1, plot=True, logx=True, logy=True, save_plot=False):
        """
        Plot a specified error metric vs tolerance.

        Parameters
        ----------
        metric_index : int
            The index of the error metric to plot on the y-axis. Must be one of:
            1 - 'norm_2_error'
            2 - 'norm_2_error_inv'
            3 - 'cond_rsrs_estimate'
            4 - 'tot_num_samples'
            5 - 'residual_size'
        logx : bool
            If True, use logarithmic scale for the x-axis (tolerance).
        logy : bool
            If True, use logarithmic scale for the y-axis (metric).

        Raises
        ------
        ValueError
            If `metric_index` is not in the range 1 to 5.
        """
        metrics = [
            "norm_2_error",
            "norm_2_error_inv",
            "cond_rsrs_estimate",
            "tot_num_samples",
            "residual_size"
        ]

        pretty_names = {
            "norm_2_error": r"$\|A - A_{\mathrm{app}}\|_2 / \|A\|_2$",
            "norm_2_error_inv": r"$\|A_{\mathrm{app}}^{-1} A - I\|_2$",
            "cond_rsrs_estimate": r"$\mathrm{cond}(A_{\mathrm{rsrs}})$ estimate",
            "tot_num_samples": "Total Number of Samples",
            "residual_size": "Residual Size"
        }

        if not (1 <= metric_index <= len(metrics)):
            raise ValueError(f"Invalid metric_index '{metric_index}'. Must be between 1 and {len(metrics)}.")

        metric = metrics[metric_index - 1]
        ylabel = pretty_names.get(metric, metric.replace("_", " ").title())
        title = f"{ylabel} vs Tolerance"

        all_stats = self.load_all_stats(kind="error")

        # Sort by tolerance for clean plotting
        all_stats.sort(key=lambda d: float(d["tolerance"]))

        # Extract x and y data
        tolerances = [float(d["tolerance"]) for d in all_stats]
        y_values = [d.get(metric, float("nan")) for d in all_stats]
        print(tolerances, y_values)
        if plot:
            # Plot
            plt.figure(figsize=(6, 4))
            plt.plot(tolerances, y_values, marker="o")

            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")

            plt.xlabel("Tolerance")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, which="both" if (logx or logy) else "major", ls='--', alpha=0.5)
            plt.tight_layout()
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                path = Path(os.getcwd()) / "results" / folder / subfolder / f"{metric}.png"
                plt.savefig(path, dpi=300)
            plt.show()
        else:
            return [tolerances, y_values]

    def plot_residual_convergence(self, log_scale=True, plot=True, save_plot=False):
        """
        Plot length of residual vectors vs tolerance for all preconditioned methods (excluding no_prec).

        Parameters:
        -----------
        log_scale : bool
            If True, use log scale on both axes; if False, use linear scale.
        """
        all_stats = self.load_all_stats()
        if not all_stats:
            print("No data found.")
            return

        # Sort by tolerance
        all_stats.sort(key=lambda d: float(d["tolerance"]))

        residual_lengths = {}

        for stat in all_stats:
            tol = float(stat["tolerance"])
            iterations = stat.get("iterations", {})
            for key, residuals in iterations.items():
                if key == "no_prec" or not isinstance(residuals, list):
                    continue

                residual_lengths.setdefault(key, []).append((tol, len(residuals)))

        if plot:
            plt.figure(figsize=(8, 5))

            for key, values in residual_lengths.items():
                values.sort(key=lambda x: x[0])
                tols, lengths = zip(*values)
                plt.plot(tols, lengths, marker='o', label=key)

            plt.xlabel("Tolerance")
            plt.ylabel("Iterations")
            plt.title("Iterations vs Tolerance")

            if log_scale:
                plt.xscale("log")
                plt.yscale("log")
            else:
                plt.xscale("log")
                plt.yscale("linear")

            plt.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                path = Path(os.getcwd()) / "results" / folder / subfolder / "gmres_iterations.png"
                plt.savefig(path, dpi=300)
            plt.show()
        else:
            return residual_lengths

    def plot_total_elapsed_time_vs_tolerance(self, logx=True, logy=True, plot=True, save_plot=False):
        """
        Plot total elapsed time without sampling vs tolerance (in seconds).

        Parameters
        ----------
        logx : bool, optional
            If True, use log scale on the x-axis (tolerance).
        logy : bool, optional
            If True, use log scale on the y-axis (time).
        """
        time_stats = self.load_all_stats(kind="time")

        if not time_stats:
            print("No time statistics found.")
            return

        # Sort by tolerance
        time_stats.sort(key=lambda d: d["tolerance"])

        # Extract and convert values
        tolerances = [d["tolerance"] for d in time_stats]
        elapsed_times_sec = [d.get("total_elapsed_time_wo_sampling", float("nan")) / 1000.0 for d in time_stats]

        if plot:
            # Plot
            plt.figure(figsize=(6, 4))
            plt.plot(tolerances, elapsed_times_sec, marker="o")

            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")

            plt.xlabel("Tolerance")
            plt.ylabel("RSRS Time (s)")
            plt.title("RSRS Elapsed Time vs Tolerance")
            plt.grid(True, which="both" if (logx or logy) else "major", ls="--", alpha=0.5)
            plt.tight_layout()
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                path = Path(os.getcwd()) / "results" / folder / subfolder / "elapsed_time_vs_tolerance.png"
                plt.savefig(path, dpi=300)
            plt.show()
        else:
            return [tolerances, elapsed_times_sec]

    def get_degrees_of_freedom(self):
        """
        Return the number of degrees of freedom (dim) from the first available time_stats file.

        Returns
        -------
        int or None
            The number of degrees of freedom, or None if not found.
        """
        time_stats = self.load_all_stats(kind="time")
        if not time_stats:
            print("No time statistics found.")
            return None

        return time_stats[0].get("dim", None)

    def plot_time_breakdown_piecharts(self, max_charts=None, save_plot=False):
        """
        Plot a pie chart of time breakdown for each tolerance.

        Parameters
        ----------
        max_charts : int or None
            Maximum number of pie charts to display. If None, shows all.
        """

        time_stats = self.load_all_stats(kind="time")
        if not time_stats:
            print("No time statistics found.")
            return

        # Sort by tolerance
        time_stats.sort(key=lambda d: float(d["tolerance"]))

        if max_charts is not None:
            time_stats = time_stats[:max_charts]

        for stat in time_stats:
            tol = stat["tolerance"]
            label = f"tol = {tol:g}"

            # Extract times
            tot_id_time = stat.get("tot_id_time", 0)
            tot_lu_time = stat.get("tot_lu_time", 0)
            extraction_time = stat.get("extraction_time", 0)

            update_times = stat.get("update_times", [])
            update_id_time = sum(entry.get("id", 0) for entry in update_times)
            update_lu_time = sum(entry.get("lu", 0) for entry in update_times)

            other = (
                stat.get("index_calculation", 0)
                + stat.get("sorting_near_field", 0)
                + stat.get("residual_calculation", 0)
            )

            # Convert all times to seconds
            data = {
                "tot_id_time": tot_id_time / 1000,
                "update_id_time": update_id_time / 1000,
                "tot_lu_time": tot_lu_time / 1000,
                "update_lu_time": update_lu_time / 1000,
                "extraction_time": extraction_time / 1000,
                "other": other / 1000,
            }

            # Remove zero entries to simplify the pie chart
            data = {k: v for k, v in data.items() if v > 0}

            plt.figure(figsize=(6, 6))
            plt.pie(
                data.values(),
                labels=[f"{k}\n{v:.2f} s" for k, v in data.items()],
                autopct="%1.1f%%",
                startangle=140
            )
            plt.title(f"Time Breakdown for {label}")
            plt.tight_layout()
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                path = Path(os.getcwd()) / "results" / folder / subfolder / f"time_breakdown_{tol:.2e}.png"
                plt.savefig(path, dpi=300)
            plt.show()

    def plot_factor_metrics(self, metric="cond", save_plot=False):
        """
        Plot condition numbers or norms for ID, LU, and Diagonal factors.

        Parameters
        ----------
        metric : str
            "cond" for condition numbers (index 0 in data), "norm" for norms (index 1).
        save_plot : bool
            Whether to save the generated plots.
        """
        assert metric in ("cond", "norm"), "metric must be 'cond' or 'norm'"
        metric_idx = 0 if metric == "cond" else 1

        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            fig, axs = plt.subplots(1, 4, figsize=(20, 4))
            fig.suptitle(f"{'Condition Numbers' if metric=='cond' else 'Norms'} for Tolerance = {tol}")

            # --- ID Rectangular Factors ---
            id_levels = data.get("id", [])
            id_vals = []
            for level in id_levels:
                clean = []
                for entry in level:
                    if entry and entry[0] and entry[0][0] and entry[0][0][metric_idx] is not None:
                        clean.append(entry[0][0][metric_idx])  # take scalar, not list
                id_vals.append(clean if clean else [])

            if any(len(v) > 0 for v in id_vals):
                axs[0].boxplot(id_vals, vert=True)
                axs[0].set_title("ID Rectangular Factor")
                axs[0].set_xlabel("Level")
                axs[0].set_ylabel(metric)
                axs[0].set_xticks(range(1, len(id_vals) + 1))
                axs[0].set_xticklabels([str(i) for i in range(len(id_vals))])
            else:
                axs[0].set_visible(False)

            # --- LU Rectangular Factors ---
            lu_levels = data.get("lu", [])
            lu_rect_vals = []
            for level in lu_levels:
                clean = []
                for entry in level:
                    if entry and entry[0] and entry[0][0] and entry[0][0][metric_idx] is not None:
                        clean.append(entry[0][0][metric_idx])
                lu_rect_vals.append(clean if clean else [])

            if any(len(v) > 0 for v in lu_rect_vals):
                axs[1].boxplot(lu_rect_vals, vert=True)
                axs[1].set_title("LU Rectangular Factor")
                axs[1].set_xlabel("Level")
                axs[1].set_ylabel(metric)
                axs[1].set_xticks(range(1, len(lu_rect_vals) + 1))
                axs[1].set_xticklabels([str(i) for i in range(len(lu_rect_vals))])
            else:
                axs[1].set_visible(False)

            # --- LU L and U Factors (Side-by-side) ---
            lu_l_vals, lu_u_vals = [], []
            for level in lu_levels:
                l_clean, u_clean = [], []
                for entry in level:
                    if entry and entry[0] and entry[0][1]:
                        if entry[0][1][0] and entry[0][1][0][metric_idx] is not None:
                            l_clean.append(entry[0][1][0][metric_idx])
                        if entry[0][1][1] and entry[0][1][1][metric_idx] is not None:
                            u_clean.append(entry[0][1][1][metric_idx])
                lu_l_vals.append(l_clean if l_clean else [])
                lu_u_vals.append(u_clean if u_clean else [])

            if any(len(v) > 0 for v in lu_l_vals) and any(len(v) > 0 for v in lu_u_vals):
                positions = np.arange(1, len(lu_l_vals) + 1)
                axs[2].boxplot(lu_l_vals, positions=positions - 0.15, widths=0.3, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'))
                axs[2].boxplot(lu_u_vals, positions=positions + 0.15, widths=0.3, patch_artist=True,
                            boxprops=dict(facecolor='lightgreen'))
                axs[2].set_title("LU L and U Factors")
                axs[2].set_xlabel("Level")
                axs[2].set_ylabel(metric)
                axs[2].set_xticks(positions)
                axs[2].set_xticklabels([str(i) for i in range(len(lu_l_vals))])
                axs[2].legend(["L", "U"])
            else:
                axs[2].set_visible(False)

            # --- Diagonal Factors (labeled L/U) ---
            dfactors = data.get("dfactors", [])
            diag_L, diag_U = [], []
            for entry in dfactors:
                if entry and entry[0] and entry[0][1]:
                    if entry[0][1][0] and entry[0][1][0][metric_idx] is not None:
                        diag_L.append(entry[0][1][0][metric_idx])
                    if entry[0][1][1] and entry[0][1][1][metric_idx] is not None:
                        diag_U.append(entry[0][1][1][metric_idx])

            if diag_L and diag_U:
                axs[3].boxplot([diag_L, diag_U], vert=True)
                axs[3].set_title("Diagonal L/U Factors")
                axs[3].set_xlabel("Factor")
                axs[3].set_ylabel(metric)
                axs[3].set_xticklabels(["L factor", "U factor"])
            else:
                axs[3].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                filename = f"{'condition_numbers' if metric=='cond' else 'norms'}_{tol:.2e}.png"
                path = Path(os.getcwd()) / "results" / folder / subfolder / filename
                plt.savefig(path, dpi=300)
            plt.show()

    def plot_lu_factors_app_cond(self, save_plot=False):
        """
        Plot (1 + res)^2 where res = norm_rect * norm_L * norm_U
        for each LU level.

        Parameters
        ----------
        save_plot : bool
            Whether to save the generated plots.
        """
        metric_idx = 1  # 1 corresponds to "norm"

        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            lu_levels = data.get("lu", [])
            prod_vals = []

            for level in lu_levels:
                level_vals = []
                for entry in level:
                    if (entry and entry[0] and entry[0][0] and entry[0][1] and
                        entry[0][1][0] and entry[0][1][1]):
                        norm_rect = entry[0][0][metric_idx]
                        norm_L = entry[0][1][0][metric_idx]
                        norm_U = entry[0][1][1][metric_idx]

                        if None not in (norm_rect, norm_L, norm_U):
                            res = norm_rect * norm_L * norm_U
                            level_vals.append((1 + res) ** 2)
                prod_vals.append(level_vals if level_vals else [])

            # --- Plot ---
            if any(len(v) > 0 for v in prod_vals):
                plt.figure(figsize=(8, 4))
                plt.boxplot(prod_vals, vert=True)
                plt.title(f"Approximate CN for Tolerance = {tol}")
                plt.xlabel("Level")
                plt.ylabel("App Cond")
                plt.xticks(range(1, len(prod_vals) + 1),
                        [str(i) for i in range(len(prod_vals))])

                plt.tight_layout()
                if save_plot:
                    folder = self.generate_folder_name()
                    subfolder = self.generate_sub_folder_name()
                    filename = f"lu_factors_app_cond_{tol:.2e}.png"
                    path = Path(os.getcwd()) / "results" / folder / subfolder / filename
                    plt.savefig(path, dpi=300)
                plt.show()

    def plot_id_factors_app_cond(self, save_plot=False):
        """
        Plot (1 + res)^2 where res = norm_rect * norm_L * norm_U
        for each LU level.

        Parameters
        ----------
        save_plot : bool
            Whether to save the generated plots.
        """

        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            id_levels = data.get("id", [])
            prod_vals = []

            for level in id_levels:
                level_vals = []
                for entry in level:
                    if (entry and entry[0] and entry[0][0] and entry[0][0][1]):
                        norm_rect = entry[0][0][1]
                        level_vals.append((1 + norm_rect) ** 2)
                prod_vals.append(level_vals if level_vals else [])

            # --- Plot ---
            if any(len(v) > 0 for v in prod_vals):
                plt.figure(figsize=(8, 4))
                plt.boxplot(prod_vals, vert=True)
                plt.title(f"Approximate CN for Tolerance = {tol}")
                plt.xlabel("Level")
                plt.ylabel("App Cond")
                plt.xticks(range(1, len(prod_vals) + 1),
                        [str(i) for i in range(len(prod_vals))])

                plt.tight_layout()
                if save_plot:
                    folder = self.generate_folder_name()
                    subfolder = self.generate_sub_folder_name()
                    filename = f"id_factors_condition_number_{tol:.2e}.png"
                    path = Path(os.getcwd()) / "results" / folder / subfolder / filename
                    plt.savefig(path, dpi=300)
                plt.show()

    def plot_d_factor_app_cond(self, save_plot=False):
        """
        Plot (1 + res)^2 where res = norm_rect * norm_L * norm_U
        for each LU level.

        Parameters
        ----------
        save_plot : bool
            Whether to save the generated plots.
        """

        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            dfactors = data.get("dfactors", [])
            prod_vals = []
            for entry in dfactors:
                if entry and entry[0] and entry[0][1]:
                    res = 1
                    if entry[0][1][0] and entry[0][1][0][1] is not None:
                        res = entry[0][1][0][1]
                    if entry[0][1][1] and entry[0][1][1][1] is not None:
                        res*=entry[0][1][1][1]
                    prod_vals.append((1+ res)**2)

            # --- Plot ---
            plt.figure(figsize=(8, 4))
            plt.boxplot(prod_vals, vert=True)
            plt.title(f"Approximate CN for Tolerance = {tol}")
            plt.xlabel("Level")
            plt.ylabel("App Cond")
            #plt.xticks(range(1, len(prod_vals) + 1),
            #        [str(i) for i in range(len(prod_vals))])

            plt.tight_layout()
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                filename = f"dfactors_condition_number_{tol:.2e}.png"
                path = Path(os.getcwd()) / "results" / folder / subfolder / filename
                plt.savefig(path, dpi=300)
            plt.show()

    def plot_condition_numbers_scatter(self, save_plot=False):
        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            fig, axs = plt.subplots(1, 4, figsize=(24, 4))
            fig.suptitle(f"Condition Numbers for Tolerance = {tol}")

            # --- ID Factors ---
            id_levels = data.get("id", [])
            id_vals = [
                [vec[0][0] for vec in level if vec and vec[0] and vec[0][0] is not None]
                for level in id_levels
            ]
            if any(id_vals):
                for i, level_vals in enumerate(id_vals):
                    sorted_vals = sorted(level_vals)
                    x = np.arange(len(sorted_vals)) + np.random.normal(0, 0.1, size=len(sorted_vals))
                    axs[0].scatter(x, sorted_vals, alpha=0.7, label=f"Level {i}", s=10)
                axs[0].set_title("ID Factor Condition Numbers")
                axs[0].set_xlabel("Sorted element index")
                axs[0].set_ylabel("cond")
                axs[0].legend()
            else:
                axs[0].set_visible(False)

            # --- LU L and U Factors ---
            lu_levels = data.get("lu", [])
            for i, level in enumerate(lu_levels):
                l_vals = [vec[0][1][0] for vec in level if vec and vec[0] and vec[0][1]]
                u_vals = [vec[0][1][1] for vec in level if vec and vec[0] and vec[0][1]]

                if l_vals:
                    x_l = np.arange(len(l_vals)) + np.random.normal(0, 0.1, size=len(l_vals))
                    axs[1].scatter(x_l, sorted(l_vals), alpha=0.7, label=f"L Level {i}", s=10, color='blue')
                if u_vals:
                    x_u = np.arange(len(u_vals)) + np.random.normal(0, 0.1, size=len(u_vals))
                    axs[1].scatter(x_u, sorted(u_vals), alpha=0.7, label=f"U Level {i}", s=10, color='green')

            axs[1].set_title("LU L and U Factors")
            axs[1].set_xlabel("Sorted element index")
            axs[1].set_ylabel("cond")
            axs[1].legend()

            # --- LU Rectangular Factors ---
            for i, level in enumerate(lu_levels):
                rect_vals = [vec[0][0] for vec in level if vec and vec[0] and vec[0][0] is not None]
                if rect_vals:
                    x_r = np.arange(len(rect_vals)) + np.random.normal(0, 0.1, size=len(rect_vals))
                    axs[2].scatter(x_r, sorted(rect_vals), alpha=0.7, label=f"Level {i}", s=10, color='red')

            axs[2].set_title("LU Rectangular Factor Condition Numbers")
            axs[2].set_xlabel("Sorted element index")
            axs[2].set_ylabel("cond")
            axs[2].legend()

            # --- Diagonal Factors ---
            dfactors = data.get("dfactors", [])
            diag_L = [vec[0][1][0] for vec in dfactors if vec and vec[0] and vec[0][1]]
            diag_U = [vec[0][1][1] for vec in dfactors if vec and vec[0] and vec[0][1]]

            if diag_L and diag_U:
                x1 = np.arange(len(diag_L)) + np.random.normal(0, 0.1, size=len(diag_L))
                x2 = np.arange(len(diag_U)) + np.random.normal(0, 0.1, size=len(diag_U))
                axs[3].scatter(x1, sorted(diag_L), alpha=0.7, label="Diagonal L Factor", s=10, color='blue')
                axs[3].scatter(x2, sorted(diag_U), alpha=0.7, label="Diagonal U Factor", s=10, color='green')
                axs[3].set_title("Diagonal L and U Factor Condition Numbers")
                axs[3].set_xlabel("Sorted element index")
                axs[3].set_ylabel("cond")
                axs[3].legend()
            else:
                axs[3].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                path = Path(os.getcwd()) / "results" / folder / subfolder / f"condition_numbers_scatter_{tol:.2e}.png"
                plt.savefig(path, dpi=300)
            plt.show()

    def plot_condition_number_summaries(self, save_plot=False):
        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        def is_valid_number(x):
            return x is not None and isinstance(x, (int, float)) and not np.isnan(x)

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            fig, axs = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(f"Condition Number Summary Stats for Tolerance = {tol}")

            def plot_summary(ax, levels_data, title):
                filtered = []
                for i, level in enumerate(levels_data):
                    clean_level = [v for v in level if is_valid_number(v)]
                    if clean_level:
                        filtered.append((i, clean_level))
                if not filtered:
                    ax.set_visible(False)
                    return
                indices, filtered_levels = zip(*filtered)

                means = [np.mean(lvl) for lvl in filtered_levels]
                medians = [np.median(lvl) for lvl in filtered_levels]
                errors = [np.std(lvl) for lvl in filtered_levels]

                x = np.arange(len(filtered_levels))
                width = 0.35

                ax.bar(x - width/2, means, width, label='Mean', yerr=errors, capsize=5)
                ax.bar(x + width/2, medians, width, label='Median')

                ax.set_title(title)
                ax.set_xlabel("Level")
                ax.set_ylabel("Condition Number")
                ax.set_xticks(x)
                ax.set_xticklabels([str(i) for i in indices])
                ax.legend()

                ymax = max(m + e for m, e in zip(means, errors))
                ymin = min(min(means), min(medians))
                ax.set_ylim(bottom=max(0.1 * ymin, 1e-12), top=1.1 * ymax)

            # --- ID Factor ---
            id_levels = data.get("id", [])
            id_vals = [
                [vec[0][0] for vec in level if vec and vec[0] and is_valid_number(vec[0][0])]
                for level in id_levels
            ]
            plot_summary(axs[0], id_vals, "ID Factor")

            # --- LU Rectangular Factor ---
            lu_levels = data.get("lu", [])
            lu_rect_vals = [
                [vec[0][0] for vec in level if vec and vec[0] and is_valid_number(vec[0][0])]
                for level in lu_levels
            ]
            plot_summary(axs[1], lu_rect_vals, "LU Rectangular Factor")

            # --- LU L Factor ---
            lu_L_vals = [
                [vec[0][1][0] for vec in level if vec and vec[0] and vec[0][1] and is_valid_number(vec[0][1][0])]
                for level in lu_levels
            ]
            plot_summary(axs[2], lu_L_vals, "LU L Factor")

            # --- LU U Factor ---
            lu_U_vals = [
                [vec[0][1][1] for vec in level if vec and vec[0] and vec[0][1] and is_valid_number(vec[0][1][1])]
                for level in lu_levels
            ]
            plot_summary(axs[3], lu_U_vals, "LU U Factor")

            # --- Diagonal Factors ---
            dfactors = data.get("dfactors", [])
            cond1_vals = [vec[0][1][0] for vec in dfactors if vec and vec[0] and vec[0][1] and is_valid_number(vec[0][1][0])]
            plot_summary(axs[4], [cond1_vals], "Diagonal Factor 1")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plot:
                folder = self.generate_folder_name()
                subfolder = self.generate_sub_folder_name()
                path = Path(os.getcwd()) / "results" / folder / subfolder / f"condition_number_summary_{tol:.2e}.png"
                plt.savefig(path, dpi=300)
            plt.show()

    def plot_max_entry_id(self, save_plot=False):
        """
        Plot condition numbers or norms for ID, LU, and Diagonal factors.

        Parameters
        ----------
        metric : str
            "cond" for condition numbers (index 0 in data), "norm" for norms (index 1).
        save_plot : bool
            Whether to save the generated plots.
        """

        stats = self.load_all_stats("condition_number")
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]

            # --- ID Rectangular Factors ---
            id_levels = data.get("id", [])
            id_vals = []
            for level in id_levels:
                clean = []
                for entry in level:
                    if entry and entry[1] and entry[1][0] and entry[1][0][1] is not None:
                        clean.append(entry[1][0][1])  # take scalar, not list
                id_vals.append(clean if clean else [])

            # --- Plot ---
            if any(len(v) > 0 for v in id_vals):
                plt.figure(figsize=(8, 4))
                plt.boxplot(id_vals, vert=True)
                plt.title(f"Max entry for Tolerance = {tol}")
                plt.xlabel("Level")
                plt.ylabel("Max inv(R11)R12_ij")
                plt.xticks(range(1, len(id_vals) + 1),
                        [str(i) for i in range(len(id_vals))])

                plt.tight_layout()
                if save_plot:
                    folder = self.generate_folder_name()
                    subfolder = self.generate_sub_folder_name()
                    filename = f"id_max_entry_{tol:.2e}.png"
                    path = Path(os.getcwd()) / "results" / folder / subfolder / filename
                    plt.savefig(path, dpi=300)
                plt.show()

    def relocate_grid(self):
        """
        Relocate the current grid to a different directory.
        """
        from pathlib import Path

        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        filename = f"grid.msh"

        src = Path(os.getcwd()) / "results/current_grid.msh"
        dst = Path(os.getcwd()) / "results" / folder / subfolder / filename

        dst.parent.mkdir(parents=True, exist_ok=True)  # create target dirs if needed
        src.rename(dst)

    def load_sols(self, tol=1e-2):
        if tol == 0.0:
            stat = self._select_error_stat()
            return self._load_solution_group(stat, "sols_no_prec")
        else:
            stat = self._select_error_stat(tol)
            return self._load_solution_group(stat, "sols_prec")

    def get_far_field(self, tol=1e-2, n_grid_points=150, plane=0, lims = [-1,1,-1,1], c=0.0):
        import bempp_cl.api
        import matplotlib
        matplotlib.rcParams["figure.figsize"] = (5.0, 4.0)
        sols = self.load_sols(tol)
        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        filename = f"grid.msh"
        grid_path = Path(os.getcwd()) / "results" / folder / subfolder / filename
        if not grid_path.exists():
            self.relocate_grid()
        grid = bempp_cl.api.import_grid(str(grid_path))
        plot_grid = np.mgrid[lims[0] : lims[1] : n_grid_points * 1j, lims[2] : lims[3] : n_grid_points * 1j]
        if plane == 0:
            # xy plane
            points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size) + c))
        elif plane ==1:
            # xz plane
            points = np.vstack((plot_grid[0].ravel(), np.zeros(plot_grid[0].size) + c, plot_grid[1].ravel()))
        else:
            # yz plane
            points = np.vstack((np.zeros(plot_grid[0].size) + c,plot_grid[0].ravel(), plot_grid[1].ravel()))

        if self.structured_operator_types[self.operator_type_index] == "BemppClLaplaceSingleLayer":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.laplace.single_layer(space, points)

        elif self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayer" or self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzCombined":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(space, points, self.kappa)

        elif self.structured_operator_types[self.operator_type_index] == "BemppClMaxwellEfie":
            space = bempp_cl.api.function_space(grid, "RWG", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.maxwell.electric_field(space, points, self.kappa)

        elif self.structured_operator_types[self.operator_type_index] == "BemppClBurtonMiller":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(space, points, self.kappa)
        
        for i, sol in enumerate(sols):
            sol_fun = bempp_cl.api.GridFunction(space, projections=sol, dual_space=dual_space)
            far_field = slp_pot * sol_fun
            if self.structured_operator_types[self.operator_type_index] == "BemppClMaxwellEfie":
                scattered_field = np.real(np.sum(far_field * far_field.conj(), axis=0))
            elif self.structured_operator_types[self.operator_type_index] == "BemppClBurtonMiller" or self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzCombined":
                scattered_field = np.real(far_field)
            else:
                scattered_field = far_field
            scattered_field = scattered_field.reshape((n_grid_points, n_grid_points))
            plt.imshow(np.abs(scattered_field.T))
            plt.title("Computed solution")
            plt.colorbar()
            sol_path = Path(os.getcwd()) / "results" / folder / subfolder / f"scattered_field_{plane}_{i}_{c}.png"
            plt.savefig(sol_path)
            plt.close()

    def get_rcs(self, tol=1e-2, number_of_angles=400, plane=0, polar=True):
        import bempp_cl.api
        plt.rcParams["figure.figsize"] = (10, 8)
        sols = self.load_sols(tol)
        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        filename = f"grid.msh"
        grid_path = Path(os.getcwd()) / "results" / folder / subfolder / filename
        if not grid_path.exists():
            self.relocate_grid()
        grid = bempp_cl.api.import_grid(str(grid_path))
        
        angles = 2*np.pi * np.linspace(0, 1, number_of_angles)
        if plane == 0:
            # xy plane
            unit_points = np.array([-np.cos(angles), -np.sin(angles), np.zeros(number_of_angles)])
        elif plane ==1:
            # xz plane
            unit_points = np.array([-np.cos(angles), np.zeros(number_of_angles), -np.sin(angles)])
        else:
            # yz plane
            unit_points = np.array([np.zeros(number_of_angles), -np.cos(angles), -np.sin(angles)])
        if self.structured_operator_types[self.operator_type_index] == "BemppClLaplaceSingleLayer":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.laplace.single_layer(space, unit_points)
        elif self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayer" or self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzCombined":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(space, unit_points, self.kappa)
        elif self.structured_operator_types[self.operator_type_index] == "BemppClBurtonMiller":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(space, unit_points, self.kappa)
        elif self.structured_operator_types[self.operator_type_index] == "BemppClMaxwellEfie":
            space = bempp_cl.api.function_space(grid, "RWG", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.maxwell.electric_field(space, unit_points, self.kappa)
        far_field = np.zeros((3, number_of_angles), dtype="complex128")
        for sol in sols:
            sol_fun = bempp_cl.api.GridFunction(space, projections=sol, dual_space=dual_space)
            scattered_field = slp_pot * sol_fun
            far_field += scattered_field
        cross_section = 10 * np.log10(4 * np.pi * np.sum(np.abs(far_field) ** 2, axis=0))

        ax = plt.subplot(111, polar=polar)

        if polar:
            angles_full = np.concatenate([angles, [angles[0]]])
            cross_section_full = np.concatenate([cross_section, [cross_section[0]]])
        else:
            angles_full = angles
            cross_section_full = cross_section
        ax.plot(angles_full, cross_section_full, lw=2)
        if polar:
            ax.set_theta_zero_location("N")  # 0° at the top
            ax.set_theta_direction(-1)       # clockwise
            ax.set_rlabel_position(225)      # move radial labels away from main lobe
        ax.set_title("Scattering Cross Section [dB]", va='bottom')

        sol_path = Path(os.getcwd()) / "results" / folder / subfolder / f"rcs_{plane}.png"
        plt.savefig(sol_path)

    def plot_field_slices_3d(
        self,
        tol=1e-2,
        plane="xz",                 # "xy", "xz", "yz"
        plane_value=None,           # z for "xy", y for "xz", x for "yz"
        plane_points=200,
        plane_extent_rel=0.0,
        # view
        elev=25,
        azim=-60,
        # appearance
        plane_alpha=1.0,
        cmap_name="viridis",
        transparent_bg=False,
        hide_axes=True,
        # framing
        box_limits=None,            # tuple (bmin, bmax)
        figsize=(10, 7),
        # save
        save_plot=True,
        out_name_prefix="field_slice_3d",
        dpi=300,
        show_colorbar=True,
        # behavior
        plot_sum=True,
        shared_color_scale=True,
        # incident handling (now consistent with your RHS generation)
        include_incident=False,     # default: scattered-only
        rhs_kind="auto",            # "auto" | "plane_wave" | "monopole" | "none"
        rhs_count=None,             # number of incident fields; default = len(sols)
        rhs_index=None,             # if set, ONLY add incident for that index (else per-solution)
        rhs_padding=1.2,            # used for monopole sources (Laplace-like)
        polarization=(-1.0, 0.0, 0.0),  # Maxwell plane-wave polarization
        # contrast controls
        contrast="linear",          # "linear" | "power" | "log"
        gamma=0.7,                  # used when contrast="power"
        clip_percentiles=(1.0, 99.5),  # used for contrast modes; None disables clipping
        # layout: make the field bigger when colorbar exists
        reserve_colorbar_space=True,
        colorbar_width=0.03,
        colorbar_pad=0.02,
    ):
        """
        Render 3D colored plane slices for:
        - each solution in self.load_sols(tol)
        - (optionally) the coherent sum of all solutions

        Can optionally add the *incident* field consistently with your RHS generation:
        - Laplace-like: monopole sources outside bounding sphere
        - Helmholtz/Maxwell: plane waves with directions from generate_directions

        Returns
        -------
        list[pathlib.Path] (full paths) of saved images (empty list if save_plot=False)
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        import meshio
        import trimesh
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        import bempp_cl.api
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # ---- local copies of your RHS helpers (so this method is self-contained) ----
        def generate_directions(n_dirs):
            if n_dirs <= 1:
                return np.array([[1.0, 0.0, 0.0]], dtype=float)
            indices = np.arange(0, n_dirs, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n_dirs)
            theta = np.pi * (1 + 5**0.5) * indices
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            dirs = np.stack((x, y, z), axis=1)
            dirs /= np.linalg.norm(dirs, axis=1)[:, None]
            return dirs

        def generate_sources(grid, n_sources, padding=1.2):
            vertices = np.array(grid.vertices)      # (3, Nv)
            center = np.mean(vertices, axis=1)      # (3,)
            max_radius = np.max(np.linalg.norm(vertices.T - center, axis=1))
            radius = padding * max_radius
            if n_sources <= 1:
                return np.array([center + np.array([radius, 0.0, 0.0])], dtype=float)

            indices = np.arange(0, n_sources, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n_sources)
            theta = np.pi * (1 + 5**0.5) * indices
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            directions = np.stack((x, y, z), axis=1)
            sources = center[None, :] + radius * directions
            return sources

        # ---- input checks ----
        plane = plane.lower()
        if plane not in ("xy", "xz", "yz"):
            raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

        if rhs_kind not in ("auto", "plane_wave", "monopole", "none"):
            raise ValueError("rhs_kind must be one of: 'auto', 'plane_wave', 'monopole', 'none'")

        if contrast not in ("linear", "power", "log"):
            raise ValueError("contrast must be one of: 'linear', 'power', 'log'")

        # ---- locate grid + load solutions ----
        sols = self.load_sols(tol)
        if len(sols) == 0:
            raise ValueError("No solutions loaded.")

        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        grid_path = Path(os.getcwd()) / "results" / folder / subfolder / "grid.msh"
        if not grid_path.exists():
            self.relocate_grid()
        if not grid_path.exists():
            raise FileNotFoundError(f"Expected mesh at {grid_path}, but it doesn't exist.")

        # ---- mesh bounds ----
        m = meshio.read(str(grid_path))
        faces = next(c.data for c in m.cells if c.type == "triangle")
        verts = m.points.astype(float)

        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        bmin, bmax = tm.bounds

        if box_limits is not None:
            bmin = np.asarray(box_limits[0], dtype=float)
            bmax = np.asarray(box_limits[1], dtype=float)

        bbox_diag = float(np.linalg.norm(bmax - bmin))
        ext = plane_extent_rel * bbox_diag

        xmin, xmax = bmin[0] - ext, bmax[0] + ext
        ymin, ymax = bmin[1] - ext, bmax[1] + ext
        zmin, zmax = bmin[2] - ext, bmax[2] + ext

        if plane_value is None:
            if plane == "xy":
                plane_value = 0.5 * (bmin[2] + bmax[2])
            elif plane == "xz":
                plane_value = 0.5 * (bmin[1] + bmax[1])
            else:
                plane_value = 0.5 * (bmin[0] + bmax[0])

        # ---- bempp setup ----
        grid = bempp_cl.api.import_grid(str(grid_path))
        op = self.structured_operator_types[self.operator_type_index]

        if op == "BemppClLaplaceSingleLayer":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space

            def make_potential(points):
                return bempp_cl.api.operators.potential.laplace.single_layer(space, points)

            is_vector_field = False

        elif op in ("BemppClHelmholtzSingleLayer", "BemppClHelmholtzCombined", "BemppClBurtonMiller"):
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space

            def make_potential(points):
                return bempp_cl.api.operators.potential.helmholtz.single_layer(space, points, self.kappa)

            is_vector_field = False

        elif op == "BemppClMaxwellEfie":
            space = bempp_cl.api.function_space(grid, "RWG", 0)
            dual_space = space

            def make_potential(points):
                return bempp_cl.api.operators.potential.maxwell.electric_field(space, points, self.kappa)

            is_vector_field = True

        else:
            raise ValueError(f"Unsupported operator type for slicing plot: {op}")

        # ---- sample points on the plane ----
        n = int(plane_points)

        if plane == "xy":
            xs = np.linspace(xmin, xmax, n)
            ys = np.linspace(ymin, ymax, n)
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            Z = plane_value * np.ones_like(X)
            pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        elif plane == "xz":
            xs = np.linspace(xmin, xmax, n)
            zs = np.linspace(zmin, zmax, n)
            X, Z = np.meshgrid(xs, zs, indexing="xy")
            Y = plane_value * np.ones_like(X)
            pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        else:  # "yz"
            ys = np.linspace(ymin, ymax, n)
            zs = np.linspace(zmin, zmax, n)
            Y, Z = np.meshgrid(ys, zs, indexing="xy")
            X = plane_value * np.ones_like(Y)
            pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

        # ---- structured triangulation of the plane (no Delaunay) ----
        ii, jj = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="xy")
        v00 = (jj * n + ii).ravel()
        v10 = (jj * n + (ii + 1)).ravel()
        v01 = ((jj + 1) * n + ii).ravel()
        v11 = ((jj + 1) * n + (ii + 1)).ravel()
        tri1 = np.stack([v00, v10, v11], axis=1)
        tri2 = np.stack([v00, v11, v01], axis=1)
        triangles = np.vstack([tri1, tri2])

        Vplane = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        tris_xyz = Vplane[triangles]

        # ---- decide incident model consistent with your RHS ----
        if not include_incident or rhs_kind == "none":
            use_incident = False
            inc_mode = "none"
        else:
            use_incident = True
            if rhs_kind == "auto":
                # Laplace-like -> monopoles, others -> plane waves
                inc_mode = "monopole" if ("Laplace" in op) else "plane_wave"
            else:
                inc_mode = rhs_kind

        # how many incident fields we generate (usually equals #RHS == #sols)
        if rhs_count is None:
            rhs_count = len(sols)

        d_list = None
        s_list = None
        pol = np.asarray(polarization, dtype=float)

        if use_incident and inc_mode == "plane_wave":
            if not hasattr(self, "kappa") or self.kappa is None:
                raise ValueError("Plane-wave incident requested, but self.kappa is None.")
            d_list = generate_directions(rhs_count)  # (rhs_count, 3)
        if use_incident and inc_mode == "monopole":
            s_list = generate_sources(grid, rhs_count, padding=rhs_padding)  # (rhs_count, 3)

        def incident_at_points(i: int):
            """Return incident evaluated at pts for solution i."""
            if not use_incident:
                return None
            if (rhs_index is not None) and (i != int(rhs_index)):
                return None

            if inc_mode == "plane_wave":
                d = d_list[i % rhs_count]
                phase = np.exp(1j * self.kappa * (d @ pts))  # (N,)
                if is_vector_field:
                    return pol[:, None] * phase[None, :]      # (3,N)
                return phase                                 # (N,)

            if inc_mode == "monopole":
                s = s_list[i % rhs_count]
                r = np.linalg.norm(pts.T - s[None, :], axis=1)  # (N,)
                return 1.0 / (4.0 * np.pi * r)                  # (N,)
            return None

        # ---- evaluate all solutions and (optionally) sum ----
        pot = make_potential(pts)

        vals_plot = []  # each entry: (N,) or (3,N)
        val_sum = None

        for i, sol in enumerate(sols):
            sol_fun = bempp_cl.api.GridFunction(space, projections=sol, dual_space=dual_space)
            val_sc = pot * sol_fun
            val_sc = np.asarray(val_sc)

            # add incident (in FIELD space) if requested
            inc = incident_at_points(i)
            if inc is not None:
                val_i = val_sc + inc
            else:
                val_i = val_sc

            # ensure shapes are consistent
            if not is_vector_field:
                val_i = val_i.reshape(-1)  # (N,)
            vals_plot.append(val_i)

            if plot_sum:
                val_sum = val_i if (val_sum is None) else (val_sum + val_i)

        def scalar_from_val(val):
            """Convert field to scalar per point for coloring."""
            if is_vector_field:
                # intensity |E|^2
                return np.real(np.sum(val * np.conj(val), axis=0)).reshape((n, n))
            else:
                # magnitude
                return np.abs(val).reshape((n, n))

        S_list = [scalar_from_val(v) for v in vals_plot]
        S_sum = scalar_from_val(val_sum) if plot_sum else None

        # ---- shared normalization (optional) ----
        global_norm = None
        if shared_color_scale:
            chunks = [S.ravel() for S in S_list]
            if plot_sum:
                chunks.append(S_sum.ravel())
            all_scalars = np.concatenate(chunks)
            finite = np.isfinite(all_scalars)
            if np.any(finite):
                vmin_g = float(np.nanmin(all_scalars[finite]))
                vmax_g = float(np.nanmax(all_scalars[finite]))
                if vmax_g <= vmin_g:
                    vmax_g = vmin_g + 1.0
            else:
                vmin_g, vmax_g = 0.0, 1.0

            # if using log/power, we still apply contrast/clip *per image* below unless you want strict global
            global_norm = colors.Normalize(vmin=vmin_g, vmax=vmax_g)

        cmap = cm.get_cmap(cmap_name)

        def make_local_norm(s_tri):
            """Create a norm based on chosen contrast mode + clipping."""
            s = np.asarray(s_tri, dtype=float)
            s = s[np.isfinite(s)]
            if s.size == 0:
                return colors.Normalize(vmin=0.0, vmax=1.0)

            if clip_percentiles is not None:
                p0, p1 = clip_percentiles
                vmin = float(np.nanpercentile(s, p0))
                vmax = float(np.nanpercentile(s, p1))
            else:
                vmin = float(np.nanmin(s))
                vmax = float(np.nanmax(s))

            if vmax <= vmin:
                vmax = vmin + 1.0

            if contrast == "linear":
                return colors.Normalize(vmin=vmin, vmax=vmax)

            if contrast == "power":
                return colors.PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax)

            # contrast == "log"
            eps = 1e-12 * vmax if vmax > 0 else 1e-12
            vmin = max(vmin, eps)
            return colors.LogNorm(vmin=vmin, vmax=vmax)

        def render_one(S, out_path):
            # per-triangle scalar (mean of its vertices)
            s_vert = S.ravel()
            s_tri = s_vert[triangles].mean(axis=1)

            # choose norm:
            # - if shared_color_scale and linear, you can use global_norm
            # - but for contrast="power"/"log" you generally want local norm (better exterior contrast)
            if shared_color_scale and contrast == "linear" and global_norm is not None:
                local_norm = global_norm
            else:
                local_norm = make_local_norm(s_tri)

            facecolors = cmap(local_norm(s_tri))
            facecolors[:, 3] = plane_alpha

            fig = plt.figure(figsize=figsize, dpi=dpi)

            if reserve_colorbar_space and show_colorbar:
                left = 0.02
                bottom = 0.02
                height = 0.96
                width = 1.0 - left - colorbar_pad - colorbar_width - 0.02
                ax = fig.add_axes([left, bottom, width, height], projection="3d")
            else:
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")

            ax.view_init(elev=elev, azim=azim)
            ax.grid(False)

            if transparent_bg:
                fig.patch.set_alpha(0.0)
                ax.set_facecolor((1, 1, 1, 0))

            coll = Poly3DCollection(tris_xyz, facecolors=facecolors, edgecolors="none", linewidths=0.0)
            try:
                coll.set_antialiased(False)
            except Exception:
                pass
            ax.add_collection3d(coll)

            if show_colorbar:
                mappable = cm.ScalarMappable(norm=local_norm, cmap=cmap)
                mappable.set_array([])
                if reserve_colorbar_space:
                    cax_left = 1.0 - colorbar_width - 0.02
                    cax = fig.add_axes([cax_left, 0.18, colorbar_width, 0.64])
                else:
                    cax = fig.add_axes([0.90, 0.18, 0.03, 0.64])
                fig.colorbar(mappable, cax=cax)

            ax.set_xlim(bmin[0], bmax[0])
            ax.set_ylim(bmin[1], bmax[1])
            ax.set_zlim(bmin[2], bmax[2])
            ax.set_box_aspect(bmax - bmin)

            if hide_axes:
                ax.set_axis_off()
            else:
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, transparent=transparent_bg)
            plt.close(fig)

        if not save_plot:
            return []

        base = Path(os.getcwd()) / "results" / folder / subfolder
        outs = []

        for i, S in enumerate(S_list):
            out_i = base / f"{out_name_prefix}_{plane}_{i:02d}.png"
            render_one(S, out_i)
            outs.append(out_i)

        if plot_sum:
            out_s = base / f"{out_name_prefix}_{plane}_sum.png"
            render_one(S_sum, out_s)
            outs.append(out_s)

        return outs
    
    def save_clipped_mesh_piece_renders(
        self,
        plane_y=None,                 # ZX plane: y=plane_y
        plane_z=None,                 # XY plane: z=plane_z
        out_dir_name="clipped_mesh_renders",
        # rendering (match your field render view!)
        elev=25,
        azim=-60,
        facecolor=(0.75, 0.75, 0.75),
        edgecolor="none",
        mesh_alpha=1.0,
        dpi=300,
        transparent=True,
        figsize=(10, 7),              # IMPORTANT: match plot_field_slices_3d
        # clipping options
        cap=False,                    # keep False for open surface meshes
        only_watertight=False,        # your mesh is not watertight
        min_faces=50,                 # discard tiny fragments
        verbose=True,
        padd = 0.0,
    ):
        """
        True mesh cutting: clip the surface mesh by planes
        - y = plane_y (ZX)
        - z = plane_z (XY)
        then split into connected components and render each as a 3D PNG (transparent).

        This produces real clipped triangle geometry (triangles are split at plane intersections),
        not a plane-projected footprint.

        Output:
        results/<folder>/<subfolder>/<out_dir_name>/piece_00.png, piece_01.png, ...

        Notes for compositing/alignment:
        - fixed figsize & dpi
        - no tight_layout
        - no bbox_inches="tight"
        - axes occupy full canvas via add_axes([0,0,1,1])
        """
        import os
        import numpy as np
        import meshio
        import trimesh
        import matplotlib.pyplot as plt
        from pathlib import Path
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # -------- mesh path exactly like plot_field_slices_3d --------
        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        grid_path = Path(os.getcwd()) / "results" / folder / subfolder / "grid.msh"
        if not grid_path.exists():
            self.relocate_grid()
        if not grid_path.exists():
            raise FileNotFoundError(f"Expected mesh at {grid_path}, but it doesn't exist.")

        # -------- load triangles --------
        m = meshio.read(str(grid_path))
        faces = next(c.data for c in m.cells if c.type == "triangle")
        verts = m.points.astype(float)

        # bbox + defaults for planes
        bmin = verts.min(axis=0) - padd
        bmax = verts.max(axis=0) + padd
        if plane_y is None:
            plane_y = 0.5 * (bmin[1] + bmax[1])
        if plane_z is None:
            plane_z = 0.5 * (bmin[2] + bmax[2])

        # build trimesh
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # -------- helpers: clip mesh by plane keeping one side --------
        # trimesh slice_mesh_plane keeps points where (x - origin)·normal >= 0
        def clip_keep_ge(mesh, coord, value):
            # keep coord >= value
            origin = np.zeros(3)
            origin[coord] = value
            normal = np.zeros(3)
            normal[coord] = 1.0
            return trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=normal, plane_origin=origin, cap=cap
            )

        def clip_keep_le(mesh, coord, value):
            # keep coord <= value  -> normal = -e_coord
            origin = np.zeros(3)
            origin[coord] = value
            normal = np.zeros(3)
            normal[coord] = -1.0
            return trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=normal, plane_origin=origin, cap=cap
            )

        # coord indices: x=0, y=1, z=2
        y_idx, z_idx = 1, 2

        # -------- clip into 4 quadrants in (y,z) --------
        pieces = []

        # (y <= plane_y, z <= plane_z)
        m00 = clip_keep_le(tm, y_idx, plane_y)
        m00 = clip_keep_le(m00, z_idx, plane_z)
        pieces.append(m00)

        # (y <= plane_y, z >= plane_z)
        m01 = clip_keep_le(tm, y_idx, plane_y)
        m01 = clip_keep_ge(m01, z_idx, plane_z)
        pieces.append(m01)

        # (y >= plane_y, z <= plane_z)
        m10 = clip_keep_ge(tm, y_idx, plane_y)
        m10 = clip_keep_le(m10, z_idx, plane_z)
        pieces.append(m10)

        # (y >= plane_y, z >= plane_z)
        m11 = clip_keep_ge(tm, y_idx, plane_y)
        m11 = clip_keep_ge(m11, z_idx, plane_z)
        pieces.append(m11)

        # split each quadrant into connected components
        comps = []
        for p in pieces:
            if p is None or len(p.faces) == 0:
                continue
            for c in p.split(only_watertight=only_watertight):
                if len(c.faces) >= min_faces:
                    comps.append(c)

        if verbose:
            print(f"plane_y={plane_y:.6g}, plane_z={plane_z:.6g}")
            print(f"Total kept components: {len(comps)} (min_faces={min_faces})")

        # -------- render each component as 3D png --------
        out_base = Path(os.getcwd()) / "results" / folder / subfolder / out_dir_name
        out_base.mkdir(parents=True, exist_ok=True)

        # fixed global framing so overlays match your field render
        bminp, bmaxp = bmin, bmax

        def render_component(comp_mesh, out_path):
            V = comp_mesh.vertices
            F = comp_mesh.faces
            tris = V[F]  # (K,3,3)

            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")  # full canvas
            ax.view_init(elev=elev, azim=azim)
            ax.grid(False)

            if transparent:
                fig.patch.set_alpha(0.0)
                ax.set_facecolor((1, 1, 1, 0))

            coll = Poly3DCollection(tris, facecolor=facecolor, edgecolor=edgecolor, alpha=mesh_alpha)
            try:
                coll.set_zsort("max")
            except Exception:
                pass
            ax.add_collection3d(coll)

            ax.set_xlim(bminp[0], bmaxp[0])
            ax.set_ylim(bminp[1], bmaxp[1])
            ax.set_zlim(bminp[2], bmaxp[2])
            ax.set_box_aspect(bmaxp - bminp)
            ax.set_axis_off()

            # IMPORTANT: no bbox_inches="tight"
            fig.savefig(out_path, dpi=dpi, transparent=transparent)
            plt.close(fig)

        saved = []
        for i, c in enumerate(comps):
            p = out_base / f"piece_{i:02d}.png"
            render_component(c, p)
            saved.append(p)

        if verbose:
            print(f"Saved {len(saved)} PNGs to {out_base}")

        box_limits = (bminp.copy(), bmaxp.copy())
        return saved, box_limits

    def composite_images(
        self,
        background_relpath: str,
        overlay_relpath: str,
        out_relpath: str = "composited.png",
        overlay_offset_px: tuple[int, int] = (0, 0),
        verbose: bool = True,
    ):
        """
        Strict pixel-aligned alpha compositing.
        Both images must have identical pixel size.

        Paths are relative to:
            results/<folder>/<subfolder>/

        Parameters
        ----------
        background_relpath : str
        overlay_relpath : str
        out_relpath : str
        overlay_offset_px : (dx, dy)
            Pixel shift of overlay (default 0,0)
        """

        import os
        from pathlib import Path
        from PIL import Image

        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        base = Path(os.getcwd()) / "results" / folder / subfolder

        bg_path = base / background_relpath
        fg_path = base / overlay_relpath
        out_path = base / out_relpath

        if not bg_path.exists():
            raise FileNotFoundError(f"Background image not found: {bg_path}")
        if not fg_path.exists():
            raise FileNotFoundError(f"Overlay image not found: {fg_path}")

        bg = Image.open(bg_path).convert("RGBA")
        fg = Image.open(fg_path).convert("RGBA")

        # --- force overlay (mesh) to be opaque wherever it exists ---
        r, g, b, a = fg.split()

        # convert any non-zero alpha to fully opaque
        a = a.point(lambda p: 255 if p > 0 else 0)

        fg = Image.merge("RGBA", (r, g, b, a))

        if verbose:
            print("Background size:", bg.size)
            print("Overlay size:   ", fg.size)

        # STRICT size check (important)
        if bg.size != fg.size:
            raise ValueError(
                "Image sizes differ. "
                "Do NOT resize for scientific overlays.\n"
                f"Background: {bg.size}, Overlay: {fg.size}\n"
                "Fix your savefig settings instead."
            )

        # Optional offset (rarely needed)
        dx, dy = overlay_offset_px
        if (dx, dy) != (0, 0):
            shifted = Image.new("RGBA", bg.size, (0, 0, 0, 0))
            shifted.paste(fg, (dx, dy), fg)
            fg = shifted

        out = Image.alpha_composite(bg, fg)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)

        if verbose:
            print("Saved:", out_path)

        return out_path

    def plot_gmres_residuals(self, log_scale=True, save_plot=False, fontsize=16):
        """
        Plot GMRES residuals comparing ONLY the first RHS (index 0).

        - Plots NO-PRECONDITIONER curve only once (since it's identical across tolerances).
        - Plots RSRS preconditioned curves for each tolerance.
        - Legend labels: "No preconditioner" and "RSRS prec (k=<tol>)"
        - Title: "GMRES residuals"
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        import os

        all_stats = self.load_all_stats(kind="error")
        if not all_stats:
            print("No data found.")
            return

        all_stats.sort(key=lambda d: float(d["tolerance"]))

        plt.figure(figsize=(10, 6))

        plotted_no_prec = False

        for stat in all_stats:
            tol = float(stat["tolerance"])
            solves = stat.get("solves", {})

            no_prec_list = solves.get("no_prec", [])
            prec_list = solves.get("prec", [])

            # need at least one RHS
            if not isinstance(prec_list, list) or len(prec_list) == 0:
                continue
            if not isinstance(no_prec_list, list) or len(no_prec_list) == 0:
                # if no_prec missing, still plot prec
                no_prec_list = None

            # first RHS residual vectors
            prec_res = prec_list[0]
            if not isinstance(prec_res, list) or len(prec_res) == 0:
                continue

            if (not plotted_no_prec) and (no_prec_list is not None):
                no_prec_res = no_prec_list[0]
                if isinstance(no_prec_res, list) and len(no_prec_res) > 0:
                    if log_scale:
                        plt.semilogy(no_prec_res, lw=2.5, label="No preconditioner")
                    else:
                        plt.plot(no_prec_res, lw=2.5, label="No preconditioner")
                    plotted_no_prec = True

            label_p = f"RSRS prec (k={tol})"
            if log_scale:
                plt.semilogy(prec_res, lw=2.0, alpha=0.9, label=label_p)
            else:
                plt.plot(prec_res, lw=2.0, alpha=0.9, label=label_p)

        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel("Residual", fontsize=fontsize)
        plt.title("GMRES residuals", fontsize=fontsize + 2)

        plt.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.5)

        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)

        plt.legend(fontsize=fontsize - 4, ncol=1, loc="best", frameon=True)
        plt.tight_layout()

        if save_plot:
            folder = self.generate_folder_name()
            subfolder = self.generate_sub_folder_name()
            path = Path(os.getcwd()) / "results" / folder / subfolder / "gmres_residuals_first_rhs.png"
            plt.savefig(path, dpi=300)

        plt.show()

    def get_existing_slice_paths(self, out_name_prefix="field_only_z", plane="xz"):
        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        base = Path(os.getcwd()) / "results" / folder / subfolder

        # Grab all images for this prefix+plane
        paths = list(base.glob(f"{out_name_prefix}_{plane}_*.png"))

        # Sort so _00, _01, ..., _sum comes last
        def key(p: Path):
            m = re.search(rf"{re.escape(out_name_prefix)}_{plane}_(\d+)\.png$", p.name)
            if m:
                return (0, int(m.group(1)))
            if p.name.endswith("_sum.png"):
                return (1, 10**9)
            return (2, p.name)

        return sorted(paths, key=key)
    
    def sample_with_python(self, init_samples):
        module = importlib.import_module("python.structured_operators")
        class_name = self.structured_operator_types[self.operator_type_index]
        OperatorClass = getattr(module, class_name)
        kappa = self.kappa
        dim_param = self.h
        geometry_type = camel_to_snake(self.geometry_types[self.geometry])
        precision = self.precision_types[self.precision_index]
        assembler = "fmm"
        OperatorClass(dim_param, kappa, geometry_type, precision, 0, init_samples, assembler)

    def summarize_gmres_cases(self, as_dataframe=False, sort_by_tolerance=True):
        """
        Summarize each case for table-making.

        For each stats file/case, returns:
        - tolerance
        - avg iterations for no preconditioner
        - avg iterations for RSRS preconditioner
        - solve_error
        - app_condition_number
        - norm_2_error
        - norm_2_error_inv
        - number of RHS solves found in each category

        Notes
        -----
        - Iteration count is taken as len(residual_history) for each RHS.
        - Averages are taken over all available RHS residual histories.
        - Missing data are returned as None.
        """
        all_stats = self.load_all_stats(kind="error")
        if not all_stats:
            return None if not as_dataframe else None

        def _avg_iterations(residual_histories):
            """
            residual_histories is expected to be a list of lists:
                [
                    [r0, r1, r2, ...],   # rhs 0
                    [r0, r1, r2, ...],   # rhs 1
                    ...
                ]
            """
            if not isinstance(residual_histories, list) or len(residual_histories) == 0:
                return None, 0

            iteration_counts = []
            for hist in residual_histories:
                if isinstance(hist, list) and len(hist) > 0:
                    iteration_counts.append(len(hist))

            if not iteration_counts:
                return None, 0

            return sum(iteration_counts) / len(iteration_counts), len(iteration_counts)

        rows = []

        for stat in all_stats:
            solves = stat.get("solves", {})
            no_prec_list = solves.get("no_prec", [])
            prec_list = solves.get("prec", [])

            avg_no_prec, n_no_prec = _avg_iterations(no_prec_list)
            avg_prec, n_prec = _avg_iterations(prec_list)

            row = {
                "tolerance": stat.get("tolerance", None),
                "avg_iters_no_prec": avg_no_prec,
                "avg_iters_prec": avg_prec,
                "num_rhs_no_prec": n_no_prec,
                "num_rhs_prec": n_prec,
                "solve_error": stat.get("solve_error", None),
                "app_condition_number": stat.get("app_condition_number", None),
                "norm_2_error": stat.get("norm_2_error", None),
                "norm_2_error_inv": stat.get("norm_2_error_inv", None),
            }
            rows.append(row)

        if sort_by_tolerance:
            def _tol_key(row):
                tol = row["tolerance"]
                try:
                    return float(tol)
                except (TypeError, ValueError):
                    return float("inf")

            rows.sort(key=_tol_key)

        if as_dataframe:
            import pandas as pd
            return pd.DataFrame(rows)

        return rows
