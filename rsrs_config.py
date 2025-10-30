import json
import numpy as np
import subprocess
from typing import List, Dict, Union
import re
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

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
        solve_tol: float = 1e-5,
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
        op_stabilisation = 0,
        oversampling_near = 8,
        oversampling_diag = 16,
        tol_ext_near = 1e-16,
        tol_ext_diag = 1e-16,
        fact_type = 1,
        n_sources: int = 1,
        save_samples: bool = True
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
            6: BemppClLaplaceSingleLayerModified
            7: BemppClLaplaceSingleLayerCP
            8: BemppClLaplaceSingleLayerMM
            9: BemppClHelmholtzSingleLayerCP
            10: BemppClLaplaceSingleLayerCPID
            11: BemppClLaplaceSingleLayerP1,
            12: KiFMMLaplaceOperatorV,
            13: BemppClLaplaceSingleLayerModifiedP1,
            14: BemppClLaplaceSingleLayerCPIDP1,
            15: BemppClHelmholtzSingleLayerCPID,
            16: BemppClMaxwellEfie
            17: BemppClHelmholtzSingleLayerP1
            18: BemppClCombinedHelmholtz
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
            "KiFMMLaplaceOperator", "KiFMMHelmholtzOperator", "BemppRsLaplaceOperator", "BemppClLaplaceSingleLayerModified",
            "BemppClLaplaceSingleLayerCP", "BemppClLaplaceSingleLayerMM", "BemppClHelmholtzSingleLayerCP", "BemppClLaplaceSingleLayerCPID",
            "BemppClLaplaceSingleLayerP1", "KiFMMLaplaceOperatorV", "BemppClLaplaceSingleLayerModifiedP1", "BemppClLaplaceSingleLayerCPIDP1",
            "BemppClHelmholtzSingleLayerCPID", "BemppClMaxwellEfie", "BemppClHelmholtzSingleLayerP1", "BemppClCombinedHelmholtz"
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
        self.geometry_types = ["SphereSurface", "CubeSurface", "CylinderSurface", "EllipsoidSurface", "Dihedral", "Device", "F16", "RidgedHorn", "EMCCAlmond", "FrigateHull", "Plane"] ## We can generate more with Gmsh and link them

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
        self.op_stabilisation = op_stabilisation
        self.fact_type = fact_type

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
            "Meshwidth": {"Meshwidth": self.h},
            "RefinementLevelAndDepth": {"RefinementLevelAndDepth": (self.ref_level, self.depth)},
        }
        return {
            "id_tols": self.id_tols,
            "dim_args": [dim_args_map[dim_type]],
            "geometry_type": self.geometry_types[self.geometry],
            "max_tree_depth": self.max_tree_depth,
            "n_sources": self.n_sources,
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
            "initial_num_samples": 20,  ## Initial num samples: useful only when sampling is done in parallel way (not active yet)
            "stabilise": stab(self.op_stabilisation),
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
            "hermitian": True,  ## Indicates if we should run RSRS for hermitian matrices (half the time and memory)
            "rank_picking": self.rank_pickings[self.rank_picking_index],
            "fact_type": self.fact_types[self.fact_type],
            "save_samples": self.save_samples
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
        ]

        if rayon_threads is not None:
            bash_lines.append(f"export RAYON_NUM_THREADS={rayon_threads}")
        else:
            # Explicitly clear RAYON_NUM_THREADS to ensure all cores are used
            bash_lines.append("unset RAYON_NUM_THREADS")

        bash_lines.append(
            f"cargo run --release '{data_type_args_json}' '{scenario_args_json}' '{rsrs_args_json}' '{output_args_json}'"
        )

        with open(filename, "w") as f:
            f.write("\n".join(bash_lines) + "\n")

        subprocess.run(["chmod", "+x", filename], check=True)


    def generate_folder_name(self) -> str:
        geom = camel_to_snake(self.geometry_types[self.geometry])
        op = self.structured_operator_types[self.operator_type_index]
        dim_key = self.dim_arg_types[self.dim_arg_type_index]

        if dim_key == "RefinementLevelAndDepth":
            if self.ref_level > 1:
                return f"{geom}_{op}_ref_level_{self.ref_level}_depth_{self.depth}_od_{self.max_tree_depth}"
            else:
                return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}"
        elif dim_key == "Meshwidth":
            return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}"
        elif dim_key == "Kappa":
            return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}_{self.kappa:.2f}"
        elif dim_key == "KappaAndMeshwidth":
            return f"{geom}_{op}_mesh_width_{rust_float_format(self.h)}_od_{self.max_tree_depth}_{self.kappa:.2f}"
        else:
            raise ValueError("Invalid dim_arg_type")

    def generate_sub_folder_name(self) -> str:
        args = self.rsrs_args()

        if self.rrqr_keys[self.rrqr_index] == "SRRQR":
            rrqr_pred = f"srrqr_" + sci_no_padding2(self.f)
        else:
            rrqr_pred = "rrqr"

        if self.op_stabilisation == 0:
            return (
                f"rsrs_null_{args['null_method']}"
                f"_toln_{args['tol_null']:.0e}"
                f"_os_{args['oversampling']}"
                f"_osdiag_{args['oversampling_diag_blocks']}"
                f"_initsam_{args['initial_num_samples']}"
                f"_mrnk_{args['min_rank']}"
                f"_mlvl_{args['min_level']}"
                f"_herm_{camel_to_snake(str(args['hermitian']))}"
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
                f"_stabilised_{sci_no_padding(self.op_stabilisation)}"
                f"_mrnk_{args['min_rank']}"
                f"_mlvl_{args['min_level']}"
                f"_herm_{camel_to_snake(str(args['hermitian']))}"
                f"_rpick_{args['rank_picking']}"
                f"_next_{args['near_block_extraction_method']}"
                f"_tolextn_{args['tol_ext_near']:.0e}"
                f"_db_ext_{args['diag_block_extraction_method']}"
                f"_tol_lstsq_{args['tol_diag_ext']:.0e}"
                f"_{rrqr_pred}"
            )

    def load_all_stats(self, kind="error"):
        """
        Load all JSON stats files of a given kind from the scenario's result directory.
        
        Parameters
        ----------
        kind : str
            Type of statistics to load. Must be one of: 'error', 'time', 'rank', condition_numbers.
            Corresponds to files named like 'error_stats_{tol}.json', etc.
        
        Returns
        -------
        List[dict]
            List of dictionaries containing parsed JSON data, each augmented with a 'tolerance' key.
        """
        valid_kinds = {"error", "time", "rank", "condition_number"}
        if kind not in valid_kinds:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of: {valid_kinds}.")

        folder = self.generate_folder_name()
        subfolder = self.generate_sub_folder_name()
        base_path = Path(os.getcwd()) / "results" / folder / subfolder

        stats = []
        pattern = re.compile(fr"{kind}_stats_(.+)\.json")

        for file in base_path.iterdir():
            if file.is_file() and file.name.startswith(f"{kind}_stats_") and file.suffix == ".json":
                match = pattern.match(file.name)
                if match:
                    tol_str = match.group(1)
                    try:
                        tolerance = float(tol_str)
                    except ValueError:
                        continue  # Skip invalid tolerance values

                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["tolerance"] = tolerance
                        stats.append(data)

        return stats

    
    def plot_errors_vs_tolerance(self, metric_index=1, plot=True, logx=True, logy=True, save_plot=False):
        """
        Plot a specified error metric vs tolerance.

        Parameters
        ----------
        metric_index : int
            The index of the error metric to plot on the y-axis. Must be one of:
            1 - 'norm_2_error'
            2 - 'norm_2_error_inv'
            3 - 'app_condition_number'
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
            "app_condition_number",
            "tot_num_samples",
            "residual_size"
        ]

        pretty_names = {
            "norm_2_error": r"$\|A - A_{\mathrm{app}}\|_2 / \|A\|_2$",
            "norm_2_error_inv": r"$\|A_{\mathrm{app}}^{-1} A - I\|_2$",
            "app_condition_number": "Approximate Condition Number",
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


    def plot_gmres_residuals(self, log_scale=True, save_plot=False):
        all_stats = self.load_all_stats()
        if not all_stats:
            print("No data found.")
            return

        all_stats.sort(key=lambda d: float(d["tolerance"]))

        no_prec_residuals = all_stats[0].get("iterations", {}).get("no_prec", [])
        if not no_prec_residuals:
            print("No residuals found for no_prec.")
            return

        plt.figure(figsize=(8, 5))
        if log_scale:
            plt.semilogy(no_prec_residuals, label="No Preconditioner", linewidth=2, color="black")
        else:
            plt.plot(no_prec_residuals, label="No Preconditioner", linewidth=2, color="black")

        for stat in all_stats:
            tol = stat["tolerance"]
            iterations = stat.get("iterations", {})
            for key, residuals in iterations.items():
                if key == "no_prec" or not isinstance(residuals, list):
                    continue
                label = f"{key} (tol={tol})"
                if log_scale:
                    plt.semilogy(residuals, label=label, alpha=0.7)
                else:
                    plt.plot(residuals, label=label, alpha=0.7)

        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.title("Residuals: No Preconditioner vs Preconditioners")
        plt.legend(fontsize="small", loc="upper right", ncol=2)
        plt.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.5)
        plt.tight_layout()
        if save_plot:
            folder = self.generate_folder_name()
            subfolder = self.generate_sub_folder_name()
            path = Path(os.getcwd()) / "results" / folder / subfolder / "gmres_residuals.png"
            plt.savefig(path, dpi=300)
        plt.show()


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
        all_stats = self.load_all_stats(kind="error")
        all_stats.sort(key=lambda d: float(d["tolerance"]))

        if tol == 0.0:
            solves = all_stats[0].get("solves", {}).get("sols_no_prec", [])
        else:
            ind = 0
            for data in all_stats:
                if data is None:
                    continue

                tol_str = data["tolerance"]
                if float(tol_str) == tol:
                    break
                else:
                    ind += 1
            solves = all_stats[ind].get("solves", {}).get("sols_prec", [])
        solutions = []
        for sol in solves:
            arr = np.array(sol)
                # Real case: 1D array
            if arr.ndim == 1:
                solutions.append(arr.astype(float))

            # Complex case: 2D array with last dimension = 2 ([Re, Im])
            elif arr.ndim == 2 and arr.shape[-1] == 2:
                complex_arr = arr[:, 0] + 1j * arr[:, 1]
                solutions.append(complex_arr)

            else:
                raise ValueError(f"Unexpected solution shape {arr.shape}")
        return solutions

    def get_far_field(self, tol=1e-2, n_grid_points=150, plane=0, lims = [-1,1,-1,1]):
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
            points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size)))
        elif plane ==1:
            # xz plane
            points = np.vstack((plot_grid[0].ravel(), np.zeros(plot_grid[0].size), plot_grid[1].ravel()))
        else:
            # yz plane
            points = np.vstack((np.zeros(plot_grid[0].size),plot_grid[0].ravel(), plot_grid[1].ravel()))
        if self.structured_operator_types[self.operator_type_index] == "BemppClLaplaceSingleLayer":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.laplace.single_layer(space, points)
        elif self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayer":
            space = bempp_cl.api.function_space(grid, "DP", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(space, points, self.kappa)
        elif self.structured_operator_types[self.operator_type_index] == "BemppClMaxwellEfie":
            space = bempp_cl.api.function_space(grid, "RWG", 0)
            dual_space = space
            slp_pot = bempp_cl.api.operators.potential.maxwell.electric_field(space, points, self.kappa)
        for i, sol in enumerate(sols):
            sol_fun = bempp_cl.api.GridFunction(space, projections=sol, dual_space=dual_space)
            far_field = slp_pot * sol_fun
            if self.structured_operator_types[self.operator_type_index] == "BemppClMaxwellEfie":
                scattered_field = np.real(np.sum(far_field * far_field.conj(), axis=0))
            else:
                scattered_field = far_field
            scattered_field = scattered_field.reshape((n_grid_points, n_grid_points))
            plt.imshow(np.abs(scattered_field.T))
            plt.title("Computed solution")
            plt.colorbar()
            sol_path = Path(os.getcwd()) / "results" / folder / subfolder / f"scattered_field_{i}.png"
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
        elif self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayer":
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

        sol_path = Path(os.getcwd()) / "results" / folder / subfolder / f"rcs.png"
        plt.savefig(sol_path)

