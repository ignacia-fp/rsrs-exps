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
        rank_picking: int = 0,
        min_rank: int = 1,
        min_level: int = 1
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
            "KiFMMLaplaceOperator", "KiFMMHelmholtzOperator", "BemppRsLaplaceOperator"
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
        self.geometry_types = ["SphereSurface", "CubeSurface", "CylinderSurface", "EllipsoidSurface"] ## We can generate more with Gmsh and link them

        ## Output Arguments: (outputs that the test the test will return)
        # In all cases, it returns the errors ||A_app^{-1}A - I||_2 and ||A_app - A||_2/||A||_2,
        # but we can also either ask for the compression results (Rank) or time that it takes
        # to complete each RSRS step.
        self.results_outputs = ["All", "Rank", "Time"]

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

        # Store other parameters
        self.h = h ## Characteristic meshwidth
        self.kappa = kappa ## Wavenumber of oscillatory problems. It must be defined for these cases. 
                           ## It also serves as a parameter to compute h when provided.

        self.ref_level = ref_level
        self.depth = depth
        self.id_tols = id_tols
        self.solve_tol = solve_tol
        self.solve = solve
        self.plot = plot
        self.dense_errors = dense_errors
        self.factors_cn = factors_cn
        self.min_rank = min_rank
        self.min_level = min_level

        if self.dim_arg_types[self.dim_arg_type_index] == "RefinementLevelAndDepth":
            if self.ref_level is None or self.depth is None:
                raise ValueError("Both `ref_level` and `depth` must be specified for 'RefinementLevelAndDepth' dim_arg_type.")
        
        if self.dense_errors:
            print("WARNING: computind dense errors is time and memory intensive. Use only for small test cases")

        if self.structured_operator_types[self.operator_type_index] == "BemppClHelmholtzSingleLayer" or self.structured_operator_types[self.operator_type_index] == "KiFMMHelmholtzOperator":
            if self.kappa == None:
                raise ValueError("A wavenumber must be provided.")
            
        if self.structured_operator_types[self.operator_type_index] != "BemppClHelmholtzSingleLayer" or self.structured_operator_types[self.operator_type_index] != "KiFMMHelmholtzOperator" or self.structured_operator_types[self.operator_type_index] != "BemppRsLaplaceOperator":
            if self.dim_arg_types[self.dim_arg_type_index] == "Kappa":
                if self.kappa != None:
                    print("WARNING: Computing h from given kappa.")
                else:
                    raise ValueError("You must provide kappa for this option.")

        if self.structured_operator_types[self.operator_type_index] == "BemppRsLaplaceOperator":
            if self.geometry_types[self.geometry] != "SphereSurface":
                print("WARNING: For Bempp-rs we only have spherical surfaces for now.")
            if self.dim_arg_types[self.dim_arg_type_index] != "RefinementLevelAndDepth":
                raise ValueError("For Bempp-rs, 'RefinementLevelAndDepth' must be used to define the mesh width.")
            if self.dense_errors:
                raise ValueError("There is no dense form of this operator.")

    def data_type_args(self) -> Dict[str, str]:
        return {
            "structured_operator_type": self.structured_operator_types[self.operator_type_index],
            "precision": self.precision_types[self.precision_index],
        }

    def scenario_args(self) -> Dict[str, Union[List[float], List[Dict], str]]:
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
            "oversampling": 8,  ## Oversampling for each individual block
            "oversampling_diag_blocks": 16,  ## Oversampling used when extracting diagonal blocks when RSRS finishes
            "initial_num_samples": 20,  ## Initial num samples: useful only when sampling is done in parallel way (not active yet)
            "null_method": self.null_methods[self.null_method_index],
            "near_block_extraction_method": self.block_extraction_methods[self.block_extraction_method_index],
            "diag_block_extraction_method": self.block_extraction_methods[self.block_extraction_method_index],
            "lu_pivot_method": self.pivot_methods[self.pivot_method_index],
            "diag_pivot_method": self.pivot_methods[self.pivot_method_index],
            "tol_null": 1e-10,  ## Tolerance when nullifying blocks
            "tol_id": self.id_tols[0],  ## ID tolerance (Irrelevant, since it is set with the scenario arguments)
            "tol_ext_near": 1e-10,  ## Tolerance used to compute pseudo inverses when extracting near field.
            "tol_diag_ext": 1e-16,  ## Tolerance used to compute pseudo inverses when extracting diagonal blocks.
            "min_rank": self.min_rank,  ## Minimum size of the box. If the box is smaller, it will be saved for the next level.
            "min_level": self.min_level, ## Level at which the algorithm stops
            "hermitian": True,  ## Indicates if we should run RSRS for hermitian matrices (half the time and memory)
            "rank_picking": self.rank_pickings[self.rank_picking_index],
        }

    def as_dict(self) -> Dict[str, Dict]:
        return {
            "data_type_args": self.data_type_args(),
            "scenario_args": self.scenario_args(),
            "output_args": self.output_args(),
            "rsrs_args": self.rsrs_args(),
        }

    def generate_bash_script(self, filename: str = "run_test.sh"):
        def json_for_bash(obj):
            return json.dumps(obj).replace("'", "'\"'\"'")

        data_type_args_json = json_for_bash(self.data_type_args())
        scenario_args_json = json_for_bash(self.scenario_args())
        output_args_json = json_for_bash(self.output_args())
        rsrs_args_json = json_for_bash(self.rsrs_args())

        bash_script = f"""#!/bin/bash
export OPENBLAS_NUM_THREADS=1
cargo run --release '{data_type_args_json}' '{scenario_args_json}' '{rsrs_args_json}' '{output_args_json}'
"""

        with open(filename, "w") as f:
            f.write(bash_script)

        subprocess.run(["chmod", "+x", filename], check=True)

    def generate_folder_name(self) -> str:
        geom = camel_to_snake(self.geometry_types[self.geometry])
        op = self.structured_operator_types[self.operator_type_index]
        dim_key = self.dim_arg_types[self.dim_arg_type_index]

        if dim_key == "RefinementLevelAndDepth":
            return f"{geom}_{op}_ref_level_{self.ref_level}_depth_{self.depth}"
        elif dim_key == "Meshwidth":
            return f"{geom}_{op}_mesh_width_{self.h:.1e}"
        elif dim_key == "Kappa":
            return f"{geom}_{op}_kappa_{self.kappa:.2f}"
        elif dim_key == "KappaAndMeshwidth":
            return f"{geom}_{op}_mesh_width_{self.h:.1e}_{self.kappa:.2f}"
        else:
            raise ValueError("Invalid dim_arg_type")

    def generate_sub_folder_name(self) -> str:
        args = self.rsrs_args()
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

    
    def plot_errors_vs_tolerance(self, metric_index=1, logx=True, logy=True):
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
        plt.show()


    def plot_gmres_residuals(self, log_scale=True):
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
        plt.show()


    def plot_residual_convergence(self, log_scale=True):
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
        plt.show()

    def plot_total_elapsed_time_vs_tolerance(self, logx=True, logy=True):
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
        plt.show()

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

    def plot_time_breakdown_piecharts(self, max_charts=None):
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
            plt.show()

    def plot_condition_numbers(self):
        stats = self.load_all_stats("condition_number")

        # Sort by tolerance
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Condition Numbers for Tolerance = {tol}")

            # --- ID Factors (use first entry only) ---
            id_levels = data.get("id", [])
            id_vals = []
            for level in id_levels:
                clean = [vec[0] for vec in level if vec and len(vec) >= 1]
                if clean:
                    id_vals.append(clean)

            if id_vals:
                axs[0].boxplot(id_vals, vert=True)
                axs[0].set_title("ID Factor Condition Numbers")
                axs[0].set_xlabel("Level")
                axs[0].set_ylabel("cond")
            else:
                axs[0].set_visible(False)

            # --- LU Factors (use second entry only) ---
            lu_levels = data.get("lu", [])
            lu_vals = []
            for level in lu_levels:
                clean = [vec[1] for vec in level if vec and len(vec) >= 2]
                if clean:
                    lu_vals.append(clean)

            if lu_vals:
                axs[1].boxplot(lu_vals, vert=True)
                axs[1].set_title("LU Factor Condition Numbers")
                axs[1].set_xlabel("Level")
                axs[1].set_ylabel("cond")
            else:
                axs[1].set_visible(False)

            # --- DFactors (use both entries) ---
            dfactors = data.get("dfactors", [])
            dfactor_vals = [(vec[0], vec[1]) for vec in dfactors if vec and len(vec) == 2]
            if dfactor_vals:
                cond1_vals, cond2_vals = zip(*dfactor_vals)
                axs[2].boxplot([cond1_vals, cond2_vals], vert=True)
                axs[2].set_title("Diagonal Factor Condition Numbers")
                axs[2].set_xlabel("Factor")
                axs[2].set_ylabel("cond")
                axs[2].set_xticklabels(["diagonal_factor_1", "diagonal_factor_2"])
            else:
                axs[2].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_condition_numbers_scatter(self):
        stats = self.load_all_stats("condition_number")

        # Sort by tolerance
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Condition Numbers for Tolerance = {tol}")

            # --- ID Factors (first entry only) ---
            id_levels = data.get("id", [])
            id_vals = [
                [vec[0] for vec in level if vec and len(vec) >= 1]
                for level in id_levels
            ]
            id_vals = [level for level in id_vals if level]  # Filter empty
            if id_vals:
                for i, level_vals in enumerate(id_vals):
                    sorted_vals = sorted(level_vals)
                    x = np.arange(len(sorted_vals))
                    jitter = np.random.normal(loc=0, scale=0.1, size=len(x))
                    axs[0].scatter(x + jitter, sorted_vals, alpha=0.7, label=f"Level {i}", s=10)
                axs[0].set_title("ID Factor Condition Numbers")
                axs[0].set_xlabel("Sorted element index")
                axs[0].set_ylabel("cond")
                axs[0].legend()
            else:
                axs[0].set_visible(False)

            # --- LU Factors (second entry only) ---
            lu_levels = data.get("lu", [])
            lu_vals = [
                [vec[1] for vec in level if vec and len(vec) >= 2]
                for level in lu_levels
            ]
            lu_vals = [level for level in lu_vals if level]  # Filter empty
            if lu_vals:
                for i, level_vals in enumerate(lu_vals):
                    sorted_vals = sorted(level_vals)
                    x = np.arange(len(sorted_vals))
                    jitter = np.random.normal(loc=0, scale=0.1, size=len(x))
                    axs[1].scatter(x + jitter, sorted_vals, alpha=0.7, label=f"Level {i}", s=10)
                axs[1].set_title("LU Factor Condition Numbers")
                axs[1].set_xlabel("Sorted element index")
                axs[1].set_ylabel("cond")
                axs[1].legend()
            else:
                axs[1].set_visible(False)

            # --- DFactors (two separate sorted scatter plots) ---
            dfactors = data.get("dfactors", [])
            dfactor_vals = [(vec[0], vec[1]) for vec in dfactors if vec and len(vec) == 2]
            if dfactor_vals:
                cond1_vals, cond2_vals = zip(*dfactor_vals)
                sorted1 = sorted(cond1_vals)
                sorted2 = sorted(cond2_vals)
                x1 = np.arange(len(sorted1)) + np.random.normal(0, 0.1, size=len(sorted1))
                x2 = np.arange(len(sorted2)) + np.random.normal(0, 0.1, size=len(sorted2))
                axs[2].scatter(x1, sorted1, alpha=0.7, label="diagonal_factor_1", s=10)
                axs[2].scatter(x2, sorted2, alpha=0.7, label="diagonal_factor_2", s=10)
                axs[2].set_title("Diagonal Factor Condition Numbers")
                axs[2].set_xlabel("Sorted element index")
                axs[2].set_ylabel("cond")
                axs[2].legend()
            else:
                axs[2].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        
    def plot_condition_number_summaries(self):
        stats = self.load_all_stats("condition_number")

        # Sort by tolerance
        stats.sort(key=lambda d: float(d["tolerance"]))

        for data in stats:
            if data is None:
                continue

            tol = data["tolerance"]
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"Condition Number Summary Stats for Tolerance = {tol}")

            def plot_summary(ax, levels_data, title):
                # Filter out non-positive values
                filtered = []
                for i, level in enumerate(levels_data):
                    clean_level = [v for v in level if v > 0]
                    if clean_level:
                        filtered.append((i, clean_level))

                if not filtered:
                    ax.set_visible(False)
                    return

                indices, filtered_levels = zip(*filtered)

                means = []
                medians = []
                errors = []

                for level in filtered_levels:
                    arr = np.array(level)
                    means.append(np.mean(arr))
                    medians.append(np.median(arr))
                    errors.append(np.std(arr))

                x = np.arange(len(filtered_levels))
                width = 0.35

                # Plot bars with error bars
                ax.bar(x - width/2, means, width, label='Mean', yerr=errors, capsize=5)
                ax.bar(x + width/2, medians, width, label='Median')

                ax.set_title(title)
                ax.set_xlabel("Level")
                ax.set_ylabel("Condition Number")
                ax.set_xticks(x)
                ax.set_xticklabels([str(i) for i in indices])
                ax.legend()

                # Compute max y considering error bars
                upper_error = [m + e for m, e in zip(means, errors)]
                ymax = max(upper_error + medians)
                ymin = min(min(means), min(medians))

                ax.set_ylim(bottom=max(0.1 * ymin, 1e-12), top=1.1 * ymax)

            # --- ID Factors ---
            id_levels = data.get("id", [])
            id_vals = [
                [vec[0] for vec in level if vec and len(vec) >= 1]
                for level in id_levels
            ]
            plot_summary(axs[0], id_vals, "ID Factor")

            # --- LU Factors ---
            lu_levels = data.get("lu", [])
            lu_vals = [
                [vec[1] for vec in level if vec and len(vec) >= 2]
                for level in lu_levels
            ]
            plot_summary(axs[1], lu_vals, "LU Factor")

            # --- DFactors Cond 1 ---
            dfactors = data.get("dfactors", [])
            cond1_vals = [vec[0] for vec in dfactors if vec and len(vec) == 2 and vec[0] > 0]
            plot_summary(axs[2], [cond1_vals], "Diagonal Factor 1")

            # --- DFactors Cond 2 ---
            cond2_vals = [vec[1] for vec in dfactors if vec and len(vec) == 2 and vec[1] > 0]
            plot_summary(axs[3], [cond2_vals], "Diagonal Factor 2")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()