# Testing framework for bempp-rsrs

This is a testing framework that interfaces Structured Operators from Python and performs RSRS on them through bempp-rsrs

## Installation

This crate retrieves bempp-cl and KiFMM operators, so both of them should be installed. Here we show an installation through an uv Virtual Environment, but any other python enviroment platform should be suitable for testing. KiFMM 
uses Python 10 and here we repeat the instructions provided in that crate.

1. Begin by installing rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Clone kifmm from https://github.com/bempp/kifmm

2. Install Maturin and pip in a new virtual environment.

```bash
uv venv --python=3.10 && source .venv/bin/activate && uv pip install maturin pip
```

2. Use the Maturin CLI to install the Python bindings into this virtual environment, which additionally will install all required dependencies.

Note that Maturin must be run from the `kifmm` crate root, not the workspace root, and that setuptools must be set to a version compatible with Mayavi.

```bash
pip install 'setuptools<69' .
```

NOTE: If the Mayavi installation fails, it can be removed by commenting the line "mayavi==4.8.1" from the pyproject.toml dependencies in the python folder. This will not affect KiFMM as it is only meant for plotting.

3. Install the Python extension of KiFMM:

```bash
maturin develop --release
```

4. After the installation of kifmm you can install bempp-cl. It is not necessary to install everything from source, except from exafmm:

```bash
git clone https://github.com/exafmm/exafmm-t.git
cd exafmm-t
CFLAGS="-O3" CXXFLAGS="-O3" ./configure
make
make install
python setup.py install
```

To install bempp-cl, in general 

```bash
uv pip install bempp-cl
```

should be enough, but sometimes you will need to install gmsh separately. In this case you should run:

```bash
uv pip install gmsh
```

NOTE: Exafmm will not work o Apple Silicon. To run these operators on your mac, you should remove `assembler="fmm"` when building an operator.



## Example of usage:

This is a compilation of examples of uses of RSRS


```python
from rsrs_config import RSRSBenchmarkConfig
print(RSRSBenchmarkConfig.__init__.__doc__)
```

    
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
            



```python
## Generate the test case either to run the test or retrieve results
config = RSRSBenchmarkConfig(operator_type=5, dim_arg_type=3, ref_level=5, depth=2)
```


```python
## Generate the shell script (disable it unless you want to run the test)
config.generate_bash_script("run_test.sh")
```


```python

```


```python
## Running test in Rust

!./run_test.sh
```


```python
print(config.plot_errors_vs_tolerance.__doc__)
```

    
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
            loglog : bool, optional
                If True, plot both axes on a logarithmic scale (log-log plot).
                Default is True.
    
            Raises
            ------
            ValueError
                If `metric_index` is not in the range 1 to 5.
            



```python
config.plot_errors_vs_tolerance(1)
```


    
![png](demo_files/demo_7_0.png)
    



```python
config.plot_errors_vs_tolerance(2)
```


    
![png](demo_files/demo_8_0.png)
    



```python
config.plot_errors_vs_tolerance(3)
```


    
![png](demo_files/demo_9_0.png)
    



```python
config.plot_errors_vs_tolerance(4, False)
```


    
![png](demo_files/demo_10_0.png)
    



```python
config.plot_errors_vs_tolerance(5, False)
```


    
![png](demo_files/demo_11_0.png)
    



```python
config.plot_gmres_residuals()
```


    
![png](demo_files/demo_12_0.png)
    



```python
config.plot_residual_convergence()
```


    
![png](demo_files/demo_13_0.png)
    

